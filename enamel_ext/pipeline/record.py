"""What one evaluation run measured, as a file: times, not scores.

Per-sample scores are recomputed from the recorded times on demand, so a rescore
at a different ``alpha`` or ``h`` can never disagree with a stored conclusion.
Rationale in docs/decisions/0006-run-record.md.
"""

from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from enamel_ext.data.schema import Provenance
from enamel_ext.data.sources import provenance_from_json, provenance_to_json
from enamel_ext.measure.calibrate import DRIFT_CAVEAT, Calibration, Drift, compare
from enamel_ext.measure.runner import PAPER_REPEATS, SolutionMeasurement
from enamel_ext.metrics import effk
from enamel_ext.metrics.score import TIMEOUT, MetricConfig, sample_score
from enamel_ext.report.hyperparams import rescore_at_alpha

__all__ = [
    "CENSORED_TOKEN",
    "COMPARABLE_FIELDS",
    "RECORD_SCHEMA_VERSION",
    "Environment",
    "ProblemRecord",
    "RunRecord",
    "SampleRecord",
    "Segment",
    "load_record",
    "record_from_json",
    "record_to_json",
    "save_record",
]

RECORD_SCHEMA_VERSION = 3

#: Environment fields that have to agree for two measurement sessions to belong
#: to one run. ``load_average`` is deliberately absent; see decision 0009.
COMPARABLE_FIELDS = ("python", "platform", "machine", "cpu_count")

#: JSON has no infinity and these files are written with ``allow_nan=False``, so
#: a right-censored time travels as this string instead.
CENSORED_TOKEN = "censored"

def _check_time(t: float, *, what: str) -> float:
    if t != t:
        raise ValueError(f"{what}: NaN timing; a failed measurement is not a slow one")
    if t < 0:
        raise ValueError(f"{what}: negative timing {t}")
    if math.isinf(t):
        return TIMEOUT
    return float(t)


def _time_to_json(t: float) -> float | str:
    return CENSORED_TOKEN if math.isinf(_check_time(t, what="time")) else float(t)


def _time_from_json(value: Any) -> float:
    if value == CENSORED_TOKEN:
        return TIMEOUT
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a time or {CENSORED_TOKEN!r}, got {value!r}")
    return _check_time(float(value), what="stored time")


def _times_to_json(times: Sequence[Sequence[float]]) -> list[list[float | str]]:
    return [[_time_to_json(t) for t in level] for level in times]


def _times_from_json(raw: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(_time_from_json(t) for t in level) for level in raw)


def _freeze_times(
    times: Sequence[Sequence[float]], *, what: str
) -> tuple[tuple[float, ...], ...]:
    out = []
    for index, level in enumerate(times):
        cases = tuple(_check_time(t, what=f"{what} level {index}") for t in level)
        if not cases:
            raise ValueError(f"{what} level {index} has no test cases")
        out.append(cases)
    if not out:
        raise ValueError(f"{what} has no levels")
    return tuple(out)

@dataclass(frozen=True)
class Environment:
    """The machine facts that decide whether two runs are comparable."""

    python: str
    platform: str
    machine: str
    cpu_count: int
    load_average: tuple[float, ...] | None = None
    calibration: Calibration | None = None

    @classmethod
    def capture(cls, calibration: Calibration | None = None) -> Environment:
        """Read the cheap facts. The probe is passed in, never taken here.

        ``capture`` is called all over the test suite and has to stay free; timing
        a probe is seconds of work and is the caller's decision.
        """
        try:
            load: tuple[float, ...] | None = tuple(round(x, 3) for x in os.getloadavg())
        except (AttributeError, OSError):  # not available on every platform
            load = None
        return cls(
            python=f"{platform.python_implementation()} {platform.python_version()}",
            platform=platform.platform(),
            machine=platform.machine(),
            cpu_count=os.cpu_count() or 0,
            load_average=load,
            calibration=calibration,
        )

    def differences(self, other: Environment) -> tuple[str, ...]:
        """Fields whose disagreement makes two measurements incomparable.

        ``load_average`` is left out on purpose: it differs between almost any two
        sessions and is noise on the measurement rather than a change in what is
        being measured. The calibration is left out too, because it is compared by
        magnitude rather than by equality; see :meth:`RunRecord.calibration_drift`.
        """
        return tuple(
            f"{field}: {getattr(self, field)!r} then {getattr(other, field)!r}"
            for field in COMPARABLE_FIELDS
            if getattr(self, field) != getattr(other, field)
        )

    def caveats(self) -> tuple[str, ...]:
        """Reasons to distrust timings taken here, in the report's own words."""
        out = []
        if 0 < self.cpu_count < 4:
            out.append(
                f"{self.cpu_count} cores: little room to keep a timed run off a busy core"
            )
        if self.load_average and self.cpu_count:
            if self.load_average[0] > self.cpu_count / 2:
                out.append(
                    f"load average {self.load_average[0]} on {self.cpu_count} cores when "
                    "the run started"
                )
        return tuple(out)


@dataclass(frozen=True)
class Segment:
    """One measurement session's contribution to a record.

    A resumed run is measured over more than one session, so the environment and
    the clock belong to the session rather than to the record. ``problem_ids`` is
    what this session measured, which is what makes every scored problem
    attributable to a machine. See docs/decisions/0009-resume.md.
    """

    started: str
    finished: str
    environment: Environment
    problem_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_ids", tuple(sorted(int(i) for i in self.problem_ids)))
        if len(set(self.problem_ids)) != len(self.problem_ids):
            raise ValueError(f"segment repeats a problem id: {self.problem_ids}")

@dataclass(frozen=True)
class SampleRecord:
    """One sample: whether it was correct, and its per-level worst-case times.

    ``level_times`` covers the timed levels only, since level 0 contributes a
    verdict and not a time; ``statuses`` covers every level, level 0 first.
    """

    index: int
    correct: bool
    level_times: tuple[tuple[float, ...], ...]
    statuses: tuple[str, ...] = ()
    verified_levels: tuple[int, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "level_times", _freeze_times(self.level_times, what=f"sample {self.index}")
        )
        object.__setattr__(self, "statuses", tuple(str(s) for s in self.statuses))
        object.__setattr__(self, "verified_levels", tuple(int(v) for v in self.verified_levels))
        if self.index < 0:
            raise ValueError(f"sample index must be >= 0, got {self.index}")

    @property
    def censored(self) -> bool:
        return any(math.isinf(t) for level in self.level_times for t in level)

    @classmethod
    def from_measurement(
        cls, index: int, measurement: SolutionMeasurement, n_levels: int
    ) -> SampleRecord:
        """Keep what scoring needs and what a reader needs to explain a 0."""
        ordered = sorted(measurement.levels, key=lambda lvl: lvl.level)
        return cls(
            index=index,
            correct=measurement.correct,
            level_times=measurement.timed_times(n_levels),
            statuses=tuple(lvl.status for lvl in ordered),
            verified_levels=measurement.verified_levels,
            detail=measurement.detail,
        )

@dataclass(frozen=True)
class ProblemRecord:
    """One problem, and the single reference run every model was scored against.

    ``reference_times`` keeps level 0 as well, because the oracle is cheap to
    store and its filter times are worth auditing later; ``time_limit`` is the
    ``T_i`` the runner actually enforced.
    """

    problem_id: int
    reference_times: tuple[tuple[float, ...], ...]
    time_limit: float
    samples: Mapping[str, tuple[SampleRecord, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_times",
            _freeze_times(self.reference_times, what=f"problem {self.problem_id} reference"),
        )
        if self.problem_id < 0:
            raise ValueError(f"problem_id must be >= 0, got {self.problem_id}")
        if len(self.reference_times) < 2:
            raise ValueError(
                f"problem {self.problem_id}: need level 0 and at least one timed level"
            )
        if any(math.isinf(t) for level in self.reference_times for t in level):
            raise ValueError(
                f"problem {self.problem_id}: the reference cannot be censored; a reference "
                "that does not finish is bad data, not a slow one"
            )
        if not math.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError(f"problem {self.problem_id}: time limit {self.time_limit}")
        frozen = {str(model): tuple(records) for model, records in self.samples.items()}
        for model, records in frozen.items():
            if not records:
                raise ValueError(f"problem {self.problem_id}: model {model!r} has no samples")
            for record in records:
                if len(record.level_times) != self.n_timed_levels:
                    raise ValueError(
                        f"problem {self.problem_id} model {model!r} sample {record.index}: "
                        f"{len(record.level_times)} timed levels, reference has "
                        f"{self.n_timed_levels}"
                    )
        object.__setattr__(self, "samples", frozen)

    @property
    def n_timed_levels(self) -> int:
        return len(self.reference_times) - 1

    def timed_reference(self) -> tuple[tuple[float, ...], ...]:
        """``t*[i,l,m]`` for the scored levels, which is what Eq. (1) needs."""
        return self.reference_times[1:]

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self.samples))

@dataclass(frozen=True)
class RunRecord:
    """Everything one run measured, plus what it would take to reproduce it.

    Holds no scores. ``failures`` names problems whose reference did not run, so
    a number computed over fewer problems can be seen for what it is.
    ``started``, ``finished`` and ``environment`` describe the session the run
    began in; ``segments`` describes each session separately, which for a resumed
    run is the only place the second machine is named. ``attempted`` is what the
    run set out to measure, which is what makes an interrupted record
    distinguishable from a finished one.
    """

    started: str
    finished: str
    environment: Environment
    metric: MetricConfig
    repeats: int
    aggregator: str
    data: Provenance
    data_fingerprint: str
    solutions: Provenance
    solutions_fingerprint: str
    problems: tuple[ProblemRecord, ...]
    failures: tuple[tuple[int, str], ...] = ()
    segments: tuple[Segment, ...] = ()
    attempted: tuple[int, ...] = ()
    schema_version: int = RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.problems, key=lambda p: p.problem_id))
        object.__setattr__(self, "problems", ordered)
        object.__setattr__(
            self, "failures", tuple((int(pid), str(why)) for pid, why in self.failures)
        )
        ids = [p.problem_id for p in ordered]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate problem ids: {dupes}")
        if not ordered and not self.failures:
            raise ValueError("a run record needs at least one problem or one failure")
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")
        self._check_attempted(ids)
        self._check_segments(ids)
        for problem in ordered:
            if problem.n_timed_levels != self.metric.n_levels:
                raise ValueError(
                    f"problem {problem.problem_id} has {problem.n_timed_levels} timed levels, "
                    f"the run's metric declares {self.metric.n_levels}"
                )
            expected = self.metric.alpha * max(max(l) for l in problem.timed_reference())
            if not math.isclose(problem.time_limit, expected, rel_tol=1e-9):
                raise ValueError(
                    f"problem {problem.problem_id}: stored time limit {problem.time_limit} is "
                    f"not alpha * worst reference case ({expected})"
                )

    def _check_attempted(self, ids: Sequence[int]) -> None:
        """Default to "attempted exactly what it holds", and never claim fewer.

        A checkpoint of an interrupted run is otherwise indistinguishable from a
        finished one. See docs/decisions/0010-checkpointing.md.
        """
        accounted = set(ids) | {pid for pid, _ in self.failures}
        attempted = sorted({int(pid) for pid in self.attempted}) or sorted(accounted)
        object.__setattr__(self, "attempted", tuple(attempted))
        unclaimed = sorted(accounted - set(attempted))
        if unclaimed:
            raise ValueError(f"record holds problems it never attempted: {unclaimed}")

    def _check_segments(self, ids: Sequence[int]) -> None:
        """Default a single-session run to one segment; otherwise account for every id.

        Every scored problem has to belong to exactly one session, since that is
        what says which machine measured it.
        """
        if not self.segments:
            object.__setattr__(
                self,
                "segments",
                (
                    Segment(
                        started=self.started,
                        finished=self.finished,
                        environment=self.environment,
                        problem_ids=tuple(ids),
                    ),
                ),
            )
            return
        chronological = tuple(sorted(self.segments, key=lambda s: s.started))
        object.__setattr__(self, "segments", chronological)
        claimed: list[int] = [pid for segment in chronological for pid in segment.problem_ids]
        if len(set(claimed)) != len(claimed):
            dupes = sorted({i for i in claimed if claimed.count(i) > 1})
            raise ValueError(f"problems measured by more than one segment: {dupes}")
        if set(claimed) != set(ids):
            unclaimed = sorted(set(ids) - set(claimed))
            unmeasured = sorted(set(claimed) - set(ids))
            raise ValueError(
                "segments must account for exactly the scored problems; "
                f"unattributed {unclaimed}, not in the record {unmeasured}"
            )

    @property
    def resumed(self) -> bool:
        return len(self.segments) > 1

    @property
    def complete(self) -> bool:
        """Whether every attempted problem has a measurement or a recorded failure."""
        return not self.missing()

    def missing(self) -> tuple[int, ...]:
        """Attempted problems with neither, which is what an interruption leaves."""
        accounted = set(self.ids()) | {pid for pid, _ in self.failures}
        return tuple(pid for pid in self.attempted if pid not in accounted)

    def drift(self) -> tuple[str, ...]:
        """How later sessions' machines differed from the first one's."""
        first = self.segments[0].environment
        out = []
        for segment in self.segments[1:]:
            for difference in first.differences(segment.environment):
                out.append(f"{segment.started}: {difference}")
        return tuple(out)

    def calibration_drift(self) -> tuple[tuple[str, Drift], ...]:
        """Each later session's probe against the first session's, where both exist.

        Compared against the first segment for the same reason :meth:`drift` is: the
        first session is the one the reference times were taken in, and every later
        session's numbers are read against those.
        """
        first = self.segments[0].environment.calibration
        if first is None:
            return ()
        return tuple(
            (segment.started, compare(first, later))
            for segment in self.segments[1:]
            for later in (segment.environment.calibration,)
            if later is not None and first.comparable(later)
        )

    def calibration_gaps(self) -> tuple[str, ...]:
        """Sessions whose drift against the first one could not be measured."""
        first = self.segments[0].environment.calibration
        if first is None:
            if any(s.environment.calibration is not None for s in self.segments[1:]):
                return ("the first session has no calibration probe to compare against",)
            return ()
        out = []
        for segment in self.segments[1:]:
            later = segment.environment.calibration
            if later is None:
                out.append(f"{segment.started}: no calibration probe")
            elif not first.comparable(later):
                out.append(
                    f"{segment.started}: calibration probe measures different work "
                    f"(v{first.version} {first.repeats}x{first.aggregator} over "
                    f"{first.replicates} replicates, then v{later.version} "
                    f"{later.repeats}x{later.aggregator} over {later.replicates})"
                )
        return tuple(out)

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self) -> Iterable[ProblemRecord]:
        return iter(self.problems)

    def __getitem__(self, problem_id: int) -> ProblemRecord:
        for problem in self.problems:
            if problem.problem_id == problem_id:
                return problem
        raise KeyError(problem_id)

    def ids(self) -> tuple[int, ...]:
        return tuple(p.problem_id for p in self.problems)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted({model for p in self.problems for model in p.samples}))

    def covered_ids(self, model: str) -> tuple[int, ...]:
        """Problems this model has samples for. Absent is not the same as failed."""
        return tuple(p.problem_id for p in self.problems if model in p.samples)

    def aligned_ids(self, models: Sequence[str] | None = None) -> tuple[int, ...]:
        """Problems every named model covers, which is what a paired test needs."""
        chosen = self.models if models is None else tuple(models)
        if not chosen:
            return ()
        shared = set(self.covered_ids(chosen[0]))
        for model in chosen[1:]:
            shared &= set(self.covered_ids(model))
        return tuple(sorted(shared))

    def sample_counts(self, model: str) -> tuple[int, ...]:
        """Distinct ``n`` across this model's problems, sorted. More than one
        entry means ``eff@k`` averages estimators of differing variance."""
        return tuple(sorted({len(self[pid].samples[model]) for pid in self.covered_ids(model)}))

    def _ids_for(self, model: str, ids: Sequence[int] | None) -> tuple[int, ...]:
        covered = self.covered_ids(model)
        if ids is None:
            if not covered:
                raise KeyError(f"model {model!r} has no samples in this run")
            return covered
        missing = sorted(set(ids) - set(covered))
        if missing:
            raise KeyError(f"model {model!r} has no samples for problems {missing}")
        return tuple(sorted(ids))

    def sample_scores(
        self,
        model: str,
        problem_id: int,
        *,
        alpha: float | None = None,
        level_weights: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """``e_{i,j}`` for every sample, recomputed from the recorded times.

        ``alpha`` and ``level_weights`` default to the run's own. Raising
        ``alpha`` above the measured one is refused for censored samples, since
        their true time was never observed.
        """
        problem = self[problem_id]
        if model not in problem.samples:
            raise KeyError(f"model {model!r} has no samples for problem {problem_id}")
        records = problem.samples[model]
        new_alpha = self.metric.alpha if alpha is None else float(alpha)
        weights = (
            self.metric.level_weights
            if level_weights is None
            else tuple(float(w) for w in level_weights)
        )
        reference = problem.timed_reference()
        if self.metric.normalization != "global":
            if new_alpha != self.metric.alpha:
                raise ValueError(
                    f"cannot rescore a {self.metric.normalization!r} run at another alpha; "
                    "the censoring guard is defined for the paper's global normalization"
                )
            config = MetricConfig(new_alpha, weights, self.metric.normalization)
            return tuple(
                sample_score(r.level_times, reference, config, correct=r.correct) for r in records
            )
        return tuple(
            rescore_at_alpha(
                r.level_times,
                reference,
                new_alpha=new_alpha,
                measured_alpha=self.metric.alpha,
                level_weights=weights,
                correct=r.correct,
            )
            for r in records
        )

    def per_problem_eff(
        self,
        model: str,
        k: int = 1,
        *,
        ids: Sequence[int] | None = None,
        alpha: float | None = None,
        level_weights: Sequence[float] | None = None,
    ) -> tuple[float, ...]:
        """``eff_i@k`` per problem, in problem-id order."""
        out = []
        for pid in self._ids_for(model, ids):
            scores = self.sample_scores(
                model, pid, alpha=alpha, level_weights=level_weights
            )
            try:
                out.append(effk.eff_at_k(scores, k))
            except ValueError as exc:
                raise ValueError(f"model {model!r} problem {pid}: {exc}") from None
        return tuple(out)

    def eff_at_k(self, model: str, k: int = 1, **kwargs: Any) -> float:
        """``eff@k``: the unweighted mean of ``eff_i@k`` over problems."""
        return effk.mean_over_problems(self.per_problem_eff(model, k, **kwargs))

    def per_problem_pass(
        self, model: str, k: int = 1, *, ids: Sequence[int] | None = None
    ) -> tuple[float, ...]:
        out = []
        for pid in self._ids_for(model, ids):
            records = self[pid].samples[model]
            correct = sum(1 for r in records if r.correct)
            try:
                out.append(effk.pass_at_k(len(records), correct, k))
            except ValueError as exc:
                raise ValueError(f"model {model!r} problem {pid}: {exc}") from None
        return tuple(out)

    def pass_at_k(self, model: str, k: int = 1, *, ids: Sequence[int] | None = None) -> float:
        return effk.mean_over_problems(self.per_problem_pass(model, k, ids=ids))

    def level_means(
        self,
        model: str,
        *,
        ids: Sequence[int] | None = None,
        alpha: float | None = None,
    ) -> tuple[float, ...]:
        """``F_l``: mean level fraction per timed level, incorrect samples at 0.

        These are the numbers the ``h`` analysis works on, so they are computed
        by scoring with one-hot weights rather than by a second formula:
        ``eff_at_h(level_means(m), h)`` then equals ``eff@1`` at that ``h``
        exactly.
        """
        chosen = self._ids_for(model, ids)
        means = []
        for level in range(self.metric.n_levels):
            weights = tuple(
                1.0 if index == level else 0.0 for index in range(self.metric.n_levels)
            )
            per_problem = []
            for pid in chosen:
                scores = self.sample_scores(model, pid, alpha=alpha, level_weights=weights)
                per_problem.append(sum(scores) / len(scores))
            means.append(effk.mean_over_problems(per_problem))
        return tuple(means)

    def censored_samples(self, model: str) -> int:
        return sum(
            1 for pid in self.covered_ids(model) for r in self[pid].samples[model] if r.censored
        )

    def incorrect_samples(self, model: str) -> int:
        return sum(
            1
            for pid in self.covered_ids(model)
            for r in self[pid].samples[model]
            if not r.correct
        )

    def caveats(self) -> tuple[str, ...]:
        """Everything that should be read alongside this run's numbers."""
        out = list(self.environment.caveats())
        if not self.complete:
            out.append(
                f"interrupted: {len(self.missing())} of {len(self.attempted)} attempted "
                "problems were never measured, so every mean here is over a prefix"
            )
        if self.repeats < PAPER_REPEATS:
            out.append(f"{self.repeats} repeats per case, below the paper's R = {PAPER_REPEATS}")
        if self.failures:
            out.append(
                f"{len(self.failures)} problems produced no reference measurement and are "
                "absent from every score"
            )
        counts = sorted({n for model in self.models for n in self.sample_counts(model)})
        if len(counts) > 1:
            out.append(f"n varies across problems or models: {counts}")
        if self.resumed:
            out.append(
                f"measured over {len(self.segments)} sessions: "
                + ", ".join(
                    f"{len(s.problem_ids)} from {s.started}" for s in self.segments
                )
            )
            out += [f"machine changed between sessions, {d}" for d in self.drift()]
        out += self._calibration_caveats()
        return tuple(out)

    def _calibration_caveats(self) -> list[str]:
        """What the probe can and cannot say, which is not the same question."""
        first = self.segments[0].environment.calibration
        out = []
        if first is not None and not first.resolves_parity():
            out.append(
                f"calibration probe resolves a differential only to "
                f"{first.resolution():.3f}, coarser than the {DRIFT_CAVEAT} a parity "
                "tolerance needs, so drift below that is undetectable on this machine"
            )
        for started, drift in self.calibration_drift():
            if drift.caveat:
                out.append(
                    f"{started}: relative speed drifted by {drift.factor:.3f}, past the "
                    f"{drift.caveat_at:.3f} these probes can resolve; overall speed "
                    f"changed by {drift.uniform:.3f}, which cancels in Eq. (1) and this "
                    "does not"
                )
        out += [f"drift unmeasured, {gap}" for gap in self.calibration_gaps()]
        return out

def _sample_to_json(record: SampleRecord) -> dict[str, Any]:
    out: dict[str, Any] = {
        "index": record.index,
        "correct": record.correct,
        "level_times": _times_to_json(record.level_times),
        "statuses": list(record.statuses),
        "verified_levels": list(record.verified_levels),
    }
    if record.detail:
        out["detail"] = record.detail
    return out


def _sample_from_json(raw: Mapping[str, Any]) -> SampleRecord:
    return SampleRecord(
        index=int(raw["index"]),
        correct=bool(raw["correct"]),
        level_times=_times_from_json(raw["level_times"]),
        statuses=tuple(raw.get("statuses", ())),
        verified_levels=tuple(int(v) for v in raw.get("verified_levels", ())),
        detail=raw.get("detail", ""),
    )


def _problem_to_json(problem: ProblemRecord) -> dict[str, Any]:
    return {
        "problem_id": problem.problem_id,
        "reference_times": _times_to_json(problem.reference_times),
        "time_limit": problem.time_limit,
        "samples": {
            model: [_sample_to_json(r) for r in problem.samples[model]]
            for model in problem.models
        },
    }


def _problem_from_json(raw: Mapping[str, Any]) -> ProblemRecord:
    return ProblemRecord(
        problem_id=int(raw["problem_id"]),
        reference_times=_times_from_json(raw["reference_times"]),
        time_limit=float(raw["time_limit"]),
        samples={
            model: tuple(_sample_from_json(r) for r in records)
            for model, records in raw["samples"].items()
        },
    )

def _calibration_to_json(calibration: Calibration) -> dict[str, Any]:
    return {
        "version": calibration.version,
        "repeats": calibration.repeats,
        "aggregator": calibration.aggregator,
        "times": {name: list(series) for name, series in calibration.times.items()},
    }


def _calibration_from_json(raw: Mapping[str, Any]) -> Calibration:
    return Calibration(
        times={
            str(name): tuple(_check_time(float(t), what=f"calibration {name}") for t in series)
            for name, series in raw["times"].items()
        },
        repeats=int(raw["repeats"]),
        aggregator=str(raw["aggregator"]),
        version=int(raw["version"]),
    )


def _environment_to_json(environment: Environment) -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": environment.python,
        "platform": environment.platform,
        "machine": environment.machine,
        "cpu_count": environment.cpu_count,
        "load_average": (
            list(environment.load_average) if environment.load_average is not None else None
        ),
    }
    if environment.calibration is not None:
        out["calibration"] = _calibration_to_json(environment.calibration)
    return out


def _environment_from_json(raw: Mapping[str, Any]) -> Environment:
    load = raw.get("load_average")
    calibration = raw.get("calibration")
    return Environment(
        python=raw["python"],
        platform=raw["platform"],
        machine=raw["machine"],
        cpu_count=int(raw["cpu_count"]),
        load_average=tuple(float(x) for x in load) if load is not None else None,
        calibration=_calibration_from_json(calibration) if calibration else None,
    )


def _segment_to_json(segment: Segment) -> dict[str, Any]:
    return {
        "started": segment.started,
        "finished": segment.finished,
        "environment": _environment_to_json(segment.environment),
        "problem_ids": list(segment.problem_ids),
    }


def _segment_from_json(raw: Mapping[str, Any]) -> Segment:
    return Segment(
        started=raw["started"],
        finished=raw["finished"],
        environment=_environment_from_json(raw["environment"]),
        problem_ids=tuple(int(i) for i in raw.get("problem_ids", ())),
    )


def record_to_json(record: RunRecord) -> str:
    payload = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "started": record.started,
        "finished": record.finished,
        "environment": _environment_to_json(record.environment),
        "segments": [_segment_to_json(s) for s in record.segments],
        "attempted": list(record.attempted),
        "metric": {
            "alpha": record.metric.alpha,
            "level_weights": list(record.metric.level_weights),
            "normalization": record.metric.normalization,
        },
        "measurement": {"repeats": record.repeats, "aggregator": record.aggregator},
        "data": {
            "provenance": provenance_to_json(record.data),
            "fingerprint": record.data_fingerprint,
        },
        "solutions": {
            "provenance": provenance_to_json(record.solutions),
            "fingerprint": record.solutions_fingerprint,
        },
        "failures": [[pid, why] for pid, why in record.failures],
        "problems": [_problem_to_json(p) for p in record.problems],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def record_from_json(text: str) -> RunRecord:
    """Parse a run record. Validation lives in the dataclasses, so a file that
    disagrees with itself fails here rather than at scoring time."""
    raw = json.loads(text)
    version = raw.get("schema_version")
    if version != RECORD_SCHEMA_VERSION:
        raise ValueError(f"record schema version {version!r}, expected {RECORD_SCHEMA_VERSION}")
    metric = raw["metric"]
    measurement = raw["measurement"]
    return RunRecord(
        started=raw["started"],
        finished=raw["finished"],
        environment=_environment_from_json(raw["environment"]),
        metric=MetricConfig(
            alpha=float(metric["alpha"]),
            level_weights=tuple(float(w) for w in metric["level_weights"]),
            normalization=metric.get("normalization", "global"),
        ),
        repeats=int(measurement["repeats"]),
        aggregator=measurement["aggregator"],
        data=provenance_from_json(raw["data"]["provenance"]),
        data_fingerprint=raw["data"]["fingerprint"],
        solutions=provenance_from_json(raw["solutions"]["provenance"]),
        solutions_fingerprint=raw["solutions"]["fingerprint"],
        problems=tuple(_problem_from_json(p) for p in raw["problems"]),
        failures=tuple((int(pid), why) for pid, why in raw.get("failures", ())),
        segments=tuple(_segment_from_json(s) for s in raw.get("segments", ())),
        attempted=tuple(int(pid) for pid in raw.get("attempted", ())),
    )


def save_record(record: RunRecord, path: Path | str) -> Path:
    """Write atomically, so a crash cannot leave a half-written record.

    Resume overwrites the file it read, and that file may be the only copy of
    hours of measurement.
    """
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    scratch = location.with_name(location.name + ".partial")
    scratch.write_text(record_to_json(record))
    os.replace(scratch, location)
    return location


def load_record(path: Path | str) -> RunRecord:
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"no run record at {location}")
    return record_from_json(location.read_text())
