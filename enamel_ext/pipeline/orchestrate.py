"""Problem-major evaluation: measure each reference once, score every model on it.

Iterating problems on the outside and models on the inside means one ``T_i`` per
problem enters every model's score, instead of each model carrying its own
reference noise. See docs/decisions/0006-run-record.md, 0009-resume.md for
extending an existing record, and 0010-checkpointing.md for saving one mid-run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from enamel_ext.data.schema import ProblemSet
from enamel_ext.measure.calibrate import Calibration, compare
from enamel_ext.measure.runner import RunConfig, evaluate_solution, measure_reference
from enamel_ext.measure.sandbox import SandboxError
from enamel_ext.metrics.score import PAPER, MetricConfig
from enamel_ext.pipeline.record import (
    RECORD_SCHEMA_VERSION,
    Environment,
    ProblemRecord,
    RunRecord,
    SampleRecord,
    Segment,
    save_record,
)
from enamel_ext.pipeline.solutions import SolutionSet

__all__ = ["resume_evaluation", "resume_mismatches", "run_evaluation", "selected_ids"]

Progress = Callable[[str], None]
Measured = tuple[Sequence[ProblemRecord], Sequence[tuple[int, str]]]
Compose = Callable[..., RunRecord]
Checkpoint = Callable[..., None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def selected_ids(
    problems: ProblemSet,
    solutions: SolutionSet,
    models: Sequence[str] | None = None,
    ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Problems to attempt: those in the data that at least one model answered."""
    chosen = tuple(models) if models is not None else solutions.models
    unknown = sorted(set(chosen) - set(solutions.models))
    if unknown:
        raise KeyError(f"no samples for models {unknown}")
    answered = {pid for model in chosen for pid in solutions.problem_ids(model)}
    available = set(problems.ids())
    if ids is not None:
        requested = {int(i) for i in ids}
        missing = sorted(requested - available)
        if missing:
            raise KeyError(f"problems {missing} are not in this problem set")
        available &= requested
    return tuple(sorted(available & answered))


def _checkpointer(compose: Compose, path: Path | None, every: int) -> Checkpoint | None:
    """Save the record-so-far every ``every`` finished problems, or never.

    Returns ``None`` when checkpointing is off, which is how ``_measure`` skips the
    cost entirely. See docs/decisions/0010-checkpointing.md.
    """
    if path is None or every <= 0:
        return None

    finished = 0

    def checkpoint(
        records: Sequence[ProblemRecord], failures: Sequence[tuple[int, str]]
    ) -> None:
        nonlocal finished
        finished += 1
        if finished % every == 0:
            save_record(compose(records, failures), path)

    return checkpoint


def _measure(
    problems: ProblemSet,
    solutions: SolutionSet,
    attempt: Sequence[int],
    chosen: Sequence[str],
    config: RunConfig,
    metric: MetricConfig,
    keep_going: bool,
    on_progress: Progress | None,
    on_checkpoint: Checkpoint | None = None,
) -> tuple[list[ProblemRecord], list[tuple[int, str]]]:
    """One measurement session over ``attempt``. Computes no scores."""
    records: list[ProblemRecord] = []
    failures: list[tuple[int, str]] = []
    for pid in attempt:
        problem = problems[pid]
        if on_progress is not None:
            on_progress(f"problem {pid}: reference")
        try:
            reference = measure_reference(problem, config, metric)
        except SandboxError as exc:
            if not keep_going:
                raise
            failures.append((pid, str(exc)))
            if on_progress is not None:
                on_progress(f"problem {pid}: reference failed, skipping")
            if on_checkpoint is not None:
                on_checkpoint(records, failures)
            continue

        by_model = {
            model: solutions.codes(model, pid)
            for model in chosen
            if solutions.codes(model, pid)
        }
        if on_progress is not None:
            total = sum(len(c) for c in by_model.values())
            on_progress(f"problem {pid}: {total} samples across {len(by_model)} models")

        # Interleaved by sample index so that no model's samples sit systematically
        # further from the reference run than another's.
        measured: dict[str, list[SampleRecord]] = {model: [] for model in by_model}
        for index in range(max(len(c) for c in by_model.values())):
            for model, codes in by_model.items():
                if index >= len(codes):
                    continue
                measured[model].append(
                    SampleRecord.from_measurement(
                        index,
                        evaluate_solution(problem, codes[index], reference, config),
                        metric.n_levels,
                    )
                )
        records.append(
            ProblemRecord(
                problem_id=pid,
                reference_times=reference.times,
                time_limit=reference.time_limit,
                samples={model: tuple(rs) for model, rs in measured.items()},
            )
        )
        if on_checkpoint is not None:
            on_checkpoint(records, failures)
    return records, failures


def run_evaluation(
    problems: ProblemSet,
    solutions: SolutionSet,
    *,
    config: RunConfig = RunConfig(),
    metric: MetricConfig = PAPER,
    models: Sequence[str] | None = None,
    ids: Sequence[int] | None = None,
    keep_going: bool = False,
    on_progress: Progress | None = None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 1,
    calibration: Calibration | None = None,
) -> RunRecord:
    """Measure and record a full run. Computes no scores.

    A problem whose reference fails is bad data rather than a low score: it stops
    the run, or with ``keep_going`` is recorded in ``failures`` and left out of
    every model's average. Given ``checkpoint``, the record-so-far is written there
    as the run goes, so an interruption leaves a record ``resume_evaluation`` takes.
    ``calibration`` is a probe the caller timed; without one, a later session's
    comparability rests on environment strings alone.
    """
    chosen = tuple(models) if models is not None else solutions.models
    if not chosen:
        raise ValueError("no models to evaluate")
    attempt = selected_ids(problems, solutions, chosen, ids)

    started = _now()
    environment = Environment.capture(calibration)
    data_fingerprint = problems.fingerprint()
    solutions_fingerprint = solutions.fingerprint()

    def compose(
        records: Sequence[ProblemRecord], failures: Sequence[tuple[int, str]]
    ) -> RunRecord:
        finished = _now()
        return RunRecord(
            started=started,
            finished=finished,
            environment=environment,
            metric=metric,
            repeats=config.repeats,
            aggregator=config.aggregator,
            data=problems.provenance,
            data_fingerprint=data_fingerprint,
            solutions=solutions.provenance,
            solutions_fingerprint=solutions_fingerprint,
            problems=tuple(records),
            failures=tuple(failures),
            segments=(
                Segment(
                    started=started,
                    finished=finished,
                    environment=environment,
                    problem_ids=tuple(p.problem_id for p in records),
                ),
            ),
            attempted=tuple(attempt),
        )

    records, failures = _measure(
        problems,
        solutions,
        attempt,
        chosen,
        config,
        metric,
        keep_going,
        on_progress,
        _checkpointer(compose, checkpoint, checkpoint_every),
    )
    if not records and not failures:
        raise ValueError("no problem was both present in the data and answered by a model")
    return compose(records, failures)


def resume_mismatches(
    record: RunRecord,
    problems: ProblemSet,
    solutions: SolutionSet,
    *,
    config: RunConfig = RunConfig(),
    metric: MetricConfig = PAPER,
    models: Sequence[str] | None = None,
    ids: Sequence[int] | None = None,
    environment: Environment | None = None,
) -> tuple[str, ...]:
    """Reasons this session cannot extend ``record``, all of them at once.

    Everything that would make the two halves different measurements of different
    things: the metric, the measurement settings, the bytes, the selection, and
    the machine, including a measured change in relative speed where both sessions
    probed for one. Rationale and what is deliberately not here in decision 0009.
    """
    out = []
    here = Environment.capture() if environment is None else environment
    if record.schema_version != RECORD_SCHEMA_VERSION:
        out.append(
            f"record schema {record.schema_version}, this build writes {RECORD_SCHEMA_VERSION}"
        )
    if record.metric != metric:
        out.append(f"metric: {record.metric} then {metric}")
    if record.repeats != config.repeats:
        out.append(f"repeats: {record.repeats} then {config.repeats}")
    if record.aggregator != config.aggregator:
        out.append(f"aggregator: {record.aggregator!r} then {config.aggregator!r}")
    if record.data_fingerprint != problems.fingerprint():
        out.append("problem set fingerprint differs: the references are not the same code")
    if record.solutions_fingerprint != solutions.fingerprint():
        out.append("solution set fingerprint differs: the samples are not the same code")
    out += [
        f"machine {difference}"
        for difference in record.environment.differences(here)
    ]
    first = record.environment.calibration
    later = here.calibration
    if first is not None and later is not None and first.comparable(later):
        drift = compare(first, later)
        if drift.refuse:
            out.append(
                f"relative speed drifted by {drift.factor:.3f}, past the "
                f"{drift.refuse_at:.3f} these calibration probes can resolve; overall "
                f"speed changed by {drift.uniform:.3f}, which cancels in Eq. (1) and "
                "this does not"
            )

    chosen = tuple(models) if models is not None else solutions.models
    try:
        attempt = set(selected_ids(problems, solutions, chosen, ids))
    except KeyError as exc:
        return tuple(out) + (str(exc.args[0]),)
    outside = sorted(set(record.ids()) - attempt)
    if outside:
        out.append(f"this selection would not attempt problems already measured: {outside}")

    dropped = sorted(set(record.models) - set(chosen))
    if dropped:
        out.append(f"models measured but not requested now: {dropped}")
    late = [
        model
        for model in sorted(set(chosen) - set(record.models))
        if any(solutions.codes(model, pid) for pid in record.ids())
    ]
    if late:
        out.append(
            f"models with samples for already-measured problems that the record does not "
            f"cover: {late}"
        )
    return tuple(out)


def resume_evaluation(
    record: RunRecord,
    problems: ProblemSet,
    solutions: SolutionSet,
    *,
    config: RunConfig = RunConfig(),
    metric: MetricConfig = PAPER,
    models: Sequence[str] | None = None,
    ids: Sequence[int] | None = None,
    keep_going: bool = False,
    on_progress: Progress | None = None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 1,
    calibration: Calibration | None = None,
) -> RunRecord:
    """Measure what ``record`` is missing and return the two sessions as one record.

    A measured problem is never re-measured; a recorded failure is retried, since
    a reference that did not run may have lost a race rather than be unrunnable.
    Returns ``record`` unchanged when there is nothing left.
    """
    chosen = tuple(models) if models is not None else solutions.models
    if not chosen:
        raise ValueError("no models to evaluate")
    environment = Environment.capture(calibration)
    mismatches = resume_mismatches(
        record,
        problems,
        solutions,
        config=config,
        metric=metric,
        models=models,
        ids=ids,
        environment=environment,
    )
    if mismatches:
        raise ValueError("cannot resume this record:\n  " + "\n  ".join(mismatches))

    attempt = tuple(
        pid for pid in selected_ids(problems, solutions, chosen, ids) if pid not in record.ids()
    )
    if not attempt:
        if on_progress is not None:
            on_progress(f"nothing left: all {len(record)} problems are already measured")
        return record

    started = _now()

    def compose(
        records: Sequence[ProblemRecord], failures: Sequence[tuple[int, str]]
    ) -> RunRecord:
        finished = _now()
        # A prior failure is only superseded once this session reaches that problem,
        # so a checkpoint taken mid-retry still records it.
        handled = {p.problem_id for p in records} | {pid for pid, _ in failures}
        return RunRecord(
            started=record.started,
            finished=finished,
            environment=record.environment,
            metric=record.metric,
            repeats=record.repeats,
            aggregator=record.aggregator,
            data=record.data,
            data_fingerprint=record.data_fingerprint,
            solutions=record.solutions,
            solutions_fingerprint=record.solutions_fingerprint,
            problems=record.problems + tuple(records),
            failures=tuple((pid, why) for pid, why in record.failures if pid not in handled)
            + tuple(failures),
            segments=record.segments
            + (
                Segment(
                    started=started,
                    finished=finished,
                    environment=environment,
                    problem_ids=tuple(p.problem_id for p in records),
                ),
            ),
            attempted=tuple(sorted(set(record.attempted) | set(attempt))),
        )

    records, failures = _measure(
        problems,
        solutions,
        attempt,
        chosen,
        config,
        metric,
        keep_going,
        on_progress,
        _checkpointer(compose, checkpoint, checkpoint_every),
    )
    return compose(records, failures)
