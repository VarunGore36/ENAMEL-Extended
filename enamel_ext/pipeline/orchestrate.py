"""Problem-major evaluation: measure each reference once, score every model on it.

Iterating problems on the outside and models on the inside means one ``T_i`` per
problem enters every model's score, instead of each model carrying its own
reference noise. See docs/decisions/0006-run-record.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Sequence

from enamel_ext.data.schema import ProblemSet
from enamel_ext.measure.runner import RunConfig, evaluate_solution, measure_reference
from enamel_ext.measure.sandbox import SandboxError
from enamel_ext.metrics.score import PAPER, MetricConfig
from enamel_ext.pipeline.record import Environment, ProblemRecord, RunRecord, SampleRecord
from enamel_ext.pipeline.solutions import SolutionSet

__all__ = ["run_evaluation", "selected_ids"]

Progress = Callable[[str], None]


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
) -> RunRecord:
    """Measure and record a full run. Computes no scores.

    A problem whose reference fails is bad data rather than a low score: it stops
    the run, or with ``keep_going`` is recorded in ``failures`` and left out of
    every model's average.
    """
    chosen = tuple(models) if models is not None else solutions.models
    if not chosen:
        raise ValueError("no models to evaluate")
    attempt = selected_ids(problems, solutions, chosen, ids)

    started = _now()
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

    if not records and not failures:
        raise ValueError("no problem was both present in the data and answered by a model")
    return RunRecord(
        started=started,
        finished=_now(),
        environment=Environment.capture(),
        metric=metric,
        repeats=config.repeats,
        aggregator=config.aggregator,
        data=problems.provenance,
        data_fingerprint=problems.fingerprint(),
        solutions=solutions.provenance,
        solutions_fingerprint=solutions.fingerprint(),
        problems=tuple(records),
        failures=tuple(failures),
    )
