"""Level-by-level evaluation of one solution against one problem.

Measures the expert reference first, because the time limit ``T_i`` and the
expected outputs both come from it, then runs the candidate under that limit.
Rationale in docs/decisions/0004-sandboxed-runner.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from enamel_ext.data.schema import GeneratedLevel, Level, MaterializedLevel, Problem
from enamel_ext.measure.sandbox import (
    CRASHED,
    ERROR,
    OK,
    TIMEOUT,
    LevelResult,
    Limits,
    SandboxError,
    run_level,
)
from enamel_ext.measure.timing import aggregate_repeats
from enamel_ext.measure.values import ABS_TOL, REL_TOL, brief, values_equal
from enamel_ext.metrics.score import PAPER, TIMEOUT as CENSORED, MetricConfig, sample_score

__all__ = [
    "PAPER_REPEATS",
    "SKIPPED",
    "WRONG_ANSWER",
    "LevelMeasurement",
    "ProblemEvaluation",
    "ReferenceMeasurement",
    "RunConfig",
    "SolutionMeasurement",
    "evaluate_problem",
    "evaluate_solution",
    "measure_reference",
    "score_solution",
]

#: R in the paper.
PAPER_REPEATS = 6

SKIPPED = "skipped"
WRONG_ANSWER = "wrong_answer"


@dataclass(frozen=True)
class RunConfig:
    """How to measure, as opposed to how to score.

    ``reference_wall_timeout`` and ``filter_wall_timeout`` cap the two runs that
    have no ``T_i`` to work from: the reference, and level 0.
    """

    repeats: int = PAPER_REPEATS
    aggregator: str = "hodges_lehmann"
    limits: Limits = Limits()
    rel_tol: float = REL_TOL
    abs_tol: float = ABS_TOL
    reference_wall_timeout: float | None = None
    filter_wall_timeout: float | None = None

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {self.repeats}")


@dataclass(frozen=True)
class LevelMeasurement:
    level: int
    status: str
    times: tuple[float, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class ReferenceMeasurement:
    """The oracle: per-case reference times, expected outputs, and ``T_i``."""

    problem_id: int
    times: tuple[tuple[float, ...], ...]
    outputs: tuple[tuple[Any, ...], ...]
    time_limit: float

    def timed(self) -> tuple[tuple[float, ...], ...]:
        return self.times[1:]


@dataclass(frozen=True)
class SolutionMeasurement:
    """One sample's per-level record. ``verified_levels`` are the levels whose
    outputs were actually compared, which a timeout can cut short."""

    problem_id: int
    correct: bool
    levels: tuple[LevelMeasurement, ...]
    detail: str = ""
    verified_levels: tuple[int, ...] = ()

    def timed_times(self, n_levels: int) -> tuple[tuple[float, ...], ...]:
        """Per-level worst cases for :func:`sample_score`, censoring what did
        not finish. Levels never reached are censored, not inferred."""
        out = []
        for index in range(1, n_levels + 1):
            found = [lvl for lvl in self.levels if lvl.level == index]
            if found and found[0].status == OK and found[0].times:
                out.append(found[0].times)
            else:
                out.append((CENSORED,))
        return tuple(out)


def _level_kwargs(problem: Problem, level: Level) -> dict[str, Any]:
    if isinstance(level, MaterializedLevel):
        return {"inputs": level.inputs}
    if isinstance(level, GeneratedLevel):
        return {
            "generator": problem.input_generator,
            "scale": level.scale,
            "seeds": level.seeds,
        }
    raise SandboxError(f"unknown level kind {type(level).__name__}")


def _aggregate(result: LevelResult, method: str) -> tuple[float, ...]:
    return tuple(aggregate_repeats(case.times, method) for case in result.cases)


def measure_reference(
    problem: Problem,
    config: RunConfig = RunConfig(),
    metric: MetricConfig = PAPER,
) -> ReferenceMeasurement:
    """Time the expert reference on every level and derive ``T_i``.

    Raises :class:`SandboxError` if the reference does not run: it is the
    oracle, so a failure there is bad data, not a low score.
    """
    if problem.n_timed_levels != metric.n_levels:
        raise SandboxError(
            f"problem {problem.problem_id} has {problem.n_timed_levels} timed levels, "
            f"metric config declares {metric.n_levels}"
        )
    times: list[tuple[float, ...]] = []
    outputs: list[tuple[Any, ...]] = []
    for level in problem.levels:
        result = run_level(
            problem.reference_solution,
            problem.entry_point,
            repeats=config.repeats,
            aggregator=config.aggregator,
            limits=config.limits,
            wall_timeout=config.reference_wall_timeout,
            **_level_kwargs(problem, level),
        )
        if not result.ok:
            raise SandboxError(
                f"problem {problem.problem_id} reference failed on level {level.level}: "
                f"{result.status} {result.detail} {result.stderr}".strip()
            )
        times.append(_aggregate(result, config.aggregator))
        outputs.append(tuple(case.output for case in result.cases))

    scored = tuple(times[1:])
    worst = max(max(level) for level in scored)
    if not worst > 0:
        raise SandboxError(
            f"problem {problem.problem_id} reference is too fast to time: worst case {worst}"
        )
    return ReferenceMeasurement(
        problem_id=problem.problem_id,
        times=tuple(times),
        outputs=tuple(outputs),
        time_limit=metric.alpha * worst,
    )


def _wrong_case(
    result: LevelResult, expected: Sequence[Any], config: RunConfig, *, complete: bool
) -> tuple[int, str] | None:
    """First case whose output disagrees with the reference, if any.

    ``complete`` demands one output per reference case; a level that stopped
    early is judged only on the cases it finished.
    """
    for index, case in enumerate(result.cases):
        if index >= len(expected):
            return index, "more cases ran than the reference produced"
        if not case.has_output:
            if complete:
                return index, "no output captured"
            continue
        if not values_equal(case.output, expected[index], config.rel_tol, config.abs_tol):
            return index, f"expected {brief(expected[index])}, got {brief(case.output)}"
    if complete and len(result.cases) < len(expected):
        return len(result.cases), "fewer cases ran than the reference produced"
    return None


def evaluate_solution(
    problem: Problem,
    code: str,
    reference: ReferenceMeasurement,
    config: RunConfig = RunConfig(),
) -> SolutionMeasurement:
    """Run ``code`` level by level under the reference's time limit.

    Stops at the first level that errors, disagrees with the reference, or
    reaches the limit; later levels are reported as skipped and score as
    censored. Level 0 runs without the limit, because it decides correctness and
    a solution that is merely slow there is not a wrong answer.
    """
    if reference.problem_id != problem.problem_id:
        raise SandboxError(
            f"reference is for problem {reference.problem_id}, not {problem.problem_id}"
        )
    levels: list[LevelMeasurement] = []
    verified: list[int] = []
    correct = True
    detail = ""

    for level in problem.levels:
        timed = level.level > 0
        result = run_level(
            code,
            problem.entry_point,
            repeats=config.repeats if timed else 1,
            time_limit=reference.time_limit if timed else None,
            aggregator=config.aggregator,
            limits=config.limits,
            wall_timeout=None if timed else config.filter_wall_timeout,
            **_level_kwargs(problem, level),
        )
        if result.status in (ERROR, CRASHED):
            correct = False
            detail = f"level {level.level}: {result.detail or result.stderr}".strip()
            levels.append(LevelMeasurement(level.level, result.status, detail=detail))
            break

        # A level that ran out of time still reports the cases it finished, and a
        # wrong answer among them outranks the timeout.
        wrong = _wrong_case(
            result, reference.outputs[level.level], config, complete=result.status == OK
        )
        if wrong is not None:
            index, why = wrong
            correct = False
            detail = f"level {level.level} case {index}: {why}"
            levels.append(LevelMeasurement(level.level, WRONG_ANSWER, detail=detail))
            break

        if result.status == TIMEOUT:
            if not timed:
                correct = False
                detail = f"level {level.level} did not finish, so correctness is unverified"
            levels.append(LevelMeasurement(level.level, TIMEOUT, detail=result.detail))
            break

        verified.append(level.level)
        times = _aggregate(result, config.aggregator)
        levels.append(LevelMeasurement(level.level, OK, times=times))
        if timed and times and max(times) >= reference.time_limit:
            detail = detail or f"level {level.level} reached the time limit"
            break

    reached = {lvl.level for lvl in levels}
    for level in problem.levels:
        if level.level not in reached:
            levels.append(LevelMeasurement(level.level, SKIPPED))
    levels.sort(key=lambda lvl: lvl.level)
    return SolutionMeasurement(
        problem.problem_id, correct, tuple(levels), detail, tuple(verified)
    )



def score_solution(
    measurement: SolutionMeasurement,
    reference: ReferenceMeasurement,
    metric: MetricConfig = PAPER,
) -> float:
    """``e_{i,j}`` for one measured solution."""
    return sample_score(
        measurement.timed_times(metric.n_levels),
        reference.timed(),
        metric,
        correct=measurement.correct,
    )


@dataclass(frozen=True)
class ProblemEvaluation:
    """All ``n`` samples of one problem, scored against one reference run."""

    problem_id: int
    reference: ReferenceMeasurement
    measurements: tuple[SolutionMeasurement, ...]
    scores: tuple[float, ...]

    @property
    def n_correct(self) -> int:
        return sum(1 for m in self.measurements if m.correct)


def evaluate_problem(
    problem: Problem,
    codes: Sequence[str],
    config: RunConfig = RunConfig(),
    metric: MetricConfig = PAPER,
) -> ProblemEvaluation:
    """Measure the reference once, then score every sample against it.

    One reference run serves all samples, as the paper's normalization intends;
    the cost is that machine drift within a problem is not cancelled.
    """
    reference = measure_reference(problem, config, metric)
    measurements = tuple(evaluate_solution(problem, code, reference, config) for code in codes)
    scores = tuple(score_solution(m, reference, metric) for m in measurements)
    return ProblemEvaluation(problem.problem_id, reference, measurements, scores)
