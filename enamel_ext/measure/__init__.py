"""Measurement backends, sandboxed execution and repeat aggregation.

Everything that touches a clock or runs untrusted code lives here, so the
metric layer stays pure and the backend can be swapped for a deterministic one
without touching scoring.
"""

from enamel_ext.measure.runner import (
    PAPER_REPEATS,
    SKIPPED,
    WRONG_ANSWER,
    LevelMeasurement,
    ProblemEvaluation,
    ReferenceMeasurement,
    RunConfig,
    SolutionMeasurement,
    evaluate_problem,
    evaluate_solution,
    measure_reference,
    score_solution,
)
from enamel_ext.measure.sandbox import (
    CRASHED,
    ERROR,
    OK,
    TIMEOUT,
    CaseResult,
    LevelResult,
    Limits,
    SandboxError,
    run_level,
)
from enamel_ext.measure.timing import AGGREGATORS, aggregate_repeats, hodges_lehmann
from enamel_ext.measure.values import brief, decode, encode, values_equal

__all__ = [
    "AGGREGATORS",
    "CRASHED",
    "ERROR",
    "OK",
    "PAPER_REPEATS",
    "SKIPPED",
    "TIMEOUT",
    "WRONG_ANSWER",
    "CaseResult",
    "LevelMeasurement",
    "LevelResult",
    "Limits",
    "ProblemEvaluation",
    "ReferenceMeasurement",
    "RunConfig",
    "SandboxError",
    "SolutionMeasurement",
    "aggregate_repeats",
    "brief",
    "decode",
    "encode",
    "evaluate_problem",
    "evaluate_solution",
    "hodges_lehmann",
    "measure_reference",
    "run_level",
    "score_solution",
    "values_equal",
]
