"""Benchmark data: problems, expert references, generators.

Fetched at setup time, never vendored; see README "Credit". Rationale in
docs/decisions/0003-data-adapter.md.
"""

from enamel_ext.data.cases import load_generator, materialize, materialize_level
from enamel_ext.data.schema import (
    PAPER_CASE_COUNTS,
    PAPER_PROBLEM_COUNT,
    UNKNOWN_LICENSE,
    GeneratedLevel,
    Level,
    MaterializedLevel,
    Problem,
    ProblemSet,
    Provenance,
)
from enamel_ext.data.sources import (
    CACHE_ENV,
    SCHEMA_VERSION,
    JsonSource,
    ProblemSource,
    default_cache_dir,
    problem_from_record,
    problem_set_from_json,
    problem_set_to_json,
    problems_from_records,
    synthetic_problem_set,
)

__all__ = [
    "CACHE_ENV",
    "PAPER_CASE_COUNTS",
    "PAPER_PROBLEM_COUNT",
    "SCHEMA_VERSION",
    "UNKNOWN_LICENSE",
    "GeneratedLevel",
    "JsonSource",
    "Level",
    "MaterializedLevel",
    "Problem",
    "ProblemSet",
    "ProblemSource",
    "Provenance",
    "default_cache_dir",
    "load_generator",
    "materialize",
    "materialize_level",
    "problem_from_record",
    "problem_set_from_json",
    "problem_set_to_json",
    "problems_from_records",
    "synthetic_problem_set",
]
