"""Turning level specs into concrete argument tuples.

Executes the benchmark's own generator source, so only run this on data whose
provenance is recorded. Model-generated code never goes through here; it runs in
the sandbox in :mod:`enamel_ext.measure`.
"""

from __future__ import annotations

from typing import Any, Callable

from enamel_ext.data.schema import Level, MaterializedLevel, Problem

__all__ = ["load_generator", "materialize_level", "materialize"]

GENERATOR_ENTRY = "make_input"


def load_generator(source: str) -> Callable[[int, int], tuple[Any, ...]]:
    """Compile generator source and return its ``make_input(seed, scale)``."""
    namespace: dict[str, Any] = {}
    exec(compile(source, "<generator>", "exec"), namespace)
    fn = namespace.get(GENERATOR_ENTRY)
    if not callable(fn):
        raise ValueError(f"generator does not define {GENERATOR_ENTRY}(seed, scale)")
    return fn


def materialize_level(problem: Problem, level: Level) -> tuple[tuple[Any, ...], ...]:
    """Argument tuples for one level, in a fixed order."""
    if isinstance(level, MaterializedLevel):
        return level.inputs
    fn = load_generator(problem.input_generator)
    cases = []
    for seed in level.seeds:
        args = fn(seed, level.scale)
        if not isinstance(args, tuple):
            raise ValueError(
                f"problem {problem.problem_id} level {level.level}: generator returned "
                f"{type(args).__name__}, expected a tuple of positional arguments"
            )
        cases.append(args)
    return tuple(cases)


def materialize(problem: Problem) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    """All levels, indexed by level."""
    return tuple(materialize_level(problem, lvl) for lvl in problem.levels)
