"""Problem, level and test-case schema. See docs/decisions/0003-data-adapter.md."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = [
    "PAPER_CASE_COUNTS",
    "PAPER_PROBLEM_COUNT",
    "UNKNOWN_LICENSE",
    "GeneratedLevel",
    "Level",
    "MaterializedLevel",
    "Problem",
    "ProblemSet",
    "Provenance",
]

#: M_0..M_3: level 0 filters correctness, levels 1-3 are timed.
PAPER_CASE_COUNTS = (8, 4, 4, 4)

#: 142 of HumanEval's 164 problems.
PAPER_PROBLEM_COUNT = 142

UNKNOWN_LICENSE = "unknown"


@dataclass(frozen=True)
class Provenance:
    """Where a problem set came from."""

    name: str
    url: str
    license: str
    retrieved: str

    def __post_init__(self) -> None:
        for name in ("name", "url", "license", "retrieved"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"provenance.{name} must not be empty")

    @property
    def redistributable(self) -> bool:
        """False until an actual license is recorded; gates vendoring."""
        return self.license != UNKNOWN_LICENSE


@dataclass(frozen=True)
class MaterializedLevel:
    """Concrete argument tuples, one per test case, as upstream ships them."""

    level: int
    inputs: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if self.level < 0:
            raise ValueError(f"level index must be >= 0, got {self.level}")
        if not self.inputs:
            raise ValueError(f"level {self.level} has no test cases")
        for i, args in enumerate(self.inputs):
            if not isinstance(args, tuple):
                raise ValueError(
                    f"level {self.level} case {i} is {type(args).__name__}, expected a "
                    f"tuple of positional arguments"
                )

    @property
    def n_cases(self) -> int:
        return len(self.inputs)


@dataclass(frozen=True)
class GeneratedLevel:
    """Test cases rebuilt from (seed, scale) by the problem's generator."""

    level: int
    scale: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        if self.level < 0:
            raise ValueError(f"level index must be >= 0, got {self.level}")
        if self.scale <= 0:
            raise ValueError(f"level {self.level} needs scale > 0, got {self.scale}")
        if not self.seeds:
            raise ValueError(f"level {self.level} has no test cases")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"level {self.level} repeats a seed, so cases collide")

    @property
    def n_cases(self) -> int:
        return len(self.seeds)


Level = MaterializedLevel | GeneratedLevel


@dataclass(frozen=True)
class Problem:
    """One problem: prompt, expert reference, and its test levels.

    The reference is also the oracle, so expected outputs are never stored.
    """

    problem_id: int
    entry_point: str
    prompt: str
    reference_solution: str
    levels: tuple[Level, ...]
    input_generator: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        if self.problem_id < 0:
            raise ValueError(f"problem_id must be >= 0, got {self.problem_id}")
        if not self.entry_point.isidentifier():
            raise ValueError(f"entry_point {self.entry_point!r} is not an identifier")
        if not self.reference_solution.strip():
            raise ValueError(f"problem {self.problem_id} has no reference solution")
        if not self.levels:
            raise ValueError(f"problem {self.problem_id} has no levels")
        want = list(range(len(self.levels)))
        got = [lvl.level for lvl in self.levels]
        if got != want:
            raise ValueError(f"problem {self.problem_id} levels must be {want}, got {got}")
        if any(isinstance(lvl, GeneratedLevel) for lvl in self.levels):
            if not self.input_generator.strip():
                raise ValueError(
                    f"problem {self.problem_id} has generated levels but no generator"
                )
        self._check_scales()

    def _check_scales(self) -> None:
        """Scale must grow across timed levels, or they cannot separate
        complexity classes and q is meaningless."""
        timed = [lvl for lvl in self.levels[1:] if isinstance(lvl, GeneratedLevel)]
        if len(timed) != len(self.levels) - 1:
            return
        scales = [lvl.scale for lvl in timed]
        if any(b <= a for a, b in zip(scales, scales[1:])):
            raise ValueError(
                f"problem {self.problem_id} timed level scales must increase, got {scales}"
            )

    @property
    def n_timed_levels(self) -> int:
        return len(self.levels) - 1

    @property
    def case_counts(self) -> tuple[int, ...]:
        return tuple(lvl.n_cases for lvl in self.levels)


def _canonical(problems: tuple[Problem, ...]) -> list[Any]:
    out = []
    for p in sorted(problems, key=lambda x: x.problem_id):
        levels: list[Any] = []
        for lvl in p.levels:
            if isinstance(lvl, GeneratedLevel):
                levels.append(["gen", lvl.level, lvl.scale, list(lvl.seeds)])
            else:
                levels.append(["mat", lvl.level, [repr(x) for x in lvl.inputs]])
        out.append(
            [
                p.problem_id,
                p.entry_point,
                p.prompt,
                p.reference_solution,
                p.input_generator,
                levels,
            ]
        )
    return out


@dataclass(frozen=True)
class ProblemSet:
    """A loaded benchmark, keyed by problem id rather than by position."""

    provenance: Provenance
    problems: tuple[Problem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "problems", tuple(self.problems))
        ids = [p.problem_id for p in self.problems]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate problem ids: {dupes}")

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self) -> Iterator[Problem]:
        return iter(self.problems)

    def __getitem__(self, problem_id: int) -> Problem:
        for p in self.problems:
            if p.problem_id == problem_id:
                return p
        raise KeyError(problem_id)

    def ids(self) -> tuple[int, ...]:
        return tuple(p.problem_id for p in self.problems)

    def fingerprint(self) -> str:
        """Digest over the fields that can change a score. Excludes provenance,
        so refetching the same data on a later date compares equal."""
        blob = json.dumps(_canonical(self.problems), sort_keys=True, default=repr)
        return hashlib.sha256(blob.encode()).hexdigest()

    def require_paper_shape(self) -> None:
        """Raise unless this is the published 142-problem, 8/4/4/4 set."""
        if len(self) != PAPER_PROBLEM_COUNT:
            raise ValueError(f"expected {PAPER_PROBLEM_COUNT} problems, got {len(self)}")
        for p in self.problems:
            if p.case_counts != PAPER_CASE_COUNTS:
                raise ValueError(
                    f"problem {p.problem_id} has case counts {p.case_counts}, "
                    f"expected {PAPER_CASE_COUNTS}"
                )
