"""Candidate solutions to score: one set of samples per model and problem.

JSON only, for the same reason as the problem cache: loading a benchmark input
must never mean unpickling one. See docs/decisions/0006-run-record.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from enamel_ext.data.schema import Problem, Provenance
from enamel_ext.data.sources import provenance_from_json, provenance_to_json

__all__ = [
    "SOLUTIONS_SCHEMA_VERSION",
    "SolutionSet",
    "load_solutions",
    "solution_set_from_json",
    "solution_set_to_json",
    "synthetic_solutions",
]

SOLUTIONS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SolutionSet:
    """``model -> problem id -> n code strings``, with where it came from.

    A problem absent for a model means no samples were generated, which is not
    the same as samples that failed; scoring skips it and the report says so.
    """

    provenance: Provenance
    samples: Mapping[str, Mapping[int, tuple[str, ...]]]

    def __post_init__(self) -> None:
        normalized: dict[str, dict[int, tuple[str, ...]]] = {}
        for model, by_problem in self.samples.items():
            if not str(model).strip():
                raise ValueError("model name must not be empty")
            if not by_problem:
                raise ValueError(f"model {model!r} has no problems")
            per_problem: dict[int, tuple[str, ...]] = {}
            for problem_id, codes in by_problem.items():
                pid = int(problem_id)
                if pid < 0:
                    raise ValueError(f"model {model!r}: problem id must be >= 0, got {pid}")
                frozen = tuple(codes)
                if not frozen:
                    raise ValueError(f"model {model!r} problem {pid} has no samples")
                for index, code in enumerate(frozen):
                    if not isinstance(code, str):
                        raise ValueError(
                            f"model {model!r} problem {pid} sample {index} is "
                            f"{type(code).__name__}, expected source text"
                        )
                per_problem[pid] = frozen
            normalized[str(model)] = per_problem
        if not normalized:
            raise ValueError("no models")
        object.__setattr__(self, "samples", normalized)

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self.samples))

    def problem_ids(self, model: str) -> tuple[int, ...]:
        return tuple(sorted(self.samples[model]))

    def codes(self, model: str, problem_id: int) -> tuple[str, ...]:
        return self.samples[model].get(problem_id, ())

    def sample_counts(self, model: str) -> tuple[int, ...]:
        """Distinct sample counts across this model's problems, sorted.

        More than one entry means ``n`` varies by problem, which the estimators
        allow but which makes ``eff@k`` mix per-problem variances.
        """
        return tuple(sorted({len(v) for v in self.samples[model].values()}))

    def common_problem_ids(self) -> tuple[int, ...]:
        """Problems every model covers, which is what a paired test needs."""
        shared: set[int] | None = None
        for model in self.samples:
            ids = set(self.samples[model])
            shared = ids if shared is None else shared & ids
        return tuple(sorted(shared or ()))

    def fingerprint(self) -> str:
        """Digest over models, problem ids and sample text. Excludes provenance."""
        canonical = [
            [model, [[pid, list(self.samples[model][pid])] for pid in self.problem_ids(model)]]
            for model in self.models
        ]
        blob = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def solution_set_to_json(solutions: SolutionSet) -> str:
    payload: dict[str, Any] = {
        "schema_version": SOLUTIONS_SCHEMA_VERSION,
        "fingerprint": solutions.fingerprint(),
        "provenance": provenance_to_json(solutions.provenance),
        "samples": {
            model: {
                str(pid): list(solutions.codes(model, pid))
                for pid in solutions.problem_ids(model)
            }
            for model in solutions.models
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def solution_set_from_json(text: str) -> SolutionSet:
    """Parse the sample format, verifying the stored fingerprint if present."""
    raw = json.loads(text)
    version = raw.get("schema_version")
    if version != SOLUTIONS_SCHEMA_VERSION:
        raise ValueError(
            f"solutions schema version {version!r}, expected {SOLUTIONS_SCHEMA_VERSION}"
        )
    prov = raw["provenance"]
    solutions = SolutionSet(
        provenance=provenance_from_json(prov),
        samples={
            model: {int(pid): tuple(codes) for pid, codes in by_problem.items()}
            for model, by_problem in raw["samples"].items()
        },
    )
    stored = raw.get("fingerprint")
    if stored is not None and stored != solutions.fingerprint():
        raise ValueError("solutions fingerprint does not match their contents")
    return solutions


def load_solutions(path: Path | str) -> SolutionSet:
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"no solution set at {location}")
    return solution_set_from_json(location.read_text())


#: Four candidates for the synthetic ``total(xs)`` problem: the reference itself,
#: a slower correct loop, a correct one that sorts its argument in place, and a
#: wrong answer. Enough shapes to exercise the whole pipeline without real data.
_SYNTHETIC_CANDIDATES = (
    "def total(xs):\n    return sum(xs)\n",
    "def total(xs):\n    acc = 0\n    for x in xs:\n        acc += x\n    return acc\n",
    "def total(xs):\n    xs.sort()\n    return sum(xs)\n",
    "def total(xs):\n    return 0\n",
)


def synthetic_solutions(
    problems: Sequence[Problem],
    models: Sequence[str] = ("reference-copy", "mixed-bag"),
) -> SolutionSet:
    """Runnable stand-in samples for the synthetic problem set.

    ``reference-copy`` submits the reference to every problem; ``mixed-bag``
    submits all four candidates, so one sample is wrong and one mutates its
    input.
    """
    by_model: dict[str, dict[int, tuple[str, ...]]] = {}
    for model in models:
        codes = _SYNTHETIC_CANDIDATES[:1] if model == "reference-copy" else _SYNTHETIC_CANDIDATES
        by_model[model] = {p.problem_id: codes for p in problems}
    return SolutionSet(
        provenance=Provenance(
            name="synthetic", url="local", license="Apache-2.0", retrieved="1970-01-01"
        ),
        samples=by_model,
    )
