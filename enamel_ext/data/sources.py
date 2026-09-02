"""Loading problem sets from a local cache or an upstream record dump.

Data is fetched at setup time, never vendored; see README "Credit". Rationale in
docs/decisions/0003-data-adapter.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from enamel_ext.data.schema import (
    GeneratedLevel,
    Level,
    MaterializedLevel,
    Problem,
    ProblemSet,
    Provenance,
)

__all__ = [
    "CACHE_ENV",
    "SCHEMA_VERSION",
    "SYNTHETIC_GENERATOR",
    "UPSTREAM_FIELDS",
    "JsonSource",
    "ProblemSource",
    "default_cache_dir",
    "problem_from_record",
    "problem_set_to_json",
    "problem_set_from_json",
    "problems_from_records",
    "provenance_from_json",
    "provenance_to_json",
    "synthetic_problem_set",
]

SCHEMA_VERSION = 1

CACHE_ENV = "ENAMEL_EXT_DATA"


def default_cache_dir() -> Path:
    """Cache root, overridable so a run can point at a pinned snapshot."""
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "cache"


class ProblemSource(Protocol):
    def load(self) -> ProblemSet: ...


def _level_to_json(lvl: Level) -> dict[str, Any]:
    if isinstance(lvl, GeneratedLevel):
        return {
            "kind": "generated",
            "level": lvl.level,
            "scale": lvl.scale,
            "seeds": list(lvl.seeds),
        }
    return {"kind": "materialized", "level": lvl.level, "inputs": [list(a) for a in lvl.inputs]}


def _level_from_json(raw: Mapping[str, Any], problem_id: int) -> Level:
    kind = raw.get("kind")
    if kind == "generated":
        return GeneratedLevel(level=raw["level"], scale=raw["scale"], seeds=tuple(raw["seeds"]))
    if kind == "materialized":
        return MaterializedLevel(
            level=raw["level"], inputs=tuple(tuple(args) for args in raw["inputs"])
        )
    raise ValueError(f"problem {problem_id}: unknown level kind {kind!r}")


def provenance_to_json(prov: Provenance) -> dict[str, str]:
    """Shared by every artifact that records where its inputs came from."""
    return {
        "name": prov.name,
        "url": prov.url,
        "license": prov.license,
        "retrieved": prov.retrieved,
    }


def provenance_from_json(raw: Mapping[str, Any]) -> Provenance:
    return Provenance(
        name=raw["name"],
        url=raw["url"],
        license=raw["license"],
        retrieved=raw["retrieved"],
    )


def problem_set_to_json(pset: ProblemSet) -> str:
    """Serialize to the cache format. Raises if any materialized input is not
    JSON-representable, because the cache must never require unpickling."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": pset.fingerprint(),
        "provenance": provenance_to_json(pset.provenance),
        "problems": [
            {
                "problem_id": p.problem_id,
                "entry_point": p.entry_point,
                "prompt": p.prompt,
                "reference_solution": p.reference_solution,
                "input_generator": p.input_generator,
                "levels": [_level_to_json(lvl) for lvl in p.levels],
            }
            for p in pset
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def problem_set_from_json(text: str) -> ProblemSet:
    """Parse the cache format. Verifies the stored fingerprint if present."""
    raw = json.loads(text)
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"cache schema version {version!r}, expected {SCHEMA_VERSION}")
    prov = raw["provenance"]
    pset = ProblemSet(
        provenance=provenance_from_json(prov),
        problems=tuple(
            Problem(
                problem_id=p["problem_id"],
                entry_point=p["entry_point"],
                prompt=p["prompt"],
                reference_solution=p["reference_solution"],
                input_generator=p.get("input_generator", ""),
                levels=tuple(_level_from_json(lvl, p["problem_id"]) for lvl in p["levels"]),
            )
            for p in raw["problems"]
        ),
    )
    stored = raw.get("fingerprint")
    if stored is not None and stored != pset.fingerprint():
        raise ValueError("cache fingerprint does not match its contents")
    return pset


class JsonSource:
    """Reads one cache file written by :func:`problem_set_to_json`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> ProblemSet:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"no problem cache at {self.path}. Fetch the benchmark data first, "
                f"or set {CACHE_ENV} to a directory that has it."
            )
        return problem_set_from_json(self.path.read_text())


#: Field names a record dump is expected to use. Override per source; the
#: upstream key names are guesses until the real dataset can be inspected.
UPSTREAM_FIELDS: Mapping[str, str] = {
    "problem_id": "problem_id",
    "entry_point": "entry_point",
    "prompt": "prompt",
    "reference_solution": "reference_solution",
    "input_generator": "input_generator",
    "levels": "levels",
}


def problem_from_record(
    record: Mapping[str, Any], fields: Mapping[str, str] = UPSTREAM_FIELDS
) -> Problem:
    """Build a Problem from one record of a dump.

    ``levels`` must already be in the cache format; converting a source's own
    level layout is the fetch script's job, so the guesswork stays in one place.
    """
    missing = [
        key
        for key, source_key in fields.items()
        if key != "input_generator" and source_key not in record
    ]
    if missing:
        raise KeyError(
            f"record is missing {sorted(fields[k] for k in missing)}; "
            f"it has {sorted(record)}"
        )
    problem_id = record[fields["problem_id"]]
    return Problem(
        problem_id=problem_id,
        entry_point=record[fields["entry_point"]],
        prompt=record[fields["prompt"]],
        reference_solution=record[fields["reference_solution"]],
        input_generator=record.get(fields["input_generator"], ""),
        levels=tuple(
            _level_from_json(lvl, problem_id) for lvl in record[fields["levels"]]
        ),
    )


def problems_from_records(
    records: Iterable[Mapping[str, Any]],
    provenance: Provenance,
    fields: Mapping[str, str] = UPSTREAM_FIELDS,
) -> ProblemSet:
    return ProblemSet(
        provenance=provenance,
        problems=tuple(problem_from_record(r, fields) for r in records),
    )


#: Generator contract: a module defining ``make_input(seed, scale) -> tuple``
#: of positional arguments for the entry point.
SYNTHETIC_GENERATOR = """
import random

def make_input(seed, scale):
    rng = random.Random(seed)
    return ([rng.randrange(1000) for _ in range(scale)],)
"""

_SYNTHETIC_REFERENCE = """
def total(xs):
    return sum(xs)
"""


def synthetic_problem_set(
    n_problems: int = 3, case_counts: tuple[int, ...] = (8, 4, 4, 4)
) -> ProblemSet:
    """A self-contained stand-in for the real benchmark, for harness tests."""
    problems = []
    for pid in range(n_problems):
        levels: list[Level] = []
        seed = pid * 1000
        for level, n_cases in enumerate(case_counts):
            seeds = tuple(range(seed + level * 100, seed + level * 100 + n_cases))
            levels.append(GeneratedLevel(level=level, scale=10 ** (level + 1), seeds=seeds))
        problems.append(
            Problem(
                problem_id=pid,
                entry_point="total",
                prompt="def total(xs):\n    'Sum a list of ints.'\n",
                reference_solution=_SYNTHETIC_REFERENCE,
                input_generator=SYNTHETIC_GENERATOR,
                levels=tuple(levels),
            )
        )
    return ProblemSet(
        provenance=Provenance(
            name="synthetic",
            url="local",
            license="Apache-2.0",
            retrieved="1970-01-01",
        ),
        problems=tuple(problems),
    )
