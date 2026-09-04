"""Run model names to published table keys.

A run's models are named by whatever produced the samples; the published tables
are keyed by the paper's display names. Rationale in
docs/decisions/0008-model-naming.md.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from enamel_ext.data.published import MODEL_IDENTIFIERS, table

__all__ = [
    "ALIASES",
    "NEAR_MISS_CUTOFF",
    "Resolution",
    "near_misses",
    "normalize",
    "published_names",
    "rename",
    "resolve",
]

#: The only run names we can map without guessing: the API identifiers Appendix
#: C.1 states, keyed to the display name the tables use. Nothing else belongs
#: here, because the paper never says what the other rows were called upstream.
ALIASES: dict[str, str] = {
    identifier: display for display, identifier in MODEL_IDENTIFIERS.items()
}

#: Similarity above which an unmatched name is reported as probably a published
#: model spelled differently, rather than as a model the paper never ran.
NEAR_MISS_CUTOFF = 0.8

_TABLES = ("greedy", "sampling", "algorithm", "implementation")


def normalize(name: str) -> str:
    """Casefolded alphanumerics only, so spacing and punctuation stop mattering."""
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def published_names() -> tuple[str, ...]:
    """Every model name the published tables use, in first-seen order."""
    seen: dict[str, None] = {}
    for name in _TABLES:
        for model in table(name):
            seen.setdefault(model, None)
    return tuple(seen)


def _index(aliases: Mapping[str, str] | None = None) -> dict[str, str]:
    """Normalized name to published display name, aliases folded in.

    Raises if two published names normalize alike, which would make the whole
    scheme ambiguous, or if an alias points at a model the paper never published.
    """
    published = published_names()
    index: dict[str, str] = {}
    for model in published:
        key = normalize(model)
        if key in index:
            raise ValueError(
                f"published names collide under normalization: "
                f"{index[key]!r} and {model!r}"
            )
        index[key] = model
    known = set(published)
    for source, target in {**ALIASES, **(aliases or {})}.items():
        if target not in known:
            raise ValueError(f"alias {source!r} names unpublished model {target!r}")
        index.setdefault(normalize(source), target)
    return index


def near_misses(
    name: str, cutoff: float = NEAR_MISS_CUTOFF, limit: int = 3
) -> tuple[str, ...]:
    """Published names close enough to ``name`` to be worth a human look."""
    index = _index()
    hits = difflib.get_close_matches(
        normalize(name), list(index), n=limit, cutoff=cutoff
    )
    return tuple(index[hit] for hit in hits)


@dataclass(frozen=True)
class Resolution:
    """What a run's model names mapped to, and what they did not.

    ``unresolved`` and ``suspect`` are kept apart on purpose: an unmatched name
    with no near miss is plausibly a model the paper never ran, which is a
    legitimate state, while one with a near miss is a naming bug that would
    silently shrink the comparison.
    """

    resolved: dict[str, str] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    suspect: dict[str, tuple[str, ...]] = field(default_factory=dict)
    collisions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        """Nothing that needs a human before the comparison can be trusted."""
        return not (self.suspect or self.collisions)


def resolve(
    names: Iterable[str],
    aliases: Mapping[str, str] | None = None,
    cutoff: float = NEAR_MISS_CUTOFF,
) -> Resolution:
    """Map run model names onto published ones, reporting what did not map."""
    index = _index(aliases)
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    suspect: dict[str, tuple[str, ...]] = {}
    for name in names:
        target = index.get(normalize(name))
        if target is None:
            unresolved.append(name)
            hits = difflib.get_close_matches(
                normalize(name), list(index), n=3, cutoff=cutoff
            )
            if hits:
                suspect[name] = tuple(index[hit] for hit in hits)
            continue
        resolved[name] = target
    hit_by: dict[str, list[str]] = {}
    for name, target in resolved.items():
        hit_by.setdefault(target, []).append(name)
    return Resolution(
        resolved=resolved,
        unresolved=tuple(unresolved),
        suspect=suspect,
        collisions={
            target: tuple(sources)
            for target, sources in hit_by.items()
            if len(sources) > 1
        },
    )


def rename(
    scores: Mapping[str, float],
    aliases: Mapping[str, str] | None = None,
    cutoff: float = NEAR_MISS_CUTOFF,
) -> tuple[dict[str, float], Resolution]:
    """``scores`` rekeyed to published names, unmatched keys carried through.

    Unmatched keys keep their original name so the comparison reports them as
    extra rather than dropping them. Raises on a collision, which has no
    well-defined answer.
    """
    report = resolve(scores, aliases, cutoff)
    if report.collisions:
        target, sources = next(iter(report.collisions.items()))
        raise ValueError(f"{list(sources)} all name published model {target!r}")
    return {report.resolved.get(name, name): value for name, value in scores.items()}, report
