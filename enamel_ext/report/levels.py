"""How much runtime difference each level can actually see.

``T_i`` is one constant per problem, set by the slowest reference case over all
levels, so a level whose reference time is a small fraction ``q`` of that one
scores almost the same for a candidate at the reference speed as for one many
times slower. This module measures ``q`` and turns it into the two numbers that
follow from it: the slowdown a level tolerates before scoring 0, and each level's
share of the score's sensitivity. Rationale in
docs/decisions/0002-reporting-layer.md.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from enamel_ext.metrics.score import PAPER, level_fraction

__all__ = [
    "PAPER_SLOWDOWNS",
    "LevelSummary",
    "describe_levels",
    "level_fraction_at",
    "limit_level",
    "limit_level_counts",
    "q_distribution",
    "q_ratios",
    "sensitivity_shares",
    "tolerated_slowdown",
]

#: Slowdown factors tabulated in README section 2.2.
PAPER_SLOWDOWNS = (2.0, 5.0, 10.0, 50.0)


def _worst_per_level(times: Sequence[Sequence[float]]) -> list[float]:
    if not times:
        raise ValueError("no levels")
    out = []
    for index, level in enumerate(times, start=1):
        if not level:
            raise ValueError(f"level {index} has no test cases")
        worst = max(level)
        if not worst > 0 or worst != worst:
            raise ValueError(f"level {index} reference worst case must be positive, got {worst}")
        out.append(worst)
    return out


def level_fraction_at(q: float, slowdown: float, alpha: float = PAPER.alpha) -> float:
    """``f[i,j,l]`` in units of the limit-setting reference time.

    ``q`` is this level's reference worst case over that constant and
    ``slowdown`` is how many times slower than the reference the candidate is,
    so this is Eq. (1) with ``t* = q``, ``t = slowdown * q`` and ``T = alpha``.
    """
    if not slowdown >= 0:
        raise ValueError(f"slowdown must be >= 0, got {slowdown}")
    return level_fraction([slowdown * q], [q], alpha)


def tolerated_slowdown(q: float, alpha: float = PAPER.alpha) -> float:
    """Slowdown at which this level first scores 0, ``alpha / q``.

    Below it the level scores something; at or above it every candidate scores
    the same 0, so the level has stopped distinguishing.
    """
    if not q > 0:
        raise ValueError(f"q must be positive, got {q}")
    return alpha / q


def limit_level(times: Sequence[Sequence[float]]) -> int:
    """1-based index of the timed level whose worst case sets ``T_i``.

    Ties go to the earliest level. Section 2.2's argument assumes this is the
    last level, which growing input scale makes likely but does not guarantee.
    """
    worst = _worst_per_level(times)
    return worst.index(max(worst)) + 1


def q_ratios(times: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Per-level ``q``: worst reference case over the largest such over levels.

    ``times[l][m]`` is the reference time for timed level ``l+1``, case ``m``, as
    :meth:`ReferenceMeasurement.timed` returns. The limit-setting level has
    ``q = 1`` by construction.
    """
    worst = _worst_per_level(times)
    largest = max(worst)
    return tuple(w / largest for w in worst)


def q_distribution(problems: Iterable[Sequence[Sequence[float]]]) -> tuple[tuple[float, ...], ...]:
    """Per-level ``q`` across problems, as one tuple per level."""
    columns: list[list[float]] = []
    for index, times in enumerate(problems):
        ratios = q_ratios(times)
        if not columns:
            columns = [[] for _ in ratios]
        elif len(ratios) != len(columns):
            raise ValueError(
                f"problem at position {index} has {len(ratios)} timed levels, "
                f"the first had {len(columns)}"
            )
        for column, value in zip(columns, ratios):
            column.append(value)
    if not columns:
        raise ValueError("no problems")
    return tuple(tuple(column) for column in columns)


def limit_level_counts(problems: Iterable[Sequence[Sequence[float]]]) -> dict[int, int]:
    """How many problems have ``T_i`` set by each level."""
    counts: dict[int, int] = {}
    for times in problems:
        level = limit_level(times)
        counts[level] = counts.get(level, 0) + 1
    if not counts:
        raise ValueError("no problems")
    return counts


@dataclass(frozen=True)
class LevelSummary:
    """One level's measured ``q`` and what it implies for discrimination.

    ``fractions`` maps a slowdown factor to the score this level awards at the
    median ``q``.
    """

    level: int
    n_problems: int
    q_min: float
    q_median: float
    q_max: float
    tolerated: float
    fractions: Mapping[float, float]


def describe_levels(
    problems: Iterable[Sequence[Sequence[float]]],
    *,
    alpha: float = PAPER.alpha,
    slowdowns: Sequence[float] = PAPER_SLOWDOWNS,
) -> tuple[LevelSummary, ...]:
    """Summarize every timed level's discrimination over a set of problems."""
    columns = q_distribution(problems)
    out = []
    for index, column in enumerate(columns, start=1):
        median = statistics.median(column)
        out.append(
            LevelSummary(
                level=index,
                n_problems=len(column),
                q_min=min(column),
                q_median=median,
                q_max=max(column),
                tolerated=tolerated_slowdown(median, alpha),
                fractions={x: level_fraction_at(median, x, alpha) for x in slowdowns},
            )
        )
    return tuple(out)


def sensitivity_shares(
    q_per_level: Sequence[float],
    level_weights: Sequence[float] = PAPER.level_weights,
    alpha: float = PAPER.alpha,
) -> tuple[float, ...]:
    """Each level's share of the score's response to a uniform slowdown.

    ``d f_l / d slowdown`` is ``-q_l / (alpha - q_l)`` while the level is
    unsaturated, so the shares are ``h_l q_l / (alpha - q_l)`` normalized. A
    candidate in a worse complexity class slows more at the larger scales, which
    concentrates the response further, so this is a floor for the last level
    rather than an estimate.
    """
    if len(q_per_level) != len(level_weights):
        raise ValueError(
            f"length mismatch: {len(q_per_level)} levels vs {len(level_weights)} weights"
        )
    if not q_per_level:
        raise ValueError("no levels")
    terms = []
    for index, (q, weight) in enumerate(zip(q_per_level, level_weights), start=1):
        if not q > 0:
            raise ValueError(f"level {index}: q must be positive, got {q}")
        if alpha - q <= 0:
            raise ValueError(
                f"level {index}: q = {q} does not leave a positive denominator at "
                f"alpha = {alpha}; the reference itself would score 0"
            )
        if weight < 0:
            raise ValueError(f"level {index}: weight must be non-negative, got {weight}")
        terms.append(weight * q / (alpha - q))
    total = sum(terms)
    if total <= 0:
        raise ValueError("every level has zero weight, so there is no response to share out")
    return tuple(term / total for term in terms)
