"""Uncertainty and rank statistics over per-problem scores.

The resampling unit is the problem, not the sample. Every function takes an
explicit ``seed``. Rationale in docs/decisions/0002-reporting-layer.md.
"""

from __future__ import annotations

import itertools
import random
from typing import Callable, NamedTuple, Sequence

__all__ = ["Interval", "bootstrap_ci", "paired_bootstrap_diff_ci", "paired_sign_test", "kendall_tau"]


class Interval(NamedTuple):
    """A point estimate with a confidence interval. ``level`` is e.g. 0.95."""

    point: float
    lo: float
    hi: float
    level: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] ({self.level:.0%})"

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _percentile(ordered: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence, ``q`` in [0, 1]."""
    if not ordered:
        raise ValueError("empty distribution")
    idx = int(round(q * (len(ordered) - 1)))
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must be in (0, 1), got {level}")


def bootstrap_ci(
    per_problem: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = _mean,
    resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over problems for one model's aggregate score.

    ``per_problem`` is ``eff_i@k`` for each problem.
    """
    _check_level(level)
    if not per_problem:
        raise ValueError("no problems to resample")
    if resamples < 1:
        raise ValueError(f"need at least one resample, got {resamples}")
    rng = random.Random(seed)
    n = len(per_problem)
    draws = sorted(
        statistic([per_problem[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    tail = (1.0 - level) / 2.0
    return Interval(statistic(per_problem), _percentile(draws, tail), _percentile(draws, 1.0 - tail), level)


def paired_bootstrap_diff_ci(
    a_per_problem: Sequence[float],
    b_per_problem: Sequence[float],
    *,
    resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap for ``mean(a) - mean(b)``, resampling problems jointly.

    This, not two independent intervals, is the comparison a leaderboard needs:
    problem difficulty is a shared term that cancels in the difference.
    """
    _check_level(level)
    if len(a_per_problem) != len(b_per_problem):
        raise ValueError(
            f"paired comparison needs the same problems: {len(a_per_problem)} vs "
            f"{len(b_per_problem)}"
        )
    if not a_per_problem:
        raise ValueError("no problems to resample")
    rng = random.Random(seed)
    n = len(a_per_problem)
    diffs = [a - b for a, b in zip(a_per_problem, b_per_problem)]
    draws = sorted(
        _mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    tail = (1.0 - level) / 2.0
    return Interval(_mean(diffs), _percentile(draws, tail), _percentile(draws, 1.0 - tail), level)


def paired_sign_test(
    a_per_problem: Sequence[float],
    b_per_problem: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for ``mean(a) - mean(b) != 0`` by sign-flip permutation.

    Exact for ``n <= 20``; sampled above that, with the observed arrangement
    counted in the null so the p-value is never 0.
    """
    if len(a_per_problem) != len(b_per_problem):
        raise ValueError("paired test needs the same problems on both sides")
    diffs = [a - b for a, b in zip(a_per_problem, b_per_problem)]
    if not diffs:
        raise ValueError("no problems to test")
    observed = abs(_mean(diffs))
    n = len(diffs)

    if n <= 20:
        patterns = itertools.product((1.0, -1.0), repeat=n)
        total = 2**n
        at_least = sum(
            1 for signs in patterns if abs(_mean([s * d for s, d in zip(signs, diffs)])) >= observed
        )
        return at_least / total

    rng = random.Random(seed)
    at_least = 1  # the observed arrangement itself
    for _ in range(resamples):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(_mean(flipped)) >= observed:
            at_least += 1
    return at_least / (resamples + 1)


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b between two rankings, tie-corrected.

    1.0 for identical orderings, -1.0 for exactly reversed. Raises if either
    ranking is entirely tied, where tau is undefined.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    n = len(x)
    if n < 2:
        raise ValueError("need at least two items to rank")

    concordant = discordant = tied_x = tied_y = 0
    for i, j in itertools.combinations(range(n), 2):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        if dx == 0 and dy == 0:
            tied_x += 1
            tied_y += 1
        elif dx == 0:
            tied_x += 1
        elif dy == 0:
            tied_y += 1
        elif (dx > 0) == (dy > 0):
            concordant += 1
        else:
            discordant += 1

    n0 = n * (n - 1) / 2
    denom = ((n0 - tied_x) * (n0 - tied_y)) ** 0.5
    if denom == 0:
        raise ValueError("one of the rankings is entirely tied; tau is undefined")
    return (concordant - discordant) / denom
