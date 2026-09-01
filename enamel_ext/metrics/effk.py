"""Unbiased estimators for ``pass@k`` and ``eff@k``.

Eq. (6), Algorithm 1 and Theorem 1 of arXiv:2406.06647v4. Rationale in
docs/decisions/0001-metric-core.md.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Sequence

__all__ = [
    "effk_weights",
    "effk_weights_exact",
    "eff_at_k",
    "pass_at_k",
    "mean_over_problems",
]


def _check_nk(n: int, k: int) -> None:
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError(f"n and k must be ints, got {type(n).__name__}/{type(k).__name__}")
    if n < 1:
        raise ValueError(f"need n >= 1, got n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")


def effk_weights(n: int, k: int) -> list[float]:
    """Weights ``lambda_r`` for ``r = k..n``, via the Algorithm 1 recurrence.

    Returned in order ``r = k..n``, so they pair with the top ``n-k+1`` order
    statistics. They sum to 1.
    """
    _check_nk(n, k)
    lam = [0.0] * (n - k + 1)
    lam[-1] = k / n
    for r in range(n - 1, k - 1, -1):
        lam[r - k] = lam[r - k + 1] * (1.0 - (k - 1) / r)
    return lam


def effk_weights_exact(n: int, k: int) -> list[Fraction]:
    """Same weights as exact rationals, straight from Eq. (6). Test ground truth
    only: ``comb`` and ``Fraction`` are both slow."""
    _check_nk(n, k)
    denom = comb(n, k)
    return [Fraction(comb(r - 1, k - 1), denom) for r in range(k, n + 1)]


def eff_at_k(scores: Sequence[float], k: int) -> float:
    """Estimate ``eff_i@k`` for one problem from ``n = len(scores)`` samples.

    ``scores`` are the per-sample scores ``e_{i,j}``; incorrect samples must
    already be 0. Order is irrelevant.
    """
    n = len(scores)
    _check_nk(n, k)
    if any(s != s for s in scores):  # NaN
        raise ValueError("scores contain NaN; an unscored sample is not the same as a 0")
    # Weights are indexed by rank r = k..n, so they pair with the top n-k+1
    # values. Starting at index 0 silently discards the largest scores.
    ordered = sorted(scores)[k - 1 :]
    return sum(w * e for w, e in zip(effk_weights(n, k), ordered))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` from Chen et al. (2021): ``1 - C(n-c, k) / C(n, k)``,
    with ``c`` correct out of ``n``. Evaluated as a product."""
    _check_nk(n, k)
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if n - c < k:
        return 1.0
    acc = 1.0
    for i in range(n - c + 1, n + 1):
        acc *= 1.0 - k / i
    return 1.0 - acc


def mean_over_problems(per_problem: Sequence[float]) -> float:
    """Aggregate per-problem ``eff_i@k`` into ``eff@k`` by an unweighted mean.

    Named rather than inlined because it is a methodological choice: any
    reweighting belongs here and nowhere else.
    """
    if not per_problem:
        raise ValueError("no problems to average over")
    return sum(per_problem) / len(per_problem)
