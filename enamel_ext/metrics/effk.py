"""Unbiased estimators for ``pass@k`` and ``eff@k``.

Reference: Qiu, Zeng, Ezick, Lott & Tong, *How Efficient is LLM-Generated Code?*
ICLR 2025 (arXiv:2406.06647v4) -- Eq. (6), Algorithm 1, Theorem 1.

``eff@k`` is the expected maximum efficiency score over ``k`` i.i.d. samples.
Given ``n >= k`` samples it is estimated by Rao--Blackwellising the bootstrap
estimator, which yields a weighted sum of order statistics::

    eff_i@k = sum_{r=k}^{n} lambda_r * e_(r),    lambda_r = C(r-1, k-1) / C(n, k)

The weights are computed by the backward recurrence of Algorithm 1 rather than
from binomial coefficients directly; see :func:`effk_weights` for why.
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

    The closed form ``lambda_r = C(r-1, k-1) / C(n, k)`` is correct but cannot be
    evaluated in floating point once ``C(n, k)`` passes ~1.8e308, which happens
    around ``n = 1030, k = n/2`` -- well inside plausible sample budgets, and the
    failure is an ``OverflowError`` rather than a quiet ``inf``. The recurrence

        lambda_n = k / n,   lambda_r = lambda_{r+1} * (1 - (k-1)/r)

    forms no binomial coefficient and stays in ``[0, 1]`` throughout.

    Returns the weights in order ``r = k, k+1, ..., n``, so they pair with the
    top ``n-k+1`` order statistics. They sum to 1 (up to float error), since
    ``sum_{r=k}^{n} C(r-1, k-1) = C(n, k)``.
    """
    _check_nk(n, k)
    lam = [0.0] * (n - k + 1)
    lam[-1] = k / n
    for r in range(n - 1, k - 1, -1):
        lam[r - k] = lam[r - k + 1] * (1.0 - (k - 1) / r)
    return lam


def effk_weights_exact(n: int, k: int) -> list[Fraction]:
    """Same weights as exact rationals, straight from Eq. (6).

    Ground truth for tests. Never use in hot paths: ``comb`` on large ``n`` is
    slow and ``Fraction`` arithmetic is slower still.
    """
    _check_nk(n, k)
    denom = comb(n, k)
    return [Fraction(comb(r - 1, k - 1), denom) for r in range(k, n + 1)]


def eff_at_k(scores: Sequence[float], k: int) -> float:
    """Estimate ``eff_i@k`` for one problem from ``n = len(scores)`` samples.

    ``scores`` are the per-sample efficiency scores ``e_{i,j}`` (see
    :mod:`enamel_ext.metrics.score`); incorrect samples must already be 0.
    Order is irrelevant -- the estimator depends only on the sorted values.

    At ``k = n`` this reduces to ``max(scores)``, i.e. the single-subset case.
    """
    n = len(scores)
    _check_nk(n, k)
    if any(s != s for s in scores):  # NaN
        raise ValueError("scores contain NaN; an unscored sample is not the same as a 0")
    # The weights are indexed by rank r = k..n, so they pair with the order
    # statistics e_(k)..e_(n) -- i.e. the top n-k+1 values, not the bottom ones.
    # The k-1 smallest samples contribute nothing: no subset of size k can have
    # its maximum below the k-th smallest value.
    ordered = sorted(scores)[k - 1 :]
    return sum(w * e for w, e in zip(effk_weights(n, k), ordered))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` from Chen et al. (2021): ``1 - C(n-c, k) / C(n, k)``.

    ``c`` is the number of correct samples out of ``n``. Evaluated as a product
    to avoid forming large binomials.
    """
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
    """Aggregate per-problem ``eff_i@k`` into ``eff@k``.

    The paper takes an unweighted mean over all 142 problems ("we define our
    efficiency metric eff@k by averaging eff_i@k over all problems i"). Kept as
    a named function rather than an inline ``mean`` because it is a
    methodological choice, not an implementation detail: it gives a problem with
    a badly calibrated time limit or a wrong reference exactly the same weight
    as a well-behaved one. Any reweighting belongs here and nowhere else.
    """
    if not per_problem:
        raise ValueError("no problems to average over")
    return sum(per_problem) / len(per_problem)
