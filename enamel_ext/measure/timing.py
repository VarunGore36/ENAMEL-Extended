"""Collapsing ``R`` timing repeats of one test case into a single estimate.

The paper uses ``R = 6`` and the Hodges-Lehmann estimator. Also holds the
stopping rule the runner censors on, which has to be the same threshold the
score uses. Rationale and the open convention question in
docs/decisions/0001-metric-core.md.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Sequence

from enamel_ext.metrics.score import TIMEOUT

__all__ = [
    "hodges_lehmann",
    "aggregate_repeats",
    "aggregate_lower_bound",
    "reaches_limit",
    "AGGREGATORS",
]


def hodges_lehmann(sample: Sequence[float]) -> float:
    """Median of the Walsh averages ``(x_a + x_b)/2`` for ``a <= b``.

    Includes the ``a == b`` terms (the one-sample pseudomedian). The paper does
    not say which convention it used; see the decision record.
    """
    if not sample:
        raise ValueError("no timing repeats to aggregate")
    if any(x != x for x in sample):
        raise ValueError("NaN in timing repeats")
    walsh = [
        (sample[a] + sample[b]) / 2.0
        for a in range(len(sample))
        for b in range(a, len(sample))
    ]
    return median(walsh)


def _min(sample: Sequence[float]) -> float:
    if not sample:
        raise ValueError("no timing repeats to aggregate")
    return min(sample)


def _median(sample: Sequence[float]) -> float:
    if not sample:
        raise ValueError("no timing repeats to aggregate")
    return median(sample)


#: Selectable aggregators. ``"hodges_lehmann"`` is the paper's choice and the
#: default; ``"min"`` is the microbenchmarking standard; ``"median"`` is a
#: robustness baseline. The choice is methodological, not a detail.
AGGREGATORS = {
    "hodges_lehmann": hodges_lehmann,
    "min": _min,
    "median": _median,
}


def aggregate_repeats(sample: Sequence[float], method: str = "hodges_lehmann") -> float:
    """Reduce ``R`` repeats of one test case to one time.

    Censoring propagates: if any repeat hit the limit, the case is reported as
    :data:`~enamel_ext.metrics.score.TIMEOUT`.
    """
    if method not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {method!r}; have {sorted(AGGREGATORS)}")
    if any(math.isinf(x) for x in sample):
        return TIMEOUT
    return AGGREGATORS[method](sample)


def aggregate_lower_bound(
    sample: Sequence[float], repeats: int, method: str = "hodges_lehmann"
) -> float:
    """Smallest aggregate still reachable after ``sample``, over ``repeats`` total.

    The unrun repeats are counted as 0, which is the best case for the candidate
    because every aggregator here is non-decreasing in each repeat. Under ``min``
    the bound is therefore 0 until the last repeat.
    """
    if repeats < len(sample):
        raise ValueError(f"{len(sample)} repeats already run, but repeats={repeats}")
    padded = list(sample) + [0.0] * (repeats - len(sample))
    return aggregate_repeats(padded, method)


def reaches_limit(
    sample: Sequence[float], repeats: int, limit: float | None, method: str = "hodges_lehmann"
) -> bool:
    """Whether the case is already certain to score 0 at this level.

    True once no completion of the remaining repeats can bring the aggregate
    back under ``limit``, so stopping here cannot cost the candidate a score it
    would otherwise have earned.
    """
    if limit is None:
        return False
    return aggregate_lower_bound(sample, repeats, method) >= limit
