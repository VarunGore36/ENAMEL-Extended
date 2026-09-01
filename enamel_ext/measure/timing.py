"""Collapsing ``R`` timing repeats of one test case into a single estimate.

The paper runs each test case ``R = 6`` times and takes the Hodges--Lehmann
estimator, whose stated appeal is a breakdown point of ~29% (robust to a couple
of scheduler hiccups) with far better efficiency than the median under a
roughly symmetric noise model.

This module is deliberately separate from the actual measurement backend: it
only turns a list of numbers into one number. The backend that produces those
numbers (wall clock here, instruction counts later) plugs in above it.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Sequence

from enamel_ext.metrics.score import TIMEOUT

__all__ = ["hodges_lehmann", "aggregate_repeats", "AGGREGATORS"]


def hodges_lehmann(sample: Sequence[float]) -> float:
    """One-sample Hodges--Lehmann estimator: the median of the Walsh averages.

    Walsh averages are ``(x_a + x_b) / 2`` for all ``a <= b``, i.e. including
    the ``a == b`` terms, which is the usual one-sample definition (the
    pseudomedian targeted by the signed-rank test). Excluding them -- ``a < b``,
    which some references use -- gives a slightly different number; the paper
    does not say which convention it used, and for ``R = 6`` the two differ by
    O(1%) on skewed samples. We take ``a <= b`` and record the choice; if a
    parity run comes out systematically off, this is one of the first knobs to
    flip.

    Cost is ``O(R^2 log R)``, which for ``R = 6`` (21 averages) is irrelevant.
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
#: default everywhere. ``"min"`` is the standard choice in microbenchmarking --
#: timing noise is one-sided (contention only ever adds time), so the minimum is
#: the best estimator of the noise-free cost and is what ``timeit`` recommends.
#: The two disagree systematically, HL being biased upward by the noise floor,
#: so which one is used is a methodological choice and not a detail. ``"median"``
#: is here as a robustness baseline.
AGGREGATORS = {
    "hodges_lehmann": hodges_lehmann,
    "min": _min,
    "median": _median,
}


def aggregate_repeats(sample: Sequence[float], method: str = "hodges_lehmann") -> float:
    """Reduce ``R`` repeats of one test case to one time.

    Censoring propagates: if any repeat hit the time limit, the case is reported
    as :data:`~enamel_ext.metrics.score.TIMEOUT`. Averaging a censored repeat
    with completed ones would invent a finite time that was never observed, and
    -- since the killed repeat's true time is only known to be ``>= T`` -- would
    bias the result downward by an unbounded amount.
    """
    if method not in AGGREGATORS:
        raise ValueError(f"unknown aggregator {method!r}; have {sorted(AGGREGATORS)}")
    if any(math.isinf(x) for x in sample):
        return TIMEOUT
    return AGGREGATORS[method](sample)
