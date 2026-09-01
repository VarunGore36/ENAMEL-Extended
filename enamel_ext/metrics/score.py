"""Per-sample efficiency score ``e_{i,j}`` -- Eq. (1)-(2) of the paper.

For problem ``i``, sample ``j``, level ``l``, test case ``m``:

* ``t[i,j,l,m]``  -- measured time of the candidate (``inf`` if killed at the limit)
* ``t*[i,l,m]``   -- measured time of the expert reference
* ``T_i = alpha * max_{l,m} t*[i,l,m]``   -- one time limit for the whole problem

The level fraction takes the *worst* case within a level::

    f[i,j,l] = (T_i - max_m t[i,j,l,m])_+ / (T_i - max_m t*[i,l,m])

and the sample score is the ``h``-weighted mean over levels, or 0 if the sample
is not correct::

    e[i,j] = sum_l h_l f[i,j,l] / sum_l h_l

Paper settings: ``alpha = 2``, ``h = (3, 3, 4)``, levels 1..3 with 4 cases each.
Level 0 (8 small but strong cases) is a correctness filter only and carries no
weight -- it decides the Boolean that gates the whole score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "TIMEOUT",
    "MetricConfig",
    "PAPER",
    "time_limit",
    "level_fraction",
    "sample_score",
]

#: Sentinel for a right-censored run (process killed at the time limit). The
#: true runtime is unknown and >= T_i; the paper's score maps it to exactly 0.
TIMEOUT = math.inf


@dataclass(frozen=True)
class MetricConfig:
    """Knobs of the efficiency score. Defaults are *not* set here -- use
    :data:`PAPER` to get the published configuration, so that any deviation from
    it is visible at the call site.

    ``alpha``
        Time-limit multiplier. Must be > 1, otherwise the denominator of the
        level containing the global slowest reference case is <= 0.
    ``level_weights``
        ``h_1..h_L``. Length fixes the number of scored levels.
    ``normalization``
        ``"global"`` -- the paper: one ``T_i`` per problem, from the slowest
        reference case over *all* levels.
        ``"per_level"`` -- variant: ``T_{i,l} = alpha * max_m t*[i,l,m]``, so
        every level is scaled by its own reference. This is the fix for the
        score-compression issue (README section 2.2), where levels whose
        reference time is a small fraction of level 3's barely discriminate.
        It is a deviation from the published metric and must never be the
        default in a parity run.
    """

    alpha: float
    level_weights: tuple[float, ...]
    normalization: str = "global"

    def __post_init__(self) -> None:
        if not self.alpha > 1.0:
            raise ValueError(f"need alpha > 1 for a positive denominator, got {self.alpha}")
        if not self.level_weights:
            raise ValueError("need at least one scored level")
        if any(w < 0 for w in self.level_weights):
            raise ValueError(f"level weights must be non-negative, got {self.level_weights}")
        if sum(self.level_weights) <= 0:
            raise ValueError("level weights sum to 0")
        if self.normalization not in ("global", "per_level"):
            raise ValueError(f"unknown normalization {self.normalization!r}")

    @property
    def n_levels(self) -> int:
        return len(self.level_weights)


#: The published configuration (alpha = 2, h = (3, 3, 4)). Levels 1-3 therefore
#: carry 30% / 30% / 40% of the weight.
PAPER = MetricConfig(alpha=2.0, level_weights=(3.0, 3.0, 4.0), normalization="global")


def _worst(times: Sequence[float], *, what: str) -> float:
    """Max over test cases within one level, with validation."""
    if not times:
        raise ValueError(f"{what}: a level has no test cases")
    if any(t != t for t in times):
        raise ValueError(f"{what}: NaN timing -- a failed measurement is not a slow one")
    if any(t < 0 for t in times):
        raise ValueError(f"{what}: negative timing")
    return max(times)


def time_limit(reference_times: Sequence[Sequence[float]], config: MetricConfig) -> float:
    """``T_i = alpha * max_{l,m} t*[i,l,m]`` over all scored levels.

    ``reference_times[l][m]`` is the reference time for level ``l+1``, case
    ``m``. Note the max ranges over *every* level, not just the hardest, so a
    single problem where level 2 happens to be slower than level 3 still gets a
    limit that admits its own reference everywhere.
    """
    if len(reference_times) != config.n_levels:
        raise ValueError(
            f"got {len(reference_times)} levels of reference times, "
            f"config declares {config.n_levels}"
        )
    worst = max(_worst(l, what="reference") for l in reference_times)
    if not math.isfinite(worst) or worst <= 0:
        raise ValueError(f"reference must have a finite positive worst-case time, got {worst}")
    return config.alpha * worst


def level_fraction(candidate: Sequence[float], reference: Sequence[float], limit: float) -> float:
    """``f[i,j,l]`` for one level: ``(T - max t)_+ / (T - max t*)``.

    ``limit`` is ``T_i`` (global normalization) or ``T_{i,l}`` (per-level).
    Unclamped above 1: code faster than the expert reference scores > 1, which
    the paper allows and which is how the reported means can be pulled up by a
    single unusually fast sample.
    """
    t = _worst(candidate, what="candidate")
    t_ref = _worst(reference, what="reference")
    denom = limit - t_ref
    if denom <= 0:
        raise ValueError(
            f"time limit {limit} does not exceed the reference worst case {t_ref}; "
            "the reference itself would score 0"
        )
    return max(limit - t, 0.0) / denom


def sample_score(
    candidate_times: Sequence[Sequence[float]],
    reference_times: Sequence[Sequence[float]],
    config: MetricConfig = PAPER,
    *,
    correct: bool = True,
) -> float:
    """``e[i,j]``: the ``h``-weighted mean of level fractions, or 0 if incorrect.

    ``correct`` is the verdict from level 0 (and from levels 1-3 producing right
    answers). Correctness is a hard gate: a wrong-but-fast sample scores 0, and
    that is the whole point of the metric.

    A level where the candidate was killed at the limit should be passed as
    ``[TIMEOUT]``; it contributes 0. The paper stops evaluating a sample after
    its first timeout, so downstream levels are also 0 -- which assumes level
    times are monotone in level. That assumption holds for the paper's
    generators but is not guaranteed in general, so the caller must supply
    ``TIMEOUT`` for the skipped levels explicitly rather than relying on this
    function to infer them.
    """
    if len(candidate_times) != config.n_levels:
        raise ValueError(
            f"got {len(candidate_times)} levels of candidate times, "
            f"config declares {config.n_levels}"
        )
    if not correct:
        return 0.0

    if config.normalization == "global":
        limits = [time_limit(reference_times, config)] * config.n_levels
    else:
        limits = [config.alpha * _worst(r, what="reference") for r in reference_times]

    total = sum(
        w * level_fraction(c, r, lim)
        for w, c, r, lim in zip(config.level_weights, candidate_times, reference_times, limits)
    )
    return total / sum(config.level_weights)
