"""Per-sample efficiency score ``e_{i,j}``, Eq. (1) and (2).

For problem ``i``, sample ``j``, level ``l``, test case ``m``:

    t[i,j,l,m]  candidate time (``inf`` if killed at the limit)
    t*[i,l,m]   expert reference time
    T_i         = alpha * max_{l,m} t*[i,l,m]
    f[i,j,l]    = (T_i - max_m t[i,j,l,m])_+ / (T_i - max_m t*[i,l,m])
    e[i,j]      = sum_l h_l f[i,j,l] / sum_l h_l,  or 0 if incorrect

Paper settings: ``alpha = 2``, ``h = (3, 3, 4)``, levels 1..3 with 4 cases each.
Level 0 is a correctness filter and carries no weight. Rationale in
docs/decisions/0001-metric-core.md.
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

#: A right-censored run: killed at the limit, true runtime unknown and >= T_i.
TIMEOUT = math.inf


@dataclass(frozen=True)
class MetricConfig:
    """Knobs of the efficiency score. No defaults; use :data:`PAPER`.

    ``alpha``
        Time-limit multiplier, must be > 1.
    ``level_weights``
        ``h_1..h_L``. Length fixes the number of scored levels.
    ``normalization``
        ``"global"`` is the paper: one ``T_i`` per problem, from the slowest
        reference case over all levels. ``"per_level"`` is a variant with
        ``T_{i,l} = alpha * max_m t*[i,l,m]``, never valid in a parity run.
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


#: The published configuration: levels 1-3 carry 30% / 30% / 40%.
PAPER = MetricConfig(alpha=2.0, level_weights=(3.0, 3.0, 4.0), normalization="global")


def _worst(times: Sequence[float], *, what: str) -> float:
    """Max over test cases within one level, with validation."""
    if not times:
        raise ValueError(f"{what}: a level has no test cases")
    if any(t != t for t in times):
        raise ValueError(f"{what}: NaN timing; a failed measurement is not a slow one")
    if any(t < 0 for t in times):
        raise ValueError(f"{what}: negative timing")
    return max(times)


def time_limit(reference_times: Sequence[Sequence[float]], config: MetricConfig) -> float:
    """``T_i = alpha * max_{l,m} t*[i,l,m]``, over every scored level.

    ``reference_times[l][m]`` is the reference time for level ``l+1``, case ``m``.
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
    """``f[i,j,l]`` for one level. Unclamped above 1: beating the reference scores
    more than 1, which at ``alpha = 2`` tops out at 2."""
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

    Pass ``[TIMEOUT]`` for a level the candidate was killed on, and for the
    levels the harness then skipped: they are not inferred here.
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
