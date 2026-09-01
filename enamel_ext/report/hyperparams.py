"""Hyperparameter sensitivity and rank stability.

``eff@k`` is *exactly linear* in the level hardnesses::

    eff@k(h) = (h_1 F_1 + ... + h_L F_L) / (h_1 + ... + h_L)

where ``F_l`` is the mean of ``f[i,j,l]`` over samples and problems. Two things
follow, both cheap enough to be a table rather than an experiment:

1. The published hardness sweeps can be *inverted* to recover ``F_l`` -- see
   ``scripts/recover_table10.py``.
2. Whether a model pair can be reordered by choosing ``h`` is decidable exactly.
   Since ``sign(eff^A(h) - eff^B(h)) = sign(sum_l h_l d_l)`` with
   ``d_l = F^A_l - F^B_l`` and ``h_l > 0``, the pair is reorderable **iff ``d``
   has mixed signs**. No search, no sweep.

``alpha`` is not linear and not free: raising it un-censors runs whose times were
never observed. :func:`rescore_at_alpha` enforces that.
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Sequence

from enamel_ext.metrics.score import MetricConfig, sample_score

__all__ = [
    "eff_at_h",
    "attainable_range",
    "HComparison",
    "compare_under_h",
    "reorderable_pairs",
    "rescore_at_alpha",
]


def eff_at_h(level_means: Sequence[float], h: Sequence[float]) -> float:
    """Aggregate score for given level means and hardnesses."""
    if len(level_means) != len(h):
        raise ValueError(f"length mismatch: {len(level_means)} level means vs {len(h)} weights")
    if not level_means:
        raise ValueError("no levels")
    if any(w <= 0 for w in h):
        raise ValueError(f"the paper requires h_l > 0, got {tuple(h)}")
    return sum(w * f for w, f in zip(h, level_means)) / sum(h)


def attainable_range(level_means: Sequence[float]) -> tuple[float, float]:
    """Range of ``eff@k`` reachable by reweighting ``h`` alone.

    The score is a convex combination of the level means, so the reachable set is
    their convex hull. With ``h_l > 0`` strictly the endpoints are approached but
    not attained; they are returned closed because the distinction is not
    material to the point being made.
    """
    if not level_means:
        raise ValueError("no levels")
    return min(level_means), max(level_means)


class HComparison(NamedTuple):
    """Outcome of comparing two models over all admissible ``h``.

    ``verdict`` is one of ``"a_always"``, ``"b_always"``, ``"tie"``, or
    ``"reorderable"``. For a reorderable pair, ``witness_a`` and ``witness_b`` are
    concrete integer hardness vectors that put each model on top -- useful in a
    paper precisely because they are exhibitable rather than asymptotic.
    """

    verdict: str
    witness_a: tuple[int, ...] | None = None
    witness_b: tuple[int, ...] | None = None

    @property
    def stable(self) -> bool:
        return self.verdict != "reorderable"


#: Relative tolerance for calling two level means different. Level means are
#: O(1), so this admits real differences down to ~1e-10 while treating the
#: round-off of a single subtraction as the zero it actually is.
_REL_SIGN = 1e-9

#: Relative tolerance for calling a weighted sum positive. Deliberately much
#: tighter than :data:`_REL_SIGN`: a witness is *built* to sit just past total
#: cancellation, so a threshold that generous would reject valid ones. Still
#: some four orders of magnitude above float eps, which is what the integer
#: weights -- large when one level's edge is small -- need.
_REL_DOT = 1e-12


def _signed_diff(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """``a - b`` per level, with differences at round-off scale forced to zero.

    Without this, a pair that weakly dominates (equal on some level) can be
    reported as reorderable purely because ``0.4 - 0.4`` came out at ``-1e-17``.
    """
    out = []
    for x, y in zip(a, b):
        d = x - y
        out.append(0.0 if abs(d) <= _REL_SIGN * (abs(x) + abs(y)) else d)
    return out


def _dot_is_positive(h: Sequence[float], d: Sequence[float]) -> bool:
    """``sum(h_l d_l) > 0``, robust to catastrophic cancellation.

    Comparing the sum against the sum of the term magnitudes is what makes this
    safe: near a crossing the two are many orders of magnitude apart, so noise
    surviving the cancellation cannot masquerade as a decision.
    """
    total = sum(w * x for w, x in zip(h, d))
    magnitude = sum(w * abs(x) for w, x in zip(h, d))
    return total > _REL_DOT * magnitude


def _integer_witness(d: Sequence[float]) -> tuple[int, ...] | None:
    """Smallest simple ``h`` (ones, with extra weight on one level) making
    ``sum(h_l d_l) > 0``, or None if no positive ``h`` can."""
    best = max(range(len(d)), key=lambda i: d[i])
    if d[best] <= 0:
        return None
    others = sum(d) - d[best]
    weight = 1 if others >= 0 else int(-others / d[best]) + 1
    for bump in range(4):  # the floor() above can land one short of the crossing
        h = [1] * len(d)
        h[best] = weight + bump
        if _dot_is_positive(h, d):
            return tuple(h)
    return None


def compare_under_h(a_level_means: Sequence[float], b_level_means: Sequence[float]) -> HComparison:
    """Decide whether ``h`` can reorder two models, exactly.

    Requires only the per-level means -- three numbers per model for the paper's
    ``L = 3``. A pair is stable when one model's level means dominate the other's
    on every level; otherwise a hardness vector exists for each ordering.
    """
    if len(a_level_means) != len(b_level_means):
        raise ValueError("both models need the same number of levels")
    if not a_level_means:
        raise ValueError("no levels")

    d = _signed_diff(a_level_means, b_level_means)
    if all(x == 0 for x in d):
        return HComparison("tie")
    if all(x >= 0 for x in d):
        return HComparison("a_always")
    if all(x <= 0 for x in d):
        return HComparison("b_always")

    return HComparison(
        "reorderable",
        witness_a=_integer_witness(d),
        witness_b=_integer_witness([-x for x in d]),
    )


def reorderable_pairs(
    models: Mapping[str, Sequence[float]],
) -> list[tuple[str, str, HComparison]]:
    """Every model pair whose ordering depends on the choice of ``h``.

    An empty result is the good outcome: it means the leaderboard is a
    consequence of the models, not of the hardness weights. Pairs are keyed in
    the iteration order of ``models``.
    """
    names = list(models)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cmp = compare_under_h(models[a], models[b])
            if not cmp.stable:
                out.append((a, b, cmp))
    return out


def rescore_at_alpha(
    candidate_times: Sequence[Sequence[float]],
    reference_times: Sequence[Sequence[float]],
    *,
    new_alpha: float,
    measured_alpha: float,
    level_weights: Sequence[float] = (3.0, 3.0, 4.0),
    correct: bool = True,
) -> float:
    """Recompute a sample's score at a different ``alpha``, without re-running.

    Lowering ``alpha`` is always safe: a run that finished under the old limit has
    a known time, and the score simply clamps to 0 if it now exceeds the new one.

    Raising ``alpha`` is only safe if nothing was censored. A run killed at the
    old limit has an unknown time somewhere in ``[T_old, inf)`` and may well have
    finished under a larger limit, so its new score is not determined by the
    recorded data. Rather than silently keeping it at 0 -- which would bias every
    larger-``alpha`` result downward -- this raises. The consequence for the
    harness: measure once at the largest ``alpha`` you will ever want to report,
    then derive all smaller ones here.
    """
    if measured_alpha <= 1.0:
        raise ValueError(f"measured_alpha must exceed 1, got {measured_alpha}")
    if new_alpha > measured_alpha and any(
        math.isinf(t) for level in candidate_times for t in level
    ):
        raise ValueError(
            f"cannot raise alpha from {measured_alpha} to {new_alpha}: this sample was "
            "censored, so its true time was never observed. Re-measure with the larger "
            "alpha, or restrict the sweep to alpha <= the measured value."
        )
    config = MetricConfig(alpha=new_alpha, level_weights=tuple(level_weights))
    return sample_score(candidate_times, reference_times, config, correct=correct)
