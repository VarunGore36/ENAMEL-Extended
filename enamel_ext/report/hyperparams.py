"""Hyperparameter sensitivity and rank stability.

``eff@k`` is exactly linear in the level hardnesses::

    eff@k(h) = (h_1 F_1 + ... + h_L F_L) / (h_1 + ... + h_L)

so the published hardness sweeps invert to recover ``F_l``
(``scripts/recover_table10.py``), and whether a model pair can be reordered by
choosing ``h`` is decidable exactly: the pair is reorderable iff
``d_l = F^A_l - F^B_l`` has mixed signs. ``alpha`` is neither linear nor free.
Rationale in docs/decisions/0002-reporting-layer.md.
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

    The score is a convex combination of the level means. Returned closed,
    although ``h_l > 0`` strictly makes the endpoints unattainable.
    """
    if not level_means:
        raise ValueError("no levels")
    return min(level_means), max(level_means)


class HComparison(NamedTuple):
    """Outcome of comparing two models over all admissible ``h``.

    ``verdict`` is ``"a_always"``, ``"b_always"``, ``"tie"`` or
    ``"reorderable"``. For a reorderable pair the witnesses are integer hardness
    vectors that put each model on top.
    """

    verdict: str
    witness_a: tuple[int, ...] | None = None
    witness_b: tuple[int, ...] | None = None

    @property
    def stable(self) -> bool:
        return self.verdict != "reorderable"


#: Relative tolerance for calling two level means different.
_REL_SIGN = 1e-9

#: Relative tolerance for calling a weighted sum positive. Tighter than
#: :data:`_REL_SIGN` because a witness is built to sit just past cancellation.
_REL_DOT = 1e-12


def _signed_diff(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """``a - b`` per level, with differences at round-off scale forced to zero."""
    out = []
    for x, y in zip(a, b):
        d = x - y
        out.append(0.0 if abs(d) <= _REL_SIGN * (abs(x) + abs(y)) else d)
    return out


def _dot_is_positive(h: Sequence[float], d: Sequence[float]) -> bool:
    """``sum(h_l d_l) > 0``, robust to catastrophic cancellation."""
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
    for bump in range(4):  # the floor above can land one short of the crossing
        h = [1] * len(d)
        h[best] = weight + bump
        if _dot_is_positive(h, d):
            return tuple(h)
    return None


def compare_under_h(a_level_means: Sequence[float], b_level_means: Sequence[float]) -> HComparison:
    """Decide whether ``h`` can reorder two models, exactly.

    Stable when one model's level means dominate the other's on every level;
    otherwise a hardness vector exists for each ordering.
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

    An empty result means the leaderboard is a consequence of the models rather
    than of the hardness weights. Pairs are keyed in the iteration order of
    ``models``.
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

    Lowering ``alpha`` is always safe. Raising it is refused when any recorded
    time is censored, since that run's true time was never observed and keeping
    it at 0 would bias every larger-``alpha`` result downward. Measure once at
    the largest ``alpha`` you will report, then derive smaller ones here.
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
