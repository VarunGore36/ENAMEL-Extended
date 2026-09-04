"""Our scores against the paper's, and what the published numbers can resolve.

Milestone 2's gate. A model pair the published table separates by less than the
tolerance is not evidence either way, so ordering is judged on the pairs it does
separate and the count is reported alongside. Rationale in
docs/decisions/0007-parity-gate.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, NamedTuple, Sequence

from enamel_ext.data.published import (
    ALPHA,
    COLUMNS,
    PAPER,
    TABLE7_EFF1_RANKING,
    TABLE7_SPEEDUP_RANKING,
    table,
)
from enamel_ext.report.stats import kendall_tau

__all__ = [
    "EFF_TOLERANCE",
    "INVERSION_MARGIN",
    "PASS_TOLERANCE",
    "Deviation",
    "Inversion",
    "PairResolution",
    "ParityResult",
    "TauFloor",
    "compare",
    "deviations",
    "differential_bound",
    "format_parity",
    "inversions",
    "published_disagreement_tau",
    "ranking_tau",
    "resolution",
    "resolvable_pairs",
    "tau_floor",
]

#: Pre-committed in decision 0007, before any measurement existed. ``eff`` is
#: loose because it carries the machine; ``pass`` is tight because it does not.
EFF_TOLERANCE = 0.05
PASS_TOLERANCE = 0.01

#: Two models each within ``tolerance`` of their published values can only swap
#: if the published gap is under twice it, so only wider pairs are gated.
INVERSION_MARGIN = 2.0


def _ranked(values: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda row: -row[1])


def _check_tolerance(tolerance: float) -> None:
    if not tolerance >= 0.0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")


def resolvable_pairs(
    published: Mapping[str, float], tolerance: float
) -> tuple[tuple[str, str], ...]:
    """Model pairs the published values separate by more than ``tolerance``.

    Each pair is ``(better, worse)`` by published value. Pairs closer than the
    tolerance are omitted: our ordering of them is neither confirmed nor
    contradicted by a table we only claim to match this closely.
    """
    _check_tolerance(tolerance)
    ranked = _ranked(published)
    return tuple(
        (a, b)
        for (a, va), (b, vb) in combinations(ranked, 2)
        if va - vb > tolerance
    )


@dataclass(frozen=True)
class PairResolution:
    """How much of the published leaderboard a tolerance can still test."""

    models: int
    pairs: int
    resolvable: int
    adjacent: int
    adjacent_resolvable: int
    tolerance: float

    @property
    def share(self) -> float:
        """Fraction of all pairs that are resolvable, 0.0 when there are none."""
        return self.resolvable / self.pairs if self.pairs else 0.0

    @property
    def adjacent_share(self) -> float:
        return (
            self.adjacent_resolvable / self.adjacent if self.adjacent else 0.0
        )


def resolution(
    published: Mapping[str, float], tolerance: float = EFF_TOLERANCE
) -> PairResolution:
    """Count the pairs a tolerance leaves as evidence, overall and adjacent.

    The two counts answer different questions. The overall share is the gate's
    power; the adjacent share is whether it says anything about neighbours, which
    is what a leaderboard is read for.
    """
    _check_tolerance(tolerance)
    ranked = _ranked(published)
    gaps = [ranked[i][1] - ranked[i + 1][1] for i in range(len(ranked) - 1)]
    return PairResolution(
        models=len(ranked),
        pairs=len(ranked) * (len(ranked) - 1) // 2,
        resolvable=len(resolvable_pairs(published, tolerance)),
        adjacent=len(gaps),
        adjacent_resolvable=sum(1 for gap in gaps if gap > tolerance),
        tolerance=tolerance,
    )


class TauFloor(NamedTuple):
    """The rank correlation a maximally-locally-wrong result still reports."""

    tau: float
    inverted: int
    adjacent: int


def tau_floor(
    published: Mapping[str, float], tolerance: float = EFF_TOLERANCE
) -> TauFloor:
    """Kendall tau after inverting as many near-tied adjacent pairs as possible.

    A tau over the whole table is dominated by the far-apart pairs, so it stays
    high even when every contested ordering is wrong. This builds that worst case
    and measures it, which is why tau is reported here as a diagnostic and not
    used as a criterion.
    """
    _check_tolerance(tolerance)
    ranked = _ranked(published)
    if len(ranked) < 2:
        raise ValueError("need at least two models to rank")
    values = [value for _, value in ranked]
    ours = list(values)
    inverted, last = 0, -2
    for i in range(len(values) - 1):
        if values[i] - values[i + 1] <= tolerance and i > last + 1:
            ours[i], ours[i + 1] = ours[i + 1], ours[i]
            inverted, last = inverted + 1, i
    return TauFloor(kendall_tau(values, ours), inverted, len(values) - 1)


def ranking_tau(first: Sequence[str], second: Sequence[str]) -> float:
    """Kendall tau between two orderings of the same models, each best first."""
    if set(first) != set(second):
        raise ValueError("rankings must cover the same models")
    if len(first) < 2:
        raise ValueError("need at least two models to rank")
    place = {model: -index for index, model in enumerate(second)}
    return kendall_tau(
        [-index for index in range(len(first))], [place[model] for model in first]
    )


def published_disagreement_tau() -> float:
    """Tau between Table 7's own two rankings of the same twelve models.

    The paper calls the pair 'very different' and argues from the difference that
    the classic speedup metric is unreasonable under censoring, so this is the
    rank correlation that accompanies a disagreement its authors treat as
    disqualifying. It is the published counterpart to ``tau_floor``, and the
    reason neither is a criterion.
    """
    return ranking_tau(TABLE7_EFF1_RANKING, TABLE7_SPEEDUP_RANKING)


class Deviation(NamedTuple):
    """One model's distance from its published value, signed ours minus theirs."""

    model: str
    ours: float
    published: float
    delta: float
    within: bool


def deviations(
    ours: Mapping[str, float],
    published: Mapping[str, float],
    tolerance: float = EFF_TOLERANCE,
) -> tuple[Deviation, ...]:
    """Per-model deviations for the models both sides have, worst first.

    Sorted by magnitude rather than by rank so that reading stops at the first
    row that is in tolerance.
    """
    _check_tolerance(tolerance)
    rows = [
        Deviation(
            model,
            ours[model],
            published[model],
            ours[model] - published[model],
            abs(ours[model] - published[model]) <= tolerance,
        )
        for model in ours
        if model in published
    ]
    return tuple(sorted(rows, key=lambda row: (-abs(row.delta), row.model)))


class Inversion(NamedTuple):
    """A resolvable pair whose published order we did not reproduce."""

    better: str
    worse: str
    published_gap: float
    our_gap: float
    gated: bool


def inversions(
    ours: Mapping[str, float],
    published: Mapping[str, float],
    tolerance: float = EFF_TOLERANCE,
    margin: float = INVERSION_MARGIN,
) -> tuple[Inversion, ...]:
    """Resolvable pairs we order the other way round, widest published gap first.

    ``gated`` marks the pairs a passing deviation check cannot excuse: two models
    each within ``tolerance`` can swap only across a gap under ``margin`` times
    it, so an inversion wider than that means the two criteria disagree.
    """
    _check_tolerance(tolerance)
    if not margin >= 1.0:
        raise ValueError(f"margin must be >= 1, got {margin}")
    out = []
    for better, worse in resolvable_pairs(published, tolerance):
        if better not in ours or worse not in ours:
            continue
        our_gap = ours[better] - ours[worse]
        if our_gap >= 0.0:
            continue
        gap = published[better] - published[worse]
        out.append(Inversion(better, worse, gap, our_gap, gap > margin * tolerance))
    return tuple(sorted(out, key=lambda row: -row.published_gap))


def differential_bound(factor: float, alpha: float = ALPHA) -> float:
    """Most a per-level fraction can move if our clock is ``factor`` times theirs.

    Eq. (1) in units of the limit-setting reference time gives
    ``f = (alpha - s q) / (alpha - q)`` for a candidate ``s`` times slower than a
    reference whose worst case is ``q <= 1`` of that constant. Scaling ``s`` by
    ``factor`` moves ``f`` by at most ``(factor - 1) alpha / (alpha - 1)``, at
    ``q = 1`` and the largest ``s`` the level still scores. A uniform slowdown
    cancels, since ``T_i`` is set by the reference; only a differential one
    survives, and only its systematic part survives averaging over problems.
    """
    if not factor >= 1.0:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if not alpha > 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    return (factor - 1.0) * alpha / (alpha - 1.0)


@dataclass(frozen=True)
class ParityResult:
    """Everything the gate looked at, and whether it passed."""

    name: str
    eff: tuple[Deviation, ...]
    passes: tuple[Deviation, ...]
    resolution: PairResolution
    inverted: tuple[Inversion, ...]
    tau: float | None
    floor: TauFloor | None
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    eff_tolerance: float
    pass_tolerance: float

    @property
    def eff_misses(self) -> tuple[Deviation, ...]:
        return tuple(row for row in self.eff if not row.within)

    @property
    def pass_misses(self) -> tuple[Deviation, ...]:
        return tuple(row for row in self.passes if not row.within)

    @property
    def gated_inversions(self) -> tuple[Inversion, ...]:
        return tuple(row for row in self.inverted if row.gated)

    @property
    def compared(self) -> int:
        """Models carrying both an eff and a pass comparison against the paper."""
        return min(len(self.eff), len(self.passes))

    @property
    def passed(self) -> bool:
        """All three criteria, over a comparison that had something in it.

        Still silent on how much coverage: a run over two models can pass this and
        is not parity, and ``missing`` is the number to read alongside. Zero
        overlap is refused outright, because criteria with nothing to check are
        vacuously satisfied rather than met. Rationale in decision 0007.
        """
        return bool(self.compared) and not (
            self.eff_misses or self.pass_misses or self.gated_inversions
        )


def _column(name: str, column: str) -> dict[str, float]:
    if column not in COLUMNS:
        raise ValueError(f"unknown column {column!r}, expected one of {COLUMNS}")
    index = COLUMNS.index(column)
    return {
        model: scores[index]
        for model, scores in table(name).items()
        if scores[index] is not None
    }


def _empty(tolerance: float) -> PairResolution:
    return PairResolution(0, 0, 0, 0, 0, tolerance)


def compare(
    eff: Mapping[str, float],
    pass_at_k: Mapping[str, float],
    *,
    name: str = "greedy",
    k: int = 1,
    eff_tolerance: float = EFF_TOLERANCE,
    pass_tolerance: float = PASS_TOLERANCE,
    margin: float = INVERSION_MARGIN,
) -> ParityResult:
    """Compare our ``eff@k`` and ``pass@k`` against one published table.

    ``k`` selects the published column, so only the values the paper prints for
    that table are available. Models we did not run are reported as ``missing``
    rather than treated as agreeing, and the pair counts describe the models
    compared rather than the whole table, so a subset run cannot borrow the full
    table's discriminating power.
    """
    published_eff = _column(name, f"eff{k}")
    published_pass = _column(name, f"pass{k}")
    ours = {model: eff[model] for model in eff if model in published_eff}
    covered = {model: published_eff[model] for model in ours}
    # A run where every model scored the same leaves tau undefined, which is a
    # result to report rather than an error: a harness that scores nothing does.
    rankable = len(ours) >= 2 and len(set(ours.values())) > 1
    tau = None
    if rankable:
        ranked = _ranked(covered)
        tau = kendall_tau(
            [value for _, value in ranked], [ours[model] for model, _ in ranked]
        )
    return ParityResult(
        name=name,
        eff=deviations(eff, published_eff, eff_tolerance),
        passes=deviations(pass_at_k, published_pass, pass_tolerance),
        resolution=resolution(covered, eff_tolerance) if ours else _empty(eff_tolerance),
        inverted=inversions(eff, published_eff, eff_tolerance, margin),
        tau=tau,
        floor=tau_floor(covered, eff_tolerance) if len(ours) >= 2 else None,
        missing=tuple(m for m in published_eff if m not in eff),
        extra=tuple(m for m in eff if m not in published_eff),
        eff_tolerance=eff_tolerance,
        pass_tolerance=pass_tolerance,
    )


def format_parity(result: ParityResult, limit: int = 8) -> list[str]:
    """The parity section, as report lines. At most ``limit`` rows per table."""
    res = result.resolution
    published = len(result.eff) + len(result.missing)
    out = [
        f"Parity against {PAPER}, {result.name} table",
        f"  compared {len(result.eff)} of {published} models"
        f"{f', {len(result.missing)} not run' if result.missing else ''}"
        f"{f', {len(result.extra)} not published' if result.extra else ''}",
        f"  eff tolerance {result.eff_tolerance:.3f}: "
        f"{res.resolvable}/{res.pairs} pairs of those resolvable "
        f"({100 * res.share:.1f}%), "
        f"{res.adjacent_resolvable}/{res.adjacent} adjacent",
        f"  verdict: {'pass' if result.passed else 'FAIL'} "
        f"({len(result.eff_misses)} eff, {len(result.pass_misses)} pass, "
        f"{len(result.gated_inversions)} order)"
        f"{'' if result.compared else ', nothing compared'}",
    ]
    if result.tau is not None and result.floor is not None:
        out.append(
            f"  kendall tau {result.tau:.3f}, but inverting "
            f"{result.floor.inverted} of {result.floor.adjacent} adjacent pairs "
            f"still reports {result.floor.tau:.3f}, so tau is not a criterion"
        )
    for label, rows, tolerance in (
        ("eff", result.eff, result.eff_tolerance),
        ("pass", result.passes, result.pass_tolerance),
    ):
        shown = [row for row in rows if not row.within][:limit] or rows[:1]
        for row in shown:
            flag = "" if row.within else "  over"
            out.append(
                f"    {label} {row.model}: ours {row.ours:.3f}, "
                f"published {row.published:.3f}, {row.delta:+.3f}{flag}"
            )
    for row in result.inverted[:limit]:
        out.append(
            f"    order {row.better} vs {row.worse}: published "
            f"{row.published_gap:+.3f}, ours {row.our_gap:+.3f}"
            f"{'  gated' if row.gated else ''}"
        )
    return out
