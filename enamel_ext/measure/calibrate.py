"""A fixed workload timed once per session, so comparability is measured not assumed.

A uniform slowdown cancels in Eq. (1), so the quantity worth detecting is the
*differential* between two sessions, which one number cannot see: the probe is a
vector of workloads with different cost mixes and the statistic is the spread of
their ratios, judged against a resolution the probe measures for itself.
See docs/decisions/0011-calibration-probe.md.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

from enamel_ext.measure.sandbox import Limits, SandboxError, run_level
from enamel_ext.measure.timing import aggregate_repeats, hodges_lehmann

__all__ = [
    "CALIBRATION_VERSION",
    "DRIFT_CAVEAT",
    "DRIFT_REFUSE",
    "REPLICATES",
    "WORKLOADS",
    "Calibration",
    "Drift",
    "compare",
    "differential",
    "probe",
    "uniform_factor",
]

#: Bumped whenever a workload or a scale changes. Two probes of different
#: versions time different work, so comparing them is refused rather than
#: approximated.
CALIBRATION_VERSION = 1

#: Independent timings per workload. Even, so :meth:`Calibration.resolution` can
#: split them into equal halves. 8 rather than 6: at 6 this VM's own probe pairs
#: fired on their own noise 0.6% of the time and at 8 they did not fire at all,
#: measured over 480 samples in docs/decisions/0011-calibration-probe.md.
REPLICATES = 8

#: Differential factors, from decision 0007's ``differential_bound``: at
#: ``alpha = 2`` a factor ``c`` moves a level fraction by up to ``2(c - 1)``, so
#: 1.025 is where drift could consume the whole 0.05 parity tolerance and 1.05 is
#: where it could consume twice it. Both are floors, not the thresholds applied:
#: a probe that cannot resolve them says so instead of firing on its own noise.
#: ``tests/test_calibrate.py`` asserts both against ``differential_bound``.
DRIFT_CAVEAT = 1.025
DRIFT_REFUSE = 1.05

_LOOP_ARITH = """
def workload(n):
    state = 12345
    total = 0
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        total += state >> 16
    return total
"""

_BULK_BUILTIN = """
def workload(n):
    blob = b"the quick brown fox jumps over the lazy dog " * n
    flip = bytes(range(255, -1, -1))
    total = blob.count(b"o")
    total += len(blob.translate(flip))
    total += sum(sorted(blob[: 8 * n]))
    return total
"""

_ALLOC_CHURN = """
def workload(n):
    total = 0
    keep = None
    for i in range(n):
        cell = [i, i + 1, i + 2]
        keep = (cell, keep) if i & 7 else None
        total += cell[2]
    return total
"""

_ATTR_CALL = """
class Cell:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def bump(self, delta):
        self.value += delta
        return self.value


def workload(n):
    cell = Cell(0)
    total = 0
    for _ in range(n):
        total += cell.bump(1)
    return total
"""

#: Name to (source, scale). Four cost mixes, because the differential worth
#: detecting is between the interpreter's eval loop and everything else:
#: interpreted integer arithmetic, C-level bulk operations, allocation and GC,
#: and call dispatch. Scales are fixed rather than adapted to the machine, or two
#: sessions would time different work and the ratio would mean nothing; they are
#: chosen so each workload lands in the same 10-15ms band.
WORKLOADS: Mapping[str, tuple[str, int]] = {
    "loop_arith": (_LOOP_ARITH, 120_000),
    "bulk_builtin": (_BULK_BUILTIN, 30_000),
    "alloc_churn": (_ALLOC_CHURN, 120_000),
    "attr_call": (_ATTR_CALL, 120_000),
}


@dataclass(frozen=True)
class Calibration:
    """Replicate timings per workload, under the settings the run will use."""

    times: Mapping[str, tuple[float, ...]]
    repeats: int
    aggregator: str
    version: int = CALIBRATION_VERSION

    def __post_init__(self) -> None:
        frozen = {
            str(name): tuple(float(t) for t in series)
            for name, series in self.times.items()
        }
        if not frozen:
            raise ValueError("a calibration needs at least one workload")
        widths = {len(series) for series in frozen.values()}
        if len(widths) != 1:
            raise ValueError(f"workloads have unequal replicate counts: {sorted(widths)}")
        if widths == {0} or min(widths) < 2:
            raise ValueError(
                "a calibration needs at least 2 replicates per workload, so it can "
                "state its own resolution"
            )
        for name, series in frozen.items():
            for t in series:
                if not t > 0:
                    raise ValueError(
                        f"workload {name!r} timed at {t}; a probe must be timeable"
                    )
        object.__setattr__(self, "times", frozen)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.times))

    @property
    def replicates(self) -> int:
        return len(self.times[self.names[0]])

    def location(self) -> Mapping[str, float]:
        """Per-workload speed estimate: Hodges-Lehmann over the replicates.

        The paper's estimator, one level up: it runs inside each replicate over
        repeats, and again here across replicates. The minimum is the more obvious
        choice, since contention only ever makes a timing slower, but what matters
        is the stability of a *ratio* between two probes rather than the bias of
        either one, and on this machine the minimum was the worst of the three
        candidates by that measure. See decision 0011.
        """
        return {name: hodges_lehmann(series) for name, series in self.times.items()}

    def resolution(self) -> float:
        """Finest differential distinguishable from this machine's own noise.

        The same statistic as :func:`differential`, computed between disjoint equal
        halves of this probe's own replicates, where the true answer is known to be
        1. Taken as the worst such split: overstating the instrument's noise costs
        sensitivity, understating it invents drift.
        """
        half = self.replicates // 2
        names = self.names
        indices = range(self.replicates)
        worst = 1.0
        for left in itertools.combinations(indices, half):
            rest = [i for i in indices if i not in left]
            # Both sides the same width, and the same estimator as location(), or
            # the split itself would be a differential. An odd replicate count
            # leaves one out of each half rather than comparing 1 against 2.
            for right in itertools.combinations(rest, half):
                if left > right:
                    continue
                a = {n: hodges_lehmann([self.times[n][i] for i in left]) for n in names}
                b = {n: hodges_lehmann([self.times[n][i] for i in right]) for n in names}
                worst = max(worst, _spread(_ratios_of(a, b, names)))
        return worst

    def resolves_parity(self) -> bool:
        """Whether this probe is fine enough to see a parity-relevant differential."""
        return self.resolution() <= DRIFT_CAVEAT

    def comparable(self, other: Calibration) -> bool:
        """Whether the two timed the same work under the same settings.

        Replicate count is part of this: an estimator over 8 replicates and one
        over 12 are not the same estimator, so their ratio would carry a bias that
        is not a machine change.
        """
        return (
            self.version == other.version
            and self.names == other.names
            and self.repeats == other.repeats
            and self.aggregator == other.aggregator
            and self.replicates == other.replicates
        )


def _ratios_of(
    a: Mapping[str, float], b: Mapping[str, float], names: Sequence[str]
) -> tuple[float, ...]:
    return tuple(b[name] / a[name] for name in names)


def _spread(ratios: Sequence[float]) -> float:
    return max(ratios) / min(ratios)


def _ratios(a: Calibration, b: Calibration) -> tuple[float, ...]:
    if not a.comparable(b):
        raise ValueError(
            f"calibrations are not comparable: v{a.version} {a.names} "
            f"{a.repeats}x{a.aggregator} over {a.replicates} replicates, then "
            f"v{b.version} {b.names} {b.repeats}x{b.aggregator} over "
            f"{b.replicates} replicates"
        )
    return _ratios_of(a.location(), b.location(), a.names)


def uniform_factor(a: Calibration, b: Calibration) -> float:
    """How much slower ``b``'s machine is overall: the geometric mean of ratios.

    This is the component that cancels in Eq. (1), since ``T_i`` is set by a
    reference measured on the same machine as its candidates. Reported so a reader
    can see that a much slower session is not thereby an incomparable one.
    """
    ratios = _ratios(a, b)
    product = 1.0
    for ratio in ratios:
        product *= ratio
    return product ** (1.0 / len(ratios))


def differential(a: Calibration, b: Calibration) -> float:
    """Spread of the per-workload ratios: ``max / min``, at least 1.

    The part of a machine change that does *not* cancel. One workload could only
    ever report the uniform factor, which is the harmless case; it takes two with
    different cost mixes to see that one kind of code sped up relative to another.
    Being a range, it is biased upward, which is why it is judged against
    :meth:`Calibration.resolution` rather than against a constant.
    """
    return _spread(_ratios(a, b))


@dataclass(frozen=True)
class Drift:
    """What two probes say about whether their sessions are commensurable."""

    factor: float
    uniform: float
    resolution: float

    @property
    def caveat_at(self) -> float:
        return max(DRIFT_CAVEAT, self.resolution)

    @property
    def refuse_at(self) -> float:
        return max(DRIFT_REFUSE, self.resolution)

    @property
    def caveat(self) -> bool:
        return self.factor > self.caveat_at

    @property
    def refuse(self) -> bool:
        return self.factor > self.refuse_at

    @property
    def resolves_parity(self) -> bool:
        """Whether the pair could have seen a parity-relevant differential at all."""
        return self.resolution <= DRIFT_CAVEAT


def compare(a: Calibration, b: Calibration) -> Drift:
    """Judge a measured differential against what the two probes can resolve.

    The thresholds are floors: a machine whose own replicate noise exceeds them
    reports that it cannot see them, rather than reporting drift it cannot
    distinguish from itself. On this 2-core VM the resolution has a median near
    1.30 and does not tighten with more replicates, so only gross drift is
    detectable; the point of carrying the number is that the report can say which
    of those two situations produced a quiet verdict. The coarser of the two
    probes sets the threshold, because the resolution is a property of the moment
    a probe was taken rather than a constant of the machine.
    """
    return Drift(
        factor=differential(a, b),
        uniform=uniform_factor(a, b),
        resolution=max(a.resolution(), b.resolution()),
    )


def probe(
    repeats: int = 6,
    aggregator: str = "hodges_lehmann",
    limits: Limits = Limits(),
    replicates: int = REPLICATES,
) -> Calibration:
    """Time every workload through the same path a candidate takes.

    Runs in the sandbox with the run's own repeats and aggregator, so the numbers
    come off the same clock, through the same process setup, as every ``t`` and
    ``t*``. Replicates are interleaved rather than blocked per workload: a machine
    that slows down partway through a blocked probe would charge that slowdown to
    whichever workload was running, which is a differential that is not real.
    Raises :class:`SandboxError` if a workload does not run, since nothing here is
    untrusted and a failure is the harness rather than the code.
    """
    if replicates < 2:
        raise ValueError(f"replicates={replicates}; a probe must state its resolution")
    times: dict[str, list[float]] = {name: [] for name in WORKLOADS}
    for _ in range(replicates):
        for name, (code, scale) in WORKLOADS.items():
            result = run_level(
                code,
                "workload",
                inputs=[(scale,)],
                repeats=repeats,
                aggregator=aggregator,
                limits=limits,
            )
            if not result.ok or not result.cases:
                raise SandboxError(
                    f"calibration workload {name!r} did not run: "
                    f"{result.status} {result.detail} {result.stderr}".strip()
                )
            elapsed = aggregate_repeats(result.cases[0].times, aggregator)
            if not elapsed > 0:
                raise SandboxError(f"calibration workload {name!r} timed at {elapsed}")
            times[name].append(elapsed)
    return Calibration(
        times={name: tuple(series) for name, series in times.items()},
        repeats=repeats,
        aggregator=aggregator,
    )
