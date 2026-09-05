"""Measure this machine's calibration-probe floor: resolution, false alarms, power.

Collects many independent probes from disjoint blocks of one long sample run, where
the true differential between any two is known to be 1, and reports how often the
applied rule fires anyway. See docs/analysis/probe-floor.md.

Collection is resumable: each invocation tops the cache up toward --samples for at
most --budget seconds, so a long collection survives being interrupted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.measure.calibrate import (  # noqa: E402
    DRIFT_CAVEAT,
    REPLICATES,
    WORKLOADS,
)
from enamel_ext.measure.sandbox import SandboxError, run_level  # noqa: E402
from enamel_ext.measure.timing import (  # noqa: E402
    aggregate_repeats,
    hodges_lehmann,
)

#: Cross-replicate location estimators to compare. The shipped one is
#: ``hodges_lehmann``; see decision 0011 for why it is not ``min``.
ESTIMATORS: Mapping[str, Callable[[Sequence[float]], float]] = {
    "min": min,
    "median": statistics.median,
    "hodges_lehmann": hodges_lehmann,
}

#: Differentials injected into one workload to measure power.
INJECTED = (1.10, 1.20, 1.40, 2.00)

#: Replicate counts to compare. REPLICATES is the shipped default.
COUNTS = (4, 6, 8, 12, 16)

Samples = dict[str, list[float]]


def load(cache: Path) -> Samples:
    if cache.exists():
        return {str(k): [float(t) for t in v] for k, v in json.loads(cache.read_text()).items()}
    return {name: [] for name in WORKLOADS}


def collect(cache: Path, target: int, budget: float, repeats: int, aggregator: str) -> Samples:
    """Append whole samples until ``target`` is reached or ``budget`` seconds pass."""
    samples = load(cache)
    have = min(len(series) for series in samples.values())
    start = time.time()
    while have < target and time.time() - start < budget:
        for name, (code, scale) in WORKLOADS.items():
            result = run_level(
                code, "workload", inputs=[(scale,)], repeats=repeats, aggregator=aggregator
            )
            if not result.ok or not result.cases:
                raise SandboxError(f"{name}: {result.status} {result.detail} {result.stderr}")
            samples[name].append(aggregate_repeats(result.cases[0].times, aggregator))
        have += 1
    cache.write_text(json.dumps(samples))
    return samples


def differential(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    ratios = [b[name] / a[name] for name in a]
    return max(ratios) / min(ratios)


def blocks_of(samples: Samples, k: int) -> list[Mapping[str, list[float]]]:
    """Disjoint consecutive blocks of ``k`` samples, each one an independent probe."""
    names = sorted(samples)
    count = min(len(samples[name]) for name in names) // k
    return [{n: samples[n][g * k : (g + 1) * k] for n in names} for g in range(count)]


def half_splits(block: Mapping[str, Sequence[float]], locate) -> list[float]:
    """The differential over every pair of disjoint equal halves, where truth is 1."""
    names = sorted(block)
    k = len(block[names[0]])
    half = k // 2
    out = []
    for left in itertools.combinations(range(k), half):
        rest = [i for i in range(k) if i not in left]
        for right in itertools.combinations(rest, half):
            if left > right:
                continue
            a = {n: locate([block[n][i] for i in left]) for n in names}
            b = {n: locate([block[n][i] for i in right]) for n in names}
            out.append(differential(a, b))
    return out


def pct(values: Sequence[float], q: float) -> float:
    return sorted(values)[int(q * (len(values) - 1))]


def fired(
    locations: Sequence[Mapping[str, float]],
    resolutions: Sequence[float],
    pairs: Sequence[tuple[int, int]],
    injected: float = 1.0,
    moved: str = "loop_arith",
) -> int:
    """Pairs whose differential clears ``max(DRIFT_CAVEAT, res_i, res_j)``.

    At ``injected == 1.0`` every firing is a false alarm, since disjoint blocks of
    one collection have a true differential of 1.
    """
    count = 0
    for i, j in pairs:
        later = dict(locations[j])
        later[moved] *= injected
        threshold = max(DRIFT_CAVEAT, resolutions[i], resolutions[j])
        count += differential(locations[i], later) > threshold
    return count


def sample_spread(samples: Samples) -> None:
    print("A single sample's own spread, in ms:")
    print(f"{'workload':>14} {'min':>7} {'p50':>7} {'p90':>7} {'max':>7} {'max/min':>8}")
    for name in sorted(samples):
        xs = sorted(samples[name])
        print(
            f"{name:>14} {xs[0] * 1000:>7.2f} {statistics.median(xs) * 1000:>7.2f} "
            f"{pct(xs, 0.9) * 1000:>7.2f} {xs[-1] * 1000:>7.2f} {xs[-1] / xs[0]:>8.3f}"
        )


def by_estimator(samples: Samples) -> None:
    """The whole rule per (replicate count, estimator): its floor, its errors."""
    print("\nThe applied rule, with the resolution measured by the same estimator:")
    print(
        f"\n{'k':>3} {'estimator':>14} {'res p50':>8} {'res p90':>8} {'res max':>8} "
        f"{'diff p50':>9} {'diff max':>9} {'false':>11} {'adjacent':>9} "
        + " ".join(f"{'x%.2f' % s:>11}" for s in INJECTED)
    )
    for k in COUNTS:
        for name, locate in ESTIMATORS.items():
            blocks = blocks_of(samples, k)
            if len(blocks) < 3:
                continue
            res = [max(half_splits(b, locate)) for b in blocks]
            loc = [{n: locate(xs) for n, xs in b.items()} for b in blocks]
            pairs = list(itertools.combinations(range(len(blocks)), 2))
            near = [(i, i + 1) for i in range(len(blocks) - 1)]
            ds = [differential(loc[i], loc[j]) for i, j in pairs]
            print(
                f"{k:>3} {name:>14} {statistics.median(res):>8.4f} {pct(res, 0.9):>8.4f} "
                f"{max(res):>8.4f} {statistics.median(ds):>9.4f} {max(ds):>9.4f} "
                f"{fired(loc, res, pairs):>5}/{len(pairs):<5} "
                f"{fired(loc, res, near):>3}/{len(near):<5} "
                + " ".join(
                    f"{fired(loc, res, pairs, s):>5}/{len(pairs):<5}" for s in INJECTED
                )
            )


def by_split_rule(samples: Samples, locate=hodges_lehmann) -> None:
    """Worst split, or a quantile of them? Understating the floor invents drift."""
    print("\nWhich half-split becomes the resolution:")
    print(f"\n{'k':>3} {'split':>8} {'res p50':>8} {'false alarms':>16}")
    rules = {
        "median": statistics.median,
        "p90": lambda s: pct(s, 0.9),
        "worst": max,
    }
    for k in COUNTS:
        blocks = blocks_of(samples, k)
        if len(blocks) < 3:
            continue
        splits = [half_splits(b, locate) for b in blocks]
        loc = [{n: locate(xs) for n, xs in b.items()} for b in blocks]
        pairs = list(itertools.combinations(range(len(blocks)), 2))
        for label, rule in rules.items():
            res = [rule(s) for s in splits]
            count = fired(loc, res, pairs)
            print(
                f"{k:>3} {label:>8} {statistics.median(res):>8.4f} "
                f"{count:>6}/{len(pairs):<5} {100 * count / len(pairs):>5.1f}%"
            )


def by_quarter(samples: Samples, k: int = REPLICATES, locate=hodges_lehmann) -> None:
    """Is the floor a property of the machine, or of the moment it was measured?"""
    names = sorted(samples)
    total = min(len(samples[n]) for n in names)
    print(f"\nResolution by quarter of the collection window (k={k}):")
    print(f"\n{'quarter':>8} {'res p50':>8} {'res max':>8} {'probes':>7}")
    for q in range(4):
        cut = {n: samples[n][q * total // 4 : (q + 1) * total // 4] for n in names}
        blocks = blocks_of(cut, k)
        res = [max(half_splits(b, locate)) for b in blocks]
        print(
            f"{q + 1:>8} {statistics.median(res):>8.4f} {max(res):>8.4f} {len(blocks):>7}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=480)
    parser.add_argument("--budget", type=float, default=150.0, help="seconds per call")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--aggregator", default="hodges_lehmann")
    parser.add_argument("--cache", type=Path, default=Path("probe-floor.json"))
    parser.add_argument("--report", action="store_true", help="skip collection")
    args = parser.parse_args(argv)

    if args.report:
        samples = load(args.cache)
    else:
        samples = collect(
            args.cache, args.samples, args.budget, args.repeats, args.aggregator
        )
    have = min(len(series) for series in samples.values()) if samples else 0
    print(f"{have}/{args.samples} samples in {args.cache}")
    if have < args.samples and not args.report:
        print("run again to continue collecting, or pass --report to use what there is")
        return 0
    if have < max(COUNTS) * 3:
        print(f"too few samples to report; {max(COUNTS) * 3} is the minimum")
        return 1
    sample_spread(samples)
    by_estimator(samples)
    by_split_rule(samples)
    by_quarter(samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
