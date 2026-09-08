"""Analyze the q distribution across levels: the §2.2 question.

Measures q = t*(level l) / t*(level 3) across all problems, reports the
distribution, and compares against the §2.2 predictions.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.schema import GeneratedLevel, ProblemSet
from enamel_ext.data.sources import problem_set_from_json
from enamel_ext.measure.runner import RunConfig
from enamel_ext.measure.sandbox import run_level
from enamel_ext.measure.timing import aggregate_repeats
from enamel_ext.metrics.score import PAPER
from enamel_ext.report.levels import (
    PAPER_SLOWDOWNS,
    describe_levels,
    limit_level_counts,
    q_distribution,
    q_ratios,
    sensitivity_shares,
)


def measure_references(pset: ProblemSet, config: RunConfig) -> list[list[list[float]]]:
    """Measure reference times for all problems and levels."""
    all_times = []
    for p in pset:
        level_times = []
        try:
            for lvl in p.levels[1:]:
                from enamel_ext.data.cases import materialize_level
                cases = materialize_level(p, lvl)
                case_times = []
                for args in cases:
                    result = run_level(
                        p.reference_solution,
                        p.entry_point,
                        inputs=[args],
                        repeats=config.repeats,
                        aggregator=config.aggregator,
                    )
                    if result.ok and result.cases:
                        case_times.append(aggregate_repeats(result.cases[0].times, config.aggregator))
                    else:
                        case_times.append(float("inf"))
                level_times.append(case_times)
        except Exception:
            level_times = [[float("inf")]] * (len(p.levels) - 1)
        all_times.append(level_times)
    return all_times


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()

    if args.problems:
        pset = problem_set_from_json(args.problems.read_text())
    else:
        from enamel_ext.data.sources import synthetic_problem_set
        pset = synthetic_problem_set()

    if args.limit > 0:
        from enamel_ext.data.schema import ProblemSet as PS, Provenance
        pset = PS(provenance=pset.provenance, problems=pset.problems[:args.limit])

    config = RunConfig(repeats=args.repeats)
    print(f"Measuring {len(pset)} problems, R={config.repeats}...")
    all_times = measure_references(pset, config)

    valid = [(i, t) for i, t in enumerate(all_times) if not any(float("inf") in c for c in t)]
    if len(valid) < len(all_times):
        print(f"  {len(all_times) - len(valid)} problems had failures, using {len(valid)}")

    times_only = [t for _, t in valid]

    print(f"\n{'='*60}")
    print("Q-DISTRIBUTION ANALYSIS (§2.2)")
    print(f"{'='*60}")

    columns = q_distribution(times_only)
    print(f"\nPer-level q (ratio of worst reference time to level 3's):")
    print(f"  {'Level':>6} {'Min':>8} {'Median':>8} {'Max':>8} {'Mean':>8} {'Stdev':>8}")
    for idx, col in enumerate(columns, start=1):
        print(f"  {idx:>6} {min(col):>8.4f} {statistics.median(col):>8.4f} {max(col):>8.4f} {statistics.mean(col):>8.4f} {statistics.stdev(col):>8.4f}")

    print(f"\nTolerance: slowdown at which level scores 0 (alpha/q at median q):")
    for idx, col in enumerate(columns, start=1):
        med = statistics.median(col)
        tol = PAPER.alpha / med
        print(f"  Level {idx}: {tol:>8.1f}x (median q={med:.4f})")

    print(f"\nSensitivity shares (level's part of score response to slowdown):")
    medians = [statistics.median(c) for c in columns]
    shares = sensitivity_shares(medians)
    for idx, share in enumerate(shares, start=1):
        print(f"  Level {idx}: {share:>6.1%}")

    print(f"\nT_i set by level:")
    counts = limit_level_counts(times_only)
    for level in sorted(counts):
        print(f"  Level {level}: {counts[level]} problems")

    print(f"\nScore at various slowdowns (at median q):")
    from enamel_ext.report.levels import level_fraction_at
    print(f"  {'Level':>6} {'q':>8} {'2x':>8} {'5x':>8} {'10x':>8} {'50x':>8}")
    for idx, col in enumerate(columns, start=1):
        med = statistics.median(col)
        scores = [level_fraction_at(med, s) for s in PAPER_SLOWDOWNS]
        print(f"  {idx:>6} {med:>8.4f} {scores[0]:>8.3f} {scores[1]:>8.3f} {scores[2]:>8.3f} {scores[3]:>8.3f}")

    print(f"\n§2.2 prediction check:")
    print(f"  Level 1 median q={medians[0]:.4f}: code 10x slower scores {level_fraction_at(medians[0], 10):.3f}")
    print(f"  Level 3 median q={medians[2]:.4f}: code 1.25x slower scores {level_fraction_at(medians[2], 1.25):.3f}")
    print(f"  60% of weight at levels 1-2 (h1+h2=6 of 10) where discrimination is weakest")
    print(f"  Sensitivity share at level 3: {shares[2]:.1%}")

    q_spread = [max(c)/min(c) for c in columns if min(c) > 0]
    print(f"\n  q spread (max/min) per level: {[f'{s:.1f}' for s in q_spread]}")
    print(f"  Wide spread means the level discriminates differently across problems")


if __name__ == "__main__":
    main()
