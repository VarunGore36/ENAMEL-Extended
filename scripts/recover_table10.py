"""Recover the per-level mean efficiency scores F_1, F_2, F_3 from Table 10.

Appendix C.6 of arXiv:2406.06647v4 sweeps each h_l from 1 to 5 for GPT-4 Turbo.
Since eff@1(h) = sum(h_l F_l) / sum(h_l) is linear in h, first differences along
one sweep isolate that level's mean. See docs/analysis/table10-recovery.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.published import (  # noqa: E402
    LEVEL_WEIGHTS,
    TABLE10_ALPHA,
    TABLE10_HARDNESS,
    TABLE3_GREEDY,
    leaderboard,
)

# Table 10 (b), (c), (d): eff@1 as one hardness varies, the others at defaults.
# Copied rather than aliased so a caller poking at SWEEPS cannot edit the table.
SWEEPS: dict[int, dict[int, float]] = {
    level: dict(row) for level, row in TABLE10_HARDNESS.items()
}
# Sum of the two hardnesses held fixed during each sweep (defaults are 3, 3, 4).
FIXED_SUM: dict[int, int] = {
    level: sum(LEVEL_WEIGHTS) - LEVEL_WEIGHTS[level - 1] for level in SWEEPS
}

PUBLISHED_EFF1 = TABLE3_GREEDY["GPT-4 Turbo"].eff1
PUBLISHED_PASS1 = TABLE3_GREEDY["GPT-4 Turbo"].pass1
# Table 10 (a): eff@1 at alpha = 1.5, 2.0, 2.5, 3.0, 3.5.
ALPHA_SPAN = max(TABLE10_ALPHA.values()) - min(TABLE10_ALPHA.values())
# Best minus fourth-best in the main leaderboard, both greedy.
TOP4_SPAN = leaderboard()[0][1] - leaderboard()[3][1]


def recover(level: int) -> tuple[float, list[float]]:
    """Return (mean estimate, per-step estimates) of F_level."""
    table = SWEEPS[level]
    fixed = FIXED_SUM[level]
    hs = sorted(table)
    steps = [
        table[b] * (b + fixed) - table[a] * (a + fixed)
        for a, b in zip(hs, hs[1:])
    ]
    return sum(steps) / len(steps), steps


def main() -> None:
    f = {}
    print("recovered mean level scores (GPT-4 Turbo, greedy):")
    for level in (1, 2, 3):
        mean, steps = recover(level)
        f[level] = mean
        pretty = ", ".join(f"{s:.3f}" for s in steps)
        print(f"  F{level} = {mean:.3f}   [per-step: {pretty}]")

    rebuilt = (3 * f[1] + 3 * f[2] + 4 * f[3]) / 10
    print(f"\nrebuilt eff@1 at h=(3,3,4): {rebuilt:.4f}  (published {PUBLISHED_EFF1})")
    assert abs(rebuilt - PUBLISHED_EFF1) < 0.002, "recovery is inconsistent with the paper"

    print(f"\nupper bound on the mean score among samples counted in pass@1 "
          f"({PUBLISHED_PASS1}):")
    for level in (1, 2, 3):
        print(f"  level {level}: <= {f[level] / PUBLISHED_PASS1:.3f}")

    span = max(f.values()) - min(f.values())
    print(f"\neff@1 reachable by reweighting h alone: "
          f"[{min(f.values()):.3f}, {max(f.values()):.3f}]  span {span:.3f}")
    print(f"  alpha sweep (1.5 -> 3.5) span:        {ALPHA_SPAN:.3f}")
    print(f"  best-to-fourth-best model span:       {TOP4_SPAN:.3f}")


if __name__ == "__main__":
    main()
