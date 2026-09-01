"""Recover the per-level mean efficiency scores F_1, F_2, F_3 from Table 10.

Appendix C.6 of arXiv:2406.06647v4 sweeps each level hardness h_l from 1 to 5
for GPT-4 Turbo. Because

    eff@1(h) = (h_1 F_1 + h_2 F_2 + h_3 F_3) / (h_1 + h_2 + h_3)

is exactly linear in h, multiplying each reported eff@1 by its denominator
recovers the numerator, and first differences along one sweep isolate that
level's mean score. No measurement required.

Writes nothing; prints the recovery and its self-consistency check. See
docs/analysis/table10-recovery.md for what it means.
"""

from __future__ import annotations

# Table 10 (b), (c), (d): eff@1 as one hardness varies, the others at defaults.
SWEEPS: dict[int, dict[int, float]] = {
    1: {1: 0.428, 2: 0.451, 3: 0.470, 4: 0.486, 5: 0.498},
    2: {1: 0.474, 2: 0.472, 3: 0.470, 4: 0.469, 5: 0.467},
    3: {1: 0.520, 2: 0.499, 3: 0.483, 4: 0.470, 5: 0.460},
}
# Sum of the two hardnesses held fixed during each sweep (defaults are 3, 3, 4).
FIXED_SUM: dict[int, int] = {1: 3 + 4, 2: 3 + 4, 3: 3 + 3}

PUBLISHED_EFF1 = 0.470
PUBLISHED_PASS1 = 0.796
# Table 10 (a): eff@1 at alpha = 1.5, 2.0, 2.5, 3.0, 3.5.
ALPHA_SPAN = 0.541 - 0.421
# Best (GPT-4 Turbo, 0.470) minus fourth-best (0.408) in the main leaderboard.
TOP4_SPAN = 0.470 - 0.408


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
    print("\nBoth hyperparameters have more leverage on the headline number than")
    print("the choice of model does. The paper reports this; what it does not")
    print("report is whether the ranking survives the sweep, since every entry")
    print("in Table 10 is a single model.")


if __name__ == "__main__":
    main()
