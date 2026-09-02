# Recovering per-level scores from Table 10

Derived 2026-09-01 from arXiv:2406.06647v4, Appendix C.6, Table 10. No new
measurement — this is arithmetic on published numbers. Reproduce with
`scripts/recover_table10.py`.

## The trick

At `k = 1` the efficiency score is exactly linear in the level hardnesses:

```
eff@1(h) = (h₁F₁ + h₂F₂ + h₃F₃) / (h₁ + h₂ + h₃)
```

where `F_l` is the mean of `f[i,j,l]` over all samples and problems (a sample
that fails correctness contributes 0 to every level). Table 10 sweeps each `h_l`
from 1 to 5 while holding the other two at their defaults. Multiplying each
reported `eff@1` by its denominator `Σh` recovers the numerator, and first
differences along a sweep isolate `F_l` directly.

The `k = 1` restriction matters: above it, the `eff@k` weights attach to order
statistics whose ordering moves with `h`, so the score is piecewise linear and
this inversion does not apply. Every published sweep is `eff@1`, so nothing here
is lost. See docs/decisions/0002-reporting-layer.md.

## Result, for GPT-4 Turbo under greedy decoding

| level | mean score `F_l` | per-step estimates from the sweep |
|---|---|---|
| 1 | 0.638 | 0.635, 0.641, 0.646, 0.630 |
| 2 | 0.453 | 0.456, 0.452, 0.459, 0.445 |
| 3 | 0.355 | 0.352, 0.355, 0.353, 0.360 |

Rebuilding the headline from these gives `eff@1 = 0.4693` against the published
0.470, and the four independent estimates within each sweep agree to ±0.008 —
about what rounding to three decimals in the table should produce. The recovery
is sound.

Dividing by `pass@1 = 0.796` bounds the mean score among samples that count as
correct: at most 0.802 on level 1, 0.569 on level 2, 0.446 on level 3.

## What this settles

**The extreme form of the score-compression concern (README §2.2) does not
hold.** The worst case in that analysis says that when a level's reference time
is a small fraction `q` of level 3's, the single per-problem limit `Tᵢ` leaves
that level nearly constant: at `q = 0.01` every candidate within 10× of the
expert scores between 0.955 and 1.005. If level 1 behaved that way for most
problems, `F₁` would sit just under `pass@1 = 0.796`. It is 0.638. So level 1
does discriminate.

**What it does not settle is why.** `F₁ = 0.638` is equally consistent with two
stories: `q` is simply not small for most problems, or `q` is small and roughly a
fifth of the samples counted in `pass@1` time out at level 1 (which the paper
permits — level 0 is the correctness gate, and a timeout at level `l ≥ 1` still
counts toward `pass@k` while zeroing the score). Those have different
implications and cannot be separated from published aggregates. Measuring the
distribution of `q` remains worth doing; the expected finding is now "moderate"
rather than "severe."

## What it opens up

Since `eff@1(h)` is a convex combination of `F₁, F₂, F₃`, reweighting `h` over
all non-negative vectors moves the headline anywhere in `[min F_l, max F_l]` —
for this model, **[0.355, 0.638], a span of 0.283**. Compare: the `α` sweep from
1.5 to 3.5 spans 0.120, and the entire gap between the best and fourth-best
model in the paper spans 0.062. Both hyperparameters have more leverage on the
number than the choice of model does.

That is not by itself a criticism: the paper is explicit that `α` and `h` encode
user preference, tells users to keep the defaults, and reports the sensitivity.
The gap is that **every entry in Table 10 is one model**, so nothing in the paper
speaks to whether the *ranking* survives the sweep — and the ranking is what
readers take away.

Linearity makes that decidable exactly, with no extra compute:

> Model A ranks above model B for every non-negative `h` if and only if
> `F^A_l ≥ F^B_l` for all three levels. If the per-level vectors cross, some
> admissible `h` flips the pair.

So rank robustness under `h` reduces to comparing three numbers per model. Once
the harness records per-level means, this is a table, not an experiment. The `α`
sweep is not free in the same way: raising `α` un-censors runs whose times were
never observed, so the harness has to measure once with a generous cap and derive
every smaller `α` by post-processing.
