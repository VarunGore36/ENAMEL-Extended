# 0002 — Reporting layer: uncertainty, hyperparameter sensitivity, rank stability

Status: accepted. Supersedes nothing. Code: `enamel_ext/report/`, tests
`tests/test_stats.py`, `tests/test_hyperparams.py`, `tests/test_levels.py` and
`tests/test_recover_table10.py`.

Everything here is post-processing over recorded per-problem or per-level
numbers. Nothing in the package executes code or reads a clock, which is why it
could be built and tested before the benchmark data is available.

## The resampling unit is the problem, not the sample

The paper reports the estimator's standard deviation (Table 11: 0.02 at `k=1`
for the Rao–Blackwellized form) but not the between-problem standard error of
the mean over the 142 problems, and no test on differences between models.
Those two are what `report.stats` adds.

Sampling noise from finite `n` per problem enters the aggregate divided by
roughly `sqrt(142)`, so 0.02 per problem contributes on the order of 0.002 to
the mean — negligible next to problem-to-problem spread. A fully nested
bootstrap would resample samples within problems as well. Not implemented, on
purpose: resampling with replacement introduces ties into a statistic that
assumes distinct draws, which biases the inner variance in a direction that is
awkward to reason about and that would have to be corrected before it could be
reported.

## Pairing, not two independent intervals

Every model is scored on the same 142 problems, so problem difficulty is a
shared nuisance term that cancels in the difference. Differencing two
independent `bootstrap_ci` results answers a different question and is far too
conservative for a leaderboard claim.
`tests/test_stats.py::test_pairing_separates_models_that_unpaired_intervals_cannot`
pins this: two models whose individual intervals overlap heavily are separated
cleanly by the paired interval, with a width less than half of either.

## The sign test is exact where it can be

For `n <= 20` all `2**n` sign patterns are enumerated; above that they are
sampled and the observed arrangement is counted in the null (`at_least = 1`),
so the p-value is never exactly 0. Valid rather than merely unbiased, which
matters when the number will be quoted.

## Kendall tau-b, with tie correction

Used to ask whether a leaderboard survives a hyperparameter change. Ties are
common once scores are rounded to three decimals, and treating them as
concordant would inflate agreement. An entirely tied ranking raises rather than
returning 0 — tau is undefined there, and 0 would read as "no agreement".

## Rank stability under `h` is decidable, not a sweep

`eff@1(h) = Σ h_l F_l / Σ h_l` is exactly linear in the hardness weights, so
`sign(eff^A − eff^B) = sign(Σ h_l d_l)` with `d_l = F^A_l − F^B_l` and
`h_l > 0`. A pair is reorderable **iff `d` has mixed signs** — no search, no
sweep, three numbers per model. `compare_under_h` returns that verdict plus
concrete integer hardness vectors that put each model on top, which is worth
more in a paper than an asymptotic statement because it is exhibitable.

The linearity itself is validated against the paper rather than assumed:
`test_reproduces_every_table10_entry_from_three_numbers` rebuilds all fifteen
published Table 10 sweep entries from the three recovered per-level means to
within 0.002. If the aggregation were anything but a weighted mean over levels,
the residuals would not all sit inside rounding error.

### The linearity is a `k = 1` fact, and the module says so

`e_{i,j}` is linear in `h` for every sample, but `eff_i@k` is `Σ_r λ_r e_{i,(r)}`
over the *sorted* sample scores, and which sample is `(r)`-th depends on `h`.
For `k = 1` the weights are uniform, the ordering drops out, and `eff_i@1` is the
plain mean over samples — linear. For `k > 1` the score is a `λ`-weighted average
of order statistics of linear functions: continuous and piecewise linear, but not
linear, and not determined by the level means at all.

The failure is not asymptotic. With two samples at `k = 2` the estimator reduces
to `max`, so level fractions `(1, 0)` and `(0, 1)` give `eff@2 = 1` at both
`h = (1, 0)` and `h = (0, 1)` and `0.5` at `h = (1, 1)`: the midpoint identity
linearity forces is violated by half the range. The same two samples have level
means `(0.5, 0.5)`, so `attainable_range` would claim the score cannot leave
`[0.5, 0.5]` while it in fact reaches 1. Both cases are tests.

This is why the functions here name `eff@1` rather than `eff@k`. It costs
nothing: the paper's headline numbers, its Table 10 sweep, and everything
`recover_table10.py` inverts are all `eff@1`. What it buys is that the rank
stability claim cannot be quietly carried over to a `k` where it does not hold.

## Two float tolerances, deliberately different

Found by asserting that a witness actually reorders instead of trusting that it
does — two of the first tests failed.

`_REL_SIGN = 1e-9` decides whether two level means differ at all. Without it,
`0.4 - 0.4` arriving as `-1e-17` turns a weakly dominant pair (equal on one
level) into a reorderable one.

`_REL_DOT = 1e-12` decides whether a weighted sum is positive, and is much
tighter because a witness is *constructed* to sit just past total cancellation;
a generous threshold would reject valid witnesses. It compares the sum against
the sum of the term magnitudes, which near a crossing are many orders of
magnitude apart, so noise surviving the cancellation cannot masquerade as a
decision. The original `> 0` test accepted `5.55e-17` as a win and returned
`h = (1,1,1)` for a pair that `(1,1,1)` ties exactly.

Both are pinned by `TestNumericalRobustness`.

## `alpha` cannot be raised after the fact

Lowering `alpha` is free: a run that finished under the old limit has a known
time, and the score clamps to 0 if it now exceeds the new limit. Raising it is
not. A run killed at the old limit has a true time somewhere in `[T_old, inf)`
and may well have finished under a larger limit, so its new score is not
determined by the recorded data. Silently keeping such a run at 0 would bias
every larger-`alpha` result downward, so `rescore_at_alpha` raises instead.

Harness consequence: **measure once at the largest `alpha` you will ever want
to report, then derive all smaller ones by post-processing.**

## Level discrimination is reported as `q`, a slowdown, and a share

`levels.py` exists because §2.2 of open-questions.md is an arithmetic claim
about Eq. (1) that only measurement can settle, and the claim needs three
numbers rather than one.

`q_ratios` normalizes each level's worst reference case by the largest one over
levels, so the limit-setting level has `q = 1` by construction. Normalizing by
*the largest* rather than by the last level is deliberate: `T_i` is defined as
`alpha * max` over all levels, so it is the largest that sets the scale, and
§2.2's reasoning silently assumes the two coincide. `limit_level_counts` reports
how often they do not, which is a fact about the data nobody has published.

From `q` follow two derived quantities. `tolerated_slowdown` is `alpha / q`, the
factor at which the level first scores 0 and past which every candidate scores
the same: it states a level's resolution as a single number. `sensitivity_shares`
differentiates the score, giving `h_l q_l / (alpha - q_l)` normalized, which is
how the response to a uniform slowdown decomposes over levels. That is §2.2's
"60% of the weight sits where discrimination is weakest" turned from a weight
ratio into a measured response ratio. It is a local statement, valid while each
level is unsaturated, and a candidate in a worse complexity class slows more at
the larger scales, so the last level's share is a floor and not an estimate.

`level_fraction_at` delegates to `metrics.score.level_fraction` with times
normalized by the limit-setting reference time rather than reimplementing Eq.
(1) in normalized form. The table in §2.2 is then a test fixture, so the
document and the scorer check each other and neither can drift alone.



## `alpha` sets the metric's dynamic range, not just its timeout tolerance

Not remarked on in the paper, and it changes how the `alpha` sweep should be
read. Since

    f = (T − t) / (T − t*),   T = alpha · max t*

we have `∂f/∂T = (t − t*) / (T − t*)²`, so `f` increases with `alpha` when the
candidate is slower than the reference and *decreases* when it is faster, with
`f → 1` from either side as `alpha → ∞`. Large `alpha` compresses every finite
score toward 1 and flattens the metric; small `alpha` is harsh. The reported
rise of `eff@1` with `alpha` is therefore mostly this compression acting on a
population that is predominantly slower than the expert reference — not a
tolerance effect. Pinned by
`test_alpha_sets_the_dynamic_range_not_just_the_timeout`.

This also means `alpha` and the unclamped upside interact: at `alpha = 2`,
instantaneous code scores 2, and the same code scores 1.5 at `alpha = 3`. Any
report that sweeps `alpha` is simultaneously changing how much credit beating
the reference earns.

## Open items

- `attainable_range` returns a closed interval although `h_l > 0` strictly makes
  the endpoints unattainable. Not material to the point being made, but it is a
  small lie in the type.
- `reorderable_pairs` is `O(models²)` with an exact per-pair test; fine at
  leaderboard scale, and there is no reason to be cleverer.
- The recovered per-level means are for one model (GPT-4 Turbo). The dominance
  table needs per-level means for every model, which the runner must record —
  it is not derivable from published aggregates.
