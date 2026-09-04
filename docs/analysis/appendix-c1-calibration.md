# What Appendix C.1's "further calibrate" step is

Resolved 2026-09-04 from arXiv:2406.06647v4. No measurement: this is a close
reading of the paper against its own notation table, plus one exact property of
Eq. (1) that the reading turns up. The property is checked in
`tests/test_calibration.py`.

## The sentence

Appendix C.1, "Code evaluation", in full:

> We use α = 2, R = 6, h₁ = h₂ = 3, h₃ = 4, M₀ = 8, M₁ = M₂ = M₃ = 4. To
> minimize server workload fluctuations, we run evaluation on virtualized cloud
> servers hosted by Google Cloud (Ubuntu 20.04.6 LTS; Intel Xeon CPU @ 2.20GHz;
> Python 3.10.12). **We use the reference time on the slowest test case for each
> problem to further calibrate the execution time of generated code.**

The bolded sentence is the entire basis for the step. It appears nowhere else in
the paper. Until now this project called it unreconstructed and treated it as a
blocker on parity, on the grounds that it sits directly on the numerator of
Eq. (1).

## What the phrase refers to

"The reference time on the slowest test case for each problem" is a maximum over
cases *and* levels — one number per problem, `max_{l,m} t*ᵢ,ₗ,ₘ`. The paper uses
that quantity in exactly one place: `Tᵢ := α · max_{l,m} t*ᵢ,ₗ,ₘ`. Eq. (1)'s
denominator uses a different quantity, the slowest case *within* a level,
`max_m t*ᵢ,ₗ,ₘ`, which is L numbers per problem and is not "for each problem".
So the referent is pinned by the paper's own wording, and it is `Tᵢ/α`.

Three readings survive that constraint. The first is that the sentence is a
prose gloss on `Tᵢ`: a raw wall-clock time becomes a score only by being offset
and divided by reference-derived quantities, both anchored on `Tᵢ`, and
"calibrate" means "put on a per-problem scale" — which §2.2 of the main text
states outright, that the score is normalized by the reference "so that the
scale of the score does not differ across problems". The second is a drift
correction: re-time the reference's slowest case alongside the candidates and
rescale the candidates by the ratio. The word "further", and the fact that the
preceding sentence is also about fluctuation control, both point this way. The
third is a speedup ratio, candidate time divided by reference time in place of
Eq. (1).

The third is excluded outright by Appendix C.3, which introduces `speedup` as a
*contrast* to `eff@k`, defines it as `t*ᵢ,ₗ,ₘ / min{tᵢ,ⱼ,ₗ,ₘ, Tᵢ}`, and reports
it separately in order to argue against it. So the choice is between a gloss and
an unstated extra step.

## Which one the paper selects

**Table 5, the nomenclature, is complete, and there is no calibration factor in
it.** It enumerates `k, n, L, zᵢ, cᵢ,ⱼ, gᵢ,ⱼ, tᵢ,ⱼ,ₗ,ₘ, fᵢ,ⱼ,ₗ, eᵢ,ⱼ, eᵢ,₍ᵣ₎,
t*ᵢ,ₗ,ₘ, Tᵢ, hₗ, Mₗ, α, R` and stops. A multiplicative correction applied to
`tᵢ,ⱼ,ₗ,ₘ` would be a symbol in that table and a term in Eq. (1), and it is
neither. `t*ᵢ,ₗ,ₘ` is defined there as one fixed quantity per case, not as a pair
of recorded and re-measured values, which is what a drift correction needs.

**Nothing else in the paper says the reference is timed more than once.** Every
occurrence of "reference" was checked; the reference times enter the metric in
exactly the two places above, and Appendix B proves Theorem 1 over `eᵢ,ⱼ` as
Eq. (1) and (2) define it, with no time transform anywhere in the argument.

So the reading the paper's own formalism supports is the first one: the sentence
restates how `Tᵢ` is built and what it is for. Under that reading there is no
step missing from this implementation. `time_limit` is `α · max_{l,m} t*`,
`level_fraction` divides by `Tᵢ − max_m t*`, and `sample_score` consumes times as
given, which is exactly what "use the reference time on the slowest test case for
each problem to calibrate the execution time of generated code" describes.

That closes the item as a blocker: the numerator of Eq. (1) is not missing a
factor, and a parity run can proceed on the metric as implemented.

## The residual

The second reading cannot be excluded as an *implementation* detail that never
made it into the formalism, and the upstream repository would settle it in a
minute — it is not reachable from this environment, so it goes on the list for
when the snapshot is pulled (decision 0005). The distinction has no consequence
for us, because this harness controls the same source of error structurally
rather than by correction: decision 0006 measures each problem's reference once
in the same run as the candidates it scores, and interleaves samples by index so
that machine drift during a run cannot line up with model identity. That is not
the same thing as a ratio correction — it does not remove drift, it stops drift
from becoming a difference between models — but it needs no unrecoverable
constant to be right, which a reconstruction of the second reading would.

## What the reading turns up: a drift correction has to scale the limit too

Working out what the second reading would actually take produced the one result
here that is not textual. Suppose the machine runs `s` times slower while a
candidate is measured than it did while the reference was measured, so work worth
`t` reference-seconds takes `s·t` wall-seconds. A correction that estimates `s`
from the reference's slowest case and reports `t_wall / s` recovers the
reference-machine time for every case that finished.

But the kill happens in wall-clock time, at `Tᵢ`, and `Tᵢ` was computed from
reference-machine times. A case is therefore censored when `s · t_true ≥ Tᵢ`,
that is when `t_true ≥ Tᵢ/s`, while the score is defined against the threshold
`Tᵢ`. For `s > 1` that is a strictly tighter threshold than the one being scored
against: every candidate whose true worst case falls in `[Tᵢ/s, Tᵢ)` is killed,
its time is never observed, and no arithmetic afterwards recovers it. It scores 0
at that level instead of `(Tᵢ − t_true)/(Tᵢ − max_m t*)`.

At the level that sets `Tᵢ` the denominator is `Tᵢ(1 − 1/α)`, so the score lost
by a candidate sitting at the bottom of that band is `(1 − 1/s)/(1 − 1/α)` —
**0.18 of that level's score at 10% drift, 0.40 at 25%, and 0.67 at 50%**, with
α = 2. Levels whose reference is fast lose less, because their denominator is
nearer `Tᵢ` and their candidates rarely time out at all, so both the magnitude
and the frequency of the error concentrate at the limit-setting level, which is
the one §2.2 shows carries the discrimination.

The fix is to calibrate the limit as well as the times: kill at wall-clock
`s · Tᵢ`. Then a case is censored exactly when `t_true ≥ Tᵢ`, the same set as
with no drift at all, and the calibrated score equals the undrifted score
exactly. Equivalently, apply the factor on the way *in*, converting `Tᵢ` into the
drifted machine's units before measuring, rather than only on the way out.

The error is also one-sided. For `s < 1`, a machine that has got faster, nothing
extra is killed and the time-only correction is exact. So a drift correction that
ignores the limit never inflates a score and only ever deflates one, and only
while the machine is slow — it does not average out over a long run, it
introduces a downward bias concentrated in exactly the periods of load that
motivated the correction.

This is the same shape as the α asymmetry in decision 0006: a censoring boundary
has to be fixed, in the units the score is defined in, *before* the measurement,
because a censored sample's true time is never observed and no post-hoc
arithmetic can recover it. Two instances of one rule is enough to state it as a
rule.

## Status

The item is closed as a blocker on parity and is no longer listed as
unreconstructed in `README.md`, `docs/open-questions.md` §2.2,
`docs/decisions/0001-metric-core.md` or `docs/decisions/0003-data-adapter.md`.
One thing remains open, and it is a question for the snapshot rather than for the
paper: whether the upstream implementation applies a per-run scaling that the
paper does not describe, and if it does, whether it scales the time limit.
