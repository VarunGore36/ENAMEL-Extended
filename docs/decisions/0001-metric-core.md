# 0001 — The metric core

Status: implemented, 2026-09-01. Covers `enamel_ext/metrics/effk.py`,
`enamel_ext/metrics/score.py`, `enamel_ext/measure/timing.py`.

This is the part of the benchmark that needs no test data, no model samples and
no clock, so it goes first and it goes in exactly as published. Everything here
is Eq. (1)–(6) of arXiv:2406.06647v4 with the paper's constants: `α = 2`,
`h = (3, 3, 4)`, three scored levels plus level 0 as a correctness filter, and
`Tᵢ = α · max over all levels and cases of the reference time`.

## Decisions

**The published configuration is a named constant, not a set of defaults.**
`MetricConfig` has no default values; `PAPER` is the only preconfigured
instance. Any deviation from the paper therefore has to be written out at the
call site, which is the property we want during a parity run — a silently
different `α` is the single easiest way to produce numbers that look plausible
and are wrong.

**Weights come from the Algorithm 1 recurrence, and Eq. (6) is kept only as
test ground truth.** The closed form `λ_r = C(r−1, k−1)/C(n, k)` cannot be
evaluated in floating point past about `n = 1030, k = n/2`, where `C(n, k)`
exceeds the float range; in CPython the failure is an `OverflowError` on the
first int-to-float contact rather than a quiet infinity. `effk_weights_exact`
computes the same weights over `Fraction` for tests, and the two agree to
within 1e-15 across the tested grid.

**Correctness is a hard gate, and the score above it is unbounded.** A wrong
sample scores 0 no matter how fast, and a sample faster than the expert
reference scores above 1 — at `α = 2`, code that finishes instantly scores 2 on
a level. Neither end is clamped. This matters for interpretation: a per-problem
`effᵢ@k` above 1 is not a bug, it means the reference was beaten, and one such
sample can pull a mean up.

**A timeout is `TIMEOUT = inf`, and censoring propagates through aggregation.**
If any of the `R` repeats of a test case was killed at the limit, the case
reports `inf` rather than aggregating the completed repeats. Mixing a censored
repeat into a Hodges–Lehmann estimate would report a finite time that was never
observed, biased downward by an unbounded amount since the killed repeat's true
time is only known to be `≥ Tᵢ`.

**Level skipping is the caller's job.** The paper stops evaluating a sample
after its first timeout, which is only equivalent to scoring the remaining
levels 0 if level times are monotone in level. That holds for the paper's
generators but is not guaranteed, so `sample_score` requires the caller to pass
`TIMEOUT` for skipped levels explicitly instead of inferring them.

**`mean_over_problems` is a named function.** The unweighted mean over the 142
problems gives a problem with a badly calibrated limit or a wrong reference the
same weight as a well-behaved one. That is the paper's choice and we keep it,
but any reweighting belongs in that one function.

## Variants, off by default

`normalization="per_level"` sets `T_{i,l} = α · maxₘ t*[i,l,m]` so each level is
scaled by its own reference. This is the candidate fix for the score-compression
issue in §2.2 of open-questions.md and it is never used in a parity run.
`test_score.py` pins both behaviours side by side: a candidate 10× slower than
the expert on a level whose reference time is 1% of level 3's scores 0.955 under
the published normalization and 0 under the variant.

The timing aggregator is selectable for the same reason. Hodges–Lehmann is the
paper's choice and the default; `min` is the standard microbenchmarking choice,
since timing noise is one-sided and the minimum therefore estimates the
noise-free cost. The two disagree systematically, with HL sitting above the
noise floor.

## Bug found while writing the tests

The first implementation of `eff_at_k` zipped the weight vector against the
*ascending* sorted scores from index 0. The weights are indexed by rank
`r = k..n`, so they must pair with the top `n−k+1` order statistics; starting at
index 0 silently discards the largest values. The wrong version still produced
weights summing to 1, still gave the right answer at `k = 1`, still looked
monotone in `k`, and still passed permutation-invariance — it failed only
against the exact enumeration in `TestTheorem1` and against brute-force
averaging of `max` over all `C(n, k)` subsets. Both of those checks are now
permanent tests. Worth noting for anyone reimplementing Eq. (6): the alignment
is the failure mode, not the arithmetic.

## Open items

The Hodges–Lehmann convention is ambiguous in the paper. We take the
one-sample form over Walsh averages `(x_a + x_b)/2` for `a ≤ b`, including the
`a = b` terms — the pseudomedian targeted by the signed-rank test. Excluding
them, which some references do, gives a slightly different number; for `R = 6`
the two differ by order 1% on skewed samples. If a parity run comes out
systematically off, flip this first.

Appendix C.1's calibration step — "we use the reference time on the slowest test
case for each problem to further calibrate the execution time of generated
code" — is still not reconstructed. It reads like a per-machine scaling
correction, but the description does not pin down what is scaled by what. No
guess is baked into the code; `sample_score` consumes times as given. This has
to be resolved before parity can mean anything, since it sits directly on the
numerator of Eq. (1).

`R = 6` repeats with a Hodges–Lehmann breakdown point near 29% tolerates one
contaminated repeat out of six and not two, which `test_timing.py` demonstrates.
That is thin for a shared or virtualized machine and is an argument for the
deterministic measurement backend, not against the estimator.
