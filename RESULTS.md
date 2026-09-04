# Results

What this project has established so far, and how it was established. The
grouping is by kind of evidence, because the kinds are not interchangeable: an
exact check of an estimator settles something permanently, a close reading of the
paper settles what the method *is* and not how well it works, arithmetic on
published numbers settles less than a measurement would, and a defect found while
building the harness says nothing about ENAMEL at all.

Nothing here has been measured on the 142 problems. That section is last and it
is empty, which is the honest state of the project and most of the reason this
file exists.

`§2.x` refers to [`docs/open-questions.md`](docs/open-questions.md). Everything
here re-runs with `python3 -m unittest discover -s tests -t .`, including the
derived numbers; `python3 scripts/recover_table10.py` prints the derivation.

## Verified exactly

**The `eff@k` estimator is correct as published and needs no changes.**
Algorithm 1's recurrence agrees with the Eq. (6) closed form computed in exact
rational arithmetic, the weights sum to exactly 1 as `Fraction`s, and the
estimator equals the brute-force mean of `max` over all `C(n,k)` subsets, which
is the identity Theorem 1 rests on. Unbiasedness and the `(k/n)·Var[max]` bound
are checked by enumeration too: all 243 outcomes of a three-point score
distribution over five samples, every `k` from 1 to 5, with the estimator's mean
matched to the true `E[max]` to twelve decimals and its variance held against the
bound. The recurrence also survives `n` where the closed form overflows in
float. Everything §2 raises concerns the score fed into the estimator, never the
estimator. See
[`docs/decisions/0001-metric-core.md`](docs/decisions/0001-metric-core.md).

**A correction to one of our own claims, not the paper's: `eff@k` is linear in
the hardness weights `h` only at `k = 1`.** The per-sample score is linear in
`h`, but `eff@k` weights order statistics whose ordering moves with `h`, so
above `k = 1` it is piecewise linear and the per-level means do not determine
it. Two samples with level fractions `(1,0)` and `(0,1)` give `eff@2 = 1` at
both `h = (1,0)` and `h = (0,1)` and `0.5` at `h = (1,1)`, while their level
means `(0.5, 0.5)` would predict a constant. Every published sweep is `eff@1`,
so nothing was lost, but all `h` analysis is now labelled.

**The upside of the score is capped, and capped hardest where the downside is
flattest.** Infinitely fast code scores `α/(α − q)` at a level whose reference
worst case is `q` times the limit-setting one: 2.0 at the level that sets `Tᵢ`,
1.053 at `q = 0.1`, 1.005 at `q = 0.01`. So "beating the expert scores above
1.0" has real headroom only at the level that already carries all of the
discrimination §2.2 is about, and with `h = (3,3,4)` and `q₁ = q₂ = 0.01` no
sample can score above 1.403 however fast it is. This is arithmetic on Eq. (1),
like the §2.2 table; how much it matters depends on the same unmeasured `q`.

**A machine-drift correction has to scale the time limit, not only the times.**
Suppose the machine is `s` times slower while a candidate runs than it was while
the reference was timed. Dividing the observed times by `s` recovers the
reference-machine time for every case that finished — but the kill happened in
wall-clock at `Tᵢ`, so a case is censored once its true time reaches `Tᵢ/s`, while
the score is defined against `Tᵢ`. Every candidate landing in `[Tᵢ/s, Tᵢ)` is
killed, its time is never observed, and it scores 0 where it should have scored up
to `(1 − 1/s)/(1 − 1/α)` of that level: 0.18 at 10% drift, 0.40 at 25%, and the
level's whole score at 2×, with α = 2. Killing at `s·Tᵢ` instead makes the
censored set identical to the undrifted one and the score exact. The error is also
one-sided, since a machine that has got *faster* loses nothing, so it does not
average out over a long run — it is a downward bias concentrated in exactly the
periods of load that would motivate the correction. Both magnitude and frequency
concentrate at the level that sets `Tᵢ`, which §2.2 identifies as the level
carrying the discrimination. Checked in `tests/test_calibration.py`. This
constrains any correction we or anyone else adds; it is not something ENAMEL does,
and the reason it came up is in the next section.

## Settled by reading the paper

**Appendix C.1's "further calibrate" step is not a step missing from Eq. (1).**
The paper says once, and nowhere else, that "we use the reference time on the
slowest test case for each problem to further calibrate the execution time of
generated code". This project had been treating that as an unreconstructed
correction sitting on the numerator of Eq. (1), and as a blocker on parity. It is
not one. "The slowest test case for each problem" is a maximum over levels as well
as cases, `max_{l,m} t*ᵢ,ₗ,ₘ`, which the paper uses in exactly one place,
`Tᵢ := α · max_{l,m} t*ᵢ,ₗ,ₘ`; Eq. (1)'s denominator uses the different, per-level
`max_m t*ᵢ,ₗ,ₘ`. Table 5's nomenclature is complete and contains no scaling
factor, `t*ᵢ,ₗ,ₘ` is defined there as one fixed quantity per case rather than a
recorded-and-re-measured pair, and Appendix C.3 rules out the remaining reading by
introducing `speedup` as a contrast to `eff@k`. So the sentence restates how `Tᵢ`
is built and what it is for, our implementation was already complete, and the
metric is not waiting on anything. What cannot be ruled out from the text is an
*implementation* detail that never entered the formalism; that is a question for
the upstream snapshot, and working out what it would take is where the drift
result above came from. See
[`docs/analysis/appendix-c1-calibration.md`](docs/analysis/appendix-c1-calibration.md).

## Derived from the paper's published numbers

At `k = 1` the score is linear in `h`, so Table 10's `h` sweep can be inverted:
multiplying each reported `eff@1` by its denominator `Σh` recovers the
numerator, and first differences along a sweep isolate one level's mean score.
For GPT-4 Turbo under greedy decoding the per-level means are
**`F = (0.638, 0.453, 0.355)`**. Rebuilding the headline from them gives
`eff@1 = 0.4693` against the published 0.470, and the four independent estimates
within each sweep agree to ±0.008, which is what rounding to three decimals
should produce. No new measurement is involved; this is arithmetic on published
numbers. See
[`docs/analysis/table10-recovery.md`](docs/analysis/table10-recovery.md).

Two things follow.

**The extreme form of the §2.2 score-compression concern does not hold.** If
level 1 were nearly constant for most problems, `F₁` would sit just under
`pass@1 = 0.796`. It is 0.638, so level 1 does discriminate. What that does not
settle is why: `q` may simply not be small for most problems, or `q` may be
small and roughly a fifth of the samples counted in `pass@1` time out at level
1, which the metric permits. Those have different implications and published
aggregates cannot separate them. The expected finding on the `q` distribution is
now "moderate" rather than "severe".

**`h` has more leverage on the headline number than the choice of model does.**
Since `eff@1` is a convex combination of the three level means, reweighting `h`
alone moves this model's score anywhere in `[0.355, 0.638]`, a span of 0.283.
The `α` sweep from 1.5 to 3.5 spans 0.120; the entire gap between the best and
fourth-best model in the paper spans 0.062. That is not itself a criticism,
since the paper is explicit that `α` and `h` encode user preference and reports
the sensitivity. The gap is that every entry in Table 10 is one model, so
nothing published speaks to whether the *ranking* survives the sweep. Linearity
makes that decidable without extra compute: model A ranks above B for every
non-negative `h` if and only if A's three level means all dominate B's, and if
the vectors cross then some admissible `h` flips the pair. Once the harness
records per-level means, rank robustness under `h` is a table rather than an
experiment. `α` is not free in the same way: raising it un-censors runs whose
times were never observed, so a run has to measure once with a generous cap and
derive smaller `α` by post-processing.

## Found while building the harness

These are findings about building this kind of measurement, not about ENAMEL.
They are worth recording because each one silently produces a plausible number:
every one reads as "the model got it wrong" or "the model was slow" rather than
as a harness bug, so nothing in the output invites a second look.

**Repeats have to run on fresh copies of the input.** With `R = 6` repeats over
one shared input, a solution that sorts, reverses or pops its argument in place
has only its first repeat measuring the intended data, and the robust aggregate
is then dominated by the five that measure something else. The copy has to be
deep, a list of lists still shares its inner lists, and it has to happen outside
the timed window. The paper's description does not address this.

**Five defects in the runner all presented as scores rather than as errors.** A
timeout swallowed a wrong answer, because the child reported the cases it
finished and the timeout branch was checked first. Level 0 ran under `Tᵢ`,
though its inputs are adversarial rather than large, so slow there is
inefficient and not incorrect; a timeout there still has to set
`correct = False`, since nothing was verified. Censoring was applied per repeat
rather than to the aggregate the score compares, and the rule that replaced it
was wrong in the same way, which is the next entry. `repr` of an int wider
than `sys.get_int_max_str_digits()` raises, so an answer like `7 ** 20000`
truncated the result file and killed the whole problem. And two subprocess
traps: `-I` implies `-E`, which discards `PYTHONHASHSEED` and lets set iteration
order differ between the run that recorded the expected output and the run that
produced the candidate's, while `communicate(timeout=)` waits for pipe EOF
rather than process exit, so a solution that forks a helper looked like a
timeout with a valid result already on disk. See
[`docs/decisions/0004-sandboxed-runner.md`](docs/decisions/0004-sandboxed-runner.md).

**A stopping rule has to censor exactly what the score gives 0.** The runner
stopped a case once its accumulated time over the `R` repeats passed `Tᵢ·R`,
which is stopping when the *mean* repeat time passes `Tᵢ`, while Eq. (1)
compares the Hodges-Lehmann aggregate. Timing noise is right-skewed, so the mean
sits above the aggregate and the two disagree in one direction only. Six repeats
at `(0.5, 0.5, 0.5, 0.5, 0.5, 4.0)·Tᵢ` have a mean of `1.08·Tᵢ` and an aggregate
of `0.5·Tᵢ`, so against a reference whose worst case is `Tᵢ/α` that candidate
matches the expert and earns the whole level — and the accumulated rule scored it
0. One outlier repeat out of six was enough, which is the contamination the
robust estimator was chosen to absorb in the first place. The rule now stops only
once no completion of the remaining repeats can bring the aggregate back under
`Tᵢ`, so "censored" and "scores 0 here" become the same statement; the price is
that the accumulated total no longer bounds the work and the wall clock has to,
which it does with room to spare, and that `min` gets no early stop at all
because its bound stays 0 until the last repeat. Third instance of one rule: a
cap or correction applied to measured times has to be applied to the threshold
those times are judged against. The other two are the drift result above and the
`α` asymmetry below. Checked in `tests/test_stopping.py`.

**Output comparison cannot go over plain JSON.** JSON cannot distinguish a tuple
from a list, and a HumanEval signature that should return `(1, 2)` returning
`[1, 2]` is a wrong answer. Containers are type-tagged on the way out and
rebuilt on the way in, so comparison uses real `==` with tolerances applied at
every depth: absolute `1e-6`, matching HumanEval's own check, since a relative
`1e-6` would accept an absolute error of 500 at magnitude `1e9`.

**A commit pins the tree; it does not pin the bytes.** Generated archives have
changed historically for reasons unrelated to repository content, so the lock
records the archive digest and a per-file digest for everything extracted. That
is what lets "repacked" be told apart from "tampered with" — a tool recording
only the archive digest cannot distinguish them. Extraction is guarded
explicitly rather than by `tarfile`'s `filter=`, which landed in 3.12 while this
targets 3.10. See
[`docs/decisions/0005-snapshot-pinning.md`](docs/decisions/0005-snapshot-pinning.md).

**A file that stores a score can disagree with itself.** The run record stores
per-level worst-case times plus a correctness verdict and never a score;
everything is recomputed at print time, which makes the report a pure function
of the record and makes the `α` sweep and the `h` section the same measurements
read at another threshold. It also exposes one asymmetry that would otherwise
pass unnoticed: lowering `α` is always sound, while raising it cannot be
answered from the record at all, because treating a censored sample as having
finished exactly at the old limit hands the slowest possible run the best score
consistent with the data. See
[`docs/decisions/0006-run-record.md`](docs/decisions/0006-run-record.md).

## Measured on the 142 problems

Nothing yet. Two things are missing, neither of them code.

The data: the 142 problems, the expert references and the input generators are
fetched from upstream at setup time and never vendored, and the environment this
was built in has no network egress, so the snapshot cannot be pulled here. The
fetcher and the lock are written and tested against synthetic archives; they
have never seen the real one, which is also why no upstream path is hard-coded.

The hardware: two cores under a hypervisor cannot produce a timing number anyone
should trust, least of all one compared against a published figure. Real
measurement runs belong on a machine chosen for it.

What is ready and waiting on those two things: `report/levels.py` computes the
`q` distribution across levels, the slowdown `α/q` past which a level scores 0
for everyone, and each level's share of the score's response to a slowdown,
which is §2.2 restated as a measured ratio; and `scripts/evaluate.py run`
produces the parity numbers with bootstrap intervals, the `α` sweep and the
reachable-`eff@1` range in one pass. The parity targets, so the comparison is
written down before the number arrives: GPT-4 Turbo `eff@1 = 0.470` /
`pass@1 = 0.796`, GPT-4 `0.454` / `0.831`, HumanEval canonical solutions
`eff@1 = 0.455`, HumanEval+ `0.513`.
