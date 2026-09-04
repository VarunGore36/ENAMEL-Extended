# Open questions and the fix list

What ENAMEL does as we read it (§1), the decisions the paper leaves to the
implementer (§2), and what we do about each one while building (§3).

These section numbers are load-bearing: the code, the tests and every file in
`docs/decisions/` cite them, so a section here may be rewritten but §2.2 stays
§2.2. Moved out of the README on 2026-09-02 with the text unchanged.

---

## 1. What ENAMEL actually does

**Level-based evaluation.** Each of the 142 problems gets four tiers of test
cases. Level 0 (`M₀ = 8` cases) is a correctness filter using small but
adversarial inputs. Levels 1–3 (`M₁ = M₂ = M₃ = 4` cases each) increase input
scale, so that algorithms of different asymptotic complexity clear different
numbers of levels. All levels of a problem share one time limit `Tᵢ`. Code that
fails any output check, or fails level 0, is scored `0` and excluded from
`pass@k`; code that clears level 0 but times out at level `l` keeps its
`pass@k` credit, and the remaining levels are skipped and treated as timeouts.

**Efficiency score.** For correct code, the per-level score is

```
f[i,j,l] = ( Tᵢ − maxₘ t[i,j,l,m] )₊  /  ( Tᵢ − maxₘ t*[i,l,m] )
```

where `t*` is the expert reference's time, the `max` is over the test cases in
that level (worst case, not average), and `Tᵢ = α · max_{l,m} t*[i,l,m]` with
`α = 2` — note the max runs over *all* levels, so `Tᵢ` is a single per-problem
constant set by the reference's slowest test case anywhere. Each test case is
run `R = 6` times and the runtime estimated by the Hodges–Lehmann estimator for
robustness to outliers. Levels are combined by a hardness-weighted mean with
`h₁ = h₂ = 3`, `h₃ = 4`. Matching the expert scores `1.0`; beating the expert
scores above `1.0`; hitting the limit scores exactly `0.0`.

**The `eff@k` metric.** `pass@k` is re-expressed as `E[maxⱼ gᵢ,ⱼ]`, which drops
the reliance on correctness being Boolean, and `eff@k := E[maxⱼ eᵢ,ⱼ]` follows
by substituting the continuous score. From `n ≥ k` samples the authors estimate
it by Rao–Blackwellizing the bootstrap estimator, giving a closed form over
order statistics with binomial weights, plus a numerically stable recurrence
(Algorithm 1) that avoids computing the coefficients directly. Theorem 1 proves
unbiasedness and a variance bound of `(k/n)·Var[max]`; empirically the
estimator's standard deviation drops from 0.20 to 0.02 at `k=1`. The final
`eff@k` is an unweighted mean of per-problem `effᵢ@k` over the 142 problems.
**This part is correct and stays.** We have already checked that Algorithm 1's
recurrence reproduces the Eq. (6) coefficients to machine precision and that
the weights sum to 1; the criticism below is about the score `eᵢ,ⱼ` that gets
fed into the estimator, not the estimator itself.

**Data.** 142 of HumanEval's 164 problems, excluding Θ(1)-time ones on two
grounds: their runtimes are too short to measure above hardware noise, and
every model solves them, so they carry no discriminative signal. One human
expert wrote a best-known-algorithm reference for each problem and then
optimized its implementation, plus a strong test-case generator seeded by
ChatGPT and hardened by hand with corner cases and non-random worst cases. The
result is a genuinely higher bar: under these tests, 11 HumanEval and 4
HumanEval+ canonical solutions are *wrong*, and 34 and 27 respectively *time
out*. As efficiency references the canonical solutions score `eff@1 = 0.455`
(HumanEval) and `0.513` (HumanEval+).

**Findings.** Across 30 models, `eff@k` is far below `pass@k` everywhere. GPT-4
Turbo leads at `eff@1 = 0.470`; no model exceeds `0.5`. Correctness rank does
not predict efficiency rank — GPT-4 Turbo beats GPT-4 on `eff@1` while losing
on `pass@1`. Two diagnostic subsets separate the failure modes: on 20 problems
where the optimal algorithm is asymptotically better, scores stay low even at
`k = 100` (ChatGPT `eff@100 = 0.483`), while on 75 problems needing only
implementation optimization, scores recover with sampling (Phind Code Llama V2:
`eff@1 = 0.351` → `eff@100 = 0.732`) — suggesting the second failure mode is
drawing a lucky sample rather than knowing better. Asking for "the most
efficient algorithm" changes almost nothing, and Self-Refine on problem #36
leaves both tested models at Θ(n) when Θ(log n) exists.

**Measurement environment.** Virtualized Google Cloud servers, Ubuntu 20.04.6,
Intel Xeon @ 2.20 GHz, CPython 3.10.12.

---

## 2. Open questions the reimplementation has to answer

Coding the paper means committing to a value for every constant and a
resolution for every ambiguity, so this section is the list of decisions the
paper leaves to the implementer, plus the places where its own numbers pull
against a stated claim. It is a work list, not a rebuttal.

The paper has an unusually candid limitations section (Appendix D), and it is
worth separating what the authors already concede from what turned up while
reading. They acknowledge: standalone-function scope, no theoretical optimality
guarantee for the references (hence scores can exceed `1.0`), no advanced
prompting study, no space-efficiency measurement, no automatic complexity
measurement, and limited scalability of expert-driven benchmark construction.
Those are stated openly, and §3 treats them as work to be done rather than as
criticism.

The rest — §2.1 through §2.8 — does not appear in Appendix D. §2.1 and §2.2 are
the two that actually change what we build.

### 2.1 The "no model exceeds 0.5" claim depends on α

The paper's central assertion is that no LLM reaches expert-level efficiency,
sharpened repeatedly into the specific form "even the strongest LLM GPT-4 Turbo
… `eff@1` below 0.5." Appendix C.6 then reports GPT-4 Turbo's `eff@1` under
different timeout factors: **0.421** at `α = 1.5`, **0.470** at the chosen
`α = 2.0`, **0.502** at `α = 2.5`, **0.525** at `α = 3.0`, **0.541** at
`α = 3.5`.

At `α = 2.5` the sharpened version of the claim no longer holds, by the paper's
own measurement. `α = 2` is not derived from anything — it is a declared
default, and Appendix C.6 explicitly frames it as a user preference knob ("if
one wants to tolerate less efficient code, then they can use a larger α"). The
underlying point survives easily: `eff@1` stays far below `pass@1` at every α
in the sweep, so "LLMs are well short of expert efficiency" is robust. It is
only the crisp numerical form of it that is threshold-dependent, and we should
not restate that form without saying which α it assumes.

Ranking is the part that actually affects us. The 0.421 → 0.541 swing across
the α range is **larger than the entire spread of the top four models** at
fixed α (GPT-4 Turbo 0.470 down to Mixtral 8x22B 0.408). The sweep covers one
model, so whether the *ordering* is stable across it is simply unknown. The
same holds for the hardness weights: `h₃` from 1 to 5 moves GPT-4 Turbo from
0.520 to 0.460, again exceeding several adjacent model gaps. The sensitivity
analysis is there; it just answers a narrower question than rank stability.

### 2.2 The score is far less continuous than advertised

`eff@k` is presented as a continuous generalization of `pass@k`. Work through
Eq. (1) with the published constants and it behaves much closer to an ordinal
"how many levels did you survive," weighted 3/3/4.

The reason is that `Tᵢ` is one constant per problem, fixed by the reference's
slowest test case at the *hardest* level, while the times entering the
numerator and denominator at easy levels are far smaller. How much this
compresses the score depends on `q`, the ratio of the reference's time at an
easy level to its time at level 3 — and `q` is small by construction, since the
whole point of the levels is that input scale grows across them. Writing `X`
for how many times slower than the expert a candidate is at that level, the
arithmetic of Eq. (1) with `α = 2` gives:

| `q` | `X = 2` | `X = 5` | `X = 10` | `X = 50` |
|---|---|---|---|---|
| 0.01 | 0.995 | 0.980 | **0.955** | 0.754 |
| 0.05 | 0.974 | 0.897 | 0.769 | 0.000 |
| 0.10 | 0.947 | 0.789 | 0.526 | 0.000 |

At `q = 0.01`, code **ten times slower than the expert** still scores 0.955 at
that level. At level 3, where the reference's time *is* the one setting `Tᵢ`,
the same formula is sharp: 1.25× slower scores 0.750, 1.5× scores 0.500, and 2×
scores exactly 0. Since `h₁ + h₂ = 6` of the total weight 10, **60% of a
problem's score sits at the levels where discrimination is weakest and 40% at
the one level where it is severe.**

If that reading holds, it reframes what the benchmark measures. The gradations
in Table 3 are then mostly gradations in *how often a model's algorithm
survives the largest input scale* — a coarser and more threshold-sensitive
quantity than a runtime ratio, and one that inherits all of §2.1's fragility
because it depends almost entirely on where `Tᵢ` falls.

Appendix C.1 mentions a further step — "we use the reference time on the slowest
test case for each problem to further calibrate the execution time of generated
code" — which reads at first like an unstated correction sitting on the numerator
of Eq. (1). It is not: "the slowest test case for each problem" is a maximum over
levels as well as cases, the paper uses that quantity in exactly one place,
`Tᵢ := α · max_{l,m} t*ᵢ,ₗ,ₘ`, and the nomenclature in Table 5 contains no
scaling factor for it to be. The sentence restates how `Tᵢ` is built. See
[`docs/analysis/appendix-c1-calibration.md`](analysis/appendix-c1-calibration.md),
which also works out what the other reading would cost: a drift correction
applied to the times but not to the kill threshold silently censors every
candidate whose true worst case lands in `[Tᵢ/s, Tᵢ)`, and the resulting error is
one-sided.

The table above is arithmetic, not measurement: what it does not tell us is the
actual distribution of `q` across the 142 problems, which is what decides whether
this matters in practice.
**Measuring that distribution is milestone 1 of this project, because much of
the rest depends on which way it goes.** The tooling for it is in place
(`enamel_ext/report/levels.py`): it reports `q` per level, the slowdown `α/q`
past which a level scores 0 for everyone, and each level's share of the score's
response to a slowdown, which is the claim above stated as a measured ratio
rather than a weight ratio. It also counts how often `Tᵢ` is set by a level
other than the last, since the reasoning here assumes it is. The table in this
section is a test fixture for that module, so it and the scorer cannot drift
apart.

### 2.3 Wall-clock time on virtualized hardware is not a portable measurement

Execution time on a shared, frequency-scaling, cache-warm, OS-scheduled machine
is a random variable whose distribution depends on hardware the benchmark does
not control — and the reported environment is *virtualized* cloud instances,
the setting most exposed to steal time and noisy neighbours. The paper
mitigates this reasonably: `R = 6` repeats, the Hodges–Lehmann estimator, and
normalization against the reference timed on the same machine, which cancels a
large amount of machine-to-machine variation.

It does not cancel all of it, and two things survive. First, the *level input
scales* and the timeout `Tᵢ` were calibrated by hand on this hardware, so a
solution comfortably inside the limit there can time out elsewhere — and by
§2.2 that flips a near-1.0 level score to exactly `0`, which is the largest
single discontinuity in the metric. Second, CPython version is part of the
measurement: 3.11's specializing interpreter substantially changed the cost of
exactly the tight interpreted loops these problems stress, so references and
candidates that differ in loop structure versus builtin usage shift *relative*
to each other, not uniformly. `t*` was measured under 3.10.12.

Cross-machine rank stability is asserted rather than demonstrated. No such
experiment appears in the paper. Consequence: published `eff@k` values are not
comparable across labs, nor across time on one lab's evolving toolchain. For a
benchmark whose stated contribution is rigor, this is the first thing to fix.

### 2.4 The reference is simultaneously the ceiling and the oracle, and there is one annotator

`eff = 1.0` means "expert level" because one expert wrote the reference and
judged it best. The paper concedes optimality cannot be guaranteed. The
compounding problem it does not raise is that the same artifact also defines
correctness: expected outputs come from the reference, which is how 11
HumanEval canonical solutions get classified as wrong. So a subtly incorrect
reference does not merely misplace the ceiling — it marks correct model output
as wrong and scores it `0`, and `pass@k` absorbs the error too. HumanEval has a
documented history of exactly this class of bug; that is why HumanEval+ exists.
A hand-written replacement is not automatically immune.

This is sharpened by the fact that a single expert made every judgment call:
which 22 problems to drop, each reference algorithm and its implementation,
each level's input scale, each generator, and which corner cases count as
"absolutely valid." The paper states it avoided corner cases "whose validity is
unclear due to the ambiguity in problem description" — an acknowledgment that
HumanEval specifications *are* ambiguous, resolved here by one person's reading
with no second annotator and no reported agreement statistic. Every calibration
constant in the benchmark is a single-annotator artifact, and the interesting
result is a *gap* whose anchor is exactly that artifact.

### 2.5 Worst-case inputs are worst-case for the algorithms the expert anticipated

The generators are the paper's strongest practical contribution, and the
comparison against random tests is convincing on its own terms: on problem #31
a random generator scores the wrong-but-fast Fermat test at 1.25 while the
expert generator correctly zeroes it. The claim in §2.2 of the paper is
broader, though — that the generators "cover the worst cases of various
algorithms" — and that is asserted, not shown.

Worst case is a property of an algorithm, not of a problem. The input family
that breaks a quadratic scan is not the one that breaks a hash-based solution
via engineered collisions, nor the one that breaks a solution whose pathology
is recursion depth or a specific branch-prediction pattern. Because the
generators were written alongside the reference, a model producing a
*structurally different* algorithm may never be shown its own worst case, and
`eff@k` is then optimistically biased for exactly the novel solutions the
benchmark most wants to characterize. Fixed per-problem inputs cannot be
adversarial in the general sense; only per-candidate search can.

Note also that the censoring handling, while correctly *described* as
censoring-invariant, is a definitional dodge rather than a statistical
treatment: mapping everything past `Tᵢ` to `0` makes the score independent of
the unknown time, but discards the information that a run was 1.05× over versus
asymptotically hopeless versus non-terminating. Three very different outcomes
collapse to one value, right where mid-tier candidates cluster.

### 2.6 No uncertainty is reported, and two comparison tables overreach

Table 3 ranks 30 models to three decimals with no confidence intervals and no
pairwise significance tests. Several adjacent pairs differ by less than 0.01
(Mixtral 8x22B 0.408 / Claude 3 Opus 0.401; Phind 0.394 / Claude 3 Haiku
0.386), and Appendix C.7 reports the estimator's own standard deviation at 0.02
for `k=1`. The greedy columns have no generation randomness to average over at
all, so their only uncertainty is execution noise — which §2.3 says is not
small — and yet greedy and sampling `eff@1` sit side by side in one table as if
commensurable. Commercial models were evaluated greedy-only, so `eff@100` is
missing precisely where the question "do frontier models close the gap given
more samples?" is most interesting.

Two smaller framing problems. Table 2 lists "ENAMEL (ours): eff@1 = 1.000"
alongside HumanEval's 0.455 and HumanEval+'s 0.513; the 1.000 is definitional,
since the reference *is* the unit, and presenting it as a measured comparison
overstates the case. Table 9 concludes ENAMEL is harder than EffiBench and
Mercury by comparing 0.336 (`1/NET`), 0.424 (`Beyond`) and 0.268 (`eff@1`) —
three differently normalized metrics, which cannot establish a difficulty
ordering.

### 2.7 HumanEval contamination confounds the central diagnostic claim

Appendix D.1 defends the choice of HumanEval by arguing that LeetCode solutions
are in pretraining corpora while "LLMs … have never seen our expert-written
efficient solutions." That defends the *efficiency labels*, and it is right to.
It does not defend the *diagnosis*.

The paper's most interesting claim is causal: models score low because they
cannot design advanced algorithms. But HumanEval's problem statements and their
naive canonical solutions have been in pretraining data since 2021, and the
naive solution is exactly what a memorizing model emits. The Self-Refine
transcripts for #36 are consistent with either story — Llama 3 70B produces the
brute-force loop that is essentially HumanEval's own approach, and Table 14
shows it *restating* that loop while claiming to have used dynamic programming
over digits, which reads at least as much like retrieval as like failed
reasoning. Distinguishing "cannot design the algorithm" from "retrieves the
memorized one before reasoning starts" requires paraphrased or fresh problems,
and that experiment is absent. Problem selection compounds it: 142 of 164 kept
by expert judgment means the benchmark is conditioned on a slice of the
distribution, not sampled from it.

Relatedly, the prompting evidence is thinner than the conclusion drawn from it.
"Fundamental capability limitation" rests on one prompt variant across two
models (Table 12) and a Self-Refine case study on **one problem** and two
models. No model in the study is given the affordances the human expert had —
execution feedback, a profiler, iteration. Concluding a capability limit from a
setup with no such affordances conflates capability with affordance.

### 2.8 Execution safety and replication cost

The paper describes no sandbox. Untrusted model-generated code from 30 models,
up to 200 samples per problem, is executed for timing; a timeout is not
isolation. This one is a straightforward gap rather than an argument — we need
a sandbox for our own runs regardless. Separately, the evaluation is large
(models × samples × 142 problems × 4 levels × 4–8 cases × 6 repeats) with no
described caching, resume, pinned image, or CI, which makes independent
replication expensive enough that it rarely happens. That matters mostly
because it is why §2.1 through §2.3 could sit unexamined for a year.

### 2.9 The published rows are not fully identified

This one surfaced from trying to line our results up against theirs, and it is
the plainest reproducibility gap in the paper. Appendix C.1 names the checkpoint
behind five of the thirty display names in Table 3 (`gpt-4-1106-preview`,
`gpt-4-0613`, and the three Claude 3 dates). For the rest it says only that "for
models that are included in Liu et al. (2023a), we re-use their generated code
samples", without listing which models those are. So for 25 rows neither the
checkpoint nor the decoding settings behind the number are recoverable from the
paper: they are whatever the EvalPlus release used, and which rows those are is
left to inference.

The sampling block has the same shape. Appendix C.1 says 200 samples per problem
for "relatively smaller models" and 100 for "larger models", and never says which
models are which. (Its third clause is not a third bucket: the "largest commercial
models" are greedy-only, and Table 3 already identifies those three by giving them
no sampling row.) So the split falls across the 27 sampling rows with nothing to
resolve it. Only Llama 3 70B Instruct is pinned, at 100, and only incidentally, by
Appendix C.7 computing Table 11 "from the 100 generated samples". Since `eff@k` is
estimated from `n` samples, a reader cannot reconstruct the estimator's variance
for the other 26.

None of this affects the paper's conclusion, which is why it reads as a
housekeeping item rather than an objection. It does affect anyone trying to
reproduce a specific number, and it is the reason our parity gate targets the
greedy column, where there is no `n` to get wrong.

---

## 3. The fix list

Each row is something from §2 and what we do about it while building. Nothing
here requires the paper's conclusions to be wrong; most of it is work the
authors either flag themselves or would likely welcome.

| Observation | What we do about it |
|---|---|
| §2.1 Crisp "below 0.5" form depends on α; rank stability unknown | Full sweep over `α` and `(h₁,h₂,h₃)` across **all** models, reporting rank correlation, not just one model's score. Any claim of the form "no model exceeds X" ships with the region of hyperparameter space where it holds |
| §2.2 Score compression at easy levels | Reproduce the published metric exactly, then measure the level-wise score distribution to test the "ordinal in disguise" reading. If confirmed: per-level normalization (`Tᵢ,ₗ` from that level's reference time) so every level discriminates, reported alongside the original for comparability |
| §2.3 Machine-dependent wall clock | Pinned container image, CPU affinity, single-threaded BLAS, disabled turbo where possible; a **hardware-independent secondary metric** from retired instructions / memory accesses (`cachegrind`, `perf`) reported next to time; cross-machine rank-stability experiment on ≥2 distinct CPUs and ≥2 CPython versions |
| §2.4 Reference is an unproven ceiling | Independent audit of all 142 references; log every score above `1.0` as a candidate improvement and re-anchor when one reproducibly beats the reference |
| §2.4 Reference doubles as oracle | A second, independently written oracle per problem; differential testing between them, with disagreements blocking the problem from scoring until adjudicated and the adjudication recorded |
| §2.4 Single annotator | Every judgment call (dropped problems, level scales, corner-case validity) re-derived independently and diffed against the original; disagreement rate reported as a benchmark-quality statistic |
| §2.5 Generators tuned to the reference | **Candidate-adaptive adversarial search** — property-based generation plus an evolutionary loop maximizing *that candidate's* own runtime, so each solution is stressed on its own worst case. We report how much scores move versus the fixed generators |
| §2.5 Timeout cliff discards information | Treat over-limit runs as censored observations with a survival-analysis lower bound rather than `0`, and report how much of the model ordering depends on that choice |
| §2.6 No uncertainty | Bootstrap confidence intervals on every `eff@k`; no "A beats B" claim without a pairwise test; greedy and sampling results never tabulated as if commensurable |
| §2.7 Contamination confound | Paraphrase/mutate problem statements, add a small fresh held-out set, and report the contamination delta explicitly |
| §2.7 Affordance vs capability | Opt-in **feedback-loop track**: give the model its own timing and profile output with a fixed iteration budget; report attainable efficiency separately from single-shot |
| *(authors' D.2)* Complexity not measured | Time across ≥5 input scales and fit a scaling exponent, reporting an estimated complexity class. The authors' objection — that high-degree polynomials are indistinguishable from exponentials — is correct, so we report a confidence band and abstain rather than guess when the fit is ambiguous |
| *(authors' D.2)* Time-only metric | Peak-memory tracking (`tracemalloc` + RSS ceiling) reported as a second axis, never folded into one number, since the time–space tradeoff has no principled exchange rate |
| §2.8 Unsandboxed execution | Container isolation, no network, read-only mounts, seccomp, hard resource caps |
| §2.8 Replication cost | Resumable runs, so a crash costs one session instead of the whole run and "start again with fewer problems" stops being the cheap option; each session's machine and clock are recorded beside the problems it contributed, and a continuation onto a differently-behaving machine is refused. See [`decisions/0009-resume.md`](decisions/0009-resume.md). Still open: content-addressed result cache, pinned dependencies, CI on a smoke subset |
| §2.9 25 of 30 rows have no stated checkpoint | Record the five the paper does state, resolve run names onto table keys through case/punctuation normalization only, and report every name we could not match instead of guessing one. See [`decisions/0008-model-naming.md`](decisions/0008-model-naming.md) |
| §2.9 `n` per model unpublished | Count the released samples per model and publish the table the paper omits; until then target the greedy column, which has no `n`, and state any sampling comparison as an interval rather than a point |

Order of work: **faithful reimplementation and parity first**, then §2.2 → §2.3
→ §2.1 → §2.4/§2.5 → §2.7. Nothing in this table gets built before the original
method runs and reproduces. A fix we cannot compare against a working baseline
is not a fix, it is a fork. What "reproduces" means is fixed in advance in
[`decisions/0007-parity-gate.md`](decisions/0007-parity-gate.md), along with what
the published numbers can and cannot resolve.
