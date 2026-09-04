# 0007 — The parity gate

Status: accepted, and pre-committed. Code: `enamel_ext/data/published.py`,
`enamel_ext/report/parity.py`, the parity section of
`enamel_ext/pipeline/summary.py`, tests `tests/test_parity.py` (94 tests, 6 of
them the gate itself, which skips until a run record exists). How a run's model
names reach the published keys is decision 0008.

README milestone 2 says parity gates the rest of the list. That is only a real
constraint if the criteria are fixed before the measurement, so this file fixes
them. Everything here was derived from the paper's published tables and from the
metric's own algebra, with no timing measured on either side.

## The tolerances, in one place

`eff@1` within **0.05** of the published value, per model. `pass@1` within
**0.01**. No inversion of a model pair the published table separates by more
than **0.10**, which is twice the `eff` tolerance. The gate's verdict is those
three and nothing else; coverage is reported beside it rather than folded into
it, because a run over three models can satisfy all three and is not parity.

## What the published numbers can resolve at all

Table 3's greedy `eff@1` column has 30 models and so 435 pairs. Of those, 356
(81.8%) are separated by more than 0.05, which is the gate's power: a comparison
against four fifths of the pairs is a real test of the leaderboard's shape.

The adjacent pairs tell the opposite story. There are 29 of them, their median
gap is 0.013, six are at or under 0.005, the smallest is 0.001, and **exactly
one survives a 0.05 tolerance**. So at any tolerance loose enough to survive a
change of machine, the published ordering of neighbours is not reproducible even
in principle. That is a property of the published spacing, not a defect in our
harness, and it is the reason ordering is checked on resolvable pairs only. A
claim about adjacent models would need agreement near 0.005, which is below what
we have any reason to expect across two machines, and settling that is
milestone 3's cross-machine experiment rather than this gate's business.

## Why rank correlation is not one of the criteria

A Kendall tau over the whole table is dominated by the pairs that are far apart,
and those are the easy ones. `parity.tau_floor` builds the worst case and
measures it: invert as many near-tied adjacent pairs as disjointness allows and
tau still reports **0.931**. A tau threshold anywhere a reasonable person would
put it therefore passes a result that has every locally contested ordering
backwards. Tau is printed in the report next to that floor so it cannot be read
as a criterion, and the criterion is the inversion count instead.

The paper supplies the other end of the same argument, which is the stronger
half because it needs no construction. Table 7 ranks its top twelve greedy models
twice, once by `eff@1` and once by the classic speedup metric, and Appendix C.3
calls the two rankings "very different" and argues from that difference that
speedup is not a reasonable metric under censoring. Kendall tau between them is
**0.848** (`parity.published_disagreement_tau`, five discordant pairs of 66). So
a disagreement the authors treat as disqualifying for an entire metric sits
*below* the tau our own maximally-wrong local ordering would still earn. There is
no threshold between the two, and a criterion has to live somewhere.

Table 7 also sharpens the resolvability point above: it prints a rank order for
neighbours separated by 0.001 (Claude 3 Sonnet over Llama 3 8B Instruct) and
0.002 (Code Llama 34B Python over Mixtral 8x7B Instruct), both far under what
this decision claims two machines can resolve. Publishing an order is not the
same as establishing one, which is why the gate judges only the pairs the table
separates widely.

## Where 0.05 comes from

Three anchors, of which one binds.

It keeps the power described above: 356 of 435 pairs stay resolvable, against
405 at 0.02 and 276 at 0.10.

It absorbs a specific amount of timing skew, and this is the anchor that binds.
A uniform slowdown cancels exactly, because `T_i` is set by the reference
measured on the same machine, so only a *differential* change in the relative
speed of different algorithms matters. Writing Eq. (1) in units of the
limit-setting reference time gives `f = (α − s·q)/(α − q)` for a candidate `s`
times slower than a reference whose worst case is `q ≤ 1` of that constant, so
scaling `s` by a factor `c` moves `f` by at most `(c − 1)·α/(α − 1)`, which is
`2(c − 1)` at `α = 2`. A tolerance of 0.05 therefore covers a systematic 2.5%
differential in the worst case, and considerably more when the differentials are
mixed in sign across problems and average down. `parity.differential_bound` is
that expression, so the justification is executable rather than prose.

It is tighter than the metric's own preference sensitivity. The paper's α sweep
moves GPT-4 Turbo's `eff@1` by 0.120 across [1.5, 3.5], and reweighting `h`
alone spans 0.283 at `k = 1` (docs/analysis/table10-recovery.md). Agreeing to
0.05 is a stronger statement than agreeing about either hyperparameter.

The honest weakness is that the second anchor cannot be checked here. This VM has
two cores and no second machine, so the differential factor between our hardware
and a 2.20GHz Xeon is unmeasured. If parity misses at 0.05 the answer is to
report the miss and the measured deviations, not to widen the tolerance
afterwards; the whole point of writing the number down first is that it cannot
move once the numbers are in.

## Why `pass@1` is the tight test, and the one signed prediction

`pass@1` is a correctness rate. Given the same problems, the same samples and the
same Python, it should not depend on the machine at all, so 0.01 is generous
rather than tight, and a miss there points at our correctness comparison, our
expected outputs, or a missing problem, all of which are our bugs.

One channel is expected to push it, in a known direction. Section 2.1 of the
paper says a code sample is excluded from `pass@k` if "the output of the code does
not match the expected output in any test case or does not pass level 0", and
that a sample which passes level 0 but exceeds the time limit at some level
`l ≥ 1` is still counted. Two things follow. Correctness is checked at every
level and not only at level 0, which is what decision 0004 already does, so that
is parity rather than a deviation. And the treatment of a sample that is correct
but slow *at level 0* is left open: the timeout clause is stated only for
`l ≥ 1`. Our level 0 runs under a wall budget with no `T_i` (decision 0004), so a
slow-but-correct sample passes for us. If theirs applied `T_i` there, it did not
pass for them. Any excursion beyond 0.01 should therefore be **positive**, and
the gate asserts the sign separately from the magnitude. A negative excursion is
the more serious signal of the two.

There is a second channel with an unknown sign, worth naming rather than
assuming away: a sample that is both wrong and over the limit at the same level
counts as incorrect for us, because a timed-out level still reports the cases it
finished and being wrong outranks being slow. Whether their implementation
compares outputs on a level it abandoned is not something the paper says. The
paper's stated rule is unconditional, so the text is on our side here.

## What Table 11 cannot contribute

The obvious place to look for a tolerance is the paper's own variance table, and
it turns out to have nothing to give.

Its Rao-Blackwellized row is exactly its vanilla row put through the paper's own
Eq. (8), at both `k`, to the printed precision: `0.20·√(1/100) = 0.0200 → 0.02`
and `0.25·√(10/100) = 0.0791 → 0.08`. Two decimals make that consistent with a
derived row rather than proof of one, and it does not make either number wrong,
but it does mean the row carries no information about the estimator's realized
variance beyond Theorem 1. A related gap: the RB estimator is a deterministic
function of a fixed sample set, so the stated protocol for the vanilla row (1000
random `k`-subsets of 100 samples) cannot be the protocol behind the RB row, and
the text gives no other.

Those figures are also per-problem rather than benchmark-level, which is provable
rather than merely likely. Eq. (1) and (2) bound a sample's score by
`α/(α − 1) = 2`, a random variable on `[0, B]` has variance at most `B²/4`, so a
per-problem standard deviation is at most 1 and the mean over 142 independent
problems is at most `1/√142 = 0.084`, well under the printed 0.20. Carried down
to benchmark scale the RB figure is `0.02/√142 ≈ 0.0017`. Greedy decoding, which
is the column the gate uses, carries no sampling noise at all. So the published
side of the comparison is quiet to well within its three printed decimals, and
the entire tolerance is about our side: timing, machine, reference behaviour and
problem coverage.

## How the three criteria relate

They are not independent, and it is better to say so than to present three
hurdles that are really one and a half.

If every model is within 0.05, then any two models move by at most 0.10 relative
to each other, so a pair separated by more than 0.10 cannot change places. The
inversion criterion is therefore *implied* by the deviation criterion, and its
job is to catch a comparison that contradicts itself rather than to add
difficulty. `tests/test_parity.py` asserts the implication against the real table
rather than trusting the algebra.

The independent content sits in the band between one and two tolerances. A pair
separated by 0.06 can invert while both models are inside 0.05, and 80 of the 435
published pairs are spaced that way, one of them adjacent. Those inversions are
counted and printed and deliberately not gated, since gating them would be
gating something the tolerance already declined to resolve.

## Coverage is reported, not gated

`ParityResult.passed` speaks only for the models compared. `missing` lists the
published models a run did not cover and `extra` lists models we ran that the
paper never published, and the pair counts in the report describe the models
actually compared rather than the full table, so a six-model run cannot borrow
the thirty-model table's discriminating power. The gate's own coverage assertions
are separate tests: all 142 problems, and no missing model.

One shape of emptiness is refused, and it is worth being precise about why it is
not a coverage rule. All three criteria are stated as absences — no deviation over
tolerance, no gated inversion — so a comparison whose model overlap with the
published table is *zero* satisfies all three by having nothing to check, and the
first version of this printed "compared 0 of 30 models" directly above
"verdict: pass". That is the likely failure mode in practice rather than a
contrived one: our run's models are named by whatever produced the samples, the
published tables are keyed by the paper's display names, and one mismatch in that
mapping empties the intersection without erroring anywhere. `passed` therefore
requires at least one model compared on both columns. The distinction being drawn
is between a weak result and an absent one: two models compared is weak evidence
and passes, nought models is not evidence and cannot.

## Cross-checks on the transcription

The published numbers are typed in by hand, so `published.py` carries checks that
a slipped digit tends to break. Table 3's sampling row minus Table 6's two
subsets leaves the 47 problems in neither, and every implied remainder lands in
range across 27 models and 6 columns. Table 7's `eff@1` column is an independent
printing of Table 3's greedy top twelve and reproduces our sorted order exactly.
Table 9's ENAMEL row repeats Table 3's greedy `eff@1` for Code Llama 34B Python.
Table 12's basic rows repeat Table 3's greedy rows for the two models it covers.
Table 10's default column reproduces Table 3's GPT-4 Turbo entry at `α = 2` and
at each default hardness. `pass@k` rises with `k` everywhere, no `eff` exceeds
`α/(α − 1)`, and every published `eff` sits below its own `pass` — the last of
which is a fact about these models rather than a law, since a sample that beats
the reference scores above 1.

## Open items

- The 47-problem remainder assumes Section 4.2's two subsets are disjoint. The
  paper implies it by calling one set "hard" and the other "seemingly easy" but
  never states it. Every implied remainder being in range is evidence, not proof.
- `n` per model is not published. Appendix C.1 gives 200 samples for "relatively
  smaller models" and 100 for "larger models" without saying which are which, and
  only Llama 3 70B Instruct is pinned, at 100, by Appendix C.7. Counting the
  released samples would settle it, and the gate uses the greedy column partly to
  avoid depending on the answer.
- If their samples are not released, our sampling numbers carry their own
  sampling noise and the `pass@1` criterion has to loosen to a stated interval
  rather than a point. The greedy column has no such problem, which is the other
  reason it is the primary target.
- Table 6 gives three targets per model over disjoint problem subsets rather than
  one, so a discrepancy can be localized to the algorithm-design problems or the
  implementation-optimization ones. Worth wiring in once the full-set comparison
  runs; `compare(name="algorithm")` already reaches those tables.
- Nothing here tests `eff@10` or `eff@100` yet, though `compare(k=10)` works. The
  order statistics make those noisier on our side and the sample-size question
  above lands on them hardest.
