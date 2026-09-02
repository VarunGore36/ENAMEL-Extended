# 0006 — The run record, the orchestrator and the report

Status: accepted. Code: `enamel_ext/pipeline/record.py`,
`enamel_ext/pipeline/solutions.py`, `enamel_ext/pipeline/orchestrate.py`,
`enamel_ext/pipeline/summary.py`, `scripts/evaluate.py`,
tests `tests/test_pipeline.py` (75 tests).

Decisions 0001 through 0005 each built one layer: the metric, the reporting
statistics, the data adapter, the sandboxed runner, the snapshot pin. Nothing
joined them, so there was no artifact a parity claim could point at. This layer
is that artifact. One command measures a set of model samples against the expert
references, writes a file describing what was measured, and prints a report
derived from that file.

## The record stores times, and every score is recomputed

A record holds per-sample, per-level worst-case times and a correctness verdict.
It holds no `eff@k`, no per-sample `e`, no per-level fraction. Every number in
the report is computed from the times at the moment it is printed.

The alternative is storing scores next to the measurements, and the failure mode
there is a file that disagrees with itself: a stored `eff@1` computed under one
`alpha` sitting beside times that yield another, with nothing in the file saying
which one the reader is looking at. Since §2.1 of the README is specifically
about how much the published numbers move with `alpha` and `h`, the sweep is not
an optional extra view of a run, it is the main one, and the record has to make
re-scoring the cheap path rather than a re-measurement.

That makes the report a pure function of the record, and `evaluate.py report`
re-derives the same text from a saved file with no measuring at all. It also
means the `alpha` and `h` columns in the report are not separate experiments:
they are the same measurements read at another threshold.

## What has to be in the file for re-scoring to be possible

Eq. (1) needs, per problem, the reference's worst case at each timed level and
the candidate's worst case at the same levels, plus `T_i`. So the reference times
keep level 0 and the candidate times do not: level 0 is a correctness filter that
contributes a verdict, and keeping a time for it would invite someone to weight
it. `timed_reference()` drops it, and `n_timed_levels` is checked against the
metric's `n_levels` so a three-level record cannot be read with a four-weight
metric.

`T_i` is stored rather than recomputed on load, and then validated against
`alpha * max(timed reference)` to a relative tolerance of `1e-9`. Storing it
alone would let an edited file rescale every score silently; recomputing it alone
would lose the fact that the candidates were actually run against a specific
limit. Keeping both and comparing them means a self-inconsistent file fails at
construction, with the two values in the message.

The environment, the data and solution fingerprints, `R`, and the aggregator are
in the file for the same reason the snapshot lock exists: a parity number that
cannot say which bytes and which machine produced it is not a parity number.
`Environment.caveats()` turns two of those facts into sentences the report
prints in its own header, so a run taken on two cores or under load says so where
the numbers are read rather than in a footnote nobody opens.

## Censoring travels as a string, and only one direction is safe

A censored level time is `math.inf` in memory, which JSON cannot represent. These
files are written with `allow_nan=False`, so `Infinity` is not silently emitted
either; the codec maps infinity to the string `"censored"` and back. That keeps
the file parseable by anything, and it makes the sentinel visible to a reader
skimming it, which `1e30` or `-1` would not.

The asymmetry that follows is the reason the guard exists. Lowering `alpha`
re-scores fine: a sample already past the old limit is past the new one too.
Raising it cannot be answered from the record, because a censored sample's true
time was never observed, and treating it as exactly the old limit would credit
the slowest possible run with the best score consistent with the data. So
`sample_scores` refuses an `alpha` above the measured one when any recorded time
is censored, and the alpha sweep in the report is filtered to values at or below
the one the run used, with a line saying that going higher needs a new run. This
is the same censoring-invariance the paper relies on, stated as a constraint on
what a saved run can be asked rather than as a property of the score.

Anything other than the paper's global normalization cannot be re-scored at
another `alpha` at all: the guard is defined for one `T_i` per problem, and
per-level normalization (the §2.2 fix) changes what a censored observation bounds.
It refuses rather than approximating.

## A reference that fails is bad data, not a slow reference

If the expert solution for a problem cannot be measured, the problem has no
`T_i`, and every candidate for it would score 0 for a reason that has nothing to
do with the candidates. The default is therefore to stop the run.

`keep_going` records the problem in `failures` with the sandbox's message and
leaves it out of every model's average. What it does not do is quietly shrink the
denominator: the report prints how many problems were scored, how many had no
reference, and which ids, because a mean over 130 problems and a mean over 142
are different quantities and the difference is invisible in the number itself.

## The loop is problem-major, and samples interleave

Iterating problems on the outside and models on the inside means the reference is
measured once per problem and that one `T_i` enters every model's score. The
other order gives each model its own reference measurement, so two models'
scores would differ by reference noise as well as by their own times, and §2.3's
point about wall-clock variance would apply to the denominator independently for
each model.

Within a problem, samples are visited by index across models rather than model by
model: sample 0 of every model, then sample 1, and so on. Machine state drifts
during a run, and iterating model by model puts the last model's samples
systematically further from the reference measurement than the first model's. The
interleaving does not remove the drift, it stops it from lining up with the thing
being compared.

`selected_ids` resolves the problem list before anything is measured, and it is
the same function the CLI uses to validate `--models` and `--ids`. An unknown
model name fails before the first subprocess starts rather than after an hour of
timing.

## The report says what it dropped

Models are aligned on the intersection of the problems they answered for paired
comparisons, and the pair count is printed next to each difference. Two models
compared on different problem sets are not comparable, and the intersection is
the smallest honest fix; the alternative, comparing each model on whatever it
covered, produces a difference that partly reflects which problems each model was
sampled on.

`level_means` is computed by re-scoring with one-hot weights, one level at a
time, rather than by averaging per-level fractions directly. That makes
`eff_at_h(level_means(m), h) == eff@1` an identity of construction rather than a
coincidence two code paths have to maintain, which matters because the hardness
section's whole argument is that the level means determine the score. It holds at
`k = 1` only, and the section labels itself `eff@1` for that reason.

The level-discrimination table is the §2.2 measurement wired to real reference
times: `q` per level, the slowdown past which a level scores 0 for everyone, each
level's share of the score's response to a slowdown, and how often `T_i` is set
by a level other than the last. On synthetic data it already reports what the
arithmetic predicts, level 3 carrying the overwhelming majority of the response.
What it needs to answer §2.2 is the real snapshot.

## The CLI has two verbs and three exit codes

`run` measures, saves a record under `runs/run-<utc>.json` unless told otherwise,
streams per-problem progress to stderr and prints the report to stdout. `report`
takes a saved record and prints the same text. Splitting them keeps the
expensive half from being re-run to change `k`, the confidence level, or the
bootstrap seed.

Exit 0 is success, 1 is a run that could not measure what it was asked to, and 2
is a usage error. The distinction is for scripts: a 2 means fix the flags and
nothing was executed, a 1 means the flags were fine and the machine or the data
was not. Synthetic inputs are the default only when neither `--problems` nor
`--solutions` is given, since the synthetic samples answer the synthetic problem
and nothing else; passing one without the other is a usage error rather than a
silent half-synthetic run.

## Open items

- No caching and no resume. A re-run measures everything again, which is the
  README's §2.8 complaint about replication cost still standing. The record is
  the right place to build resume on, since it already says exactly what was
  measured, but nothing reads it back for that yet.
- Solution sets are JSON only, and nothing yet converts model output into one.
  The format is deliberately dumb, a mapping from model name to problem id to a
  list of code strings, so that whatever produces samples does not have to know
  about this package.
- `schema_version` is checked for equality, so an older record is rejected rather
  than migrated. That is right while the schema is a week old and wrong once a
  parity number depends on a file we want to keep readable.
- The record stores per-level worst-case times, not per-case times, following
  Eq. (1)'s `max` over cases. That is enough to score and not enough to ask which
  test case was slow, which the §2.5 adversarial work will want.
- Untested at scale. The largest run so far is the synthetic problem set on two
  cores; every timing claim here waits on milestone 2.
