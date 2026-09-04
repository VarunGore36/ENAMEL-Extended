# 0009 — Resuming a run without losing what it measured

Status: accepted. Code: `enamel_ext/pipeline/record.py`,
`enamel_ext/pipeline/orchestrate.py`, `enamel_ext/pipeline/summary.py`,
`scripts/evaluate.py`, tests `tests/test_pipeline.py` (121 tests).

Decision 0006 left resume as an open item: "the record is the right place to
build resume on, since it already says exactly what was measured, but nothing
reads it back for that yet." This is that. `evaluate.py run --resume <record>`
measures what the record is missing, retries what it failed, and writes the two
sessions back as one file.

The motive is milestone 2. A parity run over 142 problems, 6 repeats, and every
sampled model is hours of wall clock, and until now a crash at hour three cost
all three hours. That is not only an inconvenience: it makes the honest reaction
to a mid-run failure "start again with fewer problems", which is exactly the
shortcut §2.8 of open-questions.md criticises the original for making tempting.

## Why splitting a run across sessions is sound for this metric

The obvious objection to resume is that a benchmark of *speed* cannot be
measured in two sittings, because the machine is not the same machine twice.

For Eq. (1) that objection is weaker than it looks, and the reason is where `T_i`
comes from. Each problem's limit is `alpha * max t*` measured from that problem's
own expert reference, in the same session as that problem's own candidates. A
candidate time and the reference time it is divided by are therefore always
neighbours in wall clock. If the second session's machine is uniformly `c` times
slower, then for every problem in it both `t` and `t*` scale by `c`, and

    f = (alpha * c * max t* - c * max t)_+ / (alpha * c * max t* - c * max t*)

cancels `c` exactly. The level fraction, and so `e_ij`, and so `eff@k`, is
unchanged. A uniform slowdown between sessions is invisible to the score.

What does not cancel is a *differential* change: a different CPython, a different
CPU, anything that speeds some code up and other code down. Then `t/t*` moves
per problem and the two halves of the record are measurements of different
things. This is the same argument decision 0007 uses to set the parity tolerance
at 0.05 from a 1.025 differential bound; here it is applied across time on one
machine instead of across two machines.

So the rule that falls out is not "refuse to resume" but "refuse to resume onto
a different machine, and say so in the file when the machine changed anyway".

## Segments, because one `Environment` per record became a lie

Before this change a `RunRecord` had one `started`, one `finished` and one
`Environment`. A resumed record with those fields alone would claim that every
problem in it was measured on the machine named in the header, at a time inside
the header's interval. Half of it would be false, and nothing in the file would
show which half.

`Segment` is one measurement session's contribution: its own `started`,
`finished`, `Environment`, and the `problem_ids` it measured.
`_check_segments` then enforces the property that makes the record trustworthy
again: every scored problem belongs to exactly one segment. Two segments
claiming one id is refused, an id no segment claims is refused, and a segment
claiming an id the record does not hold is refused. Without that check a
resumed record could credit half of a level mean to the wrong machine, and the
report would print it without hesitation.

A single-session record still has one segment, defaulted from the record's own
header, so `segments` is never empty and nothing downstream needs a special
case. `resumed` is `len(segments) > 1`, `drift()` compares each later session's
machine against the first one's, and both feed `caveats()`, which the report
prints in its header. A resumed run therefore says "measured over 2 sessions: 87
from …, 55 from …" where the numbers are read, not in a footnote.

The record keeps its top-level `started`, `environment` and so on. They now mean
the session the run *began* in, which is what the existing report line has always
described, and dropping them would have broken every reader for no gain.

## `load_average` is not a comparability field

`COMPARABLE_FIELDS` is `python`, `platform`, `machine`, `cpu_count`.
`Environment.load_average` is deliberately absent.

Load average differs between almost any two moments, so including it would
refuse essentially every resume. More to the point, it is noise *on* the
measurement rather than a change in *what is being measured*: a busy machine
makes both `t` and `t*` slower, which is the uniform case that cancels. It is
still worth recording, and `Environment.caveats()` still complains when the run
started under load, but it is a reason to distrust a number rather than a reason
to call two numbers incommensurable.

## The schema bump has no migration path

`RECORD_SCHEMA_VERSION` goes 1 to 2 and `record_from_json` still checks equality,
so a schema-1 file is rejected rather than upgraded. Writing the migration would
be easy: a schema-1 record is exactly a schema-2 record whose single segment is
its own header.

It is not written because there is nothing to migrate. 0006's own open item says
the equality check "is right while the schema is a week old and wrong once a
parity number depends on a file we want to keep readable", and no parity number
exists yet: nothing has been measured on the 142 problems. Writing a migration
now would mean maintaining a code path with no input. The moment the first real
run lands, that open item becomes live and this paragraph becomes wrong.

## A failure is retried; a success is never re-measured

`resume_evaluation` attempts exactly the selected problems the record does not
already hold. A problem with a `ProblemRecord` is never measured again, because
re-measuring it would either waste the session or, worse, replace a first-session
measurement with a second-session one and break the neighbour property above.

A problem in `failures` is always retried. `keep_going` records a problem whose
*reference* did not run, and a reference can fail for reasons that are not
properties of the code: a killed subprocess, a transient resource limit, a
machine that was momentarily out of memory. Treating that as a permanent verdict
would bake a lost race into the denominator. If it fails again it is simply
recorded again, and the retry costs one reference measurement.

A retried problem that succeeds is dropped from `failures`, since the list means
"has no reference measurement in this record" and it now has one.

## Every refusal at once

`resume_mismatches` returns all the reasons a session cannot extend a record,
and `resume_evaluation` raises with the whole list. Stopping at the first would
mean a user fixing `--repeats`, waiting, being told about `--alpha`, fixing that,
waiting, and being told about the fingerprint. The checks are cheap and none of
them measures anything, so there is no reason to be stingy.

What is checked: the schema version, the metric (`alpha`, `h`, normalization),
`R`, the aggregator, the problem-set fingerprint, the solution-set fingerprint,
the machine's comparable fields, the problem selection, and the model set.
Between them they cover every input that decides what a number means. The
fingerprints are the load-bearing pair: they are why "the references are not the
same code" is a refusal rather than something a reader has to notice.

The model-set check is the subtle one, and the obvious version of it is wrong.
Requiring `set(models) == set(record.models)` refuses a legitimate case: a model
whose samples all fall in problems the record has not reached yet is absent from
`record.models` for no other reason than that, and there is nothing wrong with
measuring it now. Two narrower checks say what is actually needed. A model the
record measured but this session does not request is refused, because its
average would end up over a subset of problems chosen by when the crash
happened. A model the record does not cover but which has samples for problems
already measured is refused, because those samples can never be taken now and
the model's average would be over the complement.

The selection check is the same idea for problems: a resume whose `--ids` or
`--limit` would not attempt something already measured is refused, since the
result would be a record wider than the run that produced it.

## Saving is atomic

`save_record` writes `<name>.partial` and `os.replace`s it over the target.
Resume overwrites the file it just read, and that file may be the only copy of
hours of measurement; a crash between truncate and flush on a plain write would
destroy exactly the thing resume exists to protect. `os.replace` is atomic within
a filesystem, so the target is either the old record or the new one. Serializing
before opening the scratch file means a record that cannot be encoded leaves no
scratch file behind either.

`--resume` with no `--out` writes back over the record it extended, since the
alternative leaves a directory of partial records and no indication which is the
longest. `--out` still overrides it for anyone who wants the intermediate kept.

## Open items

- A resumed session that measures nothing still appends an empty segment. That
  is honest about where the wall clock went and it reads oddly ("0 from …"). If
  repeated retries of a permanently broken reference become common, collapse it.
- Nothing checkpoints *within* a session. A crash still loses whatever the
  current session measured since it started, because the record is written once
  at the end. Periodic saves would need the segment to be closed and reopened,
  or a partial segment concept, and the retry rule already makes the loss
  bounded by one session.
- Comparability is judged on fields that are cheap to read, not on evidence. Two
  sessions on the same `platform` string can still differ in CPU frequency
  scaling, thermal state, or a kernel update that did not change the string. A
  stronger check would time a fixed calibration workload at the start of every
  session and compare, which is the same instrument decision 0007's differential
  bound wants and neither has yet.
- The tolerance argument above assumes the paper's global normalization. Under
  the per-level normalization of §2.2 each level has its own limit but still one
  reference per problem per level measured in the same session, so a uniform
  slowdown still cancels; this has not been re-derived carefully for the
  variant's censoring behaviour.
