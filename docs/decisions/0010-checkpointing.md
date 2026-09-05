# 0010 — Checkpointing a run, and records that admit to being partial

Status: accepted. Code: `enamel_ext/pipeline/record.py`,
`enamel_ext/pipeline/orchestrate.py`, `scripts/evaluate.py`, tests
`tests/test_pipeline.py` (19 new tests).

Decision 0009 left this open: "Nothing checkpoints *within* a session. A crash
still loses whatever the current session measured since it started, because the
record is written once at the end … the retry rule already makes the loss bounded
by one session."

That bound is vacuous exactly when it matters. On the first parity run there is
no earlier record to fall back to, so "one session" is the whole run: 142
problems, `R = 6`, every sampled model, hours of wall clock. A crash at hour
three still costs three hours, and the honest reaction is still "start again with
fewer problems" — the shortcut §2.8 of open-questions.md criticises the original
for making tempting. Resume made recovery *possible*; it did not make the first
run survivable.

So `run_evaluation` and `resume_evaluation` take a `checkpoint` path and write
the record-so-far as they go, and `evaluate.py` points that at the same file the
run will end up in.

## A checkpoint is a whole record at the final destination

Every flush writes a complete, valid `RunRecord` to the path the run would have
written at the end. Not a sidecar, not a numbered series, not a log of deltas.

The reason is that recovery should have no choices in it. `--resume` already
takes a record and writes back over it, so if a checkpoint *is* the destination
then the instruction after a crash is "run the same command with `--resume` and
the file you already have", and there is nothing to pick from and nothing to
reconstruct. A `.partial` sidecar would mean a second recovery path that only
exists for crashes, which is the path least likely to be exercised and most
likely to be wrong. Numbered checkpoints would mean a directory where the
question "which of these is the run" has an answer the user has to work out.

Overwriting the destination repeatedly is only safe because `save_record` was
already atomic — it serializes, writes `<name>.partial`, and `os.replace`s over
the target — which 0009 introduced for the resume case where the file being
overwritten is the only copy of hours of measurement. Checkpointing makes that
property load-bearing several hundred times per run instead of once.

## The hazard it creates, and `attempted`

Writing partial records to the destination puts a file on disk that looks exactly
like a finished run. Nothing in a schema-2 record distinguished the two, and the
problem count cannot: a run invoked with `--limit 40` legitimately holds 40
problems, and so does a 142-problem run that died at 40. Read the second one and
every mean in the report is over a prefix chosen by when the crash happened,
silently.

`attempted` is the ids the run set out to measure. `complete` and `missing()` are
derived from it rather than stored, which is 0006's rule about the record holding
measurements and nothing that can be recomputed from them.

Three details of how it defaults, none of them arbitrary. A record that says
nothing about `attempted` gets "exactly what it holds", so every hand-built record
and every test fixture is complete without ceremony. A record holding a problem it
never claimed to attempt is refused, because that combination has no honest
reading. And on resume the new `attempted` is the *union* of the record's and this
session's, so narrowing `--ids` or `--limit` on a continuation cannot make an
interrupted run look finished — the record remembers the largest scope it ever
claimed.

Being incomplete then surfaces through `caveats()`, which the report prints in its
header beside the resume and load caveats: "interrupted: 43 of 142 attempted
problems were never measured, so every mean here is over a prefix". The point of
the caveat channel is that it appears where the numbers are read rather than in a
footnote, and an interrupted run is the strongest case for it yet.

## One `compose` for the flush and for the return

Each entry point builds a closure `compose(records, failures) -> RunRecord` and
uses it for both the periodic flush and the value it returns. A checkpoint
therefore cannot drift into a different shape from a finished record; there is no
second construction site to keep in step.

This matters most for segments. The in-progress session's `Segment` is
*rewritten* at every flush, not appended to, so at each flush point the record
still satisfies `_check_segments`: every problem it holds belongs to exactly one
segment, and the segment covering the current session lists exactly the problems
that session has finished. Appending a segment per flush would have produced a
record with one segment per problem, all claiming the same machine, and would
have turned "measured over 2 sessions" into "measured over 87 sessions".

A checkpoint's `finished` is the moment of the flush, so the last checkpoint and
the returned record differ in that one field and nowhere else. That is the right
way round: a record on disk should not claim to have finished later than it did.

## The cost, and the knob

Writing the whole record at every problem is quadratic in bytes: the *n*th flush
writes roughly *n*/*N* of the final file, so a run writes about half of *N* times
the final size over its life. For 142 problems and a record of a few megabytes
that is a few hundred megabytes of writes across hours of measurement, against a
process that is spending that time deliberately timing code. It is not free, and
it is not close to the dominant cost.

`--checkpoint-every N` is the knob for anyone who disagrees, and `0` turns
checkpointing off entirely (as does `--no-save`). The default is `1` because the
natural unit of loss is one problem, and one problem in the parity run is minutes.
A stride of `N` loses at most the tail: the run still returns and saves everything
it measured, so the stride only affects what survives a crash.

## The destination has to be known before the run

A checkpoint needs somewhere to go, so `evaluate.py` resolves the output path
before measuring rather than after. Two consequences worth stating.

A default filename is now stamped with the run's start time instead of its finish
time. That is arguably better — it matches when the file first appears — but it is
a visible change in what `runs/run-*.json` names mean.

`_out_path` memoizes into `args.out`, because the destination is now asked for
three times: by the checkpointer before the run, by the final save after it, and
by the crash hint after that. Recomputing a timestamped default at each ask would
point the checkpoint and the final record at two different files, and the crash
hint at a third that never existed.

`--out` at a path that already holds something now clobbers it at the first
finished problem rather than at the end of the run. That is the same clobber
`--out` always had, sooner.

After a `SandboxError` the CLI prints what is salvageable, worded to be true in
all three cases: the file exists and holds measurements, nothing reached it yet,
or nothing was being written at all. It names the exact `--resume` invocation in
the first case, which is the whole point of the feature and not something a user
should have to infer.

## A bug that only a checkpoint could see

Designing this turned up a latent fault in 0009's resume code. It computed
`retried = set(attempt)` and dropped every prior failure in that set from the new
record's `failures`, on the grounds that a retried problem either succeeds (and
gets a `ProblemRecord`) or fails again (and gets a fresh entry).

That is true only once the session has finished. It says a failure is superseded
as soon as the session *plans* to retry it, and a record written before the retry
actually happens loses the reason the reference failed the first time. Nothing
ever observed this, because records were only written after the loop.

The filter is now `handled`: the problems this session has actually measured or
actually failed. Once a session completes, `handled == set(attempt)` and the
result is byte-identical to before, so nothing about finished records changes.
Mid-run, a prior failure keeps its recorded reason until the retry reaches it.

## Open items

- A stride greater than 1 makes the tail of a crashed run unrecoverable in units
  of `N` problems, and nothing warns that the file on disk is up to `N` problems
  behind the process that is writing it. Reporting the lag alongside progress
  would be cheap.
- The whole record is re-serialized per flush. An append-only journal of finished
  problems, compacted into a record at the end, would make the cost linear. It is
  not written because the quadratic cost is small against measurement and because
  a journal is a second on-disk format with its own recovery path, which is the
  thing this decision spent its argument avoiding.
- `attempted` records the *selection*, not the reason a problem is in it. A run
  narrowed by `--models` attempts fewer problems than one narrowed by `--ids` to
  the same set, and the record cannot tell them apart. Nothing needs it yet.
- Checkpointing and `keep_going` interact in a way worth watching on real data: a
  reference that fails for a transient reason is retried on every resume, so a
  problem that is genuinely unrunnable will be re-attempted once per continuation.
  0009 accepted that cost for one resume; it is unbounded across many.
