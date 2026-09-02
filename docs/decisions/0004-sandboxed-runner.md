# 0004 — Sandboxed evaluation runner

Status: accepted. Code: `enamel_ext/measure/{values,_child,sandbox,runner}.py`,
tests `tests/test_measure_runner.py` (52 tests).

This layer is the only place model-generated code runs. It turns a `Problem` plus
a solution string into per-level times and a correctness verdict, which
`enamel_ext.metrics` then scores. README section 2.8 lists the absent sandbox as
a gap in the original; this closes the process-level half of it.

## A fresh interpreter per level, and what that does not buy

Every level runs in a new `python3 -s` process with an empty temporary working
directory, a wholesale-replaced environment, `start_new_session=True`,
`RLIMIT_AS`, `RLIMIT_CPU`, and a parent-side wall-clock kill of the whole
process group.

That stops the failure modes a timeout alone does not: memory bombs, hangs
inside C code that no signal handler would interrupt, processes that outlive
their parent, and state leaking from one level into the next. It does **not**
stop network access, writes outside the working directory, or anything else
requiring kernel-level confinement; a test confirms the write escape rather than
leaving it as an assumption. `Limits` is deliberately named for what it is.
Container isolation stays on the milestone list rather than being implied by a
docstring, and the code says so in one place instead of pretending.

The obvious flag here is `-I`, and it is wrong for this job: `-I` implies `-E`,
which discards `PYTHONHASHSEED` along with everything else. Set-to-list and
dict iteration order then differ between the process that recorded the expected
output and the process that produced the candidate's, so a solution returning
`list({...})` of strings is reported as a wrong answer whenever the two seeds
disagree, and reported as correct when they happen to agree. The child instead
runs under `-s`, with the environment replaced wholesale so nothing is
inherited, `PYTHONHASHSEED=0` pinned inside it, and the repository root passed
as an argument that a one-line bootstrap puts on `sys.path` ahead of everything
else. The same bootstrap drops `''`, `'.'` and the working directory from
`sys.path`, which is what `-I` would otherwise have done. Exporting
`PYTHONPATH` was rejected for the same reason it is rejected elsewhere: an
environment variable would decide which library version measures a benchmark.

## The measuring instruments are bound before untrusted code runs

`_child` binds `time.perf_counter`, `copy.deepcopy`, the three `gc` entry
points, and `open` to module-level names at import, before anything from the
request is executed. Without that, a solution containing
`time.perf_counter = lambda: 0.0` reports 0.0 seconds for arbitrary work and
passes any limit. It is a two-line change and it closes the only channel through
which the code under test can influence its own measurement from inside the
process.

The protocol files live in a directory the child never sees as its working
directory, so a solution that enumerates or writes into `.` cannot reach the
request or forge the response.

## The response travels by file, not by stdout

Solutions print. Any protocol that shares stdout with the code under test has to
either suppress that output or parse around it, and both are fragile in a way
that shows up as a corrupt result rather than an error. The child writes JSON to
a path named in the request and stdout is discarded.

The soft rlimits are restored before that write, so lowering a limit cannot make
the child unable to report why it failed.

There are no pipes at all, and that is a deliberate second decision.
`Popen.communicate(timeout=...)` waits for end-of-file on the pipes, not for the
process to exit, so a solution that forks a helper which inherits the stderr
handle keeps the pipe open after the measurement is finished and after the
direct child has exited. The parent then reports a timeout for a run whose
result is already sitting on disk, and the solution scores 0 for something it
did correctly. Stderr goes to a file, the parent waits on `proc.wait(timeout=)`,
and a result file that exists is used even when the wall clock did expire, so a
race between finishing and being killed resolves in favour of the measurement.

## Repeats run on fresh copies of the input

This is a measurement bug the paper's description does not address. If a
solution sorts, reverses, or pops its argument in place, then with `R = 6`
repeats over one shared input only the first repeat measures the intended data.
An insertion sort would be quadratic once and near-linear five times, and the
Hodges-Lehmann estimate over those six numbers is dominated by the five that
measure the wrong thing.

`_time_case` therefore deep-copies the arguments before every call, outside the
timed window. Two consequences worth stating: the copy has to be a deep one,
since a shallow copy of a list of lists still shares the inner lists; and the
reference is timed through the same path, so whatever the copy costs is not part
of any ratio.

A related decision: the first repeat's return value is the one checked for
correctness. Later repeats are not compared against it, so a solution that is
nondeterministic across calls is scored on its first answer.

Capturing that answer means encoding it at the first call, not keeping the
reference and encoding after the loop. Holding a reference is not holding a
value: a function that appends to a module-level list and returns the list hands
back the same object every repeat, so encoding at the end reports the state
after six calls. The reference solution is subject to the identical treatment,
so byte-identical source disagrees with itself, and the mismatch reads as a
wrong answer with no plausible cause.

## Output comparison, and why JSON alone is wrong

The expected outputs come from running the reference (decision 0003), so the
candidate's return value has to cross a process boundary before it can be
compared. Plain JSON loses the distinction between a tuple and a list, and a
HumanEval signature that should return `(1, 2)` returning `[1, 2]` is a wrong
answer that JSON transport would silently accept.

`values.py` tags the container type on the way out and rebuilds the value on the
way in, so comparison happens on real Python objects. Floats travel as `repr`,
which round-trips exactly and keeps `inf` and `nan` intact. Arguments travel the
same way as return values: a materialized level whose input is a tuple, a
`bytes`, or a dict with integer keys would otherwise arrive as a list, a string,
and a dict with string keys, and the solution would be measured on data the
problem never specified.

Integers wider than 2048 bits travel as hex. JSON writes integers in decimal,
and CPython refuses to render an integer above `sys.get_int_max_str_digits()`
(4300 digits by default, and not settable below 640) as decimal at all. Any
problem whose answer is a factorial or a power reaches that width at level-3
scale, and the observed failure was not a wrong answer but a `ValueError` inside
the child's `json.dumps`, a truncated result file, and a `JSONDecodeError`
escaping `run_level` for the whole problem. Hex is exempt from the cap, so the
fix is a tag rather than a limit on what solutions may return.

Comparison is then plain `==` with one exception at every depth: floats compare
with `math.isclose`, and two NaNs compare equal because a reference that
produces NaN has not been contradicted. The tolerances are `abs_tol = 1e-6` and
`rel_tol = 1e-12`, and the split matters. HumanEval's own tests check
`abs(a - b) < 1e-6`, which is absolute; a *relative* `1e-6` accepts an absolute
error of 500 at magnitude `1e9`, so a solution that is wrong by 500 passes.
`rel_tol` is kept only to absorb accumulation slack at large magnitude and is
far tighter than double precision requires. `values_equal(True, 1)` stays true;
only container types are distinguished, because anything stricter would diverge
from what upstream's `==` accepts and break parity for no benefit.

Values with no JSON form fall back to their `repr`. That is enough for a
diagnostic and enough for comparison when the repr is meaningful, but a default
`object.__repr__` embeds an address, so two unrelated instances can compare
equal or unequal by accident. Real HumanEval signatures return built-ins, so
this has not bitten; if a problem ever returns a custom object, comparison for
that problem needs a real serializer.

Diagnostics are bounded and cannot raise. `brief` truncates at 200 characters,
because a wrong answer at level 3 otherwise carries a copy of a 32000-element
list into the results, and it renders an over-cap integer as hex, because
formatting `expected {x!r}` is itself an operation that raises for such a value.
The transport fix would have been undone by the error message describing it.

## Timeouts are three mechanisms, and only one is precise

`T_i` is on the order of microseconds to milliseconds, so:

- The child accumulates elapsed time across a case's repeats and stops the level
  at the first case whose total passes `T_i * R`. A level is scored on its worst
  case, so the remaining cases cannot change the outcome. This is the mechanism
  that produces the paper's right-censoring.
- `RLIMIT_CPU` is a backstop with integer-second granularity that also counts
  input generation, so it can only catch runaway CPU, not an over-limit run.
- The wall-clock kill catches code that never returns from a single call, which
  no in-child check can see.

The budget is the accumulated total rather than each repeat, because the score
compares the Hodges-Lehmann aggregate of the repeats against `T_i`. Censoring on
any single repeat rejects solutions the score itself would have kept: one noisy
repeat out of six, on a shared machine, is enough. For uniform repeat times the
two rules coincide exactly, so nothing changes for the typical case; they differ
only for skewed distributions, which is precisely where the aggregate is the
number that should decide.

The three mechanisms have to be ordered, or they disagree about the same
solution. `RLIMIT_CPU` reports as `SIGXCPU` rather than as a timeout, and a
child killed by a signal writes no result, so a CPU kill used to score 0 where a
wall-clock kill of the same run left the level censored and gave the earlier
levels their credit. Which one fired depended on `T_i`, with a data-dependent
crossover around half a second. `_effective_limits` therefore treats
`cpu_seconds` as a floor and raises it above the wall budget whenever the wall
budget is larger, so the precise mechanism fires first, and `-SIGXCPU` is mapped
to `timeout` rather than `crashed` for the cases where it still wins.

A wall-clock kill and an over-budget case both report `timeout`, because both
mean the same thing to the metric: the runtime is unknown and at least `T_i`.
A crash or an exception reports `error` or `crashed` instead, which the runner
turns into an incorrect verdict. Both score 0, but for different reasons, and a
harness that cannot tell "too slow" from "wrong" cannot audit its own numbers.

## The reference is measured first, and a failure there is bad data

`measure_reference` runs the expert solution on every level, aggregates its
repeats, and derives `T_i = alpha * max` over the timed levels' worst cases.
Level 0's reference times are recorded but excluded from `T_i`, matching
`metrics.score.time_limit`, which takes only the scored levels; level 0's inputs
are small and adversarial, so including them would not move the maximum but
would make the limit depend on the correctness filter.

If the reference raises, times out, or is too fast to time, `measure_reference`
raises `SandboxError`. The reference is the oracle and the denominator at once,
so a failure there is a data problem; scoring it as a low result would silently
reanchor every sample on that problem.

## Stop at the first failing level

`evaluate_solution` walks levels in order and stops at the first that times out,
errors, or disagrees with the reference. Later levels are reported `skipped` and
score as censored rather than being inferred, so the record distinguishes "was
too slow here" from "was never run". This matches the paper's treatment of
remaining levels after a timeout and saves the largest inputs, which are the
expensive ones, for solutions that have earned them.

Correctness is checked at every level, not only level 0. A solution that is
right on 8 small inputs and wrong at scale is a wrong answer, and only the
timed levels can see it.

## Level 0 decides correctness, and carries no time limit

Level 0's inputs are small and adversarial rather than large, and its purpose is
to answer whether the solution is right. Running it under `T_i` conflates two
verdicts: a solution that is slow on a hand-picked edge case is inefficient, not
incorrect, and the metric already has a way to say so. Level 0 therefore runs
once, with no `T_i`, under a wall budget only, and the timed levels keep the
limit.

The two verdicts then have to be ordered where they collide. A level that runs
out of time still reports the cases it finished, and the child keeps the first
call's output on the case that timed out, so a wrong answer among those cases is
visible. It is checked first: without that, a solution that returns `0` on the
first case and hangs on the second was recorded as correct with a censored level,
because the timeout branch broke out before comparison ever happened. Being
wrong outranks being slow.

In the other direction, a timeout at a timed level leaves `correct` true, since
slow is not wrong. A timeout at level 0 does not, because nothing was verified at
all, and calling that sample correct would inflate `pass@k` for a solution that
never produced an answer. `verified_levels` records which levels were actually
compared, so the distinction is in the data rather than implied by a status.

## One reference run serves all samples of a problem

`evaluate_problem` measures the reference once, then scores every sample against
it. That is what the paper's normalization intends, and it is `n` times cheaper
than re-measuring. The cost is that drift within a problem's evaluation is not
cancelled: if the machine slows down partway through 200 samples, the later
samples are compared against a reference timed under different conditions.
Interleaving the reference with the samples would cancel it and is the obvious
experiment once parity is established.

## Open items

- Container isolation, no-network, read-only mounts and seccomp, per the README
  milestone. The process-level layer here is the floor, not the target.
- Processes a solution leaves behind on the **success** path are not killed. The
  group kill runs when the wall clock expires and when the parent itself raises,
  but once `proc.wait()` has reaped the direct child its pid can be recycled, and
  `killpg` on a recycled pid's group would signal something unrelated. Doing this
  properly wants a cgroup or a pidfd, which is the same work as the isolation
  item above; until then a forked straggler outlives its level and can compete
  for CPU with the next one.
- Interleaved reference timing, and a measurement of how much drift there is to
  cancel on real hardware. This VM has 2 cores and cannot answer it.
- Whether `disable_gc` should default on. `timeit` disables the collector;
  leaving it enabled measures what the code actually costs, including its
  allocation behaviour. It is off by default and both settings apply to the
  reference too, so the choice is a convention rather than a bias, but the two
  should be compared once real timings exist.
- Caching and resume. The full evaluation is models x samples x 142 problems x 4
  levels x 4-8 cases x 6 repeats; nothing here persists a result yet.
