# 0011 — The calibration probe

Status: accepted. Code: `enamel_ext/measure/calibrate.py`, the calibration parts
of `enamel_ext/pipeline/record.py`, `orchestrate.py` and `summary.py`, tests
`tests/test_calibrate.py` (28 tests) plus `RecordCalibrationTest`,
`ResumeCalibrationTest` and the calibration cases in `RecordCodecTest`,
`SummaryTest` and `CliTest` in `tests/test_pipeline.py`. The measurements that
settled the two free choices are `docs/analysis/probe-floor.md`, reproducible with
`scripts/probe_floor.py`.

Decision 0007 wants to know the differential timing skew between two machines and
says plainly that it cannot check it here. Decision 0009 judges whether two
sessions of one run are commensurable by comparing four strings — `python`,
`platform`, `machine`, `cpu_count` — and lists as an open item that a stronger
check would time a fixed workload at the start of every session. This is that
instrument. The short version of what it found: it works, and on this hardware it
resolves about 1.30 where the parity tolerance needs 1.025, so its real output is
a number that says how blind it is.

## What has to be detected, which is not what a slowdown is

`T_i` in Eq. (1) is `α` times the slowest reference time for the same problem,
measured on the same machine in the same session as its candidates. Multiply every
timing on a machine by a constant `c` and both `t` and `t*` scale, so `T_i` scales,
so `f = (T_i − t)_+ / (T_i − t*)` is unchanged. A machine being uniformly twice as
slow is therefore not a comparability problem at all; it cancels exactly.

What does not cancel is a *differential* change: one kind of code getting faster
relative to another. Section 2.3 of the open questions already names the concrete
mechanism, CPython 3.11's specializing interpreter, which changed the cost of
tight interpreted loops far more than the cost of the C-level builtins those loops
compete against. A reference that leans on builtins and a candidate that leans on
the eval loop move apart under that change while a stopwatch on either one alone
would report the harmless uniform case.

This has a hard consequence for the shape of the instrument: **one timed workload
cannot measure the thing that matters.** A single number can only ever report the
uniform factor. It takes at least two workloads with different cost mixes, and the
quantity of interest is the *spread* of their ratios rather than any one of them.
The probe is a vector for that reason, not for redundancy.

## The workloads

Four, chosen to separate the interpreter's eval loop from everything else, since
that is where the known mechanism lives: `loop_arith` (interpreted integer
arithmetic in a tight loop), `bulk_builtin` (`bytes.count`, `translate`, `sorted`,
`sum` over a large buffer, so almost all C), `alloc_churn` (list and tuple
allocation with a partly-retained chain, so the allocator and the GC), and
`attr_call` (bound-method dispatch on a `__slots__` class).

Scales are fixed constants, not adapted to the machine. An adaptive scale would
mean two sessions timed different work and their ratio would measure the
adaptation rather than the machine. They were picked so all four land in the same
10-16 ms band, which the 480-sample collection below confirms they still do
(medians 13.1 to 16.1 ms). `CALIBRATION_VERSION` is bumped whenever a workload or
a scale changes, and probes of different versions refuse to be compared rather
than being compared approximately.

Replicates are interleaved — all four workloads, then all four again — rather than
blocked per workload. A machine that slows down partway through a blocked probe
would charge the slowdown to whichever workload happened to be running, which is a
differential that is not real.

The probe runs through `sandbox.run_level` with the run's own `repeats` and
`aggregator`, so its numbers come off the same clock, through the same process
setup, as every `t` and `t*` in the run. `repeats=6` and Hodges-Lehmann inside a
replicate are the paper's own settings; the probe inherits whatever the run uses,
and `comparable()` refuses a pair that used different ones.

## The statistic, and why it cannot be compared to a constant

`differential(a, b)` is `max/min` over the per-workload ratios `b_i / a_i`.
`uniform_factor(a, b)` is their geometric mean, reported alongside so a reader can
see that a much slower session is not thereby an incomparable one.

`parity.differential_bound` inverts to say that a differential factor of 1.025 is
where drift could consume the whole 0.05 parity tolerance and 1.05 is where it
could consume twice it. The temptation is to compare `differential` against 1.025
directly. That is wrong, and the reason is worth stating because it is the sort of
error that produces a confident wrong answer rather than a crash: `max/min` is a
**range statistic**. It is bounded below by 1, biased upward, and its expectation
under no drift at all grows with per-workload noise and with the number of
workloads. A range over four noisy ratios is comfortably above 1.025 on a quiet
machine that has not drifted at all.

So the statistic has to be judged against its own null, and the probe measures
that null for itself.

## The self-measured resolution

`Calibration.resolution()` computes the same `max/min` statistic between disjoint
equal halves of a single probe's own replicates, where the true differential is
known to be exactly 1, and takes the **worst** such split. Both halves are the
same width so that the two locations are the same estimator; an odd replicate
count leaves one replicate out of each split rather than comparing an estimate over
1 against an estimate over 2, which would itself be a differential. (That was a
real bug: the first version took the complement of each half, so at odd counts it
compared unequal widths, and the tuple comparison used to deduplicate splits
silently discarded most of them.)

Worst rather than median, because the two errors are not symmetric. Overstating
the instrument's noise costs sensitivity; understating it invents drift, and
inventing drift is what makes a report untrustworthy rather than merely weak. The
measurement puts numbers on that at the shipped 8 replicates: with the median split
as the threshold, 89 of 1770 probe pairs (5.0%) report drift that is not there, and
with the worst split, none do. The choice is not quite forced — the 90th-percentile
split gives 1 of 1770 — but a quantile over the 35 splits of an 8-replicate probe is
a crude thing to build a refusal on, and the maximum needs no parameter.

The applied thresholds are then `max(DRIFT_CAVEAT, res_a, res_b)` and
`max(DRIFT_REFUSE, res_a, res_b)`. The two constants are **floors, not
thresholds**. The coarser of the two probes sets the bar, because — see below — the
resolution turns out to be a property of the moment a probe was taken rather than a
constant of the machine.

## What it measures on this VM

The tables are in [`docs/analysis/probe-floor.md`](../analysis/probe-floor.md),
reproducible with `scripts/probe_floor.py`: 480 samples per workload over about
five and a half minutes, probes formed from disjoint blocks so that the true
differential between any two of them is 1 and every firing is a false alarm.

A single timing of one of these 13 ms workloads, already a Hodges-Lehmann over 6
repeats, spans 2.2× to 3.0× between its fastest and slowest observation over that
window, with p90/min between 1.48 and 1.58. That is the raw material.

At `k = 8`, the shipped setting: median resolution **1.3198**, p90 1.6559, worst
3.1292; median differential between two independent probes 1.1066; and **0 of 1770
probe pairs fired** against the applied rule. Power against a differential injected
into one workload: 2/1770 at ×1.10, 286/1770 at ×1.20, 1046/1770 at ×1.40,
1669/1770 at ×2.00.

Read that honestly. The instrument detects a doubling 94% of the time, a 40%
differential 59% of the time, and the 2.5% differential that the parity tolerance
actually cares about essentially never. **It cannot police the parity tolerance on
this hardware.** What it can do is refuse a resume across gross drift, and put in
the record the number that says it could not see less than 1.32 — which is a
better position than 0009's four strings, whose silence carries no number at all.

## Why Hodges-Lehmann across replicates, not the minimum

The minimum is the more obvious choice and was the first one. Contention,
frequency scaling and steal time only ever make a timing slower, so the fastest
replicate is the closest observation to unimpeded execution. That argument is about
the *bias* of a single location, and it is sound.

It is also not the operative question. The location is only ever used inside a
ratio between two probes, so what matters is the stability of that ratio, and an
extreme order statistic is not stable: the minimum of 8 draws from a distribution
with a long upper tail moves whenever the whole distribution moves, and the ratio
inherits the variance of both ends. Measured with the resolution computed by the
same estimator being tested, so the comparison is of the whole rule rather than of
a floor, the false alarms are:

| replicates | min | median | Hodges-Lehmann |
| --- | --- | --- | --- |
| 4 | 354/7140 | 266/7140 | 188/7140 |
| 6 | 74/3160 | 22/3160 | 20/3160 |
| 8 | 20/1770 | 0/1770 | **0/1770** |
| 12 | 6/780 | 0/780 | 0/780 |
| 16 | 1/435 | 0/435 | 0/435 |

The minimum invents drift at every replicate count tested. Hodges-Lehmann does not
pay for that safety in sensitivity either: against a ×1.40 differential at 8
replicates it detects 1046/1770 where the median detects 977, and the two are within
a percent of each other at ×2.00. The minimum's 1179 at ×1.40 looks better than
both and is not a comparable number, because its threshold is lower and that same
lower threshold is what fires on 20 undrifted pairs; sensitivity bought that way
cannot be told apart from a false-alarm rate. Hodges-Lehmann is also the same
estimator the paper already uses inside each replicate, which makes the probe one
estimator applied at two levels rather than two estimators.

The cost of dropping the minimum is real and is tested: an isolated slow replicate
used to be absorbed entirely and now costs part of its size, because
Hodges-Lehmann over one 1.0 and one 1.4 is 1.2
(`test_an_isolated_slow_replicate_costs_part_of_its_size`).

This reversed an earlier conclusion. Two 48-sample draws disagreed about which
estimator was safest — the first favoured the minimum 0/66 to 13/66, the second
went the other way 14/66 to 1/66 — and the docstring had already asserted that the
minimum "is the only estimator here that improved with replication" on the strength
of the first. Two draws were not enough to see that the disagreement was itself the
finding.

## Why 8 replicates, and why more do not help

`REPLICATES` is 8 rather than 6 because 6 is where the rule still fires on its own
noise (20 of 3160 pairs) and 8 is where it stops. The cost is a measured 5.2 s per
session, against runs measured in hours.

The number worth noticing in the full table is what does *not* happen. The median
resolution is flat in `k`: 1.309, 1.310, 1.320, 1.341, 1.323 at `k` = 4, 6, 8, 12,
16. Adding replicates does not sharpen the instrument's own noise estimate at all,
because the resolution is a maximum over the splits and the number of splits grows
combinatorially with `k`; each split gets better and there are more chances to find
a bad one, and on this machine those cancel.

What improves is the other side. The median differential between two independent
probes falls — 1.151, 1.134, 1.107, 1.082, 1.086 — so the false-alarm rate drops
because the measured quantity tightens against a threshold that stays put. This
also caps what replication can buy: past about 8 the differential is already well
inside a threshold pinned near 1.32 by the resolution, and both curves are flat, so
16 replicates cost twice as much as 8 for no additional reach. Getting below 1.32
needs a quieter machine, not a longer probe.

## The resolution is a property of the moment, not of the machine

Median resolution at `k = 8` over each quarter of the same collection window, in
order: **1.252, 1.269, 1.361, 1.499**, with worst-split values 1.45, 1.52, 1.84,
3.13. The machine became measurably noisier over five and a half minutes, on an
idle VM with nothing else scheduled. The median moved by 0.25, which is ten times
the differential the parity tolerance is trying to police.

Two consequences, both already in the code.

`compare()` takes `max(a.resolution(), b.resolution())` rather than either one or
an average, so a pair is judged by whichever probe was taken at the worse moment.
And the resolution is *reported*, in the run summary and in the record's caveats,
because a quiet verdict has two very different causes — nothing moved, or nothing
could have been seen — and only the number distinguishes them.

There is a third consequence that is not a defect but is worth naming: on hardware
this noisy `caveat_at` and `refuse_at` are both equal to the resolution, so the
two-tier design collapses to one tier and a caveat and a refusal fire together. The
tiers only separate on a machine quiet enough to resolve below 1.05.
`test_a_noisy_pair_stays_silent_about_drift_it_cannot_distinguish` asserts that
collapse directly rather than leaving it as an inference.

## Safe direction only

The probe may **add** a caveat or a refusal where the strings matched but the
machine measurably moved. It may never turn a string mismatch into a pass.
`Environment.differences()` is untouched and still excludes `calibration`, exactly
as it excludes `load_average`, so `resume_mismatches` keeps every reason it had
before and gains one. `ResumeCalibrationTest` is built around that asymmetry: a
probe cannot rescue a resume onto `machine="vax"`, and a missing, mismatched or
version-incompatible probe never refuses on its own, since absence of evidence
about the machine is not evidence about the machine.

The refusal fires on `drift.refuse`, which is drift past what twice the parity
tolerance can absorb *and* past what the probes can see. A run resumed across that
would be summing `eff` contributions measured under two different normalizations,
which is the one thing the segment machinery exists to prevent.

Drift is always measured against the **first** segment's probe rather than the
previous one, for the same reason `Segment` exists: the first session is where the
references were timed, so it defines the units the whole record is in. A chain of
five sessions each 1.1× different from its predecessor is not five small drifts, it
is one 1.6× drift from the units in force.

## What it does not do

It does not adjust anything. No score is rescaled by a measured factor, no `T_i`
is corrected. The probe's entire output is a caveat, a refusal, and a number in the
record. Correcting `eff` by a measured differential would require knowing how each
problem's reference and candidate divide between the four workloads, which is
exactly the per-problem question the probe averages over.

It also does not answer 0007's cross-machine question. That needs two machines, and
this VM is one. What it changes is that the answer will be a measurement when the
second machine exists, and that a probe from this machine is already on record to
compare against.

## Open items

- The resolution measured here is this VM's, and the non-stationarity above says
  it is not even a stable property of this VM. Nothing in the code assumes
  otherwise, but the 1.32 quoted throughout this file should be read as one
  five-minute window's answer.
- 0 false alarms in 1770 pairs is not a rate of 0. Those pairs also share blocks
  with each other, so they are not 1770 independent trials and the usual bound on
  an unobserved rate is generous here. The rule is conservative on this machine at
  this replicate count; it is not proven safe.
- All 1770 pairs come from one five-minute window, so the collection cannot speak
  to drift over hours or across a reboot, which is the separation a real resume
  has. Adjacent-block pairs — the closest thing here to two consecutive sessions —
  were clean at every replicate count from 6 up.
- Power was measured by injecting a factor into `loop_arith` alone, the largest
  single workload change the statistic can see. A differential spread over two
  workloads in the same direction is partly invisible to `max/min`, and the
  measured power is therefore an upper bound on power against real drift.
- `REPLICATES` is part of `comparable()`, so changing it from 6 to 8 makes any
  probe recorded under the old default incomparable to a new one. No run records
  exist yet, so nothing needed migrating; a future change to this constant will.
- The four workloads were chosen from the mechanism named in §2.3 rather than
  measured to be a basis for the differentials the benchmark actually contains.
  Whether four cover the reference solutions' cost mixes is answerable once the
  real problems are pinned (task #13), by correlating each reference's timing
  against the probe.
