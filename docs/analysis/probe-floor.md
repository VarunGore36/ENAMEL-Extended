# This machine's calibration-probe floor

Measured 2026-09-06 on the project VM. Reproduce with
`python scripts/probe_floor.py` (resumable; repeat until it stops asking, then
`--report`). The design decisions this measurement settled are decision
[0011](../decisions/0011-calibration-probe.md); the numbers live here.

Everything below is one machine at one moment: a 2-core Linux VM, otherwise idle,
CPython as pinned by the project. It is not a claim about hardware in general, and
§4 is specifically the finding that it is barely a claim about this machine.

## Method

480 samples per workload, collected over about five and a half minutes. One sample
is `sandbox.run_level` on each of the four `calibrate.WORKLOADS` in turn, at
`repeats=6` aggregated by Hodges-Lehmann, which is exactly the path and the
settings a candidate solution's timings go through.

Probes are then formed from **disjoint consecutive blocks** of `k` samples. Two
different blocks are two independent probes of the same machine, so the true
differential between them is 1 and every firing of the rule is a false alarm. This
buys far more statistical power per second of timing than repeatedly collecting
fresh probes: 480 samples give 60 independent 8-replicate probes and 1770 pairs of
them.

The rule under test is the applied one, not a floor in isolation:
`differential(a, b) > max(DRIFT_CAVEAT, resolution(a), resolution(b))`, with the
resolution computed by the same estimator being tested, so an estimator with a
lower floor and a proportionally lower resolution scores no better.

Power is measured by multiplying one workload (`loop_arith`) of the later probe by
an injected factor. That is the largest single-workload change the statistic can
see, so the power columns are an upper bound on power against real drift, which
would move several workloads by different amounts.

## 1. A single timing is a 2-3× random variable

| workload | min | p50 | p90 | max | max/min |
| --- | --- | --- | --- | --- | --- |
| `loop_arith` | 10.67 | 13.51 | 16.82 | 32.37 | 3.033 |
| `attr_call` | 10.46 | 13.14 | 15.89 | 28.24 | 2.700 |
| `alloc_churn` | 12.54 | 16.10 | 19.86 | 29.91 | 2.385 |
| `bulk_builtin` | 11.53 | 13.75 | 17.09 | 24.86 | 2.156 |

Milliseconds, each value already a Hodges-Lehmann over 6 repeats. The p90/min
ratios are 1.48 to 1.58, so the tails are not a handful of outliers.

All four medians land between 13.1 and 16.1 ms, which is the 10-15 ms band the
fixed scales in `WORKLOADS` were chosen for, so the scales have not rotted.

## 2. The whole rule, per replicate count and estimator

False alarms are out of all pairs of probes; `adjacent` counts only consecutive
blocks, the closest thing here to two consecutive sessions of a run.

| k | estimator | res p50 | res p90 | res max | diff p50 | false | adjacent | ×1.10 | ×1.20 | ×1.40 | ×2.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | min | 1.3161 | 1.7095 | 2.6154 | 1.1683 | 354/7140 | 6/119 | 805 | 1744 | 4233 | 6518 |
| 4 | median | 1.3086 | 1.6003 | 2.9140 | 1.1617 | 266/7140 | 7/119 | 803 | 2000 | 4906 | 6836 |
| 4 | Hodges-Lehmann | 1.3086 | 1.6003 | 2.9140 | 1.1511 | 188/7140 | 5/119 | 662 | 1937 | 4924 | 6833 |
| 6 | min | 1.3032 | 1.6129 | 2.2026 | 1.1349 | 74/3160 | 1/79 | 236 | 690 | 1940 | 3034 |
| 6 | median | 1.3786 | 1.7486 | 3.9235 | 1.1405 | 22/3160 | 0/79 | 122 | 415 | 1400 | 2999 |
| 6 | Hodges-Lehmann | 1.3105 | 1.5967 | 3.1493 | 1.1343 | 20/3160 | 0/79 | 149 | 684 | 2035 | 3062 |
| **8** | min | 1.3017 | 1.5190 | 1.9206 | 1.1225 | 20/1770 | 0/59 | 74 | 263 | 1179 | 1762 |
| **8** | median | 1.3528 | 1.6140 | 2.9653 | 1.1230 | 0/1770 | 0/59 | 16 | 246 | 977 | 1683 |
| **8** | **Hodges-Lehmann** | **1.3198** | 1.6559 | 3.1292 | **1.1066** | **0/1770** | 0/59 | 2 | 286 | 1046 | 1669 |
| 12 | min | 1.3037 | 1.4352 | 1.7219 | 1.0757 | 6/780 | 0/39 | 11 | 75 | 558 | 780 |
| 12 | median | 1.3302 | 1.6803 | 2.9331 | 1.0890 | 0/780 | 0/39 | 3 | 78 | 375 | 731 |
| 12 | Hodges-Lehmann | 1.3411 | 1.5513 | 2.3402 | 1.0820 | 0/780 | 0/39 | 0 | 55 | 445 | 737 |
| 16 | min | 1.2830 | 1.4353 | 1.6324 | 1.0556 | 1/435 | 0/29 | 4 | 29 | 298 | 435 |
| 16 | median | 1.3578 | 1.6931 | 1.9949 | 1.0875 | 0/435 | 0/29 | 0 | 34 | 197 | 427 |
| 16 | Hodges-Lehmann | 1.3234 | 1.5606 | 2.1218 | 1.0856 | 0/435 | 0/29 | 0 | 25 | 268 | 429 |

Power columns share the denominator of the `false` column in their row. The bold
row is what the code now ships.

Four things in that table decided the design.

**The minimum invents drift at every replicate count tested** (354, 74, 20, 6, 1),
where the two central estimators reach zero by 8. The minimum's usual argument is
about the bias of one location, and this rule only ever uses locations inside a
ratio, where an extreme order statistic's variance appears twice.

**Hodges-Lehmann does not pay for that safety in sensitivity, and the minimum's
extra sensitivity is not free.** At 8 replicates the two central estimators both
reach zero false alarms, and between them the power is close, with Hodges-Lehmann
ahead where the comparison is live (286 against 246 at ×1.20, 1046 against 977 at
×1.40) and the two within a percent of each other at ×2.00, where both have nearly
saturated. The minimum looks more powerful than either (1179 at ×1.40), but that is
not a comparable number: its threshold is lower, and the same lower threshold is
what fires on 20 pairs with no drift in them at all, so the extra sensitivity
cannot be told apart from the extra false alarms. Hodges-Lehmann is also the
estimator the paper already uses inside each replicate, so the probe becomes one
estimator applied at two levels.

**The resolution does not improve with replication.** Its median across `k` = 4, 6,
8, 12, 16 is 1.309, 1.310, 1.320, 1.341, 1.323 for Hodges-Lehmann: flat, if
anything drifting up. The resolution is a maximum over half-splits and the number
of splits grows combinatorially with `k`, so each split getting better and there
being more chances to draw a bad one roughly cancel.

**What improves is the measured differential**, whose median falls 1.151, 1.134,
1.107, 1.082, 1.086 across the same counts. So the false-alarm rate drops because
the measured quantity tightens against a threshold that stays put near 1.32, not
because the instrument gets sharper. That also caps what replication buys: at 16
replicates both curves are flat and the cost has doubled for no additional reach.

## 3. Worst split, not a quantile of them

The resolution is the worst of a probe's half-splits. Taking a central one instead,
with Hodges-Lehmann throughout:

| k | split | res p50 | false alarms |
| --- | --- | --- | --- |
| 4 | median | 1.2013 | 1071/7140 (15.0%) |
| 4 | worst | 1.3086 | 188/7140 (2.6%) |
| 6 | median | 1.1825 | 318/3160 (10.1%) |
| 6 | p90 | 1.2614 | 58/3160 (1.8%) |
| 6 | worst | 1.3105 | 20/3160 (0.6%) |
| **8** | median | 1.1601 | 89/1770 (5.0%) |
| **8** | p90 | 1.2366 | 1/1770 (0.1%) |
| **8** | worst | 1.3198 | 0/1770 (0.0%) |
| 12 | median | 1.1391 | 22/780 (2.8%) |
| 16 | median | 1.1122 | 34/435 (7.8%) |

A median split understates the floor by about 0.16 and reports drift that is not
there 5% of the time at the shipped replicate count. The choice of the maximum is
not forced — the 90th-percentile split is nearly as clean — but a quantile over the
35 splits of an 8-replicate probe is a crude thing to hang a refusal on, and the
maximum needs no parameter. (At `k = 4` there are only 3 splits, so the p90 and the
median are the same value.)

## 4. The floor is a property of the moment, not of the machine

Hodges-Lehmann at `k = 8`, over each quarter of the same collection window:

| quarter | res p50 | res max | probes |
| --- | --- | --- | --- |
| 1 | 1.2515 | 1.4531 | 15 |
| 2 | 1.2693 | 1.5157 | 15 |
| 3 | 1.3610 | 1.8405 | 15 |
| 4 | 1.4988 | 3.1292 | 15 |

The machine became measurably noisier over five and a half minutes, on an idle VM
with nothing else scheduled. The median resolution moved by 0.25 between the first
quarter and the last, which is ten times the 1.025 differential the parity
tolerance is trying to police.

This is the load-bearing result for the design. It is why `compare()` judges a pair
by `max(res_a, res_b)` rather than by either probe or an average of them, and why
the resolution is printed in the run summary and stored in the record: a quiet
verdict has two causes that only this number distinguishes, and the number is not
a constant that could have been written down once.

It is also the explanation for an earlier false start. Two 48-sample draws
collected minutes apart disagreed about which estimator was safest, one favouring
the minimum 0/66 to 13/66 and the other going the other way 14/66 to 1/66, and the
first draw's answer had already been written into a docstring as a fact. The
disagreement was the finding.

## What this does not measure

- One machine, one five-minute window. Nothing here speaks to drift across hours,
  a reboot, or a kernel update, which is the separation a real resumed run has.
  Adjacent-block pairs are the closest available proxy and were clean from 6
  replicates up.
- 0 false alarms in 1770 pairs is not a rate of 0, and those pairs share blocks
  with one another, so they are not 1770 independent trials.
- Power against a single moved workload is an upper bound. A differential that
  moves two workloads the same way is partly invisible to a `max/min` spread.
- Whether four workloads span the cost mixes of the benchmark's actual reference
  solutions is not tested here. That needs the real problems (task #13) and would
  be measured by correlating each reference's timing against the probe.
