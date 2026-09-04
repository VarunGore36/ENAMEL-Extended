"""Tests for the runner's stopping rule.

The rule has to censor exactly the cases the score gives 0, no others. Rationale
in docs/decisions/0004-sandboxed-runner.md.
"""

from __future__ import annotations

import unittest
from itertools import product

from enamel_ext.measure import _child
from enamel_ext.measure.sandbox import WALL_SLACK
from enamel_ext.measure.timing import (
    aggregate_lower_bound,
    aggregate_repeats,
    hodges_lehmann,
    reaches_limit,
)
from enamel_ext.measure.values import decode
from enamel_ext.metrics.score import PAPER, TIMEOUT, level_fraction

#: R and T_i. The limit is 1.0 so every time below is a multiple of it.
REPEATS = 6
LIMIT = 1.0

#: One 8x outlier in six repeats. The mean is over the limit, the Hodges-Lehmann
#: estimate is half of it, and the score compares the latter.
OUTLIER = [0.5, 0.5, 0.5, 0.5, 0.5, 4.0]

#: Largest total a case can consume while its aggregate stays under the limit,
#: found by search over ``n`` equal repeats and ``6 - n`` at zero.
HL_WORST_TOTAL = 8.0


def completions(prefix, repeats, grid):
    """Every way of filling out ``prefix`` to ``repeats`` from ``grid``."""
    for tail in product(grid, repeat=repeats - len(prefix)):
        yield list(prefix) + list(tail)


class TestAggregateLowerBound(unittest.TestCase):
    def test_a_full_sample_bounds_itself(self):
        sample = [0.3, 0.9, 0.4, 0.35, 0.32, 0.31]
        self.assertEqual(
            aggregate_lower_bound(sample, REPEATS), aggregate_repeats(sample)
        )

    def test_unrun_repeats_count_as_zero(self):
        self.assertEqual(
            aggregate_lower_bound([1.0], REPEATS),
            hodges_lehmann([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def test_the_bound_never_falls_as_repeats_arrive(self):
        """What makes "stop at the first trip" well defined: the bound is
        non-decreasing, so a case that has not tripped cannot have tripped
        earlier."""
        for sample in ([0.4] * 6, OUTLIER, [4.0, 0.1, 0.1, 0.1, 0.1, 0.1]):
            bounds = [
                aggregate_lower_bound(sample[:r], REPEATS) for r in range(REPEATS + 1)
            ]
            for earlier, later in zip(bounds, bounds[1:]):
                self.assertLessEqual(earlier, later, msg=f"{sample}")

    def test_a_censored_repeat_makes_the_bound_infinite(self):
        self.assertEqual(aggregate_lower_bound([0.1, TIMEOUT], REPEATS), TIMEOUT)

    def test_more_repeats_than_declared_is_an_error(self):
        with self.assertRaises(ValueError):
            aggregate_lower_bound([0.1] * 7, REPEATS)

    def test_min_learns_nothing_until_the_last_repeat(self):
        """Sound but useless: ``min`` can still fall on the repeat not yet run, so
        no early stop is available and only the wall clock bounds the work."""
        for r in range(REPEATS):
            self.assertEqual(aggregate_lower_bound([9.0] * r, REPEATS, "min"), 0.0)
        self.assertEqual(aggregate_lower_bound([9.0] * REPEATS, REPEATS, "min"), 9.0)


class TestStoppingAgreesWithTheScore(unittest.TestCase):
    def test_the_mean_rule_censored_a_case_that_scores_full_marks(self):
        """The defect this rule replaces. Stopping when the accumulated time
        passes ``limit * repeats`` is stopping when the mean passes the limit, and
        timing noise is right-skewed, so the mean sits above the estimate the score
        reads."""
        self.assertGreater(sum(OUTLIER) / REPEATS, LIMIT)
        self.assertLess(aggregate_repeats(OUTLIER), LIMIT)
        self.assertGreater(sum(OUTLIER), LIMIT * REPEATS)

        self.assertFalse(reaches_limit(OUTLIER, REPEATS, LIMIT))
        for r in range(1, REPEATS + 1):
            self.assertFalse(reaches_limit(OUTLIER[:r], REPEATS, LIMIT), msg=f"r={r}")

        # Against a reference whose worst case is exactly T_i / alpha, this
        # candidate matches the expert and earns the whole level.
        reference = [LIMIT / PAPER.alpha]
        self.assertEqual(level_fraction([aggregate_repeats(OUTLIER)], reference, LIMIT), 1.0)

    def test_a_stop_is_never_premature(self):
        """Soundness: once the rule trips, no completion of the remaining repeats
        brings the aggregate back under the limit."""
        grid = [0.0, 0.25, 0.5, 1.0, 2.0]
        prefixes = [
            [2.0],
            [2.0, 2.0],
            [1.5, 1.5, 1.5],
            [0.9, 1.2, 1.1, 1.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        tripped = 0
        for prefix in prefixes:
            if not reaches_limit(prefix, REPEATS, LIMIT):
                continue
            tripped += 1
            for full in completions(prefix, REPEATS, grid):
                self.assertGreaterEqual(
                    aggregate_repeats(full), LIMIT, msg=f"{prefix} -> {full}"
                )
        self.assertGreater(tripped, 0, "no prefix tripped, so nothing was tested")

    def test_a_case_under_the_limit_is_never_stopped(self):
        """Completeness in the direction that matters: if the finished aggregate
        would be under the limit, no prefix of it trips."""
        samples = [
            [0.4] * 6,
            OUTLIER,
            [0.99] * 6,
            [0.1, 0.1, 0.1, 0.1, 3.0, 3.0],
            [0.0] * 5 + [1.9],
        ]
        for sample in samples:
            if aggregate_repeats(sample) >= LIMIT:
                continue
            for r in range(1, REPEATS + 1):
                self.assertFalse(
                    reaches_limit(sample[:r], REPEATS, LIMIT), msg=f"{sample} at r={r}"
                )

    def test_a_uniformly_slow_case_stops_before_the_last_repeat(self):
        """The rule still has to save work. Two repeats over ``2 * limit`` already
        pin the Hodges-Lehmann estimate at or above the limit."""
        self.assertTrue(reaches_limit([2.0, 2.0, 2.0], REPEATS, LIMIT))
        self.assertTrue(reaches_limit([1.0] * REPEATS, REPEATS, LIMIT))

    def test_uniform_repeats_censor_the_same_cases_under_every_rule(self):
        """The claim decision 0004 makes about the typical case: with equal repeats
        the aggregate, the mean and the per-repeat rules all censor exactly the
        cases whose repeat time reaches the limit, so only skew separates them."""
        for numerator in range(1, 40):
            value = numerator / 20.0
            sample = [value] * REPEATS
            censored = value >= LIMIT
            self.assertEqual(aggregate_repeats(sample) >= LIMIT, censored, msg=f"{value}")
            self.assertEqual(sum(sample) >= LIMIT * REPEATS, censored, msg=f"{value}")
            self.assertEqual(reaches_limit(sample, REPEATS, LIMIT), censored, msg=f"{value}")

    def test_no_limit_means_no_stopping(self):
        """Level 0 and the reference run without a limit; they must run every
        repeat."""
        self.assertFalse(reaches_limit([1e6] * REPEATS, REPEATS, None))


class TestTheWallClockCoversTheBound(unittest.TestCase):
    def test_the_worst_admissible_case_fits_the_wall_slack(self):
        """The wall-clock kill is the only work bound left, so it has to be looser
        than the most a legitimately-fast case can consume."""
        worst = 0.0
        grid = [i / 200.0 for i in range(0, 801)]
        for n_slow in range(1, REPEATS + 1):
            for value in grid:
                sample = [value] * n_slow + [0.0] * (REPEATS - n_slow)
                if aggregate_repeats(sample) < LIMIT:
                    worst = max(worst, sum(sample))
        self.assertLess(worst, HL_WORST_TOTAL * LIMIT)
        self.assertLessEqual(HL_WORST_TOTAL, WALL_SLACK * REPEATS)


class TestTheChildAppliesTheRule(unittest.TestCase):
    """``_child._time_case`` against a scripted clock, so the assertions are about
    the stopping rule and not about how fast this machine happens to be."""

    def time_case(self, script, repeats=REPEATS, limit=LIMIT, aggregator="hodges_lehmann"):
        """Run the child's timing loop with each repeat taking ``script[r]``."""
        ticks = []
        for elapsed in script:
            ticks.extend([0.0, elapsed])
        clock = iter(ticks)
        original = _child._PERF
        _child._PERF = lambda: next(clock)
        try:
            return _child._time_case(lambda x: x, (1,), repeats, limit, aggregator, False)
        finally:
            _child._PERF = original

    def test_an_outlier_repeat_does_not_censor_the_case(self):
        times, _, _, status = self.time_case(OUTLIER)
        self.assertEqual(status, _child.OK)
        self.assertEqual(len(times), REPEATS)
        self.assertLess(aggregate_repeats(times), LIMIT)

    def test_a_uniformly_slow_case_is_censored_early(self):
        times, _, _, status = self.time_case([2.0] * REPEATS)
        self.assertEqual(status, _child.TIMEOUT)
        self.assertLess(len(times), REPEATS)

    def test_the_first_output_survives_an_early_stop(self):
        """The wrong-answer check outranks the timeout, so a censored case still
        has to hand back what it returned."""
        _, output, has_output, status = self.time_case([5.0] * REPEATS)
        self.assertEqual(status, _child.TIMEOUT)
        self.assertTrue(has_output)
        self.assertEqual(decode(output), 1)

    def test_an_unlimited_case_runs_every_repeat(self):
        times, _, _, status = self.time_case([9.0] * REPEATS, limit=None)
        self.assertEqual(status, _child.OK)
        self.assertEqual(len(times), REPEATS)


if __name__ == "__main__":
    unittest.main()
