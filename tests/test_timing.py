"""Tests for repeat aggregation (Hodges-Lehmann and alternatives)."""

from __future__ import annotations

import math
import unittest

from enamel_ext.measure.timing import AGGREGATORS, aggregate_repeats, hodges_lehmann
from enamel_ext.metrics.score import TIMEOUT


class TestHodgesLehmann(unittest.TestCase):
    def test_hand_computed(self):
        """[1, 2, 3]: Walsh averages are 1, 1.5, 2, 2, 2.5, 3, so the median is 2."""
        self.assertAlmostEqual(hodges_lehmann([1.0, 2.0, 3.0]), 2.0, places=15)

    def test_single_repeat(self):
        self.assertAlmostEqual(hodges_lehmann([0.42]), 0.42, places=15)

    def test_constant_sample(self):
        self.assertAlmostEqual(hodges_lehmann([1.5] * 6), 1.5, places=15)

    def test_ignores_one_wild_outlier(self):
        """The property the paper is buying. The mean here is 17.5."""
        sample = [1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
        self.assertAlmostEqual(hodges_lehmann(sample), 1.0, places=15)
        self.assertAlmostEqual(sum(sample) / len(sample), 17.5, places=12)

    def test_breakdown_at_two_of_six(self):
        """Breakdown point ~29%: two contaminated repeats out of six move it."""
        clean = [1.0] * 6
        one_bad = [1.0] * 5 + [100.0]
        two_bad = [1.0] * 4 + [100.0, 100.0]
        self.assertEqual(hodges_lehmann(one_bad), hodges_lehmann(clean))
        self.assertGreater(hodges_lehmann(two_bad), hodges_lehmann(clean))

    def test_translation_equivariant(self):
        sample = [0.3, 0.31, 0.29, 0.4, 0.305, 0.32]
        shifted = [x + 10.0 for x in sample]
        self.assertAlmostEqual(hodges_lehmann(shifted), hodges_lehmann(sample) + 10.0, places=12)

    def test_scale_equivariant(self):
        sample = [0.3, 0.31, 0.29, 0.4, 0.305, 0.32]
        scaled = [x * 1000.0 for x in sample]
        self.assertAlmostEqual(hodges_lehmann(scaled), hodges_lehmann(sample) * 1000.0, places=9)

    def test_lies_between_min_and_max(self):
        sample = [0.9, 0.2, 0.5, 0.7, 0.21, 3.0]
        self.assertGreaterEqual(hodges_lehmann(sample), min(sample))
        self.assertLessEqual(hodges_lehmann(sample), max(sample))

    def test_rejects_empty_and_nan(self):
        with self.assertRaises(ValueError):
            hodges_lehmann([])
        with self.assertRaises(ValueError):
            hodges_lehmann([1.0, float("nan")])


class TestAggregateRepeats(unittest.TestCase):
    def test_default_is_the_papers_choice(self):
        sample = [1.0, 2.0, 3.0]
        self.assertEqual(aggregate_repeats(sample), hodges_lehmann(sample))

    def test_censoring_propagates(self):
        self.assertTrue(math.isinf(aggregate_repeats([1.0, 2.0, TIMEOUT])))
        self.assertTrue(math.isinf(aggregate_repeats([TIMEOUT] * 6)))

    def test_hl_is_biased_above_min_under_one_sided_noise(self):
        """Timing noise only adds time, so HL and min disagree systematically."""
        true_cost = 1.0
        sample = [true_cost + noise for noise in (0.0, 0.02, 0.05, 0.01, 0.13, 0.04)]
        self.assertAlmostEqual(aggregate_repeats(sample, "min"), true_cost, places=12)
        self.assertGreater(aggregate_repeats(sample, "hodges_lehmann"), true_cost)

    def test_all_aggregators_agree_on_a_constant_sample(self):
        for name in AGGREGATORS:
            with self.subTest(method=name):
                self.assertAlmostEqual(aggregate_repeats([2.5] * 6, name), 2.5, places=15)

    def test_unknown_method_is_an_error(self):
        with self.assertRaises(ValueError):
            aggregate_repeats([1.0], "geometric_mean")


if __name__ == "__main__":
    unittest.main()
