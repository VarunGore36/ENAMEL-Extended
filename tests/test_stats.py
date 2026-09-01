"""Tests for the uncertainty and rank statistics."""

from __future__ import annotations

import random
import unittest

from enamel_ext.report.stats import (
    bootstrap_ci,
    kendall_tau,
    paired_bootstrap_diff_ci,
    paired_sign_test,
)


class TestBootstrapCI(unittest.TestCase):
    def test_deterministic_for_a_given_seed(self):
        vals = [0.1, 0.9, 0.3, 0.0, 0.7, 0.45]
        a = bootstrap_ci(vals, resamples=500, seed=7)
        b = bootstrap_ci(vals, resamples=500, seed=7)
        self.assertEqual(a, b)

    def test_interval_brackets_the_point_estimate(self):
        vals = [0.1, 0.9, 0.3, 0.0, 0.7, 0.45]
        ci = bootstrap_ci(vals, resamples=2000, seed=1)
        self.assertLessEqual(ci.lo, ci.point)
        self.assertLessEqual(ci.point, ci.hi)
        self.assertAlmostEqual(ci.point, sum(vals) / len(vals), places=12)

    def test_no_variance_gives_a_degenerate_interval(self):
        ci = bootstrap_ci([0.4] * 20, resamples=200, seed=3)
        self.assertAlmostEqual(ci.lo, 0.4, places=12)
        self.assertAlmostEqual(ci.hi, 0.4, places=12)

    def test_width_shrinks_with_more_problems(self):
        rng = random.Random(0)
        pool = [rng.random() for _ in range(400)]
        narrow = bootstrap_ci(pool[:400], resamples=2000, seed=5)
        wide = bootstrap_ci(pool[:25], resamples=2000, seed=5)
        self.assertLess(narrow.hi - narrow.lo, wide.hi - wide.lo)

    def test_higher_confidence_is_wider(self):
        vals = [0.1, 0.9, 0.3, 0.0, 0.7, 0.45]
        c90 = bootstrap_ci(vals, resamples=4000, seed=2, level=0.90)
        c99 = bootstrap_ci(vals, resamples=4000, seed=2, level=0.99)
        self.assertLessEqual(c90.hi - c90.lo, c99.hi - c99.lo)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([])
        with self.assertRaises(ValueError):
            bootstrap_ci([0.1], level=1.0)
        with self.assertRaises(ValueError):
            bootstrap_ci([0.1], resamples=0)


class TestPairedComparison(unittest.TestCase):
    def test_identical_models_give_a_zero_interval(self):
        vals = [0.1, 0.9, 0.3, 0.0, 0.7]
        ci = paired_bootstrap_diff_ci(vals, vals, resamples=500, seed=4)
        self.assertEqual((ci.point, ci.lo, ci.hi), (0.0, 0.0, 0.0))
        self.assertFalse(ci.excludes_zero)

    def test_constant_offset_is_recovered_exactly(self):
        """Identical differences give identical resample means, so the interval
        collapses onto the offset."""
        b = [0.1, 0.9, 0.3, 0.0, 0.7]
        a = [x + 0.05 for x in b]
        ci = paired_bootstrap_diff_ci(a, b, resamples=500, seed=4)
        for value in (ci.point, ci.lo, ci.hi):
            self.assertAlmostEqual(value, 0.05, places=12)
        self.assertTrue(ci.excludes_zero)

    def test_pairing_separates_models_that_unpaired_intervals_cannot(self):
        """A small but consistent edge is invisible to individual intervals."""
        rng = random.Random(11)
        difficulty = [rng.random() for _ in range(142)]
        b = difficulty
        a = [min(1.0, d + 0.02) for d in difficulty]

        ci_a = bootstrap_ci(a, resamples=3000, seed=1)
        ci_b = bootstrap_ci(b, resamples=3000, seed=1)
        self.assertLess(ci_a.lo, ci_b.hi)  # individual intervals overlap

        paired = paired_bootstrap_diff_ci(a, b, resamples=3000, seed=1)
        self.assertTrue(paired.excludes_zero)
        self.assertLess(paired.hi - paired.lo, (ci_a.hi - ci_a.lo) / 2)

    def test_rejects_unequal_problem_sets(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_diff_ci([0.1, 0.2], [0.1])


class TestPairedSignTest(unittest.TestCase):
    def test_no_difference_gives_p_of_one(self):
        vals = [0.1, 0.9, 0.3, 0.0, 0.7]
        self.assertAlmostEqual(paired_sign_test(vals, vals), 1.0, places=12)

    def test_exact_enumeration_for_small_n(self):
        """Only the all-plus and all-minus patterns reach the observed
        statistic, so the exact two-sided p is 2/2**10."""
        a = [1.0] * 10
        b = [0.0] * 10
        self.assertAlmostEqual(paired_sign_test(a, b), 2 / 1024, places=12)

    def test_monte_carlo_path_for_large_n(self):
        a = [1.0] * 50
        b = [0.0] * 50
        p = paired_sign_test(a, b, resamples=2000, seed=0)
        self.assertGreater(p, 0.0)  # never exactly zero: observed is counted
        self.assertLess(p, 0.01)

    def test_noise_is_not_significant(self):
        rng = random.Random(9)
        a = [rng.random() for _ in range(60)]
        b = [rng.random() for _ in range(60)]
        self.assertGreater(paired_sign_test(a, b, resamples=2000, seed=0), 0.05)

    def test_rejects_unequal_problem_sets(self):
        with self.assertRaises(ValueError):
            paired_sign_test([0.1, 0.2], [0.1])


class TestKendallTau(unittest.TestCase):
    def test_identical_orderings(self):
        self.assertAlmostEqual(kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=12)

    def test_reversed_orderings(self):
        self.assertAlmostEqual(kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, places=12)

    def test_tie_correction(self):
        """x has one tied pair: C=5, D=0, tied_x=1, so tau_b = 5/sqrt(5*6)."""
        got = kendall_tau([1, 2, 2, 3], [1, 2, 3, 4])
        self.assertAlmostEqual(got, 5 / (30**0.5), places=12)

    def test_symmetric_in_its_arguments(self):
        x = [0.47, 0.45, 0.42, 0.41]
        y = [0.52, 0.50, 0.51, 0.44]
        self.assertAlmostEqual(kendall_tau(x, y), kendall_tau(y, x), places=12)

    def test_all_tied_is_undefined_not_zero(self):
        with self.assertRaises(ValueError):
            kendall_tau([1, 1, 1], [1, 2, 3])

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            kendall_tau([1, 2], [1])
        with self.assertRaises(ValueError):
            kendall_tau([1], [1])


if __name__ == "__main__":
    unittest.main()
