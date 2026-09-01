"""Tests for the eff@k / pass@k estimators."""

from __future__ import annotations

import itertools
import math
import unittest
from fractions import Fraction

from enamel_ext.metrics.effk import (
    eff_at_k,
    effk_weights,
    effk_weights_exact,
    mean_over_problems,
    pass_at_k,
)

NK_GRID = [(1, 1), (2, 1), (2, 2), (5, 1), (5, 3), (5, 5), (10, 4), (37, 11), (100, 1), (100, 50)]


class TestWeights(unittest.TestCase):
    def test_recurrence_matches_closed_form(self):
        for n, k in NK_GRID:
            with self.subTest(n=n, k=k):
                got = effk_weights(n, k)
                want = [float(f) for f in effk_weights_exact(n, k)]
                self.assertEqual(len(got), n - k + 1)
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, delta=1e-15 * max(1.0, abs(w)))

    def test_weights_sum_to_one(self):
        for n, k in NK_GRID:
            with self.subTest(n=n, k=k):
                self.assertAlmostEqual(sum(effk_weights(n, k)), 1.0, places=12)

    def test_exact_weights_sum_to_exactly_one(self):
        for n, k in NK_GRID:
            with self.subTest(n=n, k=k):
                self.assertEqual(sum(effk_weights_exact(n, k)), Fraction(1))

    def test_weights_are_probabilities(self):
        for n, k in NK_GRID:
            for w in effk_weights(n, k):
                self.assertGreaterEqual(w, 0.0)
                self.assertLessEqual(w, 1.0)

    def test_survives_n_where_closed_form_overflows_in_float(self):
        """The reason Algorithm 1 exists: C(2000, 1000) raises on float contact."""
        n, k = 2000, 1000
        with self.assertRaises(OverflowError):
            float(math.comb(n, k))
        weights = effk_weights(n, k)
        self.assertEqual(len(weights), n - k + 1)
        self.assertTrue(all(math.isfinite(w) for w in weights))
        self.assertAlmostEqual(sum(weights), 1.0, places=10)

    def test_monotone_increasing_in_r(self):
        for n, k in [(50, 5), (50, 25), (12, 3)]:
            w = effk_weights(n, k)
            for a, b in zip(w, w[1:]):
                self.assertLessEqual(a, b + 1e-18)

    def test_k_equals_one_is_uniform(self):
        n = 17
        for w in effk_weights(n, 1):
            self.assertAlmostEqual(w, 1.0 / n, places=15)

    def test_k_equals_n_is_a_point_mass_on_the_max(self):
        w = effk_weights(9, 9)
        self.assertEqual(len(w), 1)
        self.assertAlmostEqual(w[0], 1.0, places=15)

    def test_rejects_bad_arguments(self):
        for n, k in [(0, 0), (5, 0), (5, 6), (-1, 1)]:
            with self.subTest(n=n, k=k), self.assertRaises(ValueError):
                effk_weights(n, k)
        with self.assertRaises(TypeError):
            effk_weights(5.0, 2)  # type: ignore[arg-type]


class TestEffAtK(unittest.TestCase):
    def test_equals_mean_max_over_all_k_subsets(self):
        """The defining property, and the test that pins weight-to-rank alignment."""
        scores = [0.0, 0.0, 0.17, 0.4, 0.55, 0.98, 1.4]
        n = len(scores)
        for k in range(1, n + 1):
            with self.subTest(k=k):
                subsets = list(itertools.combinations(scores, k))
                brute = sum(max(s) for s in subsets) / len(subsets)
                self.assertAlmostEqual(eff_at_k(scores, k), brute, places=14)

    def test_k_equals_n_reduces_to_max(self):
        scores = [0.0, 0.3, 0.91, 0.4]
        self.assertAlmostEqual(eff_at_k(scores, 4), max(scores), places=15)

    def test_k_equals_one_reduces_to_mean(self):
        scores = [0.0, 0.3, 0.91, 0.4]
        self.assertAlmostEqual(eff_at_k(scores, 1), sum(scores) / 4, places=15)

    def test_order_invariant(self):
        scores = [0.11, 0.0, 0.87, 0.5, 0.5]
        base = eff_at_k(scores, 3)
        for perm in itertools.permutations(scores):
            self.assertAlmostEqual(eff_at_k(list(perm), 3), base, places=15)

    def test_monotone_in_k(self):
        scores = [0.0, 0.1, 0.2, 0.55, 0.6, 0.95]
        vals = [eff_at_k(scores, k) for k in range(1, len(scores) + 1)]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a, b + 1e-15)

    def test_all_zero_stays_zero(self):
        self.assertEqual(eff_at_k([0.0] * 6, 3), 0.0)

    def test_scores_above_one_are_not_clamped(self):
        self.assertGreater(eff_at_k([0.0, 0.0, 1.8], 3), 1.0)

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            eff_at_k([0.1, float("nan"), 0.3], 2)


class TestTheorem1(unittest.TestCase):
    """Exact verification of unbiasedness and the variance bound, by enumerating
    all 3**5 outcomes of a 3-point score distribution."""

    VALUES = (0.0, 0.5, 1.2)
    PROBS = (0.5, 0.3, 0.2)
    N = 5

    def _true_expected_max(self, k: int) -> float:
        """E[max of k draws] = sum_v v * (P(X<=v)^k - P(X<v)^k)."""
        total = 0.0
        cdf = 0.0
        for v, p in zip(self.VALUES, self.PROBS):
            below = cdf
            cdf += p
            total += v * (cdf**k - below**k)
        return total

    def _outcomes(self):
        for combo in itertools.product(range(len(self.VALUES)), repeat=self.N):
            prob = 1.0
            for idx in combo:
                prob *= self.PROBS[idx]
            yield prob, [self.VALUES[idx] for idx in combo]

    def test_unbiased(self):
        outcomes = list(self._outcomes())
        for k in range(1, self.N + 1):
            with self.subTest(k=k):
                est = sum(p * eff_at_k(s, k) for p, s in outcomes)
                self.assertAlmostEqual(est, self._true_expected_max(k), places=12)

    def test_variance_bound(self):
        """Theorem 1: Var[eff-hat_i@k] <= (k/n) * Var[max of k draws]."""
        outcomes = list(self._outcomes())
        for k in range(1, self.N + 1):
            with self.subTest(k=k):
                mean = sum(p * eff_at_k(s, k) for p, s in outcomes)
                var = sum(p * (eff_at_k(s, k) - mean) ** 2 for p, s in outcomes)

                naive_mean = sum(p * max(s[:k]) for p, s in outcomes)
                naive_var = sum(p * (max(s[:k]) - naive_mean) ** 2 for p, s in outcomes)

                self.assertLessEqual(var, (k / self.N) * naive_var + 1e-12)
                self.assertLessEqual(var, naive_var + 1e-12)


class TestPassAtK(unittest.TestCase):
    def test_matches_binomial_definition(self):
        for n in range(1, 15):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    with self.subTest(n=n, c=c, k=k):
                        want = 1.0 - (
                            math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0.0
                        )
                        self.assertAlmostEqual(pass_at_k(n, c, k), want, places=12)

    def test_edges(self):
        self.assertEqual(pass_at_k(10, 0, 1), 0.0)
        self.assertEqual(pass_at_k(10, 10, 10), 1.0)
        self.assertAlmostEqual(pass_at_k(10, 1, 1), 0.1, places=12)

    def test_rejects_bad_c(self):
        for c in (-1, 11):
            with self.subTest(c=c), self.assertRaises(ValueError):
                pass_at_k(10, c, 2)


class TestMeanOverProblems(unittest.TestCase):
    def test_unweighted_mean(self):
        self.assertAlmostEqual(mean_over_problems([0.0, 1.0, 0.5]), 0.5, places=15)

    def test_empty_is_an_error_not_zero(self):
        with self.assertRaises(ValueError):
            mean_over_problems([])


if __name__ == "__main__":
    unittest.main()
