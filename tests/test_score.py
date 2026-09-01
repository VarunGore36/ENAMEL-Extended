"""Tests for the per-sample efficiency score, Eq. (1) and (2)."""

from __future__ import annotations

import unittest

from enamel_ext.metrics.score import (
    PAPER,
    TIMEOUT,
    MetricConfig,
    level_fraction,
    sample_score,
    time_limit,
)

PER_LEVEL = MetricConfig(alpha=2.0, level_weights=(3.0, 3.0, 4.0), normalization="per_level")


class TestTimeLimit(unittest.TestCase):
    def test_alpha_times_global_worst_case(self):
        self.assertAlmostEqual(time_limit([[0.1, 0.2], [0.5], [1.0, 0.9]], PAPER), 2.0)

    def test_max_ranges_over_all_levels_not_just_the_hardest(self):
        self.assertAlmostEqual(time_limit([[0.1], [3.0], [2.0]], PAPER), 6.0)

    def test_rejects_level_count_mismatch(self):
        with self.assertRaises(ValueError):
            time_limit([[1.0], [1.0]], PAPER)

    def test_rejects_censored_reference(self):
        with self.assertRaises(ValueError):
            time_limit([[1.0], [1.0], [TIMEOUT]], PAPER)

    def test_rejects_nan_and_negative(self):
        with self.assertRaises(ValueError):
            time_limit([[1.0], [1.0], [float("nan")]], PAPER)
        with self.assertRaises(ValueError):
            time_limit([[1.0], [1.0], [-0.5]], PAPER)


class TestLevelFraction(unittest.TestCase):
    def test_reference_scores_exactly_one(self):
        self.assertAlmostEqual(level_fraction([1.0], [1.0], 2.0), 1.0, places=15)

    def test_timeout_scores_zero(self):
        self.assertEqual(level_fraction([TIMEOUT], [1.0], 2.0), 0.0)

    def test_at_the_limit_scores_zero(self):
        self.assertEqual(level_fraction([2.0], [1.0], 2.0), 0.0)

    def test_beyond_the_limit_is_clamped_not_negative(self):
        self.assertEqual(level_fraction([50.0], [1.0], 2.0), 0.0)

    def test_free_code_scores_alpha_over_alpha_minus_one(self):
        """Instant code scores 2 at alpha = 2, not 1: unbounded above by design."""
        self.assertAlmostEqual(level_fraction([0.0], [1.0], 2.0), 2.0, places=15)

    def test_worst_case_within_a_level_is_what_counts(self):
        self.assertAlmostEqual(level_fraction([0.1, 0.1, 1.5], [1.0], 2.0), 0.5, places=15)

    def test_rejects_limit_below_reference(self):
        with self.assertRaises(ValueError):
            level_fraction([1.0], [1.0], 1.0)


class TestSampleScore(unittest.TestCase):
    REF = [[1.0], [1.0], [1.0]]  # flat reference: every level equally hard

    def test_reference_solution_scores_one(self):
        self.assertAlmostEqual(sample_score(self.REF, self.REF, PAPER), 1.0, places=15)

    def test_incorrect_gates_to_zero_however_fast(self):
        instant = [[0.0], [0.0], [0.0]]
        self.assertGreater(sample_score(instant, self.REF, PAPER), 1.0)
        self.assertEqual(sample_score(instant, self.REF, PAPER, correct=False), 0.0)

    def test_all_levels_timing_out_scores_zero(self):
        killed = [[TIMEOUT], [TIMEOUT], [TIMEOUT]]
        self.assertEqual(sample_score(killed, self.REF, PAPER), 0.0)

    def test_level_weight_split_is_30_30_40(self):
        pass_12 = [[1.0], [1.0], [TIMEOUT]]
        pass_3 = [[TIMEOUT], [TIMEOUT], [1.0]]
        self.assertAlmostEqual(sample_score(pass_12, self.REF, PAPER), 0.6, places=15)
        self.assertAlmostEqual(sample_score(pass_3, self.REF, PAPER), 0.4, places=15)

    def test_rejects_level_count_mismatch(self):
        with self.assertRaises(ValueError):
            sample_score([[1.0], [1.0]], self.REF, PAPER)


class TestScoreCompression(unittest.TestCase):
    """Regression test for README section 2.2.

    With ``T_i = 2.0``, a level whose own reference time is ``q`` and a candidate
    ``X`` times slower than the expert scores ``(2 - X*q)_+ / (2 - q)``, which
    stays near 1 for large ``X`` whenever ``q`` is small.
    """

    # (q, X) -> expected level fraction under the paper's global normalization
    TABLE = {
        (0.01, 2): 0.995,
        (0.01, 5): 0.980,
        (0.01, 10): 0.955,
        (0.01, 50): 0.754,
        (0.05, 2): 0.974,
        (0.05, 5): 0.897,
        (0.05, 10): 0.769,
        (0.05, 50): 0.000,
        (0.10, 2): 0.947,
        (0.10, 5): 0.789,
        (0.10, 10): 0.526,
        (0.10, 50): 0.000,
        # level 3 itself (q = 1) is the only place the score is sharp
        (1.00, 1.25): 0.750,
        (1.00, 1.50): 0.500,
        (1.00, 2.00): 0.000,
    }

    def test_global_normalization_compresses_easy_levels(self):
        limit = 2.0  # alpha=2, slowest reference case = 1.0
        for (q, x), expected in self.TABLE.items():
            with self.subTest(q=q, x=x):
                got = level_fraction([x * q], [q], limit)
                self.assertAlmostEqual(got, expected, places=3)

    def test_ten_times_slower_still_scores_above_point_nine_five(self):
        self.assertGreater(level_fraction([10 * 0.01], [0.01], 2.0), 0.95)

    def test_per_level_normalization_removes_the_compression(self):
        for q in (0.01, 0.05, 0.10, 1.00):
            with self.subTest(q=q):
                self.assertEqual(level_fraction([10 * q], [q], 2.0 * q), 0.0)
                self.assertAlmostEqual(level_fraction([1.5 * q], [q], 2.0 * q), 0.5, places=12)

    def test_variant_changes_the_sample_score(self):
        ref = [[0.01], [0.05], [1.0]]
        cand = [[0.1], [0.5], [1.0]]  # 10x slower on levels 1-2, expert on level 3
        paper = sample_score(cand, ref, PAPER)
        variant = sample_score(cand, ref, PER_LEVEL)
        self.assertAlmostEqual(paper, 0.3 * 0.955 + 0.3 * 0.769 + 0.4 * 1.0, places=3)
        self.assertAlmostEqual(variant, 0.4, places=12)
        self.assertGreater(paper - variant, 0.5)


class TestScaleInvariance(unittest.TestCase):
    """The score is invariant to a uniform rescaling of every measured time,
    because ``T_i = alpha * max t*`` scales with it. Exact only for a uniform
    factor, so it removes drift between runs, not the need to pin the machine."""

    REF = [[0.01], [0.05], [1.0]]
    CAND = [[0.02], [0.2], [1.4]]

    def test_uniform_rescaling_leaves_the_score_unchanged(self):
        base = sample_score(self.CAND, self.REF, PAPER)
        for c in (1e-3, 0.5, 2.0, 1000.0):
            with self.subTest(factor=c):
                scaled_ref = [[t * c for t in lvl] for lvl in self.REF]
                scaled_cand = [[t * c for t in lvl] for lvl in self.CAND]
                self.assertAlmostEqual(
                    sample_score(scaled_cand, scaled_ref, PAPER), base, places=12
                )

    def test_rescaling_only_the_candidate_does_change_the_score(self):
        """The invariance is not vacuous: it holds for a common factor only."""
        faster_cand = [[t * 0.5 for t in lvl] for lvl in self.CAND]
        self.assertGreater(
            sample_score(faster_cand, self.REF, PAPER),
            sample_score(self.CAND, self.REF, PAPER),
        )

    def test_invariance_also_holds_for_the_per_level_variant(self):
        base = sample_score(self.CAND, self.REF, PER_LEVEL)
        scaled_ref = [[t * 7.0 for t in lvl] for lvl in self.REF]
        scaled_cand = [[t * 7.0 for t in lvl] for lvl in self.CAND]
        self.assertAlmostEqual(
            sample_score(scaled_cand, scaled_ref, PER_LEVEL), base, places=12
        )


class TestMetricConfig(unittest.TestCase):
    def test_alpha_must_exceed_one(self):
        for alpha in (1.0, 0.5, 0.0, -1.0):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                MetricConfig(alpha=alpha, level_weights=(3.0, 3.0, 4.0))

    def test_rejects_empty_or_negative_weights(self):
        with self.assertRaises(ValueError):
            MetricConfig(alpha=2.0, level_weights=())
        with self.assertRaises(ValueError):
            MetricConfig(alpha=2.0, level_weights=(1.0, -1.0))
        with self.assertRaises(ValueError):
            MetricConfig(alpha=2.0, level_weights=(0.0, 0.0))

    def test_rejects_unknown_normalization(self):
        with self.assertRaises(ValueError):
            MetricConfig(alpha=2.0, level_weights=(1.0,), normalization="softmax")

    def test_paper_config_is_the_published_one(self):
        self.assertEqual(PAPER.alpha, 2.0)
        self.assertEqual(PAPER.level_weights, (3.0, 3.0, 4.0))
        self.assertEqual(PAPER.normalization, "global")
        self.assertEqual(PAPER.n_levels, 3)


if __name__ == "__main__":
    unittest.main()
