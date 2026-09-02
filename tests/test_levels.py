"""Tests for per-level discrimination: q, tolerated slowdown, sensitivity."""

from __future__ import annotations

import unittest

from enamel_ext.report.levels import (
    describe_levels,
    level_fraction_at,
    limit_level,
    limit_level_counts,
    q_distribution,
    q_ratios,
    sensitivity_shares,
    tolerated_slowdown,
)

#: README section 2.2: the level fraction at alpha = 2 for a candidate X times
#: slower than the reference, at a level whose reference time is q of the one
#: that sets T_i. Keyed q -> X -> published value.
README_TABLE = {
    0.01: {2: 0.995, 5: 0.980, 10: 0.955, 50: 0.754},
    0.05: {2: 0.974, 5: 0.897, 10: 0.769, 50: 0.000},
    0.10: {2: 0.947, 5: 0.789, 10: 0.526, 50: 0.000},
}

#: The same section on level 3, where q = 1 by definition.
README_SHARP = {1.25: 0.750, 1.5: 0.500, 2.0: 0.000}


def _level(*times: float) -> tuple[float, ...]:
    return times


class TestLevelFractionAt(unittest.TestCase):
    def test_reproduces_the_readme_table(self):
        """Twelve tabulated values, so the document and the code check each
        other rather than the table being asserted prose."""
        for q, row in README_TABLE.items():
            for slowdown, published in row.items():
                with self.subTest(q=q, slowdown=slowdown):
                    got = level_fraction_at(q, slowdown, 2.0)
                    self.assertAlmostEqual(got, published, delta=0.001)

    def test_the_limit_setting_level_is_sharp(self):
        for slowdown, published in README_SHARP.items():
            with self.subTest(slowdown=slowdown):
                self.assertAlmostEqual(level_fraction_at(1.0, slowdown, 2.0), published, places=9)

    def test_matching_the_reference_scores_one_at_every_q(self):
        for q in (0.001, 0.01, 0.5, 1.0):
            self.assertAlmostEqual(level_fraction_at(q, 1.0, 2.0), 1.0, places=12)

    def test_beating_the_reference_scores_above_one(self):
        self.assertGreater(level_fraction_at(1.0, 0.5, 2.0), 1.0)

    def test_decreasing_in_slowdown_and_in_q(self):
        """Larger q is sharper discrimination, which is the whole point of
        section 2.2: small q compresses the score."""
        for q in (0.01, 0.1, 1.0):
            values = [level_fraction_at(q, x, 2.0) for x in (1.0, 2.0, 5.0, 10.0)]
            self.assertEqual(values, sorted(values, reverse=True))
        for slowdown in (2.0, 5.0, 10.0):
            values = [level_fraction_at(q, slowdown, 2.0) for q in (0.01, 0.05, 0.5, 1.0)]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_a_censored_candidate_scores_zero(self):
        self.assertEqual(level_fraction_at(0.01, float("inf"), 2.0), 0.0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            level_fraction_at(0.1, -1.0, 2.0)
        with self.assertRaises(ValueError):
            level_fraction_at(2.5, 1.0, 2.0)  # denominator would be negative


class TestToleratedSlowdown(unittest.TestCase):
    def test_is_alpha_over_q(self):
        self.assertAlmostEqual(tolerated_slowdown(0.01, 2.0), 200.0)
        self.assertAlmostEqual(tolerated_slowdown(1.0, 2.0), 2.0)

    def test_is_exactly_where_the_level_stops_scoring(self):
        for q in (0.01, 0.05, 0.25, 1.0):
            x = tolerated_slowdown(q, 2.0)
            with self.subTest(q=q):
                self.assertEqual(level_fraction_at(q, x, 2.0), 0.0)
                self.assertGreater(level_fraction_at(q, x * 0.999, 2.0), 0.0)

    def test_rejects_non_positive_q(self):
        with self.assertRaises(ValueError):
            tolerated_slowdown(0.0, 2.0)


class TestQRatios(unittest.TestCase):
    def test_normalizes_by_the_largest_worst_case(self):
        times = [_level(0.01, 0.02), _level(0.1, 0.05), _level(1.0, 0.9)]
        self.assertEqual(q_ratios(times), (0.02, 0.1, 1.0))

    def test_uses_the_worst_case_within_a_level_not_the_mean(self):
        times = [_level(0.5, 0.01), _level(1.0, 1.0)]
        self.assertEqual(q_ratios(times), (0.5, 1.0))

    def test_the_limit_setting_level_is_named_even_when_it_is_not_the_last(self):
        times = [_level(0.1), _level(2.0), _level(1.0)]
        self.assertEqual(limit_level(times), 2)
        self.assertEqual(q_ratios(times), (0.05, 1.0, 0.5))

    def test_ties_name_the_earliest_level(self):
        self.assertEqual(limit_level([_level(1.0), _level(1.0)]), 1)

    def test_rejects_missing_or_impossible_times(self):
        with self.assertRaises(ValueError):
            q_ratios([])
        with self.assertRaises(ValueError):
            q_ratios([_level(1.0), ()])
        with self.assertRaises(ValueError):
            q_ratios([_level(0.0)])


class TestDistributionAcrossProblems(unittest.TestCase):
    PROBLEMS = (
        [_level(0.01), _level(0.1), _level(1.0)],
        [_level(0.02), _level(0.2), _level(1.0)],
        [_level(0.03), _level(0.3), _level(1.0)],
    )

    def test_columns_are_per_level(self):
        self.assertEqual(
            q_distribution(self.PROBLEMS),
            ((0.01, 0.02, 0.03), (0.1, 0.2, 0.3), (1.0, 1.0, 1.0)),
        )

    def test_counts_which_level_sets_the_limit(self):
        problems = list(self.PROBLEMS) + [[_level(0.1), _level(3.0), _level(1.0)]]
        self.assertEqual(limit_level_counts(problems), {3: 3, 2: 1})

    def test_refuses_problems_with_different_level_counts(self):
        problems = [self.PROBLEMS[0], [_level(0.1), _level(1.0)]]
        with self.assertRaises(ValueError):
            q_distribution(problems)

    def test_summarizes_each_level(self):
        summaries = describe_levels(self.PROBLEMS, alpha=2.0, slowdowns=(10.0,))
        self.assertEqual([s.level for s in summaries], [1, 2, 3])
        self.assertEqual([s.n_problems for s in summaries], [3, 3, 3])
        self.assertAlmostEqual(summaries[0].q_median, 0.02)
        self.assertAlmostEqual(summaries[0].q_min, 0.01)
        self.assertAlmostEqual(summaries[0].q_max, 0.03)
        self.assertAlmostEqual(summaries[0].tolerated, 100.0)
        self.assertAlmostEqual(summaries[0].fractions[10.0], 0.909, delta=0.001)
        self.assertAlmostEqual(summaries[2].tolerated, 2.0)
        self.assertEqual(summaries[2].fractions[10.0], 0.0)

    def test_rejects_no_problems(self):
        with self.assertRaises(ValueError):
            describe_levels([])


class TestSensitivityShares(unittest.TestCase):
    def test_shares_sum_to_one(self):
        shares = sensitivity_shares((0.02, 0.2, 1.0), (3.0, 3.0, 4.0), 2.0)
        self.assertAlmostEqual(sum(shares), 1.0, places=12)

    def test_small_q_concentrates_the_response_on_the_last_level(self):
        """Section 2.2's claim as a number: 40% of the weight can carry almost
        all of the score's response to a slowdown."""
        shares = sensitivity_shares((0.01, 0.05, 1.0), (3.0, 3.0, 4.0), 2.0)
        self.assertGreater(shares[2], 0.9)
        self.assertLess(shares[0] + shares[1], 0.1)

    def test_equal_q_gives_back_the_hardness_weights(self):
        shares = sensitivity_shares((1.0, 1.0, 1.0), (3.0, 3.0, 4.0), 2.0)
        for got, want in zip(shares, (0.3, 0.3, 0.4)):
            self.assertAlmostEqual(got, want, places=12)

    def test_a_zero_weight_level_contributes_nothing(self):
        shares = sensitivity_shares((1.0, 1.0), (0.0, 1.0), 2.0)
        self.assertEqual(shares, (0.0, 1.0))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            sensitivity_shares((0.1, 0.2), (1.0,), 2.0)
        with self.assertRaises(ValueError):
            sensitivity_shares((), (), 2.0)
        with self.assertRaises(ValueError):
            sensitivity_shares((0.0, 1.0), (1.0, 1.0), 2.0)
        with self.assertRaises(ValueError):
            sensitivity_shares((1.0, 2.5), (1.0, 1.0), 2.0)
        with self.assertRaises(ValueError):
            sensitivity_shares((1.0, 1.0), (0.0, 0.0), 2.0)


if __name__ == "__main__":
    unittest.main()
