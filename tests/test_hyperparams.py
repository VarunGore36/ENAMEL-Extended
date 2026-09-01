"""Tests for hyperparameter sensitivity and rank stability."""

from __future__ import annotations

import unittest

from enamel_ext.metrics.score import TIMEOUT
from enamel_ext.report.hyperparams import (
    attainable_range,
    compare_under_h,
    eff_at_h,
    reorderable_pairs,
    rescore_at_alpha,
)

#: Per-level means recovered by inverting the paper's Table 10 hardness sweeps.
#: See docs/analysis/table10-recovery.md and scripts/recover_table10.py.
RECOVERED_F = (0.638, 0.453, 0.355)

#: Table 10 (b)(c)(d) of the paper: GPT-4 Turbo eff@1 as one hardness weight is
#: swept over 1..5 with the other two held at their published values (3, 3, 4).
#: Keyed by which level's weight varies.
TABLE10 = {
    1: {1: 0.428, 2: 0.451, 3: 0.470, 4: 0.486, 5: 0.498},
    2: {1: 0.474, 2: 0.472, 3: 0.470, 4: 0.469, 5: 0.467},
    3: {1: 0.520, 2: 0.499, 3: 0.483, 4: 0.470, 5: 0.460},
}
PAPER_H = (3.0, 3.0, 4.0)


class TestEffAtH(unittest.TestCase):
    def test_reproduces_the_published_point(self):
        got = eff_at_h(RECOVERED_F, PAPER_H)
        self.assertAlmostEqual(got, 0.470, delta=0.002)

    def test_reproduces_every_table10_entry_from_three_numbers(self):
        """The strongest available check that ``eff@k`` is linear in ``h``.

        Fifteen published numbers are rebuilt from three recovered per-level
        means. If the aggregation were anything other than a weighted mean over
        levels, the residuals would not all sit inside rounding error.
        """
        for level, sweep in TABLE10.items():
            for weight, published in sweep.items():
                h = list(PAPER_H)
                h[level - 1] = float(weight)
                with self.subTest(level=level, weight=weight):
                    self.assertAlmostEqual(eff_at_h(RECOVERED_F, h), published, delta=0.002)

    def test_is_a_convex_combination(self):
        f = (0.2, 0.5, 0.9)
        for h in [(1, 1, 1), (3, 3, 4), (1, 1, 100), (100, 1, 1)]:
            self.assertLessEqual(min(f), eff_at_h(f, h))
            self.assertLessEqual(eff_at_h(f, h), max(f))

    def test_invariant_to_rescaling_h(self):
        self.assertAlmostEqual(
            eff_at_h(RECOVERED_F, (3, 3, 4)),
            eff_at_h(RECOVERED_F, (30, 30, 40)),
            places=12,
        )

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            eff_at_h((0.1, 0.2), (1, 1, 1))
        with self.assertRaises(ValueError):
            eff_at_h((), ())
        with self.assertRaises(ValueError):
            eff_at_h((0.1, 0.2), (1, 0))  # h_l > 0 strictly


class TestAttainableRange(unittest.TestCase):
    def test_hull_of_the_level_means(self):
        self.assertEqual(attainable_range(RECOVERED_F), (0.355, 0.638))

    def test_span_exceeds_the_leaderboard_spread(self):
        """0.283 of reachable range against a 0.062 spread over the paper's top
        four models: the ordering is not obviously robust to the choice of h,
        which is what compare_under_h is for."""
        lo, hi = attainable_range(RECOVERED_F)
        self.assertGreater(hi - lo, 0.062)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            attainable_range([])


class TestCompareUnderH(unittest.TestCase):
    def test_dominance_is_stable(self):
        a = (0.7, 0.6, 0.5)
        b = (0.6, 0.5, 0.4)
        self.assertEqual(compare_under_h(a, b).verdict, "a_always")
        self.assertEqual(compare_under_h(b, a).verdict, "b_always")
        self.assertTrue(compare_under_h(a, b).stable)

    def test_weak_dominance_with_one_equal_level_is_still_stable(self):
        self.assertEqual(compare_under_h((0.5, 0.6, 0.5), (0.5, 0.5, 0.4)).verdict, "a_always")

    def test_identical_models_tie(self):
        cmp = compare_under_h((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.assertEqual(cmp.verdict, "tie")
        self.assertTrue(cmp.stable)

    def test_crossing_level_means_are_reorderable(self):
        cmp = compare_under_h((0.7, 0.4, 0.3), (0.5, 0.5, 0.4))
        self.assertEqual(cmp.verdict, "reorderable")
        self.assertFalse(cmp.stable)

    def test_witnesses_actually_reorder(self):
        """A witness is only useful if it is exhibitable, so check it."""
        pairs = [
            ((0.7, 0.4, 0.3), (0.5, 0.5, 0.4)),
            ((0.9, 0.1, 0.1), (0.2, 0.8, 0.8)),
            ((0.50, 0.50, 0.60), (0.55, 0.55, 0.10)),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                cmp = compare_under_h(a, b)
                self.assertEqual(cmp.verdict, "reorderable")
                self.assertIsNotNone(cmp.witness_a)
                self.assertIsNotNone(cmp.witness_b)
                self.assertGreater(eff_at_h(a, cmp.witness_a), eff_at_h(b, cmp.witness_a))
                self.assertGreater(eff_at_h(b, cmp.witness_b), eff_at_h(a, cmp.witness_b))

    def test_the_paper_h_may_hide_a_reorderable_pair(self):
        """Two models that tie exactly at h = (3, 3, 4) yet swap under other h.
        A leaderboard reporting one h cannot show this."""
        a = (0.60, 0.60, 0.40)
        b = (0.50, 0.50, 0.55)
        self.assertAlmostEqual(eff_at_h(a, PAPER_H), eff_at_h(b, PAPER_H), places=12)
        self.assertEqual(compare_under_h(a, b).verdict, "reorderable")

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            compare_under_h((0.1, 0.2), (0.1,))
        with self.assertRaises(ValueError):
            compare_under_h((), ())


class TestReorderablePairs(unittest.TestCase):
    def test_finds_only_the_crossing_pair(self):
        models = {
            "a": (0.7, 0.4, 0.3),
            "b": (0.5, 0.5, 0.4),
            "c": (0.2, 0.2, 0.2),
        }
        found = reorderable_pairs(models)
        self.assertEqual([(x, y) for x, y, _ in found], [("a", "b")])

    def test_a_totally_ordered_board_is_empty(self):
        models = {
            "a": (0.9, 0.8, 0.7),
            "b": (0.6, 0.5, 0.4),
            "c": (0.3, 0.2, 0.1),
        }
        self.assertEqual(reorderable_pairs(models), [])

    def test_handles_fewer_than_two_models(self):
        self.assertEqual(reorderable_pairs({}), [])
        self.assertEqual(reorderable_pairs({"a": (0.1, 0.2, 0.3)}), [])


class TestRescoreAtAlpha(unittest.TestCase):
    REF = [[1.0], [1.0], [1.0]]

    def test_lowering_alpha_is_allowed_even_when_censored(self):
        candidate = [[1.5], [TIMEOUT], [TIMEOUT]]
        got = rescore_at_alpha(candidate, self.REF, new_alpha=1.8, measured_alpha=2.0)
        expected = 3 * ((1.8 - 1.5) / (1.8 - 1.0)) / 10
        self.assertAlmostEqual(got, expected, places=12)

    def test_lowering_alpha_can_clamp_a_level_to_zero(self):
        candidate = [[1.5], [1.5], [1.5]]
        self.assertEqual(rescore_at_alpha(candidate, self.REF, new_alpha=1.4, measured_alpha=2.0), 0.0)

    def test_raising_alpha_on_censored_data_is_refused(self):
        candidate = [[1.5], [TIMEOUT], [TIMEOUT]]
        with self.assertRaises(ValueError):
            rescore_at_alpha(candidate, self.REF, new_alpha=3.0, measured_alpha=2.0)

    def test_raising_alpha_is_fine_when_nothing_was_censored(self):
        candidate = [[1.5], [1.5], [1.5]]
        got = rescore_at_alpha(candidate, self.REF, new_alpha=3.0, measured_alpha=2.0)
        self.assertAlmostEqual(got, (3.0 - 1.5) / (3.0 - 1.0), places=12)

    def test_alpha_sets_the_dynamic_range_not_just_the_timeout(self):
        """A property the paper does not remark on.

        ``f = (T - t)/(T - t*)`` tends to 1 as ``T`` grows, from below when the
        candidate is slower than the reference and from above when it is faster.
        So alpha is a compression knob: large alpha pulls every finite score
        toward 1 and flattens the metric, and the reported rise of eff@1 with
        alpha is that compression, not a tolerance effect.
        """
        slow = [[1.5], [1.5], [1.5]]
        fast = [[0.5], [0.5], [0.5]]
        slow_scores = [
            rescore_at_alpha(slow, self.REF, new_alpha=a, measured_alpha=a) for a in (1.5, 2.0, 4.0, 100.0)
        ]
        fast_scores = [
            rescore_at_alpha(fast, self.REF, new_alpha=a, measured_alpha=a) for a in (1.5, 2.0, 4.0, 100.0)
        ]
        self.assertEqual(slow_scores, sorted(slow_scores))  # rises toward 1
        self.assertEqual(fast_scores, sorted(fast_scores, reverse=True))  # falls toward 1
        self.assertAlmostEqual(slow_scores[-1], 1.0, delta=0.01)
        self.assertAlmostEqual(fast_scores[-1], 1.0, delta=0.01)

    def test_incorrect_samples_score_zero_at_any_alpha(self):
        candidate = [[0.1], [0.1], [0.1]]
        got = rescore_at_alpha(
            candidate, self.REF, new_alpha=1.5, measured_alpha=2.0, correct=False
        )
        self.assertEqual(got, 0.0)

    def test_rejects_a_measured_alpha_that_is_not_a_valid_limit(self):
        with self.assertRaises(ValueError):
            rescore_at_alpha([[1.0]] * 3, self.REF, new_alpha=1.5, measured_alpha=1.0)


class TestNumericalRobustness(unittest.TestCase):
    """Regression tests for two float traps in the exact reorderability test.

    Both were found by asserting that a witness reorders rather than trusting
    that it does.
    """

    def test_a_witness_is_never_a_tie_produced_by_round_off(self):
        """``h = (1, 1, 1)`` makes these two models exactly equal, and the
        weighted difference is a total cancellation whose float residue is
        positive. A naive ``> 0`` test accepts it and reports a tie as a win."""
        a = (0.7, 0.4, 0.3)
        b = (0.5, 0.5, 0.4)
        cmp = compare_under_h(a, b)
        self.assertAlmostEqual(eff_at_h(a, (1, 1, 1)), eff_at_h(b, (1, 1, 1)), places=12)
        self.assertNotEqual(cmp.witness_b, (1, 1, 1))
        self.assertGreater(eff_at_h(b, cmp.witness_b), eff_at_h(a, cmp.witness_b))

    def test_round_off_in_a_level_difference_is_not_a_sign_change(self):
        """0.1 + 0.2 != 0.3 in binary. The models are equal on level 1 and A
        dominates on level 2, so the pair is stable; a bit of subtraction noise
        must not turn that into a reorderable pair."""
        a = (0.3, 0.5)
        b = (0.1 + 0.2, 0.4)
        self.assertNotEqual(a[0] - b[0], 0.0)  # the trap is present
        self.assertEqual(compare_under_h(a, b).verdict, "a_always")

    def test_a_large_weight_witness_still_reorders(self):
        """One level with a small edge against two with a larger one needs a
        witness weight in the hundreds, where round-off scales with the weight."""
        a = (0.500, 0.500, 0.5010)
        b = (0.505, 0.505, 0.5000)
        cmp = compare_under_h(a, b)
        self.assertEqual(cmp.verdict, "reorderable")
        self.assertGreater(max(cmp.witness_a), 5)
        self.assertGreater(eff_at_h(a, cmp.witness_a), eff_at_h(b, cmp.witness_a))
        self.assertGreater(eff_at_h(b, cmp.witness_b), eff_at_h(a, cmp.witness_b))


if __name__ == "__main__":
    unittest.main()
