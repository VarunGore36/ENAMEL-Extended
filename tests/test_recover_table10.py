"""Tests for the Table 10 inversion behind docs/analysis/table10-recovery.md.

The recovered means are quoted as a constant in three places, so the script that
produces them needs a check that fails when it drifts rather than one that
restates the answer.
"""

from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import contextmanager, redirect_stdout
from fractions import Fraction
from pathlib import Path

from enamel_ext.report.hyperparams import attainable_range, eff_at_h

REPO_ROOT = Path(__file__).resolve().parent.parent


def _script():
    """scripts/recover_table10.py loaded by path, since scripts/ is not a package."""
    location = REPO_ROOT / "scripts" / "recover_table10.py"
    spec = importlib.util.spec_from_file_location("recover_table10", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _script()

#: The means and per-step estimates tabulated in docs/analysis/table10-recovery.md.
DOCUMENTED = {
    1: (0.638, (0.635, 0.641, 0.646, 0.630)),
    2: (0.453, (0.456, 0.452, 0.459, 0.445)),
    3: (0.355, (0.352, 0.355, 0.353, 0.360)),
}


class TestRecoveredValues(unittest.TestCase):
    def test_reproduces_the_documented_means(self):
        for level, (mean, _) in DOCUMENTED.items():
            with self.subTest(level=level):
                self.assertAlmostEqual(SCRIPT.recover(level)[0], mean, places=3)

    def test_reproduces_the_documented_per_step_estimates(self):
        """Four independent estimates per level, so the spread is visible rather
        than averaged away."""
        for level, (_, steps) in DOCUMENTED.items():
            got = SCRIPT.recover(level)[1]
            self.assertEqual(len(got), len(steps))
            for index, (value, published) in enumerate(zip(got, steps)):
                with self.subTest(level=level, step=index):
                    self.assertAlmostEqual(value, published, places=3)

    def test_the_per_step_spread_is_consistent_with_rounding(self):
        """Table 10 is published to three decimals, which admits roughly this
        much disagreement between estimates; more would mean eff@1 is not
        linear in h after all."""
        for level in DOCUMENTED:
            steps = SCRIPT.recover(level)[1]
            with self.subTest(level=level):
                self.assertLess(max(steps) - min(steps), 0.017)

    def test_rebuilding_the_headline_matches_the_published_value(self):
        f = [SCRIPT.recover(level)[0] for level in (1, 2, 3)]
        self.assertAlmostEqual(eff_at_h(f, (3, 3, 4)), SCRIPT.PUBLISHED_EFF1, delta=0.002)

    def test_round_trips_to_every_published_table10_entry(self):
        """The inversion is only sound if the recovered means rebuild all
        fifteen entries, not just the one the sweeps share."""
        f = [SCRIPT.recover(level)[0] for level in (1, 2, 3)]
        for level, sweep in SCRIPT.SWEEPS.items():
            for weight, published in sweep.items():
                h = [3.0, 3.0, 4.0]
                h[level - 1] = float(weight)
                with self.subTest(level=level, weight=weight):
                    self.assertAlmostEqual(eff_at_h(f, h), published, delta=0.002)

    def test_the_levels_are_ordered_hardest_last(self):
        """Later levels use larger inputs, so their mean score should fall. A
        recovery that inverted this would point at the sweep bookkeeping."""
        f = [SCRIPT.recover(level)[0] for level in (1, 2, 3)]
        self.assertEqual(f, sorted(f, reverse=True))

    def test_level_1_discriminates_rather_than_saturating(self):
        """The extreme form of the section 2.2 concern would put F_1 just under
        pass@1; it sits well below, which is the finding the analysis records."""
        f1 = SCRIPT.recover(1)[0]
        self.assertLess(f1, SCRIPT.PUBLISHED_PASS1 - 0.1)
        self.assertGreater(f1, 0.5)


class TestDerivedQuantities(unittest.TestCase):
    """The numbers RESULTS.md and the analysis doc quote alongside the means."""

    def setUp(self):
        self.f = [SCRIPT.recover(level)[0] for level in (1, 2, 3)]

    def test_reachable_range_spans_point_283(self):
        lo, hi = attainable_range(self.f)
        self.assertAlmostEqual(lo, 0.355, places=3)
        self.assertAlmostEqual(hi, 0.638, places=3)
        self.assertAlmostEqual(hi - lo, 0.283, places=3)

    def test_h_has_more_leverage_than_alpha_or_the_leaderboard(self):
        lo, hi = attainable_range(self.f)
        self.assertAlmostEqual(SCRIPT.ALPHA_SPAN, 0.120, places=3)
        self.assertAlmostEqual(SCRIPT.TOP4_SPAN, 0.062, places=3)
        self.assertGreater(hi - lo, SCRIPT.ALPHA_SPAN)
        self.assertGreater(SCRIPT.ALPHA_SPAN, SCRIPT.TOP4_SPAN)

    def test_bounds_on_the_mean_among_correct_samples(self):
        """Dividing by pass@1 bounds the mean score among samples that count as
        correct, since the rest contribute 0 to the numerator only."""
        for level, expected in zip((1, 2, 3), (0.802, 0.569, 0.446)):
            with self.subTest(level=level):
                bound = self.f[level - 1] / SCRIPT.PUBLISHED_PASS1
                self.assertAlmostEqual(bound, expected, places=3)
                self.assertLessEqual(bound, 1.0)


class TestInversionIsExact(unittest.TestCase):
    """Property checks on the arithmetic itself, independent of the paper's
    numbers: build a sweep from a known F, invert it, get F back."""

    @staticmethod
    @contextmanager
    def _sweep(f, level, h_default=(3, 3, 4)):
        """Point the script at a sweep generated from a known ``f``."""
        weights = [
            [*h_default[: level - 1], w, *h_default[level:]] for w in range(1, 6)
        ]
        sweep = {h[level - 1]: eff_at_h(f, h) for h in weights}
        fixed = sum(h_default) - h_default[level - 1]
        saved = SCRIPT.SWEEPS[level], SCRIPT.FIXED_SUM[level]
        SCRIPT.SWEEPS[level], SCRIPT.FIXED_SUM[level] = sweep, fixed
        try:
            yield
        finally:
            SCRIPT.SWEEPS[level], SCRIPT.FIXED_SUM[level] = saved

    def test_recovers_an_arbitrary_f_exactly(self):
        f = (0.9, 0.4, 0.125)
        for level in (1, 2, 3):
            with self.subTest(level=level), self._sweep(f, level):
                mean, steps = SCRIPT.recover(level)
                self.assertAlmostEqual(mean, f[level - 1], places=12)
                for step in steps:
                    self.assertAlmostEqual(step, f[level - 1], places=12)

    def test_every_step_is_the_same_number_when_the_input_is_unrounded(self):
        """The spread in the published recovery is rounding in Table 10, not
        error in the method: given exact inputs the four estimates coincide."""
        f = (Fraction(3, 5), Fraction(4, 9), Fraction(1, 3))
        for level in (1, 2, 3):
            with self.subTest(level=level), self._sweep(f, level):
                mean, steps = SCRIPT.recover(level)
                self.assertEqual(set(steps), {f[level - 1]})
                self.assertEqual(mean, f[level - 1])

    def test_fixed_sums_match_the_paper_defaults(self):
        """FIXED_SUM is the sum of the two weights held at their defaults during
        each sweep; getting it wrong would rescale one level's estimate."""
        for level in (1, 2, 3):
            with self.subTest(level=level):
                self.assertEqual(SCRIPT.FIXED_SUM[level], 10 - (3, 3, 4)[level - 1])

    def test_each_sweep_passes_through_the_published_headline(self):
        """At its default weight every sweep must report the same eff@1."""
        for level, default in zip((1, 2, 3), (3, 3, 4)):
            with self.subTest(level=level):
                self.assertAlmostEqual(
                    SCRIPT.SWEEPS[level][default], SCRIPT.PUBLISHED_EFF1, places=3
                )


class TestScriptRuns(unittest.TestCase):
    def test_main_prints_the_recovery_and_its_own_assertion_holds(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            SCRIPT.main()
        out = buffer.getvalue()
        for fragment in ("F1 = 0.638", "F2 = 0.453", "F3 = 0.355", "0.4693"):
            self.assertIn(fragment, out)


if __name__ == "__main__":
    unittest.main()
