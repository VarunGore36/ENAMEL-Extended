"""How a machine-drift correction interacts with censoring.

Appendix C.1 says the reference time on the slowest test case is used to
"further calibrate" the execution time of generated code. One reading of that is
a per-run scaling correction; these tests pin what such a correction has to do to
be sound, since the unsound version reads as a slow candidate rather than as a
harness bug. Reasoning in docs/analysis/appendix-c1-calibration.md.
"""

import unittest

from enamel_ext.metrics.score import PAPER, TIMEOUT, level_fraction, sample_score, time_limit

#: Reference times whose worst case is at level 3, so level 3 sets ``T_i = 1.0``.
REFERENCE = [[0.05, 0.05], [0.125, 0.125], [0.25, 0.5]]

SLOWER = (1.25, 1.5, 2.0, 4.0)
FASTER = (0.25, 0.5, 0.9, 1.0)

#: Slowdowns whose censored band starts on a dyadic time, so the boundary between
#: censored and not is exact in binary floating point. 8.0 is excluded: its band
#: would reach level 2's reference worst case and censor that level too.
EXACT_BAND = (2.0, 4.0)


def candidate_at(level3_worst):
    """Matches the reference on levels 1 and 2. Level 3 gets one fast case, chosen
    to stay clear of every censored band used here, and one at ``level3_worst``."""
    return [[0.05, 0.05], [0.125, 0.125], [0.0625, level3_worst]]


def observe(true_times, slowdown, wall_limit):
    """What a harness records for a candidate whose reference-machine times are
    ``true_times``, run on a machine ``slowdown`` times slower, killed at
    ``wall_limit`` wall-clock seconds, with the survivors divided back by
    ``slowdown``."""
    recorded = []
    for level in true_times:
        row = []
        for t in level:
            wall = slowdown * t
            row.append(TIMEOUT if wall >= wall_limit else wall / slowdown)
        recorded.append(row)
    return recorded


def max_loss(slowdown, alpha):
    """Level fraction forfeited by a candidate at the bottom of the censored band,
    at the level that sets ``T_i``."""
    return (1.0 - 1.0 / slowdown) / (1.0 - 1.0 / alpha)


class TestCalibratingTheLimit(unittest.TestCase):
    """Dividing the times by the drift factor is not enough on its own: the kill
    threshold has to be multiplied by it as well."""

    def setUp(self):
        self.limit = time_limit(REFERENCE, PAPER)

    def midband(self, slowdown):
        """A true worst-case time in the middle of ``[T_i/s, T_i)``, well clear of
        both ends so the verdict does not turn on a rounding step."""
        return (self.limit / slowdown + self.limit) / 2.0

    def test_level_3_sets_the_limit(self):
        self.assertEqual(self.limit, 1.0)

    def test_scaling_the_limit_reproduces_the_undrifted_score(self):
        for s in (1.0,) + SLOWER:
            with self.subTest(slowdown=s):
                true = candidate_at(self.midband(s))
                drifted = observe(true, s, s * self.limit)
                self.assertAlmostEqual(
                    sample_score(drifted, REFERENCE, PAPER),
                    sample_score(true, REFERENCE, PAPER),
                    places=12,
                )

    def test_the_round_trip_is_exact_at_a_power_of_two(self):
        """Not a tolerance artifact: with dyadic inputs the equality is exact."""
        true = candidate_at(0.75)
        drifted = observe(true, 2.0, 2.0 * self.limit)
        self.assertEqual(
            sample_score(drifted, REFERENCE, PAPER), sample_score(true, REFERENCE, PAPER)
        )

    def test_a_faster_machine_needs_no_limit_correction(self):
        """At or below 1 nothing extra is killed, so times alone suffice."""
        for s in FASTER:
            with self.subTest(slowdown=s):
                true = candidate_at(0.75)
                drifted = observe(true, s, self.limit)
                self.assertAlmostEqual(
                    sample_score(drifted, REFERENCE, PAPER),
                    sample_score(true, REFERENCE, PAPER),
                    places=12,
                )

    def test_a_slower_machine_loses_score_it_cannot_recover(self):
        for s in SLOWER:
            with self.subTest(slowdown=s):
                true = candidate_at(self.midband(s))
                time_only = sample_score(observe(true, s, self.limit), REFERENCE, PAPER)
                self.assertLess(time_only, sample_score(true, REFERENCE, PAPER))

    def test_the_error_is_never_optimistic(self):
        """Across the sweep and across the range of candidate speeds, a time-only
        correction can only deflate a score, so it does not average out."""
        for s in FASTER + SLOWER:
            for fraction in (0.25, 0.5, 0.75, 0.9, 1.0, 1.5):
                with self.subTest(slowdown=s, fraction=fraction):
                    true = candidate_at(fraction * self.limit)
                    time_only = sample_score(observe(true, s, self.limit), REFERENCE, PAPER)
                    self.assertLessEqual(time_only, sample_score(true, REFERENCE, PAPER) + 1e-12)

    def test_the_loss_matches_the_closed_form(self):
        """At the midpoint of the band the forfeited level fraction is half of
        ``(1 - 1/s)/(1 - 1/alpha)``, weighted in by level 3's share of ``h``."""
        share = PAPER.level_weights[2] / sum(PAPER.level_weights)
        for s in SLOWER:
            with self.subTest(slowdown=s):
                true = candidate_at(self.midband(s))
                undrifted = sample_score(true, REFERENCE, PAPER)
                time_only = sample_score(observe(true, s, self.limit), REFERENCE, PAPER)
                self.assertAlmostEqual(
                    undrifted - time_only, share * max_loss(s, PAPER.alpha) / 2.0, places=12
                )


class TestTheCensoredBand(unittest.TestCase):
    """The band is ``[T_i/s, T_i)``: inside it a candidate is killed even though
    the threshold its score is defined against says it finished in time."""

    def setUp(self):
        self.limit = time_limit(REFERENCE, PAPER)
        self.censored = sample_score(candidate_at(TIMEOUT), REFERENCE, PAPER)

    def test_the_band_starts_at_the_limit_over_the_slowdown(self):
        for s in EXACT_BAND:
            with self.subTest(slowdown=s):
                bottom = self.limit / s
                at = observe(candidate_at(bottom), s, self.limit)
                just_below = observe(candidate_at(bottom - 2.0**-20), s, self.limit)
                self.assertEqual(sample_score(at, REFERENCE, PAPER), self.censored)
                self.assertGreater(sample_score(just_below, REFERENCE, PAPER), self.censored)

    def test_the_whole_level_fraction_goes(self):
        """A candidate at the bottom of the band forfeits ``max_loss`` exactly, so
        at 2x drift a candidate that matches the expert's worst case scores 0."""
        for s in EXACT_BAND:
            with self.subTest(slowdown=s):
                true = candidate_at(self.limit / s)
                lost = level_fraction(true[2], REFERENCE[2], self.limit)
                self.assertEqual(lost, max_loss(s, PAPER.alpha))
                self.assertEqual(
                    sample_score(observe(true, s, self.limit), REFERENCE, PAPER), self.censored
                )

    def test_the_documented_losses(self):
        """The figures quoted in docs/analysis/appendix-c1-calibration.md."""
        self.assertEqual(max_loss(2.0, 2.0), 1.0)
        self.assertAlmostEqual(max_loss(1.1, 2.0), 0.18, places=2)
        self.assertAlmostEqual(max_loss(1.25, 2.0), 0.40, places=12)
        self.assertAlmostEqual(max_loss(1.5, 2.0), 0.67, places=2)


class TestFastLevelsLoseLess(unittest.TestCase):
    """Both the size and the frequency of the error concentrate at the level that
    sets ``T_i``, which is the level §2.2 shows carries the discrimination."""

    def setUp(self):
        self.limit = time_limit(REFERENCE, PAPER)

    def loss_at(self, q, slowdown):
        """Level fraction forfeited at a level whose reference worst case is ``q``
        times the limit-setting one."""
        reference_worst = q * self.limit / PAPER.alpha
        return level_fraction([self.limit / slowdown], [reference_worst], self.limit)

    def test_the_loss_shrinks_as_the_level_reference_gets_faster(self):
        for s in SLOWER:
            with self.subTest(slowdown=s):
                self.assertAlmostEqual(self.loss_at(1.0, s), max_loss(s, PAPER.alpha), places=12)
                for q in (0.5, 0.1, 0.01):
                    expected = (1.0 - 1.0 / s) / (1.0 - q / PAPER.alpha)
                    self.assertAlmostEqual(self.loss_at(q, s), expected, places=12)
                    self.assertLess(self.loss_at(q, s), self.loss_at(1.0, s))

    def test_a_fast_level_rarely_reaches_the_band_at_all(self):
        """Level 1's reference worst case is a tenth of level 3's here, so code ten
        times slower than the expert is still well below the censored band."""
        self.assertEqual(max(REFERENCE[0]) / max(REFERENCE[2]), 0.1)
        self.assertLess(10.0 * max(REFERENCE[0]), self.limit / 1.25)


if __name__ == "__main__":
    unittest.main()
