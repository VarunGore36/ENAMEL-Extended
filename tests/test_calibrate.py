"""The calibration probe: what it can measure, and what it refuses to claim."""

from __future__ import annotations

import unittest

from enamel_ext.measure.calibrate import (
    CALIBRATION_VERSION,
    DRIFT_CAVEAT,
    DRIFT_REFUSE,
    REPLICATES,
    WORKLOADS,
    Calibration,
    compare,
    differential,
    probe,
    uniform_factor,
)
from enamel_ext.report.parity import EFF_TOLERANCE, differential_bound

#: Locations that are easy to scale by hand, spanning a decade like the real ones do.
BASE = {"w1": 0.010, "w2": 0.012, "w3": 0.014, "w4": 0.016}


def _cal(noise=(1.0, 1.0), locations=None, **kwargs):
    """A probe whose per-workload replicates are ``location * each noise factor``.

    ``noise`` may be one tuple for every workload or a per-workload mapping. Where
    every factor is 1.0 the location is the given time exactly; otherwise it is the
    Hodges-Lehmann of the scaled replicates.
    """
    where = BASE if locations is None else locations
    if not isinstance(noise, dict):
        noise = {name: noise for name in where}
    return Calibration(
        times={name: tuple(t * f for f in noise[name]) for name, t in where.items()},
        repeats=kwargs.pop("repeats", 6),
        aggregator=kwargs.pop("aggregator", "hodges_lehmann"),
        version=kwargs.pop("version", CALIBRATION_VERSION),
    )


def _scaled(factor, only=None):
    """Locations with every workload scaled, or just one of them."""
    return {
        name: t * factor if only is None or name == only else t
        for name, t in BASE.items()
    }


class StatisticTest(unittest.TestCase):
    def test_two_identical_probes_report_no_drift_and_no_slowdown(self):
        a, b = _cal(), _cal()
        self.assertAlmostEqual(differential(a, b), 1.0)
        self.assertAlmostEqual(uniform_factor(a, b), 1.0)

    def test_a_uniform_slowdown_is_reported_as_uniform_and_not_as_drift(self):
        """The case that cancels in Eq. (1) must not be reported as a difference."""
        a = _cal()
        b = _cal(locations=_scaled(1.4))
        self.assertAlmostEqual(differential(a, b), 1.0)
        self.assertAlmostEqual(uniform_factor(a, b), 1.4)

    def test_one_workload_moving_is_reported_at_its_own_size(self):
        a = _cal()
        b = _cal(locations=_scaled(1.2, only="w1"))
        self.assertAlmostEqual(differential(a, b), 1.2)
        self.assertAlmostEqual(uniform_factor(a, b), 1.2 ** 0.25)

    def test_the_differential_does_not_depend_on_which_probe_came_first(self):
        a, b = _cal(), _cal(locations=_scaled(1.2, only="w2"))
        self.assertAlmostEqual(differential(a, b), differential(b, a))

    def test_the_differential_is_never_below_one(self):
        for factor in (0.5, 0.9, 1.0, 1.1, 2.0):
            a, b = _cal(), _cal(locations=_scaled(factor, only="w3"))
            self.assertGreaterEqual(differential(a, b), 1.0)

    def test_a_one_workload_probe_can_only_ever_report_no_drift(self):
        """The reason the probe is a vector: one number sees only the uniform factor."""
        one = {"w1": BASE["w1"]}
        a = _cal(locations=one)
        for factor in (1.1, 2.0, 10.0):
            b = _cal(locations={"w1": BASE["w1"] * factor})
            self.assertAlmostEqual(differential(a, b), 1.0)
            self.assertAlmostEqual(uniform_factor(a, b), factor)

    def test_the_location_is_the_hodges_lehmann_of_the_replicates(self):
        """The paper's estimator again, one level up, across replicates."""
        c = _cal(noise=(1.0, 1.5, 2.0))
        # Walsh averages of (1, 1.5, 2) are 1, 1.25, 1.5, 1.5, 1.75, 2; median 1.5.
        self.assertAlmostEqual(c.location()["w1"], BASE["w1"] * 1.5)

    def test_a_uniform_factor_on_the_replicates_cancels_in_the_location_ratio(self):
        """Why the estimator's bias matters less than the stability of its ratio."""
        a = _cal(noise=(1.0, 1.5, 2.0))
        b = _cal(noise=(1.0, 1.5, 2.0), locations=_scaled(1.2, only="w1"))
        self.assertAlmostEqual(differential(a, b), 1.2)


class ResolutionTest(unittest.TestCase):
    def test_a_probe_with_repeatable_replicates_resolves_perfectly(self):
        self.assertAlmostEqual(_cal(noise=(1.0, 1.0)).resolution(), 1.0)
        self.assertTrue(_cal(noise=(1.0, 1.0)).resolves_parity())

    def test_noise_that_hits_every_workload_alike_costs_no_resolution(self):
        """A probe that slows down as a whole has not lost the ability to see drift."""
        self.assertAlmostEqual(_cal(noise=(1.0, 1.3)).resolution(), 1.0)

    def test_noise_that_hits_one_workload_is_the_resolution(self):
        c = _cal(noise={"w1": (1.0, 1.3), "w2": (1.0, 1.0), "w3": (1.0, 1.0), "w4": (1.0, 1.0)})
        self.assertAlmostEqual(c.resolution(), 1.3)
        self.assertFalse(c.resolves_parity())

    def test_an_isolated_slow_replicate_costs_part_of_its_size(self):
        """The price of not using the minimum, which would have absorbed this whole."""
        c = _cal(
            noise={
                "w1": (1.0, 1.0, 1.0, 1.4),
                "w2": (1.0, 1.0, 1.0, 1.0),
                "w3": (1.0, 1.0, 1.0, 1.0),
                "w4": (1.0, 1.0, 1.0, 1.0),
            }
        )
        # Whichever half holds the slow replicate reports the Hodges-Lehmann of one
        # 1.0 and one 1.4, which is 1.2, against 1.0 from the other half.
        self.assertAlmostEqual(c.resolution(), 1.2)

    def test_the_resolution_is_the_worst_split_not_the_average(self):
        """Overstating the instrument's noise costs sensitivity; understating invents drift."""
        c = _cal(
            noise={
                "w1": (1.0, 1.0, 1.4, 1.4),
                "w2": (1.0, 1.0, 1.0, 1.0),
                "w3": (1.0, 1.0, 1.0, 1.0),
                "w4": (1.0, 1.0, 1.0, 1.0),
            }
        )
        # One of the three splits separates the two slow replicates from the two fast
        # ones and sees 1.4; the other two see nothing, so an average would say 1.13.
        self.assertAlmostEqual(c.resolution(), 1.4)

    def test_a_probe_that_cannot_state_its_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            Calibration(times={"w1": (0.01,)}, repeats=6, aggregator="min")

    def test_an_odd_replicate_count_still_splits_into_equal_halves(self):
        """Both sides over the same width, or the split itself would be a differential."""
        c = _cal(
            noise={
                "w1": (1.0, 1.0, 1.3),
                "w2": (1.0, 1.0, 1.0),
                "w3": (1.0, 1.0, 1.0),
                "w4": (1.0, 1.0, 1.0),
            }
        )
        # Replicate 0 against replicate 2 sees the 1.3; the leftover one sits out.
        self.assertAlmostEqual(c.resolution(), 1.3)


class ComparabilityTest(unittest.TestCase):
    def test_a_probe_is_comparable_to_itself(self):
        self.assertTrue(_cal().comparable(_cal()))

    def test_every_identity_field_breaks_comparability(self):
        base = _cal()
        others = {
            "version": _cal(version=CALIBRATION_VERSION + 1),
            "repeats": _cal(repeats=3),
            "aggregator": _cal(aggregator="min"),
            "replicates": _cal(noise=(1.0, 1.0, 1.0)),
            "workloads": _cal(locations={"w1": 0.01, "w2": 0.02}),
        }
        for field, other in others.items():
            with self.subTest(field=field):
                self.assertFalse(base.comparable(other))
                with self.assertRaises(ValueError):
                    differential(base, other)
                with self.assertRaises(ValueError):
                    uniform_factor(base, other)


class ThresholdTest(unittest.TestCase):
    """The two floors are the parity tolerance restated, not free parameters."""

    def test_the_caveat_floor_is_where_drift_could_eat_the_whole_tolerance(self):
        self.assertAlmostEqual(differential_bound(DRIFT_CAVEAT), EFF_TOLERANCE)

    def test_the_refusal_floor_is_where_it_could_eat_twice_it(self):
        self.assertAlmostEqual(differential_bound(DRIFT_REFUSE), 2 * EFF_TOLERANCE)

    def test_the_default_replicate_count_can_be_halved(self):
        self.assertEqual(REPLICATES % 2, 0)


class VerdictTest(unittest.TestCase):
    #: One workload noisy enough that this pair cannot see a parity-sized change.
    NOISY = {"w1": (1.0, 1.3), "w2": (1.0, 1.0), "w3": (1.0, 1.0), "w4": (1.0, 1.0)}

    def test_a_quiet_pair_that_did_not_move_says_nothing(self):
        drift = compare(_cal(), _cal())
        self.assertAlmostEqual(drift.factor, 1.0)
        self.assertFalse(drift.caveat)
        self.assertFalse(drift.refuse)
        self.assertTrue(drift.resolves_parity)

    def test_a_quiet_pair_caveats_at_the_tolerance_and_refuses_at_twice_it(self):
        small = compare(_cal(), _cal(locations=_scaled(1.03, only="w1")))
        self.assertTrue(small.caveat)
        self.assertFalse(small.refuse)
        large = compare(_cal(), _cal(locations=_scaled(1.10, only="w1")))
        self.assertTrue(large.caveat)
        self.assertTrue(large.refuse)

    def test_a_uniform_slowdown_is_never_a_caveat_however_large(self):
        drift = compare(_cal(), _cal(locations=_scaled(3.0)))
        self.assertAlmostEqual(drift.uniform, 3.0)
        self.assertFalse(drift.caveat)
        self.assertFalse(drift.refuse)

    def test_a_noisy_pair_stays_silent_about_drift_it_cannot_distinguish(self):
        """The floors are floors: noise must not be reported as a machine change."""
        a = _cal(noise=self.NOISY)
        b = _cal(noise=self.NOISY, locations=_scaled(1.10, only="w2"))
        drift = compare(a, b)
        self.assertAlmostEqual(drift.factor, 1.10)
        self.assertAlmostEqual(drift.caveat_at, 1.3)
        self.assertAlmostEqual(drift.refuse_at, 1.3)
        self.assertFalse(drift.caveat)
        self.assertFalse(drift.refuse)
        self.assertFalse(drift.resolves_parity)

    def test_a_noisy_pair_still_reports_gross_drift(self):
        a = _cal(noise=self.NOISY)
        b = _cal(noise=self.NOISY, locations=_scaled(2.0, only="w2"))
        drift = compare(a, b)
        self.assertTrue(drift.caveat)
        self.assertTrue(drift.refuse)

    def test_the_coarser_of_the_two_probes_sets_the_threshold(self):
        quiet, noisy = _cal(), _cal(noise=self.NOISY)
        for a, b in ((quiet, noisy), (noisy, quiet)):
            with self.subTest(first=a.times["w1"]):
                self.assertAlmostEqual(compare(a, b).resolution, 1.3)


class ProbeTest(unittest.TestCase):
    """The one test that pays for real timings, kept to the smallest probe there is."""

    def test_a_probe_times_every_workload_and_records_its_settings(self):
        c = probe(repeats=1, replicates=2)
        self.assertEqual(c.names, tuple(sorted(WORKLOADS)))
        self.assertEqual(c.replicates, 2)
        self.assertEqual(c.repeats, 1)
        self.assertEqual(c.aggregator, "hodges_lehmann")
        self.assertEqual(c.version, CALIBRATION_VERSION)
        for name, series in c.times.items():
            with self.subTest(workload=name):
                self.assertTrue(all(t > 0 for t in series), series)
        self.assertGreaterEqual(c.resolution(), 1.0)
        self.assertTrue(c.comparable(c))

    def test_a_probe_that_could_not_state_its_resolution_is_refused_before_timing(self):
        for replicates in (0, 1):
            with self.subTest(replicates=replicates):
                with self.assertRaises(ValueError):
                    probe(repeats=1, replicates=replicates)


if __name__ == "__main__":
    unittest.main()
