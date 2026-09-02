"""Tests for the solution set, the run record, the orchestrator and the report."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from enamel_ext.data.schema import Provenance
from enamel_ext.data.sources import synthetic_problem_set
from enamel_ext.measure import (
    OK,
    SKIPPED,
    WRONG_ANSWER,
    LevelMeasurement,
    ReferenceMeasurement,
    SolutionMeasurement,
)
from enamel_ext.measure.runner import PAPER_REPEATS, RunConfig
from enamel_ext.measure.sandbox import SandboxError
from enamel_ext.metrics.effk import eff_at_k as raw_eff_at_k
from enamel_ext.metrics.score import PAPER, TIMEOUT, MetricConfig, sample_score
from enamel_ext.pipeline import (
    CENSORED_TOKEN,
    RECORD_SCHEMA_VERSION,
    SOLUTIONS_SCHEMA_VERSION,
    Environment,
    ProblemRecord,
    RunRecord,
    SampleRecord,
    SolutionSet,
    format_summary,
    load_record,
    load_solutions,
    orchestrate,
    record_from_json,
    record_to_json,
    run_evaluation,
    save_record,
    selected_ids,
    solution_set_from_json,
    solution_set_to_json,
    synthetic_solutions,
)
from enamel_ext.report.hyperparams import eff_at_h

PROV = Provenance(name="test", url="local", license="Apache-2.0", retrieved="1970-01-01")

REPO_ROOT = Path(__file__).resolve().parent.parent

#: One scored level keeps the hand-built records small; h then cannot matter.
ONE_LEVEL = MetricConfig(alpha=2.0, level_weights=(1.0,))

#: Three levels like the paper, equally weighted, to tell a carried config apart.
FLAT_H = MetricConfig(alpha=2.0, level_weights=(1.0, 1.0, 1.0))


def _problem(pid, reference, samples, *, alpha=2.0, filter_time=0.001):
    """``reference`` is the timed levels only; level 0 gets a nominal time."""
    limit = alpha * max(max(level) for level in reference)
    return ProblemRecord(
        problem_id=pid,
        reference_times=((filter_time,), *reference),
        time_limit=limit,
        samples=samples,
    )


def _sample(index, times, *, correct=True):
    return SampleRecord(index=index, correct=correct, level_times=times)


def _record(problems, *, metric=ONE_LEVEL, failures=(), repeats=PAPER_REPEATS, cpu_count=8):
    return RunRecord(
        started="2026-01-01T00:00:00+00:00",
        finished="2026-01-01T00:01:00+00:00",
        environment=Environment(
            python="CPython 3.10.12",
            platform="test",
            machine="x86_64",
            cpu_count=cpu_count,
        ),
        metric=metric,
        repeats=repeats,
        aggregator="hodges_lehmann",
        data=PROV,
        data_fingerprint="d" * 64,
        solutions=PROV,
        solutions_fingerprint="s" * 64,
        problems=problems,
        failures=failures,
    )


def _one_model_record(**kwargs):
    """One problem, reference worst case 1.0 so T_i = 2.0, one sample at 1.0."""
    problem = _problem(0, ((1.0,),), {"m": (_sample(0, ((1.0,),)),)})
    return _record((problem,), **kwargs)


class SolutionSetTest(unittest.TestCase):
    def test_normalizes_keys_and_sorts_models(self):
        s = SolutionSet(PROV, {"b": {"3": ["x"]}, "a": {1: ("y",)}})
        self.assertEqual(s.models, ("a", "b"))
        self.assertEqual(s.problem_ids("b"), (3,))
        self.assertEqual(s.codes("b", 3), ("x",))

    def test_absent_problem_is_no_samples_not_an_error(self):
        s = SolutionSet(PROV, {"a": {1: ("y",)}})
        self.assertEqual(s.codes("a", 2), ())

    def test_rejects_malformed_sets(self):
        for samples, why in [
            ({}, "no models"),
            ({" ": {1: ("x",)}}, "empty model name"),
            ({"a": {}}, "no problems"),
            ({"a": {1: ()}}, "no samples"),
            ({"a": {-1: ("x",)}}, "negative id"),
            ({"a": {1: (b"x",)}}, "non-str code"),
        ]:
            with self.subTest(why=why), self.assertRaises(ValueError):
                SolutionSet(PROV, samples)

    def test_common_problem_ids_is_the_intersection(self):
        s = SolutionSet(PROV, {"a": {1: ("x",), 2: ("x",)}, "b": {2: ("x",), 3: ("x",)}})
        self.assertEqual(s.common_problem_ids(), (2,))

    def test_sample_counts_are_distinct_and_sorted(self):
        s = SolutionSet(PROV, {"a": {1: ("x",), 2: ("x", "y"), 3: ("x",)}})
        self.assertEqual(s.sample_counts("a"), (1, 2))

    def test_fingerprint_tracks_code_not_provenance(self):
        a = SolutionSet(PROV, {"m": {1: ("x",)}})
        b = SolutionSet(
            Provenance(name="other", url="u", license="MIT", retrieved="2026-01-01"),
            {"m": {1: ("x",)}},
        )
        c = SolutionSet(PROV, {"m": {1: ("x ",)}})
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), c.fingerprint())

    def test_json_round_trip(self):
        s = SolutionSet(PROV, {"m": {1: ("x", "y"), 7: ("z",)}})
        self.assertEqual(solution_set_from_json(solution_set_to_json(s)), s)

    def test_rejects_a_tampered_fingerprint(self):
        raw = json.loads(solution_set_to_json(SolutionSet(PROV, {"m": {1: ("x",)}})))
        raw["samples"]["m"]["1"] = ["edited"]
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            solution_set_from_json(json.dumps(raw))

    def test_rejects_an_unknown_schema_version(self):
        raw = json.loads(solution_set_to_json(SolutionSet(PROV, {"m": {1: ("x",)}})))
        raw["schema_version"] = SOLUTIONS_SCHEMA_VERSION + 1
        with self.assertRaisesRegex(ValueError, "schema version"):
            solution_set_from_json(json.dumps(raw))

    def test_load_solutions_round_trip_and_missing_file(self):
        s = SolutionSet(PROV, {"m": {1: ("x",)}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text(solution_set_to_json(s))
            self.assertEqual(load_solutions(path), s)
            with self.assertRaises(FileNotFoundError):
                load_solutions(Path(tmp) / "absent.json")

    def test_synthetic_solutions_cover_every_problem(self):
        problems = list(synthetic_problem_set())
        s = synthetic_solutions(problems)
        self.assertEqual(s.models, ("mixed-bag", "reference-copy"))
        for model in s.models:
            self.assertEqual(s.problem_ids(model), tuple(p.problem_id for p in problems))
        self.assertEqual(s.sample_counts("reference-copy"), (1,))
        self.assertEqual(s.sample_counts("mixed-bag"), (4,))


class SampleRecordTest(unittest.TestCase):
    def test_censored_when_any_case_did_not_finish(self):
        self.assertFalse(_sample(0, ((1.0, 2.0), (3.0,))).censored)
        self.assertTrue(_sample(0, ((1.0,), (TIMEOUT,))).censored)

    def test_rejects_malformed_samples(self):
        for kwargs, why in [
            ({"index": -1, "level_times": ((1.0,),)}, "negative index"),
            ({"index": 0, "level_times": ()}, "no levels"),
            ({"index": 0, "level_times": ((),)}, "no test cases"),
            ({"index": 0, "level_times": ((-1.0,),)}, "negative time"),
            ({"index": 0, "level_times": ((math.nan,),)}, "NaN time"),
        ]:
            with self.subTest(why=why), self.assertRaises(ValueError):
                SampleRecord(correct=True, **kwargs)

    def test_from_measurement_keeps_timed_levels_and_every_status(self):
        measurement = SolutionMeasurement(
            problem_id=3,
            correct=False,
            levels=(
                LevelMeasurement(level=0, status=OK, times=(0.01,)),
                LevelMeasurement(level=1, status=OK, times=(1.0, 2.0)),
                LevelMeasurement(level=2, status=WRONG_ANSWER),
                LevelMeasurement(level=3, status=SKIPPED),
            ),
            detail="level 2: wrong answer",
            verified_levels=(0, 1),
        )
        record = SampleRecord.from_measurement(4, measurement, 3)
        self.assertEqual(record.index, 4)
        self.assertFalse(record.correct)
        self.assertEqual(record.level_times[0], (1.0, 2.0))
        self.assertTrue(all(math.isinf(t) for t in record.level_times[1]))
        self.assertEqual(record.statuses, (OK, OK, WRONG_ANSWER, SKIPPED))
        self.assertEqual(record.verified_levels, (0, 1))
        self.assertTrue(record.censored)


class ProblemRecordTest(unittest.TestCase):
    def test_timed_reference_drops_the_correctness_filter(self):
        problem = _problem(0, ((1.0,), (2.0,)), {"m": (_sample(0, ((1.0,), (2.0,))),)})
        self.assertEqual(problem.reference_times, ((0.001,), (1.0,), (2.0,)))
        self.assertEqual(problem.timed_reference(), ((1.0,), (2.0,)))
        self.assertEqual(problem.n_timed_levels, 2)
        self.assertEqual(problem.models, ("m",))

    def test_rejects_a_censored_reference(self):
        with self.assertRaisesRegex(ValueError, "reference cannot be censored"):
            ProblemRecord(
                problem_id=0,
                reference_times=((0.001,), (TIMEOUT,)),
                time_limit=2.0,
                samples={"m": (_sample(0, ((1.0,),)),)},
            )

    def test_needs_a_filter_level_and_a_timed_level(self):
        with self.assertRaisesRegex(ValueError, "level 0"):
            ProblemRecord(
                problem_id=0,
                reference_times=((1.0,),),
                time_limit=2.0,
                samples={"m": (_sample(0, ((1.0,),)),)},
            )

    def test_rejects_a_sample_with_the_wrong_level_count(self):
        with self.assertRaisesRegex(ValueError, "timed levels"):
            _problem(0, ((1.0,),), {"m": (_sample(0, ((1.0,), (1.0,))),)})

    def test_rejects_an_unusable_time_limit(self):
        for limit in (0.0, -1.0, math.inf):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "time limit"):
                ProblemRecord(
                    problem_id=0,
                    reference_times=((0.001,), (1.0,)),
                    time_limit=limit,
                    samples={"m": (_sample(0, ((1.0,),)),)},
                )

    def test_rejects_a_model_with_no_samples(self):
        with self.assertRaisesRegex(ValueError, "no samples"):
            _problem(0, ((1.0,),), {"m": ()})


class RunRecordValidationTest(unittest.TestCase):
    def test_sorts_problems_by_id(self):
        samples = {"m": (_sample(0, ((1.0,),)),)}
        record = _record((_problem(5, ((1.0,),), samples), _problem(2, ((1.0,),), samples)))
        self.assertEqual(record.ids(), (2, 5))
        self.assertEqual([p.problem_id for p in record], [2, 5])
        self.assertEqual(len(record), 2)
        self.assertEqual(record[5].problem_id, 5)
        with self.assertRaises(KeyError):
            record[7]

    def test_rejects_duplicate_problem_ids(self):
        samples = {"m": (_sample(0, ((1.0,),)),)}
        with self.assertRaisesRegex(ValueError, "duplicate problem ids: \\[1\\]"):
            _record((_problem(1, ((1.0,),), samples), _problem(1, ((2.0,),), samples)))

    def test_needs_a_problem_or_a_failure(self):
        with self.assertRaisesRegex(ValueError, "at least one problem"):
            _record(())
        self.assertEqual(_record((), failures=((3, "boom"),)).ids(), ())

    def test_rejects_repeats_below_one(self):
        with self.assertRaisesRegex(ValueError, "repeats"):
            _one_model_record(repeats=0)

    def test_rejects_a_level_count_the_metric_does_not_declare(self):
        problem = _problem(0, ((1.0,), (1.0,)), {"m": (_sample(0, ((1.0,), (1.0,))),)})
        with self.assertRaisesRegex(ValueError, "timed levels"):
            _record((problem,), metric=ONE_LEVEL)

    def test_rejects_a_stored_limit_that_is_not_alpha_times_the_worst_case(self):
        problem = ProblemRecord(
            problem_id=0,
            reference_times=((0.001,), (1.0,)),
            time_limit=3.0,
            samples={"m": (_sample(0, ((1.0,),)),)},
        )
        with self.assertRaisesRegex(ValueError, "not alpha"):
            _record((problem,), metric=ONE_LEVEL)


class RunRecordCoverageTest(unittest.TestCase):
    def setUp(self):
        one = _sample(0, ((1.0,),))
        self.record = _record(
            (
                _problem(1, ((1.0,),), {"a": (one,), "b": (one,)}),
                _problem(2, ((1.0,),), {"a": (one,)}),
                _problem(3, ((1.0,),), {"b": (one, _sample(1, ((1.0,),)))}),
            )
        )

    def test_models_and_coverage(self):
        self.assertEqual(self.record.models, ("a", "b"))
        self.assertEqual(self.record.covered_ids("a"), (1, 2))
        self.assertEqual(self.record.covered_ids("b"), (1, 3))
        self.assertEqual(self.record.covered_ids("absent"), ())

    def test_aligned_ids_is_the_intersection(self):
        self.assertEqual(self.record.aligned_ids(), (1,))
        self.assertEqual(self.record.aligned_ids(["a"]), (1, 2))
        self.assertEqual(self.record.aligned_ids([]), ())
        self.assertEqual(self.record.aligned_ids(["a", "absent"]), ())

    def test_sample_counts_expose_a_varying_n(self):
        self.assertEqual(self.record.sample_counts("a"), (1,))
        self.assertEqual(self.record.sample_counts("b"), (1, 2))

    def test_scoring_a_model_outside_its_coverage_is_an_error(self):
        with self.assertRaisesRegex(KeyError, "problems \\[3\\]"):
            self.record.per_problem_eff("a", 1, ids=[1, 3])
        with self.assertRaises(KeyError):
            self.record.per_problem_eff("absent")
        with self.assertRaises(KeyError):
            self.record.sample_scores("a", 3)


class RunRecordScoringTest(unittest.TestCase):
    """Reference worst case 1.0 at alpha 2 gives T = 2 and a denominator of 1."""

    def setUp(self):
        samples = (
            _sample(0, ((1.0,),)),                  # the reference pace, f = 1
            _sample(1, ((1.5,),)),                  # half the slack used, f = 0.5
            _sample(2, ((0.5,),)),                  # faster than the reference, f = 1.5
            _sample(3, ((TIMEOUT,),)),              # censored, f = 0
            _sample(4, ((0.5,),), correct=False),   # fast and wrong, e = 0
        )
        self.problem = _problem(0, ((1.0,),), {"m": samples})
        self.record = _record((self.problem,))

    def test_sample_scores_match_the_metric_on_the_recorded_times(self):
        scores = self.record.sample_scores("m", 0)
        self.assertEqual(scores, (1.0, 0.5, 1.5, 0.0, 0.0))
        direct = tuple(
            sample_score(r.level_times, self.problem.timed_reference(), ONE_LEVEL,
                         correct=r.correct)
            for r in self.problem.samples["m"]
        )
        self.assertEqual(scores, direct)

    def test_eff_at_k_is_the_order_statistic_estimator_over_those_scores(self):
        for k in (1, 2, 5):
            with self.subTest(k=k):
                expected = raw_eff_at_k((1.0, 0.5, 1.5, 0.0, 0.0), k)
                self.assertAlmostEqual(self.record.eff_at_k("m", k), expected)
                self.assertEqual(self.record.per_problem_eff("m", k), (expected,))

    def test_pass_at_k_counts_correctness_only(self):
        self.assertAlmostEqual(self.record.pass_at_k("m", 1), 0.8)
        self.assertEqual(self.record.censored_samples("m"), 1)
        self.assertEqual(self.record.incorrect_samples("m"), 1)

    def test_lowering_alpha_rescores_from_the_same_times(self):
        # T = 1.5, denominator 0.5: the 1.5s sample now sits exactly at the limit.
        self.assertEqual(
            self.record.sample_scores("m", 0, alpha=1.5), (1.0, 0.0, 2.0, 0.0, 0.0)
        )

    def test_raising_alpha_is_refused_while_a_sample_is_censored(self):
        with self.assertRaisesRegex(ValueError, "cannot raise alpha"):
            self.record.sample_scores("m", 0, alpha=3.0)
        clean = _record((_problem(0, ((1.0,),), {"m": (_sample(0, ((1.0,),)),)}),))
        self.assertEqual(clean.sample_scores("m", 0, alpha=3.0), (1.0,))

    def test_per_level_normalization_cannot_be_rescored_at_another_alpha(self):
        variant = MetricConfig(alpha=2.0, level_weights=(1.0,), normalization="per_level")
        record = _record((self.problem,), metric=variant)
        self.assertEqual(record.sample_scores("m", 0), (1.0, 0.5, 1.5, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "per_level"):
            record.sample_scores("m", 0, alpha=1.5)

    def test_level_weights_can_be_overridden_without_touching_alpha(self):
        problem = _problem(0, ((1.0,), (1.0,)), {"m": (_sample(0, ((1.0,), (1.5,))),)})
        record = _record((problem,), metric=MetricConfig(2.0, (1.0, 1.0)))
        self.assertEqual(record.sample_scores("m", 0), (0.75,))
        self.assertEqual(record.sample_scores("m", 0, level_weights=(1.0, 0.0)), (1.0,))
        self.assertEqual(record.sample_scores("m", 0, level_weights=(0.0, 1.0)), (0.5,))


class LevelMeansTest(unittest.TestCase):
    """``level_means`` is scored with one-hot weights, so the h identity holds."""

    def setUp(self):
        reference = ((1.0,), (1.0,), (1.0,))
        self.record = _record(
            (
                _problem(1, reference, {"m": (
                    _sample(0, ((1.0,), (1.5,), (TIMEOUT,))),
                    _sample(1, ((0.5,), (1.0,), (1.0,))),
                )}),
                _problem(2, reference, {"m": (
                    _sample(0, ((1.0,), (1.0,), (1.0,))),
                    _sample(1, ((2.0,), (2.0,), (2.0,)), correct=False),
                )}),
            ),
            metric=PAPER,
        )

    def test_means_are_per_level_averages_of_the_level_fractions(self):
        # Problem 1 level 1: f = 1.0 and 1.5, mean 1.25. Problem 2 level 1: f = 1.0
        # and 0, since the wrong sample scores 0 on every level. Mean of means 0.875.
        self.assertEqual(self.record.level_means("m"), (0.875, 0.625, 0.5))

    def test_eff_at_h_over_the_means_reproduces_eff_at_1(self):
        means = self.record.level_means("m")
        self.assertAlmostEqual(
            eff_at_h(means, PAPER.level_weights), self.record.eff_at_k("m", 1)
        )

    def test_means_respect_an_id_subset(self):
        self.assertEqual(self.record.level_means("m", ids=[2]), (0.5, 0.5, 0.5))


class CaveatsTest(unittest.TestCase):
    def test_a_clean_run_has_nothing_to_declare(self):
        self.assertEqual(_one_model_record().caveats(), ())

    def test_reports_thin_hardware_and_load(self):
        env = Environment(
            python="CPython 3.10.12",
            platform="test",
            machine="x86_64",
            cpu_count=2,
            load_average=(3.0, 1.0, 1.0),
        )
        caveats = env.caveats()
        self.assertEqual(len(caveats), 2)
        self.assertIn("2 cores", caveats[0])
        self.assertIn("load average 3.0", caveats[1])

    def test_reports_few_repeats_failures_and_a_varying_n(self):
        one = _sample(0, ((1.0,),))
        record = _record(
            (
                _problem(1, ((1.0,),), {"m": (one,)}),
                _problem(2, ((1.0,),), {"m": (one, _sample(1, ((1.0,),)))}),
            ),
            repeats=1,
            failures=((9, "reference crashed"),),
        )
        caveats = record.caveats()
        self.assertIn(f"below the paper's R = {PAPER_REPEATS}", caveats[0])
        self.assertIn("no reference measurement", caveats[1])
        self.assertIn("n varies", caveats[2])

    def test_capture_reads_the_running_machine(self):
        env = Environment.capture()
        self.assertIn("Python", env.python)
        self.assertGreater(env.cpu_count, 0)


class RecordCodecTest(unittest.TestCase):
    def setUp(self):
        self.record = _record(
            (
                _problem(1, ((1.0, 2.0),), {
                    "a": (_sample(0, ((1.0,),)), _sample(1, ((TIMEOUT,),), correct=False)),
                    "b": (_sample(0, ((3.0,),)),),
                }),
                _problem(2, ((1.0,),), {"a": (_sample(0, ((0.5,),)),)}),
            ),
            failures=((9, "reference crashed"),),
        )

    def test_round_trip_is_exact(self):
        self.assertEqual(record_from_json(record_to_json(self.record)), self.record)

    def test_censoring_travels_as_a_token_not_as_a_number(self):
        raw = json.loads(record_to_json(self.record))
        times = raw["problems"][0]["samples"]["a"][1]["level_times"]
        self.assertEqual(times, [[CENSORED_TOKEN]])
        self.assertNotIn("Infinity", record_to_json(self.record))

    def test_failures_and_schema_version_survive(self):
        parsed = record_from_json(record_to_json(self.record))
        self.assertEqual(parsed.failures, ((9, "reference crashed"),))
        self.assertEqual(parsed.schema_version, RECORD_SCHEMA_VERSION)

    def test_rejects_an_unknown_schema_version(self):
        raw = json.loads(record_to_json(self.record))
        raw["schema_version"] = RECORD_SCHEMA_VERSION + 1
        with self.assertRaisesRegex(ValueError, "schema version"):
            record_from_json(json.dumps(raw))

    def test_a_file_that_disagrees_with_itself_fails_on_load(self):
        raw = json.loads(record_to_json(self.record))
        raw["problems"][0]["time_limit"] = 99.0
        with self.assertRaisesRegex(ValueError, "not alpha"):
            record_from_json(json.dumps(raw))

    def test_rejects_a_stored_time_that_is_not_a_time(self):
        raw = json.loads(record_to_json(self.record))
        raw["problems"][0]["samples"]["b"][0]["level_times"] = [["fast"]]
        with self.assertRaisesRegex(ValueError, "expected a time"):
            record_from_json(json.dumps(raw))

    def test_save_and_load_a_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_record(self.record, Path(tmp) / "nested" / "run.json")
            self.assertTrue(path.is_file())
            self.assertEqual(load_record(path), self.record)
            with self.assertRaises(FileNotFoundError):
                load_record(Path(tmp) / "absent.json")


class SelectedIdsTest(unittest.TestCase):
    def setUp(self):
        self.problems = synthetic_problem_set()  # ids 0, 1, 2
        self.solutions = SolutionSet(
            PROV, {"a": {0: ("x",), 9: ("x",)}, "b": {1: ("x",)}}
        )

    def test_takes_problems_in_the_data_that_someone_answered(self):
        self.assertEqual(selected_ids(self.problems, self.solutions), (0, 1))
        self.assertEqual(selected_ids(self.problems, self.solutions, ["a"]), (0,))

    def test_narrows_to_a_requested_subset(self):
        self.assertEqual(selected_ids(self.problems, self.solutions, None, [1, 2]), (1,))

    def test_rejects_ids_outside_the_problem_set(self):
        with self.assertRaisesRegex(KeyError, "problems \\[7\\]"):
            selected_ids(self.problems, self.solutions, None, [1, 7])

    def test_rejects_models_with_no_samples(self):
        with self.assertRaisesRegex(KeyError, "no samples for models"):
            selected_ids(self.problems, self.solutions, ["nobody"])


def _fake_reference(problem_id, per_level=1.0, n_levels=3):
    times = ((0.001,), *((per_level,) for _ in range(n_levels)))
    return ReferenceMeasurement(
        problem_id=problem_id,
        times=times,
        outputs=tuple(((),) for _ in times),
        time_limit=PAPER.alpha * per_level,
    )


def _fake_solution(problem_id, per_level, *, correct=True, n_levels=3):
    levels = [LevelMeasurement(level=0, status=OK, times=(0.001,))]
    for index in range(1, n_levels + 1):
        levels.append(LevelMeasurement(level=index, status=OK, times=(per_level,)))
    return SolutionMeasurement(
        problem_id=problem_id,
        correct=correct,
        levels=tuple(levels),
        verified_levels=tuple(range(n_levels + 1)),
    )


class OrchestrateTest(unittest.TestCase):
    """The measuring is faked here; what is under test is the loop around it."""

    def setUp(self):
        self.problems = synthetic_problem_set()
        self.solutions = SolutionSet(
            PROV,
            {
                "fast": {0: ("a", "b"), 1: ("a",)},
                "slow": {0: ("c",), 2: ("c",)},
            },
        )
        self.paces = {"a": 0.5, "b": 1.0, "c": 1.5}

    def _run(self, **kwargs):
        seen = []

        def reference(problem, config, metric):
            seen.append(("reference", problem.problem_id))
            return _fake_reference(problem.problem_id)

        def solution(problem, code, ref, config):
            seen.append((code, problem.problem_id))
            return _fake_solution(problem.problem_id, self.paces[code])

        with mock.patch.object(orchestrate, "measure_reference", reference), \
             mock.patch.object(orchestrate, "evaluate_solution", solution):
            record = run_evaluation(self.problems, self.solutions, **kwargs)
        return record, seen

    def test_measures_each_reference_once_for_every_model(self):
        record, seen = self._run()
        self.assertEqual([pid for what, pid in seen if what == "reference"], [0, 1, 2])
        self.assertEqual(record.ids(), (0, 1, 2))
        self.assertEqual(record.models, ("fast", "slow"))
        self.assertEqual(record[0].models, ("fast", "slow"))
        self.assertEqual(record[1].models, ("fast",))
        self.assertEqual(record[2].models, ("slow",))

    def test_every_model_is_scored_against_that_one_measurement(self):
        record, _ = self._run()
        self.assertEqual(record[0].reference_times, ((0.001,), (1.0,), (1.0,), (1.0,)))
        self.assertEqual(record[0].time_limit, 2.0)
        # f = (2 - t)/(2 - 1) with the fake paces: 0.5 -> 1.5, 1.0 -> 1.0, 1.5 -> 0.5.
        self.assertEqual(record.sample_scores("fast", 0), (1.5, 1.0))
        self.assertEqual(record.sample_scores("slow", 0), (0.5,))

    def test_samples_are_interleaved_by_index_across_models(self):
        _, seen = self._run()
        on_zero = [what for what, pid in seen if pid == 0]
        self.assertEqual(on_zero, ["reference", "a", "c", "b"])

    def test_carries_the_configuration_and_provenance_of_its_inputs(self):
        record, _ = self._run(config=RunConfig(repeats=3), metric=FLAT_H)
        self.assertEqual(record.repeats, 3)
        self.assertEqual(record.aggregator, RunConfig().aggregator)
        self.assertEqual(record.metric, FLAT_H)
        self.assertEqual(record.data_fingerprint, self.problems.fingerprint())
        self.assertEqual(record.solutions_fingerprint, self.solutions.fingerprint())
        self.assertEqual(record.solutions, PROV)
        self.assertLessEqual(record.started, record.finished)

    def test_narrows_to_the_models_and_ids_it_was_given(self):
        record, seen = self._run(models=["fast"], ids=[0])
        self.assertEqual(record.ids(), (0,))
        self.assertEqual(record.models, ("fast",))
        self.assertEqual([what for what, _ in seen], ["reference", "a", "b"])

    def test_reports_progress_per_problem(self):
        lines = []
        self._run(on_progress=lines.append)
        self.assertEqual(lines[0], "problem 0: reference")
        self.assertEqual(lines[1], "problem 0: 3 samples across 2 models")

    def test_rejects_an_empty_or_unknown_model_selection(self):
        with self.assertRaisesRegex(ValueError, "no models"):
            self._run(models=[])
        with self.assertRaisesRegex(KeyError, "no samples for models"):
            self._run(models=["nobody"])

    def test_a_reference_that_does_not_run_stops_the_run(self):
        def reference(problem, config, metric):
            if problem.problem_id == 1:
                raise SandboxError("reference crashed")
            return _fake_reference(problem.problem_id)

        with mock.patch.object(orchestrate, "measure_reference", reference), \
             mock.patch.object(
                 orchestrate,
                 "evaluate_solution",
                 lambda problem, code, ref, config: _fake_solution(problem.problem_id, 1.0),
             ):
            with self.assertRaises(SandboxError):
                run_evaluation(self.problems, self.solutions)
            record = run_evaluation(self.problems, self.solutions, keep_going=True)

        self.assertEqual(record.ids(), (0, 2))
        self.assertEqual(record.failures, ((1, "reference crashed"),))
        self.assertEqual(record.covered_ids("fast"), (0,))
        self.assertIn("no reference measurement", " ".join(record.caveats()))


def _report_record(models=("fast", "slow"), *, metric=PAPER, n_problems=4):
    """A record with enough shape for every report section to have something to say."""
    reference = ((0.4,), (0.7,), (1.0,))
    paces = {"fast": 0.8, "slow": 1.4}
    problems = []
    for pid in range(n_problems):
        samples = {}
        for model in models:
            scaled = tuple((paces.get(model, 1.0) * level[0],) for level in reference)
            timed_out = model == "slow" and pid % 2 == 0
            second = (
                _sample(1, ((TIMEOUT,),) * len(reference), correct=False)
                if timed_out
                else _sample(1, scaled)
            )
            samples[model] = (_sample(0, scaled), second)
        problems.append(_problem(pid, reference, samples, alpha=metric.alpha))
    return _record(tuple(problems), metric=metric)


class SummaryTest(unittest.TestCase):
    def test_every_section_appears_for_a_two_model_run(self):
        text = format_summary(_report_record(), resamples=200)
        for heading in (
            "ENAMEL-Extended run",
            "Leaderboard",
            "Level discrimination",
            "Time limit sensitivity",
            "Hardness weights",
            "Paired comparisons",
        ):
            self.assertIn(heading, text)
        self.assertIn("fast - slow", text)
        self.assertIn("Kendall tau-b", text)
        self.assertTrue(text.endswith("\n"))

    def test_a_single_model_has_nothing_to_compare(self):
        text = format_summary(_report_record(models=("fast",)), resamples=200)
        self.assertIn("Leaderboard", text)
        self.assertNotIn("Paired comparisons", text)

    def test_a_run_with_no_scored_problem_says_so(self):
        text = format_summary(_record((), failures=((1, "boom"),)))
        self.assertIn("Nothing was scored.", text)
        self.assertIn("no reference measured for problems: 1", text)

    def test_the_alpha_sweep_is_dropped_when_one_alpha_is_usable(self):
        record = _report_record(metric=MetricConfig(1.25, PAPER.level_weights))
        text = format_summary(record, resamples=200)
        self.assertNotIn("Time limit sensitivity", text)

    def test_caveats_reach_the_header(self):
        text = format_summary(_report_record(), resamples=200)
        self.assertNotIn("caveats:", text)
        thin = _record(_report_record().problems, metric=PAPER, repeats=1, cpu_count=2)
        header = format_summary(thin, resamples=200)
        self.assertIn("caveats:", header)
        self.assertIn("below the paper's R", header)
        self.assertIn("2 cores", header)

    def test_a_zero_weight_leaves_the_h_column_unfilled(self):
        record = _report_record(metric=MetricConfig(2.0, (3.0, 3.0, 0.0)))
        text = format_summary(record, resamples=200)
        self.assertIn("Hardness weights", text)
        self.assertIn("n/a", text)

    def test_models_that_share_no_problem_cannot_be_paired(self):
        one = _sample(0, ((1.0,),))
        record = _record(
            (
                _problem(1, ((1.0,),), {"a": (one,)}),
                _problem(2, ((1.0,),), {"b": (one,)}),
            ),
            metric=ONE_LEVEL,
        )
        text = format_summary(record, resamples=200)
        self.assertIn("models share no problems", text)
        self.assertIn("a - b", text)


def _script():
    """scripts/evaluate.py loaded by path, since scripts/ is not a package."""
    location = REPO_ROOT / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _script()

    def _main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.script.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_a_real_run_measures_saves_and_reports(self):
        """The one test that actually times code, so it stays small: R = 1, one problem."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            code, out, err = self._main(
                ["run", "--repeats", "1", "--limit", "1", "--resamples", "50",
                 "--out", str(path)]
            )
            self.assertEqual(code, self.script.OK)
            self.assertIn("Leaderboard", out)
            self.assertIn("reference-copy", out)
            self.assertIn("problem 0: reference", err)
            self.assertIn(f"record written to {path}", err)
            record = load_record(path)
        self.assertEqual(record.ids(), (0,))
        self.assertEqual(record.models, ("mixed-bag", "reference-copy"))
        self.assertEqual(record.repeats, 1)
        self.assertEqual(record.metric, PAPER)
        self.assertEqual(record.incorrect_samples("reference-copy"), 0)

    def test_report_rereads_a_record_without_measuring(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_record(_report_record(), Path(tmp) / "run.json")
            code, out, _ = self._main(["report", str(path), "--resamples", "50"])
            self.assertEqual(code, self.script.OK)
            self.assertIn("Leaderboard", out)
            self.assertIn("eff@1", out)
            code, out, _ = self._main(
                ["report", str(path), "--resamples", "50", "--k", "2", "--level", "0.8"]
            )
            self.assertEqual(code, self.script.OK)
            self.assertIn("eff@2", out)
            self.assertIn("80% CI", out)

    def test_usage_errors_do_not_measure_anything(self):
        quiet = ["--no-save", "--quiet"]
        cases = {
            "missing record": ["report", "/nonexistent/run.json"],
            "problems alone": ["run", "--problems", "p.json", *quiet],
            "solutions alone": ["run", "--solutions", "s.json", *quiet],
            "unparsable ids": ["run", "--ids", "1,two", *quiet],
            "unknown id": ["run", "--ids", "999", *quiet],
            "unknown model": ["run", "--models", "nobody", *quiet],
            "unparsable hardness": ["run", "--hardness", "3,3,x", *quiet],
        }
        for why, argv in cases.items():
            with self.subTest(why=why):
                code, out, err = self._main(argv)
                self.assertEqual(code, self.script.USAGE_FAILURE)
                self.assertEqual(out, "")
                self.assertTrue(err.startswith("error: "), err)

    def test_a_run_that_cannot_measure_returns_the_run_failure_code(self):
        def raiser(*args, **kwargs):
            raise SandboxError("the reference did not run")

        with mock.patch.object(self.script, "run_evaluation", raiser):
            code, out, err = self._main(["run", "--no-save", "--quiet"])
        self.assertEqual(code, self.script.RUN_FAILURE)
        self.assertEqual(out, "")
        self.assertIn("did not finish", err)

    def test_the_default_record_path_is_timestamped_under_runs(self):
        path = self.script._default_out()
        self.assertEqual(path.parent, Path("runs"))
        self.assertTrue(path.name.startswith("run-"))
        self.assertEqual(path.suffix, ".json")
