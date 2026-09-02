"""Tests for value transport, the sandbox, and the level runner."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from enamel_ext.data import GeneratedLevel, MaterializedLevel, Problem
from enamel_ext.measure import (
    CRASHED,
    ERROR,
    OK,
    SKIPPED,
    TIMEOUT,
    WRONG_ANSWER,
    LevelMeasurement,
    Limits,
    ReferenceMeasurement,
    RunConfig,
    SandboxError,
    SolutionMeasurement,
    brief,
    decode,
    encode,
    evaluate_problem,
    evaluate_solution,
    measure_reference,
    run_level,
    score_solution,
    values_equal,
)
from enamel_ext.measure.sandbox import _effective_limits, _exit_failure
from enamel_ext.metrics.effk import eff_at_k, pass_at_k
from enamel_ext.metrics.score import MetricConfig

GEN = "def make_input(seed, scale):\n    return (list(range(seed, seed + scale)),)\n"
REF = "def total(xs):\n    return sum(xs)\n"

#: alpha = 10 instead of the paper's 2, so a noisy machine cannot censor a
#: solution that is genuinely as fast as the reference.
LOOSE = MetricConfig(alpha=10.0, level_weights=(3.0, 3.0, 4.0))


def _problem(reference: str = REF, **kw) -> Problem:
    defaults = dict(
        problem_id=0,
        entry_point="total",
        prompt="",
        reference_solution=reference,
        input_generator=GEN,
        levels=(
            GeneratedLevel(level=0, scale=8, seeds=(1, 2)),
            GeneratedLevel(level=1, scale=2000, seeds=(3,)),
            GeneratedLevel(level=2, scale=8000, seeds=(4,)),
            GeneratedLevel(level=3, scale=32000, seeds=(5,)),
        ),
    )
    defaults.update(kw)
    return Problem(**defaults)  # type: ignore[arg-type]


class TestValueTransport(unittest.TestCase):
    def test_json_would_lose_the_container_type(self):
        """A tuple and a list are different answers, so the tag is load-bearing."""
        self.assertEqual(decode(encode((1, 2))), (1, 2))
        self.assertEqual(decode(encode([1, 2])), [1, 2])
        self.assertFalse(values_equal((1, 2), [1, 2]))

    def test_round_trips_nested_containers(self):
        value = {"a": (1.5, [2, {"b": frozenset({3})}]), "c": b"\x00\xff"}
        self.assertEqual(decode(encode(value)), value)

    def test_floats_survive_exactly(self):
        for x in (0.1, 1e308, -0.0, math.inf, -math.inf):
            with self.subTest(x=x):
                self.assertEqual(repr(decode(encode(x))), repr(x))
        self.assertTrue(math.isnan(decode(encode(math.nan))))

    def test_unrepresentable_values_fall_back_to_their_repr(self):
        class Marker:
            def __init__(self, tag):
                self.tag = tag

            def __repr__(self):
                return f"Marker({self.tag})"

        self.assertEqual(decode(encode(Marker(1))), decode(encode(Marker(1))))
        self.assertNotEqual(decode(encode(Marker(1))), decode(encode(Marker(2))))
        self.assertNotEqual(decode(encode(Marker(1))), "Marker(1)")

    def test_float_comparison_is_tolerant_but_not_blind(self):
        self.assertTrue(values_equal(0.1 + 0.2, 0.3))
        self.assertFalse(values_equal(1.0, 1.1))
        self.assertTrue(values_equal([1.0, [2.0]], [1.0 + 1e-12, [2.0]]))

    def test_the_tolerance_is_absolute_not_relative(self):
        """HumanEval's own checks are ``abs(a - b) < 1e-6``. A *relative* 1e-6
        would accept an absolute error of 500 at magnitude 1e9."""
        self.assertFalse(values_equal(1e9, 1e9 + 500.0))
        self.assertTrue(values_equal(1e9, 1e9 + 1e-7))

    def test_an_int_and_a_float_are_held_to_the_same_tolerance(self):
        self.assertFalse(values_equal(10 ** 7, 10 ** 7 + 5.0))
        self.assertFalse(values_equal(10 ** 7, 10 ** 7 + 5))

    def test_huge_integers_survive_the_json_digit_cap(self):
        """CPython refuses to render a 4301-digit int as decimal, and JSON is
        decimal, so wide ints have to travel in another base."""
        value = 7 ** 20000
        self.assertGreater(len(f"{value:x}"), 4300)
        self.assertEqual(decode(json.loads(json.dumps(encode(value)))), value)
        self.assertEqual(decode(encode([value, -value])), [value, -value])

    def test_exact_comparison_is_available(self):
        self.assertFalse(values_equal(0.1 + 0.2, 0.3, rel_tol=0.0, abs_tol=0.0))

    def test_two_nans_are_not_a_wrong_answer(self):
        self.assertTrue(values_equal(math.nan, math.nan))
        self.assertFalse(values_equal(math.nan, 1.0))

    def test_diagnostics_are_bounded_and_never_raise(self):
        """A wrong answer on a 32000-element input would otherwise carry a copy of
        that input into the results, and a wide int cannot be rendered at all."""
        self.assertLess(len(brief(list(range(32000)))), 260)
        self.assertTrue(brief(list(range(32000))).endswith("..."))
        self.assertEqual(brief(7 ** 20000)[:2], "0x")
        self.assertEqual(brief([1, 2]), "[1, 2]")

    def test_comparison_keeps_python_equality_semantics(self):
        """Only container types are distinguished; ``==`` decides the rest."""
        self.assertTrue(values_equal(True, 1))
        self.assertTrue(values_equal([True], [1]))
        self.assertFalse(values_equal((1,), [1]))


class TestSandbox(unittest.TestCase):
    def test_materialized_inputs_and_repeat_count(self):
        result = run_level(REF, "total", inputs=[([1, 2, 3],), ([],)], repeats=3)
        self.assertEqual(result.status, OK)
        self.assertEqual([c.status for c in result.cases], [OK, OK])
        self.assertEqual(result.cases[0].output, 6)
        self.assertEqual(len(result.cases[0].times), 3)

    def test_generated_inputs_match_the_level_spec(self):
        result = run_level(REF, "total", generator=GEN, scale=100, seeds=[1, 2], repeats=2)
        self.assertEqual(result.status, OK)
        self.assertEqual(result.cases[0].output, sum(range(1, 101)))

    def test_each_repeat_gets_a_fresh_copy_of_the_input(self):
        """A solution that mutates its argument would otherwise make every repeat
        after the first measure different data."""
        code = (
            "def total(xs):\n"
            "    assert not xs or xs[0] != -1, 'input reused across repeats'\n"
            "    xs[0] = -1\n"
            "    return 0\n"
        )
        result = run_level(code, "total", inputs=[([1, 2],)], repeats=4)
        self.assertEqual(result.status, OK)
        self.assertEqual(len(result.cases[0].times), 4)

    def test_the_first_calls_output_is_not_overwritten_by_a_later_repeat(self):
        """The answer has to be captured at the first call, not after the last:
        anything reachable from module state can change in between."""
        code = "ACC = []\ndef total(xs):\n    ACC.append(len(xs))\n    return ACC\n"
        result = run_level(code, "total", inputs=[([1, 2],)], repeats=3)
        self.assertEqual(result.cases[0].output, [2])

    def test_the_clock_cannot_be_patched_by_the_code_under_test(self):
        """A solution that rebinds ``time.perf_counter`` would report its own
        runtime and pass any limit."""
        code = (
            "import time\n"
            "time.perf_counter = lambda: 0.0\n"
            "def total(xs):\n"
            "    return sum(xs)\n"
        )
        result = run_level(code, "total", inputs=[(list(range(100000)),)], repeats=2)
        self.assertEqual(result.status, OK)
        self.assertTrue(all(t > 0 for t in result.cases[0].times))

    def test_hash_order_is_stable_between_children(self):
        """Expected outputs are captured in one process and compared in another,
        so an unpinned hash seed would report correct solutions as wrong."""
        code = "def total(xs):\n    return list({'alpha', 'beta', 'gamma', 'delta', 'eps'})\n"
        first = run_level(code, "total", inputs=[([1],)])
        second = run_level(code, "total", inputs=[([1],)])
        self.assertEqual(first.cases[0].output, second.cases[0].output)

    def test_materialized_inputs_keep_their_python_types(self):
        """Arguments cross the boundary the way outputs do; plain JSON would turn
        a tuple into a list and an int dict key into a string."""
        code = "def probe(a, b, c):\n    return [type(a).__name__, sorted(b), c[1]]\n"
        result = run_level(code, "probe", inputs=[((1, 2), b"\x00\xff", {1: "x"})])
        self.assertEqual(result.status, OK)
        self.assertEqual(result.cases[0].output, ["tuple", [0, 255], "x"])

    def test_a_missing_entry_point_is_an_error_not_a_crash(self):
        result = run_level("def other(x):\n    return x\n", "total", inputs=[([1],)])
        self.assertEqual(result.status, ERROR)
        self.assertIn("total", result.detail)

    def test_an_exception_is_reported_per_case_and_on_the_level(self):
        result = run_level("def total(xs):\n    return 1 / 0\n", "total", inputs=[([1],)])
        self.assertEqual(result.status, ERROR)
        self.assertIn("ZeroDivisionError", result.cases[-1].detail)
        self.assertIn("ZeroDivisionError", result.detail)

    def test_over_limit_stops_the_level(self):
        code = "import time\ndef total(xs):\n    time.sleep(0.2)\n    return 0\n"
        result = run_level(code, "total", inputs=[([1],), ([2],)], repeats=4, time_limit=0.01)
        self.assertEqual(result.status, TIMEOUT)
        self.assertEqual(len(result.cases), 1)
        self.assertEqual(result.cases[0].status, TIMEOUT)

    def test_one_slow_repeat_does_not_censor_a_case_that_fits_on_aggregate(self):
        """The score compares the aggregate of the repeats against ``T_i``, so
        censoring per repeat would reject solutions the score would have kept."""
        code = (
            "import time\n"
            "CALLS = []\n"
            "def total(xs):\n"
            "    CALLS.append(1)\n"
            "    if len(CALLS) == 1:\n"
            "        time.sleep(0.03)\n"
            "    return sum(xs)\n"
        )
        result = run_level(code, "total", inputs=[([1, 2],)], repeats=6, time_limit=0.02)
        self.assertEqual(result.status, OK)
        self.assertEqual(len(result.cases[0].times), 6)
        self.assertGreater(result.cases[0].times[0], 0.02)

    def test_a_timed_out_case_still_reports_the_answer_it_computed(self):
        """Correctness is judged on the first call's output, so it has to survive
        a case that later runs out of time."""
        code = "import time\ndef total(xs):\n    time.sleep(0.05)\n    return 12345\n"
        result = run_level(code, "total", inputs=[([1],)], repeats=2, time_limit=0.001)
        self.assertEqual(result.status, TIMEOUT)
        self.assertTrue(result.cases[0].has_output)
        self.assertEqual(result.cases[0].output, 12345)

    def test_a_hang_is_killed_by_wall_clock(self):
        code = "def total(xs):\n    while True:\n        pass\n"
        result = run_level(code, "total", inputs=[([1],)], wall_timeout=2.0)
        self.assertEqual(result.status, TIMEOUT)
        self.assertIn("wall clock", result.detail)

    def test_the_kill_reaches_processes_the_solution_spawned(self):
        """A forked grandchild would otherwise outlive its own timeout."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "survived")
            code = (
                "import os, time\n"
                "def total(xs):\n"
                "    if os.fork() == 0:\n"
                "        time.sleep(1.5)\n"
                f"        open({marker!r}, 'w').close()\n"
                "        os._exit(0)\n"
                "    while True:\n"
                "        pass\n"
            )
            result = run_level(code, "total", inputs=[([1],)], wall_timeout=0.5)
            self.assertEqual(result.status, TIMEOUT)
            time.sleep(2.0)
            self.assertFalse(os.path.exists(marker))

    def test_a_process_left_behind_does_not_look_like_a_timeout(self):
        """Waiting on the child's stderr rather than on its exit reports a bogus
        timeout whenever a solution leaves a helper holding the handle open."""
        code = (
            "import os, time\n"
            "def total(xs):\n"
            "    if os.fork() == 0:\n"
            "        time.sleep(5.0)\n"
            "        os._exit(0)\n"
            "    return sum(xs)\n"
        )
        start = time.perf_counter()
        result = run_level(code, "total", inputs=[([1, 2],)], repeats=1, wall_timeout=3.0)
        self.assertEqual(result.status, OK)
        self.assertEqual(result.cases[0].output, 3)
        self.assertLess(time.perf_counter() - start, 3.0)

    def test_an_exception_in_the_harness_does_not_leak_the_child(self):
        """An interrupted run must not leave untrusted code executing."""
        real_wait = subprocess.Popen.wait
        calls: list[float | None] = []

        def failing_wait(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise KeyboardInterrupt
            return real_wait(self, timeout=timeout)

        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "survived")
            code = (
                "import time\n"
                "def total(xs):\n"
                "    time.sleep(1.5)\n"
                f"    open({marker!r}, 'w').close()\n"
                "    return 0\n"
            )
            with mock.patch.object(subprocess.Popen, "wait", failing_wait):
                with self.assertRaises(KeyboardInterrupt):
                    run_level(code, "total", inputs=[([1],)])
            time.sleep(2.0)
            self.assertFalse(os.path.exists(marker))

    def test_a_memory_bomb_hits_the_cap(self):
        code = "def total(xs):\n    return [0] * (10 ** 9)\n"
        result = run_level(
            code, "total", inputs=[([1],)], limits=Limits(address_space=256 << 20)
        )
        self.assertEqual(result.status, ERROR)
        self.assertIn("MemoryError", result.cases[-1].detail)

    def test_return_types_survive_the_boundary(self):
        result = run_level("def total(xs):\n    return (1, [2.5], {'k': None})\n", "total",
                           inputs=[([1],)])
        self.assertEqual(result.cases[0].output, (1, [2.5], {"k": None}))

    def test_the_working_directory_is_empty_and_private(self):
        code = "import os\ndef total(xs):\n    return sorted(os.listdir('.'))\n"
        result = run_level(code, "total", inputs=[([1],)])
        self.assertEqual(result.cases[0].output, [])

    def test_documents_the_gap_writes_outside_the_workdir_are_not_blocked(self):
        """Pins a known limitation: process-level limits are not containment.
        When kernel-level confinement lands, this test flips."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "escaped")
            code = f"def total(xs):\n    open({target!r}, 'w').close()\n    return 0\n"
            result = run_level(code, "total", inputs=[([1],)])
            self.assertEqual(result.status, OK)
            self.assertTrue(os.path.exists(target))

    def test_refuses_an_underspecified_level(self):
        with self.assertRaises(SandboxError):
            run_level(REF, "total", inputs=[([1],)], repeats=0)
        with self.assertRaises(SandboxError):
            run_level(REF, "total", inputs=[])
        with self.assertRaises(SandboxError):
            run_level(REF, "total", seeds=[1], scale=10)


class TestSandboxLimits(unittest.TestCase):
    def test_the_cpu_cap_stays_above_the_wall_budget(self):
        """``RLIMIT_CPU`` is integer-valued, counts input generation, and reports
        as a signal, so the wall clock has to be the mechanism that fires."""
        self.assertGreater(_effective_limits(Limits(cpu_seconds=1), 30.0).cpu_seconds, 30)
        self.assertEqual(_effective_limits(Limits(cpu_seconds=600), 30.0).cpu_seconds, 600)
        self.assertIsNone(_effective_limits(Limits(cpu_seconds=None), 30.0).cpu_seconds)

    def test_a_cpu_kill_is_a_timeout_not_a_crash(self):
        """A crash scores zero where a timeout leaves the level censored, so the
        two paths would disagree about the same slow solution."""
        self.assertEqual(_exit_failure(-signal.SIGXCPU, "").status, TIMEOUT)
        self.assertEqual(_exit_failure(-signal.SIGKILL, "").status, CRASHED)
        self.assertEqual(_exit_failure(1, "").status, CRASHED)


class TestReferenceMeasurement(unittest.TestCase):
    def test_reference_defines_the_limit_and_the_expected_outputs(self):
        problem = _problem()
        ref = measure_reference(problem, RunConfig(repeats=2), LOOSE)
        worst = max(max(level) for level in ref.timed())
        self.assertAlmostEqual(ref.time_limit, LOOSE.alpha * worst)
        self.assertEqual(len(ref.times), 4)
        self.assertEqual(ref.outputs[0][0], sum(range(1, 9)))
        self.assertTrue(all(t > 0 for level in ref.timed() for t in level))

    def test_a_broken_reference_is_bad_data_not_a_low_score(self):
        with self.assertRaises(SandboxError):
            measure_reference(_problem("def total(xs):\n    raise KeyError\n"), RunConfig(repeats=1))

    def test_level_count_must_match_the_metric_config(self):
        problem = _problem(
            levels=(
                GeneratedLevel(level=0, scale=8, seeds=(1,)),
                GeneratedLevel(level=1, scale=10, seeds=(2,)),
            )
        )
        with self.assertRaises(SandboxError):
            measure_reference(problem, RunConfig(repeats=1))


class TestEvaluateSolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = _problem()
        cls.config = RunConfig(repeats=2)
        cls.ref = measure_reference(cls.problem, cls.config, LOOSE)

    def _run(self, code: str):
        return evaluate_solution(self.problem, code, self.ref, self.config)

    def test_the_reference_scores_around_one(self):
        measurement = self._run(REF)
        self.assertTrue(measurement.correct)
        self.assertEqual([lvl.status for lvl in measurement.levels], [OK] * 4)
        self.assertGreater(score_solution(measurement, self.ref, LOOSE), 0.5)

    def test_a_wrong_answer_scores_zero_and_names_the_case(self):
        measurement = self._run("def total(xs):\n    return 0\n")
        self.assertFalse(measurement.correct)
        self.assertEqual(measurement.levels[0].status, WRONG_ANSWER)
        self.assertIn("case 0", measurement.detail)
        self.assertEqual(score_solution(measurement, self.ref, LOOSE), 0.0)

    def test_correct_on_level_zero_and_wrong_later_is_still_caught(self):
        """Level 0's small inputs are a filter, not the whole oracle."""
        code = "def total(xs):\n    return sum(xs) if len(xs) < 100 else 0\n"
        measurement = self._run(code)
        self.assertFalse(measurement.correct)
        self.assertEqual(measurement.levels[0].status, OK)
        self.assertEqual(measurement.levels[1].status, WRONG_ANSWER)

    def test_an_exception_stops_the_run_and_skips_the_rest(self):
        measurement = self._run("def total(xs):\n    raise ValueError('nope')\n")
        self.assertFalse(measurement.correct)
        self.assertIn(measurement.levels[0].status, (ERROR, CRASHED))
        self.assertEqual([lvl.status for lvl in measurement.levels[1:]], [SKIPPED] * 3)

    def test_a_timeout_censors_every_level_it_did_not_reach(self):
        """Level 0 carries no time limit, so a uniformly slow solution is timed
        out at the first scored level and censored from there on."""
        code = "import time\ndef total(xs):\n    time.sleep(0.05)\n    return sum(xs)\n"
        measurement = self._run(code)
        self.assertEqual(measurement.levels[0].status, OK)
        self.assertEqual(measurement.levels[1].status, TIMEOUT)
        self.assertEqual([lvl.status for lvl in measurement.levels[2:]], [SKIPPED] * 2)
        self.assertTrue(measurement.correct)
        self.assertTrue(all(t == math.inf for lvl in measurement.timed_times(3) for t in lvl))
        self.assertEqual(score_solution(measurement, self.ref, LOOSE), 0.0)

    def test_slow_on_level_zero_is_not_a_wrong_answer(self):
        """Level 0's inputs are adversarial rather than large; being slow there is
        an efficiency result, and correctness has to be decided on its own."""
        code = (
            "import time\n"
            "def total(xs):\n"
            "    if len(xs) < 100:\n"
            "        time.sleep(0.05)\n"
            "    return sum(xs)\n"
        )
        measurement = self._run(code)
        self.assertEqual(measurement.levels[0].status, OK)
        self.assertTrue(measurement.correct)
        self.assertIn(0, measurement.verified_levels)

    def test_a_hang_on_level_zero_is_not_reported_as_correct(self):
        """Nothing was verified, so counting the sample as correct would inflate
        pass@k on a solution that never produced an answer."""
        code = "def total(xs):\n    while True:\n        pass\n"
        measurement = evaluate_solution(
            self.problem, code, self.ref, RunConfig(repeats=1, filter_wall_timeout=2.0)
        )
        self.assertFalse(measurement.correct)
        self.assertEqual(measurement.levels[0].status, TIMEOUT)
        self.assertEqual(measurement.verified_levels, ())
        self.assertIn("unverified", measurement.detail)
        self.assertEqual(score_solution(measurement, self.ref, LOOSE), 0.0)

    def test_a_wrong_answer_is_not_hidden_by_a_later_timeout(self):
        """The child reports the cases it finished before running out of time, and
        a wrong answer among them outranks the timeout."""
        problem = _problem(
            levels=(
                GeneratedLevel(level=0, scale=8, seeds=(1, 2)),
                GeneratedLevel(level=1, scale=2000, seeds=(3, 4)),
                GeneratedLevel(level=2, scale=8000, seeds=(5,)),
                GeneratedLevel(level=3, scale=32000, seeds=(6,)),
            )
        )
        ref = measure_reference(problem, RunConfig(repeats=1), LOOSE)
        code = (
            "import time\n"
            "def total(xs):\n"
            "    if xs and xs[0] == 3:\n"
            "        return 0\n"
            "    time.sleep(0.5)\n"
            "    return sum(xs)\n"
        )
        measurement = evaluate_solution(problem, code, ref, RunConfig(repeats=1))
        self.assertFalse(measurement.correct)
        self.assertEqual(measurement.levels[1].status, WRONG_ANSWER)
        self.assertIn("case 0", measurement.detail)
        self.assertEqual(score_solution(measurement, ref, LOOSE), 0.0)

    def test_a_huge_integer_answer_is_scored_rather_than_crashing_the_problem(self):
        """An int too wide to render in decimal must neither fail to cross the
        process boundary nor raise while the disagreement is being described."""
        measurement = self._run("def total(xs):\n    return 7 ** 20000\n")
        self.assertFalse(measurement.correct)
        self.assertEqual(measurement.levels[0].status, WRONG_ANSWER)
        self.assertIn(f"{7 ** 20000:x}"[:16], measurement.detail)

    def test_a_level_at_the_limit_scores_zero_without_being_called_wrong(self):
        measurement = SolutionMeasurement(
            problem_id=0,
            correct=True,
            levels=(
                LevelMeasurement(0, OK, times=(0.0,)),
                LevelMeasurement(1, OK, times=(self.ref.time_limit,)),
                LevelMeasurement(2, OK, times=(self.ref.time_limit,)),
                LevelMeasurement(3, OK, times=(self.ref.time_limit,)),
            ),
            verified_levels=(0, 1, 2, 3),
        )
        self.assertEqual(score_solution(measurement, self.ref, LOOSE), 0.0)

    def test_a_slower_solution_scores_lower(self):
        loop = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
        fast = score_solution(self._run(REF), self.ref, LOOSE)
        slow = score_solution(self._run(loop), self.ref, LOOSE)
        self.assertLess(slow, fast)
        self.assertGreater(slow, 0.0)

    def test_a_reference_for_another_problem_is_refused(self):
        other = _problem(problem_id=1)
        with self.assertRaises(SandboxError):
            evaluate_solution(other, REF, self.ref, self.config)

    def test_materialized_levels_run_too(self):
        problem = _problem(
            input_generator="",
            levels=(
                MaterializedLevel(level=0, inputs=(([1, 2],),)),
                MaterializedLevel(level=1, inputs=((list(range(2000)),),)),
                MaterializedLevel(level=2, inputs=((list(range(8000)),),)),
                MaterializedLevel(level=3, inputs=((list(range(32000)),),)),
            ),
        )
        ref = measure_reference(problem, RunConfig(repeats=1), LOOSE)
        measurement = evaluate_solution(problem, REF, ref, RunConfig(repeats=1))
        self.assertTrue(measurement.correct)
        self.assertEqual([lvl.status for lvl in measurement.levels], [OK] * 4)


class TestEvaluateProblem(unittest.TestCase):
    def test_samples_share_one_reference_and_feed_eff_at_k(self):
        problem = _problem()
        wrong = "def total(xs):\n    return 0\n"
        result = evaluate_problem(problem, [REF, wrong, REF], RunConfig(repeats=1), LOOSE)
        self.assertEqual(result.problem_id, 0)
        self.assertEqual(result.n_correct, 2)
        self.assertEqual(result.scores[1], 0.0)
        self.assertEqual(len(result.scores), 3)
        self.assertGreater(eff_at_k(result.scores, k=2), 0.0)
        self.assertAlmostEqual(pass_at_k(3, 2, k=1), 2 / 3)


if __name__ == "__main__":
    unittest.main()
