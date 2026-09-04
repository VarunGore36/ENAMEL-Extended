"""Tests for the parity gate and for the transcription of the published tables.

Two kinds of check. The published numbers have internal consistency the paper
never states, and a mistranscribed digit usually breaks it; the comparison layer
has a claimed relationship between its criteria that a test can pin. Neither
needs a run record. See docs/decisions/0007-parity-gate.md.
"""

from __future__ import annotations

import os
import statistics
import unittest
from pathlib import Path

from enamel_ext.data import published as pub
from enamel_ext.pipeline.record import load_record
from enamel_ext.report import parity


class TranscriptionTest(unittest.TestCase):
    """Cross-checks that a wrong digit in published.py would break."""

    def test_table_6_subsets_leave_a_remainder_in_range(self):
        for model in pub.TABLE3_SAMPLING:
            scores = pub.remainder_scores(model)
            for column, value in zip(pub.COLUMNS, scores):
                with self.subTest(model=model, column=column):
                    if column.startswith("pass"):
                        self.assertGreaterEqual(value, -1e-9)
                        self.assertLessEqual(value, 1.0 + 1e-9)
                    else:
                        self.assertGreaterEqual(value, -1e-9)
                        self.assertLessEqual(value, pub.MAX_SAMPLE_SCORE + 1e-9)

    def test_the_subsets_partition_the_problem_count(self):
        self.assertEqual(
            pub.ALGORITHM_PROBLEMS + pub.IMPLEMENTATION_PROBLEMS
            + pub.REMAINDER_PROBLEMS,
            pub.PROBLEMS,
        )

    def test_table_12_basic_rows_repeat_table_3_greedy(self):
        for model, scores in pub.TABLE12_BASIC.items():
            with self.subTest(model=model):
                self.assertEqual(scores, pub.TABLE3_GREEDY[model])

    def test_table_10_defaults_agree_with_table_3(self):
        target = pub.TABLE3_GREEDY["GPT-4 Turbo"].eff1
        self.assertEqual(pub.TABLE10_ALPHA[pub.ALPHA], target)
        for level, weight in enumerate(pub.LEVEL_WEIGHTS, start=1):
            with self.subTest(level=level):
                self.assertEqual(pub.TABLE10_HARDNESS[level][weight], target)

    def test_greedy_only_models_have_no_sampling_row(self):
        for model in pub.GREEDY_ONLY:
            with self.subTest(model=model):
                self.assertIn(model, pub.TABLE3_GREEDY)
                self.assertNotIn(model, pub.TABLE3_SAMPLING)

    def test_every_sampling_model_is_also_a_greedy_model(self):
        self.assertEqual(
            set(pub.TABLE3_SAMPLING) | set(pub.GREEDY_ONLY), set(pub.TABLE3_GREEDY)
        )

    def test_the_two_table_6_subsets_cover_the_same_models(self):
        self.assertEqual(set(pub.TABLE6_ALGORITHM), set(pub.TABLE3_SAMPLING))
        self.assertEqual(set(pub.TABLE6_IMPLEMENTATION), set(pub.TABLE3_SAMPLING))

    def test_pass_at_k_rises_with_k_everywhere(self):
        for name in ("sampling", "algorithm", "implementation"):
            for model, scores in pub.table(name).items():
                with self.subTest(table=name, model=model):
                    self.assertLessEqual(scores.pass1, scores.pass10)
                    self.assertLessEqual(scores.pass10, scores.pass100)

    def test_no_published_score_exceeds_the_metric_ceiling(self):
        for name in ("greedy", "sampling", "algorithm", "implementation"):
            for model, scores in pub.table(name).items():
                for column, value in zip(pub.COLUMNS, scores):
                    if value is None or not column.startswith("eff"):
                        continue
                    with self.subTest(table=name, model=model, column=column):
                        self.assertLessEqual(value, pub.MAX_SAMPLE_SCORE)

    def test_eff_stays_under_pass_at_the_same_k(self):
        """True of the published tables, not a law: a sample beating the
        reference scores above 1, so ``eff`` may exceed ``pass`` in principle.
        No published model comes close, and a slipped digit would show here."""
        for name in ("greedy", "sampling", "algorithm", "implementation"):
            for model, scores in pub.table(name).items():
                for k in (1, 10, 100):
                    eff = getattr(scores, f"eff{k}")
                    if eff is None:
                        continue
                    with self.subTest(table=name, model=model, k=k):
                        self.assertLessEqual(eff, getattr(scores, f"pass{k}"))


class Table11Test(unittest.TestCase):
    """The published standard deviations, and what they can and cannot support."""

    def test_the_rao_blackwellized_row_is_its_own_equation_8_bound(self):
        for k, vanilla in pub.TABLE11_VANILLA_STD.items():
            with self.subTest(k=k):
                bound = pub.rb_std_bound(k, pub.TABLE11_SAMPLES, vanilla)
                self.assertEqual(
                    round(bound, 2), pub.TABLE11_RAO_BLACKWELLIZED_STD[k]
                )

    def test_the_bound_needs_k_within_the_sample_size(self):
        for k, n in ((0, 100), (101, 100), (-1, 10)):
            with self.subTest(k=k, n=n):
                with self.assertRaises(ValueError):
                    pub.rb_std_bound(k, n, 0.2)

    def test_the_figures_cannot_be_benchmark_level(self):
        """A mean over 142 bounded problems is far quieter than 0.20.

        Scores lie in ``[0, MAX_SAMPLE_SCORE]``, so a per-problem standard
        deviation is at most half that width, and the mean over problems is
        smaller again by ``sqrt(142)``.
        """
        widest = pub.benchmark_std(pub.MAX_SAMPLE_SCORE / 2.0)
        self.assertLess(widest, min(pub.TABLE11_VANILLA_STD.values()))

    def test_carried_to_benchmark_scale_the_bound_is_negligible(self):
        carried = pub.benchmark_std(pub.TABLE11_RAO_BLACKWELLIZED_STD[1])
        self.assertLess(carried, 0.002)

    def test_benchmark_std_rejects_an_empty_benchmark(self):
        with self.assertRaises(ValueError):
            pub.benchmark_std(0.2, problems=0)


def _greedy(column: str = "eff1") -> dict[str, float]:
    return {m: getattr(s, column) for m, s in pub.TABLE3_GREEDY.items()}


class Table7Test(unittest.TestCase):
    """Table 7's two rankings: a check on Table 3, and a published disagreement."""

    def test_the_eff1_column_reproduces_table_3s_greedy_order(self):
        """Appendix C.3 prints this order independently of Table 3, so a mismatch
        would mean one of the two transcriptions is wrong."""
        board = [model for model, _ in pub.leaderboard("eff1", "greedy")]
        self.assertEqual(
            tuple(board[: len(pub.TABLE7_EFF1_RANKING)]), pub.TABLE7_EFF1_RANKING
        )

    def test_the_two_rankings_cover_the_same_models(self):
        self.assertEqual(
            set(pub.TABLE7_EFF1_RANKING), set(pub.TABLE7_SPEEDUP_RANKING)
        )

    def test_the_papers_own_two_rankings_correlate_at_0_848(self):
        """The paper calls this pair 'very different' and argues from the
        difference that speedup is unreasonable under censoring."""
        self.assertEqual(round(parity.published_disagreement_tau(), 3), 0.848)

    def test_that_disagreement_is_milder_than_our_worst_case_floor(self):
        """The tau a maximally wrong local ordering still earns sits above the tau
        the paper treats as disqualifying, so no threshold between them exists."""
        floor = parity.tau_floor(_greedy(), parity.EFF_TOLERANCE)
        self.assertGreater(floor.tau, parity.published_disagreement_tau())

    def test_the_ranking_prints_orders_its_own_precision_cannot_resolve(self):
        eff = _greedy()
        gaps = [
            round(eff[a] - eff[b], 3)
            for a, b in zip(pub.TABLE7_EFF1_RANKING, pub.TABLE7_EFF1_RANKING[1:])
        ]
        self.assertEqual(sorted(gaps)[:2], [0.001, 0.002])

    def test_rankings_over_different_model_sets_are_refused(self):
        with self.assertRaises(ValueError):
            parity.ranking_tau(("a", "b"), ("a", "c"))

    def test_a_single_model_cannot_be_correlated(self):
        with self.assertRaises(ValueError):
            parity.ranking_tau(("a",), ("a",))

    def test_an_identical_ranking_correlates_perfectly(self):
        order = pub.TABLE7_EFF1_RANKING
        self.assertEqual(parity.ranking_tau(order, order), 1.0)

    def test_a_reversed_ranking_correlates_negatively(self):
        order = pub.TABLE7_EFF1_RANKING
        self.assertEqual(parity.ranking_tau(order, tuple(reversed(order))), -1.0)


class Table9Test(unittest.TestCase):
    """The cross-benchmark row, which repeats one Table 3 entry."""

    def test_the_enamel_entry_repeats_table_3(self):
        metric, value = pub.TABLE9_CROSS_BENCHMARK["ENAMEL (ours)"]
        self.assertEqual(metric, "eff@1")
        self.assertEqual(value, pub.TABLE3_GREEDY[pub.TABLE9_MODEL].eff1)

    def test_the_compared_model_is_a_published_one(self):
        self.assertIn(pub.TABLE9_MODEL, pub.TABLE3_GREEDY)


class IdentifierTest(unittest.TestCase):
    """The five checkpoints Appendix C.1 names, and the many it does not."""

    def test_every_named_identifier_belongs_to_a_published_model(self):
        for display in pub.MODEL_IDENTIFIERS:
            with self.subTest(model=display):
                self.assertIn(display, pub.TABLE3_GREEDY)

    def test_most_published_models_have_no_stated_identifier(self):
        """Why run names cannot be resolved to table keys from the paper alone:
        25 of the 30 rows are named only by whatever their source release called
        them."""
        unnamed = set(pub.TABLE3_GREEDY) - set(pub.MODEL_IDENTIFIERS)
        self.assertEqual(len(unnamed), 25)

    def test_the_greedy_only_models_are_all_named(self):
        self.assertLessEqual(set(pub.GREEDY_ONLY), set(pub.MODEL_IDENTIFIERS))

    def test_identifiers_are_distinct(self):
        values = list(pub.MODEL_IDENTIFIERS.values())
        self.assertEqual(len(values), len(set(values)))

    def test_the_model_with_a_known_sample_size_is_published(self):
        for model in pub.KNOWN_SAMPLE_SIZE:
            with self.subTest(model=model):
                self.assertIn(model, pub.TABLE3_SAMPLING)


class ResolutionTest(unittest.TestCase):
    """What the published spacing lets a tolerance test, before any measurement."""

    def setUp(self):
        self.eff = _greedy()

    def test_the_pair_count_is_every_pair(self):
        res = parity.resolution(self.eff, 0.0)
        self.assertEqual(res.models, len(self.eff))
        self.assertEqual(res.pairs, res.models * (res.models - 1) // 2)
        self.assertEqual(res.resolvable, res.pairs)
        self.assertEqual(res.adjacent, res.models - 1)

    def test_resolvable_pairs_shrink_as_the_tolerance_grows(self):
        counts = [
            parity.resolution(self.eff, d).resolvable
            for d in (0.0, 0.01, 0.02, 0.05, 0.1)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_the_gate_keeps_most_pairs_but_almost_no_neighbours(self):
        """The number that shapes decision 0007: power overall, none locally."""
        res = parity.resolution(self.eff, parity.EFF_TOLERANCE)
        self.assertEqual((res.pairs, res.resolvable), (435, 356))
        self.assertEqual((res.adjacent, res.adjacent_resolvable), (29, 1))
        self.assertEqual(round(100 * res.share, 1), 81.8)

    def test_the_neighbour_spacing_quoted_in_the_decision_holds(self):
        ranked = sorted(self.eff.values(), reverse=True)
        gaps = [
            round(ranked[i] - ranked[i + 1], 4) for i in range(len(ranked) - 1)
        ]
        self.assertEqual(len(gaps), 29)
        self.assertEqual(sum(1 for g in gaps if g <= 0.005), 6)
        self.assertEqual((min(gaps), statistics.median(gaps)), (0.001, 0.013))

    def test_the_ungated_band_is_the_size_the_decision_claims(self):
        delta = parity.EFF_TOLERANCE
        wide = parity.resolvable_pairs(self.eff, parity.INVERSION_MARGIN * delta)
        near = parity.resolvable_pairs(self.eff, delta)
        self.assertEqual(len(near) - len(wide), 80)

    def test_the_tolerances_the_decision_rejected_cost_what_it_says(self):
        counts = {
            d: parity.resolution(self.eff, d).resolvable for d in (0.02, 0.05, 0.10)
        }
        self.assertEqual(counts, {0.02: 405, 0.05: 356, 0.10: 276})

    def test_adjacent_pairs_are_a_subset_of_resolvable_ones(self):
        for d in (0.005, 0.02, 0.05):
            with self.subTest(tolerance=d):
                res = parity.resolution(self.eff, d)
                self.assertLessEqual(res.adjacent_resolvable, res.resolvable)

    def test_every_pair_is_ordered_better_first(self):
        for better, worse in parity.resolvable_pairs(self.eff, 0.02):
            with self.subTest(pair=(better, worse)):
                self.assertGreater(self.eff[better], self.eff[worse])

    def test_a_tolerance_above_the_whole_range_resolves_nothing(self):
        span = max(self.eff.values()) - min(self.eff.values())
        self.assertEqual(parity.resolvable_pairs(self.eff, span), ())

    def test_shares_are_zero_rather_than_undefined_for_one_model(self):
        res = parity.resolution({"only": 0.5}, 0.05)
        self.assertEqual((res.pairs, res.adjacent), (0, 0))
        self.assertEqual((res.share, res.adjacent_share), (0.0, 0.0))

    def test_a_negative_tolerance_is_refused(self):
        with self.assertRaises(ValueError):
            parity.resolution(self.eff, -0.01)


class TauFloorTest(unittest.TestCase):
    """Why rank correlation is reported and not gated."""

    def test_a_maximally_wrong_local_ordering_still_scores_high(self):
        floor = parity.tau_floor(_greedy(), parity.EFF_TOLERANCE)
        self.assertEqual((floor.inverted, floor.adjacent), (15, 29))
        self.assertEqual(round(floor.tau, 3), 0.931)

    def test_it_cannot_invert_more_than_disjointness_allows(self):
        eff = _greedy()
        floor = parity.tau_floor(eff, 1.0)
        self.assertEqual(floor.inverted, len(eff) // 2)

    def test_a_zero_tolerance_inverts_nothing_and_leaves_tau_at_one(self):
        floor = parity.tau_floor(_greedy(), 0.0)
        self.assertEqual(floor.inverted, 0)
        self.assertEqual(floor.tau, 1.0)

    def test_one_model_cannot_be_ranked(self):
        with self.assertRaises(ValueError):
            parity.tau_floor({"only": 0.5}, 0.05)


class DeviationTest(unittest.TestCase):
    def test_identical_scores_deviate_by_nothing(self):
        eff = _greedy()
        rows = parity.deviations(eff, eff, parity.EFF_TOLERANCE)
        self.assertEqual(len(rows), len(eff))
        self.assertTrue(all(row.delta == 0.0 and row.within for row in rows))

    def test_the_sign_is_ours_minus_theirs(self):
        rows = parity.deviations({"m": 0.5}, {"m": 0.4}, 0.05)
        self.assertAlmostEqual(rows[0].delta, 0.1)
        self.assertFalse(rows[0].within)

    def test_the_worst_row_comes_first(self):
        ours = {"a": 0.50, "b": 0.30, "c": 0.41}
        rows = parity.deviations(ours, {"a": 0.4, "b": 0.4, "c": 0.4}, 0.05)
        self.assertEqual([row.model for row in rows], ["b", "a", "c"])

    def test_the_boundary_is_inclusive(self):
        rows = parity.deviations({"m": 0.45}, {"m": 0.40}, 0.05)
        self.assertTrue(rows[0].within)

    def test_models_the_paper_did_not_publish_are_left_out(self):
        rows = parity.deviations({"ours only": 0.5}, _greedy(), 0.05)
        self.assertEqual(rows, ())


class InversionTest(unittest.TestCase):
    """The relationship between the deviation and ordering criteria."""

    def test_agreement_inverts_nothing(self):
        eff = _greedy()
        self.assertEqual(parity.inversions(eff, eff, parity.EFF_TOLERANCE), ())

    def test_a_reversed_board_inverts_every_resolvable_pair(self):
        eff = _greedy()
        flipped = {m: -v for m, v in eff.items()}
        found = parity.inversions(flipped, eff, parity.EFF_TOLERANCE)
        self.assertEqual(
            len(found), len(parity.resolvable_pairs(eff, parity.EFF_TOLERANCE))
        )

    def test_deviations_within_tolerance_cannot_produce_a_gated_inversion(self):
        """The claim decision 0007 rests on, checked against the real table.

        Two models each inside ``tolerance`` move by at most twice it between
        them, so a pair separated by more than ``INVERSION_MARGIN`` times the
        tolerance keeps its order. Ordering is therefore a consistency check on
        the deviation criterion, not a second hurdle.
        """
        eff = _greedy()
        delta = parity.EFF_TOLERANCE
        for sign in (1.0, -1.0):
            ours = {
                m: v + sign * delta * (1.0 if i % 2 else -1.0)
                for i, (m, v) in enumerate(eff.items())
            }
            with self.subTest(sign=sign):
                found = parity.inversions(ours, eff, delta)
                self.assertTrue(all(not row.gated for row in found))

    def test_an_inversion_can_survive_a_passing_deviation_check(self):
        """Which is why the near band is reported rather than assumed empty."""
        eff = _greedy()
        ours = dict(eff)
        better, worse = "Llama 3 8B Instruct", "Code Llama 34B Python"
        gap = eff[better] - eff[worse]
        self.assertGreater(gap, parity.EFF_TOLERANCE)
        self.assertLess(gap, parity.INVERSION_MARGIN * parity.EFF_TOLERANCE)
        ours[better] = eff[better] - 0.04
        ours[worse] = eff[worse] + 0.04
        found = parity.inversions(ours, eff, parity.EFF_TOLERANCE)
        self.assertEqual([(r.better, r.worse) for r in found], [(better, worse)])
        self.assertFalse(found[0].gated)

    def test_a_wide_pair_that_flips_is_gated(self):
        published = {"top": 0.50, "bottom": 0.20}
        found = parity.inversions({"top": 0.20, "bottom": 0.50}, published, 0.05)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].gated)

    def test_a_tie_in_our_scores_is_not_an_inversion(self):
        published = {"top": 0.50, "bottom": 0.20}
        ours = {"top": 0.35, "bottom": 0.35}
        self.assertEqual(parity.inversions(ours, published, 0.05), ())

    def test_pairs_we_did_not_run_are_skipped(self):
        published = {"top": 0.50, "bottom": 0.20}
        self.assertEqual(parity.inversions({"top": 0.1}, published, 0.05), ())

    def test_a_margin_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            parity.inversions(_greedy(), _greedy(), 0.05, margin=0.5)


class DifferentialBoundTest(unittest.TestCase):
    """The timing argument behind the size of the eff tolerance."""

    def test_an_identical_clock_moves_nothing(self):
        self.assertEqual(parity.differential_bound(1.0), 0.0)

    def test_the_pre_committed_tolerance_absorbs_a_two_and_a_half_percent_skew(self):
        self.assertAlmostEqual(parity.differential_bound(1.025), parity.EFF_TOLERANCE)

    def test_it_is_linear_in_the_excess(self):
        one = parity.differential_bound(1.01)
        self.assertAlmostEqual(parity.differential_bound(1.02), 2 * one)

    def test_a_larger_alpha_tolerates_more(self):
        self.assertLess(
            parity.differential_bound(1.05, alpha=3.0),
            parity.differential_bound(1.05, alpha=2.0),
        )

    def test_a_factor_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            parity.differential_bound(0.9)

    def test_an_alpha_at_one_is_refused(self):
        with self.assertRaises(ValueError):
            parity.differential_bound(1.1, alpha=1.0)


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.eff = _greedy("eff1")
        self.passes = _greedy("pass1")

    def test_the_published_table_matches_itself(self):
        result = parity.compare(self.eff, self.passes)
        self.assertTrue(result.passed)
        self.assertEqual(result.tau, 1.0)
        self.assertEqual((result.missing, result.extra), ((), ()))

    def test_a_subset_run_reports_only_its_own_pairs(self):
        few = dict(list(self.eff.items())[:6])
        result = parity.compare(few, self.passes)
        self.assertEqual(len(result.missing), len(self.eff) - 6)
        self.assertEqual(result.resolution.models, 6)
        self.assertEqual(result.resolution.pairs, 15)

    def test_coverage_alone_does_not_fail_the_gate(self):
        """``passed`` speaks for the models compared; ``missing`` is separate."""
        few = dict(list(self.eff.items())[:2])
        self.assertTrue(parity.compare(few, self.passes).passed)
        self.assertTrue(parity.compare(few, self.passes).missing)

    def test_models_we_ran_that_the_paper_did_not_are_extra(self):
        ours = dict(self.eff)
        ours["Some 2026 Model"] = 0.9
        result = parity.compare(ours, self.passes)
        self.assertEqual(result.extra, ("Some 2026 Model",))
        self.assertTrue(result.passed)

    def test_one_model_leaves_tau_undefined_rather_than_raising(self):
        result = parity.compare({"GPT-4": 0.454}, {"GPT-4": 0.831})
        self.assertIsNone(result.tau)
        self.assertIsNone(result.floor)
        self.assertTrue(result.passed)

    def test_nothing_in_common_is_reported_and_not_divided_by_zero(self):
        result = parity.compare({"x": 0.4}, {"x": 0.8})
        self.assertEqual(result.resolution.pairs, 0)
        self.assertEqual(result.resolution.share, 0.0)
        self.assertEqual(result.eff, ())

    def test_nothing_in_common_is_a_failure_rather_than_a_vacuous_pass(self):
        """Three criteria with nothing to check are satisfied by default, which is
        the one shape of emptiness the gate refuses. Unlike low coverage, it is not
        a weak result but an absent one."""
        result = parity.compare({"x": 0.4}, {"x": 0.8})
        self.assertEqual(result.compared, 0)
        self.assertEqual(
            (result.eff_misses, result.pass_misses, result.gated_inversions),
            ((), (), ()),
        )
        self.assertFalse(result.passed)

    def test_the_report_says_that_nothing_was_compared(self):
        lines = parity.format_parity(parity.compare({"x": 0.4}, {"x": 0.8}))
        verdict = next(line for line in lines if "verdict" in line)
        self.assertIn("FAIL", verdict)
        self.assertIn("nothing compared", verdict)

    def test_one_model_in_common_is_enough_to_be_judged(self):
        """The boundary: the refusal is of an empty comparison, not a small one."""
        result = parity.compare({"GPT-4": 0.454}, {"GPT-4": 0.831})
        self.assertEqual(result.compared, 1)
        self.assertTrue(result.passed)

    def test_a_run_missing_its_pass_scores_cannot_pass(self):
        """Both columns are criteria, so a run that supplies only one of them has
        not been checked against the other."""
        result = parity.compare(self.eff, {})
        self.assertEqual(result.passes, ())
        self.assertFalse(result.passed)

    def test_an_eff_miss_fails_and_a_pass_miss_fails(self):
        bad_eff = dict(self.eff)
        bad_eff["GPT-4"] = 0.10
        self.assertFalse(parity.compare(bad_eff, self.passes).passed)
        bad_pass = dict(self.passes)
        bad_pass["GPT-4"] = 0.10
        self.assertFalse(parity.compare(self.eff, bad_pass).passed)

    def test_the_tolerances_differ_between_the_two_columns(self):
        """``pass`` is the tight test because it carries no timing."""
        self.assertLess(parity.PASS_TOLERANCE, parity.EFF_TOLERANCE)
        drift = {m: v + 0.02 for m, v in self.passes.items()}
        result = parity.compare(self.eff, drift)
        self.assertEqual(len(result.pass_misses), len(self.passes))
        self.assertEqual(result.eff_misses, ())

    def test_the_sampling_table_is_reachable_at_other_k(self):
        eff10 = {m: s.eff10 for m, s in pub.TABLE3_SAMPLING.items()}
        pass10 = {m: s.pass10 for m, s in pub.TABLE3_SAMPLING.items()}
        result = parity.compare(eff10, pass10, name="sampling", k=10)
        self.assertTrue(result.passed)
        self.assertEqual(result.name, "sampling")

    def test_a_column_the_paper_never_prints_is_refused(self):
        with self.assertRaises(ValueError):
            parity.compare(self.eff, self.passes, k=5)

    def test_an_unknown_table_is_refused(self):
        with self.assertRaises(ValueError):
            parity.compare(self.eff, self.passes, name="nonesuch")

    def test_greedy_only_models_are_absent_from_the_sampling_table(self):
        eff = {m: 0.4 for m in pub.GREEDY_ONLY}
        result = parity.compare(eff, eff, name="sampling")
        self.assertEqual(set(result.extra), set(pub.GREEDY_ONLY))


class FormatTest(unittest.TestCase):
    def setUp(self):
        self.eff = _greedy("eff1")
        self.passes = _greedy("pass1")

    def test_agreement_says_pass_and_still_shows_a_row(self):
        lines = parity.format_parity(parity.compare(self.eff, self.passes))
        text = "\n".join(lines)
        self.assertIn("verdict: pass", text)
        self.assertIn(pub.PAPER, text)
        self.assertIn("eff ", text)
        self.assertIn("pass ", text)

    def test_a_miss_is_shouted_and_named(self):
        bad = dict(self.eff)
        bad["GPT-4 Turbo"] = 0.10
        text = "\n".join(parity.format_parity(parity.compare(bad, self.passes)))
        self.assertIn("FAIL", text)
        self.assertIn("GPT-4 Turbo", text)
        self.assertIn("over", text)

    def test_tau_is_printed_beside_the_floor_it_cannot_fall_below(self):
        text = "\n".join(parity.format_parity(parity.compare(self.eff, self.passes)))
        self.assertIn("not a criterion", text)
        self.assertIn("0.931", text)

    def test_missing_models_are_counted_in_the_header(self):
        few = dict(list(self.eff.items())[:6])
        text = "\n".join(parity.format_parity(parity.compare(few, self.passes)))
        self.assertIn("compared 6 of 30 models", text)
        self.assertIn("24 not run", text)

    def test_the_row_limit_is_respected(self):
        bad = {m: 0.0 for m in self.eff}
        lines = parity.format_parity(parity.compare(bad, self.passes), limit=3)
        self.assertEqual(sum(1 for line in lines if "    eff " in line), 3)

    def test_a_run_that_scored_nothing_reports_rather_than_raises(self):
        """All-zero scores leave tau undefined; the section still renders."""
        result = parity.compare({m: 0.0 for m in self.eff}, self.passes)
        self.assertIsNone(result.tau)
        self.assertFalse(result.passed)
        self.assertIn("FAIL", "\n".join(parity.format_parity(result)))

    def test_a_single_model_omits_the_tau_line(self):
        result = parity.compare({"GPT-4": 0.454}, {"GPT-4": 0.831})
        text = "\n".join(parity.format_parity(result))
        self.assertNotIn("kendall", text)
        self.assertIn("verdict: pass", text)


#: A run record to hold against the published tables. Milestone 2's gate needs
#: real measurements, so it skips until one is pointed at.
RECORD_ENV = "ENAMEL_EXT_PARITY_RECORD"


class GateTest(unittest.TestCase):
    """The gate itself. Skips without a record rather than passing vacuously."""

    def setUp(self):
        path = os.environ.get(RECORD_ENV)
        if not path:
            self.skipTest(f"no run record; set {RECORD_ENV} to a saved run")
        if not Path(path).exists():
            self.fail(f"{RECORD_ENV} points at {path}, which does not exist")
        self.record = load_record(path)
        if not set(self.record.models) & set(pub.TABLE3_GREEDY):
            self.skipTest("the run has no model the paper published")
        self.result = parity.compare(
            {m: self.record.eff_at_k(m, 1) for m in self.record.models},
            {m: self.record.pass_at_k(m, 1) for m in self.record.models},
        )

    def test_every_problem_the_paper_scored_was_scored_here(self):
        self.assertEqual(len(self.record.ids()), pub.PROBLEMS)

    def test_pass_at_1_matches_within_the_tight_tolerance(self):
        misses = [(r.model, round(r.delta, 3)) for r in self.result.pass_misses]
        self.assertEqual(misses, [], f"pass@1 outside {parity.PASS_TOLERANCE}")

    def test_any_pass_at_1_miss_is_in_the_predicted_direction(self):
        """Level 0 carries no time limit here (decision 0004), so a solution the
        paper timed out on can only add to our count, never subtract."""
        for row in self.result.pass_misses:
            with self.subTest(model=row.model):
                self.assertGreater(row.delta, 0.0)

    def test_eff_at_1_matches_within_the_loose_tolerance(self):
        misses = [(r.model, round(r.delta, 3)) for r in self.result.eff_misses]
        self.assertEqual(misses, [], f"eff@1 outside {parity.EFF_TOLERANCE}")

    def test_no_widely_separated_pair_changed_places(self):
        inverted = [(r.better, r.worse) for r in self.result.gated_inversions]
        self.assertEqual(inverted, [])

    def test_the_gate_reports_its_own_coverage(self):
        self.assertEqual(self.result.missing, (), "some published model was not run")
