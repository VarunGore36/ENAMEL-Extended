"""Resolving a run's model names onto the published table keys.

The gate compares by name, so a mismatch here empties the comparison silently.
See docs/decisions/0008-model-naming.md.
"""

from __future__ import annotations

import unittest

from enamel_ext.data import naming, published as pub


class NormalizeTest(unittest.TestCase):
    def test_case_spacing_and_punctuation_stop_mattering(self):
        for name in ("GPT-4 Turbo", "gpt4turbo", "GPT_4__TURBO", "  gpt 4 turbo  "):
            with self.subTest(name=name):
                self.assertEqual(naming.normalize(name), "gpt4turbo")

    def test_digits_are_kept_because_they_carry_the_model_size(self):
        self.assertNotEqual(
            naming.normalize("CodeGen 16B"), naming.normalize("CodeGen 6B")
        )

    def test_an_empty_name_normalizes_to_nothing(self):
        self.assertEqual(naming.normalize("---"), "")


class IndexTest(unittest.TestCase):
    def test_no_two_published_names_collide(self):
        """The whole scheme rests on this, so it is asserted rather than assumed."""
        keys = [naming.normalize(model) for model in naming.published_names()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_the_published_names_are_every_table_key(self):
        every = set()
        for name in ("greedy", "sampling", "algorithm", "implementation"):
            every |= set(pub.table(name))
        self.assertEqual(set(naming.published_names()), every)

    def test_the_shipped_aliases_are_only_the_paper_stated_identifiers(self):
        self.assertEqual(set(naming.ALIASES.values()), set(pub.MODEL_IDENTIFIERS))
        self.assertEqual(set(naming.ALIASES), set(pub.MODEL_IDENTIFIERS.values()))

    def test_an_alias_naming_an_unpublished_model_is_refused(self):
        with self.assertRaises(ValueError):
            naming.resolve(["x"], aliases={"x": "Some 2026 Model"})


class ResolveTest(unittest.TestCase):
    def test_a_published_display_name_resolves_to_itself(self):
        report = naming.resolve(naming.published_names())
        self.assertEqual(report.resolved, {m: m for m in naming.published_names()})
        self.assertEqual(report.unresolved, ())
        self.assertTrue(report.clean)

    def test_a_stated_api_identifier_resolves_to_its_display_name(self):
        for display, identifier in pub.MODEL_IDENTIFIERS.items():
            with self.subTest(model=display):
                self.assertEqual(
                    naming.resolve([identifier]).resolved, {identifier: display}
                )

    def test_a_differently_punctuated_name_resolves_without_an_alias(self):
        """Spacing and case are formatting, so matching through them is not a guess."""
        report = naming.resolve(["code_llama_34b_python"])
        self.assertEqual(
            report.resolved, {"code_llama_34b_python": "Code Llama 34B Python"}
        )

    def test_a_model_the_paper_never_ran_is_unresolved_and_not_suspect(self):
        """The legitimate case: it should end up as ``extra``, not as a naming bug."""
        report = naming.resolve(["Some 2026 Model"])
        self.assertEqual(report.unresolved, ("Some 2026 Model",))
        self.assertEqual(report.suspect, {})
        self.assertTrue(report.clean)

    def test_a_near_miss_is_flagged_and_never_applied(self):
        report = naming.resolve(["CodeLlama-34b-Python-hf"])
        self.assertEqual(report.resolved, {})
        self.assertIn("Code Llama 34B Python", report.suspect["CodeLlama-34b-Python-hf"])
        self.assertFalse(report.clean)

    def test_a_caller_supplied_alias_closes_a_near_miss(self):
        report = naming.resolve(
            ["CodeLlama-34b-Python-hf"],
            aliases={"CodeLlama-34b-Python-hf": "Code Llama 34B Python"},
        )
        self.assertEqual(
            report.resolved, {"CodeLlama-34b-Python-hf": "Code Llama 34B Python"}
        )
        self.assertTrue(report.clean)

    def test_a_display_name_outranks_an_alias_that_normalizes_the_same_way(self):
        report = naming.resolve(["GPT-4"], aliases={"gpt4": "Claude 3 Opus"})
        self.assertEqual(report.resolved, {"GPT-4": "GPT-4"})

    def test_two_names_for_one_model_are_reported_as_a_collision(self):
        report = naming.resolve(["GPT-4 Turbo", "gpt-4-1106-preview"])
        self.assertEqual(
            report.collisions,
            {"GPT-4 Turbo": ("GPT-4 Turbo", "gpt-4-1106-preview")},
        )
        self.assertFalse(report.clean)

    def test_nothing_to_resolve_is_clean(self):
        self.assertTrue(naming.resolve([]).clean)


class NearMissTest(unittest.TestCase):
    def test_a_published_name_is_its_own_closest_match(self):
        self.assertEqual(naming.near_misses("StarCoder")[0], "StarCoder")

    def test_an_unrelated_name_matches_nothing(self):
        self.assertEqual(naming.near_misses("totally-unrelated-thing"), ())

    def test_the_cutoff_can_be_loosened(self):
        tight = naming.near_misses("Vicuna", cutoff=0.99)
        loose = naming.near_misses("Vicuna", cutoff=0.6)
        self.assertEqual(tight, ())
        self.assertIn("Vicuna 7B", loose)

    def test_the_number_of_suggestions_is_capped(self):
        self.assertLessEqual(len(naming.near_misses("Code Llama 30B Python")), 3)


class RenameTest(unittest.TestCase):
    def test_scores_come_back_keyed_by_published_names(self):
        scores, report = naming.rename({"gpt-4-1106-preview": 0.47, "GPT-4": 0.454})
        self.assertEqual(scores, {"GPT-4 Turbo": 0.47, "GPT-4": 0.454})
        self.assertTrue(report.clean)

    def test_an_unmatched_key_is_carried_through_rather_than_dropped(self):
        """Dropping it would turn a naming problem into a smaller comparison."""
        scores, report = naming.rename({"Some 2026 Model": 0.9})
        self.assertEqual(scores, {"Some 2026 Model": 0.9})
        self.assertEqual(report.unresolved, ("Some 2026 Model",))

    def test_a_collision_is_refused_rather_than_silently_resolved(self):
        with self.assertRaises(ValueError):
            naming.rename({"GPT-4 Turbo": 0.47, "gpt-4-1106-preview": 0.46})

    def test_an_empty_mapping_renames_to_an_empty_mapping(self):
        scores, report = naming.rename({})
        self.assertEqual((scores, report.unresolved), ({}, ()))


if __name__ == "__main__":
    unittest.main()
