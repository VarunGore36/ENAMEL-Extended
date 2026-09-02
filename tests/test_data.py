"""Tests for the problem/reference data adapter."""

from __future__ import annotations

import unittest

from enamel_ext.data import (
    PAPER_CASE_COUNTS,
    GeneratedLevel,
    JsonSource,
    MaterializedLevel,
    Problem,
    ProblemSet,
    Provenance,
    load_generator,
    materialize,
    materialize_level,
    problem_from_record,
    problem_set_from_json,
    problem_set_to_json,
    problems_from_records,
    synthetic_problem_set,
)

PROV = Provenance(name="test", url="local", license="Apache-2.0", retrieved="2026-09-02")

GEN = """
def make_input(seed, scale):
    return (list(range(seed, seed + scale)),)
"""


def _problem(pid: int = 0, **kw) -> Problem:
    defaults = dict(
        problem_id=pid,
        entry_point="total",
        prompt="def total(xs): ...",
        reference_solution="def total(xs):\n    return sum(xs)\n",
        input_generator=GEN,
        levels=tuple(
            GeneratedLevel(level=lvl, scale=10 ** (lvl + 1), seeds=tuple(range(lvl * 10, lvl * 10 + n)))
            for lvl, n in enumerate(PAPER_CASE_COUNTS)
        ),
    )
    defaults.update(kw)
    return Problem(**defaults)  # type: ignore[arg-type]


class TestProvenance(unittest.TestCase):
    def test_unknown_license_is_not_redistributable(self):
        prov = Provenance(name="x", url="y", license="unknown", retrieved="2026-09-02")
        self.assertFalse(prov.redistributable)
        self.assertTrue(PROV.redistributable)

    def test_rejects_empty_fields(self):
        for field in ("name", "url", "license", "retrieved"):
            kw = dict(name="x", url="y", license="z", retrieved="w")
            kw[field] = "  "
            with self.subTest(field=field), self.assertRaises(ValueError):
                Provenance(**kw)


class TestLevels(unittest.TestCase):
    def test_case_count_comes_from_the_spec(self):
        self.assertEqual(GeneratedLevel(level=1, scale=10, seeds=(1, 2, 3, 4)).n_cases, 4)
        self.assertEqual(MaterializedLevel(level=0, inputs=((1,), (2,))).n_cases, 2)

    def test_duplicate_seeds_collide_and_are_rejected(self):
        with self.assertRaises(ValueError):
            GeneratedLevel(level=1, scale=10, seeds=(1, 1, 2, 3))

    def test_rejects_empty_and_nonpositive_scale(self):
        with self.assertRaises(ValueError):
            GeneratedLevel(level=1, scale=0, seeds=(1,))
        with self.assertRaises(ValueError):
            GeneratedLevel(level=1, scale=10, seeds=())
        with self.assertRaises(ValueError):
            MaterializedLevel(level=0, inputs=())

    def test_sequences_are_normalized_to_tuples(self):
        lvl = GeneratedLevel(level=1, scale=10, seeds=[3, 4])
        self.assertEqual(lvl.seeds, (3, 4))

    def test_a_bare_scalar_is_not_an_argument_tuple(self):
        """A list of scalars where arg tuples were meant would silently call the
        entry point with the wrong arity."""
        with self.assertRaises(ValueError):
            MaterializedLevel(level=0, inputs=(1, 2, 3))


class TestProblem(unittest.TestCase):
    def test_paper_shape(self):
        p = _problem()
        self.assertEqual(p.case_counts, PAPER_CASE_COUNTS)
        self.assertEqual(p.n_timed_levels, 3)

    def test_levels_must_be_contiguous_from_zero(self):
        with self.assertRaises(ValueError):
            _problem(levels=(GeneratedLevel(level=1, scale=10, seeds=(1,)),))
        with self.assertRaises(ValueError):
            _problem(
                levels=(
                    GeneratedLevel(level=0, scale=10, seeds=(1,)),
                    GeneratedLevel(level=2, scale=20, seeds=(2,)),
                )
            )

    def test_timed_scales_must_increase(self):
        levels = (
            GeneratedLevel(level=0, scale=10, seeds=(1,)),
            GeneratedLevel(level=1, scale=100, seeds=(2,)),
            GeneratedLevel(level=2, scale=100, seeds=(3,)),
        )
        with self.assertRaises(ValueError):
            _problem(levels=levels)

    def test_level_zero_scale_is_unconstrained(self):
        """Level 0 filters correctness on small adversarial inputs, so it may be
        larger than level 1."""
        _problem(
            levels=(
                GeneratedLevel(level=0, scale=999, seeds=(1,)),
                GeneratedLevel(level=1, scale=10, seeds=(2,)),
                GeneratedLevel(level=2, scale=20, seeds=(3,)),
            )
        )

    def test_generated_levels_require_a_generator(self):
        with self.assertRaises(ValueError):
            _problem(input_generator="")

    def test_materialized_levels_do_not_require_a_generator(self):
        _problem(
            input_generator="",
            levels=(MaterializedLevel(level=0, inputs=((1, 2),)),),
        )

    def test_rejects_bad_entry_point_and_empty_reference(self):
        with self.assertRaises(ValueError):
            _problem(entry_point="not an identifier")
        with self.assertRaises(ValueError):
            _problem(reference_solution="   ")


class TestProblemSet(unittest.TestCase):
    def test_lookup_is_by_id_not_position(self):
        pset = ProblemSet(provenance=PROV, problems=(_problem(41), _problem(7)))
        self.assertEqual(pset[7].problem_id, 7)
        self.assertEqual(pset.ids(), (41, 7))
        with self.assertRaises(KeyError):
            pset[0]

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            ProblemSet(provenance=PROV, problems=(_problem(3), _problem(3)))

    def test_fingerprint_ignores_problem_order_and_provenance(self):
        a = ProblemSet(provenance=PROV, problems=(_problem(1), _problem(2)))
        b = ProblemSet(provenance=PROV, problems=(_problem(2), _problem(1)))
        other_prov = ProblemSet(
            provenance=Provenance(name="n", url="u", license="MIT", retrieved="2030-01-01"),
            problems=(_problem(1), _problem(2)),
        )
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.fingerprint(), other_prov.fingerprint())

    def test_fingerprint_tracks_anything_that_moves_a_score(self):
        base = ProblemSet(provenance=PROV, problems=(_problem(1),))
        changed = ProblemSet(
            provenance=PROV,
            problems=(_problem(1, reference_solution="def total(xs):\n    return 0\n"),),
        )
        reseeded = ProblemSet(
            provenance=PROV,
            problems=(
                _problem(
                    1,
                    levels=tuple(
                        GeneratedLevel(level=lvl, scale=10 ** (lvl + 1), seeds=tuple(range(n)))
                        for lvl, n in enumerate(PAPER_CASE_COUNTS)
                    ),
                ),
            ),
        )
        self.assertNotEqual(base.fingerprint(), changed.fingerprint())
        self.assertNotEqual(base.fingerprint(), reseeded.fingerprint())

    def test_require_paper_shape(self):
        full = ProblemSet(provenance=PROV, problems=tuple(_problem(i) for i in range(142)))
        full.require_paper_shape()
        with self.assertRaises(ValueError):
            ProblemSet(provenance=PROV, problems=(_problem(0),)).require_paper_shape()

    def test_require_paper_shape_checks_case_counts(self):
        wrong = tuple(
            _problem(
                i,
                levels=tuple(
                    GeneratedLevel(level=lvl, scale=10 ** (lvl + 1), seeds=tuple(range(n)))
                    for lvl, n in enumerate((8, 4, 4, 3))
                ),
            )
            for i in range(142)
        )
        with self.assertRaises(ValueError):
            ProblemSet(provenance=PROV, problems=wrong).require_paper_shape()


class TestSerialization(unittest.TestCase):
    def test_round_trip_preserves_the_fingerprint(self):
        pset = ProblemSet(provenance=PROV, problems=(_problem(0), _problem(1)))
        back = problem_set_from_json(problem_set_to_json(pset))
        self.assertEqual(back.fingerprint(), pset.fingerprint())
        self.assertEqual(back.provenance, pset.provenance)
        self.assertEqual(back[1].levels, pset[1].levels)

    def test_round_trip_of_materialized_inputs(self):
        pset = ProblemSet(
            provenance=PROV,
            problems=(
                _problem(
                    0,
                    input_generator="",
                    levels=(MaterializedLevel(level=0, inputs=(([1, 2], "ab"), ([], "c"))),),
                ),
            ),
        )
        back = problem_set_from_json(problem_set_to_json(pset))
        self.assertEqual(back[0].levels[0].inputs, (([1, 2], "ab"), ([], "c")))

    def test_a_tampered_cache_is_rejected(self):
        pset = ProblemSet(provenance=PROV, problems=(_problem(0),))
        text = problem_set_to_json(pset).replace("return sum(xs)", "return 0")
        with self.assertRaises(ValueError):
            problem_set_from_json(text)

    def test_unknown_schema_version_is_rejected(self):
        pset = ProblemSet(provenance=PROV, problems=(_problem(0),))
        text = problem_set_to_json(pset).replace('"schema_version": 1', '"schema_version": 99')
        with self.assertRaises(ValueError):
            problem_set_from_json(text)

    def test_nonserializable_input_fails_loudly(self):
        pset = ProblemSet(
            provenance=PROV,
            problems=(
                _problem(
                    0,
                    input_generator="",
                    levels=(MaterializedLevel(level=0, inputs=((object(),),)),),
                ),
            ),
        )
        with self.assertRaises(TypeError):
            problem_set_to_json(pset)


class TestRecordAdapter(unittest.TestCase):
    RECORD = {
        "problem_id": 5,
        "entry_point": "total",
        "prompt": "def total(xs): ...",
        "reference_solution": "def total(xs):\n    return sum(xs)\n",
        "input_generator": GEN,
        "levels": [
            {"kind": "generated", "level": 0, "scale": 10, "seeds": [1, 2]},
            {"kind": "generated", "level": 1, "scale": 100, "seeds": [3, 4]},
        ],
    }

    def test_maps_a_well_formed_record(self):
        p = problem_from_record(self.RECORD)
        self.assertEqual(p.problem_id, 5)
        self.assertEqual(p.case_counts, (2, 2))

    def test_honours_a_field_rename(self):
        renamed = dict(self.RECORD)
        renamed["task_id"] = renamed.pop("problem_id")
        fields = {
            "problem_id": "task_id",
            "entry_point": "entry_point",
            "prompt": "prompt",
            "reference_solution": "reference_solution",
            "input_generator": "input_generator",
            "levels": "levels",
        }
        self.assertEqual(problem_from_record(renamed, fields).problem_id, 5)

    def test_a_missing_field_names_what_it_wanted_and_what_it_got(self):
        incomplete = {k: v for k, v in self.RECORD.items() if k != "reference_solution"}
        with self.assertRaises(KeyError) as ctx:
            problem_from_record(incomplete)
        self.assertIn("reference_solution", str(ctx.exception))

    def test_generator_is_optional(self):
        record = dict(self.RECORD)
        record.pop("input_generator")
        record["levels"] = [{"kind": "materialized", "level": 0, "inputs": [[1, 2]]}]
        self.assertEqual(problem_from_record(record).input_generator, "")

    def test_unknown_level_kind_is_rejected(self):
        record = dict(self.RECORD)
        record["levels"] = [{"kind": "pickled", "level": 0}]
        with self.assertRaises(ValueError):
            problem_from_record(record)

    def test_problems_from_records(self):
        second = dict(self.RECORD, problem_id=6)
        pset = problems_from_records([self.RECORD, second], PROV)
        self.assertEqual(pset.ids(), (5, 6))


class TestMaterialize(unittest.TestCase):
    def test_generated_cases_are_reproducible(self):
        p = _problem()
        self.assertEqual(materialize_level(p, p.levels[1]), materialize_level(p, p.levels[1]))

    def test_case_count_and_scale_match_the_spec(self):
        p = _problem()
        for level in p.levels:
            with self.subTest(level=level.level):
                cases = materialize_level(p, level)
                self.assertEqual(len(cases), level.n_cases)
                self.assertEqual(len(cases[0][0]), level.scale)

    def test_materialized_levels_pass_through(self):
        inputs = ((1, 2), (3, 4))
        p = _problem(input_generator="", levels=(MaterializedLevel(level=0, inputs=inputs),))
        self.assertEqual(materialize_level(p, p.levels[0]), inputs)

    def test_materialize_covers_every_level(self):
        p = _problem()
        self.assertEqual(tuple(len(c) for c in materialize(p)), PAPER_CASE_COUNTS)

    def test_generator_without_the_entry_point_is_rejected(self):
        with self.assertRaises(ValueError):
            load_generator("def other(seed, scale):\n    return (seed,)\n")

    def test_generator_must_return_a_tuple_of_arguments(self):
        p = _problem(input_generator="def make_input(seed, scale):\n    return [seed]\n")
        with self.assertRaises(ValueError):
            materialize_level(p, p.levels[0])


class TestSyntheticProblemSet(unittest.TestCase):
    def test_is_runnable_and_paper_shaped(self):
        pset = synthetic_problem_set(n_problems=2)
        self.assertEqual(len(pset), 2)
        for p in pset:
            with self.subTest(pid=p.problem_id):
                self.assertEqual(p.case_counts, PAPER_CASE_COUNTS)
                self.assertEqual(len(materialize(p)), 4)

    def test_round_trips_through_the_cache_format(self):
        pset = synthetic_problem_set(n_problems=2)
        self.assertEqual(
            problem_set_from_json(problem_set_to_json(pset)).fingerprint(), pset.fingerprint()
        )

    def test_seeds_do_not_collide_across_problems_or_levels(self):
        pset = synthetic_problem_set(n_problems=4)
        seen = [s for p in pset for lvl in p.levels for s in lvl.seeds]
        self.assertEqual(len(set(seen)), len(seen))


class TestJsonSource(unittest.TestCase):
    def test_missing_cache_says_how_to_fix_it(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            JsonSource("/nonexistent/problems.json").load()
        self.assertIn("ENAMEL_EXT_DATA", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
