"""Audit reference solutions for correctness.

Runs each reference solution against its test cases and reports any failures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.cases import materialize_level
from enamel_ext.data.sources import problem_set_from_json
from enamel_ext.measure.sandbox import run_level


def audit_reference(problem, verbose: bool = False) -> dict:
    """Audit a single reference solution."""
    results = {
        "problem_id": problem.problem_id,
        "entry_point": problem.entry_point,
        "levels_correct": 0,
        "levels_total": 0,
        "cases_correct": 0,
        "cases_total": 0,
        "errors": [],
    }

    for level_idx, level in enumerate(problem.levels):
        results["levels_total"] += 1
        cases = materialize_level(problem, level)
        level_correct = True

        for case_idx, args in enumerate(cases):
            results["cases_total"] += 1
            result = run_level(
                problem.reference_solution,
                problem.entry_point,
                inputs=[args],
                repeats=1,
                aggregator="hodges_lehmann",
            )

            if result.ok and result.cases and result.cases[0].status == "ok":
                results["cases_correct"] += 1
            else:
                level_correct = False
                error = {
                    "level": level_idx,
                    "case": case_idx,
                    "status": result.status,
                    "detail": result.detail[:200] if result.detail else "",
                }
                results["errors"].append(error)
                if verbose:
                    print(f"  FAIL problem {problem.problem_id} level {level_idx} case {case_idx}: {error['detail']}")

        if level_correct:
            results["levels_correct"] += 1

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.problems:
        pset = problem_set_from_json(args.problems.read_text())
    else:
        from enamel_ext.data.sources import synthetic_problem_set
        pset = synthetic_problem_set()

    if args.limit > 0:
        from enamel_ext.data.schema import ProblemSet, Provenance
        pset = ProblemSet(provenance=pset.provenance, problems=pset.problems[:args.limit])

    print(f"Auditing {len(pset)} reference solutions...")

    all_results = []
    for p in pset:
        result = audit_reference(p, verbose=args.verbose)
        all_results.append(result)
        if result["errors"]:
            print(f"  Problem {p.problem_id}: {len(result['errors'])} errors")

    total_cases = sum(r["cases_total"] for r in all_results)
    correct_cases = sum(r["cases_correct"] for r in all_results)
    total_levels = sum(r["levels_total"] for r in all_results)
    correct_levels = sum(r["levels_correct"] for r in all_results)
    problems_with_errors = sum(1 for r in all_results if r["errors"])

    print(f"\nSummary:")
    print(f"  Problems audited: {len(all_results)}")
    print(f"  Cases: {correct_cases}/{total_cases} correct ({100*correct_cases/total_cases:.1f}%)")
    print(f"  Levels: {correct_levels}/{total_levels} correct ({100*correct_levels/total_levels:.1f}%)")
    print(f"  Problems with errors: {problems_with_errors}")

    if problems_with_errors:
        print(f"\nProblems with errors:")
        for r in all_results:
            if r["errors"]:
                print(f"  Problem {r['problem_id']} ({r['entry_point']}): {len(r['errors'])} errors")


if __name__ == "__main__":
    main()
