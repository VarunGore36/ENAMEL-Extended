"""Adversarial input generation for candidate solutions.

Generates worst-case inputs by mutating existing test cases to maximize runtime.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.cases import materialize_level
from enamel_ext.data.sources import problem_set_from_json
from enamel_ext.measure.sandbox import run_level
from enamel_ext.measure.timing import aggregate_repeats


def mutate_input(args: tuple, mutation_rate: float = 0.1) -> tuple:
    """Mutate input arguments to find worst-case inputs."""
    mutated = []
    for arg in args:
        if isinstance(arg, list):
            new_arg = list(arg)
            for i in range(len(new_arg)):
                if random.random() < mutation_rate:
                    if isinstance(new_arg[i], (int, float)):
                        new_arg[i] = new_arg[i] * random.uniform(0.5, 2.0)
                        if isinstance(new_arg[i], int):
                            new_arg[i] = int(new_arg[i])
            mutated.append(type(arg)(new_arg))
        elif isinstance(arg, (int, float)):
            if random.random() < mutation_rate:
                mutated.append(arg * random.uniform(0.5, 2.0))
            else:
                mutated.append(arg)
        else:
            mutated.append(arg)
    return tuple(mutated)


def adversarial_search(
    code: str,
    entry_point: str,
    base_inputs: list[tuple],
    n_iterations: int = 100,
    repeats: int = 3,
) -> dict:
    """Search for worst-case inputs for a given solution."""
    best_time = 0.0
    best_input = None
    original_times = []

    # Measure original inputs
    for args in base_inputs[:3]:
        result = run_level(
            code, entry_point, inputs=[args], repeats=repeats, aggregator="hodges_lehmann"
        )
        if result.ok and result.cases:
            t = aggregate_repeats(result.cases[0].times, "hodges_lehmann")
            original_times.append(t)

    if not original_times:
        return {"status": "error", "detail": "no valid original inputs"}

    avg_original = sum(original_times) / len(original_times)

    # Mutate and search
    for i in range(n_iterations):
        base = random.choice(base_inputs[:3])
        mutated = mutate_input(base)

        result = run_level(
            code, entry_point, inputs=[mutated], repeats=repeats, aggregator="hodges_lehmann"
        )
        if result.ok and result.cases:
            t = aggregate_repeats(result.cases[0].times, "hodges_lehmann")
            if t > best_time:
                best_time = t
                best_input = mutated

    if best_time > avg_original:
        slowdown = best_time / avg_original
        return {
            "status": "found",
            "original_time": avg_original,
            "adversarial_time": best_time,
            "slowdown": slowdown,
            "adversarial_input": [str(x)[:100] for x in best_input],
        }
    return {
        "status": "no_improvement",
        "original_time": avg_original,
        "adversarial_time": best_time,
        "slowdown": best_time / avg_original if avg_original > 0 else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if args.problems:
        pset = problem_set_from_json(args.problems.read_text())
    else:
        from enamel_ext.data.sources import synthetic_problem_set
        pset = synthetic_problem_set()

    if args.limit > 0:
        from enamel_ext.data.schema import ProblemSet, Provenance
        pset = ProblemSet(provenance=pset.provenance, problems=pset.problems[:args.limit])

    print(f"Running adversarial search on {len(pset)} problems ({args.iterations} iterations each)...")

    results = []
    for p in pset:
        cases = materialize_level(p, p.levels[0]) if p.levels else []
        if not cases:
            continue

        result = adversarial_search(
            p.reference_solution, p.entry_point, list(cases), n_iterations=args.iterations
        )
        results.append({"problem_id": p.problem_id, **result})
        status = result["status"]
        slowdown = result.get("slowdown", 0)
        print(f"  Problem {p.problem_id}: {status} (slowdown: {slowdown:.2f}x)")

    found = [r for r in results if r["status"] == "found"]
    if found:
        avg_slowdown = sum(r["slowdown"] for r in found) / len(found)
        print(f"\nSummary:")
        print(f"  Problems tested: {len(results)}")
        print(f"  Found adversarial inputs: {len(found)}")
        print(f"  Average slowdown: {avg_slowdown:.2f}x")
    else:
        print(f"\nNo adversarial inputs found that improve on original.")


if __name__ == "__main__":
    main()
