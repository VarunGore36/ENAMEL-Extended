"""Contamination study: paraphrase problem descriptions and compare scores.

Tests whether models are memorizing problem descriptions vs understanding algorithms.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.schema import Problem, ProblemSet, Provenance
from enamel_ext.data.sources import problem_set_from_json


def paraphrase_prompt(prompt: str) -> str:
    """Simple paraphrasing of problem prompts."""
    # Rename common variables
    replacements = [
        (r'\bnumbers\b', 'nums'),
        (r'\bstring\b', 'text'),
        (r'\blist\b', 'array'),
        (r'\bfloat\b', 'decimal'),
        (r'\bint\b', 'integer'),
        (r'\bbool\b', 'boolean'),
        (r'\bList\b', 'Array'),
        (r'\bstr\b', 'String'),
        (r'\bdef (\w+)', r'def \1'),
        (r'"""([^"]+)"""', r'"""\1"""'),
    ]

    result = prompt
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    # Add a note that this is paraphrased
    if '"""' in result:
        result = result.replace('"""', '""" [Paraphrased] ', 1)

    return result


def create_paraphrased_problems(pset: ProblemSet) -> ProblemSet:
    """Create paraphrased versions of all problems."""
    new_problems = []
    for p in pset:
        paraphrased_prompt = paraphrase_prompt(p.prompt)
        new_problem = Problem(
            problem_id=p.problem_id + 1000,  # Offset IDs
            entry_point=p.entry_point,
            prompt=paraphrased_prompt,
            reference_solution=p.reference_solution,
            input_generator=p.input_generator,
            levels=p.levels,
        )
        new_problems.append(new_problem)

    return ProblemSet(
        provenance=Provenance(
            name="ENAMEL-Extended paraphrased",
            url="local",
            license="Apache-2.0",
            retrieved="2026-01-01",
        ),
        problems=tuple(new_problems),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paraphrased_problems.json"))
    args = parser.parse_args()

    if args.problems:
        pset = problem_set_from_json(args.problems.read_text())
    else:
        from enamel_ext.data.sources import synthetic_problem_set
        pset = synthetic_problem_set()

    if args.limit > 0:
        from enamel_ext.data.schema import ProblemSet as PS, Provenance
        pset = PS(provenance=pset.provenance, problems=pset.problems[:args.limit])

    print(f"Creating paraphrased versions of {len(pset)} problems...")

    paraphrased = create_paraphrased_problems(pset)

    # Show examples
    for orig, para in zip(pset.problems[:3], paraphrased.problems[:3]):
        print(f"\n--- Problem {orig.problem_id} ---")
        print(f"Original: {orig.prompt[:100]}...")
        print(f"Paraphrased: {para.prompt[:100]}...")

    # Save
    from enamel_ext.data.sources import problem_set_to_json
    output = problem_set_to_json(paraphrased)
    args.output.write_text(output)
    print(f"\nSaved {len(paraphrased)} paraphrased problems to {args.output}")


if __name__ == "__main__":
    main()
