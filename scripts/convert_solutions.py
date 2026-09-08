"""Convert upstream samples to solution set format."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.schema import Provenance
from enamel_ext.pipeline.solutions import SolutionSet, solution_set_to_json

UPSTREAM_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "upstream"
OUTPUT = _ROOT / "enamel_ext" / "data" / "cache" / "solutions.json"


def convert():
    refs_path = UPSTREAM_DIR / "samples" / "enamel-references.json"
    humaneval_path = UPSTREAM_DIR / "samples" / "humaneval-canonical.json"

    with open(refs_path, "r") as f:
        references = json.load(f)

    with open(humaneval_path, "r") as f:
        humaneval = json.load(f)

    samples: dict[str, dict[int, tuple[str, ...]]] = {}

    samples["reference"] = {}
    for idx, refs in enumerate(references):
        samples["reference"][idx] = tuple(refs)

    samples["humaneval-canonical"] = {}
    for idx, sols in enumerate(humaneval):
        samples["humaneval-canonical"][idx] = tuple(sols)

    solutions = SolutionSet(
        provenance=Provenance(
            name="ENAMEL upstream samples",
            url="https://github.com/q-rz/enamel",
            license="unknown",
            retrieved=datetime.now(timezone.utc).isoformat(),
        ),
        samples=samples,
    )

    OUTPUT.write_text(solution_set_to_json(solutions))
    print(f"wrote {len(samples)} models, {len(samples['reference'])} problems to {OUTPUT}")


if __name__ == "__main__":
    convert()
