"""Convert BigCode evaluation samples to solution set format."""

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

CACHE_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "bigcode"
OUTPUT = _ROOT / "enamel_ext" / "data" / "cache" / "bigcode_solutions.json"


def load_bigcode_samples(path: Path, model_name: str) -> dict[int, tuple[str, ...]]:
    """Load samples from BigCode evaluation JSON format."""
    with open(path) as f:
        data = json.load(f)

    samples = {}
    for pid in range(len(data)):
        codes = [c for c in data[pid] if c.strip()]
        if codes:
            samples[pid] = tuple(codes)
    return samples


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    models = {}

    # Check for downloaded files
    files = {
        "SantaCoder (BigCode)": "santacoder_humaneval.json",
        "SantaCoder-PJJ (BigCode)": "santacoder_pjj_humaneval.json",
        "StarCoder (BigCode)": "starcoder_humaneval.json",
        "GPT-4 (BigCode)": "gpt4_humaneval.json",
        "StarCoder-Co-Manual (BigCode)": "starcoder_co_manual_humaneval.json",
    }

    for model_name, filename in files.items():
        path = CACHE_DIR / filename
        if path.exists():
            samples = load_bigcode_samples(path, model_name)
            if samples:
                models[model_name] = samples
                total = sum(len(v) for v in samples.values())
                print(f"  {model_name}: {len(samples)} problems, {total} samples")

    if not models:
        print("No samples found. Download from BigCode evaluation dataset first.")
        return

    solutions = SolutionSet(
        provenance=Provenance(
            name="BigCode Evaluation Harness",
            url="https://huggingface.co/datasets/bigcode/evaluation",
            license="MIT",
            retrieved=datetime.now(timezone.utc).isoformat(),
        ),
        samples=models,
    )

    OUTPUT.write_text(solution_set_to_json(solutions))
    print(f"\nwrote {len(models)} models to {OUTPUT}")


if __name__ == "__main__":
    main()
