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
    santacoder_path = CACHE_DIR / "santacoder_humaneval.json"
    santacoder_pjj_path = CACHE_DIR / "santacoder_pjj_humaneval.json"
    starcoder_path = CACHE_DIR / "starcoder_humaneval.json"

    if santacoder_path.exists():
        samples = load_bigcode_samples(santacoder_path, "SantaCoder")
        if samples:
            models["SantaCoder (BigCode)"] = samples
            print(f"  SantaCoder (BigCode): {len(samples)} problems, {sum(len(v) for v in samples.values())} samples")

    if santacoder_pjj_path.exists():
        samples = load_bigcode_samples(santacoder_pjj_path, "SantaCoder-PJJ")
        if samples:
            models["SantaCoder-PJJ (BigCode)"] = samples
            print(f"  SantaCoder-PJJ (BigCode): {len(samples)} problems, {sum(len(v) for v in samples.values())} samples")

    if starcoder_path.exists():
        samples = load_bigcode_samples(starcoder_path, "StarCoder")
        if samples:
            models["StarCoder (BigCode)"] = samples
            print(f"  StarCoder (BigCode): {len(samples)} problems, {sum(len(v) for v in samples.values())} samples")

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
