"""Convert EvalPlus samples to solution set format for parity check."""

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

EVALPLUS_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "evalplus"
OUTPUT = _ROOT / "enamel_ext" / "data" / "cache" / "evalplus_solutions.json"

MODEL_MAP = {
    "GPT-4 Turbo": "gpt-4-1106-preview_temp_0.0",
    "GPT-4": "gpt-4_temp_0.0",
    "Phind Code Llama V2": "phind-code-llama-34b-v2_temp_0.0",
    "ChatGPT": "chatgpt_temp_0.0",
    "Code Llama 34B Python": "code-llama-34b_temp_0",
    "Code Llama 13B Python": "code-llama-13b_temp_0",
    "Code Llama 7B Python": "code-llama-7b_temp_0.0",
    "StarCoder": "starcoder_temp_0.0",
    "CodeGen 16B": "codegen-16b_temp_0.0",
    "CodeGen 6B": "codegen-6b_temp_0.0",
    "CodeGen 2B": "codegen-2b_temp_0",
    "CodeT5+ 16B": "codet5p-16b_temp_0.0",
    "Mistral 7B": "mistral-7b_temp_0.0",
    "Vicuna 13B": "vicuna-13b_temp_0.0",
    "Vicuna 7B": "vicuna-7b_temp_0.0",
    "SantaCoder": "santacoder_temp_0.0",
    "Incoder 6B": "incoder-6b_temp_0.0",
    "Incoder 1B": "incoder-1b_temp_0.0",
    "GPT-J": "gpt-j_temp_0.0",
    "GPT-Neo 2B": "gptneo-2b_temp_0.0",
    "PolyCoder": "polycoder_temp_0.0",
    "StableLM 7B": "stablelm-7b_temp_0.0",
}


def load_model_samples(archive_name: str) -> dict[int, tuple[str, ...]]:
    model_dir = EVALPLUS_DIR / archive_name
    if not model_dir.exists():
        return {}

    def find_task_dirs(root: Path) -> list[Path]:
        found = []
        for p in root.rglob("HumanEval_*"):
            if p.is_dir() and p.name.split("_")[1].isdigit():
                found.append(p)
        return found

    task_dirs = find_task_dirs(model_dir)
    if not task_dirs:
        return {}

    samples: dict[int, list[str]] = {}
    for task_dir in sorted(task_dirs):
        pid = int(task_dir.name.split("_")[1])
        if pid in samples:
            continue
        codes = []
        for py_file in sorted(task_dir.glob("*.py")):
            codes.append(py_file.read_text())
        if codes:
            samples[pid] = codes

    return {pid: tuple(codes) for pid, codes in samples.items()}


def main():
    samples: dict[str, dict[int, tuple[str, ...]]] = {}

    for model_name, archive_name in MODEL_MAP.items():
        model_samples = load_model_samples(archive_name)
        if model_samples:
            n_problems = len(model_samples)
            n_samples = sum(len(v) for v in model_samples.values())
            print(f"  {model_name}: {n_problems} problems, {n_samples} total samples")
            samples[model_name] = model_samples
        else:
            print(f"  {model_name}: not downloaded")

    if not samples:
        print("No samples found. Run fetch_evalplus.py first.")
        return

    solutions = SolutionSet(
        provenance=Provenance(
            name="EvalPlus (evalplus/evalplus)",
            url="https://github.com/evalplus/evalplus",
            license="MIT",
            retrieved=datetime.now(timezone.utc).isoformat(),
        ),
        samples=samples,
    )

    OUTPUT.write_text(solution_set_to_json(solutions))
    print(f"\nwrote {len(samples)} models to {OUTPUT}")


if __name__ == "__main__":
    main()
