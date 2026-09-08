"""Download and convert HumanEval samples from HuggingFace datasets."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "huggingface"
OUTPUT = _ROOT / "enamel_ext" / "data" / "cache" / "hf_solutions.json"

# Models to download: (name, dataset_id, filename)
MODELS = [
    ("Qwen2.5-Coder-32B-Instruct", "test-gen/humaneval_Qwen2.5-Coder-32B-Instruct_t0.1_n8_generated_code", "data/test-00000-of-00001.parquet"),
    ("Qwen2.5-Coder-7B-Instruct", "test-gen/humaneval_Qwen2.5-Coder-7B-Instruct_t0.1_n8_generated_code", "data/test-00000-of-00001.parquet"),
    ("Qwen2.5-Coder-3B-Instruct", "test-gen/humaneval_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code", "data/test-00000-of-00001.parquet"),
]


def extract_functions_from_parquet(path: Path) -> dict[int, list[str]]:
    """Extract Python function definitions from a parquet file."""
    with open(path, 'rb') as f:
        data = f.read()
    
    text = data.decode('utf-8', errors='ignore')
    
    # Find all function definitions
    pattern = r'(def\s+\w+\s*\([^)]*\):.*?)(?=\ndef\s+\w+\s*\(|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Group by problem (164 problems, 8 samples each = 1312 functions)
    samples = {}
    for i, code in enumerate(matches):
        pid = i // 8  # 8 samples per problem
        if pid not in samples:
            samples[pid] = []
        samples[pid].append(code.strip())
    
    return samples


def download_dataset(dataset_id: str, filename: str, dest: Path) -> bool:
    """Download a file from HuggingFace."""
    if dest.exists():
        return True
    url = f'https://huggingface.co/datasets/{dataset_id}/resolve/main/{filename}'
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        print(f'  Error downloading: {e}')
        return False


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    from enamel_ext.data.schema import Provenance
    from enamel_ext.pipeline.solutions import SolutionSet, solution_set_to_json
    from datetime import datetime, timezone
    
    models = {}
    
    for model_name, dataset_id, filename in MODELS:
        print(f'Processing {model_name}...')
        dest = CACHE_DIR / f'{model_name}.parquet'
        
        if not download_dataset(dataset_id, filename, dest):
            continue
        
        samples = extract_functions_from_parquet(dest)
        if samples:
            # Convert to tuple format
            model_samples = {pid: tuple(codes) for pid, codes in samples.items()}
            models[model_name] = model_samples
            total = sum(len(v) for v in model_samples.values())
            print(f'  Extracted {len(model_samples)} problems, {total} samples')
        else:
            print(f'  No functions found')
    
    if not models:
        print('No models downloaded')
        return
    
    solutions = SolutionSet(
        provenance=Provenance(
            name="HuggingFace test-gen datasets",
            url="https://huggingface.co/test-gen",
            license="unknown",
            retrieved=datetime.now(timezone.utc).isoformat(),
        ),
        samples=models,
    )
    
    OUTPUT.write_text(solution_set_to_json(solutions))
    print(f'\nwrote {len(models)} models to {OUTPUT}')


if __name__ == "__main__":
    main()
