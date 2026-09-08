"""Generate HumanEval samples from OpenAI-compatible APIs.

Outputs samples in EvalPlus directory format compatible with convert_evalplus.py.

Usage:
    python scripts/generate_samples.py --model gpt-4o --api-key sk-... --base-url https://api.openai.com/v1
    python scripts/generate_samples.py --model deepseek-coder --base-url https://api.deepseek.com/v1 --api-key sk-...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EVALPLUS_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "evalplus"


def get_humaneval_problems():
    """Load HumanEval problems from the upstream cache."""
    csv_path = _ROOT / "enamel_ext" / "data" / "cache" / "upstream" / "dataset" / "enamel.csv"
    import csv
    problems = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            task_id = row["task_id"]
            prompt = row["prompt"]
            entry_point = row["entry_point"]
            problems.append({
                "task_id": task_id,
                "prompt": prompt,
                "entry_point": entry_point,
            })
    return problems


def call_api(base_url: str, api_key: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    """Call an OpenAI-compatible API to generate a completion."""
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"Complete the Python function:\n\n{prompt}"}],
        "temperature": temperature,
        "max_tokens": 512,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def generate_samples(base_url: str, api_key: str, model: str, n_samples: int = 1) -> dict[str, list[str]]:
    """Generate samples for all HumanEval problems."""
    problems = get_humaneval_problems()
    samples = {}
    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        print(f"  [{i+1}/{len(problems)}] {task_id}...", end=" ", flush=True)
        codes = []
        for _ in range(n_samples):
            try:
                code = call_api(base_url, api_key, model, prob["prompt"])
                codes.append(code)
                print("ok", end="")
            except Exception as e:
                print(f"err: {e}", end="")
        print()
        samples[task_id] = codes
    return samples


def save_samples(samples: dict[str, list[str]], model_name: str):
    """Save samples in EvalPlus directory format."""
    out_dir = EVALPLUS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_id, codes in samples.items():
        task_dir = out_dir / task_id.replace("/", "_")
        task_dir.mkdir(exist_ok=True)
        for i, code in enumerate(codes):
            (task_dir / f"{i}.py").write_text(code)
    total = sum(len(c) for c in samples.values())
    print(f"Saved {total} samples for {len(samples)} problems to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model name/ID")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--output-name", default=None, help="Output directory name (default: model name)")
    parser.add_argument("--n-samples", type=int, default=1, help="Samples per problem")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key or OPENAI_API_KEY env var required")
        return 1

    output_name = args.output_name or args.model.replace("/", "_")
    print(f"Generating {args.n_samples} samples per problem for {args.model}...")
    samples = generate_samples(args.base_url, args.api_key, args.model, args.n_samples)
    save_samples(samples, output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
