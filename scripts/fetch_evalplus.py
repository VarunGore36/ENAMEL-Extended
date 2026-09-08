"""Download EvalPlus greedy samples for ENAMEL parity check."""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_DIR = _ROOT / "enamel_ext" / "data" / "cache" / "evalplus"

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

RELEASES_API = "https://api.github.com/repos/evalplus/evalplus/releases"


def get_asset_urls() -> dict[str, str]:
    with urllib.request.urlopen(RELEASES_API, timeout=30) as resp:
        releases = json.loads(resp.read())
    urls = {}
    for release in releases:
        for asset in release.get("assets", []):
            if asset["name"].endswith(".zip"):
                key = asset["name"].replace(".zip", "")
                urls[key] = asset["browser_download_url"]
    return urls


def download_and_extract(url: str, dest: Path) -> list[Path]:
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen(url, timeout=120) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                tmp.write(chunk)
        tmp.seek(0)
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dest)
            return [dest / name for name in zf.namelist()]


def load_samples(paths: list[Path]) -> dict[int, tuple[str, ...]]:
    samples: dict[int, list[str]] = {}
    for path in paths:
        if path.is_dir():
            for f in sorted(path.glob("*.py")):
                task_id = f.stem.replace("_", "/")
                pid = int(task_id.split("/")[-1])
                code = f.read_text()
                samples.setdefault(pid, []).append(code)
        elif path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    task_id = obj.get("task_id", "")
                    pid = int(task_id.split("/")[-1])
                    code = obj.get("solution", obj.get("completion", ""))
                    samples.setdefault(pid, []).append(code)
    return {pid: tuple(codes) for pid, codes in samples.items()}


def main():
    urls = get_asset_urls()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, archive_name in MODEL_MAP.items():
        out_dir = CACHE_DIR / archive_name
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  {model_name}: already downloaded")
            continue
        if archive_name not in urls:
            print(f"  {model_name}: archive not found")
            continue
        print(f"  {model_name}: downloading...")
        try:
            paths = download_and_extract(urls[archive_name], out_dir)
            samples = load_samples(paths)
            print(f"    extracted {sum(len(v) for v in samples.values())} samples for {len(samples)} problems")
        except Exception as e:
            print(f"    error: {e}")


if __name__ == "__main__":
    main()
