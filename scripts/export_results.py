"""Export run records to a structured JSON for the results dashboard.

Reads all run records from the runs/ directory and produces a single
site/data/results.json consumed by the frontend.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.metrics.score import MetricConfig

RUNS_DIR = _ROOT / "runs"
OUTPUT = _ROOT / "site" / "data" / "results.json"

# Model classification
OPEN_SOURCE_MODELS = {
    "Code Llama 7B Python", "Code Llama 13B Python", "Code Llama 34B Python",
    "StarCoder", "StarCoder-Co-Manual (BigCode)",
    "CodeGen 2B", "CodeGen 6B", "CodeGen 16B",
    "CodeT5+ 16B", "CodeT5+ 2B", "CodeT5+ 6B",
    "Mistral 7B",
    "Vicuna 7B", "Vicuna 13B",
    "SantaCoder", "SantaCoder (BigCode)", "SantaCoder-PJJ (BigCode)",
    "Incoder 1B", "Incoder 6B",
    "GPT-J", "GPT-Neo 2B",
    "PolyCoder", "StableLM 7B",
    "Phind Code Llama V2",
    "CodeLlama 7B", "CodeLlama 13B", "CodeLlama 34B", "CodeLlama 70B",
    "StarCoder2-15B", "StarCoder2-7B", "StarCoder2-3B",
    "DeepSeek-Coder-V2", "DeepSeek-Coder-33B", "DeepSeek-Coder-6.7B",
    "Qwen2.5-Coder-32B", "Qwen2.5-Coder-7B", "Qwen2.5-Coder-3B",
    "Llama-3.1-70B", "Llama-3.1-8B",
    "Mixtral-8x22B", "Mixtral-8x7B",
    "WizardCoder-33B", "WizardCoder-15B",
    "OpenCoder-8B",
}

COMMERCIAL_MODELS = {
    "GPT-4", "GPT-4 Turbo", "GPT-4 (BigCode)",
    "ChatGPT", "GPT-3.5-Turbo",
    "Claude 3 Opus", "Claude 3 Sonnet", "Claude 3 Haiku", "Claude Sonnet 3.5",
}


def classify_model(name: str) -> str:
    if name in OPEN_SOURCE_MODELS:
        return "open"
    if name in COMMERCIAL_MODELS:
        return "commercial"
    # Heuristic: if it has (BigCode) in name, it's open
    if "(BigCode)" in name:
        return "open"
    return "unknown"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT, text=True
        ).strip()
        return bool(out)
    except Exception:
        return False


def score_from_times(level_times, correct, metric: MetricConfig, time_limit):
    if not correct:
        return 0.0
    scores = []
    for lt in level_times:
        if isinstance(lt, str) or any(isinstance(t, str) for t in lt):
            scores.append(0.0)
        else:
            from enamel_ext.metrics.score import level_fraction
            scores.append(level_fraction(lt, [max(lt)], metric.alpha))
    if not scores:
        return 0.0
    weights = list(metric.level_weights[:len(scores)])
    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


def extract_leaderboard(record: dict, k: int = 1) -> list[dict]:
    metric_cfg = MetricConfig(
        alpha=record["metric"]["alpha"],
        level_weights=tuple(record["metric"]["level_weights"]),
    )
    models = {}
    for prob in record["problems"]:
        pid = prob["problem_id"]
        tl = prob["time_limit"]
        for model_name, samples in prob.get("samples", {}).items():
            if model_name not in models:
                models[model_name] = {"effs": [], "correct": 0, "total": 0, "censored": 0}
            m = models[model_name]
            m["total"] += 1
            sample_scores = []
            for s in samples:
                correct = s["correct"]
                if correct:
                    m["correct"] += 1
                level_times = s["level_times"]
                flat = []
                for lt in level_times:
                    if isinstance(lt, str) or (isinstance(lt, list) and any(isinstance(t, str) for t in lt)):
                        flat.append(0.0)
                        if any(isinstance(t, str) and t == "censored" for t in lt):
                            m["censored"] += 1
                    else:
                        flat.append(max(lt) if lt else 0.0)
                sc = score_from_times(level_times, correct, metric_cfg, tl)
                sample_scores.append(sc)
            if sample_scores:
                m["effs"].append(max(sample_scores) if k == 1 else sum(sorted(sample_scores, reverse=True)[:k]) / k)

    rows = []
    for model_name, m in models.items():
        eff = sum(m["effs"]) / len(m["effs"]) if m["effs"] else 0.0
        total_samples = sum(len(prob.get("samples", {}).get(model_name, [])) for prob in record["problems"])
        pas = m["correct"] / total_samples if total_samples else 0.0
        rows.append({
            "model": model_name,
            "eff1": round(eff, 4),
            "pass1": round(pas, 4),
            "problems": len(m["effs"]),
            "correct": m["correct"],
            "censored": m["censored"],
            "type": classify_model(model_name),
        })
    rows.sort(key=lambda r: -r["eff1"])
    return rows


def extract_q_distribution(record: dict) -> list[dict]:
    from enamel_ext.report.levels import describe_levels
    timed_levels = []
    for prob in record["problems"]:
        ref = prob["reference_times"]
        if ref and len(ref) >= 2:
            levels = ref[1:]
            if all(isinstance(l, list) and not any(isinstance(t, str) for t in l) for l in levels):
                timed_levels.append(levels)
    if not timed_levels:
        return []
    summaries = describe_levels(timed_levels, alpha=record["metric"]["alpha"])
    return [
        {
            "level": s.level,
            "q_median": round(s.q_median, 4),
            "q_min": round(s.q_min, 4),
            "q_max": round(s.q_max, 4),
            "tolerated_slowdown": round(s.tolerated, 1),
            "score_at_2x": round(s.fractions.get(2.0, 0), 3),
            "score_at_5x": round(s.fractions.get(5.0, 0), 3),
            "score_at_10x": round(s.fractions.get(10.0, 0), 3),
        }
        for s in summaries
    ]


def export_record(record: dict, commit: str) -> dict:
    started = record.get("started", "")
    finished = record.get("finished", "")
    env = record.get("environment", {})
    seg = record.get("segments", [{}])[0]
    problems_scored = len(record.get("problems", []))
    attempted = len(record.get("attempted", []))
    failures = len(record.get("failures", []))

    leaderboard = extract_leaderboard(record)
    q_dist = extract_q_distribution(record)

    return {
        "experiment_id": f"{started[:19]}_{commit}",
        "timestamp": started,
        "finished": finished,
        "git_commit": commit,
        "git_dirty": git_dirty(),
        "benchmark": "ENAMEL-Extended v0.0.1",
        "python": env.get("python", ""),
        "platform": env.get("platform", ""),
        "cores": env.get("cores", 0),
        "metric": {
            "alpha": record["metric"]["alpha"],
            "level_weights": record["metric"]["level_weights"],
            "normalization": record["metric"].get("normalization", "global"),
            "repeats": record.get("measurement", {}).get("repeats", 6),
            "aggregator": record.get("measurement", {}).get("aggregator", "hodges_lehmann"),
        },
        "data": {
            "name": record.get("data", {}).get("provenance", {}).get("name", ""),
            "fingerprint": record.get("data", {}).get("fingerprint", "")[:12],
        },
        "solutions": {
            "name": record.get("solutions", {}).get("provenance", {}).get("name", ""),
            "fingerprint": record.get("solutions", {}).get("fingerprint", "")[:12],
        },
        "summary": {
            "problems_attempted": attempted,
            "problems_scored": problems_scored,
            "problems_failed": failures,
            "models": len(leaderboard),
        },
        "calibration": env.get("calibration", {}),
        "leaderboard": leaderboard,
        "q_distribution": q_dist,
    }


def main():
    if not RUNS_DIR.exists():
        print("no runs/ directory")
        return

    records = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                raw = json.load(f)
            commit = git_commit()
            records.append(export_record(raw, commit))
        except Exception as e:
            print(f"  skip {path.name}: {e}")

    if not records:
        print("no valid run records")
        return

    records.sort(key=lambda r: r["timestamp"], reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_commit": git_commit(),
        "run_count": len(records),
        "latest": records[0],
        "history": records,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
