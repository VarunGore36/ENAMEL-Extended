# ENAMEL-Extended

**A rigorous reimplementation of the ENAMEL code-efficiency benchmark.**

[![Tests](https://img.shields.io/badge/tests-612%20passing-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2406.06647-b31b1b)](https://arxiv.org/abs/2406.06647)

> **pass@1 hides inefficiency.** GPT-4 reaches `pass@1 = 0.831` but only `eff@1 = 0.454`.
> ENAMEL-Extended reproduces this finding, builds a resumable measurement harness with
> bootstrap confidence intervals, and investigates the published claims.

**Live results → [enamel-extended.vercel.app](https://enamel-extended.vercel.app)**

---

## What is this?

[ENAMEL](https://arxiv.org/abs/2406.06647) (ICLR 2025) is the first serious benchmark for measuring the *efficiency* of LLM-generated code — not just whether it passes tests, but how fast it runs compared to expert-written solutions on adversarial inputs.

**ENAMEL-Extended** is a from-scratch Python reimplementation of ENAMEL. We reproduce the metric, build a measurement harness, and investigate the paper's claims with statistical rigor. All credit for the benchmark, the `eff@k` metric, and the expert reference solutions belongs to the original authors.

### Key finding from the paper

| Model | pass@1 | eff@1 | Gap |
|-------|--------|-------|-----|
| GPT-4 Turbo | 0.796 | 0.470 | 0.326 |
| GPT-4 | 0.831 | 0.454 | 0.377 |
| Llama 3 70B | 0.746 | 0.421 | 0.325 |
| Mixtral 8x22B | 0.746 | 0.408 | 0.338 |

Models that look strong on correctness are far from expert-level efficiency.

---

## Results

### Our evaluation (34 problems, 17 models)

| Rank | Model | eff@1 | pass@1 | Source |
|------|-------|-------|--------|--------|
| 1 | Phind Code Llama V2 | 0.735 | 0.882 | EvalPlus |
| 2 | ChatGPT | 0.682 | 0.882 | EvalPlus |
| 3 | GPT-4 | 0.676 | 0.941 | EvalPlus |
| 4 | GPT-4 Turbo | 0.668 | 0.882 | EvalPlus |
| 5 | Code Llama 7B | 0.500 | 0.647 | EvalPlus |
| 6 | Mistral 7B | 0.473 | 0.588 | EvalPlus |
| 7 | CodeGen 6B | 0.426 | 0.529 | EvalPlus |
| 8 | CodeGen 16B | 0.418 | 0.618 | EvalPlus |
| 9 | StarCoder | 0.397 | 0.529 | EvalPlus |
| 10 | CodeT5+ 16B | 0.347 | 0.529 | EvalPlus |

**Ranking preserved:** Kendall τ = 0.783 against published Table 3. Absolute values differ due to hardware/CPython differences.

### q-Distribution analysis (§2.2 — answered with data)

| Level | q median | Tolerated slowdown | Sensitivity share |
|-------|----------|-------------------|-------------------|
| 1 | 0.30 | 6.7× | 9% |
| 2 | 0.53 | 3.8× | 19% |
| 3 | 1.00 | 2.0× | **71%** |

**Level 3 carries 71% of the score's sensitivity** while levels 1–2 carry 60% of the weight but minimal discrimination. The "ordinal in disguise" concern from §2.2 is confirmed.

---

## Quick start

```bash
# Fetch upstream data
python3 scripts/fetch_upstream.py --allow-new-pin

# Convert to internal format
python3 scripts/convert_upstream.py
python3 scripts/convert_evalplus.py

# Run evaluation
python3 scripts/evaluate.py run \
    --problems enamel_ext/data/cache/problems.json \
    --solutions enamel_ext/data/cache/evalplus_solutions.json \
    --limit 50 --keep-going

# Generate report from existing run
python3 scripts/evaluate.py report runs/run-*.json

# Update website data
python3 scripts/export_results.py
```

---

## Architecture

```
enamel_ext/
  data/          Problem schema, provenance, generators, published tables
  measure/       Sandboxed runner, timing backends, calibration probe
  metrics/       eff@k estimator, censored scoring
  report/        Bootstrap CIs, h-sweeps, levels, parity comparison
  pipeline/      Solution sets, run record, resumable orchestrator

scripts/
  evaluate.py         Entry point: measure + score + report
  fetch_upstream.py   Fetch pinned upstream snapshot
  convert_*.py        Convert samples from various sources
  export_results.py   Export run records to website JSON
  generate_samples.py Generate samples from OpenAI-compatible APIs

site/                 Live results dashboard (Vercel)
  index.html          Overview, results, experiments, methodology, status
  data/results.json   Auto-generated from run records
```

### Design principles

- **Zero runtime dependencies** — metric core runs on bare CPython for reproducible timing
- **Resumable runs** — every problem flushes to disk; crash costs at most one problem
- **Calibration probe** — every session times a fixed workload to detect machine drift
- **Censored scoring** — timeouts score 0, not "infinity"; survival analysis, not averaging
- **Bootstrap CIs** — every reported score carries an interval; no comparison without a test

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Reimplement the metric | ✅ Done — estimator verified exactly |
| 2 | Parity gate | 🔶 Partial — 17/30 models, ranking preserved |
| 3 | Reproducible measurement | ⏳ Pending — containerization, CPU pinning |
| 4 | Honest statistics | ✅ Done — bootstrap CIs in every report |
| 5 | Reference audit | ⏳ Pending — review 142 references |
| 6 | Adversarial generation | ⏳ Pending — per-candidate worst-case search |
| 7 | Two-axis reporting | ⏳ Pending — memory axis, complexity fits |
| 8 | Contamination studies | ⏳ Pending — paraphrase deltas, feedback loop |

---

## Adding new models

### From EvalPlus releases (pre-generated)
```bash
python3 scripts/fetch_evalplus.py
python3 scripts/convert_evalplus.py
python3 scripts/evaluate.py run --solutions enamel_ext/data/cache/evalplus_solutions.json
```

### From OpenAI-compatible API
```bash
python3 scripts/generate_samples.py \
    --model gpt-4o \
    --api-key sk-... \
    --base-url https://api.openai.com/v1
```

### From local model (needs GPU)
```bash
pip install evalplus[vllm]
evalplus.codegen --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --dataset humaneval --backend vllm --greedy
```

---

## What's measured

Every `eff@k` score comes with:

- **Bootstrap 95% confidence interval** (10,000 resamples)
- **Pairwise significance test** against every other model
- **α-sweep** — score at timeout factors 1.25, 1.5, 2.0
- **h-sweep** — score under every non-negative hardness weight
- **Level discrimination** — q-distribution, sensitivity shares, tolerated slowdown
- **Calibration probe** — machine drift detection per session

---

## Scope

**Python only, but deeper.** No C++ or Rust. Staying in Python keeps direct comparison to the published numbers possible.

**Function-level only.** Repository-level and I/O-bound efficiency are out of scope.

**Open source models preferred.** Reproducible, no API costs, anyone can verify our numbers. Closed models are evaluated when pre-generated samples are available.

---

## Credit

ENAMEL is the work of **Ruizhong Qiu, Weiliang Will Zeng, James Ezick, Christopher Lott, and Hanghang Tong** (UIUC + Qualcomm AI Research), published at ICLR 2025. The benchmark, the `eff@k` metric, its Rao–Blackwellized estimator, the 142-problem selection, the expert reference solutions, and the test-case generators are all theirs.

If you use anything here, cite the original paper:

```bibtex
@inproceedings{qiu2025enamel,
  title     = {How Efficient is {LLM}-Generated Code? A Rigorous \& High-Standard Benchmark},
  author    = {Qiu, Ruizhong and Zeng, Weiliang Will and Ezick, James and
               Lott, Christopher and Tong, Hanghang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2406.06647}
}
```

**What is ours:** the harness, measurement backends, adversarial search, censored scoring, and analysis. **What is theirs:** the data, the metric, and the reference solutions.

---

## References

- Qiu et al. [How Efficient is LLM-Generated Code?](https://arxiv.org/abs/2406.06647) — ICLR 2025
- Chen et al. [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) — HumanEval
- Liu et al. [Is Your Code Generated by ChatGPT Really Correct?](https://arxiv.org/abs/2305.01210) — HumanEval+
- Hodges & Lehmann (1963) — rank tests; Casella & Robert (1996) — Rao-Blackwellisation
