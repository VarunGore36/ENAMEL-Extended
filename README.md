# ENAMEL-Extended

**Are open source code models actually efficient? Let's find out.**

[![Tests](https://img.shields.io/badge/tests-612%20passing-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-orange)]()
[![arXiv](https://img.shields.io/badge/arXiv-2406.06647-b31b1b)](https://arxiv.org/abs/2406.06647)

> Most code models pass tests but write slow code. This project measures **how efficient** open source LLM-generated code really is — not just whether it works.

**Live results → [enamel-extended.vercel.app](https://enamel-extended.vercel.app)**

---

## The problem

[ENAMEL](https://arxiv.org/abs/2406.06647) (ICLR 2025) showed that `pass@1` hides inefficiency — GPT-4 passes 83% of tests but only matches expert efficiency on 45% of problems. **This project asks the same question for open source models.**

Can you trust an open source model to write fast code? Or does it just write code that passes tests but runs 10× slower than it should?

---

## Results

### Open source model efficiency (161 problems, 13 open source models)

| Rank | Model | Size | eff@1 | pass@1 | Gap |
|------|-------|------|-------|--------|-----|
| 1 | Code Llama 7B | 7B | 0.328 | 0.398 | 0.070 |
| 2 | StarCoder | 15B | 0.277 | 0.379 | 0.102 |
| 3 | CodeGen 6B | 6B | 0.260 | 0.348 | 0.088 |
| 4 | CodeGen 16B | 16B | 0.249 | 0.335 | 0.086 |
| 5 | Mistral 7B | 7B | 0.244 | 0.317 | 0.073 |
| 6 | CodeT5+ 16B | 16B | 0.234 | 0.342 | 0.108 |
| 7 | SantaCoder | 1B | 0.179 | 0.193 | 0.014 |
| 8 | Vicuna 13B | 13B | 0.170 | 0.217 | 0.047 |
| 9 | Incoder 6B | 6B | 0.162 | 0.180 | 0.018 |
| 10 | GPT-J | 6B | 0.124 | 0.130 | 0.006 |
| 11 | Incoder 1B | 1B | 0.115 | 0.130 | 0.015 |
| 12 | Vicuna 7B | 7B | 0.099 | 0.137 | 0.038 |
| 13 | GPT-Neo 2B | 2B | 0.095 | 0.106 | 0.011 |

### Key findings

**1. No open source model reaches expert efficiency.**
The best open source model (Code Llama 7B) only matches expert efficiency on 33% of problems. Even passing tests doesn't mean your code is fast.

**2. The pass@1/eff@1 gap is real.**
Every model scores lower on efficiency than on correctness. The gap ranges from 0.01 (GPT-J) to 0.11 (CodeT5+ 16B).

**3. Bigger isn't always better.**
CodeGen 6B outperforms CodeGen 16B on efficiency. Mistral 7B beats Vicuna 13B. Size alone doesn't predict code efficiency.

**4. Specialized models help.**
Code-focused models (Code Llama, StarCoder, CodeGen) outperform general models (Vicuna, GPT-J, GPT-Neo) on efficiency.

### Reference: commercial models (from ENAMEL paper)

| Model | pass@1 | eff@1 | Gap |
|-------|--------|-------|-----|
| GPT-4 Turbo | 0.796 | 0.470 | 0.326 |
| GPT-4 | 0.831 | 0.454 | 0.377 |
| Llama 3 70B | 0.746 | 0.421 | 0.325 |
| Mixtral 8x22B | 0.746 | 0.408 | 0.338 |

Commercial models are more efficient but still far from expert level.

### q-Distribution analysis

| Level | q median | Tolerated slowdown | Sensitivity share |
|-------|----------|-------------------|-------------------|
| 1 | 0.30 | 6.7× | 9% |
| 2 | 0.53 | 3.8× | 19% |
| 3 | 1.00 | 2.0× | **71%** |

**Level 3 carries 71% of the score's sensitivity** — the hardest test cases matter most.

---

## Quick start

```bash
# Fetch upstream data
python3 scripts/fetch_upstream.py --allow-new-pin
python3 scripts/convert_upstream.py

# Download pre-generated samples
python3 scripts/fetch_evalplus.py
python3 scripts/convert_evalplus.py

# Run evaluation
python3 scripts/evaluate.py run \
    --problems enamel_ext/data/cache/problems.json \
    --solutions enamel_ext/data/cache/evalplus_solutions.json \
    --limit 50 --keep-going

# Generate report
python3 scripts/evaluate.py report runs/run-*.json
```

---

## Architecture

```
enamel_ext/
  data/          Problem schema, generators, published tables
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
```

### Design principles

- **Zero runtime dependencies** — metric core runs on bare CPython
- **Resumable runs** — every problem flushes to disk; crash costs one problem
- **Calibration probe** — every session detects machine drift
- **Censored scoring** — timeouts score 0, not "infinity"
- **Bootstrap CIs** — every score carries an interval

---

## Adding models

```bash
# From EvalPlus releases
python3 scripts/fetch_evalplus.py
python3 scripts/convert_evalplus.py

# From local model (needs GPU)
pip install evalplus[vllm]
evalplus.codegen --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --dataset humaneval --backend vllm --greedy

# From API
python3 scripts/generate_samples.py --model codellama --base-url http://localhost:8000/v1
```

---

## What's measured

Every `eff@1` score comes with:

- **Bootstrap 95% confidence interval** (10,000 resamples)
- **Pairwise significance test** against every other model
- **α-sweep** — score at timeout factors 1.25, 1.5, 2.0
- **h-sweep** — score under every non-negative hardness weight
- **Level discrimination** — q-distribution, sensitivity shares
- **Calibration probe** — machine drift detection per session

---

## Scope

**Python only.** Keeps direct comparison to published numbers possible.

**Function-level only.** Repository-level efficiency is out of scope.

**Open source models preferred.** Reproducible, no API costs, anyone can verify.

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Reimplement the metric | ✅ Done |
| 2 | Parity gate | ✅ Done — 17 models, τ=0.783 |
| 3 | Reproducible measurement | ⏳ Containerization, CPU pinning |
| 4 | Honest statistics | ✅ Done — CIs in every report |
| 5 | Reference audit | ⏳ Review 142 references |
| 6 | Adversarial generation | ⏳ Per-candidate worst-case search |
| 7 | Two-axis reporting | ⏳ Memory axis, complexity fits |
| 8 | Contamination studies | ⏳ Paraphrase deltas |

---

## Credit

ENAMEL is the work of **Ruizhong Qiu, Weiliang Will Zeng, James Ezick, Christopher Lott, and Hanghang Tong** (UIUC + Qualcomm AI Research), published at ICLR 2025.

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
