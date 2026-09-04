# ENAMEL-Extended

A from-scratch Python reimplementation of **ENAMEL**, the code-efficiency benchmark introduced in *How Efficient is LLM-Generated Code? A Rigorous & High-Standard Benchmark* by **Qiu, Zeng, Ezick, Lott & Tong** (ICLR 2025 — [arXiv:2406.06647v4](https://arxiv.org/abs/2406.06647), [q-rz/enamel](https://github.com/q-rz/enamel)) — and a place to fix whatever we run into while building it. All credit for the benchmark, the `eff@k` metric, and the expert reference solutions belongs to them; see [Credit](#credit) for the citation and for what is ours versus theirs.

ENAMEL was the first serious attempt to measure the *efficiency* of LLM-generated code rather than only its correctness, and its headline claim — that models which look strong on `pass@k` are far from expert level once timed against expert references on adversarial inputs — is well supported. GPT-4 reaches `pass@1 = 0.831` but `eff@1 = 0.454`. That gap is real, the `eff@k` estimator behind it is sound, and the paper deserves credit for building an instrument sharp enough to see it. We think the paper is right.

**The primary goal is a faithful, working reimplementation.** Reading the paper closely enough to code it turns up places where a choice is underspecified, hardware-dependent, or sharper in the write-up than the underlying numbers support. Those live in [`docs/open-questions.md`](docs/open-questions.md): what ENAMEL does (§1), the decisions the paper leaves to the implementer (§2), and what we do about each one while building (§3). **`§2.x` anywhere in this repository means a section of that file.**

That document reads as criticism because that is the honest way to write down "here is what we had to decide and why the paper didn't decide it for us." It is not an argument that ENAMEL is wrong, and the fixes are a consequence of reimplementing carefully rather than the reason for doing it. Where one changes a number we report both ours and the original.

[`RESULTS.md`](RESULTS.md) is the other half: what has actually been established so far, grouped by how it was established. Nothing has been measured on the 142 problems yet.

## Status

Built and green: the metric core, the timing layer, the sandboxed runner, the data adapter with snapshot pinning, the reporting layer, and the pipeline that ties them together. Everything is stdlib-only and the suite runs with `python3 -m unittest discover -s tests -t .` (380 tests). Design rationale is in `docs/decisions/`, one file per decision.

Blocked, and not on code: the upstream snapshot needs network access this environment does not have, and no timing number from a 2-core VM is worth reporting. Parity and the §2.2 measurement are both waiting on the data and on hardware, not on more harness.

## Layout

```
enamel_ext/
  data/          problem schema, provenance, generators, JSON cache      [built]
  measure/       sandboxed runner, timing backends, repeat aggregation   [built]
  metrics/       eff@k estimator, censored scoring                       [built]
  report/        bootstrap CIs, tests, h-sweeps, level discrimination    [built]
  pipeline/      solution sets, run record, orchestrator, text report    [built]
  adversarial/   property-based + evolutionary per-candidate input search
  models/        sampling adapters, feedback-loop track
docs/
  open-questions.md  what the paper leaves to the implementer, and our answers
  decisions/     one file per methodological decision, with rationale
  analysis/      what the paper's own text and published numbers can be made to say
scripts/         fetch, recovery, and the evaluate entry point
tests/           harness unit tests + parity tests against published numbers
rpaper1.pdf      the paper itself
```

## Running it

`scripts/evaluate.py run` is the entry point: it measures each expert reference
once, scores every model's samples against that same measurement, writes a run
record, and prints the report. `scripts/evaluate.py report <record>` re-derives
the same report from a saved record, so changing `k`, the confidence level or the
bootstrap seed costs no measurement. The record stores times and never scores, so
the α sweep and the `h` stability section are the same measurements read at
another threshold rather than separate runs — with one asymmetry: α *above* the
one a run measured is refused, because a censored sample's true time was never
observed. With no `--problems`/`--solutions` it runs on a synthetic problem set,
which is how the tests exercise it end to end. See
`docs/decisions/0006-run-record.md`.

The data itself is not in the tree: `scripts/fetch_upstream.py` fetches one
pinned commit into a git-ignored cache, verifies it against a committed lock, and
`ENAMEL_EXT_DATA` repoints the cache at that snapshot. See
`docs/decisions/0003-data-adapter.md` and `0005-snapshot-pinning.md`, and the
license question under "Credit".

## Milestones

1. **Reimplement the metric.** Eq. (1)–(6) with `α=2, h=(3,3,4), R=6, M=(8,4,4,4)`, level 0 as correctness filter, `Tᵢ = 2·max` over all levels. Along the way: measure the distribution of `q = t*(level l) / t*(level 3)` across all 142 problems to settle §2.2. *(Estimator reproduced and checked exactly; the Appendix C.1 "further calibrate" step is [resolved](docs/analysis/appendix-c1-calibration.md) and needs no code; the `q` measurement is code waiting on data.)*
2. **Parity.** Reproduce the published ranking on our hardware within a stated tolerance. Document every discrepancy. **This gates the rest of the list.** *(Harness runs end to end and the snapshot is pinned; waiting on the data and a machine worth timing on.)*
3. **Reproducible measurement.** Containerized runner, sandbox, CPU pinning, instruction-count metric, cross-machine and cross-CPython rank-stability experiment. *(Process-level isolation in place; containerization and the instruction-count metric not started.)*
4. **Honest statistics.** Censored scoring, bootstrap CIs, full hyperparameter sweep across all models, pairwise significance tests. *(In every run's report already.)*
5. **Reference audit.** All 142 references reviewed; second oracle in place; disagreement rate published; anything we beat re-anchored.
6. **Adversarial generation.** Per-candidate worst-case search; quantify how much scores move versus the fixed generators.
7. **Two-axis and complexity reporting.** Memory axis; scaling-exponent fits with abstention.
8. **Contamination and feedback studies.** Paraphrase deltas on a fresh held-out set; profiler-in-the-loop track.

Milestones 1–2 are the reimplementation. Everything after that is optional and only worth doing if the earlier steps hold up.

Done, in order of what matters: the reimplementation runs and reproduces the paper; two different machines produce the same model ranking under our harness and we can show it; every reported score carries an interval and no comparison is claimed without a test; every threshold-dependent claim states the region of hyperparameter space where it holds. If we get that far, we can also put a number on how much of ENAMEL's measured "expert gap" is algorithmic shortfall versus artifacts of fixed test inputs, a single-machine clock, an unaudited reference, and a metric whose discrimination is concentrated in one of its four levels.

## Scope

**Python only, but deeper.** No C++ or Rust. Multi-language work would help separate algorithmic skill from Python-builtin familiarity, but it roughly doubles the harness surface and [arXiv:2505.13004](https://arxiv.org/abs/2505.13004) has already staked out that direction. Staying in Python also keeps direct comparison to the published numbers possible, which we need.

**Our own harness, the paper's data.** We write the harness from scratch rather than forking `q-rz/enamel`, partly because reimplementing is the point and partly because several §3 items (the censoring model, per-level normalization, candidate-adaptive generators, two-axis scoring) are structural rather than additive and would fight an existing design. But we reuse the paper's *data*: the 142 problems, the expert references, the original generators, as fixtures and as the baseline to reproduce. **Recovering the published numbers with the original method on our own hardware is the gate for everything else.** If we cannot reproduce the paper's ranking, we do not understand the method well enough to change it.

**Function-level scope retained.** Repository-level and I/O-bound efficiency are out of scope, as they are in the original.

## Non-goals

Multi-language benchmarking, repository-level or distributed performance, training or fine-tuning models for efficiency, and running a public leaderboard.

We are also not trying to show ENAMEL's conclusion is wrong, and we should be careful not to drift into it — the incentive to find something publishable is exactly how a reimplementation turns into a rebuttal it cannot support. Our prior is that the qualitative conclusion holds: the `pass@1`/`eff@1` gap is large, it survives every α in the paper's own sweep, and the estimator behind it checks out. What we expect to change is the precision of the specific numbers and possibly the ordering of adjacent models. If hardening the method leaves even those intact, we say so and the project is still worth having done.

---

## Credit

ENAMEL is the work of **Ruizhong Qiu, Weiliang Will Zeng, James Ezick, Christopher Lott, and Hanghang Tong** (University of Illinois Urbana–Champaign and Qualcomm AI Research), published at ICLR 2025. The benchmark, the `eff@k` metric, its Rao–Blackwellized estimator and proof, the 142-problem selection, the expert reference solutions, and the strong test-case generators are all theirs. This repository reimplements their method; it does not originate it.

The part most worth singling out is the one that cannot be automated: a human expert hand-wrote a best-known-algorithm reference solution *and* a hardened test-case generator for all 142 problems — Knuth–Morris–Pratt for #10, digit DP for #36, a suffix automaton for #154, Carmichael-number corner cases for #31, and so on. That is the labor the entire benchmark rests on, and §2.4's point about single-annotator calibration is a note about auditability, not a discount on the work.

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

**What is ours:** the harness in `enamel_ext/`, the measurement backends, the per-candidate adversarial search, the censored scoring variant, and the analysis in `docs/open-questions.md` and `docs/analysis/`. **What is theirs:** everything in `enamel_ext/data/` that we reuse as fixtures. ENAMEL in turn builds on HumanEval (Chen et al., MIT) and HumanEval+/EvalPlus (Liu et al., Apache-2.0), and per the paper's Appendix C.1 some of its reference solutions are modified from those canonical solutions.

**Open item:** confirm the `q-rz/enamel` repository license before we redistribute any of their problems, references, or generators in this repo, and carry their notices forward. Until then, treat `enamel_ext/data/` as fetched at setup time rather than vendored.

## References

- Qiu, Zeng, Ezick, Lott & Tong. *How Efficient is LLM-Generated Code? A Rigorous & High-Standard Benchmark.* ICLR 2025 — [arXiv:2406.06647](https://arxiv.org/abs/2406.06647) · [code](https://github.com/q-rz/enamel)
- Chen et al. *Evaluating Large Language Models Trained on Code* (HumanEval, `pass@k`) — [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- Liu et al. *Is Your Code Generated by ChatGPT Really Correct?* (HumanEval+ / EvalPlus) — [arXiv:2305.01210](https://arxiv.org/abs/2305.01210)
- Huang et al. *EffiBench: Benchmarking the Efficiency of Automatically Generated Code* — [arXiv:2402.02037](https://arxiv.org/abs/2402.02037)
- *A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code* — [arXiv:2505.13004](https://arxiv.org/abs/2505.13004)
- Hodges & Lehmann (1963), *Estimates of location based on rank tests*; Casella & Robert (1996), *Rao-Blackwellisation of sampling schemes* — the two statistical tools underpinning `eff@k`


