# 0008 — Resolving model names onto the published tables

Status: accepted. Code: `enamel_ext/data/naming.py`, the parity section of
`enamel_ext/pipeline/summary.py`, tests `tests/test_naming.py` (24 tests).

Decision 0007 fixes what parity means once the two sides are lined up. This file
is about lining them up, which turned out to be the step most likely to fail
quietly. The gate compares by model name; the published tables are keyed by the
paper's display names (`GPT-4 Turbo`); a run's models are keyed by whatever
produced the samples. Nothing in the pipeline required those to be the same
string, so a mismatch emptied the comparison and reported nothing wrong.

## What the paper does and does not pin

Appendix C.1 names the checkpoint behind exactly five display names:
`claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`,
`gpt-4-1106-preview` and `gpt-4-0613`. Those are in `MODEL_IDENTIFIERS`.

For the rest it says only "for models that are included in Liu et al. (2023a), we
re-use their generated code samples", and does not list which models those are.
So for 25 of the 30 published rows the upstream name is not recoverable from the
paper at all: it is whatever the EvalPlus release called it. That is the reason
this cannot be a complete lookup table, and the reason the module is built to
report what it could not map rather than to guess. §2.9 of `open-questions.md`
records the same gap from the reader's side.

## Three tiers, and only the first two are automatic

**Normalization.** Casefold and drop everything that is not alphanumeric, so
`Code Llama 34B Python`, `code_llama_34b_python` and `CodeLlama34BPython` are one
key. This is not a guess: case, spacing and punctuation are formatting rather than
identity, and digits survive, so `CodeGen 16B` and `CodeGen 6B` stay distinct. The
scheme depends on no two published names colliding under it, which is asserted
against the real table rather than assumed.

**Stated aliases.** `ALIASES` is `MODEL_IDENTIFIERS` inverted and holds nothing
else. A caller can pass more, and a supplied alias naming a model the paper never
published is refused, because that typo would otherwise drop a model silently. A
display name outranks an alias that normalizes the same way.

**Everything else is reported, not resolved.** An unmatched name gets a
`difflib` similarity check against the published names, and the result splits two
ways that must not be merged: a name with no near miss is plausibly a model the
paper never ran, which is legitimate and ends up as `extra`; a name with a near
miss is probably a published model spelled differently, which is a bug. The
suggestion is printed and never applied. Whoever knows the provenance of the
samples supplies the alias.

## Two failures this closes

A run that shares no name with the published table used to produce **no parity
section at all**, because `summary._parity` returned early on an empty
intersection. Silence is the wrong output: it is indistinguishable from a run that
legitimately had nothing to compare, so the reader has no reason to look. The
section now prints whenever a name *looks* published, carrying the query and a
verdict that says nothing was compared.

Separately, `ParityResult.passed` was the conjunction of three absences, so a
comparison overlapping zero published models satisfied all three vacuously and
printed "compared 0 of 30 models" above "verdict: pass". It now requires at least
one model compared on both columns. That is deliberately not a coverage rule, and
decision 0007 keeps coverage out of the verdict: two models compared is weak
evidence and passes, nought models is not evidence and cannot.

## Collisions

Two run names can resolve to one published model, for instance `GPT-4 Turbo` and
`gpt-4-1106-preview` in the same run. `resolve` reports it and `rename` raises,
because rekeying would drop one score and there is no principled choice of which.

## Open items

- The 25 unpinned rows stay unpinned until the upstream sample sets are in hand.
  Their names come from the EvalPlus release, and reading them off that release is
  a snapshot question, not a paper question. Decision 0005 tracks the snapshot.
- The near-miss cutoff of 0.8 is a reporting threshold, not a criterion. It can
  only cause a suggestion to be printed or not printed, so it is safe to tune.
