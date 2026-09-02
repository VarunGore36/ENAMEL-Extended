# 0003 — Data adapter: problems, references, generators

Status: accepted. Supersedes nothing. Code: `enamel_ext/data/`, tests
`tests/test_data.py` (41 tests).

This layer defines what a problem *is* for the rest of the harness. It executes
the benchmark's own generator source but never model output; that lives behind
the sandbox in `enamel_ext/measure/`.

## Nothing is vendored, and provenance is a required field

The `q-rz/enamel` license is still unconfirmed (README "Credit", open item), so
the repository carries no problems, references, or generators. Data is fetched
into a cache directory that `.gitignore` excludes, and `ENAMEL_EXT_DATA`
redirects the cache so a run can be pinned to a snapshot.

`Provenance` makes name, url, license and retrieval date mandatory, with
`license = "unknown"` allowed but reported through `redistributable = False`. The
point is that an unlicensed fetch cannot quietly become a redistributable
artifact just because it sat in the tree for a while.

## The reference is the oracle, so expected outputs are not stored

Section 2.4 of the README notes that the expert reference simultaneously defines
the efficiency ceiling and the correct answer. Storing expected outputs would
create a third artifact that can disagree with both. Instead the runner computes
them by executing the reference, which makes the coupling explicit rather than
hidden in a data file, and keeps the second-oracle work of milestone 5 a change
in one place.

## Two level kinds, because parity and reproducibility pull apart

`GeneratedLevel` stores `(scale, seeds)` and rebuilds inputs on demand.
`MaterializedLevel` stores concrete argument tuples. Both are needed:

- Generated levels are reproducible by construction, cheap to store, and let the
  complexity-fitting work of milestone 7 ask for scales the paper never used.
- Materialized levels are what parity requires. The published `t*` values were
  measured on specific inputs; regenerating from a seed reproduces them only if
  our generator, our CPython, and our `random` stream all match upstream's. That
  is not something to assume, so the schema can hold the exact inputs instead.

A problem may mix the two. Scale monotonicity is enforced across timed levels
when they are all generated, because levels that do not grow cannot separate
complexity classes and would make the `q` measurement of milestone 1 meaningless.
Level 0 is exempt: it filters correctness on small adversarial inputs and may be
larger than level 1.

## The cache format is JSON, and refuses to need unpickling

Upstream ships test inputs as a pickle. Unpickling is arbitrary code execution,
and a benchmark whose stated gap in the original is the absence of a sandbox
(README section 2.8) should not open with a deserialization hole. So the cache is
JSON and `problem_set_to_json` raises on anything it cannot represent; converting
a pickle is the fetch script's job, done once, in the open.

The cost is that JSON has no tuple. Argument tuples are stored as arrays and
restored as tuples at the top level only, so an inner tuple comes back as a list.
Real HumanEval signatures take lists, strings and numbers, so this has not bitten
yet, but it is a conversion the fetch step must handle if a problem ever wants a
tuple argument.

`MaterializedLevel` also rejects any case that is not a tuple. A list of scalars
where argument tuples were meant would otherwise call the entry point with the
wrong arity and be reported as a wrong answer rather than as bad data.

## One record adapter, and it will not guess

`problem_from_record` maps a flat record through an overridable field map and
requires levels to arrive already in the cache format. Converting a source's own
level layout is explicitly the fetch script's job.

This is deliberate. The upstream field and level layout cannot be inspected from
this environment (no network egress), so any mapping written here would be a
guess. Rather than bury a guess in the library, the adapter fails with the keys
it wanted and the keys it got, and `UPSTREAM_FIELDS` is labelled as unverified.
When the real dataset is in hand, one constant and one converter change.

## Fingerprints identify data, not runs

`ProblemSet.fingerprint()` is a SHA-256 over everything that can move a score:
ids, entry points, prompts, reference source, generator source, scales, seeds and
materialized inputs. It deliberately excludes provenance, so refetching the same
data on a later date compares equal and a cache can be validated against its own
contents. `problem_set_from_json` verifies the stored fingerprint, which turns a
hand-edited reference solution into an error instead of a silent reanchoring of
the efficiency ceiling.

It is a provenance digest, not a security digest. Materialized inputs are hashed
through `repr`, so an object whose repr embeds an address would not be stable;
the JSON restriction above keeps such objects out of a real cache.

## A synthetic problem set ships with the harness

`synthetic_problem_set()` builds a runnable, paper-shaped benchmark (8/4/4/4
cases, growing scales, a trivial `total(xs)` reference) with no external data.
This is what lets the sandboxed runner be built and tested while the real data is
unavailable, and it stays useful afterwards as a fast smoke subset for CI.

## Generator contract

A generator module defines `make_input(seed, scale)` returning a tuple of
positional arguments. `materialize_level` rejects a non-tuple return rather than
wrapping it, because guessing arity is how a data bug turns into a wrong-answer
verdict.

## Open items

- Confirm the `q-rz/enamel` license, then write the fetch script and pin a
  snapshot fingerprint in the repo so parity runs are checkable.
- Decide whether level 0's 8 cases are stored materialized for parity even when
  timed levels are generated. Correctness verdicts are the input to `pass@k`, so
  they are the part least tolerant of input drift.
- The Appendix C.1 "further calibrate" step is still unreconstructed. If it turns
  out to touch inputs rather than times, it belongs in this layer.
