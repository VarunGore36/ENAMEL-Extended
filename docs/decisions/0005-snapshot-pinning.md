# 0005 — Snapshot pinning and the fetch tool

Status: accepted. Code: `enamel_ext/data/pin.py`,
`enamel_ext/data/upstream.manifest.example.json`,
`scripts/fetch_upstream.py`, tests `tests/test_pin.py` (25 tests).

Parity (milestone 2) compares our numbers against published ones, which is only
meaningful if the inputs are the same inputs. This layer is what makes "the same
inputs" checkable: it fetches one immutable snapshot of the upstream repository,
records digests, and fails loudly when a later run would silently measure
something else.

## The pin is a commit, and separately the bytes

`Manifest` refuses anything but a full 40-character commit sha. A branch or a tag
resolves to different content over time, so a manifest naming `main` records
nothing; the error says so and prints the `git ls-remote` invocation that yields
a usable value. Short shas are refused too, since they can become ambiguous as
the repository grows.

The commit is not by itself enough. GitHub generates the `tar.gz` on demand, and
those bytes have changed in the past for reasons unrelated to any repository's
contents, so the archive digest and the commit pin different things: the commit
pins the tree, the digest pins the file that was actually verified. The lock
therefore records both, plus a per-file digest for everything extracted. That
last part is what survives a repack: if the same commit is later served as a
differently compressed archive, the archive digest disagrees while every file
digest still matches, and the two facts together say "repacked", not "tampered
with". A tool that recorded only the archive digest could not tell those apart.

A digest mismatch is reported with both values and stops before extraction. It is
not resolved automatically in either direction, because both readings are
plausible and only the operator can decide which one applies.

## Trust on first use has to be asked for

The first fetch has nothing to compare against. The tool refuses it unless
`--allow-new-pin` is passed, and the same flag is required to move an existing
pin. Without that gate the failure mode is quiet: a fetch that silently records
whatever it received looks exactly like a fetch that verified something, and the
lock file then carries the authority of a check that never happened.

Moving a pin discards the old file digests rather than merging them, since they
describe a tree that is no longer the one being fetched.

## Extraction never writes through a name from the archive

`extract_archive` computes each destination itself and streams the member into
it. `tarfile.extractall` honours the member name, so `../../.ssh/authorized_keys`
in a crafted tarball is a file write outside the destination; the filters that
fix this arrived in Python 3.12 and this targets 3.10, so the guard is explicit
rather than delegated. Names are normalized and anything absolute, containing a
`..` component, or carrying a backslash raises rather than being skipped, because
a traversal attempt is a reason to abandon the archive rather than a member to
pass over.

Symlinks, hard links, devices and fifos are skipped and reported. Recreating a
symlink is the other half of the traversal problem: a link written first turns a
later ordinary write into a write wherever it points.

Bytes are counted as they are written and capped, since a header's declared size
is just another field the archive controls.

## The lock detects three different things, and says which

A manifest repointed without a re-fetch, the same commit serving different bytes,
and a cache edited after extraction are three separate problems with three
separate remedies, so `mismatch_with` and `verify_files` report them separately
instead of collapsing into "stale".

`verify_files` also reports files present under the cache that the lock does not
list. Extraction writes exactly the set the lock records, so anything else
arrived by another route, and a stray file that the harness later reads is the
kind of difference that makes a parity number wrong for a reason nobody can find
afterwards.

## The upstream layout lives in the manifest, not in the code

The paper documents no file layout: the only pointer is "available at
https://github.com/q-rz/enamel", and this environment has no egress to inspect
the repository. Any path written into the library would therefore be a guess, and
a guess in code is one that has to be found and edited later, in an unknown
number of places.

So the manifest carries the fnmatch patterns, the default takes the whole tree,
and `--inventory` prints what actually arrived. The layout-specific work then
happens once, against a real listing, in the same place the guesswork already
lives: `UPSTREAM_FIELDS` in `sources.py` carries the same caveat.

## Nothing is loaded, only reported

Decision 0003 makes the cache JSON-only to keep unpickling out of the harness.
This tool holds that line: it never imports or deserializes what it downloads.
`--inventory` names files whose loaders can execute code and says it did not
open them, so the sandbox-shaped work of converting one stays a separate,
deliberate step rather than a side effect of fetching.

## Open items

- The `q-rz/enamel` license is still unconfirmed, so an `unknown` license is
  allowed and reported: fetching into a local cache is fine, committing the
  result is not, and the report says so on every run.
- The record-to-`Problem` converter still waits on a real inventory. Until then
  the fetch stops at a verified, listed cache.
- File digests cover what `include` selected. Narrowing the patterns after a
  fetch shrinks what the lock can attest to, which is an argument for keeping the
  default broad until the layout is settled.
- No resume and no partial-download recovery: an interrupted fetch is retried
  from the start. The archive is a few megabytes, so this is not worth the state.
- Untested against the real endpoint. Egress is blocked in the development VM, so
  the tests drive the same code path over `file://` URLs and local archives; the
  first real run is on the user's machine.
