"""Fetch a pinned upstream snapshot, verify its digest, and record the pin.

Downloads one commit's tarball, extracts only regular files, and writes a lock
that later runs check the cache against. See docs/decisions/0003-data-adapter.md.
Never unpickles anything: conversion is a separate, explicit step.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from enamel_ext.data.pin import (  # noqa: E402
    Extraction,
    Lock,
    Manifest,
    PinError,
    digest_file,
    extract_archive,
)
from enamel_ext.data.sources import default_cache_dir  # noqa: E402

DEFAULT_MANIFEST = _ROOT / "enamel_ext" / "data" / "upstream.manifest.json"
EXAMPLE_MANIFEST = _ROOT / "enamel_ext" / "data" / "upstream.manifest.example.json"
DEFAULT_LOCK = _ROOT / "enamel_ext" / "data" / "upstream.lock.json"

_USER_AGENT = "enamel-ext-fetch/0.0.1 (+https://github.com/q-rz/enamel)"
_ALLOWED_SCHEMES = ("https", "http", "file")
_CHUNK = 1 << 20

OK = 0
INTEGRITY_FAILURE = 1
USAGE_FAILURE = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Exit 1: no verified snapshot. Exit 2: fix the manifest or the flags.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--dest", type=Path, default=None, help="cache root (default: the data cache dir)"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--archive", type=Path, default=None, help="use a local tarball instead of downloading"
    )
    parser.add_argument(
        "--allow-new-pin",
        action="store_true",
        help="record the digest of what was fetched as the pin (trust on first use)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify", action="store_true", help="check the cache against the lock, no network"
    )
    mode.add_argument(
        "--inventory", action="store_true", help="list what the cache holds, no network"
    )
    return parser


def _read_manifest(path: Path) -> Manifest:
    if not path.is_file():
        raise PinError(
            f"no manifest at {path}. Copy {EXAMPLE_MANIFEST.name} to {path.name} and pin a commit."
        )
    return Manifest.from_json(path.read_text(encoding="utf-8"))


def _read_lock(path: Path) -> Lock | None:
    if not path.is_file():
        return None
    return Lock.from_json(path.read_text(encoding="utf-8"))


def _download(url: str, target: Path, timeout: float) -> None:
    scheme = urlsplit(url).scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise PinError(f"refusing to fetch a {scheme or 'schemeless'} url: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with open(target, "wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)


def _verify(lock: Lock | None, dest: Path) -> int:
    if lock is None:
        print("no lock recorded, so there is nothing to verify", file=sys.stderr)
        return USAGE_FAILURE
    problems = lock.verify_files(dest)
    pin = f"{lock.repo}@{lock.revision[:12]}"
    if problems:
        print(f"{len(problems)} problem(s) in {dest} against {pin}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return INTEGRITY_FAILURE
    print(f"{len(lock.files)} file(s) in {dest} match {pin}, retrieved {lock.retrieved}")
    return OK


#: Formats whose loaders can execute code from the file, so this tool only ever
#: reports that they are present.
_EXECUTABLE_DATA = (".pkl", ".pickle", ".pt", ".pth", ".joblib", ".npy", ".npz")


def _inventory(dest: Path) -> int:
    if not dest.is_dir():
        print(f"nothing at {dest}", file=sys.stderr)
        return USAGE_FAILURE
    files = sorted(path for path in dest.rglob("*") if path.is_file())
    if not files:
        print(f"{dest} is empty")
        return OK
    print(f"{len(files)} file(s) under {dest}:")
    for path in files[:500]:
        print(f"  {path.relative_to(dest).as_posix()}  {path.stat().st_size}")
    if len(files) > 500:
        print(f"  ... and {len(files) - 500} more")
    risky = [
        path.relative_to(dest).as_posix()
        for path in files
        if path.suffix.lower() in _EXECUTABLE_DATA
    ]
    if risky:
        print(
            f"\n{len(risky)} file(s) in a format whose loader executes code "
            f"({', '.join(sorted(set(Path(p).suffix for p in risky)))}). "
            f"This tool does not load them; converting one is a separate step that "
            f"should read it in a sandbox."
        )
    return OK


def _pin_to_check(manifest: Manifest, lock: Lock | None) -> str | None:
    return manifest.archive_sha256 or (lock.archive_sha256 if lock is not None else None)


def _report(
    lock: Lock, extraction: Extraction, dest: Path, lock_path: Path, pinned: bool
) -> None:
    print(f"{lock.repo}@{lock.revision[:12]} -> {dest}")
    print(f"  archive  {lock.archive_sha256} ({lock.archive_bytes} bytes)")
    print(f"  files    {len(lock.files)} extracted, {len(extraction.skipped)} skipped")
    for entry in extraction.skipped[:10]:
        print(f"           skipped {entry}")
    print(f"  lock     {lock_path}")
    if not pinned:
        print(
            "  pin      recorded on first use; copy archive_sha256 into the manifest "
            "to make it a hard requirement"
        )
    if lock.license == "unknown":
        print("  license  unconfirmed, so this snapshot must not be committed to this repository")
    print("Next: --inventory to see the layout, then write the adapter for it.")


def _fetch(manifest: Manifest, lock: Lock | None, dest: Path, args: argparse.Namespace) -> int:
    if lock is not None:
        why = lock.mismatch_with(manifest)
        if why and not args.allow_new_pin:
            print(
                f"manifest and lock disagree: {why}. Re-run with --allow-new-pin to "
                f"move the pin, or restore the manifest.",
                file=sys.stderr,
            )
            return USAGE_FAILURE
        if why:
            lock = None
    expected = _pin_to_check(manifest, lock)
    if expected is None and not args.allow_new_pin:
        print(
            "no pin to check: the manifest records no archive_sha256 and there is no "
            "lock. Re-run with --allow-new-pin to record what is fetched as the pin.",
            file=sys.stderr,
        )
        return USAGE_FAILURE

    with tempfile.TemporaryDirectory() as staging:
        archive = Path(staging) / "snapshot.tar.gz"
        if args.archive is not None:
            shutil.copyfile(args.archive, archive)
            source = str(args.archive)
        else:
            source = manifest.resolved_archive_url()
            _download(source, archive, args.timeout)
        found, size = digest_file(archive)
        if expected is not None and found != expected:
            print(
                f"digest mismatch for {source}\n  expected {expected}\n  found    {found}",
                file=sys.stderr,
            )
            return INTEGRITY_FAILURE
        try:
            extraction = extract_archive(archive, dest, manifest.include)
        except PinError as exc:
            print(f"refusing this archive: {exc}", file=sys.stderr)
            return INTEGRITY_FAILURE

    new_lock = Lock.record(
        manifest, archive_sha256=found, archive_bytes=size, files=extraction.files
    )
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(new_lock.to_json(), encoding="utf-8")
    _report(new_lock, extraction, dest, args.lock, pinned=expected is not None)
    return OK


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dest = args.dest if args.dest is not None else default_cache_dir() / "upstream"
    try:
        if args.inventory:
            return _inventory(dest)
        lock = _read_lock(args.lock)
        if args.verify:
            return _verify(lock, dest)
        return _fetch(_read_manifest(args.manifest), lock, dest, args)
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_FAILURE
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the snapshot: {exc}", file=sys.stderr)
        return INTEGRITY_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
