"""Manifest validation, digest locking and archive extraction."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from enamel_ext.data.pin import (
    Extraction,
    FileRecord,
    Lock,
    Manifest,
    PinError,
    digest_file,
    extract_archive,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "enamel_ext" / "data" / "upstream.manifest.example.json"

SHA_A = "a" * 40
SHA_B = "b" * 40


def _manifest(**kw) -> Manifest:
    fields = {"name": "upstream", "repo": "q-rz/enamel", "revision": SHA_A}
    fields.update(kw)
    return Manifest(**fields)


def _tarball(path: Path, files: dict[str, bytes], root: str = "enamel-abc123") -> Path:
    """A GitHub-shaped tarball: everything under a single top-level directory."""
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def _special(path: Path, info: tarfile.TarInfo, root: str = "enamel-abc123") -> Path:
    """A tarball holding one non-regular member plus one ordinary file."""
    with tarfile.open(path, "w:gz") as tar:
        payload = b"ok\n"
        plain = tarfile.TarInfo(f"{root}/plain.txt")
        plain.size = len(payload)
        tar.addfile(plain, io.BytesIO(payload))
        tar.addfile(info)
    return path


def _script():
    """scripts/fetch_upstream.py loaded by path, since scripts/ is not a package."""
    location = REPO_ROOT / "scripts" / "fetch_upstream.py"
    spec = importlib.util.spec_from_file_location("fetch_upstream", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestManifest(unittest.TestCase):
    def test_a_movable_ref_is_refused_with_the_reason(self):
        """A branch resolves to different bytes over time, so it is not a snapshot."""
        for revision in ("main", "master", "HEAD", "v1.0", SHA_A[:7], ""):
            with self.assertRaises(PinError) as caught:
                _manifest(revision=revision)
            self.assertIn("revision", str(caught.exception))
        with self.assertRaises(PinError) as caught:
            _manifest(revision="main")
        self.assertIn("ls-remote", str(caught.exception))

    def test_a_commit_is_accepted_and_normalized(self):
        self.assertEqual(_manifest(revision=SHA_A.upper()).revision, SHA_A)
        self.assertEqual(_manifest().url, "https://github.com/q-rz/enamel")
        self.assertFalse(_manifest().redistributable)
        self.assertTrue(_manifest(license="MIT").redistributable)

    def test_the_archive_url_names_the_commit(self):
        url = _manifest().resolved_archive_url()
        self.assertIn(SHA_A, url)
        self.assertIn("q-rz/enamel", url)
        override = _manifest(archive_url="file:///x.tgz")
        self.assertEqual(override.resolved_archive_url(), "file:///x.tgz")

    def test_notes_are_allowed_but_stray_keys_are_not(self):
        """JSON has no comments, and a pin file is exactly the kind of file that
        needs explaining, so ``_``-prefixed keys are notes."""
        text = json.dumps({"_note": "why", "name": "u", "repo": "a/b", "revision": SHA_A})
        self.assertEqual(Manifest.from_json(text).name, "u")
        with self.assertRaises(PinError) as caught:
            Manifest.from_json(
                json.dumps({"name": "u", "repo": "a/b", "revision": SHA_A, "sha": 1})
            )
        self.assertIn("sha", str(caught.exception))
        with self.assertRaises(PinError) as caught:
            Manifest.from_json(json.dumps({"name": "u"}))
        self.assertIn("repo", str(caught.exception))
        with self.assertRaises(PinError):
            Manifest.from_json("{not json")
        with self.assertRaises(PinError):
            Manifest.from_json("[]")

    def test_a_manifest_round_trips(self):
        original = _manifest(
            license="MIT", include=("data/*.json", "LICENSE"), archive_sha256="0" * 64
        )
        self.assertEqual(Manifest.from_json(original.to_json()), original)

    def test_the_shipped_example_refuses_to_fetch_until_a_commit_is_pinned(self):
        """The placeholder must fail loudly rather than 404 halfway through a fetch."""
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(raw["repo"], "q-rz/enamel")
        self.assertEqual(raw["license"], "unknown")
        with self.assertRaises(PinError) as caught:
            Manifest.from_json(EXAMPLE.read_text(encoding="utf-8"))
        self.assertIn("revision", str(caught.exception))

    def test_a_bad_pin_is_refused_before_anything_is_fetched(self):
        with self.assertRaises(PinError):
            _manifest(archive_sha256="not-a-digest")
        with self.assertRaises(PinError):
            _manifest(include=())
        with self.assertRaises(PinError):
            _manifest(repo="enamel")


class TestLock(unittest.TestCase):
    def _lock(self, **kw) -> Lock:
        default_files = (FileRecord("b.txt", "2" * 64, 1), FileRecord("a.txt", "3" * 64, 2))
        return Lock.record(
            _manifest(**kw.pop("manifest", {})),
            archive_sha256=kw.pop("archive_sha256", "1" * 64),
            archive_bytes=kw.pop("archive_bytes", 10),
            files=kw.pop("files", default_files),
            retrieved=kw.pop("retrieved", "2026-01-02"),
        )

    def test_a_lock_round_trips_with_its_files_ordered(self):
        lock = self._lock()
        self.assertEqual([rec.path for rec in lock.files], ["a.txt", "b.txt"])
        self.assertEqual(Lock.from_json(lock.to_json()), lock)

    def test_provenance_is_as_of_the_fetch_not_as_of_now(self):
        provenance = self._lock().provenance()
        self.assertEqual(provenance.retrieved, "2026-01-02")
        self.assertFalse(provenance.redistributable)

    def test_a_moved_pin_is_reported_rather_than_silently_used(self):
        lock = self._lock()
        self.assertEqual(lock.mismatch_with(_manifest()), "")
        self.assertIn("lock records", lock.mismatch_with(_manifest(revision=SHA_B)))
        self.assertIn("manifest pins", lock.mismatch_with(_manifest(archive_sha256="4" * 64)))

    def test_a_future_lock_schema_is_refused(self):
        raw = json.loads(self._lock().to_json())
        raw["schema_version"] = 99
        with self.assertRaises(PinError):
            Lock.from_json(json.dumps(raw))
        with self.assertRaises(PinError):
            Lock.from_json(json.dumps({**raw, "schema_version": 1, "files": [{"path": "x"}]}))

    def test_verification_sees_missing_altered_and_added_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_bytes(b"aa")
            lock = self._lock(
                files=(FileRecord("a.txt", digest_file(root / "a.txt")[0], 2),
                       FileRecord("gone.txt", "5" * 64, 1)),
            )
            self.assertEqual(lock.verify_files(root), ("missing: gone.txt",))
            (root / "a.txt").write_bytes(b"ab")
            (root / "extra.txt").write_bytes(b"x")
            self.assertEqual(
                sorted(lock.verify_files(root)),
                ["changed: a.txt", "missing: gone.txt", "unexpected: extra.txt"],
            )


class TestExtraction(unittest.TestCase):
    def test_the_wrapper_directory_is_stripped_and_digests_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = _tarball(root / "s.tgz", {"data/x.json": b"[1]", "LICENSE": b"apache\n"})
            result = extract_archive(archive, root / "out")
            self.assertIsInstance(result, Extraction)
            self.assertEqual([rec.path for rec in result.files], ["LICENSE", "data/x.json"])
            self.assertEqual((root / "out" / "data" / "x.json").read_bytes(), b"[1]")
            self.assertEqual(result.files[1].sha256, hashlib.sha256(b"[1]").hexdigest())
            self.assertEqual(result.files[1].size, 3)

    def test_include_patterns_narrow_what_is_taken(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = _tarball(
                root / "s.tgz",
                {"data/x.json": b"[1]", "data/y.pkl": b"\x80", "README.md": b"#"},
            )
            result = extract_archive(archive, root / "out", ("data/*.json",))
            self.assertEqual([rec.path for rec in result.files], ["data/x.json"])
            self.assertFalse((root / "out" / "data" / "y.pkl").exists())
            self.assertFalse((root / "out" / "README.md").exists())

    def test_a_member_that_climbs_out_of_the_destination_is_refused(self):
        """``extractall`` would honour the name; nothing here writes through it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../escaped.txt", "/etc/enamel-escaped", "a/../../escaped.txt"):
                archive = _tarball(root / "s.tgz", {name: b"x"})
                with self.assertRaises(PinError):
                    extract_archive(archive, root / "out")
            self.assertFalse((root / "escaped.txt").exists())
            self.assertEqual(sorted(p.name for p in (root / "out").rglob("*")), [])

    def test_links_and_devices_are_skipped_rather_than_recreated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            link = tarfile.TarInfo("enamel-abc123/passwd")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            result = extract_archive(_special(root / "s.tgz", link), root / "out")
            self.assertEqual([rec.path for rec in result.files], ["plain.txt"])
            self.assertEqual(result.skipped, ("passwd (link)",))
            self.assertFalse((root / "out" / "passwd").exists())
            self.assertFalse((root / "out" / "passwd").is_symlink())

    def test_an_archive_that_expands_too_far_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = _tarball(root / "s.tgz", {"a": b"1234", "b": b"5678"})
            with self.assertRaises(PinError) as caught:
                extract_archive(archive, root / "out", size_limit=5)
            self.assertIn("size limit", str(caught.exception))


class TestFetchScript(unittest.TestCase):
    """End to end over ``file://`` and local archives, so nothing needs network."""

    def setUp(self):
        self.script = _script()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dest = self.root / "cache"
        self.lock = self.root / "upstream.lock.json"
        self.manifest_path = self.root / "upstream.manifest.json"
        self.archive = _tarball(
            self.root / "s.tgz", {"data/x.json": b"[1]", "data/y.pkl": b"\x80\x04truncated"}
        )

    def _write_manifest(self, **kw):
        self.manifest_path.write_text(_manifest(**kw).to_json(), encoding="utf-8")

    def _run(self, *extra: str) -> tuple[int, str, str]:
        argv = [
            "--manifest", str(self.manifest_path),
            "--lock", str(self.lock),
            "--dest", str(self.dest),
            *extra,
        ]
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.script.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_first_use_records_the_pin_and_later_runs_check_it(self):
        self._write_manifest()
        code, out, err = self._run("--archive", str(self.archive), "--allow-new-pin")
        self.assertEqual(code, 0, err)
        lock = Lock.from_json(self.lock.read_text(encoding="utf-8"))
        self.assertEqual(lock.archive_sha256, digest_file(self.archive)[0])
        self.assertEqual([rec.path for rec in lock.files], ["data/x.json", "data/y.pkl"])
        self.assertIn("recorded on first use", out)
        self.assertIn("must not be committed", out)

        self.assertEqual(self._run("--verify")[0], 0)
        (self.dest / "data" / "x.json").write_bytes(b"[2]")
        code, _, err = self._run("--verify")
        self.assertEqual(code, 1)
        self.assertIn("changed: data/x.json", err)

    def test_a_fetch_with_nothing_to_check_against_stops(self):
        """Trust on first use has to be asked for, or a silent re-pin looks like a
        successful verification."""
        self._write_manifest()
        code, _, err = self._run("--archive", str(self.archive))
        self.assertEqual(code, 2)
        self.assertIn("no pin", err)
        self.assertFalse(self.lock.exists())
        self.assertFalse(self.dest.exists())

    def test_a_digest_that_disagrees_with_the_pin_is_never_extracted(self):
        self._write_manifest(archive_sha256="0" * 64)
        code, _, err = self._run("--archive", str(self.archive))
        self.assertEqual(code, 1)
        self.assertIn("digest mismatch", err)
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.lock.exists())

    def test_moving_the_pin_takes_a_second_decision(self):
        self._write_manifest()
        self.assertEqual(self._run("--archive", str(self.archive), "--allow-new-pin")[0], 0)
        self._write_manifest(revision=SHA_B)
        code, _, err = self._run("--archive", str(self.archive))
        self.assertEqual(code, 2)
        self.assertIn("disagree", err)
        self.assertEqual(
            Lock.from_json(self.lock.read_text(encoding="utf-8")).revision, SHA_A
        )
        self.assertEqual(self._run("--archive", str(self.archive), "--allow-new-pin")[0], 0)
        self.assertEqual(
            Lock.from_json(self.lock.read_text(encoding="utf-8")).revision, SHA_B
        )

    def test_a_missing_manifest_points_at_the_example(self):
        code, _, err = self._run()
        self.assertEqual(code, 2)
        self.assertIn("upstream.manifest.example.json", err)

    def test_a_url_is_fetched_and_a_bad_scheme_is_not(self):
        self._write_manifest(archive_url=self.archive.as_uri())
        self.assertEqual(self._run("--allow-new-pin")[0], 0)
        self.assertTrue((self.dest / "data" / "x.json").is_file())
        self._write_manifest(archive_url="ftp://example.invalid/s.tgz")
        code, _, err = self._run("--allow-new-pin")
        self.assertEqual(code, 2)
        self.assertIn("ftp", err)

    def test_the_inventory_names_pickles_without_loading_them(self):
        self._write_manifest()
        self.assertEqual(self._run("--archive", str(self.archive), "--allow-new-pin")[0], 0)
        code, out, _ = self._run("--inventory")
        self.assertEqual(code, 0)
        self.assertIn("data/x.json", out)
        self.assertIn(".pkl", out)
        self.assertIn("does not load them", out)

    def test_verifying_before_anything_is_pinned_says_so(self):
        code, _, err = self._run("--verify")
        self.assertEqual(code, 2)
        self.assertIn("nothing to verify", err)


if __name__ == "__main__":
    unittest.main()
