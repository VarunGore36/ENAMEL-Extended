"""Pinning an upstream data snapshot: manifest, digest lock, safe extraction.

The fetch itself is a few lines of ``urllib`` in scripts/fetch_upstream.py; the
parts that need testing without network live here. See
docs/decisions/0003-data-adapter.md.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from enamel_ext.data.schema import UNKNOWN_LICENSE, Provenance

__all__ = [
    "DEFAULT_SIZE_LIMIT",
    "LOCK_SCHEMA_VERSION",
    "Extraction",
    "FileRecord",
    "Lock",
    "Manifest",
    "PinError",
    "digest_file",
    "extract_archive",
    "today_utc",
]

LOCK_SCHEMA_VERSION = 1

#: Refuse an archive that expands past this, so a crafted tarball cannot fill
#: the disk. Upstream's inputs are three orders of magnitude smaller.
DEFAULT_SIZE_LIMIT = 512 * 1024 * 1024

_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_REPO_RE = re.compile(r"\A[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")

_ARCHIVE_TEMPLATE = "https://codeload.github.com/{repo}/tar.gz/{revision}"

_CHUNK = 1 << 20


class PinError(RuntimeError):
    """A manifest, digest or archive member that must not be used."""


def digest_file(path: str | Path) -> tuple[str, int]:
    """``(sha256, size)`` without holding the file in memory."""
    running = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            running.update(chunk)
            size += len(chunk)
    return running.hexdigest(), size


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _json_object(text: str, what: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PinError(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PinError(f"{what} must be a JSON object, got {type(raw).__name__}")
    return raw


_MANIFEST_KEYS = frozenset(
    {"name", "repo", "revision", "license", "url", "include", "archive_url", "archive_sha256"}
)


@dataclass(frozen=True)
class Manifest:
    """What to fetch, at which commit, and what may be taken from it.

    ``include`` holds fnmatch patterns matched case-sensitively against
    archive-relative paths, where ``*`` also crosses ``/``. It defaults to
    everything because the upstream layout is not known in advance.
    """

    name: str
    repo: str
    revision: str
    license: str = UNKNOWN_LICENSE
    url: str = ""
    include: tuple[str, ...] = ("*",)
    archive_url: str = ""
    archive_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "include", tuple(self.include))
        for field in ("name", "repo", "revision", "license"):
            if not str(getattr(self, field)).strip():
                raise PinError(f"manifest.{field} must not be empty")
        if not _REPO_RE.match(self.repo):
            raise PinError(f"manifest.repo must be 'owner/name', got {self.repo!r}")
        revision = self.revision.strip().lower()
        if not _COMMIT_RE.match(revision):
            raise PinError(
                f"manifest.revision must be a full 40-character commit sha, got "
                f"{self.revision!r}: a branch or tag can move, so it does not pin a "
                f"snapshot. Find one with "
                f"'git ls-remote https://github.com/{self.repo} HEAD'."
            )
        object.__setattr__(self, "revision", revision)
        if self.archive_sha256 is not None:
            pin = str(self.archive_sha256).strip().lower()
            if not _SHA256_RE.match(pin):
                raise PinError(f"manifest.archive_sha256 is not a sha256: {self.archive_sha256!r}")
            object.__setattr__(self, "archive_sha256", pin)
        if not self.include:
            raise PinError("manifest.include must list at least one pattern")
        if not self.url:
            object.__setattr__(self, "url", f"https://github.com/{self.repo}")

    @property
    def redistributable(self) -> bool:
        """False while the license is unconfirmed; gates vendoring, not fetching."""
        return self.license != UNKNOWN_LICENSE

    def resolved_archive_url(self) -> str:
        return self.archive_url or _ARCHIVE_TEMPLATE.format(
            repo=self.repo, revision=self.revision
        )

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        """Parse a manifest. Keys beginning with ``_`` are notes and are ignored,
        since JSON has no comments and a pin file needs explaining."""
        raw = _json_object(text, "manifest")
        fields = {key: value for key, value in raw.items() if not key.startswith("_")}
        unknown = sorted(set(fields) - _MANIFEST_KEYS)
        if unknown:
            raise PinError(f"manifest has unknown keys: {unknown}")
        missing = sorted({"name", "repo", "revision"} - set(fields))
        if missing:
            raise PinError(f"manifest is missing required keys: {missing}")
        if "include" in fields and not isinstance(fields["include"], list):
            raise PinError("manifest.include must be a list of patterns")
        return cls(**fields)

    def to_json(self) -> str:
        body: dict[str, Any] = {
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "license": self.license,
            "url": self.url,
            "include": list(self.include),
        }
        if self.archive_url:
            body["archive_url"] = self.archive_url
        if self.archive_sha256:
            body["archive_sha256"] = self.archive_sha256
        return json.dumps(body, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Lock:
    """The committed record of one fetch.

    Enough to detect three different things: that the same commit now serves
    different bytes, that the extracted cache has been edited since, and that the
    manifest has been repointed without a re-fetch.
    """

    name: str
    repo: str
    revision: str
    license: str
    url: str
    retrieved: str
    archive_url: str
    archive_sha256: str
    archive_bytes: int
    files: tuple[FileRecord, ...] = ()
    schema_version: int = LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda rec: rec.path)))
        if not _SHA256_RE.match(str(self.archive_sha256).lower()):
            raise PinError(f"lock.archive_sha256 is not a sha256: {self.archive_sha256!r}")
        object.__setattr__(self, "archive_sha256", str(self.archive_sha256).lower())
        if self.schema_version != LOCK_SCHEMA_VERSION:
            raise PinError(
                f"lock schema version {self.schema_version} is not "
                f"{LOCK_SCHEMA_VERSION}; regenerate the snapshot"
            )

    @classmethod
    def record(
        cls,
        manifest: Manifest,
        *,
        archive_sha256: str,
        archive_bytes: int,
        files: Iterable[FileRecord] = (),
        retrieved: str | None = None,
    ) -> Lock:
        return cls(
            name=manifest.name,
            repo=manifest.repo,
            revision=manifest.revision,
            license=manifest.license,
            url=manifest.url,
            retrieved=retrieved or today_utc(),
            archive_url=manifest.resolved_archive_url(),
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            files=tuple(files),
        )

    def provenance(self) -> Provenance:
        """Provenance as recorded at fetch time, not as of now."""
        return Provenance(
            name=self.name, url=self.url, license=self.license, retrieved=self.retrieved
        )

    def to_json(self) -> str:
        body = {
            "schema_version": self.schema_version,
            "name": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "license": self.license,
            "url": self.url,
            "retrieved": self.retrieved,
            "archive_url": self.archive_url,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "files": [
                {"path": rec.path, "sha256": rec.sha256, "size": rec.size} for rec in self.files
            ],
        }
        return json.dumps(body, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Lock:
        raw = dict(_json_object(text, "lock"))
        try:
            files = tuple(
                FileRecord(rec["path"], rec["sha256"], int(rec["size"]))
                for rec in raw.pop("files", [])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PinError(f"lock.files is malformed: {exc}") from exc
        try:
            return cls(files=files, **raw)
        except TypeError as exc:
            raise PinError(f"lock is malformed: {exc}") from exc

    def mismatch_with(self, manifest: Manifest) -> str:
        """Why this lock does not describe ``manifest``, or ``''``."""
        if manifest.repo != self.repo or manifest.revision != self.revision:
            return (
                f"manifest points at {manifest.repo}@{manifest.revision[:12]}, "
                f"lock records {self.repo}@{self.revision[:12]}"
            )
        if manifest.archive_sha256 and manifest.archive_sha256 != self.archive_sha256:
            return (
                f"manifest pins archive {manifest.archive_sha256[:12]}, "
                f"lock records {self.archive_sha256[:12]}"
            )
        return ""

    def verify_files(self, root: str | Path) -> tuple[str, ...]:
        """Every way the extracted tree disagrees with this lock.

        Reports files added under ``root`` as well as missing and altered ones,
        because the extraction writes exactly what the lock lists, so anything
        else arrived by another route.
        """
        base = Path(root)
        problems: list[str] = []
        for record in self.files:
            target = base / record.path
            if not target.is_file():
                problems.append(f"missing: {record.path}")
                continue
            found, size = digest_file(target)
            if found != record.sha256 or size != record.size:
                problems.append(f"changed: {record.path}")
        known = {record.path for record in self.files}
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(base).as_posix()
                    if relative not in known:
                        problems.append(f"unexpected: {relative}")
        return tuple(problems)


@dataclass(frozen=True)
class Extraction:
    files: tuple[FileRecord, ...]
    skipped: tuple[str, ...]


def _strip_root(name: str) -> str:
    """Drop the single top-level directory a GitHub tarball wraps everything in."""
    parts = name.split("/", 1)
    return parts[1] if len(parts) == 2 else ""


def _safe_relative(path: str) -> str:
    """``path`` normalized, or :class:`PinError` if it could escape a destination."""
    if not path or path.startswith("/") or "\\" in path:
        raise PinError(f"archive member is not a relative posix path: {path!r}")
    normalized = posixpath.normpath(path)
    if normalized.startswith(("/", "../")) or normalized in ("..", "."):
        raise PinError(f"archive member escapes the destination: {path!r}")
    return normalized


def _member_kind(member: tarfile.TarInfo) -> str:
    if member.issym() or member.islnk():
        return "link"
    if member.ischr() or member.isblk() or member.isdev():
        return "device"
    if member.isfifo():
        return "fifo"
    return "other"


def _within(root: Path, target: Path) -> None:
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise PinError(f"archive member would be written outside {root}: {target}")


def extract_archive(
    archive: str | Path,
    dest: str | Path,
    include: Iterable[str] = ("*",),
    size_limit: int = DEFAULT_SIZE_LIMIT,
) -> Extraction:
    """Extract the regular files matching ``include`` into ``dest``.

    Each member is written through a path computed here rather than through
    ``extractall``, so a crafted name cannot choose the destination; links and
    devices are skipped rather than recreated, and the running total of bytes
    written is what the size limit checks, since a header's size can lie.
    """
    patterns = tuple(include)
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    files: list[FileRecord] = []
    skipped: list[str] = []
    total = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            relative = _strip_root(member.name)
            if not relative or member.isdir():
                continue
            if not member.isfile():
                skipped.append(f"{relative} ({_member_kind(member)})")
                continue
            relative = _safe_relative(relative)
            if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
                continue
            target = root / relative
            _within(resolved_root, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            record = _copy_member(tar, member, relative, target, size_limit - total)
            files.append(record)
            total += record.size
    return Extraction(tuple(sorted(files, key=lambda rec: rec.path)), tuple(sorted(skipped)))


def _copy_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    relative: str,
    target: Path,
    budget: int,
) -> FileRecord:
    """Stream one member to ``target``, digesting and size-capping as it goes."""
    source = tar.extractfile(member)
    if source is None:
        raise PinError(f"archive member {member.name!r} has no readable content")
    running = hashlib.sha256()
    size = 0
    with source, open(target, "wb") as out:
        while chunk := source.read(_CHUNK):
            size += len(chunk)
            if size > budget:
                raise PinError(f"archive expands past the size limit at {member.name!r}")
            running.update(chunk)
            out.write(chunk)
    return FileRecord(relative, running.hexdigest(), size)
