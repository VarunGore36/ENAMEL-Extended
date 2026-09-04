"""Out-of-process execution of one level of one solution.

A fresh interpreter per level, resource caps, and a wall-clock kill of the
process group. Process-level mitigation, not containment; see
docs/decisions/0004-sandboxed-runner.md.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from enamel_ext.measure.timing import AGGREGATORS
from enamel_ext.measure.values import decode, encode

__all__ = [
    "CRASHED",
    "ERROR",
    "OK",
    "TIMEOUT",
    "CaseResult",
    "LevelResult",
    "Limits",
    "SandboxError",
    "run_level",
]

OK = "ok"
TIMEOUT = "timeout"
ERROR = "error"
CRASHED = "crashed"

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``-I`` is not used: it implies ``-E``, which would drop ``PYTHONHASHSEED``.
#: The environment is replaced wholesale instead, and the cwd is dropped from
#: ``sys.path`` here rather than by a flag.
_BOOTSTRAP = (
    "import os, sys; "
    "sys.path[:] = [sys.argv[1]] + [p for p in sys.path if p not in ('', '.', os.getcwd())]; "
    "from enamel_ext.measure._child import main; main(sys.argv[2])"
)

#: Wall clock allowed = base + slack * (per-case limit) * repeats * cases. The
#: slack has to cover the most a case can consume before the aggregate bound in
#: :func:`~enamel_ext.measure.timing.reaches_limit` trips; under Hodges-Lehmann
#: with ``R = 6`` that is under ``8 * time_limit``, against an allowance of
#: ``WALL_SLACK * repeats`` per case.
WALL_BASE = 15.0
WALL_SLACK = 4.0

#: Seconds by which ``RLIMIT_CPU`` must exceed the wall budget.
CPU_HEADROOM = 5

_STDERR_TAIL = 2000


class SandboxError(RuntimeError):
    """The harness itself failed, as opposed to the code under test."""


@dataclass(frozen=True)
class Limits:
    """Caps applied inside the child before any untrusted code runs.

    ``cpu_seconds`` is a floor, not a ceiling: it is raised above the wall-clock
    budget when needed, because it is integer-valued, counts input generation,
    and reports as a signal rather than as a timeout.
    """

    address_space: int | None = 4 << 30
    cpu_seconds: int | None = 60
    recursion_limit: int | None = None
    disable_gc: bool = False


@dataclass(frozen=True)
class CaseResult:
    """One test case: ``repeats`` timings, plus the first call's return value."""

    status: str
    times: tuple[float, ...] = ()
    output: Any = None
    has_output: bool = False
    detail: str = ""


@dataclass(frozen=True)
class LevelResult:
    status: str
    cases: tuple[CaseResult, ...] = ()
    detail: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK


def _request(
    code: str,
    entry_point: str,
    *,
    inputs: Sequence[Sequence[Any]] | None,
    generator: str | None,
    scale: int | None,
    seeds: Sequence[int] | None,
    repeats: int,
    time_limit: float | None,
    aggregator: str,
    limits: Limits,
    result_path: str,
) -> dict[str, Any]:
    if inputs is None and not generator:
        raise SandboxError("level has neither materialized inputs nor a generator")
    return {
        "code": code,
        "entry_point": entry_point,
        "inputs": None if inputs is None else [[encode(a) for a in args] for args in inputs],
        "generator": generator,
        "scale": scale,
        "seeds": None if seeds is None else list(seeds),
        "repeats": repeats,
        "time_limit": time_limit,
        "aggregator": aggregator,
        "disable_gc": limits.disable_gc,
        "result_path": result_path,
        "limits": {
            "address_space": limits.address_space,
            "cpu_seconds": limits.cpu_seconds,
            "recursion_limit": limits.recursion_limit,
        },
    }


def _wall_timeout(time_limit: float | None, repeats: int, n_cases: int) -> float:
    per_case = time_limit if time_limit is not None else 1.0
    return WALL_BASE + WALL_SLACK * per_case * max(repeats, 1) * max(n_cases, 1)


def _effective_limits(limits: Limits, wall: float) -> Limits:
    """Keep ``RLIMIT_CPU`` above ``wall`` so the precise mechanism fires first."""
    if limits.cpu_seconds is None:
        return limits
    floor = int(math.ceil(wall)) + CPU_HEADROOM
    return limits if limits.cpu_seconds >= floor else replace(limits, cpu_seconds=floor)


def _child_env(workdir: str) -> dict[str, str]:
    """A stripped environment with ``PYTHONHASHSEED`` pinned for determinism."""
    return {
        "PATH": "",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _n_cases(inputs: Sequence[Sequence[Any]] | None, seeds: Sequence[int] | None) -> int:
    if inputs is not None:
        return len(inputs)
    return len(seeds or ())


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the whole session, then reap the direct child."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _read_tail(path: str) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, encoding="utf-8", errors="replace") as handle:
            if size > _STDERR_TAIL:
                handle.seek(size - _STDERR_TAIL)
            return handle.read()
    except OSError:
        return ""


def _load_response(path: str) -> dict[str, Any] | None:
    """Parsed response, ``None`` if the child never wrote one."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            response = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"status": CRASHED, "detail": f"unreadable result: {exc}", "cases": []}
    if not isinstance(response, dict):
        return {"status": CRASHED, "detail": f"result is a {type(response).__name__}", "cases": []}
    return response


def _signal_name(returncode: int) -> str:
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def _exit_failure(returncode: int | None, stderr: str) -> LevelResult:
    """No result file: the child died before it could report."""
    if returncode == -signal.SIGXCPU:
        return LevelResult(TIMEOUT, detail="killed by the CPU-time limit", stderr=stderr)
    if returncode is not None and returncode < 0:
        return LevelResult(
            CRASHED,
            detail=f"child killed by {_signal_name(returncode)} and wrote no result",
            stderr=stderr,
        )
    return LevelResult(
        CRASHED, detail=f"child exited with {returncode} and wrote no result", stderr=stderr
    )


def _decode_cases(raw: Any) -> tuple[CaseResult, ...]:
    out = []
    for entry in raw or ():
        out.append(
            CaseResult(
                status=entry.get("status", ERROR),
                times=tuple(float(t) for t in entry.get("times", ())),
                output=decode(entry["output"]) if entry.get("has_output") else None,
                has_output=bool(entry.get("has_output")),
                detail=entry.get("detail", ""),
            )
        )
    return tuple(out)


def run_level(
    code: str,
    entry_point: str,
    *,
    inputs: Sequence[Sequence[Any]] | None = None,
    generator: str | None = None,
    scale: int | None = None,
    seeds: Sequence[int] | None = None,
    repeats: int = 1,
    time_limit: float | None = None,
    aggregator: str = "hodges_lehmann",
    limits: Limits = Limits(),
    wall_timeout: float | None = None,
) -> LevelResult:
    """Run one level's test cases in a fresh interpreter.

    Either ``inputs`` or ``generator`` plus ``scale`` and ``seeds`` must be
    given. A case stops once no completion of its remaining repeats can bring the
    ``aggregator``'s estimate back under ``time_limit``, and the level stops with
    it: a level is scored on its worst case, so the remaining cases cannot change
    the outcome. ``aggregator`` therefore has to be the one the score will use.
    """
    if repeats < 1:
        raise SandboxError(f"repeats must be >= 1, got {repeats}")
    if aggregator not in AGGREGATORS:
        raise SandboxError(f"unknown aggregator {aggregator!r}; have {sorted(AGGREGATORS)}")
    n_cases = _n_cases(inputs, seeds)
    if n_cases == 0:
        raise SandboxError("level has no test cases")
    wall = (
        _wall_timeout(time_limit, repeats, n_cases)
        if wall_timeout is None
        else float(wall_timeout)
    )
    if not wall > 0:
        raise SandboxError(f"wall timeout must be positive, got {wall}")

    with tempfile.TemporaryDirectory(prefix="enamel-ext-io-") as iodir:
        request_path = os.path.join(iodir, "request.json")
        result_path = os.path.join(iodir, "result.json")
        stderr_path = os.path.join(iodir, "stderr.txt")
        request = _request(
            code,
            entry_point,
            inputs=inputs,
            generator=generator,
            scale=scale,
            seeds=seeds,
            repeats=repeats,
            time_limit=time_limit,
            aggregator=aggregator,
            limits=_effective_limits(limits, wall),
            result_path=result_path,
        )
        try:
            with open(request_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle)
        except (TypeError, ValueError) as exc:
            raise SandboxError(f"level input is not JSON-representable: {exc}") from exc

        timed_out = False
        with tempfile.TemporaryDirectory(prefix="enamel-ext-run-") as workdir:
            with open(stderr_path, "w", encoding="utf-8") as errfile:
                proc = subprocess.Popen(
                    [sys.executable, "-s", "-c", _BOOTSTRAP, str(REPO_ROOT), request_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=errfile,
                    cwd=workdir,
                    env=_child_env(workdir),
                    start_new_session=True,
                )
                try:
                    proc.wait(timeout=wall)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    if proc.poll() is None:
                        _kill_group(proc)

        stderr = _read_tail(stderr_path)
        response = _load_response(result_path)

    if response is None:
        if timed_out:
            return LevelResult(
                status=TIMEOUT,
                detail=f"killed after {wall:.3f}s of wall clock",
                stderr=stderr,
            )
        return _exit_failure(proc.returncode, stderr)

    try:
        cases = _decode_cases(response.get("cases"))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return LevelResult(
            status=CRASHED,
            detail=f"unreadable result: {type(exc).__name__}: {exc}",
            stderr=stderr,
        )
    return LevelResult(
        status=response.get("status", ERROR),
        cases=cases,
        detail=response.get("detail", ""),
        stderr=stderr,
    )
