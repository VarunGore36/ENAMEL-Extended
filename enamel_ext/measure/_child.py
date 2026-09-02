"""Worker process: runs one level of one solution and reports times.

Launched by :mod:`enamel_ext.measure.sandbox`; never imported in-process,
because everything it executes is untrusted. Protocol is a JSON request at the
path given on the command line, and a JSON response written to the request's
``result_path``.
"""

from __future__ import annotations

import copy
import gc
import json
import resource
import sys
import time
from typing import Any

from enamel_ext.measure.values import decode, encode

OK = "ok"
TIMEOUT = "timeout"
ERROR = "error"

#: Bound at import, before any untrusted code runs, so a solution that rebinds
#: ``time.perf_counter`` cannot report its own runtime.
_PERF = time.perf_counter
_DEEPCOPY = copy.deepcopy
_GC_DISABLE = gc.disable
_GC_ENABLE = gc.enable
_GC_ISENABLED = gc.isenabled
_OPEN = open

_RLIMITS = (("address_space", resource.RLIMIT_AS), ("cpu_seconds", resource.RLIMIT_CPU))


def _apply_limits(limits: dict[str, Any]) -> list[tuple[int, tuple[int, int]]]:
    """Lower soft rlimits, returning what to restore before writing the response."""
    saved = []
    for key, which in _RLIMITS:
        value = limits.get(key)
        if value is None:
            continue
        soft, hard = resource.getrlimit(which)
        saved.append((which, (soft, hard)))
        capped = int(value) if hard == resource.RLIM_INFINITY else min(int(value), hard)
        resource.setrlimit(which, (capped, hard))
    if limits.get("recursion_limit"):
        sys.setrecursionlimit(int(limits["recursion_limit"]))
    return saved


def _entry_point(code: str, name: str) -> Any:
    namespace: dict[str, Any] = {"__name__": "__candidate__"}
    exec(compile(code, "<solution>", "exec"), namespace)
    fn = namespace.get(name)
    if not callable(fn):
        raise ValueError(f"solution does not define {name}()")
    return fn


def _build_cases(request: dict[str, Any]) -> list[tuple[Any, ...]]:
    inputs = request.get("inputs")
    if inputs is not None:
        return [tuple(decode(arg) for arg in args) for args in inputs]
    namespace: dict[str, Any] = {}
    exec(compile(request["generator"], "<generator>", "exec"), namespace)
    make_input = namespace.get("make_input")
    if not callable(make_input):
        raise ValueError("generator does not define make_input(seed, scale)")
    cases = []
    for seed in request["seeds"]:
        args = make_input(seed, request["scale"])
        if not isinstance(args, tuple):
            raise ValueError(f"make_input returned {type(args).__name__}, expected a tuple")
        cases.append(args)
    return cases


def _time_case(
    fn: Any, args: tuple[Any, ...], repeats: int, budget: float | None, no_gc: bool
) -> tuple[list[float], Any, bool, str]:
    """Time ``repeats`` calls, each on a fresh copy of ``args``.

    Copying matters: a solution that sorts its input in place would make every
    repeat after the first measure already-sorted data. ``budget`` caps the
    accumulated time rather than a single repeat, so one noisy repeat cannot
    censor a case whose aggregate is under the limit. The first call's return
    value is encoded immediately, before a later repeat can mutate it.
    """
    times: list[float] = []
    output: Any = None
    has_output = False
    total = 0.0
    for r in range(repeats):
        call_args = _DEEPCOPY(args)
        collecting = _GC_ISENABLED()
        if no_gc:
            _GC_DISABLE()
        try:
            start = _PERF()
            value = fn(*call_args)
            elapsed = _PERF() - start
        finally:
            if no_gc and collecting:
                _GC_ENABLE()
        if r == 0:
            output = encode(value)
            has_output = True
        times.append(elapsed)
        total += elapsed
        if budget is not None and total > budget:
            return times, output, has_output, TIMEOUT
    return times, output, has_output, OK


def _run(request: dict[str, Any]) -> dict[str, Any]:
    repeats = int(request.get("repeats", 1))
    limit = request.get("time_limit")
    budget = None if limit is None else float(limit) * repeats
    no_gc = bool(request.get("disable_gc", False))
    try:
        cases = _build_cases(request)
        fn = _entry_point(request["code"], request["entry_point"])
    except BaseException as exc:
        return {"status": ERROR, "detail": f"{type(exc).__name__}: {exc}", "cases": []}

    results: list[dict[str, Any]] = []
    for args in cases:
        try:
            times, output, has_output, status = _time_case(fn, args, repeats, budget, no_gc)
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            results.append({"status": ERROR, "detail": detail})
            return {"status": ERROR, "detail": detail, "cases": results}
        entry: dict[str, Any] = {"status": status, "times": times}
        if has_output:
            entry["output"] = output
            entry["has_output"] = True
        results.append(entry)
        if status == TIMEOUT:
            return {
                "status": TIMEOUT,
                "detail": f"case {len(results) - 1} used more than {budget:.6g}s "
                f"over {repeats} repeats",
                "cases": results,
            }
    return {"status": OK, "detail": "", "cases": results}


def _serialize(response: dict[str, Any]) -> str:
    """JSON text for ``response``, or for a diagnostic if it will not encode."""
    try:
        return json.dumps(response)
    except BaseException as exc:
        return json.dumps(
            {
                "status": ERROR,
                "detail": f"result would not serialize: {type(exc).__name__}: {exc}",
                "cases": [],
            }
        )


def main(request_path: str | None = None) -> None:
    path = request_path if request_path is not None else sys.argv[1]
    with _OPEN(path, encoding="utf-8") as handle:
        request = json.load(handle)
    saved = _apply_limits(request.get("limits") or {})
    try:
        response = _run(request)
    finally:
        for which, pair in saved:
            try:
                resource.setrlimit(which, pair)
            except (ValueError, OSError):
                pass
    with _OPEN(request["result_path"], "w", encoding="utf-8") as handle:
        handle.write(_serialize(response))


if __name__ == "__main__":
    main()
