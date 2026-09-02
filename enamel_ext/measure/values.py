"""Transporting a solution's return value across the process boundary.

JSON alone cannot tell a tuple from a list, so outputs are type-tagged on the
way out and rebuilt on the way in; comparison then has plain Python ``==``
semantics. See docs/decisions/0004-sandboxed-runner.md.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["encode", "decode", "values_equal", "brief", "REL_TOL", "ABS_TOL"]

#: ``abs_tol`` matches HumanEval's own float checks, ``abs(a - b) < 1e-6``. The
#: relative term only covers accumulation slack at large magnitude, so it is far
#: tighter than double precision needs. Exact comparison is ``0.0, 0.0``.
REL_TOL = 1e-12
ABS_TOL = 1e-6

_TAG = "~"

#: CPython caps int-to-decimal conversion (``int_max_str_digits``, 4300 by
#: default, never settable below 640 digits), and JSON writes ints in decimal.
#: Wider ints travel as hex, which the cap exempts. 2048 bits is 617 digits.
_INLINE_INT_BITS = 2048

#: Characters kept per value in a diagnostic message.
_BRIEF_LIMIT = 200


def _safe_repr(value: Any) -> str:
    """``repr(value)`` where that works, a description of it where it does not.

    ``repr`` of a wide enough int raises, and hex is the one base the digit cap
    exempts, so ints keep their value even past the cap.
    """
    try:
        return repr(value)
    except BaseException as exc:
        if isinstance(value, int):
            return f"0x{value:x}"
        return f"<unreprable {type(value).__name__}: {type(exc).__name__}>"


def brief(value: Any, limit: int = _BRIEF_LIMIT) -> str:
    """Bounded repr for diagnostics, so a wrong answer on a large input does not
    carry a copy of that input into the results."""
    text = _safe_repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def encode(value: Any) -> Any:
    """Type-tagged, JSON-representable form of ``value``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if value.bit_length() <= _INLINE_INT_BITS:
            return value
        return {_TAG: "int", "v": f"{value:x}"}
    if isinstance(value, float):
        return {_TAG: "float", "v": repr(value)}
    if isinstance(value, bytes):
        return {_TAG: "bytes", "v": value.hex()}
    if isinstance(value, list):
        return {_TAG: "list", "v": [encode(x) for x in value]}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "v": [encode(x) for x in value]}
    if isinstance(value, (set, frozenset)):
        kind = "set" if isinstance(value, set) else "frozenset"
        return {
            _TAG: kind,
            "v": sorted(_safe_repr(x) for x in value),
            "e": [encode(x) for x in value],
        }
    if isinstance(value, dict):
        return {_TAG: "dict", "v": [[encode(k), encode(v)] for k, v in value.items()]}
    return {_TAG: "opaque", "v": _safe_repr(value)}


def decode(blob: Any) -> Any:
    """Inverse of :func:`encode`. Opaque values come back as their repr string."""
    if not isinstance(blob, dict) or _TAG not in blob:
        return blob
    kind = blob[_TAG]
    if kind == "int":
        return int(blob["v"], 16)
    if kind == "float":
        return float(blob["v"])
    if kind == "bytes":
        return bytes.fromhex(blob["v"])
    if kind == "list":
        return [decode(x) for x in blob["v"]]
    if kind == "tuple":
        return tuple(decode(x) for x in blob["v"])
    if kind in ("set", "frozenset"):
        items = [decode(x) for x in blob.get("e", [])]
        try:
            built = set(items)
        except TypeError:
            return _Opaque(kind + ":" + ",".join(blob["v"]))
        return built if kind == "set" else frozenset(built)
    if kind == "dict":
        try:
            return {decode(k): decode(v) for k, v in blob["v"]}
        except TypeError:
            return _Opaque(repr(blob["v"]))
    return _Opaque(blob["v"])


class _Opaque:
    """A value that could not cross the boundary; compares by repr only."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Opaque) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)

    def __repr__(self) -> str:
        return f"<opaque {self.text}>"


def values_equal(a: Any, b: Any, rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> bool:
    """``a == b`` with float tolerance applied at every depth.

    Python's own ``==`` decides everything else, so ``True == 1`` stays true; a
    list and a tuple do not, since that is a real wrong answer. Set members and
    dict keys compare exactly, because tolerance has no meaning under hashing.
    Two NaNs compare equal: the reference producing NaN is not a wrong answer.
    """
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        return all(values_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(values_equal(a[k], b[k], rel_tol, abs_tol) for k in a)
    return bool(a == b)
