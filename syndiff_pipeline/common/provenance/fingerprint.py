"""
fingerprint.py
===============
Deterministic, cross-process content hashing for the provenance graph.

Three primitives, in order of composition:

- :func:`canonical` — turn any JSON-ish Python value into a single canonical
  byte string: sorted dict keys, floats rounded to 1e-9 (no ``-0.0``),
  tuples/numpy arrays coerced to lists, ``NaN``/``inf`` rejected, UTF-8
  encoded. Two logically-equal structures always canonicalize to identical
  bytes, independent of dict insertion order, tuple-vs-list, or numpy-vs-plain
  scalar types.
- :func:`recipe_id` — hash of ``(kind, params, code_version)``. Identifies one
  materialized parameter set for one producer.
- :func:`fingerprint` — hash of ``(kind, spatial_key, recipe_id,
  sorted(input_fingerprints))``. This is the artifact's Merkle name: identical
  work is recognized by string equality, and a parameter change re-fingerprints
  exactly the affected downstream cone.

Golden tests in ``tests/test_provenance_fingerprint.py`` pin exact
``canonical()`` bytes and exact ``recipe_id``/``fingerprint`` hex outputs for
fixed inputs -- any change to the rules below is a breaking change to every
fingerprint already on disk and must bump :data:`RECIPE_SCHEMA_VERSION`.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - numpy is an existing project dependency
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]

# Bump on ANY change to producer algorithms whose output bytes could change
# for the same recipe params (e.g. a numerics fix). This is folded into every
# recipe_id via the caller-supplied ``code_version`` argument -- it is NOT
# read implicitly by canonical()/recipe_id()/fingerprint(); producers pass it
# explicitly so schema bumps are auditable per-kind.
RECIPE_SCHEMA_VERSION = 1

_FLOAT_ROUND_NDIGITS = 9

__all__ = [
    "RECIPE_SCHEMA_VERSION",
    "canonical",
    "recipe_id",
    "fingerprint",
]


def _round_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"canonical(): NaN/inf not allowed: {value!r}")
    rounded = round(value, _FLOAT_ROUND_NDIGITS)
    # Normalize -0.0 -> 0.0 so sign-of-zero never affects the byte stream.
    if rounded == 0.0:
        rounded = 0.0
    return rounded


def _canon(obj: Any) -> Any:
    """Recursively coerce *obj* into a canonical, JSON-serializable structure."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return _round_float(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        raise TypeError(
            "canonical(): raw bytes are not allowed (hex-encode or decode to str first)"
        )
    if _np is not None and isinstance(obj, _np.generic):
        # numpy scalar (np.float64, np.int64, np.bool_, ...) -> plain python.
        return _canon(obj.item())
    if _np is not None and isinstance(obj, _np.ndarray):
        return _canon(obj.tolist())
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key] = _canon(v)
        return out
    if isinstance(obj, (list, tuple)) or (
        isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))
    ):
        return [_canon(v) for v in obj]
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        # Sets and other one-shot iterables: sort for determinism if possible.
        items = list(obj)
        try:
            items = sorted(items)
        except TypeError:
            pass
        return [_canon(v) for v in items]
    raise TypeError(f"canonical(): unsupported type {type(obj)!r} for value {obj!r}")


def canonical(obj: Any) -> bytes:
    """
    Canonical byte encoding of *obj*.

    Rules (all pinned by golden tests):

    - dict keys sorted lexicographically (via ``json.dumps(sort_keys=True)``);
      non-str keys are stringified first.
    - floats rounded to 1e-9 with ``-0.0`` normalized to ``0.0``.
    - tuples, numpy arrays, and other sequences/iterables become JSON lists.
    - ``NaN``/``inf`` floats raise :class:`ValueError`.
    - output is UTF-8 bytes of compact JSON (``separators=(",", ":")``,
      ``ensure_ascii=False``).

    Raises
    ------
    TypeError
        On an unsupported value type (e.g. raw bytes, arbitrary objects).
    ValueError
        On NaN/inf floats.
    """
    canon = _canon(obj)
    text = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recipe_id(kind: str, params: Mapping[str, Any], code_version: int) -> str:
    """
    Stable id for one materialized ``(kind, params, code_version)`` recipe.

    Returns the first 16 hex characters of ``sha256(canonical(payload))``.
    """
    payload = {
        "kind": str(kind),
        "params": params,
        "code_version": int(code_version),
    }
    return _sha256_hex(canonical(payload))[:16]


def fingerprint(
    kind: str,
    spatial_key: Mapping[str, Any],
    recipe_id_: str,
    input_fps: Iterable[str],
) -> str:
    """
    Merkle fingerprint naming one artifact node.

    ``H(kind, spatial_key, recipe_id, sorted(input_fingerprints))``.
    Returns the first 24 hex characters of ``sha256(canonical(payload))``.

    Parameters
    ----------
    kind : str
        Artifact kind, see ``model.py``'s kind registry.
    spatial_key : Mapping[str, Any]
        Canonical spatial-key dict (skycell / scc / scc_ffi / event).
    recipe_id_ : str
        Output of :func:`recipe_id` for this artifact's producer.
    input_fps : Iterable[str]
        Fingerprints of the artifacts this one was built from. Order does not
        matter -- they are sorted before hashing.
    """
    payload = {
        "kind": str(kind),
        "spatial_key": spatial_key,
        "recipe_id": str(recipe_id_),
        "inputs": sorted(str(fp) for fp in input_fps),
    }
    return _sha256_hex(canonical(payload))[:24]
