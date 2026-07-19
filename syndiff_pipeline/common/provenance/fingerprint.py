"""Deterministic fingerprinting for the content-addressed provenance graph.

This module is the shared contract every other provenance module depends on.
It defines the canonical serialization and the Merkle fingerprint functions
used to name artifacts (see ``doc/template_bookkeeping_plan.md`` §9).

Dependency-free by design: stdlib only (``json``, ``hashlib``, ``subprocess``,
``math``). Nothing here may import astropy/zarr or other heavy deps, so the
scheduler and daemon can use it cheaply.

The byte output of :func:`canonical` is *golden-tested* — it must stay
byte-identical across processes, Python builds, and library versions, because
fingerprints derived from it are persisted and used for equality.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess

# Bump on ANY producer algorithm change so old products re-fingerprint
# deliberately (decision #4 in the plan). ``code_version()`` stringifies this.
RECIPE_SCHEMA_VERSION: int = 1

# Floats are rounded to this many decimal places before serialization so that
# values within ~1e-9 collapse to the same bytes (and therefore the same hash).
_FLOAT_NDIGITS: int = 9


def _normalize(obj):
    """Recursively normalize an object tree for canonical serialization.

    - ``bool`` is preserved (and checked before ``int`` since ``bool`` is a
      subclass of ``int``).
    - ``int``/``str``/``None`` pass through unchanged.
    - ``float`` is rounded to 1e-9; ``NaN``/``inf`` raise ``ValueError``;
      ``-0.0`` normalizes to ``0.0``; values that are integral after rounding
      become ``int`` so ``1.0``, ``1`` and ``1.0000000001`` all serialize
      identically.
    - ``tuple`` serializes like ``list`` (order preserved).
    - ``dict`` values are normalized; keys are left intact and sorted by
      ``json.dumps(sort_keys=True)``.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"non-finite float is not fingerprintable: {obj!r}")
        rounded = round(obj, _FLOAT_NDIGITS)
        # Normalize -0.0 -> 0.0 and collapse integral floats to int so that
        # 1.0 and 1 produce identical bytes.
        if rounded == 0.0:
            rounded = 0.0
        if rounded == int(rounded):
            return int(rounded)
        return rounded
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_normalize(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _normalize(value) for key, value in obj.items()}
    raise TypeError(f"unsupported type for canonical serialization: {type(obj)!r}")


def canonical(obj) -> bytes:
    """Deterministic canonical serialization of ``obj`` to UTF-8 bytes.

    Dict keys are sorted; lists preserve order; tuples serialize like lists;
    floats are normalized (round to 1e-9, ``-0.0`` -> ``0.0``, integral floats
    collapse to int); ``NaN``/``inf`` raise ``ValueError``. Output is
    byte-identical across processes and Python builds.
    """
    normalized = _normalize(obj)
    text = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.encode("utf-8")


def code_version() -> str:
    """Return the recipe schema version as a string."""
    return str(RECIPE_SCHEMA_VERSION)


def git_sha() -> str | None:
    """Best-effort short git SHA for forensics; ``None`` on any failure.

    Never raises: any error (not a repo, git missing, timeout) yields ``None``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def recipe_id(kind: str, params: dict, code_version: str) -> str:
    """Content-address a recipe: ``H(kind, params, code_version)[:16]``."""
    digest = hashlib.sha256(canonical([kind, params, code_version]))
    return digest.hexdigest()[:16]


def fingerprint(
    kind: str,
    spatial_key: dict,
    recipe_id: str,
    input_fingerprints: list[str],
) -> str:
    """Merkle fingerprint of an artifact node.

    ``H(kind, spatial_key, recipe_id, sorted(input_fingerprints))[:24]``. The
    input list is sorted so edge ordering never affects the fingerprint.
    """
    digest = hashlib.sha256(
        canonical([kind, spatial_key, recipe_id, sorted(input_fingerprints)])
    )
    return digest.hexdigest()[:24]
