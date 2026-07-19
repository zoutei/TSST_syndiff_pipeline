"""
fits_variants.py
================
Tri-format FITS path helpers: ``.fits.fz`` (preferred), ``.fits.gz``, ``.fits``.

All SynDiff FITS discovery, stem stripping, and manifest path keys go through
this module so readers accept any on-disk variant while writers target fpack.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Preferred → legacy read/write preference order.
FITS_FPACK_EXT = ".fits.fz"
FITS_GZIP_EXT = ".fits.gz"
FITS_PLAIN_EXT = ".fits"

FITS_STORAGE_SUFFIXES: tuple[str, ...] = (
    FITS_FPACK_EXT,
    FITS_GZIP_EXT,
    FITS_PLAIN_EXT,
)

# Longer suffixes first so strip/match is unambiguous.
_STORAGE_SUFFIXES_LONGEST_FIRST: tuple[str, ...] = tuple(
    sorted(FITS_STORAGE_SUFFIXES, key=len, reverse=True)
)


def is_fits_storage_filename(name: str) -> bool:
    """True when *name* ends with a known FITS storage suffix."""
    lower = Path(str(name)).name.lower()
    return any(lower.endswith(sfx) for sfx in _STORAGE_SUFFIXES_LONGEST_FIRST)


def is_compressed_fits_path(path: str | Path) -> bool:
    """True for ``.fits.fz`` or ``.fits.gz`` (not plain ``.fits``)."""
    lower = Path(str(path)).name.lower()
    return lower.endswith(FITS_FPACK_EXT) or lower.endswith(FITS_GZIP_EXT)


def strip_fits_storage_suffix(name: str) -> str:
    """Strip ``.fits.fz`` / ``.fits.gz`` / ``.fits`` from a basename or path."""
    base = Path(str(name)).name
    lower = base.lower()
    for sfx in _STORAGE_SUFFIXES_LONGEST_FIRST:
        if lower.endswith(sfx):
            return base[: -len(sfx)]
    return os.path.splitext(base)[0]


def fits_logical_path(path: str | Path) -> str:
    """Normalize any storage variant to ``…/stem.fits`` (logical identity)."""
    p = Path(os.path.expanduser(str(path)))
    stem = strip_fits_storage_suffix(p.name)
    return str(p.with_name(f"{stem}{FITS_PLAIN_EXT}"))


def fits_fpack_path(path: str | Path) -> str:
    """Rewrite *path* to the canonical ``.fits.fz`` write target."""
    logical = fits_logical_path(path)
    if logical.lower().endswith(FITS_PLAIN_EXT):
        return logical + ".fz"
    return logical


def storage_suffix_rank(name: str) -> int:
    """Lower rank = preferred. Unknown suffixes rank worst."""
    lower = Path(str(name)).name.lower()
    for i, sfx in enumerate(FITS_STORAGE_SUFFIXES):
        if lower.endswith(sfx):
            return i
    return len(FITS_STORAGE_SUFFIXES)


def iter_fits_variant_globs() -> tuple[str, ...]:
    """Glob patterns for directory scans (preferred first)."""
    return ("*.fits.fz", "*.fits.gz", "*.fits")


def variant_paths_for_logical(logical_fits_path: str | Path) -> list[Path]:
    """Candidate on-disk paths for a logical ``….fits`` path, preference order."""
    logical = Path(fits_logical_path(logical_fits_path))
    parent = logical.parent
    stem = strip_fits_storage_suffix(logical.name)
    return [parent / f"{stem}{sfx}" for sfx in FITS_STORAGE_SUFFIXES]


def resolve_fits_variant(path: str | Path) -> Path:
    """
    Return an on-disk FITS path for *path*, trying sibling storage variants.

    Preference when the exact path is missing: ``.fits.fz`` → ``.fits.gz`` → ``.fits``.
    If the exact path exists, it is returned as-is (even if a preferred sibling exists).
    """
    expanded = Path(os.path.expanduser(str(path)))
    if expanded.is_file():
        return expanded

    for candidate in variant_paths_for_logical(expanded):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not find FITS file: {path}")


def try_resolve_fits_variant(path: str | Path) -> Path | None:
    """Like :func:`resolve_fits_variant`, but return ``None`` when missing."""
    try:
        return resolve_fits_variant(path)
    except FileNotFoundError:
        return None


def fits_path_exists(path: str | Path) -> bool:
    """True when *path* resolves to an on-disk FITS file (any storage variant)."""
    return try_resolve_fits_variant(path) is not None


def canonical_fits_path_key(path: str | Path) -> str:
    """
    Comparison key treating ``.fits`` / ``.fits.gz`` / ``.fits.fz`` as the same file.

    Uses ``realpath`` so symlinked data roots still match.
    """
    expanded = os.path.realpath(os.path.expanduser(str(path)))
    return fits_logical_path(expanded)


def prefer_fits_path(paths: Iterable[str | Path]) -> str | None:
    """Pick the preferred existing path among candidates (fz > gz > plain)."""
    best: str | None = None
    best_rank = len(FITS_STORAGE_SUFFIXES) + 1
    for p in paths:
        ps = str(p)
        if not os.path.isfile(ps):
            continue
        rank = storage_suffix_rank(ps)
        if rank < best_rank:
            best = ps
            best_rank = rank
    return best


def resolve_stem_in_directory(directory: str | Path, stem: str) -> str | None:
    """Resolve ``{stem}.fits.fz`` / ``.fits.gz`` / ``.fits`` under *directory*."""
    root = Path(directory)
    for sfx in FITS_STORAGE_SUFFIXES:
        candidate = root / f"{stem}{sfx}"
        if candidate.is_file():
            return str(candidate)
    return None
