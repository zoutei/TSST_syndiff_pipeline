"""
ffi_naming.py
=============
TESS FFI product-id parsing and per-workspace FITS basename helpers.

Original SPOC FFIs on disk look like
``tess2020019142923-s0020-3-3-0165-s_ffic.fits``; the leading
``tess<digits>-s<sector>-<camera>-<ccd>`` substring (SPOC frame stem) uniquely
identifies the FFI epoch within one SCC. The pipeline's per-FFI FITS outputs
are written under SCC diff lanes and use the basename
``{ffi_frame_stem}_{label}.fits.fz`` so the file's directory determines what
stage it belongs to.

The helpers below are the single source of truth for that mapping.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional, Tuple

from syndiff_pipeline.common.fits_variants import (
    FITS_FPACK_EXT,
    FITS_GZIP_EXT,
    FITS_PLAIN_EXT,
    FITS_STORAGE_SUFFIXES,
    is_fits_storage_filename,
    iter_fits_variant_globs,
    resolve_stem_in_directory,
    storage_suffix_rank,
    strip_fits_storage_suffix,
)

# Canonical extension for pipeline FITS writes (CFITSIO fpack).
PIPELINE_FITS_EXT = FITS_FPACK_EXT
# Legacy read fallbacks (gzip-era and plain).
LEGACY_PIPELINE_FITS_EXT = FITS_PLAIN_EXT
LEGACY_PIPELINE_FITS_SUFFIXES = (FITS_GZIP_EXT, FITS_PLAIN_EXT)

# SPOC FFI frame stem: ``tess<digits>-s<sector>-<camera>-<ccd>``.
_SPOC_FRAME_STEM_RE = re.compile(r"^(tess\d+-s\d+-\d+-\d+)", re.I)
# Leading TESS product id token (``tess<digits>``) from any composite stem.
_TESS_PRODUCT_ID_RE = re.compile(r"^(tess\d+)", re.I)


def strip_fits_suffix(name: str) -> str:
    """Remove ``.fits.fz`` / ``.fits.gz`` / ``.fits`` from a basename or path."""
    return strip_fits_storage_suffix(name)


def is_pipeline_fits_filename(name: str) -> bool:
    """True for pipeline FITS basenames (``.fits.fz``, ``.fits.gz``, or ``.fits``)."""
    return is_fits_storage_filename(name)


def workspace_frame_fits_basename(stem: str) -> str:
    """Canonical pipeline FITS basename for a workspace frame stem."""
    return f"{stem}{PIPELINE_FITS_EXT}"


def workspace_frame_fits_path(directory: str, stem: str) -> str:
    """Absolute path for writing a pipeline FITS under *directory*."""
    return os.path.join(directory, workspace_frame_fits_basename(stem))


def resolve_pipeline_fits_path(directory: str, stem: str) -> Optional[str]:
    """
    Return an existing pipeline FITS path for *stem* under *directory*.

    Prefers ``.fits.fz`` over ``.fits.gz`` over legacy ``.fits``.
    """
    return resolve_stem_in_directory(directory, stem)


def resolve_pipeline_artifact_path(directory: str, basename: str) -> Optional[str]:
    """
    Resolve a fixed-basename pipeline artifact (e.g. ``shared_mask``).

    Tries storage variants when *basename* has no ext, otherwise tries the
    basename as given then sibling variants via stem resolution.
    """
    base = Path(str(basename)).name
    if is_pipeline_fits_filename(base):
        stem = strip_fits_suffix(base)
        return resolve_pipeline_fits_path(directory, stem)
    return resolve_pipeline_fits_path(directory, base)


def iter_pipeline_fits_paths(directory: str) -> list[str]:
    """
    Sorted pipeline FITS paths in *directory*, deduped by workspace stem.

    When multiple storage variants exist for the same stem, only the preferred
    path (``.fits.fz`` > ``.fits.gz`` > ``.fits``) is returned.
    """
    if not os.path.isdir(directory):
        return []
    by_stem: dict[str, str] = {}
    for pattern in iter_fits_variant_globs():
        for path in sorted(glob.glob(os.path.join(directory, pattern))):
            if not os.path.isfile(path):
                continue
            name = os.path.basename(path)
            if not is_pipeline_fits_filename(name):
                continue
            stem = strip_fits_suffix(name)
            existing = by_stem.get(stem)
            if existing is None or storage_suffix_rank(name) < storage_suffix_rank(
                existing
            ):
                by_stem[stem] = path
    return [by_stem[k] for k in sorted(by_stem)]


def tess_product_id_from_ffi_path(path_or_basename: str) -> Optional[str]:
    """
    Return the leading ``tess<digits>`` token from an FFI path or basename.

    Accepts the original SPOC name (``tess2020019142923-s0020-3-3-0165-s_ffic.fits``)
    or any composite workspace stem starting with the same product id
    (``tess2020019142923_hp_d``). Returns None when no match is found.
    """
    stem = strip_fits_suffix(Path(str(path_or_basename)).name)
    m = _TESS_PRODUCT_ID_RE.match(stem)
    return m.group(1) if m else None


def ffi_frame_stem_from_path(path_or_basename: str) -> str:
    """Return the SPOC frame stem ``tess<digits>-s<sector>-<camera>-<ccd>``."""
    stem = strip_fits_suffix(Path(str(path_or_basename)).name)
    m = _SPOC_FRAME_STEM_RE.match(stem)
    if not m:
        raise ValueError(
            f"cannot extract SPOC FFI frame stem from {path_or_basename!r}"
        )
    return m.group(1)


def sanitize_workspace_label(label: str) -> str:
    """Filesystem-safe label that matches ``os.path.basename(workspace_dir(...))``."""
    return str(label).replace(" ", "_")


def workspace_frame_stem(ffi_stem: str, label: str) -> str:
    """Compose the per-frame basename used for FITS files in ``ws/<label>/``."""
    return f"{ffi_stem}_{sanitize_workspace_label(label)}"


def scc_diff_artifact_stem(ffi_stem: str, label: str) -> str:
    """Stem for SCC diff lane artifacts (without FITS suffix)."""
    return workspace_frame_stem(ffi_stem, label)


def parse_workspace_frame_stem(frame_stem: str) -> Optional[Tuple[str, str]]:
    """
    Split ``{ffi_stem}_{label}`` back into ``(ffi_stem, label)``.

    *label* may contain underscores (e.g. ``epsf_r1``); the split is on the
    first ``_`` after the recognized FFI stem prefix. Returns None when
    *frame_stem* does not match the workspace pattern.
    """
    stem = strip_fits_suffix(str(frame_stem))
    m = _SPOC_FRAME_STEM_RE.match(stem)
    if not m:
        return None
    prefix = m.group(1)
    if not stem.startswith(f"{prefix}_"):
        return None
    label = stem[len(prefix) + 1 :]
    if not label:
        return None
    return prefix, label


def workspace_label_from_dir(workspace_dir_path: str) -> str:
    """Sanitized workspace label derived from a workspace directory path."""
    return sanitize_workspace_label(os.path.basename(os.path.abspath(workspace_dir_path)))


def workspace_frame_stem_for_dir(ffi_stem: str, workspace_dir_path: str) -> str:
    """``workspace_frame_stem`` that derives its label from the destination dir."""
    return workspace_frame_stem(ffi_stem, workspace_label_from_dir(workspace_dir_path))
