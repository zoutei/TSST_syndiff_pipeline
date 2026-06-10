"""
ffi_naming.py
=============
TESS FFI product-id parsing and per-workspace FITS basename helpers.

Original SPOC FFIs on disk look like
``tess2020019142923-s0020-3-3-0165-s_ffic.fits``; the leading ``tess<digits>``
substring uniquely identifies the FFI epoch (sector / camera / CCD / cadence
are encoded redundantly in path layout). The pipeline's per-FFI FITS outputs
are written into ``ws/<label>/`` workspaces and use the basename
``{ffi_product_id}_{label}.fits`` so the file's directory determines what
stage it belongs to.

The helpers below are the single source of truth for that mapping.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

# ``tess<digits>`` (case-insensitive) at the start of the FFI basename.
_TESS_PRODUCT_ID_RE = re.compile(r"^(tess\d+)", re.IGNORECASE)

# Composite workspace stem: ``tess<digits>_<label>`` (label may contain ``_``).
_WORKSPACE_FRAME_STEM_RE = re.compile(r"^(tess\d+)_(.+)$", re.IGNORECASE)


def tess_product_id_from_ffi_path(path_or_basename: str) -> Optional[str]:
    """
    Return the leading ``tess<digits>`` token from an FFI path or basename.

    Accepts the original SPOC name (``tess2020019142923-s0020-3-3-0165-s_ffic.fits``)
    or any composite workspace stem starting with the same product id
    (``tess2020019142923_hp_d``). Returns None when no match is found.
    """
    name = Path(str(path_or_basename)).name
    stem = name[:-5] if name.lower().endswith(".fits") else os.path.splitext(name)[0]
    m = _TESS_PRODUCT_ID_RE.match(stem)
    return m.group(1) if m else None


def sanitize_workspace_label(label: str) -> str:
    """Filesystem-safe label that matches ``os.path.basename(workspace_dir(...))``."""
    return str(label).replace(" ", "_")


def workspace_frame_stem(product_id: str, label: str) -> str:
    """Compose the per-frame basename used for FITS files in ``ws/<label>/``."""
    return f"{product_id}_{sanitize_workspace_label(label)}"


def parse_workspace_frame_stem(frame_stem: str) -> Optional[Tuple[str, str]]:
    """
    Split ``tess<digits>_<label>`` back into ``(product_id, label)``.

    Returns None when *frame_stem* does not match the workspace pattern.
    """
    m = _WORKSPACE_FRAME_STEM_RE.match(str(frame_stem))
    if not m:
        return None
    return m.group(1), m.group(2)


def workspace_label_from_dir(workspace_dir_path: str) -> str:
    """Sanitized workspace label derived from a workspace directory path."""
    return sanitize_workspace_label(os.path.basename(os.path.abspath(workspace_dir_path)))


def workspace_frame_stem_for_dir(product_id: str, workspace_dir_path: str) -> str:
    """``workspace_frame_stem`` that derives its label from the destination dir."""
    return workspace_frame_stem(product_id, workspace_label_from_dir(workspace_dir_path))
