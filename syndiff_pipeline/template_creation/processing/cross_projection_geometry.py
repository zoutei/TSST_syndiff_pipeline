"""Shared fixed-PS1 geometry for cross-projection padding.

Both the live ``ps1_process`` producer and the shared-store correction must
use these functions.  Coordinates are zero-based, half-open pixel coordinates
in the frame of ``reference_wcs``; no TESS WCS participates here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from astropy.wcs import WCS

PAD_SIZE = 480
EDGE_EXCLUSION = 10


def projection_identity(value: object) -> str:
    """Normalize a PS1 projection label for scientific equality checks."""
    text = str(value).strip()
    parts = text.split(".")
    if len(parts) >= 2 and parts[0] == "skycell":
        return parts[1]
    return text


def skycell_projection_identity(skycell: object) -> str | None:
    """Return the normalized projection embedded in ``skycell.PROJ.CELL``."""
    parts = str(skycell).strip().split(".")
    if len(parts) < 3 or parts[0] != "skycell":
        return None
    return projection_identity(parts[1])

_LOCATIONS = frozenset(
    {
        "top", "bottom", "left", "right",
        "top_left", "top_right", "bottom_left", "bottom_right",
    }
)


@dataclass(frozen=True)
class PaddingPatchGeometry:
    """Exact target geometry for one producer padding box."""

    wcs: WCS
    shape: tuple[int, int]
    center: tuple[float, float]  # (y, x), in reference-WCS pixel coordinates
    bounds: tuple[int, int, int, int]  # (y0, y1, x0, x1), half-open


def normalize_location(location: str) -> str:
    value = str(location).strip().lower()
    if value not in _LOCATIONS:
        raise ValueError(f"unsupported cross-projection padding location: {location!r}")
    return value


def cd_matrix_from_row(row: pd.Series) -> list[list[float]]:
    """Return the PS1 CD matrix using the producer's existing fallback order."""
    if "CD1_1" in row.index and pd.notna(row.get("CD1_1")):
        return [[float(row.get(f"CD{i}_{j}", 0.0)) for j in (1, 2)] for i in (1, 2)]
    if "CDELT1" in row.index and pd.notna(row.get("CDELT1")):
        cdelt = [float(row.get(f"CDELT{i}", 1.0)) for i in (1, 2)]
        pc = [
            [float(row.get(f"PC{i}_{j}", 0.0 if i != j else 1.0)) for j in (1, 2)]
            for i in (1, 2)
        ]
        return [[cdelt[i - 1] * pc[i - 1][j - 1] for j in (1, 2)] for i in (1, 2)]
    return [[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]]


def cell_wcs_from_row(row: pd.Series) -> WCS:
    """Build a PS1 skycell WCS from one master-skycell table row."""
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [float(row.get("CRVAL1", 0.0)), float(row.get("CRVAL2", 0.0))]
    wcs.wcs.crpix = [float(row.get("CRPIX1", 0.0)), float(row.get("CRPIX2", 0.0))]
    wcs.wcs.cd = cd_matrix_from_row(row)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def padding_patch_geometry(
    reference_wcs: WCS,
    *,
    recipient_x0: float,
    recipient_y0: float,
    cell_width: int,
    cell_height: int,
    location: str,
    pad_size: int = PAD_SIZE,
    edge_exclusion: int = EDGE_EXCLUSION,
) -> PaddingPatchGeometry:
    """Return the exact target WCS, shape, and bounds for one padding box.

    ``recipient_x0``/``recipient_y0`` locate the nominal recipient cell in
    the reference WCS frame.  Passing zero for both produces the standalone
    recipient-native geometry; passing the live row-master origin reproduces
    ``ps1_process`` placement.  Corners are deliberately explicit rather
    than substring-based, preventing a corner from becoming a full-cell box.
    """
    loc = normalize_location(location)
    width, height = int(cell_width), int(cell_height)
    p, e = int(pad_size), int(edge_exclusion)
    if width <= 0 or height <= 0 or p <= 0 or e < 0:
        raise ValueError("cell dimensions/padding constants must be positive")

    is_top = loc in {"top", "top_left", "top_right"}
    is_bottom = loc in {"bottom", "bottom_left", "bottom_right"}
    is_left = loc in {"left", "top_left", "bottom_left"}
    is_right = loc in {"right", "top_right", "bottom_right"}
    is_corner = "_" in loc
    patch = p + e

    center_x = float(recipient_x0) + width / 2
    center_y = float(recipient_y0) + height / 2
    out_width = out_height = patch
    if is_top:
        center_y = float(recipient_y0) + height + (p - e) / 2
    elif is_bottom:
        center_y = float(recipient_y0) - (p - e) / 2
    if is_left:
        center_x = float(recipient_x0) - (p - e) / 2
    elif is_right:
        center_x = float(recipient_x0) + width + (p - e) / 2
    if not is_corner:
        if is_top or is_bottom:
            out_width = width
        if is_left or is_right:
            out_height = height

    world = reference_wcs.pixel_to_world(center_x, center_y)
    target = WCS(naxis=2)
    target.wcs.crpix = [out_width / 2, out_height / 2]
    target.wcs.crval = [world.ra.degree, world.dec.degree]
    target.wcs.ctype = reference_wcs.wcs.ctype
    if reference_wcs.wcs.has_cd():
        target.wcs.cd = reference_wcs.wcs.cd.copy()
    else:
        target.wcs.pc = reference_wcs.wcs.pc.copy()
        target.wcs.cdelt = reference_wcs.wcs.cdelt.copy()

    y0 = int(center_y - out_height / 2)
    x0 = int(center_x - out_width / 2)
    return PaddingPatchGeometry(
        wcs=target,
        shape=(int(out_height), int(out_width)),
        center=(center_y, center_x),
        bounds=(y0, y0 + int(out_height), x0, x0 + int(out_width)),
    )


def valid_reprojection_footprint(reprojected: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """The producer's valid-pixel rule for a reprojected padding patch."""
    return np.isfinite(reprojected) & (np.asarray(footprint) > 0)


@dataclass(frozen=True)
class CrossProjectionWorkImageGeometry:
    """Local work-image geometry for one standalone cross-projection correction.

    ``work_bounds``/``work_shape`` describe the padding-box placement from
    ``patch`` extended inward (toward the recipient interior) by the
    convolution radius along the perpendicular axis (edges) or both axes
    (corners) -- see ``shared_convolved_cross_projection_simple_fix_plan.md``.
    All bounds are half-open ``(y0, y1, x0, x1)`` in the recipient skycell's
    own native pixel frame (``[0, cell_width) x [0, cell_height)``); they may
    be negative or extend beyond the cell on the exterior side.
    """

    patch: PaddingPatchGeometry
    work_bounds: tuple[int, int, int, int]
    work_shape: tuple[int, int]


def padding_work_image_geometry(
    reference_wcs: WCS,
    *,
    cell_width: int,
    cell_height: int,
    location: str,
    pad_size: int = PAD_SIZE,
    edge_exclusion: int = EDGE_EXCLUSION,
    inward_radius: int = 470,
) -> CrossProjectionWorkImageGeometry:
    """Return the standalone padding-box + inward-extension work-image geometry.

    The padding box itself (``patch.bounds``) is exactly
    ``padding_patch_geometry``'s recipient-native placement (matching
    ``ps1_process`` geometry). The inward extension receives convolved flux
    from the reprojected neighbor without holding any source data of its own
    (it stays zero until convolution).
    """
    r = int(inward_radius)
    if r <= 0:
        raise ValueError("inward_radius must be positive")
    loc = normalize_location(location)
    width, height = int(cell_width), int(cell_height)

    patch = padding_patch_geometry(
        reference_wcs,
        recipient_x0=0,
        recipient_y0=0,
        cell_width=width,
        cell_height=height,
        location=loc,
        pad_size=pad_size,
        edge_exclusion=edge_exclusion,
    )
    y0, y1, x0, x1 = patch.bounds

    is_top = loc in {"top", "top_left", "top_right"}
    is_bottom = loc in {"bottom", "bottom_left", "bottom_right"}
    is_left = loc in {"left", "top_left", "bottom_left"}
    is_right = loc in {"right", "top_right", "bottom_right"}

    # Extend the boundary that sits nearest the recipient interior further
    # inward by R; the far (exterior) boundary is untouched.
    if is_top:
        y0 -= r
    if is_bottom:
        y1 += r
    if is_left:
        x1 += r
    if is_right:
        x0 -= r

    return CrossProjectionWorkImageGeometry(
        patch=patch,
        work_bounds=(y0, y1, x0, x1),
        work_shape=(y1 - y0, x1 - x0),
    )


def intersect_bounds(
    bounds: tuple[int, int, int, int], *, cell_width: int, cell_height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a half-open ``(y0, y1, x0, x1)`` box to the recipient cell extent.

    Returns ``None`` when the intersection with ``[0, cell_height) x
    [0, cell_width)`` is empty.
    """
    y0, y1, x0, x1 = bounds
    cy0, cy1 = max(0, int(y0)), min(int(cell_height), int(y1))
    cx0, cx1 = max(0, int(x0)), min(int(cell_width), int(x1))
    if cy0 >= cy1 or cx0 >= cx1:
        return None
    return cy0, cy1, cx0, cx1
