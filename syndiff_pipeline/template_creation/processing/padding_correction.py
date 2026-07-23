"""Cross-projection seam correction for the shared same-projection-only convolved store.

The shared ``convolved_skycell`` cache (``convolved_store.py``) holds a
canonical cell convolved using only same-projection neighbors (see
``doc/template_bookkeeping_plan.md`` SS13). For skycells whose mapping
requires cross-projection padding, that canonical cell is missing the
neighbor's contribution near the seam -- up to ~50% flux deficit at the
immediate edge (measured in ``tests/test_seam_correction_linearity.py``).

Gaussian convolution is linear (also validated in that test), so the exact
fix is additive: reproject the cross-projection neighbor patch alone (placed
at its true position in an otherwise-zero canvas the size of the canonical
cell), convolve that canvas with the same PSF, and add the result to the
canonical cell. This module computes that small correction, caches it on
disk (content-addressed by which neighbors/locations were used -- the actual
"remembering what was padded" piece), and combines it with the canonical
cell at consumption time.

Unlike the live ``ps1_process`` sliding-window loop (``cross_projection_padding.py``),
this runs standalone at consumption time: the padding-region WCS is built
directly in the recipient skycell's own pixel frame (via its
``master_skycells_list.csv`` row), not the live loop's row-tiled master-array
frame -- geometrically equivalent (both ultimately place the padding patch at
a world-coordinate position PAD_SIZE pixels beyond the recipient cell's edge),
but does not require the live per-row processing state. The neighbor's pixel
data comes from the shared ``combined_skycell`` store (already sky-keyed and
shared across SCCs), not the live in-memory row cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.wcs import WCS

log = logging.getLogger(__name__)

PAD_SIZE = 480
EDGE_EXCLUSION = 10

_PAD_COLUMNS = (
    "pad_skycell_top", "pad_skycell_right", "pad_skycell_top_right",
    "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_bottom_left",
    "pad_skycell_bottom_right", "pad_skycell_top_left",
)
_LOCATION_BY_COLUMN = {c: c.replace("pad_skycell_", "") for c in _PAD_COLUMNS}

_SEAM_CORRECTION_DIRNAME = "ps1_seam_correction"
_ARRAYS_FILENAME = "arrays.npz"
_SIDECAR_FILENAME = "_provenance.json"


def cross_projection_padding_spec(skycell_row: pd.Series) -> list[dict[str, str]]:
    """This skycell's cross-projection padding requirements: ``[{"neighbor":
    name, "location": loc}, ...]`` (empty if none needed). Same-projection
    padding entries (interior, handled by the live loop's same-projection
    logic already baked into the canonical cell) are excluded."""
    proj = str(skycell_row.get("projection", ""))
    spec: list[dict[str, str]] = []
    for col, location in _LOCATION_BY_COLUMN.items():
        val = skycell_row.get(col)
        if pd.isna(val) or not str(val).strip():
            continue
        for cell in str(val).split("/"):
            cell = cell.strip()
            if not cell:
                continue
            parts = cell.split(".")
            if len(parts) >= 2 and parts[1] != proj:
                spec.append({"neighbor": cell, "location": location})
    return spec


def padding_spec_fingerprint(spec: list[dict[str, str]]) -> str:
    """Content-addressed key for a padding spec (order-independent)."""
    canon = sorted((d["neighbor"], d["location"]) for d in spec)
    blob = json.dumps(canon, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _get_cd_matrix(row: pd.Series) -> list[list[float]]:
    if "CD1_1" in row.index and pd.notna(row.get("CD1_1")):
        return [[float(row.get(f"CD{i}_{j}", 0.0)) for j in (1, 2)] for i in (1, 2)]
    if "CDELT1" in row.index and pd.notna(row.get("CDELT1")):
        cdelt = [float(row.get(f"CDELT{i}", 1.0)) for i in (1, 2)]
        pc = [[float(row.get(f"PC{i}_{j}", 0.0 if i != j else 1.0)) for j in (1, 2)] for i in (1, 2)]
        return [[cdelt[i - 1] * pc[i - 1][j - 1] for j in (1, 2)] for i in (1, 2)]
    return [[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]]


def _cell_wcs(row: pd.Series) -> WCS:
    """Build a skycell's own native WCS from its master_skycells_list.csv row.

    Mirrors ``cross_projection_padding.create_cell_wcs`` exactly (same CRVAL/
    CRPIX/CD source columns) but takes a row directly instead of a name+lookup.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [float(row.get("CRVAL1", 0.0)), float(row.get("CRVAL2", 0.0))]
    wcs.wcs.crpix = [float(row.get("CRPIX1", 0.0)), float(row.get("CRPIX2", 0.0))]
    wcs.wcs.cd = _get_cd_matrix(row)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return wcs


def _exclude_edge_pixels(data: np.ndarray, edge_width: int = EDGE_EXCLUSION) -> np.ndarray:
    cleaned = data.copy()
    cleaned[:edge_width, :] = np.nan
    cleaned[-edge_width:, :] = np.nan
    cleaned[:, :edge_width] = np.nan
    cleaned[:, -edge_width:] = np.nan
    return cleaned


def _standalone_padding_wcs(
    recipient_wcs: WCS, cell_width: int, cell_height: int, location: str
) -> tuple[WCS, tuple[int, int], tuple[float, float]]:
    """Padding-patch WCS bordering *location* of the recipient cell.

    Adapted from ``cross_projection_padding.create_padding_wcs`` for use
    outside the live per-row master-array loop: positions are expressed in
    the recipient cell's OWN pixel frame ([0, cell_width) x [0, cell_height)),
    not a master-array frame with a PAD_SIZE offset + cell_index tiling --
    geometrically equivalent since both ultimately resolve to a world
    coordinate PAD_SIZE pixels beyond the recipient cell's edge, evaluated via
    ``recipient_wcs.pixel_to_world``.
    """
    pad_size_adjusted = PAD_SIZE + EDGE_EXCLUSION
    cell_x_center = cell_width / 2
    cell_y_center = cell_height / 2
    cell_y_top = cell_height + (PAD_SIZE - EDGE_EXCLUSION) / 2
    cell_y_bottom = -(PAD_SIZE - EDGE_EXCLUSION) / 2
    cell_x_left = -(PAD_SIZE - EDGE_EXCLUSION) / 2
    cell_x_right = cell_width + (PAD_SIZE - EDGE_EXCLUSION) / 2

    padding_x_center, padding_y_center = cell_x_center, cell_y_center
    padding_width = padding_height = pad_size_adjusted
    if "top" in location:
        padding_y_center, padding_width = cell_y_top, cell_width
    if "bottom" in location:
        padding_y_center, padding_width = cell_y_bottom, cell_width
    if "left" in location:
        padding_x_center, padding_height = cell_x_left, cell_height
    if "right" in location:
        padding_x_center, padding_height = cell_x_right, cell_height

    padding_center_world = recipient_wcs.pixel_to_world(padding_x_center, padding_y_center)
    padding_wcs = WCS(naxis=2)
    padding_wcs.wcs.crpix = [padding_width / 2, padding_height / 2]
    padding_wcs.wcs.crval = [padding_center_world.ra.degree, padding_center_world.dec.degree]
    padding_wcs.wcs.ctype = recipient_wcs.wcs.ctype
    if recipient_wcs.wcs.has_cd():
        padding_wcs.wcs.cd = recipient_wcs.wcs.cd.copy()
    else:
        padding_wcs.wcs.pc = recipient_wcs.wcs.pc.copy()
        padding_wcs.wcs.cdelt = recipient_wcs.wcs.cdelt.copy()
    return padding_wcs, (int(padding_height), int(padding_width)), (padding_y_center, padding_x_center)


def _discover_shared_combined_fp(data_root: str | Path, projection: str, cell: str) -> str | None:
    from syndiff_pipeline.common.scc_paths import ps1_combined_zarr_path

    cell_root = ps1_combined_zarr_path(data_root) / str(projection) / str(cell)
    if not cell_root.is_dir():
        return None
    candidates = []
    try:
        for fp_dir in cell_root.iterdir():
            if fp_dir.is_dir() and (fp_dir / "arrays.npz").is_file():
                candidates.append(fp_dir)
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return candidates[0].name


def _load_combined_image(data_root: str | Path, projection: str, cell: str) -> np.ndarray | None:
    from syndiff_pipeline.template_creation.processing.combined_store import try_load_combined_cell

    fp = _discover_shared_combined_fp(data_root, projection, cell)
    if fp is None:
        return None
    loaded = try_load_combined_cell(data_root, projection, cell, fp)
    if loaded is None:
        return None
    return np.asarray(loaded["combined_image"], dtype=np.float64)


def _seam_correction_cache_dir(
    data_root: str | Path, projection: str, skycell: str, padding_fp: str
) -> Path:
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path

    # Sibling tree next to the shared convolved store (not inside it -- this
    # is a small correction patch, not a full canonical cell).
    root = ps1_convolved_zarr_path(data_root).parent / _SEAM_CORRECTION_DIRNAME
    return root / str(projection) / str(skycell) / str(padding_fp)


def compute_seam_correction(
    *,
    data_root: str | Path,
    skycell: str,
    skycell_row: pd.Series,
    spec: list[dict[str, str]],
    canonical_shape: tuple[int, int],
    psf_sigma: float,
    skycell_df: pd.DataFrame,
) -> np.ndarray | None:
    """Additive correction (same shape as the canonical cell), or ``None`` if
    no neighbor patch could be placed (e.g. neighbor not yet processed
    anywhere -- caller falls back to the uncorrected canonical cell)."""
    from reproject import reproject_interp

    from syndiff_pipeline.template_creation.processing import convolution_utils
    from syndiff_pipeline.template_creation.processing.combined_store import _projection_and_cell

    cell_height, cell_width = canonical_shape
    recipient_wcs = _cell_wcs(skycell_row)
    canvas = np.zeros(canonical_shape, dtype=np.float64)
    placed_any = False

    for item in spec:
        neighbor, location = item["neighbor"], item["location"]
        if neighbor not in skycell_df.index:
            log.warning(
                "seam correction: neighbor %s not in master skycells list; skipping (skycell=%s)",
                neighbor, skycell,
            )
            continue
        parsed = _projection_and_cell(neighbor)
        if parsed is None:
            continue
        neighbor_proj, neighbor_cell = parsed
        data = _load_combined_image(data_root, neighbor_proj, neighbor_cell)
        if data is None:
            log.warning(
                "seam correction: no combined_skycell for neighbor %s (skycell=%s, location=%s); "
                "skipping this patch -- canonical cell used uncorrected for this edge.",
                neighbor, skycell, location,
            )
            continue
        data = _exclude_edge_pixels(data)
        source_wcs = _cell_wcs(skycell_df.loc[neighbor])

        target_wcs, target_shape, (y_center, x_center) = _standalone_padding_wcs(
            recipient_wcs, cell_width, cell_height, location
        )
        reprojected, footprint = reproject_interp(
            (data, source_wcs), target_wcs, shape_out=target_shape, order="bilinear"
        )

        h, w = target_shape
        y0, y1 = int(y_center - h / 2), int(y_center - h / 2) + h
        x0, x1 = int(x_center - w / 2), int(x_center - w / 2) + w
        cy0, cy1 = max(0, y0), min(cell_height, y1)
        cx0, cx1 = max(0, x0), min(cell_width, x1)
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        py0, py1 = cy0 - y0, cy1 - y0
        px0, px1 = cx0 - x0, cx1 - x0

        valid = (~np.isnan(reprojected)) & (footprint > 0)
        patch = np.where(valid, reprojected, 0.0)
        canvas[cy0:cy1, cx0:cx1] += patch[py0:py1, px0:px1]
        placed_any = True

    if not placed_any:
        return None

    # cval=0.0 (not the function's NaN default): this canvas is deliberately
    # zero everywhere except the placed patch(es), so the correction should
    # resolve to ~0 away from them -- NaN boundary fill would instead corrupt
    # a kernel-radius-wide strip along the array's own outer edges (unrelated
    # to the patch position) once added into the canonical cell.
    correction = convolution_utils.apply_gaussian_convolution(canvas, sigma=psf_sigma, cval=0.0)
    return correction


def load_padding_aware_convolved_cell(
    data_root: str | Path,
    skycell: str,
    *,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Canonical shared convolved cell, additively seam-corrected for any
    cross-projection padding this skycell's mapping requires. Falls back to
    the uncorrected canonical cell (with a warning) if the correction can't
    be computed (e.g. a neighbor hasn't been processed by any SCC yet).
    """
    from syndiff_pipeline.template_creation.processing.combined_store import _projection_and_cell
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _try_load_shared_convolved_arrays,
    )

    canonical = _try_load_shared_convolved_arrays(data_root, skycell)
    if canonical is None:
        return None
    image, mask = canonical

    if skycell not in skycell_df.index:
        return image, mask
    spec = cross_projection_padding_spec(skycell_df.loc[skycell])
    if not spec:
        return image, mask

    parsed = _projection_and_cell(skycell)
    if parsed is None:
        return image, mask
    projection, cell = parsed
    padding_fp = padding_spec_fingerprint(spec)
    cache_dir = _seam_correction_cache_dir(data_root, projection, cell, padding_fp)
    cache_path = cache_dir / _ARRAYS_FILENAME

    correction: np.ndarray | None = None
    if cache_path.is_file():
        try:
            with np.load(cache_path) as z:
                correction = np.asarray(z["correction"], dtype=np.float64)
        except Exception:
            log.warning("seam correction cache unreadable at %s; recomputing", cache_path, exc_info=True)
            correction = None

    if correction is None:
        correction = compute_seam_correction(
            data_root=data_root,
            skycell=skycell,
            skycell_row=skycell_df.loc[skycell],
            spec=spec,
            canonical_shape=image.shape,
            psf_sigma=psf_sigma,
            skycell_df=skycell_df,
        )
        if correction is not None:
            _publish_seam_correction(cache_dir, correction, spec)

    if correction is None:
        log.warning(
            "seam correction unavailable for %s (spec=%s); using uncorrected canonical cell "
            "(may be biased low near the cross-projection seam).",
            skycell, spec,
        )
        return image, mask

    return (image + correction.astype(image.dtype, copy=False)), mask


def _publish_seam_correction(cache_dir: Path, correction: np.ndarray, spec: list[dict[str, str]]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f"_tmp_{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tmp / _ARRAYS_FILENAME, correction=correction.astype(np.float32))
        (tmp / _SIDECAR_FILENAME).write_text(json.dumps({"spec": spec}, indent=2))
        for name in (_ARRAYS_FILENAME, _SIDECAR_FILENAME):
            os.replace(tmp / name, cache_dir / name)
        tmp.rmdir()
    except Exception:
        log.warning("Failed to publish seam correction cache to %s (non-fatal)", cache_dir, exc_info=True)
