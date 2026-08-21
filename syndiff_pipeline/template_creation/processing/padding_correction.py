"""Cross-projection seam-correction primitives for the shared canonical store.

The shared ``convolved_skycell`` cache (``convolved_store.py``) holds a
canonical cell convolved using only same-projection neighbors. For skycells
whose mapping requires cross-projection padding, that canonical cell is
missing the neighbor's contribution near the seam -- up to ~50% flux deficit
at the immediate edge (measured in ``tests/test_seam_correction_linearity.py``).

This module implements the standalone, per-load correction described in
``doc/shared_convolved_cross_projection_simple_fix_plan.md``: for each
required cross-projection edge/corner, it builds a small local
same-projection/fully-padded pair over the padding box extended inward by the
convolution radius, convolves that local domain once
(``convolve_local_padding_delta``), and adds the recipient-side result into
the canonical convolved image. It does **not** touch ``ps1_process`` and does
**not** require any producer-persisted context payload -- everything needed
(the recipient's own pre-convolution mosaic and each neighbor's) is read
directly from the shared ``combined_skycell`` store at consumption time.

Unlike the live ``ps1_process`` sliding-window loop (``cross_projection_padding.py``),
this runs standalone at consumption time: the padding-region WCS is built
directly in the recipient skycell's own pixel frame (via its
``master_skycells_list.csv`` row), not the live loop's row-tiled master-array
frame -- geometrically equivalent (both ultimately place the padding patch at
a world-coordinate position PAD_SIZE pixels beyond the recipient cell's edge),
but does not require the live per-row processing state. The neighbor's pixel
data comes from the shared ``combined_skycell`` store (already sky-keyed and
shared across SCCs), not the live in-memory row cache.

Runs once per loaded shared skycell -- independent of linear/field mode,
group id, or TESS WCS -- so both downsample modes consume the identical
corrected array (``linear_downsample.py`` and ``field_downsample.py`` both
call ``load_padding_aware_convolved_cell``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing.cross_projection_geometry import (
    cell_wcs_from_row,
    cd_matrix_from_row,
    padding_patch_geometry,
    padding_work_image_geometry,
    intersect_bounds,
    valid_reprojection_footprint,
    projection_identity,
    skycell_projection_identity,
)

log = logging.getLogger(__name__)


PAD_SIZE = 480
EDGE_EXCLUSION = 10

# Bump whenever the correction *algorithm* changes shape/placement math (not
# just spec contents) -- folded into padding_spec_fingerprint so any
# telemetry/logging keyed on the spec identity distinguishes runs computed
# under an older, buggy version of the placement math. v2: fixed
# _standalone_padding_wcs's corner-location patch sizing -- "top_right" etc.
# used to match both the "top" and "right" substring checks and get resized
# to the FULL cell in both axes instead of a small corner square, pasting a
# large fraction of the diagonal neighbor's real image onto the recipient
# (a real double-count of star flux, not just a wider seam blend).
CORRECTION_ALGORITHM_VERSION = 2

_PAD_COLUMNS = (
    "pad_skycell_top", "pad_skycell_right", "pad_skycell_top_right",
    "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_bottom_left",
    "pad_skycell_bottom_right", "pad_skycell_top_left",
)
_LOCATION_BY_COLUMN = {c: c.replace("pad_skycell_", "") for c in _PAD_COLUMNS}
_COMPOSITION_COLUMNS = (
    "pad_skycell_top", "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_right",
    "pad_skycell_top_left", "pad_skycell_top_right", "pad_skycell_bottom_left",
    "pad_skycell_bottom_right",
)


class PaddingCorrectionError(RuntimeError):
    """The exact cross-projection correction cannot be constructed safely."""


def convolve_local_padding_delta(
    same_projection_input: np.ndarray,
    fully_padded_input: np.ndarray,
    *,
    local_origin_xy: tuple[int, int],
    canonical_shape: tuple[int, int],
    psf_sigma: float,
    kernel_radius: int = 470,
) -> np.ndarray:
    """Return the recipient crop of ``C(F - A)`` without pre-convolution clipping.

    ``same_projection_input`` (``A``) and ``fully_padded_input`` (``F``)
    must describe the *same finite local pixel domain*: it includes the
    recipient pixels whose convolution may change and enough exterior halo
    to contain every contributing cross-projection padding patch.  Their
    origin is expressed in the recipient skycell's native ``(x, y)`` frame;
    it may therefore be negative or extend beyond ``canonical_shape``.

    This is deliberately a small, geometry-free primitive.  The caller is
    responsible for constructing A with the exact same-projection ownership
    rules used by ``ps1_process`` and F by applying the producer's
    cross-projection replacement logic.  Keeping the whole local domain
    through convolution is essential: clipping a patch to the recipient
    image before convolution discards the exterior flux that must blur back
    into the recipient edge.
    """
    from syndiff_pipeline.template_creation.processing import convolution_utils

    a = np.asarray(same_projection_input, dtype=np.float64)
    f = np.asarray(fully_padded_input, dtype=np.float64)
    if a.ndim != 2 or f.ndim != 2 or a.shape != f.shape:
        raise PaddingCorrectionError(
            "same_projection_input and fully_padded_input must be same-shaped 2-D arrays; "
            f"got {a.shape} and {f.shape}"
        )
    cell_height, cell_width = (int(canonical_shape[0]), int(canonical_shape[1]))
    if cell_height <= 0 or cell_width <= 0:
        raise PaddingCorrectionError(f"invalid canonical shape {canonical_shape}")

    # ``ps1_process`` fills uncovered mosaic pixels with zero immediately
    # before convolution.  Apply the same convention independently to A and
    # F before forming D, so NaN represents absent flux rather than a NaN
    # that contaminates the complete kernel support.
    delta = np.nan_to_num(f, nan=0.0) - np.nan_to_num(a, nan=0.0)
    convolved_delta = convolution_utils.apply_gaussian_convolution(
        delta, sigma=psf_sigma, radius=kernel_radius, cval=0.0
    )

    x0, y0 = (int(local_origin_xy[0]), int(local_origin_xy[1]))
    x1, y1 = x0 + delta.shape[1], y0 + delta.shape[0]
    dst_x0, dst_x1 = max(0, x0), min(cell_width, x1)
    dst_y0, dst_y1 = max(0, y0), min(cell_height, y1)
    correction = np.zeros((cell_height, cell_width), dtype=np.float64)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return correction
    src_x0, src_x1 = dst_x0 - x0, dst_x1 - x0
    src_y0, src_y1 = dst_y0 - y0, dst_y1 - y0
    correction[dst_y0:dst_y1, dst_x0:dst_x1] = convolved_delta[
        src_y0:src_y1, src_x0:src_x1
    ]
    return correction


def cross_projection_padding_spec(skycell_row: pd.Series) -> list[dict[str, str]]:
    """This skycell's cross-projection padding requirements: ``[{"neighbor":
    name, "location": loc}, ...]`` (empty if none needed). Same-projection
    padding entries (interior, handled by the live loop's same-projection
    logic already baked into the canonical cell) are excluded."""
    proj = projection_identity(skycell_row.get("projection", ""))
    spec: list[dict[str, str]] = []
    for col in _COMPOSITION_COLUMNS:
        location = _LOCATION_BY_COLUMN[col]
        val = skycell_row.get(col)
        if pd.isna(val) or not str(val).strip():
            continue
        for cell in str(val).split("/"):
            cell = cell.strip()
            if not cell:
                continue
            source_projection = skycell_projection_identity(cell)
            if source_projection is not None and source_projection != proj:
                spec.append({"neighbor": cell, "location": location})
    return spec


def padding_spec_fingerprint(spec: list[dict[str, str]]) -> str:
    """Content-addressed key for a padding spec with preserved source priority.

    Includes ``CORRECTION_ALGORITHM_VERSION`` so a correction computed under
    an older, buggy version of the placement math never gets reused just
    because the spec (neighbor/location list) happens to match -- see that
    constant's docstring.
    """
    canon = [(d["neighbor"], d["location"]) for d in spec]
    blob = json.dumps(
        {"v": CORRECTION_ALGORITHM_VERSION, "spec": canon}, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _get_cd_matrix(row: pd.Series) -> list[list[float]]:
    return cd_matrix_from_row(row)


def _cell_wcs(row: pd.Series) -> WCS:
    """Build a skycell's own native WCS from its master_skycells_list.csv row.

    Mirrors ``cross_projection_padding.create_cell_wcs`` exactly (same CRVAL/
    CRPIX/CD source columns) but takes a row directly instead of a name+lookup.
    """
    return cell_wcs_from_row(row)


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
    geometry = padding_patch_geometry(
        recipient_wcs,
        recipient_x0=0,
        recipient_y0=0,
        cell_width=cell_width,
        cell_height=cell_height,
        location=location,
        pad_size=PAD_SIZE,
        edge_exclusion=EDGE_EXCLUSION,
    )
    y_center, x_center = geometry.center
    return geometry.wcs, geometry.shape, (y_center, x_center)


def _discover_shared_combined_fp(
    data_root: str | Path,
    projection: str,
    cell: str,
    *,
    combined_recipe: Mapping | None = None,
) -> str | None:
    """Return a published combined-cell fingerprint dirname, or ``None``.

    Resolution order (never trust mtime when a recipe is known -- the store
    is shared cross-sector/cross-run and a different, unrelated recipe may
    have been published *later* for the exact same sky cell):

    1. ``combined_recipe`` given: deterministically recompute the fingerprint
       that recipe maps to (``combined_store.resolve_combined_fingerprint_for_recipe``).
       This is the only way to guarantee we read back exactly what this run's
       own config would have produced.
    Recipe-qualified calls are deliberately fail-closed: a ``current``
    pointer and directory mtime are not provenance, and must never select a
    different star-removal/saturation recipe.  The compatibility fallback is
    retained only for callers that genuinely provide no recipe context.
    """
    from syndiff_pipeline.template_creation.processing.combined_store import (
        _payload_complete,
        _ps1_combined_zarr_root,
        resolve_combined_fingerprint_for_recipe,
        resolve_current_combined_ref,
    )

    if combined_recipe is not None:
        fp = resolve_combined_fingerprint_for_recipe(data_root, projection, cell, combined_recipe)
        if fp is not None:
            return fp
        log.error(
            "padding_correction: exact shared combined artifact missing for %s/%s; "
            "refusing current-pointer or mtime fallback",
            projection,
            cell,
        )
        return None

    ref = resolve_current_combined_ref(data_root, projection, cell)
    if ref is not None:
        return ref.fingerprint

    cell_root = _ps1_combined_zarr_root(data_root) / str(projection) / str(cell)
    if not cell_root.is_dir():
        return None
    candidates: list[Path] = []
    try:
        for fp_dir in cell_root.iterdir():
            if fp_dir.is_dir() and _payload_complete(fp_dir):
                candidates.append(fp_dir)
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return candidates[0].name


def _load_combined_image(
    data_root: str | Path,
    projection: str,
    cell: str,
    *,
    combined_recipe: Mapping | None = None,
) -> np.ndarray | None:
    from syndiff_pipeline.template_creation.processing.combined_store import try_load_combined_cell

    fp = _discover_shared_combined_fp(data_root, projection, cell, combined_recipe=combined_recipe)
    if fp is None:
        return None
    loaded = try_load_combined_cell(data_root, projection, cell, fp)
    if loaded is None:
        return None
    return np.asarray(loaded["combined_image"], dtype=np.float64)


def _grouped_padding_spec(spec: list[dict[str, str]]) -> dict[str, list[str]]:
    """Group an ordered padding spec by location, preserving neighbor order."""
    grouped: dict[str, list[str]] = {}
    for item in spec:
        grouped.setdefault(item["location"], []).append(item["neighbor"])
    return grouped


def _location_correction(
    *,
    location: str,
    neighbors: list[str],
    skycell: str,
    recipient_wcs: WCS,
    cell_shape: tuple[int, int],
    own_combined: np.ndarray,
    data_root: str | Path,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
    kernel_radius: int,
    combined_recipe: Mapping | None = None,
) -> np.ndarray:
    """Return one location's convolved, recipient-cropped correction.

    Builds the same-projection (``A``) / fully-padded (``F``) pair over the
    standalone work-image domain (the padding box extended inward by
    ``kernel_radius``) and hands it to ``convolve_local_padding_delta``. ``A``
    is zero outside the recipient's own native overlap strip (the exact
    same-projection input the producer would have convolved there); ``F``
    replaces that strip -- and fills the exterior padding box -- with the
    ordered, composed cross-projection source values, mirroring
    ``ps1_process``'s replacement (not addition) rule for the overlap.
    """
    from reproject import reproject_interp

    from syndiff_pipeline.template_creation.processing.combined_store import (
        _projection_and_cell,
    )

    height, width = cell_shape
    geometry = padding_work_image_geometry(
        recipient_wcs, cell_width=width, cell_height=height,
        location=location, inward_radius=kernel_radius,
    )
    wy0, wy1, wx0, wx1 = geometry.work_bounds
    py0, py1, px0, px1 = geometry.patch.bounds
    ly0, lx0 = py0 - wy0, px0 - wx0
    patch_h, patch_w = geometry.patch.shape

    full_patch = np.zeros((patch_h, patch_w), dtype=np.float64)
    cum_valid = np.zeros((patch_h, patch_w), dtype=bool)

    for neighbor in neighbors:
        if neighbor not in skycell_df.index:
            raise PaddingCorrectionError(
                f"required cross-projection source {neighbor} for {skycell}/{location} "
                "is not in the master skycells table"
            )
        source_parsed = _projection_and_cell(neighbor)
        if source_parsed is None:
            raise PaddingCorrectionError(f"cannot resolve identity of source {neighbor}")
        source_projection, source_cell = source_parsed
        source_image = _load_combined_image(
            data_root, source_projection, source_cell, combined_recipe=combined_recipe,
        )
        if source_image is None:
            raise PaddingCorrectionError(
                f"required combined skycell {neighbor} for {skycell}/{location} is unavailable"
            )
        source_image = _exclude_edge_pixels(source_image)
        source_wcs = _cell_wcs(skycell_df.loc[neighbor])
        reprojected, footprint = reproject_interp(
            (source_image, source_wcs), geometry.patch.wcs,
            shape_out=geometry.patch.shape, order="bilinear",
        )
        valid = valid_reprojection_footprint(reprojected, footprint)
        # Replacement order: mapping-table location order, then slash-source
        # order -- a later neighbor overwrites an earlier one wherever both
        # are valid, exactly as ``ps1_process`` composes these boxes.
        full_patch[valid] = reprojected[valid]
        cum_valid |= valid

    same_local = np.zeros(geometry.work_shape, dtype=np.float64)
    full_local = np.zeros(geometry.work_shape, dtype=np.float64)
    full_local[ly0:ly0 + patch_h, lx0:lx0 + patch_w] = full_patch

    # Native E-pixel overlap strip: the delta input there must be
    # (reprojected source - existing same-projection value), i.e. ``A`` must
    # carry the recipient's own combined value wherever a source actually
    # replaced it. Outside the overlap (purely exterior patch pixels), A is
    # implicitly zero -- already satisfied by the zero-initialized array.
    overlap = intersect_bounds((py0, py1, px0, px1), cell_width=width, cell_height=height)
    if overlap is not None:
        oy0, oy1, ox0, ox1 = overlap
        poy0, poy1 = oy0 - py0, oy1 - py0
        pox0, pox1 = ox0 - px0, ox1 - px0
        own_slice = own_combined[oy0:oy1, ox0:ox1]
        own_valid = np.isfinite(own_slice) & cum_valid[poy0:poy1, pox0:pox1]
        overlap_a = np.zeros((poy1 - poy0, pox1 - pox0), dtype=np.float64)
        overlap_a[own_valid] = own_slice[own_valid]
        same_local[ly0 + poy0:ly0 + poy1, lx0 + pox0:lx0 + pox1] = overlap_a

    return convolve_local_padding_delta(
        same_local, full_local, local_origin_xy=(wx0, wy0),
        canonical_shape=cell_shape, psf_sigma=psf_sigma, kernel_radius=kernel_radius,
    )


def load_padding_aware_convolved_cell(
    data_root: str | Path,
    skycell: str,
    *,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
    kernel_radius: int = 470,
    combined_recipe: Mapping | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one shared canonical convolved cell, additively seam-corrected.

    Runs the standalone correction from
    ``doc/shared_convolved_cross_projection_simple_fix_plan.md`` for every
    cross-projection padding location declared for *skycell* and adds the
    result into the shared canonical convolved image. Cells that need no
    cross-projection padding are returned unchanged. Both
    ``linear_downsample`` and ``field_downsample`` call this so the two
    modes consume an identical corrected array.

    ``combined_recipe`` (the caller's own ``combined_store.combined_recipe``
    dict, e.g. from ``combined_store.production_combined_recipe``) is
    threaded into every own-cell and cross-projection-neighbor
    ``combined_skycell`` lookup so this always resolves the recipe that
    matches the caller's own config, never "whichever fingerprint is
    newest" in the shared, cross-sector store.
    """
    from syndiff_pipeline.template_creation.processing.combined_store import (
        _projection_and_cell,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _try_load_shared_convolved_arrays,
    )

    canonical = _try_load_shared_convolved_arrays(
        data_root, skycell, psf_sigma=psf_sigma, combined_recipe=combined_recipe,
    )
    if canonical is None:
        return None
    image, mask = canonical
    if skycell not in skycell_df.index:
        return image, mask
    spec = cross_projection_padding_spec(skycell_df.loc[skycell])
    if not spec:
        return image, mask

    own_parsed = _projection_and_cell(skycell)
    if own_parsed is None:
        raise PaddingCorrectionError(f"cannot resolve projection/cell identity for {skycell}")
    own_projection, own_cell = own_parsed
    own_combined = _load_combined_image(
        data_root, own_projection, own_cell, combined_recipe=combined_recipe,
    )
    if own_combined is None:
        raise PaddingCorrectionError(
            f"required combined skycell for {skycell} is unavailable; "
            "cannot correct cross-projection seam"
        )
    if own_combined.shape != image.shape:
        raise PaddingCorrectionError(
            f"combined/convolved shape mismatch for {skycell}: "
            f"{own_combined.shape} vs {image.shape}"
        )

    recipient_wcs = _cell_wcs(skycell_df.loc[skycell])
    total_correction = np.zeros(image.shape, dtype=np.float64)
    for location, neighbors in _grouped_padding_spec(spec).items():
        total_correction += _location_correction(
            location=location, neighbors=neighbors, skycell=skycell,
            recipient_wcs=recipient_wcs, cell_shape=image.shape,
            own_combined=own_combined, data_root=data_root, skycell_df=skycell_df,
            psf_sigma=psf_sigma, kernel_radius=kernel_radius,
            combined_recipe=combined_recipe,
        )

    corrected = np.asarray(image, dtype=np.float64).copy()
    finite = np.isfinite(corrected)
    corrected[finite] += total_correction[finite]
    return corrected.astype(image.dtype, copy=False), mask
