"""Cross-projection seam-correction primitives for the shared canonical store.

The shared ``convolved_skycell`` cache (``convolved_store.py``) holds a
canonical cell convolved using only same-projection neighbors (see
``doc/template_bookkeeping_plan.md`` SS13). For skycells whose mapping
requires cross-projection padding, that canonical cell is missing the
neighbor's contribution near the seam -- up to ~50% flux deficit at the
immediate edge (measured in ``tests/test_seam_correction_linearity.py``).

Gaussian convolution is linear, but the correction must preserve the full
external halo through convolution and use the exact producer pre-padding
state.  The public loader therefore fails closed for affected legacy shared
artifacts until a context-backed correction artifact is present.

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
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing.cross_projection_geometry import (
    cell_wcs_from_row,
    cd_matrix_from_row,
    padding_patch_geometry,
    valid_reprojection_footprint,
    projection_identity,
    skycell_projection_identity,
)

log = logging.getLogger(__name__)

PAD_SIZE = 480
EDGE_EXCLUSION = 10

# Bump whenever the correction *algorithm* changes shape/placement math (not
# just spec contents) -- folded into padding_spec_fingerprint so stale cache
# entries computed under an older, buggy version are automatically orphaned
# (never looked up again) rather than silently reused. v2: fixed
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

_SEAM_CORRECTION_DIRNAME = "ps1_seam_correction"
_ARRAYS_FILENAME = "arrays.npz"
_SIDECAR_FILENAME = "_provenance.json"


class PaddingCorrectionError(RuntimeError):
    """The exact cross-projection correction cannot be constructed safely."""


@dataclass(frozen=True)
class ResolvedConvolvedCell:
    """One validated shared/legacy convolved input for either downsample mode."""

    image: np.ndarray
    mask: np.ndarray
    source_mode: str
    canonical_fingerprint: str | None
    correction_required: bool
    correction_complete: bool
    correction_fingerprint: str | None


def _correction_fingerprint(
    *, canonical_fingerprint: str, source_fingerprints: list[str], spec: list[dict[str, str]],
    psf_sigma: float, kernel_radius: int,
) -> str:
    """Identity for one ordered context-backed correction computation."""
    payload = {
        "algorithm": "context_delta_v1",
        "canonical": canonical_fingerprint,
        "sources": source_fingerprints,
        "spec": [(item["neighbor"], item["location"]) for item in spec],
        "pad_size": PAD_SIZE,
        "edge_exclusion": EDGE_EXCLUSION,
        "psf_sigma": float(psf_sigma),
        "kernel_radius": int(kernel_radius),
        "reprojection_order": "bilinear",
        "ownership": "location_then_slash_source_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def _correction_cache_dir(data_root: str | Path, projection: str, cell: str, fingerprint: str) -> Path:
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path

    return ps1_convolved_zarr_path(data_root).parent / _SEAM_CORRECTION_DIRNAME / projection / cell / fingerprint


def _correction_payload_digest(correction: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(correction, dtype=np.float32))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_validated_correction_cache(
    cache_dir: Path,
    *,
    fingerprint: str,
    canonical_fingerprint: str,
    source_fingerprints: list[str],
    spec: list[dict[str, str]],
    shape: tuple[int, int],
    psf_sigma: float,
    kernel_radius: int,
) -> np.ndarray | None:
    arrays_path = cache_dir / _ARRAYS_FILENAME
    sidecar_path = cache_dir / _SIDECAR_FILENAME
    if not arrays_path.is_file() and not sidecar_path.is_file():
        return None
    if not arrays_path.is_file() or not sidecar_path.is_file():
        raise PaddingCorrectionError(f"incomplete correction cache {cache_dir}")
    try:
        sidecar = json.loads(sidecar_path.read_text())
        with np.load(arrays_path, allow_pickle=False) as arrays:
            correction = np.asarray(arrays["correction"], dtype=np.float32)
    except Exception as exc:
        raise PaddingCorrectionError(f"unreadable correction cache {cache_dir}") from exc
    expected_spec = [(item["neighbor"], item["location"]) for item in spec]
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("correction_fingerprint") != fingerprint
        or sidecar.get("canonical_fingerprint") != canonical_fingerprint
        or sidecar.get("source_fingerprints") != source_fingerprints
        or sidecar.get("ordered_spec") != [list(item) for item in expected_spec]
        or tuple(sidecar.get("shape", ())) != tuple(shape)
        or float(sidecar.get("psf_sigma")) != float(psf_sigma)
        or int(sidecar.get("kernel_radius")) != int(kernel_radius)
        or sidecar.get("pad_size") != PAD_SIZE
        or sidecar.get("edge_exclusion") != EDGE_EXCLUSION
        or sidecar.get("ownership") != "location_then_slash_source_v1"
        or sidecar.get("payload_sha256") != _correction_payload_digest(correction)
    ):
        raise PaddingCorrectionError(f"correction cache provenance mismatch: {cache_dir}")
    if correction.shape != tuple(shape):
        raise PaddingCorrectionError(f"correction cache shape mismatch: {cache_dir}")
    return correction


def _publish_correction_cache(
    cache_dir: Path,
    *,
    fingerprint: str,
    correction: np.ndarray,
    canonical_fingerprint: str,
    source_fingerprints: list[str],
    spec: list[dict[str, str]],
    psf_sigma: float,
    kernel_radius: int,
) -> None:
    """Atomically publish an immutable, self-validating correction cache."""
    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if cache_dir.is_dir():
        return
    temporary = parent / f"_tmp_{fingerprint}_{os.getpid()}"
    temporary.mkdir(parents=False, exist_ok=False)
    stored = np.asarray(correction, dtype=np.float32)
    try:
        np.savez_compressed(temporary / _ARRAYS_FILENAME, correction=stored)
        sidecar = {
            "schema_version": 1,
            "algorithm": "context_delta_v1",
            "correction_fingerprint": fingerprint,
            "canonical_fingerprint": canonical_fingerprint,
            "source_fingerprints": source_fingerprints,
            "ordered_spec": [[item["neighbor"], item["location"]] for item in spec],
            "shape": list(stored.shape),
            "psf_sigma": float(psf_sigma),
            "kernel_radius": int(kernel_radius),
            "pad_size": PAD_SIZE,
            "edge_exclusion": EDGE_EXCLUSION,
            "reprojection_order": "bilinear",
            "ownership": "location_then_slash_source_v1",
            "payload_sha256": _correction_payload_digest(stored),
        }
        (temporary / _SIDECAR_FILENAME).write_text(json.dumps(sidecar, sort_keys=True))
        try:
            os.replace(temporary, cache_dir)
        except FileExistsError:
            # Another worker may have published the exact immutable identity.
            if not cache_dir.is_dir():
                raise
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


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


def resolve_shared_convolved_cell(
    data_root: str | Path,
    skycell: str,
    *,
    skycell_df: pd.DataFrame,
    psf_sigma: float,
    kernel_radius: int = 470,
) -> ResolvedConvolvedCell:
    """Resolve one shared canonical cell, correcting it exactly when required.

    This is the only shared-cell science path: it validates current pointers,
    requires a schema-v1 producer context for affected cells, composes all
    cross-projection sources in declared order, convolves the full local
    delta, and restores validity from the completed F input.  Missing or
    ambiguous inputs raise rather than returning a biased compatibility image.
    """
    from reproject import reproject_interp

    from syndiff_pipeline.template_creation.processing.combined_store import (
        _projection_and_cell,
        resolve_current_combined_ref,
        try_load_combined_cell,
    )
    from syndiff_pipeline.template_creation.processing.convolved_store import (
        resolve_current_convolved_ref,
        try_load_convolved_cell,
    )

    parsed = _projection_and_cell(skycell)
    if parsed is None or skycell not in skycell_df.index:
        raise PaddingCorrectionError(f"cannot resolve shared convolved cell {skycell}: missing identity/table row")
    projection, cell = parsed
    canonical_ref = resolve_current_convolved_ref(data_root, projection, cell)
    if canonical_ref is None:
        raise PaddingCorrectionError(f"missing current canonical pointer for required shared cell {skycell}")
    canonical = try_load_convolved_cell(data_root, projection, cell, canonical_ref.fingerprint)
    if canonical is None:
        raise PaddingCorrectionError(f"cannot load selected canonical artifact {canonical_ref.fingerprint} for {skycell}")
    image = np.asarray(canonical["convolved_image"])
    mask = np.asarray(canonical["convolved_mask"])
    spec = cross_projection_padding_spec(skycell_df.loc[skycell])
    if not spec:
        return ResolvedConvolvedCell(
            image=image, mask=mask, source_mode="shared", canonical_fingerprint=canonical_ref.fingerprint,
            correction_required=False, correction_complete=True, correction_fingerprint=None,
        )

    required_context = (
        "pre_cross_context", "pre_cross_context_origin_xy", "unmasked_convolved_image",
        "pre_cross_native_validity",
    )
    if any(key not in canonical for key in required_context):
        raise PaddingCorrectionError(
            f"selected canonical artifact {canonical_ref.fingerprint} for {skycell} lacks required "
            "pre-cross-projection context payload"
        )
    context = np.asarray(canonical["pre_cross_context"], dtype=np.float64)
    baseline = np.asarray(canonical["unmasked_convolved_image"], dtype=np.float64)
    a_valid = np.asarray(canonical["pre_cross_native_validity"], dtype=bool)
    height, width = image.shape
    if context.shape != (height + 2 * PAD_SIZE, width + 2 * PAD_SIZE):
        raise PaddingCorrectionError(
            f"context shape {context.shape} does not match canonical {image.shape} with halo {PAD_SIZE}"
        )
    if baseline.shape != image.shape or a_valid.shape != image.shape:
        raise PaddingCorrectionError(f"invalid baseline/validity shape in canonical artifact for {skycell}")

    recipient_wcs = _cell_wcs(skycell_df.loc[skycell])
    completed = context.copy()
    source_fingerprints: list[str] = []
    source_fingerprint_by_name: dict[str, str] = {}
    source_cache: dict[str, tuple[np.ndarray, WCS]] = {}
    coverage = np.zeros(context.shape, dtype=np.uint16)
    for priority, item in enumerate(spec):
        source_name, location = item["neighbor"], item["location"]
        source_parsed = _projection_and_cell(source_name)
        if source_parsed is None or source_name not in skycell_df.index:
            raise PaddingCorrectionError(f"required source {source_name} for {skycell}/{location} is not resolvable")
        source_projection, source_cell = source_parsed
        if source_name not in source_cache:
            source_ref = resolve_current_combined_ref(data_root, source_projection, source_cell)
            if source_ref is None:
                raise PaddingCorrectionError(f"missing current combined pointer for {source_name}")
            loaded = try_load_combined_cell(data_root, source_projection, source_cell, source_ref.fingerprint)
            if loaded is None:
                raise PaddingCorrectionError(
                    f"cannot load selected combined artifact {source_ref.fingerprint} for {source_name}"
                )
            source_cache[source_name] = (
                _exclude_edge_pixels(np.asarray(loaded["combined_image"], dtype=np.float64)),
                _cell_wcs(skycell_df.loc[source_name]),
            )
            source_fingerprint_by_name[source_name] = source_ref.fingerprint
        source_image, source_wcs = source_cache[source_name]
        source_fingerprints.append(source_fingerprint_by_name[source_name])
        patch_wcs, patch_shape, _center = _standalone_padding_wcs(
            recipient_wcs, width, height, location
        )
        reprojected, footprint = reproject_interp(
            (source_image, source_wcs), patch_wcs, shape_out=patch_shape, order="bilinear"
        )
        geometry = padding_patch_geometry(
            recipient_wcs, recipient_x0=0, recipient_y0=0, cell_width=width,
            cell_height=height, location=location, pad_size=PAD_SIZE,
            edge_exclusion=EDGE_EXCLUSION,
        )
        y0, y1, x0, x1 = geometry.bounds
        # Context is defined in recipient-native coordinates [-P:W+P,
        # -P:H+P), so conversion to array offsets is exact and independent of
        # the row-master origin stored for provenance.
        cy0, cy1 = y0 + PAD_SIZE, y1 + PAD_SIZE
        cx0, cx1 = x0 + PAD_SIZE, x1 + PAD_SIZE
        valid = valid_reprojection_footprint(reprojected, footprint)
        target = completed[cy0:cy1, cx0:cx1]
        target[valid] = reprojected[valid]
        coverage[cy0:cy1, cx0:cx1][valid] = priority + 1

    correction_fp = _correction_fingerprint(
        canonical_fingerprint=canonical_ref.fingerprint, source_fingerprints=source_fingerprints,
        spec=spec, psf_sigma=psf_sigma, kernel_radius=kernel_radius,
    )
    cache_dir = _correction_cache_dir(data_root, projection, cell, correction_fp)
    correction = _load_validated_correction_cache(
        cache_dir,
        fingerprint=correction_fp,
        canonical_fingerprint=canonical_ref.fingerprint,
        source_fingerprints=source_fingerprints,
        spec=spec,
        shape=image.shape,
        psf_sigma=psf_sigma,
        kernel_radius=kernel_radius,
    )
    if correction is None:
        correction = convolve_local_padding_delta(
            context, completed, local_origin_xy=(-PAD_SIZE, -PAD_SIZE), canonical_shape=image.shape,
            psf_sigma=psf_sigma, kernel_radius=kernel_radius,
        )
        _publish_correction_cache(
            cache_dir,
            fingerprint=correction_fp,
            correction=correction,
            canonical_fingerprint=canonical_ref.fingerprint,
            source_fingerprints=source_fingerprints,
            spec=spec,
            psf_sigma=psf_sigma,
            kernel_radius=kernel_radius,
        )
    corrected = baseline + correction
    f_valid = np.isfinite(completed[PAD_SIZE:PAD_SIZE + height, PAD_SIZE:PAD_SIZE + width])
    corrected[~f_valid] = np.nan
    finite_expected = f_valid & np.isfinite(baseline)
    if np.any(~np.isfinite(corrected[finite_expected])):
        raise PaddingCorrectionError(f"non-finite corrected pixels for {skycell}")
    return ResolvedConvolvedCell(
        image=corrected.astype(image.dtype, copy=False), mask=mask, source_mode="shared",
        canonical_fingerprint=canonical_ref.fingerprint, correction_required=True,
        correction_complete=True, correction_fingerprint=correction_fp,
    )


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


def _discover_shared_combined_fp(data_root: str | Path, projection: str, cell: str) -> str | None:
    from syndiff_pipeline.template_creation.processing.combined_store import (
        resolve_current_combined_ref,
    )

    ref = resolve_current_combined_ref(data_root, projection, cell)
    return None if ref is None else ref.fingerprint


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


def _legacy_clipped_seam_correction(
    *,
    data_root: str | Path,
    skycell: str,
    skycell_row: pd.Series,
    spec: list[dict[str, str]],
    canonical_shape: tuple[int, int],
    psf_sigma: float,
    skycell_df: pd.DataFrame,
) -> np.ndarray | None:
    """Retained only as a migration reference; never call for science output.

    It clips projected data to the recipient canvas before convolution and is
    mathematically incapable of carrying exterior flux inward.
    """
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

        valid = valid_reprojection_footprint(reprojected, footprint)
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
    """Load an unaffected canonical cell; fail closed for an affected one.

    The reverted producer-context path is intentionally not used here.  The
    future shared correction must be rebuilt wholly at the shared/downsample
    layer using the approved padded-box design, without changing
    ``ps1_process``.  Until that lands, returning the old clipped correction
    would be scientifically worse than a hard failure.
    """
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _try_load_shared_convolved_arrays,
    )

    canonical = _try_load_shared_convolved_arrays(data_root, skycell)
    if canonical is None:
        return None
    image, mask = canonical
    if skycell not in skycell_df.index or not cross_projection_padding_spec(skycell_df.loc[skycell]):
        return image, mask
    raise PaddingCorrectionError(
        f"shared canonical cell {skycell} requires the not-yet-reimplemented "
        "downsample-side cross-projection correction"
    )
