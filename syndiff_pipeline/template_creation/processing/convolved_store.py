"""Shared, sky-addressed convolved-skycell store (Phase 2, PR5 — data layer only).

Persists the *canonical* convolved skycell: the combined (band-combined +
star-removed) cell, convolved with the TESS PSF using only same-projection
padding (``apply_cross_row_padding`` — always available, sector-independent).
Cross-projection padding (different PS1 projections meeting inside one TESS
SCC's footprint) is deliberately excluded from this canonical product, because
that padding is the one genuinely SCC-specific input to ``ps1_process`` (see
``doc/template_bookkeeping_plan.md`` §13 and
``doc/bookkeeping_pr245_dataflow_map.md``).

**Status: data layer only, NOT wired into the live pipeline.** This module
provides the fingerprint/publish/load machinery for the canonical convolved
cell, mirroring ``combined_store.py``'s proven pattern exactly. It does
*not* attempt the cross-projection seam reconstruction, and
``ps1_process.py`` does not call it yet. That is a deliberate, documented
stopping point, not an oversight:

Gaussian convolution is linear, so the exact seam correction is
``convolve(canonical_gap_zero) + convolve(reprojected_patch_alone)`` — proven
to floating-point precision against the real production convolution function
in ``tests/test_seam_correction_linearity.py``. But naively leaving the
cross-projection gap zero-filled (no correction at all) produces a real,
material flux deficit near the seam (tens of percent within one truncation
radius) — validated in that same test. Wiring the *correction* into the live
per-row processing loop is real new production code touching
``cross_projection_padding.py``/``ps1_process.py``'s hot path, and per the
plan it is additionally gated on a real-SCC numeric-equivalence comparison
against today's baked-in ``convolved.zarr`` before it may ship. Both of those
are appropriately a separate, reviewed step — not bundled into this data-layer
commit.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.fingerprint import code_version, git_sha
from syndiff_pipeline.common.provenance.publish import publish_record
from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path

logger = logging.getLogger(__name__)

_ARRAYS_BASENAME = "arrays.npz"
_HEADERS_BASENAME = "headers.json"


def convolved_recipe(
    *, psf_sigma: float, radius: int = 470, mode: str = "constant"
) -> model.Recipe:
    """Build the ``convolved_skycell`` recipe (§6): psf_sigma, radius, mode.

    ``radius``/``mode`` mirror ``convolution_utils.apply_gaussian_convolution``'s
    own defaults, exposed here so a future change to either is reflected in the
    recipe identity, not silently baked into unversioned code.
    """
    params: Dict[str, Any] = {
        "psf_sigma": float(psf_sigma),
        "radius": int(radius),
        "mode": mode,
        "padding": "same_projection_only",  # documents the canonical scope (§13)
    }
    return model.Recipe(
        kind=model.CONVOLVED_SKYCELL,
        params=params,
        code_version=code_version(),
        git_sha=git_sha(),
    )


def _convolved_artifact(
    projection: str, skycell: str, recipe: model.Recipe, *, combined_fp: str
) -> model.Artifact:
    return model.Artifact(
        kind=model.CONVOLVED_SKYCELL,
        spatial_key=model.skycell_spatial_key(projection, skycell),
        recipe=recipe,
        inputs=[combined_fp],
        state="complete",
    )


def convolved_cell_dir(
    data_root: str,
    projection: str,
    skycell: str,
    recipe: model.Recipe,
    *,
    combined_fp: str,
) -> Path:
    """The fingerprinted directory for one canonical convolved cell.

    ``combined_fp`` (the input combined-skycell's fingerprint, e.g. from
    ``combined_store.combined_cell_dir(...).name``) is threaded in explicitly
    as a Merkle input: a recompute of the upstream combined cell (a config
    change, a Gaia catalog refresh) invalidates every convolved cell built
    from it, automatically.
    """
    fp = _convolved_artifact(projection, skycell, recipe, combined_fp=combined_fp).fingerprint()
    return ps1_convolved_zarr_path(data_root) / str(projection) / str(skycell) / fp


def try_load_convolved_cell(
    data_root: str,
    projection: str,
    skycell: str,
    recipe: model.Recipe,
    *,
    combined_fp: str,
) -> Optional[Dict[str, Any]]:
    """Return the cached canonical convolved cell, or ``None`` on any miss/error.

    Shape: ``{"convolved_image", "headers_data"}``. Never raises.
    """
    cell_dir = convolved_cell_dir(data_root, projection, skycell, recipe, combined_fp=combined_fp)
    arrays_path = cell_dir / _ARRAYS_BASENAME
    if not arrays_path.is_file():
        return None
    try:
        with np.load(arrays_path) as npz:
            convolved_image = npz["image"]
        headers_path = cell_dir / _HEADERS_BASENAME
        headers_data = (
            json.loads(headers_path.read_text(encoding="utf-8"))
            if headers_path.is_file()
            else {}
        )
    except Exception:
        logger.warning(
            "Convolved-store read failed for %s/%s at %s (treating as miss)",
            projection,
            skycell,
            cell_dir,
            exc_info=True,
        )
        return None
    return {"convolved_image": convolved_image, "headers_data": headers_data}


def publish_convolved_cell(
    data_root: str,
    projection: str,
    skycell: str,
    recipe: model.Recipe,
    *,
    combined_fp: str,
    convolved_image: np.ndarray,
    headers_data: Optional[dict] = None,
    produced_by: Optional[str] = None,
) -> bool:
    """Best-effort publish of a freshly computed canonical convolved cell.

    Mirrors ``combined_store.publish_combined_cell`` exactly (atomic tmp-dir
    build + ``publish_record``'s rename). Never raises.
    """
    try:
        final_dir = convolved_cell_dir(
            data_root, projection, skycell, recipe, combined_fp=combined_fp
        )
        if final_dir.exists():
            return True  # already published (identical bytes by fingerprint)
        artifact = _convolved_artifact(projection, skycell, recipe, combined_fp=combined_fp)
        artifact.location = str(final_dir)
        artifact.produced_by = produced_by
        record = artifact.to_record()

        tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(tmp_dir / _ARRAYS_BASENAME, image=np.asarray(convolved_image))
            (tmp_dir / _HEADERS_BASENAME).write_text(
                json.dumps(headers_data or {}), encoding="utf-8"
            )
            publish_record(record, tmp_dir=tmp_dir, final_dir=final_dir, data_root=data_root)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return True
    except Exception:
        logger.warning(
            "Convolved-store publish failed for %s/%s (non-fatal, will recompute next time)",
            projection,
            skycell,
            exc_info=True,
        )
        return False


__all__ = [
    "convolved_cell_dir",
    "convolved_recipe",
    "publish_convolved_cell",
    "try_load_convolved_cell",
]
