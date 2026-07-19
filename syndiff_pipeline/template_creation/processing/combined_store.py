"""Shared, sky-addressed pre-convolution "combined" skycell store (Phase 1, PR4).

Persists and reuses the band-combined + star-removed skycell image that
``ps1_process`` produces before cross-projection padding and convolution run.
Because PS1 skycells sit on a fixed sky tessellation, this product is a pure
function of the raw skycell plus a handful of processing parameters —
independent of which TESS sector/camera/ccd is being built — so a cell built
once is reused by every later overlapping sector (see
``doc/template_bookkeeping_plan.md`` §12).

**Correction vs. the plan's abstract "data/mask/uncert" triple** (found during
PR0 investigation, see ``doc/bookkeeping_pr245_dataflow_map.md``): in this
codebase, star removal runs *after* band combination, inside the subprocess
step (``process_single_cell`` / ``remove_background``), and the uncertainty
array is consumed there and does not survive downstream. The real reusable
artifact is therefore ``{combined_image (star-removed), combined_mask,
headers_data, removed_stars}`` — exactly the shape already cached in-run by
``ps1_process.band_cache`` for padding-source cells. This module persists that
same shape cross-run instead of only within one run.

**Scope (Phase 1, deliberately conservative):** only *regular* (non
padding-role) skycells are seeded from / published to this store. A skycell
that also plays a cross-projection padding-source role keeps using the
existing in-run ``band_cache`` path unchanged — sharing that role too is a
sound follow-up (it only needs a load check ahead of the JIT padding-source
dispatch) but is out of scope here to keep this change small and reviewable.

Every function here is best-effort and **must never raise** into the caller:
this is a pure optimization layer over the existing (slower) recompute path,
never a correctness dependency.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.fingerprint import code_version, git_sha
from syndiff_pipeline.common.provenance.publish import publish_record
from syndiff_pipeline.common.scc_paths import ps1_combined_zarr_path

logger = logging.getLogger(__name__)

_ARRAYS_BASENAME = "arrays.npz"
_HEADERS_BASENAME = "headers.json"
_REMOVED_STARS_BASENAME = "removed_stars.json"

# Cache the (stable, argument-free) raw-skycell recipe id: identical for every
# skycell, so there is no need to recompute the sha256 per call.
_RAW_SKYCELL_RECIPE = model.Recipe(
    kind=model.RAW_SKYCELL, params={"version_token": None}, code_version=code_version()
)


def projection_from_skycell_name(skycell_name: str) -> Optional[str]:
    """Extract the PS1 projection id from a ``skycell.PROJ.CELL``-style name."""
    try:
        return skycell_name.split(".")[1]
    except (IndexError, AttributeError):
        return None


def gaia_version_stamp(catalog_path: Optional[str]) -> str:
    """Cheap content-identity stamp for the Gaia catalog file, or ``"none"``.

    TODO: replace with a formal catalog release id once one exists.
    ``(path, size, mtime_ns)`` is the documented placeholder (mirrors decision
    #5's raw-skycell version-token philosophy): a catalog refresh at the same
    path changes ``mtime``/``size`` and therefore invalidates every downstream
    combined cell that depended on it. Never raises.
    """
    if not catalog_path:
        return "none"
    try:
        st = os.stat(catalog_path)
        return f"{catalog_path}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{catalog_path}:unknown"


def raw_skycell_input_fingerprint(projection: str, skycell: str) -> str:
    """Placeholder raw-skycell identity (plan decision #5).

    Assumes raw bytes are stable once downloaded — there is no re-download /
    batch-id tracking yet in ``ps1_download``. Upgrade this once one exists;
    until then every raw skycell at a given (projection, skycell) key
    fingerprints identically, which is safe as long as raw cells are never
    silently mutated in place (they are not, in the current pipeline).
    """
    from syndiff_pipeline.common.provenance.fingerprint import fingerprint as merkle_fingerprint

    return merkle_fingerprint(
        model.RAW_SKYCELL,
        model.skycell_spatial_key(projection, skycell),
        _RAW_SKYCELL_RECIPE.recipe_id(),
        [],
    )


def combined_recipe(
    *,
    enable_saturation_correction: bool,
    remove_saturated_stars: bool,
    bright_star_mag_threshold: float,
    gaia_version: str,
) -> model.Recipe:
    """Build the ``combined_skycell`` recipe from the current ps1_process config.

    Mirrors ``model.combined_skycell_params`` (params that matter per §6),
    computed here from plain scalars rather than a ``ResolvedTargetConfig`` so
    this processing-layer module never imports the orchestration layer.
    """
    params: Dict[str, Any] = {
        "enable_saturation_correction": bool(enable_saturation_correction),
        "remove_saturated_stars": bool(remove_saturated_stars),
        "bright_star_mag_threshold": float(bright_star_mag_threshold),
        "gaia_version": gaia_version,
    }
    return model.Recipe(
        kind=model.COMBINED_SKYCELL,
        params=params,
        code_version=code_version(),
        git_sha=git_sha(),
    )


def _combined_artifact(
    projection: str, skycell: str, recipe: model.Recipe
) -> model.Artifact:
    inputs = [raw_skycell_input_fingerprint(projection, skycell)]
    return model.Artifact(
        kind=model.COMBINED_SKYCELL,
        spatial_key=model.skycell_spatial_key(projection, skycell),
        recipe=recipe,
        inputs=inputs,
        state="complete",
    )


def combined_cell_dir(
    data_root: str, projection: str, skycell: str, recipe: model.Recipe
) -> Path:
    """The fingerprinted directory for one (projection, skycell, recipe) triple.

    The fingerprint is always derived here, from the same
    :class:`model.Artifact` construction used by both the load and publish
    paths below, so a lookup and a publish can never disagree about identity.
    """
    fp = _combined_artifact(projection, skycell, recipe).fingerprint()
    return ps1_combined_zarr_path(data_root) / str(projection) / str(skycell) / fp


def try_load_combined_cell(
    data_root: str, projection: str, skycell: str, recipe: model.Recipe
) -> Optional[Dict[str, Any]]:
    """Return the cached combined cell for this recipe, or ``None`` on any miss/error.

    Shape matches ``ps1_process.band_cache`` entries:
    ``{"combined_image", "combined_mask", "headers_data", "removed_stars"}``.
    Never raises — any read failure (partial write, corrupt file, permission
    error) is treated as a cache miss so the caller falls back to recomputing.
    """
    cell_dir = combined_cell_dir(data_root, projection, skycell, recipe)
    arrays_path = cell_dir / _ARRAYS_BASENAME
    if not arrays_path.is_file():
        return None
    try:
        with np.load(arrays_path) as npz:
            combined_image = npz["image"]
            combined_mask = npz["mask"]
        headers_path = cell_dir / _HEADERS_BASENAME
        headers_data = (
            json.loads(headers_path.read_text(encoding="utf-8"))
            if headers_path.is_file()
            else {}
        )
        stars_path = cell_dir / _REMOVED_STARS_BASENAME
        removed_stars = (
            json.loads(stars_path.read_text(encoding="utf-8"))
            if stars_path.is_file()
            else []
        )
    except Exception:
        logger.warning(
            "Combined-store read failed for %s/%s at %s (treating as miss)",
            projection,
            skycell,
            cell_dir,
            exc_info=True,
        )
        return None
    return {
        "combined_image": combined_image,
        "combined_mask": combined_mask,
        "headers_data": headers_data,
        "removed_stars": removed_stars,
    }


def publish_combined_cell(
    data_root: str,
    projection: str,
    skycell: str,
    recipe: model.Recipe,
    *,
    combined_image: np.ndarray,
    combined_mask: np.ndarray,
    headers_data: Optional[dict] = None,
    removed_stars: Optional[list] = None,
    produced_by: Optional[str] = None,
) -> bool:
    """Best-effort publish of a freshly computed combined cell. Never raises.

    Returns ``True`` if the cell is present at the fingerprinted location
    after this call (whether just published by us or already published by a
    concurrent worker/run — identical bytes by construction), ``False`` if
    publishing failed (the caller's pipeline is unaffected either way; this is
    a pure cache-population side effect).
    """
    try:
        final_dir = combined_cell_dir(data_root, projection, skycell, recipe)
        if final_dir.exists():
            return True  # another worker already published this exact fingerprint
        artifact = _combined_artifact(projection, skycell, recipe)
        artifact.location = str(final_dir)
        artifact.produced_by = produced_by
        record = artifact.to_record()

        tmp_dir = final_dir.parent / f"_tmp_{final_dir.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                tmp_dir / _ARRAYS_BASENAME,
                image=np.asarray(combined_image),
                mask=np.asarray(combined_mask),
            )
            (tmp_dir / _HEADERS_BASENAME).write_text(
                json.dumps(headers_data or {}), encoding="utf-8"
            )
            (tmp_dir / _REMOVED_STARS_BASENAME).write_text(
                json.dumps(removed_stars or [], default=str), encoding="utf-8"
            )
            publish_record(record, tmp_dir=tmp_dir, final_dir=final_dir, data_root=data_root)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return True
    except Exception:
        logger.warning(
            "Combined-store publish failed for %s/%s (non-fatal, will recompute next time)",
            projection,
            skycell,
            exc_info=True,
        )
        return False


def seed_band_cache_from_combined_store(
    data_root: str, skycell_names: Iterable[str], recipe: model.Recipe
) -> Dict[str, Dict[str, Any]]:
    """Load every available combined-store hit for *skycell_names* under *recipe*.

    Returns a dict shaped like ``ps1_process.band_cache`` (mapping
    ``skycell_name -> {"combined_image","combined_mask","headers_data",
    "removed_stars"}``) containing only the names that actually hit; the
    caller merges this into its own ``band_cache`` before dispatch. Never
    raises: a lookup failure for one name is logged and skipped, not fatal.
    """
    hits: Dict[str, Dict[str, Any]] = {}
    for name in skycell_names:
        projection = projection_from_skycell_name(name)
        if projection is None:
            continue
        try:
            loaded = try_load_combined_cell(data_root, projection, name, recipe)
        except Exception:
            logger.warning(
                "Combined-store seed lookup failed for %s (skipping)", name, exc_info=True
            )
            continue
        if loaded is not None:
            hits[name] = loaded
    return hits


__all__ = [
    "combined_cell_dir",
    "combined_recipe",
    "gaia_version_stamp",
    "projection_from_skycell_name",
    "publish_combined_cell",
    "raw_skycell_input_fingerprint",
    "seed_band_cache_from_combined_store",
    "try_load_combined_cell",
]
