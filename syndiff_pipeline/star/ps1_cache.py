"""PS1 skycell Zarr cache for the star-host workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import zarr

from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.site_config import normalize_ps1_source
from syndiff_pipeline.template_creation.processing.ps1_download import (
    ZarrWriter,
    _ARRAYS_PER_SKYCELL,
    count_complete_arrays,
    download_and_store_skycell,
    fetch_skycell_bands_masks_and_headers,
    get_projection_from_name,
    initialize_zarr_store,
)
from syndiff_pipeline.template_creation.processing.zarr_utils import (
    load_skycell_bands_masks_and_headers,
)

logger = logging.getLogger(__name__)

PS1_SKYCELLS_ZARR_BASENAME = "ps1_skycells.zarr"


def ps1_skycells_zarr_paths(
    ctx: StarEventContext,
    *,
    zarr_path_override: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Shared PS1 raw-band Zarr store (same layout as ``ps1_download``).

    Hierarchy: ``{projection_id}/{skycell_name}/{band, band_mask, band_wt}``
    at ``{data_root}/ps1_skycells.zarr`` with lock file alongside.
    """
    if zarr_path_override:
        zarr_path = Path(zarr_path_override).expanduser().resolve()
    else:
        zarr_path = Path(ctx.data_root) / PS1_SKYCELLS_ZARR_BASENAME
    lock_file = zarr_path.parent / f"{zarr_path.name}.lock"
    return zarr_path, lock_file


def _load_skycell_from_store(
    zarr_store: zarr.Group,
    projection: str,
    skycell_name: str,
) -> tuple[dict, dict, dict, dict, dict]:
    """Read one skycell using the same loader as ``ps1_process``."""
    bands, masks, weights, headers, headers_weight = load_skycell_bands_masks_and_headers(
        zarr_store,
        projection,
        skycell_name,
    )
    if bands:
        return bands, masks, weights, headers, headers_weight

    short_id = skycell_name.split(".")[-1]
    return load_skycell_bands_masks_and_headers(
        zarr_store,
        projection,
        short_id,
    )


def ensure_skycell_cached(
    skycell_name: str,
    ctx: StarEventContext,
    *,
    overwrite: bool = False,
    zarr_path_override: str | Path | None = None,
) -> Path:
    """
    Ensure *skycell_name* is present in the shared PS1 Zarr store.

    Downloads from MAST on first access and persists for later runs.
    Returns the Zarr store path.
    """
    zarr_path, lock_file = ps1_skycells_zarr_paths(ctx, zarr_path_override=zarr_path_override)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    projection = get_projection_from_name(skycell_name)
    if not projection:
        raise ValueError(f"Could not parse projection from skycell name {skycell_name!r}")

    root = initialize_zarr_store(zarr_path)
    complete = count_complete_arrays(
        root,
        projection,
        skycell_name,
        lock_file,
        overwrite=overwrite,
    )
    if complete < _ARRAYS_PER_SKYCELL or overwrite:
        logger.info(
            "Caching PS1 skycell %s to %s (%d/%d arrays present)",
            skycell_name,
            zarr_path,
            complete,
            _ARRAYS_PER_SKYCELL,
        )
        writer = ZarrWriter(root, lock_file)
        try:
            download_and_store_skycell(
                root,
                skycell_name,
                lock_file,
                writer,
                overwrite=overwrite,
            )
        finally:
            writer.close()
    else:
        logger.info("Using cached PS1 skycell %s from %s", skycell_name, zarr_path)

    return zarr_path


def load_skycell_bands_from_zarr_only(
    skycell_name: str,
    ctx: StarEventContext,
    *,
    zarr_path_override: str | Path | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """Read PS1 bands from zarr only; fail if skycell is absent or incomplete."""
    zarr_path, _lock_file = ps1_skycells_zarr_paths(ctx, zarr_path_override=zarr_path_override)
    if not zarr_path.is_dir():
        raise FileNotFoundError(
            f"PS1 zarr store not found at {zarr_path} (ps1_source=zarr_local_only)"
        )
    projection = get_projection_from_name(skycell_name)
    if not projection:
        raise ValueError(f"Could not parse projection from skycell name {skycell_name!r}")

    zarr_store = zarr.open(str(zarr_path), mode="r")
    bands, masks, weights, headers, headers_weight = _load_skycell_from_store(
        zarr_store,
        projection,
        skycell_name,
    )
    if not bands:
        raise FileNotFoundError(
            f"No PS1 band data in zarr cache for {skycell_name} at {zarr_path} "
            "(ps1_source=zarr_local_only)"
        )
    return bands, masks, weights, headers, headers_weight


def load_skycell_bands_from_cache(
    skycell_name: str,
    ctx: StarEventContext,
    *,
    overwrite: bool = False,
    zarr_path_override: str | Path | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """Load (or download-and-cache) raw PS1 bands for one skycell."""
    zarr_path = ensure_skycell_cached(
        skycell_name,
        ctx,
        overwrite=overwrite,
        zarr_path_override=zarr_path_override,
    )
    projection = get_projection_from_name(skycell_name)
    if not projection:
        raise ValueError(f"Could not parse projection from skycell name {skycell_name!r}")

    zarr_store = zarr.open(str(zarr_path), mode="r")
    bands, masks, weights, headers, headers_weight = _load_skycell_from_store(
        zarr_store,
        projection,
        skycell_name,
    )
    if not bands:
        raise FileNotFoundError(
            f"No PS1 band data in zarr cache for {skycell_name} at {zarr_path}"
        )
    return bands, masks, weights, headers, headers_weight


def load_skycell_bands_from_stream(
    skycell_name: str,
) -> tuple[dict, dict, dict, dict, dict]:
    """Fetch PS1 bands from MAST without writing to zarr."""
    bands, masks, weights, headers, headers_weight = fetch_skycell_bands_masks_and_headers(
        skycell_name
    )
    if not bands:
        raise FileNotFoundError(
            f"No PS1 band data found for {skycell_name} (ps1_source=stream)"
        )
    return bands, masks, weights, headers, headers_weight


def load_skycell_bands_for_source(
    skycell_name: str,
    ctx: StarEventContext,
    *,
    ps1_source: str,
    overwrite: bool = False,
    zarr_path_override: str | Path | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """Dispatch PS1 load by ``ps1_source`` mode."""
    mode = normalize_ps1_source(ps1_source)
    if mode == "zarr_local_only":
        return load_skycell_bands_from_zarr_only(
            skycell_name,
            ctx,
            zarr_path_override=zarr_path_override,
        )
    if mode == "zarr_download":
        return load_skycell_bands_from_cache(
            skycell_name,
            ctx,
            overwrite=overwrite,
            zarr_path_override=zarr_path_override,
        )
    if mode == "stream":
        return load_skycell_bands_from_stream(skycell_name)
    raise ValueError(f"Unsupported ps1_source: {mode!r}")
