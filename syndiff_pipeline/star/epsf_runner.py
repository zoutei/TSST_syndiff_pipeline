"""Build gridded ePSF on baseline diff images for star gepsf photometry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.scc_paths import scc_diff_dir, scc_diff_label_dir
from syndiff_pipeline.difference_imaging.orchestration.stage_params import EpsfParams
from syndiff_pipeline.difference_imaging.stages import epsf as epsf_fitting
from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
    GriddedEpsfCatalog,
    catalog_from_workspace,
    workspace_has_gridded_epsf,
)
from syndiff_pipeline.difference_imaging.support.manifest import (
    DEFAULT_MANIFEST_BASENAME,
    ordered_diff_paths_for_scc,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    resolve_pipeline_artifact_path,
)
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.difference_imaging.masking.ffi_mask import load_catalog_for_event
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.site_config import StarEpsfConfig

logger = logging.getLogger(__name__)


def _static_mask_path(lane_root: str) -> str | None:
    return resolve_pipeline_artifact_path(lane_root, SHARED_MASK_FITS_BASENAME)


# Backward-compatible alias
_shared_mask_path = _static_mask_path


def epsf_workspace_dir(ctx: StarEventContext, epsf_label: str) -> str:
    """Directory under the SCC diff lane for one gridded ePSF label."""
    lane = scc_diff_label_dir(
        ctx.data_root,
        ctx.sector,
        ctx.camera,
        ctx.ccd,
        store_name=ctx.output_store_name,
        label=epsf_label,
    )
    return str(lane.resolve())


def epsf_output_dir(ctx: StarEventContext, epsf_cfg: StarEpsfConfig) -> str:
    """Directory for the workspace label declared on an ePSF build block."""
    return epsf_workspace_dir(ctx, epsf_cfg.output)


def _scc_lane_root(ctx: StarEventContext) -> str:
    return str(
        scc_diff_dir(
            ctx.data_root,
            ctx.sector,
            ctx.camera,
            ctx.ccd,
            store_name=ctx.output_store_name,
        ).resolve()
    )


def ensure_star_epsf_catalog(
    ctx: StarEventContext,
    epsf_label: str,
    *,
    build_cfg: StarEpsfConfig | None = None,
    diffs_label: str | None = None,
    overwrite: bool = False,
    max_ffis: int | None = None,
) -> GriddedEpsfCatalog:
    """
    Load or build gridded ePSF for *epsf_label* under the SCC diff lane.

    When *build_cfg* is set and ``build_cfg.output == epsf_label``, fit on
    baseline diffs when the tree is missing or *overwrite* is true.
    """
    out_dir = epsf_workspace_dir(ctx, epsf_label)
    if not overwrite and workspace_has_gridded_epsf(out_dir):
        catalog = catalog_from_workspace(out_dir)
        if catalog is not None:
            logger.info("Reusing existing gridded ePSF at %s", out_dir)
            return catalog

    if build_cfg is None or build_cfg.output != epsf_label:
        raise FileNotFoundError(
            f"gridded ePSF workspace {epsf_label!r} not found under SCC diff lane "
            f"({_scc_lane_root(ctx)}); enable epsf block with matching output "
            "or run a diff-stage ePSF build first"
        )

    resolved_diffs = (
        str(diffs_label or build_cfg.diffs or ctx.baseline_diffs_label).strip()
    )
    manifest_path = os.path.join(ctx.event_dir, DEFAULT_MANIFEST_BASENAME)
    manifest = pd.read_csv(manifest_path)

    diff_paths = ordered_diff_paths_for_scc(
        manifest,
        ctx.data_root,
        ctx.sector,
        ctx.camera,
        ctx.ccd,
        resolved_diffs,
        store_name=ctx.output_store_name,
    )
    if max_ffis is not None and max_ffis > 0:
        diff_paths = [p for p in diff_paths if p][:max_ffis]

    existing = [p for p in diff_paths if p and os.path.isfile(p)]
    if not existing:
        raise FileNotFoundError(
            f"No baseline diff FITS for ePSF under {_scc_lane_root(ctx)}/"
            f"{resolved_diffs}"
        )

    gaia_df = pd.read_csv(ctx.gaia_catalog_path)
    if "ra" not in gaia_df.columns or "dec" not in gaia_df.columns:
        raise ValueError(
            f"Gaia catalog at {ctx.gaia_catalog_path} requires ra/dec columns"
        )

    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
    from syndiff_pipeline.common.wcs_header_cache import load_ffi_list
    from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
        ffi_path_by_stem_from_wcs_table,
    )

    ffi_list_path = scc_ffi_list_parquet(
        ctx.data_root, int(ctx.sector), int(ctx.camera), int(ctx.ccd)
    )
    if not os.path.isfile(ffi_list_path):
        raise FileNotFoundError(
            f"Star ePSF requires ffi_list.parquet at {ffi_list_path}"
        )
    ffi_list_df = load_ffi_list(ffi_list_path)
    ffi_path_by_stem = ffi_path_by_stem_from_wcs_table(manifest)

    epsf_params = EpsfParams(
        tile_nx=build_cfg.tile_nx,
        tile_ny=build_cfg.tile_ny,
        epsf_oversample=build_cfg.epsf_oversample,
        psf_size=build_cfg.psf_size,
        extract_size=build_cfg.extract_size,
        min_stars_per_tile=build_cfg.min_stars_per_tile,
        tess_mag_max=build_cfg.tess_mag_max,
        epsf_maxiters=build_cfg.epsf_maxiters,
        epsf_recentering_maxiters=build_cfg.epsf_recentering_maxiters,
        epsf_n_jobs=build_cfg.epsf_n_jobs,
        epsf_mode=build_cfg.epsf_mode,
        epsf_per_orbit=build_cfg.epsf_per_orbit,
        epsf_frames_per_anchor=build_cfg.epsf_frames_per_anchor,
        epsf_stack_before_fit=build_cfg.epsf_stack_before_fit,
        epsf_anchor_edge_fraction=build_cfg.epsf_anchor_edge_fraction,
        epsf_anchor_edge_boost=build_cfg.epsf_anchor_edge_boost,
        epsf_anchor_window_max_expand=build_cfg.epsf_anchor_window_max_expand,
        epsf_quality_bitmask=build_cfg.epsf_quality_bitmask,
        epsf_debug_plots=build_cfg.epsf_debug_plots,
    )
    cfg = SimpleNamespace(n_jobs=build_cfg.epsf_n_jobs or 8)
    lane_root = _scc_lane_root(ctx)
    mask_catalog = load_catalog_for_event(
        lane_root,
        crop_bounds=ctx.crop_bounds,
        data_root=ctx.data_root,
        sector=int(ctx.sector),
        camera=int(ctx.camera),
        ccd=int(ctx.ccd),
    )
    static_mask_path = None if mask_catalog is not None else _static_mask_path(lane_root)

    os.makedirs(out_dir, exist_ok=True)
    logger.info(
        "Fitting gridded ePSF (%dx%d) on %d baseline %s frames → %s",
        build_cfg.tile_nx,
        build_cfg.tile_ny,
        len(existing),
        resolved_diffs,
        out_dir,
    )
    epsf_stack, _tile_centers, ffi_stems, epsf_ok = epsf_fitting.fit_epsf_all_frames(
        diff_paths,
        gaia_df,
        cfg,
        epsf_params,
        out_dir,
        round_id=1,
        mask_catalog=mask_catalog,
        static_mask_path=static_mask_path,
        wcs_table=manifest,
        epsf_label=build_cfg.output,
        diffs_input=resolved_diffs,
        ffi_list_df=ffi_list_df,
        science_bounds=ctx.crop_bounds,
        ffi_path_by_stem=ffi_path_by_stem,
    )
    if epsf_stack is None or not any(epsf_ok):
        raise RuntimeError(f"ePSF fitting produced no usable frames under {out_dir}")

    catalog = catalog_from_workspace(out_dir)
    if catalog is None:
        raise RuntimeError(f"ePSF index missing after fit under {out_dir}")
    logger.info(
        "Gridded ePSF ready: %d indexed frames (%d ok)",
        len(catalog.index),
        sum(bool(x) for x in epsf_ok),
    )
    return catalog
