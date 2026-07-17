"""Build gridded ePSF on baseline diff images for star gepsf photometry."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration.stage_params import EpsfParams
from syndiff_pipeline.difference_imaging.stages import epsf as epsf_fitting
from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
    GriddedEpsfCatalog,
    catalog_from_workspace,
    workspace_has_gridded_epsf,
)
from syndiff_pipeline.difference_imaging.support.manifest import (
    DEFAULT_MANIFEST_BASENAME,
    ordered_diff_paths_for_workspace,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    DEFAULT_MANIFEST_BASENAME,
    STATIC_MASK_FITS_BASENAME,
)
from syndiff_pipeline.difference_imaging.masking.ffi_mask import load_catalog_for_event
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.site_config import StarEpsfConfig

logger = logging.getLogger(__name__)


def _static_mask_path(baseline_workspace_dir: str) -> str | None:
    for name in (STATIC_MASK_FITS_BASENAME, "shared_mask.fits", "static_mask.fits.gz"):
        path = os.path.join(baseline_workspace_dir, name)
        if os.path.isfile(path):
            return path
    return None


# Backward-compatible alias
_shared_mask_path = _static_mask_path


def epsf_workspace_dir(ctx: StarEventContext, epsf_label: str) -> str:
    """Directory under the baseline workspace for one gridded ePSF label."""
    return os.path.join(ctx.baseline_workspace_dir, epsf_label)


def epsf_output_dir(ctx: StarEventContext, epsf_cfg: StarEpsfConfig) -> str:
    """Directory for the workspace label declared on an ePSF build block."""
    return epsf_workspace_dir(ctx, epsf_cfg.output)


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
    Load or build gridded ePSF for *epsf_label* under the baseline workspace.

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
            f"gridded ePSF workspace {epsf_label!r} not found under "
            f"{ctx.baseline_workspace_dir}; enable epsf block with matching output "
            "or run a diff-stage ePSF build first"
        )

    resolved_diffs = (
        str(diffs_label or build_cfg.diffs or ctx.baseline_diffs_label).strip()
    )
    manifest_path = os.path.join(ctx.event_dir, DEFAULT_MANIFEST_BASENAME)
    manifest = pd.read_csv(manifest_path)
    baseline_run_id = None
    ws_name = Path(ctx.baseline_workspace_dir).name
    if ws_name.startswith("ws_"):
        baseline_run_id = ws_name[len("ws_") :]

    diff_paths = ordered_diff_paths_for_workspace(
        manifest,
        ctx.event_dir,
        resolved_diffs,
        manifest_path,
        run_id=baseline_run_id,
    )
    if max_ffis is not None and max_ffis > 0:
        diff_paths = [p for p in diff_paths if p][:max_ffis]

    existing = [p for p in diff_paths if p and os.path.isfile(p)]
    if not existing:
        raise FileNotFoundError(
            f"No baseline diff FITS for ePSF under {ctx.baseline_workspace_dir}/"
            f"{resolved_diffs}"
        )

    gaia_df = pd.read_csv(ctx.gaia_catalog_path)
    gaia_df = wcs_grouping.ensure_gaia_crop_xy(
        gaia_df,
        ctx.reference_ffi_path,
        ctx.crop_bounds,
        force_reproject=False,
    )

    epsf_params = EpsfParams(
        tile_nx=build_cfg.tile_nx,
        tile_ny=build_cfg.tile_ny,
        epsf_oversample=build_cfg.epsf_oversample,
        psf_size=build_cfg.psf_size,
        extract_size=build_cfg.extract_size,
        min_stars_per_tile=build_cfg.min_stars_per_tile,
        mag_max_rp=build_cfg.mag_max_rp,
        epsf_maxiters=build_cfg.epsf_maxiters,
        epsf_recentering_maxiters=build_cfg.epsf_recentering_maxiters,
        epsf_n_jobs=build_cfg.epsf_n_jobs,
    )
    cfg = SimpleNamespace(n_jobs=build_cfg.epsf_n_jobs or 8)
    mask_catalog = load_catalog_for_event(
        ctx.baseline_workspace_dir,
        crop_bounds=ctx.crop_bounds,
        data_root=ctx.data_root,
        sector=int(ctx.sector),
        camera=int(ctx.camera),
        ccd=int(ctx.ccd),
    )
    static_mask_path = None if mask_catalog is not None else _static_mask_path(
        ctx.baseline_workspace_dir
    )

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
