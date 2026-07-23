"""
Config-driven pipeline execution (YAML ``pipeline`` list).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.fits_io import write_hdul_fits, write_image_fits
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.common.download import list_local_ffis, _ffi_filename_pattern
from syndiff_pipeline.difference_imaging.stages import (
    background,
    convolved_templates as convolved_templates_runner,
    epsf as epsf_fitting,
    hotpants as hotpants_runner,
    kernel_fit as kernel_fit_runner,
    kernel_subtract as kernel_subtract_runner,
    masking,
    photometry,
    sat_template,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    is_pipeline_fits_filename,
    resolve_pipeline_artifact_path,
    resolve_pipeline_fits_path,
    tess_product_id_from_ffi_path,
    workspace_frame_fits_basename,
    workspace_frame_fits_path,
    workspace_frame_stem,
    workspace_label_from_dir,
)
from syndiff_pipeline.difference_imaging.support.manifest import (
    apply_epsf_status,
    apply_hotpants_workspace_results,
    group_ids_from_ffi_stems,
    ordered_diff_paths_for_scc,
    limit_diff_paths,
    save_frame_manifest,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import HotpantsWorkspaceDirs
from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_ffis_symlink,
    ensure_event_templates_symlink,
    event_ffis_symlink_path,
    event_templates_symlink_path,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    GAIA_CATALOG_PIPELINE_BASENAME,
    HOTPANTS_SUBSTAMP_STARS_BASENAME,
    SHARED_MASK_FITS_BASENAME,
)
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.context import PipelineInvocationContext
from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import (
    is_external_workspaces_entry,
    is_workspace_inherit_entry,
    split_pipeline,
)
from syndiff_pipeline.difference_imaging.orchestration.workspace_lock import (
    assert_workspace_config_lock,
    write_immutable_workspace_config_snapshot,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    parse_background,
    parse_centroids,
    parse_epsf,
    parse_hotpants,
    parse_kernel_fit,
    parse_kernel_subtract,
    parse_convolved_templates,
    kernel_fit_params_to_hotpants,
    HotpantsParams,
    parse_sat_template,
    parse_shared_mask,
    parse_subtract,
)
from syndiff_pipeline.difference_imaging.support.subtract import parse_subtract_expression

log = logging.getLogger(__name__)


def _subtract_load_plane(
    ws_root: str,
    product_id: str,
    frame_index: int,
    npy_stack_by_ws: dict[str, np.ndarray | None],
) -> Optional[np.ndarray]:
    """Load one per-frame 2D plane from a background workspace stack or FITS."""
    if ws_root not in npy_stack_by_ws:
        stack = None
        try:
            stack = background.load_stack(ws_root)
        except FileNotFoundError:
            stack = None
        npy_stack_by_ws[ws_root] = stack
    stack = npy_stack_by_ws[ws_root]
    if stack is not None and frame_index < len(stack):
        return stack[frame_index].astype(np.float64)
    if not product_id:
        return None
    ws_label = workspace_label_from_dir(ws_root)
    stem = workspace_frame_stem(product_id, ws_label)
    fp = resolve_pipeline_fits_path(ws_root, stem)
    if fp is not None:
        return fits.getdata(fp).astype(np.float64)
    return None


def _records_to_stem_rows(records: list) -> list:
    """Records to stem rows.
    
    Parameters
    ----------
    records : list
    
    Returns
    -------
    list"""
    return [
        {
            "ffi_product_id": r.product_id,
            "stem": r.stem,
            "success": r.success,
        }
        for r in records
    ]


def _run_background_stage(
    *,
    stage: dict,
    idx: int,
    cfg: SynDiffConfig,
    ctx: PipelineInvocationContext,
    ws_root: str,
    shared_mask: Optional[np.ndarray],
    mask_catalog,
    wcs_table: pd.DataFrame,
    processing_ffi_paths: list,
    out: str,
    crop_bounds: Optional[dict] = None,
) -> tuple:
    """Execute unified ``background`` stage; returns (shared_mask, mask_catalog)."""
    params = parse_background(stage, idx)
    inp = stage.get("inputs") or {}
    label_out = str(stage["output"]).strip()
    out_ws = _diff_stage_dir(cfg, ctx, label_out)
    os.makedirs(out_ws, exist_ok=True)
    shared_mask = _ensure_shared_mask_loaded(shared_mask, cfg=cfg)
    mask_catalog = _ensure_mask_catalog_loaded(
        ws_root,
        mask_catalog,
        shared_mask,
        crop_bounds=crop_bounds,
        cfg=cfg,
        data_root=_infer_data_root(cfg) or None,
        sector=int(cfg.sector) if cfg.sector is not None else None,
        camera=int(cfg.camera) if cfg.camera is not None else None,
        ccd=int(cfg.ccd) if cfg.ccd is not None else None,
    )
    shared_mask = mask_catalog.static

    diff_label = str(inp.get("diffs") or "").strip()
    bkg_label = str(inp.get("bkg") or "").strip()
    bkg_in_label = str(inp.get("bkg_in") or "").strip()

    diff_dir = _diff_stage_dir(cfg, ctx, diff_label) if diff_label else None
    bkg_dir = _diff_stage_dir(cfg, ctx, bkg_label) if bkg_label else None
    bkg_in_dir = _diff_stage_dir(cfg, ctx, bkg_in_label) if bkg_in_label else None

    if diff_dir:
        records = background.build_frame_records(
            processing_ffi_paths, wcs_table, diff_dir, bkg_dir
        )
    elif bkg_in_dir:
        records = background.build_frame_records_from_stack_ws(
            processing_ffi_paths, bkg_in_dir
        )
    else:
        raise RuntimeError(
            f"background stage requires inputs.diffs and/or inputs.bkg_in"
        )

    fit_flux = None
    strap_flux = None
    bkg_in_stack = None
    if bkg_in_dir:
        bkg_in_stack = background.load_stack_or_fits(bkg_in_dir, records)
    if diff_dir:
        fit_flux, strap_flux = background.load_flux_cubes(
            records, recombine_inputs=params.recombine_inputs
        )

    # Per-frame full mask when asteroid intervals are loaded; else static 2D.
    spatial_mask = shared_mask
    if mask_catalog is not None and mask_catalog.has_temporal():
        btjds = background.btjd_for_records(wcs_table, records)
        spatial_mask = np.empty((len(records),) + shared_mask.shape, dtype=np.int16)
        for i, btjd in enumerate(btjds):
            spatial_mask[i] = mask_catalog.mask_at(btjd, which="full")

    stack = background.run_background_pipeline(
        params=params,
        records=records,
        mask=spatial_mask,
        wcs_table=wcs_table,
        sector=int(cfg.sector),
        camera=int(cfg.camera),
        n_jobs=int(cfg.n_jobs),
        fit_flux=fit_flux,
        strap_flux=strap_flux,
        bkg_in_stack=bkg_in_stack,
        workspace_resolver=lambda label: _diff_stage_dir(cfg, ctx, label),
    )

    if params.write_stack:
        background.save_stack(stack, out_ws)
    if params.write_per_frame_fits:
        from syndiff_pipeline.difference_imaging.support.paths import workspace_root as _workspace_root

        ws_index_root = _workspace_root(
            cfg.output_dir, run_id=getattr(cfg, "workspace_run_id", None)
        )
        background.write_per_frame_fits(
            out_ws,
            stack,
            records,
            sck=(int(cfg.sector), int(cfg.camera), int(cfg.ccd)),
            data_root=getattr(cfg, "data_root", "") or None,
            background_params=params,
            workspace_root=ws_index_root,
        )

    stem_rows = _records_to_stem_rows(records)
    _maybe_write_background_gif(
        cfg,
        out,
        stack,
        wcs_table,
        stem_rows,
        filename=f"{label_out}_background_animation.gif",
        cbar_label=f"Background ({label_out})",
    )
    log.info("  background: wrote stack under %s shape=%s", out_ws, stack.shape)
    return shared_mask, mask_catalog


def _pipeline_plots_root(cfg: SynDiffConfig) -> str:
    """Workspace-tree path for diagnostic figures."""
    from syndiff_pipeline.difference_imaging.support.paths import (
        normalize_workspace_run_id,
        pipeline_plots_root,
    )

    sub = getattr(cfg, "pipeline_plots_dir", None)
    return pipeline_plots_root(
        cfg.output_dir,
        sub,
        run_id=normalize_workspace_run_id(getattr(cfg, "workspace_run_id", None)),
    )


def _maybe_write_background_gif(
    cfg: SynDiffConfig,
    output_dir: str,
    cube: np.ndarray,
    wcs_table: Optional[pd.DataFrame],
    stem_rows: list,
    *,
    filename: str,
    cbar_label: str,
) -> None:
    """Animated GIF of a (T, ny, nx) background cube when ``pipeline_plots`` is True."""
    if not getattr(cfg, "pipeline_plots", False):
        return
    if wcs_table is None:
        log.debug("pipeline_plots: skip background GIF %s (no wcs_table)", filename)
        return
    from syndiff_pipeline.difference_imaging.support import plot as plot_pipeline

    plot_dir = _pipeline_plots_root(cfg)
    os.makedirs(plot_dir, exist_ok=True)
    plot_pipeline.write_background_removal_animation(
        cube,
        wcs_table,
        stem_rows,
        plot_dir,
        filename=filename,
        cbar_label=cbar_label,
    )


def _subtract_load_plane_and_sigma(
    ws_root: str,
    product_id: str,
    frame_index: int,
    npy_stack_by_ws: dict[str, np.ndarray | None],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Like :func:`_subtract_load_plane` but also load a per-pixel 1σ map when the
    workspace FITS has a ``NOISE`` extension. Stacks return ``(plane, None)``.
    """
    plane = _subtract_load_plane(ws_root, product_id, frame_index, npy_stack_by_ws)
    if plane is None:
        return None, None
    if npy_stack_by_ws.get(ws_root) is not None:
        return plane, None
    ws_label = workspace_label_from_dir(ws_root)
    stem = workspace_frame_stem(product_id, ws_label)
    fp = resolve_pipeline_fits_path(ws_root, stem)
    if fp is not None:
        return photometry.read_diff_primary_and_noise_sigma(fp)
    return plane, None


def _load_template_handoff(
    cfg: SynDiffConfig, out: str, manifest_path: str | None
) -> tuple[pd.DataFrame, dict, str, float]:
    """
    Load template-pipeline handoff from SCC bookkeeping (frames + diff job).
    """
    del out, manifest_path
    if not getattr(cfg, "data_root", None):
        raise RuntimeError(
            "SCC-only diff requires data_root for template handoff (bookkeeping/diff/)"
        )
    from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
        load_scc_diff_handoff_for_config,
    )

    wcs_table, crop_bounds, ref_ffi_path, offset_threshold, _grid = (
        load_scc_diff_handoff_for_config(cfg)
    )
    log.info(
        "Loaded SCC diff handoff (bookkeeping/diff/) for s%04d/c%d/k%d",
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
    )
    return wcs_table, crop_bounds, ref_ffi_path, offset_threshold


def _cfg_ffi_leaf(cfg: SynDiffConfig) -> str:
    """Return the SCC ffi leaf directory from ``cfg.ffi_dir``.

    After the SCC layout cutover, ``SynDiffConfig.ffi_dir`` is already the leaf
    (``…/sSSSS/cC/kK/ffi``), not the legacy ``tess_ffi`` root.
    """
    return str(cfg.ffi_dir)


def _sorted_local_ffis(cfg: SynDiffConfig) -> list:
    """Sorted local ffis.
    
    Parameters
    ----------
    cfg : SynDiffConfig
    
    Returns
    -------
    list"""
    return sorted(list_local_ffis(_cfg_ffi_leaf(cfg), cfg.sector, cfg.camera, cfg.ccd))


def _ffi_paths_for_processing(cfg: SynDiffConfig) -> list:
    """Ffi paths for processing.
    
    Parameters
    ----------
    cfg : SynDiffConfig
    
    Returns
    -------
    list"""
    all_sorted = _sorted_local_ffis(cfg)
    return wcs_grouping.select_ffis_with_valid_target_wcs(
        all_sorted,
        cfg.target_ra,
        cfg.target_dec,
        max_ffis=cfg.max_ffis,
    )


def _load_gaia_catalog(
    cfg: SynDiffConfig,
) -> Optional[pd.DataFrame]:
    # When diff_config overrides the template ROI, always load the source catalog
    # so ensure_gaia_crop_xy can reproject; skip a cached pipeline CSV from another crop.
    """Load gaia catalog from the SCC diff lane or site config."""
    prefer_source_catalog = wcs_grouping.diff_crop_explicitly_configured(cfg)
    if not prefer_source_catalog:
        lane_root = _require_scc_lane_root(cfg)
        pipeline_csv = lane_root / GAIA_CATALOG_PIPELINE_BASENAME
        if pipeline_csv.is_file():
            return pd.read_csv(pipeline_csv)
    if cfg.gaia_catalog and os.path.isfile(cfg.gaia_catalog):
        return pd.read_csv(cfg.gaia_catalog)
    return None


def _load_ffi_list_for_cfg(cfg) -> pd.DataFrame | None:
    """Load SCC ``ffi_list.parquet`` for per-frame full-FFI WCS (ePSF/centroids)."""
    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
    from syndiff_pipeline.common.wcs_header_cache import load_ffi_list

    data_root = getattr(cfg, "data_root", None)
    if not data_root:
        return None
    try:
        ffi_list_path = scc_ffi_list_parquet(
            data_root, int(cfg.sector), int(cfg.camera), int(cfg.ccd)
        )
    except Exception:
        return None
    if ffi_list_path and os.path.isfile(ffi_list_path):
        return load_ffi_list(ffi_list_path)
    return None


def _ensure_gaia_crop(
    gaia_df: pd.DataFrame,
    ref_ffi_path: str,
    crop_bounds: dict,
    cfg: SynDiffConfig,
) -> pd.DataFrame:
    """Ensure gaia crop.
    
    Parameters
    ----------
    gaia_df : pd.DataFrame
    ref_ffi_path : str
    crop_bounds : dict
    cfg : SynDiffConfig
    
    Returns
    -------
    pd.DataFrame"""
    return wcs_grouping.ensure_gaia_crop_xy(
        gaia_df,
        ref_ffi_path,
        crop_bounds,
        force_reproject=wcs_grouping.diff_crop_explicitly_configured(cfg),
    )


def _path_to_group_from_wcs(wcs_table: Optional[pd.DataFrame]) -> dict[str, int]:
    """Map ``tess<digits>`` product id → group_id from ``wcs_table``."""
    path_to_group: dict[str, int] = {}
    if wcs_table is None:
        return path_to_group
    if "filename" in wcs_table.columns:
        col = "filename"
    elif "path" in wcs_table.columns:
        col = "path"
    else:
        return path_to_group
    for _, row in wcs_table.iterrows():
        pid = tess_product_id_from_ffi_path(str(row[col]))
        if not pid:
            continue
        path_to_group[pid] = int(row.get("group_id", 0))
    return path_to_group


def _tqdm_ffi_paths(ffi_paths: list, desc: str):
    """Iterate FFI paths with a tqdm bar when tqdm is installed."""
    try:
        from tqdm import tqdm

        return tqdm(ffi_paths, desc=desc, unit="frame")
    except ImportError:
        log.debug("tqdm not installed; skipping FITS load progress bar.")
        return ffi_paths


def _tqdm_frames(
    iterable,
    *,
    desc: str,
    total: Optional[int] = None,
):
    """Wrap *iterable* with tqdm (frame unit) when tqdm is installed."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, unit="frame", total=total)
    except ImportError:
        log.debug("tqdm not installed; skipping progress bar (%s).", desc)
        return iterable


def _ensure_shared_mask_loaded(
    shared_mask: Optional[np.ndarray],
    *,
    cfg: SynDiffConfig,
) -> np.ndarray:
    """Load shared mask FITS from the SCC diff lane root."""
    if shared_mask is not None:
        return shared_mask
    lane_root = _require_scc_lane_root(cfg)
    sm_path = resolve_pipeline_artifact_path(
        str(lane_root), SHARED_MASK_FITS_BASENAME
    )
    if sm_path is not None:
        mask = np.asarray(fits.getdata(sm_path), dtype=np.int16)
        log.info("  Loaded shared_mask from SCC lane (%s)", sm_path)
        return mask
    raise RuntimeError(
        "Background stages need shared_mask in memory (run shared_mask first) "
        f"or an existing shared_mask FITS under {lane_root!r}."
    )


def _ensure_ref_stars_loaded(
    ref_stars: Optional[pd.DataFrame],
    *,
    cfg: SynDiffConfig,
) -> pd.DataFrame:
    """Load hotpants substamp stars CSV from the SCC diff lane root."""
    if ref_stars is not None:
        return ref_stars
    lane_root = _require_scc_lane_root(cfg)
    rs_path = lane_root / HOTPANTS_SUBSTAMP_STARS_BASENAME
    if rs_path.is_file():
        log.info("  Loaded hotpants_substamp_stars from SCC lane (%s)", rs_path)
        return pd.read_csv(rs_path)
    raise RuntimeError(
        "Kernel fit requires hotpants_substamp_stars (run shared_mask first) or "
        f"an existing {HOTPANTS_SUBSTAMP_STARS_BASENAME!r} under {lane_root!r}."
    )


def _ensure_mask_catalog_loaded(
    ws_root: str,
    mask_catalog,
    shared_mask: Optional[np.ndarray],
    *,
    crop_bounds: Optional[dict] = None,
    asteroid_intervals=None,
    asteroid_times=None,
    data_root: str | None = None,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
    intervals_dir: str | None = None,
    cfg: SynDiffConfig | None = None,
):
    """Load MaskCatalog from memory or workspace FITS (+ SCC asteroid sidecars)."""
    from syndiff_pipeline.difference_imaging.masking.asteroids import (
        convert_intervals_to_crop_local,
        load_asteroid_products,
    )
    from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog
    from syndiff_pipeline.difference_imaging.masking.settings import default_asteroid_intervals_dir
    from syndiff_pipeline.difference_imaging.masking.tns import TRANSIENT_FIXED_BASENAME

    def _attach_asteroids(static, tns_table, iv, tm):
        if iv is None or crop_bounds is None:
            return MaskCatalog(
                static=static,
                tns_table=tns_table,
                asteroid_intervals=None,
                asteroid_times=tm,
                crop_bounds=crop_bounds,
            )
        if "y" in iv.columns and "x" in iv.columns:
            crop_iv = iv
        else:
            crop_iv = convert_intervals_to_crop_local(iv, crop_bounds, static.shape)
        return MaskCatalog(
            static=static,
            tns_table=tns_table,
            asteroid_intervals=crop_iv,
            asteroid_times=tm,
            crop_bounds=crop_bounds,
        )

    def _load_scc_asteroids():
        if asteroid_intervals is not None:
            return asteroid_intervals, asteroid_times
        if not data_root or sector is None or camera is None or ccd is None:
            return None, None
        root = (
            Path(intervals_dir)
            if intervals_dir
            else default_asteroid_intervals_dir(data_root, sector, camera, ccd)
        )
        return load_asteroid_products(root)

    if mask_catalog is not None:
        if mask_catalog.has_temporal():
            return mask_catalog
        iv, tm = _load_scc_asteroids()
        if iv is None:
            return mask_catalog
        return _attach_asteroids(
            mask_catalog.static, mask_catalog.tns_table, iv, tm
        )

    static = _ensure_shared_mask_loaded(shared_mask, cfg=cfg)
    tns_table = None
    lane_root = _require_scc_lane_root(cfg) if cfg is not None else None
    if lane_root is not None:
        tns_path = lane_root / TRANSIENT_FIXED_BASENAME
        if tns_path.is_file():
            tns_table = pd.read_parquet(tns_path)
    iv, tm = _load_scc_asteroids()
    return _attach_asteroids(static, tns_table, iv, tm)


def _infer_data_root(cfg: SynDiffConfig) -> str:
    if getattr(cfg, "data_root", None) and str(cfg.data_root).strip():
        return str(cfg.data_root)
    ffi = Path(cfg.ffi_dir) if cfg.ffi_dir else None
    if ffi is not None and ffi.name == "tess_ffi":
        return str(ffi.parent)
    gaia = Path(cfg.gaia_catalog) if cfg.gaia_catalog else None
    if gaia is not None:
        for p in gaia.parents:
            if p.name == "catalogs":
                return str(p.parent)
    return ""


def _scc_store_name(cfg: SynDiffConfig) -> str | None:
    from syndiff_pipeline.common.scc_paths import normalize_store_name

    return normalize_store_name(getattr(cfg, "output_store_name", None))


def _scc_lane_root_path(cfg: SynDiffConfig) -> Path | None:
    data_root = _infer_data_root(cfg) or None
    if not data_root:
        return None
    from syndiff_pipeline.common.scc_paths import scc_diff_dir

    return scc_diff_dir(
        data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        store_name=_scc_store_name(cfg),
    )


def _scc_label_dir_path(cfg: SynDiffConfig, label: str) -> Path | None:
    lane_root = _scc_lane_root_path(cfg)
    safe = str(label).strip()
    if lane_root is None or not safe:
        return None
    return lane_root / safe


def _require_scc_lane_root(cfg: SynDiffConfig) -> Path:
    """Return the SCC diff lane root; raise when ``data_root`` is not configured."""
    lane_root = _scc_lane_root_path(cfg)
    if lane_root is None:
        raise RuntimeError(
            "SCC-only diff requires deployment data_root and sector/camera/ccd on config"
        )
    lane_root.mkdir(parents=True, exist_ok=True)
    return lane_root


def _diff_lane_root_dir(
    cfg: SynDiffConfig,
    ctx: PipelineInvocationContext,
) -> str:
    """SCC diff lane root under ``data_root``."""
    del ctx
    return str(_require_scc_lane_root(cfg))


def _diff_stage_dir(
    cfg: SynDiffConfig,
    ctx: PipelineInvocationContext,
    label: str,
) -> str:
    """Per-label directory under the SCC diff lane."""
    del ctx
    safe = str(label).strip()
    if not safe:
        raise ValueError("diff stage label must be non-empty")
    label_dir = _require_scc_lane_root(cfg) / safe
    label_dir.mkdir(parents=True, exist_ok=True)
    return str(label_dir)


def _ordered_diff_paths(
    cfg: SynDiffConfig,
    ctx: PipelineInvocationContext,
    wcs_table: pd.DataFrame,
    label: str,
) -> list:
    data_root = _infer_data_root(cfg) or None
    if not data_root:
        raise RuntimeError(
            "SCC-only diff requires data_root to resolve per-FFI diff paths"
        )
    return ordered_diff_paths_for_scc(
        wcs_table,
        data_root,
        int(cfg.sector),
        int(cfg.camera),
        int(cfg.ccd),
        label,
        store_name=_scc_store_name(cfg),
    )


def _mask_catalog_scc_kwargs(cfg: SynDiffConfig) -> dict:
    """Common SCC asteroid-load kwargs for Hotpants/kernel/background resume."""
    return {
        "data_root": _infer_data_root(cfg) or None,
        "sector": int(cfg.sector) if cfg.sector is not None else None,
        "camera": int(cfg.camera) if cfg.camera is not None else None,
        "ccd": int(cfg.ccd) if cfg.ccd is not None else None,
    }

def _ensure_workspace_tree_symlinks(ctx: PipelineInvocationContext, cfg: SynDiffConfig) -> None:
    """Ensure ffis symlink exists in the active workspace tree (templates are SCC-absolute)."""
    out = cfg.output_dir
    run_id = ctx.workspace_run_id
    os.makedirs(ctx.workspace_root_path(), exist_ok=True)

    ffis_link = event_ffis_symlink_path(out, run_id=run_id)
    if not ffis_link.is_symlink():
        canon = event_ffis_symlink_path(out)
        if canon.is_symlink():
            try:
                ensure_event_ffis_symlink(out, canon.resolve(), run_id=run_id)
            except OSError as exc:
                log.warning("workspace ffis symlink failed: %s", exc)
        elif cfg.ffi_dir:
            ffi_leaf = _cfg_ffi_leaf(cfg)
            if os.path.isdir(ffi_leaf):
                ensure_event_ffis_symlink(out, ffi_leaf, run_id=run_id)


def _ensure_template_paths_for_kernel(
    cfg: SynDiffConfig,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    offset_threshold: float,
) -> None:
    """Ensure template paths for kernel.
    
    Parameters
    ----------
    cfg : SynDiffConfig
    wcs_table : pd.DataFrame
    crop_bounds : dict
    offset_threshold : float"""
    try:
        hotpants_runner.ensure_template_paths_from_syndiff_or_group_dirs(
            cfg,
            wcs_table,
            crop_bounds,
            offset_threshold=offset_threshold,
        )
    except hotpants_runner.SyndiffTemplateDiscoveryError as e:
        raise RuntimeError(str(e)) from e
    if not cfg.template_paths:
        raise RuntimeError(
            "template_paths empty; set template_dir or template_paths after WCS grouping."
        )


def run_config_pipeline(
    cfg: SynDiffConfig,
    *,
    validate_only: bool = False,
    diff_log_path: str | None = None,
    force_rerun: bool = False,
) -> None:
    """Run config pipeline.
    
    Parameters
    ----------
    cfg : SynDiffConfig
    validate_only : bool, optional, default ``False``
    diff_log_path : str | None, optional, default ``None``
    force_rerun : bool, optional, default ``False``"""
    validate_pipeline(cfg)
    if validate_only:
        log.info("Pipeline configuration is valid.")
        return

    effective_n_jobs = resolve_effective_n_jobs(int(cfg.n_jobs or 1))
    if effective_n_jobs != cfg.n_jobs:
        log.info(
            "effective n_jobs=%s (cfg.n_jobs=%s, SYNDIFF_REQUEST_CPUS=%s, cpu_count=%s)",
            effective_n_jobs,
            cfg.n_jobs,
            os.environ.get("SYNDIFF_REQUEST_CPUS", ""),
            __import__("multiprocessing").cpu_count(),
        )
    cfg.n_jobs = effective_n_jobs

    ctx = PipelineInvocationContext.from_config(cfg)
    out = ctx.cfg.output_dir
    ws_root = ctx.workspace_root_path()
    manifest_path = ctx.manifest_path
    os.makedirs(out, exist_ok=True)
    os.makedirs(ws_root, exist_ok=True)

    assert_workspace_config_lock(ws_root, cfg)
    _require_scc_lane_root(cfg)
    _, inherit_specs, _ = split_pipeline(cfg.pipeline)
    if inherit_specs:
        raise RuntimeError(
            "workspace_inherit is not supported under SCC-only diff storage; "
            "re-run upstream diff stages on the SCC lane instead."
        )

    _ensure_workspace_tree_symlinks(ctx, cfg)
    write_immutable_workspace_config_snapshot(ctx, cfg)

    shared_mask = None
    mask_catalog = None
    ref_stars: Optional[pd.DataFrame] = None
    gaia_df: Optional[pd.DataFrame] = None
    tile_centers = None
    processing_ffi_paths: list = []
    kernel_fit_hp: Optional[HotpantsParams] = None
    convolved_ws: Optional[str] = None

    from syndiff_pipeline.difference_imaging.stages.astrometry import (
        pipeline_needs_template_handoff,
    )

    needs_handoff = pipeline_needs_template_handoff(cfg.pipeline)
    wcs_table: Optional[pd.DataFrame] = None
    crop_bounds: Optional[dict] = None
    ref_ffi_path: Optional[str] = None
    pipeline_offset_threshold = 0.01

    if needs_handoff:
        wcs_table, crop_bounds, ref_ffi_path, pipeline_offset_threshold = (
            _load_template_handoff(cfg, out, manifest_path)
        )

    # Field-mode (geometry_mode: field) template assembly context, loaded once so
    # every template-consuming stage (shared_mask, hotpants, kernel_*) shares it.
    # Returns None for linear runs (no field_mode_assembly.json sidecar), so the
    # linear path is unaffected.
    field_ctx = None
    if needs_handoff:
        from syndiff_pipeline.difference_imaging.support.template_resolution import (
            maybe_load_field_mode_template_context,
        )

        field_ctx = maybe_load_field_mode_template_context(
            getattr(cfg, "template_dir", None), out
        )
        if field_ctx is not None:
            log.info("Field-mode template assembly active (geometry_mode: field)")
            grid = getattr(field_ctx, "mapping_grid", None)
            if grid is not None and crop_bounds is not None:
                from syndiff_pipeline.common.coordinate_preflight import (
                    validate_coordinate_contract,
                    validate_conv_pad_for_diff,
                )

                validate_coordinate_contract(grid, crop_bounds)
                from syndiff_pipeline.difference_imaging.stages.hotpants import (
                    _kernel_scale_pixels,
                )

                for stage in cfg.pipeline:
                    if not isinstance(stage, dict):
                        continue
                    kind = stage.get("kind")
                    if kind in ("kernel_fit", "hotpants"):
                        hp_probe = (
                            kernel_fit_params_to_hotpants(parse_kernel_fit(stage, 0))
                            if kind == "kernel_fit"
                            else parse_hotpants(stage, 0)
                        )
                        validate_conv_pad_for_diff(
                            grid, scale_px=float(_kernel_scale_pixels(hp_probe))
                        )
                        break

    for idx, stage in enumerate(cfg.pipeline):
        if is_external_workspaces_entry(stage) or is_workspace_inherit_entry(stage):
            continue
        kind = stage["kind"]
        log.info("=" * 70)
        log.info("Stage: %s", kind)

        if kind == "photometry":
            from syndiff_pipeline.photometry.runner import run_photometry_delegator

            site_dir = getattr(cfg, "site_config_dir", None) or out
            run_photometry_delegator(
                cfg,
                stage,
                site_dir,
                force_rerun=force_rerun,
            )
            continue

        if kind == "shared_mask":
            sm = parse_shared_mask(stage, idx)
            gaia_df = _load_gaia_catalog(cfg)
            if gaia_df is None:
                raise RuntimeError("gaia_catalog required for shared_mask.")
            gaia_df = _ensure_gaia_crop(gaia_df, ref_ffi_path, crop_bounds, cfg)

            with wcs_grouping.open_fits_memmap(ref_ffi_path) as hdul:
                ref_header = hdul[1].header
                ref_data = hdul[1].data.astype(np.float64)
                ffi_nx = int(ref_header["NAXIS1"])
                ffi_ny = int(ref_header["NAXIS2"])
            ref_crop = wcs_grouping.crop_image(ref_data, crop_bounds)

            gaia_mask_df = epsf_fitting.add_tess_flux_ratio(gaia_df.copy())
            gaia_mask_df["mag"] = gaia_mask_df["tess_mag"]

            from syndiff_pipeline.difference_imaging.masking.api import generate_shared_mask_catalog
            from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
                legacy_mask_stage_overrides,
            )
            from syndiff_pipeline.difference_imaging.masking.settings import (
                apply_stage_overrides,
                resolve_mask_settings,
            )

            data_root = _infer_data_root(cfg)
            if not data_root:
                raise RuntimeError(
                    "SCC-only shared_mask requires deployment data_root on config"
                )
            site_dir = getattr(cfg, "site_config_dir", None) or None
            lane_root = _diff_lane_root_dir(cfg, ctx)
            legacy_mask = legacy_mask_stage_overrides(stage)
            if legacy_mask:
                log.warning(
                    "shared_mask stage sets legacy mask knobs %s; prefer "
                    "config/mask_settings.yaml (shared.bright_maglim / strapsize / "
                    "ps1_min_hit_count)",
                    sorted(legacy_mask),
                )
            mask_settings, _ = resolve_mask_settings(
                stage_mask_settings=sm.mask_settings,
                site_dir=site_dir,
                ws_root=lane_root,
            )
            mask_settings = apply_stage_overrides(
                mask_settings,
                epsf_mag_lim=legacy_mask.get("epsf_mag_lim"),
                gaia_mag_bright=legacy_mask.get("gaia_mag_bright"),
                strapsize=legacy_mask.get("strapsize"),
                ps1_min_hit_count=legacy_mask.get("ps1_min_hit_count"),
            )

            ref_template_path: str | None = None
            ref_template_count_crop = None
            if int(mask_settings.shared.ps1_min_hit_count) > 0:
                ref_row = wcs_grouping.ref_manifest_row_index(wcs_table, ref_ffi_path)
                if ref_row is None:
                    raise RuntimeError(
                        f"reference FFI {ref_ffi_path!r} not found in frame manifest"
                    )
                ref_group_id = int(wcs_table.iloc[ref_row]["group_id"])
                if field_ctx is not None:
                    # Field mode: assemble the reference group's COUNT plane from
                    # the SCC field store instead of a linear template FITS.
                    from syndiff_pipeline.difference_imaging.support.template_resolution import (
                        build_field_mode_count_loader,
                    )

                    ref_template_count_crop = build_field_mode_count_loader(
                        field_ctx, crop_bounds
                    )(ref_group_id)
                    log.info(
                        "  PS1 coverage mask from assembled field COUNT (group_id=%s)",
                        ref_group_id,
                    )
                else:
                    _ensure_template_paths_for_kernel(
                        cfg,
                        wcs_table,
                        crop_bounds,
                        pipeline_offset_threshold,
                    )
                    ref_template_path = cfg.template_paths.get(ref_group_id)
                    if not ref_template_path:
                        raise RuntimeError(
                            f"No template path for reference group_id={ref_group_id}"
                        )
                    log.info(
                        "  PS1 coverage mask from reference template (group_id=%s): %s",
                        ref_group_id,
                        ref_template_path,
                    )

            plots_dir = None
            if getattr(cfg, "pipeline_plots", False):
                plots_dir = os.path.join(_pipeline_plots_root(cfg), "masks")

            lane_root = _diff_lane_root_dir(cfg, ctx)

            mask_catalog = generate_shared_mask_catalog(
                ref_image=ref_crop,
                gaia_df=gaia_mask_df,
                crop_bounds=crop_bounds,
                lane_root=lane_root,
                data_root=data_root,
                sector=int(cfg.sector),
                camera=int(cfg.camera),
                ccd=int(cfg.ccd),
                output_store_name=_scc_store_name(cfg),
                straps_csv=cfg.straps_csv or None,
                ref_ffi_path=ref_ffi_path,
                bsc_catalog_path=cfg.bsc_catalog or None,
                nx=ffi_nx,
                ny=ffi_ny,
                x_left_dead=int(getattr(cfg, "x_left_dead", 44)),
                x_right_dead=int(getattr(cfg, "x_right_dead", 44)),
                y_edge_strip=int(getattr(cfg, "y_edge_strip", 30)),
                template_path=ref_template_path,
                template_count_crop=ref_template_count_crop,
                settings=mask_settings,
                stage_mask_settings=sm.mask_settings,
                site_dir=site_dir,
                wcs_table=wcs_table,
                write_plots_dir=plots_dir,
                mask_params=sm,
            )
            shared_mask = mask_catalog.static
            ref_stars = masking.select_hotpants_ref_stars(
                gaia_df=gaia_mask_df,
                crop_bounds=crop_bounds,
                mag_min=sm.ref_mag_min,
                mag_max=sm.ref_mag_max,
                isolation_mag=sm.ref_isolation_mag,
                isolation_radius_px=sm.ref_isolation_px,
                separation_px=sm.ref_separation_px,
                output_dir=lane_root,
            )
            pipe_csv = os.path.join(lane_root, GAIA_CATALOG_PIPELINE_BASENAME)
            gaia_mask_df.to_csv(pipe_csv, index=False)
            gaia_df = gaia_mask_df.drop(columns=["mag"], errors="ignore")

        elif kind == "hotpants":
            hp = parse_hotpants(stage, idx)
            if wcs_table is None or crop_bounds is None or ref_ffi_path is None:
                raise RuntimeError(
                    "hotpants requires wcs_table, crop_bounds, and reference FFI metadata "
                    "(template handoff required: syndiff_ffi_frames.csv and "
                    "cluster_template_job.json in output_dir)."
                )
            shared_mask = _ensure_shared_mask_loaded(shared_mask, cfg=cfg)
            mask_catalog = _ensure_mask_catalog_loaded(
                ws_root,
                mask_catalog,
                shared_mask,
                crop_bounds=crop_bounds,
                cfg=cfg,
                **_mask_catalog_scc_kwargs(cfg),
            )
            shared_mask = mask_catalog.static
            if ref_stars is None:
                lane_root = _diff_lane_root_dir(cfg, ctx)
                rs_path = os.path.join(lane_root, HOTPANTS_SUBSTAMP_STARS_BASENAME)
                if not os.path.isfile(rs_path):
                    raise RuntimeError(
                        "hotpants requires hotpants_substamp_stars (run shared_mask first) or "
                        f"an existing {rs_path!r} from a prior run."
                    )
                ref_stars = pd.read_csv(rs_path)
                log.info("  Loaded hotpants_substamp_stars from prior run (%s)", rs_path)
            # Field mode uses on-demand assembly (field_ctx hoisted above);
            # linear mode discovers per-group template FITS into cfg.template_paths.
            if field_ctx is None:
                hotpants_runner.ensure_template_paths_from_syndiff_or_group_dirs(
                    cfg,
                    wcs_table,
                    crop_bounds,
                    offset_threshold=pipeline_offset_threshold,
                )
                if not cfg.template_paths:
                    raise RuntimeError(
                        "template_paths empty; set template_dir or template_paths after WCS grouping."
                    )

            inp = stage.get("inputs") or {}
            o = stage["output"]
            diffs_l = o["diffs"]
            conv_l = o["convolved"]
            bkg_l = o.get("bkg")

            diff_dir = _diff_stage_dir(cfg, ctx, diffs_l)
            conv_dir = _diff_stage_dir(cfg, ctx, conv_l)
            bkg_dir = _diff_stage_dir(cfg, ctx, bkg_l) if bkg_l else None

            dirs = HotpantsWorkspaceDirs(
                diffs=diff_dir,
                convolved=conv_dir,
                bkg=bkg_dir,
            )

            processing_ffi_paths = _ffi_paths_for_processing(cfg)
            sci_bkg_ws = _diff_stage_dir(cfg, ctx, inp["bkg"]) if inp.get("bkg") else None

            round_id = 2 if inp.get("bkg") else 1
            sci_label = str(stage.get("science", "ffi")).strip()
            sci_workspace_dir = (
                None if sci_label == "ffi" else _diff_stage_dir(cfg, ctx, sci_label)
            )
            results = hotpants_runner.hotpants_loop(
                ffi_paths=processing_ffi_paths,
                wcs_table=wcs_table,
                template_path_map={int(k): v for k, v in (cfg.template_paths or {}).items()},
                mask=shared_mask,
                crop_bounds=crop_bounds,
                hp=hp,
                cfg=cfg,
                output_dir=out,
                ref_stars_df=ref_stars,
                round_id=round_id,
                sci_bkg_ws=sci_bkg_ws,
                workspace_dirs=dirs,
                sci_workspace_dir=sci_workspace_dir,
                diffs_label=diffs_l,
                science=sci_label,
                diff_log_path=diff_log_path,
                force_rerun=force_rerun,
                field_mode_context=field_ctx,
                mask_catalog=mask_catalog,
            )
            n_ok = sum(1 for r in results if r.get("success"))
            if n_ok == 0:
                raise RuntimeError(
                    f"hotpants [{diffs_l}] round {round_id}: 0/{len(results)} frames "
                    "succeeded; check manifest path matching and template_paths."
                )
            wcs_table = apply_hotpants_workspace_results(
                wcs_table, processing_ffi_paths, results, diffs_l
            )
            save_frame_manifest(wcs_table, out, manifest_path)

            if inp.get("convolved"):
                log.warning(
                    "hotpants inputs.convolved=%r ignored in this version (convolved products are always written to output.convolved).",
                    inp["convolved"],
                )

        elif kind == "kernel_fit":
            kf_params = parse_kernel_fit(stage, idx)
            if wcs_table is None or crop_bounds is None:
                raise RuntimeError(
                    "kernel_fit requires wcs_table and crop_bounds from template handoff."
                )
            shared_mask = _ensure_shared_mask_loaded(shared_mask, cfg=cfg)
            mask_catalog = _ensure_mask_catalog_loaded(
                ws_root,
                mask_catalog,
                shared_mask,
                crop_bounds=crop_bounds,
                cfg=cfg,
                **_mask_catalog_scc_kwargs(cfg),
            )
            shared_mask = mask_catalog.static
            ref_stars = _ensure_ref_stars_loaded(ref_stars, cfg=cfg)
            ref_stars_xy = ref_stars[["x", "y"]].to_numpy(dtype=np.float64)
            hp = kernel_fit_params_to_hotpants(kf_params)
            kernel_fit_hp = hp
            kernel_fit_label = str(stage["output"]).strip()
            kernel_fit_ws = _diff_stage_dir(cfg, ctx, kernel_fit_label)
            kernel_fit_runner.run_kernel_fit(
                output_dir=out,
                manifest=wcs_table,
                crop_bounds=crop_bounds,
                shared_mask=shared_mask,
                ref_stars_xy=ref_stars_xy,
                hp=hp,
                params=kf_params,
                artifact_dir=kernel_fit_ws,
                debug_ws_dir=kernel_fit_ws,
                field_ctx=field_ctx,
                mask_catalog=mask_catalog,
            )

        elif kind == "convolved_templates":
            parse_convolved_templates(stage, idx)
            if wcs_table is None or crop_bounds is None:
                raise RuntimeError(
                    "convolved_templates requires wcs_table and crop_bounds from template handoff."
                )
            # Linear discovers per-group template FITS; field assembles from store.
            if field_ctx is None:
                _ensure_template_paths_for_kernel(
                    cfg, wcs_table, crop_bounds, pipeline_offset_threshold
                )
            hp = kernel_fit_hp or HotpantsParams()
            inp = stage.get("inputs") or {}
            kernel_fit_label = str(inp["kernel_fit"]).strip()
            kernel_fit_ws = _diff_stage_dir(cfg, ctx, kernel_fit_label)
            conv_label = str(stage["output"]).strip()
            conv_ws = _diff_stage_dir(cfg, ctx, conv_label)
            convolved_ws = conv_ws
            convolved_templates_runner.run_convolved_templates(
                kernel_fit_dir=kernel_fit_ws,
                crop_bounds=crop_bounds,
                template_paths={int(k): v for k, v in (cfg.template_paths or {}).items()},
                hp=hp,
                convolved_ws_dir=conv_ws,
                field_ctx=field_ctx,
                manifest=wcs_table,
            )

        elif kind == "kernel_subtract":
            ks_params = parse_kernel_subtract(stage, idx)
            if wcs_table is None or crop_bounds is None:
                raise RuntimeError(
                    "kernel_subtract requires wcs_table and crop_bounds from template handoff."
                )
            shared_mask = _ensure_shared_mask_loaded(shared_mask, cfg=cfg)
            mask_catalog = _ensure_mask_catalog_loaded(
                ws_root,
                mask_catalog,
                shared_mask,
                crop_bounds=crop_bounds,
                cfg=cfg,
                **_mask_catalog_scc_kwargs(cfg),
            )
            shared_mask = mask_catalog.static
            inp = stage.get("inputs") or {}
            conv_label = str(inp["convolved"]).strip()
            conv_ws = convolved_ws or _diff_stage_dir(cfg, ctx, conv_label)
            convolved_table = convolved_templates_runner.load_convolved_templates_table(
                conv_ws
            )
            o = stage["output"]
            diffs_l = str(o["diffs"]).strip()
            bkg_l = o.get("phot_bkg")
            bkg_l = str(bkg_l).strip() if bkg_l else None
            diff_dir = _diff_stage_dir(cfg, ctx, diffs_l)
            bkg_dir = _diff_stage_dir(cfg, ctx, bkg_l) if bkg_l else None
            if not processing_ffi_paths:
                processing_ffi_paths = _ffi_paths_for_processing(cfg)
            n_jobs = ks_params.kernel_subtract_n_jobs or cfg.n_jobs
            results = kernel_subtract_runner.kernel_subtract_loop(
                ffi_paths=processing_ffi_paths,
                output_dir=out,
                manifest=wcs_table,
                crop_bounds=crop_bounds,
                shared_mask=shared_mask,
                convolved_table=convolved_table,
                phot_box_size=ks_params.phot_box_size,
                diffs_dir=diff_dir,
                diffs_label=diffs_l,
                bkg_dir=bkg_dir,
                bkg_label=bkg_l,
                n_jobs=n_jobs,
                field_mode=field_ctx is not None,
                cfg=cfg,
                mask_catalog=mask_catalog,
            )
            wcs_table = apply_hotpants_workspace_results(
                wcs_table, processing_ffi_paths, results, diffs_l
            )
            save_frame_manifest(wcs_table, out, manifest_path)

        elif kind == "epsf":
            epsf_p = parse_epsf(stage, idx)
            inp = stage["inputs"]
            label_out = stage["output"]
            diff_paths = _ordered_diff_paths(cfg, ctx, wcs_table, str(inp["diffs"]))
            if cfg.max_ffis is not None:
                diff_paths, _epsf_rows = limit_diff_paths(diff_paths, cfg.max_ffis)
            if gaia_df is None:
                gaia_df = _load_gaia_catalog(cfg)
            if gaia_df is None:
                raise RuntimeError("epsf requires gaia_catalog.")
            ffi_list_df = _load_ffi_list_for_cfg(cfg)
            if ffi_list_df is None:
                raise RuntimeError(
                    "epsf requires ffi_list.parquet under data_root for per-frame WCS."
                )
            from syndiff_pipeline.difference_imaging.stages import gridded_epsf as _gridded_epsf

            ffi_path_by_stem = _gridded_epsf.ffi_path_by_stem_from_wcs_table(wcs_table)
            mask_catalog = _ensure_mask_catalog_loaded(
                ws_root,
                mask_catalog,
                shared_mask,
                crop_bounds=crop_bounds,
                cfg=cfg,
                **_mask_catalog_scc_kwargs(cfg),
            )
            shared_mask = mask_catalog.static
            lane_root = _diff_lane_root_dir(cfg, ctx)
            shared_mask_path = resolve_pipeline_artifact_path(
                lane_root, SHARED_MASK_FITS_BASENAME
            )
            ws_out = _diff_stage_dir(cfg, ctx, label_out)
            os.makedirs(ws_out, exist_ok=True)
            epsf_stack, tile_centers_new, ffi_stems, epsf_ok = (
                epsf_fitting.fit_epsf_all_frames(
                    diff_paths,
                    gaia_df,
                    cfg,
                    epsf_p,
                    ws_out,
                    round_id=1,
                    shared_mask_path=(
                        shared_mask_path
                        if shared_mask_path and os.path.isfile(shared_mask_path)
                        else None
                    ),
                    mask_catalog=mask_catalog,
                    wcs_table=wcs_table,
                    diff_log_path=diff_log_path,
                    epsf_label=label_out,
                    diffs_input=str(inp["diffs"]),
                    force_rerun=force_rerun,
                    ffi_list_df=ffi_list_df,
                    science_bounds=crop_bounds,
                    ffi_path_by_stem=ffi_path_by_stem,
                )
            )
            if tile_centers_new is not None:
                tile_centers = tile_centers_new
            wcs_table = apply_epsf_status(wcs_table, ffi_stems, epsf_ok, round_id=1)
            save_frame_manifest(wcs_table, out, manifest_path)

            epsf_smooth = epsf_fitting.prepare_epsf_stack(epsf_stack)
            epsf_fitting.save_epsf_smooth(
                epsf_smooth, ws_out, round_id=1, ffi_stem=ffi_stems
            )
            group_ids = group_ids_from_ffi_stems(wcs_table, ffi_stems)
            from syndiff_pipeline.difference_imaging.stages import gridded_epsf

            if gridded_epsf.workspace_has_gridded_epsf(ws_out):
                gridded_epsf.compute_group_epsf_gridded(
                    ws_out, group_ids, ffi_stems, epsf_ok
                )
            epsf_fitting.compute_group_epsf(
                epsf_smooth, group_ids, output_dir=ws_out
            )

            if getattr(cfg, "pipeline_plots", False):
                from syndiff_pipeline.difference_imaging.support.plot import (
                    write_gridded_epsf_workspace_plots,
                )

                crop_shape = None
                if crop_bounds is not None and "shape" in crop_bounds:
                    crop_shape = tuple(crop_bounds["shape"])
                dpi = int(getattr(cfg, "pipeline_plot_dpi", 150) or 150)
                write_gridded_epsf_workspace_plots(
                    ws_out,
                    _pipeline_plots_root(cfg),
                    epsf_label=label_out,
                    dpi=dpi,
                    crop_shape=crop_shape,
                )

        elif kind == "centroids":
            centroids_p = parse_centroids(stage, idx)
            inp = stage["inputs"]
            label_out = stage["output"]
            diff_paths = _ordered_diff_paths(cfg, ctx, wcs_table, str(inp["diffs"]))
            if cfg.max_ffis is not None:
                diff_paths, _centroids_rows = limit_diff_paths(diff_paths, cfg.max_ffis)
            if gaia_df is None:
                gaia_df = _load_gaia_catalog(cfg)
            if gaia_df is None:
                raise RuntimeError("centroids requires gaia_catalog.")
            ffi_list_df = _load_ffi_list_for_cfg(cfg)
            if ffi_list_df is None:
                raise RuntimeError(
                    "centroids requires ffi_list.parquet under data_root for per-frame WCS."
                )
            from syndiff_pipeline.difference_imaging.stages import gridded_epsf as _gridded_epsf

            ffi_path_by_stem = _gridded_epsf.ffi_path_by_stem_from_wcs_table(wcs_table)
            epsf_ws = _diff_stage_dir(cfg, ctx, str(inp["epsf"]))
            from syndiff_pipeline.difference_imaging.stages import gridded_epsf

            epsf_catalog = gridded_epsf.catalog_from_workspace(epsf_ws)
            if epsf_catalog is None:
                raise RuntimeError(
                    f"centroids requires gridded ePSF under SCC lane {inp['epsf']!r} "
                    f"({epsf_ws})"
                )
            ws_out = _diff_stage_dir(cfg, ctx, label_out)
            os.makedirs(ws_out, exist_ok=True)
            from syndiff_pipeline.difference_imaging.stages import centroids

            centroids.run_centroids_all_frames(
                diff_paths,
                gaia_df,
                epsf_catalog,
                cfg,
                centroids_p,
                ws_out,
                centroids_label=label_out,
                diffs_input=str(inp["diffs"]),
                epsf_input=str(inp["epsf"]),
                diff_log_path=diff_log_path,
                force_rerun=force_rerun,
                ffi_list_df=ffi_list_df,
                science_bounds=crop_bounds,
                ffi_path_by_stem=ffi_path_by_stem,
                wcs_table=wcs_table,
            )

        elif kind == "sat_template":
            sat_p = parse_sat_template(stage, idx)
            inp = stage["inputs"]
            label_out = stage["output"]
            ws_epsf = _diff_stage_dir(cfg, ctx, inp["epsf"])
            epsf_smooth, _ = epsf_fitting.load_epsf_smooth(ws_epsf, 1)
            group_epsf = _load_group_epsf_from_dir(ws_epsf, "group_epsf")

            if tile_centers is None and crop_bounds is not None:
                from syndiff_pipeline.difference_imaging.stages.epsf import _make_tile_grid

                ny, nx = crop_bounds["shape"]
                tiles = _make_tile_grid(ny, nx, sat_p.tile_ny, sat_p.tile_nx)
                tile_centers = [
                    (c0 + ts / 2, r0 + ts / 2) for (r0, c0, ts) in tiles
                ]

            removed_df = _load_removed_stars_in_crop(
                cfg.removed_stars_csv,
                crop_bounds,
                gaia_df,
                ref_ffi_path,
                force_reproject=wcs_grouping.diff_crop_explicitly_configured(cfg),
            )
            ws_sat = _diff_stage_dir(cfg, ctx, label_out)
            os.makedirs(ws_sat, exist_ok=True)
            sat_native, sat_hr = sat_template.build_all_group_templates(
                removed_df, group_epsf, tile_centers, crop_bounds, sat_p
            )
            sat_template.save_group_templates(sat_native, sat_hr, ws_sat, round_id=1)

        elif kind == "subtract":
            parse_subtract(stage, idx)
            inp = stage["inputs"]
            label_out = stage["output"]
            out_ws = _diff_stage_dir(cfg, ctx, label_out)
            os.makedirs(out_ws, exist_ok=True)

            expr = inp.get("expression")
            if isinstance(expr, str) and expr.strip():
                terms = parse_subtract_expression(expr)
            else:
                terms = [
                    (1, inp["science"]),
                    (-1, inp["template"]),
                ]

            if wcs_table is None:
                raise RuntimeError(
                    "subtract requires a frame manifest (wcs_table). "
                    "Template handoff required: syndiff_ffi_frames.csv in output_dir."
                )
            if any(lab == "ffi" for _, lab in terms) and crop_bounds is None:
                raise RuntimeError(
                    "subtract: label 'ffi' needs crop_bounds (template handoff required: "
                    "cluster_template_job.json with crop metadata)."
                )

            src_col = "filename" if "filename" in wcs_table.columns else "path"
            product_ids = wcs_table[src_col].map(
                lambda x: tess_product_id_from_ffi_path(str(x)) or ""
            )
            npy_cache: dict[str, np.ndarray | None] = {}

            out_label = workspace_label_from_dir(out_ws)
            n_rows = len(product_ids)
            for i, pid in _tqdm_frames(
                enumerate(product_ids),
                desc=f"subtract → {label_out}",
                total=n_rows,
            ):
                if not pid:
                    continue
                acc: np.ndarray | None = None
                acc_var: np.ndarray | None = None
                skip = False
                for sign, lab in terms:
                    if lab == "ffi":
                        row = wcs_table.iloc[i]
                        ffi_path = str(row["path"])
                        plane, err_map = hotpants_runner._load_ffi_cropped(
                            ffi_path, crop_bounds
                        )
                        plane = plane.astype(np.float64)
                        if err_map is not None and np.any(np.isfinite(err_map)):
                            sigma = np.asarray(err_map, dtype=np.float64)
                            sigma = np.where(
                                np.isfinite(sigma),
                                np.maximum(np.abs(sigma), 1e-6),
                                1e-6,
                            )
                        else:
                            sigma = None
                    else:
                        plane, sigma = _subtract_load_plane_and_sigma(
                            _diff_stage_dir(cfg, ctx, str(lab)), pid, i, npy_cache
                        )
                    if plane is None:
                        skip = True
                        break
                    vterm = sigma**2 if sigma is not None else None
                    if acc is None:
                        acc = sign * plane
                        acc_var = None if vterm is None else vterm.copy()
                    else:
                        if plane.shape != acc.shape:
                            raise RuntimeError(
                                "subtract: shape mismatch for "
                                f"{pid!r} between workspaces ({acc.shape} vs {plane.shape})"
                            )
                        acc = acc + sign * plane
                        if acc_var is not None and vterm is not None:
                            acc_var = acc_var + vterm
                        else:
                            acc_var = None
                if skip or acc is None:
                    continue
                out_stem = workspace_frame_stem(pid, out_label)
                out_fp = workspace_frame_fits_path(out_ws, out_stem)
                if acc_var is not None:
                    noise_sigma = np.sqrt(acc_var)
                    write_hdul_fits(
                        out_fp,
                        fits.HDUList(
                            [
                                fits.PrimaryHDU(acc.astype(np.float32)),
                                fits.ImageHDU(
                                    noise_sigma.astype(np.float32), name="NOISE"
                                ),
                            ]
                        ),
                    )
                else:
                    write_image_fits(out_fp, acc.astype(np.float32))

        elif kind == "background":
            if wcs_table is None:
                raise RuntimeError(
                    "background requires wcs_table from template handoff."
                )
            if not processing_ffi_paths:
                processing_ffi_paths = _ffi_paths_for_processing(cfg)
            shared_mask, mask_catalog = _run_background_stage(
                stage=stage,
                idx=idx,
                cfg=cfg,
                ctx=ctx,
                ws_root=ws_root,
                shared_mask=shared_mask,
                mask_catalog=mask_catalog,
                wcs_table=wcs_table,
                processing_ffi_paths=processing_ffi_paths,
                out=out,
                crop_bounds=crop_bounds,
            )

        else:
            raise RuntimeError(f"Unhandled stage kind {kind!r}")

    log.info("=" * 70)
    log.info("Config pipeline complete. Outputs: %s", ws_root)

def _load_group_epsf_from_dir(output_dir: str, subdir: str = "group_epsf") -> dict:
    """Load group epsf from dir.
    
    Parameters
    ----------
    output_dir : str
    subdir : str, optional, default ``'group_epsf'``
    
    Returns
    -------
    dict"""
    d = {}
    sub = os.path.join(output_dir, subdir)
    if not os.path.isdir(sub):
        return d
    for path in sorted(glob.glob(os.path.join(sub, "group_epsf_*.npy"))):
        gid = int(os.path.basename(path).replace("group_epsf_", "").replace(".npy", ""))
        d[gid] = np.load(path)
    for path in sorted(glob.glob(os.path.join(sub, "group_epsf_*.npz"))):
        gid = int(os.path.basename(path).replace("group_epsf_", "").replace(".npz", ""))
        z = np.load(path, allow_pickle=False)
        try:
            from syndiff_pipeline.difference_imaging.stages import gridded_epsf

            data = np.asarray(z["data"], dtype=np.float64)
            d[gid] = gridded_epsf.stack_from_gridded_cube(data)
        finally:
            z.close()
    return d


def _load_removed_stars_in_crop(
    removed_stars_csv: str,
    crop_bounds: dict,
    gaia_df: Optional[pd.DataFrame],
    ref_ffi_path: str | None = None,
    *,
    force_reproject: bool = False,
) -> pd.DataFrame:
    """Load removed stars in crop.
    
    Parameters
    ----------
    removed_stars_csv : str
    crop_bounds : dict
    gaia_df : Optional[pd.DataFrame]
    ref_ffi_path : str | None, optional, default ``None``
    force_reproject : bool, optional, default ``False``
    
    Returns
    -------
    pd.DataFrame"""
    if (
        gaia_df is not None
        and not force_reproject
        and "x" in gaia_df.columns
        and "y" in gaia_df.columns
    ):
        return gaia_df.copy()
    if not removed_stars_csv or not os.path.isfile(removed_stars_csv):
        log.warning("removed_stars_csv missing; empty DataFrame for sat templates.")
        return pd.DataFrame()
    df = pd.read_csv(removed_stars_csv)
    df = df.drop_duplicates(subset="source_id")
    df = df[df["source_id"] != -1].copy()
    if ref_ffi_path and "ra" in df.columns and "dec" in df.columns:
        return wcs_grouping.ensure_gaia_crop_xy(
            df,
            ref_ffi_path,
            crop_bounds,
            force_reproject=force_reproject,
        )
    if "x" not in df.columns:
        return pd.DataFrame()
    ny, nx = crop_bounds["shape"]
    in_crop = (
        (df["x"] >= 0)
        & (df["x"] < nx)
        & (df["y"] >= 0)
        & (df["y"] < ny)
    )
    return df[in_crop].copy().reset_index(drop=True)

