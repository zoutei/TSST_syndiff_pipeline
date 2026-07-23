"""Standalone WCS grouping handoff for the template pipeline."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.wcs_grouping import FRAMES_CSV_BASENAME
from syndiff_pipeline.common.download import ffi_glob_patterns, list_local_ffis, manifest_basename_from_local
from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
from syndiff_pipeline.common.wcs_header_cache import (
    ensure_scc_ffi_list,
    ffi_list_is_complete,
    header_from_cached_row,
    load_ffi_list,
)
from syndiff_pipeline.difference_imaging.support.paths import pipeline_plots_root
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import WcsGroupingStageParams

log = logging.getLogger(__name__)


def _norm_bkg_vector_path(p: Optional[str]) -> Optional[str]:
    """Norm bkg vector path."""
    if p is None or (isinstance(p, str) and not str(p).strip()):
        return None
    return str(p)


def run_wcs_grouping(
    resolved: ResolvedTargetConfig,
    *,
    ref_ffi_path: str | None = None,
    max_ffis: int | None = None,
    x_min: int | None = None,
    x_max: int | None = None,
    y_min: int | None = None,
    y_max: int | None = None,
) -> str:
    """
    Run WCS grouping for one SCC target and write cluster_template_job.json.

    Returns absolute path to the job JSON.
    """
    t = resolved.target
    wg: WcsGroupingStageParams = resolved.stages.wcs_grouping
    event_dir = resolved.event_dir
    os.makedirs(event_dir, exist_ok=True)

    ffi_leaf = resolved.ffi_dir
    all_sorted = sorted(list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd))
    if not all_sorted:
        patterns = ffi_glob_patterns(t.sector, t.camera, t.ccd)
        raise FileNotFoundError(f"No FFI files matching {patterns!r} under {ffi_leaf!r}")

    ffi_list_path = scc_ffi_list_parquet(resolved.data_root, t.sector, t.camera, t.ccd)
    ffi_list_df = load_ffi_list(ffi_list_path)
    if not ffi_list_is_complete(all_sorted, ffi_list_df):
        log.info("FFI list missing/incomplete (%s); backfilling ...", ffi_list_path)
        t0 = time.monotonic()
        ffi_list_df = ensure_scc_ffi_list(
            resolved.data_root,
            t.sector,
            t.camera,
            t.ccd,
            all_sorted,
            open_fits=wcs_grouping.open_fits_memmap,
        )
        log.info("FFI list ensure finished in %.1fs", time.monotonic() - t0)

    ffi_paths = wcs_grouping.select_ffis_with_valid_target_wcs_from_cache(
        ffi_list_df,
        all_sorted,
        t.target_ra,
        t.target_dec,
        max_ffis=max_ffis,
    )
    log.info("FFIs on disk: %d; processing: %d", len(all_sorted), len(ffi_paths))

    t0 = time.monotonic()
    wcs_table = wcs_grouping.build_wcs_table_from_cache(
        ffi_list_df, ffi_paths, t.target_ra, t.target_dec
    )
    log.info("WCS table from ffi_list built in %.2fs", time.monotonic() - t0)
    wcs_table = wcs_grouping.smooth_wcs_drift_savgol(
        wcs_table,
        window_length=wg.wcs_drift_savgol_window,
        polyorder=wg.wcs_drift_savgol_polyorder,
    )
    if wg.screen_earth_moon_angles:
        bkg_path = _norm_bkg_vector_path(wg.bkg_vector_path)
        if bkg_path:
            wcs_table = wcs_grouping.attach_tessvector_earth_moon_angles(
                wcs_table,
                sector=t.sector,
                camera=t.camera,
                tessvectors_data_path=bkg_path,
            )
        else:
            log.warning(
                "screen_earth_moon_angles enabled but bkg_vector_path unset; "
                "skipping TESSVectors attach"
            )
    wcs_table, chosen_ref = wcs_grouping.finalize_wcs_table_with_reference_anchor(
        wcs_table,
        offset_threshold=wg.offset_threshold,
        ref_ffi_path=ref_ffi_path,
        ref_earth_deg_min=float(wg.earth_deg_min),
        ref_moon_deg_min=float(wg.moon_deg_min),
        screen_earth_moon_angles=bool(wg.screen_earth_moon_angles),
    )
    log.info("Reference FFI: %s", chosen_ref)

    manifest_path = os.path.join(event_dir, FRAMES_CSV_BASENAME)
    wcs_table.to_csv(manifest_path, index=False)

    logical = manifest_basename_from_local(chosen_ref)
    if logical not in ffi_list_df.index:
        raise KeyError(f"chosen reference FFI {logical!r} missing from ffi_list")
    ref_header = header_from_cached_row(ffi_list_df.loc[logical])
    crop_bounds = wcs_grouping.resolve_crop_bounds_from_params(
        ref_header,
        x_min=x_min if x_min is not None else wg.x_min,
        x_max=x_max if x_max is not None else wg.x_max,
        y_min=y_min if y_min is not None else wg.y_min,
        y_max=y_max if y_max is not None else wg.y_max,
        crop_mode=wg.crop_mode,
        crop_box_size=wg.crop_box_size,
        target_ra=t.target_ra,
        target_dec=t.target_dec,
        x_left_dead=wg.x_left_dead,
        x_right_dead=wg.x_right_dead,
        y_edge_strip=wg.y_edge_strip,
    )

    summary_df = wcs_grouping.summarize_template_groups(wcs_table)
    out_path = wcs_grouping.write_cluster_template_job_json(
        summary_df,
        chosen_ref,
        t.sector,
        t.camera,
        t.ccd,
        wg.offset_threshold,
        event_dir,
        crop_bounds=crop_bounds,
        crop_mode=wg.crop_mode,
        crop_box_size=wg.crop_box_size if wg.crop_mode == "target_box" else None,
        geometry_mode=wg.geometry_mode if wg.geometry_mode == "field" else None,
        grouping_quantum_ps1_px=(
            wg.grouping_quantum_ps1_px if wg.geometry_mode == "field" else None
        ),
    )
    wcs_grouping.plot_wcs_drift_and_template_assignment(
        wcs_table,
        os.path.join(
            pipeline_plots_root(event_dir),
            wcs_grouping.WCS_DRIFT_LINEAR_TEMPLATE_FILENAME,
        ),
        ref_ffi_path=chosen_ref,
        sector=t.sector,
        camera=t.camera,
        ccd=t.ccd,
        target_name=t.target_name,
        include_earth_moon_panel=bool(wg.screen_earth_moon_angles),
    )
    return out_path
