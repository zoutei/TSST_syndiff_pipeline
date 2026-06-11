"""Standalone WCS grouping handoff for the template pipeline."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Optional

from astropy.io import fits

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.download import list_local_ffis, nested_ffi_dir, _ffi_filename_pattern
from syndiff_pipeline.difference_imaging.support.paths import pipeline_plots_root
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import WcsGroupingStageParams

log = logging.getLogger(__name__)


def _norm_bkg_vector_path(p: Optional[str]) -> Optional[str]:
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

    ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
    all_sorted = sorted(list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd))
    if not all_sorted:
        glob_pat = os.path.join(ffi_leaf, _ffi_filename_pattern(t.sector, t.camera, t.ccd))
        raise FileNotFoundError(f"No FFI files matching {glob_pat!r}")

    ffi_paths = wcs_grouping.select_ffis_with_valid_target_wcs(
        all_sorted, t.target_ra, t.target_dec, max_ffis=max_ffis
    )
    log.info("FFIs on disk: %d; processing: %d", len(all_sorted), len(ffi_paths))

    wcs_table = wcs_grouping.build_wcs_table(ffi_paths, t.target_ra, t.target_dec)
    wcs_table = wcs_grouping.smooth_wcs_drift_savgol(
        wcs_table,
        window_length=wg.wcs_drift_savgol_window,
        polyorder=wg.wcs_drift_savgol_polyorder,
    )
    wcs_table = wcs_grouping.attach_tessvector_earth_moon_angles(
        wcs_table,
        sector=t.sector,
        camera=t.camera,
        tessvectors_data_path=_norm_bkg_vector_path(wg.bkg_vector_path),
    )
    wcs_table, chosen_ref = wcs_grouping.finalize_wcs_table_with_reference_anchor(
        wcs_table,
        offset_threshold=wg.offset_threshold,
        ref_ffi_path=ref_ffi_path,
    )
    log.info("Reference FFI: %s", chosen_ref)

    manifest_path = os.path.join(event_dir, "syndiff_ffi_frames.csv")
    wcs_table.to_csv(manifest_path, index=False)

    with fits.open(chosen_ref, memmap=True) as hdul:
        ref_header = hdul[1].header
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
    )
    wcs_grouping.plot_wcs_drift_and_template_assignment(
        wcs_table,
        os.path.join(
            pipeline_plots_root(event_dir),
            wcs_grouping.WCS_DRIFT_TEMPLATE_DEBUG_FILENAME,
        ),
        ref_ffi_path=chosen_ref,
        sector=t.sector,
        camera=t.camera,
        ccd=t.ccd,
        target_name=t.target_name,
    )
    return out_path
