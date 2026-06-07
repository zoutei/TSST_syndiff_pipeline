"""Stage registry and in-process stage execution."""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from syndiff_pipeline.download import download_ffis, nested_ffi_dir
from syndiff_pipeline.template import pancakes, ps1_download, ps1_process
from syndiff_pipeline.template.downsample import (
    load_cluster_template_job_payload,
    main as run_downsample,
    offsets_from_cluster_job_payload,
    roi_tuple_from_cluster_job_payload,
)
from syndiff_pipeline.template_runner.handoff import run_wcs_grouping
from syndiff_pipeline.template_runner.runner_config import ResolvedTargetConfig, config_snapshot
from syndiff_pipeline.template_runner.state import STAGE_NAMES, STAGE_POOL

log = logging.getLogger(__name__)


def parse_stage_list(stages_arg: str | None) -> List[str]:
    if not stages_arg or not str(stages_arg).strip():
        return list(STAGE_NAMES)
    names = [s.strip() for s in str(stages_arg).split(",") if s.strip()]
    unknown = set(names) - set(STAGE_NAMES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    return names


def build_stage_command(
    run_id: str,
    stage: str,
    config_path: str,
    targets_path: str,
    target_label: str,
    force_rerun: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.template_runner.run_stage",
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--config",
        str(config_path),
        "--targets",
        str(targets_path),
        "--target-label",
        target_label,
    ]
    if force_rerun:
        cmd.append("--force-rerun")
    return cmd


def execute_stage(
    resolved: ResolvedTargetConfig, stage: str, force_rerun: bool = False
) -> None:
    """Run one template pipeline stage in-process."""
    t = resolved.target
    if stage == "tess_ffi_download":
        out_dir = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
        download_ffis(sector=t.sector, camera=t.camera, ccd=t.ccd, output_dir=out_dir)
        return

    if stage == "wcs_grouping":
        run_wcs_grouping(resolved)
        return

    if stage == "mapping":
        job_path = Path(resolved.handoff_dir) / "cluster_template_job.json"
        with job_path.open(encoding="utf-8") as fh:
            job = json.load(fh)
        ref_ffi = job["reference_ffi_path"]
        mp = resolved.stages.mapping
        if not mp.skip_download_catalog:
            gaia_catalog_dir = os.path.join(resolved.data_root, "catalogs")
            log.info("Downloading Gaia catalog for %s → %s", ref_ffi, gaia_catalog_dir)
            pancakes.download_gaia_catalog_for_tess_file(
                tess_file=ref_ffi,
                output_path=gaia_catalog_dir,
                gaia_credentials_file=resolved.gaia_credentials,
                force_download=force_rerun,
            )
        pancakes.process_tess_image_optimized(
            tess_file=ref_ffi,
            skycell_wcs_csv=resolved.skycell_wcs_csv,
            output_path=resolved.mapping_root,
            pad_distance=mp.pad_distance,
            edge_exclusion=mp.edge_exclusion,
            edge_buffer_large=mp.edge_buffer_large,
            edge_buffer_small=mp.edge_buffer_small,
            buffer=mp.buffer,
            tess_buffer=mp.tess_buffer,
            n_threads=mp.n_threads,
            overwrite=mp.overwrite,
            max_workers=mp.max_workers,
            oversampling_factor=mp.oversampling_factor,
        )
        return

    if stage == "ps1_download":
        pd = resolved.stages.ps1_download
        result = ps1_download.download_and_store_ps1_data(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            num_workers=pd.num_workers,
            zarr_output_dir=resolved.zarr_dir,
            use_local_files=pd.use_local_files,
            local_data_path=pd.local_data_path,
            log_level=pd.log_level,
            overwrite=pd.overwrite,
        )
        if result.get("status") != "completed":
            raise RuntimeError(f"PS1 download failed: {result.get('message', result)}")
        return

    if stage == "ps1_process":
        pp = resolved.stages.ps1_process
        result = ps1_process.run_modern_sliding_window_pipeline(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            data_root=resolved.data_root,
            projections_limit=pp.projections_limit,
            psf_sigma=pp.psf_sigma,
            enable_saturation_correction=pp.enable_saturation_correction,
            remove_saturated_stars=pp.remove_saturated_stars,
            catalog_path=pp.catalog_path,
            bright_star_mag_threshold=pp.bright_star_mag_threshold,
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return

    if stage == "downsample":
        job_path = str(Path(resolved.handoff_dir) / "cluster_template_job.json")
        payload = load_cluster_template_job_payload(job_path)
        ds = resolved.stages.downsample
        if ds.single_offset:
            offsets = np.array([[0.0, 0.0]])
            roi = roi_tuple_from_cluster_job_payload(payload)
        else:
            offsets = offsets_from_cluster_job_payload(payload)
            roi = roi_tuple_from_cluster_job_payload(payload)
        x_min, y_min, x_max, y_max = roi
        run_downsample(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            offsets=offsets,
            ignore_mask_bits=list(ds.ignore_mask_bits),
            data_root=resolved.data_root,
            mapping_dir=ds.mapping_dir or resolved.mapping_root,
            convolved_dir=ds.convolved_dir or str(Path(resolved.data_root) / "convolved_results"),
            output_base=ds.output_base or resolved.template_output_base,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            oversampling_factor=ds.oversampling_factor,
            reference_ffi_basename_expected=payload.get("reference_ffi_basename"),
            cluster_job_json_path=job_path,
        )
        return

    raise ValueError(f"Unknown stage: {stage!r}")


def stage_snapshot(resolved: ResolvedTargetConfig, stage: str) -> dict:
    snap = config_snapshot(resolved)
    snap["stage"] = stage
    snap["pool"] = STAGE_POOL.get(stage)
    return snap
