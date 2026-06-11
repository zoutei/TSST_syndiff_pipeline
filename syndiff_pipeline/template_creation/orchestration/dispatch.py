"""Stage registry and in-process stage execution."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from syndiff_pipeline.template_creation.orchestration.runner_config import (
    ResolvedTargetConfig,
    config_snapshot,
    parse_deployment_file,
)
from syndiff_pipeline.common.orchestration.deployment import (
    deployment_path_for_config,
    gaia_credentials_file,
    load_deployment,
)
log = logging.getLogger(__name__)


def _pipeline():
    from syndiff_pipeline.pipeline_spec import get_syndiff_pipeline

    return get_syndiff_pipeline()


def parse_stage_list(stages_arg: str | None) -> List[str]:
    return _pipeline().parse_stage_list(stages_arg)


def build_stage_command(
    run_id: str,
    stage: str,
    run_dir: str,
    target_label: str,
    *,
    launch_token: str,
    force_rerun: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.common.orchestration.run_stage",
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--run-dir",
        str(run_dir),
        "--target-label",
        target_label,
        "--launch-token",
        launch_token,
    ]
    if force_rerun:
        cmd.append("--force-rerun")
    return cmd


def _deployment_file_for_site(site_config_path: str) -> str:
    import yaml

    with Path(site_config_path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_deployment_file(raw)


def _download_gaia_catalog(
    *,
    site_config_path: str | None,
    tess_file: str,
    output_path: str,
    force_download: bool,
) -> None:
    from syndiff_pipeline.template_creation.processing import pancakes

    if not site_config_path:
        pancakes.download_gaia_catalog_for_tess_file(
            tess_file=tess_file,
            output_path=output_path,
            gaia_credentials_file=None,
            force_download=force_download,
        )
        return
    deployment_file = _deployment_file_for_site(site_config_path)
    deployment = load_deployment(site_config_path, deployment_file)
    deployment_path = deployment_path_for_config(site_config_path, deployment_file)
    with gaia_credentials_file(deployment, deployment_path=deployment_path) as creds_path:
        pancakes.download_gaia_catalog_for_tess_file(
            tess_file=tess_file,
            output_path=output_path,
            gaia_credentials_file=creds_path,
            force_download=force_download,
        )


_MANIFEST_META_KEYS = ("template_dir_physical", "template_dir_symlink")


def _manifest_meta_from_result(result: dict) -> dict[str, str] | None:
    meta = {key: str(result[key]) for key in _MANIFEST_META_KEYS if key in result}
    return meta or None


def _manifest_from_result(
    result: dict,
) -> tuple[int, int, list[str], dict[str, str] | None] | None:
    if not isinstance(result, dict):
        return None
    if "expected_count" not in result or "produced_count" not in result:
        return None
    artifacts = [str(p) for p in (result.get("artifacts") or [])]
    return (
        int(result["expected_count"]),
        int(result["produced_count"]),
        artifacts,
        _manifest_meta_from_result(result),
    )


def _execute_template_stage(
    resolved: ResolvedTargetConfig,
    stage: str,
    force_rerun: bool = False,
    *,
    progress_path: str | None = None,
) -> tuple[int, int, list[str], dict[str, str] | None] | None:
    """Run one template pipeline stage in-process."""
    t = resolved.target
    if stage == "tess_ffi_download":
        from syndiff_pipeline.common.download import download_ffis, nested_ffi_dir

        out_dir = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
        download_ffis(sector=t.sector, camera=t.camera, ccd=t.ccd, output_dir=out_dir)
        return None

    if stage == "wcs_grouping":
        from syndiff_pipeline.template_creation.orchestration.handoff import run_wcs_grouping

        run_wcs_grouping(resolved)
        return None

    if stage == "mapping":
        from syndiff_pipeline.template_creation.processing import pancakes

        job_path = Path(resolved.event_dir) / "cluster_template_job.json"
        with job_path.open(encoding="utf-8") as fh:
            job = json.load(fh)
        ref_ffi = job["reference_ffi_path"]
        mp = resolved.stages.mapping
        if not mp.skip_download_catalog:
            gaia_catalog_dir = os.path.join(resolved.data_root, "catalogs")
            log.info("Downloading Gaia catalog for %s → %s", ref_ffi, gaia_catalog_dir)
            _download_gaia_catalog(
                site_config_path=resolved.config_path or None,
                tess_file=ref_ffi,
                output_path=gaia_catalog_dir,
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
        return None

    if stage == "ps1_download":
        from syndiff_pipeline.template_creation.processing import ps1_download

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
        return _manifest_from_result(result)

    if stage == "ps1_process":
        from syndiff_pipeline.template_creation.processing import ps1_process
        from syndiff_pipeline.template_creation.orchestration.verify import clear_ps1_process_artifacts

        if force_rerun:
            clear_ps1_process_artifacts(resolved)
        pp = resolved.stages.ps1_process
        pd = resolved.stages.ps1_download
        result = ps1_process.run_modern_sliding_window_pipeline(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            data_root=resolved.data_root,
            projections_limit=pp.projections_limit,
            psf_sigma=pp.psf_sigma,
            ps1_source=pp.ps1_source,
            num_ingest_workers=pp.num_ingest_workers,
            use_local_files=pd.use_local_files,
            local_data_path=pd.local_data_path,
            enable_saturation_correction=pp.enable_saturation_correction,
            remove_saturated_stars=pp.remove_saturated_stars,
            catalog_path=pp.catalog_path,
            bright_star_mag_threshold=pp.bright_star_mag_threshold,
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return _manifest_from_result(result)

    if stage == "downsample":
        import numpy as np

        from syndiff_pipeline.template_creation.orchestration.verify import (
            clear_downsample_event_artifacts,
        )
        from syndiff_pipeline.template_creation.processing.downsample import (
            load_cluster_template_job_payload,
            main as run_downsample,
            offsets_from_cluster_job_payload,
            roi_tuple_from_cluster_job_payload,
        )

        os.makedirs(resolved.event_dir, exist_ok=True)
        if force_rerun:
            clear_downsample_event_artifacts(resolved)

        job_path = str(Path(resolved.event_dir) / "cluster_template_job.json")
        payload = load_cluster_template_job_payload(job_path)
        ds = resolved.stages.downsample
        if ds.single_offset:
            offsets = np.array([[0.0, 0.0]])
            roi = roi_tuple_from_cluster_job_payload(payload)
        else:
            offsets = offsets_from_cluster_job_payload(payload)
            roi = roi_tuple_from_cluster_job_payload(payload)
        x_min, y_min, x_max, y_max = roi
        result = run_downsample(
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
            allow_reference_ffi_mismatch=ds.allow_reference_ffi_mismatch,
            progress_path=progress_path,
            n_jobs=ds.n_jobs,
            skycells_per_batch=ds.skycells_per_batch,
            event_dir=resolved.event_dir,
            write_ps1_removed_stars_csv=True,
        )
        from syndiff_pipeline.common.orchestration.template_handoff import (
            ensure_event_templates_symlink,
            event_templates_symlink_path,
        )

        physical_dir = result.get("output_dir")
        if physical_dir:
            symlink_path = ensure_event_templates_symlink(
                resolved.event_dir, physical_dir
            )
            result = dict(result)
            result["template_dir_physical"] = str(physical_dir)
            result["template_dir_symlink"] = str(symlink_path)
        return _manifest_from_result(result)

    raise ValueError(f"Unknown template stage: {stage!r}")


def execute_stage(
    resolved: ResolvedTargetConfig,
    stage: str,
    force_rerun: bool = False,
    *,
    progress_path: str | None = None,
) -> tuple[int, int, list[str], dict[str, str] | None] | None:
    """Run one template pipeline stage in-process via the composed pipeline spec."""
    if stage == "diff":
        raise ValueError("diff stage must run via run_stage diff path")
    return _pipeline().require(stage).execute(
        resolved,
        force_rerun=force_rerun,
        progress_path=progress_path,
    )


def stage_snapshot(resolved: ResolvedTargetConfig, stage: str) -> dict:
    spec = _pipeline().get(stage)
    if spec is not None and spec.stage_snapshot is not None:
        return spec.stage_snapshot(resolved)
    snap = config_snapshot(resolved)
    snap["stage"] = stage
    snap["pool"] = spec.pool if spec is not None else None
    return snap
