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
    """Pipeline."""
    from syndiff_pipeline.pipeline_spec import get_syndiff_pipeline

    return get_syndiff_pipeline()


def parse_stage_list(stages_arg: str | None) -> List[str]:
    """Parse stage list.
    
    Parameters
    ----------
    stages_arg : str | None
    
    Returns
    -------
    List[str]"""
    return _pipeline().parse_stage_list(stages_arg)


def resolve_stage_name(stage: str) -> str:
    """Resolve stage name.
    
    Parameters
    ----------
    stage : str
    
    Returns
    -------
    str"""
    return _pipeline().resolve_stage_name(stage)


def build_stage_command(
    run_id: str,
    stage: str,
    run_dir: str,
    target_label: str,
    *,
    launch_token: str,
    force_rerun: bool = False,
) -> List[str]:
    """Build stage command.
    
    Parameters
    ----------
    run_id : str
    stage : str
    run_dir : str
    target_label : str
    launch_token : str
    force_rerun : bool, optional, default ``False``
    
    Returns
    -------
    List[str]"""
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
    """Deployment file for site.
    
    Parameters
    ----------
    site_config_path : str
    
    Returns
    -------
    str"""
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
    """Download gaia catalog.
    
    Parameters
    ----------
    site_config_path : str | None
    tess_file : str
    output_path : str
    force_download : bool"""
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
    """Manifest meta from result.
    
    Parameters
    ----------
    result : dict
    
    Returns
    -------
    dict[str, str] | None"""
    meta = {key: str(result[key]) for key in _MANIFEST_META_KEYS if key in result}
    return meta or None


def _manifest_from_result(
    result: dict,
) -> tuple[int, int, list[str], dict[str, str] | None] | None:
    """Manifest from result.
    
    Parameters
    ----------
    result : dict
    
    Returns
    -------
    tuple[int, int, list[str], dict[str, str] | None] | None"""
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
        from syndiff_pipeline.common.download import download_ffis

        out_dir = resolved.ffi_dir
        download_ffis(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            output_dir=out_dir,
            data_root=resolved.data_root,
            update_ffi_list=True,
        )
        return None

    if stage == "mapping":
        from syndiff_pipeline.template_creation.processing import pancakes
        from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
            resolve_scc_reference_ffi,
        )

        ref_ffi = resolve_scc_reference_ffi(resolved, force_rerun=force_rerun)
        mp = resolved.stages.mapping
        if not mp.skip_download_catalog:
            from syndiff_pipeline.common.scc_paths import scc_catalogs_dir

            gaia_catalog_dir = str(
                scc_catalogs_dir(resolved.data_root, t.sector, t.camera, t.ccd)
            )
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
            x_left_dead=mp.x_left_dead,
            x_right_dead=mp.x_right_dead,
            y_edge_strip=mp.y_edge_strip,
            template_conv_pad_spare_px=mp.template_conv_pad_spare_px,
            sci_fwhm=mp.sci_fwhm,
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
        use_shared_convolved_store = bool(pp.use_shared_convolved_store)
        write_per_scc_convolved_zarr = (
            False if use_shared_convolved_store else bool(pp.write_per_scc_convolved_zarr)
        )
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
            use_shared_convolved_store=use_shared_convolved_store,
            write_per_scc_convolved_zarr=write_per_scc_convolved_zarr,
        )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return _manifest_from_result(result)

    if stage == "remap":
        from syndiff_pipeline.template_creation.orchestration.verify import (
            mapping_master_pixels2skycells_path,
        )
        from syndiff_pipeline.template_creation.processing.field_remap import (
            run_field_remap_scc,
        )
        from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
            resolve_scc_reference_ffi,
        )

        rm = resolved.stages.remap
        wg = resolved.stages.wcs_grouping
        mp = resolved.stages.mapping
        master_path = mapping_master_pixels2skycells_path(resolved)
        with __import__("astropy.io.fits", fromlist=["open"]).open(master_path) as hdul:
            master = hdul[1].data
            full_shape = (int(master.shape[0]), int(master.shape[1]))
        ref_ffi = resolve_scc_reference_ffi(resolved, force_rerun=force_rerun)
        field_result = run_field_remap_scc(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            data_root=resolved.data_root,
            event_dir=resolved.event_dir,
            mapping_root=resolved.mapping_root,
            base_tess_shape=full_shape,
            oversampling_factor=mp.oversampling_factor,
            grouping_quantum_ps1_px=float(wg.grouping_quantum_ps1_px or 1.0),
            cache_quantum_ps1_px=float(rm.cache_quantum_ps1_px),
            keying=str(rm.keying),
            intra_skycell_R=int(rm.intra_skycell_R or 1),
            rebuild_remap_cache=bool(rm.rebuild_remap_cache),
            rebuild_inter_skycell_cache=bool(rm.rebuild_inter_skycell_cache),
            store_root=resolved.remap_output_base or None,
            scc_only=True,
            ffi_dir=resolved.ffi_dir,
            ref_ffi_path=ref_ffi,
            n_jobs=rm.n_jobs,
            progress_path=progress_path,
            raw_drift_outlier_sigma=rm.raw_drift_outlier_sigma,
            stage_regmaps_to_scratch=rm.stage_regmaps_to_scratch,
        )
        return _manifest_from_result(field_result)

    if stage == "downsample":
        from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
        from syndiff_pipeline.common.scc_paths import scc_remap_dir
        from syndiff_pipeline.template_creation.orchestration.verify import (
            mapping_master_pixels2skycells_path,
            resolve_downsample_convolved_dir,
        )
        from syndiff_pipeline.template_creation.processing.field_downsample import (
            run_field_downsample_scc,
        )
        from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
            resolve_scc_reference_ffi,
        )

        ds = resolved.stages.downsample
        wg = resolved.stages.wcs_grouping
        mp = resolved.stages.mapping
        geometry_mode = str(ds.geometry_mode or wg.geometry_mode or "field").lower()
        if geometry_mode != "field":
            raise NotImplementedError(
                "v2 template downsample supports geometry_mode='field' only; "
                "linear/event ROI templates were removed."
            )

        convolved = resolve_downsample_convolved_dir(resolved)
        remap_store_root = str(
            scc_remap_dir(
                resolved.data_root,
                t.sector,
                t.camera,
                t.ccd,
                oversampling_factor=mp.oversampling_factor,
                store_name=resolved.downsample_remap_store_name,
            )
        )

        master_path = mapping_master_pixels2skycells_path(resolved)
        mapping_grid = load_mapping_grid_from_master(master_path)
        os_factor = max(1, int(ds.oversampling_factor or 1))
        base_shape = (
            mapping_grid.array_shape_os()
            if os_factor > 1
            else mapping_grid.array_shape_native()
        )

        ref_ffi = resolve_scc_reference_ffi(resolved, force_rerun=force_rerun)
        field_result = run_field_downsample_scc(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            data_root=resolved.data_root,
            event_dir=resolved.event_dir,
            mapping_root=ds.mapping_dir or resolved.mapping_root,
            convolved_dir=convolved,
            roi_bounds=tuple(
                int(v)
                for v in (
                    mapping_grid.ffi_xmin,
                    mapping_grid.ffi_ymin,
                    mapping_grid.ffi_xmax,
                    mapping_grid.ffi_ymax,
                )
            ),
            base_tess_shape=base_shape,
            oversampling_factor=ds.oversampling_factor,
            ignore_mask_bits=list(ds.ignore_mask_bits),
            grouping_quantum_ps1_px=float(wg.grouping_quantum_ps1_px or 1.0),
            materialize_fits=bool(ds.materialize_fits),
            n_jobs=ds.n_jobs,
            rebuild_field_store=bool(getattr(ds, "rebuild_field_store", False)),
            apply_intra_skycell=bool(getattr(ds, "apply_intra_skycell", True)),
            apply_inter_skycell=bool(getattr(ds, "apply_inter_skycell", True)),
            stage_regmaps_to_scratch=ds.stage_regmaps_to_scratch,
            update_frames_csv=False,
            store_root=ds.output_base or resolved.template_output_base,
            remap_store_root=remap_store_root,
            scc_only=True,
            ffi_dir=resolved.ffi_dir,
            ref_ffi_path=ref_ffi,
            progress_path=progress_path,
            mapping_grid=mapping_grid,
        )
        field_result = dict(field_result)
        field_result["template_dir_physical"] = str(field_result["output_dir"])
        return _manifest_from_result(field_result)

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
    """Stage snapshot.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    stage : str
    
    Returns
    -------
    dict"""
    spec = _pipeline().get(stage)
    if spec is not None and spec.stage_snapshot is not None:
        return spec.stage_snapshot(resolved)
    snap = config_snapshot(resolved)
    snap["stage"] = stage
    snap["pool"] = spec.pool if spec is not None else None
    return snap
