"""Stage registry and in-process stage execution."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
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

        # Mapping owns ref selection + run_meta.json only; the linear-template
        # debug PNG is written by point-drift owners (linear downsample / remap
        # drift_source: point).
        ref_ffi = resolve_scc_reference_ffi(
            resolved, force_rerun=force_rerun, write_debug_plot=False
        )
        mp = resolved.stages.mapping
        temporal_store = None
        temporal_fingerprint = None
        tess_wcs_override = None
        if str(mp.wcs_source) == "temporal_wcs":
            from syndiff_pipeline.common.download import manifest_basename_from_local
            from syndiff_pipeline.common.scc_paths import scc_wcs_dir
            from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import (
                TemporalChebWcsStore,
            )

            temporal_dir = scc_wcs_dir(
                resolved.data_root, t.sector, t.camera, t.ccd,
                version=mp.temporal_wcs_version,
            )
            temporal_store = TemporalChebWcsStore(temporal_dir)
            # ``for_stem`` returns the production full-FFI adapter.  Keep the
            # adapter at this boundary: create_coords_for_grid() emits full
            # detector pixels and must never be paired with the crop-local raw
            # Chebyshev model.
            tess_wcs_override, ref_btjd = temporal_store.for_stem(
                manifest_basename_from_local(ref_ffi)
            )
            temporal_fingerprint = temporal_store.fingerprint

        # Gaia catalog download and the TESS<->PS1 skycell geometric mapping
        # are independent: process_tess_image_optimized() takes no Gaia
        # input, and the Gaia step is a slow (multi-minute) async TAP query
        # that was previously blocking pancakes for no reason other than
        # sharing this function. Run them concurrently instead. Different
        # output directories (gaia_catalog_dir vs resolved.mapping_root), no
        # shared state -- safe to overlap.
        gaia_thread = None
        gaia_exc: list[BaseException] = []
        from syndiff_pipeline.common.scc_paths import scc_catalogs_dir

        gaia_catalog_dir = scc_catalogs_dir(
            resolved.data_root, t.sector, t.camera, t.ccd
        )
        gaia_catalog_path = gaia_catalog_dir / (
            f"gaia_catalog_s{int(t.sector):04d}_{int(t.camera)}_{int(t.ccd)}.csv"
        )
        # The downloader itself also checks this path, but avoid starting a
        # helper thread (and avoid reporting ``gaia_download`` progress) when
        # the catalog is already complete. ``skip_download_catalog`` remains
        # an explicit escape hatch for operators who intentionally want no
        # catalog check at all.
        if (
            not mp.skip_download_catalog
            and not pancakes.gaia_catalog_cache_is_current(str(gaia_catalog_path))
        ):

            gaia_catalog_dir_str = str(gaia_catalog_dir)
            log.info(
                "Downloading Gaia catalog for %s → %s (concurrently with mapping)",
                ref_ffi, gaia_catalog_dir_str,
            )

            def _run_gaia_download() -> None:
                try:
                    _download_gaia_catalog(
                        site_config_path=resolved.config_path or None,
                        tess_file=ref_ffi,
                        output_path=gaia_catalog_dir_str,
                        force_download=force_rerun,
                    )
                except BaseException as exc:  # noqa: BLE001 -- re-raised on join below
                    gaia_exc.append(exc)

            gaia_thread = threading.Thread(
                target=_run_gaia_download, name="gaia-catalog-download", daemon=True,
            )
            gaia_thread.start()

        try:
            mapping_result = pancakes.process_tess_image_optimized(
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
                mapgrid_version=int(getattr(mp, "mapgrid_version", 3)),
                tess_wcs_override=tess_wcs_override,
            )
        finally:
            if gaia_thread is not None:
                gaia_thread.join()
                if gaia_exc:
                    raise gaia_exc[0]
        try:
            from syndiff_pipeline.template_creation.processing.mapping_plots import (
                write_mapping_projection_overlay_for_scc,
            )

            write_mapping_projection_overlay_for_scc(
                resolved, force_rerun=force_rerun
            )
        except Exception as exc:
            log.warning("Mapping projection overlay skipped: %s", exc)
        if temporal_store is not None:
            from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
                mapping_run_meta_path,
                _read_run_meta,
                _write_run_meta,
            )

            meta_path = mapping_run_meta_path(resolved)
            meta = _read_run_meta(meta_path) or {}
            meta.update({
                "wcs_source": "temporal_wcs",
                "temporal_wcs_version": mp.temporal_wcs_version,
                "temporal_wcs_fingerprint": temporal_fingerprint,
                "temporal_wcs_frame_contract_fingerprint": temporal_store.frame_contract[
                    "fingerprint"
                ],
            })
            # Mapping geometry and temporal support policy are one immutable
            # handoff for downstream PS1/remap/L5 consumers.
            if isinstance(mapping_result, dict):
                if mapping_result.get("mapping_grid") is not None:
                    meta["mapping_grid"] = mapping_result["mapping_grid"]
                if mapping_result.get("template_support_extrapolation") is not None:
                    meta["template_support_extrapolation"] = mapping_result[
                        "template_support_extrapolation"
                    ]
            _write_run_meta(meta_path, meta)
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
        # A completed external per-SCC convolved store can be supplied
        # directly to downsample.  Keep the stage visible in the DAG for
        # dependency accounting, but do not rerun PS1 processing or touch the
        # shared convolved store in that mode.
        external_convolved = getattr(resolved.stages.downsample, "convolved_dir", None)
        if external_convolved:
            path = Path(str(external_convolved))
            if not path.is_dir():
                raise RuntimeError(
                    f"Configured external convolved store does not exist: {path}"
                )
            n_data = sum(1 for entry in path.iterdir() if entry.name.endswith("_data"))
            if n_data <= 0:
                raise RuntimeError(f"Configured external convolved store is empty: {path}")
            log.info(
                "ps1_process: using completed external per-SCC convolved store "
                "%s (%d data arrays); no PS1 processing scheduled",
                path,
                n_data,
            )
            return None
        from syndiff_pipeline.template_creation.processing import ps1_process
        from syndiff_pipeline.template_creation.orchestration.verify import (
            _mapping_csv_path,
            clear_ps1_process_artifacts,
        )

        if force_rerun:
            clear_ps1_process_artifacts(resolved)
        pp = resolved.stages.ps1_process
        pd = resolved.stages.ps1_download
        mp = resolved.stages.mapping
        use_shared_convolved_store = bool(pp.use_shared_convolved_store)
        write_per_scc_convolved_zarr = (
            False if use_shared_convolved_store else bool(pp.write_per_scc_convolved_zarr)
        )
        # Must resolve the exact same mapping master skycells CSV that
        # mapping/remap/downsample use (respecting oversampling_factor and
        # store_name, e.g. OS4/tvwcs field geometry) -- passing this through
        # explicitly instead of letting run_modern_sliding_window_pipeline
        # fall back to the native OS1 CSV.
        mapping_csv_path = str(_mapping_csv_path(resolved))
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
            oversampling_factor=mp.oversampling_factor,
            mapping_csv_path=mapping_csv_path,
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
            resolve_cached_or_select_reference_ffi,
            resolve_scc_reference_ffi,
        )

        rm = resolved.stages.remap
        wg = resolved.stages.wcs_grouping
        mp = resolved.stages.mapping
        master_path = mapping_master_pixels2skycells_path(resolved)
        with __import__("astropy.io.fits", fromlist=["open"]).open(master_path) as hdul:
            master = hdul[1].data
            full_shape = (int(master.shape[0]), int(master.shape[1]))
        target_drift = None
        if str(rm.drift_source) in ("point", "point_ffi_wcs"):
            from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
                resolve_scc_point_drift_table,
                write_scc_wcs_drift_debug_plot,
            )

            # Do not reselect mapping ref on remap --force-rerun.
            ref_ffi = resolve_cached_or_select_reference_ffi(resolved)
            _wcs_table, target_drift = resolve_scc_point_drift_table(
                resolved,
                ref_ffi_path=ref_ffi,
                store_root=resolved.remap_output_base or "",
                force_rerun=force_rerun,
            )
            write_scc_wcs_drift_debug_plot(
                resolved,
                ref_ffi,
                wcs_table=_wcs_table,
                force_rerun=force_rerun,
            )
        else:
            ref_ffi = resolve_scc_reference_ffi(
                resolved, force_rerun=force_rerun, write_debug_plot=False
            )
        temporal_wcs_dir = None
        if str(rm.drift_source) == "per_skycell_temporal_wcs":
            from syndiff_pipeline.common.scc_paths import scc_wcs_dir

            temporal_wcs_dir = scc_wcs_dir(
                resolved.data_root, t.sector, t.camera, t.ccd,
                version=mp.temporal_wcs_version,
            )
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
            drift_source=str(rm.drift_source),
            target_drift=target_drift,
            temporal_wcs_dir=temporal_wcs_dir,
            apply_intra_skycell=bool(rm.apply_intra_skycell),
            apply_inter_skycell=bool(rm.apply_inter_skycell),
        )
        return _manifest_from_result(field_result)

    if stage == "downsample":
        from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
        from syndiff_pipeline.common.scc_paths import scc_remap_dir
        from syndiff_pipeline.template_creation.orchestration.verify import (
            mapping_master_pixels2skycells_path,
            resolve_downsample_convolved_dir,
        )
        from syndiff_pipeline.template_creation.processing.combined_store import (
            production_combined_recipe,
        )
        from syndiff_pipeline.template_creation.processing.field_downsample import (
            run_field_downsample_scc,
        )
        from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
            resolve_cached_or_select_reference_ffi,
            resolve_scc_reference_ffi,
        )

        ds = resolved.stages.downsample
        wg = resolved.stages.wcs_grouping
        mp = resolved.stages.mapping
        geometry_mode = str(ds.geometry_mode or wg.geometry_mode or "field").lower()
        if geometry_mode not in ("field", "linear"):
            raise NotImplementedError(
                f"v2 template downsample supports geometry_mode='field' or "
                f"'linear', got {geometry_mode!r}."
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

        if geometry_mode == "linear":
            # Never reselect mapping ref on downsample --force-rerun; point-drift
            # rebuild uses its own force_rerun keyed on the cached ref path.
            ref_ffi = resolve_cached_or_select_reference_ffi(resolved)
        else:
            ref_ffi = resolve_scc_reference_ffi(
                resolved, force_rerun=force_rerun, write_debug_plot=False
            )

        if geometry_mode == "linear":
            from syndiff_pipeline.template_creation.processing.linear_downsample import (
                run_linear_downsample_scc,
            )

            linear_result = run_linear_downsample_scc(
                resolved,
                convolved_dir=convolved,
                mapping_root=ds.mapping_dir or resolved.mapping_root,
                mapping_grid=mapping_grid,
                base_tess_shape=base_shape,
                roi_bounds=tuple(
                    int(v)
                    for v in (
                        mapping_grid.ffi_xmin,
                        mapping_grid.ffi_ymin,
                        mapping_grid.ffi_xmax,
                        mapping_grid.ffi_ymax,
                    )
                ),
                store_root=ds.output_base or resolved.template_output_base,
                ref_ffi_path=ref_ffi,
                oversampling_factor=ds.oversampling_factor,
                ignore_mask_bits=list(ds.ignore_mask_bits),
                n_jobs=ds.n_jobs,
                rebuild_field_store=bool(getattr(ds, "rebuild_field_store", False)),
                force_rerun=force_rerun,
                progress_path=progress_path,
            )
            linear_result = dict(linear_result)
            linear_result["template_dir_physical"] = str(linear_result["output_dir"])
            return _manifest_from_result(linear_result)

        # Debug/backfill-only skycell restriction, env-var gated so it can
        # never be left set in a real production config file (see
        # field_downsample.run_field_downsample_scc's only_skycells
        # parameter docstring). Unset in every real deployment; used
        # 2026-08-23 to validate H.1/H.2's patch-cache path on the subset of
        # S20/C3/K3 skycells available under the shared combined-store's
        # current recipe fingerprint, without waiting on the separate
        # cross-sector publishing gap blocking a full-SCC backfill.
        _only_skycells_env = os.environ.get("SYNDIFF_DOWNSAMPLE_ONLY_SKYCELLS")
        only_skycells = (
            {s.strip() for s in _only_skycells_env.split(",") if s.strip()}
            if _only_skycells_env
            else None
        )

        field_result = run_field_downsample_scc(
            sector=t.sector,
            camera=t.camera,
            ccd=t.ccd,
            data_root=resolved.data_root,
            event_dir=resolved.event_dir,
            mapping_root=ds.mapping_dir or resolved.mapping_root,
            convolved_dir=convolved,
            only_skycells=only_skycells,
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
            write_split_contribs=bool(getattr(ds, "write_split_contribs", False)),
            stage_regmaps_to_scratch=ds.stage_regmaps_to_scratch,
            update_frames_csv=False,
            store_root=ds.output_base or resolved.template_output_base,
            remap_store_root=remap_store_root,
            scc_only=True,
            ffi_dir=resolved.ffi_dir,
            ref_ffi_path=ref_ffi,
            progress_path=progress_path,
            mapping_grid=mapping_grid,
            psf_sigma=float(getattr(resolved.stages.ps1_process, "psf_sigma", 40.0)),
            combined_recipe=production_combined_recipe(
                resolved.stages.ps1_process,
                data_root=resolved.data_root,
                sector=t.sector,
                camera=t.camera,
                ccd=t.ccd,
            ),
        )
        field_result = dict(field_result)
        if mp.store_name == "tvwcs":
            # Debug FITS are deliberately a small representative selection;
            # NPZ/contrib artifacts remain the authoritative OS4 templates.
            import json
            from syndiff_pipeline.common.scc_paths import scc_debug_plots_dir
            from syndiff_pipeline.template_creation.processing.field_downsample import (
                materialize_field_fits_for_store,
            )
            from syndiff_pipeline.template_creation.processing.field_remap import load_remap_shifts_df

            shifts = load_remap_shifts_df(remap_store_root)
            groups = sorted(int(g) for g in shifts["group_id"].unique())
            selected = sorted(set([groups[0], groups[len(groups) // 2], groups[-1]])) if groups else []
            provenance = json.loads(
                (Path(field_result["output_dir"]) / "template_manifest.json").read_text()
            ).get("provenance", {})
            debug_fits = scc_debug_plots_dir(
                resolved.data_root, t.sector, t.camera, t.ccd,
                f"templates_tvwcs_os{int(ds.oversampling_factor)}",
            ) / "fits"
            field_result["debug_fits"] = materialize_field_fits_for_store(
                field_result["output_dir"], shifts, sector=t.sector, camera=t.camera, ccd=t.ccd,
                base_tess_shape=base_shape, roi_bounds=tuple(int(v) for v in (
                    mapping_grid.ffi_xmin, mapping_grid.ffi_ymin, mapping_grid.ffi_xmax, mapping_grid.ffi_ymax,
                )), oversampling_factor=int(ds.oversampling_factor), mapping_grid=mapping_grid,
                group_ids=selected, fits_dir=debug_fits,
                provenance=provenance,
            )
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
