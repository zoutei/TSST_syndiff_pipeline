"""Per-FFI algebraic difference with robust TESSreduce background (no Hotpants)."""

from __future__ import annotations

import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from joblib import delayed

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    BackgroundEstimateParams,
)
from syndiff_pipeline.difference_imaging.stages.convolved_templates import (
    lookup_convolved_path,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    _load_ffi_cropped,
    _write_image_fits,
)
from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
    estimate_tessreduce_residual_background,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    ffi_frame_stem_from_path,
    resolve_pipeline_fits_path,
    tess_product_id_from_ffi_path,
    workspace_frame_fits_path,
    workspace_frame_stem,
)
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    resolve_template_for_ffi,
)
from syndiff_pipeline.difference_imaging.stages.kernel_subtract_progress import (
    init_progress_pair,
    progress_path_for_diff_log,
    progress_path_for_diffs_workspace,
    record_frame_progress,
    set_progress_phase_pair,
)
from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm

log = logging.getLogger(__name__)

_KERNEL_SUBTRACT_LOKY: Optional[dict[str, Any]] = None


def _kernel_subtract_loky_initializer(payload: dict[str, Any]) -> None:
    """Kernel subtract loky initializer.
    
    Parameters
    ----------
    payload : dict[str, Any]"""
    global _KERNEL_SUBTRACT_LOKY
    _KERNEL_SUBTRACT_LOKY = payload


def _load_convolved_crop(path: str, crop_bounds: dict) -> np.ndarray:
    """Load convolved crop.
    
    Parameters
    ----------
    path : str
    crop_bounds : dict
    
    Returns
    -------
    np.ndarray"""
    from syndiff_pipeline.common.fits_io import image_hdu_data

    ox = int(crop_bounds["x_min"])
    oy = int(crop_bounds["y_min"])
    x1 = int(crop_bounds["x_max"])
    y1 = int(crop_bounds["y_max"])
    with wcs_grouping.open_fits_memmap(path) as hdul:
        data = image_hdu_data(hdul)
        if data.shape == tuple(crop_bounds["shape"]):
            return np.asarray(data, dtype=np.float64)
        return data[oy:y1, ox:x1].astype(np.float64)


def _process_one_frame(task: tuple) -> dict:
    """Process one frame.
    
    Parameters
    ----------
    task : tuple
    
    Returns
    -------
    dict"""
    global _KERNEL_SUBTRACT_LOKY
    if _KERNEL_SUBTRACT_LOKY is None:
        return {
            "success": False,
            "error_msg": "kernel_subtract worker not initialized",
            "product_id": "",
        }

    ffi_path = task[0]
    p = _KERNEL_SUBTRACT_LOKY
    crop_bounds = p["crop_bounds"]
    shared_mask = p["shared_mask"]
    mask_catalog = p.get("mask_catalog")
    btjd_by_product_id = p.get("btjd_by_product_id") or {}
    convolved_table = p["convolved_table"]
    diffs_dir = p["diffs_dir"]
    bkg_dir = p.get("bkg_dir")
    diffs_label = p["diffs_label"]
    bkg_label = p.get("bkg_label")
    output_dir = p["output_dir"]
    template_dir = p.get("template_dir")
    manifest = p["manifest"]
    sck = p.get("sck")
    data_root = p.get("data_root")
    workspace_root = p.get("workspace_root")
    downsample_fp = p.get("downsample_fp")
    cfg = p.get("cfg")
    force_rerun = bool(p.get("force_rerun", False))

    product_id = tess_product_id_from_ffi_path(ffi_path) or "unknown"
    try:
        ffi_stem = ffi_frame_stem_from_path(ffi_path)
    except ValueError:
        ffi_stem = product_id
    diff_stem = workspace_frame_stem(ffi_stem, diffs_label)
    ws_diff_out = workspace_frame_fits_path(diffs_dir, diff_stem)
    output_store_name = p.get("output_store_name")
    ks_params = BackgroundEstimateParams(
        tessreduce_smooth_gauss=float(p.get("tessreduce_smooth_gauss", 2.0)),
        tessreduce_anomaly_gauss=float(p.get("tessreduce_anomaly_gauss", 2.0)),
        tessreduce_qe_spline_degree=int(p.get("tessreduce_qe_spline_degree", 2)),
        tessreduce_qe_spline_smooth_mult=float(p.get("tessreduce_qe_spline_smooth_mult", 10.0)),
        tessreduce_boundary_k=int(p.get("tessreduce_boundary_k", 15)),
        tessreduce_boundary_sigma=float(p.get("tessreduce_boundary_sigma", 3.0)),
        tessreduce_boundary_rim_width=int(p.get("tessreduce_boundary_rim_width", 1)),
    )

    write_path: Optional[Path] = None
    if not force_rerun:
        if sck is not None and data_root:
            try:
                from syndiff_pipeline.difference_imaging.orchestration.diff_store import (
                    resolve_diff_write_path,
                )

                write_path = resolve_diff_write_path(
                    data_root=data_root,
                    sck=sck,
                    kind="diff_image",
                    stage_label=diffs_label,
                    ffi_stem=ffi_stem,
                    label=diffs_label,
                    params=ks_params,
                    output_store_name=output_store_name,
                )
                if write_path.is_file():
                    return {
                        "success": True,
                        "product_id": product_id,
                        "stem": diff_stem,
                        "skipped": True,
                        "path": str(write_path),
                        "scc_store_hit": True,
                    }
                if downsample_fp is None and cfg is not None:
                    downsample_fp = provenance_glue.resolve_downsample_fingerprint_from_cfg(
                        cfg
                    )
                prov_complete = provenance_glue.diff_image_complete_in_store(
                    sector=sck[0],
                    camera=sck[1],
                    ccd=sck[2],
                    product_id=product_id,
                    label=diffs_label,
                    params=ks_params,
                    ffi_path=ffi_path,
                    downsample_fp=downsample_fp,
                    data_root=data_root,
                    cfg=cfg,
                )
            except Exception:
                log.debug(
                    "provenance resume check failed for %s", product_id, exc_info=True
                )
                prov_complete = None
            if prov_complete is True:
                hit_path: Optional[str] = None
                if write_path is not None and write_path.is_file():
                    hit_path = str(write_path)
                else:
                    existing_after_prov = resolve_pipeline_fits_path(diffs_dir, diff_stem)
                    if existing_after_prov is not None:
                        hit_path = existing_after_prov
                if hit_path is not None:
                    return {
                        "success": True,
                        "product_id": product_id,
                        "stem": diff_stem,
                        "skipped": True,
                        "path": hit_path,
                        "provenance_hit": True,
                        "scc_store_hit": True,
                    }
                # Indexed complete but no file — fall through to process.

        if resolve_pipeline_fits_path(diffs_dir, diff_stem) is not None:
            return {
                "success": True,
                "product_id": product_id,
                "stem": diff_stem,
                "skipped": True,
            }

    try:
        template_path = None
        if bool(p.get("field_mode", False)):
            from syndiff_pipeline.difference_imaging.stages.convolved_templates import (
                lookup_convolved_path_by_group_id,
            )
            from syndiff_pipeline.difference_imaging.support.template_resolution import (
                group_id_for_ffi,
            )

            conv_path = lookup_convolved_path_by_group_id(
                convolved_table, group_id_for_ffi(manifest, ffi_path)
            )
        else:
            group_dx, group_dy, template_path = resolve_template_for_ffi(
                output_dir,
                manifest,
                ffi_path,
                template_dir=template_dir,
            )
            conv_path = lookup_convolved_path(convolved_table, group_dx, group_dy)
        ffi, _ = _load_ffi_cropped(ffi_path, crop_bounds)
        convolved = _load_convolved_crop(conv_path, crop_bounds)
        expected = tuple(crop_bounds.get("shape", ffi.shape))
        if ffi.shape != convolved.shape:
            raise ValueError(
                f"FFI shape {ffi.shape} != convolved template {convolved.shape}"
            )
        if ffi.shape != expected:
            raise ValueError(
                f"FFI shape {ffi.shape} != science grid {expected} from crop_bounds"
            )

        frame_mask_raw = (
            mask_catalog.mask_at(btjd_by_product_id.get(product_id), which="full")
            if mask_catalog is not None
            else shared_mask
        )

        diff_raw = ffi - convolved
        tessreduce_bkg, _, _ = estimate_tessreduce_residual_background(
            diff_raw,
            frame_mask_raw,
            smooth_gauss=float(p.get("tessreduce_smooth_gauss", 2.0)),
            anomaly_gauss=float(p.get("tessreduce_anomaly_gauss", 2.0)),
            qe_spline_degree=int(p.get("tessreduce_qe_spline_degree", 2)),
            qe_spline_smooth_mult=float(p.get("tessreduce_qe_spline_smooth_mult", 10.0)),
            boundary_k=int(p.get("tessreduce_boundary_k", 15)),
            boundary_sigma=float(p.get("tessreduce_boundary_sigma", 3.0)),
            boundary_rim_width=int(p.get("tessreduce_boundary_rim_width", 1)),
        )
        total_bkg = tessreduce_bkg
        diff_final = diff_raw - total_bkg

        header = wcs_grouping.crop_ffi_header(str(ffi_path), crop_bounds)
        from syndiff_pipeline.difference_imaging.orchestration.diff_store import (
            resolve_diff_write_path,
        )

        if data_root and sck is not None:
            write_path = resolve_diff_write_path(
                data_root=data_root,
                sck=sck,
                kind="diff_image",
                stage_label=diffs_label,
                ffi_stem=ffi_stem,
                label=diffs_label,
                params=ks_params,
                output_store_name=output_store_name,
            )
            scc_primary = True
        else:
            raise RuntimeError(
                "SCC-only kernel_subtract requires data_root and sector/camera/ccd"
            )
        _write_image_fits(str(write_path), diff_final, header=header)
        if sck is not None:
            try:
                if downsample_fp is None and cfg is not None:
                    downsample_fp = (
                        provenance_glue.resolve_downsample_fingerprint_from_cfg(cfg)
                    )
                inputs = provenance_glue.diff_image_input_fingerprints(
                    sector=sck[0],
                    camera=sck[1],
                    ccd=sck[2],
                    ffi_path=ffi_path,
                    downsample_fp=downsample_fp,
                    cfg=cfg,
                )
                if inputs is not None:
                    provenance_glue.emit_diff_artifact(
                        kind="diff_image",
                        sector=sck[0],
                        camera=sck[1],
                        ccd=sck[2],
                        product_id=product_id,
                        label=diffs_label,
                        # KernelFitParams is resolved in a separate upstream pipeline
                        # stage and not threaded through this loop; the recorded
                        # recipe covers this stage's own params only (deviation,
                        # see PR-D1 report).
                        params=ks_params,
                        location=str(write_path),
                        input_fingerprints=inputs,
                        data_root=data_root,
                        meta={"producer": "kernel_subtract"},
                        scc_primary=scc_primary,
                        workspace_root=workspace_root,
                        output_store_name=output_store_name,
                    )
            except Exception:
                log.debug(
                    "provenance emit (diff_image/kernel_subtract) failed for %s",
                    product_id,
                    exc_info=True,
                )
        if bkg_dir and bkg_label:
            bkg_stem = workspace_frame_stem(ffi_stem, bkg_label)
            bkg_ws_out = workspace_frame_fits_path(bkg_dir, bkg_stem)
            if data_root and sck is not None:
                bkg_write_path = resolve_diff_write_path(
                    data_root=data_root,
                    sck=sck,
                    kind="diff_background",
                    stage_label=bkg_label,
                    ffi_stem=ffi_stem,
                    label=bkg_label,
                    params=ks_params,
                    output_store_name=output_store_name,
                )
                bkg_scc_primary = True
            else:
                raise RuntimeError(
                    "SCC-only kernel_subtract background write requires data_root and s/c/k"
                )
            _write_image_fits(
                str(bkg_write_path),
                total_bkg,
                header=header,
            )
            if sck is not None:
                try:
                    ffi_fp = provenance_glue.ffi_input_fingerprint(
                        sck[0], sck[1], sck[2], ffi_path
                    )
                    bkg_inputs = provenance_glue.diff_background_input_fingerprints(
                        ffi_fp
                    )
                    if bkg_inputs is not None:
                        provenance_glue.emit_diff_artifact(
                            kind="diff_background",
                            sector=sck[0],
                            camera=sck[1],
                            ccd=sck[2],
                            product_id=product_id,
                            label=bkg_label,
                            params=ks_params,
                            location=str(bkg_write_path),
                            input_fingerprints=bkg_inputs,
                            data_root=data_root,
                            meta={"producer": "kernel_subtract"},
                            scc_primary=bkg_scc_primary,
                            workspace_root=workspace_root,
                            output_store_name=output_store_name,
                        )
                except Exception:
                    log.debug(
                        "provenance emit (diff_background/kernel_subtract) failed for %s",
                        product_id,
                        exc_info=True,
                    )

        return {
            "success": True,
            "product_id": product_id,
            "stem": diff_stem,
            "skipped": False,
        }
    except Exception as exc:
        log.warning("kernel_subtract failed for %s: %s", product_id, exc)
        return {
            "success": False,
            "product_id": product_id,
            "error_msg": str(exc),
        }


def kernel_subtract_loop(
    *,
    ffi_paths: list[str],
    output_dir: str,
    manifest,
    crop_bounds: dict,
    shared_mask: np.ndarray,
    convolved_table,
    diffs_dir: str,
    diffs_label: str,
    tessreduce_smooth_gauss: float = 2.0,
    tessreduce_anomaly_gauss: float = 2.0,
    tessreduce_qe_spline_degree: int = 2,
    tessreduce_qe_spline_smooth_mult: float = 10.0,
    tessreduce_boundary_k: int = 15,
    tessreduce_boundary_sigma: float = 3.0,
    tessreduce_boundary_rim_width: int = 1,
    bkg_dir: Optional[str] = None,
    bkg_label: Optional[str] = None,
    n_jobs: int = 1,
    field_mode: bool = False,
    cfg: Optional[Any] = None,
    mask_catalog=None,
    diff_log_path: Optional[str] = None,
    force_rerun: bool = False,
) -> list[dict]:
    """Run algebraic diff + robust TESSreduce background for each FFI.

    ``field_mode`` keys the convolved-template lookup by ``group_id`` (field
    geometry) instead of the linear ``(group_dx, group_dy)`` offsets.

    ``cfg`` (``SynDiffConfig``), when given, drives best-effort PR-D1 diff
    provenance tracking (sector/camera/ccd + ``data_root``); never changes
    what/where is written. See ``orchestration/provenance_glue.py``.

    ``force_rerun``, when set, bypasses the on-disk/provenance resume checks
    and recomputes every frame (mirrors the hotpants stage's ``force_rerun``).
    """
    os.makedirs(diffs_dir, exist_ok=True)
    if bkg_dir:
        os.makedirs(bkg_dir, exist_ok=True)

    sck = None
    data_root = None
    workspace_root = None
    output_store_name = None
    downsample_fp = None
    if cfg is not None:
        try:
            sck = (int(cfg.sector), int(cfg.camera), int(cfg.ccd))
        except Exception:
            sck = None
        data_root = getattr(cfg, "data_root", "") or None
        output_store_name = getattr(cfg, "output_store_name", None) or None
        from syndiff_pipeline.difference_imaging.support.paths import workspace_root as _workspace_root

        workspace_root = _workspace_root(
            output_dir, run_id=getattr(cfg, "workspace_run_id", None)
        )
        if sck is not None:
            try:
                downsample_fp = provenance_glue.resolve_downsample_fingerprint_from_cfg(
                    cfg
                )
            except Exception:
                log.debug(
                    "downsample fingerprint resolve failed (non-fatal)", exc_info=True
                )

    btjd_by_product_id: dict = {}
    btjd_col = None
    if manifest is not None:
        for c in ("btjd", "BTJD", "tjd", "TJD", "jd", "JD"):
            if c in manifest.columns:
                btjd_col = c
                break
        if btjd_col is not None:
            for _, row in manifest.iterrows():
                pid = tess_product_id_from_ffi_path(
                    str(row.get("ffi_path", row.get("path", "")))
                )
                if pid is None:
                    continue
                try:
                    btjd_by_product_id[pid] = float(row[btjd_col])
                except (TypeError, ValueError):
                    pass

    payload = {
        "crop_bounds": crop_bounds,
        "shared_mask": shared_mask,
        "convolved_table": convolved_table,
        "tessreduce_smooth_gauss": float(tessreduce_smooth_gauss),
        "tessreduce_anomaly_gauss": float(tessreduce_anomaly_gauss),
        "tessreduce_qe_spline_degree": int(tessreduce_qe_spline_degree),
        "tessreduce_qe_spline_smooth_mult": float(tessreduce_qe_spline_smooth_mult),
        "tessreduce_boundary_k": int(tessreduce_boundary_k),
        "tessreduce_boundary_sigma": float(tessreduce_boundary_sigma),
        "tessreduce_boundary_rim_width": int(tessreduce_boundary_rim_width),
        "diffs_dir": diffs_dir,
        "bkg_dir": bkg_dir,
        "diffs_label": diffs_label,
        "bkg_label": bkg_label,
        "output_dir": output_dir,
        "template_dir": (getattr(cfg, "template_dir", "") or None) if cfg is not None else None,
        "manifest": manifest,
        "field_mode": bool(field_mode),
        "sck": sck,
        "data_root": data_root,
        "workspace_root": workspace_root,
        "output_store_name": output_store_name,
        "downsample_fp": downsample_fp,
        "cfg": cfg,
        "mask_catalog": mask_catalog,
        "btjd_by_product_id": btjd_by_product_id,
        "force_rerun": bool(force_rerun),
    }

    tasks = [(ffi_path,) for ffi_path in ffi_paths]

    cli_progress_path = (
        str(progress_path_for_diff_log(diff_log_path)) if diff_log_path is not None else None
    )
    workspace_progress_path = str(progress_path_for_diffs_workspace(diffs_dir))
    init_progress_pair(
        workspace_progress_path,
        cli_progress_path,
        frames_total=len(tasks),
    )

    def _record_progress(result: dict) -> None:
        record_frame_progress(
            workspace_progress_path,
            cli_progress_path,
            success=bool(result.get("success")),
        )

    n_workers = max(1, min(int(n_jobs), len(tasks), multiprocessing.cpu_count()))
    if n_workers == 1:
        _kernel_subtract_loky_initializer(payload)
        results = []
        for t in tasks:
            result = _process_one_frame(t)
            _record_progress(result)
            results.append(result)
    else:
        results = parallel_map_with_optional_tqdm(
            (delayed(_process_one_frame)(t) for t in tasks),
            n_tasks=len(tasks),
            desc="kernel_subtract",
            n_jobs_eff=n_workers,
            initializer=_kernel_subtract_loky_initializer,
            initargs=(payload,),
            on_result=_record_progress,
        )

    set_progress_phase_pair(workspace_progress_path, cli_progress_path, "complete")

    ok = sum(1 for r in results if r.get("success"))
    log.info(
        "kernel_subtract: %d/%d frames succeeded (%d skipped existing)",
        ok,
        len(results),
        sum(1 for r in results if r.get("skipped")),
    )
    return results
