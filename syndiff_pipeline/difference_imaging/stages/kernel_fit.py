"""
Target-level kernel fit on the min-background FFI (HP1 → photutils → HP2).

Extracts ``kernel_solution`` from Hotpants round 2 (``hp_bgo=0``).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
from astropy.io import fits

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    HotpantsParams,
    KernelFitParams,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    _load_ffi_cropped,
    _load_template_cropped,
    _write_image_fits,
    build_hotpants_config,
    run_hotpants_frame,
)
from syndiff_pipeline.difference_imaging.stages.kernel import (
    KERNEL_FIT_META_BASENAME,
    KERNEL_R2_NPZ_BASENAME,
    build_kernel_basis,
    kernel_arrays_to_npz_dict,
    kernel_from_hotpants_result,
)
from syndiff_pipeline.difference_imaging.stages.kernel_photutils import (
    photutils_background_masked,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    workspace_frame_fits_path,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)
from syndiff_pipeline.difference_imaging.support.flux_calibration import (
    kernel_sum_at_center,
)
from syndiff_pipeline.difference_imaging.support.min_background import (
    pick_best_angle_ffi,
)
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    resolve_template_for_ffi,
)

log = logging.getLogger(__name__)


@dataclass
class KernelFitResult:
    min_bg_ffi_path: str
    product_id: str
    angle_score: float
    group_dx: float
    group_dy: float
    template_path: str
    kernel_npz_path: str
    meta_path: str
    kernel_solution: np.ndarray
    kernel_image: np.ndarray
    hp_config: Any


def kernel_fit_meta_path(output_dir: str) -> str:
    return os.path.join(output_dir, KERNEL_FIT_META_BASENAME)


def kernel_r2_npz_path(output_dir: str) -> str:
    return os.path.join(output_dir, KERNEL_R2_NPZ_BASENAME)


def load_kernel_fit_meta(output_dir: str) -> dict:
    path = kernel_fit_meta_path(output_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing kernel fit metadata: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _run_hotpants_round(
    *,
    sci: np.ndarray,
    err: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    ref_stars_xy: np.ndarray,
    hp: HotpantsParams,
    work_dir: str,
    frame_stem: str,
    collect_kernel_params: bool = True,
) -> tuple[dict, Any]:
    os.makedirs(work_dir, exist_ok=True)
    hp_config = build_hotpants_config(
        hp=hp,
        diff_dir=work_dir,
        convolved_dir=work_dir,
        frame_stem=frame_stem,
        write_stamps=False,
    )
    result = run_hotpants_frame(
        sci,
        err,
        template,
        mask,
        ref_stars_xy,
        hp_config,
        collect_kernel_params=collect_kernel_params,
    )
    return result, hp_config


def run_kernel_fit(
    *,
    output_dir: str,
    manifest,
    crop_bounds: dict,
    shared_mask: np.ndarray,
    ref_stars_xy: np.ndarray,
    hp: HotpantsParams,
    params: KernelFitParams,
    artifact_dir: Optional[str] = None,
    debug_ws_dir: Optional[str] = None,
    skip_existing: bool = True,
) -> KernelFitResult:
    """
    Fit PSF kernel on angle-ranked min-background FFI through HP1 + phot + HP2.

    Kernel NPZ and metadata are written under *artifact_dir* (typically
    ``ws/kernel_fit/``), not the event root.
    """
    meta_root = artifact_dir or debug_ws_dir or output_dir
    os.makedirs(meta_root, exist_ok=True)
    meta_path = kernel_fit_meta_path(meta_root)
    npz_path = kernel_r2_npz_path(meta_root)

    if skip_existing and os.path.isfile(meta_path) and os.path.isfile(npz_path):
        log.info("Using cached kernel fit artifacts in %s", meta_root)
        meta = load_kernel_fit_meta(meta_root)
        data = dict(np.load(npz_path, allow_pickle=False))
        ks = data["kernel_solution"]
        return KernelFitResult(
            min_bg_ffi_path=meta["min_bg_ffi_path"],
            product_id=meta["product_id"],
            angle_score=float(meta["angle_score"]),
            group_dx=float(meta["group_dx"]),
            group_dy=float(meta["group_dy"]),
            template_path=meta["template_path"],
            kernel_npz_path=npz_path,
            meta_path=meta_path,
            kernel_solution=np.asarray(ks, dtype=np.float64).ravel(),
            kernel_image=np.asarray(data["kernel_image"], dtype=np.float64),
            hp_config=None,
        )

    min_bg_path, angle_score = pick_best_angle_ffi(
        manifest, weighting_factor=params.weighting_factor
    )
    product_id = tess_product_id_from_ffi_path(min_bg_path) or "unknown"
    group_dx, group_dy, template_path = resolve_template_for_ffi(
        output_dir, manifest, min_bg_path
    )

    log.info(
        "Kernel fit on min-background FFI %s (score=%.4f) template dx=%.3f dy=%.3f",
        product_id,
        angle_score,
        group_dx,
        group_dy,
    )

    ffi, err = _load_ffi_cropped(min_bg_path, crop_bounds)
    template = _load_template_cropped(template_path, crop_bounds)
    header = wcs_grouping.crop_ffi_header(min_bg_path, crop_bounds)

    if ffi.shape != shared_mask.shape:
        raise ValueError(
            f"FFI shape {ffi.shape} != shared_mask shape {shared_mask.shape}"
        )

    basis = build_kernel_basis(hp)
    with tempfile.TemporaryDirectory(prefix="kernel_fit_") as work_root:
        hp1, _ = _run_hotpants_round(
            sci=ffi,
            err=err,
            template=template,
            mask=shared_mask,
            ref_stars_xy=ref_stars_xy,
            hp=hp,
            work_dir=os.path.join(work_root, "hp1"),
            frame_stem=f"{product_id}_hp1",
            collect_kernel_params=False,
        )
        if not hp1.get("success"):
            raise RuntimeError(
                f"Kernel-fit Hotpants round 1 failed: {hp1.get('error_msg', '')}"
            )

        phot_bkg_hp1 = photutils_background_masked(
            hp1["diff"], shared_mask, box_size=params.phot_box_size
        )
        hp1_bkg = hp1["bkg"] if hp1.get("bkg") is not None else 0.0
        sci_clean = ffi - hp1_bkg - phot_bkg_hp1

        hp2_params = replace(hp, hp_bgo=0)
        hp2, hp2_config = _run_hotpants_round(
            sci=sci_clean,
            err=err,
            template=template,
            mask=shared_mask,
            ref_stars_xy=ref_stars_xy,
            hp=hp2_params,
            work_dir=os.path.join(work_root, "hp2"),
            frame_stem=f"{product_id}_hp2",
            collect_kernel_params=params.write_kernel_params,
        )
        if not hp2.get("success"):
            raise RuntimeError(
                f"Kernel-fit Hotpants round 2 failed: {hp2.get('error_msg', '')}"
            )

        kernel_params = hp2.get("kernel_params_arrays")
        kernel_image = kernel_from_hotpants_result(
            kernel_params, hp2_config, ffi.shape
        )
        if kernel_image is None or kernel_params is None:
            raise RuntimeError("HP2 did not return kernel_solution")
        kernel_solution = np.asarray(
            kernel_params["kernel_solution"], dtype=np.float64
        ).ravel()

    reference_kernel_sum = kernel_sum_at_center(
        kernel_solution, hp2_config, ffi.shape
    )

    np.savez(
        npz_path,
        **kernel_arrays_to_npz_dict(
            kernel_image, kernel_params, basis, hp2_params
        ),
    )

    meta = {
        "min_bg_ffi_path": os.path.abspath(min_bg_path),
        "product_id": product_id,
        "angle_score": float(angle_score),
        "group_dx": float(group_dx),
        "group_dy": float(group_dy),
        "template_path": os.path.abspath(template_path),
        "kernel_npz_path": os.path.abspath(npz_path),
        "weighting_factor": float(params.weighting_factor),
        "phot_box_size": int(params.phot_box_size),
        "reference_kernel_sum": float(reference_kernel_sum),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if debug_ws_dir and params.write_debug_fits:
        os.makedirs(debug_ws_dir, exist_ok=True)
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "ffi"), ffi, header=header
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "template"),
            template,
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp1_diff"),
            hp1["diff"],
            header=header,
        )
        if hp1.get("bkg") is not None:
            _write_image_fits(
                workspace_frame_fits_path(debug_ws_dir, "hp1_bkg"),
                hp1["bkg"],
                header=header,
            )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "phot_bkg_fine_on_hp1_diff"),
            phot_bkg_hp1,
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "sci1_clean"),
            sci_clean,
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp2_diff"),
            hp2["diff"],
            header=header,
        )
        if hp2.get("bkg") is not None:
            _write_image_fits(
                workspace_frame_fits_path(debug_ws_dir, "hp2_bkg"),
                hp2["bkg"],
                header=header,
            )

    log.info("Wrote kernel fit: %s", npz_path)
    return KernelFitResult(
        min_bg_ffi_path=min_bg_path,
        product_id=product_id,
        angle_score=float(angle_score),
        group_dx=group_dx,
        group_dy=group_dy,
        template_path=template_path,
        kernel_npz_path=npz_path,
        meta_path=meta_path,
        kernel_solution=kernel_solution,
        kernel_image=kernel_image,
        hp_config=hp2_config,
    )
