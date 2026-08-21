"""
Target-level kernel fit on the min-background FFI: 3-round hotpants +
robust TESSreduce background loop (HP1 -> bkg1 -> HP2 -> bkg2 -> HP3 final).

Extracts ``kernel_solution`` from Hotpants round 3 (``hp_bgo=0``).
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
from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
    estimate_tessreduce_residual_background,
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
    """KernelFitResult."""
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
    """Kernel fit meta path.
    
    Parameters
    ----------
    output_dir : str
    
    Returns
    -------
    str"""
    return os.path.join(output_dir, KERNEL_FIT_META_BASENAME)


def kernel_r2_npz_path(output_dir: str) -> str:
    """Kernel r2 npz path.
    
    Parameters
    ----------
    output_dir : str
    
    Returns
    -------
    str"""
    return os.path.join(output_dir, KERNEL_R2_NPZ_BASENAME)


def load_kernel_fit_meta(output_dir: str) -> dict:
    """Load kernel fit meta.
    
    Parameters
    ----------
    output_dir : str
    
    Returns
    -------
    dict"""
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
    """Run hotpants round.
    
    Parameters
    ----------
    sci : np.ndarray
    err : np.ndarray
    template : np.ndarray
    mask : np.ndarray
    ref_stars_xy : np.ndarray
    hp : HotpantsParams
    work_dir : str
    frame_stem : str
    collect_kernel_params : bool, optional, default ``True``
    
    Returns
    -------
    tuple[dict, Any]"""
    os.makedirs(work_dir, exist_ok=True)
    hp_config = build_hotpants_config(
        hp=hp,
        diff_dir=work_dir,
        convolved_dir=work_dir,
        frame_stem=frame_stem,
        write_stamps=False,
        sci_shape=sci.shape,
    )
    result = run_hotpants_frame(
        sci,
        err,
        template,
        mask,
        ref_stars_xy,
        hp_config,
        collect_kernel_params=collect_kernel_params,
        oversample=getattr(hp, "oversample", None),
        use_c_extension=getattr(hp, "use_c_extension", None),
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
    field_ctx=None,
    mask_catalog=None,
    sector: Optional[int] = None,
    camera: Optional[int] = None,
    data_root: Optional[str] = None,
    ccd: Optional[int] = None,
    template_dir: Optional[str] = None,
) -> KernelFitResult:
    """
    Fit PSF kernel on angle-ranked min-background FFI through a 3-round
    hotpants + robust TESSreduce background loop (HP1 -> bkg1 -> HP2 -> bkg2
    -> HP3 final).

    Kernel NPZ and metadata are written under *artifact_dir* (typically
    ``ws/kernel_fit/``), not the event root.
    """
    from syndiff_pipeline.difference_imaging.stages.hotpants import (
        _padded_crop_bounds,
        _resolve_hotpants_mask_array,
        _resolve_linear_template_pad,
    )

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
        manifest,
        weighting_factor=params.weighting_factor,
        sector=sector,
        camera=camera,
        data_root=data_root,
        ccd=ccd,
    )
    product_id = tess_product_id_from_ffi_path(min_bg_path) or "unknown"
    if field_ctx is not None:
        from syndiff_pipeline.difference_imaging.support.template_resolution import (
            assemble_field_template_for_ffi,
            group_id_for_ffi,
        )

        group_dx, group_dy = 0.0, 0.0
        template_path = f"field:group_id={group_id_for_ffi(manifest, min_bg_path)}"
        mapping_grid = getattr(field_ctx, "mapping_grid", None)
        if mapping_grid is not None:
            # The field-store contrib canvas is already sized to (and local
            # to) the MAPGRID template-support window -- ctx.base_tess_shape
            # equals mapping_grid.array_shape_os()/array_shape_native()
            # exactly. Passing absolute FFI pixel bounds (e.g.
            # template_ffi_bounds(), which start at ffi_xmin/ffi_ymin, not 0)
            # as a *local* array crop mis-indexes the canvas -- for negative
            # ffi_ymin this silently wraps and returns an empty slice. Request
            # the whole (already correctly sized) canvas instead.
            field_template = assemble_field_template_for_ffi(
                field_ctx,
                manifest,
                min_bg_path,
                crop=None,
            )
        else:
            os_factor = max(1, int(getattr(field_ctx, "oversampling_factor", 1) or 1))
            field_template = assemble_field_template_for_ffi(
                field_ctx,
                manifest,
                min_bg_path,
                crop=(
                    int(crop_bounds["x_min"]) * os_factor,
                    int(crop_bounds["x_max"]) * os_factor,
                    int(crop_bounds["y_min"]) * os_factor,
                    int(crop_bounds["y_max"]) * os_factor,
                ),
            )
    else:
        group_dx, group_dy, template_path = resolve_template_for_ffi(
            output_dir,
            manifest,
            min_bg_path,
            template_dir=template_dir,
        )
        field_template = None

    log.info(
        "Kernel fit on min-background FFI %s (score=%.4f) template %s",
        product_id,
        angle_score,
        template_path if field_ctx is not None else f"dx={group_dx:.3f} dy={group_dy:.3f}",
    )

    ffi, err = _load_ffi_cropped(min_bg_path, crop_bounds)
    mapping_grid = getattr(field_ctx, "mapping_grid", None) if field_ctx is not None else None
    linear_pad = 0
    if field_template is not None:
        template = field_template
    elif mapping_grid is None:
        # Linear mode: validate the template's on-disk support covers
        # crop_bounds with the standard convolution margin, then load exactly
        # that padded rectangle as real pixels (see
        # hotpants._resolve_linear_template_pad).
        linear_pad = _resolve_linear_template_pad(template_path, crop_bounds) or 0
        template = _load_template_cropped(
            template_path, _padded_crop_bounds(crop_bounds, linear_pad)
        )
    else:
        template = _load_template_cropped(
            template_path, crop_bounds, preserve_template_support=True
        )
    header = wcs_grouping.crop_ffi_header(min_bg_path, crop_bounds)

    btjd = None
    for c in ("btjd", "BTJD", "tjd", "TJD", "jd", "JD"):
        if c in manifest.columns:
            try:
                row = manifest.loc[
                    manifest.apply(
                        lambda r: tess_product_id_from_ffi_path(
                            str(r.get("ffi_path", r.get("path", "")))
                        )
                        == product_id,
                        axis=1,
                    )
                ]
                if len(row):
                    btjd = float(row.iloc[0][c])
            except Exception:
                pass
            break

    hotpants_mask = np.asarray(
        _resolve_hotpants_mask_array(shared_mask, mask_catalog, btjd)
    )
    residual_mask = (
        mask_catalog.mask_at(btjd, which="full")
        if mask_catalog is not None
        else shared_mask
    )

    from syndiff_pipeline.difference_imaging.stages.hotpants import (
        _pair_hotpants_inputs,
    )

    # Use the same science/template/error pairing contract as the ordinary
    # Hotpants stage for every kernel-fit round. This pads science, error,
    # and masks to the template support (field mode via MappingGrid, linear
    # mode via the fixed convolution margin resolved above).
    raw_ffi, raw_template, raw_err = ffi, template, err
    ffi, template, err, hotpants_mask, _ = _pair_hotpants_inputs(
        raw_ffi, raw_template, raw_err, hotpants_mask, mapping_grid, linear_pad
    )
    if mapping_grid is not None or linear_pad:
        _, _, _, residual_mask, _ = _pair_hotpants_inputs(
            raw_ffi, raw_template, raw_err, residual_mask, mapping_grid, linear_pad
        )

    if ffi.shape != np.asarray(hotpants_mask).shape:
        raise ValueError(
            f"FFI shape {ffi.shape} != hotpants mask shape {np.asarray(hotpants_mask).shape}"
        )
    if err.shape != ffi.shape:
        raise ValueError(
            "kernel-fit paired inputs must have identical template-support geometry: "
            f"science={ffi.shape}, error={err.shape}"
        )
    # Masks/error stay native; the field template is at the store's own
    # oversampling (1x or Nx) and is paired against native science directly
    # by Hotpants' own oversample handling -- it is not expected to match
    # ffi.shape when oversampling_factor > 1.
    field_os_factor = (
        max(1, int(getattr(field_ctx, "oversampling_factor", 1) or 1))
        if field_ctx is not None
        else 1
    )
    expected_template_shape = (
        tuple(s * field_os_factor for s in ffi.shape) if field_os_factor > 1 else ffi.shape
    )
    if template.shape != expected_template_shape:
        raise ValueError(
            "kernel-fit paired inputs must have consistent template-support geometry: "
            f"science={ffi.shape}, template={template.shape}, "
            f"expected_template={expected_template_shape} (oversampling_factor={field_os_factor})"
        )


    def _tessreduce_bkg(diff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return estimate_tessreduce_residual_background(
            diff,
            residual_mask,
            smooth_gauss=params.tessreduce_smooth_gauss,
            anomaly_gauss=params.tessreduce_anomaly_gauss,
            qe_spline_degree=params.tessreduce_qe_spline_degree,
            qe_spline_smooth_mult=params.tessreduce_qe_spline_smooth_mult,
            boundary_k=params.tessreduce_boundary_k,
            boundary_sigma=params.tessreduce_boundary_sigma,
            boundary_rim_width=params.tessreduce_boundary_rim_width,
        )

    def _background_subtracted_convolved(hp_result: dict) -> np.ndarray:
        """Return the Hotpants model with its fitted background removed."""
        convolved = np.asarray(hp_result["convolved"], dtype=np.float64)
        hotpants_bkg = hp_result.get("bkg")
        if hotpants_bkg is None:
            return convolved
        background = np.asarray(hotpants_bkg, dtype=np.float64)
        if background.shape != convolved.shape:
            raise ValueError(
                "Hotpants convolved image and background must have identical shapes: "
                f"convolved={convolved.shape}, background={background.shape}"
            )
        return convolved - background

    basis = build_kernel_basis(hp)
    hp2_params = replace(hp, hp_bgo=0)
    with tempfile.TemporaryDirectory(prefix="kernel_fit_") as work_root:
        # Round 1: hotpants with the configured (e.g. 3rd order) background
        # order, kernel discarded -- only its convolved template matters.
        hp1, _ = _run_hotpants_round(
            sci=ffi,
            err=err,
            template=template,
            mask=hotpants_mask,
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
        conv1 = _background_subtracted_convolved(hp1)
        diff1 = ffi - conv1
        tessreduce_bkg1, tessreduce_pre_qe1, tessreduce_qe1 = _tessreduce_bkg(diff1)
        cleaned_ffi1 = ffi - tessreduce_bkg1

        # Round 2: hotpants with bgo=0 on the round-1 cleaned FFI, kernel
        # discarded -- its convolved template feeds the second background
        # estimate.
        hp2, hp2_config = _run_hotpants_round(
            sci=cleaned_ffi1,
            err=err,
            template=template,
            mask=hotpants_mask,
            ref_stars_xy=ref_stars_xy,
            hp=hp2_params,
            work_dir=os.path.join(work_root, "hp2"),
            frame_stem=f"{product_id}_hp2",
            collect_kernel_params=False,
        )
        if not hp2.get("success"):
            raise RuntimeError(
                f"Kernel-fit Hotpants round 2 failed: {hp2.get('error_msg', '')}"
            )
        conv2 = _background_subtracted_convolved(hp2)
        diff2 = ffi - conv2
        tessreduce_bkg2, tessreduce_pre_qe2, tessreduce_qe2 = _tessreduce_bkg(diff2)
        cleaned_ffi2 = ffi - tessreduce_bkg2

        # Round 3 (final): hotpants with bgo=0 on the round-2 cleaned FFI --
        # this kernel and convolved template are the ones persisted.
        hp3, hp3_config = _run_hotpants_round(
            sci=cleaned_ffi2,
            err=err,
            template=template,
            mask=hotpants_mask,
            ref_stars_xy=ref_stars_xy,
            hp=hp2_params,
            work_dir=os.path.join(work_root, "hp3"),
            frame_stem=f"{product_id}_hp3",
            collect_kernel_params=params.write_kernel_params,
        )
        if not hp3.get("success"):
            raise RuntimeError(
                f"Kernel-fit Hotpants round 3 failed: {hp3.get('error_msg', '')}"
            )

        kernel_params = hp3.get("kernel_params_arrays")
        kernel_image = kernel_from_hotpants_result(
            kernel_params, hp3_config, ffi.shape
        )
        if kernel_image is None or kernel_params is None:
            raise RuntimeError("HP3 did not return kernel_solution")
        kernel_solution = np.asarray(
            kernel_params["kernel_solution"], dtype=np.float64
        ).ravel()

    reference_kernel_sum = kernel_sum_at_center(
        kernel_solution, hp3_config, ffi.shape
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
        "tessreduce_boundary_k": int(params.tessreduce_boundary_k),
        "tessreduce_boundary_sigma": float(params.tessreduce_boundary_sigma),
        "tessreduce_boundary_rim_width": int(params.tessreduce_boundary_rim_width),
        "reference_kernel_sum": float(reference_kernel_sum),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    if debug_ws_dir and params.write_debug_fits:
        os.makedirs(debug_ws_dir, exist_ok=True)
        from syndiff_pipeline.common.grid_pairing import trim_padded_products
        from syndiff_pipeline.difference_imaging.stages.hotpants import (
            _trim_linear_pad,
        )

        # Field-mode arrays live on the padded MAPGRID=3 template-support
        # canvas; linear-mode arrays live on the crop_bounds +/- linear_pad
        # canvas (see _resolve_linear_template_pad). Both must be trimmed
        # back to crop_bounds for science-facing diagnostics.
        def _trim(arr: np.ndarray) -> np.ndarray:
            if mapping_grid is not None:
                return trim_padded_products(arr, grid=mapping_grid)
            if linear_pad:
                return _trim_linear_pad(arr, linear_pad)
            return np.asarray(arr)

        template_os_factor = (
            max(1, int(getattr(field_ctx, "oversampling_factor", 1) or 1))
            if field_ctx is not None
            else 1
        )
        if template_os_factor > 1:
            ys, xs = mapping_grid.science_slice_native()
            os_ys = slice(ys.start * template_os_factor, ys.stop * template_os_factor)
            os_xs = slice(xs.start * template_os_factor, xs.stop * template_os_factor)
            template_trimmed = np.asarray(template)[os_ys, os_xs]
        else:
            template_trimmed = _trim(template)

        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "ffi"), _trim(ffi), header=header
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "template"),
            template_trimmed,
            header=header,
        )
        # Round 1
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp1_diff"),
            _trim(hp1["diff"]),
            header=header,
        )
        if hp1.get("bkg") is not None:
            _write_image_fits(
                workspace_frame_fits_path(debug_ws_dir, "hp1_bkg"),
                _trim(hp1["bkg"]),
                header=header,
            )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp1_convolved"),
            _trim(conv1),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "ffi_minus_conv1"),
            _trim(diff1),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "tessreduce_bkg1_pre_qe"),
            _trim(tessreduce_pre_qe1),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "tessreduce_bkg1_qe_factor"),
            _trim(tessreduce_qe1),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "tessreduce_bkg1"),
            _trim(tessreduce_bkg1),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "cleaned_ffi1"),
            _trim(cleaned_ffi1),
            header=header,
        )
        # Round 2 (bgo=0)
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp2_diff"),
            _trim(hp2["diff"]),
            header=header,
        )
        if hp2.get("bkg") is not None:
            _write_image_fits(
                workspace_frame_fits_path(debug_ws_dir, "hp2_bkg"),
                _trim(hp2["bkg"]),
                header=header,
            )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp2_convolved"),
            _trim(conv2),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "ffi_minus_conv2"),
            _trim(diff2),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "tessreduce_bkg2_pre_qe"),
            _trim(tessreduce_pre_qe2),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "tessreduce_bkg2_qe_factor"),
            _trim(tessreduce_qe2),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "tessreduce_bkg2"),
            _trim(tessreduce_bkg2),
            header=header,
        )
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "cleaned_ffi2"),
            _trim(cleaned_ffi2),
            header=header,
        )
        # Round 3 (final -- kernel + convolved template persisted)
        _write_image_fits(
            workspace_frame_fits_path(debug_ws_dir, "hp3_diff"),
            _trim(hp3["diff"]),
            header=header,
        )
        if hp3.get("bkg") is not None:
            _write_image_fits(
                workspace_frame_fits_path(debug_ws_dir, "hp3_bkg"),
                _trim(hp3["bkg"]),
                header=header,
            )
        if hp3.get("convolved") is not None:
            _write_image_fits(
                workspace_frame_fits_path(debug_ws_dir, "hp3_convolved"),
                _trim(_background_subtracted_convolved(hp3)),
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
        hp_config=hp3_config,
    )
