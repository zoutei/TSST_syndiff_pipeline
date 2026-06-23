"""Per-FFI algebraic difference with photutils background (no Hotpants)."""

from __future__ import annotations

import logging
import multiprocessing
import os
from typing import Any, Optional

import numpy as np
from joblib import Parallel, delayed

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.stages.convolved_templates import (
    lookup_convolved_path,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    _load_ffi_cropped,
    _write_image_fits,
)
from syndiff_pipeline.difference_imaging.stages.kernel_photutils import (
    photutils_background_masked,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
    workspace_frame_stem,
)
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    resolve_template_for_ffi,
)

log = logging.getLogger(__name__)

_KERNEL_SUBTRACT_LOKY: Optional[dict[str, Any]] = None


def _kernel_subtract_loky_initializer(payload: dict[str, Any]) -> None:
    global _KERNEL_SUBTRACT_LOKY
    _KERNEL_SUBTRACT_LOKY = payload


def _load_convolved_crop(path: str, crop_bounds: dict) -> np.ndarray:
    from astropy.io import fits

    ox = int(crop_bounds["x_min"])
    oy = int(crop_bounds["y_min"])
    x1 = int(crop_bounds["x_max"])
    y1 = int(crop_bounds["y_max"])
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data.shape == tuple(crop_bounds["shape"]):
            return np.asarray(data, dtype=np.float64)
        return data[oy:y1, ox:x1].astype(np.float64)


def _process_one_frame(task: tuple) -> dict:
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
    convolved_table = p["convolved_table"]
    phot_box_size = p["phot_box_size"]
    diffs_dir = p["diffs_dir"]
    bkg_dir = p.get("bkg_dir")
    diffs_label = p["diffs_label"]
    bkg_label = p.get("bkg_label")
    output_dir = p["output_dir"]
    manifest = p["manifest"]

    product_id = tess_product_id_from_ffi_path(ffi_path) or "unknown"
    diff_stem = workspace_frame_stem(product_id, diffs_label)
    diff_out = os.path.join(diffs_dir, f"{diff_stem}.fits")

    if os.path.isfile(diff_out):
        return {
            "success": True,
            "product_id": product_id,
            "stem": diff_stem,
            "skipped": True,
        }

    try:
        group_dx, group_dy, _ = resolve_template_for_ffi(
            output_dir, manifest, ffi_path
        )
        conv_path = lookup_convolved_path(convolved_table, group_dx, group_dy)
        ffi, _ = _load_ffi_cropped(ffi_path, crop_bounds)
        convolved = _load_convolved_crop(conv_path, crop_bounds)
        if ffi.shape != convolved.shape:
            raise ValueError(
                f"FFI shape {ffi.shape} != convolved template {convolved.shape}"
            )

        diff_raw = ffi - convolved
        phot_bkg = photutils_background_masked(
            diff_raw, shared_mask, box_size=phot_box_size
        )

        header = wcs_grouping.crop_ffi_header(str(ffi_path), crop_bounds)
        _write_image_fits(diff_out, diff_raw, header=header)
        if bkg_dir and bkg_label:
            bkg_stem = workspace_frame_stem(product_id, bkg_label)
            _write_image_fits(
                os.path.join(bkg_dir, f"{bkg_stem}.fits"),
                phot_bkg,
                header=header,
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
    phot_box_size: int,
    diffs_dir: str,
    diffs_label: str,
    bkg_dir: Optional[str] = None,
    bkg_label: Optional[str] = None,
    n_jobs: int = 1,
) -> list[dict]:
    """Run algebraic diff + photutils background for each FFI."""
    os.makedirs(diffs_dir, exist_ok=True)
    if bkg_dir:
        os.makedirs(bkg_dir, exist_ok=True)

    payload = {
        "crop_bounds": crop_bounds,
        "shared_mask": shared_mask,
        "convolved_table": convolved_table,
        "phot_box_size": int(phot_box_size),
        "diffs_dir": diffs_dir,
        "bkg_dir": bkg_dir,
        "diffs_label": diffs_label,
        "bkg_label": bkg_label,
        "output_dir": output_dir,
        "manifest": manifest,
    }

    tasks = [(ffi_path,) for ffi_path in ffi_paths]

    n_workers = max(1, min(int(n_jobs), len(tasks), multiprocessing.cpu_count()))
    if n_workers == 1:
        _kernel_subtract_loky_initializer(payload)
        results = [_process_one_frame(t) for t in tasks]
    else:
        results = Parallel(
            n_jobs=n_workers,
            backend="loky",
            initializer=_kernel_subtract_loky_initializer,
            initargs=(payload,),
        )(delayed(_process_one_frame)(t) for t in tasks)

    ok = sum(1 for r in results if r.get("success"))
    log.info(
        "kernel_subtract: %d/%d frames succeeded (%d skipped existing)",
        ok,
        len(results),
        sum(1 for r in results if r.get("skipped")),
    )
    return results
