"""
hotpants_runner.py
==================
Hotpants differencing: FFI (cropped) vs PS1 template, Gaia reference stars.

When global ``cfg.n_jobs`` (or the stage's ``hotpants_n_jobs`` if set) is greater than 1,
:func:`hotpants_loop` uses joblib **loky** with a **worker initializer** so the
shared mask, reference-star coordinates, and template map are installed once per
process instead of being cloudpickled with every FFI task (only per-frame
arguments are serialized per task).

Supports **legacy** layout (``diff_rN/``, optional ``convolved_rN/``) and
**workspace** layout (separate dirs for diffs, convolved model, optional bkg).

Template discovery: when ``cfg.template_paths`` is empty but ``template_dir`` is set,
:func:`ensure_template_paths_from_syndiff_or_group_dirs` can fill paths from either
flat ``syndiff_template_*_dx*_dy*.fits`` names (matched to ``group_dx``/``group_dy``)
or ``group_<id>/ps1_template.fits`` (see :func:`syndiff_pipeline.difference_imaging.orchestration.config.discover_template_paths`).

Kernel artifacts (each :func:`hotpants_loop` pass): beside the diffs directory,
``{basename}_kernel_reconstruction.npz`` holds the shared raw ``basis`` stack (built
in-repo to match HOTPANTS ``kernel_vector`` / ``getKernelVec``, since pyhotpants does
not ship a Python entry point for it) and Hotpants geometry metadata;
``{basename}_kernel_params/{stem}.npz`` holds per-FFI
fitted parameters: per-stamp data from ``get_substamp_details()`` (padded
``local_kernel_solution`` and coordinates) plus the **global** ``kernel_solution``
vector from ``run_pipeline()`` / ``get_final_outputs()`` (current pyhotpants does
not include that vector in ``get_substamp_details()``). See
:func:`write_kernel_reconstruction_npz` / :func:`_calculate_kernel_basis` and
:func:`kernel_reconstruction_npz_path` / :func:`kernel_params_dir`.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from joblib import Parallel, delayed

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    sanitize_workspace_label,
    tess_product_id_from_ffi_path,
    workspace_frame_stem,
    workspace_label_from_dir,
)

warnings.filterwarnings("ignore", category=FITSFixedWarning)

log = logging.getLogger(__name__)


def write_diff_noise_mask_fits(
    out_path: str,
    diff_img: np.ndarray,
    noise_img: Optional[np.ndarray],
    mask_img: Optional[np.ndarray],
) -> None:
    """
    Write one multi-extension FITS: PRIMARY = difference, extensions NOISE (1σ)
    and MASK when provided. Uses float32 for all arrays.
    """
    primary = fits.PrimaryHDU(np.asarray(diff_img, dtype=np.float32))
    hdul: list = [primary]
    if noise_img is not None:
        hdul.append(
            fits.ImageHDU(np.asarray(noise_img, dtype=np.float32), name="NOISE")
        )
    if mask_img is not None:
        hdul.append(
            fits.ImageHDU(np.asarray(mask_img, dtype=np.float32), name="MASK")
        )
    fits.HDUList(hdul).writeto(out_path, overwrite=True)


# Loky workers: large read-only objects are set once per process via initializer
# (avoids cloudpickling them with every task).
_HOTPANTS_LOKY_PAYLOAD: Optional[dict[str, Any]] = None


def _hotpants_loky_initializer(
    mask: np.ndarray,
    ref_stars_xy: np.ndarray,
    hp: HotpantsParams,
    template_path_map: dict,
    crop_bounds: dict,
    workspace_dirs: HotpantsWorkspaceDirs,
    round_id: int,
    legacy_bkg_sidecar: bool,
    sci_workspace_dir: Optional[str],
) -> None:
    global _HOTPANTS_LOKY_PAYLOAD
    _HOTPANTS_LOKY_PAYLOAD = {
        "mask": mask,
        "ref_stars_xy": ref_stars_xy,
        "hp": hp,
        "template_path_map": template_path_map,
        "crop_bounds": crop_bounds,
        "workspace_dirs": workspace_dirs,
        "round_id": round_id,
        "legacy_bkg_sidecar": legacy_bkg_sidecar,
        "sci_workspace_dir": sci_workspace_dir,
    }


def _hotpants_loky_run_task(
    task: tuple,
) -> dict:
    """Run one Hotpants frame inside a loky worker (uses :data:`_HOTPANTS_LOKY_PAYLOAD`)."""
    global _HOTPANTS_LOKY_PAYLOAD
    p = _HOTPANTS_LOKY_PAYLOAD
    if p is None:
        return {
            "stem": None,
            "success": False,
            "error_msg": "hotpants loky worker not initialized",
        }
    ffi_path, product_id, group_id, bkg_i = task
    if product_id is None:
        return {"stem": None, "success": False, "error_msg": "not in wcs_table"}
    return _process_one_frame(
        ffi_path=ffi_path,
        product_id=product_id,
        group_id=group_id,
        hp=p["hp"],
        template_path_map=p["template_path_map"],
        mask=p["mask"],
        crop_bounds=p["crop_bounds"],
        ref_stars_xy=p["ref_stars_xy"],
        dirs=p["workspace_dirs"],
        round_id=p["round_id"],
        sci_bkg=bkg_i,
        legacy_diff_sidecar_bkg=p["legacy_bkg_sidecar"],
        sci_workspace_dir=p.get("sci_workspace_dir"),
    )


@dataclass
class HotpantsWorkspaceDirs:
    """On-disk layout for one Hotpants pass."""

    diffs: str
    convolved: str
    bkg: Optional[str] = None
    """
    If set, polynomial background FITS are written here as
    ``{tess<digits>}_{bkg_label}.fits`` (label derived from ``os.path.basename(bkg)``).
    If None, backgrounds are not persisted (still used inside Hotpants for the diff).
    """
    stamps: Optional[str] = None
    """
    If set, ``{tess<digits>}_{diffs_label}_stamps.fits`` (Hotpants stamp region)
    is written here. If None, :func:`build_hotpants_config` falls back to ``diffs``
    (legacy).
    """


def stamps_dir_for_diffs_workspace(diff_dir: str) -> str:
    """
    Hotpants stamp FITS live next to the diffs workspace directory.

    For config-driven runs, ``diff_dir`` is ``{output}/ws/{label}`` (e.g. ``hp_d``),
    so stamps are ``{output}/ws/{label}_stamps`` (e.g. ``hp_d_stamps`` under ``ws/``).
    For legacy layout, ``diff_rN_stamps`` sits beside ``diff_rN`` under ``output_dir``.
    """
    d = os.path.abspath(diff_dir)
    parent = os.path.dirname(d)
    base = os.path.basename(d)
    return os.path.join(parent, f"{base}_stamps")


def kernel_reconstruction_npz_path(diff_dir: str) -> str:
    """Path to the shared ``*_kernel_reconstruction.npz`` next to the diffs directory."""
    d = os.path.abspath(diff_dir)
    parent = os.path.dirname(d)
    base = os.path.basename(d)
    return os.path.join(parent, f"{base}_kernel_reconstruction.npz")


def kernel_params_dir(diff_dir: str) -> str:
    """Directory for per-FFI ``{stem}.npz`` kernel parameter archives."""
    d = os.path.abspath(diff_dir)
    parent = os.path.dirname(d)
    base = os.path.basename(d)
    return os.path.join(parent, f"{base}_kernel_params")


def _kernel_sigma_deg_for_basis(hp: HotpantsParams) -> tuple[int, list[float], list[int], int]:
    """
    Match :func:`build_hotpants_config`: ``rkernel``, truncated ``sigma_gauss`` /
    ``deg_fixe`` to ``hp_ngauss`` (same convention as ``HotpantsConfig``).
    """
    sci_fwhm = float(hp.sci_fwhm)
    rkernel = int(2.5 * sci_fwhm)
    ngauss = max(1, int(hp.hp_ngauss))
    sigma_full = [sci_fwhm / 2.5, sci_fwhm, sci_fwhm * 2]
    deg_full = [int(d) for d in list(hp.hp_deg_fixe)]
    n = min(ngauss, len(sigma_full), len(deg_full))
    return rkernel, sigma_full[:n], deg_full[:n], ngauss


def _calculate_kernel_basis(
    shape: tuple[int, int],
    sigma_gauss: list[float],
    deg_fixe: list[int],
) -> list[np.ndarray]:
    """
    Non-PCA HOTPANTS kernel basis images, matching ``kernel_vector`` / ``getKernelVec``
    in pyhotpants' ``alard.c``. The installed ``hotpants`` package does not expose this;
    we duplicate the small pure-numeric piece here for ``*_kernel_reconstruction.npz``.
    """
    height, width = int(shape[0]), int(shape[1])
    if height != width:
        raise ValueError(f"kernel basis must be square; got shape={shape!r}")
    if width % 2 != 1:
        raise ValueError(f"kernel basis width must be odd; got shape={shape!r}")

    half_width = width // 2
    n_comp_ker = sum(((int(d) + 1) * (int(d) + 2)) // 2 for d in deg_fixe)
    filter_x = np.zeros((n_comp_ker, width), dtype=np.float64)
    filter_y = np.zeros((n_comp_ker, width), dtype=np.float64)
    basis: list[np.ndarray] = []

    nvec = 0
    for ig, dmax in enumerate(deg_fixe):
        sigma_g = float(sigma_gauss[ig])
        for deg_x in range(int(dmax) + 1):
            for deg_y in range(int(dmax) + 1 - deg_x):
                dx = (deg_x // 2) * 2 - deg_x
                dy = (deg_y // 2) * 2 - deg_y
                sum_x = 0.0
                sum_y = 0.0

                for ix in range(width):
                    x = float(ix - half_width)
                    qe = np.exp(-x * x * sigma_g)
                    filter_x[nvec, ix] = qe * (x**deg_x)
                    filter_y[nvec, ix] = qe * (x**deg_y)
                    sum_x += filter_x[nvec, ix]
                    sum_y += filter_y[nvec, ix]

                vec = np.zeros(width * width, dtype=np.float64)
                if dx == 0 and dy == 0:
                    filter_x[nvec, :] *= 1.0 / sum_x
                    filter_y[nvec, :] *= 1.0 / sum_y

                for i in range(width):
                    for j in range(width):
                        vec[i + width * j] = filter_x[nvec, i] * filter_y[nvec, j]

                if dx == 0 and dy == 0 and nvec > 0:
                    vec -= basis[0].ravel()

                basis.append(vec.reshape(height, width))
                nvec += 1

    return basis


def write_kernel_reconstruction_npz(hp: HotpantsParams, path: str) -> bool:
    """
    Write one shared archive: stacked raw kernel ``basis`` (``n_basis``, H, W) from
    :func:`_calculate_kernel_basis`, plus scalars matching the run.

    Returns True if the file was written, False if construction failed.
    """
    rkernel, sigma_gauss, deg_fixe, _ngauss = _kernel_sigma_deg_for_basis(hp)
    size = 2 * rkernel + 1
    shape = (size, size)
    try:
        basis_list = _calculate_kernel_basis(shape, sigma_gauss, deg_fixe)
        basis = np.stack([np.asarray(b, dtype=np.float64) for b in basis_list], axis=0)
    except Exception as exc:
        log.error("_calculate_kernel_basis failed; skip %s: %s", path, exc)
        return False
    rss = int(2.5 * float(hp.sci_fwhm))
    meta: dict[str, Any] = {
        "basis": basis,
        "rkernel": np.int32(rkernel),
        "rss": np.int32(rss),
        "ko": np.int32(hp.hp_ko),
        "bgo": np.int32(hp.hp_bgo),
        "nstampx": np.int32(hp.hp_nstampx),
        "nstampy": np.int32(hp.hp_nstampy),
        "nss": np.int32(hp.hp_nss),
        "ngauss": np.int32(hp.hp_ngauss),
        "n_basis": np.int32(basis.shape[0]),
        "sci_fwhm": np.float64(hp.sci_fwhm),
        "sigma_gauss": np.asarray(sigma_gauss, dtype=np.float64),
        "deg_fixe": np.asarray(deg_fixe, dtype=np.int32),
        "hp_normalize": np.array(str(hp.hp_normalize)),
        "hp_force_convolve": np.array(str(hp.hp_force_convolve)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **meta)
    log.info("Wrote hotpants kernel reconstruction bundle: %s", path)
    return True


def _serialize_substamp_details(details: Any) -> Optional[dict[str, np.ndarray]]:
    """
    Turn ``get_substamp_details()`` into numpy arrays only (loky-friendly).

    Per-stamp ``local_kernel_solution`` rows are padded to a common width;
    true lengths are in ``local_kernel_len``.
    """
    if not isinstance(details, dict):
        return None
    out: dict[str, np.ndarray] = {}
    # Older pyhotpants might expose kernel_solution here; current versions do not
    # (global vector is taken from run_pipeline() return in run_hotpants_frame).
    ks = details.get("kernel_solution")
    if ks is not None:
        out["kernel_solution"] = np.asarray(ks, dtype=np.float64).ravel()
    substamps = (
        details.get("template_substamps")
        or details.get("substamps")
        or []
    )
    xs: list[float] = []
    ys: list[float] = []
    locals_list: list[np.ndarray] = []
    for s in substamps:
        loc = getattr(s, "local_kernel_solution", None)
        if loc is None:
            continue
        try:
            x = float(getattr(s, "x", np.nan))
            y = float(getattr(s, "y", np.nan))
        except (TypeError, ValueError):
            continue
        arr = np.asarray(loc, dtype=np.float64).ravel()
        xs.append(x)
        ys.append(y)
        locals_list.append(arr)
    if locals_list:
        lens = np.array([a.size for a in locals_list], dtype=np.int32)
        max_len = int(lens.max())
        padded = np.zeros((len(locals_list), max_len), dtype=np.float64)
        for i, a in enumerate(locals_list):
            padded[i, : a.size] = a
        out["substamp_x"] = np.asarray(xs, dtype=np.float64)
        out["substamp_y"] = np.asarray(ys, dtype=np.float64)
        out["local_kernel_solution"] = padded
        out["local_kernel_len"] = lens
    if not out:
        return None
    return out


def _save_frame_kernel_params_npz(
    stem: str,
    out_dir: str,
    arrays: dict[str, np.ndarray],
    hp: HotpantsParams,
) -> None:
    """Write ``{stem}.npz`` with fitted parameters plus a small config echo."""
    rkernel, _sig, _deg, _ = _kernel_sigma_deg_for_basis(hp)
    extra = {
        "rkernel": np.int32(rkernel),
        "ko": np.int32(hp.hp_ko),
        "bgo": np.int32(hp.hp_bgo),
        "nstampx": np.int32(hp.hp_nstampx),
        "nstampy": np.int32(hp.hp_nstampy),
        "sci_fwhm": np.float64(hp.sci_fwhm),
    }
    if "local_kernel_solution" in arrays:
        extra["n_substamps"] = np.int32(arrays["local_kernel_solution"].shape[0])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(out_path, **arrays, **extra)


def _get_hotpants_classes():
    try:
        from hotpants import Hotpants, HotpantsConfig

        return Hotpants, HotpantsConfig
    except ImportError:
        sys.modules.pop("hotpants", None)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "pyhotpants"
        if not candidate.is_dir():
            continue
        root = str(candidate.resolve())
        inserted = False
        if root not in sys.path:
            sys.path.insert(0, root)
            inserted = True
        try:
            from hotpants import Hotpants, HotpantsConfig

            return Hotpants, HotpantsConfig
        except ImportError:
            sys.modules.pop("hotpants", None)
            if inserted and sys.path and sys.path[0] == root:
                sys.path.pop(0)
            continue
    raise ImportError(
        "pyhotpants not found. Ensure the 'hotpants' directory is on sys.path "
        "or install the hotpants package."
    )


def build_hotpants_config(
    sci_fwhm: float,
    hp: HotpantsParams,
    diff_dir: str,
    convolved_dir: str,
    frame_stem: str,
    stamps_dir: Optional[str] = None,
):
    _, HotpantsConfig = _get_hotpants_classes()

    kernel_halfwidth = int(2.5 * sci_fwhm)
    substamp_halfwidth = int(2.5 * sci_fwhm)

    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(convolved_dir, exist_ok=True)
    stamp_out_dir = stamps_dir if stamps_dir is not None else diff_dir
    os.makedirs(stamp_out_dir, exist_ok=True)

    hp_config = HotpantsConfig(
        rkernel=int(kernel_halfwidth),
        ko=int(hp.hp_ko),
        bgo=int(hp.hp_bgo),
        nstampx=int(hp.hp_nstampx),
        nstampy=int(hp.hp_nstampy),
        nss=int(hp.hp_nss),
        rss=int(substamp_halfwidth),
        ngauss=int(hp.hp_ngauss),
        deg_fixe=[int(d) for d in hp.hp_deg_fixe],
        sigma_gauss=[float(sci_fwhm / 2.5), float(sci_fwhm), float(sci_fwhm * 2)],
        kf_spread_mask1=float(hp.hp_kf_spread_mask1) if getattr(hp, 'hp_kf_spread_mask1', None) is not None else 1.0,
        ks=int(hp.hp_ks) if getattr(hp, 'hp_ks', None) is not None else 0,
        kfm=int(hp.hp_kfm) if getattr(hp, 'hp_kfm', None) is not None else 0,
        fitthresh=float(hp.hp_fitthresh) if getattr(hp, 'hp_fitthresh', None) is not None else 500,
        stat_sig=float(hp.hp_stat_sig) if getattr(hp, 'hp_stat_sig', None) is not None else 3.0,
        force_convolve=str(hp.hp_force_convolve),
        normalize=str(hp.hp_normalize),
        verbose=0,
        # Diff / noise / mask: written in one multi-extension FITS via Astropy after run_pipeline.
        output_file=None,
        noise_image_file=None,
        mask_image_file=None,
        sigma_image_file=None,
        convolved_image_file=os.path.join(convolved_dir, f"{frame_stem}.fits"),
        stamp_region_file=os.path.join(stamp_out_dir, f"{frame_stem}_stamps.fits"),
    )
    return hp_config


def run_hotpants_frame(
    sci_array: np.ndarray,
    sci_err_array: np.ndarray,
    tmpl_array: np.ndarray,
    mask_array: np.ndarray,
    ref_stars_xy: np.ndarray,
    hp_config,
    *,
    science_ffi_path: Optional[str] = None,
    diff_fits_path_for_logs: Optional[str] = None,
) -> dict:
    """
    Run Hotpants on in-memory template/science arrays.

    ``science_ffi_path`` is included in log lines when pyhotpants emits its
    minimal-header FITS warning (arrays are passed, so no science FITS header).
    """
    Hotpants, _ = _get_hotpants_classes()
    result = {
        "diff": None,
        "bkg": None,
        "convolved": None,
        "noise": None,
        "mask": None,
        "success": False,
        "error_msg": "",
    }
    try:
        hp = Hotpants(
            template_data=np.ascontiguousarray(tmpl_array, dtype=np.float64),
            image_data=np.ascontiguousarray(sci_array, dtype=np.float64),
            t_error=np.zeros(tmpl_array.shape, dtype=np.float64),
            i_error=np.ascontiguousarray(sci_err_array, dtype=np.float64),
            t_mask=np.ascontiguousarray(np.isnan(tmpl_array), dtype=bool),
            i_mask=np.ascontiguousarray(mask_array > 0, dtype=bool),
            star_catalog=np.ascontiguousarray(ref_stars_xy, dtype=np.float64),
            config=hp_config,
            output_header=None,
        )
        # pyhotpants warns when saving FITS from in-memory arrays (no WCS header).
        # Record warnings so we can append the science FFI path for debugging.
        _HEADER_WARN_SUBSTR = "No FITS header available"
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            res = hp.run_pipeline()
        for w in wrec:
            msg = str(w.message)
            if _HEADER_WARN_SUBSTR in msg:
                ffi = science_ffi_path or "unknown"
                outf = getattr(hp_config, "output_file", None) or diff_fits_path_for_logs
                log.warning("%s science_ffi=%r diff_output=%r", msg, ffi, outf)
            else:
                warnings.warn(w.message, category=w.category, stacklevel=1)
        result["diff"] = res.get("diff_image")
        result["bkg"] = res.get("background")
        result["convolved"] = res.get("convolved_image")
        result["noise"] = res.get("noise_image")
        result["mask"] = res.get("output_mask")
        result["success"] = result["diff"] is not None
        if result["success"]:
            arrays = None
            try:
                details = hp.get_substamp_details()
                arrays = _serialize_substamp_details(details)
            except Exception as exc:
                log.warning("get_substamp_details failed (substamp arrays may be missing): %s", exc)
            gk = res.get("kernel_solution")
            if gk is not None:
                merged = dict(arrays) if arrays else {}
                merged["kernel_solution"] = np.asarray(gk, dtype=np.float64).ravel()
                arrays = merged
            result["kernel_params_arrays"] = arrays
    except Exception as exc:
        result["error_msg"] = str(exc)
        log.warning("hotpants failed: %s", exc)
    return result


def _load_ffi_cropped(ffi_path: str, bounds: dict) -> tuple:
    with fits.open(ffi_path, memmap=True) as hdul:
        sci = hdul[1].data.astype(np.float64)
        try:
            err = hdul[2].data.astype(np.float64)
        except Exception:
            err = np.zeros_like(sci)
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    return sci[y0:y1, x0:x1], err[y0:y1, x0:x1]


def _load_template_cropped(tmpl_path: str, bounds: dict) -> np.ndarray:
    with fits.open(tmpl_path, memmap=True) as hdul:
        if hdul[0].data is not None:
            data = hdul[0].data.astype(np.float64)
        else:
            data = hdul[1].data.astype(np.float64)
    sh = bounds.get("shape")
    if sh is not None:
        ny, nx = int(sh[0]), int(sh[1])
        if data.shape == (ny, nx):
            return data
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    return data[y0:y1, x0:x1]


def _save_bkg_fits(bkg: np.ndarray, basename: str, bkg_dir: str) -> None:
    """Write ``bkg`` as ``{bkg_dir}/{basename}.fits``."""
    os.makedirs(bkg_dir, exist_ok=True)
    fits.writeto(
        os.path.join(bkg_dir, f"{basename}.fits"),
        bkg.astype(np.float32),
        overwrite=True,
    )


def _legacy_save_bkg_sidecar(bkg: np.ndarray, diff_basename: str, diff_dir: str) -> None:
    """Write ``{diff_basename}_bkg.fits`` next to the diff (legacy layout)."""
    os.makedirs(diff_dir, exist_ok=True)
    fits.writeto(
        os.path.join(diff_dir, f"{diff_basename}_bkg.fits"),
        bkg.astype(np.float32),
        overwrite=True,
    )


def _process_one_frame(
    ffi_path,
    product_id,
    group_id,
    hp: HotpantsParams,
    template_path_map,
    mask,
    crop_bounds,
    ref_stars_xy,
    dirs: HotpantsWorkspaceDirs,
    round_id: int,
    sci_bkg=None,
    legacy_diff_sidecar_bkg: bool = False,
    sci_workspace_dir: Optional[str] = None,
):
    diffs_label = workspace_label_from_dir(dirs.diffs)
    diff_stem = workspace_frame_stem(product_id, diffs_label)

    if sci_workspace_dir:
        sci_label = workspace_label_from_dir(sci_workspace_dir)
        sci_stem = workspace_frame_stem(product_id, sci_label)
        sp = os.path.join(sci_workspace_dir, f"{sci_stem}.fits")
        if not os.path.isfile(sp):
            log.error("science workspace FITS missing for %s: %s", product_id, sp)
            return {
                "stem": diff_stem,
                "ffi_product_id": product_id,
                "group_id": group_id,
                "success": False,
                "error_msg": f"missing science FITS {sp}",
            }
        sci_crop = fits.getdata(sp).astype(np.float64)
        # Use the same cropped noise map as a raw-FFI pass. All-zero i_error makes
        # Hotpants weights degenerate and often triggers LUDCMP / clipped-stamp failures.
        _, err_crop = _load_ffi_cropped(ffi_path, crop_bounds)
        err_crop = np.asarray(err_crop, dtype=np.float64)
        if err_crop.shape != sci_crop.shape:
            log.warning(
                "Science %s shape %s != FFI err %s; using sqrt(|sci|)+1 heuristic errors",
                product_id,
                sci_crop.shape,
                err_crop.shape,
            )
            err_crop = np.sqrt(np.abs(sci_crop)) + 1.0
        else:
            err_crop = np.maximum(err_crop, np.sqrt(np.abs(sci_crop)) + 1.0)
    else:
        sci_crop, err_crop = _load_ffi_cropped(ffi_path, crop_bounds)

    if round_id > 1 and sci_bkg is not None:
        sci_crop = sci_crop - sci_bkg

    tmpl_path = template_path_map.get(group_id)
    if tmpl_path is None:
        log.error("No template for group_id=%s; frame %s skipped.", group_id, product_id)
        return {
            "stem": diff_stem,
            "ffi_product_id": product_id,
            "group_id": group_id,
            "success": False,
            "error_msg": "missing template",
        }

    tmpl_crop = _load_template_cropped(tmpl_path, crop_bounds)

    diff_out_path = os.path.join(dirs.diffs, f"{diff_stem}.fits")
    hp_config = build_hotpants_config(
        sci_fwhm=hp.sci_fwhm,
        hp=hp,
        diff_dir=dirs.diffs,
        convolved_dir=dirs.convolved,
        frame_stem=diff_stem,
        stamps_dir=dirs.stamps,
    )

    result = run_hotpants_frame(
        sci_array=sci_crop,
        sci_err_array=err_crop,
        tmpl_array=tmpl_crop,
        mask_array=mask,
        ref_stars_xy=ref_stars_xy,
        hp_config=hp_config,
        science_ffi_path=str(ffi_path),
        diff_fits_path_for_logs=diff_out_path,
    )

    if result["success"]:
        try:
            write_diff_noise_mask_fits(
                diff_out_path,
                result["diff"],
                result.get("noise"),
                result.get("mask"),
            )
        except Exception as exc:
            log.error("Failed writing %s: %s", diff_out_path, exc)
            result["success"] = False
            result["error_msg"] = str(exc)
        if legacy_diff_sidecar_bkg and result.get("bkg") is not None:
            _legacy_save_bkg_sidecar(result["bkg"], diff_stem, dirs.diffs)
        elif dirs.bkg and result.get("bkg") is not None:
            bkg_label = workspace_label_from_dir(dirs.bkg)
            bkg_basename = workspace_frame_stem(product_id, bkg_label)
            _save_bkg_fits(result["bkg"], bkg_basename, dirs.bkg)
        k_arrays = result.pop("kernel_params_arrays", None)
        if k_arrays:
            try:
                _save_frame_kernel_params_npz(
                    diff_stem, kernel_params_dir(dirs.diffs), k_arrays, hp
                )
            except Exception as exc:
                log.warning("Saving kernel params npz failed for %s: %s", diff_stem, exc)

    result["stem"] = diff_stem
    result["ffi_product_id"] = product_id
    result["group_id"] = group_id
    result["path"] = diff_out_path
    return result


def hotpants_loop(
    ffi_paths: list,
    wcs_table: pd.DataFrame,
    template_path_map: dict,
    mask: np.ndarray,
    crop_bounds: dict,
    hp: HotpantsParams,
    cfg,
    output_dir: str,
    ref_stars_df: pd.DataFrame,
    round_id: int = 1,
    sci_bkg_stack: np.ndarray = None,
    workspace_dirs: Optional[HotpantsWorkspaceDirs] = None,
    sci_workspace_dir: Optional[str] = None,
) -> list:
    """
    Run hotpants over all FFIs in parallel.

    If ``workspace_dirs`` is None, use legacy paths: ``diff_r{round_id}/``,
    ``convolved_r{round_id}/``, and ``*_bkg.fits`` sidecars in the diff directory.

    When ``sci_workspace_dir`` is set, each frame's science array is read from
    ``{sci_workspace_dir}/{stem}.fits`` (crop-sized), e.g. from a prior ``subtract``
    stage, instead of cropping the raw FFI.
    """
    if workspace_dirs is None:
        diff_base = os.path.join(output_dir, f"diff_r{round_id}")
        conv_base = os.path.join(output_dir, f"convolved_r{round_id}")
        workspace_dirs = HotpantsWorkspaceDirs(
            diffs=diff_base,
            convolved=conv_base,
            bkg=None,
        )
        legacy_bkg_sidecar = True
    else:
        legacy_bkg_sidecar = False
        if workspace_dirs.bkg:
            os.makedirs(workspace_dirs.bkg, exist_ok=True)
        os.makedirs(workspace_dirs.diffs, exist_ok=True)
        os.makedirs(workspace_dirs.convolved, exist_ok=True)

    stamps_path = stamps_dir_for_diffs_workspace(workspace_dirs.diffs)
    os.makedirs(stamps_path, exist_ok=True)
    workspace_dirs = replace(workspace_dirs, stamps=stamps_path)

    recon_path = kernel_reconstruction_npz_path(workspace_dirs.diffs)
    kparams_root = kernel_params_dir(workspace_dirs.diffs)
    os.makedirs(kparams_root, exist_ok=True)
    write_kernel_reconstruction_npz(hp, recon_path)

    ref_stars_xy = ref_stars_df[["x", "y"]].values
    path_to_row = {str(r["path"]): i for i, r in wcs_table.iterrows()}

    tasks = []
    for i, ffi_path in enumerate(ffi_paths):
        row_idx = path_to_row.get(str(ffi_path))
        if row_idx is None:
            log.warning("FFI not in wcs_table: %s", ffi_path)
            tasks.append((ffi_path, None, 0, None))
            continue
        row = wcs_table.iloc[row_idx]
        product_id = tess_product_id_from_ffi_path(ffi_path)
        if product_id is None:
            log.warning("FFI basename does not start with tess<digits>: %s", ffi_path)
            tasks.append((ffi_path, None, 0, None))
            continue
        group_id = int(row.get("group_id", 0))
        bkg_i = sci_bkg_stack[i] if (sci_bkg_stack is not None and i < len(sci_bkg_stack)) else None
        tasks.append((ffi_path, product_id, group_id, bkg_i))

    hn = hp.hotpants_n_jobs
    if hn is None:
        n_workers = max(1, int(cfg.n_jobs or 1))
    else:
        n_workers = max(1, int(hn))

    log.info(
        "Running hotpants round %s on %d frames (n_jobs=%s) ...",
        round_id,
        len(tasks),
        n_workers,
    )

    if n_workers == 1:
        def _serial_worker(args):
            ffi_path, product_id, group_id, bkg_i = args
            if product_id is None:
                return {"stem": None, "success": False, "error_msg": "not in wcs_table"}
            return _process_one_frame(
                ffi_path=ffi_path,
                product_id=product_id,
                group_id=group_id,
                hp=hp,
                template_path_map=template_path_map,
                mask=mask,
                crop_bounds=crop_bounds,
                ref_stars_xy=ref_stars_xy,
                dirs=workspace_dirs,
                round_id=round_id,
                sci_bkg=bkg_i,
                legacy_diff_sidecar_bkg=legacy_bkg_sidecar,
                sci_workspace_dir=sci_workspace_dir,
            )

        results = [_serial_worker(t) for t in tasks]
    else:
        # Process pool: pyhotpants C extension does not release the GIL, so
        # prefer="threads" would serialize CPU work across frames. loky runs
        # each frame in a separate interpreter for real multi-core speed.
        # Initializer injects large read-only arrays once per worker (not per task).
        parallel = Parallel(
            n_jobs=n_workers,
            backend="loky",
            initializer=_hotpants_loky_initializer,
            initargs=(
                mask,
                ref_stars_xy,
                hp,
                template_path_map,
                crop_bounds,
                workspace_dirs,
                round_id,
                legacy_bkg_sidecar,
                sci_workspace_dir,
            ),
        )
        results = parallel(delayed(_hotpants_loky_run_task)(t) for t in tasks)

    n_ok = sum(1 for r in results if r.get("success"))
    log.info("Round %s hotpants: %d/%d frames succeeded.", round_id, n_ok, len(results))
    return results


def collect_diff_paths(output_dir: str, round_id: int) -> list:
    """Sorted diff FITS under ``diff_r{round_id}/`` (legacy layout)."""
    import glob

    diff_dir = os.path.join(output_dir, f"diff_r{round_id}")
    paths = sorted(glob.glob(os.path.join(diff_dir, "*.fits")))
    paths = [
        p
        for p in paths
        if not p.endswith("_bkg.fits") and not p.endswith("_stamps.fits")
    ]
    return paths


def collect_diff_paths_in_dir(diff_dir: str) -> list:
    """Sorted diff FITS in an arbitrary directory (workspace layout)."""
    import glob

    paths = sorted(glob.glob(os.path.join(diff_dir, "*.fits")))
    return [
        p
        for p in paths
        if not p.endswith("_bkg.fits") and not p.endswith("_stamps.fits")
    ]


# ── syndiff_template_* filename discovery (flat template_dir) ─────────────────

_SYNDIFF_TEMPLATE_RE = re.compile(
    r"^syndiff_template_s(?P<sector>\d+)_(?P<camera>\d+)_(?P<ccd>\d+)"
    r"(?P<roi>_x(?P<x0>\d+)-(?P<x1>\d+)_y(?P<y0>\d+)-(?P<y1>\d+))?"
    r"(?:_os\d+)?"
    r"_dx(?P<dx>[+-]?\d*\.?\d+)_dy(?P<dy>[+-]?\d*\.?\d+)\.fits(?:\.gz)?$",
    re.IGNORECASE,
)


class SyndiffTemplateDiscoveryError(RuntimeError):
    """Missing or ambiguous ``syndiff_template_*.fits`` for required WCS groups."""


@dataclass(frozen=True)
class ParsedSyndiffTemplate:
    sector: int
    camera: int
    ccd: int
    x_min: Optional[int]
    x_max: Optional[int]
    y_min: Optional[int]
    y_max: Optional[int]
    dx: float
    dy: float
    path: str


def parse_syndiff_template_filename(
    path_or_basename: str,
) -> Optional[ParsedSyndiffTemplate]:
    """
    Parse a ``syndiff_template_*.fits`` basename or path.

    Example: ``syndiff_template_s0020_3_3_x1068-2092_y1039-2048_dx-0.000_dy0.010.fits``
    → sector 20, camera 3, ccd 3, ROI, dx, dy (``dy`` may omit a sign after ``dy``).

    Returns ``None`` if the name does not match.
    """
    name = Path(path_or_basename).name
    m = _SYNDIFF_TEMPLATE_RE.match(name)
    if not m:
        return None
    sec = int(m.group("sector"))
    cam = int(m.group("camera"))
    ccd = int(m.group("ccd"))
    if m.group("roi"):
        x0, x1 = int(m.group("x0")), int(m.group("x1"))
        y0, y1 = int(m.group("y0")), int(m.group("y1"))
    else:
        x0 = x1 = y0 = y1 = None
    dx = float(m.group("dx"))
    dy = float(m.group("dy"))
    p = Path(path_or_basename)
    path = str(p.resolve()) if p.is_file() else str(path_or_basename)
    return ParsedSyndiffTemplate(sec, cam, ccd, x0, x1, y0, y1, dx, dy, path)


def _syndiff_offsets_match(a: float, b: float, offset_threshold: float) -> bool:
    tol = max(1e-5, 0.01 * float(offset_threshold))
    return abs(float(a) - float(b)) <= tol


def _syndiff_roi_matches(parsed: ParsedSyndiffTemplate, crop_bounds: dict) -> bool:
    if parsed.x_min is None:
        return True
    return (
        int(crop_bounds["x_min"]) == parsed.x_min
        and int(crop_bounds["x_max"]) == parsed.x_max
        and int(crop_bounds["y_min"]) == parsed.y_min
        and int(crop_bounds["y_max"]) == parsed.y_max
    )


def _required_syndiff_template_groups(wcs_table: pd.DataFrame) -> pd.DataFrame:
    df = wcs_table.copy()
    if "wcs_ok" in df.columns:
        wok = df["wcs_ok"]
        ok = wok.apply(lambda x: x is True or str(x).lower() in ("true", "1"))
        df = df.loc[ok]
    need = ["group_id", "group_dx", "group_dy"]
    for c in need:
        if c not in df.columns:
            raise SyndiffTemplateDiscoveryError(
                f"wcs_table missing column {c!r}; run wcs_grouping first."
            )
    sub = df[need].dropna()
    sub = sub.loc[sub["group_id"] >= 0]
    if sub.empty:
        raise SyndiffTemplateDiscoveryError(
            "No valid template groups in wcs_table (check wcs_ok / group_id)."
        )
    return sub.drop_duplicates(subset=["group_id"], keep="first")


def verify_syndiff_templates(
    template_dir: str,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    *,
    sector: int,
    camera: int,
    ccd: int,
    offset_threshold: float = 0.01,
) -> dict[int, str]:
    """
    Require exactly one ``syndiff_template_*.fits`` per ``group_id`` in *wcs_table*.

    Matches sector/camera/ccd, optional ROI in the filename vs *crop_bounds*, and
    filename ``dx``/``dy`` to ``group_dx``/``group_dy`` (tolerance scales with
    *offset_threshold*).

    Returns ``group_id`` → absolute path. Raises :exc:`SyndiffTemplateDiscoveryError`
    if any group is missing or ambiguous.
    """
    root = Path(template_dir)
    if not root.is_dir():
        raise SyndiffTemplateDiscoveryError(
            f"template_dir is not a directory: {template_dir!r}"
        )

    parsed_files: list[ParsedSyndiffTemplate] = []
    for name in sorted(os.listdir(root)):
        lower = name.lower()
        if not (lower.endswith(".fits.gz") or lower.endswith(".fits")):
            continue
        full = root / name
        if not full.is_file():
            continue
        p = parse_syndiff_template_filename(str(full))
        if p is None:
            continue
        parsed_files.append(
            ParsedSyndiffTemplate(
                p.sector,
                p.camera,
                p.ccd,
                p.x_min,
                p.x_max,
                p.y_min,
                p.y_max,
                p.dx,
                p.dy,
                str(full.resolve()),
            )
        )

    if not parsed_files:
        raise SyndiffTemplateDiscoveryError(
            f"No syndiff_template_*.fits matched the expected name pattern under {root!r}."
        )

    required = _required_syndiff_template_groups(wcs_table)
    assignments: dict[int, str] = {}
    used_paths: set[str] = set()

    for _, row in required.iterrows():
        gid = int(row["group_id"])
        gdx = float(row["group_dx"])
        gdy = float(row["group_dy"])
        matches: list[ParsedSyndiffTemplate] = []
        for p in parsed_files:
            if p.path in used_paths:
                continue
            if not (p.sector == sector and p.camera == camera and p.ccd == ccd):
                continue
            if not _syndiff_roi_matches(p, crop_bounds):
                continue
            if not (
                _syndiff_offsets_match(p.dx, gdx, offset_threshold)
                and _syndiff_offsets_match(p.dy, gdy, offset_threshold)
            ):
                continue
            matches.append(p)

        if len(matches) == 0:
            raise SyndiffTemplateDiscoveryError(
                "Missing syndiff template for "
                f"group_id={gid} group_dx={gdx} group_dy={gdy} "
                f"(sector={sector} camera={camera} ccd={ccd}, "
                f"ROI x={crop_bounds.get('x_min')}–{crop_bounds.get('x_max')} "
                f"y={crop_bounds.get('y_min')}–{crop_bounds.get('y_max')}). "
                f"Scanned {len(parsed_files)} syndiff_template_*.fits under {root!r}."
            )
        if len(matches) > 1:
            paths = [m.path for m in matches]
            raise SyndiffTemplateDiscoveryError(
                f"Ambiguous syndiff templates for group_id={gid} "
                f"(group_dx={gdx} group_dy={gdy}): {paths!r}"
            )

        assignments[gid] = matches[0].path
        used_paths.add(matches[0].path)

    log.info(
        "Matched %d template group(s) from syndiff_template_*.fits under %s",
        len(assignments),
        root,
    )
    return assignments


def ensure_template_paths_from_syndiff_or_group_dirs(
    cfg,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    *,
    offset_threshold: float = 0.01,
) -> None:
    """
    If ``cfg.template_paths`` is empty and ``cfg.template_dir`` is set, fill it from:

    1. ``syndiff_template_*`` filenames (see :func:`verify_syndiff_templates`), or
    2. ``group_*/ps1_template.fits`` / ``template.fits`` via
       :func:`syndiff_pipeline.difference_imaging.orchestration.config.discover_template_paths`.

    Mutates ``cfg.template_paths`` in place. No-op if ``template_paths`` is already set.
    """
    if cfg.template_paths:
        return
    tdir = (getattr(cfg, "template_dir", "") or "").strip()
    if not tdir or not os.path.isdir(tdir):
        return

    try:
        cfg.template_paths = verify_syndiff_templates(
            tdir,
            wcs_table,
            crop_bounds,
            sector=cfg.sector,
            camera=cfg.camera,
            ccd=cfg.ccd,
            offset_threshold=float(offset_threshold),
        )
        return
    except SyndiffTemplateDiscoveryError as e:
        log.debug("Syndiff template scan did not succeed: %s", e)

    from syndiff_pipeline.difference_imaging.orchestration.config import discover_template_paths

    discovered = discover_template_paths(tdir)
    if discovered:
        cfg.template_paths = discovered
        return

    raise SyndiffTemplateDiscoveryError(
        f"Could not populate template_paths from {tdir!r}: "
        "no syndiff_template_*.fits matched required groups, and no "
        "group_*/ps1_template.fits (or template.fits) layout found."
    )
