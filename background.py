"""
background.py
=============
Hotpants-facing rough-background stacks and an optional **full spatial** path that
tracks ``TESSreduce.tessreduce.tessreduce.background()`` through ``Smooth_bkg`` /
strap QE / ``fix_background_anomalies``, then temporal smoothing via
:class:`adaptive_background.AdaptiveBackground` (TESSVectors-aware Savitzky–Golay or
adaptive median).

Numerical kernels below marked **Vendored** are lifted from the in-repo tree
``TESSreduce/tessreduce/helpers.py`` and ``TESSreduce/tessreduce/tessreduce.py`` —
SynDiff does not ``import tessreduce``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from joblib import Parallel, delayed
from photutils.detection import StarFinder
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, convolve, gaussian_filter, label, laplace
from scipy.signal import savgol_filter
from skimage.restoration import inpaint

from .adaptive_background import AdaptiveBackground
from .ffi_naming import (
    parse_workspace_frame_stem,
    tess_product_id_from_ffi_path,
    workspace_frame_stem,
    workspace_label_from_dir,
)
from .paths import BACKGROUND_STACK_NPZ_ARRAY_KEY

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Vendored from TESSreduce/tessreduce/helpers.py (strip_units, Smooth_bkg,
#     par_psf_source_mask, fix_background_anomalies) + tessreduce.py excerpts ───
# ═══════════════════════════════════════════════════════════════════════════════


def strip_units(data):
    """
    Removes astropy units from array-like data.

    Source: TESSreduce/tessreduce/helpers.py (strip_units).
    """
    if type(data) != np.ndarray:
        data = data.value
    return deepcopy(data)


def Smooth_bkg(data, gauss_smooth=0, interpolate=False, extrapolate=True):
    """
    Interpolate over masked pixels to derive a smooth background estimate.

    Source: TESSreduce/tessreduce/helpers.py (Smooth_bkg).
    """
    if (~np.isnan(data)).any():
        x = np.arange(0, data.shape[1])
        y = np.arange(0, data.shape[0])
        arr = np.ma.masked_invalid(deepcopy(data))
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~arr.mask]
        y1 = yy[~arr.mask]
        newarr = arr[~arr.mask]
        if (len(x1) > 10) & (len(y1) > 10):
            if interpolate:
                estimate = griddata(
                    (x1, y1), newarr.ravel(), (xx, yy), method="linear"
                )
                nearest = griddata(
                    (x1, y1), newarr.ravel(), (xx, yy), method="nearest"
                )
                if extrapolate:
                    estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]

                estimate = gaussian_filter(estimate, gauss_smooth)

            else:
                mask = deepcopy(arr.mask)
                mask = mask.astype(bool)
                estimate = inpaint.inpaint_biharmonic(data, mask)
                if (np.nanmedian(estimate) < 150) & (np.nanstd(estimate) < 3):
                    gauss_smooth = gauss_smooth * 4
                estimate = gaussian_filter(estimate, gauss_smooth)
        else:
            estimate = np.zeros_like(data) * np.nan
    else:
        estimate = np.zeros_like(data)

    return estimate


def _table_column_float(col) -> np.ndarray:
    """Photutils / Astropy table columns may be bare arrays or Quantity-backed."""
    return np.asarray(col.value if hasattr(col, "value") else col, dtype=float)


def par_psf_source_mask(data, prf, sigma=5):
    """
    Per-frame mask from PSF-shaped sources (photutils StarFinder).

    Source: TESSreduce/tessreduce/helpers.py (par_psf_source_mask).
    """
    mean, med, std = sigma_clipped_stats(data, sigma=3.0)

    finder = StarFinder(med + sigma * std, kernel=prf, exclude_border=False)
    res = finder.find_stars(deepcopy(data))
    m = np.ones_like(data)
    if res is not None:
        x = (_table_column_float(res["xcentroid"]) + 0.5).astype(int)
        y = (_table_column_float(res["ycentroid"]) + 0.5).astype(int)
        fwhm = (_table_column_float(res["fwhm"]) * 1.2 + 0.5).astype(int)
        fwhm[fwhm < 6] = 6
        for i in range(len(x)):
            m[
                y[i] - fwhm[i] // 2 : y[i] + fwhm[i] // 2,
                x[i] - fwhm[i] // 2 : x[i] + fwhm[i] // 2,
            ] = 0
    return m


def small_background_cube(flux: np.ndarray) -> np.ndarray:
    """
    Percentile-style constant background per frame for small cutouts.

    Source: TESSreduce/tessreduce/tessreduce.py (small_background method body).
    """
    bkg = np.zeros_like(flux)
    flux_u = strip_units(flux)
    lim = 2 * np.nanmin(flux_u, axis=(1, 2))
    ind = flux_u > lim[:, np.newaxis, np.newaxis]
    flux_u = flux_u.copy()
    flux_u[ind] = np.nan
    val = np.nanmedian(flux_u, axis=(1, 2))
    bkg[:, :, :] = val[:, np.newaxis, np.newaxis]
    return bkg


def calc_qe_strap_correction(
    flux: np.ndarray, mask: np.ndarray, bkg: np.ndarray, time_mjd: np.ndarray
) -> np.ndarray:
    """
    Multiplicative QE correction from straps vs background (per tessreduce).

    Source: TESSreduce/tessreduce/tessreduce.py (_calc_qe).
    """
    time = deepcopy(time_mjd)
    strap_data = (flux) * ((mask & 4) > 0) * (~mask & 1)
    qe = strap_data / bkg
    qe[qe == 0] = np.nan
    m, med, std = sigma_clipped_stats(qe, axis=1, sigma_upper=2)
    qes = np.ones_like(qe)
    qes[:, :, :] = med[:, np.newaxis, :]
    qes[np.isnan(qes)] = 1

    av_bkg = np.sum(bkg, axis=(1, 2)) / (bkg.shape[1] * bkg.shape[2])
    m, med, std = sigma_clipped_stats(av_bkg)
    ind = av_bkg < med + 5 * std
    breaks = np.where(np.diff(time[ind]) > 0.5)[0] + 1
    breaks = np.insert(breaks, 0, 0)
    breaks = np.append(breaks, len(time[ind]))

    new_qes = deepcopy(qes)
    ind_where = np.where(ind)[0]
    for i in range(len(breaks) - 1):
        if abs(breaks[i] - breaks[i + 1]) > 100:
            window_size = int(abs(breaks[i] - breaks[i + 1]) / 4)
            if window_size / 2 == window_size // 2:
                window_size += 1
            seg_idx = ind_where[breaks[i] : breaks[i + 1]]
            sav = savgol_filter(
                qes[ind][breaks[i] : breaks[i + 1]], window_size, 1, axis=0
            )
            new_qes[seg_idx] = sav
    new_qes[new_qes < 1.001] = 1
    return new_qes


def fix_background_anomalies(
    bkg,
    mask,
    flux=None,
    bkgmask=None,
    n_sigma=5.0,
    box_size=16,
    anom_box=30,
    anom_box_fine=4,
    dilate_r=2,
    gauss_smooth=2,
    n_jobs=-1,
):
    """
    Fix anomalies (asteroids, cosmic rays) in a background cube.

    Source: TESSreduce/tessreduce/helpers.py (fix_background_anomalies).
    """
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    T, NY, NX = bkg.shape
    mask2d = mask[0] if mask.ndim == 3 else mask
    strap = (mask2d & 4).astype(bool)
    good_cols = np.where(~strap.any(axis=0))[0]
    strap_cols = np.where(strap.any(axis=0))[0]
    has_straps = len(strap_cols) > 0 and len(good_cols) > 0
    if bkgmask is not None:
        bkgmask_arr = np.asarray(bkgmask)
        data_src = (
            np.isnan(bkgmask_arr[0])
            if bkgmask_arr.ndim == 3
            else np.isnan(bkgmask_arr)
        )
        phot_mask = strap | data_src
    else:
        phot_mask = strap | (mask2d & 1).astype(bool)

    eff_box = min(box_size, min(NY, NX) // 2)
    eff_box = max(eff_box, 4)

    yr, xr = np.ogrid[-dilate_r : dilate_r + 1, -dilate_r : dilate_r + 1]
    disk = xr**2 + yr**2 <= dilate_r**2

    def _process(i):
        frame = bkg[i].copy()
        excess = np.zeros(len(strap_cols))

        if has_straps:
            interp = np.zeros((NY, len(strap_cols)))
            for r in range(NY):
                interp[r] = np.interp(strap_cols, good_cols, frame[r, good_cols])
            excess = np.nanmedian(frame[:, strap_cols] - interp, axis=0)
            frame[:, strap_cols] -= excess

        try:
            bkg2d = Background2D(
                frame,
                box_size=eff_box,
                filter_size=3,
                mask=phot_mask,
                bkg_estimator=MedianBackground(),
                exclude_percentile=50,
            )
            trend = bkg2d.background
        except Exception:
            trend = np.full_like(frame, np.nanmedian(frame))

        resid = frame - trend

        valid_mask = ~phot_mask

        def _block_sigma(r, box):
            sigma = np.full((NY, NX), np.inf, dtype=float)
            row_starts = [min(r0, NY - box) for r0 in range(0, NY, box)]
            col_starts = [min(c0, NX - box) for c0 in range(0, NX, box)]
            for r0 in row_starts:
                for c0 in col_starts:
                    r1, c1 = r0 + box, c0 + box
                    vals = r[r0:r1, c0:c1][valid_mask[r0:r1, c0:c1]]
                    if vals.size >= 4:
                        m = np.nanmedian(vals)
                        sigma[r0:r1, c0:c1] = 1.4826 * np.nanmedian(np.abs(vals - m))
            return sigma

        sigma_coarse = _block_sigma(resid, anom_box)
        flagged_coarse = np.abs(resid) > n_sigma * sigma_coarse

        if flagged_coarse.any():
            lap = laplace(resid)
            lap_abs = np.abs(lap)
            lap_med = np.nanmedian(lap_abs)
            lap_mad = np.nanmedian(np.abs(lap_abs - lap_med))
            lap_thresh = lap_med + 3 * 1.4826 * lap_mad
            is_sharp = lap_abs > lap_thresh

            edge_border = np.zeros((NY, NX), dtype=bool)
            edge_border[0, :] = True
            edge_border[-1, :] = True
            edge_border[:, 0] = True
            edge_border[:, -1] = True
            labeled, n_comp = label(flagged_coarse)
            sharp_mask = np.zeros((NY, NX), dtype=bool)
            smooth_mask = np.zeros((NY, NX), dtype=bool)
            for comp_id in range(1, n_comp + 1):
                comp = labeled == comp_id
                if (comp & edge_border).any():
                    smooth_mask |= comp
                elif (comp & is_sharp).any():
                    sharp_mask |= comp
                else:
                    smooth_mask |= comp

            if sharp_mask.any():
                dilated = binary_dilation(sharp_mask, structure=disk)
                frame[dilated] = trend[dilated]

            if smooth_mask.any():
                eff_fine = max(min(anom_box_fine, min(NY, NX) // 2), 4)
                try:
                    bkg2d_fine = Background2D(
                        frame,
                        box_size=eff_fine,
                        filter_size=3,
                        mask=phot_mask,
                        bkg_estimator=MedianBackground(),
                        exclude_percentile=50,
                    )
                    trend_fine = bkg2d_fine.background
                except Exception:
                    trend_fine = trend
                resid_fine = frame - trend_fine
                sigma_fine = _block_sigma(resid_fine, anom_box_fine)
                flagged_fine = np.abs(resid_fine) > n_sigma * sigma_fine
                confirmed = smooth_mask & flagged_fine
                if confirmed.any():
                    dilated = binary_dilation(confirmed, structure=disk)
                    frame[dilated] = trend_fine[dilated]

        if gauss_smooth:
            fixed = gaussian_filter(frame, sigma=gauss_smooth)
        else:
            fixed = frame
        return fixed, excess

    results = Parallel(n_jobs=n_jobs)(delayed(_process)(i) for i in range(T))
    bkg_fixed = np.array([r[0] for r in results])
    excesses = [r[1] for r in results]

    if flux is not None and bkgmask is not None:
        sc = SigmaClip(sigma=3.0, maxiters=5)
        bkgmask_arr = np.asarray(bkgmask)
        exclude_mask = (
            np.any(np.isnan(bkgmask_arr), axis=0)
            if bkgmask_arr.ndim == 3
            else np.isnan(bkgmask_arr)
        )
        res_box = min(20, min(NY, NX) // 2)
        res_box = max(res_box, 4)

        def _fit_residual(residual):
            finite_vals = residual[~exclude_mask & np.isfinite(residual)]
            med = np.nanmedian(finite_vals)
            std = np.nanstd(finite_vals)
            transient_mask = exclude_mask | (residual > med + 5 * std)
            try:
                b = Background2D(
                    residual,
                    box_size=res_box,
                    filter_size=3,
                    sigma_clip=sc,
                    bkg_estimator=MedianBackground(),
                    mask=transient_mask,
                    fill_value=0.0,
                )
                return b.background
            except Exception:
                return np.full_like(residual, np.nanmedian(residual[~transient_mask]))

        residuals = flux - bkg_fixed
        corrections = Parallel(n_jobs=n_jobs)(
            delayed(_fit_residual)(residuals[i]) for i in range(T)
        )
        bkg_fixed += np.array(corrections)

    if has_straps:
        for i, excess in enumerate(excesses):
            frame_view = bkg_fixed[i]
            frame_view[:, strap_cols] += excess

    return bkg_fixed


# ═══════════════════════════════════════════════════════════════════════════════
# ── SynDiff orchestration (spatial cube ≈ tessreduce.background spatial part) ───
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "Smooth_bkg",
    "strip_units",
    "estimate_frame_background",
    "tessreduce_style_background_spatial",
    "build_flux_cube_from_hotpants",
    "load_hotpants_row_from_disk",
    "background_loop",
    "background_loop_streaming",
    "load_background_stack",
    "save_background_stack",
    "btjd_for_hotpants_order",
    "adaptive_smooth_background",
]


def _effective_num_cores(num_cores: int) -> int:
    if num_cores is None or num_cores < 1:
        return multiprocessing.cpu_count()
    return int(num_cores)


def psf_source_mask_stack(
    residual_cube: np.ndarray,
    prf_kernel_2d: np.ndarray,
    sigma: float = 5.0,
    *,
    parallel: bool = True,
    num_cores: int = -1,
) -> np.ndarray:
    """
    Apply :func:`par_psf_source_mask` frame-wise (TESSreduce ``psf_source_mask``).

    Source pattern: TESSreduce/tessreduce/tessreduce.py (psf_source_mask method).
    """
    nj = _effective_num_cores(num_cores)
    res = np.asarray(residual_cube, dtype=np.float64)
    if parallel and res.shape[0] > 1:
        masks = Parallel(n_jobs=nj)(
            delayed(par_psf_source_mask)(res[i], prf_kernel_2d, sigma)
            for i in range(res.shape[0])
        )
        return np.array(masks)
    out = np.zeros_like(res)
    for i in range(res.shape[0]):
        out[i] = par_psf_source_mask(res[i], prf_kernel_2d, sigma)
    return out


def tessreduce_style_background_spatial(
    flux: np.ndarray,
    mask: np.ndarray,
    time_mjd: np.ndarray,
    *,
    sector: int,
    camera: int,
    gauss_smooth: float = 2.0,
    calc_qe: bool = True,
    strap_iso: bool = True,
    source_hunt: bool = False,
    interpolate: bool = True,
    rerun_negative: bool = False,
    rerun_diff: bool = False,
    parallel: bool = True,
    num_cores: int = -1,
    use_error_image: bool = False,
    eflux: Optional[np.ndarray] = None,
    vector_path: Optional[str] = None,
    residual_for_source_hunt: Optional[np.ndarray] = None,
    prf_kernel_2d: Optional[np.ndarray] = None,
    source_hunt_sigma: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatial stages of ``TESSreduce.tessreduce.background()`` through
    ``fix_background_anomalies`` (no ``AdaptiveBackground`` here).

    Mirrors step order in:
      TESSreduce/tessreduce/tessreduce.py — ``background()`` ~738–817

    Returns
    -------
    bkg : ndarray (T, ny, nx)
    mask_out : ndarray — copy of ``mask`` after optional bit-8 update when ``calc_qe`` is False
    """
    if source_hunt and (
        residual_for_source_hunt is None or prf_kernel_2d is None
    ):
        raise ValueError(
            "source_hunt=True requires residual_for_source_hunt (T,ny,nx) and "
            "prf_kernel_2d (PRF kernel for photutils StarFinder)."
        )

    nj = _effective_num_cores(num_cores)
    flux_raw = np.asarray(strip_units(flux), dtype=np.float64)
    mask_arr = np.asarray(mask)
    mask_out = mask_arr.copy()
    time_mjd = np.asarray(time_mjd, dtype=float)

    if strap_iso:
        m = (mask_arr == 0) * 1.0
    else:
        m = ((mask_arr & 1 == 0) & (mask_arr & 2 == 0)) * 1.0
    m[m == 0] = np.nan

    sm = None
    if source_hunt:
        sm = psf_source_mask_stack(
            residual_for_source_hunt,
            prf_kernel_2d,
            source_hunt_sigma,
            parallel=parallel,
            num_cores=nj,
        ).astype(float)
        sm[sm == 0] = np.nan
        m = sm * m

    _bkgmask = np.array(m, copy=True)

    ny, nx = flux_raw.shape[1], flux_raw.shape[2]
    if (ny > 30) and (nx > 30):
        bkg_smth = np.zeros_like(flux_raw) * np.nan
        if parallel:
            bkg_smth = np.array(
                Parallel(n_jobs=nj)(
                    delayed(Smooth_bkg)(frame, 0, interpolate)
                    for frame in flux_raw * m
                )
            )
            if rerun_negative:
                if use_error_image and eflux is not None:
                    over_sub = (deepcopy(flux_raw) - bkg_smth) < -eflux
                else:
                    over_sub = (deepcopy(flux_raw) - bkg_smth) < -0.5
                over_sub = np.nansum(over_sub, axis=0) > 0
                strap_mask = (mask_arr & 4) > 0
                if len(strap_mask.shape) == 3:
                    strap_mask = strap_mask[0]
                if strap_iso:
                    over_sub[strap_mask] = 0
                if source_hunt | (len(mask_arr.shape) == 3):
                    m[:, over_sub[:, :]] = 1
                else:
                    m[over_sub] = 1
                _bkgmask = m
                bkg_smth = np.array(
                    Parallel(n_jobs=nj)(
                        delayed(Smooth_bkg)(frame, gauss_smooth, interpolate)
                        for frame in flux_raw * m
                    )
                )

            if rerun_diff:
                sub = deepcopy(flux_raw) - bkg_smth
                s = np.std(sub, axis=0)
                mstat, med, std = sigma_clipped_stats(s)
                resid_mask = (s > med + 5 * std) * 1.0
                resid_mask = convolve(resid_mask, np.ones((2, 2))) > 1
                if source_hunt | (len(mask_arr.shape) == 3):
                    new_mask = deepcopy(sm)
                    new_mask[:, resid_mask[:, :]] = np.nan
                else:
                    new_mask = deepcopy(resid_mask) * 1.0
                    new_mask[new_mask == 1] = np.nan
                    new_mask = abs(new_mask - 1)
                _bkgmask = new_mask
                bkg_smth = np.array(
                    Parallel(n_jobs=nj)(
                        delayed(Smooth_bkg)(frame, 0, interpolate)
                        for frame in flux_raw * new_mask
                    )
                )
        else:
            for i in range(flux_raw.shape[0]):
                bkg_smth[i] = Smooth_bkg((flux_raw * m)[i], 0, interpolate)
    else:
        log.info("Small cutout (<=30x30): small_background_cube (TESSreduce path)")
        bkg_smth = small_background_cube(flux_raw)

    bkg = np.array(bkg_smth)

    if calc_qe:
        qe = calc_qe_strap_correction(flux_raw, mask_arr, bkg, time_mjd)
        bkg = bkg * qe

    bkg = fix_background_anomalies(
        bkg,
        mask_arr,
        flux=flux_raw,
        bkgmask=_bkgmask,
        gauss_smooth=gauss_smooth,
        n_jobs=nj,
    )

    if not calc_qe:
        f = deepcopy(flux_raw)
        mstat, med, std = sigma_clipped_stats(f - bkg, axis=(1, 2))
        bkg = bkg + med[:, np.newaxis, np.newaxis]

        bkgmask_arr = np.asarray(_bkgmask)
        if len(mask_arr.shape) == 3:
            if bkgmask_arr.ndim == 3:
                new_sources = np.isnan(bkgmask_arr)
            else:
                new_sources = np.isnan(bkgmask_arr)[np.newaxis, :, :]
        else:
            if bkgmask_arr.ndim == 3:
                new_sources = np.any(np.isnan(bkgmask_arr), axis=0)
            else:
                new_sources = np.isnan(bkgmask_arr)
        mask_out = mask_arr | (new_sources.astype(mask_arr.dtype) * 8)

    return np.asarray(bkg, dtype=np.float32), mask_out


# ═══════════════════════════════════════════════════════════════════════════════
# ── Temporal smoothing (BTJD alignment + AdaptiveBackground) ───────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def _sanitize_btjd(btjd: np.ndarray) -> np.ndarray:
    """Fill NaN BTJD with linear interpolation in frame index; fallback to index."""
    t = np.asarray(btjd, dtype=float)
    if np.isnan(t).all():
        return np.arange(len(t), dtype=float)
    if not np.isnan(t).any():
        return t
    x = np.arange(len(t))
    m = np.isfinite(t)
    return np.interp(x, x[m], t[m])


def btjd_for_hotpants_order(
    wcs_table: pd.DataFrame,
    hotpants_results: List[dict],
) -> np.ndarray:
    """
    BTJD values in the same order as ``hotpants_results`` (and ``rough_bkg`` axis 0).

    Matches each result's ``ffi_product_id`` (or stem fallback) to ``wcs_table`` rows
    via the leading ``tess<digits>`` token of ``path``/``filename``.
    """
    if "btjd" not in wcs_table.columns:
        raise ValueError("wcs_table must contain a 'btjd' column")
    if "path" in wcs_table.columns:
        pids_tbl = wcs_table["path"].astype(str).map(
            lambda p: tess_product_id_from_ffi_path(p) or ""
        )
    elif "filename" in wcs_table.columns:
        pids_tbl = wcs_table["filename"].astype(str).map(
            lambda f: tess_product_id_from_ffi_path(f) or ""
        )
    else:
        raise ValueError("wcs_table must contain 'path' or 'filename'")
    btjd_series = pd.to_numeric(wcs_table["btjd"], errors="coerce")
    pid_to_btjd = dict(zip(pids_tbl, btjd_series))
    pid_to_btjd.pop("", None)

    out = []
    for r in hotpants_results:
        pid = r.get("ffi_product_id")
        if not pid:
            stem = r.get("stem")
            if stem:
                parsed = parse_workspace_frame_stem(str(stem))
                if parsed is not None:
                    pid = parsed[0]
                else:
                    pid = tess_product_id_from_ffi_path(str(stem))
        if not pid:
            out.append(np.nan)
            continue
        v = pid_to_btjd.get(str(pid), np.nan)
        out.append(float(v) if pd.notna(v) else np.nan)
    return np.asarray(out, dtype=float)


def adaptive_smooth_background(
    bkg_stack: np.ndarray,
    time_btjd: np.ndarray,
    sector: int,
    camera: int,
    *,
    vector_path: Optional[str] = None,
    method: str = "savgol",
    savgol_window: Optional[int] = None,
    savgol_polyorder: int = 2,
    w_min: int = 3,
    w_max: int = 51,
    block_size: int = 5,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Temporal smoothing of the rough background cube via TESSreduce ``AdaptiveBackground``.
    """
    time_btjd = _sanitize_btjd(np.asarray(time_btjd, dtype=float))
    if time_btjd.shape[0] != bkg_stack.shape[0]:
        raise ValueError(
            f"time_btjd length {time_btjd.shape[0]} != bkg n_frames {bkg_stack.shape[0]}"
        )
    m = str(method).lower().strip()
    if m not in ("savgol", "adaptive"):
        raise ValueError(f"method must be 'savgol' or 'adaptive', got {method!r}")

    log.info(
        "adaptive_smooth_background: cube %s BTJD finite=%d/%d sector=%s camera=%s "
        "method=%s block_size=%s n_jobs=%s",
        getattr(bkg_stack, "shape", None),
        int(np.isfinite(time_btjd).sum()),
        len(time_btjd),
        sector,
        camera,
        m,
        block_size,
        n_jobs,
    )
    time_mjd = time_btjd + 57000.0
    out_dtype = bkg_stack.dtype
    log.info(
        "adaptive_smooth_background: constructing AdaptiveBackground (TESSVectors I/O)..."
    )
    smoother = AdaptiveBackground(
        bkg_stack,
        time_mjd,
        sector=int(sector),
        camera=int(camera),
        data_path=vector_path,
        n_jobs=n_jobs,
        block_size=block_size,
    )
    if m == "savgol":
        log.info(
            "adaptive_smooth_background: calling smoother.smooth(method='savgol', "
            "window=%s polyorder=%s) ...",
            savgol_window,
            savgol_polyorder,
        )
        smoother.smooth(
            method="savgol",
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
        )
    else:
        log.info(
            "adaptive_smooth_background: calling smoother.smooth(method='adaptive') ..."
        )
        smoother.smooth(
            method="adaptive",
            w_min=w_min,
            w_max=w_max,
            n_jobs=n_jobs,
        )
    log.info(
        "adaptive_smooth_background: smooth() finished; packing output dtype=%s",
        out_dtype,
    )
    smoothed = np.asarray(smoother.smoothed, dtype=out_dtype)
    return smoothed


def build_flux_cube_from_hotpants(
    hotpants_results: list,
    *,
    recombine_hotpants: bool,
) -> np.ndarray:
    """Stack Hotpants ``diff`` planes (+ optional ``bkg``) into a (T, ny, nx) cube."""
    shape = None
    for r in hotpants_results:
        if r.get("success") and r.get("diff") is not None:
            shape = np.asarray(r["diff"]).shape
            break
    if shape is None:
        raise RuntimeError("build_flux_cube_from_hotpants: no successful diff frames.")
    cube = np.zeros((len(hotpants_results), *shape), dtype=np.float64)
    for i, r in enumerate(hotpants_results):
        if not r.get("success") or r.get("diff") is None:
            continue
        diff_img = np.asarray(r["diff"], dtype=np.float64)
        hp_bkg = (
            np.asarray(r["bkg"], dtype=np.float64)
            if r.get("bkg") is not None
            else np.zeros_like(diff_img)
        )
        if recombine_hotpants:
            cube[i] = diff_img + hp_bkg
        else:
            cube[i] = diff_img
    return cube


def estimate_frame_background(
    diff_image: np.ndarray,
    hotpants_bkg: np.ndarray,
    mask: np.ndarray,
    gauss_smooth: float = 2.0,
    recombine_hotpants: bool = False,
    *,
    interpolate: bool = True,
) -> tuple:
    """
    Single-frame rough estimate (streaming path): :func:`Smooth_bkg` on masked flux.

    Default ``interpolate=True`` matches TESSreduce ``background(..., interpolate=True)``.
    """
    if recombine_hotpants:
        to_smooth = diff_image + hotpants_bkg
    else:
        to_smooth = diff_image

    mask_bool = mask > 0
    arr_masked = to_smooth.copy().astype(np.float64)
    arr_masked[mask_bool] = np.nan

    smooth_bkg = Smooth_bkg(arr_masked, gauss_smooth, interpolate)
    return smooth_bkg, smooth_bkg


def load_hotpants_row_from_disk(
    product_id: str,
    diff_dir: str,
    bkg_dir: Optional[str],
    group_id: int = 0,
) -> dict:
    """Load one frame's Hotpants diff (and optional bkg) FITS from disk."""
    diff_label = workspace_label_from_dir(diff_dir)
    diff_stem = workspace_frame_stem(product_id, diff_label)
    dp = os.path.join(diff_dir, f"{diff_stem}.fits")
    if not os.path.isfile(dp):
        return {
            "stem": diff_stem,
            "ffi_product_id": product_id,
            "success": False,
            "diff": None,
            "bkg": None,
            "group_id": int(group_id),
        }
    diff_data = fits.getdata(dp).astype(np.float64)
    bkg_data = None
    if bkg_dir:
        bkg_label = workspace_label_from_dir(bkg_dir)
        bkg_stem = workspace_frame_stem(product_id, bkg_label)
        bp = os.path.join(bkg_dir, f"{bkg_stem}.fits")
        if os.path.isfile(bp):
            bkg_data = fits.getdata(bp).astype(np.float64)
    return {
        "stem": diff_stem,
        "ffi_product_id": product_id,
        "success": True,
        "diff": diff_data,
        "bkg": bkg_data,
        "group_id": int(group_id),
        "path": dp,
    }


def _background_frame_worker(
    task: Tuple[int, dict, np.ndarray, float, bool, bool],
) -> Tuple[int, Optional[np.ndarray]]:
    i, result, mask, gauss_smooth, recombine_hotpants, interpolate_legacy = task
    if not result.get("success") or result.get("diff") is None:
        log.debug("  Frame %s: hotpants failed — background set to zero.", i)
        return i, None

    diff_img = result["diff"].astype(np.float64)
    hp_bkg = (
        result["bkg"].astype(np.float64)
        if result.get("bkg") is not None
        else np.zeros_like(diff_img)
    )

    rough_stack_slice, _ = estimate_frame_background(
        diff_img,
        hp_bkg,
        mask,
        gauss_smooth,
        recombine_hotpants=recombine_hotpants,
        interpolate=interpolate_legacy,
    )
    return i, rough_stack_slice.astype(np.float32)


def _load_and_rough_stream_worker(
    packed: Tuple[
        int, str, str, Optional[str], int, np.ndarray, float, bool, bool
    ],
) -> Tuple[int, Optional[np.ndarray]]:
    i, product_id, diff_dir, bkg_dir, group_id, mask, gauss_smooth, recombine, interp = (
        packed
    )
    row = load_hotpants_row_from_disk(product_id, diff_dir, bkg_dir, group_id)
    return _background_frame_worker(
        (i, row, mask, gauss_smooth, recombine, interp)
    )


def _parallel_map_with_optional_tqdm(
    delayed_calls,
    n_tasks: int,
    desc: str,
    n_jobs_eff: int,
):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return Parallel(n_jobs=n_jobs_eff, backend="loky")(delayed_calls)
    try:
        gen = Parallel(n_jobs=n_jobs_eff, backend="loky", return_as="generator")(
            delayed_calls
        )
        return list(tqdm(gen, total=n_tasks, desc=desc, unit="frame"))
    except TypeError:
        log.debug("joblib Parallel(return_as=...) unavailable; running without tqdm bar.")
        return Parallel(n_jobs=n_jobs_eff, backend="loky")(delayed_calls)


def _tqdm_iter(tasks: list, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(tasks, desc=desc, unit="frame")
    except ImportError:
        return tasks


def background_loop_streaming(
    ffi_paths: Sequence[Union[str, Path]],
    diff_dir: str,
    bkg_dir: Optional[str],
    path_to_group: Dict[str, int],
    mask: np.ndarray,
    output_dir: Optional[str] = None,
    round_id: int = 1,
    gauss_smooth: float = 2.0,
    recombine_hotpants: bool = False,
    n_jobs: int = 1,
    *,
    interpolate_per_frame: bool = True,
) -> np.ndarray:
    """Per-frame load + :func:`estimate_frame_background` (RAM-friendly)."""
    ffi_paths = list(ffi_paths)
    n_frames = len(ffi_paths)
    if n_frames == 0:
        raise RuntimeError("background_loop_streaming: empty ffi_paths.")

    diff_label = workspace_label_from_dir(diff_dir)
    product_ids = [tess_product_id_from_ffi_path(p) or "" for p in ffi_paths]

    shape = None
    for pid in product_ids:
        if not pid:
            continue
        stem = workspace_frame_stem(pid, diff_label)
        dp = os.path.join(diff_dir, f"{stem}.fits")
        if os.path.isfile(dp):
            shape = fits.getdata(dp).shape
            break
    if shape is None:
        raise RuntimeError(
            "background_loop_streaming: no diff FITS found under diff_dir for any FFI."
        )

    rough_bkg_stack = np.zeros((n_frames, *shape), dtype=np.float32)
    tasks = [
        (
            i,
            pid,
            diff_dir,
            bkg_dir,
            int(path_to_group.get(pid, 0)),
            mask,
            gauss_smooth,
            recombine_hotpants,
            interpolate_per_frame,
        )
        for i, pid in enumerate(product_ids)
    ]
    n_jobs_eff = max(1, int(n_jobs or 1))
    parallel = n_jobs_eff != 1 and n_frames > 1

    if parallel:
        log.info(
            "  background_loop_streaming: per-frame load+rough bkg n_jobs=%s (loky), %d frames",
            n_jobs_eff,
            n_frames,
        )
        frame_results = _parallel_map_with_optional_tqdm(
            (delayed(_load_and_rough_stream_worker)(t) for t in tasks),
            n_frames,
            "Rough bkg (load+est)",
            n_jobs_eff,
        )
    else:
        frame_results = [
            _load_and_rough_stream_worker(t)
            for t in _tqdm_iter(tasks, "Rough bkg (load+est)")
        ]

    for i, rough_bkg in frame_results:
        if rough_bkg is not None:
            rough_bkg_stack[i] = rough_bkg

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"rough_bkg_r{round_id}")
        npz_path = f"{base}.npz"
        npy_path = f"{base}.npy"
        np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: rough_bkg_stack})
        np.save(npy_path, rough_bkg_stack)
        log.info(
            "  Rough background stack (round %s) saved to %s and %s",
            round_id,
            npz_path,
            npy_path,
        )

    return rough_bkg_stack


def background_loop(
    hotpants_results: list,
    mask: np.ndarray,
    output_dir: str = None,
    round_id: int = 1,
    gauss_smooth: float = 2.0,
    recombine_hotpants: bool = False,
    n_jobs: int = 1,
    *,
    tessreduce_spatial: bool = False,
    time_mjd: Optional[np.ndarray] = None,
    sector: int = 1,
    camera: int = 1,
    vector_path: Optional[str] = None,
    calc_qe: bool = True,
    strap_iso: bool = True,
    source_hunt: bool = False,
    interpolate: bool = True,
    rerun_negative: bool = False,
    rerun_diff: bool = False,
    parallel: Optional[bool] = None,
    use_error_image: bool = False,
    eflux: Optional[np.ndarray] = None,
    residual_for_source_hunt: Optional[np.ndarray] = None,
    prf_kernel_2d: Optional[np.ndarray] = None,
    interpolate_legacy_per_frame: bool = True,
) -> np.ndarray:
    """
    Build rough background stack from Hotpants rows.

    When ``tessreduce_spatial`` is True, runs :func:`tessreduce_style_background_spatial`
    on the full flux cube. Otherwise per-frame :func:`estimate_frame_background`.
    """
    n_frames = len(hotpants_results)

    if tessreduce_spatial:
        if time_mjd is None:
            raise ValueError("tessreduce_spatial=True requires time_mjd array (length T).")
        flux_cube = build_flux_cube_from_hotpants(
            hotpants_results, recombine_hotpants=recombine_hotpants
        )
        if len(time_mjd) != flux_cube.shape[0]:
            raise ValueError(
                f"time_mjd length {len(time_mjd)} != flux cube T={flux_cube.shape[0]}"
            )
        par = (n_jobs > 1) if parallel is None else bool(parallel)
        nj = max(1, int(n_jobs or 1))
        rough, _mask_out = tessreduce_style_background_spatial(
            flux_cube,
            mask,
            time_mjd,
            sector=sector,
            camera=camera,
            gauss_smooth=gauss_smooth,
            calc_qe=calc_qe,
            strap_iso=strap_iso,
            source_hunt=source_hunt,
            interpolate=interpolate,
            rerun_negative=rerun_negative,
            rerun_diff=rerun_diff,
            parallel=par,
            num_cores=nj,
            use_error_image=use_error_image,
            eflux=eflux,
            vector_path=vector_path,
            residual_for_source_hunt=residual_for_source_hunt,
            prf_kernel_2d=prf_kernel_2d,
        )
    else:
        shape = None
        for r in hotpants_results:
            if r.get("success") and r.get("diff") is not None:
                shape = r["diff"].shape
                break
        if shape is None:
            raise RuntimeError(
                "No successful hotpants frames found — cannot compute background."
            )

        rough = np.zeros((n_frames, *shape), dtype=np.float32)

        tasks = [
            (i, r, mask, gauss_smooth, recombine_hotpants, interpolate_legacy_per_frame)
            for i, r in enumerate(hotpants_results)
        ]
        n_jobs_eff = max(1, int(n_jobs or 1))
        parallel_workers = n_jobs_eff != 1 and n_frames > 1

        if parallel_workers:
            log.info(
                "  background_loop: per-frame rough bkg n_jobs=%s (loky), %d frames",
                n_jobs_eff,
                n_frames,
            )
            frame_results = _parallel_map_with_optional_tqdm(
                (delayed(_background_frame_worker)(t) for t in tasks),
                n_frames,
                "Rough bkg (estimate)",
                n_jobs_eff,
            )
        else:
            frame_results = [
                _background_frame_worker(t)
                for t in _tqdm_iter(tasks, "Rough bkg (estimate)")
            ]

        for i, rough_bkg in frame_results:
            if rough_bkg is not None:
                rough[i] = rough_bkg

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"rough_bkg_r{round_id}")
        npz_path = f"{base}.npz"
        npy_path = f"{base}.npy"
        np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: rough})
        np.save(npy_path, rough)
        log.info(
            "  Rough background stack (round %s) saved to %s and %s",
            round_id,
            npz_path,
            npy_path,
        )

    return rough


def load_background_stack(path: str) -> np.ndarray:
    """Load a background stack from ``.npz`` (``stack`` array) or ``.npy``."""
    if path.endswith(".npz"):
        z = np.load(path)
        if BACKGROUND_STACK_NPZ_ARRAY_KEY not in z.files:
            raise KeyError(
                f"{path!r} missing array {BACKGROUND_STACK_NPZ_ARRAY_KEY!r}; "
                f"have {list(z.files)}"
            )
        return np.asarray(z[BACKGROUND_STACK_NPZ_ARRAY_KEY])
    return np.load(path)


def save_background_stack(bkg: np.ndarray, path: str) -> None:
    """Save a background stack to ``.npz`` and ``.npy`` (same basename)."""
    path = os.path.abspath(path)
    root, ext = os.path.splitext(path)
    if ext.lower() not in (".npy", ".npz"):
        root = path
    d = os.path.dirname(root)
    if d:
        os.makedirs(d, exist_ok=True)
    npz_path = f"{root}.npz"
    npy_path = f"{root}.npy"
    arr = np.asarray(bkg, dtype=np.float32)
    np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: arr})
    np.save(npy_path, arr)
    log.info("Background stack saved to %s and %s", npz_path, npy_path)
