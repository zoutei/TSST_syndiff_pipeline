"""TESS strap QE correction and background anomaly fixing."""

from __future__ import annotations

import logging
import multiprocessing
from copy import deepcopy

import numpy as np
from astropy.stats import sigma_clipped_stats
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation, convolve, gaussian_filter, label, laplace
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)


def calc_qe_strap_correction(
    flux: np.ndarray,
    mask: np.ndarray,
    bkg: np.ndarray,
    time_mjd: np.ndarray,
    *,
    qe_floor: float = 1.001,
) -> np.ndarray:
    """Multiplicative QE map from strap columns vs background (TESSreduce _calc_qe)."""
    time = deepcopy(time_mjd)
    mask2d = mask[0] if mask.ndim == 3 else mask
    strap_data = flux * ((mask2d.astype(np.int64) & 4) > 0)
    if mask.ndim == 3:
        src = (mask[:, :, :] & 1) > 0
    else:
        src = (mask2d.astype(np.int64) & 1) > 0
    strap_data = np.where(src, 0.0, strap_data)

    qe = strap_data / bkg
    qe[qe == 0] = np.nan
    _, med, _ = sigma_clipped_stats(qe, axis=1, sigma_upper=2)
    qes = np.ones_like(qe)
    qes[:, :, :] = med[:, np.newaxis, :]
    qes[np.isnan(qes)] = 1

    av_bkg = np.sum(bkg, axis=(1, 2)) / (bkg.shape[1] * bkg.shape[2])
    _, med_b, std_b = sigma_clipped_stats(av_bkg)
    ind = av_bkg < med_b + 5 * std_b
    breaks = np.where(np.diff(time[ind]) > 0.5)[0] + 1
    breaks = np.insert(breaks, 0, 0)
    breaks = np.append(breaks, len(time[ind]))

    new_qes = deepcopy(qes)
    ind_where = np.where(ind)[0]
    for i in range(len(breaks) - 1):
        seg_len = abs(breaks[i] - breaks[i + 1])
        if seg_len > 100:
            window_size = int(seg_len / 4)
            if window_size % 2 == 0:
                window_size += 1
            seg_idx = ind_where[breaks[i] : breaks[i + 1]]
            sav = savgol_filter(
                qes[ind][breaks[i] : breaks[i + 1]], window_size, 1, axis=0
            )
            new_qes[seg_idx] = sav
    new_qes[new_qes < qe_floor] = 1.0
    return new_qes


def fix_background_anomalies(
    bkg,
    mask,
    *,
    n_sigma=5.0,
    box_size=16,
    anom_box=30,
    anom_box_fine=4,
    dilate_r=2,
    gauss_smooth=2,
    n_jobs=-1,
):
    """Fix column offsets and transients in a background cube (TESSreduce helpers)."""
    from photutils.background import Background2D, MedianBackground

    T, NY, NX = bkg.shape
    mask2d = mask[0] if mask.ndim == 3 else mask
    strap = (mask2d.astype(np.int64) & 4).astype(bool)
    good_cols = np.where(~strap.any(axis=0))[0]
    strap_cols = np.where(strap.any(axis=0))[0]
    has_straps = len(strap_cols) > 0 and len(good_cols) > 0
    phot_mask = strap | ((mask2d.astype(np.int64) & 1).astype(bool))

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

    nj = multiprocessing.cpu_count() if n_jobs is None or n_jobs < 1 else int(n_jobs)
    results = Parallel(n_jobs=nj)(delayed(_process)(i) for i in range(T))
    bkg_fixed = np.array([r[0] for r in results])
    excesses = [r[1] for r in results]

    if has_straps:
        for i, excess in enumerate(excesses):
            bkg_fixed[i][:, strap_cols] += excess

    return bkg_fixed


def strap_step(
    strap_flux: np.ndarray,
    bkg: np.ndarray,
    mask: np.ndarray,
    time_mjd: np.ndarray,
    *,
    qe_floor: float = 1.001,
    fix_anomalies: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """Apply strap QE scaling then optional anomaly fix to background stack."""
    bkg_arr = np.asarray(bkg, dtype=np.float64)
    qe = calc_qe_strap_correction(
        strap_flux, mask, bkg_arr, time_mjd, qe_floor=qe_floor
    )
    bkg_arr = bkg_arr * qe
    if fix_anomalies:
        bkg_arr = fix_background_anomalies(
            bkg_arr, mask, n_jobs=n_jobs
        )
    return np.asarray(bkg_arr, dtype=np.float32)
