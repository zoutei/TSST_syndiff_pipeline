"""
masking.py
==========
``shared_mask`` pipeline stage (thin re-exports + Hotpants ref-star selection).

Mask painters and hybrid empirical builders live in ``syndiff_pipeline.masking``.
This module re-exports public names for backward compatibility.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.masking.detector import detector_edge_mask, ps1_coverage_mask
from syndiff_pipeline.masking.geometry import size_limit
from syndiff_pipeline.masking.shared import Cat_mask, make_shared_mask
from syndiff_pipeline.masking.faint_star_squares import faint_star_squares
from syndiff_pipeline.masking.tessreduce_squares import Big_sat, Strap_mask, gaia_auto_mask

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)

__all__ = [
    "SHARED_MASK_FITS_BASENAME",
    "size_limit",
    "faint_star_squares",
    "gaia_auto_mask",
    "Big_sat",
    "Strap_mask",
    "detector_edge_mask",
    "Cat_mask",
    "ps1_coverage_mask",
    "make_shared_mask",
    "grad_clip",
    "fit_strap",
    "correct_straps",
    "select_hotpants_ref_stars",
    "load_gaia_for_masking",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── Vendored from TESSreduce rescale_straps ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def grad_clip(data: np.ndarray, box_size: int = 100) -> np.ndarray:
    """Local sigma-clip based on the gradient of a 1D array."""
    gradind = np.zeros_like(data)
    for i in range(len(data)):
        lo = max(0, i - box_size // 2)
        hi = min(len(data), i + box_size // 2)
        d = data[lo:hi]
        ind = np.isfinite(d)
        d = d[ind]
        if len(d) > 5:
            gind = ~sigma_clip(np.gradient(abs(d)) + d, sigma=2).mask
            gradind[lo:hi][ind] = gind
    return gradind > 0


def fit_strap(data: np.ndarray) -> np.ndarray:
    """Interpolate over missing/bright data in a 1D strap column."""
    x = np.arange(len(data))
    y = data.copy()
    p = np.ones_like(x) * np.nan
    if len(y[np.isfinite(y)]) > 10:
        lim = np.percentile(y[np.isfinite(y)], 50)
        y[y >= lim] = np.nan
        finite = np.isfinite(y)
        if finite.sum() > 5:
            p = interp1d(
                x[finite],
                y[finite],
                bounds_error=False,
                fill_value=np.nan,
                kind="nearest",
            )(x)
    return p


def _calc_strap_factor(i, breaks, size, av_size, normals, data):
    """Compute the QE correction factor for one strap group."""
    qe = np.ones_like(data) * np.nan
    b = int(breaks[i])
    size = size.astype(int)
    nind = np.append(normals[b - av_size : b], normals[b : b + av_size]) + 1
    nind = nind[(nind > 0) & (nind < data.shape[1] - 1)]
    norm_vec = np.nanmedian(data[:, nind], axis=1)
    norm = fit_strap(norm_vec)
    for j in range(size[i]):
        ind = normals[b] + 1 + j
        if 0 < ind < data.shape[1]:
            s1 = fit_strap(data[:, ind])
            ratio = norm / s1
            m = ~sigma_clip(ratio, sigma=2).mask
            qe[:, normals[b] + 1 + j] = np.nanmedian(ratio[m])
    return qe


def correct_straps(
    Image: np.ndarray, mask: np.ndarray, av_size: int = 5, parallel: bool = True
) -> np.ndarray:
    """
    Compute a QE correction image for TESS straps.

    Returns a 2D array of multiplicative factors (~1 outside straps).
    """
    data = deepcopy(Image)
    mask = deepcopy(mask)
    av_size = int(av_size)

    normals = np.where(np.nansum((mask & 4), axis=0) == 0)[0]
    normals = np.append(np.insert(normals, 0, -1), data.shape[1])

    breaks = np.where(np.diff(normals, append=0) > 1)[0]
    breaks[breaks == -1] = 0
    size = (np.diff(normals, append=0))[np.diff(normals, append=0) > 1]

    if len(breaks) == 0:
        return np.ones_like(Image)

    n_jobs = min(multiprocessing.cpu_count(), len(breaks)) if parallel else 1
    qe_list = Parallel(n_jobs=n_jobs)(
        delayed(_calc_strap_factor)(i, breaks, size, av_size, normals, data)
        for i in range(len(breaks))
    )
    qe = np.nanmedian(qe_list, axis=0)
    qe[np.isnan(qe)] = 1.0
    return qe


def select_hotpants_ref_stars(
    gaia_df: pd.DataFrame,
    crop_bounds: dict,
    mag_min: float = 13.5,
    mag_max: float = 14.5,
    isolation_mag: float = 13.5,
    isolation_radius_px: int = 8,
    separation_px: int = 10,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Select clean, isolated reference stars for hotpants stamp fitting.

    Expects gaia_df to have columns: x, y, tess_mag (crop-local coords).
    """
    ny, nx = crop_bounds["shape"]
    in_bounds = (
        (gaia_df["x"] >= 0)
        & (gaia_df["x"] < nx)
        & (gaia_df["y"] >= 0)
        & (gaia_df["y"] < ny)
    )
    gaia_crop = gaia_df[in_bounds].copy().reset_index(drop=True)

    cand_mask = (gaia_crop["tess_mag"] >= mag_min) & (gaia_crop["tess_mag"] <= mag_max)
    candidates = gaia_crop[cand_mask].copy().reset_index(drop=True)

    excluders = gaia_crop[gaia_crop["tess_mag"] < isolation_mag].copy().reset_index(
        drop=True
    )

    exc_xy = np.column_stack([excluders["x"].values, excluders["y"].values])
    keep_phase1 = []

    for idx, row in candidates.iterrows():
        cx, cy = row["x"], row["y"]
        if len(exc_xy) > 0:
            dists = np.sqrt((exc_xy[:, 0] - cx) ** 2 + (exc_xy[:, 1] - cy) ** 2)
            nearby = dists[(dists > 0.5) & (dists <= isolation_radius_px)]
            if len(nearby) > 0:
                continue
        keep_phase1.append(idx)

    survivors = candidates.loc[keep_phase1].copy()
    log.info("  Isolation filter: %d → %d candidates", len(candidates), len(survivors))

    survivors_sorted = survivors.sort_values("tess_mag").reset_index(drop=True)
    kept_xy_arr = []
    kept_rows_arr = []
    for _, row in survivors_sorted.iterrows():
        cx, cy = row["x"], row["y"]
        if kept_xy_arr:
            kxy = np.array(kept_xy_arr)
            dists = np.sqrt((kxy[:, 0] - cx) ** 2 + (kxy[:, 1] - cy) ** 2)
            if dists.min() < separation_px:
                continue
        kept_xy_arr.append([cx, cy])
        kept_rows_arr.append(row.to_dict())

    ref_stars = pd.DataFrame(kept_rows_arr).reset_index(drop=True)
    log.info(
        "  Separation filter: %d → %d reference stars", len(survivors), len(ref_stars)
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "hotpants_substamp_stars.csv")
        ref_stars.to_csv(out_path, index=False)
        log.info("  Reference stars saved to %s", out_path)

    return ref_stars


def load_gaia_for_masking(
    gaia_csv: str, crop_bounds: dict, mag_col: str = "tess_mag"
) -> pd.DataFrame:
    """Load a Gaia CSV and add crop-local ``x``, ``y``, ``mag`` columns."""
    df = pd.read_csv(gaia_csv)
    if "mag" not in df.columns:
        if mag_col in df.columns:
            df["mag"] = df[mag_col]
        elif "phot_rp_mean_mag" in df.columns:
            df["mag"] = df["phot_rp_mean_mag"]
        else:
            raise ValueError(f"Cannot find magnitude column in {gaia_csv}")

    if "x_ffi" in df.columns:
        df["x"] = df["x_ffi"] - crop_bounds["x_min"]
        df["y"] = df["y_ffi"] - crop_bounds["y_min"]

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Gaia DataFrame must have 'x' and 'y' columns (crop-local).")

    return df
