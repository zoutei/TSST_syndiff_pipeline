"""
temporal_smooth.py
==================
Steps 5, 8, 10, 11 of the SynDiff pipeline:

  • Optionally temporally smooth the ePSF stack across all frames (config:
    ``epsf_temporal_smooth``); otherwise use per-frame fits with all-NaN tile repair.
  • Compute per-template-group median ePSFs from that stack.
  • Temporally smooth background stacks using TESSreduce-style
    :class:`~syndiff_pipeline.adaptive_background.AdaptiveBackground` (adaptive
    median filter along time; TESSVectors Earth/Moon angles).

ePSF time-axis filtering uses ``scipy.ndimage.uniform_filter1d`` when
``epsf_temporal_smooth`` is true (``temporal_smooth_window``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .adaptive_background import AdaptiveBackground

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── ePSF temporal smoothing ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_smooth_epsf(epsf_stack: np.ndarray,
                          smooth_window: int) -> np.ndarray:
    """
    Smooth the ePSF stack along the time axis using a uniform filter.

    NaN tiles (fitting failures) are excluded from the running average:
    each tile element is smoothed independently; values that were NaN in
    the original are filled from the running window *before* smoothing.

    Parameters
    ----------
    epsf_stack : ndarray, shape (n_frames, n_tiles, over_size²)
    smooth_window : int  (number of frames in the smoothing window)

    Returns
    -------
    ndarray same shape as epsf_stack, NaN-free (medians fill residual gaps).
    """
    n_frames, n_tiles, n_pix = epsf_stack.shape
    smoothed = np.empty_like(epsf_stack)

    for t in range(n_tiles):
        tile_ts = epsf_stack[:, t, :]        # (n_frames, n_pix)

        # Per-pixel interpolation over NaN frames before smoothing
        for p in range(n_pix):
            series = tile_ts[:, p].copy()
            finite = np.isfinite(series)
            if not finite.any():
                # All NaN tile — leave as NaN, filled later by median
                smoothed[:, t, p] = np.nan
                continue
            if not finite.all():
                # Linear interpolation over NaN gaps
                x = np.arange(n_frames)
                series = np.interp(x, x[finite], series[finite])
            smoothed[:, t, p] = uniform_filter1d(series, size=smooth_window,
                                                  mode="nearest")

    # Final cleanup: tiles still all-NaN → fill with global median ePSF
    n_nan_tiles = 0
    global_med = np.nanmedian(smoothed.reshape(-1, n_pix), axis=0)
    for t in range(n_tiles):
        if np.isnan(smoothed[:, t, :]).all():
            smoothed[:, t, :] = global_med
            n_nan_tiles += 1

    if n_nan_tiles:
        log.warning(f"  {n_nan_tiles} tiles were all-NaN; filled with global median ePSF.")

    return smoothed


def prepare_epsf_stack_no_time_filter(epsf_stack: np.ndarray) -> np.ndarray:
    """
    Use fitted ePSFs without temporal interpolation or uniform filtering.

    Copies ``epsf_stack`` and replaces any tile that is NaN at every frame
    with the global median ePSF (same safety net as the end of
    ``temporal_smooth_epsf``).

    Parameters
    ----------
    epsf_stack : ndarray, shape (n_frames, n_tiles, over_size²)

    Returns
    -------
    ndarray — same shape as input; finite where inputs were finite or repairable.
    """
    n_frames, n_tiles, n_pix = epsf_stack.shape
    out = np.array(epsf_stack, copy=True)

    global_med = np.nanmedian(out.reshape(-1, n_pix), axis=0)
    n_nan_tiles = 0
    for t in range(n_tiles):
        if np.isnan(out[:, t, :]).all():
            out[:, t, :] = global_med
            n_nan_tiles += 1

    if n_nan_tiles:
        log.warning(
            f"  {n_nan_tiles} tiles were all-NaN; filled with global median ePSF."
        )

    return out


def compute_group_epsf(
    epsf_smooth: np.ndarray,
    group_ids: np.ndarray,
    output_dir: str = None,
    group_subdir: str = "group_epsf",
) -> dict:
    """
    Compute a single representative ePSF per template group by taking the
    median across all frames assigned to that group.

    Parameters
    ----------
    epsf_smooth : ndarray (n_frames, n_tiles, over_size²)
    group_ids   : ndarray of int, shape (n_frames,) — ``group_id`` for each
                  ePSF row (same order as ``epsf_smooth`` axis 0)
    output_dir  : str, optional — saves group_epsf_{gid}.npy files
    group_subdir: subdirectory under ``output_dir`` for ``group_epsf_*.npy``

    Returns
    -------
    dict {group_id (int): ndarray (n_tiles, over_size²)}
    """
    group_ids = np.asarray(group_ids)
    if group_ids.shape[0] != epsf_smooth.shape[0]:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} != n_frames "
            f"{epsf_smooth.shape[0]}"
        )
    unique_groups = [g for g in sorted(set(group_ids.tolist())) if g >= 0]

    group_epsf = {}
    for gid in unique_groups:
        frame_mask = group_ids == gid
        if frame_mask.sum() == 0:
            continue
        group_stack = epsf_smooth[frame_mask]          # (n_group_frames, n_tiles, n_pix)
        group_epsf[gid] = np.nanmedian(group_stack, axis=0)  # (n_tiles, n_pix)
        log.info(f"  Group {gid}: {frame_mask.sum()} frames → ePSF shape {group_epsf[gid].shape}")

    if output_dir:
        out_subdir = os.path.join(output_dir, group_subdir)
        os.makedirs(out_subdir, exist_ok=True)
        for gid, epsf in group_epsf.items():
            np.save(os.path.join(out_subdir, f"group_epsf_{gid}.npy"), epsf)
        log.info(f"  Group ePSFs saved to {out_subdir}/")

    return group_epsf


def save_epsf_smooth(
    epsf_smooth: np.ndarray,
    output_dir: str,
    round_id: int,
    ffi_stem: np.ndarray | list,
) -> str:
    """
    Save smoothed ePSF stack to ``epsf_rN_smooth.npz`` with ``ffi_stem`` (required).
    """
    if ffi_stem is None:
        raise TypeError("ffi_stem is required for save_epsf_smooth")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
    np.savez_compressed(
        path,
        stack=np.asarray(epsf_smooth),
        ffi_stem=np.asarray(ffi_stem, dtype=object),
    )
    log.info(f"Smoothed ePSF stack saved to {path}")
    return path


def load_epsf_smooth_stems_only(output_dir: str, round_id: int) -> list | None:
    """Load only ``ffi_stem`` from ``epsf_r{round_id}_smooth.npz``, if present."""
    base = os.path.join(output_dir, f"epsf_r{round_id}_smooth")
    npz_p = base + ".npz"
    if not os.path.isfile(npz_p):
        return None
    z = np.load(npz_p, allow_pickle=True)
    try:
        if "ffi_stem" in z.files:
            return [str(x) for x in z["ffi_stem"].tolist()]
    finally:
        z.close()
    return None


def load_epsf_smooth(output_dir: str, round_id: int) -> tuple:
    """
    Load ``epsf_r{round_id}_smooth.npz``.

    Returns
    -------
    stack : ndarray
    ffi_stem : list of str
    """
    npz_p = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
    if not os.path.isfile(npz_p):
        raise FileNotFoundError(f"No smoothed ePSF at {npz_p}")
    z = np.load(npz_p, allow_pickle=True)
    try:
        stack = np.asarray(z["stack"])
        if "ffi_stem" not in z.files:
            raise ValueError(f"{npz_p!r} missing required array 'ffi_stem'")
        ffi_stem = [str(x) for x in z["ffi_stem"].tolist()]
    finally:
        z.close()
    return stack, ffi_stem


# ═══════════════════════════════════════════════════════════════════════════════
# ── Background temporal smoothing (TESSreduce adaptive median) ─────────────────
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

    Matches each result's ``stem`` to ``wcs_table`` rows via ``path`` or ``filename``.
    """
    if "btjd" not in wcs_table.columns:
        raise ValueError("wcs_table must contain a 'btjd' column")
    if "path" in wcs_table.columns:
        stems_tbl = wcs_table["path"].astype(str).map(lambda p: Path(p).stem)
    elif "filename" in wcs_table.columns:
        stems_tbl = wcs_table["filename"].astype(str).map(lambda f: Path(f).stem)
    else:
        raise ValueError("wcs_table must contain 'path' or 'filename'")
    btjd_series = pd.to_numeric(wcs_table["btjd"], errors="coerce")
    stem_to_btjd = dict(zip(stems_tbl, btjd_series))

    out = []
    for r in hotpants_results:
        st = r.get("stem")
        if st is None:
            out.append(np.nan)
        else:
            v = stem_to_btjd.get(st, np.nan)
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
    savgol_window: int = 31,
    savgol_polyorder: int = 2,
    w_min: int = 3,
    w_max: int = 51,
    block_size: int = 5,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Temporal smoothing of the rough background cube via TESSreduce ``AdaptiveBackground``.

    Parameters
    ----------
    bkg_stack : ndarray (n_frames, ny, nx)
    time_btjd : ndarray (n_frames,) — TESS BTJD per frame (same order as stack).
    sector, camera : int — for TESSVectors lookup.
    vector_path : str or None — directory with local TessVectors CSV; else HEASARC.
    method : {'savgol', 'adaptive'}
        ``'savgol'`` — Savitzky–Golay along time (default; faster).
        ``'adaptive'`` — adaptive temporal median (``adaptive_medfilt_3d``).
    savgol_window, savgol_polyorder : int — used when ``method=='savgol'``.
    w_min, w_max : int — odd temporal window bounds for ``method=='adaptive'``.
    block_size : int — spatial block factor for ``method=='adaptive'`` (TESSreduce default 5).
    n_jobs : int — joblib parallelism for ``method=='adaptive'`` (``-1`` = all cores).

    Returns
    -------
    ndarray — same shape as ``bkg_stack``; dtype matches input where possible.
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
    log.info("adaptive_smooth_background: constructing AdaptiveBackground (TESSVectors I/O)...")
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
        log.info("adaptive_smooth_background: calling smoother.smooth(method='adaptive') ...")
        smoother.smooth(
            method="adaptive",
            w_min=w_min,
            w_max=w_max,
            n_jobs=n_jobs,
        )
    log.info("adaptive_smooth_background: smooth() finished; packing output dtype=%s", out_dtype)
    smoothed = np.asarray(smoother.smoothed, dtype=out_dtype)
    return smoothed


def save_bkg_smooth(bkg_smooth: np.ndarray,
                     output_dir: str, round_id: int) -> str:
    """Save smoothed background stack to output_dir/bkg_smooth_rN.npy."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"bkg_smooth_r{round_id}.npy")
    np.save(path, bkg_smooth)
    log.info(f"Smoothed background stack saved to {path}")
    return path


def compute_final_background(
    bkg_smooth_r1: np.ndarray,
    bkg_smooth_r2: np.ndarray,
    smooth_fn: Callable[[np.ndarray], np.ndarray],
    output_dir: str = None,
) -> np.ndarray:
    """
    Combine round-1 and round-2 smooth backgrounds and apply temporal smoothing
    (via ``smooth_fn``) to produce the final background stack.

    ``bkg_final = smooth_fn(bkg_smooth_r1 + bkg_smooth_r2)``

    Parameters
    ----------
    bkg_smooth_r1, bkg_smooth_r2 : ndarray (n_frames, ny, nx)
    smooth_fn : callable — ``combined_cube -> smoothed_cube``
    output_dir : str, optional

    Returns
    -------
    ndarray (n_frames, ny, nx) — bkg_final
    """
    combined = bkg_smooth_r1 + bkg_smooth_r2
    bkg_final = smooth_fn(combined)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "bkg_final.npy")
        np.save(path, bkg_final)
        log.info(f"Final background stack saved to {path}")

    return bkg_final
