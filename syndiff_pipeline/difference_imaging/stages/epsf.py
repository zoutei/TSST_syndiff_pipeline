"""
epsf_fitting.py
===============
Fit empirical PSFs (ePSF) on difference images with photutils, tiling each
frame into tile_nx × tile_ny sub-regions and building a per-frame
:class:`~photutils.psf.GriddedPSFModel`.

Persists per-frame ``*_gridded_epsf.npz`` archives, legacy smooth stacks for
downstream stages, and per-template-group median ePSFs.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from syndiff_pipeline.difference_imaging.support.ffi_naming import parse_workspace_frame_stem, strip_fits_suffix, tess_product_id_from_ffi_path

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── ePSF stack bundle (stack + ffi_stem per axis-0 row) ───────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def epsf_stack_bundle_base(output_dir: str, round_id: int) -> str:
    """Base path without extension for ``epsf_stack_r{round_id}.npz``."""
    return os.path.join(output_dir, f"epsf_stack_r{round_id}")


def save_epsf_stack_bundle(
    stack: np.ndarray,
    ffi_stems: list,
    output_dir: str,
    round_id: int,
) -> str:
    """Save epsf stack bundle.
    
    Parameters
    ----------
    stack : np.ndarray
    ffi_stems : list
    output_dir : str
    round_id : int
    
    Returns
    -------
    str"""
    os.makedirs(output_dir, exist_ok=True)
    path = epsf_stack_bundle_base(output_dir, round_id) + ".npz"
    np.savez_compressed(
        path,
        stack=np.asarray(stack),
        ffi_stem=np.asarray(ffi_stems, dtype=object),
    )
    log.info("  ePSF stack saved to %s  shape=%s", path, stack.shape)
    return path


def load_epsf_stack_bundle(output_dir: str, round_id: int) -> tuple:
    """
    Load round ``round_id`` ePSF stack from ``epsf_stack_r{round_id}.npz``.

    Returns
    -------
    stack : ndarray, shape (n_frames, n_tiles, n_pix)
    ffi_stem : list of str
    """
    base = epsf_stack_bundle_base(output_dir, round_id)
    npz_p = base + ".npz"
    if not os.path.isfile(npz_p):
        raise FileNotFoundError(f"No ePSF stack at {npz_p}")
    z = np.load(npz_p, allow_pickle=True)
    try:
        stack = np.asarray(z["stack"])
        if "ffi_stem" not in z.files:
            raise ValueError(f"{npz_p!r} missing required array 'ffi_stem'")
        raw = z["ffi_stem"]
        ffi_stem = [str(x) for x in raw.tolist()]
    finally:
        z.close()
    return stack, ffi_stem


# ═══════════════════════════════════════════════════════════════════════════════
# ── Smoothed / repaired ePSF stack (saved alongside fitting workspace) ────────
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_epsf_stack(epsf_stack: np.ndarray) -> np.ndarray:
    """
    Per-frame fitted ePSFs with all-NaN tiles repaired (global median ePSF per pixel).

    No temporal filtering — aligns with using raw fits before downstream grouping.

    Parameters
    ----------
    epsf_stack : ndarray, shape (n_frames, n_tiles, over_size²)
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
            "  prepare_epsf_stack: %d tiles were all-NaN; filled with global median ePSF.",
            n_nan_tiles,
        )

    return out


def compute_group_epsf(
    epsf_smooth: np.ndarray,
    group_ids: np.ndarray,
    output_dir: str | None = None,
    group_subdir: str = "group_epsf",
) -> dict:
    """
    One median ePSF per template group (median across frames in that group).

    Parameters
    ----------
    epsf_smooth : ndarray (n_frames, n_tiles, over_size²)
    group_ids : ndarray (n_frames,)
    output_dir : str, optional — saves ``group_epsf_{gid}.npy`` under ``group_subdir``
    """
    group_ids = np.asarray(group_ids)
    if group_ids.shape[0] != epsf_smooth.shape[0]:
        raise ValueError(
            f"group_ids length {group_ids.shape[0]} != n_frames "
            f"{epsf_smooth.shape[0]}"
        )
    unique_groups = [g for g in sorted(set(group_ids.tolist())) if g >= 0]

    group_epsf: dict[int, np.ndarray] = {}
    for gid in unique_groups:
        frame_mask = group_ids == gid
        if frame_mask.sum() == 0:
            continue
        group_stack = epsf_smooth[frame_mask]
        group_epsf[gid] = np.nanmedian(group_stack, axis=0)
        log.info(
            "  Group %s: %d frames → ePSF shape %s",
            gid,
            int(frame_mask.sum()),
            group_epsf[gid].shape,
        )

    if output_dir:
        out_subdir = os.path.join(output_dir, group_subdir)
        os.makedirs(out_subdir, exist_ok=True)
        for gid, epsf in group_epsf.items():
            np.save(os.path.join(out_subdir, f"group_epsf_{gid}.npy"), epsf)
        log.info("  Group ePSFs saved to %s/", out_subdir)

    return group_epsf


def save_epsf_smooth(
    epsf_smooth: np.ndarray,
    output_dir: str,
    round_id: int,
    ffi_stem: np.ndarray | list,
) -> str:
    """Save stack to ``epsf_rN_smooth.npz`` with ``ffi_stem`` (one per axis-0 row)."""
    if ffi_stem is None:
        raise TypeError("ffi_stem is required for save_epsf_smooth")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
    np.savez_compressed(
        path,
        stack=np.asarray(epsf_smooth),
        ffi_stem=np.asarray(ffi_stem, dtype=object),
    )
    log.info("Smoothed ePSF stack saved to %s", path)
    return path


def load_epsf_smooth_stems_only(output_dir: str, round_id: int) -> list | None:
    """Load only ``ffi_stem`` from ``epsf_r{round_id}_smooth.npz``, if present."""
    npz_p = os.path.join(output_dir, f"epsf_r{round_id}_smooth.npz")
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
    """Load ``epsf_r{round_id}_smooth.npz``. Returns ``stack``, ``ffi_stem`` list."""
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
# ── Per-frame gridded ePSF fitting (photutils) ────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# shared_mask bit values used for ePSF star rejection (see masking.Source_mask):
#   value 2 — very bright star crosses (Big_sat)
#   value 4 — TESS straps
# Catalog Gaia sources (value 1) must NOT be excluded — those are the ePSF stars.
EPSF_SHARED_MASK_BITS = 2 | 4


def _load_shared_mask_2d(shared_mask_path: str | None, shape: tuple[int, int]) -> np.ndarray | None:
    """
    Load boolean bad-pixel mask for ePSF star extraction, if available.

    Only bits with values 2 and 4 (bright-star crosses and straps) are used.
    Catalog-source bit 1 is ignored so Gaia ePSF stars are not removed.
    """
    if not shared_mask_path or not os.path.isfile(shared_mask_path):
        return None
    try:
        data = fits.getdata(shared_mask_path)
        if data is None:
            return None
        mask = np.asarray(data)
        if mask.shape != shape:
            return None
        return (mask.astype(np.int64) & EPSF_SHARED_MASK_BITS) != 0
    except Exception as exc:
        log.debug("shared mask for ePSF not loaded: %s", exc)
        return None


def fit_epsf_all_frames(diff_paths: list,
                         gaia_df: pd.DataFrame,
                         col_corr_2d: np.ndarray,
                         cfg,
                         epsf,
                         output_dir: str = None,
                         round_id: int = 1,
                         *,
                         shared_mask_path: str | None = None,
                         diff_log_path: str | None = None,
                         epsf_label: str | None = None,
                         diffs_input: str | None = None) -> tuple:
    """
    Fit gridded ePSF on every difference image in diff_paths.

    Parameters
    ----------
    diff_paths  : list of str (FITS files from hotpants)
    gaia_df     : pd.DataFrame (crop-local Gaia catalog)
    col_corr_2d : 2D ndarray — legacy arg; mask uses shared_mask when given
    cfg         : SynDiffConfig
    epsf        : EpsfParams
    output_dir  : str, optional
    round_id    : int
    shared_mask_path : optional path to shared_mask.fits for star masking

    Returns
    -------
    epsf_stack, tile_centers, ffi_stems, epsf_ok
    """
    from syndiff_pipeline.difference_imaging.stages import gridded_epsf

    mask_2d = None
    first_path = next((p for p in diff_paths if p and os.path.exists(p)), None)
    if shared_mask_path and first_path:
        shape = fits.getdata(first_path).shape
        mask_2d = _load_shared_mask_2d(shared_mask_path, shape)
    elif col_corr_2d is not None and first_path is None and col_corr_2d is not None:
        mask_2d = col_corr_2d <= 0

    if output_dir is None:
        raise ValueError("fit_epsf_all_frames requires output_dir for gridded ePSF")

    result = gridded_epsf.fit_gridded_epsf_all_frames(
        diff_paths,
        gaia_df,
        cfg,
        epsf,
        output_dir,
        mask_2d=mask_2d,
        round_id=round_id,
        diff_log_path=diff_log_path,
        epsf_label=epsf_label,
        diffs_input=diffs_input,
        publish_scc=bool(getattr(cfg, "publish_scc", False)),
    )
    epsf_stack, tile_centers, ffi_stems, epsf_ok = result
    save_epsf_stack_bundle(epsf_stack, ffi_stems, output_dir, round_id)
    return epsf_stack, tile_centers, ffi_stems, epsf_ok


def fit_epsf_tiled(diff_image: np.ndarray,
                   gaia_df: pd.DataFrame,
                   col_corr_2d: np.ndarray,
                   cfg,
                   epsf,
                   frame_label: str = "") -> tuple:
    """Fit gridded ePSF on a single difference image (test / debug helper)."""
    from syndiff_pipeline.difference_imaging.stages import gridded_epsf

    mask_2d = None
    if col_corr_2d is not None and col_corr_2d.shape == diff_image.shape:
        mask_2d = col_corr_2d <= 0
    _model, tile_centers, stack = gridded_epsf.build_gridded_psf_for_frame(
        diff_image,
        gaia_df,
        epsf,
        mask_2d=mask_2d,
        frame_label=frame_label,
    )
    if stack is None:
        n_tiles = epsf.tile_ny * epsf.tile_nx
        over_size = 2 * epsf.psf_size + 1
        return (
            np.full((n_tiles, over_size ** 2), np.nan),
            tile_centers,
        )
    flat = gridded_epsf.stack_from_gridded_cube(stack)
    return flat, tile_centers


def tess_mag_from_gaia_phot(
    g: np.ndarray, bp: np.ndarray, rp: np.ndarray
) -> np.ndarray:
    """
    TESS magnitude from Gaia ``phot_g_mean_mag`` / BP / RP (TGLC polynomial).

    Where G, BP, and RP are finite: color polynomial (same as TGLC ``ffi.py``).
    Otherwise (missing color): ``tess_mag = G - 0.430`` when G is finite.
    Non-finite polynomial values for valid color use ``G - 0.430``.

    For a catalog DataFrame including ``tess_flux`` / ``tess_flux_ratio``, use
    :func:`add_tess_flux_ratio` instead.
    """
    g = np.asarray(g, dtype=float)
    bp = np.asarray(bp, dtype=float)
    rp = np.asarray(rp, dtype=float)
    tess = np.full_like(g, np.nan, dtype=float)
    color_ok = np.isfinite(g) & np.isfinite(bp) & np.isfinite(rp)
    dif = np.where(color_ok, bp - rp, np.nan)
    tess_poly = (
        g
        - 0.00522555 * dif ** 3
        + 0.0891337 * dif ** 2
        - 0.633923 * dif
        + 0.0324473
    )
    tess[color_ok] = tess_poly[color_ok]
    bad_poly = color_ok & ~np.isfinite(tess_poly)
    tess[bad_poly] = g[bad_poly] - 0.430
    g_only = np.isfinite(g) & ~color_ok
    tess[g_only] = g[g_only] - 0.430
    return tess


def add_tess_flux_ratio(gaia_df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of ``gaia_df`` with ``tess_mag`` (via :func:`tess_mag_from_gaia_phot`),
    ``tess_flux``, and ``tess_flux_ratio``.

    Merges any pre-existing ``tess_mag`` with photometry: NaN rows are filled
    from G/BP/RP; see :func:`tess_mag_from_gaia_phot` for the conversion.

    Parameters
    ----------
    gaia_df : pd.DataFrame
        Must have ``phot_g_mean_mag`` when ``tess_mag`` is absent or all NaN.
        ``phot_bp_mean_mag`` and ``phot_rp_mean_mag`` are optional (per column or per row).
        Optionally ``tess_mag`` (pre-computed; NaNs may be filled).

    Returns
    -------
    pd.DataFrame with columns ``tess_mag``, ``tess_flux``, ``tess_flux_ratio``.
    """
    df = gaia_df.copy()
    n = len(df)

    if "phot_g_mean_mag" not in df.columns:
        if "tess_mag" in df.columns and df["tess_mag"].notna().any():
            g = np.full(n, np.nan, dtype=float)
            bp = np.full(n, np.nan, dtype=float)
            rp = np.full(n, np.nan, dtype=float)
        else:
            raise ValueError(
                "add_tess_flux_ratio requires phot_g_mean_mag when tess_mag is "
                "absent or all NaN"
            )
    else:
        g = df["phot_g_mean_mag"].values.astype(float)

    if "phot_bp_mean_mag" in df.columns:
        bp = df["phot_bp_mean_mag"].values.astype(float)
    else:
        bp = np.full(n, np.nan, dtype=float)

    if "phot_rp_mean_mag" in df.columns:
        rp = df["phot_rp_mean_mag"].values.astype(float)
    else:
        rp = np.full(n, np.nan, dtype=float)

    synthesized = tess_mag_from_gaia_phot(g, bp, rp)
    if "tess_mag" not in df.columns or df["tess_mag"].isna().all():
        df["tess_mag"] = synthesized
    else:
        tm = df["tess_mag"].values.astype(float)
        fill = ~np.isfinite(tm)
        if fill.any():
            tm[fill] = synthesized[fill]
        df["tess_mag"] = tm

    df["tess_flux"] = 10.0 ** (-df["tess_mag"].values / 2.5)
    max_flux = np.nanmax(df["tess_flux"].values)
    df["tess_flux_ratio"] = df["tess_flux"] / max_flux if max_flux > 0 else df["tess_flux"]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ── Median mask (column correction) ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Bad CCD columns in CCD coordinates (0-based)
_BAD_COLS_CCD = [171, 172, 1024]


def _column_correction_from_full_vals(
    vals: np.ndarray, x0: int, x1: int, nx_crop: int
) -> np.ndarray:
    """Per-crop-column correction: crop column j maps to full-chip column ``x0 + j``."""
    out = np.ones(nx_crop, dtype=np.float64)
    for j in range(nx_crop):
        ffi_x = x0 + j
        if 0 <= ffi_x < len(vals):
            out[j] = float(vals[ffi_x])
    return out


def build_median_mask_correction(median_mask_path: str,
                                  camera: int, ccd: int,
                                  crop_bounds: dict) -> np.ndarray:
    """
    Load the TGLC median mask FITS and extract the column-correction 1D array
    for the crop region.

    The median_mask.fits file has one row per (camera, ccd) combination.
    The correction is tiled into a 2D array matching the crop shape, then
    bad columns are zeroed out.

    Parameters
    ----------
    median_mask_path : str
    camera, ccd : int
    crop_bounds : dict  (from wcs_grouping.get_crop_bounds)

    Returns
    -------
    2D ndarray of shape (ny_crop, nx_crop), float64
        Values are the column correction factors; 0 = bad column.
    """
    ny_crop, nx_crop = crop_bounds["shape"]
    col_corr = np.ones(nx_crop, dtype=np.float64)

    if not os.path.exists(median_mask_path):
        log.warning(f"median_mask.fits not found at {median_mask_path}. "
                    "Using uniform column correction = 1.")
        return np.tile(col_corr, (ny_crop, 1))

    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap

    with open_fits_memmap(median_mask_path) as hdul:
        # Find the row matching (camera, ccd)
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        if data is None:
            log.warning("median_mask.fits has no data. Using uniform correction.")
            return np.tile(col_corr, (ny_crop, 1))

        # Attempt structured table lookup
        cam_col = [c for c in data.dtype.names if "cam" in c.lower()]
        ccd_col = [c for c in data.dtype.names if "ccd" in c.lower()]
        if cam_col and ccd_col:
            row_mask = (data[cam_col[0]] == camera) & (data[ccd_col[0]] == ccd)
            if row_mask.any():
                row = data[row_mask][0]
                # Column correction values are stored after the metadata columns
                val_keys = [k for k in data.dtype.names if k not in cam_col + ccd_col]
                vals = np.array([row[k] for k in val_keys], dtype=np.float64)
                x0, x1 = crop_bounds["x_min"], crop_bounds["x_max"]
                col_corr = _column_correction_from_full_vals(vals, x0, x1, nx_crop)
            else:
                log.warning(f"No median_mask row for camera={camera}, ccd={ccd}.")
        else:
            # Plain 2D array — use row index = (camera-1)*4 + (ccd-1)
            row_idx = (camera - 1) * 4 + (ccd - 1)
            if data.ndim == 2 and row_idx < data.shape[0]:
                vals = data[row_idx].astype(np.float64)
                x0, x1 = crop_bounds["x_min"], crop_bounds["x_max"]
                col_corr = _column_correction_from_full_vals(vals, x0, x1, nx_crop)

    # Zero out known bad columns (convert from CCD coords to crop-local)
    x_min = crop_bounds["x_min"]
    for bad_col in _BAD_COLS_CCD:
        local = bad_col - x_min
        if 0 <= local < nx_crop:
            col_corr[local] = 0.0

    col_corr_2d = np.tile(col_corr, (ny_crop, 1))
    return col_corr_2d


# ═══════════════════════════════════════════════════════════════════════════════
# ── Tile machinery ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tile_grid(ny: int, nx: int, tile_ny: int, tile_nx: int) -> list:
    """
    Return list of (r0, c0, tile_size) tuples for a square tile grid.
    """
    tile_h = ny // tile_ny
    tile_w = nx // tile_nx
    tile_size = min(tile_h, tile_w)
    tiles = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            r0 = i * tile_size
            c0 = j * tile_size
            tiles.append((r0, c0, tile_size))
    return tiles

