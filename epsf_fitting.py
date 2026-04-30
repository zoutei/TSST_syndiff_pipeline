"""
epsf_fitting.py
===============
Steps 4 & 10 of the SynDiff pipeline:

  Fit an empirical PSF (ePSF) on each difference image using TGLC's
  get_psf / fit_psf, tiling the image into tile_nx × tile_ny sub-regions.

Uses ``tglc.effective_psf`` (TGLC — TESS Gaia Light Curve toolkit).
Install or clone TGLC and ensure ``tglc`` is importable (e.g. ``PYTHONPATH``).
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
from joblib import Parallel, delayed

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
# ── TGLC import helper ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _get_tglc():
    """Import tglc modules needed for ePSF fitting."""
    try:
        import tglc.ffi as tglc_ffi
        from tglc.effective_psf import get_psf, fit_psf
        return tglc_ffi, get_psf, fit_psf
    except ImportError as exc:
        raise ImportError(
            "The ``tglc`` package is required for ePSF fitting. "
            "Install TGLC or add its source tree to PYTHONPATH."
        ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# ── Gaia catalog preparation ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

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

    with fits.open(median_mask_path) as hdul:
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
# ── CustomSource (compatible with TGLC's get_psf / fit_psf) ──────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class CustomSource:
    """
    Minimal Source-compatible object for tglc.effective_psf.get_psf / fit_psf.

    When source.__class__ is temporarily set to tglc.ffi.Source before calling
    get_psf, the 6-DOF background model is enabled (uses source.mask for column
    correction and source.gaia for star positions).

    Tiled ePSF driver; same mathematics as standard TGLC/TGLC-FFI workflows.
    """

    PEDESTAL = 100.0  # counts added to lift background pixels for fitting

    def __init__(self, image_tile: np.ndarray,
                 gaia_tile: pd.DataFrame,
                 col_corr_1d: np.ndarray,
                 sector: int = 20):
        """
        Parameters
        ----------
        image_tile  : 2D array (tile_size, tile_size)
        gaia_tile   : DataFrame with tess_flux_ratio, x_local, y_local, tess_mag
                      (coordinates in tile-local pixels)
        col_corr_1d : 1D array of length tile_size (column correction values)
        sector      : int TESS sector
        """
        import numpy.ma as ma

        h, w = image_tile.shape
        if h != w:
            raise ValueError(
                "image_tile must be square for TGLC get_psf/fit_psf "
                f"(got {h}×{w})"
            )
        if len(col_corr_1d) != w:
            raise ValueError(
                f"col_corr_1d length {len(col_corr_1d)} != tile width {w}"
            )
        self.size   = h
        self.sector = sector
        self.time   = np.array([0.0])

        # Shift image by pedestal before fitting (absorbed by 6-DOF background)
        self.flux = (image_tile + self.PEDESTAL)[np.newaxis, :, :].copy()

        # mask: masked_array; .data = column correction, .mask = bad pixels
        mask_data_2d = np.tile(col_corr_1d[:w], (h, 1))
        bad_pix      = (mask_data_2d == 0)
        self.mask    = ma.masked_array(mask_data_2d, mask=bad_pix)

        # gaia: astropy Table with fields expected by get_psf
        t = Table()
        t["tess_flux_ratio"]        = np.array(gaia_tile["tess_flux_ratio"])
        t["tess_mag"]               = np.array(gaia_tile["tess_mag"])
        t[f"sector_{sector}_x"]     = np.array(gaia_tile["x_local"])
        t[f"sector_{sector}_y"]     = np.array(gaia_tile["y_local"])
        self.gaia = t


# ═══════════════════════════════════════════════════════════════════════════════
# ── Tile machinery ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _make_tile_grid(ny: int, nx: int, tile_ny: int, tile_nx: int) -> list:
    """
    Return list of (r0, c0, tile_size) tuples for a square tile grid.

    ``tile_size = min(ny // tile_ny, nx // tile_nx)`` so each tile is square,
    as required by TGLC ``get_psf`` / ``fit_psf`` (they use ``source.size`` as
    one side and assume ``height == width``). Uncovered strips at the image
    right/top match the ePSF notebook ``make_tile_grid``.
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


def _build_tile_catalog(gaia_df: pd.DataFrame,
                         r0: int, c0: int,
                         tile_size: int,
                         psf_size: int, os_factor: int,
                         margin: int = None) -> pd.DataFrame:
    """
    Extract Gaia stars falling within (or near) the current tile and translate
    to tile-local coordinates.
    """
    if margin is None:
        margin = psf_size * os_factor

    x_lo = c0 - margin
    x_hi = c0 + tile_size + margin
    y_lo = r0 - margin
    y_hi = r0 + tile_size + margin

    in_tile = (
        (gaia_df["x"] >= x_lo) & (gaia_df["x"] < x_hi) &
        (gaia_df["y"] >= y_lo) & (gaia_df["y"] < y_hi)
    )
    tile_cat = gaia_df[in_tile].copy()
    tile_cat = tile_cat.rename(columns={"x": "x_local", "y": "y_local"})
    tile_cat["x_local"] = tile_cat["x_local"] - c0
    tile_cat["y_local"] = tile_cat["y_local"] - r0
    return tile_cat.reset_index(drop=True)


def _fit_one_tile(image: np.ndarray,
                  gaia_df: pd.DataFrame,
                  col_corr_2d: np.ndarray,
                  r0: int, c0: int, tile_size: int,
                  cfg,
                  tglc_ffi, get_psf, fit_psf,
                  diag: Optional[dict] = None) -> np.ndarray:
    """
    Fit ePSF for a single tile.

    Returns
    -------
    1D ndarray of shape (over_size²,) — normalized ePSF coefficients,
    or NaN array if fitting failed.
    """
    over_size = 2 * cfg.psf_size + 1
    nan_epsf  = np.full(over_size ** 2, np.nan)

    # Extract square tile (TGLC requires height == width)
    tile_img    = image[r0:r0 + tile_size, c0:c0 + tile_size]
    tile_corr   = col_corr_2d[r0:r0 + tile_size, c0:c0 + tile_size]
    col_corr_1d = tile_corr[0, :]  # use first row as representative 1D correction

    # Stars for this tile
    tile_cat = _build_tile_catalog(
        gaia_df, r0, c0, tile_size,
        cfg.psf_size, cfg.epsf_oversample,
    )
    if len(tile_cat) < 3:
        if diag is not None:
            diag["n_starved"] = diag.get("n_starved", 0) + 1
        return nan_epsf

    # Ensure we have tess_flux_ratio
    if "tess_flux_ratio" not in tile_cat.columns:
        tile_cat = add_tess_flux_ratio(tile_cat)

    source = CustomSource(tile_img, tile_cat, col_corr_1d, sector=cfg.sector)

    try:
        # Temporarily spoof class to enable 6-DOF background in get_psf
        source.__class__ = tglc_ffi.Source
        A, _star_info, _over_size, _xr, _yr = get_psf(
            source, factor=cfg.epsf_oversample, psf_size=cfg.psf_size,
        )
        source.__class__ = CustomSource
        e_psf_full = fit_psf(A, source, _over_size, power=0.8, time=0)
    except Exception as exc:
        source.__class__ = CustomSource
        log.debug(f"  tile ({r0},{c0}) fit failed: {exc}")
        if diag is not None:
            diag["n_exc"] = diag.get("n_exc", 0) + 1
            if diag.get("first_exc") is None:
                diag["first_exc"] = str(exc)
        return nan_epsf

    if np.isnan(e_psf_full).any():
        if diag is not None:
            diag["n_nan_psf"] = diag.get("n_nan_psf", 0) + 1
        return nan_epsf

    # Extract just the PSF coefficients (first over_size² elements)
    psf_coeffs = e_psf_full[:over_size ** 2]
    norm = np.sum(psf_coeffs)
    if norm > 0:
        psf_coeffs = psf_coeffs / norm
    return psf_coeffs


# ═══════════════════════════════════════════════════════════════════════════════
# ── Per-frame ePSF fitting ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def fit_epsf_tiled(diff_image: np.ndarray,
                   gaia_df: pd.DataFrame,
                   col_corr_2d: np.ndarray,
                   cfg,
                   frame_label: str = "") -> tuple:
    """
    Fit ePSF on one difference image using a tile_ny × tile_nx grid.

    Parameters
    ----------
    diff_image  : 2D ndarray (ny_crop, nx_crop)
    gaia_df     : pd.DataFrame with x, y (crop-local), tess_mag,
                  tess_flux_ratio, phot_g/bp/rp columns
    col_corr_2d : 2D ndarray (ny_crop, nx_crop) — column correction
    cfg         : SynDiffConfig
    frame_label : str, optional — basename for logging (e.g. diff FITS name)

    Returns
    -------
    epsf_tiles  : ndarray (n_tiles, over_size²) — one normalized ePSF per tile
                  (NaN for failed tiles)
    tile_centers: list of (cx, cy) in crop-local pixels
    """
    tglc_ffi, get_psf, fit_psf = _get_tglc()

    gaia_df = add_tess_flux_ratio(gaia_df)

    ny, nx = diff_image.shape
    tiles = _make_tile_grid(ny, nx, cfg.tile_ny, cfg.tile_nx)
    n_tiles = len(tiles)
    over_size = 2 * cfg.psf_size + 1

    epsf_tiles   = np.full((n_tiles, over_size ** 2), np.nan)
    tile_centers = []

    diag = {
        "n_starved": 0,
        "n_exc": 0,
        "n_nan_psf": 0,
        "first_exc": None,
    }
    n_ok = 0
    for idx, (r0, c0, tile_size) in enumerate(tiles):
        cx = c0 + tile_size / 2
        cy = r0 + tile_size / 2
        tile_centers.append((cx, cy))

        coeffs = _fit_one_tile(
            diff_image, gaia_df, col_corr_2d,
            r0, c0, tile_size, cfg,
            tglc_ffi, get_psf, fit_psf,
            diag=diag,
        )
        epsf_tiles[idx] = coeffs
        if not np.isnan(coeffs).all():
            n_ok += 1

    log.debug(f"  ePSF tiles: {n_ok}/{n_tiles} fitted")
    if n_ok == 0:
        suffix = f" ({frame_label})" if frame_label else ""
        first = diag.get("first_exc")
        extra = f' First TGLC error: "{first}"' if first else ""
        log.warning(
            "ePSF: all %d tiles failed%s — tiles with <3 Gaia stars: %d; "
            "TGLC exceptions: %d; NaN PSF from fit_psf: %d.%s",
            n_tiles,
            suffix,
            diag["n_starved"],
            diag["n_exc"],
            diag["n_nan_psf"],
            extra,
        )

    # Fill failed tiles with median of successful ones
    good_tiles = epsf_tiles[~np.isnan(epsf_tiles).any(axis=1)]
    if len(good_tiles) > 0:
        med = np.median(good_tiles, axis=0)
        for idx in range(n_tiles):
            if np.isnan(epsf_tiles[idx]).any():
                epsf_tiles[idx] = med
    return epsf_tiles, tile_centers


def fit_epsf_all_frames(diff_paths: list,
                         gaia_df: pd.DataFrame,
                         col_corr_2d: np.ndarray,
                         cfg,
                         output_dir: str = None,
                         round_id: int = 1) -> tuple:
    """
    Fit ePSF on every difference image in diff_paths.

    Parameters
    ----------
    diff_paths  : list of str (FITS files from hotpants_runner)
    gaia_df     : pd.DataFrame (crop-local Gaia catalog with tess_flux_ratio)
    col_corr_2d : 2D ndarray  (column correction map for the crop)
    cfg         : SynDiffConfig
    output_dir  : str, optional — if given, saves epsf_stack_r{round_id}.npz
    round_id    : int

    Returns
    -------
    epsf_stack  : ndarray (n_frames, n_tiles, over_size²)
    tile_centers: list of (cx, cy) [same for all frames — from first valid frame]
    ffi_stems   : list of str — stem per ``diff_paths`` row (axis-0 identity)
    epsf_ok     : list of bool — True if difference image loaded and ePSF fitted
    """
    over_size = 2 * cfg.psf_size + 1
    n_tiles   = cfg.tile_ny * cfg.tile_nx
    n_frames  = len(diff_paths)

    epsf_stack   = np.full((n_frames, n_tiles, over_size ** 2), np.nan)
    tile_centers = None
    ffi_stems    = [Path(p).stem for p in diff_paths]
    epsf_ok      = []

    for i, diff_path in enumerate(diff_paths):
        if not os.path.exists(diff_path):
            log.warning(f"  diff frame missing: {diff_path}")
            epsf_ok.append(False)
            continue
        try:
            diff_img = fits.getdata(diff_path).astype(np.float64)
        except Exception as exc:
            log.warning(f"  Cannot load {diff_path}: {exc}")
            epsf_ok.append(False)
            continue

        tiles_i, centers_i = fit_epsf_tiled(
            diff_img, gaia_df, col_corr_2d, cfg,
            frame_label=os.path.basename(diff_path),
        )
        epsf_stack[i] = tiles_i
        if tile_centers is None:
            tile_centers = centers_i
        epsf_ok.append(True)

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            log.info(f"  ePSF round {round_id}: {i + 1}/{n_frames} frames done")

    if tile_centers is None:
        # Fallback: compute from grid geometry
        ny, nx = col_corr_2d.shape
        tiles = _make_tile_grid(ny, nx, cfg.tile_ny, cfg.tile_nx)
        tile_centers = [
            (c0 + ts / 2, r0 + ts / 2) for (r0, c0, ts) in tiles
        ]

    if output_dir:
        save_epsf_stack_bundle(epsf_stack, ffi_stems, output_dir, round_id)

    return epsf_stack, tile_centers, ffi_stems, epsf_ok
