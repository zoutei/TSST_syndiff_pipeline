"""
masking.py
==========
``shared_mask`` pipeline stage:

  1. Build a shared bitmask (Gaia catalog, very-bright-star crosses, TESS straps).
  2. Select clean, isolated hotpants reference stars.

Bright-star, saturation-cross, and strap masking follow the **TESSreduce**
conventions; the low-level helpers are vendored into this module.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from copy import deepcopy
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from astropy.stats import sigma_clip
from joblib import Parallel, delayed
import multiprocessing

from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Vendored from TESSreduce cat_mask ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def size_limit(x, y, image):
    """Return boolean index of pixels inside image boundaries."""
    yy, xx = image.shape
    return (y > 0) & (y < yy - 1) & (x > 0) & (x < xx - 1)


def gaia_auto_mask(table: pd.DataFrame, Image: np.ndarray, scale: float = 1.0) -> dict:
    """
    Build a magnitude-keyed mask dict from a Gaia catalog.
    Each magnitude bin gets a square kernel of increasing size.
    Returns dict with key 'all' containing the union mask.

    Expects table columns: x, y, mag (all in crop-local pixels).
    """
    image = np.zeros_like(Image)
    x = (np.round(table["x"].values, 0)).astype(int)
    y = (np.round(table["y"].values, 0)).astype(int)
    m = table["mag"].values
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]

    masks = {}
    mags = [
        [18, 17], [17, 16], [16, 15], [15, 14], [14, 13.5],
        [13.5, 12], [12, 10], [10, 9], [9, 8], [8, 7],
    ]
    sizes = (np.array([3, 4, 5, 6, 7, 8, 10, 14, 16, 18]) * scale).astype(int)

    for i, mag_range in enumerate(mags):
        mag_ind = (m > mag_range[1]) & (m <= mag_range[0])
        magim = np.zeros_like(image)
        magim[y[mag_ind], x[mag_ind]] = 1.0
        sz = sizes[i]
        if sz > 0:
            k = np.ones((sz, sz))
            conv = fftconvolve(magim, k, mode="same")
            masks[str(mag_range[0])] = (conv > 0.1) * 1.0

    masks["all"] = np.zeros_like(image, dtype=float)
    for key in masks:
        masks["all"] += masks[key]
    masks["all"] = (masks["all"] > 0.1) * 1.0
    return masks


def Big_sat(table: pd.DataFrame, Image: np.ndarray, scale: float = 1.0) -> list:
    """
    Build cross + circular body masks for stars brighter than mag 7.

    Expects table columns: x, y, mag (crop-local pixels).  Gaia and BSC rows
    may be concatenated (TESSreduce ``Cat_mask`` convention).
    Returns list of 2D mask arrays.
    """
    image = np.zeros_like(Image)
    sat = table[table["mag"].values < 7].copy()
    x = (np.round(sat["x"].values, 0)).astype(int)
    y = (np.round(sat["y"].values, 0)).astype(int)
    m = sat["mag"].values
    ind = size_limit(x, y, image)
    x, y, m = x[ind], y[ind], m[ind]

    satmasks = []
    for i in range(len(x)):
        mag = m[i]
        mask = np.zeros_like(image, dtype=float)

        body = int(13 * scale)
        length = int(20 * scale)
        width = int(3 * scale)

        if mag <= 5 and mag > 4:
            body = int(15 * scale)
            length = int(60 * scale)
            width = int(5 * scale)
        elif mag <= 4:
            body = int(22 * scale)
            length = int(115 * scale)
            width = int(7 * scale)

        kernel = np.zeros((body * 2 + 1, body * 2 + 1))
        yy, xx = np.where(kernel == 0)
        dist = np.sqrt((yy - body) ** 2 + (xx - body) ** 2)
        kernel[yy[dist <= body + 1], xx[dist <= body + 1]] = 1
        stamp = np.zeros_like(image)
        stamp[y[i], x[i]] = 1
        conv = fftconvolve(stamp, kernel, mode="same")
        mask = (conv > 0.1) * 1.0

        for r0, r1, c0, c1 in [
            (max(0, y[i] - length), y[i] + length, max(0, x[i] - width), x[i] + width),
            (max(0, y[i] - width), y[i] + width, max(0, x[i] - length), x[i] + length),
        ]:
            mask[r0:r1, c0:c1] = 1

        satmasks.append(mask)

    return satmasks


def Strap_mask(Image: np.ndarray, col_offset: int, straps_csv: str,
               size: int = 4) -> np.ndarray:
    """
    Build a strap mask for TESS CCDs.

    Parameters
    ----------
    Image : 2D array (crop-local, used for shape)
    col_offset : int
        x_min of the crop region in FFI coordinates, used to align strap columns.
    straps_csv : str
        Path to tess_straps.csv (column 'Column' lists strap pixel columns in CCD coords).
    size : int
        Kernel width for strap dilation.
    """
    strap_mask = np.zeros_like(Image)

    if not straps_csv or not os.path.isfile(straps_csv):
        from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
            tess_straps_csv,
        )

        straps_csv = str(tess_straps_csv())

    if not os.path.exists(straps_csv):
        log.warning(f"tess_straps.csv not found at {straps_csv}. Strap masking disabled.")
        return strap_mask

    straps_df = pd.read_csv(straps_csv)
    # Columns in the CSV are in CCD coordinates; translate to crop-local
    straps = straps_df["Column"].values - col_offset + 44
    strap_in_crop = straps[(straps > 0) & (straps < Image.shape[1])]
    strap_mask[:, strap_in_crop.astype(int)] = 1

    k_size = max(1, int(size))
    if k_size % 2 == 0:
        k_size += 1
    big_strap = fftconvolve(strap_mask, np.ones((k_size, k_size)), mode="same") > 0.5
    return big_strap.astype(int)


def detector_edge_mask(
    shape: tuple[int, int],
    crop_bounds: dict,
    *,
    nx: int,
    ny: int,
    x_left_dead: int = 44,
    x_right_dead: int = 44,
    y_edge_strip: int = 30,
) -> np.ndarray:
    """
    Mask TESS detector non-science regions intersecting the crop (bit 8).

    Usable FFI area is ``x in [x_left_dead, nx - x_right_dead)``,
    ``y in [0, ny - y_edge_strip)``.
    """
    ny_crop, nx_crop = shape
    x_min = int(crop_bounds["x_min"])
    y_min = int(crop_bounds["y_min"])
    x_usable_lo = int(x_left_dead)
    x_usable_hi = nx - int(x_right_dead)
    y_usable_hi = ny - int(y_edge_strip)

    edge = np.zeros((ny_crop, nx_crop), dtype=bool)
    for j in range(nx_crop):
        x_ffi = x_min + j
        if x_ffi < x_usable_lo or x_ffi >= x_usable_hi:
            edge[:, j] = True
    for i in range(ny_crop):
        y_ffi = y_min + i
        if y_ffi >= y_usable_hi:
            edge[i, :] = True
    return edge


def Cat_mask(data_image: np.ndarray,
             gaia_df: pd.DataFrame,
             straps_csv: str,
             maglim: float = 13.0,
             scale: float = 1.0,
             strapsize: int = 6,
             col_offset: int = 0,
             bsc_df: pd.DataFrame | None = None) -> np.ndarray:
    """
    Build the full bitmask for one image.

    Bit layout:
      bit 1 (value 1) — catalog sources (gaia_auto_mask)
      bit 2 (value 2) — very bright star crosses (Big_sat, mag < 7; Gaia + BSC)
      bit 4 (value 4) — TESS straps
      bit 8 (value 8) — detector edge dead zones (applied in make_shared_mask)
      bit 16 (value 16) — insufficient PS1 coverage (applied in make_shared_mask)

    Parameters
    ----------
    data_image : 2D ndarray (crop-local)
    gaia_df : pd.DataFrame  with columns x, y, mag (crop-local coords)
    straps_csv : str
    maglim : float
        Only mask stars with mag < maglim.
    scale : float
        Scale factor for mask sizes.
    strapsize : int
        Strap kernel width.
    col_offset : int
        x_min of the crop (for strap alignment).
    bsc_df : pd.DataFrame, optional
        Bright Star Catalogue rows in crop-local coords with ``vmag``.

    Returns
    -------
    int ndarray of same shape as data_image
    """
    gaia_sub = gaia_df[gaia_df["mag"] < maglim].copy()

    mg = gaia_auto_mask(gaia_sub, data_image, scale)
    bit1 = (mg["all"] > 0).astype(int)  # catalog mask

    sat_table = gaia_sub
    if bsc_df is not None and len(bsc_df) > 0:
        bsc_sat = bsc_df.copy()
        bsc_sat["mag"] = bsc_sat["vmag"]
        sat_table = pd.concat(
            [gaia_sub, bsc_sat[["x", "y", "mag"]]],
            ignore_index=True,
        )
    sat_list = Big_sat(sat_table, data_image, scale)
    if len(sat_list) > 0:
        bit2 = (np.nansum(sat_list, axis=0) > 0).astype(int) * 2
    else:
        bit2 = np.zeros_like(data_image, dtype=int)

    if strapsize > 0:
        bit4 = Strap_mask(data_image, col_offset, straps_csv, size=strapsize).astype(int) * 4
    else:
        bit4 = np.zeros_like(data_image, dtype=int)

    return bit1 | bit2 | bit4


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
            p = interp1d(x[finite], y[finite], bounds_error=False,
                         fill_value=np.nan, kind="nearest")(x)
    return p


def _calc_strap_factor(i, breaks, size, av_size, normals, data):
    """Compute the QE correction factor for one strap group."""
    qe = np.ones_like(data) * np.nan
    b = int(breaks[i])
    size = size.astype(int)
    nind = np.append(normals[b - av_size:b], normals[b:b + av_size]) + 1
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


def correct_straps(Image: np.ndarray, mask: np.ndarray,
                   av_size: int = 5, parallel: bool = True) -> np.ndarray:
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


# ═══════════════════════════════════════════════════════════════════════════════
# ── New pipeline functions ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def ps1_coverage_mask(count_crop: np.ndarray, *, min_hit_count: int = 5000) -> np.ndarray:
    """True where PS1 hit count is below *min_hit_count*."""
    return count_crop < int(min_hit_count)


def make_shared_mask(ref_image: np.ndarray,
                     gaia_df: pd.DataFrame,
                     crop_bounds: dict,
                     straps_csv: str,
                     maglim: float = 13.0,
                     strapsize: int = 6,
                     output_dir: str = None,
                     *,
                     ref_ffi_path: str | None = None,
                     bsc_catalog_path: str | None = None,
                     nx: int | None = None,
                     ny: int | None = None,
                     x_left_dead: int = 44,
                     x_right_dead: int = 44,
                     y_edge_strip: int = 30,
                     template_path: str | None = None,
                     ps1_min_hit_count: int = 5000) -> np.ndarray:
    """
    Build the shared bitmask for the cropped region.

    Parameters
    ----------
    ref_image : 2D ndarray  (already cropped to crop_bounds)
    gaia_df : pd.DataFrame
        Must have columns 'x', 'y' in crop-local coords and 'mag'.
    crop_bounds : dict  (from wcs_grouping.get_crop_bounds)
    straps_csv : str    (path to tess_straps.csv)
    maglim : float      (mask stars brighter than this)
    strapsize : int
    output_dir : str, optional — if given, writes shared_mask.fits
    ref_ffi_path : str, optional
        Reference FFI for BSC WCS projection (required when BSC is used).
    bsc_catalog_path : str, optional
        Override path to decompressed BSC5 ``catalog``; default is bundled asset.
    nx, ny : int, optional
        Full FFI dimensions for detector edge masking.
    x_left_dead, x_right_dead, y_edge_strip : int
        Dead-zone strip sizes (TESS FFI layout).
    template_path : str, optional
        Reference WCS-group syndiff template; when set, masks pixels with
        ``COUNT < ps1_min_hit_count`` (bit 16).
    ps1_min_hit_count : int
        Minimum PS1 hit count per TESS pixel; ``0`` disables PS1 coverage masking.

    Returns
    -------
    int ndarray, same shape as ref_image
    """
    bsc_in_crop = None
    if ref_ffi_path is not None:
        from syndiff_pipeline.common.bsc_catalog import (
            load_bright_star_catalog,
            project_bsc_to_crop,
        )

        bsc_full = load_bright_star_catalog(bsc_catalog_path)
        bsc_in_crop = project_bsc_to_crop(bsc_full, ref_ffi_path, crop_bounds)
        if len(bsc_in_crop):
            log.info("  BSC: %d stars in crop for saturation crosses", len(bsc_in_crop))

    mask = Cat_mask(
        data_image=ref_image,
        gaia_df=gaia_df,
        straps_csv=straps_csv,
        maglim=maglim,
        scale=1.0,
        strapsize=strapsize,
        col_offset=crop_bounds["x_min"],
        bsc_df=bsc_in_crop,
    )

    if nx is not None and ny is not None:
        edge = detector_edge_mask(
            ref_image.shape,
            crop_bounds,
            nx=int(nx),
            ny=int(ny),
            x_left_dead=int(x_left_dead),
            x_right_dead=int(x_right_dead),
            y_edge_strip=int(y_edge_strip),
        )
        mask = mask | (edge.astype(np.int16) * 8)

    if template_path and int(ps1_min_hit_count) > 0:
        from syndiff_pipeline.common.template_coverage import load_template_count_cropped

        count_crop = load_template_count_cropped(template_path, crop_bounds)
        if count_crop is not None:
            if count_crop.shape != ref_image.shape:
                raise ValueError(
                    f"Template COUNT crop shape {count_crop.shape} != ref_image "
                    f"{ref_image.shape} for {template_path!r}"
                )
            no_ps1 = ps1_coverage_mask(count_crop, min_hit_count=ps1_min_hit_count)
            mask = mask | (no_ps1.astype(np.int16) * 16)
            log.info(
                "  PS1 coverage: %d pixels with COUNT < %d",
                int(no_ps1.sum()),
                int(ps1_min_hit_count),
            )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, SHARED_MASK_FITS_BASENAME)
        hdu = fits.PrimaryHDU(mask.astype(np.int16))
        hdu.writeto(out_path, overwrite=True)
        log.info(f"Shared mask written to {out_path}  "
                 f"(masked pixels: {(mask > 0).sum()} / {mask.size})")

    return mask


def select_hotpants_ref_stars(gaia_df: pd.DataFrame,
                               crop_bounds: dict,
                               mag_min: float = 13.5,
                               mag_max: float = 14.5,
                               isolation_mag: float = 13.5,
                               isolation_radius_px: int = 8,
                               separation_px: int = 10,
                               output_dir: str = None) -> pd.DataFrame:
    """
    Select clean, isolated reference stars for hotpants stamp fitting.

    Expects gaia_df to have columns: x, y, tess_mag (crop-local coords).

    Algorithm
    ---------
    Phase 1 — Isolation check:
        Excluders = all Gaia stars with tess_mag < isolation_mag.
        For each candidate (mag_min ≤ tess_mag ≤ mag_max):
            reject if any excluder (other than itself, dist > 0.5 px)
            falls within isolation_radius_px pixels.

    Phase 2 — Pairwise separation (greedy keep-brightest):
        Sort surviving candidates by tess_mag ascending (smallest = brightest).
        Iterate: keep a star only if no already-kept star is within
        separation_px pixels.

    Parameters
    ----------
    gaia_df : pd.DataFrame  with x, y (crop-local), tess_mag
    crop_bounds : dict
    mag_min, mag_max : float
    isolation_mag : float
    isolation_radius_px : int
    separation_px : int
    output_dir : str, optional

    Returns
    -------
    pd.DataFrame  with subset of gaia_df rows; reset index.
    """
    # Restrict to stars within crop bounds
    ny, nx = crop_bounds["shape"]
    in_bounds = (
        (gaia_df["x"] >= 0) & (gaia_df["x"] < nx) &
        (gaia_df["y"] >= 0) & (gaia_df["y"] < ny)
    )
    gaia_crop = gaia_df[in_bounds].copy().reset_index(drop=True)

    # Candidates: magnitude in [mag_min, mag_max]
    cand_mask = (gaia_crop["tess_mag"] >= mag_min) & (gaia_crop["tess_mag"] <= mag_max)
    candidates = gaia_crop[cand_mask].copy().reset_index(drop=True)

    # Excluders: all stars brighter than isolation_mag
    excluders = gaia_crop[gaia_crop["tess_mag"] < isolation_mag].copy().reset_index(drop=True)

    # ── Phase 1: isolation filter ─────────────────────────────────────────────
    exc_xy = np.column_stack([excluders["x"].values, excluders["y"].values])
    keep_phase1 = []

    for idx, row in candidates.iterrows():
        cx, cy = row["x"], row["y"]
        if len(exc_xy) > 0:
            dists = np.sqrt((exc_xy[:, 0] - cx) ** 2 + (exc_xy[:, 1] - cy) ** 2)
            # Ignore self (distance < 0.5 px)
            nearby = dists[(dists > 0.5) & (dists <= isolation_radius_px)]
            if len(nearby) > 0:
                continue
        keep_phase1.append(idx)

    survivors = candidates.loc[keep_phase1].copy()
    log.info(f"  Isolation filter: {len(candidates)} → {len(survivors)} candidates")

    # ── Phase 2: greedy separation / keep-brightest ───────────────────────────
    survivors_sorted = survivors.sort_values("tess_mag").reset_index(drop=True)
    kept_xy = []
    kept_indices = []

    for _, row in survivors_sorted.iterrows():
        cx, cy = row["x"], row["y"]
        if kept_xy:
            kxy = np.array(kept_xy)
            dists = np.sqrt((kxy[:, 0] - cx) ** 2 + (kxy[:, 1] - cy) ** 2)
            if dists.min() < separation_px:
                continue
        kept_xy.append([cx, cy])
        kept_indices.append(row.name if hasattr(row, "name") else len(kept_xy) - 1)

    # survivors_sorted was reset_index'd; rebuild from it
    ref_stars = survivors_sorted.iloc[: len(kept_xy)].copy()
    # Rebuild properly
    final_rows = []
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
    log.info(f"  Separation filter: {len(survivors)} → {len(ref_stars)} reference stars")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "hotpants_substamp_stars.csv")
        ref_stars.to_csv(out_path, index=False)
        log.info(f"  Reference stars saved to {out_path}")

    return ref_stars


def load_gaia_for_masking(gaia_csv: str,
                          crop_bounds: dict,
                          mag_col: str = "tess_mag") -> pd.DataFrame:
    """
    Helper: load a Gaia CSV and add crop-local 'x', 'y', 'mag' columns.

    Expects the CSV to have 'x' and 'y' already in crop-local coordinates
    (as produced by wcs_grouping.build_unique_gaia_catalog / template-job Gaia CSV), or raw pixel
    columns that are rebased here.

    Parameters
    ----------
    gaia_csv : str
    crop_bounds : dict
    mag_col : str  (column to copy into 'mag')
    """
    df = pd.read_csv(gaia_csv)
    if "mag" not in df.columns:
        if mag_col in df.columns:
            df["mag"] = df[mag_col]
        elif "phot_rp_mean_mag" in df.columns:
            df["mag"] = df["phot_rp_mean_mag"]
        else:
            raise ValueError(f"Cannot find magnitude column in {gaia_csv}")

    # If x/y are in FFI coords (x_ffi, y_ffi), rebase
    if "x_ffi" in df.columns:
        df["x"] = df["x_ffi"] - crop_bounds["x_min"]
        df["y"] = df["y_ffi"] - crop_bounds["y_min"]

    # Ensure we have x and y
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Gaia DataFrame must have 'x' and 'y' columns (crop-local).")

    return df
