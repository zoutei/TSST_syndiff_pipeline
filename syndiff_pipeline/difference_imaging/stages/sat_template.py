"""
sat_template.py
===============
``sat_template`` pipeline stage — model images of removed (saturated) Gaia stars
using the smoothed empirical ePSF.  Two resolutions are produced:

  • Native-resolution template at cfg.epsf_oversample (default 2×), used for
    subtraction in rough background and later Hotpants rounds.
  • High-resolution template at cfg.high_res_os (default 9×), block-sum
    downsampled to native pixels for storage and optional downstream use.

Stamps are shifted, flux-scaled copies of the tiled empirical ePSF.
"""

import logging
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import shift as nd_shift, zoom as nd_zoom

from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    iter_pipeline_fits_paths,
    strip_fits_suffix,
    workspace_frame_fits_path,
)

log = logging.getLogger(__name__)

# TESS zero point (AB mag system, approximate)
TESS_ZEROPOINT = 20.44


# ═══════════════════════════════════════════════════════════════════════════════
# ── Utility functions ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def tess_mag_to_flux(tess_mag: np.ndarray) -> np.ndarray:
    """Convert TESS magnitudes to flux (arbitrary units, using TESS zero point)."""
    return 10.0 ** (-0.4 * (np.asarray(tess_mag, dtype=float) - TESS_ZEROPOINT))


def get_tile_epsf_at_position(epsf_tiles: np.ndarray,
                               tile_centers: list,
                               x: float, y: float,
                               over_size: int) -> np.ndarray:
    """
    Return the ePSF of the nearest tile to position (x, y).

    Parameters
    ----------
    epsf_tiles   : ndarray (n_tiles, over_size²)
    tile_centers : list of (cx, cy)
    x, y         : crop-local pixel position
    over_size    : int — oversampled PSF grid side length

    Returns
    -------
    2D ndarray (over_size, over_size)
    """
    centers = np.array(tile_centers)     # (n_tiles, 2)
    dists = np.sqrt((centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2)
    best = np.argmin(dists)
    return epsf_tiles[best].reshape(over_size, over_size)


def _block_sum_downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """
    Block-sum downsample a 2D oversampled array by `factor`.

    Input shape must be divisible by factor; if not, it is cropped.
    """
    h, w = arr.shape
    h_trim = (h // factor) * factor
    w_trim = (w // factor) * factor
    arr = arr[:h_trim, :w_trim]
    return arr.reshape(h_trim // factor, factor,
                       w_trim // factor, factor).sum(axis=(1, 3))


def place_star_epsf(canvas_os: np.ndarray,
                    x_crop: float, y_crop: float,
                    flux: float,
                    epsf_2d: np.ndarray,
                    os_factor: int) -> None:
    """
    Add a scaled ePSF stamp into an oversampled canvas (in-place).

    Sub-pixel placement is handled by shifting the oversampled ePSF array
    by the fractional pixel offset converted to oversampled pixels.

    Parameters
    ----------
    canvas_os : 2D ndarray  (ny_os, nx_os) — modified in place
    x_crop, y_crop : float  (crop-local native-pixel coordinates of the star)
    flux       : float      (star flux in native-pixel units)
    epsf_2d    : 2D ndarray (over_size, over_size) — normalized ePSF
    os_factor  : int        (oversampling factor)
    """
    over_size = epsf_2d.shape[0]
    half_os   = over_size // 2      # half-size in oversampled pixels

    # Integer pixel position in the oversampled canvas
    x_os_center = int(round(x_crop * os_factor))
    y_os_center = int(round(y_crop * os_factor))

    # Fractional offset: remaining sub-pixel shift in oversampled pixels
    frac_x = (x_crop * os_factor) - x_os_center
    frac_y = (y_crop * os_factor) - y_os_center

    # Shift ePSF by fractional offset
    shifted = nd_shift(epsf_2d, [frac_y, frac_x], order=1, mode="constant", cval=0.0)

    # Bounds in the canvas
    ny_os, nx_os = canvas_os.shape
    r0 = y_os_center - half_os;  r1 = r0 + over_size
    c0 = x_os_center - half_os;  c1 = c0 + over_size

    # Clip to canvas
    sr0 = max(0, -r0);  er0 = over_size - max(0, r1 - ny_os)
    sc0 = max(0, -c0);  ec0 = over_size - max(0, c1 - nx_os)
    r0, r1 = max(0, r0), min(ny_os, r1)
    c0, c1 = max(0, c0), min(nx_os, c1)

    if r1 > r0 and c1 > c0 and er0 > sr0 and ec0 > sc0:
        canvas_os[r0:r1, c0:c1] += flux * shifted[sr0:er0, sc0:ec0]


# ═══════════════════════════════════════════════════════════════════════════════
# ── Template builders ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def build_sat_template(removed_stars_df: pd.DataFrame,
                        group_epsf: np.ndarray,
                        tile_centers: list,
                        crop_bounds: dict,
                        os_factor: int,
                        over_size: int) -> np.ndarray:
    """
    Build the native-resolution saturated-star template using the smoothed ePSF.

    Stars are placed on an oversampled canvas (ny*os_factor, nx*os_factor)
    then block-sum downsampled to native (ny, nx).

    Parameters
    ----------
    removed_stars_df : pd.DataFrame
        Rows from the removed_stars CSV that fall inside the crop region.
        Must have columns: x (crop-local), y (crop-local), tess_mag.
    group_epsf   : ndarray (n_tiles, over_size²) — ePSF for this group
    tile_centers : list of (cx, cy)
    crop_bounds  : dict  (from wcs_grouping.get_crop_bounds)
    os_factor    : int   (= cfg.epsf_oversample, default 2)
    over_size    : int   (= 2*cfg.psf_size + 1)

    Returns
    -------
    2D ndarray (ny, nx) — native-resolution template
    """
    ny, nx = crop_bounds["shape"]
    canvas_os = np.zeros((ny * os_factor, nx * os_factor), dtype=np.float64)

    stars_in_crop = _filter_stars_to_crop(removed_stars_df, crop_bounds)
    log.info(f"  build_sat_template: placing {len(stars_in_crop)} stars "
             f"(os={os_factor}, over_size={over_size})")

    for _, row in stars_in_crop.iterrows():
        x_c = float(row["x"])
        y_c = float(row["y"])
        flux = tess_mag_to_flux(float(row["tess_mag"]))
        epsf_2d = get_tile_epsf_at_position(group_epsf, tile_centers, x_c, y_c, over_size)
        place_star_epsf(canvas_os, x_c, y_c, flux, epsf_2d, os_factor)

    template_native = _block_sum_downsample(canvas_os, os_factor)
    return template_native


def build_sat_template_highres(removed_stars_df: pd.DataFrame,
                                 group_epsf: np.ndarray,
                                 tile_centers: list,
                                 crop_bounds: dict,
                                 epsf_os: int,
                                 high_res_os: int,
                                 over_size: int) -> np.ndarray:
    """
    Build the high-resolution saturated-star template (oversampled canvas,
    then block-sum downsampled to native resolution).

    The ePSF is zoomed from epsf_os → high_res_os before stamp placement,
    so the canvas is at resolution (ny*high_res_os, nx*high_res_os).

    Parameters
    ----------
    removed_stars_df : pd.DataFrame
    group_epsf   : ndarray (n_tiles, over_size²)
    tile_centers : list of (cx, cy)
    crop_bounds  : dict
    epsf_os      : int   (cfg.epsf_oversample, e.g. 2)
    high_res_os  : int   (cfg.high_res_os, e.g. 9)
    over_size    : int   (e.g. 23)

    Returns
    -------
    2D ndarray (ny, nx) native-resolution template
    """
    zoom_factor = high_res_os / epsf_os
    over_size_hr = int(round(over_size * zoom_factor))
    if over_size_hr % 2 == 0:
        over_size_hr += 1   # keep it odd

    ny, nx = crop_bounds["shape"]
    canvas_os = np.zeros((ny * high_res_os, nx * high_res_os), dtype=np.float64)

    stars_in_crop = _filter_stars_to_crop(removed_stars_df, crop_bounds)
    log.info(f"  build_sat_template_highres: placing {len(stars_in_crop)} stars "
             f"(os={high_res_os}, zoom_factor={zoom_factor:.2f})")

    for _, row in stars_in_crop.iterrows():
        x_c = float(row["x"])
        y_c = float(row["y"])
        flux = tess_mag_to_flux(float(row["tess_mag"]))
        epsf_2d = get_tile_epsf_at_position(group_epsf, tile_centers, x_c, y_c, over_size)

        # Zoom ePSF to high_res_os
        epsf_hr = nd_zoom(epsf_2d, zoom_factor, order=1)
        # Crop / pad to over_size_hr × over_size_hr
        epsf_hr = _center_crop_or_pad(epsf_hr, over_size_hr)
        norm = epsf_hr.sum()
        if norm > 0:
            epsf_hr /= norm

        place_star_epsf(canvas_os, x_c, y_c, flux, epsf_hr, high_res_os)

    template_native = _block_sum_downsample(canvas_os, high_res_os)
    return template_native


def build_all_group_templates(removed_stars_df: pd.DataFrame,
                               group_epsf_dict: dict,
                               tile_centers: list,
                               crop_bounds: dict,
                               sat) -> tuple:
    """
    Build native and high-resolution templates for every template group.

    Parameters
    ----------
    removed_stars_df : pd.DataFrame  (removed stars within crop, with x, y, tess_mag)
    group_epsf_dict  : dict  {group_id: ndarray (n_tiles, over_size²)}
    tile_centers     : list of (cx, cy)
    crop_bounds      : dict
    sat              : SatTemplateParams

    Returns
    -------
    sat_tmpl_native : dict {group_id: 2D ndarray (ny, nx)}
    sat_tmpl_hr     : dict {group_id: 2D ndarray (ny, nx)}
    """
    over_size = 2 * sat.psf_size + 1
    sat_tmpl_native = {}
    sat_tmpl_hr     = {}

    for gid, group_epsf in group_epsf_dict.items():
        log.info(f"Building sat templates for group {gid} ...")
        sat_tmpl_native[gid] = build_sat_template(
            removed_stars_df, group_epsf, tile_centers,
            crop_bounds, os_factor=sat.epsf_oversample, over_size=over_size,
        )
        sat_tmpl_hr[gid] = build_sat_template_highres(
            removed_stars_df, group_epsf, tile_centers,
            crop_bounds, epsf_os=sat.epsf_oversample,
            high_res_os=sat.high_res_os, over_size=over_size,
        )

    return sat_tmpl_native, sat_tmpl_hr


def save_group_templates(sat_tmpl_native: dict, sat_tmpl_hr: dict,
                          output_dir: str, round_id: int = 1) -> None:
    """Save both sets of group templates to output_dir."""
    for tag, tmpl_dict in [("native", sat_tmpl_native), ("hr", sat_tmpl_hr)]:
        sub = os.path.join(output_dir, f"sat_tmpl_{tag}_r{round_id}")
        os.makedirs(sub, exist_ok=True)
        for gid, tmpl in tmpl_dict.items():
            out_path = workspace_frame_fits_path(sub, f"group_{gid}")
            fits.writeto(out_path, tmpl.astype(np.float32), overwrite=True)
        log.info(f"  Sat templates ({tag}) saved to {sub}/")


def load_group_templates(output_dir: str, round_id: int = 1) -> tuple:
    """
    Reload group templates (native + hr) from output_dir.

    Returns
    -------
    sat_tmpl_native, sat_tmpl_hr : dict {group_id: 2D ndarray}
    """
    def _load(sub):
        d = {}
        subdir = os.path.join(output_dir, sub)
        for path in iter_pipeline_fits_paths(subdir):
            stem = strip_fits_suffix(os.path.basename(path))
            if not stem.startswith("group_"):
                continue
            gid = int(stem.replace("group_", "", 1))
            d[gid] = fits.getdata(path).astype(np.float64)
        return d

    native = _load(f"sat_tmpl_native_r{round_id}")
    hr     = _load(f"sat_tmpl_hr_r{round_id}")
    return native, hr


# ═══════════════════════════════════════════════════════════════════════════════
# ── Internal helpers ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_stars_to_crop(df: pd.DataFrame, crop_bounds: dict) -> pd.DataFrame:
    """Keep only stars whose crop-local (x, y) fall within the crop region."""
    ny, nx = crop_bounds["shape"]
    in_crop = (
        (df["x"] >= 0) & (df["x"] < nx) &
        (df["y"] >= 0) & (df["y"] < ny)
    )
    return df[in_crop].copy().reset_index(drop=True)


def _center_crop_or_pad(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Crop or zero-pad a 2D square array to target_size × target_size."""
    h, w = arr.shape
    out = np.zeros((target_size, target_size), dtype=arr.dtype)
    # Center overlap
    y0 = max(0, (target_size - h) // 2);  y1 = y0 + min(h, target_size)
    x0 = max(0, (target_size - w) // 2);  x1 = x0 + min(w, target_size)
    sy0 = max(0, (h - target_size) // 2); sy1 = sy0 + (y1 - y0)
    sx0 = max(0, (w - target_size) // 2); sx1 = sx0 + (x1 - x0)
    out[y0:y1, x0:x1] = arr[sy0:sy1, sx0:sx1]
    return out
