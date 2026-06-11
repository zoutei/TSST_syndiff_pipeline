"""
photometry.py
=============
``forced_photometry`` pipeline stage — forced PSF photometry on difference images.
When ``cfg.n_jobs`` > 1, cutout I/O and per-epoch ``psf_flux`` use joblib **loky**
(process pool); use ``n_jobs: 1`` for a fully serial run.

**Multiple sky targets** (``additional_forced_targets``): :func:`run_forced_photometry_multi`
reads each difference FITS once per epoch and runs ``psf_flux`` for every source
(``phot_snap='brightest'`` adds one full-epoch scan pass before flux). A single
target still uses :func:`_run_forced_photometry_single` so cutouts are reused and
FITS are not read twice.

**FITS inputs:** Multi-extension files may include extension ``NOISE`` (per-pixel
ERROR, treated like TESSreduce ``ecut``: the fitter uses ``residual² / error``).
When absent, photometry uses unit ``error`` (same as TESSreduce ``use_error_image=False``
with flat weights).
Supports two modes:
  • 'epsf' — use the fitted empirical ePSF (EpsfLocator wrapper)
  • 'prf'  — use the official TESS PRF (TESS_PRF from the PRF package)

The ``create_psf`` class and ``polynomial_surface`` are vendored from the
publicly available **TESSreduce** project.  ``EpsfLocator`` is a thin wrapper
that implements the same ``.locate(col, row, shape)`` API as the official PRF
locator so ``create_psf`` can run with either PRF or empirical ePSF.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.optimize import minimize

from syndiff_pipeline.common.joblib_progress import (
    parallel_map_with_optional_tqdm,
    tqdm_iter,
)
from syndiff_pipeline.difference_imaging.stages.photometry_progress import (
    init_progress_pair,
    progress_path_for_diff_log,
    progress_path_for_output_workspace,
    record_epoch_progress,
    reset_epochs_done_pair,
    set_progress_phase_pair,
)
from scipy.ndimage import shift as nd_shift
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


@dataclass
class ForcedPhotTargetSpec:
    """One forced-photometry source for :func:`run_forced_photometry_multi`."""

    target_xy: np.ndarray
    csv_basename: str = "lightcurve.csv"
    plot_source_label: str = "primary"
    plot_png_path: Optional[str] = None
    plot_gif_diff_path: Optional[str] = None
    plot_gif_science_path: Optional[str] = None
    plot_gif_pair_path: Optional[str] = None
    tag: str = "primary"


def read_diff_primary_and_noise_sigma(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load PRIMARY difference image and optional per-pixel ERROR map (same role as
    TESSreduce ``ecut`` / ``flux_err`` for weighting).

    Looks for extension ``NOISE``; if not found but a second HDU exists, uses HDU 1.
    Shape mismatch returns ``None`` for the error array.
    """
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        noise: Optional[np.ndarray] = None
        if len(hdul) > 1:
            for hdu in hdul[1:]:
                if hdu.data is None:
                    continue
                name = str(hdu.header.get("EXTNAME", "")).strip().upper()
                if name == "NOISE":
                    noise = np.asarray(hdu.data, dtype=np.float64)
                    break
            if noise is None and hdul[1].data is not None:
                noise = np.asarray(hdul[1].data, dtype=np.float64)
        if noise is not None and noise.shape != data.shape:
            log.warning(
                "NOISE shape %s != PRIMARY %s in %s; ignoring NOISE for photometry",
                noise.shape,
                data.shape,
                path,
            )
            noise = None
    return data, noise


def per_frame_target_crop_xy(
    wcs_table: pd.DataFrame,
    ra: float,
    dec: float,
    crop_bounds: dict,
) -> np.ndarray:
    """
    For each manifest row, open that FFI and map (ra, dec) to **crop-local** (x, y).

    Uses the same column as the pipeline manifest: ``path`` or ``filename``.
    Rows with missing paths or WCS failures get ``(nan, nan)``.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    path_col = "path" if "path" in wcs_table.columns else "filename"
    coord_rd = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
    x_min = float(crop_bounds["x_min"])
    y_min = float(crop_bounds["y_min"])
    n = len(wcs_table)
    out = np.full((n, 2), np.nan, dtype=np.float64)
    for i in range(n):
        p = wcs_table.iloc[i].get(path_col)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        ps = str(p).strip()
        if not ps or not os.path.isfile(ps):
            continue
        try:
            with fits.open(ps, memmap=True) as hdul:
                wcs = WCS(hdul[1].header, fix=False)
                x_ffi, y_ffi = wcs.world_to_pixel(coord_rd)
            out[i, 0] = float(x_ffi) - x_min
            out[i, 1] = float(y_ffi) - y_min
        except Exception as exc:
            log.debug("  per_frame_target_crop_xy row %s: %s", i, exc)
    return out


def resolve_forced_target_xy(
    spec: dict,
    primary_xy: np.ndarray,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
) -> np.ndarray:
    """
    Build per-epoch crop-local (x, y) for one normalized forced-target spec.

    ``spec`` must include ``position_mode`` (``sky``, ``offset``, or ``fixed``)
    from :func:`~syndiff_pipeline.difference_imaging.orchestration.config.normalize_additional_forced_targets`.
    """
    mode = str(spec.get("position_mode", "sky"))
    n_epochs = len(wcs_table)
    if mode == "sky":
        return per_frame_target_crop_xy(
            wcs_table, float(spec["ra"]), float(spec["dec"]), crop_bounds
        )
    if mode == "offset":
        offset = np.array([float(spec["dx"]), float(spec["dy"])], dtype=np.float64)
        primary = np.asarray(primary_xy, dtype=np.float64)
        if primary.shape != (n_epochs, 2):
            raise ValueError(
                f"primary_xy shape {primary.shape} != ({n_epochs}, 2) for offset target"
            )
        return primary + offset
    if mode == "fixed":
        xy = np.array([float(spec["x"]), float(spec["y"])], dtype=np.float64)
        return np.broadcast_to(xy, (n_epochs, 2)).copy()
    raise ValueError(f"unknown forced target position_mode {mode!r}")


def _tessreduce_error_plane(
    ecut: Optional[np.ndarray],
    shape: tuple[int, ...],
) -> np.ndarray:
    """
    Per-pixel ``error`` for ``create_psf.psf_flux`` / ``psf_position``.

    TESSreduce passes ``flux_err`` (or flat 0.1) **without squaring**; the
    objective is ``sum((residual)**2 / error)``. Non-finite or ~zero entries
    fall back to 1.0 to avoid division blow-ups.
    """
    if ecut is None:
        return np.ones(shape, dtype=np.float64)
    e = np.asarray(ecut, dtype=np.float64)
    e = np.where(np.isfinite(e), np.abs(e), 1.0)
    e = np.where(e > 1e-30, e, 1.0)
    return e


def _tessreduce_brightest_weight(
    cut: np.ndarray,
    ecut: Optional[np.ndarray],
) -> float:
    """
    TESSreduce ``snap='brightest'`` weights frames by ``|sum(cut/ecut)|`` in a 3×3
    patch at the stamp center (target is centered in the extracted cutout).
    """
    h, w = cut.shape
    hc, wc = h // 2, w // 2
    y0, y1 = max(0, hc - 1), min(h, hc + 2)
    x0, x1 = max(0, wc - 1), min(w, wc + 2)
    patch = cut[y0:y1, x0:x1]
    if ecut is None:
        denom = np.ones_like(patch, dtype=np.float64)
    else:
        denom = ecut[y0:y1, x0:x1]
    denom = _tessreduce_error_plane(denom, patch.shape)
    return float(np.abs(np.nansum(patch / denom)))


# ═══════════════════════════════════════════════════════════════════════════════
# ── Vendored from TESSreduce ``psf_photom`` (polynomial_surface, create_psf) ──
# ═══════════════════════════════════════════════════════════════════════════════

def polynomial_surface(x, y, coeffs, order=2):
    """Evaluate an n-order 2D polynomial surface at grid positions (x, y)."""
    z = np.zeros_like(x, dtype=float)
    ind = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            z += coeffs[ind] * (x ** i) * (y ** j)
            ind += 1
    return z


class create_psf:
    """
    Forced PSF photometry using a user-supplied PRF/ePSF locator.

    Vendored from TESSreduce ``psf_photom``.

    The `prf` argument must implement:
        prf.locate(col: float, row: float, shape: tuple) -> 2D ndarray

    Both TESS_PRF and EpsfLocator satisfy this interface.
    """

    def __init__(self, prf, size: int):
        self.prf       = prf
        self.size      = size
        self.source_x  = 0.0
        self.source_y  = 0.0
        self.cent      = size / 2.0 - 0.5
        self.psf       = None
        self.flux      = None
        self.eflux     = None
        self.surface   = None
        self.image_residual = None

    def source(self, shiftx=0.0, shifty=0.0, ext_shift=None):
        if ext_shift is None:
            ext_shift = [0, 0]
        centx_s = self.cent + shiftx
        centy_s = self.cent + shifty
        psf = self.prf.locate(centx_s - ext_shift[1],
                              centy_s - ext_shift[0],
                              (self.size, self.size))
        psf = nd_shift(psf, ext_shift)
        self.psf = psf / np.nansum(psf)

    def minimize_position(self, coeff, image, error, ext_shift):
        self.source_x = coeff[0]
        self.source_y = coeff[1]
        self.source(shiftx=self.source_x, shifty=self.source_y, ext_shift=ext_shift)
        diff = np.abs(image - self.psf) ** 2
        return np.nansum(diff / error)

    def psf_position(self, image, error=None, limx=0.8, limy=0.8, ext_shift=None):
        if error is None:
            error = np.ones_like(image)
        if ext_shift is None:
            ext_shift = [0, 0]
        if np.nansum(image) > 0:
            normimage = image / np.nansum(image)
            coeff  = [self.source_x, self.source_y]
            lims   = [[-limx, limx], [-limy, limy]]
            res = minimize(
                self.minimize_position, coeff,
                args=(normimage, error, ext_shift),
                method="Powell", bounds=lims,
            )
            self.source_x = res.x[0]
            self.source_y = res.x[1]
            self.psf_fit  = res
        else:
            self.psf_fit = None

    def minimize_psf_flux(self, coeff, image, error=None, surface=True,
                           order=2, kernel=None):
        if surface:
            x  = np.arange(image.shape[1])
            y  = np.arange(image.shape[0])
            yy, xx = np.meshgrid(y, x)
            s  = polynomial_surface(xx, yy, coeff[1:], order)
        else:
            s = 0
        if kernel is not None:
            psf = fftconvolve(self.psf, kernel, mode="same")
        else:
            psf = self.psf
        return np.nansum((image - psf * coeff[0] - s) ** 2 / error)

    def psf_flux(self, image, error=None, ext_shift=None, surface=True,
                 poly_order=3, kernel=None):
        if error is None:
            error = np.ones_like(image)
        if self.psf is None:
            self.source(shiftx=self.source_x, shifty=self.source_y)
        if (ext_shift is not None) and np.isfinite(ext_shift).all():
            self.source(ext_shift=ext_shift)

        mask = np.zeros_like(self.psf)
        mask[self.psf > np.nanpercentile(self.psf, 90)] = 1
        f0 = np.nansum(image * mask)

        if surface:
            num_coeffs = (poly_order + 1) * (poly_order + 2) // 2
            initial = np.zeros(num_coeffs + 1)
            initial[0] = f0
        else:
            initial = np.array([f0])

        res = minimize(
            self.minimize_psf_flux, initial,
            args=(image, error, surface, poly_order, kernel),
            method="BFGS",
        )
        error_val = np.sqrt(np.diag(res["hess_inv"]))
        self.res   = res
        self.flux  = res.x[0]
        self.eflux = error_val[0]

        if surface:
            x  = np.arange(image.shape[1])
            y  = np.arange(image.shape[0])
            yy, xx = np.meshgrid(y, x)
            s  = polynomial_surface(xx, yy, res.x[1:], poly_order)
        else:
            s = image * 0
        self.surface = s
        self.image_residual = image - self.psf * self.flux - s


# ═══════════════════════════════════════════════════════════════════════════════
# ── EpsfLocator — drop-in ePSF replacement for TESS_PRF ──────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class EpsfLocator:
    """
    Wrap a 2D empirical ePSF array so it can be used as a drop-in replacement
    for TESS_PRF inside create_psf.source().

    Implements the .locate(col, row, shape) interface compatible with TESS_PRF.

    The ePSF is oversampled by os_factor.  locate() computes the sub-pixel
    offset, shifts the oversampled ePSF via scipy.ndimage.shift, then
    block-sum downsamples to native resolution.
    """

    def __init__(self, epsf_2d: np.ndarray, os_factor: int):
        """
        Parameters
        ----------
        epsf_2d   : 2D ndarray (over_size, over_size) — normalized oversampled ePSF
        os_factor : int  (e.g. 2)
        """
        self.epsf_os  = epsf_2d.copy()
        self.os_factor = os_factor
        self.over_size = epsf_2d.shape[0]

    def locate(self, col: float, row: float, shape: tuple) -> np.ndarray:
        """
        Compute the native-resolution PSF stamp centred at sub-pixel
        position (col, row) within a stamp of `shape`.

        Parameters
        ----------
        col, row : float  — fractional position of the source within the stamp
        shape    : (ny_stamp, nx_stamp)

        Returns
        -------
        2D ndarray (ny_stamp, nx_stamp)
        """
        ny_stamp, nx_stamp = shape
        os = self.os_factor
        over_size = self.over_size

        # Centre of the native-pixel stamp in oversampled pixels
        cx_os = col  * os
        cy_os = row  * os

        # Centre of the oversampled ePSF array
        half = over_size / 2.0

        # Sub-pixel offset from ePSF centre to desired position
        dx_os = cx_os - half
        dy_os = cy_os - half

        # Shift ePSF in oversampled space
        shifted = nd_shift(self.epsf_os, [dy_os, dx_os], order=1,
                           mode="constant", cval=0.0)

        # Block-sum downsample to native size
        h_trim = (over_size // os) * os
        w_trim = (over_size // os) * os
        shifted = shifted[:h_trim, :w_trim]
        native = shifted.reshape(h_trim // os, os, w_trim // os, os).sum(axis=(1, 3))

        # Crop or pad to match requested stamp shape
        out = np.zeros((ny_stamp, nx_stamp), dtype=np.float64)
        nh, nw = native.shape
        y0 = (ny_stamp - nh) // 2;  y1 = y0 + nh
        x0 = (nx_stamp - nw) // 2;  x1 = x0 + nw
        # Clamp
        sy0 = max(0, -y0);  sy1 = nh - max(0, y1 - ny_stamp)
        sx0 = max(0, -x0);  sx1 = nw - max(0, x1 - nx_stamp)
        y0, y1 = max(0, y0), min(ny_stamp, y1)
        x0, x1 = max(0, x0), min(nx_stamp, x1)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = native[sy0:sy1, sx0:sx1]

        norm = out.sum()
        if norm > 0:
            out /= norm
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# ── PSF kernel builder (epsf or prf) ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def build_psf_kernel(
    phot,
    cfg,
    epsf_smooth: np.ndarray,
    tile_centers: list,
    target_x: float,
    target_y: float,
    over_size: int,
    crop_bounds: dict,
):
    """
    Return a PSF locator object (either EpsfLocator or TESS_PRF) based on
    ``phot.psf_type``.

    Parameters
    ----------
    phot         : ForcedPhotometryParams
    cfg          : SynDiffConfig (camera, ccd, sector for PRF)
    epsf_smooth  : ndarray (n_tiles, over_size²) — per-tile group ePSF
    tile_centers : list of (cx, cy)
    target_x, target_y : float  (crop-local pixel position of the target)
    over_size    : int
    crop_bounds  : dict

    Returns
    -------
    object with .locate(col, row, shape) method
    """
    from syndiff_pipeline.difference_imaging.stages.sat_template import get_tile_epsf_at_position

    if phot.psf_type == "epsf":
        epsf_2d = get_tile_epsf_at_position(
            epsf_smooth, tile_centers, target_x, target_y, over_size,
        )
        return EpsfLocator(epsf_2d, phot.epsf_oversample)

    if phot.psf_type == "prf":
        try:
            from PRF import TESS_PRF
        except ImportError:
            raise ImportError(
                "The PRF package is required for psf_type='prf'. "
                "Install with: pip install PRF"
            )
        col_ffi = target_x + crop_bounds["x_min"]
        row_ffi = target_y + crop_bounds["y_min"]
        return TESS_PRF(cfg.camera, cfg.ccd, cfg.sector, col_ffi, row_ffi)

    raise ValueError(f"Unknown psf_type '{phot.psf_type}'. Must be 'epsf' or 'prf'.")


# ═══════════════════════════════════════════════════════════════════════════════
# ── Target pixel helpers ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def get_target_pixel_crop(wcs, target_ra: float, target_dec: float,
                           crop_bounds: dict) -> tuple:
    """
    Convert target RA/Dec to crop-local pixel coordinates.

    Parameters
    ----------
    wcs        : astropy WCS of the reference FFI
    target_ra, target_dec : float (degrees)
    crop_bounds : dict

    Returns
    -------
    (x_crop, y_crop) : float, float
    """
    from astropy.coordinates import SkyCoord

    coord  = SkyCoord(ra=target_ra, dec=target_dec, unit="deg")
    x_ffi, y_ffi = wcs.world_to_pixel(coord)
    x_crop = float(x_ffi) - crop_bounds["x_min"]
    y_crop = float(y_ffi) - crop_bounds["y_min"]
    return x_crop, y_crop


def _extract_cutout(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Extract a centred square cutout of `size` from `image` at (x, y)."""
    half = size // 2
    ix, iy = int(round(x)), int(round(y))
    ny, nx = image.shape
    r0 = max(0, iy - half);  r1 = min(ny, iy - half + size)
    c0 = max(0, ix - half);  c1 = min(nx, ix - half + size)
    cutout = np.full((size, size), np.nan)
    dy0 = r0 - (iy - half);  dx0 = c0 - (ix - half)
    cutout[dy0:dy0 + (r1 - r0), dx0:dx0 + (c1 - c0)] = image[r0:r1, c0:c1]
    return cutout


def photometry_marker_xy(
    stamp_size: int,
    source_offset_x: float,
    source_offset_y: float,
) -> tuple[float, float]:
    """Stamp-pixel coordinates of the PSF-flux anchor (center + phot_snap offsets)."""
    cent = stamp_size / 2.0 - 0.5
    return cent + float(source_offset_x), cent + float(source_offset_y)


def _science_cutout_worker(
    task: Tuple[int, Optional[str], float, float, dict, int],
) -> Tuple[int, Optional[np.ndarray]]:
    """Load cropped FFI science and extract one stamp (epoch index, cutout)."""
    from syndiff_pipeline.difference_imaging.stages.hotpants import _load_ffi_cropped

    i, ffi_path, tx, ty, crop_bounds, phot_cutout_size = task
    if ffi_path is None or not os.path.isfile(str(ffi_path)):
        return i, None
    if not (np.isfinite(tx) and np.isfinite(ty)):
        return i, None
    try:
        sci, _ = _load_ffi_cropped(str(ffi_path), crop_bounds)
        return i, _extract_cutout(sci, float(tx), float(ty), phot_cutout_size)
    except Exception as exc:
        log.debug("  science cutout frame %s: %s", i, exc)
        return i, None


def _diff_cutout_worker(
    task: Tuple[int, Optional[str], float, float, int],
) -> Tuple[int, Optional[np.ndarray]]:
    """Load one difference FITS and extract a square stamp (epoch index, cutout)."""
    i, path, tx, ty, stamp_size = task
    if path is None or not os.path.isfile(str(path)):
        return i, None
    if not (np.isfinite(tx) and np.isfinite(ty)):
        return i, None
    try:
        data, _ = read_diff_primary_and_noise_sigma(str(path))
        return i, _extract_cutout(data, float(tx), float(ty), int(stamp_size))
    except Exception as exc:
        log.debug("  diff cutout frame %s: %s", i, exc)
        return i, None


def _extract_diff_cutouts_for_epochs(
    diff_paths: list,
    target_xy: np.ndarray,
    stamp_size: int,
    n_jobs: int,
) -> list:
    """Per-epoch difference-image stamps at ``target_xy`` (chronological order)."""
    n = len(diff_paths)
    txy = np.asarray(target_xy, dtype=np.float64)
    tasks = [
        (
            i,
            diff_paths[i],
            float(txy[i, 0]),
            float(txy[i, 1]),
            int(stamp_size),
        )
        for i in range(n)
    ]
    cutouts: list = [None] * n
    parallel = int(n_jobs or 1) != 1 and n > 1
    if parallel:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_diff_cutout_worker)(t) for t in tasks
        )
        for i, cut in results:
            cutouts[i] = cut
    else:
        for t in tasks:
            i, cut = _diff_cutout_worker(t)
            cutouts[i] = cut
    return cutouts


def _extract_science_cutouts_for_epochs(
    wcs_table: pd.DataFrame,
    target_xy: np.ndarray,
    crop_bounds: dict,
    phot_cutout_size: int,
    n_jobs: int,
) -> list:
    """Per-manifest-row cropped science stamps at ``target_xy`` (chronological order)."""
    path_col = "path" if "path" in wcs_table.columns else "filename"
    n = len(wcs_table)
    txy = np.asarray(target_xy, dtype=np.float64)
    tasks = [
        (
            i,
            wcs_table.iloc[i].get(path_col),
            float(txy[i, 0]),
            float(txy[i, 1]),
            crop_bounds,
            int(phot_cutout_size),
        )
        for i in range(n)
    ]
    cutouts: list = [None] * n
    parallel = int(n_jobs or 1) != 1 and n > 1
    if parallel:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_science_cutout_worker)(t) for t in tasks
        )
        for i, cut in results:
            cutouts[i] = cut
    else:
        for t in tasks:
            i, cut = _science_cutout_worker(t)
            cutouts[i] = cut
    return cutouts


def _write_cutout_debug_gifs(
    diff_paths: list,
    wcs_table: pd.DataFrame,
    target_xy: np.ndarray,
    crop_bounds: dict,
    phot,
    cfg,
    *,
    source_offset_x: float = 0.0,
    source_offset_y: float = 0.0,
    plot_gif_diff_path: Optional[str] = None,
    plot_gif_science_path: Optional[str] = None,
    plot_gif_pair_path: Optional[str] = None,
) -> None:
    """Write diff, science, and side-by-side stamp GIFs when paths are set."""
    from syndiff_pipeline.difference_imaging.support import plot as plot_mod

    if not (
        plot_gif_diff_path or plot_gif_science_path or plot_gif_pair_path
    ):
        return

    dpi = int(getattr(cfg, "pipeline_plot_dpi", 150) or 150)
    stamp_size = int(getattr(phot, "phot_debug_stamp_size", 25) or 25)
    n_jobs = int(getattr(cfg, "n_jobs", 1) or 1)
    marker_xy = photometry_marker_xy(stamp_size, source_offset_x, source_offset_y)
    btjd = None
    if wcs_table is not None and "btjd" in wcs_table.columns:
        btjd = np.asarray(wcs_table["btjd"].values, dtype=float)

    diff_cutouts: Optional[list] = None
    if plot_gif_diff_path or plot_gif_pair_path:
        diff_cutouts = _extract_diff_cutouts_for_epochs(
            diff_paths,
            target_xy,
            stamp_size,
            n_jobs,
        )

    sci_cutouts: Optional[list] = None
    if plot_gif_science_path or plot_gif_pair_path:
        sci_cutouts = _extract_science_cutouts_for_epochs(
            wcs_table,
            target_xy,
            crop_bounds,
            stamp_size,
            n_jobs,
        )

    if plot_gif_diff_path and diff_cutouts is not None:
        plot_mod.write_stamp_animation(
            diff_cutouts,
            plot_gif_diff_path,
            btjd=btjd,
            stamp_size=stamp_size,
            cmap="RdBu_r",
            scale_mode="symmetric",
            cbar_label="Diff stamp",
            dpi=dpi,
            marker_xy=marker_xy,
        )
    if plot_gif_science_path and sci_cutouts is not None:
        plot_mod.write_stamp_animation(
            sci_cutouts,
            plot_gif_science_path,
            btjd=btjd,
            stamp_size=stamp_size,
            cmap="viridis",
            scale_mode="percentile",
            cbar_label="Science stamp",
            dpi=dpi,
            marker_xy=marker_xy,
        )
    if plot_gif_pair_path and diff_cutouts is not None and sci_cutouts is not None:
        plot_mod.write_dual_stamp_animation(
            diff_cutouts,
            sci_cutouts,
            plot_gif_pair_path,
            btjd=btjd,
            stamp_size=stamp_size,
            dpi=dpi,
            marker_xy=marker_xy,
        )


def _locator_bundle_for_parallel(prf_or_epsf, phot, cfg, crop_bounds, target_x, target_y):
    """
    Picklable description of the PSF locator for joblib workers.

    TESS_PRF objects may not pickle reliably; workers reconstruct from metadata.
    """
    if phot.psf_type == "epsf":
        return (
            "epsf",
            np.ascontiguousarray(prf_or_epsf.epsf_os, dtype=np.float64),
            int(prf_or_epsf.os_factor),
        )
    if phot.psf_type == "prf":
        col_ffi = float(target_x + crop_bounds["x_min"])
        row_ffi = float(target_y + crop_bounds["y_min"])
        return (
            "prf",
            int(cfg.sector),
            int(cfg.camera),
            int(cfg.ccd),
            col_ffi,
            row_ffi,
        )
    raise ValueError(f"Unknown psf_type {phot.psf_type!r}")


def _locator_from_bundle(bundle: tuple) -> Any:
    kind = bundle[0]
    if kind == "epsf":
        _, arr, os_factor = bundle
        return EpsfLocator(np.asarray(arr, dtype=np.float64), int(os_factor))
    if kind == "prf":
        _, sector, camera, ccd, col_ffi, row_ffi = bundle
        from PRF import TESS_PRF

        return TESS_PRF(camera, ccd, sector, col_ffi, row_ffi)
    raise ValueError(f"Unknown locator bundle kind {kind!r}")


def _forced_phot_cutout_worker(
    task: Tuple[int, Optional[str], float, float, int],
) -> Tuple[int, Optional[np.ndarray], float, Optional[np.ndarray]]:
    """Load one diff FITS; return (index, cutout, tess_brightest_weight, ecut_cut)."""
    i, path, target_x, target_y, phot_cutout_size = task
    if path is None or not os.path.exists(path):
        return i, None, -1.0, None
    if not (np.isfinite(target_x) and np.isfinite(target_y)):
        return i, None, -1.0, None
    try:
        data, sigma_full = read_diff_primary_and_noise_sigma(path)
        cut = _extract_cutout(data, float(target_x), float(target_y), phot_cutout_size)
        sigma_cut = None
        if sigma_full is not None:
            sigma_cut = _extract_cutout(
                sigma_full, float(target_x), float(target_y), phot_cutout_size
            )
    except Exception as exc:
        log.warning("  Cannot read %s: %s", path, exc)
        return i, None, -1.0, None
    finite = cut[np.isfinite(cut)]
    if len(finite) == 0:
        return i, cut, -1.0, sigma_cut
    tw = _tessreduce_brightest_weight(cut, sigma_cut)
    return i, cut, tw, sigma_cut


def _forced_phot_flux_worker(
    task: Tuple[
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
        float,
        tuple,
        int,
        int,
        float,
        int,
        str,
    ],
) -> Tuple[int, dict]:
    """
    Run ``psf_flux`` for one epoch in an isolated ``create_psf`` instance.

    Returns (frame_index, record_dict).
    """
    (
        i,
        cut,
        sigma_cut,
        source_x,
        source_y,
        locator_bundle,
        phot_cutout_size,
        phot_bkg_poly_order,
        btjd,
        group_id,
        path,
    ) = task
    if cut is None:
        return i, {
            "btjd": btjd,
            "flux": np.nan,
            "eflux": np.nan,
            "filename": path or "",
            "group_id": group_id,
        }
    prf = _locator_from_bundle(locator_bundle)
    psf_obj = create_psf(prf, phot_cutout_size)
    psf_obj.source_x = float(source_x)
    psf_obj.source_y = float(source_y)
    error = _tessreduce_error_plane(sigma_cut, cut.shape)
    try:
        psf_obj.psf_flux(
            cut,
            error=error,
            surface=True,
            poly_order=phot_bkg_poly_order,
        )
        flux, eflux = psf_obj.flux, psf_obj.eflux
    except Exception as exc:
        log.debug("  psf_flux failed for frame %s: %s", i, exc)
        flux, eflux = np.nan, np.nan
    return i, {
        "btjd": btjd,
        "flux": flux,
        "eflux": eflux,
        "filename": path or "",
        "group_id": group_id,
    }


def _offsets_after_source_only(locator_bundle: tuple, phot_cutout_size: int) -> Tuple[float, float]:
    """Match ``create_psf.source()`` defaults when ``psf_position`` is not used."""
    prf = _locator_from_bundle(locator_bundle)
    psf_obj = create_psf(prf, phot_cutout_size)
    psf_obj.source()
    return float(psf_obj.source_x), float(psf_obj.source_y)


def _forced_phot_brightest_scan_multi_worker(
    task: Tuple[int, Optional[str], Tuple[Tuple[float, float], ...], int],
) -> Tuple[int, List[Tuple[Optional[np.ndarray], Optional[np.ndarray], float]]]:
    """
    One epoch: read FITS once, extract each source's cutout and brightest weight.

    Returns (frame_index, per_source list of (cut, sigma_cut, tess_weight)).
    """
    i, path, coords_per_source, phot_cutout_size = task
    n_src = len(coords_per_source)
    empty = [(None, None, -1.0)] * n_src
    if path is None or not os.path.exists(path):
        return i, list(empty)
    try:
        data, sigma_full = read_diff_primary_and_noise_sigma(path)
    except Exception as exc:
        log.warning("  Cannot read %s: %s", path, exc)
        return i, list(empty)
    out: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], float]] = []
    for tx, ty in coords_per_source:
        if not (np.isfinite(tx) and np.isfinite(ty)):
            out.append((None, None, -1.0))
            continue
        try:
            cut = _extract_cutout(data, float(tx), float(ty), phot_cutout_size)
            sigma_cut = None
            if sigma_full is not None:
                sigma_cut = _extract_cutout(
                    sigma_full, float(tx), float(ty), phot_cutout_size
                )
        except Exception:
            out.append((None, None, -1.0))
            continue
        finite = cut[np.isfinite(cut)]
        if len(finite) == 0:
            out.append((cut, sigma_cut, -1.0))
            continue
        tw = _tessreduce_brightest_weight(cut, sigma_cut)
        out.append((cut, sigma_cut, tw))
    return i, out


def _forced_phot_multi_flux_worker(
    task: Tuple[
        int,
        Optional[str],
        float,
        int,
        int,
        int,
        tuple,
        str,
    ],
) -> Tuple[int, List[dict], List[Optional[np.ndarray]]]:
    """
    One epoch: read FITS once; run ``psf_flux`` for each source (isolated ``create_psf``).

    ``per_source`` entries are
    (locator_bundle, tx_i, ty_i, source_x, source_y).

    Returns cutout stamps per source (same order as records) for debug GIFs.
    """
    (
        i,
        path,
        btjd,
        group_id,
        phot_cutout_size,
        phot_bkg_poly_order,
        per_source,
        path_str,
    ) = task

    def _nan_record() -> dict:
        return {
            "btjd": btjd,
            "flux": np.nan,
            "eflux": np.nan,
            "filename": path_str or "",
            "group_id": group_id,
        }

    if path is None or not os.path.exists(path):
        return i, [_nan_record() for _ in per_source], [None] * len(per_source)

    try:
        data, sigma_full = read_diff_primary_and_noise_sigma(path)
    except Exception as exc:
        log.warning("  Cannot read %s: %s", path, exc)
        return i, [_nan_record() for _ in per_source], [None] * len(per_source)

    records: List[dict] = []
    cuts: List[Optional[np.ndarray]] = []
    for locator_bundle, tx, ty, sx, sy in per_source:
        if not (np.isfinite(tx) and np.isfinite(ty)):
            records.append(_nan_record())
            cuts.append(None)
            continue
        try:
            cut = _extract_cutout(data, float(tx), float(ty), phot_cutout_size)
            sigma_cut = None
            if sigma_full is not None:
                sigma_cut = _extract_cutout(
                    sigma_full, float(tx), float(ty), phot_cutout_size
                )
        except Exception:
            records.append(_nan_record())
            cuts.append(None)
            continue

        prf = _locator_from_bundle(locator_bundle)
        psf_obj = create_psf(prf, phot_cutout_size)
        psf_obj.source_x = float(sx)
        psf_obj.source_y = float(sy)
        error = _tessreduce_error_plane(sigma_cut, cut.shape)
        try:
            psf_obj.psf_flux(
                cut,
                error=error,
                surface=True,
                poly_order=phot_bkg_poly_order,
            )
            flux, eflux = psf_obj.flux, psf_obj.eflux
        except Exception as exc:
            log.debug("  psf_flux failed for frame %s: %s", i, exc)
            flux, eflux = np.nan, np.nan
        records.append(
            {
                "btjd": btjd,
                "flux": flux,
                "eflux": eflux,
                "filename": path_str or "",
                "group_id": group_id,
            }
        )
        cuts.append(cut)
    return i, records, cuts


def _sigma_clipped_mean(values: np.ndarray, *, n_sigma: float) -> float:
    """Mean of finite ``values`` after rejecting points outside mean ± n_sigma·std."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(np.nan)
    if v.size == 1:
        return float(v[0])
    mu = float(np.nanmean(v))
    sig = float(np.nanstd(v))
    if not np.isfinite(sig) or sig <= 0.0:
        return mu
    lo, hi = mu - n_sigma * sig, mu + n_sigma * sig
    clipped = v[(v >= lo) & (v <= hi)]
    if clipped.size == 0:
        return float(np.nanmean(v))
    return float(np.nanmean(clipped))


def _centered_time_average_btjd(
    btjd_sorted: np.ndarray,
    flux_sorted: np.ndarray,
    *,
    window_hours: float,
    n_sigma_clip: Optional[float] = 3.0,
) -> np.ndarray:
    """
    For each sorted epoch ``t``, return the mean of ``flux`` over samples with
    ``|btjd - t| <= window_hours/2`` (centered moving average in BTJD days).

    When ``n_sigma_clip`` is not None, each window uses a mean with points outside
    ``mean ± n_sigma_clip·std`` removed first (3σ clipping by default).
    """
    t = np.asarray(btjd_sorted, dtype=float)
    f = np.asarray(flux_sorted, dtype=float)
    half_days = (window_hours / 24.0) / 2.0
    n = t.size
    if n == 0:
        return f
    out = np.empty(n, dtype=float)
    for i in range(n):
        sel = (t >= t[i] - half_days) & (t <= t[i] + half_days)
        vals = f[sel]
        if n_sigma_clip is None:
            out[i] = np.nanmean(vals)
        else:
            out[i] = _sigma_clipped_mean(vals, n_sigma=float(n_sigma_clip))
    return out


def _robust_std_1d(x: np.ndarray) -> float:
    """MAD-based robust scale (same construction as PRF LC comparison notebook)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * max(mad, 1e-12 * (1.0 + abs(med))))


def _binned_sigma_clip_btjd(
    btjd: np.ndarray,
    flux: np.ndarray,
    *,
    bin_width_days: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin by BTJD, σ-clip within each bin using median and MAD-scale robust std,
    then average clipped fluxes / times per bin (matches
    ``scripts/recipe_prf_lightcurve_compare.ipynb`` ``binned_sigma_clip``).
    """
    btjd = np.asarray(btjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    mask = np.zeros_like(flux, dtype=bool)
    if btjd.size == 0:
        return mask, np.array([]), np.array([])

    tmin = float(np.nanmin(btjd))
    tmax = float(np.nanmax(btjd))
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        return mask, np.array([]), np.array([])

    bins = np.arange(tmin, tmax + bin_width_days, bin_width_days)
    inds = np.digitize(btjd, bins)

    binned_avg_flux_list: list[float] = []
    binned_avg_time_list: list[float] = []

    for b in np.unique(inds):
        in_bin = inds == b
        if np.sum(in_bin) < 1:
            continue
        f = flux[in_bin]
        t = btjd[in_bin]
        if np.sum(in_bin) < 3:
            mask[in_bin] = True
            binned_avg_flux_list.append(float(np.nanmean(f)))
            binned_avg_time_list.append(float(np.nanmean(t)))
            continue
        med = float(np.median(f))
        rob_std = _robust_std_1d(f)
        keep = np.abs(f - med) <= sigma * rob_std
        mask[in_bin] = keep
        if np.any(keep):
            binned_avg_flux_list.append(float(np.nanmean(f[keep])))
            binned_avg_time_list.append(float(np.nanmean(t[keep])))
        else:
            binned_avg_flux_list.append(float("nan"))
            binned_avg_time_list.append(float("nan"))

    b_avg_f = np.asarray(binned_avg_flux_list, dtype=float)
    b_avg_t = np.asarray(binned_avg_time_list, dtype=float)
    return mask, b_avg_t, b_avg_f


def write_lightcurve_diagnostic_plot(
    lc_df: pd.DataFrame,
    output_dir: str,
    *,
    dpi: int = 150,
    title_line: str = "",
    bin_width_days: float = 0.5,
    bin_sigma: float = 3.0,
    zoom_ylim_pad_frac: float = 0.08,
    png_path: Optional[str] = None,
) -> Optional[str]:
    """
    Write ``lightcurve_control.png``: BTJD vs flux with ``eflux`` error bars on
    the top panel, and a bottom panel with the same comparison notebook-style
    **binned σ-clip**: epochs kept after per-bin robust clipping, native flux
    scale, plus large markers for per-bin averages of the clipped points.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(
            "pipeline_plots: matplotlib is not installed; skipping light curve plot."
        )
        return None

    need = ("btjd", "flux", "eflux")
    if not all(c in lc_df.columns for c in need):
        log.warning(
            "pipeline_plots: light curve missing required columns %s; skip plot.", need
        )
        return None

    ok = lc_df["flux"].notna() & lc_df["btjd"].notna()
    n = int(ok.sum())
    if ok.any():
        x = lc_df.loc[ok, "btjd"].to_numpy(dtype=float)
        y = lc_df.loc[ok, "flux"].to_numpy(dtype=float)
        yerr = lc_df.loc[ok, "eflux"].to_numpy(dtype=float)
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        yers = yerr[order]
        mask_kept, binned_t, binned_f = _binned_sigma_clip_btjd(
            xs,
            ys,
            bin_width_days=bin_width_days,
            sigma=bin_sigma,
        )
    else:
        log.warning(
            "pipeline_plots: no finite flux values; writing empty light curve plot."
        )
        xs = np.array([], dtype=float)
        ys = np.array([], dtype=float)
        yers = np.array([], dtype=float)
        mask_kept = np.zeros(0, dtype=bool)
        binned_t = np.array([], dtype=float)
        binned_f = np.array([], dtype=float)

    subtitle = f"{n} epochs"

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(7, 6.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    ax_top.errorbar(
        xs,
        ys,
        yerr=yers,
        fmt="o",
        capsize=2,
        color="0.35",
        ecolor="0.55",
        ms=4,
        alpha=0.75,
        label="per epoch",
        zorder=2,
    )
    ax_top.set_ylabel("Difference-image flux")
    ax_top.grid(True, alpha=0.35)
    ax_top.legend(loc="best", fontsize=8)
    if title_line:
        ax_top.set_title(f"{title_line}\n{subtitle}")
    else:
        ax_top.set_title(f"SynDiff forced photometry — {subtitle}")

    ax_bot.set_xlabel("BTJD")
    bin_note = (
        f"Binned σ-clip ({bin_width_days:g} d bins · MAD vs median · σ={bin_sigma:g})"
    )
    ax_bot.set_title(bin_note, fontsize=10, color="0.35")

    if np.any(mask_kept):
        ax_bot.errorbar(
            xs[mask_kept],
            ys[mask_kept],
            yerr=yers[mask_kept],
            fmt="o",
            capsize=2,
            color="0.35",
            ecolor="0.55",
            ms=4,
            alpha=0.75,
            label="per epoch (kept)",
            zorder=2,
        )

    fin_b = np.isfinite(binned_f) & np.isfinite(binned_t)
    if np.any(fin_b):
        ax_bot.plot(
            binned_t[fin_b],
            binned_f[fin_b],
            "o",
            ms=9,
            color="tab:blue",
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="binned avg",
            zorder=3,
            linestyle="None",
        )

    ax_bot.set_ylabel("Difference-image flux")
    ax_bot.grid(True, alpha=0.35)
    ax_bot.legend(loc="best", fontsize=8)

    y_parts: list[np.ndarray] = []
    if np.any(mask_kept):
        y_parts.append(ys[mask_kept])
    if np.any(fin_b):
        y_parts.append(binned_f[fin_b])
    if y_parts:
        yy = np.concatenate(y_parts)
        yy = yy[np.isfinite(yy)]
        if yy.size > 0:
            lo = float(np.nanmin(yy))
            hi = float(np.nanmax(yy))
            span = hi - lo
            pad = max(span * zoom_ylim_pad_frac, 1e-6 * (abs(hi) + abs(lo) + 1.0))
            if span <= 0 or not np.isfinite(span):
                pad = max(abs(lo), abs(hi), 1.0) * zoom_ylim_pad_frac
            ax_bot.set_ylim(lo - pad, hi + pad)

    if png_path is not None:
        out_path = os.path.expanduser(png_path)
    else:
        out_path = os.path.join(output_dir, "lightcurve_control.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log.info("  pipeline_plots: light curve figure %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# ── Main photometry function ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def _run_forced_photometry_single(
    diff_paths: list,
    target_xy: np.ndarray,
    epsf_r2_smooth: np.ndarray,
    tile_centers: list,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    cfg,
    phot,
    output_dir: str,
    *,
    ref_frame_index: Optional[int] = None,
    lightcurve_plot_path: Optional[str] = None,
    plot_gif_diff_path: Optional[str] = None,
    plot_gif_science_path: Optional[str] = None,
    plot_gif_pair_path: Optional[str] = None,
    plot_title_suffix: Optional[str] = None,
    plot_source_label: Optional[str] = None,
    lightcurve_csv_filename: Optional[str] = None,
    output_label: Optional[str] = None,
    diffs_input: Optional[str] = None,
    diff_log_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Original single-target path: one FITS read per epoch during cutouts, then
    flux reuses cutouts (no second read). Used when only one source is requested.
    """
    csv_name = lightcurve_csv_filename or "lightcurve.csv"
    if os.path.basename(csv_name) != csv_name or ".." in csv_name:
        raise ValueError(
            f"lightcurve_csv_filename must be a plain basename, got {csv_name!r}"
        )
    txy = np.asarray(target_xy, dtype=np.float64)
    if txy.ndim != 2 or txy.shape[1] != 2:
        raise ValueError("target_xy must have shape (n_epochs, 2)")
    n_epochs = len(diff_paths)
    if txy.shape[0] != n_epochs:
        raise ValueError(
            f"target_xy length {txy.shape[0]} != len(diff_paths) {n_epochs}"
        )

    over_size = 2 * phot.psf_size + 1

    if epsf_r2_smooth.ndim == 3:
        group_epsf = np.nanmedian(epsf_r2_smooth, axis=0)
    else:
        group_epsf = epsf_r2_smooth

    tx_med = float(np.nanmedian(txy[:, 0]))
    ty_med = float(np.nanmedian(txy[:, 1]))
    if not (np.isfinite(tx_med) and np.isfinite(ty_med)):
        raise ValueError(
            "forced photometry: need at least one finite (x, y) in target_xy"
        )

    prf_or_epsf = build_psf_kernel(
        phot,
        cfg,
        group_epsf,
        tile_centers,
        tx_med,
        ty_med,
        over_size,
        crop_bounds,
    )
    psf_obj = create_psf(prf_or_epsf, phot.phot_cutout_size)

    n_jobs = int(getattr(cfg, "n_jobs", 1) or 1)
    parallel = n_jobs != 1 and n_epochs > 1
    snap = str(phot.phot_snap or "brightest").lower()

    cli_progress_path = (
        str(progress_path_for_diff_log(diff_log_path))
        if diff_log_path is not None
        else None
    )
    track_progress = output_label is not None
    workspace_progress_path: Optional[str] = None
    if track_progress:
        workspace_progress_path = str(progress_path_for_output_workspace(output_dir))
        init_progress_pair(
            workspace_progress_path,
            cli_progress_path,
            output_label=str(output_label),
            diffs_input=str(diffs_input or ""),
            n_sources=1,
            epochs_total=n_epochs,
            phase="cutouts",
        )
    tqdm_base = f"photometry {output_label}" if track_progress else "photometry"

    def _record_epoch() -> None:
        if workspace_progress_path:
            record_epoch_progress(workspace_progress_path, cli_progress_path)

    best_idx = None
    best_tw = -1.0
    cutouts: list = []
    sigma_cutouts: list = []

    if not parallel:
        for i, path in enumerate(tqdm_iter(diff_paths, desc=f"{tqdm_base} cutouts")):
            if path is None or not os.path.exists(path):
                cutouts.append(None)
                sigma_cutouts.append(None)
                _record_epoch()
                continue
            tx_i, ty_i = float(txy[i, 0]), float(txy[i, 1])
            if not (np.isfinite(tx_i) and np.isfinite(ty_i)):
                cutouts.append(None)
                sigma_cutouts.append(None)
                _record_epoch()
                continue
            try:
                data, sigma_full = read_diff_primary_and_noise_sigma(path)
                cut = _extract_cutout(data, tx_i, ty_i, phot.phot_cutout_size)
                sigma_cut = None
                if sigma_full is not None:
                    sigma_cut = _extract_cutout(
                        sigma_full, tx_i, ty_i, phot.phot_cutout_size
                    )
            except Exception as exc:
                log.warning("  Cannot read %s: %s", path, exc)
                cutouts.append(None)
                sigma_cutouts.append(None)
                _record_epoch()
                continue
            cutouts.append(cut)
            sigma_cutouts.append(sigma_cut)
            if cut is not None:
                tw = _tessreduce_brightest_weight(cut, sigma_cut)
                if tw > best_tw:
                    best_tw = tw
                    best_idx = i
            _record_epoch()
    else:
        log.info(
            "  forced_photometry: cutouts n_jobs=%s (loky), %d epochs",
            n_jobs,
            n_epochs,
        )
        cut_tasks = [
            (i, path, float(txy[i, 0]), float(txy[i, 1]), phot.phot_cutout_size)
            for i, path in enumerate(diff_paths)
        ]
        cut_results = parallel_map_with_optional_tqdm(
            (delayed(_forced_phot_cutout_worker)(t) for t in cut_tasks),
            n_tasks=n_epochs,
            desc=f"{tqdm_base} cutouts",
            n_jobs_eff=n_jobs,
            on_result=lambda _r: _record_epoch(),
        )
        cut_results.sort(key=lambda r: r[0])
        cutouts = [None] * n_epochs
        sigma_cutouts = [None] * n_epochs
        for i, cut, tw, sigc in cut_results:
            cutouts[i] = cut
            sigma_cutouts[i] = sigc
            if cut is not None and tw > best_tw:
                best_tw = tw
                best_idx = i

    psf_obj.source()
    if snap == "ref":
        ri = ref_frame_index
        if (
            ri is not None
            and 0 <= ri < len(cutouts)
            and cutouts[ri] is not None
        ):
            rc = cutouts[ri]
            psf_obj.psf_position(
                rc, error=_tessreduce_error_plane(sigma_cutouts[ri], rc.shape)
            )
            log.info(
                "  PSF position fit on ref frame %s: dx=%.3f, dy=%.3f",
                ri,
                psf_obj.source_x,
                psf_obj.source_y,
            )
        else:
            log.warning(
                "  phot_snap='ref' but ref cutout unavailable; using default (0,0) offsets"
            )
    elif snap == "brightest":
        if best_idx is not None and cutouts[best_idx] is not None:
            bc = cutouts[best_idx]
            psf_obj.psf_position(
                bc,
                error=_tessreduce_error_plane(
                    sigma_cutouts[best_idx], bc.shape
                ),
            )
            log.info(
                "  PSF position fit on brightest frame %s: dx=%.3f, dy=%.3f",
                best_idx,
                psf_obj.source_x,
                psf_obj.source_y,
            )
    elif snap != "fixed":
        log.warning(
            "  Unknown phot_snap=%r; using 'fixed' (source at stamp centre only)",
            snap,
        )

    btjd_col = (
        wcs_table["btjd"].values
        if "btjd" in wcs_table.columns
        else np.full(n_epochs, np.nan)
    )
    gid_col = (
        wcs_table["group_id"].values
        if "group_id" in wcs_table.columns
        else np.zeros(n_epochs, int)
    )

    locator_bundle = _locator_bundle_for_parallel(
        prf_or_epsf, phot, cfg, crop_bounds, tx_med, ty_med
    )
    sx = float(psf_obj.source_x)
    sy = float(psf_obj.source_y)

    if track_progress:
        reset_epochs_done_pair(workspace_progress_path, cli_progress_path, phase="flux")

    if not parallel:
        records = []
        for i, (path, cut) in enumerate(
            zip(tqdm_iter(diff_paths, desc=tqdm_base), cutouts)
        ):
            if cut is None:
                records.append(
                    {
                        "btjd": btjd_col[i] if i < len(btjd_col) else np.nan,
                        "flux": np.nan,
                        "eflux": np.nan,
                        "filename": path if path else "",
                        "group_id": int(gid_col[i]) if i < len(gid_col) else -1,
                    }
                )
                _record_epoch()
                continue

            error = _tessreduce_error_plane(sigma_cutouts[i], cut.shape)
            try:
                psf_obj.psf_flux(
                    cut,
                    error=error,
                    surface=True,
                    poly_order=phot.phot_bkg_poly_order,
                )
                flux, eflux = psf_obj.flux, psf_obj.eflux
            except Exception as exc:
                log.debug("  psf_flux failed for frame %s: %s", i, exc)
                flux, eflux = np.nan, np.nan

            records.append(
                {
                    "btjd": btjd_col[i] if i < len(btjd_col) else np.nan,
                    "flux": flux,
                    "eflux": eflux,
                    "filename": path or "",
                    "group_id": int(gid_col[i]) if i < len(gid_col) else -1,
                }
            )
            _record_epoch()
    else:
        log.info("  forced_photometry: psf_flux n_jobs=%s (loky)", n_jobs)
        flux_tasks = []
        for i, path in enumerate(diff_paths):
            cut = cutouts[i]
            btjd = float(btjd_col[i]) if i < len(btjd_col) else float(np.nan)
            gid = int(gid_col[i]) if i < len(gid_col) else -1
            flux_tasks.append(
                (
                    i,
                    cut,
                    sigma_cutouts[i],
                    sx,
                    sy,
                    locator_bundle,
                    int(phot.phot_cutout_size),
                    int(phot.phot_bkg_poly_order),
                    btjd,
                    gid,
                    str(path) if path else "",
                )
            )
        flux_results = parallel_map_with_optional_tqdm(
            (delayed(_forced_phot_flux_worker)(t) for t in flux_tasks),
            n_tasks=n_epochs,
            desc=tqdm_base,
            n_jobs_eff=n_jobs,
            on_result=lambda _r: _record_epoch(),
        )
        flux_results.sort(key=lambda r: r[0])
        records = [rec for _, rec in flux_results]

    if track_progress:
        set_progress_phase_pair(workspace_progress_path, cli_progress_path, "complete")

    lc_df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, csv_name)
    lc_df.to_csv(out_path, index=False)
    log.info(f"Light curve saved to {out_path}  ({len(lc_df)} epochs)")

    if getattr(cfg, "pipeline_plots", False):
        dpi = int(getattr(cfg, "pipeline_plot_dpi", 150) or 150)
        base = (
            f"Sector {cfg.sector} cam{cfg.camera} ccd{cfg.ccd} · "
            f"{phot.psf_type.upper()}"
        )
        parts = [base]
        if plot_source_label:
            parts.append(str(plot_source_label))
        if plot_title_suffix:
            parts.append(str(plot_title_suffix))
        title = " · ".join(parts)
        write_lightcurve_diagnostic_plot(
            lc_df,
            output_dir,
            dpi=dpi,
            title_line=title,
            png_path=lightcurve_plot_path,
        )
        _write_cutout_debug_gifs(
            diff_paths,
            wcs_table,
            txy,
            crop_bounds,
            phot,
            cfg,
            source_offset_x=sx,
            source_offset_y=sy,
            plot_gif_diff_path=plot_gif_diff_path,
            plot_gif_science_path=plot_gif_science_path,
            plot_gif_pair_path=plot_gif_pair_path,
        )

    return lc_df


def run_forced_photometry_multi(
    diff_paths: list,
    targets: List[ForcedPhotTargetSpec],
    epsf_r2_smooth: np.ndarray,
    tile_centers: list,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    cfg,
    phot,
    output_dir: str,
    *,
    ref_frame_index: Optional[int] = None,
    plot_title_suffix: Optional[str] = None,
    output_label: Optional[str] = None,
    diffs_input: Optional[str] = None,
    diff_log_path: Optional[str] = None,
) -> List[pd.DataFrame]:
    """
    Forced PSF photometry for **multiple** sources sharing the same difference-image list.

    Builds a PRF/ePSF locator per source (median WCS position), resolves ``phot_snap``
    offsets per source, then loads each FITS **once per epoch** and runs ``psf_flux``
    for every source (see :class:`ForcedPhotTargetSpec`).

    Returns one light-curve DataFrame per target, in the same order as ``targets``.
    """
    if not targets:
        raise ValueError("run_forced_photometry_multi: targets list is empty")

    n_epochs = len(diff_paths)
    n_src = len(targets)

    if n_src == 1:
        sp = targets[0]
        return [
            _run_forced_photometry_single(
                diff_paths,
                sp.target_xy,
                epsf_r2_smooth,
                tile_centers,
                wcs_table,
                crop_bounds,
                cfg,
                phot,
                output_dir,
                ref_frame_index=ref_frame_index,
                lightcurve_plot_path=sp.plot_png_path,
                plot_gif_diff_path=sp.plot_gif_diff_path,
                plot_gif_science_path=sp.plot_gif_science_path,
                plot_gif_pair_path=sp.plot_gif_pair_path,
                plot_title_suffix=plot_title_suffix,
                plot_source_label=sp.plot_source_label,
                lightcurve_csv_filename=sp.csv_basename,
                output_label=output_label,
                diffs_input=diffs_input,
                diff_log_path=diff_log_path,
            )
        ]

    for spec in targets:
        cname = spec.csv_basename
        if os.path.basename(cname) != cname or ".." in cname:
            raise ValueError(
                f"csv_basename must be a plain basename, got {cname!r} "
                f"(source tag {spec.tag!r})"
            )
        txy = np.asarray(spec.target_xy, dtype=np.float64)
        if txy.ndim != 2 or txy.shape[1] != 2:
            raise ValueError(
                f"target_xy must have shape (n_epochs, 2); got {txy.shape} for {spec.tag!r}"
            )
        if txy.shape[0] != n_epochs:
            raise ValueError(
                f"target {spec.tag!r}: target_xy length {txy.shape[0]} != "
                f"len(diff_paths) {n_epochs}"
            )

    over_size = 2 * phot.psf_size + 1
    if epsf_r2_smooth.ndim == 3:
        group_epsf = np.nanmedian(epsf_r2_smooth, axis=0)
    else:
        group_epsf = epsf_r2_smooth

    locator_bundles: list[tuple] = []
    for spec in targets:
        txy = np.asarray(spec.target_xy, dtype=np.float64)
        tx_med = float(np.nanmedian(txy[:, 0]))
        ty_med = float(np.nanmedian(txy[:, 1]))
        if not (np.isfinite(tx_med) and np.isfinite(ty_med)):
            raise ValueError(
                f"forced photometry [{spec.tag}]: need at least one finite (x, y) "
                "in target_xy"
            )
        prf_or_epsf = build_psf_kernel(
            phot,
            cfg,
            group_epsf,
            tile_centers,
            tx_med,
            ty_med,
            over_size,
            crop_bounds,
        )
        locator_bundles.append(
            _locator_bundle_for_parallel(
                prf_or_epsf, phot, cfg, crop_bounds, tx_med, ty_med
            )
        )

    n_jobs = int(getattr(cfg, "n_jobs", 1) or 1)
    parallel = n_jobs != 1 and n_epochs > 1
    snap = str(phot.phot_snap or "brightest").lower()
    phot_size = int(phot.phot_cutout_size)
    poly_order = int(phot.phot_bkg_poly_order)

    cli_progress_path = (
        str(progress_path_for_diff_log(diff_log_path))
        if diff_log_path is not None
        else None
    )
    track_progress = output_label is not None
    workspace_progress_path: Optional[str] = None
    if track_progress:
        workspace_progress_path = str(progress_path_for_output_workspace(output_dir))
        init_progress_pair(
            workspace_progress_path,
            cli_progress_path,
            output_label=str(output_label),
            diffs_input=str(diffs_input or ""),
            n_sources=n_src,
            epochs_total=n_epochs,
            phase="cutouts" if snap == "brightest" else "flux",
        )
    tqdm_base = f"photometry {output_label}" if track_progress else "photometry"

    def _record_epoch() -> None:
        if workspace_progress_path:
            record_epoch_progress(workspace_progress_path, cli_progress_path)

    sx = np.zeros(n_src, dtype=np.float64)
    sy = np.zeros(n_src, dtype=np.float64)

    if snap == "fixed":
        for s in range(n_src):
            sx[s], sy[s] = _offsets_after_source_only(locator_bundles[s], phot_size)

    elif snap == "ref":
        ri = ref_frame_index
        ref_ok = (
            ri is not None
            and 0 <= ri < n_epochs
            and diff_paths[ri] is not None
            and os.path.exists(str(diff_paths[ri]))
        )
        if ref_ok:
            path_ref = str(diff_paths[ri])
            try:
                data, sigma_full = read_diff_primary_and_noise_sigma(path_ref)
            except Exception as exc:
                log.warning(
                    "  phot_snap='ref': cannot read ref frame %s: %s; "
                    "using fixed offsets for all sources",
                    path_ref,
                    exc,
                )
                ref_ok = False
            if ref_ok:
                for s, spec in enumerate(targets):
                    txy = np.asarray(spec.target_xy, dtype=np.float64)
                    tx_i, ty_i = float(txy[ri, 0]), float(txy[ri, 1])
                    if not (np.isfinite(tx_i) and np.isfinite(ty_i)):
                        log.warning(
                            "  phot_snap='ref' but ref cutout unavailable for [%s]; "
                            "using default offsets",
                            spec.tag,
                        )
                        sx[s], sy[s] = _offsets_after_source_only(
                            locator_bundles[s], phot_size
                        )
                        continue
                    cut = _extract_cutout(data, tx_i, ty_i, phot_size)
                    sigma_cut = None
                    if sigma_full is not None:
                        sigma_cut = _extract_cutout(
                            sigma_full, tx_i, ty_i, phot_size
                        )
                    psf_obj = create_psf(
                        _locator_from_bundle(locator_bundles[s]), phot_size
                    )
                    psf_obj.source()
                    psf_obj.psf_position(
                        cut,
                        error=_tessreduce_error_plane(sigma_cut, cut.shape),
                    )
                    sx[s] = float(psf_obj.source_x)
                    sy[s] = float(psf_obj.source_y)
                    log.info(
                        "  PSF position fit [%s] on ref frame %s: dx=%.3f, dy=%.3f",
                        spec.tag,
                        ri,
                        psf_obj.source_x,
                        psf_obj.source_y,
                    )
        if not ref_ok:
            log.warning(
                "  phot_snap='ref' but ref cutout unavailable; using default (0,0) offsets"
            )
            for s in range(n_src):
                sx[s], sy[s] = _offsets_after_source_only(locator_bundles[s], phot_size)

    elif snap == "brightest":
        best_tw = [-1.0] * n_src
        best_cut: list[Optional[np.ndarray]] = [None] * n_src
        best_sig: list[Optional[np.ndarray]] = [None] * n_src
        best_idx: list[Optional[int]] = [None] * n_src

        scan_tasks = []
        for i, path in enumerate(diff_paths):
            coords = tuple(
                (
                    float(np.asarray(targets[s].target_xy, dtype=np.float64)[i, 0]),
                    float(np.asarray(targets[s].target_xy, dtype=np.float64)[i, 1]),
                )
                for s in range(n_src)
            )
            scan_tasks.append((i, path, coords, phot_size))

        if not parallel:
            for t in tqdm_iter(scan_tasks, desc=f"{tqdm_base} scan"):
                i, per_src = _forced_phot_brightest_scan_multi_worker(t)
                for s, (cut, sigc, tw) in enumerate(per_src):
                    if tw > best_tw[s]:
                        best_tw[s] = tw
                        best_idx[s] = i
                        best_cut[s] = cut
                        best_sig[s] = sigc
                _record_epoch()
        else:
            log.info(
                "  forced_photometry: brightest scan n_jobs=%s (loky), %d epochs, %d sources",
                n_jobs,
                n_epochs,
                n_src,
            )
            scan_results = parallel_map_with_optional_tqdm(
                (
                    delayed(_forced_phot_brightest_scan_multi_worker)(t)
                    for t in scan_tasks
                ),
                n_tasks=n_epochs,
                desc=f"{tqdm_base} scan",
                n_jobs_eff=n_jobs,
                on_result=lambda _r: _record_epoch(),
            )
            scan_results.sort(key=lambda r: r[0])
            for i, per_src in scan_results:
                for s, (cut, sigc, tw) in enumerate(per_src):
                    if tw > best_tw[s]:
                        best_tw[s] = tw
                        best_idx[s] = i
                        best_cut[s] = cut
                        best_sig[s] = sigc

        for s, spec in enumerate(targets):
            if best_cut[s] is not None:
                psf_obj = create_psf(
                    _locator_from_bundle(locator_bundles[s]), phot_size
                )
                psf_obj.source()
                psf_obj.psf_position(
                    best_cut[s],
                    error=_tessreduce_error_plane(best_sig[s], best_cut[s].shape),
                )
                sx[s] = float(psf_obj.source_x)
                sy[s] = float(psf_obj.source_y)
                log.info(
                    "  PSF position fit [%s] on brightest frame %s: dx=%.3f, dy=%.3f",
                    spec.tag,
                    best_idx[s],
                    psf_obj.source_x,
                    psf_obj.source_y,
                )
            else:
                sx[s], sy[s] = _offsets_after_source_only(locator_bundles[s], phot_size)

    else:
        log.warning(
            "  Unknown phot_snap=%r; using 'fixed' (source at stamp centre only)",
            snap,
        )
        for s in range(n_src):
            sx[s], sy[s] = _offsets_after_source_only(locator_bundles[s], phot_size)

    btjd_col = (
        wcs_table["btjd"].values
        if "btjd" in wcs_table.columns
        else np.full(n_epochs, np.nan)
    )
    gid_col = (
        wcs_table["group_id"].values
        if "group_id" in wcs_table.columns
        else np.zeros(n_epochs, int)
    )

    records_cols: list[list[Optional[dict]]] = [
        [None] * n_epochs for _ in range(n_src)
    ]
    cutouts_cols: list[list[Optional[np.ndarray]]] = [
        [None] * n_epochs for _ in range(n_src)
    ]

    if track_progress and snap == "brightest":
        reset_epochs_done_pair(workspace_progress_path, cli_progress_path, phase="flux")

    flux_tasks = []
    for i, path in enumerate(diff_paths):
        btjd = float(btjd_col[i]) if i < len(btjd_col) else float(np.nan)
        gid = int(gid_col[i]) if i < len(gid_col) else -1
        per_source = tuple(
            (
                locator_bundles[s],
                float(np.asarray(targets[s].target_xy, dtype=np.float64)[i, 0]),
                float(np.asarray(targets[s].target_xy, dtype=np.float64)[i, 1]),
                float(sx[s]),
                float(sy[s]),
            )
            for s in range(n_src)
        )
        flux_tasks.append(
            (
                i,
                path,
                btjd,
                gid,
                phot_size,
                poly_order,
                per_source,
                str(path) if path else "",
            )
        )

    if not parallel:
        for t in tqdm_iter(flux_tasks, desc=tqdm_base):
            i, recs, cuts = _forced_phot_multi_flux_worker(t)
            for s, rec in enumerate(recs):
                records_cols[s][i] = rec
            for s, cut in enumerate(cuts):
                cutouts_cols[s][i] = cut
            _record_epoch()
    else:
        log.info(
            "  forced_photometry: multi flux n_jobs=%s (loky), %d epochs, %d sources",
            n_jobs,
            n_epochs,
            n_src,
        )
        flux_results = parallel_map_with_optional_tqdm(
            (delayed(_forced_phot_multi_flux_worker)(t) for t in flux_tasks),
            n_tasks=n_epochs,
            desc=tqdm_base,
            n_jobs_eff=n_jobs,
            on_result=lambda _r: _record_epoch(),
        )
        flux_results.sort(key=lambda r: r[0])
        for i, recs, cuts in flux_results:
            for s, rec in enumerate(recs):
                records_cols[s][i] = rec
            for s, cut in enumerate(cuts):
                cutouts_cols[s][i] = cut

    if track_progress:
        set_progress_phase_pair(workspace_progress_path, cli_progress_path, "complete")

    os.makedirs(output_dir, exist_ok=True)
    out_dfs: List[pd.DataFrame] = []
    plot_on = getattr(cfg, "pipeline_plots", False)
    dpi = int(getattr(cfg, "pipeline_plot_dpi", 150) or 150)
    base_title = (
        f"Sector {cfg.sector} cam{cfg.camera} ccd{cfg.ccd} · "
        f"{phot.psf_type.upper()}"
    )

    for s, spec in enumerate(targets):
        rec_list = records_cols[s]
        assert all(r is not None for r in rec_list)
        lc_df = pd.DataFrame(rec_list)
        out_path = os.path.join(output_dir, spec.csv_basename)
        lc_df.to_csv(out_path, index=False)
        log.info(
            "Light curve saved to %s  (%d epochs) [%s]",
            out_path,
            len(lc_df),
            spec.tag,
        )

        if plot_on:
            parts = [base_title]
            if spec.plot_source_label:
                parts.append(str(spec.plot_source_label))
            if plot_title_suffix:
                parts.append(str(plot_title_suffix))
            title = " · ".join(parts)
            write_lightcurve_diagnostic_plot(
                lc_df,
                output_dir,
                dpi=dpi,
                title_line=title,
                png_path=spec.plot_png_path,
            )
            _write_cutout_debug_gifs(
                diff_paths,
                wcs_table,
                np.asarray(spec.target_xy, dtype=np.float64),
                crop_bounds,
                phot,
                cfg,
                source_offset_x=float(sx[s]),
                source_offset_y=float(sy[s]),
                plot_gif_diff_path=spec.plot_gif_diff_path,
                plot_gif_science_path=spec.plot_gif_science_path,
                plot_gif_pair_path=spec.plot_gif_pair_path,
            )

        out_dfs.append(lc_df)

    return out_dfs


def run_forced_photometry(
    diff_paths: list,
    target_xy: np.ndarray,
    epsf_r2_smooth: np.ndarray,
    tile_centers: list,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    cfg,
    phot,
    output_dir: str,
    *,
    ref_frame_index: Optional[int] = None,
    lightcurve_plot_path: Optional[str] = None,
    plot_gif_diff_path: Optional[str] = None,
    plot_gif_science_path: Optional[str] = None,
    plot_gif_pair_path: Optional[str] = None,
    plot_title_suffix: Optional[str] = None,
    plot_source_label: Optional[str] = None,
    lightcurve_csv_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Forced PSF photometry on difference-image FITS (SynDiff pipeline).

    Uses the single-target implementation (one FITS read per epoch, cutouts
    reused for flux). For multiple sources on the same diff list, the pipeline
    calls :func:`run_forced_photometry_multi`.
    """
    return _run_forced_photometry_single(
        diff_paths,
        target_xy,
        epsf_r2_smooth,
        tile_centers,
        wcs_table,
        crop_bounds,
        cfg,
        phot,
        output_dir,
        ref_frame_index=ref_frame_index,
        lightcurve_plot_path=lightcurve_plot_path,
        plot_gif_diff_path=plot_gif_diff_path,
        plot_gif_science_path=plot_gif_science_path,
        plot_gif_pair_path=plot_gif_pair_path,
        plot_title_suffix=plot_title_suffix,
        plot_source_label=plot_source_label,
        lightcurve_csv_filename=lightcurve_csv_filename,
    )
