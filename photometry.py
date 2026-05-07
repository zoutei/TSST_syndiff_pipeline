"""
photometry.py
=============
``forced_photometry`` pipeline stage — forced PSF photometry on difference images.
When ``cfg.n_jobs`` > 1, cutout I/O and per-epoch ``psf_flux`` use joblib **loky**
(process pool); use ``n_jobs: 1`` for a fully serial run.

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
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.ndimage import shift as nd_shift
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


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

def build_psf_kernel(cfg,
                     epsf_smooth: np.ndarray,
                     tile_centers: list,
                     target_x: float,
                     target_y: float,
                     over_size: int,
                     crop_bounds: dict):
    """
    Return a PSF locator object (either EpsfLocator or TESS_PRF) based on
    cfg.psf_type.

    Parameters
    ----------
    cfg          : SynDiffConfig
    epsf_smooth  : ndarray (n_tiles, over_size²) — per-tile group ePSF
    tile_centers : list of (cx, cy)
    target_x, target_y : float  (crop-local pixel position of the target)
    over_size    : int
    crop_bounds  : dict

    Returns
    -------
    object with .locate(col, row, shape) method
    """
    from .sat_template import get_tile_epsf_at_position

    if cfg.psf_type == "epsf":
        epsf_2d = get_tile_epsf_at_position(
            epsf_smooth, tile_centers, target_x, target_y, over_size,
        )
        return EpsfLocator(epsf_2d, cfg.epsf_oversample)

    elif cfg.psf_type == "prf":
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

    else:
        raise ValueError(f"Unknown psf_type '{cfg.psf_type}'. Must be 'epsf' or 'prf'.")


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


def _locator_bundle_for_parallel(prf_or_epsf, cfg, crop_bounds, target_x, target_y):
    """
    Picklable description of the PSF locator for joblib workers.

    TESS_PRF objects may not pickle reliably; workers reconstruct from metadata.
    """
    if cfg.psf_type == "epsf":
        return (
            "epsf",
            np.ascontiguousarray(prf_or_epsf.epsf_os, dtype=np.float64),
            int(prf_or_epsf.os_factor),
        )
    if cfg.psf_type == "prf":
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
    raise ValueError(f"Unknown psf_type {cfg.psf_type!r}")


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


def write_lightcurve_diagnostic_plot(
    lc_df: pd.DataFrame,
    output_dir: str,
    *,
    dpi: int = 150,
    title_line: str = "",
    smooth_window_hours: float = 6.0,
    smooth_n_sigma_clip: Optional[float] = 3.0,
    zoom_ylim_pad_frac: float = 0.08,
    png_path: Optional[str] = None,
) -> Optional[str]:
    """
    Write ``lightcurve_control.png``: BTJD vs flux with ``eflux`` error bars, a
    centered moving average (default 6 h) with optional 3σ rejection inside each
    time window before averaging, and a second panel with the same series but
    y-limits from the min/max of that average (plus a small margin).
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

    ok = lc_df["flux"].notna()
    if not ok.any():
        log.warning("pipeline_plots: no finite flux values; skipping light curve plot.")
        return None

    x = lc_df.loc[ok, "btjd"].to_numpy(dtype=float)
    y = lc_df.loc[ok, "flux"].to_numpy(dtype=float)
    yerr = lc_df.loc[ok, "eflux"].to_numpy(dtype=float)

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    yers = yerr[order]
    y_smooth = _centered_time_average_btjd(
        xs,
        ys,
        window_hours=smooth_window_hours,
        n_sigma_clip=smooth_n_sigma_clip,
    )

    n = int(ok.sum())
    clip_note = (
        f" · {smooth_n_sigma_clip:g}σ-clip mean"
        if smooth_n_sigma_clip is not None
        else ""
    )
    subtitle = f"{n} epochs · {smooth_window_hours:g} h centered mean{clip_note}"

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(7, 6.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    def _plot_panel(ax, *, set_title: bool) -> None:
        ax.errorbar(
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
        ax.plot(
            xs,
            y_smooth,
            "-",
            color="tab:blue",
            lw=1.8,
            label=(
                f"{smooth_window_hours:g} h mean ({smooth_n_sigma_clip:g}σ clip)"
                if smooth_n_sigma_clip is not None
                else f"{smooth_window_hours:g} h mean"
            ),
            zorder=3,
        )
        ax.set_ylabel("Difference-image flux")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        if set_title:
            if title_line:
                ax.set_title(f"{title_line}\n{subtitle}")
            else:
                ax.set_title(f"SynDiff forced photometry — {subtitle}")

    _plot_panel(ax_top, set_title=True)
    _plot_panel(ax_bot, set_title=False)
    ax_bot.set_xlabel("BTJD")
    zoom_src = (
        f"{smooth_window_hours:g} h mean (σ-clip) min/max"
        if smooth_n_sigma_clip is not None
        else f"{smooth_window_hours:g} h mean min/max"
    )
    ax_bot.set_title(
        f"Zoom: y-range from {zoom_src}",
        fontsize=10,
        color="0.35",
    )

    smin = np.nanmin(y_smooth)
    smax = np.nanmax(y_smooth)
    if np.isfinite(smin) and np.isfinite(smax):
        span = smax - smin
        pad = max(span * zoom_ylim_pad_frac, 1e-6 * (abs(smax) + abs(smin) + 1.0))
        if span <= 0 or not np.isfinite(span):
            pad = max(abs(smin), abs(smax), 1.0) * zoom_ylim_pad_frac
        ax_bot.set_ylim(smin - pad, smax + pad)

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

def run_forced_photometry(
    diff_paths: list,
    target_xy: np.ndarray,
    epsf_r2_smooth: np.ndarray,
    tile_centers: list,
    wcs_table: pd.DataFrame,
    crop_bounds: dict,
    cfg,
    output_dir: str,
    *,
    ref_frame_index: Optional[int] = None,
    lightcurve_plot_path: Optional[str] = None,
    plot_title_suffix: Optional[str] = None,
    lightcurve_csv_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Forced PSF photometry on difference-image FITS (SynDiff pipeline).

    Per-epoch **cutouts** use rows of ``target_xy`` (crop-local x, y), e.g. from
    each FFI WCS in the frame manifest. The PSF locator (PRF / ePSF) is built at
    the **median** crop position across epochs.

    **Error / noise:** Same convention as TESSreduce ``create_psf``: the optimizer
    uses ``sum((residual)**2 / error)`` where ``error`` is the ``NOISE`` HDU
    (equivalent to ``ecut`` / ``flux_err``) when present, or ones otherwise.

    **snap** (``cfg.phot_snap``): ``brightest`` uses TESSreduce's
    ``|sum(cut/ecut)|`` in a central 3×3 patch to pick the reference epoch;
    ``ref`` fits position on ``ref_frame_index``; ``fixed`` uses (0, 0) offsets
    only.
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

    over_size = 2 * cfg.psf_size + 1

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
        cfg, group_epsf, tile_centers,
        tx_med, ty_med, over_size, crop_bounds,
    )
    psf_obj = create_psf(prf_or_epsf, cfg.phot_cutout_size)

    n_jobs = int(getattr(cfg, "n_jobs", 1) or 1)
    parallel = n_jobs != 1 and n_epochs > 1
    snap = str(getattr(cfg, "phot_snap", "brightest") or "brightest").lower()

    best_idx = None
    best_tw = -1.0
    cutouts: list = []
    sigma_cutouts: list = []

    if not parallel:
        for i, path in enumerate(diff_paths):
            if path is None or not os.path.exists(path):
                cutouts.append(None)
                sigma_cutouts.append(None)
                continue
            tx_i, ty_i = float(txy[i, 0]), float(txy[i, 1])
            if not (np.isfinite(tx_i) and np.isfinite(ty_i)):
                cutouts.append(None)
                sigma_cutouts.append(None)
                continue
            try:
                data, sigma_full = read_diff_primary_and_noise_sigma(path)
                cut = _extract_cutout(data, tx_i, ty_i, cfg.phot_cutout_size)
                sigma_cut = None
                if sigma_full is not None:
                    sigma_cut = _extract_cutout(
                        sigma_full, tx_i, ty_i, cfg.phot_cutout_size
                    )
            except Exception as exc:
                log.warning("  Cannot read %s: %s", path, exc)
                cutouts.append(None)
                sigma_cutouts.append(None)
                continue
            cutouts.append(cut)
            sigma_cutouts.append(sigma_cut)
            if cut is not None:
                tw = _tessreduce_brightest_weight(cut, sigma_cut)
                if tw > best_tw:
                    best_tw = tw
                    best_idx = i
    else:
        log.info(
            "  forced_photometry: cutouts n_jobs=%s (loky), %d epochs",
            n_jobs,
            n_epochs,
        )
        cut_tasks = [
            (i, path, float(txy[i, 0]), float(txy[i, 1]), cfg.phot_cutout_size)
            for i, path in enumerate(diff_paths)
        ]
        cut_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_forced_phot_cutout_worker)(t) for t in cut_tasks
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
        prf_or_epsf, cfg, crop_bounds, tx_med, ty_med
    )
    sx = float(psf_obj.source_x)
    sy = float(psf_obj.source_y)

    if not parallel:
        records = []
        for i, (path, cut) in enumerate(zip(diff_paths, cutouts)):
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
                continue

            error = _tessreduce_error_plane(sigma_cutouts[i], cut.shape)
            try:
                psf_obj.psf_flux(
                    cut,
                    error=error,
                    surface=True,
                    poly_order=cfg.phot_bkg_poly_order,
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
                    int(cfg.phot_cutout_size),
                    int(cfg.phot_bkg_poly_order),
                    btjd,
                    gid,
                    str(path) if path else "",
                )
            )
        flux_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_forced_phot_flux_worker)(t) for t in flux_tasks
        )
        flux_results.sort(key=lambda r: r[0])
        records = [rec for _, rec in flux_results]

    lc_df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, csv_name)
    lc_df.to_csv(out_path, index=False)
    log.info(f"Light curve saved to {out_path}  ({len(lc_df)} epochs)")

    if getattr(cfg, "pipeline_plots", False):
        dpi = int(getattr(cfg, "pipeline_plot_dpi", 150) or 150)
        base = (
            f"Sector {cfg.sector} cam{cfg.camera} ccd{cfg.ccd} · "
            f"{cfg.psf_type.upper()}"
        )
        title = f"{base} · {plot_title_suffix}" if plot_title_suffix else base
        write_lightcurve_diagnostic_plot(
            lc_df,
            output_dir,
            dpi=dpi,
            title_line=title,
            png_path=lightcurve_plot_path,
        )

    return lc_df
