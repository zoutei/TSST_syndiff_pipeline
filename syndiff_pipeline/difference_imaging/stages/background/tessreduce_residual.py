"""Direct production port of ``dev/background/tessreduce_smooth_bkg_steps``.

Keep this module algorithmically aligned with that notebook.  In particular,
the anomaly repair and residual-surface routines intentionally do not reuse the
older pipeline strap helper, which implements a different algorithm.

The gap-fill step additionally ports the "robust" boundary sigma-clip from
``dev/background/tessreduce_smooth_bkg_steps_s50_robust.ipynb``
(``sanitize_boundary_outliers`` + ``robust_trend_residual_gap_fill``,
``fill_method="robust"``, ``interpolate=False`` / "biharmonic_robust"
variant): before biharmonic inpainting fills the masked region, pixels on the
valid side of the mask boundary that are KNN-sigma-clip outliers relative to
their local neighborhood are folded into the fit mask, so an anomalous rim
pixel can't bias the smooth trend. This is the sole background-removal method
used by ``kernel_fit`` and ``background_estimate`` (see
``estimate_tessreduce_residual_background`` below, which both stages call).
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import binary_dilation, gaussian_filter, label as ndi_label, laplace
from scipy.spatial import cKDTree
from skimage import restoration as inpaint

FAINT_CAT = 32
STRAP_BIT = 4

# Notebook defaults for the boundary KNN sigma-clip (s50_robust variant).
BOUNDARY_CLIP_K = 15
BOUNDARY_CLIP_SIGMA = 3.0
BOUNDARY_CLIP_RIM_WIDTH = 1


def _fit_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(np.int64, copy=False)
    return (m == 0) | (m == FAINT_CAT)


def sanitize_boundary_outliers(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    k: int = BOUNDARY_CLIP_K,
    sigma_thresh: float = BOUNDARY_CLIP_SIGMA,
    rim_width: int = BOUNDARY_CLIP_RIM_WIDTH,
) -> np.ndarray:
    """Notebook ``sanitize_boundary_outliers`` exactly.

    Fold anomalous valid pixels bordering a masked region into the mask,
    using a KNN local median/MAD sigma-clip computed from nearby valid
    pixels (excluding the rim itself).
    """
    sanitized_mask = np.asarray(mask, dtype=bool).copy()
    dilated = binary_dilation(sanitized_mask, iterations=rim_width)
    rim = dilated & ~sanitized_mask
    rim_coords = np.argwhere(rim)
    valid_pool = ~sanitized_mask & ~rim
    valid_coords = np.argwhere(valid_pool)

    if len(valid_coords) < k or rim_coords.size == 0:
        return sanitized_mask

    tree = cKDTree(valid_coords)
    _, indices = tree.query(rim_coords, k=k)
    neighbor_coords = valid_coords[indices]
    neighbor_vals = data[neighbor_coords[..., 0], neighbor_coords[..., 1]]
    local_median = np.nanmedian(neighbor_vals, axis=1)
    local_mad = 1.4826 * np.nanmedian(
        np.abs(neighbor_vals - local_median[:, np.newaxis]), axis=1
    )
    rim_values = data[rim_coords[:, 0], rim_coords[:, 1]]
    is_outlier = (
        np.isfinite(rim_values)
        & np.isfinite(local_median)
        & (
            np.abs(rim_values - local_median)
            > sigma_thresh * np.maximum(local_mad, 1e-5)
        )
    )
    outlier_coords = rim_coords[is_outlier]
    if outlier_coords.size:
        sanitized_mask[outlier_coords[:, 0], outlier_coords[:, 1]] = True
    return sanitized_mask


def smooth_bkg_decomposed(
    data: np.ndarray,
    *,
    gauss_smooth: float = 0.0,
    boundary_k: int = BOUNDARY_CLIP_K,
    boundary_sigma: float = BOUNDARY_CLIP_SIGMA,
    boundary_rim_width: int = BOUNDARY_CLIP_RIM_WIDTH,
) -> np.ndarray:
    """Notebook ``robust_trend_residual_gap_fill(interpolate=False,
    fill_method="robust")`` biharmonic branch: KNN-sigma-clip the mask
    boundary (``sanitize_boundary_outliers``) before biharmonic inpainting,
    then optional Gaussian smoothing (unchanged from the legacy variant).
    """
    data = np.asarray(data, dtype=np.float64)
    if not (~np.isnan(data)).any():
        return np.zeros_like(data)
    arr = np.ma.masked_invalid(deepcopy(data))
    if arr.count() <= 10:
        return np.zeros_like(data)
    invalid_mask = arr.mask.astype(bool)
    safe_invalid_mask = sanitize_boundary_outliers(
        np.nan_to_num(data, nan=0.0),
        invalid_mask,
        k=boundary_k,
        sigma_thresh=boundary_sigma,
        rim_width=boundary_rim_width,
    )
    fill_input = data.copy()
    fill_input[safe_invalid_mask] = np.nan
    filled = np.asarray(
        inpaint.inpaint_biharmonic(
            np.nan_to_num(fill_input, nan=0.0), safe_invalid_mask
        ),
        dtype=np.float64,
    )
    gs = float(gauss_smooth)
    if gs > 0:
        if np.nanmedian(filled) < 150 and np.nanstd(filled) < 3:
            gs *= 4
        filled = gaussian_filter(filled, gs)
    return np.asarray(filled, dtype=np.float64)


def _block_sigma(resid: np.ndarray, box: int, valid_mask: np.ndarray) -> np.ndarray:
    ny, nx = resid.shape
    sigma = np.full((ny, nx), np.inf, dtype=float)
    for r0 in [min(v, ny - box) for v in range(0, ny, box)]:
        for c0 in [min(v, nx - box) for v in range(0, nx, box)]:
            vals = resid[r0:r0 + box, c0:c0 + box][valid_mask[r0:r0 + box, c0:c0 + box]]
            if vals.size >= 4:
                med = np.nanmedian(vals)
                sigma[r0:r0 + box, c0:c0 + box] = 1.4826 * np.nanmedian(np.abs(vals - med))
    return sigma


def _fit_residual_bkg(residual: np.ndarray, exclude_mask: np.ndarray, res_box: int, n_sigma: float = 5.0) -> np.ndarray:
    """Exact notebook residual-surface implementation."""
    if (~exclude_mask).sum() < 4:
        return np.zeros_like(residual)
    finite = residual[~exclude_mask & np.isfinite(residual)]
    med = np.nanmedian(finite) if finite.size else 0.0
    std = np.nanstd(finite) if finite.size else 0.0
    transient = exclude_mask | (np.abs(residual - med) > 5 * std)
    try:
        corr = Background2D(residual, box_size=res_box, filter_size=3,
                            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
                            bkg_estimator=MedianBackground(), mask=transient,
                            fill_value=0.0).background
    except Exception:
        valid = residual[~transient]
        corr = np.full_like(residual, np.nanmedian(valid) if valid.size else 0.0)
    corr_resid = residual - corr
    _, _, corr_std = sigma_clipped_stats(corr_resid[~exclude_mask])
    flagged = np.abs(corr_resid) > n_sigma * corr_std
    if flagged.any():
        lap_abs = np.abs(laplace(corr_resid))
        lap_med = np.nanmedian(lap_abs)
        lap_mad = np.nanmedian(np.abs(lap_abs - lap_med))
        sharp = lap_abs > lap_med + 3 * 1.4826 * lap_mad
        labeled, n_components = ndi_label(flagged)
        if n_components:
            labels = labeled.ravel()
            sizes = np.bincount(labels, minlength=n_components + 1)
            sharp_counts = np.bincount(labels, weights=sharp.ravel(), minlength=n_components + 1)
            suppress = np.flatnonzero((sharp_counts / np.maximum(sizes, 1))[1:] >= 0.3) + 1
            if len(suppress):
                corr[np.isin(labeled, suppress)] = 0.0
    return corr


def _strap_fit_pixels(mask: np.ndarray) -> np.ndarray:
    return (mask == STRAP_BIT) | (mask == (STRAP_BIT | FAINT_CAT))


def _sigma_clip_mask(values: np.ndarray, *, sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    use = np.ones(values.shape, dtype=bool)
    for _ in range(maxiters):
        if use.sum() < 4:
            break
        _, med, std = sigma_clipped_stats(values[use], sigma=sigma, maxiters=1)
        if not np.isfinite(std) or std == 0:
            break
        new_use = use & (np.abs(values - med) <= sigma * std)
        if new_use.sum() == use.sum():
            break
        use = new_use
    return use


def _qe_spline_map(flux: np.ndarray, background: np.ndarray, mask: np.ndarray, *, degree: int = 2, smooth_mult: float = 10.0) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(np.int64, copy=False)
    ny, _ = background.shape
    rows = np.arange(ny, dtype=np.float64)
    strap_fit = _strap_fit_pixels(m)
    qe_map = np.ones_like(background, dtype=np.float64)
    for col in np.where((m & STRAP_BIT).any(axis=0))[0]:
        good = strap_fit[:, col] & np.isfinite(flux[:, col]) & np.isfinite(background[:, col]) & (background[:, col] != 0)
        if good.sum() < max(10, degree + 1):
            continue
        values, yy = flux[good, col] / background[good, col], rows[good]
        use = _sigma_clip_mask(values, sigma=3.0, maxiters=5)
        if use.sum() < degree + 1:
            continue
        qfit, yfit = values[use], yy[use]
        smoothing = max(yfit.size * float(np.nanvar(qfit)) * smooth_mult, 1e-6)
        try:
            fitted = UnivariateSpline(yfit, qfit, k=degree, s=smoothing)(rows)
        except Exception:
            continue
        fitted[~np.isfinite(fitted)] = 1.0
        fitted[fitted < 1.0] = 1.0
        qe_map[:, col] = fitted
    return qe_map


# Growth-curve search uses r in range(2, 20); keep a stamp that contains those rings
# plus the r=3 SEP ellipse. Full-CCD arrays are not required per detection.
_SEP_RING_RMAX = 19
_SEP_ELLIPSE_R = 3.0


def _sep_object_stamp_slices(x: float, y: float, a: float, b: float, ny: int, nx: int) -> tuple[int, int, int, int]:
    pad = max(_SEP_RING_RMAX + 1, int(np.ceil(_SEP_ELLIPSE_R * max(a, b, 0.0) + 2)))
    x0 = max(0, int(np.floor(x - pad)))
    y0 = max(0, int(np.floor(y - pad)))
    x1 = min(nx, int(np.ceil(x + pad)) + 1)
    y1 = min(ny, int(np.ceil(y + pad)) + 1)
    return y0, y1, x0, x1


def _accumulate_sep_object_mask(
    sep_mask: np.ndarray,
    obj,
    lap_sub: np.ndarray,
    lap_err: np.ndarray,
    noise: float,
) -> None:
    """Same growth-curve mask as the full-frame loop, on a small stamp around ``obj``."""
    import sep

    ny, nx = lap_sub.shape
    x = float(obj["x"])
    y = float(obj["y"])
    a = float(obj["a"])
    b = float(obj["b"])
    theta = float(obj["theta"])
    y0, y1, x0, x1 = _sep_object_stamp_slices(x, y, a, b, ny, nx)
    if y1 <= y0 or x1 <= x0:
        return
    ap = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    sep.mask_ellipse(ap, x - x0, y - y0, a, b, theta, r=_SEP_ELLIPSE_R)
    if not ap.sum():
        return
    lap_c = lap_sub[y0:y1, x0:x1]
    err_c = lap_err[y0:y1, x0:x1]
    if (lap_c / (err_c + 1e-10))[ap].mean() <= 2.0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    true_r = next(
        (
            r - 1
            for r in range(2, 20)
            if (dist >= r - 0.5).any()
            and lap_c[(dist >= r - 0.5) & (dist < r + 0.5)].mean() < noise
        ),
        None,
    )
    if true_r is not None and 2 <= true_r <= 5:
        sep_mask[y0:y1, x0:x1] |= dist <= true_r


def fix_bkg_frame_decomposed(bkg_i: np.ndarray, flux_i: np.ndarray, bkgmask_i: np.ndarray, mask: np.ndarray, *, gauss_smooth: float = 2.0, n_sigma: float = 5.0, force_anomaly_repair: bool = False) -> np.ndarray:
    """Notebook ``fix_bkg_frame_decomposed`` production path.

    Background2D / inpaint still run on the full CCD. SEP detections only
    allocate a small stamp (growth-curve r<=19 plus the r=3 ellipse).
    """
    import sep

    mask2d = np.asarray(mask)
    if mask2d.ndim == 3:
        mask2d = mask2d[0]
    mask2d = mask2d.astype(np.int64, copy=False)
    ny, nx = bkg_i.shape
    src_mask = (mask2d & 1).astype(bool)
    strap = (mask2d & 4).astype(bool)
    strap_cols = np.where(strap.any(axis=0))[0]
    good_cols = np.where(~strap.any(axis=0))[0]
    has_straps = len(strap_cols) > 0 and len(good_cols) > 0
    data_src = np.isnan(bkgmask_i)
    phot_mask = strap | data_src
    masked_ref = flux_i * (~src_mask).astype(float)
    masked_ref[masked_ref == 0] = np.nan
    is_high_bkg = bool(np.nanmedian(masked_ref) > 200.0)
    if force_anomaly_repair:
        is_high_bkg = False
    eff_box = max(4, min(16, min(ny, nx) // 2))
    disk_y, disk_x = np.ogrid[-2:3, -2:3]
    disk = disk_x**2 + disk_y**2 <= 4
    frame = bkg_i.copy()
    # The notebook invokes this function with skip_strap_correction=True;
    # strap QE is deliberately applied only in the final B-spline step.
    try:
        trend = Background2D(frame, box_size=eff_box, filter_size=3, mask=phot_mask,
                             bkg_estimator=MedianBackground(), exclude_percentile=50).background
    except Exception:
        trend = np.full_like(frame, np.nanmedian(frame))
    residual = frame - trend
    valid = ~phot_mask
    if not is_high_bkg:
        coarse = (np.abs(residual) > n_sigma * _block_sigma(residual, 30, valid)) & valid
        lap_abs = np.abs(laplace(frame)).astype(np.float64)
        lap_bkg = sep.Background(lap_abs)
        lap_sub, lap_err = lap_abs - lap_bkg.back(), lap_bkg.rms()
        try:
            objects = sep.extract(lap_sub, thresh=3.0, err=lap_err)
        except Exception:
            objects = []
        sep_mask = np.zeros((ny, nx), dtype=bool)
        noise = np.nanmedian(lap_err)
        for obj in objects:
            _accumulate_sep_object_mask(sep_mask, obj, lap_sub, lap_err, noise)
        lap_med, lap_mad = np.nanmedian(lap_abs), np.nanmedian(np.abs(lap_abs - np.nanmedian(lap_abs)))
        is_sharp = lap_abs > lap_med + 3 * 1.4826 * lap_mad
        edge = np.zeros((ny, nx), dtype=bool); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
        labeled, count = ndi_label(coarse)
        sharp_mask = sep_mask.copy()
        if count:
            flat = labeled.ravel()
            touch_edge = np.zeros(count + 1, bool); touch_sep = np.zeros(count + 1, bool); touch_sharp = np.zeros(count + 1, bool)
            np.bitwise_or.at(touch_edge, flat, edge.ravel()); np.bitwise_or.at(touch_sep, flat, sep_mask.ravel()); np.bitwise_or.at(touch_sharp, flat, is_sharp.ravel())
            labels = ~(touch_edge & ~touch_sep) & touch_sharp & touch_sep; labels[0] = False
            sharp_mask |= labels[labeled]
        smooth_mask = coarse & ~sharp_mask
        if sharp_mask.any() or smooth_mask.any():
            try:
                fine = Background2D(frame, box_size=max(min(4, min(ny, nx)//2), 4), filter_size=3,
                                    mask=phot_mask | sharp_mask, bkg_estimator=MedianBackground(), exclude_percentile=50).background
            except Exception:
                fine = trend
            frame[sharp_mask & valid] = fine[sharp_mask & valid]
            confirmed = smooth_mask & ((np.abs(frame - fine) > n_sigma * _block_sigma(frame - fine, 4, valid)) & valid)
            frame[binary_dilation(confirmed, structure=disk) & valid] = fine[binary_dilation(confirmed, structure=disk) & valid]
    gaussian = gaussian_filter(frame, sigma=2.0 if is_high_bkg else gauss_smooth)
    return gaussian + _fit_residual_bkg(flux_i - gaussian, np.isnan(bkgmask_i), max(4, min(20, min(ny, nx)//2)), n_sigma)


def estimate_tessreduce_residual_background(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    smooth_gauss: float = 2.0,
    anomaly_gauss: float = 2.0,
    qe_spline_degree: int = 2,
    qe_spline_smooth_mult: float = 10.0,
    force_anomaly_repair: bool = False,
    boundary_k: int = BOUNDARY_CLIP_K,
    boundary_sigma: float = BOUNDARY_CLIP_SIGMA,
    boundary_rim_width: int = BOUNDARY_CLIP_RIM_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notebook ``run_tessreduce_variant`` (``biharmonic_robust``) arithmetic
    for one input frame.

    This is the single shared background estimator used by both
    ``kernel_fit`` and ``background_estimate``; ``boundary_k``/``boundary_sigma``/
    ``boundary_rim_width`` control the KNN sigma-clip applied to the mask
    boundary before biharmonic gap-filling (see ``sanitize_boundary_outliers``).
    """
    flux = np.asarray(residual, dtype=np.float64)
    if flux.ndim != 2:
        raise ValueError(f"residual background expects 2-D image, got {flux.shape}")
    fit = _fit_mask(mask)
    if fit.shape != flux.shape:
        raise ValueError(f"mask shape {fit.shape} != residual shape {flux.shape}")
    bkgmask = np.where(fit, 1.0, np.nan)
    smooth = smooth_bkg_decomposed(
        flux * bkgmask,
        gauss_smooth=smooth_gauss,
        boundary_k=boundary_k,
        boundary_sigma=boundary_sigma,
        boundary_rim_width=boundary_rim_width,
    )
    pre_qe = fix_bkg_frame_decomposed(
        smooth, flux, bkgmask, mask, gauss_smooth=anomaly_gauss,
        force_anomaly_repair=force_anomaly_repair,
    )
    qe = _qe_spline_map(flux, pre_qe, mask, degree=qe_spline_degree, smooth_mult=qe_spline_smooth_mult)
    return pre_qe * qe, pre_qe, qe
