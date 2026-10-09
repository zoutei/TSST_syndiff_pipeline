"""Direct production port of ``dev/background/tessreduce_smooth_bkg_steps``.

Keep this module algorithmically aligned with that notebook.  In particular,
the anomaly repair and residual-surface routines intentionally do not reuse the
older pipeline strap helper, which implements a different algorithm.

The gap-fill step additionally ports the "robust" boundary sigma-clip from
``dev/background/tessreduce_smooth_bkg_steps_s50_robust.ipynb``
(``sanitize_boundary_outliers`` + ``robust_trend_residual_gap_fill``,
``fill_method="robust"``, ``interpolate=False`` / "biharmonic_robust"
variant): before gap-filling the masked region, pixels on the
valid side of the mask boundary that are KNN-sigma-clip outliers relative to
their local neighborhood are folded into the fit mask, so an anomalous rim
pixel can't bias the smooth trend. This is the sole background-removal method
used by ``kernel_fit`` and ``background_estimate`` (see
``estimate_tessreduce_residual_background`` below, which both stages call).

The gap fill itself (``smooth_bkg_decomposed``'s ``fill_method``) defaults to
an exact harmonic (Laplace) solve (``harmonic_inpaint``) rather than
``skimage.restoration.inpaint_biharmonic``. Over the large merged mask holes
this pipeline produces (strap columns x bright-star mask crosses; masked
pixels are ~58% of the frame), biharmonic inpainting matches both boundary
values AND slopes and has no maximum principle, so it extrapolates rim
gradients into the hole with either sign -- this was identified as the direct
cause of the diffuse +-4-8 e/s black/white patches seen in difference images.
The harmonic solve obeys the maximum principle (fill values are bounded by
the rim/valid values) and cannot overshoot; validated full-frame it reduces
negative patches 34->1 and masked high-pass RMS 1.22->0.62 at essentially
identical runtime. Pass ``fill_method="biharmonic"`` to
``smooth_bkg_decomposed``/``estimate_tessreduce_residual_background`` to
restore the previous behaviour for comparison.
"""

from __future__ import annotations

import logging
from typing import Optional
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import binary_dilation, gaussian_filter, label as ndi_label, laplace
from scipy.spatial import cKDTree
from skimage import restoration as inpaint

log = logging.getLogger(__name__)

FAINT_CAT = 32
STRAP_BIT = 4
# Catalogue star masks whose surroundings still carry the star's own PSF wing (bit 1 BRIGHT_CAT crosses, bit 2
# SAT_CROSS circles); see ``star_mask_pad_px``.
STAR_MASK_BITS = 1 | 2

# Notebook defaults for the boundary KNN sigma-clip (s50_robust variant).
BOUNDARY_CLIP_K = 15
BOUNDARY_CLIP_SIGMA = 3.0
BOUNDARY_CLIP_RIM_WIDTH = 1


def _fit_mask(
    mask: np.ndarray,
    star_mask_pad_px: int = 0,
    extra_exclude: np.ndarray | None = None,
) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    m = m.astype(np.int64, copy=False)
    fit = (m == 0) | (m == FAINT_CAT)
    if star_mask_pad_px > 0:
        r = int(star_mask_pad_px)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        fit &= ~binary_dilation((m & STAR_MASK_BITS) != 0, structure=disk)
    if extra_exclude is not None:
        ex = np.asarray(extra_exclude, dtype=bool)
        if ex.shape != fit.shape:
            raise ValueError(f"extra_exclude shape {ex.shape} != mask shape {fit.shape}")
        fit &= ~ex
    return fit


def parse_star_wing_radii(radii) -> tuple[np.ndarray, np.ndarray]:
    """Validate a ``[[mag_hi, radius_px], ...]`` table (``mag_hi`` strictly increasing, radius >= 1)."""
    rows = [tuple(r) for r in (radii or [])]
    if not rows or any(len(r) != 2 for r in rows):
        raise ValueError(f"star wing radii must be a non-empty list of [mag_hi, radius_px] pairs, got {radii!r}")
    mag_hi = np.array([float(r[0]) for r in rows])
    rad = np.array([int(r[1]) for r in rows])
    if np.any(np.diff(mag_hi) <= 0):
        raise ValueError(f"star wing radii: mag_hi must be strictly increasing, got {mag_hi.tolist()}")
    if np.any(rad < 1) or np.any(rad != np.array([float(r[1]) for r in rows])):
        raise ValueError(f"star wing radii: radius_px must be integers >= 1, got {[r[1] for r in rows]}")
    return mag_hi, rad


def star_wing_exclusion(
    shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    tess_mag: np.ndarray,
    radii,
) -> np.ndarray:
    """Pixels to drop from the background fit: a disk around every catalogue star, sized by magnitude.

    ``radii`` is ``[[mag_hi, radius_px], ...]`` with ``mag_hi`` increasing; a star with ``tess_mag < mag_hi`` of the
    first matching row gets that radius, stars at or fainter than the last ``mag_hi`` get none. ``x``/``y`` are
    crop-local 0-based pixel positions (the lane's ``gaia_catalog_pipeline.csv``). The disks are centred on the star:
    the bright side of the TESS wing flips with radius (toward the optical axis inside ~7 px, away beyond ~11 px), and a
    shifted disk tested no better on S24 C2K2 (dev_runs/maskfoot_20261001).
    """
    mag_hi, rad = parse_star_wing_radii(radii)
    ny, nx = int(shape[0]), int(shape[1])
    out = np.zeros((ny, nx), dtype=bool)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(tess_mag, dtype=float)
    k = np.searchsorted(mag_hi, t, side="right")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & (k < len(mag_hi))
    for xi, yi, r in zip(x[ok], y[ok], rad[k[ok]]):
        i0, i1 = max(0, int(np.floor(yi - r))), min(ny, int(np.ceil(yi + r)) + 1)
        j0, j1 = max(0, int(np.floor(xi - r))), min(nx, int(np.ceil(xi + r)) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        yy, xx = np.ogrid[i0:i1, j0:j1]
        out[i0:i1, j0:j1] |= (xx - xi) ** 2 + (yy - yi) ** 2 <= r * r
    return out


def star_wing_exclusion_from_catalog(catalog_csv: str, shape: tuple[int, int], radii) -> np.ndarray:
    """``star_wing_exclusion`` for the stars of a lane catalogue CSV (columns ``x``, ``y``, ``tess_mag``)."""
    import pandas as pd

    cat = pd.read_csv(catalog_csv, usecols=["x", "y", "tess_mag"])
    return star_wing_exclusion(shape, cat["x"].to_numpy(), cat["y"].to_numpy(), cat["tess_mag"].to_numpy(), radii)


def faint_star_exclusion_from_catalog(
    catalog_csv: str,
    shape: tuple[int, int],
    radii,
    tmag_min: float,
    bp_rp_min: Optional[float] = None,
) -> np.ndarray:
    """Disks around faint (optionally red) catalogue stars, dropped from the background fit.

    Selection first: ``tess_mag >= tmag_min`` and, when ``bp_rp_min`` is not None, a finite
    ``phot_bp_mean_mag - phot_rp_mean_mag >= bp_rp_min`` (NaN colour is not selected). The selected stars then get the
    ``[[mag_hi, radius_px], ...]`` disks of ``star_wing_exclusion``. Columns: ``x``, ``y``, ``tess_mag`` and, for the
    colour cut, ``phot_bp_mean_mag``/``phot_rp_mean_mag``.
    """
    import pandas as pd

    parse_star_wing_radii(radii)
    cols = ["x", "y", "tess_mag"]
    if bp_rp_min is not None:
        cols += ["phot_bp_mean_mag", "phot_rp_mean_mag"]
        header = list(pd.read_csv(catalog_csv, nrows=0).columns)
        missing = [c for c in cols if c not in header]
        if missing:
            raise ValueError(f"{catalog_csv}: faint-star colour cut (bp_rp_min={bp_rp_min}) needs columns {missing}")
    cat = pd.read_csv(catalog_csv, usecols=cols)
    t = cat["tess_mag"].to_numpy(dtype=float)
    sel = np.isfinite(t) & (t >= float(tmag_min))
    if bp_rp_min is not None:
        c = cat["phot_bp_mean_mag"].to_numpy(dtype=float) - cat["phot_rp_mean_mag"].to_numpy(dtype=float)
        sel &= np.isfinite(c) & (c >= float(bp_rp_min))
    return star_wing_exclusion(
        shape, cat["x"].to_numpy()[sel], cat["y"].to_numpy()[sel], t[sel], radii
    )


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


def _sparse_cg(A: sp.spmatrix, b: np.ndarray, *, M, maxiter: int) -> np.ndarray:
    """``scipy.sparse.linalg.cg`` across the ``tol``/``rtol`` keyword rename.

    SciPy >=1.12 renamed the convergence-tolerance keyword ``tol`` to
    ``rtol`` (and removed ``tol`` outright in >=1.14); older SciPy only
    accepts ``tol``. Try the modern keyword first so we don't silently run
    with a different tolerance than intended on either version.
    """
    try:
        sol, info = spla.cg(A, b, rtol=1e-6, maxiter=maxiter, M=M)
    except TypeError:
        sol, info = spla.cg(A, b, tol=1e-6, maxiter=maxiter, M=M)
    if info != 0:
        # info > 0: maxiter reached without converging; info < 0: bad input.
        # Either way ``sol`` is not the harmonic solution, and silently
        # returning it would put a wrong background on one frame out of
        # thousands with nothing in the log to find it by.
        log.warning(
            "harmonic_inpaint CG did not converge (info=%s, n=%d, maxiter=%d); "
            "the returned fill is the unconverged iterate",
            info,
            b.size,
            maxiter,
        )
    return sol


def harmonic_inpaint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact Laplace (harmonic) inpainting of ``mask`` pixels, Dirichlet BC from the
    unmasked (valid) pixels, dropping missing neighbors at the frame edge
    (Neumann there). Solved with Jacobi-preconditioned CG on the sparse SPD
    5-point Laplacian (full frame ~2.4M unknowns -- too large for spsolve).

    Replaces ``skimage.restoration.inpaint_biharmonic``: harmonic obeys the
    maximum principle, so the fill cannot overshoot the rim values, whereas
    biharmonic matches rim slopes as well and extrapolates them into large
    holes, which is what produced the +-4-8 e/s patches at strap x bright-star
    mask holes.

    Invalid pixels whose 4-connected component touches no valid pixel
    anywhere (fully enclosed by other invalid pixels / the frame edge, so no
    Dirichlet data reaches them) are excluded from the linear system and set
    to the frame median of valid pixels instead.
    """
    ny, nx = image.shape
    invalid = np.asarray(mask, dtype=bool)
    valid = ~invalid
    out = np.asarray(image, dtype=np.float64).copy()

    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, ncomp = ndi_label(invalid, structure=struct4)
    touches_valid = np.zeros(ncomp + 1, dtype=bool)
    padded_valid = np.zeros((ny + 2, nx + 2), dtype=bool)
    padded_valid[1:-1, 1:-1] = valid
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = padded_valid[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]
        sel = invalid & shifted
        if sel.any():
            touches_valid[labeled[sel]] = True
    isolated_labels = np.flatnonzero(~touches_valid[1:]) + 1
    isolated_mask = (
        np.isin(labeled, isolated_labels) if isolated_labels.size
        else np.zeros_like(invalid)
    )
    frame_median = float(np.median(image[valid])) if valid.any() else 0.0
    if isolated_mask.any():
        out[isolated_mask] = frame_median

    attached = invalid & ~isolated_mask
    coords = np.argwhere(attached)
    n = coords.shape[0]
    if n == 0:
        return out

    id_map = -np.ones((ny, nx), dtype=np.int64)
    id_map[attached] = np.arange(n)
    y, x = coords[:, 0], coords[:, 1]
    k_idx = np.arange(n)
    diag = np.zeros(n)
    b = np.zeros(n)
    rows = []
    cols = []
    data_parts = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yn, xn = y + dy, x + dx
        inb = (yn >= 0) & (yn < ny) & (xn >= 0) & (xn < nx)
        diag[inb] += 1.0
        yin, xin, kk = yn[inb], xn[inb], k_idx[inb]
        nb_id = id_map[yin, xin]
        is_inv_nb = nb_id >= 0
        if is_inv_nb.any():
            rows.append(kk[is_inv_nb])
            cols.append(nb_id[is_inv_nb])
            data_parts.append(-np.ones(int(is_inv_nb.sum())))
        is_val_nb = ~is_inv_nb
        if is_val_nb.any():
            np.add.at(b, kk[is_val_nb], image[yin[is_val_nb], xin[is_val_nb]])
    # Diagonal block appended last, in lockstep across rows/cols/data -- it must
    # stay last in ALL THREE lists (an earlier version put k_idx first in
    # rows/cols but last in data, silently misaligning every entry).
    rows.append(k_idx)
    cols.append(k_idx)
    data_parts.append(diag)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data_parts)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    M = sp.diags(1.0 / A.diagonal())
    sol = _sparse_cg(A, b, M=M, maxiter=20000)
    out[attached] = sol
    return out


def smooth_bkg_decomposed(
    data: np.ndarray,
    *,
    gauss_smooth: float = 0.0,
    boundary_k: int = BOUNDARY_CLIP_K,
    boundary_sigma: float = BOUNDARY_CLIP_SIGMA,
    boundary_rim_width: int = BOUNDARY_CLIP_RIM_WIDTH,
    fill_method: str = "harmonic",
) -> np.ndarray:
    """Notebook ``robust_trend_residual_gap_fill(interpolate=False,
    fill_method="robust")`` branch: KNN-sigma-clip the mask boundary
    (``sanitize_boundary_outliers``) before gap-filling, then optional
    Gaussian smoothing (unchanged from the legacy variant).

    ``fill_method`` selects the gap fill: ``"harmonic"`` (default) is the
    exact Laplace solve (see ``harmonic_inpaint``); ``"biharmonic"`` restores
    the original ``skimage.restoration.inpaint_biharmonic`` behaviour.
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
    if fill_method == "harmonic":
        filled = np.asarray(
            harmonic_inpaint(
                np.nan_to_num(fill_input, nan=0.0), safe_invalid_mask
            ),
            dtype=np.float64,
        )
    elif fill_method == "biharmonic":
        filled = np.asarray(
            inpaint.inpaint_biharmonic(
                np.nan_to_num(fill_input, nan=0.0), safe_invalid_mask
            ),
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"unknown fill_method {fill_method!r}; expected 'harmonic' or 'biharmonic'"
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


def _fit_residual_bkg(residual: np.ndarray, exclude_mask: np.ndarray, res_box: int, n_sigma: float = 5.0,
                      exclude_percentile: Optional[float] = None) -> np.ndarray:
    """Exact notebook residual-surface implementation."""
    if (~exclude_mask).sum() < 4:
        return np.zeros_like(residual)
    finite = residual[~exclude_mask & np.isfinite(residual)]
    med = np.nanmedian(finite) if finite.size else 0.0
    std = np.nanstd(finite) if finite.size else 0.0
    transient = exclude_mask | (np.abs(residual - med) > 5 * std)
    # photutils' default exclude_percentile (10) is passed implicitly; only an explicit value is forwarded.
    bg_kwargs = {} if exclude_percentile is None else {"exclude_percentile": float(exclude_percentile)}
    try:
        corr = Background2D(residual, box_size=res_box, filter_size=3,
                            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
                            bkg_estimator=MedianBackground(), mask=transient,
                            fill_value=0.0, **bg_kwargs).background
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


def fix_bkg_frame_decomposed(bkg_i: np.ndarray, flux_i: np.ndarray, bkgmask_i: np.ndarray, mask: np.ndarray, *, gauss_smooth: float = 2.0, n_sigma: float = 5.0, force_anomaly_repair: bool = False, residual_exclude_percentile: Optional[float] = None) -> np.ndarray:
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
    return gaussian + _fit_residual_bkg(flux_i - gaussian, np.isnan(bkgmask_i), max(4, min(20, min(ny, nx)//2)), n_sigma,
                                          residual_exclude_percentile)


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
    fill_method: str = "harmonic",
    star_mask_pad_px: int = 0,
    extra_exclude: np.ndarray | None = None,
    residual_exclude_percentile: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Notebook ``run_tessreduce_variant`` (``biharmonic_robust``) arithmetic
    for one input frame.

    This is the single shared background estimator used by both
    ``kernel_fit`` and ``background_estimate``; ``boundary_k``/``boundary_sigma``/
    ``boundary_rim_width`` control the KNN sigma-clip applied to the mask
    boundary before gap-filling (see ``sanitize_boundary_outliers``).
    ``fill_method`` (``"harmonic"`` default, or ``"biharmonic"`` for the
    legacy behaviour) is forwarded to ``smooth_bkg_decomposed``.

    ``star_mask_pad_px`` (default 0 = unchanged) grows the catalogue star masks (bits 1|2) by a disk of that radius
    before they are excluded from the fit. The mask circles end inside the star's PSF wing (bit-2 radii 9/8/7/6 px
    for T 8-10/10-11/11-12/12-13), so without padding the fill is solved from rim pixels that still carry the wing,
    which lifts the background under every masked star (localbg_20260930). The padding only shrinks the set of fit
    pixels; every later step (anomaly repair, residual surface) takes its exclusion from the same fit mask.

    ``extra_exclude`` (bool, mask-shaped; default None = unchanged) drops further pixels from the fit, e.g. the
    magnitude-sized star disks of ``star_wing_exclusion`` (stage key ``tessreduce_star_wing_radii``).

    ``residual_exclude_percentile`` (float in (0, 100]; default None = photutils' default 10, bit-identical to before)
    is forwarded as ``exclude_percentile`` to the residual-surface ``Background2D`` only when set (stage key
    ``tessreduce_residual_exclude_percentile``).
    """
    flux = np.asarray(residual, dtype=np.float64)
    if flux.ndim != 2:
        raise ValueError(f"residual background expects 2-D image, got {flux.shape}")
    fit = _fit_mask(mask, star_mask_pad_px, extra_exclude)
    if fit.shape != flux.shape:
        raise ValueError(f"mask shape {fit.shape} != residual shape {flux.shape}")
    bkgmask = np.where(fit, 1.0, np.nan)
    smooth = smooth_bkg_decomposed(
        flux * bkgmask,
        gauss_smooth=smooth_gauss,
        boundary_k=boundary_k,
        boundary_sigma=boundary_sigma,
        boundary_rim_width=boundary_rim_width,
        fill_method=fill_method,
    )
    pre_qe = fix_bkg_frame_decomposed(
        smooth, flux, bkgmask, mask, gauss_smooth=anomaly_gauss,
        force_anomaly_repair=force_anomaly_repair,
        residual_exclude_percentile=residual_exclude_percentile,
    )
    qe = _qe_spline_map(flux, pre_qe, mask, degree=qe_spline_degree, smooth_mult=qe_spline_smooth_mult)
    return pre_qe * qe, pre_qe, qe
