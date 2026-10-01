"""Template-kernel numerics used by the ``final`` stage: blended convolution, photometric matching, FITS writer.

Verbatim ports (numerics unchanged) of the e2e copies ``perband/tk/{convolve,match,make_diff}.py`` (themselves copies of
the Phase A dev ``template_kernel`` modules), minus everything that imported the dev fitter pin or hard-coded F1 paths:

* convolve:  ``load_grid``, ``subcell_centres_sci``, ``hat_weights``, ``convolve_blended``, ``block_sum``, ``trim``
* match:     ``cheb_terms``, ``cheb_design``, ``eval_a``, ``fit_match``, ``bright_radius``, ``bright_star_mask``,
             ``selection``, ``run_match``, ``load_science``, ``load_hp_planes``
* make_diff: ``write_fits`` (the ``_write`` helper: lossless GZIP_1 + exact round-trip assert)

Difference: the Tmag<13 exclusion footprints come from an explicit ``stars`` table (columns ``x``, ``y``, ``tmag`` in
science-local pixels; the fit scene's stars) instead of a module-global Gaia table that the e2e script monkey-patched in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.fft
from astropy.io import fits
from scipy.signal import fftconvolve

OS = 4
SCI_BOUNDS = dict(x_min=44, x_max=2092, y_min=0, y_max=2048, shape=(2048, 2048))


# ------------------------------------------------------------------ geometry / convolution
def load_grid(mapping_root: Path):
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    masters = sorted(Path(mapping_root).glob("*master_pixels2skycells_os4.fits*"))
    if len(masters) != 1:
        raise FileNotFoundError(f"expected one os4 master in {mapping_root}: {masters}")
    return load_mapping_grid_from_master(masters[0])


def subcell_centres_sci(grid, axis: str) -> np.ndarray:
    """Science-local native coordinate of every template subcell centre along x or y.

    Mapping convention (common/mapping_grid.create_coords_for_grid):
        ffi = ffi_min + (I + 0.5)/F - 0.5 ; science-local = ffi - science_min_ffi.
    """
    F = int(grid.oversampling)
    if axis == "x":
        n, fmin, smin = grid.width_os, grid.ffi_xmin, grid.science_xmin_ffi
    else:
        n, fmin, smin = grid.height_os, grid.ffi_ymin, grid.science_ymin_ffi
    I = np.arange(n, dtype=np.float64)
    return (fmin - smin) + (I + 0.5) / F - 0.5


def hat_weights(coord: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """(n_nodes, len(coord)) piecewise-linear node weights, replicating EM.bilinear_cell
    (searchsorted side='right', cell index clipped to [0, n-2], weight clipped to [0, 1])."""
    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.size
    j0 = np.clip(np.searchsorted(nodes, coord, side="right") - 1, 0, n - 2)
    w = np.clip((coord - nodes[j0]) / (nodes[j0 + 1] - nodes[j0]), 0.0, 1.0)
    H = np.zeros((n, coord.size), dtype=np.float64)
    idx = np.arange(coord.size)
    H[j0, idx] += 1.0 - w
    H[j0 + 1, idx] += w
    return H


def block_sum(a: np.ndarray, F: int = OS) -> np.ndarray:
    H, W = a.shape
    return a.reshape(H // F, F, W // F, F).sum(axis=(1, 3))


def trim(native_padded: np.ndarray, grid) -> np.ndarray:
    from syndiff_pipeline.common.grid_pairing import trim_padded_products
    return np.asarray(trim_padded_products(native_padded, grid=grid))


def convolve_blended(T: np.ndarray, K: np.ndarray, grid, node_x, node_y, *,
                     workers: int = 4, verbose: bool = False, return_stats: bool = False):
    """sum_n (w_n T) (*) K_n on the full padded OS grid (float64 output, same shape as T).

    w_n(X, Y) = the fitter's bilinear node weight at the SOURCE subcell centre (flat extrapolation outside the node
    grid); (*) is true convolution with fftconvolve semantics (odd N, centre (N-1)/2 = zero offset).  sum_n w_n = 1 and
    each K_n sums to 1, so flux is conserved up to what leaves the padded array."""
    import time
    T = np.asarray(T)
    nr, nc, N, _ = K.shape
    assert len(node_x) == nc and len(node_y) == nr
    m = (N - 1) // 2
    H, W = T.shape
    hx = hat_weights(subcell_centres_sci(grid, "x"), node_x)   # (nc, W)
    hy = hat_weights(subcell_centres_sci(grid, "y"), node_y)   # (nr, H)
    out = np.zeros((H, W), dtype=np.float64)
    stats = []
    t0 = time.time()
    with scipy.fft.set_workers(workers):
        for i in range(nr):
            rows = np.nonzero(hy[i])[0]
            if rows.size == 0:
                continue
            r0, r1 = rows[0], rows[-1] + 1
            for j in range(nc):
                cols = np.nonzero(hx[j])[0]
                if cols.size == 0:
                    continue
                c0, c1 = cols[0], cols[-1] + 1
                A = T[r0:r1, c0:c1].astype(np.float64) * hy[i, r0:r1, None] * hx[j, None, c0:c1]
                C = fftconvolve(A, K[i, j], mode="full")          # index p <-> r0 - m + p
                orow0, ocol0 = r0 - m, c0 - m
                pr0, pc0 = max(0, -orow0), max(0, -ocol0)
                pr1 = C.shape[0] - max(0, orow0 + C.shape[0] - H)
                pc1 = C.shape[1] - max(0, ocol0 + C.shape[1] - W)
                out[orow0 + pr0:orow0 + pr1, ocol0 + pc0:ocol0 + pc1] += C[pr0:pr1, pc0:pc1]
                if return_stats:
                    stats.append(dict(i=i, j=j, sum_wT=float(A.sum()), sum_K=float(K[i, j].sum()),
                                      sum_conv_all=float(C.sum()), sum_conv_kept=float(C[pr0:pr1, pc0:pc1].sum())))
                del A, C
            if verbose:
                print(f"  node row {i}/{nr} done ({time.time() - t0:.1f}s)", flush=True)
    if return_stats:
        return out, stats
    return out


# ------------------------------------------------------------------ photometric matching
def load_science(ffi, bkg_path):
    """sci - bkg (science-local 2048^2), exactly as the Hotpants frame builds it."""
    from syndiff_pipeline.difference_imaging.stages import hotpants as HP
    sci, _ = HP._load_ffi_cropped(str(ffi), SCI_BOUNDS)
    bkg = np.asarray(fits.getdata(bkg_path, 1), dtype=float)
    assert bkg.shape == sci.shape
    return sci - bkg


def load_hp_planes(hp_d_path):
    with fits.open(hp_d_path) as h:
        return (np.asarray(h[1].data, dtype=np.float64), np.asarray(h[2].data, dtype=np.float64),
                np.asarray(h[3].data))


def bright_radius(tmag):
    """Exclusion radius [native px] for a Tmag<13 star removed from the template.
    r = 3 px at Tmag 13, growing 3 px per magnitude brighter, capped at 40 px."""
    return np.clip(3.0 + 3.0 * (13.0 - np.asarray(tmag)), 3.0, 40.0)


def bright_star_mask(stars, shape=(2048, 2048), tmag_lim: float = 13.0, scale: float = 1.0):
    """True where a pixel lies in the footprint of a Tmag<tmag_lim star of ``stars`` (columns x, y, tmag)."""
    g = stars[stars.tmag < tmag_lim]
    H, W = shape
    m = np.zeros(shape, dtype=bool)
    for x, y, r in zip(g.x.to_numpy(), g.y.to_numpy(), scale * bright_radius(g.tmag.to_numpy())):
        if x < -r or y < -r or x > W - 1 + r or y > H - 1 + r:
            continue
        x0, x1 = max(0, int(np.floor(x - r))), min(W, int(np.ceil(x + r)) + 1)
        y0, y1 = max(0, int(np.floor(y - r))), min(H, int(np.ceil(y + r)) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    return m


def cheb_terms(ko: int):
    return [(p, q) for d in range(ko + 1) for p in range(d + 1) for q in [d - p]]


def cheb_design(x, y, ko: int):
    """(npix, nterm) Chebyshev T_p(xn) T_q(yn), p+q <= ko."""
    from numpy.polynomial import chebyshev as Ch
    xn = (np.asarray(x, dtype=np.float64) - 1023.5) / 1024.0
    yn = (np.asarray(y, dtype=np.float64) - 1023.5) / 1024.0
    Tx = np.stack([Ch.chebval(xn, np.eye(ko + 1)[p]) for p in range(ko + 1)], axis=-1)
    Ty = np.stack([Ch.chebval(yn, np.eye(ko + 1)[q]) for q in range(ko + 1)], axis=-1)
    return np.stack([Tx[..., p] * Ty[..., q] for p, q in cheb_terms(ko)], axis=-1)


def eval_a(coef, ko: int, shape=(2048, 2048)):
    H, W = shape
    out = np.empty(shape, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    for r0 in range(0, H, 256):
        yy, xx = np.mgrid[r0:min(H, r0 + 256), 0:W]
        out[r0:r0 + yy.shape[0]] = cheb_design(xx, yy, ko) @ coef
    return out


def fit_match(target, C, noise, good, ko: int, *, clip: float = 5.0, max_iter: int = 20):
    """Weighted LSQ target ~= a(x,y) C + b on `good` pixels; iterative clip on |r|/noise."""
    yy, xx = np.nonzero(good)
    t = target[yy, xx]
    c = C[yy, xx]
    s = noise[yy, xx]
    Pm = cheb_design(xx, yy, ko)                       # (n, nt)
    A = np.concatenate([Pm * c[:, None], np.ones((len(t), 1))], axis=1)
    w = 1.0 / s ** 2
    keep = np.ones(len(t), dtype=bool)
    hist = []
    for it in range(max_iter):
        Aw = A[keep] * w[keep, None]
        coef = np.linalg.solve(Aw.T @ A[keep], Aw.T @ t[keep])
        z = (t - A @ coef) / s
        # robust scale of the normalised residual (not assumed to be 1)
        med = np.median(z[keep])
        mad = 1.4826 * np.median(np.abs(z[keep] - med))
        new = np.abs(z - med) < clip * mad
        hist.append(dict(it=it, n_keep=int(keep.sum()), chi2_red=float(np.mean(z[keep] ** 2)),
                         z_med=float(med), z_mad=float(mad)))
        if np.array_equal(new, keep):
            break
        keep = new
    nt = Pm.shape[1]
    cov = np.linalg.inv((A[keep] * w[keep, None]).T @ A[keep])
    res = dict(ko=ko, terms=cheb_terms(ko), coef_a=coef[:nt].tolist(), b=float(coef[nt]),
               coef_err=np.sqrt(np.diag(cov)).tolist(), n_good=int(len(t)),
               n_keep=int(keep.sum()), frac_clipped=float(1 - keep.mean()),
               chi2_red_keep=float(np.mean(z[keep] ** 2)), z_mad_keep=float(hist[-1]["z_mad"]),
               n_iter=len(hist), history=hist)
    keep_img = np.zeros(target.shape, dtype=bool)
    keep_img[yy[keep], xx[keep]] = True
    return res, keep_img


def selection(noise, mask, stars, *, bright_scale: float = 1.0):
    # Hotpants output mask: accept 0 and pure FLAG_OK_CONV (64), as pyhotpants itself does
    # (pure/utils.py "reject any flag except pure FLAG_OK_CONV") and as the evaluator does.
    good = np.isin(mask, (0, 64)) & (noise > 0) & np.isfinite(noise)
    bm = bright_star_mask(stars, noise.shape, scale=bright_scale)
    return good & ~bm, dict(n_hp_good=int(good.sum()), n_bright_excl=int((good & bm).sum()))


def run_match(target, C, noise, mask, ko: int, stars, out_dir: Path | None = None, tag: str = ""):
    good, sel = selection(noise, mask, stars)
    good &= np.isfinite(target) & np.isfinite(C)
    res, keep = fit_match(target, C, noise, good, ko)
    res.update(sel)
    a = eval_a(np.asarray(res["coef_a"]), ko, target.shape)
    res["a_stats"] = dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                          p1=float(np.percentile(a, 1)), p99=float(np.percentile(a, 99)))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"match{tag}.json").write_text(json.dumps(res, indent=1))
    return res, a, keep


# ------------------------------------------------------------------ output
def write_fits(path: Path, primary, pairs):
    """pairs: [(array, header)]; lossless GZIP_1 + exact round-trip assert (as finish.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus = [fits.PrimaryHDU(header=primary)]
    for arr, hdr in pairs:
        hdus.append(fits.CompImageHDU(data=arr, header=hdr, compression_type="GZIP_1", quantize_level=0))
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)
    with fits.open(path) as chk:
        for i, (arr, _) in enumerate(pairs, 1):
            assert np.array_equal(chk[i].data, arr, equal_nan=True), (path, i)
