# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Per-stamp local background for the scene fit (``scene_fit --stamp-bg``), 2026-09-30.

Motivation: in the production OOF baseline F1 faint stars (T9.5-13) show a flat ~ -0.17 e-/s/px
offset in the 3-6 px annulus: background-like, per cutout, which a CCD-wide Chebyshev surface
cannot give one cutout. A naive free per-stamp pedestal absorbs the star's own r^-2 halo and
biases bright-star fluxes (+0.47%/mag, 09-17 "chromatic halo = pedestal"), so:

- ``annulus``: one constant per stamp estimated BEFORE the fit from its own data at
  ``r_in <= r <= r_out`` px from the catalogue position (3-sigma-clipped median, iterated;
  pixels within ``nb_radius`` (scaled up for brighter neighbours) of any scene neighbour brighter
  than ``nb_frac`` x the target (catalogue tess_flux) are masked). Subtracted from the data as a
  FIXED offset: no gradient, it cannot trade against the PSF wing.
- ``fit`` (in scene_fit.island_solve): one free constant per stamp solved exactly with the
  fluxes, with a Gaussian prior toward the annulus estimate (sigma = 3 x its robust standard
  error by default).

Pixel partition ("cells"): every union pixel belongs to exactly ONE stamp's pedestal: the stamp
whose centre (integer cx, cy) is nearest (ties -> lower scene index), among the stamps that
contain the pixel. An isolated stamp's cell is its whole square; overlapping stamps split their
overlap along the perpendicular bisector. The cell owner is always in the same overlap island,
so the ``fit`` pedestals stay island-local. (The export's ``owner`` map -- first copy in export
order -- is arbitrary within overlaps, so it is not used for this.)
"""

from __future__ import annotations

import numpy as np

MODES = ("none", "annulus", "fit")
MIN_PIXELS = 8           # fewer usable annulus pixels -> no estimate (NaN, no prior)


def stamp_offsets(S: int):
    k = np.arange(S * S)
    return (k % S - S // 2).astype(np.float64), (k // S - S // 2).astype(np.float64)


def cell_owner(uid, U: int, S: int) -> np.ndarray:
    """(N, S*S) scene index of the stamp whose pedestal each copy's union pixel belongs to."""
    uid = np.asarray(uid)
    N = uid.shape[0]
    ox, oy = stamp_offsets(S)
    d2 = np.broadcast_to(ox * ox + oy * oy, uid.shape).ravel()
    star = np.repeat(np.arange(N), S * S)
    u = uid.ravel()
    order = np.lexsort((star, d2, u))            # by pixel, then distance, then star index
    u_s = u[order]
    first = np.r_[True, u_s[1:] != u_s[:-1]]
    best = np.full(U + 1, -1, np.int64)
    best[u_s[first]] = star[order][first]
    out = best[u]
    dummy = u >= U                               # off-array copies: own stamp (weight is 0 anyway)
    out[dummy] = star[dummy]
    return out.reshape(uid.shape)


def clipped_median(v, clip: float, max_iter: int = 10):
    """Iterated sigma-clipped median; sigma = 1.4826 MAD. Returns (median, sigma, n_used)."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    keep = np.ones(v.size, bool)
    for _ in range(max_iter):
        if keep.sum() < 3:
            break
        med = np.median(v[keep])
        sig = 1.4826 * np.median(np.abs(v[keep] - med))
        if sig <= 0:
            break
        new = np.abs(v - med) <= clip * sig
        if np.array_equal(new, keep):
            break
        keep = new
    if keep.sum() == 0:
        return np.nan, np.nan, 0
    med = float(np.median(v[keep]))
    sig = float(1.4826 * np.median(np.abs(v[keep] - med)))
    return med, sig, int(keep.sum())


def annulus_estimates(data, valid, x0, y0, cx, cy, flux, *, S: int, r_in: float = 5.0,
                      r_out: float = 7.0, clip: float = 3.0, nb_frac: float = 0.01,
                      nb_radius: float = 3.0):
    """Per-stamp annulus background. Returns dict of (N,) arrays:
    ``est`` (e-/s, NaN without MIN_PIXELS usable pixels), ``se`` (robust standard error of the
    median = 1.253 sigma / sqrt(n)), ``sigma``, ``n_annulus`` (valid annulus pixels),
    ``n_masked`` (removed as near neighbours), ``n_clipped`` (removed by the clip), ``n_used``.

    Neighbour mask: every other scene star j with ``flux_j > nb_frac * flux_i`` masks the
    pixels within ``nb_radius * (1 + 0.5 log10(max(flux_j / flux_i, 1)))`` px of it (bigger
    for brighter neighbours; catalogue fluxes, positions x0, y0).
    """
    data = np.asarray(data, np.float64)
    valid = np.asarray(valid, bool)
    x0, y0 = np.asarray(x0, np.float64), np.asarray(y0, np.float64)
    flux = np.nan_to_num(np.asarray(flux, np.float64), nan=0.0)
    N = data.shape[0]
    ox, oy = stamp_offsets(S)
    px = np.asarray(cx, np.float64)[:, None] + ox[None]
    py = np.asarray(cy, np.float64)[:, None] + oy[None]
    out = {k: np.full(N, np.nan) for k in ("est", "se", "sigma")}
    for k in ("n_annulus", "n_masked", "n_clipped", "n_used"):
        out[k] = np.zeros(N, np.int64)
    # neighbour candidates: stars whose position is within r_out + max mask radius of the stamp
    reach = r_out + nb_radius * 3.0 + 1.0
    order = np.argsort(x0)
    xs = x0[order]
    for i in range(N):
        r = np.hypot(px[i] - x0[i], py[i] - y0[i])
        ann = valid[i] & (r >= r_in) & (r <= r_out) & np.isfinite(data[i])
        out["n_annulus"][i] = int(ann.sum())
        lo, hi = np.searchsorted(xs, [x0[i] - reach, x0[i] + reach])
        cand = order[lo:hi]
        cand = cand[(cand != i) & (np.abs(y0[cand] - y0[i]) <= reach)]
        ratio = np.ones(cand.size)             # target without a flux: every neighbour masks
        if cand.size and flux[i] > 0:
            ratio = flux[cand] / flux[i]
            cand, ratio = cand[ratio > nb_frac], ratio[ratio > nb_frac]
        mask = np.zeros_like(ann)
        for j, q in zip(cand, ratio):
            rad = nb_radius * (1.0 + 0.5 * np.log10(max(q, 1.0)))
            mask |= np.hypot(px[i] - x0[j], py[i] - y0[j]) <= rad
        use = ann & ~mask
        out["n_masked"][i] = int((ann & mask).sum())
        if use.sum() < MIN_PIXELS:
            continue
        med, sig, n = clipped_median(data[i][use], clip)
        out["est"][i], out["sigma"][i], out["n_used"][i] = med, sig, n
        out["n_clipped"][i] = int(use.sum()) - n
        out["se"][i] = 1.253 * sig / np.sqrt(max(n, 1))
    return out


def summary(est: dict) -> dict:
    ok = np.isfinite(est["est"])
    n_use = est["n_used"][ok].sum()
    return {
        "n_stamps": int(ok.size), "n_estimated": int(ok.sum()),
        "est_median": float(np.median(est["est"][ok])) if ok.any() else float("nan"),
        "est_p16_p84": [float(v) for v in np.percentile(est["est"][ok], [16, 84])] if ok.any() else [],
        "se_median": float(np.median(est["se"][ok])) if ok.any() else float("nan"),
        "clipped_fraction": float(est["n_clipped"][ok].sum() / max(n_use + est["n_clipped"][ok].sum(), 1)),
        "masked_fraction": float(est["n_masked"].sum() / max(est["n_annulus"].sum(), 1)),
    }
