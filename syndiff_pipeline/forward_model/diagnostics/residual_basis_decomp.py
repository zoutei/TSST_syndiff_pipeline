# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Decompose each star's fixed (time-averaged) residual onto physically
interpretable basis vectors, to separate *which* model defect it represents.

Background: at the stage-2 plateau, 99.8% of the excess chi variance in
r in [1.5,3.5) is a per-star pattern that is fixed in time (see
HANDOFF_epsf_wing_stage2.md "Excess-variance diagnosis"). Ruled out so far:
bad frames, flux-solve scatter, trajectory jitter, sub-pixel rendering phase,
and spatial-variation grid coarseness (2x2 vs 3x3). The remaining candidates
differ in the *shape* of the residual they produce, which is what this
measures.

To first order, a small model error shows up as a residual proportional to
the derivative of the model with respect to the mis-set quantity:

    flux error      ->  B_f = M                      (even, same profile)
    position error  ->  B_x = dM/dx,  B_y = dM/dy    (ODD / dipole)
    width error     ->  B_w = 2M + (x-x0) dM/dx + (y-y0) dM/dy   (EVEN)

``B_w`` is the dilation generator: for M(x) = f P((x-x0)/s) / s^2 (flux
preserved under a width change), -dM/ds at s=1 is exactly that combination.

SIGN, and it is easy to get backwards -- an earlier version of the print label
below had it inverted. Since ``B_w = -dM/ds``, data that are broader by a factor
``s`` give ``data ~ M + (s-1) dM/ds = M - (s-1) B_w``, so the residual
``data - model`` is ``-(s-1) B_w`` and the fitted coefficient is ``a_w = -(s-1)``.
**A POSITIVE ``a_w`` therefore means the data are NARROWER than the model**, and
``s = 1 - a_w``.
So a star whose PSF is genuinely broader than the model (brighter-fatter,
chromatic width, focus) loads on ``B_w``, while a star whose assumed centroid
or PSF-matching kernel is off loads on ``B_x``/``B_y``. These are linearly
independent and, being opposite in parity, largely uncorrelated -- which is
what makes the split informative rather than a fitting degeneracy.

The fitted coefficients are directly interpretable: ``a_x``/``a_y`` in
pixels, ``a_w`` as a fractional width error. Whatever variance is left after
projecting all four out is, by construction, *not* explainable as a flux,
position, or width error of a single centered source -- e.g. contamination
from a second source, or a genuine PSF shape mismatch beyond a width scale.

Item 2 (brighter-fatter) falls out of the same numbers: brighter-fatter
predicts ``a_w`` grows with stellar flux. Chromatic width predicts ``a_w``
tracks BP-RP colour instead. Both are reported, along with the partial
correlation of each controlling for the other, since brightness and colour
are themselves correlated in a magnitude-limited sample.

Read-only: evaluates a frozen checkpoint, touches no model/loss/optimizer
code and runs no training.

Usage:
    python -m syndiff_pipeline.forward_model.diagnostics.residual_basis_decomp \\
        [run_dir] [params_name] [--mag-lo 9] [--mag-hi 11] [--clean-only]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import _bootstrap  # noqa: F401
from .. import fit as FIT
from .excess_variance_diag import _forward_and_flux, _radius_grid
from .radial_profile import build_full_context

R_ANNULUS_LO = 1.5
R_ANNULUS_HI = 3.5


def build_basis(model2d: np.ndarray, x0: float, y0: float) -> dict[str, np.ndarray]:
    """Flux / position / width basis vectors from a single model stamp.

    ``x0, y0`` are the source's sub-pixel position *relative to the stamp
    centre pixel*; the width generator's moment arm has to be measured from
    where the star actually is, not from the stamp centre, or a centring
    offset leaks into the width channel.
    """
    S = model2d.shape[0]
    c = S // 2
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    dx = xx - (c + x0)
    dy = yy - (c + y0)
    # np.gradient returns d/drow, d/dcol -> (d/dy, d/dx)
    gy, gx = np.gradient(model2d)
    return {
        "flux": model2d,
        "x": gx,
        "y": gy,
        "width": 2.0 * model2d + dx * gx + dy * gy,
    }


def weighted_lstsq(
    resid: np.ndarray, basis: dict[str, np.ndarray], weight: np.ndarray,
) -> dict[str, float]:
    """GLS fit of ``resid`` onto the basis; returns coefficients + variance split."""
    names = list(basis.keys())
    A = np.stack([basis[n].ravel() for n in names], axis=1)  # (n_pix, n_basis)
    r = resid.ravel()
    w = weight.ravel()
    ok = np.isfinite(r) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(A), axis=1)
    if ok.sum() < len(names) + 2:
        return {n: float("nan") for n in names} | {"frac_explained": float("nan"),
                                                   "resid_rms_before": float("nan"),
                                                   "resid_rms_after": float("nan")}
    A, r, w = A[ok], r[ok], w[ok]
    sw = np.sqrt(w)
    Aw, rw = A * sw[:, None], r * sw
    coef, *_ = np.linalg.lstsq(Aw, rw, rcond=None)
    pred = A @ coef
    ss_before = float(np.sum(w * r**2))
    ss_after = float(np.sum(w * (r - pred) ** 2))
    out = {n: float(c) for n, c in zip(names, coef)}
    out["frac_explained"] = 1.0 - ss_after / ss_before if ss_before > 0 else float("nan")
    out["resid_rms_before"] = float(np.sqrt(ss_before / w.sum()))
    out["resid_rms_after"] = float(np.sqrt(ss_after / w.sum()))
    # Per-channel explained fraction: drop one basis vector at a time.
    for i, n in enumerate(names):
        keep = [j for j in range(len(names)) if j != i]
        c2, *_ = np.linalg.lstsq(Aw[:, keep], rw, rcond=None)
        ss_wo = float(np.sum(w * (r - A[:, keep] @ c2) ** 2))
        out[f"frac_{n}"] = (ss_wo - ss_after) / ss_before if ss_before > 0 else float("nan")
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rho + two-sided p (t-approximation); avoids a scipy dependency."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 4:
        return float("nan"), float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho**2))
    # two-sided p via a normal approximation to the t distribution
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))
    return rho, float(p)


def partial_spearman(a: np.ndarray, b: np.ndarray, ctrl: np.ndarray) -> tuple[float, float]:
    """Spearman of a vs b after linearly removing ctrl from both (rank space)."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(ctrl)
    a, b, c = a[ok], b[ok], ctrl[ok]
    if a.size < 5:
        return float("nan"), float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rc = np.argsort(np.argsort(c)).astype(float)
    def resid(v):
        A = np.stack([np.ones_like(rc), rc], axis=1)
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    return spearman(resid(ra), resid(rb))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, nargs="?",
                   default=Path("dev/forward_epsf_wcs/output/runs/scale1_mag911_600"))
    p.add_argument("params_name", type=str, nargs="?", default="params_stage2.npz")
    p.add_argument("--mag-lo", type=float, default=9.0)
    p.add_argument("--mag-hi", type=float, default=11.0)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--region", type=str, default="1536,1536,2048,2048")
    p.add_argument("--epsf-grid", type=str, default=None)
    p.add_argument("--clean-only", action="store_true",
                   help="restrict to n_members==1 groups (no attached companion)")
    p.add_argument("--annulus-only", action="store_true",
                   help="fit only r in [1.5,3.5) instead of the whole stamp")
    args = p.parse_args(argv)

    params = FIT.load_params_npz(args.run_dir / args.params_name)
    if args.epsf_grid:
        n_rows, n_cols = (int(v) for v in args.epsf_grid.lower().split("x"))
    else:
        n_rows = int(params["epsf_base_raw"].shape[0])
        n_cols = int(params["epsf_base_raw"].shape[1])
    print(f"checkpoint {args.run_dir / args.params_name}  epsf_grid={n_rows}x{n_cols}")

    rebuilt = build_full_context(
        mag_lo=args.mag_lo, mag_hi=args.mag_hi, n_frames=args.n_frames,
        region_str=args.region, epsf_grid=(n_rows, n_cols),
    )
    fd, groups, mags = rebuilt["fd"], rebuilt["groups"], rebuilt["mags"]
    stars = rebuilt["expanded_stars"]
    out = _forward_and_flux(fd, params)

    data = np.asarray(fd.data)
    model = out["model"]
    var = out["var"]
    w_mask = out["w_mask"]
    S = data.shape[-1]
    rad = _radius_grid(S)
    fit_mask2d = ((rad >= R_ANNULUS_LO) & (rad < R_ANNULUS_HI)) if args.annulus_only \
        else np.ones_like(rad, dtype=bool)

    n_members = groups.valid.sum(axis=1)
    gis = np.where(n_members == 1)[0] if args.clean_only else np.arange(groups.n_groups)

    # colour, where available
    bp = stars["phot_bp_mean_mag"].to_numpy(dtype=float) if "phot_bp_mean_mag" in stars else None
    rp = stars["phot_rp_mean_mag"].to_numpy(dtype=float) if "phot_rp_mean_mag" in stars else None

    rows = []
    for gi in gis:
        slots = np.where(groups.valid[gi])[0]
        if slots.size == 0:
            continue
        slot = int(slots[0])
        star = int(groups.members[gi, slot])
        m = w_mask[gi] & fit_mask2d[None, :, :]
        if m.sum() == 0:
            continue
        # fixed (time-averaged) component
        cnt = m.sum(axis=0)
        ok2d = cnt > 0
        if ok2d.sum() < 12:
            continue
        resid_fixed = np.where(ok2d, (data[gi] - model[gi]).sum(axis=0) / np.maximum(cnt, 1), 0.0)
        model_fixed = np.where(ok2d, (model[gi] * m).sum(axis=0) / np.maximum(cnt, 1), 0.0)
        var_mean = np.where(ok2d, (var[gi] * m).sum(axis=0) / np.maximum(cnt, 1), np.inf)
        weight = np.where(ok2d, cnt / np.maximum(var_mean, 1e-12), 0.0)

        x0 = float(np.mean(out["x_slot"][gi, slot, :])) - float(fd.ctx.stamp_center_x[gi])
        y0 = float(np.mean(out["y_slot"][gi, slot, :])) - float(fd.ctx.stamp_center_y[gi])
        basis = build_basis(model_fixed, x0, y0)
        fit = weighted_lstsq(resid_fixed, basis, weight)

        peak = float(np.max(np.abs(model_fixed)))
        # NB: ``fit`` has its own "flux" key (the flux *basis* coefficient), so the
        # star's fitted flux is stored under a distinct name -- **fit is splatted
        # last and would otherwise silently overwrite it.
        rows.append({
            "gi": int(gi), "star": star, "mag": float(mags[star]),
            "star_flux": float(np.mean(out["flux"][gi, :, slot])),
            "color": float(bp[star] - rp[star]) if (bp is not None and rp is not None) else np.nan,
            "peak": peak, **fit,
        })

    if not rows:
        raise SystemExit("no usable groups")

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    n = len(rows)
    print(f"\nn_groups analysed: {n}  ({'clean only' if args.clean_only else 'all groups'}; "
          f"{'annulus' if args.annulus_only else 'full stamp'})")

    print("\n=== how much of each star's FIXED residual is explained by "
          "flux/position/width ===")
    fe = col("frac_explained")
    print(f"  total frac explained : median={np.nanmedian(fe):.3f}  "
          f"p16={np.nanpercentile(fe,16):.3f} p84={np.nanpercentile(fe,84):.3f}")
    for ch in ("flux", "x", "y", "width"):
        v = col(f"frac_{ch}")
        print(f"  unique to {ch:<6}     : median={np.nanmedian(v):+.4f}  "
              f"p84={np.nanpercentile(v,84):+.4f}")
    print(f"  UNEXPLAINED leftover : median={1-np.nanmedian(fe):.3f}  "
          f"(residual RMS {np.nanmedian(col('resid_rms_before')):.2f} -> "
          f"{np.nanmedian(col('resid_rms_after')):.2f} counts)")

    ax, ay, aw = col("x"), col("y"), col("width")
    print("\n=== implied physical mis-set (fixed component) ===")
    print(f"  position |a_x| median={np.nanmedian(np.abs(ax)):.4f} px  "
          f"p84={np.nanpercentile(np.abs(ax),84):.4f} px")
    print(f"  position |a_y| median={np.nanmedian(np.abs(ay)):.4f} px  "
          f"p84={np.nanpercentile(np.abs(ay),84):.4f} px")
    print(f"  width    a_w   median={np.nanmedian(aw):+.5f}  "
          f"p16={np.nanpercentile(aw,16):+.5f} p84={np.nanpercentile(aw,84):+.5f}  "
          f"(fractional; >0 = data NARROWER than model)")

    print("\n=== item 2: does the width error scale with brightness "
          "(brighter-fatter) or colour (chromatic)? ===")
    mag, flux, color = col("mag"), col("star_flux"), col("color")
    r_mf = np.corrcoef(mag, np.log10(np.where(flux > 0, flux, np.nan)))[0, 1]
    print(f"  [sanity] corr(tess_mag, log10 star_flux) = {r_mf:+.4f} (must be near -1)")
    logflux = np.log10(np.where(flux > 0, flux, np.nan))
    for label, v in (("tess_mag", mag), ("log10 flux", logflux), ("BP-RP colour", color)):
        rho, pv = spearman(v, aw)
        print(f"  a_w vs {label:<13}: Spearman rho={rho:+.3f}  p={pv:.3g}")
    rho, pv = partial_spearman(aw, logflux, color)
    print(f"  a_w vs log10 flux | colour controlled : rho={rho:+.3f}  p={pv:.3g}")
    rho, pv = partial_spearman(aw, color, logflux)
    print(f"  a_w vs colour     | flux   controlled : rho={rho:+.3f}  p={pv:.3g}")

    print("\n=== brightness-binned width error (brighter-fatter would be monotonic) ===")
    order = np.argsort(mag)
    nb = 4
    print(f"{'mag bin':>16} {'n':>4} {'median a_w':>12} {'median |a_x|':>13}")
    for b in range(nb):
        sel = order[b * n // nb:(b + 1) * n // nb]
        if sel.size == 0:
            continue
        print(f"{mag[sel].min():6.2f}-{mag[sel].max():<6.2f} {sel.size:5d} "
              f"{np.nanmedian(aw[sel]):12.5f} {np.nanmedian(np.abs(ax[sel])):13.4f}")

    print("\n=== worst 10 groups by unexplained residual (candidates for "
          "contamination / genuine shape mismatch) ===")
    left = col("resid_rms_after")
    worst = np.argsort(-left)[:10]
    print(f"{'gi':>5} {'mag':>6} {'a_x':>8} {'a_y':>8} {'a_w':>9} {'expl':>6} {'left_rms':>9}")
    for i in worst:
        r = rows[i]
        print(f"{r['gi']:5d} {r['mag']:6.2f} {r['x']:8.4f} {r['y']:8.4f} "
              f"{r['width']:9.5f} {r['frac_explained']:6.3f} {r['resid_rms_after']:9.2f}")


if __name__ == "__main__":
    main()
