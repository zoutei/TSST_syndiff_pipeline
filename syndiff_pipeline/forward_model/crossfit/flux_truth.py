# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""A4 (part 2): out-of-fold flux vs Gaia-XP synthetic TESS magnitude.

    python -m syndiff_pipeline.forward_model.crossfit.flux_truth --merged merged.npz --xp xp_tess.csv --scene-dir <scene> --out dm.json

Model (plan A4), fitted per star i on isolated, non-M trainees with an out-of-fold flux:

    dm_i = -2.5 log10 f_i + Z - T^XP_i  ~  beta_T (T_i - 11) + sum_{k=1..3} c_k (u_i - u0)^k + (const absorbed in Z)

  f_i     out-of-fold re-solved flux (e-/s), from merge_oof (flux_oof)
  T^XP_i  XP synthetic TESS magnitude (xp_tess; its arbitrary constant is absorbed in Z)
  T_i     catalogue TESS magnitude (scene tess_mag)
  u_i     colour = Gaia BP-RP (scene bundle); u0 = its median over the fitted stars
  Z       zero point (fitted)
  beta_T  brightness slope (mag per mag); G3-style target |beta_T| <= 5e-4
  c_k     cubic colour polynomial: absorbs XP truncation at 1020 nm, bandpass and colour-model residuals

Fit: robust (Huber, scale 0.01 mag) least squares, equal weight per star (no S/N weights: they fake brightness trends).
Isolated: no other scene star with tess_mag < T_i + 2 within 6 px. RUWE is not available locally and is not cut.
Also reported: the axis-region (< 900 px from the optical axis) minus far-side (> 1800 px) median of the
colour/brightness-corrected dm, and M stars (BP-RP > 2) separately (response-curve choice is +-5% for them).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def isolated(x, y, T, rad=6.0, dmag=2.0):
    tr = cKDTree(np.c_[x, y])
    iso = np.ones(len(x), bool)
    for i, nb in enumerate(tr.query_ball_point(np.c_[x, y], rad)):
        for j in nb:
            if j != i and T[j] < T[i] + dmag:
                iso[i] = False
                break
    return iso


def design(T, u, u0):
    du = u - u0
    return np.c_[np.ones_like(T), T - 11.0, du, du ** 2, du ** 3]


def robust_fit(A, y, scale=0.01):
    p0 = np.linalg.lstsq(A, y, rcond=None)[0]
    r = least_squares(lambda p: A @ p - y, p0, loss="huber", f_scale=scale)
    J = r.jac
    res = A @ r.x - y
    s2 = (1.4826 * np.median(np.abs(res - np.median(res)))) ** 2
    cov = np.linalg.pinv(J.T @ J) * s2
    return r.x, np.sqrt(np.diag(cov)), res


def run(merged, xp, scene_dir, axis, flux_key="flux_oof", sel_key="ok_oof"):
    m = dict(np.load(merged))
    xpt = pd.read_csv(xp).set_index("source_id").reindex(m["source_id"])
    z = np.load(Path(scene_dir) / "scene_bundle.npz")
    assert (z["source_id"] == m["source_id"]).all()
    x, y, T = m["x"], m["y"], m["T"]
    meta = json.loads((Path(scene_dir) / "scene_meta.json").read_text())
    bp = np.asarray(np.load(meta["source_bundle"])["bp_rp"], float)[z["star_bundle_index"]]
    f = m[flux_key]
    TX = xpt["T_XP"].values
    iso = isolated(x, y, T)
    base = m["trainee"] & m[sel_key] & np.isfinite(f) & (f > 0) & np.isfinite(TX) & np.isfinite(bp) & iso
    mstar = bp > 2.0
    dm_raw = -2.5 * np.log10(np.where(f > 0, f, np.nan)) - TX
    out = {"n_base": int(base.sum()), "n_mstar": int((base & mstar).sum()), "rho_axis": None}
    s = base & ~mstar
    u0 = float(np.median(bp[s]))
    A = design(T[s], bp[s], u0)
    p, pe, res = robust_fit(A, dm_raw[s])
    out.update(Z=float(p[0]), beta_T=float(p[1]), beta_T_err=float(pe[1]), colour_poly=p[2:].tolist(),
               colour_poly_err=pe[2:].tolist(), u0=u0, rms_mmag=float(1e3 * 1.4826 * np.median(np.abs(res))), n_fit=int(s.sum()))
    corr = dm_raw - design(T, bp, u0) @ p           # colour/brightness-corrected dm for every star
    r = np.hypot(x - axis[0], y - axis[1])
    near, far = s & (r < 900), s & (r > 1800)
    out["axis_minus_far_permil"] = float(1e3 * np.log(10) / 2.5 * (np.median(corr[near]) - np.median(corr[far]))) \
        if near.sum() > 20 and far.sum() > 20 else None
    out["n_near_far"] = [int(near.sum()), int(far.sum())]
    tb = {}
    for lo, hi in ((8, 9), (9, 10), (10, 11), (11, 12), (12, 13)):
        b = s & (T >= lo) & (T < hi)
        tb[f"{lo}-{hi}"] = dict(n=int(b.sum()), med_mmag=float(1e3 * np.median(corr[b] + p[1] * (T[b] - 11))) if b.sum() else None)
    out["dm_by_T_colour_corrected_mmag"] = tb
    ms = base & mstar
    out["mstar_med_resid_mmag"] = float(1e3 * np.median(corr[ms])) if ms.sum() > 5 else None
    return out, dict(dm_raw=dm_raw, corr=corr, fit=s, mstar=ms, T=T, bp=bp, x=x, y=y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merged", required=True)
    ap.add_argument("--xp", required=True)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--axis", required=True, help="optical axis x,y in science px")
    ap.add_argument("--flux", default="flux_oof", help="flux_oof or flux_in")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ax = tuple(float(v) for v in a.axis.split(","))
    out, arr = run(a.merged, a.xp, a.scene_dir, ax, a.flux, "ok_oof" if a.flux == "flux_oof" else "ok_in")
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(a.out).with_suffix(".npz"), **arr)
    print(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
