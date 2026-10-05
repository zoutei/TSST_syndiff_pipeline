# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""V0.4 convergence gate: is a fit's held-out chi^2 flat at the end of training? (D16(b), 2026-09-30)

    python -m syndiff_pipeline.forward_model.crossfit.converge lcurve1.json [lcurve2.json ...] [--window 1500] [--tol 0.01]

Input: lcurve.py outputs (median core chi^2 of out-of-fold and in-fold trainees per Tmag bin at each checkpoint).
For each file and Tmag bin, fit a straight line to ln(median chi^2) against ePSF training step over the last
``--window`` steps:

    ln chi2_b(s) ~ a_b + g_b * s                       (s in units of 1000 ePSF steps)

  chi2_b(s)  median core chi^2 of bin b's held-out (or in-fold) stars at checkpoint step s
  g_b        fractional change per 1000 steps (fitted; its error from the fit residuals)

Two conditions per held-out bin, both required ("converged AND non-rising", user D16(b) 2026-09-30):
  flat        g_b <= tol (default 1% per 1000 steps) over the last ``--window`` steps
  non-rising  chi2_b(end) <= (1 + rise_tol) * min_s chi2_b(s) over the whole run (default rise_tol 2%)
A slope test alone is not enough: the 09-29 K=2 moment-blind fits rose x1.45-2.2 and then flattened, so they
pass "flat" while having overfit. A fit passes when every bin passes both. Needs >= 4 checkpoints in the window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def slopes(rows, key, window):
    last = rows[-1]["epsf_step"]
    sel = [r for r in rows if r["epsf_step"] >= last - window]
    out = {}
    for b in rows[0][key]:
        s = np.array([r["epsf_step"] for r in sel], float) / 1000.0
        y = np.log([r[key][b] for r in sel])
        if len(s) < 4:
            out[b] = dict(n=len(s), g=None, g_err=None)
            continue
        A = np.c_[np.ones_like(s), s]
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        cov = np.linalg.inv(A.T @ A) * (r @ r) / max(len(s) - 2, 1)
        out[b] = dict(n=len(s), g=float(coef[1]), g_err=float(np.sqrt(cov[1, 1])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lcurves", nargs="+")
    ap.add_argument("--window", type=float, default=1500, help="ePSF steps at the end of training to fit")
    ap.add_argument("--tol", type=float, default=0.01, help="max allowed held-out rise per 1000 steps (fraction)")
    ap.add_argument("--rise-tol", type=float, default=0.02, help="max allowed end/min - 1 of held-out chi2 (fraction)")
    a = ap.parse_args(argv)
    report = {}
    for p in a.lcurves:
        rows = json.load(open(p))
        ho, inf = slopes(rows, "oof", a.window), slopes(rows, "inf", a.window)
        for b, v in ho.items():
            ser = [r["oof"][b] for r in rows]
            v["end_over_min"] = float(ser[-1] / min(ser))
            v["step_of_min"] = rows[int(np.argmin(ser))]["epsf_step"]
            v["flat"] = v["g"] is not None and v["g"] <= a.tol
            v["non_rising"] = v["end_over_min"] <= 1 + a.rise_tol
        ok = all(v["flat"] and v["non_rising"] for v in ho.values())
        report[p] = dict(converged=ok, rise_tol=a.rise_tol, last_step=rows[-1]["epsf_step"], window=a.window, tol=a.tol,
                         heldout=ho, infold=inf)
        print(Path(p).stem.ljust(24), "PASS" if ok else "FAIL",
              " ".join(f"{b}:" + ("n/a" if v["g"] is None else f"{100*v['g']:+.1f}%/k,end/min {v['end_over_min']:.2f}@{v['step_of_min']}")
                       for b, v in ho.items()))
    print(json.dumps({k: v["converged"] for k, v in report.items()}))
    return report


if __name__ == "__main__":
    main()
