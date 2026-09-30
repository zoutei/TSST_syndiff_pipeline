# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""A3 (part 2): merge per-fold scores into one out-of-fold (OOF) and one in-fold value per star, and summarise.

    python -m syndiff_pipeline.forward_model.crossfit.merge_oof --scores fold0.npz fold1.npz --out merged.npz

Each input is a score_oof output of the fit that held fold k out (``held_fold`` = k). For star i with fold f_i:
  OOF value     = from the fit with held_fold == f_i (never trained on i)
  in-fold value = from a fit with held_fold != f_i (trained on i); with 2 folds exactly one.

Summary statistics are plain medians over stars (no S/N weights). Uncertainty of a median of n values with robust
scatter s: 1.2533 s / sqrt(n). Paired comparisons use per-star log ratios.

  R_mem(T) = median_i chi2_OOF,i / chi2_in-fold,i       (memorisation ratio, per Tmag bin)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TBINS = ((8.0, 9.0), (9.0, 10.0), (10.0, 11.0), (11.0, 12.0), (12.0, 13.0))


def merge(paths):
    S = [dict(np.load(p)) for p in paths]
    sid = S[0]["source_id"]
    for s in S[1:]:
        assert (s["source_id"] == sid).all()
    fold = S[0]["fold"]
    N = len(sid)
    held = [int(s["held_fold"]) for s in S]
    assert sorted(held) == sorted(set(held)), "one score per held fold"
    out = {k: S[0][k] for k in ("source_id", "fold", "T", "x", "y", "colour", "role0", "trainee", "pix_disjoint")}
    for key in ("c2", "flux", "ok"):
        o = np.full(N, np.nan) if key != "ok" else np.zeros(N, bool)
        i_ = np.full(N, np.nan) if key != "ok" else np.zeros(N, bool)
        for s, h in zip(S, held):
            o[s["oof"]] = s[key][s["oof"]]          # trainees of fold h, scored by the fit that held h out
            i_[s["infold"]] = s[key][s["infold"]]   # trainees of other folds (2 folds: each star once)
        out[f"{key}_oof"] = o
        out[f"{key}_in"] = i_
    return out


def med_err(v):
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan, np.nan, int(v.size)
    m = np.median(v)
    s = 1.4826 * np.median(np.abs(v - m))
    return float(m), float(1.2533 * s / np.sqrt(v.size)), int(v.size)


def summary(m, sel=None):
    base = m["trainee"] & m["ok_oof"] & m["ok_in"]
    if sel is not None:
        base &= sel
    out = {}
    for lo, hi in TBINS:
        s = base & (m["T"] >= lo) & (m["T"] < hi)
        oof, oofe, n = med_err(m["c2_oof"][s])
        inn, inne, _ = med_err(m["c2_in"][s])
        lr, lre, _ = med_err(np.log(m["c2_oof"][s] / m["c2_in"][s]))
        out[f"{lo:g}-{hi:g}"] = dict(n=n, c2_oof=oof, c2_oof_err=oofe, c2_in=inn, c2_in_err=inne,
                                    R_mem=float(np.exp(lr)), R_mem_err=float(np.exp(lr) * lre))
    return out


def paired(a, b, sel=None):
    """Per-bin median per-star ratios a/b of OOF chi2 and OOF flux (two merged tables, same stars)."""
    base = a["trainee"] & a["ok_oof"] & b["ok_oof"]
    if sel is not None:
        base &= sel
    out = {}
    for lo, hi in TBINS + ((8.0, 13.0),):
        s = base & (a["T"] >= lo) & (a["T"] < hi)
        lc, lce, n = med_err(np.log(a["c2_oof"][s] / b["c2_oof"][s]))
        fr = a["flux_oof"][s] / b["flux_oof"][s] - 1
        fm, fe, _ = med_err(fr)
        out[f"{lo:g}-{hi:g}"] = dict(n=n, chi2_ratio=float(np.exp(lc)), chi2_ratio_err=float(np.exp(lc) * lce),
                                    flux_ratio_minus1_permil=1e3 * fm, flux_err_permil=1e3 * fe,
                                    flux_scatter_permil=float(1e3 * 1.4826 * np.nanmedian(np.abs(fr - fm))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = merge(a.scores)
    np.savez_compressed(a.out, **m)
    sm = dict(all=summary(m), pix_disjoint=summary(m, m["pix_disjoint"]), inputs=a.scores)
    Path(a.out).with_suffix(".json").write_text(json.dumps(sm, indent=1))
    print(json.dumps(sm["all"], indent=0))


if __name__ == "__main__":
    main()
