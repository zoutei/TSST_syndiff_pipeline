# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""A4 (part 1): Gaia-XP synthetic TESS magnitudes for a scene's stars.

Run with the isolated XP venv (gaiaxpy 2.1.4; the syndiff env has no gaiaxpy):

    /astro/armin/koji/syndiff/dev_runs/xp_colour_20260925/envs/xp/bin/python -m syndiff_pipeline.forward_model.crossfit.xp_tess \
        --scene-dir <scene> --out <xp_tess.csv> [--bulk-dir DIR ...] [--download-dir DIR]

The synthetic magnitude is photon-weighted through the TESS response R(lambda):

    T_XP = -2.5 log10( sum_lambda f_lambda * lambda * R(lambda) * dlambda ) + C

f_lambda: GaiaXPy ``calibrate`` flux (W m^-2 nm^-1) on 336..1020 nm, 2 nm steps; lambda in nm; R: the response curve
(``--response``; default TESS v2.0, Vanderspek 2020-07-30, its sha256 is recorded). C is an arbitrary constant chosen
so that median(T_XP - tess_mag) = 0 for 10 <= tess_mag < 12 (cosmetic: every use fits its own zero point).
XP stops at 1020 nm; the response beyond that is lost (``resp_frac_gt1020`` records how much of R's integral that is,
star-independent). The star-dependent part of the truncation is a colour effect: always fit a colour term.
Stars with BP-RP > 2 (M dwarfs/giants) are flagged: the SVO vs v2 response choice alone moves their synthetic flux
by +-5% (bandpass_rizy_20260929); report them separately.

Output columns: source_id, T_XP, lam_tess_mean, bp_rp_scene, tess_mag, m_star (bp_rp > 2), has_xp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RESP = "/astro/armin/koji/syndiff/dev_runs/xp_colour_20260925/tess-response-function-v2.0.csv"
INDEX = "/astro/armin/koji/syndiff/dev_runs/chroma_v2_20260928/colour/scripts/xp_bulk_index.json"
CDN = "https://cdn.gea.esac.esa.int/Gaia/gdr3/Spectroscopy/xp_continuous_mean_spectrum/"
BULK_DIRS = ["/astro/armin/koji/syndiff/dev_runs/xp_colour_20260925/xp_bulk",
             "/astro/armin/koji/syndiff/dev_runs/xp_dimension_20260925/xp_bulk_F2"]
LAM = np.arange(336.0, 1021.0, 2.0)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731


def needed_files(sids):
    idx = json.load(open(INDEX))
    hp = np.unique(np.asarray(sids, np.int64) // (34359738368 * 256))  # HEALPix level 8 = source_id // (2^35 * 4^4)
    return sorted({f for lo, hi, f, _ in idx for h in hp if lo <= h <= hi})


def locate(files, bulk_dirs, download_dir):
    out = []
    for f in files:
        hit = next((Path(d) / f for d in bulk_dirs if (Path(d) / f).exists()), None)
        if hit is None:
            dd = Path(download_dir)
            dd.mkdir(parents=True, exist_ok=True)
            hit = dd / f
            if not hit.exists():
                log("downloading", f)
                urllib.request.urlretrieve(CDN + f, hit.with_suffix(".part"))
                hit.with_suffix(".part").rename(hit)
        out.append(hit)
    return out


def filter_rows(paths, sids):
    keep = set(int(s) for s in sids)
    parts = []
    for p in paths:
        for ch in pd.read_csv(p, comment="#", chunksize=200_000):
            ch = ch[ch["source_id"].isin(keep)]
            if len(ch):
                parts.append(ch)
        log("filtered", Path(p).name, sum(len(x) for x in parts))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def parse_arrays(df):
    """Bulk files write arrays '[a,b,...]'; GaiaXPy wants ndarrays (and a 0-based index)."""
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.startswith("[").any():
            df[c] = df[c].apply(lambda s: np.array(json.loads(s.replace("null", "NaN").replace("nan", "NaN")), float)
                                if isinstance(s, str) else s)
    return df.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bulk-dir", action="append", default=None)
    ap.add_argument("--download-dir", default="/astro/armin/koji/syndiff/dev_runs/heldout_harness_20260929/xp_bulk")
    ap.add_argument("--response", default=RESP)
    ap.add_argument("--prefiltered", default=None, help="CSV already filtered to (a superset of) the scene's ids")
    ap.add_argument("--batch", type=int, default=2000)
    a = ap.parse_args(argv)

    from gaiaxpy import calibrate

    sc = Path(a.scene_dir)
    z = np.load(sc / "scene_bundle.npz")
    meta = json.loads((sc / "scene_meta.json").read_text())
    b = np.load(meta["source_bundle"])
    sids = z["source_id"].astype(np.int64)
    bp_rp = np.asarray(b["bp_rp"], float)[z["star_bundle_index"]]
    if a.prefiltered:
        df = filter_rows([a.prefiltered], sids)
        files = [a.prefiltered]
    else:
        files = [str(p) for p in locate(needed_files(sids), a.bulk_dir or BULK_DIRS, a.download_dir)]
        df = filter_rows(files, sids)
    df = parse_arrays(df)
    log("stars with XP", len(df), "of", len(sids))

    tr = np.genfromtxt(a.response, delimiter=",", comments="#")
    tr = tr[np.all(np.isfinite(tr), 1)]
    R = np.interp(LAM, tr[:, 0], tr[:, 1], left=0, right=0)
    tot = np.trapezoid(tr[:, 1], tr[:, 0])
    beyond = np.trapezoid(np.where(tr[:, 0] > 1020, tr[:, 1], 0), tr[:, 0]) / tot

    rows = []
    for i in range(0, len(df), a.batch):
        sp, _ = calibrate(df.iloc[i:i + a.batch].reset_index(drop=True), sampling=LAM, save_file=False)
        F = np.vstack(sp["flux"].values)
        nph = F * LAM * R
        s = nph.sum(1) * 2.0
        rows.append(pd.DataFrame(dict(source_id=sp["source_id"].values.astype(np.int64),
                                      S=s, lam_tess_mean=(nph * LAM).sum(1) / nph.sum(1))))
        log("calibrated", i + a.batch)
    xp = pd.concat(rows, ignore_index=True)
    t = pd.DataFrame(dict(source_id=sids, tess_mag=z["tess_mag"].astype(float), bp_rp_scene=bp_rp))
    t = t.merge(xp, on="source_id", how="left")
    t["has_xp"] = np.isfinite(t["S"]) & (t["S"] > 0)
    t["T_XP"] = np.where(t["has_xp"], -2.5 * np.log10(t["S"].clip(lower=1e-300)), np.nan)
    ref = t["has_xp"] & (t["tess_mag"] >= 10) & (t["tess_mag"] < 12)
    C = float(np.median(t.loc[ref, "tess_mag"] - t.loc[ref, "T_XP"]))
    t["T_XP"] += C
    t["m_star"] = t["bp_rp_scene"] > 2.0
    t.drop(columns="S").to_csv(a.out, index=False, float_format="%.6f")
    info = dict(scene=str(sc), n=len(t), n_xp=int(t["has_xp"].sum()), response=a.response,
                response_sha256=hashlib.sha256(Path(a.response).read_bytes()).hexdigest(),
                resp_frac_gt1020=float(beyond), sampling_nm=[float(LAM[0]), float(LAM[-1]), 2.0], C=C,
                bulk_files=files, n_m_star=int(t["m_star"].sum()),
                median_T_XP_minus_tmag_by_bprp={f"{lo}-{hi}": float(np.nanmedian((t["T_XP"] - t["tess_mag"])[
                    t["has_xp"] & (t["bp_rp_scene"] >= lo) & (t["bp_rp_scene"] < hi)]))
                    for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 5.0))})
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=1))
    log(json.dumps({k: info[k] for k in ("n", "n_xp", "resp_frac_gt1020", "n_m_star")}))


if __name__ == "__main__":
    main()
