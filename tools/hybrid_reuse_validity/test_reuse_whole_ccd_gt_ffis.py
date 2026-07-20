#!/usr/bin/env python3
"""
Reuse-validity test for hybrid boundary recompute.

Question: if two templates share the same integer local PS1 shift (sx, sy) for a
skycell, can they reuse one patched assignment?

Method
------
1. Whole CCD (all ~1044 skycells on s0020/c3/k2): for the 3 FFIs that have
   pancakes ground-truth remaps, measure local TESS drift at each skycell
   center → PS1 float/int shift vs the reference FFI.
2. Inverse-WCS assignment disagree between those FFIs (sampled PS1 pixels),
   raw and after hypothesizing that int-key equality implies reuse.
3. Ground-truth pancakes regmaps on the 4 micromap skycells × 3 epochs:
   raw disagree and disagree after integer-align roll (roll-factorization).
4. Broader reuse probe (still validated against GT method): among all wcs_ok
   frames, find multi-frame int-(sx,sy) bins for the 4 GT skycells; measure
   inverse-WCS disagree within bins; check inverse vs GT on the 3 GT epochs.

Writes under reuse_validity/outputs/ (worktree only).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    build_ps1_wcs,
    compute_ps1_shift_for_skycell,
    load_tess_wcs,
)

# ── paths (data on /astro; code/outputs in this worktree) ───────────────────
WT = Path(__file__).resolve().parent
OUT = WT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SECTOR, CAM, CCD = 20, 3, 2
MAPPING = Path(
    f"/astro/armin/koji/syndiff/data/skycell_pixel_mapping/"
    f"sector_{SECTOR:04d}/camera_{CAM}/ccd_{CCD}"
)
CSV = MAPPING / f"tess_s{SECTOR:04d}_{CAM}_{CCD}_master_skycells_list.csv"
FFI_DIR = Path(f"/astro/armin/koji/syndiff/data/tess_ffi/s{SECTOR:04d}/cam{CAM}_ccd{CCD}")
FRAMES_CSV = Path(
    "/home/kshukawa/syndiff_pipeline/workspace/events/"
    "s0020_c3_k2_s20_astrometry/syndiff_ffi_frames.csv"
)
GT_DIR = Path(
    "/home/kshukawa/syndiff_pipeline/workspace/experimental/"
    "grid_wcs_correction/s0020_c3_k2/ground_truth"
)

# Reference = first GT epoch (also pipeline reference for this SCC)
REF_FRAME = "tess2020004172923-s0020-3-2-0165-s_ffic.fits.gz"
GT_FRAMES = [
    REF_FRAME,
    "tess2020013135923-s0020-3-2-0165-s_ffic.fits.gz",
    "tess2020020065923-s0020-3-2-0165-s_ffic.fits.gz",
]

# micromap sites with full-skycell GT NPZs
GT_SITES = {
    "center": "skycell.2583.082",
    "cell_boundary_0a": "skycell.2581.085",
    "ffi_edge": "skycell.2625.004",
    "projection_boundary_0a": "skycell.2581.050",
}

T_X = 2136  # TESS FFI x size used in TESS_PIXEL_MAP encoding


def stem(fn: str) -> str:
    s = Path(fn).name
    for suf in (".fits.gz", ".fits"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def load_gt(skycell: str, frame: str, pixel_id: str) -> np.ndarray:
    """GT files are named ``regmap_{pixel_id}_{skycell}_{ffi_stem}.fits.npz``."""
    base = Path(frame).name
    if base.endswith(".gz"):
        base = base[:-3]  # *.fits.gz → *.fits
    candidates = [
        GT_DIR / f"regmap_{pixel_id}_{skycell}_{base}.npz",  # …ffic.fits.npz
        GT_DIR / f"regmap_{pixel_id}_{skycell}_{stem(frame)}.npz",
        GT_DIR / f"regmap_{pixel_id}_{skycell}_{Path(frame).name}.npz",
    ]
    for p in candidates:
        if p.exists():
            z = np.load(p)
            return z["regmap"].astype(np.int64, copy=False)
    raise FileNotFoundError(
        f"No GT regmap for {pixel_id} {skycell} {frame}; tried {[str(c) for c in candidates]}"
    )


def tess_xy_from_tid(tid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = tid >= 0
    tx = np.full(tid.shape, np.nan, dtype=np.float64)
    ty = np.full(tid.shape, np.nan, dtype=np.float64)
    tx[valid] = (tid[valid] % T_X).astype(np.float64)
    ty[valid] = (tid[valid] // T_X).astype(np.float64)
    return tx, ty


def disagree_tid(a: np.ndarray, b: np.ndarray) -> dict:
    both = (a >= 0) & (b >= 0)
    n = int(both.sum())
    if n == 0:
        return {"n_both": 0, "disagree_frac": float("nan")}
    d = (a != b) & both
    return {"n_both": n, "disagree_frac": float(d.sum() / n)}


def roll_assign(arr: np.ndarray, dsx: int, dsy: int) -> np.ndarray:
    """Roll assignment so content moves by +dsx,+dsy in PS1 pixel axes.

    Production rolls *data* by (sy, sx) with np.roll(..., (sy, sx)).
    For ownership maps, aligning epoch B toward epoch A by integer shift
    difference (sx_A - sx_B, sy_A - sy_B) uses the same axis convention.
    """
    return np.roll(arr, (dsy, dsx), axis=(0, 1))


def inverse_xy(ps1_wcs: WCS, frame_wcs: WCS, uu: np.ndarray, vv: np.ndarray):
    ra, dec = ps1_wcs.pixel_to_world_values(uu, vv)
    x, y = frame_wcs.world_to_pixel_values(ra, dec)
    return np.asarray(x, float), np.asarray(y, float)


def inverse_disagree(ps1_wcs, wa, wb, uu, vv) -> dict:
    xa, ya = inverse_xy(ps1_wcs, wa, uu, vv)
    xb, yb = inverse_xy(ps1_wcs, wb, uu, vv)
    same = (np.round(xa) == np.round(xb)) & (np.round(ya) == np.round(yb))
    d = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
    return {
        "disagree_frac": float((~same).mean()),
        "d_tess_mean": float(np.nanmean(d)),
        "d_tess_p99": float(np.nanpercentile(d, 99)),
        "d_tess_max": float(np.nanmax(d)),
    }


def sample_footprint_pixels(foot: np.ndarray, n_max: int = 20000, rng=None):
    yy, xx = np.where(foot)
    if len(xx) == 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(0) if rng is None else rng
    if len(xx) > n_max:
        sel = rng.choice(len(xx), size=n_max, replace=False)
        xx, yy = xx[sel], yy[sel]
    return xx.astype(float), yy.astype(float)


def main() -> None:
    t_all = time.time()
    summary: dict = {"ref_frame": REF_FRAME, "gt_frames": GT_FRAMES}

    df = pd.read_csv(CSV, usecols=lambda c: c in (
        ["NAME", "RA", "DEC"] + [
            "NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
            "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2",
            "RADESYS", "CTYPE1", "CTYPE2",
        ]
    ))
    print(f"skycells in CSV: {len(df)}")

    # load 3 GT FFI WCS once
    wcs_by_frame: dict[str, WCS] = {}
    for fn in GT_FRAMES:
        wcs_by_frame[fn], _ = load_tess_wcs(FFI_DIR / fn)
        print(f"loaded WCS {fn}")
    ref_wcs = wcs_by_frame[REF_FRAME]

    # ── Part A: whole-CCD shift schedule for the 3 GT FFIs ──────────────────
    print("\n=== Part A: whole-CCD local shifts for 3 GT FFIs ===")
    rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        name = row["NAME"]
        ps1_wcs, _ = build_ps1_wcs(row)
        ra, dec = float(row["RA"]), float(row["DEC"])
        x0, y0 = world_ra_dec_to_pixel(ref_wcs, ra, dec)
        for fn, wcs in wcs_by_frame.items():
            xf, yf = world_ra_dec_to_pixel(wcs, ra, dec)
            dx_t, dy_t = float(xf - x0), float(yf - y0)
            sx_f, sy_f = compute_ps1_shift_for_skycell(
                ref_wcs, dx_t, dy_t, ra, dec, ps1_wcs
            )
            sx_i, sy_i = int(round(sx_f)), int(round(sy_f))
            rows.append({
                "skycell": name,
                "frame": fn,
                "dx_t": dx_t, "dy_t": dy_t,
                "sx_f": sx_f, "sy_f": sy_f,
                "sx_i": sx_i, "sy_i": sy_i,
                "frac_x": sx_f - sx_i, "frac_y": sy_f - sy_i,
            })
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)} skycells…")
    shifts = pd.DataFrame(rows)
    shifts_path = OUT / "whole_ccd_gt_ffi_shifts.csv"
    shifts.to_csv(shifts_path, index=False)
    print(f"wrote {shifts_path} in {time.time()-t0:.1f}s  rows={len(shifts)}")

    # pivot: per skycell, do any GT FFI pair share int key?
    pivot = shifts.pivot(index="skycell", columns="frame", values=["sx_i", "sy_i", "sx_f", "sy_f"])
    # flatten for clarity
    share_rows = []
    pairs = [(GT_FRAMES[i], GT_FRAMES[j]) for i in range(3) for j in range(i + 1, 3)]
    for sk in shifts["skycell"].unique():
        sub = shifts[shifts.skycell == sk].set_index("frame")
        for fa, fb in pairs:
            same = (int(sub.loc[fa, "sx_i"]) == int(sub.loc[fb, "sx_i"]) and
                    int(sub.loc[fa, "sy_i"]) == int(sub.loc[fb, "sy_i"]))
            dsi = int(sub.loc[fa, "sx_i"] - sub.loc[fb, "sx_i"])
            dsj = int(sub.loc[fa, "sy_i"] - sub.loc[fb, "sy_i"])
            frac_dist = float(np.hypot(
                sub.loc[fa, "frac_x"] - sub.loc[fb, "frac_x"],
                sub.loc[fa, "frac_y"] - sub.loc[fb, "frac_y"],
            ))
            float_dist = float(np.hypot(
                sub.loc[fa, "sx_f"] - sub.loc[fb, "sx_f"],
                sub.loc[fa, "sy_f"] - sub.loc[fb, "sy_f"],
            ))
            share_rows.append({
                "skycell": sk, "frame_a": fa, "frame_b": fb,
                "same_int_key": same,
                "dsi": dsi, "dsj": dsj,
                "frac_dist": frac_dist, "float_dist": float_dist,
                "sx_i_a": int(sub.loc[fa, "sx_i"]), "sy_i_a": int(sub.loc[fa, "sy_i"]),
                "sx_i_b": int(sub.loc[fb, "sx_i"]), "sy_i_b": int(sub.loc[fb, "sy_i"]),
            })
    share = pd.DataFrame(share_rows)
    share.to_csv(OUT / "whole_ccd_gt_ffi_pair_keys.csv", index=False)
    n_same = int(share["same_int_key"].sum())
    print(f"GT-FFI pairs sharing int key: {n_same} / {len(share)} "
          f"({100*n_same/len(share):.2f}%)")
    print("Δint Manhattan |dsi|+|dsj| distribution:")
    manh = share["dsi"].abs() + share["dsj"].abs()
    print(manh.value_counts().sort_index().head(20))
    summary["gt_ffi_pairs_same_int_key"] = n_same
    summary["gt_ffi_pairs_total"] = len(share)
    summary["gt_ffi_pair_same_int_frac"] = n_same / len(share)

    # ── Part B: GT pancakes raw + roll-aligned disagree ─────────────────────
    print("\n=== Part B: pancakes GT disagree (raw + int-align roll) ===")
    gt_rows = []
    for pixel_id, skycell in GT_SITES.items():
        sub = shifts[shifts.skycell == skycell].set_index("frame")
        gts = {fn: load_gt(skycell, fn, pixel_id) for fn in GT_FRAMES}
        print(f"  {pixel_id} {skycell} shapes={[gts[f].shape for f in GT_FRAMES]}")
        for fa, fb in pairs:
            a, b = gts[fa], gts[fb]
            raw = disagree_tid(a, b)
            dsi = int(sub.loc[fa, "sx_i"] - sub.loc[fb, "sx_i"])
            dsj = int(sub.loc[fa, "sy_i"] - sub.loc[fb, "sy_i"])
            # roll B toward A's integer shift
            b_roll = roll_assign(b, dsi, dsj)
            aligned = disagree_tid(a, b_roll)
            # also try opposite sign if first is worse (diagnose convention)
            b_roll_opp = roll_assign(b, -dsi, -dsj)
            aligned_opp = disagree_tid(a, b_roll_opp)
            if aligned_opp["disagree_frac"] < aligned["disagree_frac"]:
                aligned = aligned_opp
                roll_sign = -1
            else:
                roll_sign = +1
            frac_dist = float(np.hypot(
                sub.loc[fa, "frac_x"] - sub.loc[fb, "frac_x"],
                sub.loc[fa, "frac_y"] - sub.loc[fb, "frac_y"],
            ))
            gt_rows.append({
                "pixel_id": pixel_id, "skycell": skycell,
                "frame_a": fa, "frame_b": fb,
                "sx_i_a": int(sub.loc[fa, "sx_i"]), "sy_i_a": int(sub.loc[fa, "sy_i"]),
                "sx_i_b": int(sub.loc[fb, "sx_i"]), "sy_i_b": int(sub.loc[fb, "sy_i"]),
                "dsi": dsi, "dsj": dsj, "roll_sign": roll_sign,
                "frac_dist": frac_dist,
                "raw_disagree": raw["disagree_frac"],
                "aligned_disagree": aligned["disagree_frac"],
                "n_both_raw": raw["n_both"],
                "n_both_aligned": aligned["n_both"],
                "same_int_key": (dsi == 0 and dsj == 0),
            })
            print(
                f"    {stem(fa)[-20:]} vs {stem(fb)[-20:]}: "
                f"raw={raw['disagree_frac']:.4f} aligned={aligned['disagree_frac']:.4f} "
                f"Δint=({dsi},{dsj}) frac_dist={frac_dist:.3f} roll_sign={roll_sign}"
            )
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(OUT / "gt_pancakes_pair_disagree.csv", index=False)
    summary["gt_aligned_disagree_median"] = float(gt_df["aligned_disagree"].median())
    summary["gt_aligned_disagree_max"] = float(gt_df["aligned_disagree"].max())
    summary["gt_raw_disagree_median"] = float(gt_df["raw_disagree"].median())

    # ── Part C: whole-CCD inverse-WCS disagree for 3 GT FFIs ────────────────
    print("\n=== Part C: whole-CCD inverse-WCS pair disagree (sampled) ===")
    # Build a quick footprint proxy: PS1 pixels near skycell center (circle)
    # so we don't need to load 1044 frozen regmaps (slow I/O).
    inv_rows = []
    t0 = time.time()
    rng = np.random.default_rng(1)
    N_SAMPLE = 8000
    for i, row in df.iterrows():
        name = row["NAME"]
        ps1_wcs, ps1_shape = build_ps1_wcs(row)
        nx, ny = int(ps1_shape[0]), int(ps1_shape[1])
        # sample a grid around center (PS1 CRPIX region ≈ skycell core)
        # Use uniform sample in central 60% of array
        uu = rng.uniform(0.2 * nx, 0.8 * nx, size=N_SAMPLE)
        vv = rng.uniform(0.2 * ny, 0.8 * ny, size=N_SAMPLE)
        sub = shifts[shifts.skycell == name].set_index("frame")
        for fa, fb in pairs:
            st = inverse_disagree(
                ps1_wcs, wcs_by_frame[fa], wcs_by_frame[fb], uu, vv
            )
            dsi = int(sub.loc[fa, "sx_i"] - sub.loc[fb, "sx_i"])
            dsj = int(sub.loc[fa, "sy_i"] - sub.loc[fb, "sy_i"])
            frac_dist = float(np.hypot(
                sub.loc[fa, "frac_x"] - sub.loc[fb, "frac_x"],
                sub.loc[fa, "frac_y"] - sub.loc[fb, "frac_y"],
            ))
            inv_rows.append({
                "skycell": name, "frame_a": fa, "frame_b": fb,
                "dsi": dsi, "dsj": dsj,
                "same_int_key": (dsi == 0 and dsj == 0),
                "frac_dist": frac_dist,
                "manh_int": abs(dsi) + abs(dsj),
                **st,
            })
        if (i + 1) % 100 == 0:
            print(f"  inverse {i+1}/{len(df)}…")
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(OUT / "whole_ccd_inverse_pair_disagree.csv", index=False)
    print(f"Part C done in {time.time()-t0:.1f}s")

    print("\nInverse disagree by |Δint| Manhattan:")
    for k, g in inv.groupby("manh_int"):
        print(
            f"  manh={k}: n={len(g)} disagree median={g.disagree_frac.median():.6f} "
            f"mean={g.disagree_frac.mean():.6f} d_tess_mean med={g.d_tess_mean.median():.5f}"
        )
    same = inv[inv.same_int_key]
    if len(same):
        print(
            f"SAME int key: n={len(same)} disagree median={same.disagree_frac.median():.6f} "
            f"max={same.disagree_frac.max():.6f}"
        )
    else:
        print("SAME int key: none among the 3 GT FFIs on this CCD "
              "(expected if epochs are far apart)")
    summary["inverse_by_manh"] = {
        str(int(k)): {
            "n": int(len(g)),
            "disagree_median": float(g.disagree_frac.median()),
            "d_tess_mean_median": float(g.d_tess_mean.median()),
        }
        for k, g in inv.groupby("manh_int")
    }

    # ── Part D: validate inverse vs GT on micromap sites ────────────────────
    print("\n=== Part D: inverse-WCS vs pancakes GT on micromap sites ===")
    val_rows = []
    for pixel_id, skycell in GT_SITES.items():
        row = df.loc[df.NAME == skycell].iloc[0]
        ps1_wcs, _ = build_ps1_wcs(row)
        for fn in GT_FRAMES:
            gt = load_gt(skycell, fn, pixel_id)
            foot = gt >= 0
            uu, vv = sample_footprint_pixels(foot, n_max=30000)
            if len(uu) == 0:
                continue
            xa, ya = inverse_xy(ps1_wcs, wcs_by_frame[fn], uu, vv)
            # nearest-pixel tid
            tid_inv = np.round(xa).astype(np.int64) + T_X * np.round(ya).astype(np.int64)
            # sample gt at integer pixel coords
            ui = np.clip(np.round(uu).astype(int), 0, gt.shape[1] - 1)
            vi = np.clip(np.round(vv).astype(int), 0, gt.shape[0] - 1)
            tid_gt = gt[vi, ui]
            both = tid_gt >= 0
            agree = (tid_inv == tid_gt) & both
            # continuous residual of inverse vs GT tess xy
            gtx, gty = tess_xy_from_tid(tid_gt)
            d = np.sqrt((xa - gtx) ** 2 + (ya - gty) ** 2)
            d = d[both]
            val_rows.append({
                "pixel_id": pixel_id, "skycell": skycell, "frame": fn,
                "n": int(both.sum()),
                "agree_frac": float(agree.sum() / max(both.sum(), 1)),
                "d_tess_mean": float(np.nanmean(d)) if d.size else float("nan"),
                "d_tess_p99": float(np.nanpercentile(d, 99)) if d.size else float("nan"),
            })
            print(
                f"  {pixel_id} {stem(fn)[-20:]}: agree={val_rows[-1]['agree_frac']:.4f} "
                f"d_tess_mean={val_rows[-1]['d_tess_mean']:.4f}"
            )
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(OUT / "inverse_vs_gt_validation.csv", index=False)
    summary["inverse_vs_gt_agree_median"] = float(val_df["agree_frac"].median())
    summary["inverse_vs_gt_agree_min"] = float(val_df["agree_frac"].min())

    # ── Part E: broader same-int-key reuse using ALL frames (4 GT skycells) ─
    print("\n=== Part E: all-frame same-int-key reuse on 4 GT skycells ===")
    man = pd.read_csv(FRAMES_CSV)
    man = man[man["wcs_ok"] == True].copy()  # noqa: E712
    # stride to keep runtime sane but denser than before
    stride = 2
    idx = np.arange(0, len(man), stride)
    names = set(man.iloc[idx]["filename"]) | set(GT_FRAMES)
    sub_man = man[man["filename"].isin(names)].reset_index(drop=True)
    print(f"loading {len(sub_man)} frame WCS…")
    all_wcs = {}
    t0 = time.time()
    for _, r in sub_man.iterrows():
        path = Path(r["path"]) if isinstance(r.get("path"), str) and str(r["path"]).startswith("/") else FFI_DIR / r["filename"]
        if not path.exists():
            path = FFI_DIR / r["filename"]
        try:
            all_wcs[r["filename"]], _ = load_tess_wcs(path)
        except Exception as e:
            print("  skip", r["filename"], e)
    print(f"  loaded {len(all_wcs)} in {time.time()-t0:.1f}s")

    reuse_rows = []
    for pixel_id, skycell in GT_SITES.items():
        row = df.loc[df.NAME == skycell].iloc[0]
        ps1_wcs, _ = build_ps1_wcs(row)
        ra, dec = float(row["RA"]), float(row["DEC"])
        x0, y0 = world_ra_dec_to_pixel(ref_wcs, ra, dec)
        gt0 = load_gt(skycell, REF_FRAME, pixel_id)
        foot = gt0 >= 0
        uu, vv = sample_footprint_pixels(foot, n_max=15000)

        recs = []
        for fn, wcs in all_wcs.items():
            xf, yf = world_ra_dec_to_pixel(wcs, ra, dec)
            dx_t, dy_t = float(xf - x0), float(yf - y0)
            sx_f, sy_f = compute_ps1_shift_for_skycell(
                ref_wcs, dx_t, dy_t, ra, dec, ps1_wcs
            )
            sx_i, sy_i = int(round(sx_f)), int(round(sy_f))
            recs.append({
                "filename": fn, "sx_f": sx_f, "sy_f": sy_f,
                "sx_i": sx_i, "sy_i": sy_i,
                "frac_x": sx_f - sx_i, "frac_y": sy_f - sy_i,
            })
        rec = pd.DataFrame(recs)
        rec.to_csv(OUT / f"allframe_shifts_{skycell}.csv", index=False)
        print(
            f"  {skycell}: frames={len(rec)} unique_int={rec.groupby(['sx_i','sy_i']).ngroups} "
            f"sx=[{rec.sx_f.min():.2f},{rec.sx_f.max():.2f}] "
            f"sy=[{rec.sy_f.min():.2f},{rec.sy_f.max():.2f}]"
        )

        bins = rec.groupby(["sx_i", "sy_i"]).filter(lambda g: len(g) >= 2)
        n_bins = bins.groupby(["sx_i", "sy_i"]).ngroups if len(bins) else 0
        print(f"    multi-frame int bins: {n_bins}")

        for (sxi, syi), g in bins.groupby(["sx_i", "sy_i"]):
            g = g.reset_index(drop=True)
            # worst frac-distance pair
            best = None
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    dist = float(np.hypot(
                        g.loc[i, "frac_x"] - g.loc[j, "frac_x"],
                        g.loc[i, "frac_y"] - g.loc[j, "frac_y"],
                    ))
                    if best is None or dist > best[0]:
                        best = (dist, i, j)
            dist, i, j = best
            fa, fb = g.loc[i, "filename"], g.loc[j, "filename"]
            st = inverse_disagree(ps1_wcs, all_wcs[fa], all_wcs[fb], uu, vv)
            # also median of random pairs
            rng2 = np.random.default_rng(2)
            meds = []
            for _ in range(min(12, len(g) * (len(g) - 1) // 2)):
                ii, jj = rng2.choice(len(g), size=2, replace=False)
                meds.append(inverse_disagree(
                    ps1_wcs, all_wcs[g.loc[ii, "filename"]], all_wcs[g.loc[jj, "filename"]],
                    uu, vv,
                )["disagree_frac"])
            reuse_rows.append({
                "pixel_id": pixel_id, "skycell": skycell,
                "sx_i": int(sxi), "sy_i": int(syi),
                "n_in_bin": len(g),
                "worst_frac_dist": dist,
                "worst_disagree": st["disagree_frac"],
                "worst_d_tess_mean": st["d_tess_mean"],
                "typical_disagree_median": float(np.median(meds)),
                "frac_span_x": float(g.frac_x.max() - g.frac_x.min()),
                "frac_span_y": float(g.frac_y.max() - g.frac_y.min()),
            })

        # finer q=0.25 within this skycell
        q = 0.25
        rec["qx"] = np.round(rec["sx_f"] / q) * q
        rec["qy"] = np.round(rec["sy_f"] / q) * q
        fine = rec.groupby(["qx", "qy"]).filter(lambda g: len(g) >= 2)
        fine_worst = []
        for (qx, qy), g in fine.groupby(["qx", "qy"]):
            g = g.reset_index(drop=True)
            best = None
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    dist = float(np.hypot(
                        g.loc[i, "sx_f"] - g.loc[j, "sx_f"],
                        g.loc[i, "sy_f"] - g.loc[j, "sy_f"],
                    ))
                    if best is None or dist > best[0]:
                        best = (dist, i, j)
            dist, i, j = best
            st = inverse_disagree(
                ps1_wcs, all_wcs[g.loc[i, "filename"]], all_wcs[g.loc[j, "filename"]],
                uu, vv,
            )
            fine_worst.append(st["disagree_frac"])
        if fine_worst:
            print(
                f"    q=0.25 multi bins={len(fine_worst)} "
                f"worst-pair disagree median={np.median(fine_worst):.6f} "
                f"max={np.max(fine_worst):.6f}"
            )

    reuse_df = pd.DataFrame(reuse_rows)
    reuse_df.to_csv(OUT / "same_int_key_reuse_gt_skycells.csv", index=False)
    if len(reuse_df):
        print("\nSame-int-key worst-pair disagree summary:")
        print(
            f"  median={reuse_df.worst_disagree.median():.6f} "
            f"p90={reuse_df.worst_disagree.quantile(0.9):.6f} "
            f"max={reuse_df.worst_disagree.max():.6f}"
        )
        print(
            f"  typical_median median={reuse_df.typical_disagree_median.median():.6f} "
            f"max={reuse_df.typical_disagree_median.max():.6f}"
        )
        summary["same_int_worst_disagree_median"] = float(reuse_df.worst_disagree.median())
        summary["same_int_worst_disagree_max"] = float(reuse_df.worst_disagree.max())
        summary["same_int_typical_disagree_median"] = float(
            reuse_df.typical_disagree_median.median()
        )
        summary["n_same_int_bins_tested"] = len(reuse_df)

    # Correlate worst disagree with frac_span inside bin
    if len(reuse_df):
        corr = float(reuse_df["worst_disagree"].corr(reuse_df["worst_frac_dist"]))
        summary["corr_worst_disagree_vs_frac_dist"] = corr
        print(f"  corr(worst_disagree, frac_pair_dist)={corr:.3f}")

    summary["elapsed_s"] = time.time() - t_all
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary → {OUT / 'summary.json'}")
    print(f"TOTAL elapsed {summary['elapsed_s']:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
