"""Standard per-run residual diagnostic: stacked residual/flux by colour x magnitude in a 3x3 CCD grid.

Input is the run's ``raster_res_model.npz`` (``res``/``model`` = per-star residual/flux and model/flux
rasterised on a 33x33, 0.25 px, +-4 px grid; ``colour`` = BP-RP, ``x``/``y`` = CCD position, ``flux``),
as written by ``dev_runs/magwidth_20260924/raster_model.py`` for every scene_fit run.

Figure layout: the outer 3x3 is the CCD (y up, cell edges 0/683/1365/2048). Inside each cell, rows are
Tmag bins (bright on top) and columns are BP-RP bins (blue -> red). Each stamp is a 5-sigma-clipped mean
of residual/flux in 1e-4 of the star's flux per px^2 (same units/stack as colour_residual.png), and its
title gives n and the stamp's width terms cxx/cyy (1e-3 px^2; negative = data sharper than the model),
fitted with the cell's own median model image.

Two figures:
  resgrid_<label>.png      absolute residual
  resgrid_<label>_rel.png  minus the cell's all-star stack (isolates the colour / magnitude dependence)
plus resgrid_<label>.json with n, cxx, cyy per (cell, mag, colour).

Usage: python -m syndiff_pipeline.forward_model.diagnostics.residual_grid RUN_DIR [LABEL] [--out-dir DIR]
(Moved from dev/forward_epsf_wcs/diagnostics/residual_grid.py on 2026-10-07; numerics unchanged.)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

MAG_BINS = [(8.0, 10.0), (10.0, 12.0), (12.0, 13.0)]
COL_BINS = [(-1.0, 0.75), (0.75, 1.25), (1.25, 4.0)]
COL_NAMES = ["blue <0.75", "mid 0.75-1.25", "red >1.25"]
NCELL = 3
MIN_N = 5

C1 = np.linspace(-4, 4, 33)
H = C1[1] - C1[0]
XX, YY = np.meshgrid(C1, C1)
RR = np.hypot(XX, YY)
W = RR < 3.5


def tmag_from_flux(fl):
    return -2.5 * np.log10(np.maximum(fl, 1e-9) / 15000.0) + 10.0  # 15000 e-/s at T=10


def rstack(A):
    med = np.nanmedian(A, 0)
    mad = 1.4826 * np.nanmedian(np.abs(A - med), 0) + 1e-12
    return np.nanmean(np.where(np.abs(A - med) < 5 * mad, A, np.nan), 0)


def basis(P):
    gy, gx = np.gradient(P, H)
    return np.stack([P, gx, gy, np.gradient(gx, H, axis=1), np.gradient(gy, H, axis=0),
                     np.gradient(gx, H, axis=0), np.ones_like(P)], -1)


def width_terms(D, P):
    B = basis(np.nan_to_num(P))[W]
    d = D[W]
    k = np.isfinite(d)
    if k.sum() < 50:
        return np.nan, np.nan
    co, *_ = np.linalg.lstsq(B[k], d[k], rcond=None)
    return 1e3 * co[3], 1e3 * co[4]


def load(run: Path):
    e = np.load(run / "raster_res_model.npz")
    RS, MD, fl, col, x, y = e["res"], e["model"], e["flux"], e["colour"], e["x"], e["y"]
    T = tmag_from_flux(fl)
    # as width_analysis.py: drop stars whose fitted flux is so low that model/f blows up
    ok = np.isfinite(T) & (T < 13.5) & np.isfinite(col) & (np.nanmax(np.abs(MD), axis=(1, 2)) < 5)
    return RS[ok], MD[ok], T[ok], col[ok], x[ok], y[ok]


def compute(run: Path):
    RS, MD, T, col, x, y = load(run)
    edges = np.linspace(0, 2048, NCELL + 1)
    cells = {}
    for iy in range(NCELL):
        for ix in range(NCELL):
            inc = (x >= edges[ix]) & (x < edges[ix + 1]) & (y >= edges[iy]) & (y < edges[iy + 1])
            inc &= (T >= MAG_BINS[0][0]) & (T < MAG_BINS[-1][1])
            P = np.nanmedian(MD[inc], 0)
            allstack = rstack(RS[inc])
            stamps = {}
            for im, (a, b) in enumerate(MAG_BINS):
                for ic, (c0, c1) in enumerate(COL_BINS):
                    s = inc & (T >= a) & (T < b) & (col >= c0) & (col < c1)
                    n = int(s.sum())
                    img = rstack(RS[s]) if n >= MIN_N else np.full(RR.shape, np.nan)
                    cxx, cyy = width_terms(img, P) if n >= MIN_N else (np.nan, np.nan)
                    rel = img - allstack
                    rxx, ryy = width_terms(rel, P) if n >= MIN_N else (np.nan, np.nan)
                    stamps[im, ic] = dict(img=img, rel=rel, n=n, cxx=cxx, cyy=cyy, rxx=rxx, ryy=ryy)
            cells[iy, ix] = dict(stamps=stamps, n=int(inc.sum()), edges=(edges[ix], edges[ix + 1], edges[iy], edges[iy + 1]))
    return cells


def plot(cells, path: Path, title: str, key: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nm, nc = len(MAG_BINS), len(COL_BINS)
    vals = np.concatenate([1e4 * np.abs(st[key][W]) for c in cells.values() for st in c["stamps"].values()
                           if st["n"] >= MIN_N])
    vals = vals[np.isfinite(vals)]
    lim = float(np.percentile(vals, 99.5)) if vals.size else 1.0
    fig = plt.figure(figsize=(3 * nc * NCELL * 0.62 + 1.5, 3 * nm * NCELL * 0.66 + 1.0))
    outer = fig.add_gridspec(NCELL, NCELL, wspace=0.10, hspace=0.22, left=0.03, right=0.92, top=0.885, bottom=0.03)
    im = None
    for (iy, ix), c in cells.items():
        inner = outer[NCELL - 1 - iy, ix].subgridspec(nm, nc, wspace=0.04, hspace=0.30)
        x0, x1, y0, y1 = c["edges"]
        for (jm, jc), st in c["stamps"].items():
            ax = fig.add_subplot(inner[jm, jc])
            ax.set_xticks([]); ax.set_yticks([])
            im = ax.imshow(1e4 * np.where(RR < 3.6, st[key], np.nan), origin="lower", extent=[-4, 4, -4, 4],
                           cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            wx, wy = (st["cxx"], st["cyy"]) if key == "img" else (st["rxx"], st["ryy"])
            ax.set_title(f"n={st['n']}  {wx:+.1f}/{wy:+.1f}" if st["n"] >= MIN_N else f"n={st['n']}", fontsize=6.5, pad=2)
            if jc == 0:
                a, b = MAG_BINS[jm]
                ax.set_ylabel(f"T {a:g}-{b:g}", fontsize=7)
            if jm == 0 and jc == 1:
                ax.text(0.5, 1.42, f"x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}  (n={c['n']})", transform=ax.transAxes,
                        ha="center", fontsize=8.5, fontweight="bold")
            if jm == nm - 1:
                ax.set_xlabel(COL_NAMES[jc], fontsize=7)
    cax = fig.add_axes([0.935, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="residual / flux, 1e-4 per px²")
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("label", nargs="?")
    p.add_argument("--out-dir")
    a = p.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)  # empty (n<5) bins
    run = Path(a.run_dir)
    label = a.label or run.name
    out = Path(a.out_dir) if a.out_dir else run
    out.mkdir(parents=True, exist_ok=True)
    cells = compute(run)
    mags = " / ".join(f"{a:g}-{b:g}" for a, b in MAG_BINS)
    sub = (f"rows = Tmag {mags} (bright on top);  cols = BP-RP: blue < {COL_BINS[0][1]:g}, "
           f"mid {COL_BINS[1][0]:g}-{COL_BINS[1][1]:g}, red > {COL_BINS[2][0]:g};  outer 3x3 = CCD cells (y up)\n"
           "stamp title: n, cxx/cyy (1e-3 px², − = data sharper than model)")
    plot(cells, out / f"resgrid_{label}.png", f"{label}: stacked residual/flux by colour x mag across the CCD\n{sub}", "img")
    plot(cells, out / f"resgrid_{label}_rel.png",
         f"{label}: same, minus each cell's all-star stack (colour / magnitude dependence only)\n{sub}", "rel")
    js = {f"{iy},{ix}": {f"{jm},{jc}": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                         for k, v in st.items() if k not in ("img", "rel")}
                         for (jm, jc), st in c["stamps"].items()} for (iy, ix), c in cells.items()}
    (out / f"resgrid_{label}.json").write_text(json.dumps(
        dict(mag_bins=MAG_BINS, colour_bins=COL_BINS, ncell=NCELL, units_width="1e-3 px^2", cells=js), indent=1,
        default=float))
    print("wrote", out / f"resgrid_{label}.png")


if __name__ == "__main__":
    main()
