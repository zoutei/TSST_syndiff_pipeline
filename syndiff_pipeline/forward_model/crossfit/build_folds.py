# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""A1: fold assignment + per-fold held-out scenes.

    python -m syndiff_pipeline.forward_model.crossfit.build_folds --scene-dir <scene> --name F1 --out <dir> \
        [--seed 20260929] [--extra-catalogue ids.csv --ffi <ffic.fits.fz>]

Writes, under ``<out>``:
  folds.npz            tile map + scene-star table (source_id, fold, role0, T, colour, x, y, pix_disjoint)
  folds.csv            source_id,fold,in_scene,role0,T,colour,x,y,pix_disjoint for the scene stars and, with
                       --extra-catalogue, every catalogue source on the CCD (fold = tile of its position).
                       This is the file the G4 scorer reads with --folds (it merges on source_id).
  scenes/<name>_fold<k>/scene_bundle.npz   fold-k stars (trainees only) demoted to nuisance (role 2): modelled
                       and flux-solved, template stop-gradient, so fit k never trains on them. Scored later on the
                       UNMODIFIED scene; "fit k" means "fit with fold k held out".
  balance.json         star counts per fold by Tmag bin, 3x3 CCD cell and colour tertile.

pix_disjoint (per star, w.r.t. the fit that holds it out): its r <= 3 px core shares no pixel with the full
15 x 15 stamp box of any trainee of that fit.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import folds as FO

TBINS = [8, 9, 10, 11, 12, 13]


def project_catalogue(cat: pd.DataFrame, ffi: Path, scene_z) -> tuple[np.ndarray, np.ndarray, dict]:
    """ra/dec -> science-local px with the FFI header WCS, then shifted by the median offset to the scene's x0/y0
    (the scene positions come from the fitted WCS; the header WCS is only good to ~0.1-1 px, enough for tiles)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(ffi) as hd:
        h = next(x.header for x in hd if x.header.get("NAXIS") == 2 and "CTYPE1" in x.header)
    w = WCS(h)
    col, row = w.all_world2pix(cat["ra"].values, cat["dec"].values, 0)
    x, y = col - 44.0, row
    idx = pd.Index(cat["source_id"].values)
    loc = idx.get_indexer(scene_z["source_id"])
    k = loc >= 0
    dx = scene_z["x0"][k] - x[loc[k]]
    dy = scene_z["y0"][k] - y[loc[k]]
    off = dict(n_matched=int(k.sum()), dx_med=float(np.median(dx)), dy_med=float(np.median(dy)),
               dx_mad=float(np.median(np.abs(dx - np.median(dx)))), dy_mad=float(np.median(np.abs(dy - np.median(dy)))))
    return x + off["dx_med"], y + off["dy_med"], off


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--pattern", choices=("group", "diagonal"), default="group",
                    help="group = seeded shuffle inside group x group super-tiles (K=2); diagonal = (ix + 2 iy) mod K (K=5)")
    ap.add_argument("--extra-catalogue", default=None, help="CSV with source_id,ra,dec[,T,BP,RP] of the CCD")
    ap.add_argument("--ffi", default=None, help="FFI FITS whose header WCS projects --extra-catalogue")
    ap.add_argument("--extra-positions", default=None,
                    help="CSV with source_id,x,y (science-local px) of more sources, e.g. the G4 scorer's stars file")
    ap.add_argument("--no-scenes", action="store_true")
    ap.add_argument("--scene-folds", default=None, help="comma list of folds to write scenes for (default all)")
    a = ap.parse_args(argv)

    src, out = Path(a.scene_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = dict(np.load(src / "scene_bundle.npz"))
    meta = json.loads((src / "scene_meta.json").read_text())
    bundle = np.load(meta["source_bundle"])
    colour = np.asarray(bundle["bp_rp"], float)[z["star_bundle_index"]]
    x, y = z["x0"].astype(float), z["y0"].astype(float)
    T = z["tess_mag"].astype(float)
    role0 = z["role"].copy()
    S = int(z["stamp"])

    tmap = (FO.tile_map_diagonal(a.seed, a.tile, a.n_folds) if a.pattern == "diagonal"
            else FO.tile_map(a.seed, a.tile, a.group, a.n_folds))
    fold = FO.fold_of(tmap, a.tile, x, y)
    trainee = role0 != 2

    # coverage of each fit's trainees (fit k trains on trainees with fold != k)
    cov = [FO.stamp_coverage(z["cx"][trainee & (fold != k)], z["cy"][trainee & (fold != k)], S)
           for k in range(a.n_folds)]
    pdis = np.zeros(len(x), bool)
    for k in range(a.n_folds):
        s = fold == k
        pdis[s] = FO.core_disjoint(cov[k], x[s], y[s])

    rows = pd.DataFrame(dict(source_id=z["source_id"], fold=fold, in_scene=True, role0=role0, T=T, colour=colour,
                             x=x, y=y, pix_disjoint=pdis))
    proj = None
    if a.extra_catalogue:
        cat = pd.read_csv(a.extra_catalogue)
        ex, ey, proj = project_catalogue(cat, Path(a.ffi), z)       # match on the full catalogue for the offset
        on = ((ex >= -0.5) & (ex < FO.NPIX - 0.5) & (ey >= -0.5) & (ey < FO.NPIX - 0.5)
              & ~cat["source_id"].isin(rows["source_id"]).values)
        cat, ex, ey = cat[on], ex[on], ey[on]
        ef = FO.fold_of(tmap, a.tile, ex, ey)
        epd = np.zeros(len(ex), bool)
        for k in range(a.n_folds):
            s = ef == k
            epd[s] = FO.core_disjoint(cov[k], ex[s], ey[s])
        ecol = (cat["BP"] - cat["RP"]).values if {"BP", "RP"} <= set(cat) else np.full(len(ex), np.nan)
        rows = pd.concat([rows, pd.DataFrame(dict(
            source_id=cat["source_id"].values, fold=ef, in_scene=False, role0=-1,
            T=cat["T"].values if "T" in cat else np.nan, colour=ecol, x=ex, y=ey, pix_disjoint=epd))],
            ignore_index=True)
        proj["n_extra_on_ccd"] = int(on.sum())
    if a.extra_positions:
        ep = pd.read_csv(a.extra_positions)
        ep = ep[~ep["source_id"].isin(rows["source_id"])]
        px_, py_ = ep["x"].values.astype(float), ep["y"].values.astype(float)
        pf = FO.fold_of(tmap, a.tile, px_, py_)
        ppd = np.zeros(len(ep), bool)
        for k in range(a.n_folds):
            s = pf == k
            ppd[s] = FO.core_disjoint(cov[k], px_[s], py_[s])
        rows = pd.concat([rows, pd.DataFrame(dict(
            source_id=ep["source_id"].values, fold=pf, in_scene=False, role0=-1,
            T=ep["tmag"].values if "tmag" in ep else np.nan,
            colour=ep["bp_rp"].values if "bp_rp" in ep else np.nan, x=px_, y=py_, pix_disjoint=ppd))],
            ignore_index=True)
    rows.to_csv(out / "folds.csv", index=False, float_format="%.4f")

    np.savez(out / "folds.npz", tile_map=tmap, tile=a.tile, group=a.group, pattern=a.pattern, seed=a.seed, n_folds=a.n_folds,
             source_id=z["source_id"], fold=fold, role0=role0, T=T, colour=colour, x=x, y=y, pix_disjoint=pdis)

    # balance table (trainees only: the stars whose OOF scores matter for the ePSF)
    cell = np.clip((x // 683).astype(int), 0, 2) * 3 + np.clip((y // 683).astype(int), 0, 2)
    ct = np.nanquantile(colour[trainee], [1 / 3, 2 / 3])
    ctb = np.digitize(colour, ct)
    bal = {"Tmag": {}, "cell": {}, "colour_tercile": {}, "colour_tercile_edges": ct.tolist()}
    tb = np.digitize(T, TBINS)
    for b in np.unique(tb):
        lab = f"{TBINS[b - 1] if b else '-inf'}-{TBINS[b] if b < len(TBINS) else 'inf'}"
        bal["Tmag"][lab] = [int((trainee & (tb == b) & (fold == k)).sum()) for k in range(a.n_folds)]
    for c in range(9):
        bal["cell"][str(c)] = [int((trainee & (cell == c) & (fold == k)).sum()) for k in range(a.n_folds)]
    for c in range(3):
        bal["colour_tercile"][str(c)] = [int((trainee & (ctb == c) & (fold == k)).sum()) for k in range(a.n_folds)]
    bal["trainees_per_fold"] = [int((trainee & (fold == k)).sum()) for k in range(a.n_folds)]
    bal["pix_disjoint_frac_trainees"] = [float(pdis[trainee & (fold == k)].mean()) for k in range(a.n_folds)]
    bal["projection"] = proj
    bal["scene"] = str(src)
    bal["args"] = vars(a)
    (out / "balance.json").write_text(json.dumps(bal, indent=1))
    print(json.dumps({k: bal[k] for k in ("Tmag", "trainees_per_fold", "pix_disjoint_frac_trainees", "projection")}))

    if a.no_scenes:
        return
    for k in (range(a.n_folds) if a.scene_folds is None else [int(v) for v in a.scene_folds.split(",")]):
        d = out / "scenes" / f"{a.name}_fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        held = trainee & (fold == k)
        zz = dict(z)
        role = role0.copy()
        role[held] = 2
        zz["role"] = role
        np.savez(d / "scene_bundle.npz", **zz)
        np.savez(d / "heldout.npz", held=held, source_id=z["source_id"], role0=role0, fold=fold, pix_disjoint=pdis)
        m = dict(meta)
        m["n_roles"] = {"contrib": int((role == 0).sum()), "anchor": int((role == 1).sum()),
                        "nuisance": int((role == 2).sum())}
        m["crossfit"] = dict(source_scene=str(src), fold_held=k, n_held=int(held.sum()), seed=a.seed, tile=a.tile,
                             group=a.group, n_folds=a.n_folds)
        (d / "scene_meta.json").write_text(json.dumps(m, indent=1))
    shutil.copy(Path(__file__), out / "build_folds.py.snapshot")


if __name__ == "__main__":
    main()
