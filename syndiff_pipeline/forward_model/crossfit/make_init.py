# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""A2: leak-free initial params for fold fits.

    python -m syndiff_pipeline.forward_model.crossfit.make_init --donor-fit <other-sector fit out/> --scene-dir <target scene> \
        --route xsec|smooth --out <dir>

The init must not have seen any star of the target frame. Both routes take the ePSF from a fit of ANOTHER sector's
frame on the same CCD (the donor), never from a fit of the target frame (C5, #14, A3 all saw every target star):

  xsec    donor epsf_base_raw + donor chroma_g8, as trained.
  smooth  node-mean of the donor's decoded ePSF (every node = the mean over nodes, so the donor's node-to-node detail
          is gone) and chroma_g8 = 0 (colour learnt from scratch).

In both, wcs_coeff = the target scene's own initial WCS (the bundle's params0), no temporal modes.
Agreement of the two routes' out-of-fold scores is the leak/convergence test (V0.2).

  photutils  no donor. The ePSF is a photutils EPSFBuilder build on the TARGET frame's own bootstrap difference image
             (--hp-d), recentred to the scene_fit core-centroid gauge; the WCS is a free-x,y photometry centroid fit
             against epoch-propagated Gaia (see ``photutils_init``). With --fold k only the training-fold stars
             (fold != k, role != 2) enter either step, so the start never sees the held-out stars:

    python -m syndiff_pipeline.forward_model.crossfit.make_init --route photutils --scene-dir <scene> \
        --hp-d <hp_d.fits.fz> [--fold k] --out <dir>

Writes <out>/params_init.npz and a fit_meta.json beside it that tells scene_fit's warm-start check the colour gauge
and extras of chroma_g8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax  # noqa: F401  (before pandas/pyarrow anywhere below: pyarrow-first segfaults XLA)
import numpy as np

from .. import epsf_model as EM


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--donor-fit", default=None, help="fit out/ dir of another sector's frame, same CCD "
                    "(xsec/smooth; for photutils only its fit_meta.json colour-model keys are copied, if given)")
    ap.add_argument("--scene-dir", required=True, help="target scene (for the initial WCS)")
    ap.add_argument("--route", choices=("xsec", "smooth", "photutils"), required=True)
    ap.add_argument("--out", required=True)
    ph = ap.add_argument_group("photutils route")
    ph.add_argument("--hp-d", default=None, help="target frame bootstrap hp_d (HDU1 diff, HDU2 noise, HDU3 mask)")
    ph.add_argument("--fold", type=int, default=None, help="held-out fold k of the K-fold harness (default: all stars)")
    ph.add_argument("--n-folds", type=int, default=5)
    ph.add_argument("--fold-seed", type=int, default=20260929,
                    help="tile-map seed; the harness used F1 20260929, F2 20260930, S22 20260931 (prefer --folds-csv)")
    ph.add_argument("--folds-csv", default=None, help="harness folds.csv (source_id,fold): authoritative fold of each star")
    ph.add_argument("--fold-tile", type=int, default=128)
    ph.add_argument("--fold-pattern", choices=("diagonal", "group"), default="diagonal")
    ph.add_argument("--min-sep-px", type=float, default=6.0, help="ePSF pool isolation radius (0 = off)")
    ph.add_argument("--n-jobs", type=int, default=6)
    ph.add_argument("--phot-init", choices=("header", "scene", "linear"), default="header",
                    help="initial x,y of the free-xy photometry: FFI header WCS of the bundle's Gaia positions "
                         "(default; independent of the earlier fit), the scene's x0,y0, or the pure linear WCS "
                         "(off by up to ~15 px at the corners: not usable)")
    ph.add_argument("--ffi", default=None, help="target FFI for --phot-init header (default: found beside the lane)")
    ph.add_argument("--xy-bounds", type=float, default=2.0, help="max |fit - init| in x,y [px] in the photometry")
    ph.add_argument("--wcs-tmag-max", type=float, default=13.0)
    ph.add_argument("--mask-source", choices=("scene", "hpd"), default="scene",
                    help="reject mask: the scene's shared mask (default) or the hp_d mask plane")
    a = ap.parse_args(argv)

    if a.route == "photutils":
        if not a.hp_d:
            raise SystemExit("--route photutils needs --hp-d")
        from .photutils_init import build_photutils_init
        dm = json.loads((Path(a.donor_fit) / "fit_meta.json").read_text()) if a.donor_fit else None
        fm = build_photutils_init(scene_dir=a.scene_dir, hp_d=a.hp_d, out=a.out, fold=a.fold, n_folds=a.n_folds,
                                  seed=a.fold_seed, tile=a.fold_tile, pattern=a.fold_pattern, min_sep_px=a.min_sep_px,
                                  n_jobs=a.n_jobs, phot_init=a.phot_init, ffi=a.ffi, xy_bounds=a.xy_bounds, wcs_tmag_max=a.wcs_tmag_max, donor_meta=dm,
                                  mask_source=a.mask_source, folds_csv=a.folds_csv)
        print(json.dumps({k: v for k, v in fm["crossfit_init"].items()
                          if k in ("route", "fold", "star_counts", "wcs")}, default=str))
        return
    if not a.donor_fit:
        raise SystemExit(f"--route {a.route} needs --donor-fit")
    donor = Path(a.donor_fit)
    dm = json.loads((donor / "fit_meta.json").read_text())
    dp = np.load(donor / "params.npz")
    scene_meta = json.loads((Path(a.scene_dir) / "scene_meta.json").read_text())
    if Path(dm["scene_dir"]).resolve() == Path(a.scene_dir).resolve():
        raise SystemExit("donor fit was trained on the target scene: not leak-free")
    if dm.get("scene_meta", {}).get("frame_stem") == scene_meta.get("frame_stem"):
        raise SystemExit("donor fit is the target frame: not leak-free")
    b = np.load(scene_meta["source_bundle"])
    out = {
        "wcs_coeff": np.asarray(b["params0_wcs_coeff"], np.float32),
        "epsf_modes": np.zeros((0,) + dp["epsf_base_raw"].shape, np.float32),
        "w_coeff": np.zeros((0, 1), np.float32),
    }
    if a.route == "xsec":
        out["epsf_base_raw"] = dp["epsf_base_raw"].astype(np.float32)
        out["chroma_g8"] = dp["chroma_g8"].astype(np.float32)
    else:
        base = np.asarray(EM.decode_epsf_base(dp["epsf_base_raw"]), np.float64)
        mean = base.mean(axis=(0, 1), keepdims=True)
        mean = mean / mean.sum(axis=(-2, -1), keepdims=True)
        out["epsf_base_raw"] = np.asarray(EM.encode_epsf_base(np.broadcast_to(mean, base.shape).astype(np.float32)))
        out["chroma_g8"] = np.zeros_like(dp["chroma_g8"], np.float32)
    if "epsf_repr" in dp.files:
        out["epsf_repr"] = dp["epsf_repr"]
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    np.savez(o / "params_init.npz", **out)
    meta = {"chroma_g8_gauge": dm.get("chroma_g8_gauge", "raw"), "chroma_g8_extras": dm.get("chroma_g8_extras", ""),
            "chroma_g8_blur_order": dm.get("chroma_g8_blur_order", 0), "chroma_g8_no_dil": dm.get("chroma_g8_no_dil", False),
            "crossfit_init": dict(route=a.route, donor_fit=str(donor), donor_scene=dm["scene_dir"],
                                  donor_frame=dm.get("scene_meta", {}).get("frame_stem"),
                                  target_scene=a.scene_dir, target_frame=scene_meta.get("frame_stem"),
                                  wcs_from="target bundle params0_wcs_coeff")}
    (o / "fit_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["crossfit_init"]), {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
