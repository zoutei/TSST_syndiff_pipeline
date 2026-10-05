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

Writes <out>/params_init.npz and a fit_meta.json beside it that tells scene_fit's warm-start check the colour gauge
and extras of chroma_g8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import epsf_model as EM


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--donor-fit", required=True, help="fit out/ dir of another sector's frame, same CCD")
    ap.add_argument("--scene-dir", required=True, help="target scene (for the initial WCS)")
    ap.add_argument("--route", choices=("xsec", "smooth"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

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
