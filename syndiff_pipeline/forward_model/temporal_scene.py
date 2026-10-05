# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Build a scene for another FFI of the same SCC with EXACTLY the reference scene's stars.

Temporal ePSF study (2026-09-29): every per-frame scene keeps the reference scene's star list, roles, stamp
squares, overlap islands and static extra masks (bad columns, nuisance cores), so two fits differ only
in the frame's pixels.  What changes per frame:

  data, noise, finite   from the frame's hp_d (same lane, same crop geometry)
  valid                 reference valid  AND  frame finite  AND  frame mask (MaskCatalog at the frame's BTJD,
                        bits edge/PS1/TNS/asteroid; bright is static)  AND  NOT frame-specific isolated spikes
                        (|frame - reference| > 10 sigma, all 8 neighbours < 3 sigma, r >= 3 px from the square centre)

The source bundle (static TAN+Chebyshev WCS, Gaia positions at the reference epoch) is shared; the per-frame WCS
is re-fitted through ``wcs_coeff`` (the scene fit's stage 1 trains only the WCS).  Proper motion over one sector
(<= 13 d) is < 1 mas, far below 0.001 px.

usage: python -m syndiff_pipeline.forward_model.temporal_scene --ref-scene DIR --frame-stem STEM --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax  # noqa: F401  (before pandas/pyarrow)
import numpy as np
from astropy.io import fits

from . import scene_export as SE


def build(ref_scene: Path, frame_stem: str, out_dir: Path, *, data_root: str, spike_sigma: float = 10.0) -> dict:
    z = dict(np.load(ref_scene / "scene_bundle.npz"))
    meta = json.loads((ref_scene / "scene_meta.json").read_text())
    ws = Path(meta["workspace"])
    hp = ws / "hp_d" / f"{frame_stem}_hp_d.fits.fz"
    with fits.open(hp) as h:
        cal = np.asarray(h[1].data, np.float32)
        noise = np.asarray(h[2].data, np.float32)
        btjd = 0.5 * (h[1].header.get("TSTART", np.nan) + h[1].header.get("TSTOP", np.nan))
    if not np.isfinite(btjd):
        raise ValueError(f"no TSTART/TSTOP in {hp}")
    mask, mask_src = SE.load_frame_mask(ws, data_root=Path(data_root), sector=meta["sector"], camera=meta["camera"],
                                        ccd=meta["ccd"], btjd=btjd, shape=cal.shape)
    other = (mask & (SE.BIT_EDGE | SE.BIT_PS1 | SE.BIT_TNS | SE.BIT_ASTEROID)) != 0
    finite = np.isfinite(cal) & np.isfinite(noise) & (noise > 0)

    S = int(z["stamp"]); N = z["role"].shape[0]
    k = np.arange(S * S); dx = k % S - S // 2; dy = k // S - S // 2; r = np.hypot(dx, dy)
    gx = z["cx"][:, None].astype(np.int64) + dx[None]; gy = z["cy"][:, None].astype(np.int64) + dy[None]
    H, W = cal.shape
    inarr = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    gxc = np.clip(gx, 0, W - 1); gyc = np.clip(gy, 0, H - 1)
    fin = inarr & finite[gyc, gxc]
    data = np.where(fin, cal[gyc, gxc], 0.0).astype(np.float32)
    nz = np.where(fin, noise[gyc, gxc], 1.0).astype(np.float32)
    oth = inarr & other[gyc, gxc]

    # frame-specific isolated spikes, relative to the reference frame's pixels
    ref_ok = z["finite"].astype(bool) & fin
    dev = np.where(ref_ok, (data - z["data"]) / np.sqrt(nz ** 2 + z["noise"] ** 2), 0.0)
    a3 = np.abs(dev).reshape(N, S, S)
    pad = np.pad(a3, ((0, 0), (1, 1), (1, 1)))
    nb = np.zeros_like(a3)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a or b:
                nb = np.maximum(nb, pad[:, 1 + a:1 + a + S, 1 + b:1 + b + S])
    spike = ((a3 > spike_sigma) & (nb < 3)).reshape(N, S * S) & (r >= 3)[None]
    uid = z["uid"]
    kill = np.zeros(int(z["n_union"]) + 1, bool)
    kill[uid[spike & z["valid"].astype(bool)]] = True
    kill[uid[oth & z["valid"].astype(bool)]] = True
    kill[uid[~fin & z["valid"].astype(bool)]] = True
    valid = z["valid"].astype(bool) & ~kill[uid]

    z.update(data=data, noise=nz, finite=fin, valid=valid)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scene_bundle.npz", **z)
    meta = dict(meta)
    meta.update(frame_stem=frame_stem, frame_btjd=float(btjd), mask_source=mask_src,
                temporal_scene=dict(ref_scene=str(ref_scene), ref_frame_stem=json.loads(
                    (ref_scene / "scene_meta.json").read_text())["frame_stem"],
                    n_spike_px=int(spike.sum()), n_mask_px=int((oth & z["valid"].astype(bool)).sum()),
                    valid_ref=int(np.load(ref_scene / "scene_bundle.npz")["valid"].sum()),
                    valid_frame=int(valid.sum())))
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    return meta["temporal_scene"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ref-scene", required=True)
    p.add_argument("--frame-stem", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-root", default="/astro/armin/koji/syndiff/data")
    p.add_argument("--spike-sigma", type=float, default=10.0)
    a = p.parse_args(argv)
    info = build(Path(a.ref_scene), a.frame_stem, Path(a.out_dir), data_root=a.data_root, spike_sigma=a.spike_sigma)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
