# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Local-background measurement around scene-fit stars (localbg_20260930, step 1).

    python -m syndiff_pipeline.forward_model.crossfit.bg_annulus --fit-out <fit dir> --base-scene <scene> --out <npz> \
        [--folds folds.npz --held-fold k]

Renders the fit on the UNMODIFIED scene (score_oof.build: the fit's own colour/gauge/env settings, fresh state) and
records per star, over its own 15 x 15 stamp pixels that are valid:

  b_ann     median residual d - m over the annulus r_lo <= r <= r_hi px (default 5-7) from the catalogue position (e-/s)
  b_mean    mean of the same (e-/s)
  b_in      median residual over 3 <= r < 5 px (e-/s)
  sig_ann   median pixel noise sigma in the annulus (e-/s)
  wing_ann  median of the star's OWN model f_i T_i in the annulus (e-/s)
  nbr_ann   median of the OTHER stars' model in the annulus (e-/s)
  dflux_bg  flux change the least-squares solve would make if a constant b_ann were added to the star's stamp:
                df_i = b_ann * sum_p w_p T_ip / sum_p w_p T_ip^2,   w_p = 1 / sigma_p^2 over valid stamp pixels
            reported as df_i / f_i (fractional). T_ip = star i's unit-flux template at pixel p.
  plus flux, T, colour (the fit's colour input), bp_rp, x, y, r_axis, role0, n_ann, and fold/oof/infold if --folds.

d - m is the full-scene residual: data minus every modelled star's template x re-solved flux, so it is what no star
in the model explains. No S/N weights are used in any summary statistic.
"""
from __future__ import annotations

import jax  # noqa: F401
import jax.numpy as jnp

import argparse
import json
from pathlib import Path

import numpy as np

from .. import loss as L
from .. import scene_fit as SF
from .score_oof import build


def own_templates(meta, scene, params):
    """Per-star unit-flux templates, exactly as make_model's templates() (without stop_gradient)."""
    cref = meta["colour_ref"]
    cref = {int(k): v for k, v in cref.items()} if isinstance(cref, dict) else cref
    d2 = meta.get("chroma_delta2_mean", 0.0)
    d2 = {int(k): v for k, v in d2.items()} if isinstance(d2, dict) else d2
    axis = tuple(meta["chroma_axis"]) if meta.get("chroma_axis") else None
    extras = SF.g8_extras_tuple(meta.get("chroma_g8_extras", "") or "", int(meta.get("chroma_g8_blur_order", 0) or 0),
                                bool(meta.get("chroma_g8_no_dil", False)))
    ctxs = scene.contexts(cref, chroma_axis=axis, chroma_g8_gauge=meta.get("chroma_g8_gauge", "mean"),
                          chroma_g8_no_dil=bool(meta.get("chroma_g8_no_dil", False)), chroma_g8_extras=extras,
                          delta2_mean=d2)
    S2 = scene.S * scene.S
    T = np.zeros((scene.N, S2), np.float32)
    for _, idx, ctx in ctxs:
        T[np.asarray(idx)] = np.asarray(L.forward_model(params, ctx)[0]).reshape(-1, S2)
    return T


def measure(fit_out, base_scene, folds=None, held_fold=None, r_lo=5.0, r_hi=7.0):
    meta, scene, diag, params, _ = build(Path(fit_out), Path(base_scene))
    N = scene.N
    pix = np.ones(scene.U + 1, np.float32)
    pix[scene.U] = 0.0
    st = {"pix_active_u": jnp.asarray(pix), "star_free": jnp.ones((N,), jnp.float32),
          "f_fixed": jnp.zeros((N,), jnp.float32), "f_prior": jnp.zeros((N,), jnp.float32)}
    f0, _, _ = diag(params, st)
    st, _ = SF.update_flux_prior(f0, st, scene)
    f, _, chi = diag(params, st)
    f = np.asarray(f, np.float64)
    z = scene.z
    sig = np.sqrt(np.asarray(L.pixel_variance(jnp.asarray(z["noise"])), np.float64))
    res = np.asarray(chi, np.float64) * sig
    data = np.asarray(z["data"], np.float64)
    model = data - res
    T = own_templates(meta, scene, params).astype(np.float64)
    own = f[:, None] * T
    S = int(z["stamp"])
    k = np.arange(S * S)
    px = z["cx"][:, None] + (k % S - S // 2)[None]
    py = z["cy"][:, None] + (k // S - S // 2)[None]
    r = np.hypot(px - z["x0"][:, None], py - z["y0"][:, None])
    valid = np.asarray(z["valid"], bool)
    ann = valid & (r >= r_lo) & (r <= r_hi)
    inner = valid & (r >= 3.0) & (r < r_lo)

    def rowmed(a, m):
        out = np.full(N, np.nan)
        for i in range(N):
            if m[i].any():
                out[i] = np.median(a[i][m[i]])
        return out

    b_ann = rowmed(res, ann)
    w = np.where(valid, 1.0 / sig ** 2, 0.0)
    dflux = b_ann * (w * T).sum(1) / np.maximum((w * T * T).sum(1), 1e-30)
    ax = scene.optical_axis()
    out = dict(
        b_ann=b_ann, b_mean=np.where(ann.any(1), (res * ann).sum(1) / np.maximum(ann.sum(1), 1), np.nan),
        b_in=rowmed(res, inner), sig_ann=rowmed(sig, ann), wing_ann=rowmed(own, ann),
        nbr_ann=rowmed(model - own, ann), n_ann=ann.sum(1), flux=f, dflux_bg=dflux / np.where(f > 0, f, np.nan),
        T=z["tess_mag"].astype(float), x=z["x0"].astype(float), y=z["y0"].astype(float),
        colour=np.asarray(scene.colour, float)[z["star_bundle_index"]],
        bp_rp=np.asarray(scene.src.bp_rp, float)[z["star_bundle_index"]],
        r_axis=np.hypot(z["x0"] - ax[0], z["y0"] - ax[1]), role0=scene.role.copy(), source_id=z["source_id"],
        r_lo=r_lo, r_hi=r_hi)
    if folds is not None:
        fz = np.load(folds)
        assert (fz["source_id"] == z["source_id"]).all()
        tr = scene.role != SF.ROLE_NUISANCE
        out.update(fold=fz["fold"], oof=tr & (fz["fold"] == held_fold) if held_fold is not None else np.zeros(N, bool),
                   infold=tr & (fz["fold"] != held_fold) if held_fold is not None else tr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-out", required=True)
    ap.add_argument("--base-scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--held-fold", type=int, default=None)
    ap.add_argument("--r-lo", type=float, default=5.0)
    ap.add_argument("--r-hi", type=float, default=7.0)
    a = ap.parse_args(argv)
    out = measure(a.fit_out, a.base_scene, a.folds, a.held_fold, a.r_lo, a.r_hi)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    tr = out["role0"] != SF.ROLE_NUISANCE
    print(json.dumps(dict(n=int(tr.sum()), median_b_ann=float(np.nanmedian(out["b_ann"][tr])),
                          median_sig=float(np.nanmedian(out["sig_ann"][tr])),
                          median_dflux_permil=float(1e3 * np.nanmedian(out["dflux_bg"][tr])))))


if __name__ == "__main__":
    main()
