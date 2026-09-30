# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Residual rasters for the standard residual grid, rendered correctly for A3 fits and out of fold.

    python -m syndiff_pipeline.forward_model.crossfit.raster_oof --out <dir>/raster_res_model.npz --base-scene <scene> \
        --fit <fit out dir>[:<held fold>] [--fit ...] [--folds folds.npz]

For each --fit, the params are rendered on the UNMODIFIED base scene (score_oof.build: the fit's own colour file,
colour_ref, gauge and extras; magwidth's raster_model.py ignores those and mis-renders A3). With ':k' only the
trainees of fold k are kept (out of fold for that fit); several --fit entries are concatenated, so two folds give
one out-of-fold raster set. Without ':k' all trainees are kept (in-sample). Output format = raster_model.py's
(res/model per star over flux on the 33 x 33, 0.25 px grid; colour = Gaia BP-RP; x, y, flux, role), which
diagnostics/residual_grid.py reads.
"""
from __future__ import annotations

import jax  # noqa: F401
import jax.numpy as jnp

import argparse
from pathlib import Path

import numpy as np

from .. import scene_fit as SF
from ..diagnostics import stacked_temporal_residuals as STR
from .score_oof import build

HW, NG = 4.0, 33


def rasterise_fit(fit_out, base_scene, keep_fold=None, folds=None):
    meta, scene, diag, params, _ = build(Path(fit_out), Path(base_scene))
    N = scene.N
    pix = np.ones(scene.U + 1, np.float32)
    pix[scene.U] = 0.0
    st = {"pix_active_u": jnp.asarray(pix), "star_free": jnp.ones((N,), jnp.float32),
          "f_fixed": jnp.zeros((N,), jnp.float32), "f_prior": jnp.zeros((N,), jnp.float32)}
    f0, _, _ = diag(params, st)
    st, _ = SF.update_flux_prior(f0, st, scene)
    f, _, chi = diag(params, st)
    f = np.asarray(f)
    chi = np.asarray(chi)
    z = scene.z
    res = chi * np.sqrt(np.asarray(z["noise"], float) ** 2 + 1e-6)
    model = np.asarray(z["data"], float) - res
    S = int(z["stamp"])
    k = np.arange(S * S)
    px = (z["cx"][:, None] + (k % S - S // 2)[None]).astype(float)
    py = (z["cy"][:, None] + (k // S - S // 2)[None]).astype(float)
    bp = np.asarray(scene.src.bp_rp, float)[z["star_bundle_index"]]
    sel = (scene.role != SF.ROLE_NUISANCE) & np.isfinite(bp) & (f > 0)
    if keep_fold is not None:
        fz = np.load(folds)
        assert (fz["source_id"] == z["source_id"]).all()
        sel &= fz["fold"] == keep_fold
    idx = np.flatnonzero(sel)
    valid = np.asarray(z["valid"], bool)
    R = np.full((len(idx), NG, NG), np.nan, np.float32)
    M = R.copy()
    for j, i in enumerate(idx):
        args = (np.array([z["x0"][i]]), np.array([z["y0"][i]]), px[i], py[i], valid[i].astype(float)[None])
        R[j] = STR.rasterize_packed_values((res[i] / f[i])[None], *args, oversample=4, half_width=HW)[0]
        M[j] = STR.rasterize_packed_values((model[i] / f[i])[None], *args, oversample=4, half_width=HW)[0]
    return dict(res=R, model=M, colour=bp[idx], x=z["x0"][idx], y=z["y0"][idx], flux=f[idx],
                role=scene.role[idx], tess_mag=z["tess_mag"][idx], source_id=z["source_id"][idx])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-scene", required=True)
    ap.add_argument("--fit", action="append", required=True)
    ap.add_argument("--folds", default=None)
    a = ap.parse_args(argv)
    parts = []
    for spec in a.fit:
        path, _, k = spec.partition(":")
        parts.append(rasterise_fit(path, a.base_scene, int(k) if k else None, a.folds))
        print(spec, len(parts[-1]["flux"]), flush=True)
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)


if __name__ == "__main__":
    main()
