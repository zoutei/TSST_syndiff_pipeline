# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""A3: score one fit's params on the UNMODIFIED scene (every star free) and tag each star in-fold / out-of-fold.

    python -m syndiff_pipeline.forward_model.crossfit.score_oof --fit-out <fit dir> --base-scene <scene> --out <score.npz> \
        [--folds <folds.npz> --held-fold k] [--params params_stage3.npz]

The model is rebuilt from the fit's own fit_meta.json: colour file, colour_ref and <delta^2> (they depend on the
trained population, so they are NOT recomputed on the base scene), colour gauge/extras, optical axis. A pre-09-24
meta without g8 keys means gauge 'mean', no extras (as scene_fit.resolve_g8_defaults). The fine-nbr prior only
enters the loss, not the rendering, so it does not matter here.

Per star (index = base-scene star):
  c2      core chi^2: mean over valid pixels with r <= 3 px of ((d - m) / sigma)^2, m = the full scene model with every
          star's re-solved flux (plan A3 formula)
  ncore   number of those pixels
  flux    re-solved flux (e-/s)
  oof     star is a trainee (role0 != 2) of the base scene whose fold == held fold  -> out of fold for this fit
  infold  trainee whose fold != held fold -> trained by this fit
  pix_disjoint  from folds.npz
No per-star S/N weights are used anywhere (a S/N-dependent weight fakes a brightness trend, bfwidth_20260929).
"""
from __future__ import annotations

import jax  # noqa: F401  (import jax before pandas/pyarrow: XLA segfault otherwise)
import jax.numpy as jnp

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np

from .. import fit as FIT
from .. import loss as L
from .. import scene_fit as SF

CORE_R = 3.0


MODEL_ENV = (("epsf_nodes", "SYNDIFF_EPSF_NODES"), ("local_poly_hard", "SYNDIFF_EPSF_LOCAL_POLY_HARD"))


def configure_model_env(meta: dict) -> dict:
    """Restore model options that the fitting code reads from the environment, from the fit's own fit_meta.

    Branch epsf-localpoly-exp (prior bake-off, 2026-09-30) switches a coarser node grid (``epsf_nodes``, read in
    SF.Scene.__init__) and hard local-polynomial smoothing (``local_poly_hard``, a loss.py global) by env var. A fit
    scored with the wrong setting renders a different model and still gives plausible numbers (hard smoothing mismatch
    is silent), so: set both from fit_meta before the Scene/model are built; refuse if the shell already holds a
    different non-zero value; refuse if this code cannot honour a non-zero setting. Fits without the keys = 0/0 =
    unchanged behaviour.
    """
    want = {key: int(meta.get(key, 0) or 0) for key, _ in MODEL_ENV}
    for key, var in MODEL_ENV:
        have = int(os.environ.get(var, "0") or 0)
        if have and have != want[key]:
            raise RuntimeError(f"{var}={have} in the environment but the fit was trained with {key}={want[key]}; "
                               f"unset it (the scorer sets it from fit_meta)")
    if want["epsf_nodes"] and "SYNDIFF_EPSF_NODES" not in inspect.getsource(SF.Scene.__init__):
        raise RuntimeError(f"fit uses epsf_nodes={want['epsf_nodes']} but this code has no node-grid option; "
                           f"score it with the branch that trained it (epsf-localpoly-exp)")
    os.environ["SYNDIFF_EPSF_NODES"] = str(want["epsf_nodes"])
    if hasattr(L, "set_local_poly_hard"):
        L.set_local_poly_hard(want["local_poly_hard"])
    elif want["local_poly_hard"]:
        raise RuntimeError(f"fit uses local_poly_hard={want['local_poly_hard']} but this code has no "
                           f"set_local_poly_hard; score it with the branch that trained it (epsf-localpoly-exp)")
    return want


def build(fit_out: Path, base_scene: Path, params_name: str = "params_stage3.npz"):
    meta = json.loads((fit_out / "fit_meta.json").read_text())
    configure_model_env(meta)
    scene = SF.Scene(Path(base_scene))
    if meta.get("colour_file"):
        scene.use_colour_file(meta["colour_file"])
    cref = meta["colour_ref"]
    cref = {int(k): v for k, v in cref.items()} if isinstance(cref, dict) else cref
    d2 = meta.get("chroma_delta2_mean", 0.0)
    d2 = {int(k): v for k, v in d2.items()} if isinstance(d2, dict) else d2
    axis = tuple(meta["chroma_axis"]) if meta.get("chroma_axis") else None
    extras = SF.g8_extras_tuple(meta.get("chroma_g8_extras", "") or "", int(meta.get("chroma_g8_blur_order", 0) or 0),
                                bool(meta.get("chroma_g8_no_dil", False)))
    _, diag = SF.make_model(
        scene, colour_ref=cref, huber_delta=meta["huber_delta"], ridge=meta["ridge"],
        prior_kappa=meta["prior_kappa"], lambda_lap=meta["lambda_lap"], lambda_pixel=meta["lambda_pixel_lap"],
        chroma_axis=axis, lambda_fine_nbr=meta.get("lambda_fine_nbr", 0.0),
        fine_nbr_mode=meta.get("fine_nbr_mode") or "plain",
        chroma_g8_gauge=meta.get("chroma_g8_gauge", "mean"), chroma_g8_no_dil=bool(meta.get("chroma_g8_no_dil", False)),
        chroma_g8_extras=extras, delta2_mean=d2)
    pp = fit_out / params_name
    for alt in ("params.npz", "params_latest.npz"):     # a running fit has only params_latest.npz
        if not pp.exists():
            pp = fit_out / alt
    params = FIT.load_params_npz(pp)
    params = SF.set_chroma_model(params, meta.get("chroma_model", "global8"), halo=bool(meta.get("chroma_halo", False)),
                                 g8_init=None, g8_extras=extras, g8_source_extras=extras) if "chroma_g8" in params else params
    return meta, scene, jax.jit(diag), params, str(pp)


def core_chi2(scene, chi):
    z = scene.z
    S = int(z["stamp"])
    k = np.arange(S * S)
    dx, dy = k % S - S // 2, k // S - S // 2
    px = z["cx"][:, None] + dx[None]
    py = z["cy"][:, None] + dy[None]
    r = np.hypot(px - z["x0"][:, None], py - z["y0"][:, None])
    m = np.asarray(z["valid"], bool) & (r <= CORE_R)
    n = m.sum(1)
    return np.where(m, chi ** 2, 0).sum(1) / np.maximum(n, 1), n


def score(fit_out, base_scene, folds=None, held_fold=None, params_name="params_stage3.npz", state="fresh"):
    meta, scene, diag, params, ppath = build(Path(fit_out), Path(base_scene), params_name)
    N = scene.N
    if state == "fresh":
        pix = np.ones(scene.U + 1, np.float32)
        pix[scene.U] = 0.0
        st = {"pix_active_u": jnp.asarray(pix), "star_free": jnp.ones((N,), jnp.float32),
              "f_fixed": jnp.zeros((N,), jnp.float32), "f_prior": jnp.zeros((N,), jnp.float32)}
        f0, _, _ = diag(params, st)
        st, _ = SF.update_flux_prior(f0, st, scene)          # as scene_fit does before training
    else:                                                    # the fit's own state (reproduces its in-scene diag)
        sz = np.load(state)
        st = {k: jnp.asarray(sz[k]) for k in ("pix_active_u", "star_free", "f_fixed", "f_prior")}
    f, _, chi = diag(params, st)
    f = np.asarray(f, np.float64)
    chi = np.asarray(chi, np.float64)
    c2, n = core_chi2(scene, chi)
    z = scene.z
    role0 = scene.role.copy()
    out = dict(c2=c2, ncore=n, flux=f, source_id=z["source_id"], T=z["tess_mag"].astype(float),
               x=z["x0"].astype(float), y=z["y0"].astype(float), role0=role0,
               colour=np.asarray(scene.colour, float)[z["star_bundle_index"]],
               free=np.asarray(st["star_free"]) > 0)
    if folds is not None:
        fz = np.load(folds)
        assert (fz["source_id"] == z["source_id"]).all(), "folds.npz is for another scene"
        fold = fz["fold"]
        trainee = role0 != SF.ROLE_NUISANCE
        out.update(fold=fold, pix_disjoint=fz["pix_disjoint"], trainee=trainee)
        if held_fold is None:
            out.update(oof=np.zeros(N, bool), infold=trainee)
        else:
            out.update(oof=trainee & (fold == held_fold), infold=trainee & (fold != held_fold), held_fold=held_fold)
    out["ok"] = (f > 0) & (n >= 15)
    info = dict(fit_out=str(fit_out), params=ppath, base_scene=str(base_scene), held_fold=held_fold,
                colour_file=meta.get("colour_file"), fine_nbr_mode=meta.get("fine_nbr_mode"),
                chroma_g8_extras=meta.get("chroma_g8_extras"), state=str(state))
    return out, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-out", required=True)
    ap.add_argument("--base-scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--held-fold", type=int, default=None)
    ap.add_argument("--params", default="params_stage3.npz")
    ap.add_argument("--state", default="fresh", help="'fresh' (all stars free) or a state npz")
    a = ap.parse_args(argv)
    out, info = score(a.fit_out, a.base_scene, a.folds, a.held_fold, a.params, a.state)
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o, **out)
    o.with_suffix(".json").write_text(json.dumps(info, indent=1))
    s = out["ok"] & out.get("oof", np.zeros(len(out["c2"]), bool))
    msg = {"median_c2_all_trainees": float(np.median(out["c2"][out["ok"] & (out["role0"] != 2)]))}
    if s.any():
        msg["median_c2_oof"] = float(np.median(out["c2"][s]))
        msg["n_oof"] = int(s.sum())
    print(json.dumps(msg))


if __name__ == "__main__":
    main()
