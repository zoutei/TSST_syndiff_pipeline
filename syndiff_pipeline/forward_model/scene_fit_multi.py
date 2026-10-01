# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Joint multi-frame scene fit: ONE static ePSF (+ colour model) for several FFIs of one SCC.

Temporal ePSF study (dev_runs/temporal_epsf_20260929). Every frame's scene must come from
``temporal_scene.py`` off the same reference scene, so all frames share stars, roles, stamp squares,
overlap islands and tier tables; only data / noise / valid / finite differ. Per frame:

  * its own ``wcs_coeff`` (trainable; ``params["wcs_coeff"]`` is (F, n_coeff, 1)),
  * a FIXED jitter blur ``epsf_blur[t]`` = (6, 6, 3) node covariance (Sxx, Syy, Sxy) in px^2 relative to the
    reference ePSF, from spacecraft telemetry (``--blur-file``; zeros = no temporal PSF model),
  * its own exact per-island flux solve.

The static ePSF and colour leaves are shared. Loss = pixel-weighted mean data term over all frames + the
usual ePSF regularisers once. Frames are evaluated sequentially (``jax.lax.map``) to bound memory.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax  # noqa: F401  (before pandas/pyarrow)
import jax.numpy as jnp
import numpy as np
import optax

from . import _bootstrap  # noqa: F401
from . import fit as FIT
from . import loss as L
from . import scene_fit as SF


def _log(msg: str) -> None:
    print(f"[scene_fit_multi {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_multi_model(scene: SF.Scene, frames: list[dict], *, colour_ref, huber_delta, ridge, prior_kappa,
                     lambda_lap, lambda_pixel, chroma_axis, lambda_fine_nbr, chroma_g8_gauge,
                     chroma_g8_no_dil, chroma_g8_extras, delta2_mean):
    """frames: list of per-frame scene_bundle dicts (data, noise, valid, finite); structure from ``scene``."""
    S2 = scene.S * scene.S
    N, U = scene.N, scene.U
    ctxs = scene.contexts(colour_ref, chroma_axis=chroma_axis, chroma_g8_gauge=chroma_g8_gauge,
                          chroma_g8_no_dil=chroma_g8_no_dil, chroma_g8_extras=chroma_g8_extras,
                          delta2_mean=delta2_mean)
    tiers = scene.tier_tables()
    z = scene.z
    for key in ("uid", "role", "pair_i", "pair_j", "island"):
        for fz in frames:
            if not np.array_equal(fz[key], z[key]):
                raise ValueError(f"frame scene differs from the reference scene in {key!r}")
    DATA = jnp.asarray(np.stack([fz["data"] for fz in frames]).astype(np.float32))
    VAR = L.pixel_variance(jnp.asarray(np.stack([fz["noise"] for fz in frames]).astype(np.float32)))
    VALID = jnp.asarray(np.stack([fz["valid"] for fz in frames]).astype(np.float32))
    FINITE = jnp.asarray(np.stack([fz["finite"] for fz in frames]).astype(np.float32))
    owner = jnp.asarray(z["owner"], jnp.float32)
    uid = jnp.asarray(z["uid"], jnp.int32)
    pi = jnp.asarray(z["pair_i"], jnp.int32)
    pj = jnp.asarray(z["pair_j"], jnp.int32)
    lj = jnp.asarray(z["pair_lj"].astype(np.int32))
    pmask = jnp.asarray(z["pair_mask"], jnp.float32)
    nuis = jnp.asarray(scene.role == SF.ROLE_NUISANCE, jnp.float32)
    core = jnp.asarray(z["core"], jnp.float32)

    def templates(params):
        T = jnp.zeros((N, S2), jnp.float32)
        for r, idx, ctx in ctxs:
            t = L.forward_model(params, ctx)[0].reshape(idx.shape[0], S2)
            if r == SF.ROLE_NUISANCE:
                t = jax.lax.stop_gradient(t)
            T = T.at[idx].set(t)
        return T

    def solve(T, st, data, var, valid, finite):
        w = valid * st["pix_active_u"][uid] / var
        diag = jnp.sum(w * T * T, axis=1)
        bvec = jnp.sum(w * T * data, axis=1)
        Tj = jnp.take_along_axis(T[pj], lj, axis=1)
        aij = jnp.sum(pmask * w[pi] * T[pi] * Tj, axis=1)
        lam = prior_kappa * nuis * jnp.sum(finite / var * T * T, axis=1)
        diag = diag * (1.0 + ridge) + lam + 1e-20
        bvec = bvec + lam * st["f_prior"]
        f = jnp.zeros((N,), jnp.float32)
        for t in tiers:
            K, n = t["K"], t["n"]
            A = jnp.zeros((n, K, K), jnp.float32)
            A = A.at[t["row"], t["slot"], t["slot"]].add(diag[t["s"]])
            A = A.at[t["prow"], t["psi"], t["psj"]].add(aij[t["p"]])
            A = A.at[t["prow"], t["psj"], t["psi"]].add(aij[t["p"]])
            bb = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(bvec[t["s"]])
            free = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(st["star_free"][t["s"]])
            ff = jnp.zeros((n, K), jnp.float32).at[t["row"], t["slot"]].set(st["f_fixed"][t["s"]])
            fixed = 1.0 - free
            A2 = A * free[:, :, None] * free[:, None, :] + jax.vmap(jnp.diag)(fixed)
            b2 = free * (bb - jnp.einsum("nij,nj->ni", A, ff * fixed)) + fixed * ff
            x = jnp.linalg.solve(A2, b2[..., None])[..., 0]
            f = f.at[t["s"]].set(x[t["row"], t["slot"]])
        return f

    def frame_terms(params, st, blur, xs):
        wcs, data, var, valid, finite, bl = xs
        p = dict(params)
        p["wcs_coeff"] = wcs
        p["epsf_blur"] = bl * blur        # blur=0.0 switches the temporal model off
        T = templates(p)
        f = solve(T, st, data, var, valid, finite)
        mu = jnp.zeros((U + 1,), jnp.float32).at[uid].add(f[:, None] * T)[uid]
        chi = (data - mu) / jnp.sqrt(var)
        pw = valid * owner * st["pix_active_u"][uid]
        ell = 0.5 * L.huber_rho(chi, huber_delta) + 0.5 * jnp.log(var)
        cv = core[None] * valid
        chi2_core = jnp.sum(cv * chi * chi, axis=1) / jnp.clip(jnp.sum(cv, axis=1), 1.0, None)
        return jnp.sum(pw * ell), jnp.sum(pw), f, chi2_core

    def loss_fn(params, st, blurs, blur_on):
        xs = (params["wcs_coeff"], DATA, VAR, VALID, FINITE, blurs)
        num, den, _, _ = jax.lax.map(lambda x: frame_terms(params, st, blur_on, x), xs)
        data_term = jnp.sum(num) / jnp.clip(jnp.sum(den), 1.0, None)
        base = L.decoded_epsf_base(params)          # static (unblurred) ePSF for the regularisers
        lap = L.node_smoothness_penalty(base[None])
        pix_lap = L.pixel_laplacian_penalty(base)
        fine = (L.fine_neighbour_penalty(base[None]) if lambda_fine_nbr > 0 else jnp.zeros((), jnp.float32))
        loss = data_term + lambda_lap * lap + lambda_pixel * pix_lap + lambda_fine_nbr * fine
        per_frame = num / jnp.clip(den, 1.0, None)
        return loss, {"loss": loss, "data_term": data_term, "lap": lap, "pixel_lap": pix_lap,
                      "fine_nbr": fine, "per_frame": per_frame}

    def diagnose(params, st, blurs, blur_on):
        xs = (params["wcs_coeff"], DATA, VAR, VALID, FINITE, blurs)
        num, den, f, chi2 = jax.lax.map(lambda x: frame_terms(params, st, blur_on, x), xs)
        return num / jnp.clip(den, 1.0, None), f, chi2

    return loss_fn, diagnose


def blur_from_eps(eps_file: str, frame_idx: list[int], grid, ref_idx: int | None = None) -> np.ndarray:
    """eps file (Nf, NG>=3, nB) from phot_affine (--gen blur/blurdil: gens 0..2 = dSxx, dSyy, dSxy relative to the
    reference ePSF; spatial basis 1, Xn, Yn[, ...]) -> (F, n_rows, n_cols, 3) node covariances."""
    E_all = np.nan_to_num(np.load(eps_file)[:, :3])
    E = E_all[frame_idx]                                          # (F, 3, nB)
    if ref_idx is not None:                                       # blur relative to the frame the static ePSF came from
        E = E - E_all[ref_idx][None]
    X = (np.asarray(grid.node_col_ccd) - 1068.0) / 1024.0
    Y = (np.asarray(grid.node_row_ccd) - 1024.0) / 1024.0
    YY, XX = np.meshgrid(Y, X, indexing="ij")                    # (rows, cols)
    B = [np.ones_like(XX), XX, YY, XX * XX, XX * YY, YY * YY][: E.shape[-1]]
    out = np.zeros((len(frame_idx), len(Y), len(X), 3))
    for k, b in enumerate(B):
        out += E[:, None, None, :, k] * b[None, :, :, None]
    return np.nan_to_num(out).astype(np.float32)


def run(args):
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    scene_dirs = [s for s in args.scene_dirs.split(",") if s]
    frame_idx = [int(v) for v in args.frame_idx.split(",")] if args.frame_idx else list(range(len(scene_dirs)))
    scene = SF.Scene(Path(scene_dirs[0]))
    frames = []
    for sd in scene_dirs:
        zz = np.load(Path(sd) / "scene_bundle.npz")
        frames.append({k: zz[k] for k in ("data", "noise", "valid", "finite", "uid", "role", "pair_i", "pair_j", "island")})
    F = len(frames)
    if args.colour_file:
        _log(f"colour file: {scene.use_colour_file(args.colour_file)}")
    colour_ref = scene.colour_ref()
    delta2_mean = scene.delta2_mean(colour_ref)
    chroma_axis = scene.optical_axis()
    g8_extras = SF.g8_extras_tuple(args.chroma_g8_extras)
    loss_fn, diagnose = make_multi_model(
        scene, frames, colour_ref=colour_ref, huber_delta=args.huber_delta, ridge=args.ridge,
        prior_kappa=args.prior_kappa, lambda_lap=args.lambda_lap, lambda_pixel=args.lambda_pixel_lap,
        chroma_axis=chroma_axis, lambda_fine_nbr=args.lambda_fine_nbr, chroma_g8_gauge=args.chroma_g8_gauge,
        chroma_g8_no_dil=False, chroma_g8_extras=g8_extras, delta2_mean=delta2_mean)

    params = FIT.load_params_npz(Path(args.init_params_file))
    params = SF.set_chroma_model(params, "global8", halo=False, g8_extras=g8_extras, g8_source_extras=g8_extras)
    wcs0 = []
    for k, sd in enumerate(scene_dirs):
        wf = args.init_wcs_dirs.split(",")[k] if args.init_wcs_dirs else None
        wcs0.append(np.asarray(FIT.load_params_npz(Path(wf) / "params_stage3.npz")["wcs_coeff"]) if wf
                    else np.asarray(params["wcs_coeff"]))
    params["wcs_coeff"] = jnp.asarray(np.stack(wcs0))
    sz = np.load(args.init_state_file)
    st = {k: jnp.asarray(sz[k]) for k in ("pix_active_u", "star_free", "f_fixed", "f_prior")}
    blurs = (blur_from_eps(args.blur_file, frame_idx, scene.src.epsf_grid, args.blur_ref_idx) if args.blur_file
             else np.zeros((F, scene.src.epsf_grid.n_rows, scene.src.epsf_grid.n_cols, 3), np.float32))
    blurs = jnp.asarray(blurs)
    blur_on = jnp.asarray(0.0 if args.no_blur else 1.0, jnp.float32)
    _log(f"{F} frames; blur file {args.blur_file} (on={not args.no_blur}); node trace dS range "
         f"{float(jnp.min(blurs[..., 0] + blurs[..., 1])):.4f}..{float(jnp.max(blurs[..., 0] + blurs[..., 1])):.4f} px2")

    meta = dict(vars(args), n_frames=F, colour_ref=colour_ref, chroma_delta2_mean=delta2_mean,
                chroma_axis=chroma_axis, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                code_sha=os.popen(f"git -C {Path(__file__).parent} rev-parse --short HEAD").read().strip())
    (out / "fit_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    diag_j = jax.jit(diagnose)
    pf0, _, _ = diag_j(params, st, blurs, blur_on)
    _log(f"initial per-frame data terms: {np.round(np.asarray(pf0), 4).tolist()}")

    keys = tuple(params.keys())
    hist = out / "history.jsonl"
    steps = [int(s) for s in args.steps_per_stage.split(",")]
    lrs = [float(s) for s in args.lr_per_stage.split(",")]
    for stage in (1, 2, 3):
        n = steps[stage - 1]
        if n <= 0:
            continue
        labels = FIT._leaf_labels(stage, freeze_wcs=False, param_keys=keys)
        opt = FIT.make_stage_optimizer(stage, lrs[stage - 1], epsf_lr_scale=args.epsf_lr_scale, param_keys=keys,
                                       chroma_lr_scale=args.chroma_lr_scale, grad_clip=args.grad_clip)
        opt_state = opt.init(params)

        @jax.jit
        def step_fn(p, o, s, _labels=labels):
            (lv, met), g = jax.value_and_grad(
                lambda q: loss_fn(FIT.stop_grad_frozen_params(q, _labels), s, blurs, blur_on), has_aux=True)(p)
            upd, o = opt.update(g, o, p)
            return optax.apply_updates(p, upd), o, met, optax.global_norm(g)

        _log(f"=== stage {stage}: {n} steps lr {lrs[stage - 1]}")
        t0 = time.time()
        for step in range(n):
            ts = time.time()
            params, opt_state, met, gn = step_fn(params, opt_state, st)
            lv = float(met["loss"])
            if not np.isfinite(lv):
                raise FloatingPointError(f"non-finite loss stage {stage} step {step}")
            rec = {"stage": stage, "step": step, "loss": lv, "data_term": float(met["data_term"]),
                   "per_frame": np.asarray(met["per_frame"]).round(5).tolist(), "grad_norm": float(gn),
                   "step_s": time.time() - ts}
            with hist.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if step % args.log_every == 0:
                _log(f"s{stage} step {step:5d} loss {lv:.6f} data {rec['data_term']:.6f} |g| {float(gn):.3g} {rec['step_s']:.2f}s")
            if (step + 1) % args.checkpoint_every == 0:
                FIT.save_params_npz(out / "params_latest.npz", params)
        FIT.save_params_npz(out / f"params_stage{stage}.npz", params)
        _log(f"stage {stage} done in {(time.time() - t0) / 60:.1f} min")
    pf, f, chi2 = diag_j(params, st, blurs, blur_on)
    FIT.save_params_npz(out / "params.npz", params)
    np.savez(out / "flux_solved.npz", per_frame_data_term=np.asarray(pf), flux=np.asarray(f), chi2_core=np.asarray(chi2),
             frame_idx=np.asarray(frame_idx), source_id=scene.z["source_id"], role=scene.role,
             tess_mag=scene.z["tess_mag"], **{k: np.asarray(v) for k, v in st.items()})
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    _log(f"done; final per-frame data terms {np.round(np.asarray(pf), 4).tolist()}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene-dirs", required=True, help="comma list of temporal_scene dirs (same reference scene)")
    p.add_argument("--frame-idx", default="", help="comma list: each scene's row in --blur-file")
    p.add_argument("--blur-file", default=None, help="phot_affine eps npy (Nf, NG, nB), gens 0..2 = dSxx, dSyy, dSxy")
    p.add_argument("--blur-ref-idx", type=int, default=None, help="subtract this frame's blur (the static ePSF's own frame)")
    p.add_argument("--no-blur", action="store_true", help="control: static ePSF, no temporal model")
    p.add_argument("--init-params-file", required=True)
    p.add_argument("--init-state-file", required=True)
    p.add_argument("--init-wcs-dirs", default="", help="comma list of per-frame fit out dirs (wcs_coeff warm start)")
    p.add_argument("--colour-file", default=None)
    p.add_argument("--chroma-g8-gauge", default="raw")
    p.add_argument("--chroma-g8-extras", default="dil_r,sq0,sq1,q1_0,q1_x,q1_y,q2_0,q2_x,q2_y")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps-per-stage", default="200,0,1500")
    p.add_argument("--lr-per-stage", default="1e-3,3e-4,1e-4")
    p.add_argument("--epsf-lr-scale", type=float, default=2.5)
    p.add_argument("--chroma-lr-scale", type=float, default=10.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--huber-delta", type=float, default=1e6)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--prior-kappa", type=float, default=0.1)
    p.add_argument("--lambda-lap", type=float, default=1e-2)
    p.add_argument("--lambda-pixel-lap", type=float, default=1e-2)
    p.add_argument("--lambda-fine-nbr", type=float, default=1e8)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=50)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
