# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""scene_fit Chebyshev background (--bg-cheb-order), ported 2026-09-30 from scene-bg-cheb 5855d64.

- basis/terms
- island_solve (Schur complement) == an independent float64 dense solve of the joint normal equations
- off path: loss + fluxes bit-identical to dev main f70c1ed on a real scene (skipped without it)
- background + brightness width coexist in a real scene_fit run (skipped without the scene)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import scene_fit as SF

jax.config.update("jax_platform_name", "cpu")

REF_COMMIT = "f70c1ed"
F2_SCENE = Path("/home/kshukawa/syndiff_pipeline/dev/forward_epsf_wcs/output/runs/"
                "sffi_colour_20260924/scenes/F2_s20c3k3")
F2_WARM = Path("/astro/armin/koji/syndiff/dev_runs/sffi_rawgauge_20260924/runs/F2_D1_raw_dilR/out")
need_f2 = pytest.mark.skipif(not (F2_SCENE.exists() and F2_WARM.exists()), reason="F2 scene not available")


def test_bg_terms_and_basis():
    assert SF.bg_terms(0) == [(0, 0)]
    assert len(SF.bg_terms(2)) == 6 and len(SF.bg_terms(3)) == 10
    x, y = np.array([0.0, 1024.0, 1536.0]), np.array([2048.0, 1024.0, 512.0])
    B = SF.bg_basis(x, y, 2)
    u, v = (x - 1024) / 1024, (y - 1024) / 1024
    want = {(0, 0): 1 + 0 * u, (1, 0): u, (0, 1): v, (2, 0): 2 * u * u - 1, (1, 1): u * v, (0, 2): 2 * v * v - 1}
    for m, ij in enumerate(SF.bg_terms(2)):
        np.testing.assert_allclose(B[:, m], want[ij], atol=1e-12)


# ---------------------------------------------------------------- toy scene for the solver

def _toy(order, S=5, seed=0):
    rng = np.random.default_rng(seed)
    # an overlapping pair (0, 1) + isolated stars spread over the CCD (star 3 rejected/fixed)
    iso = [[1500, 400], [900, 1800], [1800, 1700], [300, 1000], [1000, 1000], [1700, 1000],
           [300, 1900], [1100, 200], [600, 600], [1400, 1400]]
    centres = np.array([[200, 300], [202, 301]] + iso)
    N, h = len(centres), S // 2
    off = np.arange(S) - h
    gx = centres[:, 0][:, None, None] + off[None, None, :]
    gy = centres[:, 1][:, None, None] + off[None, :, None]
    key = (gy * 4096 + gx).reshape(N, S * S)
    uniq, inv = np.unique(key, return_inverse=True)
    uid = inv.reshape(N, S * S).astype(np.int32)
    U = uniq.size
    owner = np.zeros(N * S * S, bool)
    _, first = np.unique(inv, return_index=True)
    owner[first] = True
    owner = owner.reshape(N, S * S)
    ux, uy = uniq % 4096, uniq // 4096
    # templates: Gaussians at sub-pixel offsets
    T = np.zeros((N, S * S))
    for i in range(N):
        dx, dy = rng.uniform(-0.4, 0.4, 2) + (centres[i] - centres[i])
        xx, yy = np.meshgrid(off, off)
        g = np.exp(-((xx - dx) ** 2 + (yy - dy) ** 2) / (2 * 0.9 ** 2))
        T[i] = (g / g.sum()).ravel()
    f_true = rng.uniform(5e3, 5e4, N)
    c_true = rng.normal(0, 3.0, len(SF.bg_terms(max(order, 0)))) + np.r_[20.0, np.zeros(len(SF.bg_terms(max(order, 0))) - 1)]
    phi_u = SF.bg_basis(ux, uy, max(order, 0))                  # (U, M)
    sky = phi_u @ c_true if order >= 0 else np.zeros(U)
    mu = sky.copy()
    for i in range(N):
        np.add.at(mu, uid[i], f_true[i] * T[i])
    var_u = 4.0 + 0.01 * mu
    data_u = mu + rng.normal(0, np.sqrt(var_u))
    # pairs: 0-1 overlap
    pi, pj = np.array([0]), np.array([1])
    lj = np.zeros((1, S * S), np.int64); pm = np.zeros((1, S * S))
    pos1 = {u: k for k, u in enumerate(uid[1])}
    for k, u in enumerate(uid[0]):
        if u in pos1:
            lj[0, k] = pos1[u]; pm[0, k] = 1.0
    n_iso = N - 2
    z = {"tiers": np.array([1, 2]), "star_tier": np.r_[1, 1, np.zeros(n_iso, int)],
         "star_row": np.r_[0, 0, np.arange(n_iso)], "star_slot": np.zeros(N, int),
         "pair_i": pi, "pair_j": pj,
         "tier1_star_idx": np.arange(2, N)[:, None], "tier2_star_idx": np.array([[0, 1]])}
    z["star_slot"][1] = 1
    tiers = SF.Scene.tier_tables(SimpleNamespace(z=z))
    a = dict(data=jnp.asarray(data_u[uid], jnp.float32), var=jnp.asarray(var_u[uid], jnp.float32),
             valid=jnp.ones((N, S * S), jnp.float32), finite=jnp.ones((N, S * S), jnp.float32),
             owner=jnp.asarray(owner, jnp.float32), uid=jnp.asarray(uid), pi=jnp.asarray(pi, jnp.int32),
             pj=jnp.asarray(pj, jnp.int32), lj=jnp.asarray(lj, jnp.int32), pmask=jnp.asarray(pm, jnp.float32),
             nuis=jnp.zeros(N, jnp.float32))
    free = np.ones(N, np.float32); free[3] = 0.0              # star 3 rejected: flux held fixed
    f_fixed = np.where(free > 0, 0.0, f_true * 1.01)
    pix = np.ones(U + 1, np.float32); pix[U] = 0.0
    st = {"pix_active_u": jnp.asarray(pix), "star_free": jnp.asarray(free),
          "f_fixed": jnp.asarray(f_fixed, jnp.float32), "f_prior": jnp.zeros(N, jnp.float32)}
    phi = jnp.asarray(SF.bg_basis(ux[uid], uy[uid], order), jnp.float32) if order >= 0 else None
    ref = dict(T=T, uid=uid, U=U, data_u=data_u, var_u=var_u, phi_u=phi_u, free=free, f_fixed=f_fixed,
               z=z, S=S, ux=ux, uy=uy, centres=centres)
    return jnp.asarray(T, jnp.float32), st, a, tiers, phi, ref


def _dense_reference(ref, order):
    """float64 weighted least squares over union pixels: unknowns = free fluxes (+ c)."""
    T, uid, U = ref["T"], ref["uid"], ref["U"]
    N = T.shape[0]
    cols = []
    for i in range(N):
        col = np.zeros(U); np.add.at(col, uid[i], T[i]); cols.append(col)
    X = np.stack(cols, 1)                                     # (U, N)
    y = ref["data_u"] - X[:, ref["free"] == 0] @ ref["f_fixed"][ref["free"] == 0]
    D = X[:, ref["free"] > 0]
    if order >= 0:
        D = np.hstack([D, ref["phi_u"]])
    w = 1.0 / ref["var_u"]
    sol = np.linalg.solve(D.T @ (w[:, None] * D), D.T @ (w * y))
    nf = int((ref["free"] > 0).sum())
    f = ref["f_fixed"].copy(); f[ref["free"] > 0] = sol[:nf]
    return f, (sol[nf:] if order >= 0 else None)


@pytest.mark.parametrize("order", [-1, 0, 2])
def test_island_solve_matches_float64_dense(order):
    T, st, a, tiers, phi, ref = _toy(order)
    f, c, b = jax.jit(lambda T, st: SF.island_solve(T, st, a, tiers, ridge=0.0, prior_kappa=0.0, phi=phi))(T, st)
    assert b is None
    f_ref, c_ref = _dense_reference(ref, order)
    np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-4)
    if order < 0:
        assert c is None
    else:
        np.testing.assert_allclose(np.asarray(c), c_ref, rtol=0, atol=5e-3)
    assert float(f[3]) == pytest.approx(float(ref["f_fixed"][3]), rel=1e-6)   # fixed star untouched


def test_island_solve_gradient_through_background():
    T, st, a, tiers, phi, _ = _toy(2)
    g = jax.grad(lambda T: jnp.sum(SF.island_solve(T, st, a, tiers, ridge=0.0, prior_kappa=0.0,
                                                    phi=phi)[1] ** 2))(T)
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0


def test_resume_keeps_bg_order(tmp_path):
    (tmp_path / "fit_meta.json").write_text(json.dumps({"bg_cheb_order": 2}))
    a = SF.build_parser().parse_args(["--scene-dir", "x", "--out-dir", str(tmp_path), "--resume"])
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.bg_cheb_order == 2
    b = SF.build_parser().parse_args(["--scene-dir", "x", "--out-dir", str(tmp_path)])
    SF.resolve_g8_defaults(b, tmp_path)
    assert b.bg_cheb_order == -1


# ---------------------------------------------------------------- real scene (skipped without it)

def _ref_scene_fit_module(tmp_path):
    repo = Path(SF.__file__).resolve().parents[1]
    src = subprocess.run(["git", "-C", str(repo), "show", f"{REF_COMMIT}:forward_epsf_wcs/scene_fit.py"],
                         check=True, capture_output=True, text=True).stdout
    path = tmp_path / "scene_fit_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("syndiff_pipeline.forward_model._scene_fit_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _f2_setup():
    from syndiff_pipeline.forward_model import fit as FIT
    scene = SF.Scene(F2_SCENE)
    params = SF.set_chroma_model(FIT.load_params_npz(F2_WARM / "params.npz"), "global8", halo=False,
                                 g8_extras=("dil_r",), g8_source_extras=("dil_r",))
    sz = np.load(F2_WARM / "state_stage3.npz")
    st = {k: jnp.asarray(sz[k]) for k in ("pix_active_u", "star_free", "f_fixed", "f_prior")}
    kw = dict(colour_ref=scene.colour_ref(), huber_delta=1e6, ridge=1e-6, prior_kappa=0.1, lambda_lap=1e-2,
              lambda_pixel=1e-2, chroma_axis=(2102.73, 2106.62), lambda_fine_nbr=1e8,
              chroma_g8_gauge="raw", chroma_g8_extras=("dil_r",))
    return scene, params, st, kw


@need_f2
@pytest.mark.skip(reason='dev-repo harness: git-checks-out a dev reference commit to compare against; parity is covered by the migration goldens (dev_runs/fwd_migration_20260930)')
def test_bg_off_bit_identical_to_main_on_real_scene(tmp_path):
    ref = _ref_scene_fit_module(tmp_path)
    scene, params, st, kw = _f2_setup()
    l_new, d_new = SF.make_model(scene, **kw)
    l_ref, d_ref = ref.make_model(scene, **kw)
    v_new, a_new = jax.jit(l_new)(params, st)
    v_ref, a_ref = jax.jit(l_ref)(params, st)
    assert np.array_equal(np.asarray(v_new), np.asarray(v_ref))
    assert "bg_coef" not in a_new
    f_new = np.asarray(jax.jit(d_new)(params, st)[0])
    f_ref = np.asarray(jax.jit(d_ref)(params, st)[0])
    assert np.array_equal(f_new, f_ref)


@need_f2
@pytest.mark.parametrize("stamp_bg", ["none", "annulus", "fit"])
def test_bright_width_and_background_coexist(tmp_path, stamp_bg):
    out = tmp_path / "run"
    SF.main(["--scene-dir", str(F2_SCENE), "--out-dir", str(out),
             "--init-params-file", str(F2_WARM / "params.npz"),
             "--init-state-file", str(F2_WARM / "state_stage3.npz"),
             "--colour-file=", "--steps-per-stage", "0,0,2",
             "--bright-width", "lin", "--bright-width-init", "0.8e-3",
             "--bg-cheb-order", "2", "--stamp-bg", stamp_bg])
    meta = json.loads((out / "fit_meta.json").read_text())
    assert meta["stamp_bg"] == stamp_bg and meta["stamp_bg_model"]["mode"] == stamp_bg
    if stamp_bg != "none":
        assert meta["stamp_bg_model"]["n_estimated"] > 0.5 * meta["stamp_bg_model"]["n_stamps"]
    if stamp_bg == "fit":
        assert np.isfinite(meta["stamp_bg_model"]["fit_median"])
        assert "stamp_bg" in np.load(out / "flux_solved.npz").files
    # the fit can be rebuilt from fit_meta.json alone (the OOF renderer's contract)
    import inspect
    from syndiff_pipeline.forward_model import fit as FIT
    sig = inspect.signature(SF.make_model).parameters
    assert all(k in meta for k in ("bright_width", "bright_width_q_file", "bright_width_q_ref",
                                   "bright_width_gen_per_dsigma", "bg_cheb_order", "stamp_bg",
                                   "stamp_bg_prior_sigma"))
    assert all(k in sig for k in ("bright_width", "bright_width_q_file", "bright_width_q_ref",
                                  "bright_width_gen_per_dsigma", "bg_cheb_order", "stamp_bg"))
    p = FIT.load_params_npz(out / "params.npz")
    assert "bright_width" in p
    assert meta["bg_cheb_order"] == 2 and len(meta["bg_terms"]) == 6
    assert len(meta["bg_coef_final"]) == 6 and np.all(np.isfinite(meta["bg_coef_final"]))
    bw = meta["bright_width_model"]
    assert bw["form"] == "lin" and np.isfinite(bw["b_final"]) and bw["generator"] == "blur_raw"
    recs = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    steps = [r for r in recs if "loss" in r]
    assert steps and all("bg_coef" in r and "bright_width_b" in r for r in steps)
    assert all(np.isfinite(r["loss"]) for r in steps)
    assert "bg_coef" in np.load(out / "flux_solved.npz").files
