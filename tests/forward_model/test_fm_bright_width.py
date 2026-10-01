# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Brightness-dependent PSF width (bright_width.py, 2026-09-30).

(1) leaf absent -> loss/grad/jaxpr/templates bit-identical to dev main f70c1ed
(2) b = 0 -> same templates and loss as no leaf
(3) Gaussian calibration: --bright-width-init b reproduces DeltaSigma = b (q - q_ref)/1e4 to < 2%
(4) gradients finite, b trainable (recovers an injected b)
(5) warm start / checkpoint without the leaf -> zero-initialised
(6) q = 2 s * f * peak fraction on a toy scene, flux-source precedence, q_ref weighting
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from syndiff_pipeline.forward_model import bright_width as BW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit as FIT
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import scene_fit as SF

jax.config.update("jax_platform_name", "cpu")

REF_COMMIT = "f70c1ed"        # dev main: colour model A3 + moment-blind prior
AXIS = (-55.66, 2098.93)
A3_EXTRAS = ("dil_r", "sq0", "sq1", "q1_0", "q1_x", "q1_y", "q2_0", "q2_x", "q2_y")
A3_VALS = [0.012, -0.03, 0.004, -0.02, 0.006, -0.015, 0.02, -0.004, 0.01,
           0.003, -0.002, 0.004, 0.001, -0.002, 0.003, 0.0005, -0.001]


def _a3_ctx(ctx, gauge="raw"):
    return L.replace(ctx, chroma_axis=AXIS, chroma_g8_gauge=gauge, chroma_g8_extras=A3_EXTRAS,
                     chroma_delta2_mean=0.2)


def _a3_params(params):
    p = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
    p["chroma_g8"] = jnp.asarray(np.asarray(A3_VALS, np.float32))
    return p


def _no_colour(params):
    return {k: v for k, v in params.items() if not k.startswith("chroma")}


def _fixture():
    from test_fm_chroma import _chroma_fixture
    return _chroma_fixture(n_frames=4)


def _with_q(ctx, seed=0, q_ref=3e4):
    n = int(ctx.x_lin.shape[0])
    q = np.random.default_rng(seed).uniform(0.0, 1.2e5, n).astype(np.float32)
    return L.replace(ctx, bright_q=jnp.asarray(q), bright_q_ref=float(q_ref))


def _loss(mod, fx, params, ctx):
    loss, _ = mod.total_loss(
        params, ctx, fx["data"], fx["noise"], fx["weight"],
        fx["wcs_second_diff"], fx["w_second_diff"],
        epsf_modes_init=fx["epsf_modes_init"], stamp_active=fx["stamp_active"],
    )
    return loss


def _ref_loss_module(tmp_path):
    repo = Path(L.__file__).resolve().parents[1]
    src = subprocess.run(["git", "-C", str(repo), "show", f"{REF_COMMIT}:forward_epsf_wcs/loss.py"],
                         check=True, capture_output=True, text=True).stdout
    path = tmp_path / "loss_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("syndiff_pipeline.forward_model._loss_ref_bw", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- (1) leaf absent

@pytest.mark.parametrize("colour", ["a3", "none"])
@pytest.mark.skip(reason='dev-repo harness: git-checks-out a dev reference commit to compare against; parity is covered by the migration goldens (dev_runs/fwd_migration_20260930)')
def test_leaf_absent_bit_identical_to_main(tmp_path, colour):
    ref = _ref_loss_module(tmp_path)
    fx = _fixture()
    if colour == "a3":
        ctx, params = _a3_ctx(fx["ctx"]), _a3_params(fx["params"])
    else:
        ctx, params = fx["ctx"], _no_colour(fx["params"])
    ctx = _with_q(ctx)            # q on the context is inert without the leaf
    ref_ctx = ref.StaticContext(**{f.name: getattr(ctx, f.name) for f in fields(ref.StaticContext)})
    v_new, g_new = jax.value_and_grad(lambda p: _loss(L, fx, p, ctx))(params)
    v_ref, g_ref = jax.value_and_grad(lambda p: _loss(ref, fx, p, ref_ctx))(params)
    assert np.array_equal(np.asarray(v_new), np.asarray(v_ref))
    for k in params:
        assert np.array_equal(np.asarray(g_new[k]), np.asarray(g_ref[k])), k
    assert str(jax.make_jaxpr(lambda p: _loss(L, fx, p, ctx))(params)) == \
        str(jax.make_jaxpr(lambda p: _loss(ref, fx, p, ref_ctx))(params))
    t_new = L.forward_model(params, ctx)[0]
    t_ref = ref.forward_model(params, ref_ctx)[0]
    assert np.array_equal(np.asarray(t_new), np.asarray(t_ref))


def test_merge_is_identity_without_leaf():
    sentinel = (1, 2, 3, {"blur_raw": 4})
    assert BW.merge_slot_terms({"x": 0}, SimpleNamespace(), None, sentinel) is sentinel
    assert BW.merge_slot_terms({"x": 0}, SimpleNamespace(), None, None) is None


# ---------------------------------------------------------------- (2) b = 0

@pytest.mark.parametrize("colour", ["a3", "none"])
def test_b_zero_matches_no_leaf(colour):
    fx = _fixture()
    if colour == "a3":
        ctx, params = _a3_ctx(fx["ctx"]), _a3_params(fx["params"])
    else:
        ctx, params = fx["ctx"], _no_colour(fx["params"])
    ctx = _with_q(ctx)
    p0 = dict(params, bright_width=jnp.zeros((1,), jnp.float32))
    t_a = np.asarray(L.forward_model(params, ctx)[0])
    t_b = np.asarray(L.forward_model(p0, ctx)[0])
    np.testing.assert_allclose(t_b, t_a, rtol=0, atol=1e-6 * np.abs(t_a).max())
    np.testing.assert_allclose(float(_loss(L, fx, p0, ctx)), float(_loss(L, fx, params, ctx)), rtol=1e-6)


def test_leaf_without_q_refuses():
    fx = _fixture()
    p = dict(_no_colour(fx["params"]), bright_width=jnp.zeros((1,), jnp.float32))
    with pytest.raises(ValueError, match="bright_q"):
        L.forward_model(p, fx["ctx"])


# ---------------------------------------------------------------- (3) calibration

def _gauss_scene(sigma_px=1.1, centred=True):
    from test_fm_chroma import _scaffold
    ctx, _, static, _ = _scaffold(uniform_nodes=True, bp_rp=np.ones(3, np.float32), colour_ref=0.0)
    gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:gsz, 0:gsz] - EM.NODE_CENTER_INDEX
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (EM.OVERSAMPLE * sigma_px) ** 2)).astype(np.float32)
    base = np.broadcast_to(g / g.sum(), (2, 2, gsz, gsz)).copy()
    epsf = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.zeros((3, 2, 2, gsz, gsz), jnp.float32))
    params = L.init_params(static, epsf, n_wcs_basis=3, n_w_basis=3)
    if not centred:
        params["wcs_coeff"] = params["wcs_coeff"].at[0, :].set(3.0)
    return ctx, _no_colour(params)


def _moments(t):
    t = np.asarray(t, np.float64)
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    f = t.sum(); mx = (t * xx).sum() / f; my = (t * yy).sum() / f
    return ((t * (xx - mx) ** 2).sum() / f, (t * (yy - my) ** 2).sum() / f,
            (t * (xx - mx) * (yy - my)).sum() / f)


def _dsigma(ctx, params, b, dq, gauge="raw", colour=False):
    q_ref = 2.0e4
    n = int(ctx.x_lin.shape[0])
    ctx = L.replace(ctx, chroma_g8_gauge=gauge, bright_q=jnp.full((n,), q_ref + dq, jnp.float32),
                    bright_q_ref=q_ref)
    p = dict(params)
    if colour:   # A3 colour leaf at zero: the term must merge into its blur field identically
        ctx = _a3_ctx(ctx, gauge)
        p["chroma_g8"] = jnp.zeros((17,), jnp.float32)
    t0 = L.forward_model(p, ctx)[0][0, 0, 0]
    p["bright_width"] = jnp.asarray([b / BW.LEAF_UNIT], jnp.float32)
    t1 = L.forward_model(p, ctx)[0][0, 0, 0]
    m0, m1 = _moments(t0), _moments(t1)
    return m1[0] - m0[0], m1[1] - m0[1], m1[2] - m0[2]


@pytest.mark.parametrize("gauge", ["raw", "mean"])
@pytest.mark.parametrize("colour", [False, True])
def test_gaussian_calibration(colour, gauge):
    """Rendered stamp, centred Gaussian. gauge = the colour gauge; the brightness term always
    rides on blur_raw, so it is calibrated under either (the A3 default is raw)."""
    ctx, params = _gauss_scene()
    b, dq = 0.8e-3, 1.0e5                     # bfwidth law, q - q_ref = 1e5 e- -> DeltaSigma 8e-3 px^2
    want = b * dq / BW.Q_UNIT
    dxx, dyy, dxy = _dsigma(ctx, params, b, dq, gauge=gauge, colour=colour)
    assert abs(dxx / want - 1) < 0.02, dxx / want
    assert abs(dyy / want - 1) < 0.02, dyy / want
    assert abs(dxy) < 0.01 * want
    # linear in (q - q_ref), sign included
    mxx, myy, _ = _dsigma(ctx, params, b, -dq, gauge=gauge, colour=colour)
    assert abs(mxx / -want - 1) < 0.02 and abs(myy / -want - 1) < 0.02


@pytest.mark.parametrize("sigma", [0.8, 1.1, 1.5])
def test_generator_conversion_on_node_grid(sigma):
    """DeltaSigma = 2 w for P + w * blur_raw[P] on the node grid (GEN_PER_DSIGMA = 0.5)."""
    gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:gsz, 0:gsz] - EM.NODE_CENTER_INDEX
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (EM.OVERSAMPLE * sigma) ** 2)); g /= g.sum()
    c = np.asarray(EM.node_coord_1d(n_grid=gsz), np.float64)

    def sxx(a):
        f = a.sum(); m = (a * c[None, :]).sum() / f
        return (a * (c[None, :] - m) ** 2).sum() / f

    w = 0.5 * 8e-3
    for fn in (EM.blur_generator, EM.chroma_blur_field_raw):
        G = np.asarray(fn(jnp.asarray(g[None, None], jnp.float32)), np.float64)[0, 0]
        assert abs((sxx(g + w * G) - sxx(g)) / (w / BW.GEN_PER_DSIGMA) - 1) < 0.01


def test_mean_gauge_blur_is_not_a_calibrated_blur():
    """Why the term never uses the mean-gauge 'blur' generator."""
    ctx, params = _gauss_scene(0.8)
    ctx = L.replace(ctx, chroma_g8_gauge="mean", chroma_axis=AXIS, bright_q=None)
    assert BW.blur_field_name(ctx) == "blur_raw"


# ---------------------------------------------------------------- (4) gradients, trainable

def test_gradient_finite_and_nonzero():
    fx = _fixture()
    ctx = _with_q(_a3_ctx(fx["ctx"]))
    p = dict(_a3_params(fx["params"]), bright_width=jnp.asarray([0.8], jnp.float32))
    g = jax.grad(lambda q: _loss(L, fx, q, ctx))(p)
    for k, v in g.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    assert float(np.abs(np.asarray(g["bright_width"])).max()) > 0


def test_b_trainable_recovers_injected_value():
    ctx, params = _gauss_scene()
    ctx = L.replace(ctx, chroma_g8_gauge="raw",
                    bright_q=jnp.asarray([1.5e5, 2.0e4, 8.0e4], jnp.float32), bright_q_ref=2.0e4)
    truth = np.asarray(L.forward_model(dict(params, bright_width=jnp.asarray([0.8])), ctx)[0])
    scale = 1.0 / float(np.abs(truth).max())

    def obj(leaf):
        t = L.forward_model(dict(params, bright_width=leaf), ctx)[0]
        return jnp.sum(((t - truth) * scale) ** 2)

    opt = optax.adam(0.05)
    leaf = jnp.zeros((1,), jnp.float32)
    st = opt.init(leaf)
    step = jax.jit(lambda lf, s: (lambda g: opt.update(g, s, lf))(jax.grad(obj)(lf)))
    for _ in range(150):
        upd, st = step(leaf, st)
        leaf = optax.apply_updates(leaf, upd)
    assert abs(float(leaf[0]) - 0.8) < 0.02, float(leaf[0])


def test_leaf_trains_at_stage3_in_chroma_bucket():
    keys = FIT.STAGE_LEAVES + ("chroma_g8", "bright_width")
    for stage in (1, 2):
        assert FIT._leaf_labels(stage, freeze_wcs=False, param_keys=keys)["bright_width"] == "frozen"
    assert FIT._leaf_labels(3, freeze_wcs=False, param_keys=keys)["bright_width"] == "train_chroma"


# ---------------------------------------------------------------- (5) warm start

def test_set_bright_width_init_rules():
    base = {"epsf_base_raw": jnp.zeros((2, 2, 3, 3))}
    p, src = BW.set_bright_width(base, "lin", None)
    assert src == "zero" and np.array_equal(np.asarray(p["bright_width"]), [0.0])
    p2, src = BW.set_bright_width(dict(base, bright_width=jnp.asarray([0.7])), "lin", None)
    assert src == "warm_start" and np.allclose(np.asarray(p2["bright_width"]), [0.7])
    p3, src = BW.set_bright_width(dict(base, bright_width=jnp.asarray([0.7])), "lin", 0.8)
    assert src == "cli" and np.allclose(np.asarray(p3["bright_width"]), [0.8])
    p4, src = BW.set_bright_width(dict(base, bright_width=jnp.asarray([0.7])), "none", None)
    assert "bright_width" not in p4


def test_checkpoint_roundtrip_and_old_checkpoint_zero_init(tmp_path):
    fx = _fixture()
    params = _a3_params(fx["params"])
    FIT.save_params_npz(tmp_path / "old.npz", params)
    old = FIT.load_params_npz(tmp_path / "old.npz")
    assert "bright_width" not in old
    p, src = BW.set_bright_width(old, "lin", None)
    assert src == "zero" and float(p["bright_width"][0]) == 0.0
    p["bright_width"] = jnp.asarray([0.63], jnp.float32)
    FIT.save_params_npz(tmp_path / "new.npz", p)
    back = FIT.load_params_npz(tmp_path / "new.npz")
    assert np.allclose(np.asarray(back["bright_width"]), [0.63])


def test_cli_units():
    a = SF.build_parser().parse_args(["--scene-dir", "x", "--out-dir", "y", "--bright-width", "lin",
                                      "--bright-width-init", "0.8e-3", "--no-bright-width-train"])
    assert a.bright_width == "lin" and a.bright_width_train is False
    assert np.isclose(a.bright_width_init / BW.LEAF_UNIT, 0.8)
    d = SF.build_parser().parse_args(["--scene-dir", "x", "--out-dir", "y"])
    SF.resolve_g8_defaults(d, Path("/nonexistent"))
    assert d.bright_width == "none" and d.bright_width_train is True


# ---------------------------------------------------------------- (6) q on a toy scene

def _toy_scene(n=5):
    S = 5
    core = np.zeros((S, S), bool); core[1:4, 1:4] = True
    rng = np.random.default_rng(1)
    return SimpleNamespace(
        S=S, N=n, role=np.array([0, 0, 1, 2, 0][:n]),
        z={"tess_mag": np.array([8.0, 9.0, 10.0, 11.0, 12.0][:n]),
           "tess_flux": 10 ** (-0.4 * np.array([8.0, 9.0, 10.0, 11.0, 12.0][:n])),
           "source_id": np.arange(100, 100 + n), "core": core.ravel(),
           "noise": np.abs(rng.normal(1.0, 0.1, (n, S * S)))},
    )


def _toy_render(n=5, S=5):
    T = np.full((n, S * S), 0.01, np.float32)
    peaks = np.array([0.30, 0.25, 0.40, 0.20, 0.35][:n], np.float32)
    T[np.arange(n), 12] = peaks
    f = np.array([2.0e5, 8.0e4, 3.0e4, 1.0e4, 5.0e3][:n], np.float32)
    return (lambda p, st: (jnp.asarray(T), jnp.asarray(f))), peaks, f


def _qargs(**kw):
    return SimpleNamespace(**dict(dict(init_from=None, init_params_file=None), **kw))


def test_q_from_warm_start_solve(tmp_path):
    scene = _toy_scene()
    render, peaks, f = _toy_render()
    st = {"f_prior": jnp.zeros(5)}
    q, flux, peak, meta = SF.bright_width_q(scene, {"a": jnp.zeros(1)}, st, render,
                                            _qargs(init_params_file=str(tmp_path / "p.npz")))
    np.testing.assert_allclose(peak, peaks, rtol=1e-6)
    np.testing.assert_allclose(q, 2.0 * f * peaks, rtol=1e-6)
    assert meta["q_source"] == "warm_start_initial_solve"


def test_q_prefers_flux_solved_and_falls_back_per_star(tmp_path):
    scene = _toy_scene()
    render, peaks, _ = _toy_render()
    fs = np.array([1e5, np.nan, 2e4, -50.0, 1e3])
    np.savez(tmp_path / "flux_solved.npz", flux=fs, source_id=scene.z["source_id"])
    q, flux, _, meta = SF.bright_width_q(scene, {}, {}, render, _qargs(init_from=str(tmp_path)))
    assert meta["q_source"].startswith("flux_solved:") and meta["n_catalogue_fallback"] == 1
    cat = BW.catalogue_flux(scene.z["tess_mag"])
    want = 2.0 * np.array([1e5, cat[1], 2e4, 0.0, 1e3]) * peaks
    np.testing.assert_allclose(q, want, rtol=1e-6)


def test_q_cap():
    scene = _toy_scene()
    render, peaks, f = _toy_render()
    q, _, _, meta = SF.bright_width_q(scene, {}, {"f_prior": jnp.zeros(5)}, render,
                                      _qargs(init_params_file="x/p.npz", bright_width_qmax=3e4))
    np.testing.assert_allclose(q, np.minimum(2.0 * f * peaks, 3e4), rtol=1e-6)
    assert meta["n_capped"] == 2 and meta["q_cap"] == 3e4


def test_q_catalogue_without_warm_start():
    scene = _toy_scene()
    render, peaks, _ = _toy_render()
    q, _, _, meta = SF.bright_width_q(scene, {}, {}, render, _qargs())
    assert meta["q_source"] == "catalogue"
    np.testing.assert_allclose(q[2], 2.0 * 15000.0 * peaks[2], rtol=1e-6)   # Tmag 10


def test_q_ref_is_weighted_mean_over_contributors():
    scene = _toy_scene()
    q = np.array([4e4, 2e4, 9e9, 9e9, 1e3])
    flux = np.array([2e5, 8e4, 1.0, 1.0, 5e3])
    got = SF.bright_width_q_ref(scene, q, flux)
    s2 = np.median(scene.z["noise"][:, scene.z["core"]] ** 2, axis=1)
    w = flux ** 2 / s2
    m = scene.role == 0
    assert np.isclose(got, np.sum(w[m] * q[m]) / np.sum(w[m]))


def test_slot_weight_indexes_stars_by_occupied_slot():
    ctx = SimpleNamespace(bright_q=jnp.asarray([1e4, 5e4, 9e4], jnp.float32), bright_q_ref=3e4,
                          chroma_g8_gauge="raw")
    occ = jnp.asarray([2, 0, 1, 2])
    p = {"bright_width": jnp.asarray([0.8], jnp.float32)}
    w = np.asarray(BW.slot_weight(p, ctx, occ))
    np.testing.assert_allclose(w, 0.5 * 0.8e-3 * (np.array([9e4, 1e4, 5e4, 9e4]) - 3e4) / 1e4, rtol=1e-6)
    d, sx, sy, fl = BW.merge_slot_terms(p, ctx, occ, None)
    assert set(fl) == {"blur_raw"} and not np.any(np.asarray(sx)) and not np.any(np.asarray(d))
    d, sx, sy, fl = BW.merge_slot_terms(p, ctx, occ, ("d", "x", "y", {"blur_raw": jnp.ones(4)}))
    np.testing.assert_allclose(np.asarray(fl["blur_raw"]), 1.0 + w, rtol=1e-6)
