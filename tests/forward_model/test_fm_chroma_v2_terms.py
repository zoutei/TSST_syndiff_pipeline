# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Colour model v2 chroma_g8 extras (2026-09-28): sq0/sq1, q1_*/q2_* elongation planes, dilp_x/dilp_y.

Contract: forward_epsf_wcs/docs/CHROMA_V2_CONTRACT.md (loss.py section).
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
import pytest

from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import loss as L

jax.config.update("jax_platform_name", "cpu")

AXIS = (-55.66, 2098.93)            # CCD2/4-like optical axis, sci px
REF_COMMIT = "bb232c8"              # dev main at merge: colour model #14 + moment-blind fine-nbr default
NEW = ("sq0", "sq1", "q1_0", "q1_x", "q1_y", "q2_0", "q2_x", "q2_y", "dilp_x", "dilp_y")
C14 = [0.012, -0.03, 0.004, -0.02, 0.006, -0.015, 0.02, -0.004, 0.01]   # 8 base + dil_r


def _g8_ctx(ctx, extras, delta2_mean=0.0):
    return L.replace(ctx, chroma_axis=AXIS, chroma_g8_gauge="raw", chroma_g8_extras=tuple(extras),
                     chroma_delta2_mean=float(delta2_mean))


def _g8_params(params, values):
    p = {k: v for k, v in params.items() if k not in L.CHROMA_LEAVES}
    p["chroma_g8"] = jnp.asarray(np.asarray(values, np.float32))
    return p


def _fixture(n_frames=4):
    from test_fm_chroma import _chroma_fixture
    return _chroma_fixture(n_frames=n_frames)


def _loss(mod, fx, params, ctx):
    loss, _ = mod.total_loss(
        params, ctx, fx["data"], fx["noise"], fx["weight"],
        fx["wcs_second_diff"], fx["w_second_diff"],
        epsf_modes_init=fx["epsf_modes_init"], stamp_active=fx["stamp_active"],
    )
    return loss


def _ref_loss_module(tmp_path):
    """loss.py as of REF_COMMIT, imported inside the package so its relative imports resolve."""
    repo = Path(L.__file__).resolve().parents[1]
    src = subprocess.run(["git", "-C", str(repo), "show", f"{REF_COMMIT}:forward_epsf_wcs/loss.py"],
                         check=True, capture_output=True, text=True).stdout
    path = tmp_path / "loss_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("syndiff_pipeline.forward_model._loss_ref_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclass() resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. No-op
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason='dev-repo harness: git-checks-out a dev reference commit to compare against; parity is covered by the migration goldens (dev_runs/fwd_migration_20260930)')
def test_c14_loss_and_grad_bit_identical_to_ref_commit(tmp_path):
    ref = _ref_loss_module(tmp_path)
    fx = _fixture()
    ctx = _g8_ctx(fx["ctx"], ("dil_r",), delta2_mean=0.37)  # delta2_mean must be inert without sq*
    params = _g8_params(fx["params"], C14)
    ref_ctx = ref.StaticContext(**{f.name: getattr(ctx, f.name) for f in fields(ref.StaticContext)})

    v_new, g_new = jax.value_and_grad(lambda p: _loss(L, fx, p, ctx))(params)
    v_ref, g_ref = jax.value_and_grad(lambda p: _loss(ref, fx, p, ref_ctx))(params)
    assert np.array_equal(np.asarray(v_new), np.asarray(v_ref))
    for k in params:
        assert np.array_equal(np.asarray(g_new[k]), np.asarray(g_ref[k])), k
    # and the traced program itself is unchanged (no extra ops, not merely equal numbers)
    j_new = jax.make_jaxpr(lambda p: _loss(L, fx, p, ctx))(params)
    j_ref = jax.make_jaxpr(lambda p: _loss(ref, fx, p, ref_ctx))(params)
    assert str(j_new) == str(j_ref)


def test_new_extras_at_zero_match_c14():
    fx = _fixture()
    base = _loss(L, fx, _g8_params(fx["params"], C14), _g8_ctx(fx["ctx"], ("dil_r",)))
    ctx = _g8_ctx(fx["ctx"], ("dil_r",) + NEW, delta2_mean=0.37)
    got = _loss(L, fx, _g8_params(fx["params"], C14 + [0.0] * len(NEW)), ctx)
    np.testing.assert_allclose(float(got), float(base), rtol=1e-6)


def test_absent_extras_create_no_fields():
    ctx = _fake_ctx(); occ = jnp.arange(ctx.x_lin.shape[0])
    f = L._chroma_g8_slot_terms({"chroma_g8": jnp.asarray(C14, jnp.float32)}, ctx, occ)[3]
    assert "aniso_raw" not in f and "shear_raw" not in f
    ctx.chroma_g8_extras = ("dil_r", "q2_0")
    f = L._chroma_g8_slot_terms({"chroma_g8": jnp.asarray(C14 + [0.01], jnp.float32)}, ctx, occ)[3]
    assert "shear_raw" in f and "aniso_raw" not in f


# ---------------------------------------------------------------------------
# slot-level algebra
# ---------------------------------------------------------------------------

def _fake_ctx(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        chroma_delta=jnp.asarray(rng.normal(0.0, 0.6, n), jnp.float32),
        chroma_axis=AXIS,
        x_lin=jnp.asarray(rng.uniform(0, 2048, n), jnp.float32),
        y_lin=jnp.asarray(rng.uniform(0, 2048, n), jnp.float32),
        chroma_g8_gauge="raw", chroma_g8_no_dil=False, chroma_g8_extras=("dil_r",),
        chroma_delta2_mean=0.0,
    )


def _vec(extras, **vals):
    return jnp.asarray(C14 + [vals.get(n, 0.0) for n in extras[1:]], jnp.float32)


def test_planes_follow_the_contract_formulas():
    ctx = _fake_ctx(); occ = jnp.arange(200)
    ctx.chroma_g8_extras = ("dil_r", "astig0") + NEW
    v = dict(astig0=0.004, q1_0=0.01, q1_x=-0.02, q1_y=0.03, q2_0=-0.005, q2_x=0.015, q2_y=0.007,
             dilp_x=0.011, dilp_y=-0.013)
    f = L._chroma_g8_slot_terms({"chroma_g8": _vec(ctx.chroma_g8_extras, **v)}, ctx, occ)[3]
    d = np.asarray(ctx.chroma_delta)
    x, y = np.asarray(ctx.x_lin), np.asarray(ctx.y_lin)
    X, Y = (x - 1024) / 1024, (y - 1024) / 1024
    vx, vy = AXIS[0] - x, AXIS[1] - y
    r, phi = np.hypot(vx, vy) / 1000, np.arctan2(vy, vx)
    tol = dict(rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(f["aniso_raw"], d * (0.004 * np.cos(2 * phi) + 0.01 - 0.02 * X + 0.03 * Y), **tol)
    np.testing.assert_allclose(f["shear_raw"], d * (0.004 * np.sin(2 * phi) - 0.005 + 0.015 * X + 0.007 * Y), **tol)
    np.testing.assert_allclose(f["dilation_raw"], d * (C14[6] + C14[8] * r + 0.011 * X - 0.013 * Y), **tol)


def test_dilp_dropped_with_no_dil():
    ctx = _fake_ctx(); occ = jnp.arange(200)
    ctx.chroma_g8_extras = ("dil_r", "dilp_x", "dilp_y"); ctx.chroma_g8_no_dil = True
    f = L._chroma_g8_slot_terms({"chroma_g8": _vec(ctx.chroma_g8_extras, dilp_x=0.01)}, ctx, occ)[3]
    assert "dilation_raw" not in f


# ---------------------------------------------------------------------------
# 3. <delta^2> centring
# ---------------------------------------------------------------------------

def test_delta2_mean_centres_the_quadratic_shift():
    ctx = _fake_ctx(); occ = jnp.arange(200)
    ctx.chroma_g8_extras = ("dil_r", "sq0", "sq1")
    c = jnp.asarray([0.0] * 9 + [0.02, 0.0], jnp.float32)   # sq0 only, s(r) = 0
    d = np.asarray(ctx.chroma_delta, np.float64)
    vx, vy = AXIS[0] - np.asarray(ctx.x_lin), AXIS[1] - np.asarray(ctx.y_lin)
    rr = np.hypot(vx, vy)

    def radial(m):
        ctx.chroma_delta2_mean = m
        _, sx, sy, _ = L._chroma_g8_slot_terms({"chroma_g8": c}, ctx, occ)
        return (np.asarray(sx) * vx + np.asarray(sy) * vy) / rr   # shift along n_axis

    m2 = float(np.mean(d * d))
    s = radial(m2)
    np.testing.assert_allclose(s, 0.02 * (d * d - m2), rtol=1e-4, atol=1e-7)
    assert abs(s.mean()) < 1e-6 * np.abs(s).max() + 1e-8
    assert abs(radial(0.0).mean()) > 0.01 * m2           # uncentred it carries a mean shift


# ---------------------------------------------------------------------------
# 2. Generator widths on a Gaussian (full render)
# ---------------------------------------------------------------------------

def _gauss_scene(sigma_px=1.1):
    from test_fm_chroma import _scaffold
    ctx, _, static, _ = _scaffold(uniform_nodes=True, bp_rp=np.ones(3, np.float32), colour_ref=0.0)
    gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:gsz, 0:gsz] - EM.NODE_CENTER_INDEX
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (EM.OVERSAMPLE * sigma_px) ** 2)).astype(np.float32)
    base = np.broadcast_to(g / g.sum(), (2, 2, gsz, gsz)).copy()
    epsf = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.zeros((3, 2, 2, gsz, gsz), jnp.float32))
    params = L.init_params(static, epsf, n_wcs_basis=3, n_w_basis=3)
    params["wcs_coeff"] = params["wcs_coeff"].at[0, :].set(3.0)
    return ctx, params


def _moments(ctx, params, extras, **vals):
    ctx = _g8_ctx(ctx, ("dil_r",) + extras)
    p = dict(params); p["chroma_g8"] = jnp.asarray([0.0] * 9 + [vals.get(n, 0.0) for n in extras], jnp.float32)
    t = np.asarray(L.forward_model(p, ctx)[0][0, 0, 0], np.float64)   # star 0, frame 0
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    f = t.sum(); mx = (t * xx).sum() / f; my = (t * yy).sum() / f
    return ((t * (xx - mx) ** 2).sum() / f, (t * (yy - my) ** 2).sum() / f, (t * (xx - mx) * (yy - my)).sum() / f)


def test_q1_is_an_x_y_elongation():
    ctx, params = _gauss_scene()
    sxx0, syy0, sxy0 = _moments(ctx, params, ())
    e0 = sxx0 - syy0
    e = {}
    for sgn in (1.0, -1.0):
        sxx, syy, sxy = _moments(ctx, params, ("q1_0",), q1_0=0.05 * sgn)
        e[sgn] = sxx - syy - e0
        assert abs(sxy - sxy0) < 0.05 * abs(e[sgn])
        assert abs((sxx + syy) - (sxx0 + syy0)) < 0.1 * abs(e[sgn])
    assert abs(e[1.0]) > 1e-3 and np.sign(e[1.0]) == -np.sign(e[-1.0])
    np.testing.assert_allclose(-e[-1.0], e[1.0], rtol=0.1)


def test_q2_is_a_45_degree_elongation():
    ctx, params = _gauss_scene()
    sxx0, syy0, sxy0 = _moments(ctx, params, ())
    sxx, syy, sxy = _moments(ctx, params, ("q2_0",), q2_0=0.05)
    dxy = sxy - sxy0
    assert abs(dxy) > 1e-3
    assert abs((sxx - syy) - (sxx0 - syy0)) < 0.05 * abs(dxy)


def test_dilp_x_changes_round_width_linearly_in_X():
    ctx, params = _gauss_scene()
    X = (float(ctx.x_lin[0]) - 1024.0) / 1024.0
    sxx0, syy0, _ = _moments(ctx, params, ())
    w = {}
    for a in (0.03, 0.06):
        sxx, syy, _ = _moments(ctx, params, ("dilp_x",), dilp_x=a)
        w[a] = (sxx + syy) - (sxx0 + syy0)
        assert abs((sxx - syy) - (sxx0 - syy0)) < 0.05 * abs(w[a])
    np.testing.assert_allclose(w[0.06], 2 * w[0.03], rtol=0.05)
    # same effect as the base dilation slot evaluated at this star's X (linear in X)
    p = dict(params); p["chroma_g8"] = jnp.asarray([0.0] * 6 + [0.03 * X, 0.0, 0.0], jnp.float32)
    t_base = L.forward_model(p, _g8_ctx(ctx, ("dil_r",)))[0][0, 0, 0]
    p["chroma_g8"] = jnp.asarray([0.0] * 9 + [0.03], jnp.float32)
    t_plane = L.forward_model(p, _g8_ctx(ctx, ("dil_r", "dilp_x")))[0][0, 0, 0]
    np.testing.assert_allclose(np.asarray(t_plane), np.asarray(t_base), rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Gradients
# ---------------------------------------------------------------------------

def test_gradients_of_every_new_extra_are_finite_and_nonzero():
    fx = _fixture()
    extras = ("dil_r",) + NEW
    rng = np.random.default_rng(3)
    vals = C14 + list(0.01 * rng.normal(size=len(NEW)))
    ctx = _g8_ctx(fx["ctx"], extras, delta2_mean=float(np.mean(np.asarray(fx["ctx"].chroma_delta) ** 2)))
    g = jax.grad(lambda p: _loss(L, fx, p, ctx))(_g8_params(fx["params"], vals))["chroma_g8"]
    g = np.asarray(g)[9:]
    assert np.all(np.isfinite(g))
    for name, gi in zip(NEW, g):
        assert gi != 0.0, name
