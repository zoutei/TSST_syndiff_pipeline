"""bright_width forms aniso / quad / onset (2026-10-12, bwidth_campaign_20261012): stamp-level calibration through
loss.forward_model, the lin special case, set_bright_width leaves, stage-3 training bucket, onset q transform."""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syndiff_pipeline.forward_model import bright_width as BW  # noqa: E402
from syndiff_pipeline.forward_model import epsf_model as EM  # noqa: E402
from syndiff_pipeline.forward_model import fit as FIT  # noqa: E402
from syndiff_pipeline.forward_model import loss as L  # noqa: E402
from test_fm_bright_width import _gauss_scene, _moments, _a3_ctx  # noqa: E402

Q_REF = 2.0e4


def _render(ctx, params, q, extra=None, colour=False):
    n = int(ctx.x_lin.shape[0])
    ctx = L.replace(ctx, chroma_g8_gauge="raw", bright_q=jnp.full((n,), q, jnp.float32), bright_q_ref=Q_REF)
    p = dict(params)
    if colour:
        ctx = _a3_ctx(ctx); p["chroma_g8"] = jnp.zeros((17,), jnp.float32)
    t0 = L.forward_model(p, ctx)[0][0, 0, 0]
    p.update(extra or {})
    t1 = L.forward_model(p, ctx)[0][0, 0, 0]
    m0, m1 = _moments(t0), _moments(t1)
    return m1[0] - m0[0], m1[1] - m0[1], m1[2] - m0[2]


def _leaf(v):
    return jnp.asarray([v / BW.LEAF_UNIT], jnp.float32)


@pytest.mark.parametrize("colour", [False, True])
def test_aniso_calibration(colour):
    ctx, params = _gauss_scene()
    b, d, dq = 0.6e-3, 0.3e-3, 1.0e5
    dxx, dyy, dxy = _render(ctx, params, Q_REF + dq, dict(bright_width=_leaf(b), bright_width_xy=_leaf(d)), colour)
    u = dq / BW.Q_UNIT
    assert abs(dxx / ((b + d) * u) - 1) < 0.03, dxx / ((b + d) * u)
    assert abs(dyy / ((b - d) * u) - 1) < 0.05, dyy / ((b - d) * u)
    assert abs(dxy) < 0.01 * b * u


def test_aniso_generator_moments_on_node_grid():
    gsz = EM.NODE_GRID_SIZE
    yy, xx = np.mgrid[0:gsz, 0:gsz] - EM.NODE_CENTER_INDEX
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (EM.OVERSAMPLE * 1.1) ** 2)); g /= g.sum()
    c = np.asarray(EM.node_coord_1d(n_grid=gsz), np.float64)

    def s(a, ax):
        f = a.sum(); cc = c[None, :] if ax == "x" else c[:, None]
        m = (a * cc).sum() / f
        return (a * (cc - m) ** 2).sum() / f
    w = 4e-3
    G = np.asarray(EM.chroma_blur_xy_field_raw(jnp.asarray(g[None, None], jnp.float32)), np.float64)[0, 0]
    assert abs((s(g + w * G, "x") - s(g, "x")) / (2 * w) - 1) < 0.01
    assert abs((s(g + w * G, "y") - s(g, "y")) / (-2 * w) - 1) < 0.01


def test_quad_calibration_and_zero_c_is_lin():
    ctx, params = _gauss_scene()
    b, c, dq = 0.5e-3, 0.02e-3, 1.0e5
    q = Q_REF + dq
    dxx, dyy, _ = _render(ctx, params, q, dict(bright_width=_leaf(b), bright_width_q2=_leaf(c)))
    want = b * dq / BW.Q_UNIT + c * (q ** 2 - Q_REF ** 2) / BW.Q_UNIT ** 2
    assert abs(dxx / want - 1) < 0.03 and abs(dyy / want - 1) < 0.03
    lin = _render(ctx, params, q, dict(bright_width=_leaf(b)))
    zc = _render(ctx, params, q, dict(bright_width=_leaf(b), bright_width_q2=_leaf(0.0)))
    zd = _render(ctx, params, q, dict(bright_width=_leaf(b), bright_width_xy=_leaf(0.0)))
    np.testing.assert_allclose(zc, lin, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(zd, lin, rtol=1e-6, atol=1e-12)


def test_onset_q():
    q = np.array([0.0, 5e3, 1e4, 3e4])
    np.testing.assert_allclose(BW.onset_q(q, 1e4), [0, 0, 0, 2e4])
    np.testing.assert_allclose(BW.onset_q(q, 0.0), q)


def test_set_bright_width_extra_leaves():
    p, _ = BW.set_bright_width({}, "aniso", 1.0)
    assert set(p) == {"bright_width", "bright_width_xy"} and float(p["bright_width_xy"][0]) == 0.0
    p2, _ = BW.set_bright_width(dict(p, bright_width_xy=jnp.asarray([0.3], jnp.float32)), "aniso", None)
    assert abs(float(p2["bright_width_xy"][0]) - 0.3) < 1e-7
    p3, _ = BW.set_bright_width(p2, "quad", None)
    assert set(p3) == {"bright_width", "bright_width_q2"}
    p4, _ = BW.set_bright_width(p3, "onset", None)
    assert set(p4) == {"bright_width"}
    p5, _ = BW.set_bright_width(p3, "none", None)
    assert not set(p5) & set(BW.BRIGHT_WIDTH_LEAVES)
    assert FIT.BRIGHT_WIDTH_LEAVES == BW.BRIGHT_WIDTH_LEAVES


def test_extra_leaves_train_at_stage3_in_chroma_bucket():
    keys = ("wcs_coeff", "epsf_base_raw", "bright_width", "bright_width_xy", "bright_width_q2")
    for st, want in ((2, "frozen"), (3, "train_chroma")):
        lab = FIT._leaf_labels(st, freeze_wcs=False, param_keys=keys)
        assert lab["bright_width_xy"] == want and lab["bright_width_q2"] == want


def test_lin_weight_unchanged():
    ctx = type("C", (), dict(bright_q=jnp.asarray([1e4, 5e4]), bright_q_ref=2e4))()
    w = BW.slot_weight({"bright_width": jnp.asarray([0.8])}, ctx, jnp.arange(2))
    np.testing.assert_allclose(np.asarray(w), 0.5 * 0.8e-3 / 1e4 * (np.array([1e4, 5e4]) - 2e4), rtol=1e-6)
