# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Per-node local-polynomial (Anderson & King) smoothness and the sharp high-pass neighbour mode (2026-09-30)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import loss as L

G = 63


def gauss(sx, sy, x0=0.0, y0=0.0):
    c = (np.arange(G) - (G - 1) / 2) / 4.0
    yy, xx = np.meshgrid(c, c, indexing="ij")
    e = np.exp(-0.5 * ((xx - x0) ** 2 / sx ** 2 + (yy - y0) ** 2 / sy ** 2))
    return e / e.sum()


def test_5x5_quartic_is_anderson_king_eq8():
    k = L.savgol2d_kernel(4, 5)
    assert k[2, 2] == pytest.approx(0.441632, abs=2e-6)
    assert k[0, 0] == pytest.approx(0.041632, abs=2e-6)
    assert k[2, 1] == pytest.approx(0.200816, abs=2e-6)
    assert k.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("order,size", [(1, 7), (2, 7), (4, 7), (4, 5)])
def test_kernel_reproduces_polynomials_up_to_its_order(order, size):
    y, x = np.mgrid[:G, :G].astype(float) / 10.0
    k = L.savgol2d_kernel(order, size)
    for d in range(order + 1):
        for i in range(d + 1):
            f = jnp.asarray(x ** i * y ** (d - i), jnp.float32)[None, None]
            sm = L.local_poly_smooth(f, size, (order, order, order))
            h = size // 2
            np.testing.assert_allclose(np.asarray(sm)[0, 0, h:-h, h:-h], np.asarray(f)[0, 0, h:-h, h:-h],
                                       rtol=1e-3, atol=1e-3)


def test_nearly_blind_to_core_width_and_shift_but_penalises_junk():
    E = jnp.asarray(gauss(0.46, 0.5), jnp.float32)[None, None]
    gen = L._low_order_generators(E[0, 0])
    width = gen[3] + gen[4]
    rng = np.random.default_rng(0)
    junk = jnp.asarray(rng.normal(size=(G, G)) * float(jnp.std(width)), jnp.float32)
    stripe = jnp.asarray(np.where(np.arange(G) % 2, 1.0, -1.0)[None, :] * np.ones((G, 1)) * float(jnp.std(width)), jnp.float32)
    r = lambda v: float(L.local_poly_penalty(v[None, None])) / float(jnp.mean(v ** 2))
    assert r(width) < 0.05 and r(gen[1]) < 0.02          # a real width/shift change is almost free
    assert r(junk) > 0.3 and r(stripe) > 0.5              # grid-scale noise and stripes are not
    # the 5x5 AK kernel passes a sample-scale stripe untouched (why the default window is 7)
    assert float(L.local_poly_penalty(stripe[None, None], 5, (4, 4, 4))) < 1e-6 * float(jnp.mean(stripe ** 2))


def test_penalty_grad_finite_and_nodes_independent():
    E = jnp.asarray(np.stack([np.stack([gauss(0.45 + 0.02 * i, 0.5) for j in range(2)]) for i in range(2)]), jnp.float32)
    g = jax.grad(lambda e: L.local_poly_penalty(e))(E)
    assert bool(jnp.all(jnp.isfinite(g)))
    # changing one node leaves the others' gradient unchanged: no coupling between nodes
    E2 = E.at[0, 0].add(1e-4 * jnp.asarray(np.random.default_rng(1).normal(size=(G, G)), jnp.float32))
    g2 = jax.grad(lambda e: L.local_poly_penalty(e))(E2)
    np.testing.assert_allclose(np.asarray(g2[1, 1]), np.asarray(g[1, 1]), rtol=1e-6, atol=1e-12)


def test_highpass_mode_blind_to_width_penalises_checkerboard():
    E = jnp.asarray(np.stack([np.stack([gauss(0.46 + 0.03 * j, 0.5) for j in range(2)])]), jnp.float32)
    plain = float(L.fine_nbr_penalty(E[None], "plain")); hp = float(L.fine_nbr_penalty(E[None], "highpass"))
    assert hp < 0.1 * plain                               # a pure width difference barely passes 1 c/px
    cb = jnp.asarray(np.where((np.add.outer(np.arange(G), np.arange(G))) % 2, 1e-4, -1e-4), jnp.float32)
    E2 = E.at[0, 1].add(cb)
    assert float(L.fine_nbr_penalty(E2[None], "highpass")) > 0.5 * float(jnp.mean(cb ** 2)) * 0.25


def test_local_poly_hard_decode_smooths_and_keeps_gauges(monkeypatch):
    from syndiff_pipeline.forward_model import epsf_model as EM
    rng = np.random.default_rng(4)
    E = np.stack([np.stack([gauss(0.46 + 0.02 * i, 0.5) * (1 + 0.05 * rng.normal(size=(G, G))) for j in range(2)]) for i in range(2)])
    E = np.clip(E, 1e-9, None); E /= E.sum(axis=(-2, -1), keepdims=True)
    params = {"epsf_base_raw": EM.encode_epsf_base(jnp.asarray(E, jnp.float32))}
    try:
        L.set_local_poly_hard(0); off = np.asarray(L.decoded_epsf_base(params))
        L.set_local_poly_hard(7); on = np.asarray(L.decoded_epsf_base(params))
    finally:
        L.set_local_poly_hard(0)
    # one smoothing pass is not a projection (Q(Q(E)) != Q(E)): a 7x7 pass removes ~70% of the penalised structure,
    # a 5x5 pass ~90% (as in AK, who smooth once per iteration); the flux-rule/centring gauges do not add it back
    assert float(L.local_poly_penalty(jnp.asarray(on))) < 0.5 * float(L.local_poly_penalty(jnp.asarray(off)))
    np.testing.assert_allclose(on.sum(axis=(-2, -1)), 1.0, atol=1e-4)
    cls = np.asarray(EM.phase_class_sums(jnp.asarray(on)))            # flux rule still holds
    assert np.allclose(cls, cls.reshape(*cls.shape[:-2], -1).mean(-1)[..., None, None], rtol=2e-3)


def test_resample_nodes_bilinear_and_endpoints():
    from syndiff_pipeline.forward_model import scene_fit as SF
    ox = np.linspace(0, 2047, 6); oy = np.linspace(0, 2047, 6)
    f = (np.add.outer(oy, 2 * ox))[..., None]                         # linear in position -> resampled exactly
    nx = ny = np.linspace(0, 2047, 3)
    out = SF.resample_nodes(f, ox, oy, nx, ny)[..., 0]
    np.testing.assert_allclose(out, np.add.outer(ny, 2 * nx), rtol=1e-12)
