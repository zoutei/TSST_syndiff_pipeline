# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Per-stamp local background (scene_fit --stamp-bg annulus|fit), 2026-09-30.

(a) off -> bit-identical (island_solve without ped; make_model default on F2 in test_scene_bg_cheb)
(b) the annulus estimator recovers an injected constant under a skewed positive contaminant tail
    better than a least-squares plane
(c) 'fit' with a tight prior == 'annulus'; with no prior == the unconstrained LSQ pedestal (float64)
(d) coexistence with --bright-width and --bg-cheb-order: test_scene_bg_cheb::test_bright_width_and_background_coexist
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import scene_fit as SF
from syndiff_pipeline.forward_model import stamp_bg as SB
from test_fm_scene_bg_cheb import _toy

jax.config.update("jax_platform_name", "cpu")


# ---------------------------------------------------------------- (b) estimator

def _contaminated_stamp(rng, c0, S=15):
    ox, oy = SB.stamp_offsets(S)
    r2 = ox ** 2 + oy ** 2
    star = 3e4 * np.exp(-r2 / (2 * 0.8 ** 2)) / (2 * np.pi * 0.8 ** 2)
    img = star + c0 + rng.normal(0, 1.0, S * S)
    # skewed positive tail: faint unresolved sources on ~15% of pixels
    hit = rng.random(S * S) < 0.15
    img[hit] += rng.exponential(4.0, hit.sum())
    return img


def test_annulus_beats_lsq_plane_on_skewed_tail():
    rng = np.random.default_rng(0)
    S, c0 = 15, -0.17
    ox, oy = SB.stamp_offsets(S)
    r = np.hypot(ox, oy)
    ann = (r >= 5) & (r <= 7)
    err_a, err_p = [], []
    for _ in range(200):
        img = _contaminated_stamp(rng, c0)
        est = SB.annulus_estimates(img[None], np.ones((1, S * S), bool), [0.0], [0.0], [0], [0], [3e4], S=S)
        err_a.append(est["est"][0] - c0)
        X = np.c_[np.ones(ann.sum()), ox[ann], oy[ann]]
        err_p.append(np.linalg.lstsq(X, img[ann], rcond=None)[0][0] - c0)
    err_a, err_p = np.array(err_a), np.array(err_p)
    assert np.mean(np.abs(err_a)) < 0.5 * np.mean(np.abs(err_p))
    assert abs(np.mean(err_a)) < abs(np.mean(err_p))


def test_annulus_masks_bright_neighbours_and_reports_clipping():
    S = 15
    rng = np.random.default_rng(1)
    ox, oy = SB.stamp_offsets(S)
    img = 2.0 + rng.normal(0, 0.5, S * S)
    nb = (6.0, 0.0)                                      # a bright neighbour sitting on the annulus
    img += 5e4 * np.exp(-((ox - nb[0]) ** 2 + (oy - nb[1]) ** 2) / (2 * 0.8 ** 2)) / (2 * np.pi * 0.64)
    data = np.stack([img, img])
    est = SB.annulus_estimates(data, np.ones((2, S * S), bool), [0.0, 6.0], [0.0, 0.0], [0, 6], [0, 0],
                               [1e4, 5e4], S=S)
    assert est["n_masked"][0] > 0 and abs(est["est"][0] - 2.0) < 0.2
    assert est["n_used"][0] + est["n_clipped"][0] + est["n_masked"][0] == est["n_annulus"][0]
    summ = SB.summary(est)
    assert 0.0 <= summ["clipped_fraction"] < 1 and summ["masked_fraction"] > 0


def test_cells_partition_union_pixels():
    T, st, a, tiers, phi, ref = _toy(-1)
    cells = SB.cell_owner(ref["uid"], ref["U"], ref["S"])
    uid = ref["uid"]
    # every union pixel has exactly one owner copy (the copy inside its own cell)
    own = cells == np.arange(uid.shape[0])[:, None]
    counts = np.bincount(uid[own], minlength=ref["U"])
    assert np.all(counts[np.unique(uid)] == 1)
    # all copies of a pixel agree on the cell; nearest centre wins in the overlap
    for u in np.unique(uid):
        assert len(set(cells[uid == u])) == 1


# ---------------------------------------------------------------- (c) the solver

def _ped(ref, est_mu, sigma):
    N = ref["T"].shape[0]
    cells = SB.cell_owner(ref["uid"], ref["U"], ref["S"])
    z = ref["z"]
    pw = np.zeros(N) if sigma is None else np.full(N, 1.0 / sigma ** 2)
    return cells, dict(cell_star=jnp.asarray(cells, jnp.int32),
                       cell_slot=jnp.asarray(z["star_slot"][cells], jnp.int32),
                       own_cell=jnp.asarray(cells == np.arange(N)[:, None], jnp.float32),
                       prior_w=jnp.asarray(pw, jnp.float32), prior_mu=jnp.asarray(est_mu, jnp.float32))


def _solve(T, st, a, tiers, phi, ped):
    return jax.jit(lambda T, st: SF.island_solve(T, st, a, tiers, ridge=0.0, prior_kappa=0.0,
                                                 phi=phi, ped=ped))(T, st)


@pytest.mark.parametrize("order", [-1, 2])
def test_tight_prior_equals_annulus(order):
    T, st, a, tiers, phi, ref = _toy(order)
    N = ref["T"].shape[0]
    mu = np.random.default_rng(5).normal(-0.2, 0.3, N)
    cells, ped = _ped(ref, mu, 1e-3)
    f_fit, c_fit, b_fit = _solve(T, st, a, tiers, phi, ped)
    a1 = dict(a, data=a["data"] - jnp.asarray(mu, jnp.float32)[cells])          # 'annulus' = fixed offset
    f_ann, c_ann, _ = _solve(T, st, a1, tiers, phi, None)
    np.testing.assert_allclose(np.asarray(b_fit), mu, atol=1e-3)
    np.testing.assert_allclose(np.asarray(f_fit), np.asarray(f_ann), rtol=1e-4)
    if order >= 0:
        np.testing.assert_allclose(np.asarray(c_fit), np.asarray(c_ann), atol=1e-2)


def test_no_prior_equals_float64_lsq_pedestal():
    T, st, a, tiers, phi, ref = _toy(-1)
    N, U, uid = ref["T"].shape[0], ref["U"], ref["uid"]
    cells, ped = _ped(ref, np.zeros(N), None)
    f, c, b = _solve(T, st, a, tiers, None, ped)
    assert c is None
    # dense float64: unknowns = free fluxes + one pedestal per stamp cell, over union pixels
    X = np.zeros((U, N))
    for i in range(N):
        np.add.at(X[:, i], uid[i], ref["T"][i])
    P = np.zeros((U, N))
    P[uid.ravel(), cells.ravel()] = 1.0
    free = ref["free"] > 0
    y = ref["data_u"] - X[:, ~free] @ ref["f_fixed"][~free]
    D = np.hstack([X[:, free], P])
    w = 1.0 / ref["var_u"]
    sol = np.linalg.solve(D.T @ (w[:, None] * D), D.T @ (w * y))
    f_ref = ref["f_fixed"].copy(); f_ref[free] = sol[:free.sum()]
    np.testing.assert_allclose(np.asarray(f), f_ref, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(b), sol[free.sum():], atol=2e-3)


def test_ped_gradients_finite():
    T, st, a, tiers, phi, ref = _toy(2)
    _, ped = _ped(ref, np.zeros(ref["T"].shape[0]), 0.5)
    g = jax.grad(lambda T: jnp.sum(SF.island_solve(T, st, a, tiers, ridge=0.0, prior_kappa=0.0,
                                                    phi=phi, ped=ped)[0]))(T)
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).max()) > 0


def test_off_is_the_plain_solve():
    T, st, a, tiers, phi, ref = _toy(-1)
    f0, c0, b0 = _solve(T, st, a, tiers, None, None)
    assert c0 is None and b0 is None
    # same traced program as a solve that never heard of pedestals (the default arguments)
    j_off = jax.make_jaxpr(lambda T: SF.island_solve(T, st, a, tiers, ridge=0.0, prior_kappa=0.0, ped=None)[0])(T)
    j_def = jax.make_jaxpr(lambda T: SF.island_solve(T, st, a, tiers, ridge=0.0, prior_kappa=0.0)[0])(T)
    assert str(j_off) == str(j_def)
