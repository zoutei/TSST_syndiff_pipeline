# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Task T3: give the temporal ePSF mode a per-node (spatial) amplitude field.

``w_coeff`` generalises from ``(n_modes, n_basis)`` (legacy: one global
``w_k(t)`` shared by every ePSF node) to ``(n_modes, n_rows, n_cols,
n_basis)`` (T3: a per-node ``w_k(t, x, y)`` field, bilinear-blended to each
star's position with the same node-blend machinery the ePSF base/modes
already use). Dispatch is shape-based (``w_coeff.ndim``), so every existing
2-D checkpoint/caller must be completely unaffected.

Covers the task's required gates: bit-identity of the dispatch helpers for a
legacy 2-D ``w_coeff``, the per-node zero-time-mean gauge for a spatial (4-D)
``w_coeff``, a synthetic case where a known per-node temporal amplitude is
recovered (both a closed-form exact-reduction check and a short Adam fit),
gradient finiteness through the full ``total_loss``, a checkpoint round trip,
and a K=2/4x4-grid end-to-end exercise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import fit as FIT
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model.groups import GroupSet
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import sci2idl_exponents


# ---------------------------------------------------------------------------
# 1. Unit tests for the shape-dispatch helpers themselves.
# ---------------------------------------------------------------------------

def test_w_field_from_coeff_2d_matches_legacy_formula_exactly():
    """The ndim==2 branch must be textually identical to the pre-T3 code."""
    rng = np.random.default_rng(0)
    n_modes, n_basis, n_frames = 2, 5, 9
    w_coeff = jnp.asarray(rng.normal(size=(n_modes, n_basis)).astype(np.float32))
    frame_basis = jnp.asarray(rng.normal(size=(n_frames, n_basis)).astype(np.float32))

    got = L.w_field_from_coeff(w_coeff, frame_basis)
    legacy = (w_coeff @ frame_basis.T).T
    legacy = legacy - jnp.mean(legacy, axis=0, keepdims=True)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(legacy))
    assert got.shape == (n_frames, n_modes)


def test_w_field_from_coeff_4d_shape_and_per_node_zero_mean_gauge():
    rng = np.random.default_rng(1)
    n_modes, n_rows, n_cols, n_basis, n_frames = 2, 4, 4, 6, 11
    w_coeff = jnp.asarray(
        rng.normal(size=(n_modes, n_rows, n_cols, n_basis)).astype(np.float32)
    )
    frame_basis = jnp.asarray(rng.normal(size=(n_frames, n_basis)).astype(np.float32))

    got = L.w_field_from_coeff(w_coeff, frame_basis)
    assert got.shape == (n_frames, n_rows, n_cols, n_modes)
    mean_over_t = np.asarray(jnp.mean(got, axis=0))
    # Independently zero at EVERY node, not just on average over the field.
    np.testing.assert_allclose(mean_over_t, np.zeros_like(mean_over_t), atol=1e-6)
    # Every node's curve genuinely differs -- not a broadcast copy of one
    # shared curve (that would defeat the entire point of T3).
    flat = np.asarray(got).reshape(n_frames, n_rows * n_cols, n_modes)
    assert not np.allclose(flat[:, 0, :], flat[:, 1, :])


def test_w_field_from_coeff_rejects_bad_ndim():
    bad = jnp.zeros((2, 3, 4))
    with pytest.raises(ValueError):
        L.w_field_from_coeff(bad, jnp.zeros((5, 4)))


def test_node_field_for_frame_dispatch_matches_broadcast_and_per_node_forms():
    rng = np.random.default_rng(2)
    K, n_rows, n_cols, G = 2, 4, 4, 5
    base = jnp.asarray(rng.normal(size=(n_rows, n_cols, G, G)).astype(np.float32))
    modes = jnp.asarray(rng.normal(size=(K, n_rows, n_cols, G, G)).astype(np.float32))

    w_scalar = jnp.asarray(rng.normal(size=(K,)).astype(np.float32))
    got_legacy = L._node_field_for_frame(base, modes, w_scalar)
    want_legacy = base + jnp.einsum("kijxy,k->ijxy", modes, w_scalar)
    np.testing.assert_array_equal(np.asarray(got_legacy), np.asarray(want_legacy))

    w_node = jnp.asarray(rng.normal(size=(n_rows, n_cols, K)).astype(np.float32))
    got_spatial = L._node_field_for_frame(base, modes, w_node)
    want_spatial = base + jnp.einsum("kijxy,ijk->ijxy", modes, w_node)
    np.testing.assert_array_equal(np.asarray(got_spatial), np.asarray(want_spatial))

    # Sanity: broadcasting the same scalar to every node must agree with the
    # legacy (uniform) form exactly.
    w_node_uniform = jnp.broadcast_to(w_scalar, (n_rows, n_cols, K))
    got_uniform = L._node_field_for_frame(base, modes, w_node_uniform)
    np.testing.assert_allclose(np.asarray(got_uniform), np.asarray(got_legacy), atol=1e-5)


def test_init_params_w_spatial_flag_shapes_and_zero_init():
    static, epsf0, n_wcs, n_w = _tiny_static_and_epsf(n_rows=3, n_cols=3, n_modes=2)
    p2d = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w)
    assert p2d["w_coeff"].shape == (2, n_w)

    p4d = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w, w_spatial=True)
    assert p4d["w_coeff"].shape == (2, 3, 3, n_w)
    assert float(jnp.abs(p4d["w_coeff"]).max()) == 0.0


# ---------------------------------------------------------------------------
# Shared fixture helpers: minimal WCS/ePSF static scaffolding + a StaticContext
# with stars placed at caller-chosen (region-local) pixel positions.
# ---------------------------------------------------------------------------

def _tiny_static_and_epsf(*, n_rows: int, n_cols: int, n_modes: int, seed: int = 0):
    degree = 1
    exps = tuple(sci2idl_exponents(degree))
    static = CW.ChebWcsStatic(
        ra0_deg=180.0, dec0_deg=0.0,
        cd_inv=np.array([[-20.0, 0.0], [0.0, 20.0]], dtype=float),
        crpix=np.array([51.0, 51.0], dtype=float),
        center=np.array([50.0, 50.0], dtype=float),
        half_extents=np.array([50.0, 50.0], dtype=float),
        poly_degree=degree, exponents=exps,
    )
    rng = np.random.default_rng(seed)
    Gsz = EM.NODE_GRID_SIZE
    base = rng.random((n_rows, n_cols, Gsz, Gsz)).astype(np.float32)
    base /= base.reshape(n_rows, n_cols, -1).sum(axis=-1)[..., None, None]
    modes = rng.normal(size=(n_modes, n_rows, n_cols, Gsz, Gsz)).astype(np.float32) * 0.05
    epsf0 = EM.EpsfGridParams(base=jnp.asarray(base), modes=jnp.asarray(modes))
    n_wcs_basis, n_w_basis = 2, 4
    return static, epsf0, n_wcs_basis, n_w_basis


def _build_ctx(
    static: CW.ChebWcsStatic,
    *,
    node_x: np.ndarray,
    node_y: np.ndarray,
    star_x: np.ndarray,
    star_y: np.ndarray,
    n_frames: int,
    n_wcs_basis: int,
    n_w_basis: int,
    seed: int = 0,
) -> L.StaticContext:
    """One star per group (K=1 slots), square (non-packed) path."""
    n_stars = len(star_x)
    grid = EM.EpsfGridStatic(
        node_x=node_x, node_y=node_y, node_col_ccd=node_x, node_row_ccd=node_y,
    )
    members = np.arange(n_stars, dtype=np.int64)[:, None]
    valid = np.ones((n_stars, 1), dtype=bool)
    groups = GroupSet(n_stars, 1, members, valid, np.ones(n_stars, dtype=bool), 0)
    rng = np.random.default_rng(seed)
    wcs_fb = rng.normal(size=(n_frames, n_wcs_basis)).astype(np.float32)
    w_fb = rng.normal(size=(n_frames, n_w_basis)).astype(np.float32)
    n_terms = static.n_terms
    return L.build_static_context(
        cheb_static=static, wcs_frame_basis=wcs_fb, w_frame_basis=w_fb,
        epsf_grid=grid, groups=groups,
        ra=np.zeros(n_stars, dtype=np.float32), dec=np.zeros(n_stars, dtype=np.float32),
        x_lin=np.asarray(star_x, dtype=np.float32), y_lin=np.asarray(star_y, dtype=np.float32),
        cheb_basis=np.zeros((n_stars, n_terms), dtype=np.float32),
        stamp_center_x=np.round(np.asarray(star_x)).astype(np.int64),
        stamp_center_y=np.round(np.asarray(star_y)).astype(np.int64),
        t_exp_sec=1426.0,
        stamp_snr_weight=np.ones(n_stars, dtype=np.float32),
        fit_radius=np.full(n_stars, 3.0, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 2. Synthetic recovery, part A: closed-form exact reduction at a star sitting
#    exactly on a node -- the spatial model, queried at that node, must render
#    BIT-IDENTICALLY to the legacy global model carrying the same value.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("corner", [(0, 0), (3, 3), (1, 2)])
def test_forward_model_spatial_w_reduces_to_legacy_at_the_matching_node(corner):
    n_rows, n_cols, n_modes, n_frames = 4, 4, 2, 5
    static, epsf0, n_wcs, n_w = _tiny_static_and_epsf(
        n_rows=n_rows, n_cols=n_cols, n_modes=n_modes,
    )
    node_x = np.array([10.0, 30.0, 50.0, 70.0])
    node_y = np.array([10.0, 30.0, 50.0, 70.0])
    i0, j0 = corner
    ctx = _build_ctx(
        static, node_x=node_x, node_y=node_y,
        star_x=np.array([node_x[j0]]), star_y=np.array([node_y[i0]]),
        n_frames=n_frames, n_wcs_basis=n_wcs, n_w_basis=n_w,
    )

    rng = np.random.default_rng(7)
    w_value = rng.normal(size=(n_modes, n_w)).astype(np.float32) * 0.3

    params_legacy = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w)
    params_legacy["w_coeff"] = jnp.asarray(w_value)

    params_spatial = L.init_params(
        static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w, w_spatial=True,
    )
    w4d = np.zeros((n_modes, n_rows, n_cols, n_w), dtype=np.float32)
    w4d[:, i0, j0, :] = w_value
    params_spatial["w_coeff"] = jnp.asarray(w4d)

    t_legacy, _, _, _ = L.forward_model(params_legacy, ctx)
    t_spatial, _, _, _ = L.forward_model(params_spatial, ctx)
    np.testing.assert_allclose(np.asarray(t_legacy), np.asarray(t_spatial), atol=1e-5, rtol=1e-5)
    # And it must actually be non-trivial (w perturbs the rendered template).
    params_zero = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w)
    t_zero, _, _, _ = L.forward_model(params_zero, ctx)
    assert not np.allclose(np.asarray(t_zero), np.asarray(t_legacy), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Synthetic recovery, part B: a genuine Adam fit recovers an unknown
#    per-node amplitude field (K=2, 4x4 grid) from noiseless synthetic data.
#    do_recenter=False makes forward_model exactly LINEAR in w_coeff (through
#    the modes contraction), so the fit is a well-posed, non-flaky convex
#    problem -- appropriate for a fast, deterministic unit test.
# ---------------------------------------------------------------------------

def test_adam_recovers_known_per_node_w_field_k2_4x4_grid():
    n_rows, n_cols, n_modes, n_frames = 4, 4, 2, 60
    static, epsf0, n_wcs, n_w = _tiny_static_and_epsf(
        n_rows=n_rows, n_cols=n_cols, n_modes=n_modes, seed=3,
    )
    node_x = np.array([10.0, 30.0, 50.0, 70.0])
    node_y = np.array([10.0, 30.0, 50.0, 70.0])
    # One star exactly AT each of the 16 nodes: every node's curve is then
    # independently identifiable (its own star's bilinear weight is one-hot),
    # so a node with no star -- which would have structurally zero gradient,
    # by design (see the CPU warm-start's "under-populated nodes" regularizer)
    # -- cannot be mistaken for a fit failure in this unit test.
    gy, gx = np.meshgrid(node_y, node_x, indexing="ij")
    star_x = gx.ravel()
    star_y = gy.ravel()
    ctx = _build_ctx(
        static, node_x=node_x, node_y=node_y, star_x=star_x, star_y=star_y,
        n_frames=n_frames, n_wcs_basis=n_wcs, n_w_basis=n_w, seed=9,
    )

    rng = np.random.default_rng(11)
    w_true = (rng.normal(size=(n_modes, n_rows, n_cols, n_w)) * 0.3).astype(np.float32)

    params_true = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w, w_spatial=True)
    params_true["w_coeff"] = jnp.asarray(w_true)
    true_templates, _, _, _ = L.forward_model(params_true, ctx, do_recenter=False)
    true_templates = jax.lax.stop_gradient(true_templates)

    params0 = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w, w_spatial=True)

    def loss_fn(w_coeff):
        params = dict(params0)
        params["w_coeff"] = w_coeff
        templates, _, _, _ = L.forward_model(params, ctx, do_recenter=False)
        return jnp.mean((templates - true_templates) ** 2)

    opt = optax.adam(0.1)
    w = jnp.zeros_like(params_true["w_coeff"])
    opt_state = opt.init(w)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(400):
        val, grad = grad_fn(w)
        updates, opt_state = opt.update(grad, opt_state, w)
        w = optax.apply_updates(w, updates)

    assert np.isfinite(float(val))
    assert val < 1e-6, f"Adam fit did not converge (final MSE {float(val):.3e})"

    w_recovered = np.asarray(w)
    corr = np.corrcoef(w_recovered.ravel(), w_true.ravel())[0, 1]
    assert corr > 0.99, f"recovered w_coeff poorly correlated with truth (r={corr:.3f})"
    np.testing.assert_allclose(w_recovered, w_true, atol=0.03, rtol=0.1)

    # The physical claim T3 is built on: different nodes' recovered curves
    # really do differ from one another (not just noise on a shared curve).
    node_curves = w_recovered.reshape(n_modes, n_rows * n_cols, n_w)
    assert not np.allclose(node_curves[:, 0, :], node_curves[:, -1, :], atol=1e-3)


# ---------------------------------------------------------------------------
# 4. Gradient finiteness through the full total_loss with a spatial w_coeff.
# ---------------------------------------------------------------------------

def test_total_loss_gradient_finite_with_spatial_w_coeff():
    n_rows, n_cols, n_modes, n_frames = 3, 3, 2, 8
    static, epsf0, n_wcs, n_w = _tiny_static_and_epsf(
        n_rows=n_rows, n_cols=n_cols, n_modes=n_modes, seed=5,
    )
    node_x = np.array([10.0, 40.0, 70.0])
    node_y = np.array([10.0, 40.0, 70.0])
    star_x = np.array([15.0, 55.0])
    star_y = np.array([25.0, 65.0])
    ctx = _build_ctx(
        static, node_x=node_x, node_y=node_y, star_x=star_x, star_y=star_y,
        n_frames=n_frames, n_wcs_basis=n_wcs, n_w_basis=n_w, seed=6,
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w, w_spatial=True)
    rng = np.random.default_rng(13)
    params["w_coeff"] = jnp.asarray(
        (rng.normal(size=(n_modes, n_rows, n_cols, n_w)) * 0.1).astype(np.float32)
    )

    n_groups = int(ctx.members.shape[0])
    data = jnp.asarray(rng.random((n_groups, n_frames, 7, 7)).astype(np.float32))
    noise = jnp.ones_like(data)
    weight = jnp.ones_like(data)
    wcs_second_diff = T.second_difference_matrix(n_wcs)
    w_second_diff = T.second_difference_matrix(n_w)
    epsf_modes_init = params["epsf_modes"]

    def scalar(p):
        loss, _ = L.total_loss(
            p, ctx, data, noise, weight, wcs_second_diff, w_second_diff,
            epsf_modes_init=epsf_modes_init,
        )
        return loss

    value, grads = jax.value_and_grad(scalar)(params)
    assert np.isfinite(float(value))
    for key, g in grads.items():
        arr = np.asarray(g)
        assert np.isfinite(arr).all(), f"non-finite gradient for {key}"
    assert np.any(np.asarray(grads["w_coeff"]) != 0.0)
    assert grads["w_coeff"].shape == params["w_coeff"].shape


# ---------------------------------------------------------------------------
# 5. Checkpoint round trip with a spatial (4-D) w_coeff.
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip_spatial_w_coeff(tmp_path):
    n_rows, n_cols, n_modes = 4, 4, 2
    static, epsf0, n_wcs, n_w = _tiny_static_and_epsf(
        n_rows=n_rows, n_cols=n_cols, n_modes=n_modes, seed=8,
    )
    params = L.init_params(static, epsf0, n_wcs_basis=n_wcs, n_w_basis=n_w, w_spatial=True)
    rng = np.random.default_rng(14)
    params["w_coeff"] = jnp.asarray(
        rng.normal(size=(n_modes, n_rows, n_cols, n_w)).astype(np.float32)
    )
    path = tmp_path / "ckpt.npz"
    FIT.save_params_npz(path, params)
    loaded = FIT.load_params_npz(path)
    assert loaded["w_coeff"].shape == params["w_coeff"].shape
    np.testing.assert_array_equal(np.asarray(loaded["w_coeff"]), np.asarray(params["w_coeff"]))


# ---------------------------------------------------------------------------
# 6. End-to-end wiring: train_loop.run_stages_from_bundle(w_spatial=True)
#    builds the right shape from a 2-D bundle params0, an --init-params
#    checkpoint carrying a matching 4-D w_coeff survives the merge, and the
#    stage-3 checkpoint round-trips it (mirrors
#    test_chroma.test_isolated_stage_handoff_preserves_non_zero_chroma).
# ---------------------------------------------------------------------------

def test_train_loop_w_spatial_end_to_end_and_isolated_stage_handoff(tmp_path):
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import fit_bundle as FB
    from syndiff_pipeline.forward_model import train_loop as TL
    from test_fm_fit_bundle import _tiny_bundle

    bundle = _tiny_bundle()
    assert bundle.params0["w_coeff"].ndim == 2  # bundle export is always 2-D

    n_modes, n_w = bundle.params0["w_coeff"].shape
    n_rows, n_cols = bundle.epsf_base.shape[0], bundle.epsf_base.shape[1]
    rng = np.random.default_rng(21)
    inject_w = (0.02 * rng.normal(size=(n_modes, n_rows, n_cols, n_w))).astype(np.float32)
    seed = dict(FB.params0_as_jnp(bundle))
    seed["w_coeff"] = jnp.asarray(inject_w)
    seed_path = tmp_path / "seed.npz"
    FIT.save_params_npz(seed_path, {k: np.asarray(v) for k, v in seed.items()})

    out = tmp_path / "out"
    TL.run_stages_from_bundle(
        bundle, out_dir=out, stage=3, start_stage=3,
        steps_per_stage=[0, 0, 1], lr_per_stage=[1e-2, 3e-4, 1e-4],
        log_every=1, checkpoint_every=0, reject_every=0, stage1_core_stamp=0,
        w_spatial=True, init_params=seed_path,
    )
    got = FIT.load_params_npz(out / "params_stage3.npz")
    assert got["w_coeff"].shape == (n_modes, n_rows, n_cols, n_w)
    assert np.isfinite(np.asarray(got["w_coeff"])).all()
    # Injected value must have survived the merge (not silently reset to
    # zero/2-D) -- Adam may have moved it a little in one step, so check it
    # is close to, not identical to, the seed.
    np.testing.assert_allclose(np.asarray(got["w_coeff"]), inject_w, atol=0.05)


def test_train_loop_w_spatial_requires_matching_init_params_shape(tmp_path):
    """A 2-D --init-params checkpoint must not silently merge into a
    freshly-upgraded 4-D params tree (or vice versa) -- exact shape match is
    required, same contract as every other optional leaf."""
    pytest.importorskip("optax")
    from syndiff_pipeline.forward_model import fit_bundle as FB
    from syndiff_pipeline.forward_model import train_loop as TL
    from test_fm_fit_bundle import _tiny_bundle

    bundle = _tiny_bundle()
    seed = dict(FB.params0_as_jnp(bundle))  # w_coeff stays 2-D
    seed_path = tmp_path / "seed_2d.npz"
    FIT.save_params_npz(seed_path, {k: np.asarray(v) for k, v in seed.items()})

    with pytest.raises(SystemExit):
        TL.run_stages_from_bundle(
            bundle, out_dir=tmp_path / "out", stage=3, start_stage=3,
            steps_per_stage=[0, 0, 1], lr_per_stage=[1e-2, 3e-4, 1e-4],
            log_every=1, checkpoint_every=0, reject_every=0, stage1_core_stamp=0,
            w_spatial=True, init_params=seed_path,
        )
