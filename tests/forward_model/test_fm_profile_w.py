# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Task PW: profile the temporal ePSF mode amplitude out in closed form,
per frame, jointly with the per-stamp fluxes.

Gates covered (in order):
  1. bit-identical when off (solver, ``LossWeights`` default, ``total_loss``);
  2. synthetic recovery of a known per-frame amplitude, K=1 and K=2 modes;
  3. blended groups (``K_members >= 2``) and the padded slots;
  4. pedestal on and off;
  5. gradient finiteness/correctness through the profiled solve;
  6. the zero-time-mean gauge and its exact base-shift counterpart;
  7. optimizer labels + checkpoint/export round trip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import fit as FIT
from syndiff_pipeline.forward_model import flux_solve as FS
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import temporal as TB


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _linear_problem(*, n_modes=2, n_members=1, packed=True, seed=0, blended_pad=False):
    """A synthetic problem that is EXACTLY the profiled model:
    ``data = sum_j f_j (A_j + sum_m w_m B_{j,m}) [+ b]``."""
    rng = np.random.default_rng(seed)
    n_groups, n_frames, P, S = 4, 6, 12, 4
    shape = (n_groups, n_members, n_frames, P) if packed else (n_groups, n_members, n_frames, S, S)
    A = rng.uniform(0.05, 1.0, size=shape).astype(np.float32)
    B = (rng.normal(size=(n_modes,) + shape) * 0.15).astype(np.float32)
    if blended_pad:
        # Last member slot of the first group is padding: all-zero template.
        A[0, -1] = 0.0
        B[:, 0, -1] = 0.0
    w_true = (rng.normal(size=(n_frames, n_modes)) * 0.4).astype(np.float32)
    f_true = rng.uniform(2.0, 6.0, size=(n_groups, n_frames, n_members)).astype(np.float32)
    if blended_pad:
        f_true[0, :, -1] = 0.0
    if packed:
        T = A + np.einsum("fm,mgkfp->gkfp", w_true, B)
        data = np.einsum("gkfp,gfk->gfp", T, f_true)
    else:
        T = A + np.einsum("fm,mgkfxy->gkfxy", w_true, B)
        data = np.einsum("gkfxy,gfk->gfxy", T, f_true)
    weight = np.ones_like(data, dtype=np.float32)
    return (jnp.asarray(A), jnp.asarray(B), jnp.asarray(data.astype(np.float32)),
            jnp.asarray(weight), w_true, f_true)


# --------------------------------------------------------------------------
# 1. bit-identical when off
# --------------------------------------------------------------------------

def test_zero_modes_is_bit_identical_to_plain_flux_solve():
    A, B, data, weight, _, _ = _linear_problem(n_modes=2)
    B0 = jnp.zeros((0,) + A.shape, dtype=A.dtype)
    flux, ped, w = FS.solve_group_fluxes_profile_w(A, B0, data, weight)
    ref = FS.solve_group_fluxes(A, data, weight, ridge=1e-6)
    np.testing.assert_array_equal(np.asarray(flux), np.asarray(ref))
    assert ped is None and w.shape == (data.shape[1], 0)


def test_lossweights_default_profile_w_is_off():
    lw = L.LossWeights()
    assert lw.profile_w is False
    assert lw.profile_w_iters == 2
    assert lw.ridge_w is None


def test_solve_block_model_off_path_is_bit_identical():
    """``_solve_block_model`` with ``profile_w=False`` must reproduce the
    pre-PW inline code exactly (both with and without the pedestal)."""
    A, _, data, weight, _, _ = _linear_problem(n_modes=1)
    for pedestal in (False, True):
        lw = L.LossWeights(stamp_pedestal=pedestal)
        model, w_solved = L._solve_block_model(A, None, data, weight, lw)
        assert w_solved is None
        ref = FS.solve_fluxes(
            A, data, weight, flux_objective="l2", ridge=lw.ridge,
            huber_delta=lw.huber_delta, irls_iterations=lw.huber_irls_iters,
            pedestal=pedestal,
        )
        if pedestal:
            ref_model = FS.model_stamps_with_pedestal(A, ref[0], ref[1])
        else:
            ref_model = FS.model_stamps(A, ref)
        np.testing.assert_array_equal(np.asarray(model), np.asarray(ref_model))


def test_forward_model_probe_baseline_matches_w_zero_render():
    """``return_mode_templates=True`` must return the ``w = 0`` render itself
    as ``templates`` -- bit-identically to a plain call with a zero override."""
    from test_fm_gpu_flux_export import _packed_bundle
    bundle = _packed_bundle(6)
    fd_params = {k: jnp.asarray(v) for k, v in bundle.params0.items()}
    ctx = _ctx_from_bundle(bundle)
    n_modes = int(np.asarray(fd_params["epsf_modes"]).shape[0])
    zero_w = jnp.zeros((int(ctx.wcs_frame_basis.shape[0]), n_modes), dtype=jnp.float32)
    t_ref, _, _, _ = L.forward_model(fd_params, ctx, w_of_t_override=zero_w)
    t_probe, _, _, _, modes_t = L.forward_model(fd_params, ctx, return_mode_templates=True)
    np.testing.assert_array_equal(np.asarray(t_ref), np.asarray(t_probe))
    assert modes_t.shape == (n_modes,) + t_ref.shape


def _ctx_from_bundle(bundle):
    groups = bundle.group_set()
    return L.build_static_context(
        cheb_static=bundle.cheb_static,
        wcs_frame_basis=bundle.wcs_frame_basis,
        w_frame_basis=bundle.w_frame_basis,
        epsf_grid=bundle.epsf_grid,
        groups=groups,
        ra=bundle.ra, dec=bundle.dec,
        stamp_center_x=bundle.stamp_center_x,
        stamp_center_y=bundle.stamp_center_y,
        t_exp_sec=bundle.t_exp_sec,
        stamp_snr_weight=bundle.stamp_snr_weight,
        fit_radius=bundle.fit_radius_stage23,
        stamp_active=np.ones((bundle.n_groups, bundle.n_frames), dtype=np.float32),
        x_lin=bundle.x_lin, y_lin=bundle.y_lin, cheb_basis=bundle.cheb_basis,
        pix_x=bundle.pix_x, pix_y=bundle.pix_y, pix_valid=bundle.pix_valid,
    )


# --------------------------------------------------------------------------
# 2/3/4. synthetic recovery: K modes, blended groups, pedestal on/off
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_modes", [1, 2])
@pytest.mark.parametrize("packed", [True, False])
def test_recovers_known_per_frame_amplitude(n_modes, packed):
    A, B, data, weight, w_true, f_true = _linear_problem(
        n_modes=n_modes, packed=packed, seed=n_modes,
    )
    flux, ped, w = FS.solve_group_fluxes_profile_w(A, B, data, weight, iterations=3)
    assert ped is None
    np.testing.assert_allclose(np.asarray(w), w_true, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(flux), f_true, atol=2e-4, rtol=2e-4)


def test_recovers_amplitude_with_blended_group_and_padding_slot():
    A, B, data, weight, w_true, f_true = _linear_problem(
        n_modes=2, n_members=3, blended_pad=True, seed=7,
    )
    flux, _, w = FS.solve_group_fluxes_profile_w(A, B, data, weight, iterations=3)
    np.testing.assert_allclose(np.asarray(w), w_true, atol=3e-4, rtol=3e-4)
    # Real members recovered; the all-zero padding slot stays at zero
    # (its normal-equation block is pure ridge).
    np.testing.assert_allclose(np.asarray(flux)[:, :, :2], f_true[:, :, :2], atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(np.asarray(flux)[0, :, 2], 0.0, atol=1e-6)


@pytest.mark.parametrize("pedestal", [False, True])
def test_recovers_amplitude_with_and_without_pedestal(pedestal):
    A, B, data, weight, w_true, f_true = _linear_problem(n_modes=2, seed=11)
    b_true = np.zeros(data.shape[:2], dtype=np.float32)
    if pedestal:
        rng = np.random.default_rng(12)
        b_true = rng.uniform(-1.5, 1.5, size=data.shape[:2]).astype(np.float32)
        data = data + jnp.asarray(b_true)[:, :, None]
    flux, ped, w = FS.solve_group_fluxes_profile_w(
        A, B, data, weight, iterations=3, pedestal=pedestal,
    )
    np.testing.assert_allclose(np.asarray(w), w_true, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(np.asarray(flux), f_true, atol=3e-4, rtol=3e-4)
    if pedestal:
        np.testing.assert_allclose(np.asarray(ped), b_true, atol=3e-4, rtol=3e-4)
    else:
        assert ped is None


def test_huber_irls_objective_also_recovers_the_amplitude():
    A, B, data, weight, w_true, _ = _linear_problem(n_modes=1, seed=13)
    _, _, w = FS.solve_group_fluxes_profile_w(
        A, B, data, weight, iterations=3,
        flux_objective="huber_irls", huber_delta=50.0, irls_iterations=2,
    )
    np.testing.assert_allclose(np.asarray(w), w_true, atol=1e-3, rtol=1e-3)


def test_schur_pieces_are_additive_over_group_chunks():
    """The whole reason ``profile_w_schur_pieces`` is public: a full-population
    solve accumulates it chunk by chunk."""
    A, B, data, weight, _, _ = _linear_problem(n_modes=2, seed=17)
    w0 = jnp.zeros((data.shape[1], 2), dtype=jnp.float32)
    S, r, _, _ = FS.profile_w_schur_pieces(A, B, data, weight, w0)
    S1, r1, _, _ = FS.profile_w_schur_pieces(A[:2], B[:, :2], data[:2], weight[:2], w0)
    S2, r2, _, _ = FS.profile_w_schur_pieces(A[2:], B[:, 2:], data[2:], weight[2:], w0)
    np.testing.assert_allclose(np.asarray(S), np.asarray(S1 + S2), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(r), np.asarray(r1 + r2), rtol=1e-4, atol=1e-4)


def test_templates_at_w_matches_explicit_sum():
    A, B, _, _, w_true, _ = _linear_problem(n_modes=2, seed=19)
    got = FS.templates_at_w(A, B, jnp.asarray(w_true))
    want = np.asarray(A) + np.einsum("fm,mgkfp->gkfp", w_true, np.asarray(B))
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# 5. gradients
# --------------------------------------------------------------------------

def test_gradient_through_profiled_solve_is_finite_and_nonzero():
    """The base/chroma leaves still need gradients, and the profiled ``w``
    depends on them -- this differentiates THROUGH the solve (no envelope
    shortcut), so the derivative must be finite and must actually move."""
    A, B, data, weight, _, _ = _linear_problem(n_modes=2, seed=23)

    def scalar(templates):
        flux, _, w = FS.solve_group_fluxes_profile_w(
            templates, B, data, weight, iterations=2,
        )
        return jnp.sum(flux ** 2) + jnp.sum(w ** 2)

    value, grad = jax.jit(jax.value_and_grad(scalar))(A)
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(grad)).all()
    assert np.any(np.asarray(grad) != 0.0)


def test_profiled_w_gradient_wrt_mode_templates_is_finite():
    A, B, data, weight, _, _ = _linear_problem(n_modes=1, seed=29)

    def scalar(mode_templates):
        _, _, w = FS.solve_group_fluxes_profile_w(A, mode_templates, data, weight, iterations=2)
        return jnp.sum(w ** 2)

    grad = jax.grad(scalar)(B)
    assert np.isfinite(np.asarray(grad)).all()
    assert np.any(np.asarray(grad) != 0.0)


def test_profiled_w_gradient_matches_finite_difference():
    """Differentiating through the solve (not the envelope theorem) is the
    documented choice; check it against a central finite difference of the
    same finite-iteration computation."""
    A, B, data, weight, _, _ = _linear_problem(n_modes=1, seed=31)
    direction = jnp.asarray(np.random.default_rng(0).normal(size=A.shape).astype(np.float32))

    def scalar(eps):
        _, _, w = FS.solve_group_fluxes_profile_w(
            A + eps * direction, B, data, weight, iterations=2,
        )
        return jnp.sum(w ** 2)

    analytic = float(jax.grad(scalar)(jnp.float32(0.0)))
    h = 1e-3
    numeric = float((scalar(jnp.float32(h)) - scalar(jnp.float32(-h))) / (2 * h))
    assert np.isfinite(analytic) and np.isfinite(numeric)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=1e-5)


# --------------------------------------------------------------------------
# 6. gauge
# --------------------------------------------------------------------------

def test_gauge_zero_time_mean_and_base_shift_leave_the_field_identical():
    rng = np.random.default_rng(41)
    n_modes, n_rows, n_cols, G = 2, 2, 2, 5
    base = jnp.asarray(rng.normal(size=(n_rows, n_cols, G, G)).astype(np.float32))
    modes = jnp.asarray(rng.normal(size=(n_modes, n_rows, n_cols, G, G)).astype(np.float32))
    w = jnp.asarray((rng.normal(size=(9, n_modes)) + 3.0).astype(np.float32))

    w_g, const = FS.gauge_zero_time_mean(w)
    np.testing.assert_allclose(np.asarray(jnp.mean(w_g, axis=0)), 0.0, atol=1e-5)
    base2 = FS.shift_base_by_modes(base, modes, const)

    # The rendered field at every frame must be unchanged: the constant went
    # into the base, nowhere else.
    field_before = base[None] + jnp.einsum("tk,kijxy->tijxy", w, modes)
    field_after = base2[None] + jnp.einsum("tk,kijxy->tijxy", w_g, modes)
    np.testing.assert_allclose(
        np.asarray(field_before), np.asarray(field_after), rtol=1e-4, atol=1e-4,
    )


def test_gauge_zero_time_mean_weighted():
    w = jnp.asarray(np.array([[0.0], [2.0], [4.0]], dtype=np.float32))
    fw = jnp.asarray(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    w_g, const = FS.gauge_zero_time_mean(w, fw)
    np.testing.assert_allclose(np.asarray(const), [2.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(w_g).ravel(), [-2.0, 0.0, 2.0], atol=1e-6)


# --------------------------------------------------------------------------
# 7. optimizer labels, and the loss/export round trip
# --------------------------------------------------------------------------

def test_freeze_w_labels_w_coeff_frozen():
    keys = ("wcs_coeff", "epsf_base_raw", "epsf_modes", "w_coeff")
    default = FIT._leaf_labels(3, freeze_wcs=False, param_keys=keys)
    frozen = FIT._leaf_labels(3, freeze_wcs=False, freeze_w=True, param_keys=keys)
    assert default["w_coeff"] == "train_w"
    assert frozen["w_coeff"] == "frozen"
    # Everything else is untouched by the new flag.
    for k in ("wcs_coeff", "epsf_base_raw", "epsf_modes"):
        assert default[k] == frozen[k]


def test_total_loss_profile_w_runs_and_ignores_w_coeff():
    """With PW on, perturbing ``w_coeff`` must not change the loss at all
    (the spline has left the model), while the loss itself stays finite."""
    from test_fm_gpu_flux_export import _packed_bundle
    bundle = _packed_bundle(6)
    ctx = _ctx_from_bundle(bundle)
    params = {k: jnp.asarray(v) for k, v in bundle.params0.items()}
    data = jnp.asarray(bundle.data)
    noise = jnp.asarray(bundle.noise)
    weight = jnp.asarray(bundle.weight)
    wsd = TB.second_difference_matrix(int(ctx.wcs_frame_basis.shape[1]))
    wwd = TB.second_difference_matrix(int(ctx.w_frame_basis.shape[1]))
    lw = L.LossWeights(profile_w=True, profile_w_iters=1)

    def loss_of(p):
        v, _ = L.total_loss(
            p, ctx, data, noise, weight, wsd, wwd,
            epsf_modes_init=jnp.asarray(bundle.epsf_modes), weights=lw,
        )
        return v

    v0 = float(loss_of(params))
    bumped = dict(params)
    bumped["w_coeff"] = params["w_coeff"] + 1.0
    v1 = float(loss_of(bumped))
    assert np.isfinite(v0)
    assert v0 == v1

    g = jax.grad(loss_of)(params)
    assert np.all(np.asarray(g["w_coeff"]) == 0.0)
    assert np.isfinite(np.asarray(g["epsf_base_raw"])).all()


def test_total_loss_chunked_and_unchunked_agree_under_profile_w():
    from test_fm_gpu_flux_export import _packed_bundle
    bundle = _packed_bundle(8)
    ctx = _ctx_from_bundle(bundle)
    params = {k: jnp.asarray(v) for k, v in bundle.params0.items()}
    args = (ctx, jnp.asarray(bundle.data), jnp.asarray(bundle.noise), jnp.asarray(bundle.weight),
            TB.second_difference_matrix(int(ctx.wcs_frame_basis.shape[1])),
            TB.second_difference_matrix(int(ctx.w_frame_basis.shape[1])))
    lw = L.LossWeights(profile_w=True, profile_w_iters=2)
    v_full, _ = L.total_loss(params, *args, epsf_modes_init=jnp.asarray(bundle.epsf_modes), weights=lw)
    v_chunk, _ = L.total_loss(params, *args, epsf_modes_init=jnp.asarray(bundle.epsf_modes),
                              weights=lw, stamp_chunk=4)
    np.testing.assert_allclose(float(v_full), float(v_chunk), rtol=2e-4, atol=1e-6)


def test_gpu_flux_export_round_trip_profile_w(tmp_path):
    pytest.importorskip("optax")
    from test_fm_gpu_flux_export import _packed_bundle, _run_export
    from syndiff_pipeline.forward_model import train_loop as TL

    n_t, n_want = 8, 4
    bundle = _packed_bundle(n_t)
    off = _run_export(bundle, tmp_path, n_want=n_want, gpu_flux_frame_block=n_want, tag="pw_off")
    assert "w_of_t" not in off.files

    out_on = tmp_path / "pw_on"
    TL.run_stages_from_bundle(
        bundle, out_dir=out_on, stage=1, start_stage=1,
        steps_per_stage=[0], lr_per_stage=[1e-2], log_every=1, checkpoint_every=0,
        reject_every=0, stage1_core_stamp=0, n_frames_per_stage=[n_want],
        gpu_flux_frame_block=n_want, profile_w=True,
    )
    on = np.load(out_on / "flux_solved.npz")
    assert "w_of_t" in on.files and "w_of_t_time_mean" in on.files
    n_modes = int(np.asarray(bundle.params0["epsf_modes"]).shape[0])
    assert on["w_of_t"].shape == (n_t, n_modes)
    assert np.isfinite(on["w_of_t"]).all()
    assert np.isfinite(on["flux"]).all()
    # Checkpoint round trip: params still load and still carry w_coeff.
    from syndiff_pipeline.forward_model import fit as _FIT
    reloaded = _FIT.load_params_npz(out_on / "params.npz")
    assert "w_coeff" in reloaded
    meta = (out_on / "fit_meta.json").read_text()
    assert '"profile_w": true' in meta
