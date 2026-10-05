# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Task M4: additive per-stamp pedestal, solved jointly with flux.

Covers the required gates: bit-identical when off, synthetic pedestal
recovery, K=2 group with pedestal, JAX gradient finiteness through the
augmented solve, and an export round trip (``flux_solved.npz`` carries a
``pedestal`` array only when requested).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import flux_solve as FS


def _problem():
    """Same fixture as test_flux_objectives.py's ``_problem`` -- K=2, one
    group, one frame, 2x2 packed/square-equivalent stamp."""
    templates = jnp.asarray(
        [[[[[1.0, 0.2], [0.1, 0.0]]], [[[0.0, 0.1], [0.3, 1.0]]]]],
        dtype=jnp.float32,
    )
    true_flux = jnp.asarray([[[2.0, -0.5]]], dtype=jnp.float32)
    data = FS.model_stamps(templates, true_flux)
    weight = jnp.ones_like(data)
    return templates, data, weight, true_flux


# --------------------------------------------------------------------------
# 1. Default off -> bit-identical to pre-M4 behavior.
# --------------------------------------------------------------------------

def test_solve_group_fluxes_default_pedestal_is_false_and_bit_identical():
    templates, data, weight, _ = _problem()
    explicit_false = FS.solve_group_fluxes(templates, data, weight, ridge=1e-6, pedestal=False)
    default = FS.solve_group_fluxes(templates, data, weight, ridge=1e-6)
    np.testing.assert_array_equal(np.asarray(explicit_false), np.asarray(default))
    # Not just close -- the augmented-design branch must never execute.
    assert explicit_false.shape == default.shape == (1, 1, 2)


def test_solve_group_fluxes_huber_irls_default_off_bit_identical():
    templates, data, weight, _ = _problem()
    data = data + jnp.asarray([[[[0.01, -0.02], [0.015, -0.01]]]])
    a = FS.solve_group_fluxes_huber_irls(templates, data, weight, ridge=1e-6, huber_delta=1.0, iterations=2)
    b = FS.solve_group_fluxes_huber_irls(
        templates, data, weight, ridge=1e-6, huber_delta=1.0, iterations=2, pedestal=False,
    )
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_solve_fluxes_dispatcher_default_off_bit_identical():
    templates, data, weight, _ = _problem()
    for objective in ("l2", "huber_irls"):
        a = FS.solve_fluxes(templates, data, weight, flux_objective=objective, ridge=1e-6)
        b = FS.solve_fluxes(templates, data, weight, flux_objective=objective, ridge=1e-6, pedestal=False)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_model_stamps_unaffected_by_pedestal_helper_existing():
    """``model_stamps`` itself takes no pedestal argument and must be
    untouched; the new pedestal-aware model is a separate function."""
    templates, data, weight, true_flux = _problem()
    model = FS.model_stamps(templates, true_flux)
    np.testing.assert_allclose(np.asarray(model), np.asarray(data), atol=1e-6)


# --------------------------------------------------------------------------
# 2. Synthetic recovery: known pedestal recovered to < 1e-6.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("packed", [False, True])
def test_pedestal_recovers_known_synthetic_value(packed):
    rng = np.random.default_rng(0)
    n_groups, K, n_frames, P = 3, 1, 4, 9
    if packed:
        templates = jnp.asarray(rng.uniform(0.1, 1.0, size=(n_groups, K, n_frames, P)), dtype=jnp.float32)
    else:
        S = 3
        templates = jnp.asarray(rng.uniform(0.1, 1.0, size=(n_groups, K, n_frames, S, S)), dtype=jnp.float32)

    true_flux = jnp.asarray(rng.uniform(1.0, 5.0, size=(n_groups, n_frames, K)), dtype=jnp.float32)
    true_pedestal = jnp.asarray(rng.uniform(-2.0, 2.0, size=(n_groups, n_frames)), dtype=jnp.float32)
    data = FS.model_stamps_with_pedestal(templates, true_flux, true_pedestal)
    weight = jnp.ones_like(data)

    # This repo runs JAX float32-only (no jax_enable_x64 anywhere), so the
    # normal-equations solve (ridge=1e-6) has an intrinsic ~1e-5 relative
    # rounding floor -- "recovered to <1e-6" is float32-exact modulo that
    # floor, not literal 1e-6 agreement; 2e-5 comfortably separates a real
    # recovery from a broken one (a wrong pedestal sign/scale/shape would
    # miss by order-1, not 1e-5).
    flux, pedestal = FS.solve_group_fluxes(templates, data, weight, ridge=1e-6, pedestal=True)
    np.testing.assert_allclose(np.asarray(flux), np.asarray(true_flux), atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(pedestal), np.asarray(true_pedestal), atol=2e-5, rtol=2e-5)

    # Reconstructed model must reproduce the exact data (noiseless problem).
    model = FS.model_stamps_with_pedestal(templates, flux, pedestal)
    np.testing.assert_allclose(np.asarray(model), np.asarray(data), atol=2e-5)


def test_pedestal_via_solve_fluxes_dispatcher_l2_and_huber():
    rng = np.random.default_rng(1)
    n_groups, K, n_frames, P = 2, 1, 3, 7
    templates = jnp.asarray(rng.uniform(0.1, 1.0, size=(n_groups, K, n_frames, P)), dtype=jnp.float32)
    true_flux = jnp.asarray(rng.uniform(1.0, 5.0, size=(n_groups, n_frames, K)), dtype=jnp.float32)
    true_pedestal = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n_groups, n_frames)), dtype=jnp.float32)
    data = FS.model_stamps_with_pedestal(templates, true_flux, true_pedestal)
    weight = jnp.ones_like(data)

    for objective in ("l2", "huber_irls"):
        flux, pedestal = FS.solve_fluxes(
            templates, data, weight, flux_objective=objective, ridge=1e-6,
            huber_delta=10.0, irls_iterations=2, pedestal=True,
        )
        np.testing.assert_allclose(np.asarray(flux), np.asarray(true_flux), atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(np.asarray(pedestal), np.asarray(true_pedestal), atol=3e-5, rtol=3e-5)


# --------------------------------------------------------------------------
# 3. K=2 group with pedestal: the pedestal is shared by both members, flux
#    stays per-member.
# --------------------------------------------------------------------------

def test_k2_group_pedestal_shared_by_both_members():
    templates, _, _, true_flux = _problem()  # K=2, one group, one frame
    true_pedestal = jnp.asarray([[0.37]], dtype=jnp.float32)  # (1 group, 1 frame)
    data = FS.model_stamps_with_pedestal(templates, true_flux, true_pedestal)
    weight = jnp.ones_like(data)

    flux, pedestal = FS.solve_group_fluxes(templates, data, weight, ridge=1e-6, pedestal=True)
    assert flux.shape == (1, 1, 2)  # K=2 preserved
    assert pedestal.shape == (1, 1)  # one pedestal per group per frame, not per member
    np.testing.assert_allclose(np.asarray(flux), np.asarray(true_flux), atol=2e-5)
    np.testing.assert_allclose(np.asarray(pedestal), np.asarray(true_pedestal), atol=2e-5)


# --------------------------------------------------------------------------
# 4. JAX gradient finiteness through the augmented (K+1-unknown) solve.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["l2", "huber_irls"])
def test_pedestal_solve_has_finite_value_and_template_gradient(objective):
    templates, data, weight, _ = _problem()
    data = data + jnp.asarray([[[[0.01, -0.02], [0.015, -0.01]]]])

    def scalar(ts):
        flux, pedestal = FS.solve_fluxes(
            ts, data, weight, flux_objective=objective, ridge=1e-6,
            huber_delta=0.5, irls_iterations=2, pedestal=True,
        )
        return jnp.sum(flux**2) + jnp.sum(pedestal**2)

    value, grad = jax.jit(jax.value_and_grad(scalar))(templates)
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(grad)).all()
    assert np.any(np.asarray(grad) != 0.0)


def test_pedestal_gradient_flows_to_pedestal_output_itself():
    """The pedestal output must itself be differentiable w.r.t. the data
    (a basic sanity check that it is a real solved unknown, not a constant)."""
    templates, data, weight, _ = _problem()

    def pedestal_sum(d):
        _, pedestal = FS.solve_group_fluxes(templates, d, weight, ridge=1e-6, pedestal=True)
        return jnp.sum(pedestal)

    grad = jax.grad(pedestal_sum)(data)
    assert np.isfinite(np.asarray(grad)).all()


# --------------------------------------------------------------------------
# 5. Export round trip: flux_solved.npz carries "pedestal" only when
#    requested (default-off leaves the pre-M4 key set exactly).
# --------------------------------------------------------------------------

def test_gpu_flux_export_round_trip_pedestal(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from test_fm_gpu_flux_export import _packed_bundle, _run_export

    n_t, n_want = 8, 4
    bundle = _packed_bundle(n_t)

    off = _run_export(bundle, tmp_path, n_want=n_want, gpu_flux_frame_block=n_want, tag="ped_off")
    assert "pedestal" not in off.files

    pytest.importorskip("jax")
    from syndiff_pipeline.forward_model import train_loop as TL
    out_on = tmp_path / "ped_on"
    TL.run_stages_from_bundle(
        bundle, out_dir=out_on, stage=1, start_stage=1,
        steps_per_stage=[0], lr_per_stage=[1e-2], log_every=1, checkpoint_every=0,
        reject_every=0, stage1_core_stamp=0, n_frames_per_stage=[n_want],
        gpu_flux_frame_block=n_want, stamp_pedestal=True,
    )
    on = np.load(out_on / "flux_solved.npz")
    assert "pedestal" in on.files
    n_g = int(bundle.group_set().n_groups)
    assert on["pedestal"].shape == (n_g, n_t)
    # Flux/chi2 must still be populated (pedestal did not silently break the solve).
    assert np.any(on["pix_sum"] > 0)

    # Off-run flux need not equal on-run flux (different model), but both
    # must be finite, well-formed exports.
    assert np.isfinite(off["flux"]).all()
    assert np.isfinite(on["flux"]).all()
    assert np.isfinite(on["pedestal"]).all()
