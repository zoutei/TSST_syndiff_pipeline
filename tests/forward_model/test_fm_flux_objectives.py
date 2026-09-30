# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Objective-aligned, differentiable profile-flux solve tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import flux_solve as FS


def _problem():
    templates = jnp.asarray(
        [[[[[1.0, 0.2], [0.1, 0.0]]], [[[0.0, 0.1], [0.3, 1.0]]]]],
        dtype=jnp.float32,
    )
    true_flux = jnp.asarray([[[2.0, -0.5]]], dtype=jnp.float32)
    data = FS.model_stamps(templates, true_flux)
    weight = jnp.ones_like(data)
    return templates, data, weight, true_flux


def test_huber_irls_matches_l2_in_quadratic_regime():
    templates, data, weight, _ = _problem()
    data = data + jnp.asarray([[[[0.01, -0.02], [0.015, -0.01]]]])
    l2 = FS.solve_fluxes(templates, data, weight, flux_objective="l2", ridge=1e-6)
    huber = FS.solve_fluxes(
        templates, data, weight, flux_objective="huber_irls",
        huber_delta=10.0, irls_iterations=2, ridge=1e-6,
    )
    np.testing.assert_allclose(huber, l2, rtol=2e-6, atol=2e-6)


def test_huber_irls_improves_flux_with_outlier():
    templates = jnp.ones((1, 1, 1, 9), dtype=jnp.float32)
    data = jnp.ones((1, 1, 9), dtype=jnp.float32) * 3.0
    data = data.at[..., 0].set(30.0)
    weight = jnp.ones_like(data)
    l2 = FS.solve_fluxes(templates, data, weight, flux_objective="l2")
    huber = FS.solve_fluxes(
        templates, data, weight, flux_objective="huber_irls",
        huber_delta=1.0, irls_iterations=4,
    )
    assert abs(float(huber[0, 0, 0]) - 3.0) < abs(float(l2[0, 0, 0]) - 3.0)
    assert float(huber[0, 0, 0]) < 3.2


def test_huber_irls_square_and_packed_are_equivalent():
    templates, data, weight, _ = _problem()
    kwargs = dict(flux_objective="huber_irls", huber_delta=0.5, irls_iterations=3)
    square = FS.solve_fluxes(templates, data, weight, **kwargs)
    packed = FS.solve_fluxes(
        templates.reshape(1, 2, 1, 4), data.reshape(1, 1, 4),
        weight.reshape(1, 1, 4), **kwargs,
    )
    np.testing.assert_allclose(square, packed, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("objective", ["l2", "huber_irls"])
def test_flux_objective_has_finite_value_and_template_gradient(objective):
    templates, data, weight, _ = _problem()

    def scalar(ts):
        flux = FS.solve_fluxes(
            ts, data, weight, flux_objective=objective,
            huber_delta=0.5, irls_iterations=2,
        )
        return jnp.sum(flux**2)

    value, grad = jax.jit(jax.value_and_grad(scalar))(templates)
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(grad)).all()


def test_flux_objective_validation():
    templates, data, weight, _ = _problem()
    with pytest.raises(ValueError, match="unknown flux_objective"):
        FS.solve_fluxes(templates, data, weight, flux_objective="cauchy")


def test_cli_hyphenated_huber_objective_alias():
    templates, data, weight, _ = _problem()
    kwargs = dict(huber_delta=0.5, irls_iterations=2)
    hyphenated = FS.solve_fluxes(
        templates, data, weight, flux_objective="huber-irls", **kwargs,
    )
    internal = FS.solve_fluxes(
        templates, data, weight, flux_objective="huber_irls", **kwargs,
    )
    np.testing.assert_allclose(hyphenated, internal, rtol=1e-6, atol=1e-6)
