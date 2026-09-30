# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for ``flux_solve.enclosed_flux_renormalize`` (task T1-3)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from syndiff_pipeline.forward_model import flux_solve as FS


def test_no_op_when_footprint_sums_are_equal():
    flux = jnp.asarray([100.0, 200.0, 300.0])
    same = jnp.asarray([0.9, 0.9, 0.9])
    out = FS.enclosed_flux_renormalize(flux, same, same)
    np.testing.assert_allclose(np.asarray(out), np.asarray(flux))


def test_renormalizes_proportionally_to_footprint_ratio():
    flux = jnp.asarray([100.0, 100.0])
    template_sum = jnp.asarray([0.81, 0.90])  # narrower at t=0 -> less enclosed
    base_sum = jnp.asarray([0.90, 0.90])
    out = np.asarray(FS.enclosed_flux_renormalize(flux, template_sum, base_sum))
    # t=0: template encloses less than base -> renormalized flux scaled DOWN
    assert out[0] < 100.0
    np.testing.assert_allclose(out[0], 100.0 * 0.81 / 0.90)
    # t=1: ratio is 1 -> unchanged
    np.testing.assert_allclose(out[1], 100.0)


def test_broadcasts_static_base_footprint_sum():
    flux = jnp.asarray([[10.0, 20.0, 30.0]])  # (1 group, 3 frames)
    template_sum = jnp.asarray([[0.8, 0.9, 1.0]])
    base_sum = jnp.asarray([0.9])  # per-group static, broadcasts over frames
    out = np.asarray(FS.enclosed_flux_renormalize(flux, template_sum, base_sum[:, None]))
    expected = np.asarray(flux) * np.asarray(template_sum) / 0.9
    np.testing.assert_allclose(out, expected)


def test_removes_a_pure_normalization_drift_by_construction():
    """If a star's TRUE flux is constant and the profile-fit flux drifts
    purely because the footprint-enclosed template normalization drifts
    (mechanism A), renormalizing recovers the constant value exactly.
    """
    true_flux = 500.0
    drift = np.array([0.97, 1.0, 1.03, 0.99])  # template_footprint_sum / base_footprint_sum
    base_sum = 0.85
    template_sum = drift * base_sum
    observed_flux = jnp.asarray(true_flux / drift)  # what mechanism (A) alone would report
    out = np.asarray(FS.enclosed_flux_renormalize(
        observed_flux, jnp.asarray(template_sum), jnp.asarray(base_sum)
    ))
    np.testing.assert_allclose(out, true_flux, rtol=1e-5)
