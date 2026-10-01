# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Focused tests for hierarchically standardized continuous stamp weights."""

from __future__ import annotations

from unittest import mock

import jax.numpy as jnp
import numpy as np

from syndiff_pipeline.forward_model import stamp_reject as SR
from test_fm_stamp_reject_v2 import _tiny_ctx_and_fd


def test_group_scales_shrink_short_series_more_than_long_series():
    rng = np.random.default_rng(12)
    d = np.vstack([
        rng.normal(0, 0.5, 100),
        rng.normal(0, 0.5, 100),
    ])
    pool = np.ones_like(d, dtype=bool)
    pool[0, 4:] = False
    scales, pooled, counts = SR.hierarchical_group_log_scales(
        d, pool, shrinkage_frames=20,
    )
    assert counts.tolist() == [4, 100]
    raw_short = SR.level2_mad_scale(d[0:1], pool[0:1])
    raw_long = SR.level2_mad_scale(d[1:2], pool[1:2])
    assert abs(scales[0] - pooled) < abs(raw_short - pooled)
    assert abs(scales[1] - raw_long) < abs(pooled - raw_long)


def test_continuous_weights_are_one_sided_and_never_zero():
    d = np.array([[-2.0, 0.0, 2.0, 4.0, 40.0]])
    weights = SR.one_sided_robust_stamp_weights(
        d, np.array([1.0]), n_sigma=3.0, min_weight=0.05,
    )
    np.testing.assert_allclose(weights[0, :3], 1.0)
    assert weights[0, 3] == np.float32(0.75)
    assert 0.0 < weights[0, 4] < weights[0, 3]


def test_standardized_gate_only_physical_baseline_hard_zeros():
    fd, params = _tiny_ctx_and_fd(1, 20)
    baseline = np.ones((1, 20), dtype=np.float32)
    baseline[0, 2] = 0.0
    chi2 = np.full((1, 20), 300.0)
    chi2[0, 5] = 30000.0

    with mock.patch.object(
        SR, "per_stamp_chi2_red",
        return_value=(jnp.asarray(chi2), jnp.full((1, 20), 100.0)),
    ):
        stats = SR.refresh_stamp_active_standardized(
            params, [fd], mask_baselines=[baseline], state=None,
            n_sigma=3.0, min_weight=0.05,
        )
    applied = np.asarray(fd.ctx.stamp_active)
    assert applied[0, 2] == 0.0
    assert 0.0 < applied[0, 5] < 1.0
    assert np.count_nonzero(applied == 0.0) == 1
    assert stats["n_rejected"] == 1.0
    assert stats["n_downweighted"] >= 1.0


def test_calibration_ignores_previous_residual_weights():
    fd, params = _tiny_ctx_and_fd(2, 30)
    chi2 = np.tile(300.0 * np.exp(np.linspace(-0.1, 0.1, 30)), (2, 1))
    chi2[0, 8] *= 8.0
    state = SR.StandardizedGateState()

    with mock.patch.object(
        SR, "per_stamp_chi2_red",
        return_value=(jnp.asarray(chi2), jnp.full((2, 30), 100.0)),
    ):
        first = SR.refresh_stamp_active_standardized(
            params, [fd], state=state, hysteresis_n=1,
        )
        # The previous residual decision is now present in ctx.stamp_active.
        # Repeating identical residuals must yield identical calibration.
        second = SR.refresh_stamp_active_standardized(
            params, [fd], state=state, hysteresis_n=1,
        )
    assert second["pooled_scale"] == first["pooled_scale"]
    assert second["median_group_scale"] == first["median_group_scale"]


def test_persistent_state_hysteresis_and_reentry():
    fd, params = _tiny_ctx_and_fd(1, 20)
    spiked = np.full((1, 20), 300.0)
    spiked[0, 5] = 30000.0
    clean = np.full((1, 20), 300.0)
    pix = jnp.full((1, 20), 100.0)
    state = SR.StandardizedGateState()

    with mock.patch.object(SR, "per_stamp_chi2_red", return_value=(jnp.asarray(spiked), pix)):
        SR.refresh_stamp_active_standardized(params, [fd], state=state, hysteresis_n=2)
        assert np.asarray(fd.ctx.stamp_active)[0, 5] == 1.0
        SR.refresh_stamp_active_standardized(params, [fd], state=state, hysteresis_n=2)
        assert 0.0 < np.asarray(fd.ctx.stamp_active)[0, 5] < 1.0

    with mock.patch.object(SR, "per_stamp_chi2_red", return_value=(jnp.asarray(clean), pix)):
        SR.refresh_stamp_active_standardized(params, [fd], state=state, hysteresis_n=2)
        assert np.asarray(fd.ctx.stamp_active)[0, 5] < 1.0
        SR.refresh_stamp_active_standardized(params, [fd], state=state, hysteresis_n=2)
        assert np.asarray(fd.ctx.stamp_active)[0, 5] == 1.0
