"""Focused tests for the second-pass TessReduce-like estimator."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
    _fit_mask,
    _qe_spline_map,
    estimate_tessreduce_residual_background,
)


def test_fit_mask_accepts_only_clear_and_faint_catalog_pixels():
    mask = np.array([[0, 32, 4, 36, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(_fit_mask(mask), [[True, True, False, False, False]])


def test_qe_spline_only_changes_strap_columns():
    flux = np.ones((24, 4), dtype=float)
    background = np.ones_like(flux)
    mask = np.zeros_like(flux, dtype=np.uint8)
    mask[:, 2] = 4
    flux[:, 2] = 1.0 + 0.01 * np.arange(24)

    qe = _qe_spline_map(flux, background, mask)

    np.testing.assert_array_equal(qe[:, :2], 1.0)
    np.testing.assert_array_equal(qe[:, 3], 1.0)
    assert np.nanmedian(qe[:, 2]) > 1.0


def test_estimator_returns_finite_component_without_straps():
    image = np.full((32, 32), 2.0)
    mask = np.zeros_like(image, dtype=np.uint8)

    component, pre_qe, qe = estimate_tessreduce_residual_background(image, mask)

    assert np.isfinite(component).all()
    assert np.isfinite(pre_qe).all()
    np.testing.assert_array_equal(qe, 1.0)
