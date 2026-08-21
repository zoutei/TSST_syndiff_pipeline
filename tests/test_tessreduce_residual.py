"""Focused tests for the second-pass TessReduce-like estimator."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
    _accumulate_sep_object_mask,
    _fit_mask,
    _qe_spline_map,
    _sep_object_stamp_slices,
    estimate_tessreduce_residual_background,
    sanitize_boundary_outliers,
    smooth_bkg_decomposed,
)


def test_fit_mask_accepts_only_clear_and_faint_catalog_pixels():
    mask = np.array([[0, 32, 4, 36, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(_fit_mask(mask), [[True, True, False, False, False]])


def test_sanitize_boundary_outliers_flags_anomalous_rim_pixel():
    # A 20x20 field of ~0 with a masked 6x6 block in the middle; one rim
    # pixel just outside the mask is a huge outlier relative to its local
    # neighborhood and should be folded into the sanitized mask.
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=0.1, size=(20, 20))
    mask = np.zeros((20, 20), dtype=bool)
    mask[7:13, 7:13] = True
    outlier_rc = (6, 9)  # directly above the masked block
    data[outlier_rc] = 500.0

    sanitized = sanitize_boundary_outliers(data, mask, k=8, sigma_thresh=3.0, rim_width=1)

    assert sanitized[outlier_rc]
    assert not mask[outlier_rc]
    # Original mask footprint is preserved (only additive).
    assert (sanitized & mask == mask).all()


def test_sanitize_boundary_outliers_leaves_rim_alone_when_no_outlier():
    data = np.zeros((20, 20))
    mask = np.zeros((20, 20), dtype=bool)
    mask[7:13, 7:13] = True

    sanitized = sanitize_boundary_outliers(data, mask, k=8, sigma_thresh=3.0, rim_width=1)

    np.testing.assert_array_equal(sanitized, mask)


def test_smooth_bkg_decomposed_rejects_boundary_outlier_before_inpainting():
    rng = np.random.default_rng(1)
    data = rng.normal(loc=100.0, scale=0.05, size=(24, 24))
    mask = np.zeros((24, 24), dtype=bool)
    mask[9:15, 9:15] = True
    data[mask] = np.nan
    outlier_rc = (8, 12)
    data[outlier_rc] = 1.0e4

    filled_robust = smooth_bkg_decomposed(
        data.copy(), gauss_smooth=0.0, boundary_k=8, boundary_sigma=3.0, boundary_rim_width=1
    )

    assert np.isfinite(filled_robust).all()
    # The inpainted center should stay near the ~100 background level, not be
    # dragged toward the injected outlier.
    assert abs(filled_robust[11, 11] - 100.0) < 5.0


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


def test_force_anomaly_repair_runs_on_high_median_image():
    image = np.full((48, 48), 600.0)
    image[20:24, 20:24] += 80.0
    mask = np.zeros_like(image, dtype=np.uint8)

    component, pre_qe, qe = estimate_tessreduce_residual_background(
        image, mask, force_anomaly_repair=True
    )

    assert np.isfinite(component).all()
    assert np.isfinite(pre_qe).all()
    np.testing.assert_array_equal(qe, 1.0)


def _full_frame_sep_mask(obj, lap_sub, lap_err, noise):
    """Pre-optimization loop: full-CCD ellipse + distance map per object."""
    import sep

    ny, nx = lap_sub.shape
    yy, xx = np.mgrid[:ny, :nx]
    ap = np.zeros((ny, nx), dtype=bool)
    sep.mask_ellipse(ap, obj["x"], obj["y"], obj["a"], obj["b"], obj["theta"], r=3.0)
    sep_mask = np.zeros((ny, nx), dtype=bool)
    if not ap.sum() or (lap_sub / (lap_err + 1e-10))[ap].mean() <= 2.0:
        return sep_mask
    dist = np.sqrt((xx - obj["x"]) ** 2 + (yy - obj["y"]) ** 2)
    true_r = next(
        (
            r - 1
            for r in range(2, 20)
            if (dist >= r - 0.5).any()
            and lap_sub[(dist >= r - 0.5) & (dist < r + 0.5)].mean() < noise
        ),
        None,
    )
    if true_r is not None and 2 <= true_r <= 5:
        sep_mask |= dist <= true_r
    return sep_mask


def test_sep_object_mask_stamp_matches_full_frame():
    rng = np.random.default_rng(0)
    ny, nx = 128, 160
    lap_sub = rng.normal(0.0, 1.0, (ny, nx))
    yy, xx = np.mgrid[:ny, :nx]
    objects = [
        {"x": 40.2, "y": 50.7, "a": 1.4, "b": 1.1, "theta": 0.2},
        {"x": 3.0, "y": 4.0, "a": 2.0, "b": 1.5, "theta": 0.0},
        {"x": 155.4, "y": 120.1, "a": 1.8, "b": 1.6, "theta": 1.1},
    ]
    for obj in objects:
        blob = np.exp(-(((xx - obj["x"]) / 1.5) ** 2 + ((yy - obj["y"]) / 1.5) ** 2) / 2.0)
        lap_sub += 12.0 * blob
    lap_err = np.full((ny, nx), 1.0)
    noise = 1.0

    stamp = np.zeros((ny, nx), dtype=bool)
    ref = np.zeros((ny, nx), dtype=bool)
    for obj in objects:
        _accumulate_sep_object_mask(stamp, obj, lap_sub, lap_err, noise)
        ref |= _full_frame_sep_mask(obj, lap_sub, lap_err, noise)

    np.testing.assert_array_equal(stamp, ref)
    y0, y1, x0, x1 = _sep_object_stamp_slices(40.2, 50.7, 1.4, 1.1, ny, nx)
    assert (y1 - y0) < ny and (x1 - x0) < nx
