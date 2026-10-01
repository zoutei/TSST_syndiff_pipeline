"""Focused tests for the second-pass TessReduce-like estimator."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from skimage import restoration as inpaint

from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
    _accumulate_sep_object_mask,
    _fit_mask,
    parse_star_wing_radii,
    star_wing_exclusion,
    _qe_spline_map,
    _sep_object_stamp_slices,
    estimate_tessreduce_residual_background,
    harmonic_inpaint,
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


def _local_rim(mask, iterations=1):
    from scipy.ndimage import binary_dilation

    return binary_dilation(mask, iterations=iterations) & ~mask


def test_harmonic_inpaint_obeys_maximum_principle_where_biharmonic_overshoots():
    # Smooth "sky" trend plus genuine per-pixel scatter (the real-world
    # regime: rim values carry real point-to-point gradient/noise, not a
    # perfectly flat plateau) with a large hole. This is the failure mode
    # identified in the investigation: skimage's biharmonic inpainting
    # matches boundary value AND slope with no maximum principle, so for a
    # large hole it can extrapolate past the bounds set by its own rim; the
    # exact harmonic solve is a weighted average of its boundary and can
    # never do so.
    rng = np.random.default_rng(20)
    ny, nx = 80, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    trend = 100.0 + 0.1 * xx - 0.05 * yy
    data = trend + rng.normal(0.0, 1.5, size=(ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[15:65, 15:65] = True

    rim = _local_rim(mask)
    lo, hi = data[rim].min(), data[rim].max()

    filled_h = harmonic_inpaint(data.copy(), mask)
    filled_b = np.asarray(inpaint.inpaint_biharmonic(data.copy(), mask), dtype=np.float64)

    # Harmonic: strictly bounded (to numerical CG tolerance) by the local rim.
    assert filled_h[mask].max() <= hi + 1e-4
    assert filled_h[mask].min() >= lo - 1e-4

    # Biharmonic: on this same input, it overshoots that same rim range by a
    # clear margin (this is the artefact being fixed).
    overshoot = max(0.0, filled_b[mask].max() - hi, lo - filled_b[mask].min())
    assert overshoot > 0.5


def test_harmonic_inpaint_matches_dense_laplace_solve():
    # Small frame, a few holes (none touching the frame border, so all are
    # ordinary interior Dirichlet problems) -- solve the identical discrete
    # 5-point Laplacian system independently with a dense direct solve and
    # check harmonic_inpaint agrees to ~1e-6.
    rng = np.random.default_rng(7)
    ny, nx = 40, 40
    data = rng.normal(50.0, 5.0, size=(ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[5:10, 5:9] = True
    mask[20:26, 15:22] = True
    mask[30:33, 30:36] = True

    filled = harmonic_inpaint(data.copy(), mask)

    invalid = mask
    coords = np.argwhere(invalid)
    n = coords.shape[0]
    id_map = -np.ones((ny, nx), dtype=np.int64)
    id_map[invalid] = np.arange(n)
    A_dense = np.zeros((n, n))
    b = np.zeros(n)
    for k, (y, x) in enumerate(coords):
        deg = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yn, xn = y + dy, x + dx
            if not (0 <= yn < ny and 0 <= xn < nx):
                continue
            deg += 1
            if invalid[yn, xn]:
                A_dense[k, id_map[yn, xn]] -= 1.0
            else:
                b[k] += data[yn, xn]
        A_dense[k, k] += deg
    sol_dense = np.linalg.solve(A_dense, b)

    np.testing.assert_allclose(filled[invalid], sol_dense, atol=1e-6, rtol=1e-6)


def test_harmonic_inpaint_isolated_component_uses_frame_median_no_nan():
    # A wholly-invalid array has no valid pixel anywhere for the mask to
    # reach (the pathological "isolated" case the docstring calls out) --
    # every masked pixel is dropped from the linear system and filled with
    # the frame median of valid pixels, which here is 0.0 since there are
    # none. Must not raise, divide-by-zero, or leave any NaN/inf behind.
    image = np.full((10, 10), 12345.0)
    mask = np.ones((10, 10), dtype=bool)

    out = harmonic_inpaint(image, mask)

    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out, 0.0)


def test_smooth_bkg_decomposed_fill_method_dispatch():
    rng = np.random.default_rng(1)
    data = rng.normal(loc=100.0, scale=0.05, size=(24, 24))
    mask = np.zeros((24, 24), dtype=bool)
    mask[9:15, 9:15] = True
    data[mask] = np.nan
    outlier_rc = (8, 12)
    data[outlier_rc] = 1.0e4

    default_result = smooth_bkg_decomposed(
        data.copy(), gauss_smooth=0.0, boundary_k=8, boundary_sigma=3.0, boundary_rim_width=1
    )
    harmonic_result = smooth_bkg_decomposed(
        data.copy(), gauss_smooth=0.0, boundary_k=8, boundary_sigma=3.0, boundary_rim_width=1,
        fill_method="harmonic",
    )
    np.testing.assert_array_equal(default_result, harmonic_result)

    biharmonic_result = smooth_bkg_decomposed(
        data.copy(), gauss_smooth=0.0, boundary_k=8, boundary_sigma=3.0, boundary_rim_width=1,
        fill_method="biharmonic",
    )

    # Reproduce the exact pre-existing biharmonic code path by hand and
    # check bit-for-bit agreement.
    invalid_mask = np.isnan(data)
    safe_invalid_mask = sanitize_boundary_outliers(
        np.nan_to_num(data, nan=0.0), invalid_mask, k=8, sigma_thresh=3.0, rim_width=1
    )
    fill_input = data.copy()
    fill_input[safe_invalid_mask] = np.nan
    expected_biharmonic = np.asarray(
        inpaint.inpaint_biharmonic(np.nan_to_num(fill_input, nan=0.0), safe_invalid_mask),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(biharmonic_result, expected_biharmonic)

    # The two fill methods actually differ here (otherwise this test would
    # not be exercising anything real).
    assert not np.array_equal(harmonic_result, biharmonic_result)

    with pytest.raises(ValueError):
        smooth_bkg_decomposed(data.copy(), fill_method="not_a_real_method")


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


def test_fit_mask_star_pad_grows_only_star_masks():
    mask = np.zeros((21, 21), dtype=np.uint8)
    mask[10, 10] = 2          # one SAT_CROSS (bit 2) pixel
    mask[0:2, :] = 4          # strap rows: not a star mask, must not grow
    mask[20, 20] = 32         # faint square pixel stays a fit pixel unless inside the pad
    base = _fit_mask(mask)
    padded = _fit_mask(mask, star_mask_pad_px=3)
    yy, xx = np.mgrid[:21, :21]
    disk = (yy - 10) ** 2 + (xx - 10) ** 2 <= 9
    np.testing.assert_array_equal(padded, base & ~disk)
    np.testing.assert_array_equal(_fit_mask(mask, star_mask_pad_px=0), base)
    assert padded[20, 20] and padded[2, 10] == base[2, 10]


def test_star_mask_pad_removes_wing_lift_under_masked_star():
    # flat sky 1.0 + a star wing that extends past its mask circle: without padding the gap fill is solved
    # from the wing-carrying rim and the background under the star is lifted; padding past the wing removes it.
    n = 81
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - 40, xx - 40)
    image = 1.0 + 2.0 * np.exp(-r / 2.5)
    mask = np.where(r <= 6, 2, 0).astype(np.uint8)
    lifted, _, _ = estimate_tessreduce_residual_background(image, mask)
    padded, _, _ = estimate_tessreduce_residual_background(image, mask, star_mask_pad_px=8)
    assert lifted[40, 40] - 1.0 > 0.05
    assert abs(padded[40, 40] - 1.0) < 0.5 * (lifted[40, 40] - 1.0)


def test_star_wing_exclusion_radius_by_magnitude():
    radii = [[9.0, 5], [12.0, 3], [13.0, 1]]
    x = np.array([10.0, 30.0, 50.0, 70.0])
    y = np.array([10.0, 10.0, 10.0, 10.0])
    t = np.array([8.5, 9.0, 12.5, 13.0])   # 9.0 falls in the 12.0 row; 13.0 is past the last mag_hi -> no disk
    ex = star_wing_exclusion((21, 81), x, y, t, radii)
    yy, xx = np.mgrid[:21, :81]
    want = np.zeros((21, 81), bool)
    for xi, r in ((10, 5), (30, 3), (50, 1)):
        want |= (xx - xi) ** 2 + (yy - 10) ** 2 <= r * r
    np.testing.assert_array_equal(ex, want)


def test_star_wing_exclusion_clips_at_edges_and_skips_nan():
    ex = star_wing_exclusion((10, 10), np.array([0.0, np.nan, -3.0]), np.array([0.0, 5.0, -3.0]),
                             np.array([10.0, 10.0, 10.0]), [[11.0, 4]])
    yy, xx = np.mgrid[:10, :10]
    want = (xx ** 2 + yy ** 2 <= 16) | ((xx + 3) ** 2 + (yy + 3) ** 2 <= 16)
    np.testing.assert_array_equal(ex, want)


@pytest.mark.parametrize("bad", [[], [[10.0, 5], [9.0, 3]], [[10.0, 0]], [[10.0, 2.5]], [[10.0]]])
def test_parse_star_wing_radii_rejects_bad_tables(bad):
    with pytest.raises(ValueError):
        parse_star_wing_radii(bad)


def test_fit_mask_extra_exclude_and_estimator_wing_lift():
    mask = np.zeros((21, 21), dtype=np.uint8)
    extra = np.zeros((21, 21), bool)
    extra[5:8, 5:8] = True
    np.testing.assert_array_equal(_fit_mask(mask, extra_exclude=extra), ~extra)
    with pytest.raises(ValueError):
        _fit_mask(mask, extra_exclude=np.zeros((5, 5), bool))
    # same wing-lift case as the pad test: a magnitude-sized disk past the wing removes the lift
    n = 81
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - 40, xx - 40)
    image = 1.0 + 2.0 * np.exp(-r / 2.5)
    mask = np.where(r <= 6, 2, 0).astype(np.uint8)
    lifted, _, _ = estimate_tessreduce_residual_background(image, mask)
    disk = star_wing_exclusion((n, n), np.array([40.0]), np.array([40.0]), np.array([9.0]), [[13.0, 14]])
    fixed, _, _ = estimate_tessreduce_residual_background(image, mask, extra_exclude=disk)
    assert lifted[40, 40] - 1.0 > 0.05
    assert abs(fixed[40, 40] - 1.0) < 0.5 * (lifted[40, 40] - 1.0)


def test_extra_exclude_follows_linear_pad_like_the_residual_mask():
    """kernel_fit pads the residual mask to the Hotpants support (linear mode: constant True margin); the
    star-wing exclusion must be padded the same way or _fit_mask rejects it (2048 vs 2064 on the paper lanes)."""
    import numpy as np
    from syndiff_pipeline.difference_imaging.stages.hotpants import _pair_hotpants_inputs
    from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import _fit_mask, star_wing_exclusion

    n, pad = 64, 8
    sci = np.zeros((n, n)); err = np.ones((n, n)); tmpl = np.zeros((n + 2 * pad, n + 2 * pad))
    mask = np.zeros((n, n), dtype=np.int16)
    ex = star_wing_exclusion((n, n), np.array([30.0]), np.array([20.0]), np.array([9.0]), [[13.0, 5]])
    _, _, _, mask_p, _ = _pair_hotpants_inputs(sci, tmpl, err, mask, None, pad)
    _, _, _, ex_p, _ = _pair_hotpants_inputs(sci, tmpl, err, ex, None, pad)
    ex_p = np.asarray(ex_p, dtype=bool)
    assert ex_p.shape == np.asarray(mask_p).shape == (n + 2 * pad, n + 2 * pad)
    assert np.array_equal(ex_p[pad:-pad, pad:-pad], ex) and ex_p[:pad].all() and ex_p[:, :pad].all()
    fit = _fit_mask(mask_p, 0, ex_p)
    assert not fit[pad + 20, pad + 30] and fit.shape == ex_p.shape


def test_star_wing_exclusion_follows_field_mode_padding():
    # kernel_fit pads the background-fit exclusion with the same pairing as the residual mask; in field mode
    # (MappingGrid) the science crop sits inside the template support and every fabricated pixel is excluded.
    from syndiff_pipeline.common.mapping_grid import MappingGrid
    from syndiff_pipeline.difference_imaging.stages.hotpants import _pair_hotpants_inputs

    grid = MappingGrid.from_ffi_shape(2048, 2048)
    sshape = grid.science_ffi_bounds()["shape"]
    tshape = grid.template_ffi_bounds()["shape"]
    ex = star_wing_exclusion(sshape, np.array([100.0]), np.array([200.0]), np.array([10.0]), [[13.0, 9]])
    sci = np.zeros(sshape)
    tmpl = np.zeros(tshape)
    _, _, _, ex_p, _ = _pair_hotpants_inputs(sci, tmpl, sci, ex, grid, 0)
    ex_p = np.asarray(ex_p, dtype=bool)
    assert ex_p.shape == tuple(tshape)
    ys, xs = grid.science_slice_native()
    np.testing.assert_array_equal(ex_p[ys, xs], ex)
    pad = np.ones(tshape, bool)
    pad[ys, xs] = False
    assert ex_p[pad].all()
    mask = np.zeros(tshape, dtype=np.int16)
    fit = _fit_mask(mask, extra_exclude=ex_p)
    assert not fit[pad].any() and fit[ys, xs].sum() == (~ex).sum()
