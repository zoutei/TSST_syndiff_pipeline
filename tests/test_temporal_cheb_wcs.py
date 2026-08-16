import numpy as np
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import (
    TemporalChebWcs,
    fit_per_ffi_chebyshev,
    fit_temporal_coefficients,
)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_temporal_cheb_save_load_and_per_ffi_fit(tmp_path):
    w = _wcs()
    ra = np.linspace(9.98, 10.02, 60)
    dec = np.linspace(19.98, 20.02, 60)
    ref = TemporalChebWcs.from_reference_wcs(w, center=[100, 100], half_extents=[100, 100])
    x, y = ref.linear_pixels(ra, dec)
    x = x + 0.2
    y = y - 0.1
    fit = fit_per_ffi_chebyshev(w, ra, dec, x, y, center=[100, 100], half_extents=[100, 100])
    assert fit["keep_mask"].sum() >= 55
    assert np.median(fit["residual"][fit["keep_mask"]]) < 1e-6
    ref.coeff_matrix[:, 0] = np.r_[fit["coeff_x"], fit["coeff_y"]]
    path = tmp_path / "temporal.npz"
    ref.save(path)
    loaded = TemporalChebWcs.load(path)
    assert np.allclose(loaded.world_to_pixel_values(ra, dec, 0.5), ref.world_to_pixel_values(ra, dec, 0.5))


def test_temporal_coefficients_reject_nan_rows():
    t = np.linspace(0, 1, 8)
    values = np.column_stack([0.2 * t, -0.1 * t])
    values[3] = np.nan
    result = fit_temporal_coefficients(t, values, n_interior=1)
    assert not result["valid_mask"][3]
    assert result["coeff_matrix"].shape[0] == 2


def test_temporal_wcs_pixel_inverse_broadcasts_scalar_coordinate():
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    world = model.all_pix2world(np.array([0.0, 100.0, 200.0]), 0.0, 0.5)
    assert world.shape == (3, 2)
    assert np.isfinite(world).all()


def test_temporal_wcs_at_time_accepts_astropy_pixel_array_form():
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    world = model.at_time(0.5).all_pix2world(
        np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]]), 0
    )
    assert world.shape == (3, 2)
    assert np.isfinite(world).all()


def test_temporal_wcs_at_time_separate_arrays_return_astropy_tuple():
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    ra, dec = model.at_time(0.5).all_pix2world(
        np.array([0.0, 100.0, 200.0]), np.array([0.0, 100.0, 200.0]), 0
    )
    assert ra.shape == dec.shape == (3,)
    assert np.isfinite(ra).all() and np.isfinite(dec).all()


def test_temporal_wcs_pixel_inverse_round_trips_a_small_correction():
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    model.coeff_matrix[0, :] = 0.2
    model.coeff_matrix[model.n_terms, :] = -0.1
    ra = np.array([9.98, 10.0, 10.02])
    dec = np.array([19.98, 20.0, 20.02])
    x, y = model.world_to_pixel_values(ra, dec, 0.5)
    got_ra, got_dec = model.pixel_to_world(x, y, 0.5)
    assert np.allclose(got_ra, ra, atol=1e-8)
    assert np.allclose(got_dec, dec, atol=1e-8)


def test_temporal_wcs_pixel_to_world_to_pixel_round_trip():
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    model.coeff_matrix[0, :] = 0.2
    model.coeff_matrix[model.n_terms, :] = -0.1
    x = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
    y = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
    ra, dec = model.pixel_to_world(x, y, 0.5)
    got_x, got_y = model.world_to_pixel_values(ra, dec, 0.5)
    assert np.allclose(got_x, x, atol=1e-6)
    assert np.allclose(got_y, y, atol=1e-6)


def test_temporal_wcs_large_grid_fast_inverse_round_trip():
    """Large arrays use chunked, fully converged inversion."""
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    model.coeff_matrix[0, :] = 0.2
    model.coeff_matrix[model.n_terms, :] = -0.1
    x = np.linspace(0.0, 200.0, 100_001)
    y = np.linspace(200.0, 0.0, 100_001)
    ra, dec = model.pixel_to_world(x, y, 0.5)
    got_x, got_y = model.world_to_pixel_values(ra, dec, 0.5)
    assert np.max(np.hypot(got_x - x, got_y - y)) < 1e-6


def test_temporal_wcs_dense_detector_round_trip_is_fully_converged():
    """The inverse must converge well below the downstream pixel gate."""
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    model.coeff_matrix[0, :] = 0.2
    model.coeff_matrix[model.n_terms, :] = -0.1
    axis = np.linspace(0.0, 200.0, 17)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    ra, dec = model.pixel_to_world(xx, yy, 0.5)
    got_x, got_y = model.world_to_pixel_values(ra, dec, 0.5)
    assert np.max(np.hypot(got_x - xx, got_y - yy)) < 1e-9


def test_temporal_wcs_world_round_trip_is_fully_converged():
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100]
    )
    model.coeff_matrix[0, :] = 0.2
    model.coeff_matrix[model.n_terms, :] = -0.1
    x = np.linspace(0.0, 200.0, 31)
    y = np.linspace(200.0, 0.0, 31)
    ra, dec = model.pixel_to_world(x, y, 0.5)
    got_x, got_y = model.world_to_pixel_values(ra, dec, 0.5)
    got_ra, got_dec = model.pixel_to_world(got_x, got_y, 0.5)
    assert np.max(np.hypot(got_ra - ra, got_dec - dec)) < 1e-11
