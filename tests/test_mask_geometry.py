"""Geometry YAML and radius helpers."""

from syndiff_pipeline.masking.geometry import (
    cross_geometry_from_mag,
    empirical_circle_radius,
    load_geometry,
    radius_from_mag,
)


def test_load_packaged_geometry():
    geo = load_geometry()
    assert len(geo["circle_bins"]) >= 9
    assert geo["circle_mag_min"] == 9.0


def test_empirical_circle_radius_bins():
    assert empirical_circle_radius(11.5) == 7
    assert empirical_circle_radius(8.0) == 0  # crosses, not circles
    assert empirical_circle_radius(11.5, scale=2.0) == 14


def test_radius_from_mag_shared_tns_asteroid():
    # bright uses cross body
    assert radius_from_mag(5.0) >= 20
    # mid catalog
    assert radius_from_mag(12.5) == 6
    # faint default
    assert radius_from_mag(19.0) == 2


def test_cross_geometry():
    b, L, w = cross_geometry_from_mag(5.5)
    assert b > 0 and L > 0 and w > 0
    b0, L0, w0 = cross_geometry_from_mag(10.0)
    assert (b0, L0, w0) == (0, 0, 0)
