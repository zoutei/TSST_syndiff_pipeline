"""Geometry YAML and radius helpers."""

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking.geometry import (
    big_sat_empirical,
    cross_geometry_from_mag,
    empirical_circle_radius,
    load_geometry,
    radius_from_mag,
    size_limit,
)


def test_load_packaged_geometry():
    geo = load_geometry()
    assert len(geo["circle_bins"]) >= 9
    assert geo["circle_mag_min"] == 7.5


def test_empirical_circle_radius_bins():
    assert empirical_circle_radius(11.5) == 7
    assert empirical_circle_radius(7.0) == 0  # very bright: crosses, not circles
    assert empirical_circle_radius(8.0) == 9  # mid-bright tier circles
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
    b0, L0, w0 = cross_geometry_from_mag(8.0)
    assert (b0, L0, w0) == (0, 0, 0)


def test_size_limit_margin_admits_off_array_rows():
    image = np.zeros((100, 100))
    x = np.array([-30, 5, 130])
    y = np.array([50, 5, 50])
    assert list(size_limit(x, y, image)) == [False, True, False]
    assert list(size_limit(x, y, image, margin=40)) == [True, True, True]


def test_big_sat_empirical_paints_from_star_just_outside_array():
    """A very bright star centered just outside the array must still paint in.

    Regression for the shared-mask boundary bug: a T<4 star's cross (arm
    length ~97 px) painted from a center a few px outside a diff crop should
    still leave marked pixels inside the crop.
    """
    image = np.zeros((200, 200))
    table = pd.DataFrame({"x": [-5.0], "y": [100.0], "mag": [3.0]})

    # Without a margin the star (x=-5, just outside the array) would be
    # dropped before painting; confirm size_limit itself would reject it.
    assert not size_limit(np.array([-5]), np.array([100]), image).any()

    mask = big_sat_empirical(table, image)
    assert mask.sum() > 0
    assert mask[100, 0] == 1  # cross arm reaches the crop's left edge
