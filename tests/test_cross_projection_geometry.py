"""Shared producer/consumer geometry contract for cross-projection padding."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing.cross_projection_geometry import (
    EDGE_EXCLUSION,
    PAD_SIZE,
    intersect_bounds,
    padding_patch_geometry,
    padding_work_image_geometry,
)
from syndiff_pipeline.template_creation.processing.cross_projection_padding import (
    CELL_OVERLAP,
    create_padding_wcs,
)
from syndiff_pipeline.template_creation.processing.padding_correction import (
    _standalone_padding_wcs,
)

W, H = 6481, 6429
PATCH = PAD_SIZE + EDGE_EXCLUSION


@pytest.fixture
def wcs() -> WCS:
    value = WCS(naxis=2)
    value.wcs.crval = [180.0, -30.0]
    value.wcs.crpix = [W / 2, H / 2]
    value.wcs.cd = [[-1.0 / 3600, 0.0], [0.0, 1.0 / 3600]]
    value.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return value


@pytest.mark.parametrize(
"location,bounds,shape",
[
    ("top", (H - EDGE_EXCLUSION, H + PAD_SIZE, 0, W), (PATCH, W)),
    ("bottom", (-PAD_SIZE, EDGE_EXCLUSION, 0, W), (PATCH, W)),
    ("left", (0, H, -PAD_SIZE, EDGE_EXCLUSION), (H, PATCH)),
    ("right", (0, H, W - EDGE_EXCLUSION, W + PAD_SIZE), (H, PATCH)),
    ("top_left", (H - EDGE_EXCLUSION, H + PAD_SIZE, -PAD_SIZE, EDGE_EXCLUSION), (PATCH, PATCH)),
    ("top_right", (H - EDGE_EXCLUSION, H + PAD_SIZE, W - EDGE_EXCLUSION, W + PAD_SIZE), (PATCH, PATCH)),
    ("bottom_left", (-PAD_SIZE, EDGE_EXCLUSION, -PAD_SIZE, EDGE_EXCLUSION), (PATCH, PATCH)),
    ("bottom_right", (-PAD_SIZE, EDGE_EXCLUSION, W - EDGE_EXCLUSION, W + PAD_SIZE), (PATCH, PATCH)),
],
)
def test_recipient_native_bounds_and_shape(wcs, location, bounds, shape):
    geometry = padding_patch_geometry(
        wcs,
        recipient_x0=0,
        recipient_y0=0,
        cell_width=W,
        cell_height=H,
        location=location,
    )
    assert geometry.bounds == bounds
    assert geometry.shape == shape


@pytest.mark.parametrize("location", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_live_producer_and_standalone_share_corner_geometry(wcs, location):
    config = SimpleNamespace(cell_width=W, cell_height=H)
    cell_index = 2
    producer_wcs, producer_shape, producer_center = create_padding_wcs(
        wcs, config, location, cell_index
    )
    standalone_wcs, standalone_shape, standalone_center = _standalone_padding_wcs(
        wcs, W, H, location
    )
    assert producer_shape == standalone_shape == (PATCH, PATCH)

    recipient_x0 = PAD_SIZE + cell_index * (W - CELL_OVERLAP)
    expected = padding_patch_geometry(
        wcs,
        recipient_x0=recipient_x0,
        recipient_y0=PAD_SIZE,
        cell_width=W,
        cell_height=H,
        location=location,
    )
    assert producer_center == expected.center
    # Same WCS construction: compare a point well away from CRPIX roundoff.
    np.testing.assert_allclose(
        producer_wcs.pixel_to_world_values(25.0, 25.0),
        expected.wcs.pixel_to_world_values(25.0, 25.0),
        rtol=0,
        atol=1e-12,
    )
    assert standalone_center != producer_center  # different reference frames by design
    np.testing.assert_allclose(
        standalone_wcs.pixel_to_world_values(25.0, 25.0),
        padding_patch_geometry(
            wcs,
            recipient_x0=0,
            recipient_y0=0,
            cell_width=W,
            cell_height=H,
            location=location,
        ).wcs.pixel_to_world_values(25.0, 25.0),
        rtol=0,
        atol=1e-12,
    )


def test_invalid_location_is_rejected(wcs):
    with pytest.raises(ValueError, match="unsupported"):
        padding_patch_geometry(
            wcs,
            recipient_x0=0,
            recipient_y0=0,
            cell_width=W,
            cell_height=H,
            location="diagonalish",
        )


@pytest.mark.parametrize(
    "location,expected_work_bounds,expected_recipient_slice",
    [
        # edges: inward extension is one-dimensional, perpendicular to the edge.
        ("top", (H - PAD_SIZE, H + PAD_SIZE, 0, W), (H - PAD_SIZE, H, 0, W)),
        ("bottom", (-PAD_SIZE, PAD_SIZE, 0, W), (0, PAD_SIZE, 0, W)),
        ("left", (0, H, -PAD_SIZE, PAD_SIZE), (0, H, 0, PAD_SIZE)),
        ("right", (0, H, W - PAD_SIZE, W + PAD_SIZE), (0, H, W - PAD_SIZE, W)),
        # corners: inward extension applies along both axes.
        ("top_left", (H - PAD_SIZE, H + PAD_SIZE, -PAD_SIZE, PAD_SIZE), (H - PAD_SIZE, H, 0, PAD_SIZE)),
        ("bottom_right", (-PAD_SIZE, PAD_SIZE, W - PAD_SIZE, W + PAD_SIZE), (0, PAD_SIZE, W - PAD_SIZE, W)),
    ],
)
def test_padding_work_image_geometry_extends_inward_by_radius(
    wcs, location, expected_work_bounds, expected_recipient_slice
):
    """R=470 (== PAD_SIZE - EDGE_EXCLUSION here) keeps the arithmetic simple:
    the work image spans exactly [near edge - PAD_SIZE, near edge + PAD_SIZE)
    along an affected axis, and the recipient-side slice is exactly one
    PAD_SIZE-wide strip inside the cell."""
    geometry = padding_work_image_geometry(
        wcs, cell_width=W, cell_height=H, location=location,
        inward_radius=PAD_SIZE - EDGE_EXCLUSION,
    )
    assert geometry.work_bounds == expected_work_bounds
    assert geometry.work_shape == (
        expected_work_bounds[1] - expected_work_bounds[0],
        expected_work_bounds[3] - expected_work_bounds[2],
    )
    recipient_slice = intersect_bounds(geometry.work_bounds, cell_width=W, cell_height=H)
    assert recipient_slice == expected_recipient_slice
    # The padding-box placement itself must be untouched (matches ps1_process).
    assert geometry.patch.bounds == padding_patch_geometry(
        wcs, recipient_x0=0, recipient_y0=0, cell_width=W, cell_height=H, location=location,
    ).bounds


def test_intersect_bounds_returns_none_when_disjoint():
    assert intersect_bounds((-500, -480, 0, 10), cell_width=100, cell_height=100) is None


def test_padding_work_image_geometry_rejects_nonpositive_radius(wcs):
    with pytest.raises(ValueError, match="inward_radius"):
        padding_work_image_geometry(
            wcs, cell_width=W, cell_height=H, location="top", inward_radius=0,
        )


def test_projection_identity_normalizes_table_and_skycell_spellings():
    from syndiff_pipeline.template_creation.processing.cross_projection_geometry import (
        projection_identity,
        skycell_projection_identity,
    )

    assert projection_identity("1234") == "1234"
    assert projection_identity("skycell.1234") == "1234"
    assert skycell_projection_identity("skycell.1234.056") == "1234"
    assert skycell_projection_identity("not-a-skycell") is None
