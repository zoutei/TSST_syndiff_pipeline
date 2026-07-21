"""Tests for coordinate_preflight helpers."""

from __future__ import annotations

import pytest

from syndiff_pipeline.common.coordinate_preflight import (
    CoordinatePreflightError,
    validate_coordinate_contract,
)
from syndiff_pipeline.common.mapping_grid import MappingGrid


def test_crop_bounds_mismatch_raises():
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    bad = dict(grid.science_ffi_bounds())
    bad["y_min"] = 1
    with pytest.raises(CoordinatePreflightError, match="y_min"):
        validate_coordinate_contract(grid, bad)
