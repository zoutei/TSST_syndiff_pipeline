"""Golden-vector tests for MappingGrid v2 coordinate contract."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.common.coordinate_preflight import (
    CoordinatePreflightError,
    assert_wcs_uses_ffi_coords,
    validate_conv_pad_for_diff,
    validate_coordinate_contract,
)
from syndiff_pipeline.common.mapping_grid import (
    MAPGRID_VERSION,
    MappingGrid,
    MappingGridError,
    compute_conv_pad_native,
    compute_rkernel,
    create_coords_for_grid,
)


@pytest.fixture
def default_grid() -> MappingGrid:
  return MappingGrid.from_ffi_shape(2048, 2048)


class TestDefaultGeometry:
    def test_from_ffi_shape_defaults(self, default_grid: MappingGrid):
        g = default_grid
        assert g.ffi_xmin == 44
        assert g.ffi_xmax == 2004
        assert g.ffi_ymin == -8
        assert g.ffi_ymax == 2018
        assert g.conv_pad_native == 8
        assert g.array_shape_native() == (2026, 1960)
        assert g.science_ffi_bounds()["shape"] == (2018, 1960)

    def test_rkernel_and_conv_pad_defaults(self):
        assert compute_rkernel(1.88) == 4
        assert compute_conv_pad_native(4, template_conv_pad_spare_px=4) == 8


class TestGoldenVectors:
    """Table from padded_scc_v2_implementation.md §4.4."""

    @pytest.mark.parametrize(
        "ffi_x, ffi_y, expected_lx, expected_ly",
        [
            (44, 0, 0, 8),
            (44, -1, 0, 7),
            (44, -8, 0, 0),
            (2003, 2017, 1959, 2025),
        ],
    )
    def test_ffi_to_local(
        self,
        default_grid: MappingGrid,
        ffi_x: int,
        ffi_y: int,
        expected_lx: int,
        expected_ly: int,
    ):
        assert default_grid.ffi_to_local(ffi_x, ffi_y) == (expected_lx, expected_ly)

    def test_flat_round_trip_corners(self, default_grid: MappingGrid):
        g = default_grid
        corners = [
            g.ffi_to_local(44, -8),
            g.ffi_to_local(44, 0),
            g.ffi_to_local(2003, 2017),
            (g.width_native // 2, g.conv_pad_native + g.science_ffi_bounds()["shape"][0] // 2),
        ]
        for lx, ly in corners:
            flat = g.local_to_flat(lx, ly)
            assert g.flat_to_local(flat) == (lx, ly)
            assert g.flat_to_ffi(flat) == g.local_to_ffi(lx, ly)

    def test_science_vs_template_bounds(self, default_grid: MappingGrid):
        sci = default_grid.science_ffi_bounds()
        tmpl = default_grid.template_ffi_bounds()
        assert sci["y_min"] == 0
        assert tmpl["y_min"] == -8
        assert sci["shape"][0] + default_grid.conv_pad_native == tmpl["shape"][0]


class TestSerialization:
    def test_from_fits_header(self, default_grid: MappingGrid):
        hdr = default_grid.to_fits_header_updates()
        loaded = MappingGrid.from_fits_header(hdr)
        assert loaded == default_grid

    def test_rejects_mapgrid_v1(self, default_grid: MappingGrid):
        hdr = default_grid.to_fits_header_updates()
        hdr["MAPGRID"] = 1
        with pytest.raises(MappingGridError, match="MAPGRID"):
            MappingGrid.from_fits_header(hdr)

    def test_from_sidecar_v3(self, default_grid: MappingGrid):
        doc = {
            "schema_version": 3,
            "mapping_grid": default_grid.to_mapping_dict(),
        }
        loaded = MappingGrid.from_sidecar(doc)
        assert loaded == default_grid

    def test_rejects_sidecar_v2(self, default_grid: MappingGrid):
        with pytest.raises(MappingGridError, match="schema_version"):
            MappingGrid.from_sidecar({"schema_version": 2, "roi_bounds": [0, 0, 2048, 2048]})


class TestCreateCoordsForGrid:
    def test_wcs_coords_use_ffi_not_local(self, default_grid: MappingGrid):
        tpix, flat = create_coords_for_grid(default_grid)
        assert_wcs_uses_ffi_coords(tpix, default_grid)
        # Pad row local (0,0) -> FFI (44, -8)
        ty0, tx0 = tpix[0]
        assert ty0 < 0
        assert int(round(tx0)) == 44

    def test_local_mistake_fails_preflight(self, default_grid: MappingGrid):
        mistaken = np.array([[7.0, 0.0], [0.0, 44.0]])
        with pytest.raises(CoordinatePreflightError):
            assert_wcs_uses_ffi_coords(mistaken, default_grid)

    def test_wcs_at_ffi_44_minus4_differs_from_local_0_4(self, default_grid: MappingGrid):
        tpix, _ = create_coords_for_grid(default_grid)
        lx, ly = 0, 4
        ffi_x, ffi_y = default_grid.local_to_ffi(lx, ly)
        ty_ffi, tx_ffi = tpix[default_grid.local_to_flat(lx, ly)]
        assert (ty_ffi, tx_ffi) == (float(ffi_y), float(ffi_x))
        assert ffi_y == -4
        assert (ly, lx) != (ffi_y, ffi_x)


class TestPreflight:
    def test_validate_coordinate_contract(self, default_grid: MappingGrid):
        validate_coordinate_contract(
            default_grid,
            default_grid.science_ffi_bounds(),
            default_grid.to_fits_header_updates(),
        )

    def test_validate_conv_pad_ok(self, default_grid: MappingGrid):
        validate_conv_pad_for_diff(default_grid, scale_px=1.88)

    def test_validate_conv_pad_fails_oversize_kernel(self, default_grid: MappingGrid):
        with pytest.raises(CoordinatePreflightError, match="rkernel"):
            validate_conv_pad_for_diff(default_grid, scale_px=10.0)


class TestOversampling:
    def test_array_shape_os(self):
        g = MappingGrid.from_ffi_shape(2048, 2048, oversampling=2)
        assert g.array_shape_os() == (4052, 3920)

    def test_flat_os_width(self):
        g = MappingGrid.from_ffi_shape(2048, 2048, oversampling=2)
        flat = g.local_to_flat(1, 2, oversampled=True)
        assert flat == 2 * g.width_os + 1
