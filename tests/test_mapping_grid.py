"""Golden-vector tests for the strict MAPGRID=3 coordinate contract."""

from __future__ import annotations

import hashlib
import json

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
        assert g.ffi_xmin == 36
        assert g.ffi_xmax == 2012
        assert g.ffi_ymin == -8
        assert g.ffi_ymax == 2026
        assert g.conv_pad_native == 8
        assert g.array_shape_native() == (2034, 1976)
        assert g.science_ffi_bounds()["shape"] == (2018, 1960)

    def test_rkernel_and_conv_pad_defaults(self):
        assert compute_rkernel(1.88) == 4
        assert compute_conv_pad_native(4, template_conv_pad_spare_px=4) == 8


class TestGoldenVectors:
    """Table from padded_scc_v2_implementation.md §4.4."""

    @pytest.mark.parametrize(
        "ffi_x, ffi_y, expected_lx, expected_ly",
        [
            (36, 0, 0, 8),
            (36, -1, 0, 7),
            (36, -8, 0, 0),
            (2011, 2017, 1975, 2025),
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
        assert sci["shape"][0] + 2 * default_grid.conv_pad_native == tmpl["shape"][0]


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

    def test_serializes_explicit_baseline_geometry_and_fingerprint(self, default_grid: MappingGrid):
        payload = default_grid.to_mapping_dict()
        assert payload["coordinate_frame"] == "full_ffi"
        assert payload["science_ymin"] == 0
        assert payload["template_ymin"] == -8
        assert payload["pad_left"] == payload["pad_right"] == 8
        assert payload["geometry_fingerprint"] == default_grid.geometry_fingerprint
        assert MappingGrid.from_mapping_dict(payload) == default_grid

    def test_rejects_geometry_fingerprint_mismatch(self, default_grid: MappingGrid):
        payload = default_grid.to_mapping_dict()
        payload["geometry_fingerprint"] = "invalid"
        with pytest.raises(MappingGridError, match="fingerprint"):
            MappingGrid.from_mapping_dict(payload)

    def test_mapgrid3_recipe_fingerprints_complete_paired_geometry(self):
        grid = MappingGrid.from_science_bounds(2, 0, 6, 4, pad=1, oversampling=2)
        recipe = grid.geometry_recipe()

        assert recipe["mapgrid_version"] == 3
        assert recipe["template_bounds_ffi"] == {
            "x_min": 1, "x_max": 7, "y_min": -1, "y_max": 5
        }
        assert recipe["science_bounds_ffi"] == {
            "x_min": 2, "x_max": 6, "y_min": 0, "y_max": 4
        }
        assert recipe["template_support_bounds_ffi"] == recipe["template_bounds_ffi"]
        assert recipe["physical_template_bounds_ffi"] == recipe["template_bounds_ffi"]
        assert recipe["pad_native"] == {
            "left": 1, "right": 1, "bottom": 1, "top": 1
        }
        assert recipe["support_policy"] == "bounded_support_pad"
        assert recipe["science_pad_policy"] == "neutral_invalid"
        assert recipe["science_slice_native"] == [[1, 5], [1, 5]]
        assert recipe["science_slice_os"] == [[2, 10], [2, 10]]
        assert recipe["effective_support_pad_native"] == 1
        expected_fp = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        assert grid.geometry_fingerprint == expected_fp

    def test_mapgrid2_recipe_is_rejected(self, default_grid: MappingGrid):
        recipe = default_grid.geometry_recipe()
        assert recipe["mapgrid_version"] == 3
        recipe["mapgrid_version"] = 2
        with pytest.raises(MappingGridError, match="MAPGRID=3"):
            MappingGrid.from_mapping_dict(recipe)

    def test_rejects_missing_and_unknown_mapgrid(self, default_grid: MappingGrid):
        payload = default_grid.to_mapping_dict()
        payload.pop("mapgrid_version")
        with pytest.raises(MappingGridError, match="mapgrid_version"):
            MappingGrid.from_mapping_dict(payload)
        for version in (0, 1, 2, 4, 99):
            payload["mapgrid_version"] = version
            with pytest.raises(MappingGridError, match="MAPGRID=3"):
                MappingGrid.from_mapping_dict(payload)

    def test_rejects_sidecar_v2(self, default_grid: MappingGrid):
        with pytest.raises(MappingGridError, match="schema_version"):
            MappingGrid.from_sidecar({"schema_version": 2, "roi_bounds": [0, 0, 2048, 2048]})


class TestCreateCoordsForGrid:
    def test_wcs_coords_use_ffi_not_local(self, default_grid: MappingGrid):
        tpix, flat = create_coords_for_grid(default_grid)
        assert_wcs_uses_ffi_coords(tpix, default_grid)
        # Pad corner local (0,0) -> FFI (36, -8)
        ty0, tx0 = tpix[0]
        assert ty0 < 0
        assert int(round(tx0)) == 36

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

    def test_preflight_samples_all_physical_edges(self, default_grid: MappingGrid):
        tpix, _ = create_coords_for_grid(default_grid, 1)
        assert_wcs_uses_ffi_coords(tpix, default_grid)
        tpix[-1, 1] += 1
        with pytest.raises(CoordinatePreflightError, match="edge sample"):
            assert_wcs_uses_ffi_coords(tpix, default_grid)

    def test_validate_conv_pad_ok(self, default_grid: MappingGrid):
        validate_conv_pad_for_diff(default_grid, scale_px=1.88)

    def test_validate_conv_pad_fails_oversize_kernel(self, default_grid: MappingGrid):
        with pytest.raises(CoordinatePreflightError, match="rkernel"):
            validate_conv_pad_for_diff(default_grid, scale_px=10.0)


class TestOversampling:
    def test_array_shape_os(self):
        g = MappingGrid.from_ffi_shape(2048, 2048, oversampling=2)
        assert g.array_shape_os() == (4068, 3952)

    def test_flat_os_width(self):
        g = MappingGrid.from_ffi_shape(2048, 2048, oversampling=2)
        flat = g.local_to_flat(1, 2, oversampled=True)
        assert flat == 2 * g.width_os + 1
