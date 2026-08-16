"""Pad/trim pairing contract for diff stages (Phase 0b)."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.common.grid_pairing import (
    pad_mask_bottom,
    prepare_science_template_pairing,
    trim_padded_products,
    zero_pad_science_bottom,
    pad_mask_to_template,
)
from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError


@pytest.fixture
def grid() -> MappingGrid:
    return MappingGrid.from_ffi_shape(2048, 2048)


class TestZeroPadScienceBottom:
    def test_removed_api_fails_loudly(self, grid: MappingGrid):
        sci = np.ones(grid.science_ffi_bounds()["shape"], dtype=np.float32)
        with pytest.raises(MappingGridError, match="MAPGRID=3"):
            zero_pad_science_bottom(sci, grid.conv_pad_native)


class TestPrepareScienceTemplatePairing:
    def test_pairing_shapes_match(self, grid: MappingGrid):
        sci_shape = grid.science_ffi_bounds()["shape"]
        tmpl_shape = grid.template_ffi_bounds()["shape"]
        sci = np.zeros(sci_shape, dtype=np.float64)
        tmpl = np.ones(tmpl_shape, dtype=np.float64)
        sci_p, tmpl_p = prepare_science_template_pairing(sci, tmpl, grid)
        assert sci_p.shape == tmpl_p.shape == tmpl_shape

    def test_wrong_science_shape_raises(self, grid: MappingGrid):
        sci = np.zeros((10, 10))
        tmpl = np.ones(grid.template_ffi_bounds()["shape"])
        with pytest.raises(MappingGridError, match="science shape"):
            prepare_science_template_pairing(sci, tmpl, grid)


class TestTrimPaddedProducts:
    def test_trim_restores_science_shape(self, grid: MappingGrid):
        full = np.arange(np.prod(grid.template_ffi_bounds()["shape"])).reshape(
            grid.template_ffi_bounds()["shape"]
        )
        trimmed = trim_padded_products(full, grid=grid)
        assert trimmed.shape == grid.science_ffi_bounds()["shape"]

    def test_hotpants_like_output_is_science_2018x1960(self, grid: MappingGrid):
        """§16 / PR-4c: trimmed pairing product is science shape; bottom not all-zero."""
        sci = np.ones(grid.science_ffi_bounds()["shape"], dtype=np.float64)
        sci[0, :] = 3.0  # science bottom row
        tmpl = np.ones(grid.template_ffi_bounds()["shape"], dtype=np.float64)
        sci_p, tmpl_p = prepare_science_template_pairing(sci, tmpl, grid)
        assert sci_p.shape == tmpl_p.shape == grid.template_ffi_bounds()["shape"]
        # Hotpants-like product on padded grid, then trim to science.
        product = sci_p - tmpl_p
        out = trim_padded_products(product, grid=grid)
        assert out.shape == (2018, 1960)
        assert out.shape == grid.science_ffi_bounds()["shape"]
        assert not np.all(out[0, :] == 0)

    def test_round_trip_pad_trim(self, grid: MappingGrid):
        sci = np.random.default_rng(0).normal(size=grid.science_ffi_bounds()["shape"])
        tmpl = np.random.default_rng(1).normal(size=grid.template_ffi_bounds()["shape"])
        sci_p, _ = prepare_science_template_pairing(sci, tmpl, grid)
        product = sci_p * 2.0
        out = trim_padded_products(product, grid=grid)
        np.testing.assert_allclose(out, sci * 2.0)


class TestPadMaskBottom:
    def test_removed_api_fails_loudly(self, grid: MappingGrid):
        mask = np.zeros(grid.science_ffi_bounds()["shape"], dtype=bool)
        with pytest.raises(MappingGridError, match="MAPGRID=3"):
            pad_mask_bottom(mask, grid.conv_pad_native)

    def test_zero_pad_science_bottom_marks_pad_rows_good_by_contrast(
        self, grid: MappingGrid
    ):
        """Documents why pad_mask_bottom exists: the science helper is wrong for masks."""
        mask = np.zeros(grid.science_ffi_bounds()["shape"], dtype=bool)
        with pytest.raises(MappingGridError, match="MAPGRID=3"):
            zero_pad_science_bottom(mask, grid.conv_pad_native)

    def test_no_pad_returns_bool_array(self, grid: MappingGrid):
        mask = np.zeros(grid.science_ffi_bounds()["shape"], dtype=bool)
        with pytest.raises(MappingGridError, match="MAPGRID=3"):
            pad_mask_bottom(mask, 0)

    def test_negative_pad_raises(self, grid: MappingGrid):
        mask = np.zeros((4, 4), dtype=bool)
        with pytest.raises(MappingGridError, match="MAPGRID=3"):
            pad_mask_bottom(mask, -1)


class TestOversamplingPad:
  """OS>1: pad native first, then upsample (contract stub for Phase 0b)."""

  def test_native_pad_before_os_factor(self, grid: MappingGrid):
      sci = np.ones(grid.science_ffi_bounds()["shape"])
      with pytest.raises(MappingGridError, match="MAPGRID=3"):
          zero_pad_science_bottom(sci, grid.conv_pad_native)


class TestFourSidedMapgrid3Pairing:
    @pytest.fixture
    def v3(self) -> MappingGrid:
        return MappingGrid.from_science_bounds(10, 0, 18, 6, pad=2)

    def test_all_edges_and_corner_are_neutral_and_trim_exactly(self, v3):
        sci = np.arange(48, dtype=float).reshape(6, 8)
        tmpl = np.ones(v3.template_ffi_bounds()["shape"])
        padded, _ = prepare_science_template_pairing(sci, tmpl, v3)
        assert padded.shape == tmpl.shape == (10, 12)
        ys, xs = v3.science_slice_native()
        assert np.all(padded[:2] == 0)
        assert np.all(padded[-2:] == 0)
        assert np.all(padded[:, :2] == 0)
        assert np.all(padded[:, -2:] == 0)
        np.testing.assert_array_equal(trim_padded_products(padded, grid=v3), sci)

    def test_mask_excludes_all_fabricated_edges(self, v3):
        mask = np.zeros((6, 8), dtype=bool)
        padded = pad_mask_to_template(mask, v3)
        assert padded.shape == (10, 12)
        assert np.all(padded[:2]) and np.all(padded[-2:])
        assert np.all(padded[:, :2]) and np.all(padded[:, -2:])
        assert not np.any(padded[v3.science_slice_native()])

    def test_explicit_convolution_reference_trim_preserves_edge_pixels(self, v3):
        """Convolve on T, then canonical-trim; S edge values remain present."""
        sci = np.zeros((6, 8), dtype=float)
        sci[0, 0] = 7.0
        sci[-1, -1] = 11.0
        padded = prepare_science_template_pairing(
            sci, np.zeros(v3.template_ffi_bounds()["shape"]), v3
        )[0]
        kernel = np.ones((3, 3), dtype=float) / 9.0
        from scipy.signal import convolve2d
        ref = convolve2d(padded, kernel, mode="same", boundary="fill")
        out = trim_padded_products(ref, grid=v3)
        assert out.shape == sci.shape
        assert out[0, 0] > 0 and out[-1, -1] > 0
