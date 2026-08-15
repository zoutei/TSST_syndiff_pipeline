"""Pad/trim pairing contract for diff stages (Phase 0b)."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.common.grid_pairing import (
    pad_mask_bottom,
    prepare_science_template_pairing,
    trim_padded_products,
    zero_pad_science_bottom,
)
from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError


@pytest.fixture
def grid() -> MappingGrid:
    return MappingGrid.from_ffi_shape(2048, 2048)


class TestZeroPadScienceBottom:
    def test_pad_shapes(self, grid: MappingGrid):
        sci = np.ones(grid.science_ffi_bounds()["shape"], dtype=np.float32)
        padded = zero_pad_science_bottom(sci, grid.conv_pad_native)
        assert padded.shape == grid.template_ffi_bounds()["shape"]
        assert np.all(padded[: grid.conv_pad_native, :] == 0)
        np.testing.assert_array_equal(padded[grid.conv_pad_native :, :], sci)


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
        trimmed = trim_padded_products(full, grid.conv_pad_native)
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
        out = trim_padded_products(product, grid.conv_pad_native)
        assert out.shape == (2018, 1960)
        assert out.shape == grid.science_ffi_bounds()["shape"]
        assert not np.all(out[0, :] == 0)

    def test_round_trip_pad_trim(self, grid: MappingGrid):
        sci = np.random.default_rng(0).normal(size=grid.science_ffi_bounds()["shape"])
        tmpl = np.random.default_rng(1).normal(size=grid.template_ffi_bounds()["shape"])
        sci_p, _ = prepare_science_template_pairing(sci, tmpl, grid)
        product = sci_p * 2.0
        out = trim_padded_products(product, grid.conv_pad_native)
        np.testing.assert_allclose(out, sci * 2.0)


class TestPadMaskBottom:
    def test_pad_rows_are_marked_bad(self, grid: MappingGrid):
        mask = np.zeros(grid.science_ffi_bounds()["shape"], dtype=bool)
        padded = pad_mask_bottom(mask, grid.conv_pad_native)
        assert padded.shape == grid.template_ffi_bounds()["shape"]
        assert np.all(padded[: grid.conv_pad_native, :] == True)  # noqa: E712
        np.testing.assert_array_equal(padded[grid.conv_pad_native :, :], mask)

    def test_zero_pad_science_bottom_marks_pad_rows_good_by_contrast(
        self, grid: MappingGrid
    ):
        """Documents why pad_mask_bottom exists: the science helper is wrong for masks."""
        mask = np.zeros(grid.science_ffi_bounds()["shape"], dtype=bool)
        wrongly_padded = zero_pad_science_bottom(mask, grid.conv_pad_native)
        assert np.all(wrongly_padded[: grid.conv_pad_native, :] == False)  # noqa: E712

    def test_no_pad_returns_bool_array(self, grid: MappingGrid):
        mask = np.zeros(grid.science_ffi_bounds()["shape"], dtype=bool)
        out = pad_mask_bottom(mask, 0)
        assert out.shape == mask.shape
        assert out.dtype == bool

    def test_negative_pad_raises(self, grid: MappingGrid):
        mask = np.zeros((4, 4), dtype=bool)
        with pytest.raises(MappingGridError, match="pad_rows"):
            pad_mask_bottom(mask, -1)


class TestOversamplingPad:
  """OS>1: pad native first, then upsample (contract stub for Phase 0b)."""

  def test_native_pad_before_os_factor(self, grid: MappingGrid):
      sci = np.ones(grid.science_ffi_bounds()["shape"])
      padded_native = zero_pad_science_bottom(sci, grid.conv_pad_native)
      f = 2
      padded_os = np.repeat(
          np.repeat(padded_native, f, axis=0), f, axis=1
      )
      expected_h = grid.height_native * f
      assert padded_os.shape[0] == expected_h
