"""Tests for convolution_utils.apply_gaussian_convolution."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing.convolution_utils import (
    apply_gaussian_convolution,
)


class TestConvolutionUtils(unittest.TestCase):
    def test_cval_zero_conserves_flux_on_tight_cutout(self):
        cutout = np.zeros((951, 951), dtype=np.float32)
        cutout[470:481, 470:481] = 1000.0
        input_flux = float(cutout.sum())

        out = apply_gaussian_convolution(cutout, sigma=60.0, radius=470, cval=0.0)
        self.assertFalse(np.isnan(out).any())
        self.assertAlmostEqual(float(np.sum(out)), input_flux, delta=1.0)

    def test_default_cval_nan_contaminates_tight_cutout(self):
        cutout = np.zeros((951, 951), dtype=np.float32)
        cutout[470:481, 470:481] = 1000.0
        input_flux = float(cutout.sum())

        out = apply_gaussian_convolution(cutout, sigma=60.0, radius=470)
        self.assertGreater(np.isnan(out).mean(), 0.9)
        self.assertLess(float(np.nansum(out)), 0.1 * input_flux)


if __name__ == "__main__":
    unittest.main()
