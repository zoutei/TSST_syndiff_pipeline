# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for photutils -> forward ePSF conversion."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model.init_study.photutils_to_forward_epsf import (
    gridded_stack_to_epsf_base,
    photutils_density_to_flux_fraction,
    resample_oversampled_stamp_to_node,
)


def test_photutils_density_to_flux_fraction():
    os = 4
    stamp = np.ones((25, 25), dtype=np.float64) * (os ** 2 / 625.0)
    frac = photutils_density_to_flux_fraction(stamp, oversample=os)
    assert frac.shape == (25, 25)
    assert abs(float(np.sum(frac)) - 1.0) < 1e-10


def test_resample_to_node_sum_one():
    os = 4
    n = 25
    stamp = np.zeros((n, n), dtype=np.float64)
    stamp[n // 2, n // 2] = float(os ** 2)
    frac = photutils_density_to_flux_fraction(stamp, oversample=os)
    node = resample_oversampled_stamp_to_node(frac, oversample_src=os)
    assert node.shape == (EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)
    assert abs(float(np.sum(node)) - 1.0) < 1e-5


def test_gridded_stack_shape():
    stack = np.ones((4, 25, 25), dtype=np.float64) * 16.0
    base = gridded_stack_to_epsf_base(stack, tile_ny=2, tile_nx=2, oversample=4)
    assert base.shape == (2, 2, EM.NODE_GRID_SIZE, EM.NODE_GRID_SIZE)
    for i in range(2):
        for j in range(2):
            assert abs(float(np.sum(base[i, j])) - 1.0) < 1e-4
