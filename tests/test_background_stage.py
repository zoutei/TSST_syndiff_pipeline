"""Tests for unified background stage helpers."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from syndiff_pipeline.difference_imaging.stages.background import io


def test_stack_from_bkg_records_builds_cube():
    with tempfile.TemporaryDirectory() as tmp:
        ny, nx = 8, 10
        records = []
        for i in range(3):
            stem = f"tess1234567890_bkg{i}"
            path = os.path.join(tmp, f"{stem}.fits")
            fits.writeto(path, np.full((ny, nx), float(i), dtype=np.float32))
            records.append(
                io.FrameRecord(
                    index=i,
                    product_id=f"tess1234567890",
                    stem=stem,
                    diff_path="",
                    bkg_path=path,
                    success=True,
                )
            )
        stack = io.stack_from_bkg_records(records)
        assert stack.shape == (3, ny, nx)
        assert stack[2, 0, 0] == pytest.approx(2.0)


def test_load_stack_or_fits_prefers_stack_file():
    with tempfile.TemporaryDirectory() as tmp:
        arr = np.ones((2, 4, 4), dtype=np.float32)
        io.save_stack(arr, tmp)
        records = [
            io.FrameRecord(
                index=0,
                product_id="tess1",
                stem="tess1_x",
                diff_path="",
                bkg_path="",
                success=False,
            )
        ]
        loaded = io.load_stack_or_fits(tmp, records)
        np.testing.assert_array_equal(loaded, arr)
