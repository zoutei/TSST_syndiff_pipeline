"""Tests for gridded ePSF diagnostic plots."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages import gridded_epsf
from syndiff_pipeline.difference_imaging.support.plot import (
    write_gridded_epsf_frame_plot,
    write_gridded_epsf_workspace_plots,
)


class TestGriddedEpsfPlots(unittest.TestCase):
    def test_frame_plot_and_workspace_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "epsf_r1")
            os.makedirs(ws, exist_ok=True)
            ny, nx = 15, 15
            yy, xx = np.mgrid[0:ny, 0:nx]
            stamp = np.exp(-((xx - 7) ** 2 + (yy - 7) ** 2) / 8.0)
            stack = np.stack([stamp, stamp * 0.9, stamp * 1.1, stamp * 0.8])
            grid_xypos = [(256.0, 256.0), (768.0, 256.0), (256.0, 768.0), (768.0, 768.0)]
            npz_path = gridded_epsf.gridded_epsf_npz_path(ws, "tess123")
            gridded_epsf.save_gridded_epsf_npz(npz_path, stack, grid_xypos, 2)
            gridded_epsf.save_gridded_epsf_index(ws, {"tess123": npz_path})

            plot_dir = os.path.join(tmp, "debug_plots")
            png = os.path.join(plot_dir, "one.png")
            out = write_gridded_epsf_frame_plot(
                npz_path,
                png,
                title="tess123",
                crop_shape=(1024, 1024),
            )
            self.assertEqual(out, png)
            self.assertTrue(os.path.isfile(png))
            self.assertGreater(os.path.getsize(png), 1000)

            written = write_gridded_epsf_workspace_plots(
                ws,
                plot_dir,
                epsf_label="epsf_r1",
                max_frames=1,
                crop_shape=(1024, 1024),
            )
            self.assertGreaterEqual(len(written), 2)
            summary = os.path.join(plot_dir, "epsf_epsf_r1_index.json")
            self.assertIn(summary, written)
            self.assertTrue(os.path.isfile(summary))

    def test_discovers_npz_when_index_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "epsf_r1")
            os.makedirs(ws, exist_ok=True)
            ny, nx = 11, 11
            stamp = np.ones((ny, nx), dtype=np.float64)
            stack = np.stack([stamp, stamp, stamp, stamp])
            grid_xypos = [(128.0, 128.0), (384.0, 128.0), (128.0, 384.0), (384.0, 384.0)]
            npz_path = gridded_epsf.gridded_epsf_npz_path(ws, "tess999")
            gridded_epsf.save_gridded_epsf_npz(npz_path, stack, grid_xypos, 2)
            gridded_epsf.save_gridded_epsf_index(ws, {})

            plot_dir = os.path.join(tmp, "debug_plots")
            written = write_gridded_epsf_workspace_plots(
                ws, plot_dir, epsf_label="epsf_r1", max_frames=1
            )
            self.assertGreaterEqual(len(written), 2)


if __name__ == "__main__":
    unittest.main()
