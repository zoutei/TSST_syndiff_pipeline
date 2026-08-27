"""Tests for gridded ePSF diagnostic plots."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages import gridded_epsf
from syndiff_pipeline.difference_imaging.support.plot import (
    gridded_epsf_frame_plot_path,
    select_evenly_spaced_stems,
    spatial_tile_subplot_grid,
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

            plot_dir = os.path.join(tmp, "debug_plots", "epsf_r1")
            png = os.path.join(plot_dir, "one.png")
            out = write_gridded_epsf_frame_plot(
                npz_path,
                png,
                title="tess123",
            )
            self.assertEqual(out, png)
            self.assertTrue(os.path.isfile(png))
            self.assertGreater(os.path.getsize(png), 1000)

            written = write_gridded_epsf_workspace_plots(
                ws,
                plot_dir,
                epsf_label="epsf_r1",
                max_frames=1,
            )
            self.assertGreaterEqual(len(written), 2)
            summary = os.path.join(plot_dir, "epsf_r1_index.json")
            self.assertIn(summary, written)
            self.assertTrue(os.path.isfile(summary))
            expected_png = gridded_epsf_frame_plot_path(plot_dir, "epsf_r1", "tess123")
            self.assertIn(expected_png, written)

    def test_workspace_plots_prefer_anchor_stems_over_interpolated(self):
        # Regression: orbit-binned mode writes an npz per frame (anchor or
        # interpolated/blended), but only anchors have a real per-tile fit
        # (n_stars). With many interpolated frames per anchor, unrestricted
        # evenly-spaced selection almost always lands on a frame with no
        # star count to show. The anchor-stem sidecar must be consulted so
        # selection prefers real fits.
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "epsf_r1")
            os.makedirs(ws, exist_ok=True)
            ny, nx = 15, 15
            yy, xx = np.mgrid[0:ny, 0:nx]
            stamp = np.exp(-((xx - 7) ** 2 + (yy - 7) ** 2) / 8.0)
            stack = np.stack([stamp])
            grid_xypos = [(256.0, 256.0)]

            index = {}
            anchor_stems = {"anchor_1"}
            for stem in ["interp_a", "interp_b", "anchor_1", "interp_c", "interp_d"]:
                npz_path = gridded_epsf.gridded_epsf_npz_path(ws, stem)
                n_stars = [42] if stem in anchor_stems else None
                gridded_epsf.save_gridded_epsf_npz(npz_path, stack, grid_xypos, 2, n_stars=n_stars)
                index[stem] = npz_path
            gridded_epsf.save_gridded_epsf_index(ws, index)
            gridded_epsf.save_gridded_epsf_anchor_stems(ws, anchor_stems)

            plot_dir = os.path.join(tmp, "debug_plots", "epsf_r1")
            written = write_gridded_epsf_workspace_plots(
                ws, plot_dir, epsf_label="epsf_r1", max_frames=1,
            )
            expected_png = gridded_epsf_frame_plot_path(plot_dir, "epsf_r1", "anchor_1")
            self.assertIn(expected_png, written)

    def test_anchor_stems_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(gridded_epsf.load_gridded_epsf_anchor_stems(tmp), set())
            gridded_epsf.save_gridded_epsf_anchor_stems(tmp, {"a", "b"})
            self.assertEqual(gridded_epsf.load_gridded_epsf_anchor_stems(tmp), {"a", "b"})

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

            plot_dir = os.path.join(tmp, "debug_plots", "epsf_r1")
            written = write_gridded_epsf_workspace_plots(
                ws, plot_dir, epsf_label="epsf_r1", max_frames=1
            )
            self.assertGreaterEqual(len(written), 2)

    def test_group_plots_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "epsf_r1")
            group_dir = os.path.join(ws, "group_epsf")
            os.makedirs(group_dir, exist_ok=True)
            ny, nx = 9, 9
            stamp = np.ones((ny, nx), dtype=np.float64)
            stack = np.stack([stamp, stamp, stamp, stamp])
            grid_xypos = [(128.0, 128.0), (384.0, 128.0), (128.0, 384.0), (384.0, 384.0)]
            frame_npz = gridded_epsf.gridded_epsf_npz_path(ws, "tess111")
            gridded_epsf.save_gridded_epsf_npz(frame_npz, stack, grid_xypos, 2)
            gridded_epsf.save_gridded_epsf_index(ws, {"tess111": frame_npz})
            group_npz = os.path.join(group_dir, "group_epsf_0.npz")
            gridded_epsf.save_gridded_epsf_npz(group_npz, stack, grid_xypos, 2)

            plot_dir = os.path.join(tmp, "debug_plots", "epsf_r1")
            written = write_gridded_epsf_workspace_plots(
                ws, plot_dir, epsf_label="epsf_r1", max_frames=1
            )
            self.assertFalse(any("group" in os.path.basename(p) for p in written))

    def test_select_evenly_spaced_stems_by_btjd(self):
        stems = ["stem_c", "stem_a", "stem_b"]
        wcs = pd.DataFrame(
            {
                "product_id": ["stem_a", "stem_b", "stem_c"],
                "btjd": [100.0, 200.0, 300.0],
            }
        )
        picked = select_evenly_spaced_stems(stems, wcs_table=wcs, max_frames=2)
        self.assertEqual(picked, ["stem_a", "stem_c"])

    def test_spatial_tile_subplot_grid_bottom_left_is_low_xy(self):
        grid_xypos = np.array(
            [
                [100.0, 200.0],
                [300.0, 200.0],
                [100.0, 400.0],
                [300.0, 400.0],
            ]
        )
        n_rows, n_cols, placements = spatial_tile_subplot_grid(grid_xypos)
        self.assertEqual((n_rows, n_cols), (2, 2))
        by_node = {k: (row, col) for k, row, col in placements}
        # node 0 at (100, 200) -> bottom-left
        self.assertEqual(by_node[0], (1, 0))
        # node 3 at (300, 400) -> top-right
        self.assertEqual(by_node[3], (0, 1))


if __name__ == "__main__":
    unittest.main()
