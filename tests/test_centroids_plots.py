"""Tests for centroid debug residual FITS helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.support.plot import (
    centroids_residual_fits_path,
    select_pipeline_debug_stems,
)


class TestCentroidsPlots(unittest.TestCase):
    def test_select_debug_stems_from_epsf_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            epsf_plot_dir = os.path.join(tmp, "debug_plots", "epsf_r1")
            os.makedirs(epsf_plot_dir, exist_ok=True)
            summary = os.path.join(epsf_plot_dir, "epsf_r1_index.json")
            with open(summary, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "frames_plotted": ["stem_b", "stem_a", "stem_c"],
                    },
                    fh,
                )
            picked = select_pipeline_debug_stems(
                ["stem_a", "stem_b", "stem_d"],
                reference_plot_dir=epsf_plot_dir,
                reference_label="epsf_r1",
                max_frames=10,
            )
            self.assertEqual(picked, ["stem_b", "stem_a"])

    def test_select_debug_stems_fallback_to_btjd(self):
        stems = ["stem_c", "stem_a", "stem_b"]
        wcs = pd.DataFrame(
            {
                "product_id": ["stem_a", "stem_b", "stem_c"],
                "btjd": [100.0, 200.0, 300.0],
            }
        )
        picked = select_pipeline_debug_stems(stems, wcs_table=wcs, max_frames=2)
        self.assertEqual(picked, ["stem_a", "stem_c"])

    def test_centroids_residual_fits_path(self):
        plot_dir = "/tmp/debug_plots/centroids_r1"
        self.assertEqual(
            centroids_residual_fits_path(plot_dir, "centroids_r1", "stem_a"),
            os.path.join(
                plot_dir, "centroids_r1_stem_a_epsf_photometry_residual.fits"
            ),
        )


if __name__ == "__main__":
    unittest.main()
