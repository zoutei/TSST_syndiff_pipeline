"""Tests for forced_photometry methods list parsing and CSV naming."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    ForcedPhotometryParams,
    PsfPhotometryMethodParams,
    parse_forced_photometry,
)
from syndiff_pipeline.difference_imaging.stages import photometry as ph


class TestPhotometryMethodsConfig(unittest.TestCase):
    def test_parse_two_psf_methods(self):
        p = parse_forced_photometry(
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "hp_d"},
                "output": "lc",
                "methods": [
                    {"name": "prf", "type": "psf", "psf_type": "prf"},
                    {
                        "name": "epsf",
                        "type": "psf",
                        "psf_type": "epsf",
                        "inputs": {"epsf": "epsf_r1"},
                    },
                ],
            },
            0,
        )
        self.assertEqual(len(p.methods), 2)
        self.assertIsInstance(p.methods[0], PsfPhotometryMethodParams)
        self.assertEqual(p.methods[1].epsf_workspace, "epsf_r1")

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            parse_forced_photometry(
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "x"},
                    "output": "y",
                    "methods": [
                        {"name": "x", "type": "psf", "psf_type": "prf"},
                        {"name": "x", "type": "aperture"},
                    ],
                },
                0,
            )


class TestPhotometryMultiPsfCsvs(unittest.TestCase):
    def test_two_psf_methods_write_two_csvs(self):
        n_ep = 2
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame({"btjd": [100.0, 101.0], "group_id": [0, 0]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
        epsf = np.zeros((1, 121))
        tiles = [(50.0, 50.0)]
        cfg = SimpleNamespace(
            sector=20,
            camera=3,
            ccd=3,
            n_jobs=1,
            pipeline_plots=False,
            pipeline_plot_dpi=150,
        )
        stage = ForcedPhotometryParams(
            methods=[
                PsfPhotometryMethodParams(
                    name="prf_a",
                    psf_type="epsf",
                    psf_size=5,
                    phot_cutout_size=15,
                    phot_bkg_poly_order=1,
                    phot_snap="fixed",
                    tile_nx=1,
                    tile_ny=1,
                ),
                PsfPhotometryMethodParams(
                    name="prf_b",
                    psf_type="epsf",
                    psf_size=5,
                    phot_cutout_size=15,
                    phot_bkg_poly_order=1,
                    phot_snap="fixed",
                    tile_nx=1,
                    tile_ny=1,
                ),
            ]
        )
        target_specs = [(xy.copy(), None, "primary", {"position_mode": "sky"})]

        g = np.linspace(-1, 1, 11)
        xx, yy = np.meshgrid(g, g)
        k = np.exp(-(xx**2 + yy**2) / 0.5)
        k /= k.sum()
        loc = ph.EpsfLocator(k.astype(np.float64), os_factor=2)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                ph, "read_diff_primary_and_noise_sigma", return_value=(np.zeros((32, 32)), None)
            ):
                with patch.object(ph, "build_psf_kernel", return_value=loc):
                    with patch.object(ph.os.path, "exists", return_value=True):
                        ph.run_forced_photometry_stage(
                            diff_paths=paths,
                            target_specs=target_specs,
                            phot_stage=stage,
                            epsf_by_workspace={"epsf_ws": epsf},
                            stage_epsf_workspace="epsf_ws",
                            tile_centers=tiles,
                            wcs_table=wcs,
                            crop_bounds=crop_bounds,
                            cfg=cfg,
                            output_dir=tmp,
                        )
            self.assertTrue((Path(tmp) / "lightcurve_prf_a.csv").is_file())
            self.assertTrue((Path(tmp) / "lightcurve_prf_b.csv").is_file())


if __name__ == "__main__":
    unittest.main()
