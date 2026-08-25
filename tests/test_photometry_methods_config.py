"""Tests for forced_photometry methods list parsing and CSV naming."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    AperturePhotometryMethodParams,
    ForcedPhotometryParams,
    PsfPhotometryMethodParams,
    parse_forced_photometry,
)
from syndiff_pipeline.difference_imaging.stages import photometry as ph


class TestPhotometryMethodsConfig(unittest.TestCase):
    def test_forced_photometry_can_exclude_primary_target(self):
        params = parse_forced_photometry(
            {
                "kind": "forced_photometry",
                "include_primary_target": False,
                "methods": [{"name": "ffiwcs", "type": "psf", "psf_type": "epsf", "position_source": "native_wcs"}],
            },
            0,
        )
        self.assertFalse(params.include_primary_target)

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
        self.assertIsNone(p.methods[1].fitter)
        self.assertIsNone(p.methods[1].psf_grouper_min_separation)

    def test_parse_fitter_tessreduce_and_null_poly(self):
        p = parse_forced_photometry(
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "hp_d", "epsf": "epsf_r1"},
                "output": "lc",
                "methods": [
                    {
                        "name": "epsf_bkg",
                        "type": "psf",
                        "psf_type": "epsf",
                        "fitter": "tessreduce",
                        "phot_bkg_poly_order": None,
                    },
                    {
                        "name": "ap3",
                        "type": "aperture",
                        "subtract_sky": False,
                        "mask_sky_with_shared_mask": True,
                    },
                ],
            },
            0,
        )
        self.assertEqual(p.methods[0].fitter, "tessreduce")
        self.assertIsNone(p.methods[0].phot_bkg_poly_order)
        self.assertIsInstance(p.methods[1], AperturePhotometryMethodParams)
        self.assertFalse(p.methods[1].subtract_sky)
        self.assertTrue(p.methods[1].mask_sky_with_shared_mask)

    def test_fitter_rejected_on_prf(self):
        with self.assertRaises(ValueError) as ctx:
            parse_forced_photometry(
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "x"},
                    "output": "y",
                    "methods": [
                        {
                            "name": "prf",
                            "type": "psf",
                            "psf_type": "prf",
                            "fitter": "tessreduce",
                        },
                    ],
                },
                0,
            )
        self.assertIn("fitter", str(ctx.exception))

    def test_bad_fitter_rejected(self):
        with self.assertRaises(ValueError):
            parse_forced_photometry(
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "x"},
                    "output": "y",
                    "methods": [
                        {
                            "name": "epsf",
                            "type": "psf",
                            "psf_type": "epsf",
                            "fitter": "create_psf",
                        },
                    ],
                },
                0,
            )

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
    def test_two_prf_methods_write_two_csvs(self):
        n_ep = 2
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame({"btjd": [100.0, 101.0], "group_id": [0, 0]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
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
                    psf_type="prf",
                    psf_size=5,
                    phot_cutout_size=15,
                    phot_bkg_poly_order=1,
                    phot_snap="fixed",
                    tile_nx=1,
                    tile_ny=1,
                ),
                PsfPhotometryMethodParams(
                    name="prf_b",
                    psf_type="prf",
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
        target_specs_by_method = {"prf_a": target_specs, "prf_b": target_specs}

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
                            target_specs_by_method=target_specs_by_method,
                            phot_stage=stage,
                            epsf_by_workspace={},
                            stage_epsf_workspace=None,
                            tile_centers=tiles,
                            wcs_table=wcs,
                            crop_bounds=crop_bounds,
                            cfg=cfg,
                            output_dir=tmp,
                        )
            self.assertTrue((Path(tmp) / "lightcurve_prf_a.csv").is_file())
            self.assertTrue((Path(tmp) / "lightcurve_prf_b.csv").is_file())

    def test_epsf_without_gridded_raises(self):
        n_ep = 1
        paths = ["/fake/diff_0.fits"]
        wcs = pd.DataFrame({"btjd": [100.0], "group_id": [0]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
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
                    name="epsf",
                    psf_type="epsf",
                    phot_snap="fixed",
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                ph.run_forced_photometry_stage(
                    diff_paths=paths,
                    target_specs_by_method={"epsf": [(xy, None, "primary", {})]},
                    phot_stage=stage,
                    epsf_by_workspace={"epsf_ws": np.zeros((1, 121))},
                    stage_epsf_workspace="epsf_ws",
                    tile_centers=[(50.0, 50.0)],
                    wcs_table=wcs,
                    crop_bounds=crop_bounds,
                    cfg=cfg,
                    output_dir=tmp,
                    gridded_epsf_by_workspace={},
                )
            self.assertIn("gridded", str(ctx.exception).lower())

    def test_dual_epsf_photutils_and_tessreduce_write_two_csvs(self):
        n_ep = 2
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame({"btjd": [100.0, 101.0], "group_id": [0, 0]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
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
                PsfPhotometryMethodParams(name="epsf", psf_type="epsf", fitter="photutils"),
                PsfPhotometryMethodParams(
                    name="epsf_bkg",
                    psf_type="epsf",
                    fitter="tessreduce",
                    phot_bkg_poly_order=0,
                    phot_snap="fixed",
                    phot_cutout_size=15,
                ),
            ]
        )
        catalog = MagicMock()
        catalog.load_model.return_value = None  # epochs → NaN flux; still write CSVs
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ph.os.path, "exists", return_value=True):
                ph.run_forced_photometry_stage(
                    diff_paths=paths,
                    target_specs_by_method={
                        "epsf": [(xy.copy(), None, "primary", {})],
                        "epsf_bkg": [(xy.copy(), None, "primary", {})],
                    },
                    phot_stage=stage,
                    epsf_by_workspace={},
                    stage_epsf_workspace="epsf_r1",
                    tile_centers=[],
                    wcs_table=wcs,
                    crop_bounds=crop_bounds,
                    cfg=cfg,
                    output_dir=tmp,
                    gridded_epsf_by_workspace={"epsf_r1": catalog},
                )
            self.assertTrue((Path(tmp) / "lightcurve_epsf.csv").is_file())
            self.assertTrue((Path(tmp) / "lightcurve_epsf_bkg.csv").is_file())


class TestCreatePsfSurfaceArgs(unittest.TestCase):
    def test_null_means_no_surface(self):
        surface, order = ph._create_psf_surface_args(None)
        self.assertFalse(surface)
        self.assertEqual(order, 0)

    def test_zero_is_constant_surface(self):
        surface, order = ph._create_psf_surface_args(0)
        self.assertTrue(surface)
        self.assertEqual(order, 0)


class TestForcedPhotGrouper(unittest.TestCase):
    def test_grouper_none_by_default(self):
        phot = PsfPhotometryMethodParams(name="epsf", psf_type="epsf")
        self.assertIsNone(phot.psf_grouper_min_separation)

        captured = {}

        class _FakePSFPhotometry:
            def __init__(self, *args, **kwargs):
                captured["grouper"] = kwargs.get("grouper")

            def __call__(self, *args, **kwargs):
                raise RuntimeError("stop after construction")

        image = np.zeros((32, 32), dtype=np.float64)
        model = MagicMock()
        with patch("photutils.psf.PSFPhotometry", _FakePSFPhotometry):
            flux, *_ = ph.forced_phot_gridded_epoch(image, model, 16.0, 16.0, phot)
        self.assertTrue(np.isnan(flux))
        self.assertIsNone(captured["grouper"])


if __name__ == "__main__":
    unittest.main()
