"""Tests for the combined multi-method gridded-ePSF forced photometry pass.

Several forced_photometry methods (e.g. xy_free / ffi_wcs / temporal_wcs, all
psf_type: epsf with the default fitter: photutils) now share one read of
each diff FITS per epoch instead of each method re-reading the whole
diff_paths list on its own -- see
run_forced_photometry_gridded_multi_method / _forced_phot_gridded_multi_flux_worker
in syndiff_pipeline/difference_imaging/stages/photometry.py.
"""

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
    ForcedPhotometryParams,
    PsfPhotometryMethodParams,
)
from syndiff_pipeline.difference_imaging.stages import photometry as ph


def _cfg(**kw):
    base = dict(n_jobs=1, pipeline_plots=False, pipeline_plot_dpi=150)
    base.update(kw)
    return SimpleNamespace(**base)


class TestGriddedMultiMethodSharedIO(unittest.TestCase):
    def setUp(self):
        self.n_epochs = 3
        self.diff_paths = [f"/fake/diff_{i}.fits" for i in range(self.n_epochs)]
        self.wcs = pd.DataFrame(
            {
                "btjd": [100.0, 101.0, 102.0],
                "group_id": [0, 0, 1],
            }
        )
        self.catalog = MagicMock()
        self.catalog.workspace_dir = "epsf_r1"
        self.catalog.load_model.return_value = "MODEL"

        self.method_a = PsfPhotometryMethodParams(
            name="a", psf_type="epsf", fixed_position=False
        )
        self.method_b = PsfPhotometryMethodParams(
            name="b", psf_type="epsf", fixed_position=True
        )
        self.xy_a = np.full((self.n_epochs, 2), (10.0, 20.0), dtype=np.float64)
        self.xy_b = np.full((self.n_epochs, 2), (30.0, 40.0), dtype=np.float64)

    def _fake_forced_phot(self, image, model, x, y, phot, error=None):
        # Flux uniquely encodes (x, y) so the test can tell which method's
        # target_xy actually reached the fit.
        return float(x) + float(y) / 100.0, 0.0, float(x), float(y)

    def test_reads_each_diff_path_once_regardless_of_method_count(self):
        entries = [
            (self.method_a, self.catalog, self.xy_a, "lc_a.csv", None, "a"),
            (self.method_b, self.catalog, self.xy_b, "lc_b.csv", None, "b"),
        ]
        read_mock = MagicMock(return_value=(np.zeros((8, 8)), None))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ph, "read_diff_primary_and_noise_sigma", read_mock):
                with patch.object(ph, "forced_phot_gridded_epoch", side_effect=self._fake_forced_phot):
                    with patch.object(ph.os.path, "exists", return_value=True):
                        dfs = ph.run_forced_photometry_gridded_multi_method(
                            diff_paths=self.diff_paths,
                            entries=entries,
                            wcs_table=self.wcs,
                            cfg=_cfg(),
                            output_dir=tmp,
                        )
            # One read per epoch total, not one per (epoch, method).
            self.assertEqual(read_mock.call_count, self.n_epochs)
            # One model load per epoch too, since both methods share the
            # same catalog/workspace within an epoch.
            self.assertEqual(self.catalog.load_model.call_count, self.n_epochs)

            self.assertEqual(len(dfs), 2)
            df_a, df_b = dfs
            self.assertTrue(np.allclose(df_a["flux"], 10.0 + 20.0 / 100.0))
            self.assertTrue(np.allclose(df_b["flux"], 30.0 + 40.0 / 100.0))
            self.assertTrue(np.allclose(df_a["x_fit"], 10.0))
            self.assertTrue(np.allclose(df_b["x_fit"], 30.0))
            self.assertTrue((Path(tmp) / "lc_a.csv").is_file())
            self.assertTrue((Path(tmp) / "lc_b.csv").is_file())

    def test_uppercase_BTJD_does_not_reopen_fits_for_timestamps(self):
        """SCC frames.csv copies BTJD (uppercase). The combined pass must use
        that column instead of a serial read_diff_btjd pre-pass over every hp_d.
        """
        wcs = pd.DataFrame(
            {
                "BTJD": [200.0, 201.0, 202.0],
                "group_id": [0, 0, 1],
            }
        )
        entries = [
            (self.method_a, self.catalog, self.xy_a, "lc_a.csv", None, "a"),
        ]
        read_mock = MagicMock(return_value=(np.zeros((8, 8)), None))
        btjd_mock = MagicMock(side_effect=AssertionError("read_diff_btjd must not run"))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ph, "read_diff_primary_and_noise_sigma", read_mock):
                with patch.object(ph, "read_diff_btjd", btjd_mock):
                    with patch.object(ph, "forced_phot_gridded_epoch", side_effect=self._fake_forced_phot):
                        with patch.object(ph.os.path, "exists", return_value=True):
                            dfs = ph.run_forced_photometry_gridded_multi_method(
                                diff_paths=self.diff_paths,
                                entries=entries,
                                wcs_table=wcs,
                                cfg=_cfg(),
                                output_dir=tmp,
                            )
            self.assertEqual(read_mock.call_count, self.n_epochs)
            btjd_mock.assert_not_called()
            self.assertTrue(np.allclose(dfs[0]["btjd"], [200.0, 201.0, 202.0]))

    def test_matches_running_each_method_separately(self):
        """The combined pass must give bit-identical output to running
        _run_forced_photometry_gridded_single once per method (the old,
        unbatched behaviour) -- only the I/O pattern changes, not the math.
        """
        self.catalog.load_model.return_value = SimpleNamespace(
            x_0=SimpleNamespace(fixed=False), y_0=SimpleNamespace(fixed=False)
        )
        with tempfile.TemporaryDirectory() as tmp_combined, tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            read_mock = MagicMock(return_value=(np.ones((8, 8)) * 5.0, None))
            with patch.object(ph, "read_diff_primary_and_noise_sigma", read_mock):
                with patch.object(ph.os.path, "exists", return_value=True):
                    with patch("photutils.psf.PSFPhotometry") as MockPSF:
                        instance = MockPSF.return_value
                        table = {
                            "flux_fit": [7.0],
                            "flux_err": [0.5],
                            "x_fit": [11.0],
                            "y_fit": [21.0],
                        }
                        fake_result = MagicMock()
                        fake_result.colnames = list(table.keys())
                        fake_result.__getitem__.side_effect = table.__getitem__
                        instance.return_value = fake_result

                        dfs = ph.run_forced_photometry_gridded_multi_method(
                            diff_paths=self.diff_paths,
                            entries=[
                                (self.method_a, self.catalog, self.xy_a, "lc_a.csv", None, "a"),
                            ],
                            wcs_table=self.wcs,
                            cfg=_cfg(),
                            output_dir=tmp_combined,
                        )
                        df_combined = dfs[0]

                        df_single = ph._run_forced_photometry_gridded_single(
                            diff_paths=self.diff_paths,
                            target_xy=self.xy_a,
                            gridded_catalog=self.catalog,
                            wcs_table=self.wcs,
                            cfg=_cfg(),
                            phot=self.method_a,
                            output_dir=tmp_a,
                            lightcurve_csv_filename="lc_single.csv",
                        )
            pd.testing.assert_frame_equal(
                df_combined.reset_index(drop=True), df_single.reset_index(drop=True)
            )

    def test_fixed_position_does_not_leak_across_shared_model(self):
        """forced_phot_gridded_epoch must reset .fixed both ways so a model
        instance reused across methods in one epoch (see model_cache in
        _forced_phot_gridded_multi_flux_worker) can't inherit a stale flag.
        """
        model = SimpleNamespace(x_0=SimpleNamespace(fixed=None), y_0=SimpleNamespace(fixed=None))

        class _StopAfterConstruction:
            def __init__(self, *a, **kw):
                pass

            def __call__(self, *a, **kw):
                raise RuntimeError("stop after construction")

        with patch("photutils.psf.PSFPhotometry", _StopAfterConstruction):
            ph.forced_phot_gridded_epoch(
                np.zeros((8, 8)), model, 1.0, 1.0, self.method_b  # fixed_position=True
            )
            self.assertTrue(model.x_0.fixed)
            self.assertTrue(model.y_0.fixed)

            ph.forced_phot_gridded_epoch(
                np.zeros((8, 8)), model, 1.0, 1.0, self.method_a  # fixed_position=False
            )
            self.assertFalse(model.x_0.fixed)
            self.assertFalse(model.y_0.fixed)


class TestForcedPhotometryStageMultiPositionSource(unittest.TestCase):
    """End-to-end through run_forced_photometry_stage: methods with
    different resolved position_source but the same underlying epsf
    workspace share one I/O pass, and each still gets its own target_xy.
    """

    def test_two_position_sources_one_stage_correct_per_method_xy(self):
        n_ep = 2
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame({"btjd": [100.0, 101.0], "group_id": [0, 0]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy_native = np.full((n_ep, 2), (5.0, 6.0), dtype=np.float64)
        xy_temporal = np.full((n_ep, 2), (50.0, 60.0), dtype=np.float64)

        stage = ForcedPhotometryParams(
            methods=[
                PsfPhotometryMethodParams(name="ffiwcs", psf_type="epsf"),
                PsfPhotometryMethodParams(name="temporalwcs", psf_type="epsf"),
            ]
        )
        target_specs_by_method = {
            "ffiwcs": [(xy_native, None, "primary", {})],
            "temporalwcs": [(xy_temporal, None, "primary", {})],
        }
        catalog = MagicMock()
        catalog.workspace_dir = "epsf_r1"
        catalog.load_model.return_value = "MODEL"

        def _fake_forced_phot(image, model, x, y, phot, error=None):
            return float(x) + float(y) / 100.0, 0.0, float(x), float(y)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                ph, "read_diff_primary_and_noise_sigma", return_value=(np.zeros((8, 8)), None)
            ):
                with patch.object(ph, "forced_phot_gridded_epoch", side_effect=_fake_forced_phot):
                    with patch.object(ph.os.path, "exists", return_value=True):
                        results = ph.run_forced_photometry_stage(
                            diff_paths=paths,
                            target_specs_by_method=target_specs_by_method,
                            phot_stage=stage,
                            epsf_by_workspace={},
                            stage_epsf_workspace="epsf_r1",
                            tile_centers=[],
                            wcs_table=wcs,
                            crop_bounds=crop_bounds,
                            cfg=_cfg(),
                            output_dir=tmp,
                            gridded_epsf_by_workspace={"epsf_r1": catalog},
                        )
            df_native = results["ffiwcs"][0]
            df_temporal = results["temporalwcs"][0]
            self.assertTrue(np.allclose(df_native["x_fit"], 5.0))
            self.assertTrue(np.allclose(df_temporal["x_fit"], 50.0))
            self.assertTrue(np.allclose(df_native["flux"], 5.0 + 6.0 / 100.0))
            self.assertTrue(np.allclose(df_temporal["flux"], 50.0 + 60.0 / 100.0))


class TestBtjdColumnFromWcsTable(unittest.TestCase):
    def test_prefers_uppercase_BTJD(self):
        wcs = pd.DataFrame({"BTJD": [1.5, 2.5], "group_id": [0, 1]})
        got = ph._btjd_column_from_wcs_table(wcs, 2)
        self.assertTrue(np.allclose(got, [1.5, 2.5]))

    def test_lowercase_btjd_still_works(self):
        wcs = pd.DataFrame({"btjd": [9.0]})
        got = ph._btjd_column_from_wcs_table(wcs, 1)
        self.assertTrue(np.allclose(got, [9.0]))

    def test_missing_column_is_nan(self):
        wcs = pd.DataFrame({"group_id": [0, 1]})
        got = ph._btjd_column_from_wcs_table(wcs, 2)
        self.assertEqual(len(got), 2)
        self.assertTrue(np.all(~np.isfinite(got)))


if __name__ == "__main__":
    unittest.main()
