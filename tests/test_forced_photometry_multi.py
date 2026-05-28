"""Tests for multi-target forced photometry (file-major FITS batching)."""
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

import photometry as ph
from stage_params import ForcedPhotometryParams


def _tiny_epsf_locator():
    """Normalized stub ePSF for ``create_psf`` (avoids PRF / Gaia dependencies)."""
    g = np.linspace(-1, 1, 11)
    xx, yy = np.meshgrid(g, g)
    k = np.exp(-(xx**2 + yy**2) / 0.5)
    k /= k.sum()
    return ph.EpsfLocator(k.astype(np.float64), os_factor=2)


class TestForcedPhotometryMultiReadCount(unittest.TestCase):
    def setUp(self) -> None:
        self.read_calls = 0

    def _fake_read(self, path: str):
        self.read_calls += 1
        return np.zeros((32, 32), dtype=np.float64), np.ones((32, 32), dtype=np.float64)

    def _minimal_cfg(self, *, n_jobs: int = 1):
        return SimpleNamespace(
            sector=20,
            camera=3,
            ccd=3,
            n_jobs=n_jobs,
            pipeline_plots=False,
            pipeline_plot_dpi=150,
        )

    def _minimal_phot(self, *, phot_snap: str = "fixed") -> ForcedPhotometryParams:
        return ForcedPhotometryParams(
            psf_type="epsf",
            psf_size=5,
            phot_cutout_size=15,
            phot_bkg_poly_order=1,
            epsf_oversample=2,
            tile_nx=1,
            tile_ny=1,
            phot_snap=phot_snap,
        )

    def test_multi_two_sources_fixed_snap_reads_each_file_once(self):
        """S=2, E=3, fixed snap: 3 FITS reads total (not 6)."""
        n_ep = 3
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame(
            {
                "btjd": [100.0, 101.0, 102.0],
                "group_id": [0, 0, 0],
            }
        )
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
        epsf = np.zeros((1, 121))
        tiles = [(50.0, 50.0)]
        cfg = self._minimal_cfg(n_jobs=1)
        phot = self._minimal_phot(phot_snap="fixed")

        t0 = ph.ForcedPhotTargetSpec(
            target_xy=xy.copy(),
            csv_basename="lightcurve.csv",
            plot_source_label="primary",
            tag="primary",
        )
        t1 = ph.ForcedPhotTargetSpec(
            target_xy=xy.copy(),
            csv_basename="lightcurve_b.csv",
            plot_source_label="b",
            tag="extra[0]",
        )

        loc = _tiny_epsf_locator()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ph, "read_diff_primary_and_noise_sigma", side_effect=self._fake_read):
                with patch.object(ph, "build_psf_kernel", return_value=loc):
                    with patch.object(ph.os.path, "exists", return_value=True):
                        ph.run_forced_photometry_multi(
                            paths,
                            [t0, t1],
                            epsf,
                            tiles,
                            wcs,
                            crop_bounds,
                            cfg,
                            phot,
                            output_dir=tmp,
                            ref_frame_index=None,
                            plot_title_suffix="lc_ws",
                        )

        self.assertEqual(
            self.read_calls,
            n_ep,
            "multi-target fixed snap should read each epoch once in the flux phase",
        )

    def test_multi_brightest_snap_reads_twice_per_epoch(self):
        """Brightest scan + flux: 2 reads per epoch (batch path, S>=2)."""
        n_ep = 2
        paths = [f"/fake/diff_{i}.fits" for i in range(n_ep)]
        wcs = pd.DataFrame(
            {
                "btjd": [100.0, 101.0],
                "group_id": [0, 0],
            }
        )
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        xy = np.full((n_ep, 2), 16.0, dtype=np.float64)
        epsf = np.zeros((1, 121))
        tiles = [(50.0, 50.0)]
        cfg = self._minimal_cfg(n_jobs=1)
        phot = self._minimal_phot(phot_snap="brightest")

        targets = [
            ph.ForcedPhotTargetSpec(
                target_xy=xy.copy(),
                csv_basename="a.csv",
                tag="a",
            ),
            ph.ForcedPhotTargetSpec(
                target_xy=xy.copy(),
                csv_basename="b.csv",
                tag="b",
            ),
        ]

        loc = _tiny_epsf_locator()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ph, "read_diff_primary_and_noise_sigma", side_effect=self._fake_read):
                with patch.object(ph, "build_psf_kernel", return_value=loc):
                    with patch.object(ph.os.path, "exists", return_value=True):
                        ph.run_forced_photometry_multi(
                            paths,
                            targets,
                            epsf,
                            tiles,
                            wcs,
                            crop_bounds,
                            cfg,
                            phot,
                            output_dir=tmp,
                            plot_title_suffix="x",
                        )

        self.assertEqual(self.read_calls, 2 * n_ep)


if __name__ == "__main__":
    unittest.main()
