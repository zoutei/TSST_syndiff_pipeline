"""Tests for syndiff_pipeline.star.windowed_photometry."""

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

from syndiff_pipeline.difference_imaging.stages.photometry import EpsfLocator
from syndiff_pipeline.star import diff_runner
from syndiff_pipeline.star.identifiers import ResolvedHost
from syndiff_pipeline.star import windowed_photometry


def _resolved_host(gaia_id: int = 1060421588522505216) -> ResolvedHost:
    return ResolvedHost(
        input_kind="gaia",
        input_value=gaia_id,
        tic_id=None,
        gaia_source_id=gaia_id,
        ra=0.0,
        dec=0.0,
        phot_g_mean_mag=None,
        phot_bp_mean_mag=None,
        phot_rp_mean_mag=None,
        resolution_method="test",
        label=None,
    )



def _make_epsf_locator(psf_size: int = 11, os_factor: int = 2) -> EpsfLocator:
    over_size = 2 * psf_size + 1
    yy, xx = np.mgrid[0:over_size, 0:over_size]
    center = (over_size - 1) / 2.0
    model = np.exp(-0.5 * (((xx - center) / 1.0) ** 2 + ((yy - center) / 1.0) ** 2))
    model /= model.sum()
    return EpsfLocator(model, os_factor)


class TestReadStarDiffStamp(unittest.TestCase):
    def test_round_trip_via_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stamp.fits.fz")
            stamp = np.arange(64, dtype=np.float32).reshape(8, 8)
            written = diff_runner.write_star_diff_stamp(
                path,
                stamp,
                window_origin=(100, 200),
                host_local_xy=(104.5, 206.25),
            )
            data, header = windowed_photometry.read_star_diff_stamp(written)
            np.testing.assert_allclose(data, stamp, rtol=1e-6)
            self.assertEqual(header["xmin"], 100)
            self.assertEqual(header["ymin"], 200)
            self.assertAlmostEqual(header["host_x"], 4.5)
            self.assertAlmostEqual(header["host_y"], 6.25)


class TestApertureFluxOnStamp(unittest.TestCase):
    def test_recovers_injected_point_source_flux(self):
        host_xy = (12.0, 12.0)
        background = 7.5
        source_flux = 42.0
        stamp = np.full((25, 25), background, dtype=np.float64)
        stamp[int(round(host_xy[1])), int(round(host_xy[0]))] += source_flux
        result = windowed_photometry.aperture_flux_on_stamp(
            stamp,
            host_xy,
            tar_ap=3.0,
            sky_in=5.0,
            sky_out=9.0,
        )
        self.assertAlmostEqual(result["sky_median"], background, places=6)
        self.assertAlmostEqual(result["flux"], source_flux, places=6)
        self.assertAlmostEqual(result["aperture_sum_raw"], background * 9 + source_flux, places=6)


class TestPsfFluxOnStamp(unittest.TestCase):
    def test_recovers_injected_psf_flux(self):
        host_xy = (12.0, 12.0)
        background = 5.0
        source_flux = 30.0
        epsf = _make_epsf_locator()
        psf_stamp = epsf.locate(host_xy[0], host_xy[1], (25, 25))
        stamp = background + source_flux * psf_stamp
        result = windowed_photometry.psf_flux_on_stamp(
            stamp,
            host_xy,
            epsf,
            psf_size=11,
        )
        self.assertAlmostEqual(result["flux"], source_flux, delta=4.0)
        self.assertTrue(np.isfinite(result["flux_err"]))


class TestRunWindowedForcedPhotometry(unittest.TestCase):
    def test_batch_light_curve_and_csv_output(self):
        host = _resolved_host()
        host_xy = (12.0, 12.0)
        background = 4.0
        fluxes = [10.0, 20.0, 15.0]
        btjds = [2459000.1, 2459000.2, 2459000.3]

        with tempfile.TemporaryDirectory() as tmp:
            stamp_paths = []
            for i, flux in enumerate(fluxes):
                stamp = np.full((25, 25), background, dtype=np.float64)
                stamp[int(round(host_xy[1])), int(round(host_xy[0]))] += flux
                path = os.path.join(tmp, "stamps", f"frame_{i}.fits.fz")
                written = diff_runner.write_star_diff_stamp(
                    path,
                    stamp.astype(np.float32),
                    window_origin=(50 + i, 60 + i),
                    host_local_xy=(50 + i + host_xy[0], 60 + i + host_xy[1]),
                )
                stamp_paths.append(written)

            out_dir = os.path.join(tmp, "lc")
            dfs = windowed_photometry.run_windowed_forced_photometry(
                stamp_paths,
                host=host,
                methods=[
                    {
                        "name": "ap3",
                        "type": "aperture",
                        "tar_ap": 3,
                        "sky_in": 5,
                        "sky_out": 9,
                    },
                ],
                output_dir=out_dir,
                time_values=btjds,
            )

            self.assertEqual(set(dfs.keys()), {"ap3"})
            ap_df = dfs["ap3"]
            self.assertEqual(len(ap_df), len(fluxes))
            np.testing.assert_allclose(ap_df["btjd"].values, btjds)
            recovered = ap_df["flux_wo_sky"].values
            np.testing.assert_allclose(recovered, fluxes, rtol=0, atol=1e-3)

            ap_csv = os.path.join(
                out_dir,
                f"lightcurve_ap3_gaia_{host.gaia_source_id}.csv",
            )
            self.assertTrue(os.path.isfile(ap_csv))

    def test_psf_batch_writes_csv(self):
        host = _resolved_host()
        host_xy = (12.0, 12.0)
        background = 5.0
        fluxes = [30.0, 45.0]
        epsf = _make_epsf_locator()

        with tempfile.TemporaryDirectory() as tmp:
            stamp_paths = []
            for i, flux in enumerate(fluxes):
                psf_model = epsf.locate(host_xy[0], host_xy[1], (25, 25))
                stamp = background + flux * psf_model
                path = os.path.join(tmp, "stamps", f"frame_{i}.fits.fz")
                written = diff_runner.write_star_diff_stamp(
                    path,
                    stamp.astype(np.float32),
                    window_origin=(10 + i, 20 + i),
                    host_local_xy=(10 + i + host_xy[0], 20 + i + host_xy[1]),
                )
                stamp_paths.append(written)

            out_dir = os.path.join(tmp, "lc")
            dfs = windowed_photometry.run_windowed_forced_photometry(
                stamp_paths,
                host=host,
                methods=[
                    {
                        "name": "prf",
                        "type": "psf",
                        "epsf_model": epsf,
                        "psf_size": 11,
                    },
                ],
                output_dir=out_dir,
            )
            prf_df = dfs["prf"]
            self.assertEqual(len(prf_df), len(fluxes))
            for got, expected in zip(prf_df["flux"].values, fluxes):
                self.assertAlmostEqual(got, expected, delta=max(6.0, expected * 0.15))
            prf_csv = os.path.join(
                out_dir,
                f"lightcurve_prf_gaia_{host.gaia_source_id}.csv",
            )
            self.assertTrue(os.path.isfile(prf_csv))


if __name__ == "__main__":
    unittest.main()
