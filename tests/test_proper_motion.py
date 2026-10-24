"""Opt-in Gaia proper-motion propagation (common/proper_motion.py and its call sites)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.proper_motion import (
    GAIA_DR3_EPOCH_BTJD,
    frame_epoch_btjd,
    propagate_gaia_df,
    propagate_radec,
)
from syndiff_pipeline.common.provenance.model import epsf_recipe_params
from syndiff_pipeline.common.wcs_header_cache import extract_ffi_header_record
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    CentroidsParams,
    EpsfParams,
    parse_centroids,
    parse_epsf,
    parse_per_ffi_wcs,
    parse_temporal_wcs,
)
from syndiff_pipeline.difference_imaging.wcs.sci2idl import join_stars

from test_gaia_science_xy import _sip_ffi_header, _write_test_ffi

EPOCH_4YR = GAIA_DR3_EPOCH_BTJD + 4.0 * 365.25


class TestHelper(unittest.TestCase):
    def test_matches_astropy_space_motion(self):
        ra = np.array([214.84, 10.0, 300.0, 45.0])
        dec = np.array([70.7, -20.0, 85.0, 5.0])
        pmra = np.array([-3.9, 120.0, 40.0, -80.0])
        pmdec = np.array([-7.0, 60.0, -90.0, 30.0])
        ra2, dec2, n = propagate_radec(ra, dec, pmra, pmdec, EPOCH_4YR)
        self.assertEqual(n, 0)
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, pm_ra_cosdec=pmra * u.mas / u.yr,
                     pm_dec=pmdec * u.mas / u.yr, obstime=Time(2457389.0, format="jd"))
        ref = c.apply_space_motion(dt=4.0 * u.yr)
        sep = SkyCoord(ra2 * u.deg, dec2 * u.deg).separation(ref).arcsec
        # linear tangent-plane step vs great-circle: second order (pm*dt)^2 tan(dec)
        self.assertLess(sep.max(), 1e-5, sep)
        # dec 5..70 deg, pm<=130 mas/yr: tighter than 1e-6 arcsec for the non-polar stars
        self.assertLess(sep[:2].max(), 1e-6, sep)

    def test_nan_pm_passthrough(self):
        ra2, dec2, n = propagate_radec([10.0, 20.0], [5.0, 6.0], [np.nan, 1.0], [2.0, np.nan], EPOCH_4YR)
        np.testing.assert_array_equal(ra2, [10.0, 20.0])
        np.testing.assert_array_equal(dec2, [5.0, 6.0])
        self.assertEqual(n, 2)

    def test_zero_dt_identity(self):
        ra2, dec2, _ = propagate_radec([10.0], [5.0], [100.0], [100.0], GAIA_DR3_EPOCH_BTJD)
        self.assertEqual((ra2[0], dec2[0]), (10.0, 5.0))

    def test_df_keeps_raw(self):
        df = pd.DataFrame({"ra": [10.0], "dec": [5.0], "pmra": [100.0], "pmdec": [-50.0]})
        out = propagate_gaia_df(df, EPOCH_4YR)
        self.assertEqual(out["ra_j2016"][0], 10.0)
        self.assertNotEqual(out["ra"][0], 10.0)
        self.assertEqual(df["ra"][0], 10.0)  # input untouched
        with self.assertRaises(ValueError):
            propagate_gaia_df(df.drop(columns=["pmra"]), EPOCH_4YR)

    def test_frame_epoch(self):
        h = fits.Header({"TSTART": 1844.5, "TSTOP": 1844.7})
        self.assertAlmostEqual(frame_epoch_btjd(h), 1844.6)
        h = fits.Header({"DATE-OBS": "2019-12-27T00:00:00", "DATE-END": "2019-12-27T12:00:00"})
        want = Time("2019-12-27T06:00:00", format="isot", scale="utc").jd - 2457000.0
        self.assertAlmostEqual(frame_epoch_btjd(h), want, places=6)
        self.assertTrue(np.isnan(frame_epoch_btjd(fits.Header())))


class TestProjection(unittest.TestCase):
    def _setup(self, tmp):
        hdr = _sip_ffi_header()
        hdr["TSTART"] = EPOCH_4YR - 0.01
        hdr["TSTOP"] = EPOCH_4YR + 0.01
        path = Path(tmp) / "tess_s0020_3_3_ffic.fits"
        _write_test_ffi(path, hdr)
        row = extract_ffi_header_record(path, open_fits=wcs_grouping.open_fits_memmap)
        ffi_list_df = pd.DataFrame([row]).set_index("filename")
        bounds = {"x_min": 0, "x_max": 200, "y_min": 0, "y_max": 200, "shape": (200, 200)}
        gaia = pd.DataFrame({"source_id": [1, 2], "ra": [100.0, 100.001], "dec": [20.0, 20.001],
                             "pmra": [500.0, np.nan], "pmdec": [-300.0, np.nan]})
        return str(path), ffi_list_df, bounds, gaia

    def test_off_identical_and_on_shifts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, lst, bounds, gaia = self._setup(tmp)
            base = wcs_grouping.gaia_science_xy_for_frame(gaia, path, lst, bounds)
            off = wcs_grouping.gaia_science_xy_for_frame(
                gaia, path, lst, bounds, apply_proper_motion=False)
            pd.testing.assert_frame_equal(base, off)
            on = wcs_grouping.gaia_science_xy_for_frame(
                gaia, path, lst, bounds, apply_proper_motion=True)
            # star 1: pmra 500 mas/yr * 4 yr = 2" = 2/21 px eastward -> x decreases (CD1_1<0 here: 0.36"/px)
            dx = on["x"][0] - base["x"][0]
            dy = on["y"][0] - base["y"][0]
            self.assertAlmostEqual(np.hypot(dx, dy), np.hypot(2.0, -1.2) / 0.36, places=2)
            # star 2 (NaN pm) unmoved
            self.assertEqual(on["x"][1], base["x"][1])
            self.assertEqual(on["ra_j2016"][0], 100.0)

    def test_join_stars_epoch(self):
        gaia = pd.DataFrame({"source_id": [1], "ra": [10.0], "dec": [5.0], "x": [50.0], "y": [60.0],
                             "pmra": [100.0], "pmdec": [0.0]})
        phot = Table({"x_init": [50.01], "y_init": [60.0]})
        off = join_stars(phot, gaia)
        on = join_stars(phot, gaia, epoch_btjd=EPOCH_4YR)
        self.assertEqual(off["ra"][0], 10.0)
        self.assertGreater(on["ra"][0], 10.0)
        self.assertEqual(on["ra_j2016"][0], 10.0)
        with self.assertRaises(ValueError):
            join_stars(phot, gaia.drop(columns=["pmra"]), epoch_btjd=EPOCH_4YR)


class TestParams(unittest.TestCase):
    def test_default_off_and_fingerprint(self):
        self.assertIsNone(parse_epsf({"kind": "epsf"}, 0).apply_proper_motion)
        base = epsf_recipe_params(parse_epsf({"kind": "epsf"}, 0))
        self.assertNotIn("apply_proper_motion", base)
        self.assertEqual(base, epsf_recipe_params(parse_epsf({"kind": "epsf", "apply_proper_motion": False}, 0)))
        self.assertEqual(base, epsf_recipe_params(EpsfParams()))
        on = epsf_recipe_params(parse_epsf({"kind": "epsf", "apply_proper_motion": True}, 0))
        self.assertTrue(on["apply_proper_motion"])
        self.assertNotEqual(base, on)

    def test_all_stages_parse(self):
        for fn, kind in ((parse_epsf, "epsf"), (parse_centroids, "centroids"),
                         (parse_per_ffi_wcs, "per_ffi_wcs"), (parse_temporal_wcs, "temporal_wcs")):
            self.assertTrue(fn({"kind": kind, "apply_proper_motion": True}, 0).apply_proper_motion)
            self.assertIsNone(fn({"kind": kind}, 0).apply_proper_motion)
            with self.assertRaises(ValueError):
                fn({"kind": kind, "apply_proper_motion": "yes"}, 0)

    def test_centroids_fingerprint_dict_unchanged(self):
        import dataclasses
        d = dataclasses.asdict(CentroidsParams())
        for k in getattr(CentroidsParams, "_RECIPE_OMIT_WHEN_NONE"):
            if d.get(k) is None:
                d.pop(k)
        self.assertNotIn("apply_proper_motion", d)


if __name__ == "__main__":
    unittest.main()
