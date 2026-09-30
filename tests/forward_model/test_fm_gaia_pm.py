# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for Gaia proper-motion propagation."""

from __future__ import annotations

import numpy as np
from astropy.time import Time

from syndiff_pipeline.forward_model.gaia_pm import (
    GAIA_DR3_EPOCH,
    angular_shift_arcsec,
    apply_pm_to_dataframe,
    btjd_to_time,
    propagate_ra_dec,
)


def test_btjd_to_time_offset():
    t = btjd_to_time(1325.0)
    assert abs(t.jd - (1325.0 + 2457000.0)) < 1e-6


def test_zero_pm_no_shift():
    ra = np.array([100.0, 200.0])
    dec = np.array([-10.0, 20.0])
    pmra = np.zeros(2)
    pmdec = np.zeros(2)
    obstime = Time(2020.0, format="jyear")
    ra1, dec1 = propagate_ra_dec(ra, dec, pmra, pmdec, obstime)
    assert np.allclose(ra1, ra)
    assert np.allclose(dec1, dec)


def test_pm_shift_scale_barnard_star_order():
    # ~10 arcsec/yr PM over ~4 yr -> tens of arcsec (sanity on units).
    ra = np.array([269.45])
    dec = np.array([4.67])
    pmra = np.array([-798.0])
    pmdec = np.array([10328.0])
    obstime = Time(2020.0, format="jyear")
    ra1, dec1 = propagate_ra_dec(ra, dec, pmra, pmdec, obstime, ref_epoch=GAIA_DR3_EPOCH)
    shift = angular_shift_arcsec(ra, dec, ra1, dec1)[0]
    assert 30.0 < shift < 60.0


def test_apply_pm_dataframe_missing_columns():
    import pandas as pd

    df = pd.DataFrame({"ra": [1.0], "dec": [2.0], "source_id": [1]})
    out, stats = apply_pm_to_dataframe(df, 1325.0)
    assert not stats["applied"]
    assert out["ra"].iloc[0] == 1.0
