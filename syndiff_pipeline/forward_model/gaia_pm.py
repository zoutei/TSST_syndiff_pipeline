# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Gaia DR3 proper-motion propagation for forward-modeled astrometry."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

try:
    from erfa import ErfaWarning
except ImportError:  # pragma: no cover
    ErfaWarning = Warning  # type: ignore[misc, assignment]

GAIA_DR3_EPOCH = Time(2016.0, format="jyear")
_BTJD_OFFSET = 2457000.0

_PM_COLS = ("pm", "pmra", "pmra_error", "pmdec", "pmdec_error")


def btjd_to_time(btjd: float | np.ndarray) -> Time:
    """Convert TESS BTJD to astropy Time (TDB, JD)."""
    jd = np.asarray(btjd, dtype=np.float64) + _BTJD_OFFSET
    return Time(jd, format="jd", scale="tdb")


def propagate_ra_dec(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pmra_mas_yr: np.ndarray,
    pmdec_mas_yr: np.ndarray,
    obstime: Time,
    *,
    ref_epoch: Time = GAIA_DR3_EPOCH,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate catalog positions from ``ref_epoch`` to ``obstime`` (degrees)."""
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    pmra = np.asarray(pmra_mas_yr, dtype=np.float64)
    pmdec = np.asarray(pmdec_mas_yr, dtype=np.float64)
    finite = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(pmra)
        & np.isfinite(pmdec)
    )
    ra_out = ra.copy()
    dec_out = dec.copy()
    if not np.any(finite):
        return ra_out, dec_out
    coords = SkyCoord(
        ra=ra[finite] * u.deg,
        dec=dec[finite] * u.deg,
        pm_ra_cosdec=pmra[finite] * u.mas / u.yr,
        pm_dec=pmdec[finite] * u.mas / u.yr,
        obstime=ref_epoch,
    )
    # ERFA pmsafe Note 6: infinite distance assumed when parallax is absent.
    # Expected for most Gaia rows; tangential PM is unchanged at TESS scales.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*distance overridden \(Note 6\).*",
            category=ErfaWarning,
        )
        prop = coords.apply_space_motion(obstime)
    ra_out[finite] = prop.ra.deg
    dec_out[finite] = prop.dec.deg
    return ra_out, dec_out


def angular_shift_arcsec(
    ra0_deg: np.ndarray,
    dec0_deg: np.ndarray,
    ra1_deg: np.ndarray,
    dec1_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle separation between two sky positions (arcsec)."""
    c0 = SkyCoord(ra=ra0_deg * u.deg, dec=dec0_deg * u.deg)
    c1 = SkyCoord(ra=ra1_deg * u.deg, dec=dec1_deg * u.deg)
    return c0.separation(c1).to_value(u.arcsec)


def apply_pm_to_dataframe(
    df: pd.DataFrame,
    target_btjd: float,
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> tuple[pd.DataFrame, dict]:
    """
    Replace ``ra``/``dec`` with positions propagated to ``target_btjd``.

    Rows without finite ``pmra``/``pmdec`` keep catalog epoch coordinates.
  """
    out = df.copy()
    if "pmra" not in out.columns or "pmdec" not in out.columns:
        return out, {"applied": False, "reason": "no_pm_columns"}
    obstime = btjd_to_time(target_btjd)
    ra0 = out[ra_col].to_numpy(dtype=np.float64)
    dec0 = out[dec_col].to_numpy(dtype=np.float64)
    pmra = out["pmra"].to_numpy(dtype=np.float64)
    pmdec = out["pmdec"].to_numpy(dtype=np.float64)
    ra1, dec1 = propagate_ra_dec(ra0, dec0, pmra, pmdec, obstime)
    out[ra_col] = ra1
    out[dec_col] = dec1
    shift = angular_shift_arcsec(ra0, dec0, ra1, dec1)
    finite = np.isfinite(shift)
    stats: dict = {
        "applied": True,
        "target_btjd": float(target_btjd),
        "ref_epoch_jyear": float(GAIA_DR3_EPOCH.jyear),
        "n_rows": int(len(out)),
        "n_pm_finite": int(np.isfinite(pmra).sum()),
        "shift_arcsec_max": float(np.nanmax(shift[finite])) if finite.any() else 0.0,
        "shift_arcsec_median": float(np.nanmedian(shift[finite])) if finite.any() else 0.0,
        "shift_arcsec_p99": float(np.nanpercentile(shift[finite], 99)) if finite.any() else 0.0,
    }
    if finite.any():
        idx = int(np.nanargmax(shift))
        stats["max_shift_source_id"] = (
            int(out.iloc[idx]["source_id"]) if "source_id" in out.columns else None
        )
        stats["max_shift_tess_mag"] = (
            float(out.iloc[idx]["tess_mag"]) if "tess_mag" in out.columns else None
        )
    return out, stats


def pm_shift_summary_for_catalog(
    df: pd.DataFrame,
    target_btjd: float,
    *,
    tess_mag_lt: float | None = None,
) -> dict:
    """Summarize PM shifts without mutating ``df``."""
    sub = df
    if tess_mag_lt is not None and "tess_mag" in sub.columns:
        sub = sub[sub["tess_mag"] < tess_mag_lt]
    if len(sub) == 0 or "pmra" not in sub.columns:
        return {"n": 0}
    ra0 = sub["ra"].to_numpy(dtype=np.float64)
    dec0 = sub["dec"].to_numpy(dtype=np.float64)
    pmra = sub["pmra"].to_numpy(dtype=np.float64)
    pmdec = sub["pmdec"].to_numpy(dtype=np.float64)
    obstime = btjd_to_time(target_btjd)
    ra1, dec1 = propagate_ra_dec(ra0, dec0, pmra, pmdec, obstime)
    shift = angular_shift_arcsec(ra0, dec0, ra1, dec1)
    finite = np.isfinite(shift)
    if not finite.any():
        return {"n": len(sub), "n_finite_pm": 0}
    pm_tot = np.sqrt(pmra**2 + pmdec**2)
    idx = int(np.nanargmax(shift))
    return {
        "n": int(len(sub)),
        "n_finite_pm": int(finite.sum()),
        "shift_arcsec_max": float(shift[finite].max()),
        "shift_arcsec_median": float(np.median(shift[finite])),
        "pm_mas_yr_max": float(pm_tot[finite].max()),
        "max_source_id": int(sub.iloc[idx]["source_id"]) if "source_id" in sub.columns else None,
        "max_tess_mag": float(sub.iloc[idx]["tess_mag"]) if "tess_mag" in sub.columns else None,
    }
