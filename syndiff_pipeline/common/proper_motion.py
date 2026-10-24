"""Linear Gaia proper-motion propagation to a TESS frame epoch (opt-in).

Gaia DR3 positions are at the reference epoch J2016.0 (JD 2457389.0 TCB, i.e.
BTJD 389.0; TCB vs TDB differs by far less than one day of motion, ignored).
The SynDiff WCS path (ePSF star positions, centroid initial positions, the sky
positions the per-FFI / temporal WCS is fitted to) can use these positions
propagated to each FFI's epoch.

Model: linear motion on the local tangent plane,

    ra  += pmra  * dt / cos(dec) / 3.6e6   [deg]   (pmra already includes cos dec)
    dec += pmdec * dt / 3.6e6              [deg]

with ``dt`` in Julian years (365.25 d) and pm in mas/yr.  Parallax and radial
velocity are ignored (parallax shifts are < 1 mpx at TESS scale for nearly all
stars; the line-of-sight perspective term is negligible).  Stars with a
non-finite ``pmra`` or ``pmdec`` receive no shift.  The one-step linear update
differs from a great-circle space-motion solution at second order,
~ (pm*dt)^2 tan(dec) / 2 (~1e-4 arcsec for 1 arcsec/yr over 4 yr).
"""

from __future__ import annotations

import numpy as np

GAIA_DR3_EPOCH_BTJD = 2457389.0 - 2457000.0  # J2016.0
JULIAN_YEAR_DAYS = 365.25
_MAS_PER_DEG = 3.6e6


def frame_epoch_btjd(header) -> float:
    """Mid-exposure BTJD of a TESS FFI header.

    Uses the mean of ``TSTART``/``TSTOP`` (BTJD); falls back to the midpoint of
    ``DATE-OBS``/``DATE-END`` (UTC) and finally to ``DATE-OBS``.  NaN if none.
    """
    try:
        t0, t1 = float(header.get("TSTART")), float(header.get("TSTOP"))
        if np.isfinite(t0) and np.isfinite(t1):
            return 0.5 * (t0 + t1)
    except (TypeError, ValueError):
        pass
    from astropy.time import Time

    d0, d1 = header.get("DATE-OBS"), header.get("DATE-END")
    if d0 and d1:
        return float(0.5 * (Time(d0, format="isot", scale="utc").jd
                            + Time(d1, format="isot", scale="utc").jd) - 2457000.0)
    if d0:
        return float(Time(d0, format="isot", scale="utc").jd - 2457000.0)
    return float("nan")


def propagate_radec(ra, dec, pmra, pmdec, epoch_btjd: float):
    """Return ``(ra, dec, n_no_pm)`` propagated from J2016.0 to *epoch_btjd*.

    Inputs in degrees and mas/yr (arrays or Series); outputs are float64
    ndarrays.  *n_no_pm* counts stars left unshifted for NaN proper motion.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    pmra = np.asarray(pmra, dtype=np.float64)
    pmdec = np.asarray(pmdec, dtype=np.float64)
    if not np.isfinite(epoch_btjd):
        raise ValueError("propagate_radec: non-finite frame epoch")
    dt = (float(epoch_btjd) - GAIA_DR3_EPOCH_BTJD) / JULIAN_YEAR_DAYS
    ok = np.isfinite(pmra) & np.isfinite(pmdec)
    dra = np.where(ok, pmra, 0.0) * dt / np.cos(np.deg2rad(dec)) / _MAS_PER_DEG
    ddec = np.where(ok, pmdec, 0.0) * dt / _MAS_PER_DEG
    return ra + dra, dec + ddec, int((~ok).sum())


def propagate_gaia_df(df, epoch_btjd: float, *, ra_col="ra", dec_col="dec",
                      pmra_col="pmra", pmdec_col="pmdec"):
    """Copy of *df* with ``ra``/``dec`` propagated; raw kept in ``ra_j2016``/``dec_j2016``."""
    for c in (pmra_col, pmdec_col):
        if c not in df.columns:
            raise ValueError(f"apply_proper_motion requires Gaia column {c!r}")
    out = df.copy()
    ra, dec, n_no_pm = propagate_radec(
        df[ra_col].to_numpy(), df[dec_col].to_numpy(),
        df[pmra_col].to_numpy(), df[pmdec_col].to_numpy(), epoch_btjd,
    )
    out["ra_j2016"] = df[ra_col].to_numpy()
    out["dec_j2016"] = df[dec_col].to_numpy()
    out[ra_col] = ra
    out[dec_col] = dec
    if n_no_pm:
        import logging
        logging.getLogger(__name__).info(
            "proper motion: %d / %d stars have NaN pm (no shift applied)", n_no_pm, len(df))
    return out


def pm_kwargs(params) -> dict:
    """``{"apply_proper_motion": True}`` when *params* opts in, else ``{}`` (call unchanged when off)."""
    return {"apply_proper_motion": True} if getattr(params, "apply_proper_motion", None) else {}
