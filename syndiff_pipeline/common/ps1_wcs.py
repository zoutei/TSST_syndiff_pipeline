"""PS1 skycell WCS pixel convention.

PS1 (IPP) writes the stack/warp header ``CRPIX1``/``CRPIX2`` in its own pixel
convention, where the first pixel centre is at 0.5. FITS readers (astropy,
``calculate_radec``) assume 1.0, so a PS1 header read as FITS puts every star
0.5 PS1 px too low in x and y. Reading it with ``CRPIX + 0.5`` puts the
header on the light (proof: PS1 DR2 stack catalogue ``xPos`` vs its own sky
position, 57 skycells; dev_runs/ps1_wcs_convention_20261020/README.md).

Every PS1 header or skycell-table read that converts between PS1 pixels and
sky goes through this module, so the offset is applied in exactly one place:

* :func:`ps1_wcs_from_header` for raw PS1 FITS headers (``headers_data``,
  ``headers.json``, zarr ``attrs["header"]``);
* :func:`apply_ps1_crpix_convention` for the bundled ``skycell_wcs.csv`` table
  (raw PS1 CRPIX). The master skycell CSV written by the mapping inherits the
  corrected values and carries :data:`PS1_CRPIX_COLUMN`, so it is read as
  stored and never corrected twice.

``SYNDIFF_PS1_CRPIX_CONVENTION=fits`` restores the old (uncorrected) reading,
for reproducing outputs made before the fix and for A/B tests. The effective
offset is recorded in mapping outputs (:data:`PS1_CRPIX_COLUMN`, FITS key
:data:`PS1_CRPIX_FITS_KEY`) and in stage fingerprints.
"""
from __future__ import annotations

import os
from typing import Any, Union

import numpy as np

ENV_VAR = "SYNDIFF_PS1_CRPIX_CONVENTION"
CONVENTIONS = {"ipp": 0.5, "fits": 0.0}
DEFAULT_CONVENTION = "ipp"

# Marker written next to MAPGRID/GEOMFP: CSV column (value = offset applied, float)
# and FITS keyword in the master/regmap headers. Absent = legacy (0.0).
PS1_CRPIX_COLUMN = "PS1CRPIX"
PS1_CRPIX_FITS_KEY = "PS1CRPIX"


def ps1_crpix_convention() -> str:
    """Active convention name: ``"ipp"`` (default, +0.5) or ``"fits"`` (legacy, +0)."""
    name = os.environ.get(ENV_VAR, DEFAULT_CONVENTION).strip().lower()
    if name not in CONVENTIONS:
        raise ValueError(f"{ENV_VAR} must be one of {sorted(CONVENTIONS)}, got {name!r}")
    return name


def ps1_crpix_offset() -> float:
    """Offset added to raw PS1 ``CRPIX1``/``CRPIX2`` under the active convention."""
    return CONVENTIONS[ps1_crpix_convention()]


def ps1_wcs_from_header(header: Union[str, Any]):
    """Celestial astropy ``WCS`` of a raw PS1 skycell header, with the CRPIX offset applied.

    ``header`` is a FITS header string (``headers_data`` / ``headers.json``) or an
    ``astropy.io.fits.Header``. Use this instead of ``WCS(header)`` for any PS1 header.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    if isinstance(header, (str, bytes)):
        header = fits.Header.fromstring(header if isinstance(header, str) else header.decode())
    w = WCS(header)
    if w.naxis > 2:
        w = w.celestial
    off = ps1_crpix_offset()
    if off:
        w.wcs.crpix = np.asarray(w.wcs.crpix, dtype=float) + off
    return w


# PS1 pole cells (projection 2643) have CRVAL2 = 89.999999999997 in the real image headers, so wcslib uses
# LONPOLE = 180 for them.  The bundled skycell_wcs.csv stores CRVAL2 = 90.0 exactly, for which wcslib switches to
# LONPOLE = 0 (FITS Paper II), i.e. a 180 deg RA flip about the pole.  Every WCS built from a table row must
# therefore carry LONPOLE = 180 when CRVAL2 >= 90.  (dev_runs/s26_ownwcs_20261027/mapping_pole_diag/lonpole_check)
POLE_LONPOLE_DEG = 180.0


def table_row_needs_pole_lonpole(crval2) -> bool:
    """True for a skycell-table row whose CRVAL2 sits on (or numerically above) the pole."""
    return float(crval2) >= 90.0


def set_pole_lonpole_header(header):
    """Add ``LONPOLE=180`` to a skycell-table-derived ``fits.Header`` with ``CRVAL2 >= 90`` (in place; returns it)."""
    if table_row_needs_pole_lonpole(header["CRVAL2"]):
        header["LONPOLE"] = (POLE_LONPOLE_DEG, "PS1 pole cell, as real header")
    return header


def set_pole_lonpole_wcs(wcs):
    """Set ``wcs.wcs.lonpole = 180`` on a table-derived ``astropy`` WCS with ``CRVAL2 >= 90`` (in place; returns it)."""
    if table_row_needs_pole_lonpole(wcs.wcs.crval[1]):
        wcs.wcs.lonpole = POLE_LONPOLE_DEG
    return wcs


def apply_ps1_crpix_convention(df):
    """Return a copy of a raw PS1 skycell table with ``CRPIX1/2`` corrected and the marker column set.

    For the bundled ``skycell_wcs.csv`` (raw PS1 values) only. A table that already
    carries :data:`PS1_CRPIX_COLUMN` is returned unchanged (never corrected twice).
    """
    if PS1_CRPIX_COLUMN in df.columns:
        return df
    out = df.copy()
    off = ps1_crpix_offset()
    out["CRPIX1"] = out["CRPIX1"].astype(float) + off
    out["CRPIX2"] = out["CRPIX2"].astype(float) + off
    out[PS1_CRPIX_COLUMN] = off
    return out


def table_ps1_crpix_offset(df) -> float:
    """Offset already contained in a skycell table's CRPIX (0.0 for legacy tables without the marker)."""
    if PS1_CRPIX_COLUMN not in df.columns or len(df) == 0:
        return 0.0
    vals = np.unique(np.asarray(df[PS1_CRPIX_COLUMN], dtype=float))
    if len(vals) != 1:
        raise ValueError(f"mixed {PS1_CRPIX_COLUMN} values in one skycell table: {vals.tolist()}")
    return float(vals[0])


# Band preference for the single PS1 header used per skycell (WCS + MJD-OBS epoch).  The four band stacks share one
# WCS but have different input exposures, so MJD-OBS differs by up to ~1 yr between bands; ``i`` carries the largest
# template band weight in every recipe so far.  Dict order of ``headers_data`` is the thread-arrival order of the
# bands, so ``next(iter(...))`` picked a different header (epoch) from run to run.
REFERENCE_BAND_ORDER = ("i", "r", "z", "y")


def ps1_reference_header(headers):
    """The one deterministic PS1 header of a skycell: band ``i`` if present, else r, z, y, else the first sorted key.

    ``headers`` maps band -> header string (``headers_data`` / ``headers.json``)."""
    if not headers:
        raise ValueError("no PS1 band headers")
    for b in REFERENCE_BAND_ORDER:
        if b in headers:
            return headers[b]
    return headers[sorted(headers)[0]]
