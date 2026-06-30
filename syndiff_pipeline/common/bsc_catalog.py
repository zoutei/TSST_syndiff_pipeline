"""Bright Star Catalogue (BSC5) loader and crop projection."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
    bright_star_catalog_path,
)

log = logging.getLogger(__name__)

# 0-based (start, end) slices; end exclusive (VizieR ybsc5.readme byte positions).
_BSC_COLSPECS = [
    (0, 4),  # HR (bytes 1-4)
    (75, 77),  # RAh (bytes 76-77)
    (77, 79),  # RAm (bytes 78-79)
    (79, 83),  # RAs (bytes 80-83)
    (83, 84),  # DE- sign (byte 84)
    (84, 86),  # DEd (bytes 85-86)
    (86, 88),  # DEm (bytes 87-88)
    (88, 90),  # DEs (bytes 89-90)
    (102, 107),  # Vmag (bytes 103-107)
]
_BSC_NAMES = ["HR", "RAh", "RAm", "RAs", "DE_sign", "DEd", "DEm", "DEs", "Vmag"]


def _parse_bsc_fwf_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert fixed-width BSC table to ``hr``, ``ra``, ``dec``, ``vmag``."""
    df = raw.copy()
    ra_deg = (
        df["RAh"].fillna(0) + df["RAm"].fillna(0) / 60.0 + df["RAs"].fillna(0) / 3600.0
    ) * 15.0
    df["ra"] = ra_deg
    df.loc[df[["RAh", "RAm", "RAs"]].isna().all(axis=1), "ra"] = np.nan

    dec_abs = df["DEd"].fillna(0) + df["DEm"].fillna(0) / 60.0 + df["DEs"].fillna(0) / 3600.0
    df["dec"] = np.where(df["DE_sign"] == "-", -dec_abs, dec_abs)
    df.loc[df[["DEd", "DEm", "DEs"]].isna().all(axis=1), "dec"] = np.nan

    out = pd.DataFrame(
        {
            "hr": pd.to_numeric(df["HR"], errors="coerce"),
            "ra": df["ra"].astype(float),
            "dec": df["dec"].astype(float),
            "vmag": pd.to_numeric(df["Vmag"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["hr", "ra", "dec", "vmag"]).reset_index(drop=True)
    out["hr"] = out["hr"].astype(int)
    return out


@functools.lru_cache(maxsize=2)
def load_bright_star_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load the decompressed VizieR BSC5 ``catalog`` fixed-width file.

    Returns columns: ``hr``, ``ra``, ``dec``, ``vmag`` (degrees, float).
    """
    catalog_path = Path(path) if path is not None else bright_star_catalog_path()
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Missing Bright Star Catalogue: {catalog_path}. "
            "Ensure syndiff_pipeline/resources/bsc5/catalog is present."
        )

    raw = pd.read_fwf(
        catalog_path,
        colspecs=_BSC_COLSPECS,
        names=_BSC_NAMES,
        header=None,
    )
    return _parse_bsc_fwf_table(raw)


def filter_catalog_to_ffi_footprint(
    df: pd.DataFrame,
    ref_ffi_path: str,
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
    margin_deg: float = 1.0,
) -> pd.DataFrame:
    """
    Keep catalog rows whose sky position is near the FFI field of view.

    BSC is full-sky; projecting every row through TESS SIP WCS and keeping only
    in-bounds pixels can assign spurious on-chip coordinates to stars tens of
    degrees off the field (common at high declination).  Pre-filter by angular
    distance from the image center to the farthest corner, plus ``margin_deg``.
    """
    import warnings

    if df.empty:
        return df.copy()

    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap

    with open_fits_memmap(ref_ffi_path) as hdul:
        hdr = hdul[1].header
        nx = int(hdr["NAXIS1"])
        ny = int(hdr["NAXIS2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(hdr)

    center = SkyCoord(wcs.wcs.crval[0] * u.deg, wcs.wcs.crval[1] * u.deg)
    corner_ra, corner_dec = wcs.pixel_to_world_values(
        [0, nx - 1, nx - 1, 0],
        [0, 0, ny - 1, ny - 1],
    )
    corner_coords = SkyCoord(corner_ra * u.deg, corner_dec * u.deg)
    max_sep_deg = (
        max(float(center.separation(c).deg) for c in corner_coords) + float(margin_deg)
    )

    coords = SkyCoord(df[ra_col].values * u.deg, df[dec_col].values * u.deg)
    keep = center.separation(coords).deg <= max_sep_deg
    n_drop = int((~keep).sum())
    if n_drop:
        log.info(
            "BSC footprint filter: kept %d / %d rows within %.2f deg of field center",
            int(keep.sum()),
            len(df),
            max_sep_deg,
        )
    return df.loc[keep].copy().reset_index(drop=True)


def project_bsc_to_crop(
    bsc_df: pd.DataFrame,
    ref_ffi_path: str,
    crop_bounds: dict,
) -> pd.DataFrame:
    """Project BSC ``ra``/``dec`` to crop-local ``x``/``y``; keep in-bounds rows."""
    from syndiff_pipeline.common import wcs_grouping

    in_footprint = filter_catalog_to_ffi_footprint(bsc_df, ref_ffi_path)
    return wcs_grouping.ensure_gaia_crop_xy(
        in_footprint,
        ref_ffi_path,
        crop_bounds,
        ra_col="ra",
        dec_col="dec",
    )
