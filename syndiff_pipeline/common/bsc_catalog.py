"""Bright Star Catalogue (BSC5) loader and crop projection."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
    bright_star_catalog_path,
)

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


def project_bsc_to_crop(
    bsc_df: pd.DataFrame,
    ref_ffi_path: str,
    crop_bounds: dict,
) -> pd.DataFrame:
    """Project BSC ``ra``/``dec`` to crop-local ``x``/``y``; keep in-bounds rows."""
    from syndiff_pipeline.common import wcs_grouping

    return wcs_grouping.ensure_gaia_crop_xy(
        bsc_df,
        ref_ffi_path,
        crop_bounds,
        ra_col="ra",
        dec_col="dec",
    )
