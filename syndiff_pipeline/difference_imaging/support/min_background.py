"""Min-background FFI selection from manifest Earth/Moon angles."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _earth_moon_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Earth moon columns.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    tuple[str, str]"""
    for earth_col, moon_col in (
        ("Earth_Camera_Angle", "Moon_Camera_Angle"),
        ("earth_deg", "moon_deg"),
    ):
        if earth_col in df.columns and moon_col in df.columns:
            return earth_col, moon_col
    raise KeyError(
        "syndiff_ffi_frames.csv must include Earth/Moon angle columns "
        "(Earth_Camera_Angle/Moon_Camera_Angle or earth_deg/moon_deg)."
    )


def angle_score_series(df: pd.DataFrame, weighting_factor: float) -> pd.Series:
    """Angle score series.
    
    Parameters
    ----------
    df : pd.DataFrame
    weighting_factor : float
    
    Returns
    -------
    pd.Series"""
    earth_col, moon_col = _earth_moon_columns(df)
    earth = pd.to_numeric(df[earth_col], errors="coerce")
    moon = pd.to_numeric(df[moon_col], errors="coerce")
    wf = float(weighting_factor)
    return (earth + moon * wf) / (1.0 + wf)


def _manifest_needs_angle_enrichment(df: pd.DataFrame) -> bool:
    try:
        earth_col, moon_col = _earth_moon_columns(df)
    except KeyError:
        return True
    earth = pd.to_numeric(df[earth_col], errors="coerce")
    moon = pd.to_numeric(df[moon_col], errors="coerce")
    return not (earth.notna().any() and moon.notna().any())


def _btjd_series_from_date_obs(series: pd.Series) -> pd.Series:
    from astropy.time import Time

    out = pd.Series(np.nan, index=series.index, dtype=float)
    for idx, val in series.items():
        if pd.isna(val) or not str(val).strip():
            continue
        try:
            t = Time(str(val), format="isot", scale="utc")
            try:
                out.at[idx] = float(t.btjd)
            except AttributeError:
                out.at[idx] = float(t.jd) - 2457000.0
        except Exception:
            continue
    return out


def _merge_ffi_list_timing(
    manifest: pd.DataFrame,
    *,
    data_root: str,
    sector: int,
    camera: int,
    ccd: int,
) -> pd.DataFrame:
    from syndiff_pipeline.common.download import manifest_basename_from_local
    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
    from syndiff_pipeline.common.wcs_header_cache import load_ffi_list

    out = manifest.copy()
    ffi_list_df = load_ffi_list(scc_ffi_list_parquet(data_root, sector, camera, ccd))
    if ffi_list_df.empty or "path" not in out.columns:
        return out
    if "DATE-OBS" not in out.columns:
        out["DATE-OBS"] = pd.NA
    if "btjd" not in out.columns:
        out["btjd"] = np.nan
    for idx, row in out.iterrows():
        logical = manifest_basename_from_local(str(row["path"]))
        if logical not in ffi_list_df.index:
            continue
        date_obs = ffi_list_df.loc[logical].get("date_obs")
        if pd.notna(date_obs) and (
            pd.isna(out.at[idx, "DATE-OBS"]) or not str(out.at[idx, "DATE-OBS"]).strip()
        ):
            out.at[idx, "DATE-OBS"] = date_obs
    missing_btjd = pd.to_numeric(out["btjd"], errors="coerce").isna()
    if missing_btjd.any() and out["DATE-OBS"].notna().any():
        out.loc[missing_btjd, "btjd"] = _btjd_series_from_date_obs(
            out.loc[missing_btjd, "DATE-OBS"]
        )
    return out


def ensure_manifest_earth_moon_angles(
    manifest: pd.DataFrame,
    *,
    sector: int,
    camera: int,
    data_root: Optional[str] = None,
    ccd: Optional[int] = None,
    tessvectors_data_path: Optional[str] = None,
) -> pd.DataFrame:
    """Attach Earth/Moon camera angles when missing from a frames manifest."""
    if not _manifest_needs_angle_enrichment(manifest):
        return manifest

    out = manifest.copy()
    if data_root is not None and ccd is not None:
        out = _merge_ffi_list_timing(
            out,
            data_root=data_root,
            sector=sector,
            camera=camera,
            ccd=ccd,
        )
    elif "btjd" not in out.columns or not pd.to_numeric(out["btjd"], errors="coerce").notna().any():
        if "DATE-OBS" in out.columns and out["DATE-OBS"].notna().any():
            out["btjd"] = _btjd_series_from_date_obs(out["DATE-OBS"])
        else:
            log.warning(
                "Cannot enrich manifest with Earth/Moon angles: missing btjd/DATE-OBS "
                "and no data_root/ccd for ffi_list merge."
            )
            return manifest

    from syndiff_pipeline.common.wcs_grouping import attach_tessvector_earth_moon_angles

    out = attach_tessvector_earth_moon_angles(
        out,
        sector=sector,
        camera=camera,
        tessvectors_data_path=tessvectors_data_path,
    )
    if _manifest_needs_angle_enrichment(out):
        log.warning(
            "Manifest still lacks usable Earth/Moon angles after TESSVectors attach."
        )
    return out


def _usable_manifest_rows(df: pd.DataFrame) -> pd.Series:
    """Usable manifest rows.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.Series"""
    mask = pd.Series(True, index=df.index)
    if "wcs_ok" in df.columns:
        mask &= df["wcs_ok"].astype(str).str.lower().isin({"true", "1", "yes", "t"})
    earth_col, moon_col = _earth_moon_columns(df)
    mask &= pd.to_numeric(df[earth_col], errors="coerce").notna()
    mask &= pd.to_numeric(df[moon_col], errors="coerce").notna()
    if "path" in df.columns:
        mask &= df["path"].astype(str).str.strip().ne("")
    return mask


def pick_best_angle_ffi(
    manifest: pd.DataFrame,
    *,
    weighting_factor: float,
    sector: Optional[int] = None,
    camera: Optional[int] = None,
    data_root: Optional[str] = None,
    ccd: Optional[int] = None,
    tessvectors_data_path: Optional[str] = None,
) -> tuple[str, float]:
    """Return ``(absolute ffi path, score)`` for the highest angle-ranked row."""
    table = manifest
    if sector is not None and camera is not None and _manifest_needs_angle_enrichment(manifest):
        table = ensure_manifest_earth_moon_angles(
            manifest,
            sector=sector,
            camera=camera,
            data_root=data_root,
            ccd=ccd,
            tessvectors_data_path=tessvectors_data_path,
        )
    try:
        usable = _usable_manifest_rows(table)
    except KeyError:
        usable = pd.Series(False, index=table.index)
    if not usable.any():
        from syndiff_pipeline.common.wcs_grouping import (
            choose_reference_ffi_path,
            try_resolve_existing_fits_path,
        )

        log.warning(
            "No manifest rows with Earth/Moon angles; falling back to WCS drift "
            "reference pick for kernel-fit min-background FFI."
        )
        ffi_path = choose_reference_ffi_path(table)
        resolved = try_resolve_existing_fits_path(ffi_path)
        return str(resolved if resolved is not None else ffi_path), float("nan")
    scores = angle_score_series(table, weighting_factor)
    sub_scores = scores[usable]
    idx = int(sub_scores.idxmax())
    path_col = "path" if "path" in table.columns else "filename"
    ffi_path = os.path.abspath(
        os.path.expanduser(str(table.loc[idx, path_col]))
    )
    from syndiff_pipeline.common.wcs_grouping import try_resolve_existing_fits_path

    resolved = try_resolve_existing_fits_path(ffi_path)
    return str(resolved if resolved is not None else ffi_path), float(scores.loc[idx])
