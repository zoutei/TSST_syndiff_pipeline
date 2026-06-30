"""Min-background FFI selection from manifest Earth/Moon angles."""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


def _earth_moon_columns(df: pd.DataFrame) -> tuple[str, str]:
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
    earth_col, moon_col = _earth_moon_columns(df)
    earth = pd.to_numeric(df[earth_col], errors="coerce")
    moon = pd.to_numeric(df[moon_col], errors="coerce")
    wf = float(weighting_factor)
    return (earth + moon * wf) / (1.0 + wf)


def _usable_manifest_rows(df: pd.DataFrame) -> pd.Series:
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
    manifest: pd.DataFrame, *, weighting_factor: float
) -> tuple[str, float]:
    """Return ``(absolute ffi path, score)`` for the highest angle-ranked row."""
    usable = _usable_manifest_rows(manifest)
    if not usable.any():
        from syndiff_pipeline.common.wcs_grouping import (
            choose_reference_ffi_path,
            try_resolve_existing_fits_path,
        )

        log.warning(
            "No manifest rows with Earth/Moon angles; falling back to WCS drift "
            "reference pick for kernel-fit min-background FFI."
        )
        ffi_path = choose_reference_ffi_path(manifest)
        resolved = try_resolve_existing_fits_path(ffi_path)
        return str(resolved if resolved is not None else ffi_path), float("nan")
    scores = angle_score_series(manifest, weighting_factor)
    sub_scores = scores[usable]
    idx = int(sub_scores.idxmax())
    path_col = "path" if "path" in manifest.columns else "filename"
    ffi_path = os.path.abspath(
        os.path.expanduser(str(manifest.loc[idx, path_col]))
    )
    from syndiff_pipeline.common.wcs_grouping import try_resolve_existing_fits_path

    resolved = try_resolve_existing_fits_path(ffi_path)
    return str(resolved if resolved is not None else ffi_path), float(scores.loc[idx])
