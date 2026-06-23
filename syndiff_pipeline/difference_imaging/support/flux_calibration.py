"""Hotpants kernel calibration metadata and light-curve photometric calibration.

Calibration workflow (light-curve CSV columns):

1. ``kernel_ref`` — 3σ-clipped mean of per-epoch ``kernel_sum`` (Hotpants flux scale).
2. ``zp_ref`` — ``25 + 2.5 * log10(kernel_ref)`` (template ZP anchor).
3. ``flux`` — ``flux_uncal * kernel_ref / kernel_sum`` (kernel-equalized).
4. ``tess_zp`` — per-epoch ``25 + 2.5 * log10(kernel_sum)`` from Hotpants.
5. ``tmag`` — ``-2.5 * log10(flux_uncal) + tess_zp`` (uncalibrated rate, e⁻/s).
6. ``flux_jy`` — ``2416 * 10^(-0.4 * tmag)`` (TESS handbook).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    PHOT_CALIB_CSV_BASENAME,
    meta_workspace_dir_from_diffs_dir,
)

log = logging.getLogger(__name__)

TEMPLATE_ZP = 25.0
TESS_JY_ZP = 2416.0
_LN10 = math.log(10.0)
_MAG_ERR_SCALE = 2.5 / _LN10

PHOT_CALIB_COLUMNS = (
    "product_id",
    "stem",
    "kernel_sum",
    "tess_zp",
    "success",
)


def tess_zp_from_kernel_sum(kernel_sum: float) -> float:
    """
    Hotpants-derived TESS AB zero point (template ZP = 25).

    ``tess_zp = TEMPLATE_ZP + 2.5 * log10(kernel_sum)``
    """
    validate_kernel_sum(kernel_sum)
    return float(TEMPLATE_ZP + 2.5 * np.log10(float(kernel_sum)))


def kernel_sum_at_center(
    kernel_solution: np.ndarray,
    hp_config,
    image_shape: tuple[int, int],
) -> float:
    """Hotpants kernel sum at image centre (photometric flux scale factor)."""
    from syndiff_pipeline.difference_imaging.stages.kernel import kernel_image_at_coords

    kimg = kernel_image_at_coords(
        kernel_solution, hp_config, image_shape, at_coords=None
    )
    return float(np.nansum(kimg))


def validate_kernel_sum(kernel_sum: float) -> None:
    if not np.isfinite(kernel_sum) or float(kernel_sum) <= 0.0:
        raise ValueError(f"invalid kernel_sum={kernel_sum!r}")


def sigma_clipped_mean(values: np.ndarray, *, sigma: float = 3.0) -> float:
    """Mean of finite values after astropy sigma clipping."""
    from astropy.stats import sigma_clipped_stats

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if v.size == 1:
        return float(v[0])
    mean, _, _ = sigma_clipped_stats(v, sigma=sigma)
    return float(mean)


def kernel_ref_from_kernel_sums(
    kernel_sums: np.ndarray, *, sigma: float = 3.0
) -> float:
    """3σ-clipped mean of positive finite kernel sums."""
    v = np.asarray(kernel_sums, dtype=float)
    valid = np.isfinite(v) & (v > 0.0)
    if not valid.any():
        return float("nan")
    return sigma_clipped_mean(v[valid], sigma=sigma)


def tess_mag_from_cts_per_s(flux: float, tess_zp: float) -> float:
    """TESS AB magnitude from count rate (e⁻/s) and per-epoch ``tess_zp``."""
    flux = float(flux)
    tess_zp = float(tess_zp)
    if not np.isfinite(flux) or flux <= 0.0 or not np.isfinite(tess_zp):
        return float("nan")
    return float(-2.5 * np.log10(flux) + tess_zp)


def tess_mag_err(flux: float, eflux: float) -> float:
    """1σ magnitude uncertainty from flux uncertainty."""
    flux = float(flux)
    eflux = float(eflux)
    if not np.isfinite(flux) or flux <= 0.0 or not np.isfinite(eflux):
        return float("nan")
    return float(_MAG_ERR_SCALE * eflux / flux)


def flux_jy_from_tess_mag(tmag: float) -> float:
    """Flux density in Jy from TESS magnitude (handbook zeropoint 2416 Jy)."""
    tmag = float(tmag)
    if not np.isfinite(tmag):
        return float("nan")
    return float(TESS_JY_ZP * 10.0 ** (-0.4 * tmag))


def flux_jy_err(flux_jy: float, etmag: float) -> float:
    """1σ Jy uncertainty from magnitude uncertainty."""
    flux_jy = float(flux_jy)
    etmag = float(etmag)
    if not np.isfinite(flux_jy) or not np.isfinite(etmag):
        return float("nan")
    return float(flux_jy * (_LN10 / 2.5) * etmag)


def stamp_diff_calib_metadata(
    header: fits.Header,
    kernel_sum: float,
    tess_zp: float,
) -> fits.Header:
    """Add non-destructive Hotpants calibration keywords to a FITS header."""
    hdr = fits.Header(header)
    hdr["FLUXSCAL"] = (
        float(kernel_sum),
        "Hotpants kernel sum at centre (flux scale factor)",
    )
    hdr["KERNZPT"] = (
        float(tess_zp),
        "Kernel-derived AB zero point",
    )
    return hdr


def phot_calib_csv_path(meta_dir: str) -> str:
    return os.path.join(meta_dir, PHOT_CALIB_CSV_BASENAME)


def build_phot_calib_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for res in results:
        product_id = res.get("ffi_product_id") or res.get("product_id")
        if not product_id:
            continue
        kernel_sum = res.get("kernel_sum")
        tess_zp = res.get("tess_zp")
        rows.append(
            {
                "product_id": str(product_id),
                "stem": str(res.get("stem") or ""),
                "kernel_sum": float(kernel_sum) if kernel_sum is not None else np.nan,
                "tess_zp": float(tess_zp) if tess_zp is not None else np.nan,
                "success": bool(res.get("success")),
            }
        )
    return rows


def write_phot_calib_table(meta_dir: str, results: list[dict[str, Any]]) -> str:
    """Write ``phot_calib.csv`` under the meta workspace."""
    os.makedirs(meta_dir, exist_ok=True)
    path = phot_calib_csv_path(meta_dir)
    rows = build_phot_calib_rows(results)
    pd.DataFrame(rows, columns=list(PHOT_CALIB_COLUMNS)).to_csv(path, index=False)
    log.info("Wrote photometric calibration table: %s (%d rows)", path, len(rows))
    return path


def load_phot_calib_table(meta_dir: str) -> Optional[pd.DataFrame]:
    path = phot_calib_csv_path(meta_dir)
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def apply_kernel_calibration(
    lc_df: pd.DataFrame,
    phot_calib: Optional[pd.DataFrame] = None,
    *,
    flux_col: str = "flux",
    eflux_col: str = "eflux",
) -> pd.DataFrame:
    """
    Join Hotpants calibration and add kernel-normalized flux, TESS mag, and Jy.

    Uncalibrated measurements are stored as ``flux_uncal`` / ``eflux_uncal``.
    Kernel-equalized values are written to ``flux`` / ``eflux``.
    ``tmag`` and ``flux_jy`` use ``flux_uncal`` with per-epoch ``tess_zp``.

    When ``kernel_sum`` is already present on ``lc_df``, it is used directly;
    otherwise rows are joined from ``phot_calib``.
    """
    out = lc_df.copy()
    out = out.drop(columns=["sci_zp", "flux_cal", "eflux_cal"], errors="ignore")

    if "flux_uncal" in out.columns and out["flux_uncal"].notna().any():
        uncal_flux = out["flux_uncal"].to_numpy(dtype=float)
    elif flux_col in out.columns:
        out["flux_uncal"] = out[flux_col]
        uncal_flux = out["flux_uncal"].to_numpy(dtype=float)
    else:
        log.warning(
            "apply_kernel_calibration: no %r or flux_uncal column; skipping",
            flux_col,
        )
        return out

    if "eflux_uncal" in out.columns and out["eflux_uncal"].notna().any():
        uncal_eflux = out["eflux_uncal"].to_numpy(dtype=float)
    elif eflux_col in out.columns:
        out["eflux_uncal"] = out[eflux_col]
        uncal_eflux = out["eflux_uncal"].to_numpy(dtype=float)
    else:
        uncal_eflux = np.full(len(out), np.nan, dtype=float)
        out["eflux_uncal"] = uncal_eflux

    if "kernel_sum" not in out.columns or not out["kernel_sum"].notna().any():
        if phot_calib is None or phot_calib.empty:
            log.warning("apply_kernel_calibration: no kernel_sum and no phot_calib")
            return out
        if "filename" not in out.columns:
            log.warning("apply_kernel_calibration: lc_df has no filename column")
            return out
        calib = phot_calib.drop_duplicates("product_id", keep="first").set_index(
            "product_id"
        )

        def _lookup(product_id: str, column: str) -> float:
            if not product_id or product_id not in calib.index:
                return float("nan")
            val = calib.at[product_id, column]
            return float(val) if pd.notna(val) else float("nan")

        product_ids = out["filename"].astype(str).map(
            lambda f: tess_product_id_from_ffi_path(f) or ""
        )
        out["kernel_sum"] = product_ids.map(lambda p: _lookup(p, "kernel_sum"))
        if "tess_zp" not in out.columns or not out["tess_zp"].notna().any():
            out["tess_zp"] = product_ids.map(lambda p: _lookup(p, "tess_zp"))

    ks = out["kernel_sum"].to_numpy(dtype=float)
    valid = np.isfinite(ks) & (ks > 0.0) & np.isfinite(uncal_flux)
    if not valid.any():
        log.warning(
            "apply_kernel_calibration: no epochs with finite kernel_sum and flux_uncal"
        )
        return out

    if "tess_zp" not in out.columns or not out["tess_zp"].notna().any():
        tz = np.full(len(out), np.nan, dtype=float)
        valid_tz = np.isfinite(ks) & (ks > 0.0)
        tz[valid_tz] = np.array(
            [tess_zp_from_kernel_sum(float(v)) for v in ks[valid_tz]], dtype=float
        )
        out["tess_zp"] = tz
    else:
        missing_tz = out["tess_zp"].isna() & np.isfinite(ks) & (ks > 0.0)
        if missing_tz.any():
            out.loc[missing_tz, "tess_zp"] = out.loc[missing_tz, "kernel_sum"].map(
                lambda v: tess_zp_from_kernel_sum(float(v))
            )

    kernel_ref = kernel_ref_from_kernel_sums(ks[valid])
    out["kernel_ref"] = kernel_ref
    out["zp_ref"] = (
        tess_zp_from_kernel_sum(kernel_ref) if np.isfinite(kernel_ref) else np.nan
    )

    factor = np.full(len(out), np.nan, dtype=float)
    valid_factor = np.isfinite(kernel_ref) & np.isfinite(ks) & (ks > 0.0)
    factor[valid_factor] = kernel_ref / ks[valid_factor]
    out["flux"] = uncal_flux * factor
    out["eflux"] = uncal_eflux * factor

    tess_zp_arr = out["tess_zp"].to_numpy(dtype=float)
    valid_mag = np.isfinite(uncal_flux) & (uncal_flux > 0.0) & np.isfinite(tess_zp_arr)
    tmag = np.full(len(out), np.nan, dtype=float)
    tmag[valid_mag] = -2.5 * np.log10(uncal_flux[valid_mag]) + tess_zp_arr[valid_mag]

    valid_err = valid_mag & np.isfinite(uncal_eflux)
    etmag = np.full(len(out), np.nan, dtype=float)
    etmag[valid_err] = _MAG_ERR_SCALE * uncal_eflux[valid_err] / uncal_flux[valid_err]

    flux_jy = np.full(len(out), np.nan, dtype=float)
    valid_jy = np.isfinite(tmag)
    flux_jy[valid_jy] = TESS_JY_ZP * 10.0 ** (-0.4 * tmag[valid_jy])

    eflux_jy = np.full(len(out), np.nan, dtype=float)
    valid_jye = valid_jy & np.isfinite(etmag)
    eflux_jy[valid_jye] = flux_jy[valid_jye] * (_LN10 / 2.5) * etmag[valid_jye]

    out["tmag"] = tmag
    out["etmag"] = etmag
    out["flux_jy"] = flux_jy
    out["eflux_jy"] = eflux_jy

    log.info(
        "Applied kernel calibration: kernel_ref=%.6f zp_ref=%.4f (%d/%d epochs)",
        kernel_ref,
        float(out["zp_ref"].iloc[0]) if np.isfinite(kernel_ref) else float("nan"),
        int(valid.sum()),
        len(out),
    )
    return out


def apply_zp_calibration_if_available(
    lc_df: pd.DataFrame,
    diffs_dir: Optional[str],
    *,
    flux_col: str = "flux",
    eflux_col: str = "eflux",
) -> pd.DataFrame:
    """Load paired ``phot_calib.csv`` and apply kernel calibration when present."""
    if not diffs_dir:
        return lc_df
    meta_dir = meta_workspace_dir_from_diffs_dir(diffs_dir)
    phot_calib = load_phot_calib_table(meta_dir)
    if phot_calib is None:
        return lc_df
    return apply_kernel_calibration(
        lc_df, phot_calib, flux_col=flux_col, eflux_col=eflux_col
    )
