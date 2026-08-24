"""Sci2Idl polynomial WCS fitting on top of a reference linear WCS."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.wcs.sip_poly_fit import (
    TESS_FFI_NAXIS,
    iterative_clip_du_dv,
    n_sci2idl_terms,
    poly_eval,
)
from syndiff_pipeline.difference_imaging.wcs.wcs_conversion import radec_to_uv

log = logging.getLogger(__name__)

GAIA_CATALOG_BASENAME = "gaia_catalog_pipeline.csv"


@dataclass
class StarSelectionConfig:
    flags_ok: int = 0
    qfit_min: float = 0.0
    qfit_max: float = 0.2
    pos_err_min: float = 0.0
    pos_err_max: float = 0.05
    cfit_abs_max: float = 0.05
    min_stars: int = 50
    clip_n_sigma: float = 3.0
    clip_max_iter: int = 3


@dataclass
class FitConfig:
    sip_degree: int = 5
    sip_fallback: tuple[int | None, ...] = (2, None)


@dataclass
class Sci2IdlFitResult:
    linear_wcs: WCS
    coeff_x: list[float]
    coeff_y: list[float]
    poly_degree: int
    rotation_fit_x: bool
    rotation_fit_y: bool
    keep_mask: np.ndarray


def crop_bounds_from_header(header: fits.Header) -> dict[str, Any]:
    x_min = int(header["XMIN"])
    y_min = int(header["YMIN"])
    x_max = int(header["XMAX"])
    y_max = int(header["YMAX"])
    ny = y_max - y_min
    nx = x_max - x_min
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "shape": (ny, nx),
    }


def join_stars(
    phot: Table,
    gaia: pd.DataFrame,
    *,
    max_sep_px: float = 0.25,
) -> pd.DataFrame:
    df = phot.to_pandas()
    gcols = [c for c in ("source_id", "ra", "dec", "x", "y") if c in gaia.columns]
    g = gaia[gcols].copy()
    # `phot` (the centroids stage's own output) already carries source_id/ra/
    # dec from ITS OWN Gaia match; drop those before merging so the join
    # doesn't collide into pandas' "_x"/"_y" suffixes (which would leave no
    # bare "ra"/"dec"/"source_id" column for select_good_stars to read) when
    # the exact x_init/y_init == x/y merge actually finds a match.
    df_bare = df.drop(columns=[c for c in gcols if c in df.columns])
    exact = df_bare.merge(g, left_on=["x_init", "y_init"], right_on=["x", "y"], how="inner")
    if len(exact) > 0:
        return exact

    if not {"x", "y", "ra", "dec"}.issubset(g.columns):
        return exact

    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)]))
    xy = np.column_stack([df["x_init"].to_numpy(dtype=float), df["y_init"].to_numpy(dtype=float)])
    dist, idx = tree.query(xy, k=1)
    keep = np.isfinite(dist) & (dist <= max_sep_px)
    if not np.any(keep):
        log.warning(
            "join_stars: exact merge empty and no Gaia neighbors within %.3f px",
            max_sep_px,
        )
        return exact

    matched = df.loc[keep].copy().reset_index(drop=True)
    g_hit = g.iloc[idx[keep]].reset_index(drop=True)
    for col in gcols:
        matched[col] = g_hit[col].to_numpy()
    matched["gaia_match_sep_px"] = dist[keep]
    return matched


def select_good_stars(df: pd.DataFrame, cfg: StarSelectionConfig) -> pd.DataFrame:
    mask = (
        (df["flags"] == cfg.flags_ok)
        & np.isfinite(df["qfit"])
        & (df["qfit"] >= cfg.qfit_min)
        & (df["qfit"] <= cfg.qfit_max)
        & np.isfinite(df["x_err"])
        & np.isfinite(df["y_err"])
        & (df["x_err"] > cfg.pos_err_min)
        & (df["y_err"] > cfg.pos_err_min)
        & (df["x_err"] < cfg.pos_err_max)
        & (df["y_err"] < cfg.pos_err_max)
        & np.isfinite(df["cfit"])
        & (np.abs(df["cfit"]) < cfg.cfit_abs_max)
        & np.isfinite(df["ra"])
        & np.isfinite(df["dec"])
        & np.isfinite(df["x_fit"])
        & np.isfinite(df["y_fit"])
    )
    return df.loc[mask].copy()


def uv_from_linear_wcs(
    wcs: WCS,
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-frame pixel offsets (u, v) via wcs_conversion."""
    return radec_to_uv(ra, dec, wcs.to_header(relax=True))


def build_frame_stars(
    stars_qc: pd.DataFrame,
    linear_wcs: WCS,
    *,
    stem: str,
    btjd: float,
) -> pd.DataFrame:
    if stars_qc.empty:
        return stars_qc
    x_fit = stars_qc["x_fit"].to_numpy(dtype=float)
    y_fit = stars_qc["y_fit"].to_numpy(dtype=float)
    ra = stars_qc["ra"].to_numpy(dtype=float)
    dec = stars_qc["dec"].to_numpy(dtype=float)
    crpix1 = float(linear_wcs.wcs.crpix[0])
    crpix2 = float(linear_wcs.wcs.crpix[1])
    stars = stars_qc.copy()
    stars["xprime"] = x_fit - (crpix1 - 1.0)
    stars["yprime"] = y_fit - (crpix2 - 1.0)
    u, v = uv_from_linear_wcs(linear_wcs, ra, dec)
    stars["u"] = u
    stars["v"] = v
    stars["stem"] = stem
    stars["btjd"] = btjd
    return stars


def fit_sci2idl_distortion(
    stars: pd.DataFrame,
    linear_wcs: WCS,
    cfg: FitConfig,
    *,
    fit_coeffs0: bool = True,
    rotation_fit_x: bool = True,
    rotation_fit_y: bool = True,
    n_sigma: float = 3.0,
    max_iter: int = 20,
) -> Sci2IdlFitResult:
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_fit = stars["x_fit"].to_numpy(dtype=float)
    y_fit = stars["y_fit"].to_numpy(dtype=float)
    u, v = uv_from_linear_wcs(linear_wcs, ra, dec)
    crpix1 = float(linear_wcs.wcs.crpix[0])
    crpix2 = float(linear_wcs.wcs.crpix[1])
    xprime = x_fit - (crpix1 - 1.0)
    yprime = y_fit - (crpix2 - 1.0)

    # Always use fixed tesswcs CCD scale (not crop NAXIS) so monomials stay O(1).
    coord_scale = float(TESS_FFI_NAXIS)

    coeff_x, coeff_y, keep_mask, _, _ = iterative_clip_du_dv(
        xprime,
        yprime,
        u,
        v,
        cfg.sip_degree,
        n_sigma=n_sigma,
        max_iter=max_iter,
        fit_coeffs0=fit_coeffs0,
        rotation_fit_x=rotation_fit_x,
        rotation_fit_y=rotation_fit_y,
        coord_scale=coord_scale,
    )
    return Sci2IdlFitResult(
        linear_wcs=linear_wcs,
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        poly_degree=cfg.sip_degree,
        rotation_fit_x=rotation_fit_x,
        rotation_fit_y=rotation_fit_y,
        keep_mask=keep_mask,
    )


def sci2idl_du_dv_px(
    result: Sci2IdlFitResult,
    stars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    u, v, xprime, yprime = sci2idl_prime_coords(stars, result.linear_wcs)
    u_fit = poly_eval(result.coeff_x, xprime, yprime, result.poly_degree)
    v_fit = poly_eval(result.coeff_y, xprime, yprime, result.poly_degree)
    return u - u_fit, v - v_fit


def sci2idl_prime_coords(
    stars: pd.DataFrame,
    linear_wcs: WCS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_fit = stars["x_fit"].to_numpy(dtype=float)
    y_fit = stars["y_fit"].to_numpy(dtype=float)
    u, v = uv_from_linear_wcs(linear_wcs, ra, dec)
    crpix1 = float(linear_wcs.wcs.crpix[0])
    crpix2 = float(linear_wcs.wcs.crpix[1])
    xprime = x_fit - (crpix1 - 1.0)
    yprime = y_fit - (crpix2 - 1.0)
    return u, v, xprime, yprime


def warmstart_frame(
    stars: pd.DataFrame,
    linear_wcs: WCS,
    *,
    sip_degree: int,
    star_cfg: StarSelectionConfig | None = None,
) -> Sci2IdlFitResult:
    cfg = star_cfg or StarSelectionConfig()
    return fit_sci2idl_distortion(
        stars,
        linear_wcs,
        FitConfig(sip_degree=sip_degree, sip_fallback=()),
        fit_coeffs0=True,
        rotation_fit_x=True,
        rotation_fit_y=True,
        n_sigma=cfg.clip_n_sigma,
        max_iter=cfg.clip_max_iter,
    )


def warmstart_table_row(
    stem: str,
    btjd: float,
    result: Sci2IdlFitResult,
    *,
    fit_ok: bool = True,
    n_stars_qc: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stem": stem,
        "btjd": btjd,
        "n_keep": int(result.keep_mask.sum()),
        "fit_ok": bool(fit_ok),
        "n_stars_qc": int(n_stars_qc if n_stars_qc is not None else len(result.keep_mask)),
        "message": message,
    }
    for i, val in enumerate(result.coeff_x):
        row[f"c{i}_x"] = val
    for i, val in enumerate(result.coeff_y):
        row[f"c{i}_y"] = val
    return row


def warmstart_coeff_vectors(row: pd.Series, sip_degree: int) -> tuple[list[float], list[float]]:
    n = n_sci2idl_terms(sip_degree)
    cx = [float(row[f"c{i}_x"]) for i in range(n)]
    cy = [float(row[f"c{i}_y"]) for i in range(n)]
    return cx, cy
