# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""
Fit crop-local WCS from centroids + Gaia, compare to TESS crop WCS on a pixel grid.

Intended for dev / validation before any pipeline-stage promotion.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import Sip, WCS
from astropy.wcs.utils import (
    _linear_wcs_fit,
    _sip_fit,
    celestial_frame_to_wcs,
    fit_wcs_from_points,
)
import astropy.units as u
from scipy.optimize import least_squares

from syndiff_pipeline.common.fits_variants import try_resolve_fits_variant
from syndiff_pipeline.common.wcs_grouping import crop_ffi_header, resolve_existing_fits_path
from syndiff_pipeline.difference_imaging.stages.centroids import (
    load_centroids_index,
    photresults_ecsv_path,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import tess_product_id_from_ffi_path

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.cheb_poly_fit import (
    cheb_model_eval,
    chebyshev_to_monomial_coeffs,
    compare_design_conditions,
    crop_half_extents,
    iterative_clip_cheb_du_dv,
    residual_chebyshev_to_sci2idl_monomials,
)
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.sip_poly_fit import (
    TESS_FFI_NAXIS,
    build_sip_header_from_sci2idl,
    coeff_table_rows,
    iterative_clip_du_dv,
    poly_eval,
    sci2idl_to_sip_updates,
)
from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.wcs_conversion import radec_to_uvprime  # noqa: F401  (used by other dev scripts)

log = logging.getLogger(__name__)

GAIA_CATALOG_BASENAME = "gaia_catalog_pipeline.csv"


@dataclass
class StarSelectionConfig:
    flags_ok: int = 0
    qfit_min: float = 0.0
    qfit_max: float = 0.2
    pos_err_min: float = 0.0  # reject exactly-zero formal errors
    pos_err_max: float = 0.05
    cfit_abs_max: float = 0.05
    min_stars: int = 50
    clip_n_sigma: float = 3.0
    clip_max_iter: int = 3


@dataclass
class FitConfig:
    sip_degree: int = 4
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


@dataclass
class ChebyshevFitResult:
    linear_wcs: WCS
    coeff_x: list[float]
    coeff_y: list[float]
    poly_degree: int
    sx: float
    sy: float
    identity_linear: bool
    keep_mask: np.ndarray
    mono_coeff_x: list[float]
    mono_coeff_y: list[float]


@dataclass
class FrameResult:
    stem: str
    segment: str
    btjd: float
    n_stars_in: int
    n_stars_fit: int
    fit_ok: bool
    sip_degree_used: int | None
    tess_vs_stars_median_px: float
    tess_vs_stars_p95_px: float
    fit_vs_stars_median_px: float
    fit_vs_stars_p95_px: float
    grid_tess_vs_fit_median_px: float
    grid_tess_vs_fit_p95_px: float
    grid_tess_vs_fit_max_px: float
    message: str = ""


def _btjd_from_header(header: fits.Header) -> float:
    date = header.get("DATE-OBS")
    if not date:
        return float("nan")
    return float(Time(date, format="isot", scale="utc").jd - 2457000.0)


def crop_bounds_from_header(header: fits.Header) -> dict[str, Any]:
    """Read crop bounds stamped on hp_d / diff products."""
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


def assign_orbit_chains(btjd: Sequence[float], gap_min: float = 45.0) -> np.ndarray:
    btjd_arr = np.asarray(btjd, dtype=float)
    chains = np.zeros(len(btjd_arr), dtype=int)
    chain = 0
    for i in range(1, len(btjd_arr)):
        gap = (btjd_arr[i] - btjd_arr[i - 1]) * 24.0 * 60.0
        if np.isfinite(gap) and gap > gap_min:
            chain += 1
        chains[i] = chain
    return chains


def select_orbit_segments(
    stems: Sequence[str],
    btjd: Sequence[float],
    *,
    orbit_index: int = 1,
    n_each: int = 10,
    gap_min: float = 45.0,
) -> dict[str, list[str]]:
    """Return start/mid/end stem lists from one orbit chain (0-based index)."""
    order = np.argsort(btjd)
    stems_sorted = [stems[i] for i in order]
    btjd_sorted = [btjd[i] for i in order]
    chains = assign_orbit_chains(btjd_sorted, gap_min=gap_min)
    mask = chains == orbit_index
    chain_stems = [s for s, m in zip(stems_sorted, mask) if m]
    if len(chain_stems) < n_each:
        raise RuntimeError(
            f"Orbit chain {orbit_index} has {len(chain_stems)} frames; need {n_each}."
        )
    mid = len(chain_stems) // 2
    return {
        "start": chain_stems[:n_each],
        "mid": chain_stems[mid - n_each // 2 : mid - n_each // 2 + n_each],
        "end": chain_stems[-n_each:],
    }


def load_gaia_catalog(workspace: Path) -> pd.DataFrame:
    for rel in (GAIA_CATALOG_BASENAME, f"../{GAIA_CATALOG_BASENAME}"):
        path = workspace / rel
        if path.is_file():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Gaia catalog not found under {workspace}")


def resolve_hp_d_path(workspace: Path, stem: str) -> Path | None:
    """Resolve ``hp_d/{stem}_hp_d.fits[.fz|.gz]`` under *workspace*."""
    return try_resolve_fits_variant(workspace / "hp_d" / f"{stem}_hp_d.fits")


def sip_coeff_table(
    wcs: WCS,
    *,
    camera: int | None = None,
    ccd: int | None = None,
) -> pd.DataFrame:
    """
    Tabulate fitted SIP coeffs in Calc_TESS_distortion / siaf-style rows.

    Columns: siaf_index, exponent_x/y, A/B (forward), AP/BP (inverse).
    Missing terms are 0. Linear WCS keys are in :func:`linear_wcs_param_table`.
    """
    sip = wcs.sip
    cols = [
        "CAMERA", "CCD", "siaf_index", "exponent_x", "exponent_y",
        "A", "B", "AP", "BP",
    ]
    if sip is None:
        return pd.DataFrame(columns=cols)

    a_order = int(sip.a_order)
    b_order = int(sip.b_order)
    ap_order = int(getattr(sip, "ap_order", 0) or 0)
    bp_order = int(getattr(sip, "bp_order", 0) or 0)
    poly_degree = max(a_order, b_order, ap_order, bp_order)

    def _get(mat: np.ndarray | None, i: int, j: int) -> float:
        if mat is None:
            return 0.0
        if i < mat.shape[0] and j < mat.shape[1]:
            return float(mat[i, j])
        return 0.0

    rows: list[dict[str, Any]] = []
    for tot in range(poly_degree + 1):
        exp_x = tot
        for j in range(tot + 1):
            i = exp_x
            rows.append(
                {
                    "CAMERA": camera,
                    "CCD": ccd,
                    "siaf_index": tot * 10 + j,
                    "exponent_x": i,
                    "exponent_y": j,
                    "A": _get(sip.a, i, j),
                    "B": _get(sip.b, i, j),
                    "AP": _get(getattr(sip, "ap", None), i, j),
                    "BP": _get(getattr(sip, "bp", None), i, j),
                }
            )
            exp_x -= 1
    return pd.DataFrame(rows)


def linear_wcs_param_table(wcs: WCS) -> pd.DataFrame:
    """One-row table of CRPIX / CRVAL / CD (or PC·CDELT) for a fitted WCS."""
    crpix = np.asarray(wcs.wcs.crpix, dtype=float)
    crval = np.asarray(wcs.wcs.crval, dtype=float)
    if wcs.wcs.has_cd():
        cd = np.asarray(wcs.wcs.cd, dtype=float)
    else:
        pc = np.asarray(wcs.wcs.get_pc(), dtype=float)
        cdelt = np.asarray(wcs.wcs.cdelt, dtype=float)
        cd = pc * cdelt
    return pd.DataFrame(
        [
            {
                "CRPIX1": crpix[0],
                "CRPIX2": crpix[1],
                "CRVAL1": crval[0],
                "CRVAL2": crval[1],
                "CD1_1": cd[0, 0],
                "CD1_2": cd[0, 1],
                "CD2_1": cd[1, 0],
                "CD2_2": cd[1, 1],
                "CTYPE1": str(wcs.wcs.ctype[0]),
                "CTYPE2": str(wcs.wcs.ctype[1]),
                "A_ORDER": int(wcs.sip.a_order) if wcs.sip is not None else None,
                "B_ORDER": int(wcs.sip.b_order) if wcs.sip is not None else None,
            }
        ]
    )


def join_stars(
    phot: Table,
    gaia: pd.DataFrame,
    *,
    max_sep_px: float = 0.25,
) -> pd.DataFrame:
    """
    Attach Gaia ``ra``/``dec`` (and ids) to photometry rows.

    Prefer an exact ``(x_init, y_init) == (x, y)`` merge (centroids were seeded
    from that catalog). If the on-disk Gaia CSV was regenerated, fall back to
    nearest-neighbor matching within *max_sep_px*.
    """
    df = phot.to_pandas()
    gcols = [c for c in ("source_id", "ra", "dec", "x", "y") if c in gaia.columns]
    g = gaia[gcols].copy()
    exact = df.merge(g, left_on=["x_init", "y_init"], right_on=["x", "y"], how="inner")
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
            "join_stars: exact merge empty and no Gaia neighbors within %.3f px "
            "(median NN=%.4f)",
            max_sep_px,
            float(np.median(dist)) if len(dist) else float("nan"),
        )
        return exact

    matched = df.loc[keep].copy().reset_index(drop=True)
    g_hit = g.iloc[idx[keep]].reset_index(drop=True)
    for col in gcols:
        matched[col] = g_hit[col].to_numpy()
    matched["gaia_match_sep_px"] = dist[keep]
    log.info(
        "join_stars: exact merge empty; NN-matched %d/%d stars (med sep=%.4f px)",
        int(keep.sum()),
        len(df),
        float(np.median(dist[keep])),
    )
    return matched


def select_good_stars(df: pd.DataFrame, cfg: StarSelectionConfig) -> pd.DataFrame:
    """Keep photutils-clean stars using fit-quality columns (not TESS WCS)."""
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


def star_selection_cut_counts(
    df: pd.DataFrame,
    cfg: StarSelectionConfig,
) -> pd.DataFrame:
    """Per-cut drop counts for notebook reporting (sequential after flags==ok)."""
    base = (df["flags"] == cfg.flags_ok) & np.isfinite(df["ra"]) & np.isfinite(df["dec"])
    base &= np.isfinite(df["x_fit"]) & np.isfinite(df["y_fit"])
    n0 = int(base.sum())
    rows: list[dict[str, object]] = [{"cut": "flags==0 & finite coords", "n_pass": n0, "n_drop": int((~base).sum())}]

    m = base & np.isfinite(df["qfit"]) & (df["qfit"] >= cfg.qfit_min) & (df["qfit"] <= cfg.qfit_max)
    rows.append({"cut": f"{cfg.qfit_min} <= qfit <= {cfg.qfit_max}", "n_pass": int(m.sum()), "n_drop": int((base & ~m).sum())})
    prev = m

    m = prev & np.isfinite(df["x_err"]) & np.isfinite(df["y_err"])
    m &= (df["x_err"] > cfg.pos_err_min) & (df["y_err"] > cfg.pos_err_min)
    m &= (df["x_err"] < cfg.pos_err_max) & (df["y_err"] < cfg.pos_err_max)
    rows.append(
        {
            "cut": f"{cfg.pos_err_min} < x_err,y_err < {cfg.pos_err_max}",
            "n_pass": int(m.sum()),
            "n_drop": int((prev & ~m).sum()),
        }
    )
    prev = m

    m = prev & np.isfinite(df["cfit"]) & (np.abs(df["cfit"]) < cfg.cfit_abs_max)
    rows.append(
        {
            "cut": f"|cfit| < {cfg.cfit_abs_max}",
            "n_pass": int(m.sum()),
            "n_drop": int((prev & ~m).sum()),
        }
    )
    return pd.DataFrame(rows)


def _sip_coef_names(degree: int, *, min_order: int) -> list[str]:
    """SIP coefficient keys ``i_j`` with ``min_order <= i+j < degree+1``."""
    return [
        f"{i}_{j}"
        for i in range(degree + 1)
        for j in range(degree + 1)
        if min_order <= (i + j) < (degree + 1)
    ]


def _sip_only_fit(params, lon, lat, u, v, w_obj, order, coef_names):
    """SIP residual objective with CRPIX/CD held fixed in *w_obj*."""
    from astropy.modeling.models import SIP

    a_params = params[: len(coef_names)]
    b_params = params[len(coef_names) :]
    crpix = np.asarray(w_obj.wcs.crpix, dtype=float)
    cdx = np.asarray(w_obj.wcs.cd, dtype=float)

    a_coeff, b_coeff = {}, {}
    for i, name in enumerate(coef_names):
        a_coeff["A_" + name] = a_params[i]
        b_coeff["B_" + name] = b_params[i]

    sip = SIP(crpix=crpix, a_order=order, b_order=order, a_coeff=a_coeff, b_coeff=b_coeff)
    fuv, guv = sip(u, v)

    xo, yo = np.dot(cdx, np.array([u + fuv - crpix[0], v + guv - crpix[1]]))

    x, y = w_obj.all_world2pix(lon, lat, 0)
    x, y = np.dot(w_obj.wcs.cd, (x - w_obj.wcs.crpix[0], y - w_obj.wcs.crpix[1]))

    return np.concatenate((x - xo, y - yo))


def fit_wcs_from_points_sip(
    xy: tuple[np.ndarray, np.ndarray],
    world_coords: SkyCoord,
    *,
    projection: str = "TAN",
    sip_degree: int | None = None,
    sip_min_order: int = 2,
    refit_linear_with_sip: bool = True,
) -> WCS:
    """
    Like :func:`astropy.wcs.utils.fit_wcs_from_points`, but ``sip_min_order``
    controls the lowest SIP total degree included (0 = constant+linear free).

    When ``sip_min_order < 2``, set ``refit_linear_with_sip=False`` so the
    linear WCS from the first stage is held fixed (avoids CD/SIP degeneracy).
    """
    xp, yp = xy
    try:
        lon, lat = world_coords.data.lon.deg, world_coords.data.lat.deg
    except AttributeError:
        unit_sph = world_coords.unit_spherical
        lon, lat = unit_sph.lon.deg, unit_sph.lat.deg

    wcs = celestial_frame_to_wcs(frame=world_coords.frame, projection=projection)
    if wcs.wcs.has_pc():
        wcs.wcs.cd = wcs.wcs.pc
        wcs.wcs.__delattr__("pc")

    xpmin, xpmax, ypmin, ypmax = xp.min(), xp.max(), yp.min(), yp.max()
    wcs.pixel_shape = (
        1 if xpmax <= 0.0 else int(np.ceil(xpmax)),
        1 if ypmax <= 0.0 else int(np.ceil(ypmax)),
    )

    sc1 = SkyCoord(lon.min() * u.deg, lat.max() * u.deg)
    sc2 = SkyCoord(lon.max() * u.deg, lat.min() * u.deg)
    pa = sc1.position_angle(sc2)
    sep = sc1.separation(sc2)
    midpoint_sc = sc1.directional_offset_by(pa, sep / 2)
    wcs.wcs.crval = (midpoint_sc.data.lon.deg, midpoint_sc.data.lat.deg)
    wcs.wcs.crpix = ((xpmax + xpmin) / 2.0, (ypmax + ypmin) / 2.0)

    if xpmin == xpmax:
        xpmin, xpmax = xpmin - 0.5, xpmax + 0.5
    if ypmin == ypmax:
        ypmin, ypmax = ypmin - 0.5, ypmax + 0.5

    p0 = np.concatenate([wcs.wcs.cd.flatten(), wcs.wcs.crpix.flatten()])
    fit = least_squares(
        _linear_wcs_fit,
        p0,
        args=(lon, lat, xp, yp, wcs),
        bounds=[
            [-np.inf, -np.inf, -np.inf, -np.inf, xpmin + 1, ypmin + 1],
            [np.inf, np.inf, np.inf, np.inf, xpmax + 1, ypmax + 1],
        ],
    )
    wcs.wcs.crpix = np.array(fit.x[4:6])
    wcs.wcs.cd = np.array(fit.x[0:4].reshape((2, 2)))

    if not sip_degree:
        return wcs

    degree = sip_degree
    if "-SIP" not in wcs.wcs.ctype[0]:
        wcs.wcs.ctype = [x + "-SIP" for x in wcs.wcs.ctype]

    coef_names = _sip_coef_names(degree, min_order=sip_min_order)
    if refit_linear_with_sip:
        p0 = np.concatenate(
            (
                np.array(wcs.wcs.crpix),
                wcs.wcs.cd.flatten(),
                np.zeros(2 * len(coef_names)),
            )
        )
        fit = least_squares(
            _sip_fit,
            p0,
            args=(lon, lat, xp, yp, wcs, degree, coef_names),
            bounds=[
                [xpmin + 1, ypmin + 1] + [-np.inf] * (4 + 2 * len(coef_names)),
                [xpmax + 1, ypmax + 1] + [np.inf] * (4 + 2 * len(coef_names)),
            ],
        )
        coef_fit = (
            list(fit.x[6 : 6 + len(coef_names)]),
            list(fit.x[6 + len(coef_names) :]),
        )
        wcs.wcs.cd = fit.x[2:6].reshape((2, 2))
        wcs.wcs.crpix = fit.x[0:2]
    else:
        p0 = np.zeros(2 * len(coef_names))
        fit = least_squares(
            _sip_only_fit,
            p0,
            args=(lon, lat, xp, yp, wcs, degree, coef_names),
        )
        coef_fit = (
            list(fit.x[: len(coef_names)]),
            list(fit.x[len(coef_names) :]),
        )

    a_vals = np.zeros((degree + 1, degree + 1))
    b_vals = np.zeros((degree + 1, degree + 1))
    for coef_name in coef_names:
        a_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[0].pop(0)
        b_vals[int(coef_name[0])][int(coef_name[2])] = coef_fit[1].pop(0)

    wcs.sip = Sip(
        a_vals,
        b_vals,
        np.zeros((degree + 1, degree + 1)),
        np.zeros((degree + 1, degree + 1)),
        wcs.wcs.crpix,
    )
    return wcs


def iterative_clip_stars(
    df: pd.DataFrame,
    wcs: WCS,
    *,
    n_sigma: float,
    max_iter: int,
) -> pd.DataFrame:
    active = df.copy()
    for _ in range(max_iter):
        if len(active) < 10:
            break
        ra = active["ra"].to_numpy(dtype=float)
        dec = active["dec"].to_numpy(dtype=float)
        x_fit = active["x_fit"].to_numpy(dtype=float)
        y_fit = active["y_fit"].to_numpy(dtype=float)
        px, py = wcs.world_to_pixel_values(ra, dec)
        resid = np.hypot(px - x_fit, py - y_fit)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)))
        scale = max(1.4826 * mad, 1e-6)
        keep = resid < med + n_sigma * scale
        if keep.all():
            break
        active = active.loc[keep].copy()
    return active


def fit_crop_wcs(
    stars: pd.DataFrame,
    shape: tuple[int, int],
    cfg: FitConfig,
) -> tuple[WCS | None, int | None, str]:
    """Fit crop WCS with Astropy ``fit_wcs_from_points`` + SIP."""
    xy = np.column_stack([stars["x_fit"].to_numpy(), stars["y_fit"].to_numpy()])
    world = SkyCoord(
        stars["ra"].to_numpy(dtype=float) * u.deg,
        stars["dec"].to_numpy(dtype=float) * u.deg,
    )
    last = ""
    for sip_deg in (cfg.sip_degree, *cfg.sip_fallback):
        try:
            wcs = fit_wcs_from_points(
                xy.T,
                world,
                projection="TAN",
                sip_degree=sip_deg,
            )
            wcs.array_shape = shape
            return wcs, sip_deg, ""
        except Exception as exc:
            last = str(exc)
    return None, None, last


def _uvprime_from_linear_wcs(
    wcs: WCS,
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-frame pixel offsets (u, v) from a linear TAN WCS."""
    x, y = wcs.world_to_pixel_values(ra, dec)
    crpix1 = float(wcs.wcs.crpix[0])
    crpix2 = float(wcs.wcs.crpix[1])
    return x - (crpix1 - 1.0), y - (crpix2 - 1.0)


def fit_sci2idl_distortion(
    stars: pd.DataFrame,
    linear_wcs: WCS,
    cfg: FitConfig,
    *,
    fit_coeffs0: bool = False,
    rotation_fit_x: bool = False,
    rotation_fit_y: bool = False,
    n_sigma: float = 3.0,
    max_iter: int = 20,
) -> Sci2IdlFitResult:
    """
    Fit Calc_TESS_distortion-style Sci2Idl polynomials on top of a linear WCS.

    With ``rotation_fit_x=False`` / ``rotation_fit_y=False`` (Calc default), the
    first three coefficients per axis are anchored to ``[0, 1, 0]`` / ``[0, 0, 1]``.
    Set both ``rotation_fit_*`` to True to leave constant + linear terms free.
    """
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_fit = stars["x_fit"].to_numpy(dtype=float)
    y_fit = stars["y_fit"].to_numpy(dtype=float)

    u, v = _uvprime_from_linear_wcs(linear_wcs, ra, dec)
    crpix1 = float(linear_wcs.wcs.crpix[0])
    crpix2 = float(linear_wcs.wcs.crpix[1])
    xprime = x_fit - (crpix1 - 1.0)
    yprime = y_fit - (crpix2 - 1.0)

    shape = getattr(linear_wcs, "pixel_shape", None)
    if shape is not None and len(shape) >= 2:
      coord_scale = float(max(int(shape[0]), int(shape[1]), 1))
    else:
      arr = getattr(linear_wcs, "array_shape", None)
      if arr is not None and len(arr) >= 2:
        coord_scale = float(max(int(arr[-1]), int(arr[0]), 1))
      else:
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


def sci2idl_wcs(result: Sci2IdlFitResult, shape: tuple[int, int]) -> WCS:
    """Build an Astropy WCS with linear header + SIP keys from Sci2Idl coeffs."""
    hdr = build_sip_header_from_sci2idl(
        result.linear_wcs.to_header(relax=True),
        result.coeff_x,
        result.coeff_y,
        poly_degree=result.poly_degree,
        fold_linear=result.rotation_fit_x and result.rotation_fit_y,
    )
    for i in (1, 2):
        ctype = str(hdr.get(f"CTYPE{i}", ""))
        if ctype and "-SIP" not in ctype and ctype.endswith("TAN"):
            hdr[f"CTYPE{i}"] = ctype + "-SIP"
    wcs = WCS(hdr)
    wcs.array_shape = shape
    return wcs


def fit_chebyshev_distortion(
    stars: pd.DataFrame,
    linear_wcs: WCS,
    cfg: FitConfig,
    *,
    identity_linear: bool = True,
    n_sigma: float = 3.0,
    max_iter: int = 20,
    sx: float | None = None,
    sy: float | None = None,
) -> ChebyshevFitResult:
    """
    Fit 2D product Chebyshev distortion on top of a linear WCS.

    Default ``identity_linear=True`` fits residual Chebyshev series on
    ``(u-x', v-y')`` (all degrees 0..N) and evaluates as identity + residual.
    Domain half-extents default to crop NAXIS/2.
    """
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_fit = stars["x_fit"].to_numpy(dtype=float)
    y_fit = stars["y_fit"].to_numpy(dtype=float)

    u, v = _uvprime_from_linear_wcs(linear_wcs, ra, dec)
    crpix1 = float(linear_wcs.wcs.crpix[0])
    crpix2 = float(linear_wcs.wcs.crpix[1])
    xprime = x_fit - (crpix1 - 1.0)
    yprime = y_fit - (crpix2 - 1.0)

    shape = getattr(linear_wcs, "pixel_shape", None)
    if shape is not None and len(shape) >= 2:
        naxis1 = int(shape[0])
        naxis2 = int(shape[1])
    else:
        arr = getattr(linear_wcs, "array_shape", None)
        if arr is not None and len(arr) >= 2:
            naxis2 = int(arr[0])
            naxis1 = int(arr[1])
        else:
            naxis1 = int(TESS_FFI_NAXIS)
            naxis2 = int(TESS_FFI_NAXIS)

    sx_use, sy_use = crop_half_extents(naxis1=naxis1, naxis2=naxis2)
    if sx is not None:
        sx_use = float(sx)
    if sy is not None:
        sy_use = float(sy)

    coeff_x, coeff_y, keep_mask, _, _ = iterative_clip_cheb_du_dv(
        xprime,
        yprime,
        u,
        v,
        cfg.sip_degree,
        sx=sx_use,
        sy=sy_use,
        n_sigma=n_sigma,
        max_iter=max_iter,
        identity_linear=identity_linear,
    )
    if identity_linear:
        mono_x = residual_chebyshev_to_sci2idl_monomials(
            coeff_x, cfg.sip_degree, sx=sx_use, sy=sy_use, axis="x"
        )
        mono_y = residual_chebyshev_to_sci2idl_monomials(
            coeff_y, cfg.sip_degree, sx=sx_use, sy=sy_use, axis="y"
        )
    else:
        mono_x = chebyshev_to_monomial_coeffs(
            coeff_x, cfg.sip_degree, sx=sx_use, sy=sy_use
        )
        mono_y = chebyshev_to_monomial_coeffs(
            coeff_y, cfg.sip_degree, sx=sx_use, sy=sy_use
        )
    return ChebyshevFitResult(
        linear_wcs=linear_wcs,
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        poly_degree=cfg.sip_degree,
        sx=sx_use,
        sy=sy_use,
        identity_linear=identity_linear,
        keep_mask=keep_mask,
        mono_coeff_x=mono_x,
        mono_coeff_y=mono_y,
    )


def chebyshev_wcs(result: ChebyshevFitResult, shape: tuple[int, int]) -> WCS:
    """Build Astropy WCS with SIP keys from Chebyshev→monomial conversion.

    Folds constant+linear Sci2Idl terms into CD/CRVAL so they are not dropped
    when only SIP A/B (degree >= 2) are written.
    """
    hdr = build_sip_header_from_sci2idl(
        result.linear_wcs.to_header(relax=True),
        result.mono_coeff_x,
        result.mono_coeff_y,
        poly_degree=result.poly_degree,
        fold_linear=True,
    )
    for i in (1, 2):
        ctype = str(hdr.get(f"CTYPE{i}", ""))
        if ctype and "-SIP" not in ctype and ctype.endswith("TAN"):
            hdr[f"CTYPE{i}"] = ctype + "-SIP"
    wcs = WCS(hdr)
    wcs.array_shape = shape
    return wcs


def chebyshev_du_dv_px(
    result: ChebyshevFitResult,
    stars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Residuals ``u - u_fit``, ``v - v_fit`` for a Chebyshev distortion fit."""
    u, v, xprime, yprime = sci2idl_prime_coords(stars, result.linear_wcs)
    u_fit = cheb_model_eval(
        result.coeff_x,
        xprime,
        yprime,
        result.poly_degree,
        sx=result.sx,
        sy=result.sy,
        identity_linear=result.identity_linear,
        axis="x",
    )
    v_fit = cheb_model_eval(
        result.coeff_y,
        xprime,
        yprime,
        result.poly_degree,
        sx=result.sx,
        sy=result.sy,
        identity_linear=result.identity_linear,
        axis="y",
    )
    return u - u_fit, v - v_fit


def chebyshev_residual_sq_px(
    result: ChebyshevFitResult,
    stars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    du, dv = chebyshev_du_dv_px(result, stars)
    return du, dv, du * du + dv * dv


def chebyshev_coeff_table(
    result: ChebyshevFitResult,
    *,
    camera: int | None = None,
    ccd: int | None = None,
) -> pd.DataFrame:
    """Tabulate Chebyshev coeffs plus converted SIP monomials."""
    sip = sci2idl_to_sip_updates(
        result.mono_coeff_x, result.mono_coeff_y, result.poly_degree
    )
    rows: list[dict[str, Any]] = []
    for idx, (siaf_index, exp_x, exp_y) in enumerate(coeff_table_rows(result.poly_degree)):
        rows.append(
            {
                "CAMERA": camera,
                "CCD": ccd,
                "siaf_index": siaf_index,
                "exponent_x": exp_x,
                "exponent_y": exp_y,
                "cheb_x": float(result.coeff_x[idx]),
                "cheb_y": float(result.coeff_y[idx]),
                "mono_x": float(result.mono_coeff_x[idx]),
                "mono_y": float(result.mono_coeff_y[idx]),
                "SIP_A": sip.get(f"A_{exp_x}_{exp_y}", 0.0),
                "SIP_B": sip.get(f"B_{exp_x}_{exp_y}", 0.0),
            }
        )
    return pd.DataFrame(rows)


def distortion_design_condition_table(
    stars: pd.DataFrame,
    linear_wcs: WCS,
    degrees: Sequence[int],
    *,
    sx: float,
    sy: float,
    monomial_coord_scale: float | None = None,
    min_total_degree: int = 0,
) -> pd.DataFrame:
    """Compare Chebyshev vs scaled-monomial design condition numbers.

    Default ``min_total_degree=0`` matches the residual Chebyshev fit (full series
    on ``u-x'``). Pass ``min_total_degree=2`` to compare Sci2Idl's free monomial
    subspace only.
    """
    _, _, xprime, yprime = sci2idl_prime_coords(stars, linear_wcs)
    scale = float(
        TESS_FFI_NAXIS if monomial_coord_scale is None else monomial_coord_scale
    )
    rows: list[dict[str, Any]] = []
    for deg in degrees:
        stats = compare_design_conditions(
            xprime,
            yprime,
            int(deg),
            sx=sx,
            sy=sy,
            monomial_coord_scale=scale,
            min_total_degree=min_total_degree,
        )
        rows.append(
            {
                "degree": int(deg),
                "min_total_degree": int(min_total_degree),
                "n_terms": int(stats["n_terms"]),
                "n_stars": int(stats["n_stars"]),
                "cheb_cond": stats["cheb_cond"],
                "mono_cond": stats["mono_cond"],
                "cond_ratio_mono_over_cheb": (
                    stats["mono_cond"] / stats["cheb_cond"]
                    if np.isfinite(stats["cheb_cond"]) and stats["cheb_cond"] > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def sci2idl_prime_coords(
    stars: pd.DataFrame,
    linear_wcs: WCS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return u, v, xprime, yprime for Sci2Idl residual evaluation."""
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_fit = stars["x_fit"].to_numpy(dtype=float)
    y_fit = stars["y_fit"].to_numpy(dtype=float)
    u, v = _uvprime_from_linear_wcs(linear_wcs, ra, dec)
    crpix1 = float(linear_wcs.wcs.crpix[0])
    crpix2 = float(linear_wcs.wcs.crpix[1])
    xprime = x_fit - (crpix1 - 1.0)
    yprime = y_fit - (crpix2 - 1.0)
    return u, v, xprime, yprime


def sci2idl_du_dv_px(
    result: Sci2IdlFitResult,
    stars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Calc_TESS_distortion residual convention: ``u - u_fit``, ``v - v_fit``."""
    u, v, xprime, yprime = sci2idl_prime_coords(stars, result.linear_wcs)
    u_fit = poly_eval(result.coeff_x, xprime, yprime, result.poly_degree)
    v_fit = poly_eval(result.coeff_y, xprime, yprime, result.poly_degree)
    return u - u_fit, v - v_fit


def sci2idl_coeff_table(
    result: Sci2IdlFitResult,
    *,
    camera: int | None = None,
    ccd: int | None = None,
) -> pd.DataFrame:
    """Tabulate all Sci2Idl coefficients (including anchored degree-0/1 terms)."""
    sip = sci2idl_to_sip_updates(result.coeff_x, result.coeff_y, result.poly_degree)
    rows: list[dict[str, Any]] = []
    for idx, (siaf_index, exp_x, exp_y) in enumerate(coeff_table_rows(result.poly_degree)):
        rows.append(
            {
                "CAMERA": camera,
                "CCD": ccd,
                "siaf_index": siaf_index,
                "exponent_x": exp_x,
                "exponent_y": exp_y,
                "coeff_x": float(result.coeff_x[idx]),
                "coeff_y": float(result.coeff_y[idx]),
                "SIP_A": sip.get(f"A_{exp_x}_{exp_y}", 0.0),
                "SIP_B": sip.get(f"B_{exp_x}_{exp_y}", 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_du_dv_panels_arrays(
    out_path: Path,
    *,
    du_fit: np.ndarray,
    dv_fit: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    du_cut: np.ndarray | None = None,
    dv_cut: np.ndarray | None = None,
    x_cut: np.ndarray | None = None,
    y_cut: np.ndarray | None = None,
    title: str,
    residual_ylim: float = 0.4,
) -> None:
    """Notebook-style du/dv panels from precomputed residual arrays."""
    style_fit = dict(s=4, c="C2", alpha=0.55, linewidths=0, label="fit stars")
    style_cut = dict(s=4, c="C1", alpha=0.45, linewidths=0, label="clipped")

    all_res = [du_fit, dv_fit]
    if du_cut is not None and dv_cut is not None:
        all_res.extend([du_cut, dv_cut])
    p95 = float(np.percentile(np.abs(np.concatenate(all_res)), 95))
    ylim = max(residual_ylim, p95 * 1.05)
    residual_limits = (-ylim, ylim)

    med_du = float(np.median(np.abs(du_fit)))
    med_dv = float(np.median(np.abs(dv_fit)))
    full_title = f"{title}\nmed|du|={med_du:.3f}  med|dv|={med_dv:.3f} px"

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=False)
    panels = (
        (axes[0], y_fit, dv_fit, y_cut, dv_cut, "y [px]", "dv_pix", "dv vs y"),
        (axes[1], x_fit, du_fit, x_cut, du_cut, "x [px]", "du_pix", "du vs x"),
        (axes[2], x_fit, dv_fit, x_cut, dv_cut, "x [px]", "dv_pix", "dv vs x"),
        (axes[3], y_fit, du_fit, y_cut, du_cut, "y [px]", "du_pix", "du vs y"),
    )
    for i, (ax, xg, yg, xc, yc, xlabel, ylabel, panel_title) in enumerate(panels):
        if xc is not None and yc is not None:
            ax.scatter(xc, yc, **style_cut)
        ax.scatter(xg, yg, **style_fit)
        ax.axhline(0.0, color="k", lw=1.5)
        ax.set_ylim(residual_limits)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_title(f"{full_title}\n{panel_title}")
        else:
            ax.set_title(panel_title)
        if i == 0 and du_cut is not None:
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def star_residuals_px(wcs: WCS, stars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_obs = stars["x_fit"].to_numpy(dtype=float)
    y_obs = stars["y_fit"].to_numpy(dtype=float)
    x_pred, y_pred = wcs.world_to_pixel_values(ra, dec)
    dx = x_pred - x_obs
    dy = y_pred - y_obs
    r = np.hypot(dx, dy)
    return dx, dy, r


def star_du_dv_px(wcs: WCS, stars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Observed minus predicted pixel residuals (notebook / distortion convention)."""
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    x_obs = stars["x_fit"].to_numpy(dtype=float)
    y_obs = stars["y_fit"].to_numpy(dtype=float)
    x_pred, y_pred = wcs.world_to_pixel_values(ra, dec)
    du = x_obs - x_pred
    dv = y_obs - y_pred
    return du, dv


def star_residual_sq_px(wcs: WCS, stars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return du, dv, and du^2 + dv^2 in pixel units."""
    du, dv = star_du_dv_px(wcs, stars)
    return du, dv, du * du + dv * dv


def sci2idl_residual_sq_px(
    result: Sci2IdlFitResult,
    stars: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return du, dv, and du^2 + dv^2 for Sci2Idl (u-u_fit convention)."""
    du, dv = sci2idl_du_dv_px(result, stars)
    return du, dv, du * du + dv * dv


def plot_residual_sq_on_ffi(
    out_path: Path,
    *,
    image: np.ndarray,
    stars: pd.DataFrame,
    resid_sq: np.ndarray,
    cut_mask: np.ndarray,
    title: str,
    label: str = "du^2 + dv^2 [px^2]",
    vmax_percentile: float = 95.0,
    image_percentiles: tuple[float, float] = (5.0, 99.5),
) -> None:
    """
    Overlay star residuals on a crop FFI.

    *stars* x/y are FITS-style 1-based crop-local pixels.  *cut_mask* is True for
    sigma-clipped rejects (drawn with open circles on top).
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"expected 2-D image, got shape {img.shape}")

    x = stars["x_fit"].to_numpy(dtype=float) - 1.0
    y = stars["y_fit"].to_numpy(dtype=float) - 1.0
    cut_mask = np.asarray(cut_mask, dtype=bool)
    keep = ~cut_mask

    vmin_img, vmax_img = np.nanpercentile(img, image_percentiles)
    vmax_res = float(np.nanpercentile(resid_sq[keep] if np.any(keep) else resid_sq, vmax_percentile))
    vmax_res = max(vmax_res, 1e-12)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        img,
        origin="lower",
        cmap="gray",
        vmin=vmin_img,
        vmax=vmax_img,
        interpolation="nearest",
    )
    sc = ax.scatter(
        x[keep],
        y[keep],
        c=resid_sq[keep],
        s=10,
        cmap="inferno",
        vmin=0.0,
        vmax=vmax_res,
        linewidths=0,
        alpha=0.9,
    )
    if np.any(cut_mask):
        ax.scatter(
            x[cut_mask],
            y[cut_mask],
            s=28,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.0,
            label=f"sigma-clipped ({int(cut_mask.sum())})",
        )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(label)
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(-0.5, img.shape[0] - 0.5)
    ax.set_xlabel("x [px, 0-based crop]")
    ax.set_ylabel("y [px, 0-based crop]")
    ax.set_title(f"{title}\n{label}")
    if np.any(cut_mask):
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_grid_points(nx: int, ny: int, n_side: int = 10, margin_frac: float = 0.08) -> np.ndarray:
    """Return (n_side^2, 2) interior pixel grid."""
    mx = margin_frac * nx
    my = margin_frac * ny
    xs = np.linspace(mx, nx - 1 - mx, n_side)
    ys = np.linspace(my, ny - 1 - my, n_side)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def grid_wcs_residuals(
    wcs_a: WCS,
    wcs_b: WCS,
    grid_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each grid pixel in *wcs_a*, convert to sky and back through *wcs_b*.

    Returns dx, dy, r in pixels of *wcs_a*'s pixel system.
    """
    ra, dec = wcs_a.pixel_to_world_values(grid_xy[:, 0], grid_xy[:, 1])
    xb, yb = wcs_b.world_to_pixel_values(ra, dec)
    dx = xb - grid_xy[:, 0]
    dy = yb - grid_xy[:, 1]
    r = np.hypot(dx, dy)
    return dx, dy, r


def resolve_spoc_ffi_path(ffi_dir: Path, stem: str, sector: int, camera: int, ccd: int) -> str | None:
    product = tess_product_id_from_ffi_path(stem) or stem
    patterns = [
        f"{product}-s{sector:04d}-{camera}-{ccd}-*_ffic.fits.fz",
        f"{product}-s{sector:04d}-{camera}-{ccd}-*_ffic.fits.gz",
        f"{product}-s{sector:04d}-{camera}-{ccd}-*_ffic.fits",
        f"{product}*_ffic.fits.fz",
        f"{product}*_ffic.fits.gz",
        f"{product}*_ffic.fits",
    ]
    cam_dirs = [
        ffi_dir / f"s{sector:04d}" / f"cam{camera}_ccd{ccd}",
        ffi_dir / f"s{sector:04d}" / f"camera{camera}_ccd{ccd}",
        ffi_dir,
    ]
    for cam_dir in cam_dirs:
        if not cam_dir.is_dir():
            continue
        for pat in patterns:
            matches = sorted(cam_dir.glob(pat))
            if matches:
                return str(matches[0])
    return None


def tess_crop_wcs_for_frame(
    hp_d_path: Path,
    ffi_dir: Path | None,
    sector: int,
    camera: int,
    ccd: int,
    crop_bounds: dict[str, Any],
    stem: str,
) -> tuple[WCS, str]:
    """
    Prefer fresh crop WCS from the SPOC FFI; fall back to the hp_d header WCS.

    The hp_d header is already ``crop_ffi_header`` output from the epoch FFI, so
    it is a valid per-epoch TESS crop baseline when raw FFIs are offline.
    """
    if ffi_dir is not None:
        ffi_path = resolve_spoc_ffi_path(ffi_dir, stem, sector, camera, ccd)
        if ffi_path is not None:
            resolved = resolve_existing_fits_path(ffi_path)
            if resolved:
                hdr = crop_ffi_header(resolved, crop_bounds)
                return WCS(hdr), f"spoc:{resolved}"

    hdr = fits.getheader(hp_d_path, ext=1)
    return WCS(hdr), "hp_d_header"


def plot_du_dv_panels(
    out_path: Path,
    *,
    wcs: WCS,
    stars_fit: pd.DataFrame,
    stars_cut: pd.DataFrame | None,
    title: str,
    residual_ylim: float = 0.4,
) -> None:
    """
    Notebook-style distortion residual panels (obs − pred).

    Panels: dv vs y, du vs x, dv vs x, du vs y.
    """
    style_fit = dict(s=4, c="C2", alpha=0.55, linewidths=0, label="fit stars")
    style_cut = dict(s=4, c="C1", alpha=0.45, linewidths=0, label="clipped")

    du_fit, dv_fit = star_du_dv_px(wcs, stars_fit)
    x_fit = stars_fit["x_fit"].to_numpy(dtype=float)
    y_fit = stars_fit["y_fit"].to_numpy(dtype=float)

    du_cut = dv_cut = x_cut = y_cut = None
    if stars_cut is not None and len(stars_cut) > 0:
        du_cut, dv_cut = star_du_dv_px(wcs, stars_cut)
        x_cut = stars_cut["x_fit"].to_numpy(dtype=float)
        y_cut = stars_cut["y_fit"].to_numpy(dtype=float)

    all_res = [du_fit, dv_fit]
    if du_cut is not None:
        all_res.extend([du_cut, dv_cut])
    p95 = float(np.percentile(np.abs(np.concatenate(all_res)), 95))
    ylim = max(residual_ylim, p95 * 1.05)
    residual_limits = (-ylim, ylim)

    med_du = float(np.median(np.abs(du_fit)))
    med_dv = float(np.median(np.abs(dv_fit)))
    full_title = f"{title}\nmed|du|={med_du:.3f}  med|dv|={med_dv:.3f} px"

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=False)
    panels = (
        (axes[0], y_fit, dv_fit, y_cut, dv_cut, "y [px]", "dv_pix", "dv vs y"),
        (axes[1], x_fit, du_fit, x_cut, du_cut, "x [px]", "du_pix", "du vs x"),
        (axes[2], x_fit, dv_fit, x_cut, dv_cut, "x [px]", "dv_pix", "dv vs x"),
        (axes[3], y_fit, du_fit, y_cut, du_cut, "y [px]", "du_pix", "du vs y"),
    )
    for i, (ax, xg, yg, xc, yc, xlabel, ylabel, panel_title) in enumerate(panels):
        if xc is not None and yc is not None:
            ax.scatter(xc, yc, **style_cut)
        ax.scatter(xg, yg, **style_fit)
        ax.axhline(0.0, color="k", lw=1.5)
        ax.set_ylim(residual_limits)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_title(f"{full_title}\n{panel_title}")
        else:
            ax.set_title(panel_title)
        if i == 0 and (stars_cut is not None and len(stars_cut) > 0):
            ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_frame_diagnostics(
    out_dir: Path,
    stem: str,
    segment: str,
    btjd: float,
    stars: pd.DataFrame,
    stars_cut: pd.DataFrame | None,
    wcs_tess: WCS,
    wcs_fit: WCS,
    grid_xy: np.ndarray,
    grid_dx: np.ndarray,
    grid_dy: np.ndarray,
    grid_r: np.ndarray,
    n_side: int,
    *,
    sip_degree_used: int | None = None,
    residual_ylim: float = 0.4,
) -> None:
    frame_dir = out_dir / "frames" / segment / stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Star residuals: TESS vs measured
    _, _, r_tess = star_residuals_px(wcs_tess, stars)
    dx_fit, dy_fit, r_fit = star_residuals_px(wcs_fit, stars)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    sc = ax.scatter(
        stars["x_fit"],
        stars["y_fit"],
        c=r_tess,
        s=6,
        cmap="viridis",
        vmin=0,
        vmax=np.percentile(r_tess, 95),
    )
    ax.set_title(f"TESS WCS vs stars\nmed={np.median(r_tess):.3f} px")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="|residual| [px]")

    ax = axes[1]
    sc = ax.scatter(
        stars["x_fit"],
        stars["y_fit"],
        c=r_fit,
        s=6,
        cmap="viridis",
        vmin=0,
        vmax=np.percentile(r_fit, 95),
    )
    ax.set_title(f"Fitted WCS vs stars\nmed={np.median(r_fit):.3f} px")
    ax.set_xlabel("x [px]")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="|residual| [px]")

    ax = axes[2]
    step = max(1, len(stars) // 400)
    sub = stars.iloc[::step]
    dx_t, dy_t, _ = star_residuals_px(wcs_tess, sub)
    ax.quiver(
        sub["x_fit"],
        sub["y_fit"],
        dx_t,
        dy_t,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="C0",
        alpha=0.5,
        label="TESS",
    )
    ax.quiver(
        sub["x_fit"],
        sub["y_fit"],
        star_residuals_px(wcs_fit, sub)[0],
        star_residuals_px(wcs_fit, sub)[1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="C3",
        alpha=0.5,
        label="Fit",
    )
    ax.set_title("Star residual vectors")
    ax.set_xlabel("x [px]")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.suptitle(f"{stem}  segment={segment}  BTJD={btjd:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(frame_dir / "star_residuals.png", dpi=150)
    plt.close(fig)

    # 10x10 grid residual maps (TESS vs Fit)
    gx = grid_xy[:, 0].reshape(n_side, n_side)
    gy = grid_xy[:, 1].reshape(n_side, n_side)
    gdx = grid_dx.reshape(n_side, n_side)
    gdy = grid_dy.reshape(n_side, n_side)
    gr = grid_r.reshape(n_side, n_side)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    vmax = max(0.05, float(np.percentile(grid_r, 95)))
    im0 = axes[0, 0].pcolormesh(gx, gy, gr, cmap="magma", shading="auto", vmin=0, vmax=vmax)
    axes[0, 0].set_title(f"Grid |TESS−Fit| [px]  med={np.median(grid_r):.4f}")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(gx, gy, gdx, cmap="RdBu_r", shading="auto")
    axes[0, 1].set_title("Grid Δx (Fit−TESS) [px]")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].pcolormesh(gx, gy, gdy, cmap="RdBu_r", shading="auto")
    axes[1, 0].set_title("Grid Δy (Fit−TESS) [px]")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1, 0])

    axes[1, 1].quiver(
        grid_xy[:, 0],
        grid_xy[:, 1],
        grid_dx,
        grid_dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="k",
    )
    axes[1, 1].set_title("Grid residual vectors")
    axes[1, 1].set_xlim(0, wcs_tess.pixel_shape[0])
    axes[1, 1].set_ylim(0, wcs_tess.pixel_shape[1])
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlabel("x [px]")
    axes[1, 1].set_ylabel("y [px]")

    fig.suptitle(f"{stem} — {n_side}×{n_side} grid TESS vs fitted WCS", fontsize=11)
    fig.tight_layout()
    fig.savefig(frame_dir / "grid_tess_vs_fit.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(grid_r, bins=30, color="C0", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.axvline(np.median(grid_r), color="C3", ls="--", label=f"median={np.median(grid_r):.4f}")
    ax.axvline(np.percentile(grid_r, 95), color="C2", ls=":", label=f"p95={np.percentile(grid_r, 95):.4f}")
    ax.set_xlabel("|TESS − Fit| at grid points [px]")
    ax.set_ylabel("count")
    ax.set_title(f"{stem} grid residual histogram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(frame_dir / "grid_residual_hist.png", dpi=150)
    plt.close(fig)

    # Notebook-style du/dv panels: TESS (before) and fitted WCS (after)
    base_title = f"{stem}  segment={segment}  BTJD={btjd:.3f}"
    plot_du_dv_panels(
        frame_dir / "tess_du_dv_panels.png",
        wcs=wcs_tess,
        stars_fit=stars,
        stars_cut=stars_cut,
        title=f"{base_title}  WCS=TESS",
        residual_ylim=residual_ylim,
    )
    sip_label = f"Fit SIP-{sip_degree_used}" if sip_degree_used is not None else "Fit"
    plot_du_dv_panels(
        frame_dir / "fit_du_dv_panels.png",
        wcs=wcs_fit,
        stars_fit=stars,
        stars_cut=stars_cut,
        title=f"{base_title}  WCS={sip_label}",
        residual_ylim=residual_ylim,
    )


def plot_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, color in (("tess_vs_stars_median_px", "C0"), ("fit_vs_stars_median_px", "C3")):
        for seg, marker in (("start", "o"), ("mid", "s"), ("end", "^")):
            sub = summary[summary["segment"] == seg]
            ax.scatter(
                sub["btjd"],
                sub[label],
                marker=marker,
                s=40,
                alpha=0.85,
                label=f"{seg} {label.split('_')[0]}",
            )
    ax.set_xlabel("BTJD")
    ax.set_ylabel("median star residual [px]")
    ax.set_title("Star residuals: TESS vs fitted WCS")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "summary_star_residuals_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for seg, marker in (("start", "o"), ("mid", "s"), ("end", "^")):
        sub = summary[summary["segment"] == seg]
        ax.scatter(
            sub["btjd"],
            sub["grid_tess_vs_fit_median_px"],
            marker=marker,
            s=40,
            label=f"{seg} med",
        )
        ax.scatter(
            sub["btjd"],
            sub["grid_tess_vs_fit_p95_px"],
            marker=marker,
            s=25,
            alpha=0.5,
            label=f"{seg} p95",
        )
    ax.set_xlabel("BTJD")
    ax.set_ylabel("grid TESS−Fit residual [px]")
    ax.set_title(f"{len(summary)} frames — {summary['grid_n_side'].iloc[0]}×{summary['grid_n_side'].iloc[0]} grid")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "summary_grid_residuals_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    segments = ["start", "mid", "end"]
    x = np.arange(len(segments))
    width = 0.35
    med_tess = [summary.loc[summary.segment == s, "tess_vs_stars_median_px"].median() for s in segments]
    med_fit = [summary.loc[summary.segment == s, "fit_vs_stars_median_px"].median() for s in segments]
    ax.bar(x - width / 2, med_tess, width, label="TESS vs stars", color="C0")
    ax.bar(x + width / 2, med_fit, width, label="Fit vs stars", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(segments)
    ax.set_ylabel("median residual [px]")
    ax.set_title("Orbit-2 sample segments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "summary_segment_boxless.png", dpi=150)
    plt.close(fig)


def process_frame(
    workspace: Path,
    stem: str,
    segment: str,
    *,
    centroids_label: str,
    gaia: pd.DataFrame,
    ffi_dir: Path | None,
    sector: int,
    camera: int,
    ccd: int,
    star_cfg: StarSelectionConfig,
    fit_cfg: FitConfig,
    grid_n_side: int,
    out_dir: Path,
    residual_ylim: float = 0.4,
) -> FrameResult:
    hp_d_path = resolve_hp_d_path(workspace, stem)
    ecsv_path = workspace / centroids_label / f"{stem}_photresults.ecsv"
    if hp_d_path is None:
        return FrameResult(
            stem, segment, float("nan"), 0, 0, False, None,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            message=f"missing hp_d for stem={stem} under {workspace / 'hp_d'}",
        )
    if not ecsv_path.is_file():
        return FrameResult(
            stem, segment, float("nan"), 0, 0, False, None,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            message=f"missing centroids: {ecsv_path}",
        )

    hdr = fits.getheader(hp_d_path, ext=1)
    btjd = _btjd_from_header(hdr)
    crop_bounds = crop_bounds_from_header(hdr)
    ny, nx = crop_bounds["shape"]

    wcs_tess, tess_source = tess_crop_wcs_for_frame(
        hp_d_path, ffi_dir, sector, camera, ccd, crop_bounds, stem
    )

    phot = Table.read(ecsv_path)
    merged = join_stars(phot, gaia)
    good = select_good_stars(merged, star_cfg)
    n_in = len(good)
    if n_in < star_cfg.min_stars:
        return FrameResult(
            stem, segment, btjd, n_in, 0, False, None,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            message=f"only {n_in} stars after QC",
        )

    clipped = iterative_clip_stars(
        good, wcs_tess, n_sigma=star_cfg.clip_n_sigma, max_iter=star_cfg.clip_max_iter
    )
    stars_cut = good.loc[~good.index.isin(clipped.index)].copy()
    wcs_fit, sip_used, err = fit_crop_wcs(clipped, (ny, nx), fit_cfg)
    if wcs_fit is None:
        return FrameResult(
            stem, segment, btjd, n_in, len(clipped), False, None,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            message=f"fit failed: {err}",
        )

    _, _, r_tess = star_residuals_px(wcs_tess, clipped)
    _, _, r_fit = star_residuals_px(wcs_fit, clipped)

    grid_xy = make_grid_points(nx, ny, n_side=grid_n_side)
    gdx, gdy, gr = grid_wcs_residuals(wcs_tess, wcs_fit, grid_xy)

    plot_frame_diagnostics(
        out_dir,
        stem,
        segment,
        btjd,
        clipped,
        stars_cut if len(stars_cut) else None,
        wcs_tess,
        wcs_fit,
        grid_xy,
        gdx,
        gdy,
        gr,
        grid_n_side,
        sip_degree_used=sip_used,
        residual_ylim=residual_ylim,
    )

    # Save fitted header
    hdr_out = out_dir / "headers" / f"{stem}_wcs.fits"
    hdr_out.parent.mkdir(parents=True, exist_ok=True)
    wcs_fit.to_header(relax=True)
    fits.PrimaryHDU(data=np.zeros((ny, nx), dtype=np.float32), header=wcs_fit.to_header(relax=True)).writeto(
        hdr_out, overwrite=True
    )

    return FrameResult(
        stem=stem,
        segment=segment,
        btjd=btjd,
        n_stars_in=n_in,
        n_stars_fit=len(clipped),
        fit_ok=True,
        sip_degree_used=sip_used,
        tess_vs_stars_median_px=float(np.median(r_tess)),
        tess_vs_stars_p95_px=float(np.percentile(r_tess, 95)),
        fit_vs_stars_median_px=float(np.median(r_fit)),
        fit_vs_stars_p95_px=float(np.percentile(r_fit, 95)),
        grid_tess_vs_fit_median_px=float(np.median(gr)),
        grid_tess_vs_fit_p95_px=float(np.percentile(gr, 95)),
        grid_tess_vs_fit_max_px=float(np.max(gr)),
        message=tess_source,
    )


def build_frame_list(
    workspace: Path,
    centroids_label: str,
    stems: Iterable[str] | None,
    *,
    orbit_index: int,
    n_each: int,
) -> list[tuple[str, str]]:
    """Return [(stem, segment), ...]."""
    if stems is not None:
        return [(s, "manual") for s in stems]

    index = load_centroids_index(str(workspace / centroids_label))
    all_stems = sorted(index.keys())
    btjd_list = []
    for s in all_stems:
        hp = resolve_hp_d_path(workspace, s)
        if hp is None:
            btjd_list.append(float("nan"))
            continue
        btjd_list.append(_btjd_from_header(fits.getheader(hp, ext=1)))
    segments = select_orbit_segments(
        all_stems, btjd_list, orbit_index=orbit_index, n_each=n_each
    )
    out: list[tuple[str, str]] = []
    for seg, seg_stems in segments.items():
        for s in seg_stems:
            out.append((s, seg))
    return out


def run_batch(
    workspace: Path,
    out_dir: Path,
    *,
    centroids_label: str = "centroids_r1",
    stems: Sequence[str] | None = None,
    orbit_index: int = 1,
    n_each: int = 10,
    sector: int = 22,
    camera: int = 3,
    ccd: int = 3,
    ffi_dir: Path | None = None,
    grid_n_side: int = 10,
    residual_ylim: float = 0.4,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    gaia = load_gaia_catalog(workspace)
    star_cfg = StarSelectionConfig()
    fit_cfg = FitConfig()

    if ffi_dir is None:
        import yaml

        cfg_path = workspace / "diff_config.yaml"
        if cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text())
            fd = cfg.get("ffi_dir")
            if fd:
                ffi_dir = Path(fd)

    frames = build_frame_list(
        workspace, centroids_label, stems,
        orbit_index=orbit_index, n_each=n_each,
    )
    log.info("Processing %d frames", len(frames))

    rows: list[dict[str, Any]] = []
    for stem, segment in frames:
        log.info("Frame %s (%s)", stem, segment)
        res = process_frame(
            workspace,
            stem,
            segment,
            centroids_label=centroids_label,
            gaia=gaia,
            ffi_dir=ffi_dir,
            sector=sector,
            camera=camera,
            ccd=ccd,
            star_cfg=star_cfg,
            fit_cfg=fit_cfg,
            grid_n_side=grid_n_side,
            out_dir=out_dir,
            residual_ylim=residual_ylim,
        )
        rows.append({**res.__dict__, "grid_n_side": grid_n_side})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "frame_list.json", "w", encoding="utf-8") as fh:
        json.dump(
            [{"stem": s, "segment": seg} for s, seg in frames],
            fh,
            indent=2,
        )
    plot_summary(out_dir, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fit WCS from centroids and compare to TESS.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centroids-label", default="centroids_r1")
    parser.add_argument("--orbit-index", type=int, default=1, help="0-based orbit chain index")
    parser.add_argument("--n-each", type=int, default=10, help="Frames per start/mid/end segment")
    parser.add_argument("--sector", type=int, default=22)
    parser.add_argument("--camera", type=int, default=3)
    parser.add_argument("--ccd", type=int, default=3)
    parser.add_argument("--ffi-dir", type=Path, default=None)
    parser.add_argument("--grid-n-side", type=int, default=10)
    parser.add_argument(
        "--residual-ylim",
        type=float,
        default=0.4,
        help="Symmetric |du|/|dv| axis limit in pixels (auto-expands if p95 is larger)",
    )
    parser.add_argument("--stems", nargs="*", default=None, help="Explicit stems (skip auto selection)")
    args = parser.parse_args(argv)

    run_batch(
        args.workspace,
        args.out,
        centroids_label=args.centroids_label,
        stems=args.stems,
        orbit_index=args.orbit_index,
        n_each=args.n_each,
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        ffi_dir=args.ffi_dir,
        grid_n_side=args.grid_n_side,
        residual_ylim=args.residual_ylim,
    )


if __name__ == "__main__":
    main()
