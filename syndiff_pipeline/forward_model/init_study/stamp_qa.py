# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Hotpants-style stamp QA for forward_epsf_wcs init studies.

Two-phase rejection on free-xy simultaneous PSF photometry (gridded ePSF):

Phase 1 (norm / kernel-sum analogue)
    For each source stamp, take ``norm = flux_fit / tess_flux`` (fitted scale).
    Estimate ``mu, sigma`` of the norm population with iterative sigma-clip,
    then two-sided reject::

        |norm_i - mu| / sigma >= ker_sig_reject

Phase 2 (residual variance)
    For surviving sources, build the model image, form a stamp residual, and
    score::

        chi2 = mean( (model - data)^2 / var )

    One-sided reject (poor fits only)::

        (chi2_i - mu_chi) > ker_sig_reject * sigma_chi

Positions are free (photutils PSFPhotometry fits x, y, flux). Sources that
fall in the same photutils SourceGrouper group are fit simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.nddata import NDData
from astropy.stats import sigma_clip
from astropy.table import Table

log = logging.getLogger(__name__)


@dataclass
class StampQaConfig:
    fit_shape: int = 11
    aperture_radius: float = 4.0
    grouper_min_separation: float = 7.0
    ker_sig_reject: float = 2.5
    sigma_clip_sigma: float = 3.0
    sigma_clip_maxiters: int = 5
    noise_floor: float = 1.0


@dataclass
class StampQaResult:
    table: pd.DataFrame
    phase1_mu: float
    phase1_sigma: float
    phase2_mu: float
    phase2_sigma: float
    ker_sig_reject: float
    n_input: int
    n_after_phase1: int
    n_after_phase2: int


def load_gridded_model_from_pooled_npz(
    path: Path,
    *,
    crop_origin: tuple[float, float] = (0.0, 0.0),
) -> Any:
    """Load pooled ePSF npz as ``GriddedPSFModel`` in full-crop coordinates.

    ``grid_xypos`` in the pooled build are relative to the cropped image; add
    ``crop_origin`` (x0, y0) so they match science / hp_d pixel coordinates.
    """
    from photutils.psf import GriddedPSFModel

    path = Path(path)
    z = np.load(path, allow_pickle=True)
    stack = np.asarray(z["data"], dtype=np.float64)
    grid = np.asarray(z["grid_xypos"], dtype=np.float64)
    oversampling = int(np.asarray(z["oversampling"]))
    x0, y0 = float(crop_origin[0]), float(crop_origin[1])
    grid_abs = np.column_stack([grid[:, 0] + x0, grid[:, 1] + y0])
    meta = {"grid_xypos": grid_abs, "oversampling": oversampling}
    return GriddedPSFModel(NDData(data=stack, meta=meta))


def _sigma_stats(values: np.ndarray, *, sigma: float, maxiters: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float(np.nanmean(arr)) if arr.size else float("nan"), float("nan")
    clipped = sigma_clip(arr, sigma=sigma, maxiters=maxiters, masked=True)
    good = np.asarray(clipped.compressed(), dtype=float)
    if good.size < 2:
        return float(np.nanmean(arr)), float(np.nanstd(arr))
    return float(np.mean(good)), float(np.std(good, ddof=1))


def phase1_norm_reject(
    norms: np.ndarray,
    *,
    ker_sig_reject: float = 2.5,
    sigma: float = 3.0,
    maxiters: int = 5,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Two-sided Hotpants-style norm clip. Returns keep mask, mu, sigma, diff."""
    norms = np.asarray(norms, dtype=float)
    mu, sig = _sigma_stats(norms, sigma=sigma, maxiters=maxiters)
    if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
        diff = np.full(norms.shape, np.nan)
        return np.isfinite(norms), mu, sig, diff
    diff = np.abs(norms - mu) / sig
    keep = np.isfinite(diff) & (diff < float(ker_sig_reject))
    return keep, mu, sig, diff


def phase2_chi2_reject(
    chi2: np.ndarray,
    *,
    active: np.ndarray | None = None,
    ker_sig_reject: float = 2.5,
    sigma: float = 3.0,
    maxiters: int = 5,
) -> tuple[np.ndarray, float, float]:
    """One-sided residual clip: reject only high chi2. Returns keep, mu, sigma."""
    chi2 = np.asarray(chi2, dtype=float)
    if active is None:
        active = np.isfinite(chi2) & (chi2 >= 0)
    else:
        active = np.asarray(active, dtype=bool) & np.isfinite(chi2) & (chi2 >= 0)
    mu, sig = _sigma_stats(chi2[active], sigma=sigma, maxiters=maxiters)
    keep = active.copy()
    if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
        return keep, mu, sig
    # One-sided: keep low residuals; clip only (chi2 - mu) > k * sig
    bad = active & ((chi2 - mu) > float(ker_sig_reject) * sig)
    keep[bad] = False
    return keep, mu, sig


def _stamp_chi2(
    data: np.ndarray,
    model: np.ndarray,
    noise: np.ndarray | None,
    x: float,
    y: float,
    *,
    fit_shape: int,
    noise_floor: float,
) -> float:
    """Mean weighted residual variance in a fit_shape stamp around (x, y)."""
    h = int(fit_shape) // 2
    xi, yi = int(np.floor(x)), int(np.floor(y))
    ny, nx = data.shape
    y0, y1 = max(0, yi - h), min(ny, yi + h + 1)
    x0, x1 = max(0, xi - h), min(nx, xi + h + 1)
    if y1 <= y0 or x1 <= x0:
        return -1.0
    d = data[y0:y1, x0:x1]
    m = model[y0:y1, x0:x1]
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(m)):
        return -1.0
    if noise is None:
        var = np.maximum(np.abs(d), noise_floor)
    else:
        n = noise[y0:y1, x0:x1]
        var = np.maximum(np.asarray(n, dtype=float) ** 2, noise_floor**2)
    resid = (m - d) ** 2 / var
    return float(np.mean(resid))


def _set_gridded_model_xy_fixed(epsf_model, fix_xy: bool) -> None:
    """Toggle whether GriddedPSFModel holds x/y at init (flux-only fit)."""
    epsf_model.x_0.fixed = bool(fix_xy)
    epsf_model.y_0.fixed = bool(fix_xy)


def run_gridded_photometry(
    image: np.ndarray,
    epsf_model,
    stars: pd.DataFrame,
    *,
    noise: np.ndarray | None = None,
    cfg: StampQaConfig | None = None,
    fix_xy: bool = False,
    compute_stamp_chi2: bool = True,
    mask: np.ndarray | None = None,
    error: np.ndarray | None = None,
    xy_bounds: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """PSFPhotometry on a GriddedPSFModel; simultaneous SourceGrouper groups.

  Parameters
  ----------
  fix_xy
      If True, hold each source at init (x, y) and fit flux only.
  compute_stamp_chi2
      Stamp QA scores (norm / chi2); skip for light-curve baselines.
  mask, error
      Optional boolean bad-pixel mask (True = ignore) and 1-sigma error image passed to
      ``PSFPhotometry`` (both default None: unchanged behaviour).
  xy_bounds
      Max allowed |x_fit - x_init|, |y_fit - y_init| in px (``PSFPhotometry(xy_bounds=...)``); None = unbounded.

  Returns ``(table, model_image)``. ``model_image`` is ``None`` when there are
  no stars or when ``compute_stamp_chi2`` is False.
    """
    from photutils.psf import PSFPhotometry, SourceGrouper

    cfg = cfg or StampQaConfig()
    if stars.empty:
        return pd.DataFrame(), None

    xcol_in = "x_init" if "x_init" in stars.columns else "x"
    ycol_in = "y_init" if "y_init" in stars.columns else "y"

    init = Table()
    init["x"] = np.asarray(stars[xcol_in], dtype=float)
    init["y"] = np.asarray(stars[ycol_in], dtype=float)
    if "tess_flux" in stars.columns:
        init["flux"] = np.asarray(stars["tess_flux"], dtype=float)
    else:
        init["flux"] = np.full(len(stars), 1000.0)

    meta_cols = [c for c in ("source_id", "tess_mag", "tess_flux", "ra", "dec") if c in stars.columns]
    for c in meta_cols:
        init[c] = stars[c].to_numpy()

    _set_gridded_model_xy_fixed(epsf_model, fix_xy)

    phot = PSFPhotometry(
        epsf_model,
        fit_shape=int(cfg.fit_shape),
        aperture_radius=float(cfg.aperture_radius),
        grouper=SourceGrouper(min_separation=float(cfg.grouper_min_separation)),
        local_bkg_estimator=None,
        xy_bounds=None if xy_bounds is None else float(xy_bounds),
    )
    result = phot(
        np.asarray(image, dtype=np.float64),
        mask=None if mask is None else np.asarray(mask, dtype=bool),
        error=None if error is None else np.asarray(error, dtype=np.float64),
        init_params=init,
    )
    tbl = result if hasattr(result, "colnames") else result.to_table()
    df = tbl.to_pandas()

    # photutils drops non-fit columns from init_params; reattach by row order.
    # PSFPhotometry returns one row per input source in the same order as init.
    if len(df) == len(stars):
        for c in meta_cols:
            if c not in df.columns:
                df[c] = stars[c].to_numpy()
    else:
        log.warning(
            "photometry row count %s != input stars %s; tess_flux/meta not reattached",
            len(df),
            len(stars),
        )

    xcol = "x_fit" if "x_fit" in df.columns else "x_0"
    ycol = "y_fit" if "y_fit" in df.columns else "y_0"
    fcol = "flux_fit" if "flux_fit" in df.columns else "flux_0"

    df["flux_fit"] = df[fcol].to_numpy(dtype=float)
    df["x_fit"] = df[xcol].to_numpy(dtype=float)
    df["y_fit"] = df[ycol].to_numpy(dtype=float)
    if "x_init" not in df.columns:
        df["x_init"] = init["x"]
    if "y_init" not in df.columns:
        df["y_init"] = init["y"]

    model_img = None
    if compute_stamp_chi2:
        try:
            model_img = phot.make_model_image(image.shape, psf_shape=int(cfg.fit_shape))
        except Exception:
            model_img = phot.make_model_image(image.shape)
        model_arr = np.asarray(model_img, dtype=float)
        chi2 = np.full(len(df), np.nan)
        for i, row in df.iterrows():
            chi2[i] = _stamp_chi2(
                image,
                model_arr,
                noise,
                float(row[xcol]),
                float(row[ycol]),
                fit_shape=cfg.fit_shape,
                noise_floor=cfg.noise_floor,
            )
        df["stamp_chi2"] = chi2

        tess_flux = (
            df["tess_flux"].to_numpy(dtype=float)
            if "tess_flux" in df.columns
            else np.full(len(df), np.nan)
        )
        flux_fit = df[fcol].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = flux_fit / tess_flux
        df["norm"] = norm

    _set_gridded_model_xy_fixed(epsf_model, False)
    return df, model_img


def run_free_xy_photometry(
    image: np.ndarray,
    epsf_model,
    stars: pd.DataFrame,
    *,
    noise: np.ndarray | None = None,
    cfg: StampQaConfig | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Free-xy GriddedPSF photometry (stamp QA default)."""
    return run_gridded_photometry(
        image,
        epsf_model,
        stars,
        noise=noise,
        cfg=cfg,
        fix_xy=False,
        compute_stamp_chi2=True,
    )


def apply_hotpants_stamp_qa(
    phot_df: pd.DataFrame,
    *,
    cfg: StampQaConfig | None = None,
) -> StampQaResult:
    """Apply phase-1 (norm) then phase-2 (chi2) rejection to a photometry table."""
    cfg = cfg or StampQaConfig()
    df = phot_df.copy()
    n_input = len(df)

    keep1, mu1, sig1, diff = phase1_norm_reject(
        df["norm"].to_numpy(dtype=float),
        ker_sig_reject=cfg.ker_sig_reject,
        sigma=cfg.sigma_clip_sigma,
        maxiters=cfg.sigma_clip_maxiters,
    )
    df["norm_diff"] = diff
    df["pass_phase1"] = keep1

    keep2, mu2, sig2 = phase2_chi2_reject(
        df["stamp_chi2"].to_numpy(dtype=float),
        active=keep1,
        ker_sig_reject=cfg.ker_sig_reject,
        sigma=cfg.sigma_clip_sigma,
        maxiters=cfg.sigma_clip_maxiters,
    )
    df["pass_phase2"] = keep2
    df["keep"] = keep2

    return StampQaResult(
        table=df,
        phase1_mu=mu1,
        phase1_sigma=sig1,
        phase2_mu=mu2,
        phase2_sigma=sig2,
        ker_sig_reject=float(cfg.ker_sig_reject),
        n_input=n_input,
        n_after_phase1=int(np.sum(keep1)),
        n_after_phase2=int(np.sum(keep2)),
    )
