"""Forced photometry on small per-frame star diff stamps."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.stages.photometry import (
    _build_aperture_masks,
    _tessreduce_error_plane,
    aperture_flux_on_cutout,
    create_psf,
)
from syndiff_pipeline.star.identifiers import ResolvedHost


def read_star_diff_stamp(path: str) -> tuple[np.ndarray, dict]:
    """Read a stamp FITS written by :func:`~syndiff_pipeline.star.diff_runner.write_star_diff_stamp`."""
    with fits.open(path, memmap=True) as hdul:
        stamp = np.asarray(hdul[0].data, dtype=np.float64)
        hdr = hdul[0].header
        header = {
            "xmin": int(hdr["XMIN"]),
            "ymin": int(hdr["YMIN"]),
            "host_x": float(hdr["HOSTX"]),
            "host_y": float(hdr["HOSTY"]),
        }
    return stamp, header


def aperture_flux_on_stamp(
    stamp: np.ndarray,
    host_xy: tuple[float, float],
    *,
    tar_ap: float = 3.0,
    sky_in: float = 5.0,
    sky_out: float = 9.0,
) -> dict:
    """Local-background-subtracted aperture photometry on a stamp at *host_xy*."""
    data = np.asarray(stamp, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"stamp must be 2D, got shape {data.shape}")

    host_x, host_y = float(host_xy[0]), float(host_xy[1])
    center_x = int(round(host_x))
    center_y = int(round(host_y))
    shape = data.shape
    ap_tar, ap_sky, n_tar = _build_aperture_masks(
        shape,
        center_y,
        center_x,
        int(tar_ap),
        int(sky_in),
        int(sky_out),
    )
    flux, sky, flux_wo_sky, eflux = aperture_flux_on_cutout(
        data,
        ap_tar,
        ap_sky,
        n_tar,
    )
    annulus = data * ap_sky
    from astropy.stats import sigma_clipped_stats

    sky_median, _, _ = sigma_clipped_stats(annulus)
    return {
        "flux": float(flux_wo_sky),
        "flux_err": float(eflux),
        "sky_median": float(sky_median),
        "aperture_sum_raw": float(flux),
        "sky": float(sky),
        "flux_with_sky": float(flux),
        "flux_wo_sky": float(flux_wo_sky),
        "eflux": float(eflux),
    }


def psf_flux_on_stamp(
    stamp: np.ndarray,
    host_xy: tuple[float, float],
    epsf_model,
    *,
    psf_size: int = 11,
    phot_bkg_poly_order: int = 3,
) -> dict:
    """Fit a PSF/ePSF model at *host_xy* on a small stamp using ``create_psf``."""
    data = np.asarray(stamp, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"stamp must be 2D, got shape {data.shape}")

    size = int(data.shape[0])
    if data.shape[1] != size:
        raise ValueError(f"stamp must be square, got shape {data.shape}")

    fit_size = min(size, 2 * int(psf_size) + 1)
    if fit_size < size:
        half = fit_size // 2
        cx = int(round(host_xy[0]))
        cy = int(round(host_xy[1]))
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = min(size, y0 + fit_size)
        x1 = min(size, x0 + fit_size)
        if y1 - y0 < fit_size:
            y0 = max(0, y1 - fit_size)
        if x1 - x0 < fit_size:
            x0 = max(0, x1 - fit_size)
        data = data[y0:y1, x0:x1]
        host_x = float(host_xy[0]) - x0
        host_y = float(host_xy[1]) - y0
        size = int(data.shape[0])
    else:
        host_x, host_y = float(host_xy[0]), float(host_xy[1])

    cent = size / 2.0 - 0.5
    shiftx = host_x - cent
    shifty = host_y - cent

    psf_obj = create_psf(epsf_model, size)
    psf_obj.source(shiftx=shiftx, shifty=shifty)
    error = _tessreduce_error_plane(None, data.shape)
    psf_obj.psf_flux(
        data,
        error=error,
        surface=True,
        poly_order=int(phot_bkg_poly_order),
    )
    return {
        "flux": float(psf_obj.flux),
        "flux_err": float(psf_obj.eflux),
        "eflux": float(psf_obj.eflux),
    }


def _lightcurve_csv_name(method_name: str, host: ResolvedHost) -> str:
    host_id = int(host.gaia_source_id)
    return f"lightcurve_{method_name}_gaia_{host_id}.csv"


def _method_type(method: dict) -> str:
    return str(method.get("type", "aperture")).strip().lower()


def _epoch_time_value(
    index: int,
    time_values: Optional[list],
) -> float:
    if time_values is None or index >= len(time_values):
        return float(np.nan)
    try:
        return float(time_values[index])
    except (TypeError, ValueError):
        return float(np.nan)


def _aperture_record(
    stamp_path: str,
    header: dict,
    result: dict,
    *,
    btjd: float,
    group_id: int = -1,
) -> dict:
    return {
        "btjd": btjd,
        "flux": result["flux_with_sky"],
        "flux_wo_sky": result["flux_wo_sky"],
        "sky": result["sky"],
        "eflux": result["eflux"],
        "filename": stamp_path,
        "group_id": group_id,
        "xmin": header["xmin"],
        "ymin": header["ymin"],
        "host_x": header["host_x"],
        "host_y": header["host_y"],
    }


def _psf_record(
    stamp_path: str,
    header: dict,
    result: dict,
    *,
    btjd: float,
    group_id: int = -1,
) -> dict:
    return {
        "btjd": btjd,
        "flux": result["flux"],
        "eflux": result["eflux"],
        "filename": stamp_path,
        "group_id": group_id,
        "xmin": header["xmin"],
        "ymin": header["ymin"],
        "host_x": header["host_x"],
        "host_y": header["host_y"],
    }


def run_windowed_forced_photometry(
    stamp_paths: list[str],
    *,
    host: ResolvedHost,
    methods: list[dict],
    output_dir: str,
    time_values: Optional[list] = None,
    group_ids: Optional[list] = None,
) -> dict[str, pd.DataFrame]:
    """Run forced photometry on per-frame star diff stamps for one host."""
    if not stamp_paths:
        raise ValueError("stamp_paths must not be empty")
    if not methods:
        raise ValueError("methods must not be empty")

    os.makedirs(output_dir, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for method in methods:
        method_name = str(method["name"]).strip()
        mtype = _method_type(method)
        records: list[dict] = []

        for i, stamp_path in enumerate(stamp_paths):
            btjd = _epoch_time_value(i, time_values)
            gid = (
                int(group_ids[i])
                if group_ids is not None and i < len(group_ids)
                else -1
            )

            try:
                stamp, header = read_star_diff_stamp(stamp_path)
                host_xy = (header["host_x"], header["host_y"])
            except Exception:
                nan_rec = {
                    "btjd": btjd,
                    "flux": np.nan,
                    "eflux": np.nan,
                    "filename": stamp_path,
                    "group_id": gid,
                }
                if mtype == "aperture":
                    nan_rec.update(
                        {
                            "flux_wo_sky": np.nan,
                            "sky": np.nan,
                        }
                    )
                records.append(nan_rec)
                continue

            if mtype == "aperture":
                result = aperture_flux_on_stamp(
                    stamp,
                    host_xy,
                    tar_ap=float(method.get("tar_ap", 3.0)),
                    sky_in=float(method.get("sky_in", 5.0)),
                    sky_out=float(method.get("sky_out", 9.0)),
                )
                records.append(
                    _aperture_record(
                        stamp_path,
                        header,
                        result,
                        btjd=btjd,
                        group_id=gid,
                    )
                )
            elif mtype in {"psf", "prf"}:
                epsf_model = method.get("epsf_model")
                if epsf_model is None:
                    raise ValueError(
                        f"method {method_name!r} (type={mtype}) requires "
                        "'epsf_model' in the method dict"
                    )
                result = psf_flux_on_stamp(
                    stamp,
                    host_xy,
                    epsf_model,
                    psf_size=int(method.get("psf_size", 11)),
                    phot_bkg_poly_order=int(method.get("phot_bkg_poly_order", 3)),
                )
                records.append(
                    _psf_record(
                        stamp_path,
                        header,
                        result,
                        btjd=btjd,
                        group_id=gid,
                    )
                )
            else:
                raise ValueError(f"unsupported photometry method type {mtype!r}")

        lc_df = pd.DataFrame(records)
        csv_name = method.get("csv_basename") or _lightcurve_csv_name(method_name, host)
        if os.path.basename(csv_name) != csv_name or ".." in csv_name:
            raise ValueError(f"csv_basename must be a plain basename, got {csv_name!r}")
        out_path = os.path.join(output_dir, csv_name)
        lc_df.to_csv(out_path, index=False)
        out[method_name] = lc_df

    return out
