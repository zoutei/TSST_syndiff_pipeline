"""Per-frame QC metrics for per-FFI WCS fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.wcs.sci2idl import Sci2IdlFitResult, sci2idl_du_dv_px
from syndiff_pipeline.difference_imaging.wcs.sip_poly_fit import poly_eval


def compute_frame_qc_metrics(
    result: Sci2IdlFitResult,
    stars: pd.DataFrame,
    *,
    stem: str,
    btjd: float,
    fit_ok: bool,
    message: str = "",
) -> dict[str, float | str | bool | int]:
    n_qc = len(stars)
    keep = result.keep_mask
    n_fit = int(keep.sum())
    du_all, dv_all = sci2idl_du_dv_px(result, stars)
    hypot_all = np.hypot(du_all, dv_all)

    row: dict[str, float | str | bool | int] = {
        "stem": stem,
        "btjd": float(btjd),
        "n_stars_qc": n_qc,
        "n_stars_fit": n_fit,
        "n_stars_clipped": n_qc - n_fit,
        "clip_fraction": float((n_qc - n_fit) / n_qc) if n_qc else float("nan"),
        "fit_ok": bool(fit_ok),
        "message": message,
        "med_abs_du_all": float(np.median(np.abs(du_all))),
        "med_abs_dv_all": float(np.median(np.abs(dv_all))),
    }
    if n_fit < 1:
        for key in (
            "med_abs_du", "med_abs_dv", "med_hypot", "rms_hypot",
            "p50_hypot", "p68_hypot", "p95_hypot", "p99_hypot", "max_hypot",
            "mad_du", "mad_dv", "med_du", "med_dv",
        ):
            row[key] = float("nan")
        return row

    du = du_all[keep]
    dv = dv_all[keep]
    hypot = hypot_all[keep]
    row.update({
        "med_abs_du": float(np.median(np.abs(du))),
        "med_abs_dv": float(np.median(np.abs(dv))),
        "med_hypot": float(np.median(hypot)),
        "rms_hypot": float(np.sqrt(np.mean(hypot**2))),
        "p50_hypot": float(np.percentile(hypot, 50)),
        "p68_hypot": float(np.percentile(hypot, 68)),
        "p95_hypot": float(np.percentile(hypot, 95)),
        "p99_hypot": float(np.percentile(hypot, 99)),
        "max_hypot": float(np.max(hypot)),
        "mad_du": float(np.median(np.abs(du - np.median(du)))),
        "mad_dv": float(np.median(np.abs(dv - np.median(dv)))),
        "med_du": float(np.median(du)),
        "med_dv": float(np.median(dv)),
    })
    return row


def build_stars_audit_table(
    stars: pd.DataFrame,
    result: Sci2IdlFitResult,
) -> pd.DataFrame:
    du, dv = sci2idl_du_dv_px(result, stars)
    u_fit = poly_eval(result.coeff_x, stars["xprime"], stars["yprime"], result.poly_degree)
    v_fit = poly_eval(result.coeff_y, stars["xprime"], stars["yprime"], result.poly_degree)
    out = stars.copy()
    out["used_in_fit"] = result.keep_mask
    out["du"] = du
    out["dv"] = dv
    out["hypot_resid"] = np.hypot(du, dv)
    out["u_fit"] = u_fit
    out["v_fit"] = v_fit
    return out
