"""Orchestrate spatial → temporal → strap background steps."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.stages.background import io, spatial, strap, temporal

log = logging.getLogger(__name__)


@dataclass
class BackgroundStepSpatialParams:
    enabled: bool = True
    box_size: int = 16
    filter_size: int = 3
    exclude_percentile: float = 50.0
    exclude_straps: bool = True
    n_jobs: int = -1
    save: Optional[str] = None


@dataclass
class BackgroundStepTemporalParams:
    enabled: bool = True
    method: str = "savgol"
    savgol_window: Optional[int] = None
    savgol_polyorder: int = 2
    gap_thresh_days: float = 0.5
    tile_size: int = 256
    w_min: int = 3
    w_max: int = 51
    block_size: int = 5
    vector_path: Optional[str] = None
    save: Optional[str] = None


@dataclass
class BackgroundStepStrapParams:
    enabled: bool = True
    qe_floor: float = 1.001
    fix_anomalies: bool = True
    save: Optional[str] = None


@dataclass
class BackgroundParams:
    recombine_inputs: bool = True
    write_per_frame_fits: bool = True
    write_stack: bool = True
    spatial: BackgroundStepSpatialParams = field(
        default_factory=BackgroundStepSpatialParams
    )
    temporal: BackgroundStepTemporalParams = field(
        default_factory=BackgroundStepTemporalParams
    )
    strap: BackgroundStepStrapParams = field(default_factory=BackgroundStepStrapParams)


def btjd_for_records(
    wcs_table: pd.DataFrame,
    records: List[io.FrameRecord],
) -> np.ndarray:
    from syndiff_pipeline.difference_imaging.support.ffi_naming import (
        parse_workspace_frame_stem,
        tess_product_id_from_ffi_path,
    )

    if "btjd" not in wcs_table.columns:
        raise ValueError("wcs_table must contain a 'btjd' column")
    col = "path" if "path" in wcs_table.columns else "filename"
    pids_tbl = wcs_table[col].astype(str).map(
        lambda p: tess_product_id_from_ffi_path(p) or ""
    )
    btjd_series = pd.to_numeric(wcs_table["btjd"], errors="coerce")
    pid_to_btjd = dict(zip(pids_tbl, btjd_series))
    pid_to_btjd.pop("", None)

    out = []
    for rec in records:
        v = pid_to_btjd.get(rec.product_id, np.nan)
        out.append(float(v) if pd.notna(v) else np.nan)
    return np.asarray(out, dtype=float)


def _save_intermediate(
    stack: np.ndarray,
    records: List[io.FrameRecord],
    ws_path: str,
    *,
    write_per_frame_fits: bool,
    write_stack: bool,
) -> None:
    os.makedirs(ws_path, exist_ok=True)
    if write_stack:
        io.save_stack(stack, ws_path)
    if write_per_frame_fits:
        io.write_per_frame_fits(ws_path, stack, records)


def run_background_pipeline(
    *,
    params: BackgroundParams,
    records: List[io.FrameRecord],
    mask: np.ndarray,
    wcs_table: pd.DataFrame,
    sector: int,
    camera: int,
    n_jobs: int,
    fit_flux: Optional[np.ndarray] = None,
    strap_flux: Optional[np.ndarray] = None,
    bkg_in_stack: Optional[np.ndarray] = None,
    workspace_resolver,
) -> np.ndarray:
    """
    Run enabled background steps and return final (T, ny, nx) stack.

    ``workspace_resolver(label)`` returns absolute path for a workspace label.
    """
    if not (
        params.spatial.enabled or params.temporal.enabled or params.strap.enabled
    ):
        raise ValueError("background stage: at least one step must be enabled")

    if params.strap.enabled and strap_flux is None:
        raise ValueError("strap step requires diffs flux cube (inputs.diffs)")

    if bkg_in_stack is not None and not params.spatial.enabled:
        stack = np.asarray(bkg_in_stack, dtype=np.float32)
    elif params.spatial.enabled:
        if fit_flux is None:
            raise ValueError("spatial step requires diffs flux cube")
        log.info(
            "background: spatial step on cube %s (n_jobs=%s)",
            fit_flux.shape,
            params.spatial.n_jobs if params.spatial.n_jobs > 0 else n_jobs,
        )
        nj = params.spatial.n_jobs if params.spatial.n_jobs > 0 else n_jobs
        stack = spatial.spatial_step(
            fit_flux,
            mask,
            box_size=params.spatial.box_size,
            filter_size=params.spatial.filter_size,
            exclude_percentile=params.spatial.exclude_percentile,
            exclude_straps=params.spatial.exclude_straps,
            n_jobs=nj,
        )
        if params.spatial.save:
            _save_intermediate(
                stack,
                records,
                workspace_resolver(params.spatial.save),
                write_per_frame_fits=params.write_per_frame_fits,
                write_stack=params.write_stack,
            )
    else:
        raise ValueError(
            "background: bkg_in required when spatial step is disabled"
        )

    if params.temporal.enabled:
        time_btjd = btjd_for_records(wcs_table, records)
        log.info(
            "background: temporal step method=%s cube %s",
            params.temporal.method,
            stack.shape,
        )
        stack = temporal.temporal_step(
            stack,
            time_btjd,
            sector,
            camera,
            method=params.temporal.method,
            savgol_window=params.temporal.savgol_window,
            savgol_polyorder=params.temporal.savgol_polyorder,
            gap_thresh_days=params.temporal.gap_thresh_days,
            tile_size=params.temporal.tile_size,
            w_min=params.temporal.w_min,
            w_max=params.temporal.w_max,
            block_size=params.temporal.block_size,
            vector_path=params.temporal.vector_path,
            n_jobs=n_jobs,
        )
        if params.temporal.save:
            _save_intermediate(
                stack,
                records,
                workspace_resolver(params.temporal.save),
                write_per_frame_fits=params.write_per_frame_fits,
                write_stack=params.write_stack,
            )

    if params.strap.enabled:
        time_mjd = btjd_for_records(wcs_table, records) + 57000.0
        log.info("background: strap step on cube %s", stack.shape)
        stack = strap.strap_step(
            strap_flux,
            stack,
            mask,
            time_mjd,
            qe_floor=params.strap.qe_floor,
            fix_anomalies=params.strap.fix_anomalies,
            n_jobs=n_jobs,
        )
        if params.strap.save:
            _save_intermediate(
                stack,
                records,
                workspace_resolver(params.strap.save),
                write_per_frame_fits=params.write_per_frame_fits,
                write_stack=params.write_stack,
            )

    return np.asarray(stack, dtype=np.float32)
