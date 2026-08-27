"""
Per-frame gridded empirical PSF fitting with photutils.

Builds a :class:`~photutils.psf.GriddedPSFModel` per difference image from
Gaia stars on a tile_ny × tile_nx grid (see starpositioningscript reference).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from joblib import delayed
from photutils.psf import EPSFBuilder, GriddedPSFModel, extract_stars

from syndiff_pipeline.common.joblib_progress import (
    parallel_map_with_optional_tqdm,
    tqdm_iter,
)
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    parse_workspace_frame_stem,
    strip_fits_suffix,
    tess_product_id_from_ffi_path,
)

log = logging.getLogger(__name__)

# photutils logs per-star exclusion at WARNING on crowded fields; not actionable.
_PHOTUTILS_EPSF_LOGGER = "photutils.psf.epsf_builder"


def _suppress_photutils_epsf_noise() -> None:
    """Silence photutils per-star ePSF exclusion chatter (uses logging, not warnings)."""
    logging.getLogger(_PHOTUTILS_EPSF_LOGGER).setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*has been excluded from ePSF fitting.*",
    )


_suppress_photutils_epsf_noise()

GRIDDED_EPSF_INDEX_BASENAME = "gridded_epsf_index.json"
GRIDDED_EPSF_NPZ_SUFFIX = "_gridded_epsf.npz"
GRIDDED_EPSF_ANCHORS_BASENAME = "gridded_epsf_anchors.json"

# Per-process state for loky workers (initialized once per worker, not per frame).
_WORKER_CTX: dict[str, Any] = {}


def _init_gridded_epsf_worker(
    gaia_df: pd.DataFrame,
    epsf_params,
    output_dir: str,
    mask_2d: np.ndarray | None,
    skip_existing: bool = True,
    sck: tuple | None = None,
    data_root: str | None = None,
    epsf_label: str | None = None,
    workspace_root: str | None = None,
    output_store_name: str | None = None,
    mask_catalog=None,
    btjd_by_stem: dict | None = None,
    diff_image_fps: dict[str, str] | None = None,
    ffi_list_df: pd.DataFrame | None = None,
    science_bounds: dict | None = None,
    ffi_path_by_stem: dict[str, str] | None = None,
) -> None:
    """Load shared ePSF inputs once per loky worker (see starpositioningscript)."""
    _suppress_photutils_epsf_noise()
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "gaia_df": gaia_df,
            "epsf_params": epsf_params,
            "output_dir": output_dir,
            "mask_2d": mask_2d,
            "skip_existing": bool(skip_existing),
            "sck": sck,
            "data_root": data_root,
            "epsf_label": epsf_label,
            "workspace_root": workspace_root,
            "output_store_name": output_store_name,
            "mask_catalog": mask_catalog,
            "btjd_by_stem": btjd_by_stem or {},
            "diff_image_fps": dict(diff_image_fps or {}),
            "ffi_list_df": ffi_list_df,
            "science_bounds": science_bounds,
            "ffi_path_by_stem": dict(ffi_path_by_stem or {}),
        }
    )


def _resolve_epsf_frame_mask(ctx: dict, ffi_stem: str) -> np.ndarray | None:
    """Per-FFI ePSF reject mask from MaskCatalog (preferred) or static fallback."""
    catalog = ctx.get("mask_catalog")
    if catalog is not None:
        from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask

        btjd = ctx.get("btjd_by_stem", {}).get(ffi_stem)
        return epsf_reject_mask(catalog.mask_at(btjd, which="full"))
    return ctx.get("mask_2d")


def _configure_blas_threads(n_workers: int) -> None:
    """Match starpositioningscript: let BLAS use cores not claimed by other workers."""
    cpu_cap = os.cpu_count() or 1
    per_worker = max(1, cpu_cap // max(1, n_workers))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(per_worker)


def gridded_epsf_npz_path(output_dir: str, ffi_stem: str) -> str:
    """Path for one frame's gridded PSF archive."""
    return os.path.join(output_dir, f"{ffi_stem}{GRIDDED_EPSF_NPZ_SUFFIX}")


def _ensure_tess_mag_column(gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Add/refresh a ``tess_mag`` column via ``epsf.tess_mag_from_gaia_phot``."""
    from syndiff_pipeline.difference_imaging.stages.epsf import (
        tess_mag_from_gaia_phot,
    )

    df = gaia_df.copy()

    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(df), np.nan)

    g = _col("phot_g_mean_mag")
    bp = _col("phot_bp_mean_mag")
    rp = _col("phot_rp_mean_mag")
    df["tess_mag"] = tess_mag_from_gaia_phot(g, bp, rp)
    return df


def _filter_gaia_for_epsf(
    gaia_df: pd.DataFrame,
    *,
    tess_mag_max: float | None,
    tess_mag_min: float | None = None,
) -> pd.DataFrame:
    """
    Brightness pre-filter for ePSF star catalogs, on derived ``tess_mag``.

    Standing policy (2026-08-22): star selection always filters on
    ``tess_mag`` (see ``_ensure_tess_mag_column``), never raw Gaia
    ``phot_rp_mean_mag``. ``tess_mag_min`` additionally drops stars brighter
    than the given magnitude (e.g. to skip saturated/non-linear stars and
    shrink the per-tile star count for faster ePSF fitting).

    Expects ``ra``/``dec`` (science-array ``x``/``y`` are computed per frame).
    """
    df = _ensure_tess_mag_column(gaia_df)
    mag = pd.to_numeric(df["tess_mag"], errors="coerce")
    keep = pd.Series(True, index=df.index)
    if tess_mag_max is not None:
        keep &= mag < float(tess_mag_max)
    if tess_mag_min is not None:
        keep &= mag > float(tess_mag_min)
    df = df.loc[keep].copy()
    return df.reset_index(drop=True)


def apply_epsf_isolation_filter(
    gaia_df: pd.DataFrame,
    *,
    mag_min: float | None,
    mag_max: float | None,
    min_sep_px: float,
    neighbor_mag_max: float,
    mag_col: str = "tess_mag",
) -> pd.DataFrame:
    """
    Keep ``mag_col`` window rows isolated by >= ``min_sep_px`` from any
    ``mag_col`` neighbor brighter than ``neighbor_mag_max``.

    Direct port of ``dev/forward_epsf_wcs.isolated_forced_phot.
    select_isolated_stars``'s isolation rule (a separate, non-production GPU
    fitter): the candidate window and the neighbor pool are both drawn from
    the SAME full, unfiltered ``gaia_df`` -- a star between ``mag_max`` and
    ``neighbor_mag_max`` isn't itself a candidate but still counts as a
    contaminating neighbor. ``gaia_df`` must already carry per-frame ``x``/
    ``y`` (call after :func:`~syndiff_pipeline.common.wcs_grouping.
    gaia_science_xy_for_frame`, before the tile-section loop -- isolation is
    evaluated globally, not per-tile).
    """
    if len(gaia_df) == 0 or "x" not in gaia_df.columns or mag_col not in gaia_df.columns:
        return gaia_df.iloc[0:0]
    mag = pd.to_numeric(gaia_df[mag_col], errors="coerce").to_numpy(dtype=float)
    x = gaia_df["x"].to_numpy(dtype=float)
    y = gaia_df["y"].to_numpy(dtype=float)
    finite = np.isfinite(mag) & np.isfinite(x) & np.isfinite(y)
    cand = finite.copy()
    if mag_max is not None:
        cand &= mag < float(mag_max)
    if mag_min is not None:
        cand &= mag > float(mag_min)
    neigh = finite & (mag < float(neighbor_mag_max))

    from scipy.spatial import cKDTree

    xy_n = np.column_stack([x[neigh], y[neigh]])
    xy_c = np.column_stack([x[cand], y[cand]])
    if xy_n.shape[0] < 2 or xy_c.shape[0] == 0:
        return gaia_df.iloc[0:0]
    tree = cKDTree(xy_n)
    dist, _ = tree.query(xy_c, k=2)
    nn = dist[:, 1] if dist.ndim == 2 else np.full(int(cand.sum()), np.inf)
    keep_local = nn >= float(min_sep_px)
    cand_idx = np.flatnonzero(cand)[keep_local]
    return gaia_df.iloc[cand_idx].reset_index(drop=True)


def _resolve_tess_mag_max(epsf_params) -> float:
    """
    Bright-end cutoff for ePSF star selection, on derived ``tess_mag``.

    Legacy frozen configs wrote ``tess_mag_max: null`` for experiment B (no
    narrow mag window); treat that as the reference default, not "use all
    Gaia".
    """
    mag_max = getattr(epsf_params, "tess_mag_max", 12.95)
    if mag_max is None:
        return 12.95
    return float(mag_max)


def _resolve_tess_mag_min(epsf_params) -> float | None:
    """Faint-end cutoff for ePSF star selection; ``None`` means no lower bound."""
    mag_min = getattr(epsf_params, "tess_mag_min", None)
    if mag_min is None:
        return None
    return float(mag_min)


def prepare_gaia_for_gridded_epsf(
    gaia_df: pd.DataFrame,
    epsf_params,
) -> pd.DataFrame:
    """
    One-time Gaia table for the frame-parallel ePSF loop.

    Call once in the parent process before ``Parallel`` (see
    ``starpositioningscript.py`` main block). Per-frame workers receive this
    pre-filtered table; section loops only apply spatial cuts. Star
    selection always filters on derived ``tess_mag`` (standing policy,
    2026-08-22), never raw Gaia ``phot_rp_mean_mag``.
    """
    isolation_sep = getattr(epsf_params, "epsf_isolation_min_sep_px", None)
    if isolation_sep is not None:
        # Defer the magnitude-window cut to build_gridded_psf_for_frame's
        # apply_epsf_isolation_filter, which needs the FULL (unfiltered)
        # candidate+neighbor pool together with per-frame x/y to replicate
        # dev/forward_epsf_wcs's select_isolated_stars exactly -- narrowing
        # to the mag window here would silently drop the fainter
        # (tess_mag_max..neighbor_mag_max) neighbors the isolation check
        # needs.
        out = _ensure_tess_mag_column(gaia_df)
        if "ra" not in out.columns or "dec" not in out.columns:
            raise ValueError("Gaia catalog for ePSF requires ra, dec columns")
        log.info(
            "ePSF Gaia catalog: %d stars loaded (isolation filter deferred to "
            "per-frame: tess_mag window (%s, %s), min_sep_px=%s, "
            "neighbor_mag_max=%s)",
            len(out),
            _resolve_tess_mag_min(epsf_params),
            _resolve_tess_mag_max(epsf_params),
            isolation_sep,
            getattr(epsf_params, "epsf_isolation_neighbor_mag_max", 13.0),
        )
        return out
    mag_max = _resolve_tess_mag_max(epsf_params)
    mag_min = _resolve_tess_mag_min(epsf_params)
    out = _filter_gaia_for_epsf(
        gaia_df, tess_mag_max=mag_max, tess_mag_min=mag_min
    )
    if "ra" not in out.columns or "dec" not in out.columns:
        raise ValueError("Gaia catalog for ePSF requires ra, dec columns")
    n = len(out)
    log.info(
        "ePSF Gaia catalog: %d stars after %s < tess_mag < %s pre-filter",
        n,
        mag_min,
        mag_max,
    )
    return out


def _section_bounds(
    ny: int,
    nx: int,
    tile_ny: int,
    tile_nx: int,
    i: int,
    j: int,
) -> tuple[int, int, int, int]:
    """Section pixel bounds matching starpositioningscript grid layout."""
    step_x = nx // tile_nx
    step_y = ny // tile_ny
    x_min = j * step_x
    x_max = (j + 1) * step_x
    y_min = i * step_y
    y_max = (i + 1) * step_y
    return x_min, x_max, y_min, y_max


def _stars_in_section(
    gaia_df: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    margin: float = 0.0,
) -> pd.DataFrame:
    """
    Gaia rows in a grid section.

    Base cut matches ``starpositioningscript.py`` (``>= x_min``, ``< x_max``).
    Optional *margin* excludes edge stars (experiment B: ``extract_size//2 + 2``).
    """
    if margin <= 0.0:
        in_sec = (
            (gaia_df["x"] >= x_min)
            & (gaia_df["x"] < x_max)
            & (gaia_df["y"] >= y_min)
            & (gaia_df["y"] < y_max)
        )
        return gaia_df.loc[in_sec].copy()

    x0 = x_min + margin
    x1 = x_max - margin
    y0 = y_min + margin
    y1 = y_max - margin
    if x1 <= x0 or y1 <= y0:
        return gaia_df.iloc[0:0].copy()
    in_sec = (
        (gaia_df["x"] >= x0)
        & (gaia_df["x"] < x1)
        & (gaia_df["y"] >= y0)
        & (gaia_df["y"] < y1)
    )
    return gaia_df.loc[in_sec].copy()


def _filter_stars_off_mask(
    stars_df: pd.DataFrame,
    mask_2d: np.ndarray,
    *,
    ny: int,
    nx: int,
) -> pd.DataFrame:
    """Drop Gaia rows whose integer pixel positions fall on masked pixels."""
    if len(stars_df) == 0:
        return stars_df
    xi = np.clip(np.round(stars_df["x"].values).astype(int), 0, nx - 1)
    yi = np.clip(np.round(stars_df["y"].values).astype(int), 0, ny - 1)
    good = ~mask_2d[yi, xi]
    return stars_df.loc[good].copy()


def _filter_stars_geometric_mask(
    stars_tbl: Table,
    section_mask: np.ndarray,
    box_radius: int,
) -> Table:
    """Keep stars whose mask cutout has enough unmasked 2D area (reference script)."""
    if len(stars_tbl) == 0:
        return stars_tbl
    mask = np.asarray(section_mask, dtype=bool)
    ny, nx = mask.shape
    valid: list[int] = []
    for idx, row in enumerate(stars_tbl):
        cx = int(round(float(row["x"])))
        cy = int(round(float(row["y"])))
        y0 = max(0, cy - box_radius)
        y1 = min(ny, cy + box_radius + 1)
        x0 = max(0, cx - box_radius)
        x1 = min(nx, cx + box_radius + 1)
        cutout = mask[y0:y1, x0:x1]
        unmasked_y, unmasked_x = np.where(~cutout)
        if (
            len(unmasked_x) >= 10
            and len(np.unique(unmasked_x)) >= 3
            and len(np.unique(unmasked_y)) >= 3
        ):
            valid.append(idx)
    return stars_tbl[valid]


def fit_epsf_section_multi(
    frames: list[tuple[np.ndarray, Table, np.ndarray | None]],
    *,
    extract_size: int,
    oversampling: int,
    maxiters: int,
    recentering_maxiters: int = 20,
    smoothing_kernel: str = "quadratic",
    builder_fit_shape: int = 5,
    recentering_boxsize: int = 3,
    use_mask: bool = False,
    star_box_radius: int = 7,
    star_usage_out: dict | None = None,
) -> np.ndarray | None:
    """
    Fit one grid-section ePSF stamp, pooling stars from one or more frames.

    *frames* is a list of ``(section_data, stars_tbl, section_mask)`` tuples
    (one per input frame); ``photutils.psf.extract_stars`` natively accepts
    matched lists of ``NDData``/star tables and pools the extracted stars
    into a single fit. :func:`fit_epsf_section` is a thin single-frame
    wrapper (list of length 1) kept for existing per-frame call sites.

    When *star_usage_out* is provided (a dict), it is populated with
    ``"used_xy"``/``"excluded_xy"`` lists of section-local ``(x, y)``
    positions -- the exact candidate positions ``extract_stars`` handed to
    ``EPSFBuilder``, split by whether the final fit excluded that star
    (``EPSFBuildResult.excluded_star_indices``). Meaningful only for the
    single-frame case (:func:`fit_epsf_section`); with multiple pooled
    frames, section-local ``(x, y)`` isn't unique across frames.

    Returns oversampled 2D stamp array or None on failure.
    """
    nddatas: list[NDData] = []
    tables: list[Table] = []
    for section_data, stars_tbl, section_mask in frames:
        frame_stars = stars_tbl
        if len(frame_stars) == 0:
            continue
        if section_mask is not None and star_box_radius > 0:
            frame_stars = _filter_stars_geometric_mask(
                frame_stars, section_mask, int(star_box_radius)
            )
        if len(frame_stars) == 0:
            continue
        data = np.asarray(section_data, dtype=np.float64)
        mask = None
        if use_mask and section_mask is not None:
            mask = np.asarray(section_mask, dtype=bool)
            if mask.shape != data.shape:
                mask = None
        nddatas.append(NDData(data=data, mask=mask))
        tables.append(frame_stars)
    if not nddatas:
        return None
    try:
        _suppress_photutils_epsf_noise()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(nddatas) == 1:
                extracted = extract_stars(nddatas[0], tables[0], size=int(extract_size))
            else:
                extracted = extract_stars(nddatas, tables, size=int(extract_size))
            if extracted is None or len(extracted) == 0:
                return None
            builder = EPSFBuilder(
                oversampling=int(oversampling),
                maxiters=int(maxiters),
                recentering_maxiters=int(recentering_maxiters),
                smoothing_kernel=str(smoothing_kernel),
                fit_shape=int(builder_fit_shape),
                recentering_boxsize=int(recentering_boxsize),
                progress_bar=False,
            )
            result = builder(extracted)
            epsf = result.epsf
        if star_usage_out is not None:
            try:
                all_stars = extracted.all_stars
                excluded = set(result.excluded_star_indices)
                star_usage_out["used_xy"] = [
                    (float(s.center[0]), float(s.center[1]))
                    for i, s in enumerate(all_stars)
                    if i not in excluded
                ]
                star_usage_out["excluded_xy"] = [
                    (float(s.center[0]), float(s.center[1]))
                    for i, s in enumerate(all_stars)
                    if i in excluded
                ]
            except Exception:
                log.debug("fit_epsf_section_multi: star_usage_out capture failed", exc_info=True)
        stamp = np.asarray(epsf.data, dtype=np.float64)
        if not np.all(np.isfinite(stamp)):
            return None
        return stamp
    except Exception as exc:
        log.debug("fit_epsf_section_multi failed: %s", exc)
        return None


def fit_epsf_section(
    section_data: np.ndarray,
    stars_tbl: Table,
    *,
    extract_size: int,
    oversampling: int,
    maxiters: int,
    recentering_maxiters: int = 20,
    smoothing_kernel: str = "quadratic",
    builder_fit_shape: int = 5,
    recentering_boxsize: int = 3,
    section_mask: np.ndarray | None = None,
    use_mask: bool = False,
    star_box_radius: int = 7,
    star_usage_out: dict | None = None,
) -> np.ndarray | None:
    """
    Fit one grid-section ePSF stamp with photutils (single frame).

    Returns oversampled 2D stamp array or None on failure. See
    :func:`fit_epsf_section_multi` for *star_usage_out*.
    """
    if len(stars_tbl) == 0:
        return None
    return fit_epsf_section_multi(
        [(section_data, stars_tbl, section_mask)],
        extract_size=extract_size,
        oversampling=oversampling,
        maxiters=maxiters,
        recentering_maxiters=recentering_maxiters,
        smoothing_kernel=smoothing_kernel,
        builder_fit_shape=builder_fit_shape,
        recentering_boxsize=recentering_boxsize,
        use_mask=use_mask,
        star_box_radius=star_box_radius,
        star_usage_out=star_usage_out,
    )


def _tile_centers_from_shape(
    ny: int, nx: int, tile_ny: int, tile_nx: int
) -> list[tuple[float, float]]:
    """Grid-node centers in crop-local pixels (matches section layout)."""
    centers: list[tuple[float, float]] = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            x_center = j * (nx / tile_nx) + (nx / (2 * tile_nx))
            y_center = i * (ny / tile_ny) + (ny / (2 * tile_ny))
            centers.append((float(x_center), float(y_center)))
    return centers


def build_gridded_psf_for_frame(
    diff_image: np.ndarray,
    gaia_df: pd.DataFrame,
    epsf_params,
    *,
    mask_2d: np.ndarray | None = None,
    frame_label: str = "",
    star_usage_out: dict | None = None,
) -> tuple[GriddedPSFModel | None, list[tuple[float, float]], np.ndarray | None, list[int]]:
    """
    Build a spatially varying PSF model for one difference image.

    Algorithm follows ``starpositioningscript.py`` ``processing()``:
    section grid → per-section ``extract_stars`` + ``EPSFBuilder`` → mean
    fallback for failed sections → ``GriddedPSFModel``.

    *gaia_df* must already be prepared via :func:`prepare_gaia_for_gridded_epsf`.

    When *star_usage_out* is provided (a dict), it is populated with
    ``"used_xy"``/``"excluded_xy"`` lists of frame-local ``(x, y)`` --
    pooled across every tile that reached ``EPSFBuilder`` (see
    :func:`fit_epsf_section`'s own *star_usage_out*) -- for the debug-plot
    star-selection overlay. Only covers this single-frame fitting path, not
    :func:`fit_anchor_pooled`'s multi-frame pooling.

    Returns ``(model, grid_xypos, stack, n_stars_per_tile)`` -- the last
    element is the post-mask candidate star count per tile (same order as
    ``grid_xypos``/``stack``), for debug-plot annotation; a fallback-filled
    tile still reports its own (too-few) count, not the fallback source's.
    """
    ny, nx = diff_image.shape
    isolation_sep = getattr(epsf_params, "epsf_isolation_min_sep_px", None)
    if isolation_sep is not None:
        # Global (whole-frame) isolation pass, not per-tile -- mirrors
        # dev/forward_epsf_wcs.isolated_forced_phot.select_isolated_stars.
        # gaia_df here still carries the FULL candidate+neighbor pool
        # (prepare_gaia_for_gridded_epsf deferred the mag-window cut for
        # exactly this reason).
        n_before = len(gaia_df)
        gaia_df = apply_epsf_isolation_filter(
            gaia_df,
            mag_min=_resolve_tess_mag_min(epsf_params),
            mag_max=_resolve_tess_mag_max(epsf_params),
            min_sep_px=float(isolation_sep),
            neighbor_mag_max=float(
                getattr(epsf_params, "epsf_isolation_neighbor_mag_max", 13.0)
            ),
            mag_col="tess_mag",
        )
        log.debug(
            "ePSF isolation filter (%s): %d -> %d candidate stars "
            "(min_sep_px=%s, neighbor_mag_max=%s)",
            frame_label or "frame",
            n_before,
            len(gaia_df),
            isolation_sep,
            getattr(epsf_params, "epsf_isolation_neighbor_mag_max", 13.0),
        )
    tile_ny = int(epsf_params.tile_ny)
    tile_nx = int(epsf_params.tile_nx)
    oversampling = int(epsf_params.epsf_oversample)
    min_stars = int(getattr(epsf_params, "min_stars_per_tile", 5))
    maxiters = int(getattr(epsf_params, "epsf_maxiters", 15))
    recentering_maxiters = int(getattr(epsf_params, "epsf_recentering_maxiters", 20))
    extract_size = int(
        getattr(epsf_params, "extract_size", None) or epsf_params.psf_size
    )
    smoothing_kernel = str(getattr(epsf_params, "epsf_smoothing_kernel", "quadratic"))
    builder_fit_shape = int(getattr(epsf_params, "epsf_builder_fit_shape", 5))
    recentering_boxsize = int(getattr(epsf_params, "epsf_recentering_boxsize", 3))
    star_box_radius = int(getattr(epsf_params, "epsf_star_box_radius", 7))
    use_section_mask = bool(getattr(epsf_params, "epsf_use_section_mask", True))
    stamp_border_crop = int(getattr(epsf_params, "epsf_stamp_border_crop", 0))
    # experiment B section edge margin: extract_size // 2 + 2
    star_margin = float(extract_size) / 2.0 + 2.0

    step_x = nx // tile_nx
    step_y = ny // tile_ny
    if step_x < extract_size or step_y < extract_size:
        log.warning(
            "ePSF grid cells smaller than extract_size=%s for %s",
            extract_size,
            frame_label or "frame",
        )

    epsf_grid: dict[tuple[int, int], np.ndarray | str] = {}
    grid_xypos: list[tuple[float, float]] = []
    n_stars_grid: dict[tuple[int, int], int] = {}
    used_xy_frame: list[tuple[float, float]] = []
    excluded_xy_frame: list[tuple[float, float]] = []

    for i in range(tile_ny):
        for j in range(tile_nx):
            x_min, x_max, y_min, y_max = _section_bounds(ny, nx, tile_ny, tile_nx, i, j)

            x_center = j * (nx / tile_nx) + (nx / (2 * tile_nx))
            y_center = i * (ny / tile_ny) + (ny / (2 * tile_ny))
            grid_xypos.append((float(x_center), float(y_center)))

            section = diff_image[y_min:y_max, x_min:x_max]
            sec_stars = _stars_in_section(
                gaia_df, x_min, x_max, y_min, y_max, margin=star_margin
            )
            section_mask = None
            if mask_2d is not None:
                section_mask = np.asarray(mask_2d[y_min:y_max, x_min:x_max], dtype=bool)
                sec_stars = _filter_stars_off_mask(sec_stars, mask_2d, ny=ny, nx=nx)
            n_stars_grid[(i, j)] = len(sec_stars)
            if len(sec_stars) < min_stars:
                epsf_grid[(i, j)] = "too_few"
                continue

            stars_tbl = Table()
            stars_tbl["x"] = np.asarray(sec_stars["x"].values - x_min, dtype=float)
            stars_tbl["y"] = np.asarray(sec_stars["y"].values - y_min, dtype=float)

            tile_usage: dict | None = {} if star_usage_out is not None else None
            stamp = fit_epsf_section(
                np.asarray(section, dtype=np.float64),
                stars_tbl,
                extract_size=extract_size,
                oversampling=oversampling,
                maxiters=maxiters,
                recentering_maxiters=recentering_maxiters,
                smoothing_kernel=smoothing_kernel,
                builder_fit_shape=builder_fit_shape,
                recentering_boxsize=recentering_boxsize,
                section_mask=section_mask,
                use_mask=use_section_mask and section_mask is not None,
                star_box_radius=star_box_radius,
                star_usage_out=tile_usage,
            )
            if tile_usage:
                used_xy_frame.extend(
                    (x + x_min, y + y_min) for x, y in tile_usage.get("used_xy", [])
                )
                excluded_xy_frame.extend(
                    (x + x_min, y + y_min) for x, y in tile_usage.get("excluded_xy", [])
                )
            if stamp is None:
                epsf_grid[(i, j)] = "fit_failed"
            else:
                epsf_grid[(i, j)] = stamp

    if star_usage_out is not None:
        star_usage_out["used_xy"] = used_xy_frame
        star_usage_out["excluded_xy"] = excluded_xy_frame

    valid = [v for v in epsf_grid.values() if isinstance(v, np.ndarray)]
    n_stars_list = [
        n_stars_grid.get((i, j), 0) for i in range(tile_ny) for j in range(tile_nx)
    ]
    if not valid:
        suffix = f" ({frame_label})" if frame_label else ""
        log.warning("ePSF: all grid sections failed%s", suffix)
        return None, grid_xypos, None, n_stars_list

    fallback = np.mean(valid, axis=0)
    psf_list: list[np.ndarray] = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            result = epsf_grid.get((i, j), "too_few")
            if isinstance(result, np.ndarray):
                psf_list.append(result)
            else:
                psf_list.append(fallback)

    if stamp_border_crop > 0:
        cropped: list[np.ndarray] = []
        for arr in psf_list:
            bc = int(stamp_border_crop)
            if bc * 2 >= arr.shape[0] or bc * 2 >= arr.shape[1]:
                cropped.append(arr)
            else:
                cropped.append(arr[bc:-bc, bc:-bc])
        psf_list = cropped

    stack = np.array(psf_list, dtype=np.float64)
    meta = {"grid_xypos": grid_xypos, "oversampling": oversampling}
    nddata_grid = NDData(data=stack, meta=meta)
    model = GriddedPSFModel(nddata_grid)
    return model, grid_xypos, stack, n_stars_list


def save_gridded_epsf_npz(
    path: str,
    stack: np.ndarray,
    grid_xypos: list[tuple[float, float]],
    oversampling: int,
    n_stars: list[int] | None = None,
) -> str:
    """
    Write one frame's gridded PSF cube.

    ``n_stars`` (optional, same order as ``grid_xypos``) is the per-tile
    candidate star count for debug-plot annotation -- omitted for
    interpolated/blended frames, which have no fit of their own to count
    stars for (see :func:`write_gridded_epsf_frame_plot`).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    xy = np.asarray(grid_xypos, dtype=np.float64)
    kwargs = {}
    if n_stars is not None:
        kwargs["n_stars"] = np.asarray(n_stars, dtype=np.int64)
    np.savez_compressed(
        path,
        data=np.asarray(stack, dtype=np.float64),
        grid_xypos=xy,
        oversampling=int(oversampling),
        **kwargs,
    )
    return path


def load_gridded_psf_model(path: str) -> GriddedPSFModel:
    """Load a per-frame :class:`GriddedPSFModel` from npz."""
    z = np.load(path, allow_pickle=False)
    try:
        stack = np.asarray(z["data"], dtype=np.float64)
        grid_xypos = [tuple(row) for row in np.asarray(z["grid_xypos"])]
        oversampling = int(z["oversampling"])
    finally:
        z.close()
    meta = {"grid_xypos": grid_xypos, "oversampling": oversampling}
    return GriddedPSFModel(NDData(data=stack, meta=meta))


def save_gridded_epsf_index(output_dir: str, index: dict[str, str]) -> str:
    """Persist ffi_stem → npz path mapping."""
    path = os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    return path


def register_gridded_epsf_index_entry(output_dir: str, ffi_stem: str) -> None:
    """Append one frame to ``gridded_epsf_index.json`` (worker-safe, file-locked)."""
    import fcntl

    index_path = os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME)
    os.makedirs(output_dir, exist_ok=True)
    npz_path = gridded_epsf_npz_path(output_dir, ffi_stem)
    with open(index_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read()
            if raw.strip():
                index = json.loads(raw)
            else:
                index = {}
            index[str(ffi_stem)] = npz_path
            fh.seek(0)
            fh.truncate(0)
            json.dump(index, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_gridded_epsf_index(output_dir: str) -> dict[str, str]:
    """Load ffi_stem → npz path mapping; empty dict if missing."""
    path = os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k): str(v) for k, v in raw.items()}


def save_gridded_epsf_anchor_stems(output_dir: str, anchor_stems) -> str:
    """Persist the set of ffi_stems that were directly fit (orbit_binned mode).

    Lets debug-plot selection prefer real fits over interpolated/blended
    frames, which have no fit of their own and so no per-tile star count
    (see :func:`syndiff_pipeline.difference_imaging.support.plot.
    write_gridded_epsf_workspace_plots`). Absent for per_frame mode, where
    every frame is a real fit and no preference is needed.
    """
    path = os.path.join(output_dir, GRIDDED_EPSF_ANCHORS_BASENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sorted(str(s) for s in anchor_stems), fh, indent=2)
    return path


def load_gridded_epsf_anchor_stems(output_dir: str) -> set[str]:
    """Load the anchor-stem set written by :func:`save_gridded_epsf_anchor_stems`; empty if missing."""
    path = os.path.join(output_dir, GRIDDED_EPSF_ANCHORS_BASENAME)
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(s) for s in raw}


def workspace_has_gridded_epsf(output_dir: str) -> bool:
    """True when the workspace contains a gridded ePSF index."""
    return os.path.isfile(os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME))


@dataclass(frozen=True)
class GriddedEpsfCatalog:
    """Per-frame gridded PSF lookup for forced photometry."""

    workspace_dir: str
    index: dict[str, str]

    def path_for_stem(self, ffi_stem: str) -> str | None:
        """Return npz path for a tess product stem, if present."""
        p = self.index.get(ffi_stem)
        if p and os.path.isfile(p):
            return p
        alt = gridded_epsf_npz_path(self.workspace_dir, ffi_stem)
        if os.path.isfile(alt):
            return alt
        return None

    def load_model(self, ffi_stem: str) -> GriddedPSFModel | None:
        """Load gridded model for one frame."""
        path = self.path_for_stem(ffi_stem)
        if path is None:
            return None
        return load_gridded_psf_model(path)


def catalog_from_workspace(output_dir: str) -> GriddedEpsfCatalog | None:
    """Build catalog from workspace dir; None if no gridded index."""
    if not workspace_has_gridded_epsf(output_dir):
        return None
    return GriddedEpsfCatalog(
        workspace_dir=output_dir,
        index=load_gridded_epsf_index(output_dir),
    )


def stack_from_gridded_cube(stack: np.ndarray) -> np.ndarray:
    """Flatten (n_grid, ny, nx) cube to (n_grid, ny*nx) for legacy tile stacks."""
    n_grid = stack.shape[0]
    return stack.reshape(n_grid, -1)


def _diff_path_to_stem(diff_path: str) -> str:
    stem = strip_fits_suffix(Path(str(diff_path)).name)
    parsed = parse_workspace_frame_stem(stem)
    if parsed is not None:
        return parsed[0]
    return tess_product_id_from_ffi_path(stem) or stem


def ffi_path_by_stem_from_wcs_table(wcs_table: pd.DataFrame | None) -> dict[str, str]:
    """Map TESS product id → science FFI path from a frame manifest."""
    if wcs_table is None or len(wcs_table) == 0:
        return {}
    if "path" in wcs_table.columns:
        col = "path"
    elif "filename" in wcs_table.columns:
        col = "filename"
    else:
        return {}
    out: dict[str, str] = {}
    for _, row in wcs_table.iterrows():
        ffi_path = str(row[col])
        stem = (
            str(row["product_id"])
            if "product_id" in wcs_table.columns and pd.notna(row.get("product_id"))
            else tess_product_id_from_ffi_path(ffi_path) or ""
        )
        if stem:
            out[stem] = ffi_path
    return out


def _is_valid_gridded_epsf_npz(path: str) -> bool:
    """True when *path* is a readable gridded ePSF per-frame archive."""
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(k in z.files for k in ("data", "grid_xypos", "oversampling"))
    except (OSError, ValueError):
        return False


def _load_gridded_epsf_stack(path: str) -> np.ndarray | None:
    """Load the PSF cube from a per-frame npz, or ``None`` if unreadable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return np.asarray(z["data"], dtype=np.float64)
    except (OSError, ValueError):
        return None


def _upstream_diff_image_stage(cfg, diffs_input: str | None) -> dict | None:
    """Hotpants/background_estimate stage that produced diffs feeding ePSF."""
    pipeline = getattr(cfg, "pipeline", None)
    if not pipeline:
        return None
    from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import (
        split_pipeline,
    )

    _, _, stages = split_pipeline(pipeline)
    diffs_label = (diffs_input or "").strip() or None
    if diffs_label:
        for _, st in stages:
            if st.get("kind") not in ("hotpants", "background_estimate"):
                continue
            o = st.get("output") or {}
            if str(o.get("diffs", "")).strip() == diffs_label:
                return st
    for _, st in stages:
        if st.get("kind") in ("hotpants", "background_estimate"):
            return st
    return None


def _diff_image_stage_params(stage: dict):
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
        parse_hotpants,
        parse_background_estimate,
    )

    kind = stage.get("kind")
    try:
        if kind == "hotpants":
            return parse_hotpants(stage, 0)
        if kind == "background_estimate":
            return parse_background_estimate(stage, 0)
    except Exception:
        log.debug("diff_image stage param reparse failed", exc_info=True)
    return None


def _diff_image_stage_label(stage: dict) -> str | None:
    o = stage.get("output") or {}
    label = str(o.get("diffs", "")).strip()
    return label or None


def _load_frames_df_for_provenance(cfg) -> pd.DataFrame | None:
    output_dir = getattr(cfg, "output_dir", "") or ""
    if not output_dir:
        return None
    try:
        from syndiff_pipeline.difference_imaging.support.manifest import (
            load_frame_manifest,
        )

        manifest_path = getattr(cfg, "manifest", "") or None
        return load_frame_manifest(output_dir, manifest_path or None)
    except Exception:
        log.debug("load_frame_manifest for epsf provenance failed", exc_info=True)
        return None


def _diff_image_fp_for_product(
    *,
    sector: int,
    camera: int,
    ccd: int,
    product_id: str,
    cfg,
    diff_stage: dict,
    frames_df: pd.DataFrame | None,
    downsample_fp: str | None,
) -> str | None:
    """Reconstruct one ``diff_image`` fingerprint (real edges, no ``loc:``)."""
    params = _diff_image_stage_params(diff_stage)
    label = _diff_image_stage_label(diff_stage)
    if params is None or not label:
        return None
    ffi_dir = getattr(cfg, "ffi_dir", "") or ""
    ffi_path = provenance_glue.resolve_ffi_path_for_product_id(
        frames_df, product_id, ffi_dir=ffi_dir or None
    )
    if not ffi_path:
        return None
    ds_fp = downsample_fp
    if ds_fp is None:
        ds_fp = provenance_glue.resolve_downsample_fingerprint_from_cfg(cfg)
    if ds_fp is None:
        return None
    inputs = provenance_glue.diff_image_input_fingerprints(
        sector=sector,
        camera=camera,
        ccd=ccd,
        ffi_path=ffi_path,
        downsample_fp=ds_fp,
    )
    if inputs is None:
        return None
    return provenance_glue.diff_kind_fingerprint(
        "diff_image",
        sector=sector,
        camera=camera,
        ccd=ccd,
        product_id=product_id,
        label=label,
        params=params,
        input_fingerprints=inputs,
    )


def build_diff_image_fps(
    cfg,
    diff_paths: list[str],
    *,
    diffs_input: str | None,
    sck: tuple | None,
) -> dict[str, str]:
    if sck is None:
        return {}
    diff_stage = _upstream_diff_image_stage(cfg, diffs_input)
    if diff_stage is None:
        return {}
    frames_df = _load_frames_df_for_provenance(cfg)
    if frames_df is None:
        return {}
    downsample_fp = provenance_glue.resolve_downsample_fingerprint_from_cfg(cfg)
    sector, camera, ccd = int(sck[0]), int(sck[1]), int(sck[2])
    out: dict[str, str] = {}
    for diff_path in diff_paths:
        if not diff_path:
            continue
        stem = _diff_path_to_stem(diff_path)
        pid = tess_product_id_from_ffi_path(stem) or stem
        fp = _diff_image_fp_for_product(
            sector=sector,
            camera=camera,
            ccd=ccd,
            product_id=pid,
            cfg=cfg,
            diff_stage=diff_stage,
            frames_df=frames_df,
            downsample_fp=downsample_fp,
        )
        if fp is not None:
            out[pid] = fp
    return out


def _upstream_epsf_stage(cfg, epsf_label: str | None) -> dict | None:
    """The ``epsf`` stage whose output feeds a downstream consumer (e.g. centroids)."""
    pipeline = getattr(cfg, "pipeline", None)
    if not pipeline:
        return None
    from syndiff_pipeline.difference_imaging.orchestration.pipeline_entries import (
        split_pipeline,
    )

    _, _, stages = split_pipeline(pipeline)
    label = (epsf_label or "").strip() or None
    if label:
        for _, st in stages:
            if st.get("kind") == "epsf" and str(st.get("output", "")).strip() == label:
                return st
    for _, st in stages:
        if st.get("kind") == "epsf":
            return st
    return None


def _epsf_fp_for_product(
    *,
    sector: int,
    camera: int,
    ccd: int,
    product_id: str,
    epsf_stage: dict,
    diff_image_fp: str | None,
) -> str | None:
    """Reconstruct one ``epsf`` fingerprint from its stage params + upstream diff_image_fp."""
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import parse_epsf

    label = str(epsf_stage.get("output", "")).strip()
    if not label:
        return None
    try:
        params = parse_epsf(epsf_stage, 0)
    except Exception:
        log.debug("epsf stage param reparse failed", exc_info=True)
        return None
    inputs = provenance_glue.epsf_input_fingerprints(diff_image_fp)
    if inputs is None:
        return None
    return provenance_glue.diff_kind_fingerprint(
        "epsf",
        sector=sector,
        camera=camera,
        ccd=ccd,
        product_id=product_id,
        label=label,
        params=params,
        input_fingerprints=inputs,
    )


def build_epsf_fps(
    cfg,
    diff_paths: list[str],
    *,
    epsf_label: str | None,
    sck: tuple | None,
    diff_image_fps: dict[str, str] | None = None,
) -> dict[str, str]:
    """Reconstruct ``{product_id: epsf_fp}`` for a downstream stage's bookkeeper skip-check."""
    if sck is None:
        return {}
    epsf_stage = _upstream_epsf_stage(cfg, epsf_label)
    if epsf_stage is None:
        return {}
    sector, camera, ccd = int(sck[0]), int(sck[1]), int(sck[2])
    diff_image_fps = diff_image_fps or {}
    out: dict[str, str] = {}
    for diff_path in diff_paths:
        if not diff_path:
            continue
        stem = _diff_path_to_stem(diff_path)
        pid = tess_product_id_from_ffi_path(stem) or stem
        fp = _epsf_fp_for_product(
            sector=sector,
            camera=camera,
            ccd=ccd,
            product_id=pid,
            epsf_stage=epsf_stage,
            diff_image_fp=diff_image_fps.get(pid),
        )
        if fp is not None:
            out[pid] = fp
    return out


def _fit_one_frame_task(
    frame_idx: int,
    diff_path: str,
) -> tuple[int, str, bool, list[tuple[float, float]] | None, np.ndarray | None, bool, str | None]:
    """Worker: fit one frame and write npz.

    Shared inputs come from :func:`_init_gridded_epsf_worker` (not pickled per frame).

    Returns
    -------
    frame_idx, ffi_stem, ok, grid_xypos, stack, skipped_existing, npz_path
        ``npz_path`` is the actual on-disk location of the valid npz (SCC-lane
        or workspace-local) when ``ok`` is True, else None.
    """
    ctx = _WORKER_CTX
    gaia_base = ctx["gaia_df"]
    epsf_params = ctx["epsf_params"]
    output_dir = ctx["output_dir"]

    ffi_stem = _diff_path_to_stem(diff_path) if diff_path else f"frame_{frame_idx}"
    product_id = tess_product_id_from_ffi_path(ffi_stem) or ffi_stem
    ffi_path_by_stem = ctx.get("ffi_path_by_stem") or {}
    ffi_path = ffi_path_by_stem.get(ffi_stem) or ffi_path_by_stem.get(product_id)
    ffi_list_df = ctx.get("ffi_list_df")
    science_bounds = ctx.get("science_bounds")
    if ffi_path is None or ffi_list_df is None or science_bounds is None:
        log.warning(
            "  ePSF: missing ffi_list/science_bounds for %s", ffi_stem
        )
        return frame_idx, ffi_stem, False, None, None, False, None
    from syndiff_pipeline.common.wcs_grouping import gaia_science_xy_for_frame

    try:
        gaia_df = gaia_science_xy_for_frame(
            gaia_base, ffi_path, ffi_list_df, science_bounds
        )
    except Exception as exc:
        log.warning("  ePSF: Gaia projection failed for %s: %s", ffi_stem, exc)
        return frame_idx, ffi_stem, False, None, None, False, None

    mask_2d = _resolve_epsf_frame_mask(ctx, ffi_stem)
    ws_out_path = gridded_epsf_npz_path(output_dir, ffi_stem)
    epsf_label = str(ctx.get("epsf_label") or "epsf")
    data_root = ctx.get("data_root")
    output_store_name = ctx.get("output_store_name")
    from syndiff_pipeline.difference_imaging.orchestration.diff_store import (
        resolve_diff_write_path,
    )

    sck = ctx.get("sck")
    if data_root and sck is not None:
        write_path = resolve_diff_write_path(
            data_root=data_root,
            sck=sck,
            kind="epsf",
            stage_label=epsf_label,
            ffi_stem=ffi_stem,
            label=epsf_label,
            params=epsf_params,
            output_store_name=output_store_name,
            suffix=".npz",
        )
        scc_primary = True
    else:
        raise RuntimeError(
            "SCC-only gridded_epsf requires deployment data_root and sector/camera/ccd"
        )
    if ctx.get("skip_existing", True):
        if _is_valid_gridded_epsf_npz(write_path):
            return frame_idx, ffi_stem, True, None, None, True, str(write_path)
        if sck is not None and data_root:
            diff_image_fp = (ctx.get("diff_image_fps") or {}).get(product_id)
            inputs = provenance_glue.epsf_input_fingerprints(diff_image_fp)
            prov_complete = None
            if inputs is not None:
                try:
                    prov_complete = provenance_glue.artifact_complete_in_store(
                        kind="epsf",
                        sector=sck[0],
                        camera=sck[1],
                        ccd=sck[2],
                        product_id=product_id,
                        label=epsf_label,
                        params=epsf_params,
                        input_fingerprints=inputs,
                        data_root=data_root,
                    )
                except Exception:
                    log.debug(
                        "provenance resume check (epsf) failed for %s", ffi_stem, exc_info=True
                    )
            if prov_complete is True:
                hit_path: str | None = None
                if _is_valid_gridded_epsf_npz(write_path):
                    hit_path = str(write_path)
                if hit_path is not None:
                    return frame_idx, ffi_stem, True, None, None, True, hit_path
                # Indexed complete but no locatable file — fall through to process.
    if diff_path is None or not os.path.exists(diff_path):
        log.warning("  diff frame missing: %s", diff_path)
        return frame_idx, ffi_stem, False, None, None, False, None
    try:
        diff_img = fits.getdata(diff_path).astype(np.float64)
    except Exception as exc:
        log.warning("  Cannot load %s: %s", diff_path, exc)
        return frame_idx, ffi_stem, False, None, None, False, None

    model, grid_xypos, stack, n_stars = build_gridded_psf_for_frame(
        diff_img,
        gaia_df,
        epsf_params,
        mask_2d=mask_2d,
        frame_label=os.path.basename(diff_path),
    )
    if model is None or stack is None:
        return frame_idx, ffi_stem, False, grid_xypos, None, False, None

    save_gridded_epsf_npz(
        write_path,
        stack,
        grid_xypos,
        int(epsf_params.epsf_oversample),
        n_stars=n_stars,
    )

    if sck is not None:
        try:
            diff_image_fp = (ctx.get("diff_image_fps") or {}).get(product_id)
            inputs = provenance_glue.epsf_input_fingerprints(
                diff_image_fp,
                diff_image_path=diff_path,
            )
            if inputs is not None:
                provenance_glue.emit_diff_artifact(
                    kind="epsf",
                    sector=sck[0],
                    camera=sck[1],
                    ccd=sck[2],
                    product_id=product_id,
                    label=epsf_label,
                    params=epsf_params,
                    location=write_path,
                    input_fingerprints=inputs,
                    data_root=data_root,
                    is_fits=False,
                    scc_primary=scc_primary,
                    workspace_root=ctx.get("workspace_root"),
                    output_store_name=output_store_name,
                )
        except Exception:
            log.debug("provenance emit (epsf) failed for %s", ffi_stem, exc_info=True)

    return frame_idx, ffi_stem, True, grid_xypos, stack, False, str(write_path)


def fit_gridded_epsf_all_frames(
    diff_paths: list[str],
    gaia_df: pd.DataFrame,
    cfg,
    epsf_params,
    output_dir: str,
    *,
    mask_2d: np.ndarray | None = None,
    mask_catalog=None,
    btjd_by_stem: dict | None = None,
    round_id: int = 1,
    diff_log_path: str | None = None,
    epsf_label: str | None = None,
    diffs_input: str | None = None,
    skip_existing: bool = True,
    workspace_root: str | None = None,
    ffi_list_df: pd.DataFrame | None = None,
    science_bounds: dict | None = None,
    ffi_path_by_stem: dict[str, str] | None = None,
    wcs_table: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[str], list[bool]]:
    """
    Fit gridded ePSF on every difference image (thread-parallel over frames).

    When ``mask_catalog`` is provided, each frame resolves an ePSF reject mask
    via in-memory ``mask_at(btjd)`` (no static-mask FITS I/O per frame).
    ``mask_2d`` remains a static fallback when no catalog is available.

    Returns
    -------
    epsf_stack : ndarray (n_frames, n_tiles, n_pix) — legacy flat tile layout
    tile_centers : list of (cx, cy)
    ffi_stems : list of str
    epsf_ok : list of bool
    """
    n_frames = len(diff_paths)
    epsf_n_jobs = getattr(epsf_params, "epsf_n_jobs", None)
    n_workers = resolve_effective_n_jobs(
        int(getattr(cfg, "n_jobs", 1) or 1),
        stage_n_jobs=epsf_n_jobs,
    )
    _configure_blas_threads(n_workers)

    from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
        init_progress_pair,
        progress_path_for_diff_log,
        progress_path_for_output_workspace,
        refresh_progress_pair_from_artifacts,
        set_progress_phase_pair,
    )

    gaia_df = prepare_gaia_for_gridded_epsf(gaia_df, epsf_params)
    os.makedirs(output_dir, exist_ok=True)
    if ffi_path_by_stem is None and wcs_table is not None:
        ffi_path_by_stem = ffi_path_by_stem_from_wcs_table(wcs_table)

    track_progress = epsf_label is not None
    cli_progress_path = (
        str(progress_path_for_diff_log(diff_log_path))
        if track_progress and diff_log_path is not None
        else None
    )
    workspace_progress_path: str | None = None
    if track_progress:
        workspace_progress_path = str(progress_path_for_output_workspace(output_dir))
        init_progress_pair(
            workspace_progress_path,
            cli_progress_path,
            epsf_label=str(epsf_label),
            diffs_input=str(diffs_input or "?"),
            round_id=round_id,
            frames_total=n_frames,
            output_dir=output_dir,
        )
        refresh_progress_pair_from_artifacts(
            workspace_progress_path, cli_progress_path
        )

    tasks = [(i, p) for i, p in enumerate(diff_paths)]

    tqdm_desc = f"epsf {epsf_label}" if track_progress else "ePSF gridded"

    def _on_frame_done(result: tuple) -> None:
        if not track_progress or workspace_progress_path is None:
            return
        if result[5]:
            return
        from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
            record_frame_progress,
        )

        _ok = bool(result[2])
        record_frame_progress(
            workspace_progress_path, cli_progress_path, success=_ok
        )

    try:
        prov_sck = (int(cfg.sector), int(cfg.camera), int(cfg.ccd))
    except Exception:
        prov_sck = None
    prov_data_root = getattr(cfg, "data_root", "") or None
    prov_output_store_name = getattr(cfg, "output_store_name", None) or None
    if workspace_root is None:
        from syndiff_pipeline.difference_imaging.support.paths import workspace_root as _workspace_root

        prov_workspace_root = _workspace_root(
            cfg.output_dir, run_id=getattr(cfg, "workspace_run_id", None)
        )
    else:
        prov_workspace_root = workspace_root

    diff_image_fps = build_diff_image_fps(
        cfg,
        diff_paths,
        diffs_input=diffs_input,
        sck=prov_sck,
    )

    worker_initargs = (
        gaia_df,
        epsf_params,
        output_dir,
        mask_2d,
        skip_existing,
        prov_sck,
        prov_data_root,
        epsf_label,
        prov_workspace_root,
        prov_output_store_name,
        mask_catalog,
        btjd_by_stem or {},
        diff_image_fps,
        ffi_list_df,
        science_bounds,
        ffi_path_by_stem or {},
    )
    results: list[tuple] = []
    if n_workers <= 1 or n_frames <= 1:
        _init_gridded_epsf_worker(*worker_initargs)
        log.info(
            "ePSF [%s] round %s: starting %d frames (n_jobs=1)",
            epsf_label or "?",
            round_id,
            n_frames,
        )
        for t in tqdm_iter(tasks, desc=tqdm_desc):
            result = _fit_one_frame_task(*t)
            _on_frame_done(result)
            results.append(result)
    else:
        log.info(
            "ePSF [%s] round %s: starting %d frames (n_jobs=%s, backend=loky)",
            epsf_label or "?",
            round_id,
            n_frames,
            n_workers,
        )
        delayed_calls = [delayed(_fit_one_frame_task)(i, p) for i, p in tasks]
        # Parent updates progress when each result is yielded (no worker NFS locks).
        results = parallel_map_with_optional_tqdm(
            delayed_calls,
            n_tasks=n_frames,
            desc=tqdm_desc,
            n_jobs_eff=n_workers,
            initializer=_init_gridded_epsf_worker,
            initargs=worker_initargs,
            on_result=_on_frame_done,
        )

    if track_progress and workspace_progress_path is not None:
        refresh_progress_pair_from_artifacts(
            workspace_progress_path, cli_progress_path
        )
        set_progress_phase_pair(workspace_progress_path, cli_progress_path, "complete")

    results.sort(key=lambda r: r[0])
    ffi_stems = [r[1] for r in results]
    epsf_ok = [r[2] for r in results]
    tile_centers: list[tuple[float, float]] | None = None
    index: dict[str, str] = {}

    stacks: list[np.ndarray | None] = []
    for _idx, stem, ok, centers, stack, _skipped, npz_path in results:
        if ok:
            path = npz_path or gridded_epsf_npz_path(output_dir, stem)
            index[stem] = path
            if stack is None:
                stack = _load_gridded_epsf_stack(path)
        if ok and stack is not None:
            stacks.append(stack)
            if tile_centers is None and centers is not None:
                tile_centers = centers
        else:
            stacks.append(None)

    if tile_centers is None:
        first_path = next((p for p in diff_paths if p and os.path.exists(p)), None)
        if mask_2d is not None:
            ny, nx = mask_2d.shape
        elif first_path is not None:
            ny, nx = fits.getdata(first_path).shape
        else:
            ny, nx = 1024, 1024
        tile_centers = _tile_centers_from_shape(
            ny, nx, int(epsf_params.tile_ny), int(epsf_params.tile_nx)
        )

    save_gridded_epsf_index(output_dir, index)

    n_tiles = epsf_params.tile_ny * epsf_params.tile_nx
    n_pix = 0
    for s in stacks:
        if s is not None:
            n_pix = s.reshape(s.shape[0], -1).shape[1]
            break
    if n_pix == 0:
        n_pix = (2 * epsf_params.psf_size + 1) ** 2

    epsf_stack = np.full((n_frames, n_tiles, n_pix), np.nan)
    good_rows = [stack_from_gridded_cube(s) for s in stacks if s is not None]
    med_row = np.nanmedian(np.stack(good_rows), axis=0) if good_rows else None

    for i, stack in enumerate(stacks):
        if stack is not None:
            epsf_stack[i] = stack_from_gridded_cube(stack)
        elif med_row is not None:
            epsf_stack[i] = med_row

    n_ok = sum(epsf_ok)
    n_skipped = sum(1 for r in results if r[5])
    if epsf_label:
        if n_skipped:
            log.info(
                "ePSF [%s] round %s: %d/%d frames succeeded (%d skipped existing)",
                epsf_label,
                round_id,
                n_ok,
                n_frames,
                n_skipped,
            )
        else:
            log.info(
                "ePSF [%s] round %s: %d/%d frames succeeded",
                epsf_label,
                round_id,
                n_ok,
                n_frames,
            )
    elif n_skipped:
        log.info(
            "  ePSF round %s: %d/%d frames succeeded (%d skipped existing)",
            round_id,
            n_ok,
            n_frames,
            n_skipped,
        )
    else:
        log.info("  ePSF round %s: %d/%d frames succeeded", round_id, n_ok, n_frames)
    return epsf_stack, tile_centers, ffi_stems, epsf_ok


def compute_group_epsf_gridded(
    output_dir: str,
    group_ids: np.ndarray,
    ffi_stems: list[str],
    epsf_ok: list[bool],
    *,
    group_subdir: str = "group_epsf",
) -> dict[int, np.ndarray]:
    """
    Median gridded PSF cube per template group across frames.

    Saves ``group_epsf_{gid}.npz`` with keys data, grid_xypos, oversampling.
    """
    index = load_gridded_epsf_index(output_dir)
    group_ids = np.asarray(group_ids)
    unique_groups = [g for g in sorted(set(group_ids.tolist())) if g >= 0]
    group_epsf: dict[int, np.ndarray] = {}

    out_subdir = os.path.join(output_dir, group_subdir)
    os.makedirs(out_subdir, exist_ok=True)

    for gid in unique_groups:
        cubes: list[np.ndarray] = []
        grid_xypos = None
        oversampling = 2
        for stem, frame_gid, ok in zip(ffi_stems, group_ids, epsf_ok):
            if not ok or int(frame_gid) != int(gid):
                continue
            path = index.get(stem) or gridded_epsf_npz_path(output_dir, stem)
            if not os.path.isfile(path):
                continue
            z = np.load(path, allow_pickle=False)
            try:
                cubes.append(np.asarray(z["data"], dtype=np.float64))
                if grid_xypos is None:
                    grid_xypos = np.asarray(z["grid_xypos"])
                    oversampling = int(z["oversampling"])
            finally:
                z.close()
        if not cubes:
            continue
        med = np.nanmedian(np.stack(cubes, axis=0), axis=0)
        group_epsf[gid] = med
        out_path = os.path.join(out_subdir, f"group_epsf_{gid}.npz")
        np.savez_compressed(
            out_path,
            data=med,
            grid_xypos=grid_xypos,
            oversampling=oversampling,
        )
        log.info("  Group %s: %d frames → gridded ePSF cube %s", gid, len(cubes), med.shape)

    return group_epsf
