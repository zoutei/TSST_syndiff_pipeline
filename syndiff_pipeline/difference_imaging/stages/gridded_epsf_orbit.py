"""
Orbit-binned gridded ePSF.

Instead of fitting one :class:`~photutils.psf.GriddedPSFModel` per FFI (the
``per_frame`` legacy path in ``gridded_epsf.py``, the dominant cost of the
``epsf`` stage), fit only a handful of **anchor** models per orbit (default
5), each built from a batch of representative FFIs (default 20) drawn from a
contiguous, quality-filtered window around the anchor's target time. Every
other FFI in that orbit resolves to a BTJD-interpolated blend of the two
bracketing anchors, written through the *existing*
``gridded_epsf_index.json`` (``ffi_stem -> npz path``) contract -- many stems
already share one npz path today, which is the mechanism that makes "N
anchors cover the whole orbit" work with zero changes to
``GriddedEpsfCatalog``/``centroids.py``/``photometry.py`` consumers.

Anchor placement uses an edge-weighted density (denser near orbit start/end,
where drift changes fastest) -- the idea ``TemporalWcsParams.edge_densify_
knots``/``edge_fraction`` names but never implements (see
``temporal_wcs.py``); this module actually implements it, independently.
Orbit segmentation reuses ``shift_schedule._split_orbit_segments_from_csv``
directly (the same MIT ``TESS_orbit_times.csv`` partitioner
``temporal_wcs.py::_orbit_bounds`` wraps) sourced from the already-cached
``ffi_list`` ``DATE-OBS`` column, so orbit indices agree with
``temporal_wcs`` without re-opening any FITS file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from joblib import delayed

from syndiff_pipeline.common.download import manifest_basename_from_local
from syndiff_pipeline.common.ffi_quality import dquality_for_stem, quality_ok_mask
from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.common.wcs_grouping import gaia_science_xy_for_frame
from syndiff_pipeline.common.wcs_header_cache import header_from_cached_row
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
    _configure_blas_threads,
    _diff_path_to_stem,
    _filter_stars_off_mask,
    _is_valid_gridded_epsf_npz,
    _load_gridded_epsf_stack,
    _section_bounds,
    _stars_in_section,
    _tile_centers_from_shape,
    build_diff_image_fps,
    build_gridded_psf_for_frame,
    fit_epsf_section_multi,
    gridded_epsf_npz_path,
    prepare_gaia_for_gridded_epsf,
    save_gridded_epsf_anchor_stems,
    save_gridded_epsf_index,
    save_gridded_epsf_npz,
    stack_from_gridded_cube,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)

log = logging.getLogger(__name__)


# Per-process state for loky workers (initialized once per worker, not per
# task -- see fit_gridded_epsf_orbit_binned's parallel dispatch for why).
_ORBIT_WORKER_CTX: dict[str, Any] = {}


def _init_orbit_epsf_worker(
    gaia_base: pd.DataFrame,
    epsf_params,
    mask_catalog,
    mask_2d: Optional[np.ndarray],
    ffi_list_df: pd.DataFrame,
    science_bounds: dict,
    ffi_path_by_stem: dict[str, str],
    debug_plot_dir: Optional[str] = None,
    epsf_label: str = "epsf",
) -> None:
    """Load shared anchor-fitting inputs once per loky worker."""
    _ORBIT_WORKER_CTX.clear()
    _ORBIT_WORKER_CTX.update(
        {
            "gaia_base": gaia_base,
            "epsf_params": epsf_params,
            "mask_catalog": mask_catalog,
            "mask_2d": mask_2d,
            "ffi_list_df": ffi_list_df,
            "science_bounds": science_bounds,
            "ffi_path_by_stem": dict(ffi_path_by_stem or {}),
            "debug_plot_dir": debug_plot_dir,
            "epsf_label": epsf_label,
        }
    )


def _run_anchor_fit_task(payload: dict) -> tuple[Optional[list], Optional[np.ndarray], list[int]]:
    """Worker: fit one anchor. Reads shared state from _ORBIT_WORKER_CTX
    (set by _init_orbit_epsf_worker), not from closure capture."""
    ctx = _ORBIT_WORKER_CTX
    gaia_base = ctx["gaia_base"]
    epsf_params = ctx["epsf_params"]
    mask_catalog = ctx["mask_catalog"]
    mask_2d = ctx["mask_2d"]
    ffi_list_df = ctx["ffi_list_df"]
    science_bounds = ctx["science_bounds"]
    ffi_path_by_stem = ctx["ffi_path_by_stem"]
    debug_plot_dir = ctx.get("debug_plot_dir")
    epsf_label = ctx.get("epsf_label", "epsf")

    stack_before_fit = bool(epsf_params.epsf_stack_before_fit)
    window_masks = [
        _frame_reject_mask(mask_catalog=mask_catalog, mask_2d=mask_2d, btjd=bt)
        for bt in payload["window_btjds"]
    ]
    if stack_before_fit:
        return fit_anchor_stacked(
            window_diff_paths=payload["window_diff_paths"],
            window_masks=window_masks,
            anchor_ffi_path=payload["anchor_ffi_path"],
            gaia_base=gaia_base,
            epsf_params=epsf_params,
            ffi_list_df=ffi_list_df,
            science_bounds=science_bounds,
            frame_label=payload["stem"],
            debug_plot_dir=debug_plot_dir if bool(getattr(epsf_params, "epsf_debug_plots", True)) else None,
            epsf_label=epsf_label,
        )
    window_ffi_paths = [
        ffi_path_by_stem.get(s) or payload["anchor_ffi_path"]
        for s in payload["window_stems"]
    ]
    return fit_anchor_pooled(
        window_diff_paths=payload["window_diff_paths"],
        window_masks=window_masks,
        window_ffi_paths=window_ffi_paths,
        gaia_base=gaia_base,
        epsf_params=epsf_params,
        ffi_list_df=ffi_list_df,
        science_bounds=science_bounds,
        frame_label=payload["stem"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── Anchor placement (pure, unit-testable) ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════


def anchor_target_phases(
    n_anchors: int,
    edge_fraction: float,
    edge_boost: float,
    *,
    n_grid: int = 4001,
) -> np.ndarray:
    """
    ``n_anchors`` explicit target phases in ``[0, 1]`` over one orbit.

    Canonical 5-anchor placement (the production default,
    ``epsf_per_orbit=5``): the two orbit endpoints, the midpoint, and two
    more at ``edge_fraction`` in from each end -- e.g. ``edge_fraction=0.12``
    gives ``[0.0, 0.12, 0.5, 0.88, 1.0]``. Generalizes to other odd
    ``n_anchors`` by keeping both endpoints and the midpoint fixed and
    filling the remaining anchors as symmetric pairs spaced between
    ``edge_fraction`` and the midpoint. ``edge_boost`` is accepted for
    backward config compatibility but unused by this explicit-endpoint
    scheme (kept from an earlier smooth-density design; see git history).
    """
    del edge_boost, n_grid
    if n_anchors <= 0:
        return np.array([], dtype=float)
    if n_anchors == 1:
        return np.array([0.5], dtype=float)
    if n_anchors == 2:
        return np.array([0.0, 1.0], dtype=float)

    phases = [0.0, 1.0]
    remaining = n_anchors - 2
    if remaining % 2 == 1:
        phases.append(0.5)
        remaining -= 1
    n_pairs = remaining // 2
    edge_fraction = float(edge_fraction)
    for k in range(n_pairs):
        depth = (
            edge_fraction
            if n_pairs == 1
            else edge_fraction + k * (0.5 - edge_fraction) / n_pairs
        )
        phases.append(depth)
        phases.append(1.0 - depth)
    return np.array(sorted(phases), dtype=float)


@dataclass(frozen=True)
class AnchorSelection:
    """One anchor's placement + selected fit window, in orbit-local frame positions."""

    anchor_index: int
    anchor_frame_pos: int
    target_phase: float
    target_btjd: float
    window_frame_pos: tuple[int, ...]


def _grow_contiguous_window(
    anchor_pos: int,
    quality_ok: np.ndarray,
    frames_per_anchor: int,
    max_expand: int,
    n: int,
) -> list[int]:
    """Expand outward from *anchor_pos*, collecting quality-good positions."""
    collected: list[int] = []
    if quality_ok[anchor_pos]:
        collected.append(anchor_pos)
    radius = 1
    while len(collected) < frames_per_anchor and radius <= max_expand:
        left = anchor_pos - radius
        right = anchor_pos + radius
        if left < 0 and right >= n:
            break
        if left >= 0 and quality_ok[left]:
            collected.append(left)
        if len(collected) >= frames_per_anchor:
            break
        if right < n and quality_ok[right]:
            collected.append(right)
        radius += 1
    return sorted(collected)


def select_anchor_frames(
    *,
    btjds: np.ndarray,
    quality_ok: np.ndarray,
    n_anchors: int,
    frames_per_anchor: int,
    edge_fraction: float,
    edge_boost: float,
    max_expand: int,
) -> list[AnchorSelection]:
    """
    Place ``n_anchors`` anchors in one orbit and select each one's fit window.

    *btjds*/*quality_ok* are orbit-local arrays (index 0 = first frame of the
    orbit segment). Each target phase maps to its nearest actual FFI (any
    quality -- this fixes the anchor's identity/npz stem even if that exact
    frame later gets excluded from its own fit window); the fit window then
    grows outward from that position, excluding quality-flagged frames, until
    *frames_per_anchor* good frames are collected or *max_expand* is hit (a
    warning is logged and the run falls back to best-available count).
    """
    n = len(btjds)
    if n == 0:
        return []
    n_anchors = max(1, min(int(n_anchors), n))
    finite = np.isfinite(btjds)
    if not finite.any():
        return []
    candidate_idx = np.where(finite)[0]
    t0, t1 = float(btjds[candidate_idx[0]]), float(btjds[candidate_idx[-1]])
    span = t1 - t0
    phases = anchor_target_phases(n_anchors, edge_fraction, edge_boost)

    out: list[AnchorSelection] = []
    n_good = int(np.count_nonzero(quality_ok))
    for k, phase in enumerate(phases):
        target_btjd = t0 + float(phase) * span
        anchor_pos = int(
            candidate_idx[np.argmin(np.abs(btjds[candidate_idx] - target_btjd))]
        )
        window = _grow_contiguous_window(
            anchor_pos, quality_ok, frames_per_anchor, max_expand, n
        )
        want = min(frames_per_anchor, n_good)
        if len(window) < want:
            log.warning(
                "epsf orbit-binned: anchor %d/%d only collected %d/%d frames "
                "(window expansion exhausted at radius bound %d)",
                k,
                n_anchors,
                len(window),
                frames_per_anchor,
                max_expand,
            )
        if not window:
            window = [anchor_pos]
        out.append(
            AnchorSelection(
                anchor_index=k,
                anchor_frame_pos=anchor_pos,
                target_phase=float(phase),
                target_btjd=float(target_btjd),
                window_frame_pos=tuple(window),
            )
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# ── Orbit segmentation (reuses the shared MIT-orbit-CSV primitive) ─────────
# ═══════════════════════════════════════════════════════════════════════════


def _date_obs_and_dquality(
    ffi_list_df: pd.DataFrame, ffi_path_by_stem: dict[str, str], product_id: str
) -> tuple[Optional[str], int]:
    ffi_path = ffi_path_by_stem.get(product_id)
    if ffi_path is None or ffi_list_df is None:
        return None, 0
    logical = manifest_basename_from_local(ffi_path)
    if logical not in ffi_list_df.index:
        return None, 0
    row = ffi_list_df.loc[logical]
    try:
        hdr = header_from_cached_row(row)
    except Exception:
        return None, 0
    date_obs = hdr.get("DATE-OBS")
    dq = dquality_for_stem(ffi_list_df, logical)
    return (str(date_obs) if date_obs else None), dq


def _resolve_btjd_by_stem(
    product_ids: list[str],
    btjd_by_stem: dict,
    ffi_list_df: pd.DataFrame,
    ffi_path_by_stem: dict[str, str],
) -> dict[str, float]:
    """
    ``btjd_by_stem`` filled in with a DATE-OBS-derived BTJD for any
    product id missing (or non-finite).

    Linear-mode frame manifests (``bookkeeping/diff_linear/.../frames.csv``)
    carry no ``btjd``/``BTJD``/... column at all (unlike field-mode/
    temporal-WCS runs), so ``btjd_by_stem_from_manifest`` can return an
    entirely empty dict -- orbit segmentation and anchor placement need a
    real BTJD for every frame regardless of manifest shape. Uses the same
    conversion as ``temporal_wcs.py::_btjd_from_date_obs``.
    """
    from astropy.time import Time

    resolved = dict(btjd_by_stem or {})
    n_filled = 0
    for pid in product_ids:
        existing = resolved.get(pid)
        if existing is not None and np.isfinite(existing):
            continue
        date_obs, _dq = _date_obs_and_dquality(ffi_list_df, ffi_path_by_stem, pid)
        if not date_obs:
            continue
        try:
            resolved[pid] = float(
                Time(str(date_obs), format="isot", scale="utc").jd - 2457000.0
            )
            n_filled += 1
        except Exception:
            log.debug(
                "epsf orbit-binned: BTJD conversion failed for %s (DATE-OBS=%r)",
                pid,
                date_obs,
            )
    if n_filled:
        log.info(
            "epsf orbit-binned: derived BTJD from DATE-OBS for %d/%d frames "
            "(manifest had no btjd column)",
            n_filled,
            len(product_ids),
        )
    return resolved


def _orbit_segments(
    diff_paths: list[str],
    sector: int,
    ffi_list_df: pd.DataFrame,
    ffi_path_by_stem: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    """``[start, end)`` orbit segments for *diff_paths* (assumed chronological)."""
    from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
        ensure_tess_orbit_times_csv,
    )
    from syndiff_pipeline.template_creation.processing.shift_schedule import (
        _split_orbit_segments_from_csv,
    )

    stems = [_diff_path_to_stem(p) if p else "" for p in diff_paths]
    product_ids = [tess_product_id_from_ffi_path(s) or s for s in stems]
    dates = [
        _date_obs_and_dquality(ffi_list_df, ffi_path_by_stem, pid)[0]
        for pid in product_ids
    ]
    if all(not d for d in dates):
        raise RuntimeError(
            "epsf orbit_binned requires DATE-OBS for every frame (ffi_list is "
            "missing header_cards) -- rebuild ffi_list.parquet or use "
            "epsf_mode: per_frame"
        )
    bounds = _split_orbit_segments_from_csv(
        int(sector), dates, ensure_tess_orbit_times_csv()
    )
    return bounds, product_ids


# ═══════════════════════════════════════════════════════════════════════════
# ── Anchor fitting: stacked (default) and pooled modes ──────────────────────
# ═══════════════════════════════════════════════════════════════════════════


def _load_grid_xypos(path: str) -> Optional[list[tuple[float, float]]]:
    """Read only ``grid_xypos`` from a per-frame gridded-ePSF npz."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return [tuple(row) for row in np.asarray(z["grid_xypos"])]
    except (OSError, ValueError, KeyError):
        return None


def _nanmean_combine(images: list[np.ndarray]) -> np.ndarray:
    """NaN-aware mean-combine (documented policy; see module Notes)."""
    stack = np.stack(images, axis=0)
    with np.errstate(invalid="ignore", all="ignore"):
        out = np.nanmean(stack, axis=0)
    return np.nan_to_num(out, nan=0.0)


def _frame_reject_mask(
    *, mask_catalog, mask_2d: Optional[np.ndarray], btjd: Optional[float]
) -> Optional[np.ndarray]:
    if mask_catalog is not None:
        from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask

        return epsf_reject_mask(mask_catalog.mask_at(btjd, which="full"))
    return mask_2d


def fit_anchor_stacked(
    *,
    window_diff_paths: list[str],
    window_masks: list[Optional[np.ndarray]],
    anchor_ffi_path: str,
    gaia_base: pd.DataFrame,
    epsf_params,
    ffi_list_df: pd.DataFrame,
    science_bounds: dict,
    frame_label: str = "",
    debug_plot_dir: Optional[str] = None,
    epsf_label: str = "epsf",
) -> tuple[Optional[list[tuple[float, float]]], Optional[np.ndarray], list[int]]:
    """
    Mean-combine *window_diff_paths* into one synthetic frame (F4: pre-
    averaging forfeits per-frame recentering -- documented risk, see module
    Notes) and reuse :func:`build_gridded_psf_for_frame` unchanged. Reject
    masks are the **union** (logical OR) of every window frame's mask (F5),
    not one representative frame's -- avoids leaking transient artifacts
    (asteroids, cosmic rays) unique to the other frames into the average
    unmasked. Gaia positions use the anchor's own assigned FFI's WCS (dithers
    are integer-pixel and already reconciled upstream of this stage).

    When *debug_plot_dir* is set, writes this anchor's DS9 region + star-
    selection PNG (blue=used by EPSFBuilder, red=cut-selected but excluded)
    inline from the fit just computed -- no re-fit.
    """
    images = []
    for p in window_diff_paths:
        try:
            images.append(np.asarray(fits.getdata(p), dtype=np.float64))
        except Exception as exc:
            log.warning("epsf orbit-binned: cannot load %s: %s", p, exc)
    if not images:
        return None, None, []
    synthetic = _nanmean_combine(images)
    real_masks = [m for m in window_masks if m is not None]
    union_mask = np.logical_or.reduce(real_masks) if real_masks else None
    gaia_df = gaia_science_xy_for_frame(
        gaia_base, anchor_ffi_path, ffi_list_df, science_bounds
    )
    star_usage: Optional[dict] = {} if debug_plot_dir else None
    _model, grid_xypos, stack, n_stars = build_gridded_psf_for_frame(
        synthetic,
        gaia_df,
        epsf_params,
        mask_2d=union_mask,
        frame_label=frame_label,
        star_usage_out=star_usage,
    )
    if debug_plot_dir and star_usage:
        try:
            from syndiff_pipeline.difference_imaging.support.ds9_regions import (
                write_epsf_star_selection_region,
            )
            from syndiff_pipeline.difference_imaging.support.plot import (
                epsf_star_selection_png_path,
                epsf_star_selection_region_path,
                write_epsf_star_selection_plot,
            )

            used_xy = star_usage.get("used_xy", [])
            excluded_xy = star_usage.get("excluded_xy", [])
            write_epsf_star_selection_region(
                used_xy,
                excluded_xy,
                epsf_star_selection_region_path(debug_plot_dir, epsf_label, frame_label),
            )
            write_epsf_star_selection_plot(
                synthetic,
                used_xy,
                excluded_xy,
                epsf_star_selection_png_path(debug_plot_dir, epsf_label, frame_label),
                title=f"{epsf_label} · {frame_label} · star selection",
            )
        except Exception:
            log.warning(
                "epsf orbit-binned: star-selection debug output failed for %s",
                frame_label,
                exc_info=True,
            )
    return grid_xypos, stack, n_stars


def fit_anchor_pooled(
    *,
    window_diff_paths: list[str],
    window_masks: list[Optional[np.ndarray]],
    window_ffi_paths: list[str],
    gaia_base: pd.DataFrame,
    epsf_params,
    ffi_list_df: pd.DataFrame,
    science_bounds: dict,
    frame_label: str = "",
) -> tuple[list[tuple[float, float]], Optional[np.ndarray], list[int]]:
    """
    Pool per-frame Gaia-star extractions (each frame keeps its own
    recentering) across the whole window into one ``EPSFBuilder`` call per
    tile, via :func:`fit_epsf_section_multi`. Generalizes
    :func:`build_gridded_psf_for_frame`'s single-frame section loop.

    Returns ``(grid_xypos, stack, n_stars_per_tile)`` -- ``n_stars_per_tile``
    is the pooled candidate-star count (summed across the whole window) per
    tile, for debug-plot annotation.
    """
    images: list[np.ndarray] = []
    gaia_dfs: list[pd.DataFrame] = []
    masks: list[Optional[np.ndarray]] = []
    for path, ffi_path, mask in zip(window_diff_paths, window_ffi_paths, window_masks):
        try:
            img = np.asarray(fits.getdata(path), dtype=np.float64)
        except Exception as exc:
            log.warning("epsf orbit-binned (pooled): cannot load %s: %s", path, exc)
            continue
        images.append(img)
        gaia_dfs.append(
            gaia_science_xy_for_frame(gaia_base, ffi_path, ffi_list_df, science_bounds)
        )
        masks.append(mask)
    if not images:
        return [], None, []

    ny, nx = images[0].shape
    tile_ny = int(epsf_params.tile_ny)
    tile_nx = int(epsf_params.tile_nx)
    oversampling = int(epsf_params.epsf_oversample)
    min_stars = int(getattr(epsf_params, "min_stars_per_tile", 5))
    maxiters = int(getattr(epsf_params, "epsf_maxiters", 15))
    recentering_maxiters = int(getattr(epsf_params, "epsf_recentering_maxiters", 20))
    extract_size = int(getattr(epsf_params, "extract_size", None) or epsf_params.psf_size)
    smoothing_kernel = str(getattr(epsf_params, "epsf_smoothing_kernel", "quadratic"))
    builder_fit_shape = int(getattr(epsf_params, "epsf_builder_fit_shape", 5))
    recentering_boxsize = int(getattr(epsf_params, "epsf_recentering_boxsize", 3))
    star_box_radius = int(getattr(epsf_params, "epsf_star_box_radius", 7))
    use_section_mask = bool(getattr(epsf_params, "epsf_use_section_mask", True))
    stamp_border_crop = int(getattr(epsf_params, "epsf_stamp_border_crop", 0))
    star_margin = float(extract_size) / 2.0 + 2.0

    epsf_grid: dict[tuple[int, int], Any] = {}
    grid_xypos: list[tuple[float, float]] = []
    n_stars_grid: dict[tuple[int, int], int] = {}
    for i in range(tile_ny):
        for j in range(tile_nx):
            x_min, x_max, y_min, y_max = _section_bounds(ny, nx, tile_ny, tile_nx, i, j)
            x_center = j * (nx / tile_nx) + (nx / (2 * tile_nx))
            y_center = i * (ny / tile_ny) + (ny / (2 * tile_ny))
            grid_xypos.append((float(x_center), float(y_center)))

            section_frames: list[tuple[np.ndarray, Table, Optional[np.ndarray]]] = []
            for img, gdf, mask_2d in zip(images, gaia_dfs, masks):
                sec_stars = _stars_in_section(
                    gdf, x_min, x_max, y_min, y_max, margin=star_margin
                )
                section_mask = None
                if mask_2d is not None:
                    section_mask = np.asarray(
                        mask_2d[y_min:y_max, x_min:x_max], dtype=bool
                    )
                    sec_stars = _filter_stars_off_mask(sec_stars, mask_2d, ny=ny, nx=nx)
                if len(sec_stars) == 0:
                    continue
                stars_tbl = Table()
                stars_tbl["x"] = np.asarray(sec_stars["x"].values - x_min, dtype=float)
                stars_tbl["y"] = np.asarray(sec_stars["y"].values - y_min, dtype=float)
                section_frames.append(
                    (np.asarray(img[y_min:y_max, x_min:x_max], dtype=np.float64), stars_tbl, section_mask)
                )

            n_stars_grid[(i, j)] = sum(len(t) for _, t, _ in section_frames)
            if n_stars_grid[(i, j)] < min_stars:
                epsf_grid[(i, j)] = "too_few"
                continue
            stamp = fit_epsf_section_multi(
                section_frames,
                extract_size=extract_size,
                oversampling=oversampling,
                maxiters=maxiters,
                recentering_maxiters=recentering_maxiters,
                smoothing_kernel=smoothing_kernel,
                builder_fit_shape=builder_fit_shape,
                recentering_boxsize=recentering_boxsize,
                use_mask=use_section_mask,
                star_box_radius=star_box_radius,
            )
            epsf_grid[(i, j)] = stamp if stamp is not None else "fit_failed"

    valid = [v for v in epsf_grid.values() if isinstance(v, np.ndarray)]
    n_stars_list = [
        n_stars_grid.get((i, j), 0) for i in range(tile_ny) for j in range(tile_nx)
    ]
    if not valid:
        log.warning(
            "epsf orbit-binned (pooled): all grid sections failed%s",
            f" ({frame_label})" if frame_label else "",
        )
        return grid_xypos, None, n_stars_list
    fallback = np.mean(valid, axis=0)
    psf_list = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            result = epsf_grid.get((i, j), "too_few")
            psf_list.append(result if isinstance(result, np.ndarray) else fallback)
    if stamp_border_crop > 0:
        bc = stamp_border_crop
        psf_list = [
            arr if bc * 2 >= arr.shape[0] or bc * 2 >= arr.shape[1] else arr[bc:-bc, bc:-bc]
            for arr in psf_list
        ]
    return grid_xypos, np.array(psf_list, dtype=np.float64), n_stars_list


# ═══════════════════════════════════════════════════════════════════════════
# ── Provenance fingerprints for orbit-binned frames (F1) ────────────────────
# ═══════════════════════════════════════════════════════════════════════════


def anchor_epsf_fingerprint(
    *,
    sector: int,
    camera: int,
    ccd: int,
    anchor_product_id: str,
    epsf_label: str,
    epsf_params,
    window_diff_image_fps: list[Optional[str]],
) -> Optional[str]:
    """
    An anchor's ``epsf`` fingerprint from every diff image in its own fit
    window (not just its own frame) -- correctly invalidates the anchor when
    *any* window member's upstream diff image changes.
    """
    if not window_diff_image_fps or any(fp is None for fp in window_diff_image_fps):
        return None
    primary, *extra = window_diff_image_fps
    inputs = provenance_glue.epsf_input_fingerprints(primary, *extra)
    if inputs is None:
        return None
    return provenance_glue.diff_kind_fingerprint(
        "epsf",
        sector=sector,
        camera=camera,
        ccd=ccd,
        product_id=anchor_product_id,
        label=epsf_label,
        params=epsf_params,
        input_fingerprints=inputs,
    )


def interpolated_epsf_fingerprint(
    *,
    sector: int,
    camera: int,
    ccd: int,
    product_id: str,
    epsf_label: str,
    epsf_params,
    neighbor_anchor_fps: list[Optional[str]],
) -> Optional[str]:
    """
    An interpolated (non-anchor) frame's ``epsf`` fingerprint from its one or
    two bracketing anchors' *own* ``epsf`` fingerprints -- not its own diff
    image, which the artifact does not actually depend on in isolation (F1:
    the previous per-frame-only scheme was simply incorrect for these
    frames). Each anchor fingerprint already encodes that anchor's full
    window + config, so this correctly invalidates whenever either
    neighboring anchor's fit changes for any reason.
    """
    if not neighbor_anchor_fps or any(fp is None for fp in neighbor_anchor_fps):
        return None
    primary, *extra = neighbor_anchor_fps
    inputs = provenance_glue.epsf_input_fingerprints(primary, *extra)
    if inputs is None:
        return None
    return provenance_glue.diff_kind_fingerprint(
        "epsf",
        sector=sector,
        camera=camera,
        ccd=ccd,
        product_id=product_id,
        label=epsf_label,
        params=epsf_params,
        input_fingerprints=inputs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── Main entry point ─────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_epsf_write_path(
    *,
    stem: str,
    epsf_params,
    epsf_label: str,
    output_dir: str,
    data_root: Optional[str],
    sck: Optional[tuple],
    output_store_name: Optional[str],
) -> str:
    if data_root and sck is not None:
        from syndiff_pipeline.difference_imaging.orchestration.diff_store import (
            resolve_diff_write_path,
        )

        return str(
            resolve_diff_write_path(
                data_root=data_root,
                sck=sck,
                kind="epsf",
                stage_label=epsf_label,
                ffi_stem=stem,
                label=epsf_label,
                params=epsf_params,
                output_store_name=output_store_name,
                suffix=".npz",
            )
        )
    return gridded_epsf_npz_path(output_dir, stem)


def fit_gridded_epsf_orbit_binned(
    diff_paths: list[str],
    gaia_df: pd.DataFrame,
    cfg,
    epsf_params,
    output_dir: str,
    *,
    mask_2d: Optional[np.ndarray] = None,
    mask_catalog=None,
    btjd_by_stem: Optional[dict] = None,
    round_id: int = 1,
    diff_log_path: Optional[str] = None,
    epsf_label: Optional[str] = None,
    diffs_input: Optional[str] = None,
    skip_existing: bool = True,
    ffi_list_df: Optional[pd.DataFrame] = None,
    science_bounds: Optional[dict] = None,
    ffi_path_by_stem: Optional[dict[str, str]] = None,
    wcs_table: Optional[pd.DataFrame] = None,
    debug_plot_dir: Optional[str] = None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[str], list[bool]]:
    """
    Fit orbit-binned gridded ePSF: a handful of anchor models per orbit,
    every other frame resolved to a BTJD-interpolated blend.

    Same return contract as
    :func:`gridded_epsf.fit_gridded_epsf_all_frames` (legacy flat
    ``(n_frames, n_tiles, n_pix)`` stack, tile centers, stems, per-frame ok
    flags) so callers/downstream stages (``prepare_epsf_stack``,
    ``save_epsf_stack_bundle``, ``compute_group_epsf``) work unchanged
    regardless of ``epsf_mode``.

    When ``debug_plot_dir`` is set and ``epsf_params.epsf_debug_plots`` is
    True, each anchor's own real fit (not the blended/interpolated frames,
    which have none) writes a DS9 region + star-selection PNG inline --
    reusing the fit already computed, no re-fit. Anchor frames are
    naturally few and bounded (``epsf_per_orbit`` per orbit), so no
    separate frame-selection step is needed the way centroids' debug
    residual FITS needs one. Only the default ``epsf_stack_before_fit:
    true`` (stacked) anchor-fit path is covered -- not
    :func:`fit_anchor_pooled`.
    """
    n_frames = len(diff_paths)
    epsf_label_str = str(epsf_label or "epsf")
    if n_frames == 0:
        return np.zeros((0, 0, 0)), [], [], []

    epsf_n_jobs = getattr(epsf_params, "epsf_n_jobs", None)
    n_workers = resolve_effective_n_jobs(
        int(getattr(cfg, "n_jobs", 1) or 1), stage_n_jobs=epsf_n_jobs
    )
    _configure_blas_threads(n_workers)

    gaia_base = prepare_gaia_for_gridded_epsf(gaia_df, epsf_params)
    os.makedirs(output_dir, exist_ok=True)

    if ffi_path_by_stem is None and wcs_table is not None:
        from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
            ffi_path_by_stem_from_wcs_table,
        )

        ffi_path_by_stem = ffi_path_by_stem_from_wcs_table(wcs_table)
    ffi_path_by_stem = ffi_path_by_stem or {}

    if btjd_by_stem is None and wcs_table is not None:
        from syndiff_pipeline.difference_imaging.stages.epsf import (
            btjd_by_stem_from_manifest,
        )

        btjd_by_stem = btjd_by_stem_from_manifest(wcs_table)
    btjd_by_stem = btjd_by_stem or {}

    if ffi_list_df is None or science_bounds is None:
        raise RuntimeError(
            "epsf_mode=orbit_binned requires ffi_list_df and science_bounds"
        )

    sector = int(getattr(cfg, "sector"))
    try:
        prov_sck = (int(cfg.sector), int(cfg.camera), int(cfg.ccd))
    except Exception:
        prov_sck = None
    prov_data_root = getattr(cfg, "data_root", "") or None
    prov_output_store_name = getattr(cfg, "output_store_name", None) or None
    prov_run_id = getattr(cfg, "run_id", "") or None

    diff_image_fps = build_diff_image_fps(
        cfg, diff_paths, diffs_input=diffs_input, sck=prov_sck
    )

    stems = [
        _diff_path_to_stem(p) if p else f"frame_{i}" for i, p in enumerate(diff_paths)
    ]
    product_ids = [tess_product_id_from_ffi_path(s) or s for s in stems]
    btjd_by_stem = _resolve_btjd_by_stem(
        product_ids, btjd_by_stem, ffi_list_df, ffi_path_by_stem
    )

    segment_bounds, _pids = _orbit_segments(
        diff_paths, sector, ffi_list_df, ffi_path_by_stem
    )

    from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
        init_progress_pair,
        progress_path_for_diff_log,
        progress_path_for_output_workspace,
        record_frame_progress,
        refresh_progress_pair_from_artifacts,
        set_progress_phase_pair,
    )

    track_progress = epsf_label is not None
    cli_progress_path = (
        str(progress_path_for_diff_log(diff_log_path))
        if track_progress and diff_log_path is not None
        else None
    )
    workspace_progress_path: Optional[str] = None
    if track_progress:
        workspace_progress_path = str(progress_path_for_output_workspace(output_dir))
        init_progress_pair(
            workspace_progress_path,
            cli_progress_path,
            epsf_label=epsf_label_str,
            diffs_input=str(diffs_input or "?"),
            round_id=round_id,
            frames_total=n_frames,
            output_dir=output_dir,
        )
        refresh_progress_pair_from_artifacts(workspace_progress_path, cli_progress_path)

    # ── Pass 1: orbit segmentation + anchor placement (cheap, header-only) ──
    orbit_anchor_lists: dict[int, list[AnchorSelection]] = {}
    orbit_local_range: dict[int, tuple[int, int]] = {}
    for orbit_idx, (start, end) in enumerate(segment_bounds):
        start, end = int(start), int(end)
        if end <= start:
            continue
        orbit_local_range[orbit_idx] = (start, end)
        local_btjds = np.array(
            [btjd_by_stem.get(product_ids[p], np.nan) for p in range(start, end)],
            dtype=float,
        )
        local_dq = np.array(
            [
                _date_obs_and_dquality(ffi_list_df, ffi_path_by_stem, product_ids[p])[1]
                for p in range(start, end)
            ],
            dtype=int,
        )
        quality_ok = np.array(
            [
                quality_ok_mask(dq, epsf_params.epsf_quality_bitmask) and np.isfinite(bt)
                for dq, bt in zip(local_dq, local_btjds)
            ]
        )
        n_local = end - start
        n_anchors_orbit = max(1, min(int(epsf_params.epsf_per_orbit), n_local))
        frames_per_anchor_orbit = max(
            1, min(int(epsf_params.epsf_frames_per_anchor), n_local)
        )
        if (
            n_anchors_orbit < epsf_params.epsf_per_orbit
            or frames_per_anchor_orbit < epsf_params.epsf_frames_per_anchor
        ):
            log.warning(
                "epsf orbit-binned: orbit %d has only %d frames (< "
                "epsf_per_orbit=%d / epsf_frames_per_anchor=%d); relaxed to "
                "%d anchors x up to %d frames each (short orbit or a "
                "max_ffis debug crop -- see F6)",
                orbit_idx,
                n_local,
                epsf_params.epsf_per_orbit,
                epsf_params.epsf_frames_per_anchor,
                n_anchors_orbit,
                frames_per_anchor_orbit,
            )
        orbit_anchor_lists[orbit_idx] = select_anchor_frames(
            btjds=local_btjds,
            quality_ok=quality_ok,
            n_anchors=n_anchors_orbit,
            frames_per_anchor=frames_per_anchor_orbit,
            edge_fraction=float(epsf_params.epsf_anchor_edge_fraction),
            edge_boost=float(epsf_params.epsf_anchor_edge_boost),
            max_expand=int(epsf_params.epsf_anchor_window_max_expand),
        )

    n_anchor_total = sum(len(a) for a in orbit_anchor_lists.values())
    log.info(
        "ePSF [%s] round %s (orbit_binned): %d orbits, %d anchors, %d frames "
        "total (n_jobs=%s)",
        epsf_label_str,
        round_id,
        len(orbit_anchor_lists),
        n_anchor_total,
        n_frames,
        n_workers,
    )

    # ── Pass 2: resolve each anchor's write path/fingerprint, skip-check ────
    anchor_write_path: dict[tuple[int, int], str] = {}
    anchor_fp: dict[tuple[int, int], Optional[str]] = {}
    anchor_window_global: dict[tuple[int, int], list[int]] = {}
    fit_tasks: list[tuple[tuple[int, int], dict]] = []
    anchor_stack: dict[tuple[int, int], Optional[np.ndarray]] = {}
    anchor_n_stars: dict[tuple[int, int], list[int]] = {}
    anchor_grid_xypos: Optional[list[tuple[float, float]]] = None

    for orbit_idx, anchors in orbit_anchor_lists.items():
        start, _end = orbit_local_range[orbit_idx]
        for a_idx, anchor in enumerate(anchors):
            key = (orbit_idx, a_idx)
            global_pos = start + anchor.anchor_frame_pos
            stem = stems[global_pos]
            pid = product_ids[global_pos]
            window_global = [start + w for w in anchor.window_frame_pos]
            anchor_window_global[key] = window_global
            window_fps = [diff_image_fps.get(product_ids[p]) for p in window_global]

            fp = (
                anchor_epsf_fingerprint(
                    sector=prov_sck[0],
                    camera=prov_sck[1],
                    ccd=prov_sck[2],
                    anchor_product_id=pid,
                    epsf_label=epsf_label_str,
                    epsf_params=epsf_params,
                    window_diff_image_fps=window_fps,
                )
                if prov_sck is not None
                else None
            )
            anchor_fp[key] = fp

            write_path = _resolve_epsf_write_path(
                stem=stem,
                epsf_params=epsf_params,
                epsf_label=epsf_label_str,
                output_dir=output_dir,
                data_root=prov_data_root,
                sck=prov_sck,
                output_store_name=prov_output_store_name,
            )
            anchor_write_path[key] = write_path

            if skip_existing and _is_valid_gridded_epsf_npz(write_path):
                loaded = _load_gridded_epsf_stack(write_path)
                anchor_stack[key] = loaded
                if loaded is not None and anchor_grid_xypos is None:
                    anchor_grid_xypos = _load_grid_xypos(write_path)
                continue

            fit_tasks.append(
                (
                    key,
                    {
                        "window_diff_paths": [diff_paths[p] for p in window_global],
                        "window_stems": [stems[p] for p in window_global],
                        "window_btjds": [
                            btjd_by_stem.get(product_ids[p]) for p in window_global
                        ],
                        "anchor_ffi_path": ffi_path_by_stem.get(pid)
                        or ffi_path_by_stem.get(stem),
                        "stem": stem,
                        "pid": pid,
                    },
                )
            )

    if fit_tasks:
        worker_initargs = (
            gaia_base,
            epsf_params,
            mask_catalog,
            mask_2d,
            ffi_list_df,
            science_bounds,
            ffi_path_by_stem,
            debug_plot_dir,
            epsf_label_str,
        )
        if n_workers <= 1 or len(fit_tasks) <= 1:
            _init_orbit_epsf_worker(*worker_initargs)
            fit_results = [_run_anchor_fit_task(payload) for _key, payload in fit_tasks]
        else:
            # Shared per-worker state (gaia_base, mask_catalog, ...) goes
            # through the loky initializer, not per-task closure capture:
            # a nested closure gets re-pickled on every task, and joblib's
            # automatic memmapping of the large repeatedly-pickled arrays
            # (MaskCatalog._buf/.static in particular) makes them read-only
            # in the worker -- MaskCatalog.mask_at's buffer-reuse write then
            # raises "assignment destination is read-only" (see
            # gridded_epsf.py's _init_gridded_epsf_worker/_WORKER_CTX for
            # the same pattern already solving this for per_frame mode).
            calls = [delayed(_run_anchor_fit_task)(payload) for _key, payload in fit_tasks]
            fit_results = parallel_map_with_optional_tqdm(
                calls,
                n_tasks=len(calls),
                desc=f"epsf {epsf_label_str} (anchors)",
                n_jobs_eff=n_workers,
                initializer=_init_orbit_epsf_worker,
                initargs=worker_initargs,
            )
        for (key, payload), (grid_xypos, stack, n_stars) in zip(fit_tasks, fit_results):
            anchor_stack[key] = stack
            anchor_n_stars[key] = n_stars
            if stack is not None and anchor_grid_xypos is None:
                anchor_grid_xypos = grid_xypos
            if track_progress and workspace_progress_path is not None:
                record_frame_progress(
                    workspace_progress_path, cli_progress_path, success=stack is not None
                )
            if stack is not None:
                save_gridded_epsf_npz(
                    anchor_write_path[key],
                    stack,
                    grid_xypos,
                    int(epsf_params.epsf_oversample),
                    n_stars=n_stars,
                )
                orbit_idx, a_idx = key
                pid = product_ids[
                    orbit_local_range[orbit_idx][0]
                    + orbit_anchor_lists[orbit_idx][a_idx].anchor_frame_pos
                ]
                fp = anchor_fp.get(key)
                if prov_sck is not None and fp is not None:
                    window_global = anchor_window_global[key]
                    window_fps = [diff_image_fps.get(product_ids[p]) for p in window_global]
                    inputs = provenance_glue.epsf_input_fingerprints(
                        window_fps[0], *window_fps[1:]
                    ) if window_fps else None
                    if inputs is not None:
                        try:
                            anchor_meta = {}
                            if prov_run_id:
                                anchor_meta["run_id"] = prov_run_id
                            provenance_glue.emit_diff_artifact(
                                kind="epsf",
                                sector=prov_sck[0],
                                camera=prov_sck[1],
                                ccd=prov_sck[2],
                                product_id=pid,
                                label=epsf_label_str,
                                params=epsf_params,
                                location=anchor_write_path[key],
                                input_fingerprints=inputs,
                                data_root=prov_data_root,
                                is_fits=False,
                                output_store_name=prov_output_store_name,
                                meta=anchor_meta or None,
                            )
                        except Exception:
                            log.debug(
                                "provenance emit (epsf orbit anchor) failed for %s",
                                pid,
                                exc_info=True,
                            )
    else:
        for key in anchor_write_path:
            if track_progress and workspace_progress_path is not None and key in anchor_stack:
                record_frame_progress(
                    workspace_progress_path,
                    cli_progress_path,
                    success=anchor_stack[key] is not None,
                )

    # ── Pass 3: index materialization for every frame (anchor / blend / clamp) ──
    index: dict[str, str] = {}
    anchor_stems: set[str] = set()
    stems_out: list[str] = list(stems)
    epsf_ok: list[bool] = [False] * n_frames
    stacks_by_pos: list[Optional[np.ndarray]] = [None] * n_frames

    for orbit_idx, anchors in orbit_anchor_lists.items():
        start, end = orbit_local_range[orbit_idx]
        if not anchors:
            continue
        anchor_global_positions = [start + a.anchor_frame_pos for a in anchors]
        anchor_btjds = [
            btjd_by_stem.get(product_ids[p], np.nan) for p in anchor_global_positions
        ]
        order = np.argsort(anchor_global_positions)
        sorted_anchor_pos = [anchor_global_positions[i] for i in order]
        sorted_anchor_keys = [(orbit_idx, int(order[i])) for i in range(len(order))]
        sorted_anchor_btjds = [anchor_btjds[i] for i in order]

        for local_i, global_pos in enumerate(range(start, end)):
            stem = stems[global_pos]
            pid = product_ids[global_pos]
            if global_pos in anchor_global_positions:
                a_idx = anchor_global_positions.index(global_pos)
                key = (orbit_idx, a_idx)
                stack = anchor_stack.get(key)
                write_path = anchor_write_path[key]
                index[stem] = write_path
                stacks_by_pos[global_pos] = stack
                epsf_ok[global_pos] = stack is not None
                if stack is not None:
                    anchor_stems.add(stem)
                continue

            frame_btjd = btjd_by_stem.get(pid, np.nan)
            before_i = None
            after_i = None
            for i, apos in enumerate(sorted_anchor_pos):
                if apos < global_pos:
                    before_i = i
                elif apos > global_pos and after_i is None:
                    after_i = i

            if before_i is not None and after_i is not None:
                key_a, key_b = sorted_anchor_keys[before_i], sorted_anchor_keys[after_i]
                stack_a, stack_b = anchor_stack.get(key_a), anchor_stack.get(key_b)
                t_a, t_b = sorted_anchor_btjds[before_i], sorted_anchor_btjds[after_i]
                fp_a, fp_b = anchor_fp.get(key_a), anchor_fp.get(key_b)
                neighbor_fps = [fp_a, fp_b]
                clamp_key = None
            elif before_i is not None:
                key_a = sorted_anchor_keys[before_i]
                stack_a, stack_b = anchor_stack.get(key_a), None
                neighbor_fps = [anchor_fp.get(key_a)]
                clamp_key = key_a
            elif after_i is not None:
                key_a = sorted_anchor_keys[after_i]
                stack_a, stack_b = anchor_stack.get(key_a), None
                neighbor_fps = [anchor_fp.get(key_a)]
                clamp_key = key_a
            else:
                stack_a = stack_b = None
                neighbor_fps = []
                clamp_key = None

            if clamp_key is not None:
                # Before first / after last anchor in the orbit: clamp
                # directly to that anchor's own npz -- no new file.
                index[stem] = anchor_write_path[clamp_key]
                stacks_by_pos[global_pos] = stack_a
                epsf_ok[global_pos] = stack_a is not None
            elif stack_a is not None and stack_b is not None and np.isfinite(t_a) and np.isfinite(t_b) and t_b > t_a:
                weight = float(np.clip((frame_btjd - t_a) / (t_b - t_a), 0.0, 1.0))
                blended = (1.0 - weight) * stack_a + weight * stack_b
                write_path = _resolve_epsf_write_path(
                    stem=stem,
                    epsf_params=epsf_params,
                    epsf_label=epsf_label_str,
                    output_dir=output_dir,
                    data_root=prov_data_root,
                    sck=prov_sck,
                    output_store_name=prov_output_store_name,
                )
                if not (skip_existing and _is_valid_gridded_epsf_npz(write_path)):
                    # n_stars intentionally omitted: this frame has no fit of
                    # its own (interpolated/blended from two anchors), and a
                    # synthetic blended count would misrepresent it as a real
                    # per-tile candidate count in the debug-plot title.
                    save_gridded_epsf_npz(
                        write_path,
                        blended,
                        anchor_grid_xypos,
                        int(epsf_params.epsf_oversample),
                    )
                    if prov_sck is not None:
                        interp_fp = interpolated_epsf_fingerprint(
                            sector=prov_sck[0],
                            camera=prov_sck[1],
                            ccd=prov_sck[2],
                            product_id=pid,
                            epsf_label=epsf_label_str,
                            epsf_params=epsf_params,
                            neighbor_anchor_fps=neighbor_fps,
                        )
                        if interp_fp is not None:
                            inputs = provenance_glue.epsf_input_fingerprints(
                                neighbor_fps[0], *neighbor_fps[1:]
                            )
                            if inputs is not None:
                                try:
                                    blend_meta = {"producer": "gridded_epsf_orbit_blend"}
                                    if prov_run_id:
                                        blend_meta["run_id"] = prov_run_id
                                    provenance_glue.emit_diff_artifact(
                                        kind="epsf",
                                        sector=prov_sck[0],
                                        camera=prov_sck[1],
                                        ccd=prov_sck[2],
                                        product_id=pid,
                                        label=epsf_label_str,
                                        params=epsf_params,
                                        location=write_path,
                                        input_fingerprints=inputs,
                                        data_root=prov_data_root,
                                        is_fits=False,
                                        output_store_name=prov_output_store_name,
                                        meta=blend_meta,
                                    )
                                except Exception:
                                    log.debug(
                                        "provenance emit (epsf orbit blend) failed for %s",
                                        pid,
                                        exc_info=True,
                                    )
                index[stem] = write_path
                stacks_by_pos[global_pos] = blended
                epsf_ok[global_pos] = True
            else:
                stacks_by_pos[global_pos] = None
                epsf_ok[global_pos] = False

            if track_progress and workspace_progress_path is not None:
                record_frame_progress(
                    workspace_progress_path,
                    cli_progress_path,
                    success=epsf_ok[global_pos],
                )

    # Frames outside any orbit segment (shouldn't happen, but fail safe).
    for global_pos in range(n_frames):
        if stems_out[global_pos] not in index and stacks_by_pos[global_pos] is None:
            epsf_ok[global_pos] = False

    if track_progress and workspace_progress_path is not None:
        refresh_progress_pair_from_artifacts(workspace_progress_path, cli_progress_path)
        set_progress_phase_pair(workspace_progress_path, cli_progress_path, "complete")

    save_gridded_epsf_index(output_dir, index)
    save_gridded_epsf_anchor_stems(output_dir, anchor_stems)

    if anchor_grid_xypos is None:
        first_path = next((p for p in diff_paths if p and os.path.exists(p)), None)
        if mask_2d is not None:
            ny, nx = mask_2d.shape
        elif first_path is not None:
            ny, nx = fits.getdata(first_path).shape
        else:
            ny, nx = 1024, 1024
        anchor_grid_xypos = _tile_centers_from_shape(
            ny, nx, int(epsf_params.tile_ny), int(epsf_params.tile_nx)
        )

    n_tiles = epsf_params.tile_ny * epsf_params.tile_nx
    n_pix = 0
    for s in stacks_by_pos:
        if s is not None:
            n_pix = s.reshape(s.shape[0], -1).shape[1]
            break
    if n_pix == 0:
        n_pix = (2 * epsf_params.psf_size + 1) ** 2

    epsf_stack = np.full((n_frames, n_tiles, n_pix), np.nan)
    good_rows = [
        stack_from_gridded_cube(s) for s in stacks_by_pos if s is not None
    ]
    med_row = np.nanmedian(np.stack(good_rows), axis=0) if good_rows else None
    for i, s in enumerate(stacks_by_pos):
        if s is not None:
            epsf_stack[i] = stack_from_gridded_cube(s)
        elif med_row is not None:
            epsf_stack[i] = med_row

    n_ok = sum(epsf_ok)
    log.info(
        "ePSF [%s] round %s (orbit_binned): %d/%d frames ok (%d anchors fit fresh, "
        "%d anchors reused)",
        epsf_label_str,
        round_id,
        n_ok,
        n_frames,
        len(fit_tasks),
        n_anchor_total - len(fit_tasks),
    )

    if debug_plots_enabled(epsf_params):
        try:
            write_orbit_debug_plots(
                output_dir=output_dir,
                epsf_label=epsf_label_str,
                segment_bounds=segment_bounds,
                orbit_anchor_lists=orbit_anchor_lists,
                orbit_local_range=orbit_local_range,
                product_ids=product_ids,
                btjd_by_stem=btjd_by_stem,
                ffi_list_df=ffi_list_df,
                ffi_path_by_stem=ffi_path_by_stem,
                epsf_params=epsf_params,
            )
        except Exception:
            log.debug("epsf orbit debug plots failed (best-effort)", exc_info=True)

    return epsf_stack, anchor_grid_xypos, stems_out, epsf_ok


# ═══════════════════════════════════════════════════════════════════════════
# ── Debug plot (item 11) ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════


def debug_plots_enabled(epsf_params) -> bool:
    return bool(getattr(epsf_params, "epsf_debug_plots", True))


def write_orbit_debug_plots(
    *,
    output_dir: str,
    epsf_label: str,
    segment_bounds,
    orbit_anchor_lists: dict[int, list[AnchorSelection]],
    orbit_local_range: dict[int, tuple[int, int]],
    product_ids: list[str],
    btjd_by_stem: dict,
    ffi_list_df,
    ffi_path_by_stem: dict,
    epsf_params,
) -> None:
    """
    One diagnostic plot per orbit: FFI BTJD rug (quality-excluded frames
    marked), anchor target-phase vs. assigned-FFI markers, each anchor's
    selected window bracketed, and the interpolation blend weight for
    non-anchor frames. Best-effort -- a missing/broken matplotlib must never
    invalidate the (already-written) science artifacts, matching
    ``temporal_wcs.py::_write_debug_plots``'s convention.
    """
    try:
        import matplotlib.pyplot as plt

        debug_dir = os.path.join(output_dir, "debug_plots")
        os.makedirs(debug_dir, exist_ok=True)
        bitmask = int(epsf_params.epsf_quality_bitmask)

        for orbit_idx, (start, end) in orbit_local_range.items():
            anchors = orbit_anchor_lists.get(orbit_idx, [])
            local_positions = list(range(start, end))
            local_btjd = np.array(
                [btjd_by_stem.get(product_ids[p], np.nan) for p in local_positions]
            )
            local_dq = np.array(
                [
                    _date_obs_and_dquality(ffi_list_df, ffi_path_by_stem, product_ids[p])[1]
                    for p in local_positions
                ]
            )
            good = np.array(
                [
                    quality_ok_mask(dq, bitmask) and np.isfinite(bt)
                    for dq, bt in zip(local_dq, local_btjd)
                ]
            )
            finite = np.isfinite(local_btjd)

            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.plot(
                local_btjd[finite & good],
                np.zeros(int((finite & good).sum())),
                "|",
                color="tab:blue",
                ms=10,
                label="good",
            )
            ax.plot(
                local_btjd[finite & ~good],
                np.zeros(int((finite & ~good).sum())),
                "x",
                color="0.6",
                ms=6,
                label="quality-excluded",
            )

            colors = plt.cm.tab10(np.linspace(0, 1, max(len(anchors), 1)))
            anchor_pos_sorted = sorted(start + a.anchor_frame_pos for a in anchors)
            anchor_bt_sorted = [
                btjd_by_stem.get(product_ids[p], np.nan) for p in anchor_pos_sorted
            ]
            for a_idx, anchor in enumerate(anchors):
                color = colors[a_idx % len(colors)]
                ax.axvline(anchor.target_btjd, color=color, ls="--", lw=1, alpha=0.7)
                assigned_btjd = local_btjd[anchor.anchor_frame_pos]
                if np.isfinite(assigned_btjd):
                    ax.plot([assigned_btjd], [0.0], marker="*", color=color, ms=14, zorder=5)
                window_bt = [
                    local_btjd[w]
                    for w in anchor.window_frame_pos
                    if np.isfinite(local_btjd[w])
                ]
                if window_bt:
                    ax.hlines(
                        0.02 + 0.03 * a_idx,
                        min(window_bt),
                        max(window_bt),
                        color=color,
                        lw=3,
                        alpha=0.6,
                    )

            weights: list[float] = []
            weight_times: list[float] = []
            for i, global_pos in enumerate(range(start, end)):
                if global_pos in anchor_pos_sorted:
                    continue
                bt = local_btjd[i]
                if not np.isfinite(bt):
                    continue
                before = [
                    j for j, apos in enumerate(anchor_pos_sorted) if apos < global_pos
                ]
                after = [
                    j for j, apos in enumerate(anchor_pos_sorted) if apos > global_pos
                ]
                if before and after:
                    t_a, t_b = anchor_bt_sorted[before[-1]], anchor_bt_sorted[after[0]]
                    if np.isfinite(t_a) and np.isfinite(t_b) and t_b > t_a:
                        weights.append(float(np.clip((bt - t_a) / (t_b - t_a), 0.0, 1.0)))
                        weight_times.append(float(bt))
            if weights:
                order = np.argsort(weight_times)
                ax2 = ax.twinx()
                ax2.plot(
                    np.asarray(weight_times)[order],
                    np.asarray(weights)[order],
                    "-",
                    color="tab:green",
                    lw=1,
                    alpha=0.7,
                )
                ax2.set_ylabel("blend weight (anchor→anchor)", color="tab:green")
                ax2.set_ylim(-0.05, 1.05)

            ax.set_yticks([])
            ax.set_xlabel("BTJD")
            ax.set_title(
                f"epsf {epsf_label} orbit {orbit_idx}: {len(anchors)} anchors, "
                f"{int(good.sum())}/{len(good)} good frames"
            )
            ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    debug_dir, f"epsf_orbit_{orbit_idx:02d}_anchor_selection.png"
                ),
                dpi=130,
            )
            plt.close(fig)
    except Exception as exc:  # diagnostics must never invalidate science output
        log.warning("epsf orbit debug plots skipped: %s", exc)
