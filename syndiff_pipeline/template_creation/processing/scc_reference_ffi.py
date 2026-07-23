"""SCC-scoped reference FFI selection for template mapping (no event RA/Dec)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.download import (
    ffi_glob_patterns,
    list_local_ffis,
    manifest_basename_from_local,
)
from syndiff_pipeline.common.scc_paths import (
    scc_bookkeeping_stage_dir,
    scc_debug_plots_dir,
    scc_ffi_list_parquet,
)
from syndiff_pipeline.common.wcs_header_cache import (
    ensure_scc_ffi_list,
    ffi_list_is_complete,
    header_from_cached_row,
    load_ffi_list,
    median_crval_from_cache,
    wcs_from_cached_row,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig

log = logging.getLogger(__name__)

RUN_META_FILENAME = "run_meta.json"
POINT_DRIFT_TABLE_BASENAME = "point_drift_table.csv"
POINT_DRIFT_GROUPS_BASENAME = "point_drift_groups.csv"
POINT_DRIFT_META_BASENAME = "point_drift_meta.json"


def scc_wcs_drift_debug_plot_path(resolved: ResolvedTargetConfig) -> Path:
    """SCC ``debug_plots/wcs_drift_linear_template.png`` (point-drift owners write this)."""
    t = resolved.target
    return (
        scc_debug_plots_dir(resolved.data_root, t.sector, t.camera, t.ccd)
        / wcs_grouping.WCS_DRIFT_LINEAR_TEMPLATE_FILENAME
    )


def _screen_earth_moon_angles(resolved: ResolvedTargetConfig) -> bool:
    mp = resolved.stages.mapping
    wg = resolved.stages.wcs_grouping
    return bool(mp.screen_earth_moon_angles or wg.screen_earth_moon_angles)


def _bkg_vector_path(resolved: ResolvedTargetConfig) -> str | None:
    mp = resolved.stages.mapping
    wg = resolved.stages.wcs_grouping
    path = (mp.bkg_vector_path or wg.bkg_vector_path or "").strip()
    return path or None


def _maybe_attach_earth_moon_angles(
    wcs_table: pd.DataFrame,
    resolved: ResolvedTargetConfig,
) -> pd.DataFrame:
    if not _screen_earth_moon_angles(resolved):
        return wcs_table
    bkg_path = _bkg_vector_path(resolved)
    if not bkg_path:
        log.warning(
            "screen_earth_moon_angles enabled but bkg_vector_path unset; "
            "skipping TESSVectors attach"
        )
        return wcs_table
    t = resolved.target
    return wcs_grouping.attach_tessvector_earth_moon_angles(
        wcs_table,
        sector=t.sector,
        camera=t.camera,
        tessvectors_data_path=bkg_path,
    )


def _load_scc_ffi_context(
    resolved: ResolvedTargetConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Sorted local FFI paths + complete ``ffi_list`` parquet dataframe."""
    t = resolved.target
    ffi_paths = sorted(list_local_ffis(resolved.ffi_dir, t.sector, t.camera, t.ccd))
    if not ffi_paths:
        patterns = ffi_glob_patterns(t.sector, t.camera, t.ccd)
        raise FileNotFoundError(
            f"No FFI files matching {patterns!r} under {resolved.ffi_dir!r}"
        )

    ffi_list_path = scc_ffi_list_parquet(resolved.data_root, t.sector, t.camera, t.ccd)
    ffi_list_df = load_ffi_list(ffi_list_path)
    if not ffi_list_is_complete(ffi_paths, ffi_list_df):
        log.info("FFI list missing/incomplete (%s); backfilling ...", ffi_list_path)
        t0 = time.monotonic()
        ffi_list_df = ensure_scc_ffi_list(
            resolved.data_root,
            t.sector,
            t.camera,
            t.ccd,
            ffi_paths,
            open_fits=wcs_grouping.open_fits_memmap,
        )
        log.info("FFI list ensure finished in %.1fs", time.monotonic() - t0)
    return ffi_paths, ffi_list_df


def _build_smoothed_wcs_table(
    resolved: ResolvedTargetConfig,
    ffi_list_df: pd.DataFrame,
    ffi_paths: list[str],
    anchor_ra: float,
    anchor_dec: float,
) -> pd.DataFrame:
    """Build WCS table from cache, SG-smooth, optionally attach Earth/Moon angles."""
    mp = resolved.stages.mapping
    t0 = time.monotonic()
    wcs_table = wcs_grouping.build_wcs_table_from_cache(
        ffi_list_df, ffi_paths, float(anchor_ra), float(anchor_dec)
    )
    log.info("WCS table from ffi_list built in %.2fs", time.monotonic() - t0)
    wcs_table = wcs_grouping.smooth_wcs_drift_savgol(
        wcs_table,
        window_length=mp.wcs_drift_savgol_window,
        polyorder=mp.wcs_drift_savgol_polyorder,
    )
    return _maybe_attach_earth_moon_angles(wcs_table, resolved)


def mapping_run_meta_path(resolved: ResolvedTargetConfig) -> Path:
    """Path to persisted mapping reference-FFI bookkeeping for one SCC."""
    t = resolved.target
    os_factor = int(resolved.stages.mapping.oversampling_factor)
    base = scc_bookkeeping_stage_dir(
        resolved.data_root, t.sector, t.camera, t.ccd, "mapping"
    )
    if os_factor != 1:
        base = base / f"oversampling_{os_factor}"
    return base / RUN_META_FILENAME


def _write_run_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_run_meta(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read mapping run_meta %s: %s", path, exc)
        return None


def resolve_scc_reference_ffi(
    resolved: ResolvedTargetConfig,
    *,
    force_rerun: bool = False,
    override_path: str | None = None,
    write_debug_plot: bool = False,
) -> str:
    """
    Choose the mapping-epoch reference FFI for one SCC and persist bookkeeping.

    Returns the absolute path to the on-disk reference FFI (``.fits``,
    ``.fits.gz``, or ``.fits.fz`` — whichever exists).

    By default this does **not** write ``wcs_drift_linear_template.png``; that
    PNG is owned by point-drift callers (linear downsample / remap
    ``drift_source: point``). Pass ``write_debug_plot=True`` only for legacy
    callers that still want a median-CRVAL diagnostic plot.
    """
    t = resolved.target
    meta_path = mapping_run_meta_path(resolved)
    mp = resolved.stages.mapping

    explicit = (override_path or mp.reference_ffi or "").strip()
    if explicit and not force_rerun:
        ref = _resolve_existing_ffi_path(resolved, explicit)
        if ref:
            if write_debug_plot:
                write_scc_wcs_drift_debug_plot(resolved, ref, force_rerun=False)
            return ref

    if not force_rerun:
        cached = _read_run_meta(meta_path)
        if cached:
            ref = str(cached.get("reference_ffi_path") or "").strip()
            resolved_ref = _resolve_existing_ffi_path(resolved, ref) if ref else None
            if resolved_ref:
                if os.path.abspath(resolved_ref) != os.path.abspath(ref):
                    log.info(
                        "Cached SCC reference FFI remapped %s → %s",
                        ref,
                        resolved_ref,
                    )
                    payload = dict(cached)
                    payload["reference_ffi_path"] = resolved_ref
                    payload["reference_ffi_basename"] = os.path.basename(resolved_ref)
                    _write_run_meta(meta_path, payload)
                else:
                    log.info("Using cached SCC reference FFI from %s", meta_path)
                if write_debug_plot:
                    write_scc_wcs_drift_debug_plot(
                        resolved, resolved_ref, force_rerun=False
                    )
                return resolved_ref

    ffi_paths, ffi_list_df = _load_scc_ffi_context(resolved)

    if explicit:
        ref = _resolve_existing_ffi_path(resolved, explicit)
        if not ref:
            raise FileNotFoundError(f"reference_ffi override not found: {explicit!r}")
        selection_rule = "override"
    else:
        t0 = time.monotonic()
        anchor_ra, anchor_dec = median_crval_from_cache(ffi_list_df, ffi_paths)
        log.info(
            "SCC median-CRVAL anchor (%.4f, %.4f) in %.2fs",
            anchor_ra,
            anchor_dec,
            time.monotonic() - t0,
        )
        wcs_table = _build_smoothed_wcs_table(
            resolved, ffi_list_df, ffi_paths, anchor_ra, anchor_dec
        )
        ref = wcs_grouping.choose_reference_ffi_path(
            wcs_table,
            earth_deg_min=float(mp.earth_deg_min),
            moon_deg_min=float(mp.moon_deg_min),
            max_smoothed_residual=float(mp.max_smoothed_residual),
            selection_mode=str(mp.reference_ffi_selection or "drift_arc_midpoint"),
            screen_earth_moon_angles=_screen_earth_moon_angles(resolved),
        )
        selection_mode = str(mp.reference_ffi_selection or "drift_arc_midpoint").strip().lower()
        if selection_mode == "drift_arc_midpoint":
            selection_rule = "scc_drift_arc_midpoint"
        else:
            selection_rule = "scc_median_crval_anchor"

    ref_resolved = _resolve_existing_ffi_path(resolved, ref) or os.path.abspath(ref)
    ref_abs = os.path.abspath(ref_resolved)
    payload = {
        "reference_ffi_path": ref_abs,
        "reference_ffi_basename": os.path.basename(ref_abs),
        "selection_rule": selection_rule,
        "reference_ffi_selection": str(mp.reference_ffi_selection or "drift_arc_midpoint"),
        "oversampling_factor": int(mp.oversampling_factor),
        "sector": t.sector,
        "camera": t.camera,
        "ccd": t.ccd,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_run_meta(meta_path, payload)
    log.info("SCC reference FFI (%s): %s", selection_rule, ref_abs)
    if write_debug_plot:
        write_scc_wcs_drift_debug_plot(resolved, ref_abs, force_rerun=force_rerun)
    return ref_abs


def write_scc_wcs_drift_debug_plot(
    resolved: ResolvedTargetConfig,
    ref_ffi_path: str,
    *,
    wcs_table: pd.DataFrame | None = None,
    force_rerun: bool = False,
) -> Path | None:
    """
    Write the WCS drift / template-group debug plot.

    Preferred callers pass a point-drift ``wcs_table`` that already has
    ``group_id`` (ref-FFI-center anchor). Without ``wcs_table``, falls back to
    a median-CRVAL rebuild for legacy/compat use.
    """
    plot_path = scc_wcs_drift_debug_plot_path(resolved)
    # Point-drift callers pass wcs_table and always overwrite so a stale
    # median-CRVAL PNG (from older mapping runs) cannot stick around.
    # Legacy path (no table): skip if PNG exists unless force_rerun.
    if wcs_table is None and plot_path.is_file() and not force_rerun:
        return plot_path

    t = resolved.target
    mp = resolved.stages.mapping
    wg = resolved.stages.wcs_grouping

    if wcs_table is not None:
        table = wcs_table
        if "group_id" not in table.columns:
            table, _chosen = wcs_grouping.finalize_wcs_table_with_reference_anchor(
                table,
                offset_threshold=float(wg.offset_threshold or 0.01),
                ref_ffi_path=ref_ffi_path,
                ref_earth_deg_min=float(mp.earth_deg_min),
                ref_moon_deg_min=float(mp.moon_deg_min),
                ref_max_smoothed_residual=float(mp.max_smoothed_residual),
                screen_earth_moon_angles=_screen_earth_moon_angles(resolved),
            )
    else:
        ffi_paths, ffi_list_df = _load_scc_ffi_context(resolved)
        anchor_ra, anchor_dec = median_crval_from_cache(ffi_list_df, ffi_paths)
        table = _build_smoothed_wcs_table(
            resolved, ffi_list_df, ffi_paths, anchor_ra, anchor_dec
        )
        table, _chosen = wcs_grouping.finalize_wcs_table_with_reference_anchor(
            table,
            offset_threshold=float(wg.offset_threshold or 0.01),
            ref_ffi_path=ref_ffi_path,
            ref_earth_deg_min=float(mp.earth_deg_min),
            ref_moon_deg_min=float(mp.moon_deg_min),
            ref_max_smoothed_residual=float(mp.max_smoothed_residual),
            screen_earth_moon_angles=_screen_earth_moon_angles(resolved),
        )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    wcs_grouping.plot_wcs_drift_and_template_assignment(
        table,
        str(plot_path),
        ref_ffi_path=ref_ffi_path,
        sector=t.sector,
        camera=t.camera,
        ccd=t.ccd,
        target_name=t.target_name,
        include_earth_moon_panel=_screen_earth_moon_angles(resolved),
    )
    log.info("SCC WCS drift debug plot: %s", plot_path)
    return plot_path


def _point_drift_meta_path(store_root: Path) -> Path:
    return store_root / POINT_DRIFT_META_BASENAME


def _point_drift_table_path(store_root: Path) -> Path:
    return store_root / POINT_DRIFT_TABLE_BASENAME


def _point_drift_groups_path(store_root: Path) -> Path:
    return store_root / POINT_DRIFT_GROUPS_BASENAME


def _point_drift_dict_from_table(wcs_table: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Map manifest filename → ``(delta_x, delta_y)`` from a point-drift table."""
    out: dict[str, tuple[float, float]] = {}
    path_col = "filename" if "filename" in wcs_table.columns else "path"
    for _, row in wcs_table.iterrows():
        key = str(row.get(path_col) or "").strip()
        if not key:
            continue
        dx = row.get("delta_x")
        dy = row.get("delta_y")
        if pd.isna(dx) or pd.isna(dy):
            continue
        out[key] = (float(dx), float(dy))
    return out


def resolve_scc_point_drift_table(
    resolved: ResolvedTargetConfig,
    *,
    ref_ffi_path: str,
    store_root: str | Path,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build (or load) an SCC FFI-center point-drift table for ``drift_source: point``.

    The anchor is the center pixel of the **already-resolved** reference FFI
    (not re-selected here). Returns the full ``wcs_table`` and a
    ``(n_frames, 2)`` ``target_drift`` array in the same FFI sort order used by
    ``_build_shift_schedule_for_scc``.
    """
    t = resolved.target
    wg = resolved.stages.wcs_grouping
    store = Path(store_root)
    store.mkdir(parents=True, exist_ok=True)
    meta_path = _point_drift_meta_path(store)
    table_path = _point_drift_table_path(store)

    if not force_rerun and meta_path.is_file() and table_path.is_file():
        try:
            with meta_path.open(encoding="utf-8") as fh:
                cached_meta = json.load(fh)
            cached_ref = str(cached_meta.get("reference_ffi_path") or "").strip()
            if cached_ref and os.path.abspath(cached_ref) == os.path.abspath(ref_ffi_path):
                wcs_table = pd.read_csv(table_path)
                drift = _target_drift_array_for_ffi_order(
                    wcs_table,
                    sorted(
                        list_local_ffis(resolved.ffi_dir, t.sector, t.camera, t.ccd)
                    ),
                )
                log.info("Using cached SCC point-drift table from %s", table_path)
                return wcs_table, drift
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            log.warning("Could not load cached point-drift table: %s", exc)

    ffi_paths, ffi_list_df = _load_scc_ffi_context(resolved)

    ref_logical = manifest_basename_from_local(ref_ffi_path)
    if ref_logical not in ffi_list_df.index:
        raise ValueError(f"Reference FFI {ref_logical!r} missing from ffi_list cache")
    ref_row = ffi_list_df.loc[ref_logical]
    if not bool(ref_row.get("wcs_ok", False)):
        raise ValueError(f"Reference FFI has invalid WCS in ffi_list: {ref_logical!r}")
    ref_header = header_from_cached_row(ref_row)
    ref_wcs = wcs_from_cached_row(ref_row)
    naxis1 = int(ref_header["NAXIS1"])
    naxis2 = int(ref_header["NAXIS2"])
    anchor_x = naxis1 / 2.0
    anchor_y = naxis2 / 2.0
    anchor_ra, anchor_dec = ref_wcs.pixel_to_world_values(anchor_x, anchor_y)

    wcs_table = _build_smoothed_wcs_table(
        resolved, ffi_list_df, ffi_paths, float(anchor_ra), float(anchor_dec)
    )
    wcs_table = wcs_grouping.reanchor_wcs_drift_to_reference(wcs_table, ref_ffi_path)
    offset_threshold = float(wg.offset_threshold or 0.01)
    wcs_table = wcs_grouping.assign_template_groups(wcs_table, offset_threshold)
    summary = wcs_grouping.summarize_template_groups(wcs_table)

    wcs_table.to_csv(table_path, index=False)
    summary.to_csv(_point_drift_groups_path(store), index=False)
    meta_payload = {
        "reference_ffi_path": os.path.abspath(ref_ffi_path),
        "reference_ffi_basename": os.path.basename(ref_ffi_path),
        "anchor_pixel": [anchor_x, anchor_y],
        "anchor_ra_deg": float(anchor_ra),
        "anchor_dec_deg": float(anchor_dec),
        "offset_threshold": offset_threshold,
        "n_groups": int(len(summary)),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_run_meta(meta_path, meta_payload)
    log.info(
        "SCC point-drift table: %d frames, %d groups",
        len(wcs_table),
        len(summary),
    )
    target_drift = _target_drift_array_for_ffi_order(wcs_table, ffi_paths)
    return wcs_table, target_drift


def _target_drift_array_for_ffi_order(
    wcs_table: pd.DataFrame, ffi_paths: list[str]
) -> np.ndarray:
    """Build ``(n_frames, 2)`` drift array aligned to sorted local FFI paths."""
    drift_by_key = _point_drift_dict_from_table(wcs_table)
    out = np.full((len(ffi_paths), 2), np.nan, dtype=np.float64)
    for i, p in enumerate(ffi_paths):
        logical = manifest_basename_from_local(p)
        hit = drift_by_key.get(logical)
        if hit is not None:
            out[i, 0] = hit[0]
            out[i, 1] = hit[1]
    return out


def load_mapping_reference_ffi(resolved: ResolvedTargetConfig) -> str | None:
    """Load persisted mapping reference FFI path, or ``None`` if absent.

    Accepts bookkeeping that still names ``.fits.gz`` / ``.fits`` after a
    migration to ``.fits.fz`` (or the reverse).
    """
    cached = _read_run_meta(mapping_run_meta_path(resolved))
    if not cached:
        return None
    ref = str(cached.get("reference_ffi_path") or "").strip()
    if not ref:
        return None
    return _resolve_existing_ffi_path(resolved, ref)


def resolve_cached_or_select_reference_ffi(
    resolved: ResolvedTargetConfig,
) -> str:
    """Load mapping ref FFI from cache, or select once without writing the debug PNG.

    Downstream stages (linear downsample, remap ``drift_source: point``) must
    use this instead of ``resolve_scc_reference_ffi(..., force_rerun=...)`` so
    stage ``--force-rerun`` cannot reselect the mapping reference FFI.
    """
    ref = load_mapping_reference_ffi(resolved)
    if ref:
        return ref
    return resolve_scc_reference_ffi(
        resolved, force_rerun=False, write_debug_plot=False
    )


def _resolve_existing_ffi_path(resolved: ResolvedTargetConfig, value: str) -> str | None:
    """Resolve a basename or absolute path to an on-disk FFI (any storage suffix)."""
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    resolved_variant = wcs_grouping.try_resolve_existing_fits_path(candidate)
    if resolved_variant is not None:
        return str(resolved_variant.resolve())
    from syndiff_pipeline.common.download import resolve_local_ffi_path

    local = resolve_local_ffi_path(resolved.ffi_dir, raw)
    if local:
        return local
    # Basename-only: try stem lookup under ffi_dir via variant helper.
    local_variant = wcs_grouping.try_resolve_existing_fits_path(
        Path(resolved.ffi_dir) / Path(raw).name
    )
    if local_variant is not None:
        return str(local_variant.resolve())
    return None
