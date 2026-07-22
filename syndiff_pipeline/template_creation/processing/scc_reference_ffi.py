"""SCC-scoped reference FFI selection for template mapping (no event RA/Dec)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.download import ffi_glob_patterns, list_local_ffis
from syndiff_pipeline.common.scc_paths import scc_bookkeeping_stage_dir, scc_ffi_list_parquet
from syndiff_pipeline.common.wcs_header_cache import (
    ensure_scc_ffi_list,
    ffi_list_is_complete,
    load_ffi_list,
    median_crval_from_cache,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig

log = logging.getLogger(__name__)

RUN_META_FILENAME = "run_meta.json"


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
) -> str:
    """
    Choose the mapping-epoch reference FFI for one SCC and persist bookkeeping.

    Returns the absolute path to the on-disk reference FFI (``.fits``,
    ``.fits.gz``, or ``.fits.fz`` — whichever exists).
    """
    t = resolved.target
    meta_path = mapping_run_meta_path(resolved)
    mp = resolved.stages.mapping

    explicit = (override_path or mp.reference_ffi or "").strip()
    if explicit and not force_rerun:
        ref = _resolve_existing_ffi_path(resolved, explicit)
        if ref:
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
                return resolved_ref

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
        t0 = time.monotonic()
        wcs_table = wcs_grouping.build_wcs_table_from_cache(
            ffi_list_df, ffi_paths, anchor_ra, anchor_dec
        )
        log.info("WCS table from ffi_list built in %.2fs", time.monotonic() - t0)
        wcs_table = wcs_grouping.smooth_wcs_drift_savgol(
            wcs_table,
            window_length=mp.wcs_drift_savgol_window,
            polyorder=mp.wcs_drift_savgol_polyorder,
        )
        bkg_path = mp.bkg_vector_path
        if bkg_path:
            wcs_table = wcs_grouping.attach_tessvector_earth_moon_angles(
                wcs_table,
                sector=t.sector,
                camera=t.camera,
                tessvectors_data_path=bkg_path,
            )
        ref = wcs_grouping.choose_reference_ffi_path(
            wcs_table,
            earth_deg_min=float(mp.earth_deg_min),
            moon_deg_min=float(mp.moon_deg_min),
            max_smoothed_residual=float(mp.max_smoothed_residual),
            selection_mode=str(mp.reference_ffi_selection or "drift_arc_midpoint"),
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
    return ref_abs


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
