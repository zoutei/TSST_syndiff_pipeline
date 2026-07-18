"""SCC-scoped reference FFI selection for template mapping (no event RA/Dec)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.download import ffi_glob_patterns, list_local_ffis
from syndiff_pipeline.common.scc_paths import scc_bookkeeping_stage_dir, scc_wcs_cache_parquet
from syndiff_pipeline.common.wcs_header_cache import load_or_build_wcs_cache
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


def _median_crval_anchor(ffi_paths: list[str]) -> tuple[float, float]:
    """Return median CRVAL across usable FFIs (chip-center sky anchor)."""
    rvals: list[float] = []
    dvals: list[float] = []
    for ffi_path in ffi_paths:
        info = wcs_grouping.extract_wcs_from_ffi(ffi_path)
        if not info.get("wcs_ok"):
            continue
        hdr = info["header"]
        try:
            rvals.append(float(hdr["CRVAL1"]))
            dvals.append(float(hdr["CRVAL2"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not rvals:
        raise RuntimeError("No usable WCS headers for SCC median-CRVAL anchor")
    return float(np.median(rvals)), float(np.median(dvals))


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

    Returns the absolute path to the reference FFI FITS file.
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
            if ref and wcs_grouping.fits_path_exists(ref):
                log.info("Using cached SCC reference FFI from %s", meta_path)
                return os.path.abspath(ref)

    ffi_paths = sorted(list_local_ffis(resolved.ffi_dir, t.sector, t.camera, t.ccd))
    if not ffi_paths:
        patterns = ffi_glob_patterns(t.sector, t.camera, t.ccd)
        raise FileNotFoundError(
            f"No FFI files matching {patterns!r} under {resolved.ffi_dir!r}"
        )

    cache_path = scc_wcs_cache_parquet(resolved.data_root, t.sector, t.camera, t.ccd)
    load_or_build_wcs_cache(
        ffi_paths,
        cache_path,
        open_fits=wcs_grouping.open_fits_memmap,
    )

    if explicit:
        ref = _resolve_existing_ffi_path(resolved, explicit)
        if not ref:
            raise FileNotFoundError(f"reference_ffi override not found: {explicit!r}")
        selection_rule = "override"
    else:
        anchor_ra, anchor_dec = _median_crval_anchor(ffi_paths)
        wcs_table = wcs_grouping.build_wcs_table(ffi_paths, anchor_ra, anchor_dec)
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
        )
        selection_rule = "scc_median_crval_anchor"

    ref_abs = os.path.abspath(ref)
    payload = {
        "reference_ffi_path": ref_abs,
        "reference_ffi_basename": os.path.basename(ref_abs),
        "selection_rule": selection_rule,
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
    """Load persisted mapping reference FFI path, or ``None`` if absent."""
    cached = _read_run_meta(mapping_run_meta_path(resolved))
    if not cached:
        return None
    ref = str(cached.get("reference_ffi_path") or "").strip()
    if ref and wcs_grouping.fits_path_exists(ref):
        return os.path.abspath(ref)
    return None


def _resolve_existing_ffi_path(resolved: ResolvedTargetConfig, value: str) -> str | None:
    """Resolve a basename or absolute path to an on-disk FFI."""
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    from syndiff_pipeline.common.download import resolve_local_ffi_path

    local = resolve_local_ffi_path(resolved.ffi_dir, raw)
    return local
