"""Artifact verification for template pipeline stages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.fits_variants import (
    FITS_STORAGE_SUFFIXES,
    strip_fits_storage_suffix,
    try_resolve_fits_variant,
)
from syndiff_pipeline.common.scc_paths import (
    ps1_convolved_zarr_path,
    ps1_skycells_zarr_path,
    scc_convolved_zarr,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import PIPELINE_FITS_EXT
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig, resolve_config, RunnerConfig
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    CHECKPOINT_STAGES,
    checkpoint_stage_indexed,
)
from syndiff_pipeline.common.orchestration.targets import Target

log = logging.getLogger(__name__)

# Bump when the manifest JSON schema changes; a mismatch invalidates a manifest.
MANIFEST_SCHEMA_VERSION = 2

# Removed L4b-lite policies; manifests/sidecars claiming these must be rebuilt.
_DEPRECATED_L4B_POLICIES = frozenset(
    {"lite", "abutting_under_type1_wcs", "abutting_border"}
)


class AbsenceProbeResult(Enum):
    """Fast pre-check before scheduling a full background artifact verify."""

    ABSENT = "absent"
    MAYBE_PRESENT = "maybe"
    UNKNOWN = "unknown"


@dataclass
class VerifyResult:
    """VerifyResult."""
    stage: str
    ok: bool
    message: str
    path: str | None = None
    # Tri-state marker: True when completeness cannot be determined (e.g. a
    # required external manifest is unavailable). ``unknown`` results have
    # ``ok=False`` but callers may choose not to force a needless rerun.
    unknown: bool = False


# ---------------------------------------------------------------------------
# Completion manifests
#
# verify.py is intentionally decoupled from the run directory layout: the caller
# passes ``manifest_path`` explicitly. We never import the run-layout module here.
# ---------------------------------------------------------------------------


def _diff_stage_context(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig | None = None,
    *,
    meta: dict | None = None,
) -> "StageRunContext":
    """Diff stage context.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    runner_cfg : RunnerConfig | None, optional, default ``None``
    meta : dict | None, optional, default ``None``
    
    Returns
    -------
    'StageRunContext'"""
    from syndiff_pipeline.common.orchestration.spec import StageRunContext

    cfg = runner_cfg
    if cfg is None:
        raise ValueError("diff stage verification requires RunnerConfig")
    return StageRunContext(
        run_id="",
        runs_root="",
        target_label=resolved.target.label(),
        target=resolved.target,
        runner_cfg=cfg,
        meta=dict(meta or {}),
    )


def diff_config_fingerprint(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig,
    *,
    meta: dict | None = None,
) -> str:
    """Diff config fingerprint.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    runner_cfg : RunnerConfig
    meta : dict | None, optional, default ``None``
    
    Returns
    -------
    str"""
    from syndiff_pipeline.difference_imaging.orchestration.stages import _diff_config_fingerprint

    return _diff_config_fingerprint(_diff_stage_context(resolved, runner_cfg, meta=meta))


def config_fingerprint(
    resolved: ResolvedTargetConfig,
    stage: str,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> str:
    """Stable hash of the stage params that affect this stage's outputs."""
    if stage == "diff":
        if runner_cfg is None:
            raise ValueError("diff config fingerprint requires RunnerConfig")
        return diff_config_fingerprint(resolved, runner_cfg, meta=meta)
    parts: list[str] = [stage]
    t = resolved.target
    parts.extend([str(t.sector), str(t.camera), str(t.ccd)])
    if stage == "mapping":
        mp = resolved.stages.mapping
        parts.extend([str(mp.oversampling_factor), str(mp.pad_distance), str(mp.overwrite)])
    elif stage == "ps1_process":
        pp = resolved.stages.ps1_process
        parts.extend(
            [
                str(pp.projections_limit),
                str(pp.psf_sigma),
                str(pp.enable_saturation_correction),
                str(pp.remove_saturated_stars),
                str(pp.bright_star_mag_threshold),
            ]
        )
    elif stage == "remap":
        rm = resolved.stages.remap
        mp = resolved.stages.mapping
        parts.extend(
            [
                str(mp.oversampling_factor),
                str(rm.cache_quantum_ps1_px),
                str(rm.keying),
                str(rm.intra_skycell_R),
                str(rm.store_name or ""),
            ]
        )
    elif stage == "downsample":
        ds = resolved.stages.downsample
        parts.extend(
            [
                str(ds.oversampling_factor),
                str(ds.single_offset),
                str(list(ds.ignore_mask_bits)),
                str(ds.output_base or resolved.template_output_base),
                str(ds.output_store_name or ""),
                str(resolved.downsample_remap_store_name or ""),
                str(bool(ds.apply_intra_skycell)),
                str(bool(ds.apply_inter_skycell)),
            ]
        )
    elif stage == "ps1_download":
        pd = resolved.stages.ps1_download
        parts.extend([str(pd.overwrite), str(pd.use_local_files)])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _downsample_manifest_meta(
    resolved: ResolvedTargetConfig,
    meta: dict | None,
) -> dict | None:
    """Downsample manifest meta.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    meta : dict | None
    
    Returns
    -------
    dict | None"""
    if meta and "template_dir_physical" in meta and "template_dir_symlink" in meta:
        return meta
    from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
        template_dir_meta_from_event_dir,
    )

    derived = template_dir_meta_from_event_dir(resolved.event_dir)
    if derived:
        return {**(meta or {}), **derived}
    return meta


def write_manifest(
    manifest_path,
    resolved: ResolvedTargetConfig,
    stage: str,
    produced_paths,
    expected_count: int,
    produced_count: int,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> dict:
    """Atomically write a completion manifest (tmp file + rename).

    Schema: schema_version, stage, expected_count, produced_count, artifacts
    (list of paths), config_fingerprint, completed_at (iso utc).
    """
    if stage == "downsample":
        meta = _downsample_manifest_meta(resolved, meta)
    path = Path(manifest_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "expected_count": int(expected_count),
        "produced_count": int(produced_count),
        "artifacts": [str(p) for p in (produced_paths or [])],
        "config_fingerprint": config_fingerprint(
            resolved, stage, runner_cfg=runner_cfg, meta=meta
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if meta:
        for key in ("template_dir_physical", "template_dir_symlink"):
            if key in meta:
                payload[key] = str(meta[key])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return payload


def read_manifest(manifest_path) -> dict | None:
    """Read a manifest JSON, returning None if absent or malformed."""
    if manifest_path is None:
        return None
    path = Path(manifest_path)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest_valid(
    manifest: dict,
    resolved: ResolvedTargetConfig,
    stage: str,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> bool:
    """True if *manifest* is well-formed, matches the current config, and all
    listed artifacts still exist on disk."""
    if not isinstance(manifest, dict):
        return False
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    if manifest.get("stage") != stage:
        return False
    if manifest.get("config_fingerprint") != config_fingerprint(
        resolved, stage, runner_cfg=runner_cfg, meta=meta
    ):
        return False
    expected = manifest.get("expected_count")
    produced = manifest.get("produced_count")
    if not isinstance(expected, int) or not isinstance(produced, int):
        return False
    if expected > 0 and produced < expected:
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        if not Path(str(artifact)).exists():
            return False
    return True


# Backward-compatible aliases (older call sites used these names).
stage_config_fingerprint = config_fingerprint
read_stage_manifest = read_manifest


def check_manifests_only(
    resolved: ResolvedTargetConfig,
    stage: str,
    *,
    manifest_path: str | Path | None = None,
    stable_manifest_path: str | Path | None = None,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> bool | None:
    """Fast manifest check without on-disk artifact scanning.

    Returns ``True`` when a valid manifest proves completeness, ``False`` when
    manifests exist but do not prove completeness, and ``None`` when no
    manifest was found (full verify required).
    """
    saw_manifest = False
    for candidate in (manifest_path, stable_manifest_path):
        if candidate is None:
            continue
        manifest = read_manifest(candidate)
        if manifest is None:
            continue
        saw_manifest = True
        if manifest_valid(
            manifest, resolved, stage, runner_cfg=runner_cfg, meta=meta
        ):
            return True
    if saw_manifest:
        return False
    return None


def copy_manifest_to_stable(
    source_manifest_path: str | Path,
    stable_manifest_path: str | Path,
) -> bool:
    """Atomically copy a per-run manifest to the stable cross-run path."""
    manifest = read_manifest(source_manifest_path)
    if manifest is None:
        return False
    dest = Path(stable_manifest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return True


def write_stable_manifest(
    resolved: ResolvedTargetConfig,
    stage: str,
    stable_manifest_path: str | Path,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> None:
    """Collect artifacts and write the stable under-runs-root manifest."""
    stable_path = Path(stable_manifest_path)
    existing = read_manifest(stable_path)
    if existing is not None and manifest_valid(
        existing, resolved, stage, runner_cfg=runner_cfg, meta=meta
    ):
        return
    expected, produced, artifacts = collect_stage_artifacts(
        resolved, stage, runner_cfg=runner_cfg, meta=meta
    )
    write_manifest(
        stable_path,
        resolved,
        stage,
        artifacts,
        expected,
        produced,
        runner_cfg=runner_cfg,
        meta=meta,
    )


def verify_tess_ffi_download(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify tess ffi download.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    VerifyResult"""
    from syndiff_pipeline.common.download import (
        expected_ffi_basenames,
        list_local_ffis,
        local_ffi_manifest_basenames,
    )

    t = resolved.target
    # resolve_config sets ffi_dir to the SCC ffi leaf already.
    ffi_leaf = resolved.ffi_dir
    local_files = list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd)
    expected = expected_ffi_basenames(
        t.sector, t.camera, t.ccd, output_dir=ffi_leaf, local_only=True
    )
    if expected is None:
        if not local_files:
            return VerifyResult(
                "tess_ffi_download",
                False,
                "No FFI files found and tesscurl manifest unavailable",
                ffi_leaf,
                unknown=True,
            )
        return VerifyResult(
            "tess_ffi_download",
            False,
            f"Cannot verify completeness ({len(local_files)} local files; tesscurl manifest unavailable)",
            ffi_leaf,
            unknown=True,
        )
    if not expected:
        return VerifyResult(
            "tess_ffi_download",
            False,
            "tesscurl manifest has no FFIs for this SCC",
            ffi_leaf,
        )

    existing = local_ffi_manifest_basenames(local_files)
    missing = [bn for bn in expected if bn not in existing]
    if missing:
        return VerifyResult(
            "tess_ffi_download",
            False,
            f"Partial FFI download: {len(existing)}/{len(expected)} files ({len(missing)} missing)",
            ffi_leaf,
        )
    return VerifyResult(
        "tess_ffi_download",
        True,
        f"All {len(expected)} FFI files present",
        ffi_leaf,
    )


def verify_wcs_grouping(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify wcs grouping.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    VerifyResult"""
    job_path = Path(resolved.event_dir) / "cluster_template_job.json"
    if not job_path.is_file():
        return VerifyResult("wcs_grouping", False, "Missing cluster_template_job.json", str(job_path))
    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return VerifyResult("wcs_grouping", False, f"Invalid JSON: {exc}", str(job_path))
    ref = payload.get("reference_ffi_path")
    if not ref or not wcs_grouping.fits_path_exists(ref):
        return VerifyResult("wcs_grouping", False, "reference_ffi_path missing or not found", str(job_path))
    return VerifyResult("wcs_grouping", True, "Valid cluster_template_job.json", str(job_path))


def verify_mapping(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify mapping artifacts under the SCC oversampling leaf."""
    from syndiff_pipeline.common.mapping_grid import (
        MappingGridError,
        load_mapping_grid_from_master,
    )

    t = resolved.target
    suffix = ""
    os_factor = resolved.stages.mapping.oversampling_factor
    mapping_root = Path(resolved.mapping_root)
    if os_factor > 1:
        suffix = f"_os{os_factor}"
    # Prefer flat SCC leaf (post-migration); fall back to legacy nested layout.
    csv_flat = (
        mapping_root
        / f"tess_s{t.sector:04d}_{t.camera}_{t.ccd}_master_skycells_list{suffix}.csv"
    )
    csv_nested = (
        mapping_root
        / f"sector_{t.sector:04d}"
        / f"camera_{t.camera}"
        / f"ccd_{t.ccd}"
        / f"tess_s{t.sector:04d}_{t.camera}_{t.ccd}_master_skycells_list{suffix}.csv"
    )
    csv_path = csv_flat if csv_flat.is_file() else csv_nested
    if not csv_path.is_file():
        return VerifyResult("mapping", False, "Master skycells CSV missing", str(csv_flat))

    try:
        master_path = mapping_master_pixels2skycells_path(resolved)
    except FileNotFoundError as exc:
        return VerifyResult(
            "mapping",
            False,
            f"Master pixels2skycells FITS missing: {exc}",
            str(csv_path),
        )
    if not master_path.is_file():
        return VerifyResult(
            "mapping",
            False,
            f"Master pixels2skycells FITS missing: {master_path}",
            str(csv_path),
        )
    try:
        grid = load_mapping_grid_from_master(master_path)
    except MappingGridError as exc:
        return VerifyResult(
            "mapping",
            False,
            f"Master FITS is not MAPGRID v2 (rebuild mapping): {exc}",
            str(master_path),
        )
    return VerifyResult(
        "mapping",
        True,
        f"MAPGRID v2 master OK (shape={grid.array_shape_os()})",
        str(master_path),
    )


_PS1_DOWNLOAD_BANDS = ("r", "i", "z", "y")


def _ps1_download_expected_array_names() -> list[str]:
    """Ps1 download expected array names.
    
    Returns
    -------
    list[str]"""
    names: list[str] = []
    for band in _PS1_DOWNLOAD_BANDS:
        names.extend([band, f"{band}_mask", f"{band}_wt"])
    return names


def _projection_from_skycell_name(skycell_name: str) -> str | None:
    """Projection from skycell name.
    
    Parameters
    ----------
    skycell_name : str
    
    Returns
    -------
    str | None"""
    try:
        return skycell_name.split(".")[1]
    except (IndexError, AttributeError):
        return None


def _expected_ps1_download_skycells(resolved: ResolvedTargetConfig) -> list[str]:
    """Expected ps1 download skycells.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    list[str]"""
    from syndiff_pipeline.template_creation.processing.csv_utils import get_all_padding_cells, load_csv_data

    csv_path = _mapping_csv_path(resolved)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Master skycells CSV missing: {csv_path}")
    df = load_csv_data(str(csv_path))
    if "NAME" not in df.columns:
        raise ValueError(f"Master skycells CSV missing NAME column: {csv_path}")

    unique_skycells = sorted(df["NAME"].astype(str).unique())
    try:
        padding_map = get_all_padding_cells(str(csv_path), list(unique_skycells))
        padding_cells: set[str] = set()
        for cells in padding_map.values():
            padding_cells.update(cells)
        unique_skycells = sorted(set(unique_skycells) | padding_cells)
    except Exception as exc:
        log.warning("Could not load padding skycells for %s: %s", csv_path, exc)
    return unique_skycells


_ZARR_META_NAMES = frozenset({".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"})


def _zarr_array_has_chunks(array_dir: Path) -> bool:
    """True if a Zarr array directory contains at least one materialized chunk.

    Metadata-only and decompression-free: we never open the array or read a
    chunk's bytes. This is the fast on-disk proxy for the writer's
    ``ps1_download._array_complete_unlocked`` check (which exists + non-empty +
    one readable chunk). Supports the v3 layout (chunks under ``c/``) and the v2
    layout (chunk keys directly under the array dir alongside ``.zarray``).

    Reading a chunk's compressed bytes is what made verification take ~30 min on
    NFS; a directory listing is orders of magnitude cheaper.
    """
    # Fast path (Zarr v3): a single scandir of the chunk root. We avoid an extra
    # is_dir() stat because NFS metadata latency dominates this hot loop.
    try:
        with os.scandir(array_dir / "c") as it:
            return any(True for _ in it)
    except FileNotFoundError:
        pass  # No v3 chunk root; fall through to the v2 layout probe.
    except NotADirectoryError:
        return False
    except OSError:
        return False
    # Zarr v2 fallback: any non-metadata entry under the array dir is a chunk key.
    try:
        with os.scandir(array_dir) as it:
            return any(entry.name not in _ZARR_META_NAMES for entry in it)
    except OSError:
        return False


def _ps1_download_skycell_complete(zarr_path: Path, skycell_name: str) -> bool:
    """All expected PS1 arrays for *skycell_name* exist with chunks on disk.

    Mirrors ``ps1_download.skycell_array_status`` / ``expected_array_names``: the
    skycell is complete iff every band, mask, and weight array is present and has
    at least one chunk written. Pure filesystem metadata, no Zarr open.
    """
    projection_id = _projection_from_skycell_name(skycell_name)
    if not projection_id:
        return False
    skycell_dir = zarr_path / projection_id / skycell_name
    if not skycell_dir.is_dir():
        return False
    return all(
        _zarr_array_has_chunks(skycell_dir / name)
        for name in _ps1_download_expected_array_names()
    )


def verify_ps1_download(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify ps1 download.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    VerifyResult"""
    zarr_path = ps1_skycells_zarr_path(resolved.data_root)
    if not zarr_path.exists():
        return VerifyResult(
            "ps1_download",
            False,
            "Shared zarr store missing",
            str(zarr_path),
        )
    try:
        expected_skycells = _expected_ps1_download_skycells(resolved)
    except FileNotFoundError as exc:
        return VerifyResult("ps1_download", False, str(exc), str(zarr_path))
    except ValueError as exc:
        return VerifyResult("ps1_download", False, str(exc), str(zarr_path))

    started = time.monotonic()
    complete = sum(
        1 for skycell in expected_skycells if _ps1_download_skycell_complete(zarr_path, skycell)
    )
    elapsed = time.monotonic() - started
    log.info(
        "verify_ps1_download: %d/%d skycells complete in %.2fs (%s)",
        complete,
        len(expected_skycells),
        elapsed,
        zarr_path,
    )
    if complete < len(expected_skycells):
        return VerifyResult(
            "ps1_download",
            False,
            f"Partial PS1 zarr: {complete}/{len(expected_skycells)} skycells complete",
            str(zarr_path),
        )
    return VerifyResult(
        "ps1_download",
        True,
        f"PS1 zarr complete ({complete}/{len(expected_skycells)} skycells)",
        str(zarr_path),
    )


def _mapping_csv_path(resolved: ResolvedTargetConfig) -> Path:
    """Mapping CSV path under the SCC oversampling leaf (flat or legacy nested)."""
    t = resolved.target
    suffix = ""
    os_factor = resolved.stages.mapping.oversampling_factor
    mapping_root = Path(resolved.mapping_root)
    if os_factor > 1:
        suffix = f"_os{os_factor}"
    csv_flat = (
        mapping_root
        / f"tess_s{t.sector:04d}_{t.camera}_{t.ccd}_master_skycells_list{suffix}.csv"
    )
    if csv_flat.is_file() or not (
        mapping_root / f"sector_{t.sector:04d}" / f"camera_{t.camera}" / f"ccd_{t.ccd}"
    ).is_dir():
        return csv_flat
    return (
        mapping_root
        / f"sector_{t.sector:04d}"
        / f"camera_{t.camera}"
        / f"ccd_{t.ccd}"
        / f"tess_s{t.sector:04d}_{t.camera}_{t.ccd}_master_skycells_list{suffix}.csv"
    )


def _convolved_zarr_path(resolved: ResolvedTargetConfig) -> Path:
    """Per-SCC legacy convolved.zarr path (removed-stars CSV sibling)."""
    t = resolved.target
    return scc_convolved_zarr(resolved.data_root, t.sector, t.camera, t.ccd)


def ps1_process_uses_shared_convolved_store(resolved: ResolvedTargetConfig) -> bool:
    return bool(getattr(resolved.stages.ps1_process, "use_shared_convolved_store", False))


def resolve_ps1_process_checkpoint_location(resolved: ResolvedTargetConfig) -> Path:
    """Canonical ps1_process artifact root for checkpoints and verify."""
    if ps1_process_uses_shared_convolved_store(resolved):
        return ps1_convolved_zarr_path(resolved.data_root)
    return _convolved_zarr_path(resolved)


def resolve_downsample_convolved_dir(resolved: ResolvedTargetConfig) -> str:
    """Resolve convolved inputs for downsample (explicit override, else checkpoint location)."""
    ds = resolved.stages.downsample
    if ds.convolved_dir:
        return str(ds.convolved_dir)
    return str(resolve_ps1_process_checkpoint_location(resolved))


def ps1_process_removed_stars_csv_path(resolved: ResolvedTargetConfig) -> Path:
    """Ps1 process removed stars csv path.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    Path"""
    return Path(str(_convolved_zarr_path(resolved)).replace(".zarr", "_removed_stars.csv"))


def event_dir_ps1_removed_stars_csv_path(resolved: ResolvedTargetConfig) -> Path:
    """Event dir ps1 removed stars csv path.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    Path"""
    from syndiff_pipeline.template_creation.processing.downsample import (
        PS1_REMOVED_STARS_CSV_FILENAME,
    )

    return Path(resolved.event_dir) / PS1_REMOVED_STARS_CSV_FILENAME


def clear_downsample_event_artifacts(resolved: ResolvedTargetConfig) -> list[str]:
    """Remove event-dir artifacts written by the downsample stage.

    Never deletes the shared SCC ``field_templates/`` store. Field mode may
    rewrite event-local schedule/group sidecars on the next run; SCC contribs
    are reused unless ``stages.downsample.rebuild_field_store: true``.
    """
    removed: list[str] = []
    event = Path(resolved.event_dir)
    csv_path = event_dir_ps1_removed_stars_csv_path(resolved)
    candidates = [
        csv_path,
        event / "template_group_shifts.parquet",
        event / "template_group_shifts.json",
        event / "shift_schedule.npz",
        event / "shift_schedule.json",
    ]
    for path in candidates:
        if path.is_file():
            path.unlink()
            removed.append(str(path))
            log.info("Force rerun: removed file %s", path)
    return removed


def mapping_master_pixels2skycells_path(resolved: ResolvedTargetConfig) -> Path:
    """Path to mapping's master TESS→skycell FITS for this SCC."""
    from syndiff_pipeline.template_creation.processing.field_remap import (
        _master_pixels2skycells_path,
    )

    t = resolved.target
    os_factor = int(resolved.stages.mapping.oversampling_factor or 1)
    return _master_pixels2skycells_path(
        Path(resolved.mapping_root),
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=os_factor,
    )


def _count_npz_files(cache_dir: Path) -> int:
    """Count materialized ``*.npz`` files under *cache_dir* (recursive).

    Schema-v3 Exact caches live in skycell / pair subfolders; flat root ``*.npz``
    alone is the legacy layout.
    """
    if not cache_dir.is_dir():
        return 0
    return sum(1 for p in cache_dir.rglob("*.npz") if p.is_file())


def _has_flat_exact_npz(cache_dir: Path) -> bool:
    """True if any ``*.npz`` sits directly under *cache_dir* (legacy flat layout)."""
    if not cache_dir.is_dir():
        return False
    return any(p.is_file() for p in cache_dir.glob("*.npz"))

def _legacy_manifest_rejection_reason(payload: dict, *, source: str) -> str | None:
    """Reject pre-simplification manifests that used geometry toggles."""
    if payload.get("include_abutting_border_exact"):
        return (
            f"{source} uses removed L4b-lite (include_abutting_border_exact); "
            "rebuild remap with intra-skycell (exact_cache_l4a/) and inter-skycell "
            "(exact_cache_l4b/) caches"
        )
    if "apply_hybrid_exact" in payload or "l4b_policy" in payload:
        return (
            f"{source} uses removed geometry toggles (apply_hybrid_exact / l4b_policy); "
            "rebuild remap (stages.remap.rebuild_remap_cache: true and "
            "rebuild_inter_skycell_cache: true)"
        )
    if "hybrid_R" in payload and "intra_skycell_R" not in payload:
        return (
            f"{source} uses legacy hybrid_R; rebuild remap for schema v2 manifest "
            "(intra_skycell_R)"
        )
    policy = str(payload.get("l4b_policy", ""))
    if policy in _DEPRECATED_L4B_POLICIES:
        return (
            f"{source} has deprecated l4b_policy={policy!r}; rebuild remap"
        )
    return None


def _l4b_lite_rejection_reason(payload: dict, *, source: str) -> str | None:
    """Alias for legacy manifest rejection (L4b-lite / toggle manifests)."""
    return _legacy_manifest_rejection_reason(payload, source=source)


def _polluted_legacy_exact_cache_reason(read_root: Path) -> str | None:
    """Reject monolithic ``exact_cache/`` when pure ``exact_cache_l4a/`` is absent."""
    from syndiff_pipeline.template_creation.processing.field_remap import (
        EXACT_CACHE_L4A_DIRNAME,
        EXACT_CACHE_LEGACY_DIRNAME,
    )

    legacy = read_root / EXACT_CACHE_LEGACY_DIRNAME
    l4a = read_root / EXACT_CACHE_L4A_DIRNAME
    if legacy.is_dir() and any(legacy.glob("*.npz")):
        if not l4a.is_dir() or _count_npz_files(l4a) == 0:
            return (
                f"legacy polluted {EXACT_CACHE_LEGACY_DIRNAME}/ present without pure "
                f"{EXACT_CACHE_L4A_DIRNAME}/; archive or delete legacy cache and rebuild "
                "L4a (stages.remap.rebuild_remap_cache: true); do not reuse exact_cache/ "
                "as L4a"
            )
    return None


def _verify_remap_exact_caches(
    read_root: Path,
    payload: dict,
    *,
    require_intra_skycell: bool = True,
    require_inter_skycell: bool = True,
) -> str | None:
    """Validate intra + inter exact cache layout and NPZ counts.

    ``require_intra_skycell`` / ``require_inter_skycell`` control whether each
    layer's on-disk cache must be present and complete (used by downsample when
    geometry toggles skip a layer). Remap verify keeps both required.
    """
    from syndiff_pipeline.template_creation.processing.field_remap import (
        EXACT_CACHE_L4A_DIRNAME,
        EXACT_CACHE_L4B_DIRNAME,
        REMAP_SCHEMA_VERSION,
    )

    schema = int(payload.get("schema_version", 0))
    if schema < REMAP_SCHEMA_VERSION:
        return (
            f"remap_manifest schema_version={schema} is outdated; "
            f"expected >={REMAP_SCHEMA_VERSION}; rebuild remap"
        )

    polluted = _polluted_legacy_exact_cache_reason(read_root)
    if polluted:
        return polluted

    l4a_dir = read_root / EXACT_CACHE_L4A_DIRNAME
    l4b_dir = read_root / EXACT_CACHE_L4B_DIRNAME

    if schema >= 3:
        check_flat_l4a = require_intra_skycell or l4a_dir.is_dir()
        check_flat_l4b = require_inter_skycell or l4b_dir.is_dir()
        if (check_flat_l4a and _has_flat_exact_npz(l4a_dir)) or (
            check_flat_l4b and _has_flat_exact_npz(l4b_dir)
        ):
            return (
                "legacy flat Exact NPZ files found under exact_cache_l4a/ or "
                "exact_cache_l4b/ root; wipe Exact dirs and rebuild remap for "
                "schema-v3 skycell/pair subfolders"
            )
        for name in (
            "shift_epochs.parquet",
            "pair_epochs.parquet",
            "epoch_group_members.parquet",
            "gid_epoch_index.npz",
            "group_id_per_frame.npy",
        ):
            if not (read_root / name).is_file():
                return (
                    f"missing remap epoch artifact {name}; rebuild remap "
                    "(schema v3)"
                )
        n_l4a_manifest = payload.get("n_shift_epochs", payload.get("n_intra_skycell_keys"))
        n_l4b_manifest = payload.get(
            "n_pair_epochs", payload.get("n_inter_skycell_pair_states")
        )
    else:
        n_l4a_manifest = payload.get("n_intra_skycell_keys") or payload.get("n_exact_keys")
        n_l4b_manifest = payload.get("n_inter_skycell_pair_states") or payload.get(
            "n_l4b_pair_states"
        )

    if require_intra_skycell:
        if not l4a_dir.is_dir() or _count_npz_files(l4a_dir) == 0:
            if isinstance(n_l4a_manifest, int) and n_l4a_manifest > 0:
                return (
                    f"missing {EXACT_CACHE_L4A_DIRNAME}/ "
                    f"(manifest n_shift_epochs/n_intra={n_l4a_manifest}); rebuild "
                    "intra-skycell (stages.remap.rebuild_remap_cache: true)"
                )

        if isinstance(n_l4a_manifest, int) and n_l4a_manifest > 0:
            on_disk = _count_npz_files(l4a_dir)
            if on_disk < n_l4a_manifest:
                return (
                    f"intra-skycell cache incomplete: {on_disk}/{n_l4a_manifest} NPZ under "
                    f"{EXACT_CACHE_L4A_DIRNAME}/; rebuild remap"
                )

    if require_inter_skycell:
        if not l4b_dir.is_dir():
            return (
                f"missing {EXACT_CACHE_L4B_DIRNAME}/; rebuild inter-skycell "
                "(stages.remap.rebuild_inter_skycell_cache: true)"
            )
        if isinstance(n_l4b_manifest, int) and n_l4b_manifest > 0:
            on_disk = _count_npz_files(l4b_dir)
            if on_disk < n_l4b_manifest:
                return (
                    f"inter-skycell cache incomplete: {on_disk}/{n_l4b_manifest} NPZ under "
                    f"{EXACT_CACHE_L4B_DIRNAME}/; rebuild remap"
                )
        elif _count_npz_files(l4b_dir) == 0:
            return (
                f"inter-skycell requires nonempty {EXACT_CACHE_L4B_DIRNAME}/; rebuild remap"
            )
    return None


def _geometry_mode_for_resolved(resolved: ResolvedTargetConfig) -> str:
    job = Path(resolved.event_dir) / "cluster_template_job.json"
    if job.is_file():
        try:
            import json

            payload = json.loads(job.read_text())
            mode = payload.get("geometry_mode")
            if mode:
                return str(mode).lower()
        except Exception:
            pass
    return str(
        getattr(resolved.stages.wcs_grouping, "geometry_mode", None)
        or getattr(resolved.stages.downsample, "geometry_mode", None)
        or "field"
    ).lower()


def verify_downsample_field_mode(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify SCC field_templates store for geometry_mode: field."""
    from syndiff_pipeline.template_creation.processing.field_remap import (
        REMAP_MANIFEST_NAME,
        remap_root,
        resolve_remap_read_root,
    )
    from syndiff_pipeline.template_creation.processing.field_templates import (
        field_templates_root,
        verify_field_store,
    )

    t = resolved.target
    ds = resolved.stages.downsample
    mp = resolved.stages.mapping
    store = field_templates_root(
        resolved.data_root,
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=ds.oversampling_factor,
        store_name=ds.output_store_name,
    )
    # Prefer absolute override when output_base is set (matches dispatch write path).
    if ds.output_base:
        store = Path(ds.output_base)
    remap_store = remap_root(
        resolved.data_root,
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=mp.oversampling_factor,
        store_name=resolved.downsample_remap_store_name,
    )
    try:
        read_root, _legacy = resolve_remap_read_root(remap_store, store)
    except FileNotFoundError as exc:
        return VerifyResult("downsample", False, str(exc), str(remap_store))
    manifest_path = read_root / REMAP_MANIFEST_NAME
    if not manifest_path.is_file():
        return VerifyResult(
            "downsample",
            False,
            f"Field downsample requires {REMAP_MANIFEST_NAME} from remap stage",
            str(read_root),
        )
    import json

    try:
        remap_payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return VerifyResult(
            "downsample",
            False,
            f"Unreadable {REMAP_MANIFEST_NAME}: {exc}",
            str(manifest_path),
        )
    lite_reason = _l4b_lite_rejection_reason(
        remap_payload, source=REMAP_MANIFEST_NAME
    )
    if lite_reason:
        return VerifyResult("downsample", False, lite_reason, str(manifest_path))
    ds = resolved.stages.downsample
    cache_reason = _verify_remap_exact_caches(
        read_root,
        remap_payload,
        require_intra_skycell=bool(getattr(ds, "apply_intra_skycell", True)),
        require_inter_skycell=bool(getattr(ds, "apply_inter_skycell", True)),
    )
    if cache_reason:
        return VerifyResult("downsample", False, cache_reason, str(read_root))

    assembly_path = store / "field_mode_assembly.json"
    if assembly_path.is_file():
        try:
            assembly_payload = json.loads(assembly_path.read_text())
            lite_reason = _l4b_lite_rejection_reason(
                assembly_payload, source="field_mode_assembly.json"
            )
            if lite_reason:
                return VerifyResult("downsample", False, lite_reason, str(assembly_path))
        except (OSError, json.JSONDecodeError) as exc:
            return VerifyResult(
                "downsample",
                False,
                f"Unreadable field_mode_assembly.json: {exc}",
                str(assembly_path),
            )

    # The SCC store is shared across events; each event records exactly the
    # crop-filtered keys it required in field_contrib_keys.json (written only
    # after a successful build). Verify against THAT — not the full-chip
    # template_group_shifts, which would false-fail every cropped run.
    marker = Path(resolved.event_dir) / "field_contrib_keys.json"
    if not marker.is_file():
        return VerifyResult(
            "downsample",
            False,
            "Field downsample incomplete: missing field_contrib_keys.json marker",
            str(store),
        )
    try:
        keys_payload = json.loads(marker.read_text())
        lite_reason = _l4b_lite_rejection_reason(
            keys_payload, source="field_contrib_keys.json"
        )
        if lite_reason:
            return VerifyResult("downsample", False, lite_reason, str(marker))
        raw_keys = keys_payload.get("keys", [])
        if raw_keys and any(len(k) != 4 for k in raw_keys):
            return VerifyResult(
                "downsample",
                False,
                "Field mode requires group-qualified contrib keys "
                "[group_id, skycell, sx, sy]; rebuild downsample",
                str(marker),
            )
        required = [
            (int(k[0]), str(k[1]), int(k[2]), int(k[3])) for k in raw_keys
        ]
    except Exception as exc:
        return VerifyResult(
            "downsample", False, f"Unreadable field_contrib_keys.json: {exc}", str(store)
        )
    result = verify_field_store(
        store, required_keys=required, require_nonempty=False
    )
    if not result["ok"]:
        return VerifyResult(
            "downsample",
            False,
            f"Field store incomplete: {'; '.join(result['reasons'])}",
            str(store),
        )
    if required:
        from syndiff_pipeline.template_creation.processing.field_templates import (
            contrib_path,
            load_contrib,
        )

        nonempty = 0
        for key in required:
            if len(key) == 4:
                gid_i, skycell, sx_i, sy_i = key
                p = contrib_path(store, skycell, sx_i, sy_i, group_id=int(gid_i))
            else:
                skycell, sx_i, sy_i = key
                p = contrib_path(store, skycell, sx_i, sy_i)
            if p.is_file() and len(load_contrib(p)["indices"]) > 0:
                nonempty += 1
        if nonempty == 0:
            return VerifyResult(
                "downsample",
                False,
                f"Field store has {len(required)} keys but all contribs empty",
                str(store),
            )
    return VerifyResult(
        "downsample",
        True,
        f"Field store OK ({store.name})",
        str(store),
    )


def _shared_convolved_cell_root(shared_root: Path, full_skycell_name: str) -> Path | None:
    from syndiff_pipeline.template_creation.processing.combined_store import _projection_and_cell

    parsed = _projection_and_cell(full_skycell_name)
    if parsed is None:
        return None
    projection, cell = parsed
    return shared_root / projection / cell


def _clear_shared_convolved_cells(
    resolved: ResolvedTargetConfig,
    shared_root: Path,
) -> list[str]:
    """Remove published shared-store cells for this SCC's expected skycells only."""
    removed: list[str] = []
    try:
        expected = expected_ps1_process_skycells(resolved)
    except Exception as exc:
        log.warning(
            "Force rerun: could not resolve expected skycells for shared convolved clear: %s",
            exc,
        )
        return removed
    for name in expected:
        cell_root = _shared_convolved_cell_root(shared_root, name)
        if cell_root is None:
            log.warning(
                "Force rerun: skipping unparseable skycell %r for shared convolved clear",
                name,
            )
            continue
        if cell_root.is_dir():
            shutil.rmtree(cell_root)
            removed.append(str(cell_root))
            log.info("Force rerun: removed shared convolved cell %s", cell_root)
    return removed


def clear_ps1_process_artifacts(resolved: ResolvedTargetConfig) -> list[str]:
    """Clear ps1_process outputs so a force-rerun rebuilds from scratch.

    Legacy mode (``use_shared_convolved_store=False``): removes the per-SCC
    ``convolved.zarr`` tree. Shared mode: scope-clears only the skycell cells
    listed in this SCC's mapping CSV under the shared ``ps1_convolved.zarr``
    store (never the whole store). The per-SCC removed-stars CSV is always
    removed when present.
    """
    removed: list[str] = []
    if ps1_process_uses_shared_convolved_store(resolved):
        shared_root = resolve_ps1_process_checkpoint_location(resolved)
        removed.extend(_clear_shared_convolved_cells(resolved, shared_root))
    else:
        legacy_zarr = _convolved_zarr_path(resolved)
        if legacy_zarr.is_dir():
            shutil.rmtree(legacy_zarr)
            removed.append(str(legacy_zarr))
            log.info("Force rerun: removed directory %s", legacy_zarr)
        elif legacy_zarr.is_file():
            legacy_zarr.unlink()
            removed.append(str(legacy_zarr))
            log.info("Force rerun: removed file %s", legacy_zarr)

    csv_path = ps1_process_removed_stars_csv_path(resolved)
    if csv_path.is_file():
        csv_path.unlink()
        removed.append(str(csv_path))
        log.info("Force rerun: removed file %s", csv_path)
    return removed


def _skycell_name(entry) -> str:
    """Normalize a skycell identifier to its plain name.

    ``expected_convolved_skycells`` (and the underlying task list) may yield a
    ``(name, index)`` tuple; the stored Zarr arrays are keyed by the name alone,
    so we always compare on the name. Defensive even though the source now
    returns strings.
    """
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    return str(entry)


def expected_ps1_process_skycells(resolved: ResolvedTargetConfig) -> list[str]:
    """Expected ps1 process skycells.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    list[str]"""
    from syndiff_pipeline.template_creation.processing.csv_utils import load_csv_data
    from syndiff_pipeline.template_creation.processing.ps1_process import expected_convolved_skycells

    t = resolved.target
    try:
        names = expected_convolved_skycells(
            resolved.data_root,
            t.sector,
            t.camera,
            t.ccd,
            projections_limit=resolved.stages.ps1_process.projections_limit,
        )
        return sorted({_skycell_name(n) for n in names})
    except Exception as exc:
        log.debug("Falling back to CSV row skycell list for ps1_process verify: %s", exc)
        csv_path = _mapping_csv_path(resolved)
        df = load_csv_data(str(csv_path))
        if "projection" not in df.columns or "NAME" not in df.columns:
            raise ValueError(f"Master skycells CSV missing projection/NAME: {csv_path}") from exc
        projections = sorted(df["projection"].astype(str).unique())
        limit = resolved.stages.ps1_process.projections_limit
        if limit:
            projections = projections[: int(limit)]
        names = df[df["projection"].astype(str).isin(projections)]["NAME"].astype(str)
        return sorted(set(names))


def _count_convolved_data_arrays(zarr_path: Path, expected_names: list[str]) -> tuple[int, list[str]]:
    """(saved, missing) over expected skycells using metadata-only scandir.

    A skycell is "saved" iff its ``<name>_data`` array exists with at least one
    materialized chunk. No Zarr open and no chunk decompression, so this stays
    fast on NFS even for stores with thousands of arrays.
    """
    missing: list[str] = []
    saved = 0
    for name in expected_names:
        if _zarr_array_has_chunks(zarr_path / f"{name}_data"):
            saved += 1
        else:
            missing.append(name)
    return saved, missing


def _shared_convolved_cell_published(shared_root: Path, full_skycell_name: str) -> bool:
    from syndiff_pipeline.template_creation.processing.combined_store import _projection_and_cell

    parsed = _projection_and_cell(full_skycell_name)
    if parsed is None:
        return False
    projection, cell = parsed
    cell_root = shared_root / projection / cell
    if not cell_root.is_dir():
        return False
    try:
        for fp_dir in cell_root.iterdir():
            if fp_dir.is_dir() and (fp_dir / "arrays.npz").is_file():
                return True
    except OSError:
        return False
    return False


def _count_shared_convolved_cells(
    shared_root: Path, expected_names: list[str]
) -> tuple[int, list[str]]:
    missing: list[str] = []
    saved = 0
    for name in expected_names:
        if _shared_convolved_cell_published(shared_root, name):
            saved += 1
        else:
            missing.append(name)
    return saved, missing


def _store_has_any_data_array(zarr_path: Path) -> bool:
    """True if the convolved store contains any ``*_data`` array directory."""
    try:
        with os.scandir(zarr_path) as it:
            return any(entry.name.endswith("_data") for entry in it)
    except OSError:
        return False


def verify_ps1_process(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig | None = None,
) -> VerifyResult:
    """Verify ps1 process.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    runner_cfg : RunnerConfig | None, optional, default ``None``
    
    Returns
    -------
    VerifyResult"""
    zarr_path = resolve_ps1_process_checkpoint_location(resolved)
    if runner_cfg is not None and runner_cfg.bookkeeping_trust_index:
        indexed = checkpoint_stage_indexed(resolved, "ps1_process")
        if indexed:
            return VerifyResult("ps1_process", True, "indexed", str(zarr_path))
        return VerifyResult(
            "ps1_process", False, "scc_assembly not indexed", str(zarr_path)
        )
    shared_store = ps1_process_uses_shared_convolved_store(resolved)
    if not zarr_path.exists():
        label = "Shared convolved store missing" if shared_store else "Convolved zarr missing"
        return VerifyResult("ps1_process", False, label, str(zarr_path))
    try:
        expected = expected_ps1_process_skycells(resolved)
    except Exception as exc:
        return VerifyResult("ps1_process", False, str(exc), str(zarr_path))
    if not expected:
        return VerifyResult("ps1_process", False, "No expected skycells from mapping CSV", str(zarr_path))

    started = time.monotonic()
    if shared_store:
        saved, missing = _count_shared_convolved_cells(zarr_path, expected)
    else:
        saved, missing = _count_convolved_data_arrays(zarr_path, expected)
    elapsed = time.monotonic() - started
    log.info(
        "verify_ps1_process: %d/%d skycells saved in %.2fs (%s)",
        saved,
        len(expected),
        elapsed,
        zarr_path,
    )

    if saved == 0:
        if shared_store:
            msg = f"Shared convolved store has no published cells: 0/{len(expected)} skycells saved"
        elif _store_has_any_data_array(zarr_path):
            msg = (
                f"Convolved zarr has *_data arrays but none cover expected skycells "
                f"(or all empty): 0/{len(expected)} skycells saved"
            )
        else:
            msg = f"Convolved zarr store is empty (no *_data arrays): 0/{len(expected)} skycells saved"
        return VerifyResult("ps1_process", False, msg, str(zarr_path))

    if saved < len(expected):
        return VerifyResult(
            "ps1_process",
            False,
            f"Partial convolved zarr: {saved}/{len(expected)} skycells saved"
            + (f" (missing e.g. {missing[:3]})" if missing else ""),
            str(zarr_path),
        )
    return VerifyResult(
        "ps1_process",
        True,
        f"Convolved zarr complete ({saved}/{len(expected)} skycells)",
        str(zarr_path),
    )


def _downsample_expected_basenames(resolved: ResolvedTargetConfig) -> tuple[list[str], Path]:
    """Per-offset FITS basenames ``downsample.save_fits_outputs`` writes, plus the
    output base dir. Honors ``single_offset`` (a single ``[0, 0]`` offset) and the
    ROI/oversampling filename tags. Raises on a missing/invalid cluster job JSON.
    """
    import numpy as np

    from syndiff_pipeline.template_creation.processing.downsample import (
        load_cluster_template_job_payload,
        offsets_from_cluster_job_payload,
        roi_tuple_from_cluster_job_payload,
    )

    t = resolved.target
    ds = resolved.stages.downsample
    job_path = Path(resolved.event_dir) / "cluster_template_job.json"
    payload = load_cluster_template_job_payload(str(job_path))
    roi = roi_tuple_from_cluster_job_payload(payload)
    if ds.single_offset:
        offsets = np.array([[0.0, 0.0]])
    else:
        offsets = offsets_from_cluster_job_payload(payload)
    x_min, y_min, x_max, y_max = roi
    roi_part = ""
    if not (x_min == 0 and y_min == 0):
        roi_part = f"_x{x_min}-{x_max}_y{y_min}-{y_max}"
    os_factor = ds.oversampling_factor
    os_part = f"_os{os_factor}" if os_factor > 1 else ""
    base = Path(ds.output_base or resolved.template_output_base)
    basenames = [
        f"syndiff_template_s{t.sector:04d}_{t.camera}_{t.ccd}{roi_part}{os_part}"
        f"_dx{float(dx):.3f}_dy{float(dy):.3f}{PIPELINE_FITS_EXT}"
        for dx, dy in offsets
    ]
    return basenames, base


def _downsample_fits_filename_candidates(basename: str) -> list[str]:
    """Canonical ``.fits.fz`` basename plus legacy ``.fits.gz`` / ``.fits``."""
    stem = strip_fits_storage_suffix(basename)
    return [f"{stem}{sfx}" for sfx in FITS_STORAGE_SUFFIXES]


def _find_downsample_fits(base: Path, t, basename: str) -> str | None:
    """Locate a per-offset FITS under any ``sector..._ccd<ccd>*`` output dir.

    The writer's output directory carries the full ROI suffix (which depends on
    the base frame shape) while the *filename* only tags ROI when x_min/y_min are
    nonzero, so we glob across matching dirs and match on the authoritative
    filename rather than reconstructing the exact directory name.
    """
    for bn in _downsample_fits_filename_candidates(basename):
        pattern = f"sector{t.sector:04d}_camera{t.camera}_ccd{t.ccd}*/{bn}"
        matches = sorted(base.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def expected_downsample_fits_paths(resolved: ResolvedTargetConfig) -> list[Path]:
    """Resolved per-offset FITS paths. Found files report their real path; missing
    ones report a canonical expected path (useful for manifest/report listings)."""
    basenames, base = _downsample_expected_basenames(resolved)
    t = resolved.target
    paths: list[Path] = []
    for bn in basenames:
        found = _find_downsample_fits(base, t, bn)
        if found:
            paths.append(Path(found))
        else:
            paths.append(base / f"sector{t.sector:04d}_camera{t.camera}_ccd{t.ccd}" / bn)
    return paths


def verify_downsample(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify SCC templates store (full-chip) or legacy per-event downsample outputs."""
    store = Path(resolved.template_output_base)
    if store.is_dir() and any(store.iterdir()):
        return VerifyResult(
            "downsample",
            True,
            f"SCC templates store present under {store.name}/",
            str(store),
        )
    # New SCC-scoped linear mode: go straight to the SCC-scoped check.
    # Deliberately bypasses _geometry_mode_for_resolved (and its
    # cluster_template_job.json / event_dir probe) -- that probe is what
    # stalled the verify scan on greenfield SCCs (see linear_downsample
    # module docstring). Only "linear" is fast-pathed here; "field" is left
    # alone since it's also the DownsampleStageParams dataclass default and
    # can't be distinguished from "not configured" without the job-file check
    # legacy event-scoped stores rely on. Also gated on `output_base` being
    # unset: the legacy event-scoped linear path (flat offset FITS under a
    # caller-chosen output_base, e.g. the old `shifted_downsampled/`
    # convention) also uses geometry_mode="linear" but is a different,
    # still-supported store layout -- the new SCC-scoped writer never sets
    # output_base (it uses output_store_name under templates_{name}/ instead).
    ds = resolved.stages.downsample
    if not ds.output_base and str(getattr(ds, "geometry_mode", None) or "").lower() == "linear":
        return verify_downsample_linear_mode(resolved)
    legacy = _verify_downsample_legacy(resolved)
    return VerifyResult(
        "downsample",
        legacy.ok,
        legacy.message,
        legacy.path,
        unknown=legacy.unknown,
    )


def verify_downsample_linear_mode(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify SCC linear-mode templates store (geometry_mode: linear).

    SCC-scoped only -- deliberately never touches ``event_dir`` /
    ``cluster_template_job.json`` (that legacy per-event check is what made
    this stage's verify scan stall on greenfield SCCs; see
    ``linear_downsample.run_linear_downsample_scc``).
    """
    import json

    from syndiff_pipeline.template_creation.processing.linear_downsample import (
        LINEAR_ASSEMBLY_BASENAME,
    )

    ds = resolved.stages.downsample
    store = (
        Path(ds.output_base)
        if ds.output_base
        else Path(resolved.template_output_base)
    )
    if not store.is_dir():
        return VerifyResult("downsample", False, f"No linear templates store at {store}", str(store))

    assembly_path = store / LINEAR_ASSEMBLY_BASENAME
    if not assembly_path.is_file():
        return VerifyResult(
            "downsample", False, f"Missing {LINEAR_ASSEMBLY_BASENAME}", str(assembly_path)
        )
    try:
        payload = json.loads(assembly_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return VerifyResult("downsample", False, f"Unreadable {LINEAR_ASSEMBLY_BASENAME}: {exc}", str(assembly_path))

    expected = [store / name for name in payload.get("artifacts", [])]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        return VerifyResult(
            "downsample", False,
            f"Partial linear downsample: {len(expected) - len(missing)}/{len(expected)} group FITS present",
            missing[0],
        )
    n_groups = int(payload.get("n_groups", len(expected)))
    return VerifyResult(
        "downsample", True, f"All {n_groups} linear group FITS present", str(store)
    )


def _verify_downsample_legacy(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify downsample (legacy event-scoped linear offset FITS, or field SCC store).

    The new SCC-scoped linear mode is routed to ``verify_downsample_linear_mode``
    directly from ``verify_downsample`` (before this function is called) --
    this function's "linear" fallback below is the older, still-supported
    event-scoped flat-offset-FITS convention (``output_base`` explicitly set).
    """
    mode = _geometry_mode_for_resolved(resolved)
    if mode == "field":
        return verify_downsample_field_mode(resolved)

    t = resolved.target
    try:
        basenames, base = _downsample_expected_basenames(resolved)
    except Exception as exc:
        out_base = Path(resolved.stages.downsample.output_base or resolved.template_output_base)
        return VerifyResult("downsample", False, f"Cannot determine expected offsets: {exc}", str(out_base))

    found: list[str] = []
    missing: list[str] = []
    for bn in basenames:
        match = _find_downsample_fits(base, t, bn)
        if match:
            found.append(match)
        else:
            missing.append(bn)

    n_expected = len(basenames)
    sample = found[0] if found else str(base)
    if missing:
        return VerifyResult(
            "downsample",
            False,
            f"Partial downsample: {len(found)}/{n_expected} offset FITS present "
            f"({len(missing)} missing)",
            sample,
        )

    if ps1_process_removed_stars_csv_path(resolved).is_file():
        csv_path = event_dir_ps1_removed_stars_csv_path(resolved)
        if not csv_path.is_file():
            return VerifyResult(
                "downsample",
                False,
                f"Missing {csv_path.name} in event_dir",
                str(csv_path),
            )

    return VerifyResult(
        "downsample",
        True,
        f"All {n_expected} offset FITS present",
        sample,
    )


def verify_remap(resolved: ResolvedTargetConfig) -> VerifyResult:
    """Verify SCC remap store (shift schedule, groups, remap_manifest)."""
    import json

    from syndiff_pipeline.template_creation.processing.field_remap import (
        REMAP_MANIFEST_NAME,
        remap_root,
    )

    t = resolved.target
    mp = resolved.stages.mapping
    rm = resolved.stages.remap
    store = remap_root(
        resolved.data_root,
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=mp.oversampling_factor,
        store_name=rm.store_name,
    )
    manifest_path = store / REMAP_MANIFEST_NAME
    if not manifest_path.is_file():
        return VerifyResult(
            "remap",
            False,
            f"Missing {REMAP_MANIFEST_NAME}",
            str(store),
        )
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return VerifyResult(
            "remap",
            False,
            f"Unreadable {REMAP_MANIFEST_NAME}: {exc}",
            str(manifest_path),
        )
    lite_reason = _l4b_lite_rejection_reason(payload, source=REMAP_MANIFEST_NAME)
    if lite_reason:
        return VerifyResult("remap", False, lite_reason, str(manifest_path))
    cache_reason = _verify_remap_exact_caches(
        store,
        payload,
    )
    if cache_reason:
        return VerifyResult("remap", False, cache_reason, str(store))
    if float(payload.get("cache_quantum_ps1_px", -1)) != float(rm.cache_quantum_ps1_px):
        return VerifyResult(
            "remap",
            False,
            "remap_manifest cache_quantum_ps1_px does not match config",
            str(manifest_path),
        )
    if str(payload.get("keying", "")) != str(rm.keying):
        return VerifyResult(
            "remap",
            False,
            "remap_manifest keying does not match config",
            str(manifest_path),
        )
    schedule = store / "shift_schedule.npz"
    groups = store / "template_group_shifts.parquet"
    missing = [p.name for p in (schedule, groups) if not p.is_file()]
    if missing:
        return VerifyResult(
            "remap",
            False,
            f"Remap store incomplete: missing {', '.join(missing)}",
            str(store),
        )
    return VerifyResult(
        "remap",
        True,
        f"Remap store OK ({store.name})",
        str(manifest_path),
    )


def verify_diff(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig | None = None,
    *,
    meta: dict | None = None,
) -> VerifyResult:
    """Verify diff.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    runner_cfg : RunnerConfig | None, optional, default ``None``
    meta : dict | None, optional, default ``None``
    
    Returns
    -------
    VerifyResult"""
    from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
        _scc_lane_root,
        diff_workspace_complete,
        frozen_diff_config_for_verify,
        resolve_diff_site_config_path,
    )
    from syndiff_pipeline.difference_imaging.support.manifest import manifest_path_from_output_dir

    if runner_cfg is None:
        return VerifyResult(
            "diff",
            False,
            "diff verification requires RunnerConfig with diff_config_path",
            resolved.event_dir,
        )
    if not runner_cfg.diff_config_path:
        return VerifyResult(
            "diff",
            False,
            "diff verification requires diff_config_path on RunnerConfig",
            resolved.event_dir,
        )
    cfg = frozen_diff_config_for_verify(
        resolve_diff_site_config_path(meta=meta, runner_cfg=runner_cfg),
        resolved.target,
        meta=meta,
    )
    event_dir = Path(resolved.event_dir)
    manifest_csv = manifest_path_from_output_dir(str(event_dir), None)
    lane_root = _scc_lane_root(cfg)
    if diff_workspace_complete(cfg, event_dir):
        detail = str(lane_root) if lane_root is not None else ""
        return VerifyResult("diff", True, "SCC diff lane complete", detail)

    data_root = getattr(cfg, "data_root", "") or ""
    if not data_root:
        return VerifyResult(
            "diff",
            False,
            "deployment missing data_root for SCC diff verification",
            str(event_dir),
        )
    if not Path(manifest_csv).is_file():
        return VerifyResult("diff", False, "Missing frame manifest CSV", manifest_csv)

    lane_detail = str(lane_root) if lane_root is not None else "SCC diff lane path unavailable"
    return VerifyResult(
        "diff",
        False,
        "SCC diff lane incomplete (bookkeeping or final stage outputs missing)",
        lane_detail,
    )


def verify_photometry(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig | None = None,
    *,
    meta: dict | None = None,
) -> VerifyResult:
    """Verify photometry outputs under ``phot_{run_id}/``."""
    from syndiff_pipeline.photometry.orchestration.verify import photometry_complete
    from syndiff_pipeline.photometry.site_config import (
        load_photometry_site_policy,
        resolve_photometry_config_path,
        resolve_photometry_run_config,
    )

    if runner_cfg is None or not getattr(runner_cfg, "photometry_config_path", ""):
        return VerifyResult(
            "photometry",
            False,
            "photometry verification requires photometry_config_path on RunnerConfig",
            resolved.event_dir,
        )
    policy = load_photometry_site_policy(
        resolve_photometry_config_path(meta=meta, runner_cfg=runner_cfg)
    )
    run_config = resolve_photometry_run_config(
        policy, resolved.target, site_dir=Path(policy.config_path).parent
    )
    event_dir = Path(resolved.event_dir)
    if photometry_complete(run_config, event_dir, run_config.photometry_run_id):
        from syndiff_pipeline.difference_imaging.support.paths import photometry_root

        phot_root = photometry_root(str(event_dir), run_config.photometry_run_id)
        return VerifyResult("photometry", True, "Photometry outputs present", str(phot_root))
    return VerifyResult(
        "photometry",
        False,
        "Photometry outputs missing under phot_{run_id}/",
        str(event_dir),
    )


def stage_absence_probe(
    resolved: ResolvedTargetConfig,
    stage: str,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> AbsenceProbeResult:
    """Fast filesystem probe: skip full verify when outputs cannot exist."""
    from syndiff_pipeline.common.download import list_local_ffis, tesscurl_script_path
    from syndiff_pipeline.common.orchestration.event_ws_symlinks import event_templates_symlink_path
    from syndiff_pipeline.common.wcs_grouping import CLUSTER_TEMPLATE_JOB_FILENAME

    if stage == "mapping":
        csv_path = _mapping_csv_path(resolved)
        return (
            AbsenceProbeResult.MAYBE_PRESENT
            if csv_path.is_file()
            else AbsenceProbeResult.ABSENT
        )

    if stage == "tess_ffi_download":
        t = resolved.target
        ffi_leaf = resolved.ffi_dir
        if list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd):
            return AbsenceProbeResult.MAYBE_PRESENT
        cached = tesscurl_script_path(ffi_leaf, t.sector)
        if Path(cached).is_file():
            return AbsenceProbeResult.ABSENT
        return AbsenceProbeResult.ABSENT

    if stage == "ps1_download":
        zarr_path = ps1_skycells_zarr_path(resolved.data_root)
        return (
            AbsenceProbeResult.MAYBE_PRESENT
            if zarr_path.exists()
            else AbsenceProbeResult.ABSENT
        )

    if stage == "ps1_process":
        zarr_path = _convolved_zarr_path(resolved)
        return (
            AbsenceProbeResult.MAYBE_PRESENT
            if zarr_path.exists()
            else AbsenceProbeResult.ABSENT
        )

    if stage == "downsample":
        store = Path(resolved.template_output_base)
        if store.is_dir() and any(store.iterdir()):
            return AbsenceProbeResult.MAYBE_PRESENT
        job_path = Path(resolved.event_dir) / CLUSTER_TEMPLATE_JOB_FILENAME
        templates_link = event_templates_symlink_path(resolved.event_dir)
        if job_path.is_file() or (
            templates_link.is_symlink() and templates_link.resolve().is_dir()
        ):
            return AbsenceProbeResult.MAYBE_PRESENT
        return AbsenceProbeResult.ABSENT

    if stage == "remap":
        from syndiff_pipeline.template_creation.processing.field_remap import (
            REMAP_MANIFEST_NAME,
            remap_root,
        )

        t = resolved.target
        mp = resolved.stages.mapping
        store = remap_root(
            resolved.data_root,
            t.sector,
            t.camera,
            t.ccd,
            oversampling_factor=mp.oversampling_factor,
            store_name=resolved.stages.remap.store_name,
        )
        if (store / REMAP_MANIFEST_NAME).is_file():
            return AbsenceProbeResult.MAYBE_PRESENT
        return AbsenceProbeResult.ABSENT

    if stage == "diff":
        if runner_cfg is None or not runner_cfg.diff_config_path:
            return AbsenceProbeResult.UNKNOWN
        from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
            _diff_frame_manifest_available,
            _scc_lane_root,
            frozen_diff_config_for_verify,
            resolve_diff_site_config_path,
        )

        cfg = frozen_diff_config_for_verify(
            resolve_diff_site_config_path(meta=meta, runner_cfg=runner_cfg),
            resolved.target,
            meta=meta,
        )

        if _diff_frame_manifest_available(cfg, resolved.event_dir):
            return AbsenceProbeResult.MAYBE_PRESENT
        lane_root = _scc_lane_root(cfg)
        if lane_root is not None and lane_root.is_dir() and any(lane_root.iterdir()):
            return AbsenceProbeResult.MAYBE_PRESENT
        return AbsenceProbeResult.ABSENT

    return AbsenceProbeResult.UNKNOWN


VERIFY_FUNCS = {
    "tess_ffi_download": verify_tess_ffi_download,
    "mapping": verify_mapping,
    "ps1_download": verify_ps1_download,
    "ps1_process": verify_ps1_process,
    "remap": verify_remap,
    "downsample": verify_downsample,
    "diff": verify_diff,
    "photometry": verify_photometry,
}


def verify_stage(
    resolved: ResolvedTargetConfig,
    stage: str,
    runner_cfg: RunnerConfig | None = None,
    *,
    meta: dict | None = None,
) -> VerifyResult:
    """Verify stage.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    stage : str
    runner_cfg : RunnerConfig | None, optional, default ``None``
    meta : dict | None, optional, default ``None``
    
    Returns
    -------
    VerifyResult"""
    if (
        runner_cfg is not None
        and runner_cfg.bookkeeping_trust_index
        and stage in CHECKPOINT_STAGES
    ):
        indexed = checkpoint_stage_indexed(resolved, stage)
        return VerifyResult(
            stage,
            indexed,
            "indexed" if indexed else "not indexed",
            "",
        )
    fn = VERIFY_FUNCS.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage: {stage!r}")
    if stage == "diff":
        return fn(resolved, runner_cfg, meta=meta)
    if stage == "photometry":
        return fn(resolved, runner_cfg, meta=meta)
    return fn(resolved)


def stage_complete(
    resolved: ResolvedTargetConfig,
    stage: str,
    manifest_path: str | None = None,
    stable_manifest_path: str | None = None,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> bool:
    """Return True if the stage outputs are complete.

    Manifest-first: when *manifest_path* (per-run) or *stable_manifest_path*
    (cross-run) points to a valid manifest (well-formed, schema version ok,
    config fingerprint matches, and every listed artifact still exists on disk),
    the stage is complete. Otherwise fall back to the hardened on-disk check
    ``verify_stage(resolved, stage).ok``. An ``unknown`` on-disk result is treated
    conservatively (not complete).
    """
    if (
        runner_cfg is not None
        and runner_cfg.bookkeeping_trust_index
        and stage in CHECKPOINT_STAGES
    ):
        return checkpoint_stage_indexed(resolved, stage)
    for candidate in (manifest_path, stable_manifest_path):
        if candidate is None:
            continue
        manifest = read_manifest(candidate)
        if manifest is not None and manifest_valid(
            manifest, resolved, stage, runner_cfg=runner_cfg, meta=meta
        ):
            return True
    result = verify_stage(resolved, stage, runner_cfg, meta=meta)
    if result.unknown:
        return False
    return result.ok


def collect_stage_artifacts(
    resolved: ResolvedTargetConfig,
    stage: str,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> tuple[int, int, list[str]]:
    """Return (expected_count, produced_count, artifact_paths) for manifest writing."""
    if stage == "diff":
        from syndiff_pipeline.difference_imaging.orchestration.stages import DIFF_STAGE

        if runner_cfg is None:
            raise ValueError("diff artifact collection requires RunnerConfig")
        ctx = _diff_stage_context(resolved, runner_cfg, meta=meta)
        return DIFF_STAGE.collect_artifacts(ctx)
    if stage == "remap":
        from syndiff_pipeline.template_creation.processing.field_remap import (
            REMAP_MANIFEST_NAME,
            remap_root,
        )

        t = resolved.target
        mp = resolved.stages.mapping
        store = remap_root(
            resolved.data_root,
            t.sector,
            t.camera,
            t.ccd,
            oversampling_factor=mp.oversampling_factor,
            store_name=resolved.stages.remap.store_name,
        )
        manifest = store / REMAP_MANIFEST_NAME
        ok = manifest.is_file()
        return 1, int(ok), [str(manifest)] if ok else [str(store)]
    if stage == "downsample":
        from syndiff_pipeline.common.wcs_grouping import _event_job_path

        has_event_job = Path(_event_job_path(resolved.event_dir)).is_file()
        # SCC-only field builds have no event handoff; collect store markers.
        # Linear / event-bound runs keep the FITS path (matches dispatch).
        if _geometry_mode_for_resolved(resolved) == "field" and not has_event_job:
            from syndiff_pipeline.template_creation.processing.field_templates import (
                MANIFEST_NAME,
                field_templates_root,
            )

            t = resolved.target
            ds = resolved.stages.downsample
            store = field_templates_root(
                resolved.data_root,
                t.sector,
                t.camera,
                t.ccd,
                oversampling_factor=ds.oversampling_factor,
                store_name=ds.output_store_name,
            )
            if ds.output_base:
                store = Path(ds.output_base)
            paths = [
                store / "field_mode_assembly.json",
                store / MANIFEST_NAME,
            ]
            existing = [str(p) for p in paths if p.is_file()]
            return len(paths), len(existing), existing

        paths = expected_downsample_fits_paths(resolved)
        if ps1_process_removed_stars_csv_path(resolved).is_file():
            paths.append(event_dir_ps1_removed_stars_csv_path(resolved))
        from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
            event_templates_symlink_path,
        )

        symlink = event_templates_symlink_path(resolved.event_dir)
        if symlink.is_symlink() and symlink.resolve().is_dir():
            paths.append(symlink)
        existing = [str(p) for p in paths if p.is_file() or p.is_symlink()]
        return len(paths), len(existing), existing
    if stage == "ps1_process":
        zarr_path = resolve_ps1_process_checkpoint_location(resolved)
        if runner_cfg is not None and runner_cfg.bookkeeping_trust_index:
            indexed = checkpoint_stage_indexed(resolved, stage)
            return (1, int(indexed), [str(zarr_path)])
        expected = expected_ps1_process_skycells(resolved)
        if not zarr_path.exists():
            return len(expected), 0, [str(zarr_path)]
        if ps1_process_uses_shared_convolved_store(resolved):
            saved, _missing = _count_shared_convolved_cells(zarr_path, expected)
        else:
            saved, _missing = _count_convolved_data_arrays(zarr_path, expected)
        return len(expected), saved, [str(zarr_path)]
    if stage == "mapping":
        csv_path = _mapping_csv_path(resolved)
        ok = csv_path.is_file()
        return 1, int(ok), [str(csv_path)] if ok else []
    if stage == "wcs_grouping":
        from syndiff_pipeline.common.wcs_grouping import (
            CLUSTER_TEMPLATE_JOB_FILENAME,
            WCS_DRIFT_TEMPLATE_DEBUG_FILENAME,
        )

        from syndiff_pipeline.difference_imaging.support.paths import pipeline_plots_root

        job_path = Path(resolved.event_dir) / CLUSTER_TEMPLATE_JOB_FILENAME
        plot_path = (
            Path(pipeline_plots_root(resolved.event_dir))
            / WCS_DRIFT_TEMPLATE_DEBUG_FILENAME
        )
        ok = job_path.is_file()
        artifacts = [str(job_path)] if ok else []
        if plot_path.is_file():
            artifacts.append(str(plot_path))
        return 1, int(ok), artifacts
    if stage == "ps1_download":
        expected = _expected_ps1_download_skycells(resolved)
        zarr_path = ps1_skycells_zarr_path(resolved.data_root)
        result = verify_ps1_download(resolved)
        produced = 0
        if result.ok:
            produced = len(expected)
        return len(expected), produced, [str(zarr_path)]
    if stage == "tess_ffi_download":
        from syndiff_pipeline.common.download import expected_ffi_basenames, list_local_ffis

        t = resolved.target
        ffi_leaf = resolved.ffi_dir
        expected = expected_ffi_basenames(t.sector, t.camera, t.ccd, output_dir=ffi_leaf) or []
        files = list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd)
        return len(expected), len(files), [str(p) for p in files]
    result = verify_stage(resolved, stage, runner_cfg, meta=meta)
    path = result.path or ""
    return 1, int(result.ok), [path] if path else []


def persist_completion_manifests(
    resolved: ResolvedTargetConfig,
    stage: str,
    manifest_paths: list[str | Path],
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> list[str]:
    """Write completion manifests for a stage already verified complete on disk.

    The caller supplies explicit manifest paths (per-run, stable, etc.) so this
    module stays decoupled from run-directory layout.
    """
    expected, produced, artifacts = collect_stage_artifacts(
        resolved, stage, runner_cfg=runner_cfg, meta=meta
    )
    written: list[str] = []
    for manifest_path in manifest_paths:
        write_manifest(
            manifest_path,
            resolved,
            stage,
            artifacts,
            expected,
            produced,
            runner_cfg=runner_cfg,
            meta=meta,
        )
        written.append(str(manifest_path))
    return written


def verify_target(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig,
    stages: Optional[List[str]] = None,
    *,
    meta: dict | None = None,
) -> List[VerifyResult]:
    """Verify target.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    runner_cfg : RunnerConfig
    stages : Optional[List[str]], optional, default ``None``
    meta : dict | None, optional, default ``None``
    
    Returns
    -------
    List[VerifyResult]"""
    if stages is None:
        from syndiff_pipeline.pipeline_spec import STAGE_NAMES

        stages = list(STAGE_NAMES)
    return [verify_stage(resolved, s, runner_cfg, meta=meta) for s in stages]


def verify_all(cfg: RunnerConfig, targets: List[Target], stages: Optional[List[str]] = None) -> List[VerifyResult]:
    """Verify all.
    
    Parameters
    ----------
    cfg : RunnerConfig
    targets : List[Target]
    stages : Optional[List[str]], optional, default ``None``
    
    Returns
    -------
    List[VerifyResult]"""
    out: List[VerifyResult] = []
    for t in targets:
        resolved = resolve_config(t, cfg)
        for r in verify_target(resolved, cfg, stages):
            out.append(r)
    return out
