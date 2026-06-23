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

from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig, resolve_config, RunnerConfig
from syndiff_pipeline.common.orchestration.targets import Target

log = logging.getLogger(__name__)

# Bump when the manifest JSON schema changes; a mismatch invalidates a manifest.
MANIFEST_SCHEMA_VERSION = 2


class AbsenceProbeResult(Enum):
    """Fast pre-check before scheduling a full background artifact verify."""

    ABSENT = "absent"
    MAYBE_PRESENT = "maybe"
    UNKNOWN = "unknown"


@dataclass
class VerifyResult:
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
    elif stage == "downsample":
        ds = resolved.stages.downsample
        parts.extend(
            [
                str(ds.oversampling_factor),
                str(ds.single_offset),
                str(list(ds.ignore_mask_bits)),
                str(ds.output_base or resolved.template_output_base),
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
    from syndiff_pipeline.common.download import (
        expected_ffi_basenames,
        list_local_ffis,
        nested_ffi_dir,
    )

    t = resolved.target
    ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
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

    existing = {Path(p).name for p in local_files}
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
    job_path = Path(resolved.event_dir) / "cluster_template_job.json"
    if not job_path.is_file():
        return VerifyResult("wcs_grouping", False, "Missing cluster_template_job.json", str(job_path))
    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return VerifyResult("wcs_grouping", False, f"Invalid JSON: {exc}", str(job_path))
    ref = payload.get("reference_ffi_path")
    if not ref or not Path(ref).is_file():
        return VerifyResult("wcs_grouping", False, "reference_ffi_path missing or not found", str(job_path))
    return VerifyResult("wcs_grouping", True, "Valid cluster_template_job.json", str(job_path))


def verify_mapping(resolved: ResolvedTargetConfig) -> VerifyResult:
    t = resolved.target
    suffix = ""
    os_factor = resolved.stages.mapping.oversampling_factor
    mapping_root = Path(resolved.mapping_root)
    if os_factor > 1:
        mapping_root = mapping_root / f"oversampling_{os_factor}"
        suffix = f"_os{os_factor}"
    csv_path = (
        mapping_root
        / f"sector_{t.sector:04d}"
        / f"camera_{t.camera}"
        / f"ccd_{t.ccd}"
        / f"tess_s{t.sector:04d}_{t.camera}_{t.ccd}_master_skycells_list{suffix}.csv"
    )
    if csv_path.is_file():
        return VerifyResult("mapping", True, "Master skycells CSV exists", str(csv_path))
    return VerifyResult("mapping", False, "Master skycells CSV missing", str(csv_path))


_PS1_DOWNLOAD_BANDS = ("r", "i", "z", "y")


def _ps1_download_expected_array_names() -> list[str]:
    names: list[str] = []
    for band in _PS1_DOWNLOAD_BANDS:
        names.extend([band, f"{band}_mask", f"{band}_wt"])
    return names


def _projection_from_skycell_name(skycell_name: str) -> str | None:
    try:
        return skycell_name.split(".")[1]
    except (IndexError, AttributeError):
        return None


def _expected_ps1_download_skycells(resolved: ResolvedTargetConfig) -> list[str]:
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
    zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
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
    t = resolved.target
    suffix = ""
    os_factor = resolved.stages.mapping.oversampling_factor
    mapping_root = Path(resolved.mapping_root)
    if os_factor > 1:
        mapping_root = mapping_root / f"oversampling_{os_factor}"
        suffix = f"_os{os_factor}"
    return (
        mapping_root
        / f"sector_{t.sector:04d}"
        / f"camera_{t.camera}"
        / f"ccd_{t.ccd}"
        / f"tess_s{t.sector:04d}_{t.camera}_{t.ccd}_master_skycells_list{suffix}.csv"
    )


def _convolved_zarr_path(resolved: ResolvedTargetConfig) -> Path:
    t = resolved.target
    return (
        Path(resolved.data_root)
        / "convolved_results"
        / f"sector_{t.sector:04d}_camera_{t.camera}_ccd_{t.ccd}.zarr"
    )


def ps1_process_removed_stars_csv_path(resolved: ResolvedTargetConfig) -> Path:
    return Path(str(_convolved_zarr_path(resolved)).replace(".zarr", "_removed_stars.csv"))


def event_dir_ps1_removed_stars_csv_path(resolved: ResolvedTargetConfig) -> Path:
    from syndiff_pipeline.template_creation.processing.downsample import (
        PS1_REMOVED_STARS_CSV_FILENAME,
    )

    return Path(resolved.event_dir) / PS1_REMOVED_STARS_CSV_FILENAME


def clear_downsample_event_artifacts(resolved: ResolvedTargetConfig) -> list[str]:
    """Remove event-dir artifacts written by the downsample stage."""
    removed: list[str] = []
    csv_path = event_dir_ps1_removed_stars_csv_path(resolved)
    if csv_path.is_file():
        csv_path.unlink()
        removed.append(str(csv_path))
        log.info("Force rerun: removed file %s", csv_path)
    return removed


def clear_ps1_process_artifacts(resolved: ResolvedTargetConfig) -> list[str]:
    removed: list[str] = []
    for path in (_convolved_zarr_path(resolved), ps1_process_removed_stars_csv_path(resolved)):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
            log.info("Force rerun: removed directory %s", path)
        elif path.is_file():
            path.unlink()
            removed.append(str(path))
            log.info("Force rerun: removed file %s", path)
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


def _store_has_any_data_array(zarr_path: Path) -> bool:
    """True if the convolved store contains any ``*_data`` array directory."""
    try:
        with os.scandir(zarr_path) as it:
            return any(entry.name.endswith("_data") for entry in it)
    except OSError:
        return False


def verify_ps1_process(resolved: ResolvedTargetConfig) -> VerifyResult:
    zarr_path = _convolved_zarr_path(resolved)
    if not zarr_path.exists():
        return VerifyResult("ps1_process", False, "Convolved zarr missing", str(zarr_path))
    try:
        expected = expected_ps1_process_skycells(resolved)
    except Exception as exc:
        return VerifyResult("ps1_process", False, str(exc), str(zarr_path))
    if not expected:
        return VerifyResult("ps1_process", False, "No expected skycells from mapping CSV", str(zarr_path))

    started = time.monotonic()
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
        if _store_has_any_data_array(zarr_path):
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
        f"_dx{float(dx):.3f}_dy{float(dy):.3f}.fits.gz"
        for dx, dy in offsets
    ]
    return basenames, base


def _downsample_fits_filename_candidates(basename: str) -> list[str]:
    """Canonical ``.fits.gz`` basename plus legacy uncompressed ``.fits``."""
    if basename.endswith(".fits.gz"):
        return [basename, basename[:-3]]
    return [basename]


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


def verify_diff(
    resolved: ResolvedTargetConfig,
    runner_cfg: RunnerConfig | None = None,
    *,
    meta: dict | None = None,
) -> VerifyResult:
    from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
        diff_workspace_complete,
        diff_workspace_root,
        frozen_diff_config_for_verify,
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
        runner_cfg.diff_config_path,
        resolved.target,
        meta=meta,
    )
    event_dir = Path(resolved.event_dir)
    ws_dir = diff_workspace_root(cfg, event_dir)
    manifest_csv = manifest_path_from_output_dir(str(event_dir), None)
    if diff_workspace_complete(cfg, event_dir):
        return VerifyResult(
            "diff",
            True,
            f"Frame manifest and final pipeline outputs present under {ws_dir.name}/",
            str(ws_dir),
        )
    if not Path(manifest_csv).is_file():
        return VerifyResult("diff", False, "Missing frame manifest CSV", manifest_csv)
    if not ws_dir.is_dir():
        return VerifyResult(
            "diff",
            False,
            f"Missing workspace tree {ws_dir.name}/ under event_dir",
            str(ws_dir),
        )
    return VerifyResult(
        "diff",
        False,
        f"Final pipeline outputs missing under {ws_dir.name}/",
        str(ws_dir),
    )


def stage_absence_probe(
    resolved: ResolvedTargetConfig,
    stage: str,
    *,
    runner_cfg: RunnerConfig | None = None,
    meta: dict | None = None,
) -> AbsenceProbeResult:
    """Fast filesystem probe: skip full verify when outputs cannot exist."""
    from syndiff_pipeline.common.download import list_local_ffis, nested_ffi_dir, tesscurl_script_path
    from syndiff_pipeline.common.orchestration.event_ws_symlinks import event_templates_symlink_path
    from syndiff_pipeline.common.wcs_grouping import CLUSTER_TEMPLATE_JOB_FILENAME

    if stage == "wcs_grouping":
        job_path = Path(resolved.event_dir) / CLUSTER_TEMPLATE_JOB_FILENAME
        return (
            AbsenceProbeResult.MAYBE_PRESENT
            if job_path.is_file()
            else AbsenceProbeResult.ABSENT
        )

    if stage == "mapping":
        csv_path = _mapping_csv_path(resolved)
        return (
            AbsenceProbeResult.MAYBE_PRESENT
            if csv_path.is_file()
            else AbsenceProbeResult.ABSENT
        )

    if stage == "tess_ffi_download":
        t = resolved.target
        ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
        if list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd):
            return AbsenceProbeResult.MAYBE_PRESENT
        cached = tesscurl_script_path(ffi_leaf, t.sector)
        if Path(cached).is_file():
            return AbsenceProbeResult.ABSENT
        return AbsenceProbeResult.ABSENT

    if stage == "ps1_download":
        zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
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
        job_path = Path(resolved.event_dir) / CLUSTER_TEMPLATE_JOB_FILENAME
        templates_link = event_templates_symlink_path(resolved.event_dir)
        if job_path.is_file() or (
            templates_link.is_symlink() and templates_link.resolve().is_dir()
        ):
            return AbsenceProbeResult.MAYBE_PRESENT
        return AbsenceProbeResult.ABSENT

    if stage == "diff":
        if runner_cfg is None or not runner_cfg.diff_config_path:
            return AbsenceProbeResult.UNKNOWN
        from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
            diff_workspace_root,
            frozen_diff_config_for_verify,
        )
        from syndiff_pipeline.difference_imaging.support.manifest import (
            manifest_path_from_output_dir,
        )

        cfg = frozen_diff_config_for_verify(
            runner_cfg.diff_config_path,
            resolved.target,
            meta=meta,
        )
        event_dir = Path(resolved.event_dir)
        manifest_csv = Path(manifest_path_from_output_dir(str(event_dir), None))
        ws_dir = diff_workspace_root(cfg, event_dir)
        if ws_dir.is_dir() or manifest_csv.is_file():
            return AbsenceProbeResult.MAYBE_PRESENT
        return AbsenceProbeResult.ABSENT

    return AbsenceProbeResult.UNKNOWN


VERIFY_FUNCS = {
    "tess_ffi_download": verify_tess_ffi_download,
    "wcs_grouping": verify_wcs_grouping,
    "mapping": verify_mapping,
    "ps1_download": verify_ps1_download,
    "ps1_process": verify_ps1_process,
    "downsample": verify_downsample,
    "diff": verify_diff,
}


def verify_stage(
    resolved: ResolvedTargetConfig,
    stage: str,
    runner_cfg: RunnerConfig | None = None,
    *,
    meta: dict | None = None,
) -> VerifyResult:
    fn = VERIFY_FUNCS.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage: {stage!r}")
    if stage == "diff":
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
    if stage == "downsample":
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
        expected = expected_ps1_process_skycells(resolved)
        zarr_path = _convolved_zarr_path(resolved)
        if not zarr_path.exists():
            return len(expected), 0, [str(zarr_path)]
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
        zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
        result = verify_ps1_download(resolved)
        produced = 0
        if result.ok:
            produced = len(expected)
        return len(expected), produced, [str(zarr_path)]
    if stage == "tess_ffi_download":
        from syndiff_pipeline.common.download import expected_ffi_basenames, list_local_ffis, nested_ffi_dir

        t = resolved.target
        ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
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
    if stages is None:
        from syndiff_pipeline.pipeline_spec import STAGE_NAMES

        stages = list(STAGE_NAMES)
    return [verify_stage(resolved, s, runner_cfg, meta=meta) for s in stages]


def verify_all(cfg: RunnerConfig, targets: List[Target], stages: Optional[List[str]] = None) -> List[VerifyResult]:
    out: List[VerifyResult] = []
    for t in targets:
        resolved = resolve_config(t, cfg)
        for r in verify_target(resolved, cfg, stages):
            out.append(r)
    return out
