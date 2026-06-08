"""Artifact verification for template pipeline stages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from syndiff_pipeline.download import expected_ffi_basenames, list_local_ffis, nested_ffi_dir
from syndiff_pipeline.template.csv_utils import get_all_padding_cells, load_csv_data
from syndiff_pipeline.template.downsample import (
    load_cluster_template_job_payload,
    offsets_from_cluster_job_payload,
    roi_tuple_from_cluster_job_payload,
)
from syndiff_pipeline.template.ps1_process import expected_convolved_skycells
from syndiff_pipeline.template_runner.runner_config import ResolvedTargetConfig, resolve_config, RunnerConfig
from syndiff_pipeline.template_runner.state import STAGE_NAMES
from syndiff_pipeline.template_runner.targets import Target

log = logging.getLogger(__name__)

# Bump when the manifest JSON schema changes; a mismatch invalidates a manifest.
MANIFEST_SCHEMA_VERSION = 1


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


def config_fingerprint(resolved: ResolvedTargetConfig, stage: str) -> str:
    """Stable hash of the stage params that affect this stage's outputs.

    A change to any fingerprinted param invalidates a stale manifest so a new
    config never reuses outputs produced under a different configuration.
    """
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


def write_manifest(
    manifest_path,
    resolved: ResolvedTargetConfig,
    stage: str,
    produced_paths,
    expected_count: int,
    produced_count: int,
) -> dict:
    """Atomically write a completion manifest (tmp file + rename).

    Schema: schema_version, stage, expected_count, produced_count, artifacts
    (list of paths), config_fingerprint, completed_at (iso utc).
    """
    path = Path(manifest_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "expected_count": int(expected_count),
        "produced_count": int(produced_count),
        "artifacts": [str(p) for p in (produced_paths or [])],
        "config_fingerprint": config_fingerprint(resolved, stage),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
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


def manifest_valid(manifest: dict, resolved: ResolvedTargetConfig, stage: str) -> bool:
    """True if *manifest* is well-formed, matches the current config, and all
    listed artifacts still exist on disk."""
    if not isinstance(manifest, dict):
        return False
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    if manifest.get("stage") != stage:
        return False
    if manifest.get("config_fingerprint") != config_fingerprint(resolved, stage):
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


def verify_tess_ffi_download(resolved: ResolvedTargetConfig) -> VerifyResult:
    t = resolved.target
    ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
    expected = expected_ffi_basenames(t.sector, t.camera, t.ccd, output_dir=ffi_leaf)
    if expected is None:
        files = list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd)
        if not files:
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
            f"Cannot verify completeness ({len(files)} local files; tesscurl manifest unavailable)",
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

    existing = {Path(p).name for p in list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd)}
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
    job_path = Path(resolved.handoff_dir) / "cluster_template_job.json"
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


def _ps1_download_array_complete(group, array_name: str) -> bool:
    if array_name not in group:
        return False
    array = group[array_name]
    if array.shape == (0,) or array.size == 0:
        return False
    try:
        _ = array[0:1, 0:1]
    except Exception:
        return False
    return True


def _ps1_download_skycell_complete(root, skycell_name: str) -> bool:
    projection_id = _projection_from_skycell_name(skycell_name)
    if not projection_id or projection_id not in root or skycell_name not in root[projection_id]:
        return False
    group = root[projection_id][skycell_name]
    return all(_ps1_download_array_complete(group, name) for name in _ps1_download_expected_array_names())


def verify_ps1_download(resolved: ResolvedTargetConfig) -> VerifyResult:
    zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
    try:
        expected_skycells = _expected_ps1_download_skycells(resolved)
    except FileNotFoundError as exc:
        return VerifyResult("ps1_download", False, str(exc), str(zarr_path))
    except ValueError as exc:
        return VerifyResult("ps1_download", False, str(exc), str(zarr_path))

    if not zarr_path.exists():
        return VerifyResult(
            "ps1_download",
            False,
            f"Shared zarr store missing (0/{len(expected_skycells)} skycells)",
            str(zarr_path),
        )
    try:
        import zarr

        root = zarr.open(str(zarr_path), mode="r")
        complete = sum(
            1 for skycell in expected_skycells if _ps1_download_skycell_complete(root, skycell)
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
    except Exception as exc:
        return VerifyResult("ps1_download", False, f"Cannot verify PS1 zarr: {exc}", str(zarr_path))


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


def expected_ps1_process_skycells(resolved: ResolvedTargetConfig) -> list[str]:
    t = resolved.target
    try:
        return expected_convolved_skycells(
            resolved.data_root,
            t.sector,
            t.camera,
            t.ccd,
            projections_limit=resolved.stages.ps1_process.projections_limit,
        )
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


def _count_convolved_data_arrays(zarr_root, expected_names: list[str]) -> tuple[int, list[str]]:
    missing: list[str] = []
    saved = 0
    for name in expected_names:
        key = f"{name}_data"
        if key not in zarr_root:
            missing.append(name)
            continue
        if int(zarr_root[key].size) <= 0:
            missing.append(name)
            continue
        saved += 1
    return saved, missing


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
    try:
        import zarr

        root = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:
        return VerifyResult("ps1_process", False, f"Cannot verify convolved zarr: {exc}", str(zarr_path))

    # Detect an empty/all-empty store up front so the message is explicit.
    all_data_keys = [str(k) for k in root.array_keys() if str(k).endswith("_data")]
    non_empty_total = sum(1 for k in all_data_keys if int(root[k].size) > 0)
    if non_empty_total == 0:
        if all_data_keys:
            msg = (
                f"Convolved zarr has {len(all_data_keys)} *_data arrays but all are empty: "
                f"0/{len(expected)} skycells saved"
            )
        else:
            msg = f"Convolved zarr store is empty (no *_data arrays): 0/{len(expected)} skycells saved"
        return VerifyResult("ps1_process", False, msg, str(zarr_path))

    saved, missing = _count_convolved_data_arrays(root, expected)
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
    t = resolved.target
    ds = resolved.stages.downsample
    job_path = Path(resolved.handoff_dir) / "cluster_template_job.json"
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
        f"_dx{float(dx):.3f}_dy{float(dy):.3f}.fits"
        for dx, dy in offsets
    ]
    return basenames, base


def _find_downsample_fits(base: Path, t, basename: str) -> str | None:
    """Locate a per-offset FITS under any ``sector..._ccd<ccd>*`` output dir.

    The writer's output directory carries the full ROI suffix (which depends on
    the base frame shape) while the *filename* only tags ROI when x_min/y_min are
    nonzero, so we glob across matching dirs and match on the authoritative
    filename rather than reconstructing the exact directory name.
    """
    pattern = f"sector{t.sector:04d}_camera{t.camera}_ccd{t.ccd}*/{basename}"
    matches = sorted(base.glob(pattern))
    return str(matches[0]) if matches else None


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
    return VerifyResult(
        "downsample",
        True,
        f"All {n_expected} offset FITS present",
        sample,
    )


VERIFY_FUNCS = {
    "tess_ffi_download": verify_tess_ffi_download,
    "wcs_grouping": verify_wcs_grouping,
    "mapping": verify_mapping,
    "ps1_download": verify_ps1_download,
    "ps1_process": verify_ps1_process,
    "downsample": verify_downsample,
}


def verify_stage(resolved: ResolvedTargetConfig, stage: str) -> VerifyResult:
    fn = VERIFY_FUNCS.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage: {stage!r}")
    return fn(resolved)


def stage_complete(
    resolved: ResolvedTargetConfig,
    stage: str,
    manifest_path: str | None = None,
) -> bool:
    """Return True if the stage outputs are complete.

    Manifest-first: when *manifest_path* points to a valid manifest (well-formed,
    schema version ok, config fingerprint matches, and every listed artifact still
    exists on disk), the stage is complete. Otherwise fall back to the hardened
    on-disk check ``verify_stage(resolved, stage).ok``. An ``unknown`` on-disk
    result is treated conservatively (not complete).
    """
    if manifest_path is not None:
        manifest = read_manifest(manifest_path)
        if manifest is not None and manifest_valid(manifest, resolved, stage):
            return True
    result = verify_stage(resolved, stage)
    if result.unknown:
        return False
    return result.ok


def collect_stage_artifacts(resolved: ResolvedTargetConfig, stage: str) -> tuple[int, int, list[str]]:
    """Return (expected_count, produced_count, artifact_paths) for manifest writing."""
    if stage == "downsample":
        paths = expected_downsample_fits_paths(resolved)
        existing = [str(p) for p in paths if p.is_file()]
        return len(paths), len(existing), existing
    if stage == "ps1_process":
        expected = expected_ps1_process_skycells(resolved)
        zarr_path = _convolved_zarr_path(resolved)
        try:
            import zarr

            root = zarr.open(str(zarr_path), mode="r")
            saved, _missing = _count_convolved_data_arrays(root, expected)
            return len(expected), saved, [str(zarr_path)]
        except Exception:
            return len(expected), 0, [str(zarr_path)]
    if stage == "mapping":
        csv_path = _mapping_csv_path(resolved)
        ok = csv_path.is_file()
        return 1, int(ok), [str(csv_path)] if ok else []
    if stage == "wcs_grouping":
        job_path = Path(resolved.handoff_dir) / "cluster_template_job.json"
        ok = job_path.is_file()
        return 1, int(ok), [str(job_path)] if ok else []
    if stage == "ps1_download":
        expected = _expected_ps1_download_skycells(resolved)
        zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
        result = verify_ps1_download(resolved)
        produced = 0
        if result.ok:
            produced = len(expected)
        return len(expected), produced, [str(zarr_path)]
    if stage == "tess_ffi_download":
        t = resolved.target
        ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
        expected = expected_ffi_basenames(t.sector, t.camera, t.ccd, output_dir=ffi_leaf) or []
        files = list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd)
        return len(expected), len(files), [str(p) for p in files]
    result = verify_stage(resolved, stage)
    path = result.path or ""
    return 1, int(result.ok), [path] if path else []


def verify_target(resolved: ResolvedTargetConfig, stages: Optional[List[str]] = None) -> List[VerifyResult]:
    stages = stages or list(STAGE_NAMES)
    return [verify_stage(resolved, s) for s in stages]


def verify_all(cfg: RunnerConfig, targets: List[Target], stages: Optional[List[str]] = None) -> List[VerifyResult]:
    out: List[VerifyResult] = []
    for t in targets:
        resolved = resolve_config(t, cfg)
        for r in verify_target(resolved, stages):
            out.append(r)
    return out
