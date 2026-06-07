"""Artifact verification for template pipeline stages."""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from syndiff_pipeline.download import list_local_ffis, nested_ffi_dir
from syndiff_pipeline.template.csv_utils import load_csv_data
from syndiff_pipeline.template_runner.runner_config import ResolvedTargetConfig, resolve_config, RunnerConfig
from syndiff_pipeline.template_runner.state import STAGE_NAMES
from syndiff_pipeline.template_runner.targets import Target

log = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    stage: str
    ok: bool
    message: str
    path: str | None = None


def verify_tess_ffi_download(resolved: ResolvedTargetConfig) -> VerifyResult:
    t = resolved.target
    ffi_leaf = nested_ffi_dir(t.sector, t.camera, t.ccd, root=resolved.ffi_dir)
    files = list_local_ffis(ffi_leaf, t.sector, t.camera, t.ccd)
    if files:
        return VerifyResult("tess_ffi_download", True, f"{len(files)} FFI files", ffi_leaf)
    return VerifyResult("tess_ffi_download", False, "No FFI files found", ffi_leaf)


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


def verify_ps1_download(resolved: ResolvedTargetConfig) -> VerifyResult:
    zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
    if not zarr_path.exists():
        return VerifyResult("ps1_download", False, "Shared zarr store missing", str(zarr_path))
    try:
        import zarr

        root = zarr.open(str(zarr_path), mode="r")
        if len(list(root.groups())) == 0 and len(root.array_keys()) == 0:
            return VerifyResult("ps1_download", False, "Zarr store empty", str(zarr_path))
    except Exception as exc:
        return VerifyResult("ps1_download", False, f"Cannot open zarr: {exc}", str(zarr_path))
    return VerifyResult("ps1_download", True, "Shared zarr store present", str(zarr_path))


def _mapping_csv_path(resolved: ResolvedTargetConfig) -> Path:
    """Path to the master skycells CSV (matches ``verify_mapping`` layout)."""
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


def _expected_ps1_skycell_count(resolved: ResolvedTargetConfig) -> int:
    """Skycells that ``ps1_process`` should write for this target/config."""
    csv_path = _mapping_csv_path(resolved)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Master skycells CSV missing: {csv_path}")
    df = load_csv_data(str(csv_path))
    if "projection" not in df.columns:
        raise ValueError(f"Master skycells CSV missing projection column: {csv_path}")
    projections = sorted(df["projection"].astype(str).unique())
    limit = resolved.stages.ps1_process.projections_limit
    if limit:
        projections = projections[: int(limit)]
    return int(len(df[df["projection"].astype(str).isin(projections)]))


def _count_convolved_data_arrays(zarr_root) -> tuple[int, list[str]]:
    """Return (non-empty *_data array count, all *_data array names)."""
    data_keys = [str(k) for k in zarr_root.array_keys() if str(k).endswith("_data")]
    non_empty = 0
    for key in data_keys:
        if int(zarr_root[key].size) > 0:
            non_empty += 1
    return non_empty, data_keys


def verify_ps1_process(resolved: ResolvedTargetConfig) -> VerifyResult:
    zarr_path = _convolved_zarr_path(resolved)
    if not zarr_path.exists():
        return VerifyResult("ps1_process", False, "Convolved zarr missing", str(zarr_path))
    try:
        import zarr

        root = zarr.open(str(zarr_path), mode="r")
        saved, data_keys = _count_convolved_data_arrays(root)
        if saved == 0:
            if data_keys:
                msg = f"Convolved zarr has {len(data_keys)} *_data arrays but all are empty"
            else:
                msg = "Convolved zarr store is empty (no *_data arrays)"
            return VerifyResult("ps1_process", False, msg, str(zarr_path))

        expected = _expected_ps1_skycell_count(resolved)
        if saved < expected:
            return VerifyResult(
                "ps1_process",
                False,
                f"Partial convolved zarr: {saved}/{expected} skycells saved",
                str(zarr_path),
            )
        return VerifyResult(
            "ps1_process",
            True,
            f"Convolved zarr complete ({saved}/{expected} skycells)",
            str(zarr_path),
        )
    except FileNotFoundError as exc:
        return VerifyResult("ps1_process", False, str(exc), str(zarr_path))
    except Exception as exc:
        return VerifyResult("ps1_process", False, f"Cannot verify convolved zarr: {exc}", str(zarr_path))


def verify_downsample(resolved: ResolvedTargetConfig) -> VerifyResult:
    t = resolved.target
    base = Path(resolved.stages.downsample.output_base or resolved.template_output_base)
    matches: list[str] = []
    for d in base.glob(f"sector{t.sector:04d}_camera{t.camera}_ccd{t.ccd}*"):
        matches.extend(glob.glob(str(d / "syndiff_template_*.fits")))
    if matches:
        return VerifyResult("downsample", True, f"{len(matches)} template FITS", matches[0])
    return VerifyResult("downsample", False, "No syndiff_template_*.fits found", str(base))


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
