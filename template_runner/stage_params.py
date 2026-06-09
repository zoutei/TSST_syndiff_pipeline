"""Strict allow-list validation for template-runner stage YAML params."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, FrozenSet, Type

WCS_GROUPING_ALLOWED = frozenset(
    {
        "offset_threshold",
        "wcs_drift_savgol_window",
        "wcs_drift_savgol_polyorder",
        "bkg_vector_path",
        "crop_quadrant",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "x_left_dead",
        "x_right_dead",
        "y_edge_strip",
    }
)
MAPPING_ALLOWED = frozenset(
    {
        "buffer",
        "tess_buffer",
        "pad_distance",
        "edge_exclusion",
        "edge_buffer_large",
        "edge_buffer_small",
        "n_threads",
        "max_workers",
        "oversampling_factor",
        "overwrite",
        "skip_download_catalog",
        "executor",
        "condor_request_cpus",
        "condor_request_memory",
        "condor_requirements",
        "condor_rank",
    }
)
PS1_DOWNLOAD_ALLOWED = frozenset(
    {"num_workers", "use_local_files", "local_data_path", "overwrite", "log_level"}
)
PS1_PROCESS_ALLOWED = frozenset(
    {
        "projections_limit",
        "psf_sigma",
        "ps1_source",
        "num_ingest_workers",
        "enable_saturation_correction",
        "remove_saturated_stars",
        "catalog_path",
        "bright_star_mag_threshold",
        "executor",
        "condor_request_cpus",
        "condor_request_memory",
        "condor_requirements",
        "condor_rank",
    }
)
DOWNSAMPLE_ALLOWED = frozenset(
    {
        "ignore_mask_bits",
        "oversampling_factor",
        "mapping_dir",
        "convolved_dir",
        "output_base",
        "single_offset",
        "allow_reference_ffi_mismatch",
    }
)


def _merge_dataclass(cls: Type, data: Dict[str, Any]):
    valid = {f.name for f in fields(cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def validate_stage_keys(stage_dict: dict, allowed: FrozenSet[str], stage_name: str) -> None:
    unknown = set(stage_dict) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in stages.{stage_name}: {sorted(unknown)}")


@dataclass
class WcsGroupingStageParams:
    offset_threshold: float = 0.01
    wcs_drift_savgol_window: int | None = 11
    wcs_drift_savgol_polyorder: int = 2
    bkg_vector_path: str | None = None
    crop_quadrant: str = "full"
    x_min: int | None = None
    x_max: int | None = None
    y_min: int | None = None
    y_max: int | None = None
    x_left_dead: int = 44
    x_right_dead: int = 44
    y_edge_strip: int = 30


@dataclass
class MappingStageParams:
    buffer: int = 200
    tess_buffer: int = 150
    pad_distance: int = 480
    edge_exclusion: int = 10
    edge_buffer_large: int = 410
    edge_buffer_small: int = 70
    n_threads: int = 8
    max_workers: int | None = None
    oversampling_factor: int = 1
    overwrite: bool = True
    skip_download_catalog: bool = False
    executor: str = "condor"
    condor_request_cpus: int = 16
    condor_request_memory: int = 100_000
    condor_requirements: str | None = "Memory <= 500000 && LoadAvg < 10"
    condor_rank: str | None = "-LoadAvg"


@dataclass
class Ps1DownloadStageParams:
    num_workers: int = 8
    use_local_files: bool = False
    local_data_path: str = "data/ps1_skycells"
    overwrite: bool = False
    log_level: str = "INFO"


@dataclass
class Ps1ProcessStageParams:
    projections_limit: int | None = None
    psf_sigma: float = 60.0
    ps1_source: str = "zarr"
    num_ingest_workers: int = 16
    enable_saturation_correction: bool = True
    remove_saturated_stars: bool = False
    catalog_path: str | None = None
    bright_star_mag_threshold: float = 13.0
    executor: str = "condor"
    condor_request_cpus: int = 64
    condor_request_memory: int = 500_000
    condor_requirements: str | None = "Memory >= 500000 && LoadAvg < 10"
    condor_rank: str | None = "-LoadAvg"


@dataclass
class DownsampleStageParams:
    ignore_mask_bits: list = None  # type: ignore[assignment]
    oversampling_factor: int = 1
    mapping_dir: str | None = None
    convolved_dir: str | None = None
    output_base: str | None = None
    single_offset: bool = False
    allow_reference_ffi_mismatch: bool = False

    def __post_init__(self):
        if self.ignore_mask_bits is None:
            object.__setattr__(self, "ignore_mask_bits", [12])


@dataclass
class ResourcePoolParams:
    max_concurrent: int = 1


@dataclass
class TemplateStageParams:
    wcs_grouping: WcsGroupingStageParams
    mapping: MappingStageParams
    ps1_download: Ps1DownloadStageParams
    ps1_process: Ps1ProcessStageParams
    downsample: DownsampleStageParams


def parse_stage_params(stages_raw: dict) -> TemplateStageParams:
    stages_raw = stages_raw or {}
    wg = stages_raw.get("wcs_grouping", {}) or {}
    mp = stages_raw.get("mapping", {}) or {}
    pd = stages_raw.get("ps1_download", {}) or {}
    pp = stages_raw.get("ps1_process", {}) or {}
    ds = stages_raw.get("downsample", {}) or {}
    validate_stage_keys(wg, WCS_GROUPING_ALLOWED, "wcs_grouping")
    validate_stage_keys(mp, MAPPING_ALLOWED, "mapping")
    validate_stage_keys(pd, PS1_DOWNLOAD_ALLOWED, "ps1_download")
    validate_stage_keys(pp, PS1_PROCESS_ALLOWED, "ps1_process")
    validate_stage_keys(ds, DOWNSAMPLE_ALLOWED, "downsample")
    if pp.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.ps1_process.executor must be 'local' or 'condor'")
    ps1_source = pp.get("ps1_source", "zarr")
    if ps1_source not in ("zarr", "stream"):
        raise ValueError("stages.ps1_process.ps1_source must be 'zarr' or 'stream'")
    if mp.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.mapping.executor must be 'local' or 'condor'")
    return TemplateStageParams(
        wcs_grouping=_merge_dataclass(WcsGroupingStageParams, wg),
        mapping=_merge_dataclass(MappingStageParams, mp),
        ps1_download=_merge_dataclass(Ps1DownloadStageParams, pd),
        ps1_process=_merge_dataclass(Ps1ProcessStageParams, pp),
        downsample=_merge_dataclass(DownsampleStageParams, ds),
    )
