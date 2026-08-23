"""Strict allow-list validation for template-runner stage YAML params."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, FrozenSet, Type

WCS_GROUPING_ALLOWED = frozenset(
    {
        "offset_threshold",
        "wcs_drift_savgol_window",
        "wcs_drift_savgol_polyorder",
        "bkg_vector_path",
        "crop_mode",
        "crop_box_size",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "x_left_dead",
        "x_right_dead",
        "y_edge_strip",
        "geometry_mode",
        "grouping_quantum_ps1_px",
        "screen_earth_moon_angles",
        "earth_deg_min",
        "moon_deg_min",
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
        "reference_ffi",
        "reference_ffi_selection",
        "bkg_vector_path",
        "wcs_drift_savgol_window",
        "wcs_drift_savgol_polyorder",
        "screen_earth_moon_angles",
        "earth_deg_min",
        "moon_deg_min",
        "max_smoothed_residual",
        "executor",
        "condor_request_cpus",
        "condor_request_memory",
        "host_stats_min_mem_mb",
        "host_stats_max_load15",
        "x_left_dead",
        "x_right_dead",
        "y_edge_strip",
        "template_conv_pad_spare_px",
        "sci_fwhm",
        "store_name",
        "wcs_source",
        "temporal_wcs_version",
        "mapgrid_version",
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
        "use_shared_convolved_store",
        "write_per_scc_convolved_zarr",
        "executor",
        "condor_request_cpus",
        "condor_request_memory",
        "host_stats_min_mem_mb",
        "host_stats_max_load15",
    }
)
DIFF_ALLOWED = frozenset({"executor"})
STAR_ALLOWED = frozenset({"executor"})
PHOTOMETRY_ALLOWED = frozenset({"executor"})
REMAP_ALLOWED = frozenset(
    {
        "cache_quantum_ps1_px",
        "keying",
        "intra_skycell_R",
        "rebuild_remap_cache",
        "rebuild_inter_skycell_cache",
        "raw_drift_outlier_sigma",
        "n_jobs",
        "stage_regmaps_to_scratch",
        "executor",
        "condor_request_cpus",
        "condor_request_memory",
        "condor_request_disk",
        "host_stats_min_mem_mb",
        "host_stats_max_load15",
        "store_name",
        "drift_source",
        "apply_intra_skycell",
        "apply_inter_skycell",
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
        "n_jobs",
        "skycells_per_batch",
        "log_level",
        "stage_regmaps_to_scratch",
        "checkpoint_skycells",
        "executor",
        "condor_request_cpus",
        "condor_request_memory",
        "condor_request_disk",
        "host_stats_min_mem_mb",
        "host_stats_max_load15",
        "geometry_mode",
        "materialize_fits",
        "rebuild_field_store",
        "prematerialize_top_n",
        "apply_intra_skycell",
        "apply_inter_skycell",
        "write_split_contribs",
        "remap_store_name",
        "output_store_name",
    }
)


def _merge_dataclass(cls: Type, data: Dict[str, Any]):
    """Merge dataclass.
    
    Parameters
    ----------
    data : Dict[str, Any]"""
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
    """Validate stage keys.
    
    Parameters
    ----------
    stage_dict : dict
    allowed : FrozenSet[str]
    stage_name : str"""
    unknown = set(stage_dict) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in stages.{stage_name}: {sorted(unknown)}")


@dataclass
class WcsGroupingStageParams:
    """WcsGroupingStageParams."""
    offset_threshold: float = 0.01
    wcs_drift_savgol_window: int | None = 11
    wcs_drift_savgol_polyorder: int = 2
    bkg_vector_path: str | None = None
    crop_mode: str = "full"
    crop_box_size: int = 1024
    x_min: int | None = None
    x_max: int | None = None
    y_min: int | None = None
    y_max: int | None = None
    x_left_dead: int = 44
    x_right_dead: int = 44
    y_edge_strip: int = 30
    # field (default) = SCC signature groups; linear = target-anchored dx/dy groups
    geometry_mode: str = "field"
    grouping_quantum_ps1_px: float = 1.0
    screen_earth_moon_angles: bool = False
    earth_deg_min: float = 45.0
    moon_deg_min: float = 25.0


@dataclass
class MappingStageParams:
    """MappingStageParams."""
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
    reference_ffi: str | None = None
    reference_ffi_selection: str = "drift_arc_midpoint"
    bkg_vector_path: str | None = None
    wcs_drift_savgol_window: int | None = 11
    wcs_drift_savgol_polyorder: int = 2
    screen_earth_moon_angles: bool = False
    earth_deg_min: float = 45.0
    moon_deg_min: float = 25.0
    max_smoothed_residual: float = 0.05
    executor: str = "condor"
    condor_request_cpus: int = 16
    condor_request_memory: int = 100_000
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0
    x_left_dead: int = 44
    x_right_dead: int = 44
    y_edge_strip: int = 30
    template_conv_pad_spare_px: int = 4
    sci_fwhm: float = 1.88
    # Named mapping lane (for example ``tvwcs`` -> mapping_tvwcs/).
    store_name: str | None = None
    # Geometry source is explicit in frozen configs so mapping cannot silently
    # fall back to the SPOC header WCS for a distortion-aware lane.
    wcs_source: str = "spoc_ffi_wcs"
    temporal_wcs_version: str = "temporal_cheb5_bspline_v1"
    # Mapping geometry schema. MAPGRID=3 is the only supported contract.
    mapgrid_version: int = 3

    def __post_init__(self) -> None:
        from syndiff_pipeline.common.scc_paths import normalize_store_name

        object.__setattr__(self, "store_name", normalize_store_name(self.store_name))
        source = str(self.wcs_source or "spoc_ffi_wcs").strip().lower()
        if source not in ("spoc_ffi_wcs", "temporal_wcs"):
            raise ValueError(
                "stages.mapping.wcs_source must be 'spoc_ffi_wcs' or 'temporal_wcs', "
                f"got {self.wcs_source!r}"
            )
        object.__setattr__(self, "wcs_source", source)
        version = int(self.mapgrid_version)
        if version != 3:
            raise ValueError("stages.mapping.mapgrid_version must be exactly 3")
        object.__setattr__(self, "mapgrid_version", version)
        version = str(self.temporal_wcs_version or "").strip()
        if not version:
            raise ValueError("stages.mapping.temporal_wcs_version must be non-empty")
        object.__setattr__(self, "temporal_wcs_version", version)


@dataclass
class Ps1DownloadStageParams:
    """Ps1DownloadStageParams."""
    num_workers: int = 8
    use_local_files: bool = False
    local_data_path: str = "data/ps1_skycells"
    overwrite: bool = False
    log_level: str = "INFO"


@dataclass
class Ps1ProcessStageParams:
    """Ps1ProcessStageParams."""
    projections_limit: int | None = None
    psf_sigma: float = 40.0
    ps1_source: str = "zarr"
    num_ingest_workers: int = 16
    enable_saturation_correction: bool = False
    remove_saturated_stars: bool = True
    catalog_path: str | None = None
    bright_star_mag_threshold: float = 13.0
    use_shared_convolved_store: bool = False
    write_per_scc_convolved_zarr: bool = True
    executor: str = "condor"
    condor_request_cpus: int = 64
    condor_request_memory: int = 300_000
    host_stats_min_mem_mb: int = 300_000
    host_stats_max_load15: float = 10.0

    def __post_init__(self) -> None:
        if self.use_shared_convolved_store and self.write_per_scc_convolved_zarr:
            raise ValueError(
                "stages.ps1_process: use_shared_convolved_store=True requires "
                "write_per_scc_convolved_zarr=False (hard cut on per-SCC convolved.zarr)"
            )


@dataclass
class DiffStageParams:
    """DiffStageParams."""
    executor: str = "condor"


@dataclass
class StarStageParams:
    """StarStageParams."""
    executor: str = "condor"


@dataclass
class PhotometryStageParams:
    """PhotometryStageParams."""
    executor: str = "condor"


@dataclass
class RemapStageParams:
    """RemapStageParams."""
    cache_quantum_ps1_px: float = 1.0
    keying: str = "absolute"
    # Intra-skycell (L4a) boundary dilation radius; inter-skycell (L4b) gated by
    # ``apply_inter_skycell``.
    intra_skycell_R: int = 1
    # Pre-SG MAD gate on raw TESS drift (per orbit). ``None`` disables.
    raw_drift_outlier_sigma: float | None = 5.0
    rebuild_remap_cache: bool = False
    rebuild_inter_skycell_cache: bool = False
    n_jobs: int = 16
    # Defaults to False (regmaps read directly from NFS). Set true explicitly
    # to stage regmaps to the execute host's local scratch disk first, but
    # only when that disk is known to be ample for the regmap set (os4+
    # regmaps for large SCCs can exceed a default execute-host scratch disk
    # and cause ENOSPC); see resolve_stage_regmaps_to_scratch.
    stage_regmaps_to_scratch: bool | None = None
    executor: str = "condor"
    condor_request_cpus: int = 32
    condor_request_memory: int = 128_000
    condor_request_disk: int | None = None  # MB; None → omit request_disk
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0
    # Named remap lane → remap_{store_name}/; None → remap/
    store_name: str | None = None
    drift_source: str = "per_skycell_temporal_wcs"
    apply_intra_skycell: bool = True
    apply_inter_skycell: bool = True

    def __post_init__(self):
        from syndiff_pipeline.common.scc_paths import normalize_store_name

        object.__setattr__(self, "store_name", normalize_store_name(self.store_name))
        ds = str(self.drift_source or "per_skycell_temporal_wcs").strip().lower()
        # Keep old frozen YAML readable. New configs should use the explicit
        # names; dispatch accepts both during migration of existing stores.
        if ds not in (
            "per_skycell_temporal_wcs",
            "point_ffi_wcs",
            "per_skycell",
            "point",
        ):
            raise ValueError(
                "stages.remap.drift_source must be 'per_skycell_temporal_wcs' "
                "or 'point_ffi_wcs' (legacy aliases: 'per_skycell', 'point'), "
                f"got {self.drift_source!r}"
            )
        object.__setattr__(self, "drift_source", ds)


@dataclass
class DownsampleStageParams:
    """DownsampleStageParams."""
    ignore_mask_bits: list = None  # type: ignore[assignment]
    oversampling_factor: int = 1
    mapping_dir: str | None = None
    convolved_dir: str | None = None
    output_base: str | None = None
    single_offset: bool = False
    allow_reference_ffi_mismatch: bool = False
    n_jobs: int = 16
    skycells_per_batch: int = 20
    log_level: str = "INFO"
    # Defaults to False (regmaps read directly from NFS); set true explicitly
    # to stage to local scratch disk when it's known to be ample.
    stage_regmaps_to_scratch: bool | None = None
    checkpoint_skycells: bool = False
    executor: str = "local"
    condor_request_cpus: int = 16
    condor_request_memory: int = 128_000
    condor_request_disk: int | None = None  # MB; None → omit request_disk
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0
    geometry_mode: str = "field"
    materialize_fits: bool = False
    rebuild_field_store: bool = False
    prematerialize_top_n: int | None = None
    apply_intra_skycell: bool = True
    apply_inter_skycell: bool = True
    # H.1 (2026-08-23): also write the group-independent interior +
    # per-group seam-delta contrib split (interior_contribs/,
    # seam_delta_contribs/), alongside the existing plain contribs/ --
    # additive, does not touch/replace existing contribs/. Set true to
    # backfill an already-complete store for the H.2 patch-cache
    # convolution path; see convolved_templates_patch_cache.py and
    # spicy-squishing-ritchie.md Part H.
    write_split_contribs: bool = False
    # INPUT: which remap lane to read; None → inherit stages.remap.store_name
    remap_store_name: str | None = None
    # OUTPUT: which templates lane to write; None → templates/
    output_store_name: str | None = None

    def __post_init__(self):
        """Post init."""
        from syndiff_pipeline.common.scc_paths import normalize_store_name

        if self.ignore_mask_bits is None:
            object.__setattr__(self, "ignore_mask_bits", [12])
        level = (self.log_level or "INFO").upper()
        if level not in ("INFO", "DEBUG"):
            raise ValueError(f"log_level must be INFO or DEBUG, got {self.log_level!r}")
        object.__setattr__(self, "log_level", level)
        # Both false is valid for coarse/bootstrap templates (plain per-skycell roll).
        object.__setattr__(
            self, "remap_store_name", normalize_store_name(self.remap_store_name)
        )
        object.__setattr__(
            self, "output_store_name", normalize_store_name(self.output_store_name)
        )


@dataclass
class ResourcePoolParams:
    """ResourcePoolParams."""
    max_concurrent: int = 1


@dataclass
class TemplateStageParams:
    """TemplateStageParams."""
    wcs_grouping: WcsGroupingStageParams
    mapping: MappingStageParams
    ps1_download: Ps1DownloadStageParams
    ps1_process: Ps1ProcessStageParams
    remap: RemapStageParams
    downsample: DownsampleStageParams
    diff: DiffStageParams = field(default_factory=DiffStageParams)
    star: StarStageParams = field(default_factory=StarStageParams)
    photometry: PhotometryStageParams = field(default_factory=PhotometryStageParams)


def _filter_allowed_keys(stage_dict: dict, allowed: FrozenSet[str]) -> dict:
    """Keep only allow-listed keys (for non-strict frozen-config loading)."""
    return {k: v for k, v in stage_dict.items() if k in allowed}


def parse_stage_params(stages_raw: dict, *, strict: bool = True) -> TemplateStageParams:
    """Parse stage params.

    Parameters
    ----------
    stages_raw : dict
    strict : bool, optional
        When False, unknown stage names and keys are dropped instead of
        raising (for frozen run configs written by newer feature branches).

    Returns
    -------
    TemplateStageParams"""
    stages_raw = stages_raw or {}
    if strict and "templates" in stages_raw:
        raise ValueError(
            "stages.templates was renamed to stages.downsample; update your config"
        )
    wg = stages_raw.get("wcs_grouping", {}) or {}
    mp = stages_raw.get("mapping", {}) or {}
    pd = stages_raw.get("ps1_download", {}) or {}
    pp = stages_raw.get("ps1_process", {}) or {}
    rm = stages_raw.get("remap", {}) or stages_raw.get("skycell_remap", {}) or {}
    if strict:
        ds = stages_raw.get("downsample", {}) or {}
    else:
        ds = stages_raw.get("downsample", {}) or stages_raw.get("templates", {}) or {}
    df = stages_raw.get("diff", {}) or {}
    st = stages_raw.get("star", {}) or {}
    ph = stages_raw.get("photometry", {}) or {}
    if not strict:
        wg = _filter_allowed_keys(wg, WCS_GROUPING_ALLOWED)
        mp = _filter_allowed_keys(mp, MAPPING_ALLOWED)
        pd = _filter_allowed_keys(pd, PS1_DOWNLOAD_ALLOWED)
        pp = _filter_allowed_keys(pp, PS1_PROCESS_ALLOWED)
        rm = _filter_allowed_keys(rm, REMAP_ALLOWED)
        ds = _filter_allowed_keys(ds, DOWNSAMPLE_ALLOWED)
        df = _filter_allowed_keys(df, DIFF_ALLOWED)
        st = _filter_allowed_keys(st, STAR_ALLOWED)
        ph = _filter_allowed_keys(ph, PHOTOMETRY_ALLOWED)
    validate_stage_keys(wg, WCS_GROUPING_ALLOWED, "wcs_grouping")
    validate_stage_keys(mp, MAPPING_ALLOWED, "mapping")
    validate_stage_keys(pd, PS1_DOWNLOAD_ALLOWED, "ps1_download")
    validate_stage_keys(pp, PS1_PROCESS_ALLOWED, "ps1_process")
    validate_stage_keys(rm, REMAP_ALLOWED, "remap")
    validate_stage_keys(ds, DOWNSAMPLE_ALLOWED, "downsample")
    validate_stage_keys(df, DIFF_ALLOWED, "diff")
    validate_stage_keys(st, STAR_ALLOWED, "star")
    validate_stage_keys(ph, PHOTOMETRY_ALLOWED, "photometry")
    if pp.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.ps1_process.executor must be 'local' or 'condor'")
    ps1_source = pp.get("ps1_source", "zarr")
    if ps1_source not in ("zarr", "stream"):
        raise ValueError("stages.ps1_process.ps1_source must be 'zarr' or 'stream'")
    if mp.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.mapping.executor must be 'local' or 'condor'")
    if df.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.diff.executor must be 'local' or 'condor'")
    if st.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.star.executor must be 'local' or 'condor'")
    if ph.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.photometry.executor must be 'local' or 'condor'")
    if rm.get("executor", "condor") not in ("local", "condor"):
        raise ValueError("stages.remap.executor must be 'local' or 'condor'")
    if ds.get("executor", "local") not in ("local", "condor"):
        raise ValueError("stages.downsample.executor must be 'local' or 'condor'")
    remap_keying = str(rm.get("keying", "absolute"))
    if remap_keying not in ("absolute", "phase"):
        raise ValueError("stages.remap.keying must be 'absolute' or 'phase'")
    return TemplateStageParams(
        wcs_grouping=_merge_dataclass(WcsGroupingStageParams, wg),
        mapping=_merge_dataclass(MappingStageParams, mp),
        ps1_download=_merge_dataclass(Ps1DownloadStageParams, pd),
        ps1_process=_merge_dataclass(Ps1ProcessStageParams, pp),
        remap=_merge_dataclass(RemapStageParams, rm),
        downsample=_merge_dataclass(DownsampleStageParams, ds),
        diff=_merge_dataclass(DiffStageParams, df),
        star=_merge_dataclass(StarStageParams, st),
        photometry=_merge_dataclass(PhotometryStageParams, ph),
    )
