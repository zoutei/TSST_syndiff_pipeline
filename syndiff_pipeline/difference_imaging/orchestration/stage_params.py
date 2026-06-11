"""
Per-pipeline-stage parameters parsed from flat YAML keys.

Unknown keys on a stage mapping raise :exc:`ValueError` during validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, FrozenSet, Optional, Type, TypeVar


T = TypeVar("T")


def _pick_optional_str(stage: dict, name: str) -> Optional[str]:
    if name not in stage:
        return None
    v = stage[name]
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _merge_dataclass(cls: Type[T], stage: dict) -> T:
    names = {f.name for f in fields(cls)}
    kw = {n: stage[n] for n in names if n in stage}
    base = cls()
    return cls(**{**base.__dict__, **kw})  # type: ignore[arg-type]


def validate_stage_keys(
    stage: dict,
    pipeline_idx: int,
    kind: str,
    allowed: FrozenSet[str],
) -> None:
    unknown = set(stage.keys()) - allowed
    if unknown:
        raise ValueError(
            f"pipeline[{pipeline_idx}] ({kind}): unknown keys {sorted(unknown)!r}; "
            f"allowed: {sorted(allowed)!r}"
        )


# ── Structural + param key sets ───────────────────────────────────────────────

SHARED_MASK_ALLOWED = frozenset(
    {
        "kind",
        "gaia_mag_bright",
        "strapsize",
        "ref_mag_min",
        "ref_mag_max",
        "ref_isolation_mag",
        "ref_isolation_px",
        "ref_separation_px",
    }
)

HOTPANTS_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "science",
        "sci_fwhm",
        "hp_ko",
        "hp_bgo",
        "hp_nstampx",
        "hp_nstampy",
        "hp_nss",
        "hp_ngauss",
        "hp_deg_fixe",
        "hp_fitthresh",
        "hp_stat_sig",
        "hp_kf_spread_mask1",
        "hp_ks",
        "hp_kfm",
        "hp_force_convolve",
        "hp_normalize",
        "hotpants_n_jobs",
        "write_convolved",
        "write_bkg",
        "write_stamps",
        "write_kernel_params",
    }
)

EPSF_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "tile_nx",
        "tile_ny",
        "epsf_oversample",
        "psf_size",
    }
)

SAT_TEMPLATE_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "high_res_os",
        "epsf_oversample",
        "psf_size",
        "tile_nx",
        "tile_ny",
    }
)

SUBTRACT_ALLOWED = frozenset({"kind", "inputs", "output"})

BACKGROUND_ROUGH_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "round_id",
        "stream_load_rough",
        "write_per_frame_fits",
        "bkg_tessreduce_spatial_pipeline",
        "bkg_r1_recombine_hotpants",
        "bkg_gauss_smooth",
        "bkg_calc_qe",
        "bkg_strap_iso",
        "bkg_source_hunt",
        "bkg_interpolate",
        "bkg_rerun_negative",
        "bkg_rerun_diff",
        "bkg_use_error_image",
        "bkg_vector_path",
    }
)

BACKGROUND_ADAPTIVE_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "round_id",
        "write_per_frame_fits",
        "bkg_adaptive_method",
        "bkg_adaptive_savgol_window",
        "bkg_adaptive_savgol_polyorder",
        "bkg_adaptive_w_min",
        "bkg_adaptive_w_max",
        "bkg_adaptive_block_size",
        "bkg_vector_path",
    }
)

BACKGROUND_ESTIMATE_ALLOWED = frozenset(
    BACKGROUND_ROUGH_ALLOWED | BACKGROUND_ADAPTIVE_ALLOWED | {"mode"}
)

FORCED_PHOTOMETRY_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "psf_type",
        "phot_cutout_size",
        "phot_debug_stamp_size",
        "phot_bkg_poly_order",
        "phot_snap",
        "psf_size",
        "epsf_oversample",
        "tile_nx",
        "tile_ny",
    }
)


@dataclass
class SharedMaskParams:
    gaia_mag_bright: float = 13.0
    strapsize: int = 6
    ref_mag_min: float = 13.5
    ref_mag_max: float = 14.5
    ref_isolation_mag: float = 13.5
    ref_isolation_px: int = 8
    ref_separation_px: int = 10


@dataclass
class HotpantsParams:
    sci_fwhm: float = 1.0
    hp_ko: int = 2
    hp_bgo: int = 3
    hp_nstampx: int = 10
    hp_nstampy: int = 10
    hp_nss: int = 100
    hp_ngauss: int = 3
    hp_deg_fixe: list = field(default_factory=lambda: [6, 4, 2])
    hp_fitthresh: float = 5.0
    hp_stat_sig: float = 3.0
    hp_kf_spread_mask1: float = 0.0
    hp_ks: float = 3.0
    hp_kfm: float = 0.75
    hp_force_convolve: str = "t"
    hp_normalize: str = "i"
    hotpants_n_jobs: Optional[int] = None
    write_convolved: bool = True
    write_bkg: bool = True
    write_stamps: bool = True
    write_kernel_params: bool = True


@dataclass
class EpsfParams:
    tile_nx: int = 4
    tile_ny: int = 4
    epsf_oversample: int = 2
    psf_size: int = 11


@dataclass
class SatTemplateParams:
    high_res_os: int = 9
    epsf_oversample: int = 2
    psf_size: int = 11
    tile_nx: int = 4
    tile_ny: int = 4


@dataclass
class BackgroundSpatialParams:
    """Rough / spatial TESSreduce background (``background_rough`` / estimate rough leg)."""

    bkg_tessreduce_spatial_pipeline: bool = True
    bkg_r1_recombine_hotpants: bool = False
    bkg_gauss_smooth: float = 2.0
    bkg_calc_qe: bool = True
    bkg_strap_iso: bool = True
    bkg_source_hunt: bool = True
    bkg_interpolate: bool = True
    bkg_rerun_negative: bool = False
    bkg_rerun_diff: bool = False
    bkg_use_error_image: bool = False
    bkg_vector_path: Optional[str] = None


@dataclass
class BackgroundAdaptiveParams:
    bkg_adaptive_method: str = "savgol"
    bkg_adaptive_savgol_window: Optional[int] = None
    bkg_adaptive_savgol_polyorder: int = 2
    bkg_adaptive_w_min: int = 3
    bkg_adaptive_w_max: int = 51
    bkg_adaptive_block_size: int = 5
    bkg_vector_path: Optional[str] = None


@dataclass
class ForcedPhotometryParams:
    psf_type: str = "epsf"
    phot_cutout_size: int = 15
    phot_debug_stamp_size: int = 25
    phot_bkg_poly_order: int = 3
    phot_snap: str = "brightest"
    psf_size: int = 11
    epsf_oversample: int = 2
    tile_nx: int = 4
    tile_ny: int = 4


def parse_shared_mask(stage: dict, pipeline_idx: int) -> SharedMaskParams:
    validate_stage_keys(stage, pipeline_idx, "shared_mask", SHARED_MASK_ALLOWED)
    return _merge_dataclass(SharedMaskParams, stage)


def parse_hotpants(stage: dict, pipeline_idx: int) -> HotpantsParams:
    validate_stage_keys(stage, pipeline_idx, "hotpants", HOTPANTS_ALLOWED)
    hp = _merge_dataclass(HotpantsParams, stage)
    if "hotpants_n_jobs" in stage:
        v = stage["hotpants_n_jobs"]
        hp.hotpants_n_jobs = None if v is None else int(v)
    return hp


def parse_epsf(stage: dict, pipeline_idx: int) -> EpsfParams:
    validate_stage_keys(stage, pipeline_idx, "epsf", EPSF_ALLOWED)
    return _merge_dataclass(EpsfParams, stage)


def parse_sat_template(stage: dict, pipeline_idx: int) -> SatTemplateParams:
    validate_stage_keys(stage, pipeline_idx, "sat_template", SAT_TEMPLATE_ALLOWED)
    return _merge_dataclass(SatTemplateParams, stage)


def parse_subtract(stage: dict, pipeline_idx: int) -> None:
    validate_stage_keys(stage, pipeline_idx, "subtract", SUBTRACT_ALLOWED)


def parse_background_rough(stage: dict, pipeline_idx: int) -> BackgroundSpatialParams:
    validate_stage_keys(stage, pipeline_idx, "background_rough", BACKGROUND_ROUGH_ALLOWED)
    p = _merge_dataclass(BackgroundSpatialParams, stage)
    p.bkg_vector_path = _pick_optional_str(stage, "bkg_vector_path")
    return p


def parse_background_adaptive(stage: dict, pipeline_idx: int) -> BackgroundAdaptiveParams:
    validate_stage_keys(
        stage, pipeline_idx, "background_adaptive", BACKGROUND_ADAPTIVE_ALLOWED
    )
    p = _merge_dataclass(BackgroundAdaptiveParams, stage)
    p.bkg_vector_path = _pick_optional_str(stage, "bkg_vector_path")
    return p


def parse_background_estimate(
    stage: dict, pipeline_idx: int
) -> tuple[BackgroundSpatialParams, BackgroundAdaptiveParams]:
    validate_stage_keys(
        stage, pipeline_idx, "background_estimate", BACKGROUND_ESTIMATE_ALLOWED
    )
    sp_names = {f.name for f in fields(BackgroundSpatialParams)}
    ad_names = {f.name for f in fields(BackgroundAdaptiveParams)}
    spatial = _merge_dataclass(
        BackgroundSpatialParams, {k: stage[k] for k in stage if k in sp_names}
    )
    adaptive = _merge_dataclass(
        BackgroundAdaptiveParams, {k: stage[k] for k in stage if k in ad_names}
    )
    spatial.bkg_vector_path = _pick_optional_str(stage, "bkg_vector_path")
    adaptive.bkg_vector_path = _pick_optional_str(stage, "bkg_vector_path")
    return spatial, adaptive


def parse_forced_photometry(stage: dict, pipeline_idx: int) -> ForcedPhotometryParams:
    validate_stage_keys(
        stage, pipeline_idx, "forced_photometry", FORCED_PHOTOMETRY_ALLOWED
    )
    if "psf_type" not in stage:
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry: required key 'psf_type' "
            "(\"epsf\" or \"prf\")"
        )
    pt = str(stage["psf_type"]).strip().lower()
    if pt not in ("epsf", "prf"):
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry: psf_type must be "
            f"'epsf' or 'prf', got {stage['psf_type']!r}"
        )
    p = _merge_dataclass(ForcedPhotometryParams, stage)
    p.psf_type = pt
    return p


def validate_stage_for_kind(stage: dict, pipeline_idx: int, kind: str) -> None:
    """Strict key allow-list for *kind* (no merge). Used from validate_pipeline."""
    parsers = {
        "shared_mask": lambda: parse_shared_mask(stage, pipeline_idx),
        "hotpants": lambda: parse_hotpants(stage, pipeline_idx),
        "epsf": lambda: parse_epsf(stage, pipeline_idx),
        "sat_template": lambda: parse_sat_template(stage, pipeline_idx),
        "subtract": lambda: parse_subtract(stage, pipeline_idx),
        "background_rough": lambda: parse_background_rough(stage, pipeline_idx),
        "background_adaptive": lambda: parse_background_adaptive(stage, pipeline_idx),
        "background_estimate": lambda: parse_background_estimate(stage, pipeline_idx),
        "forced_photometry": lambda: parse_forced_photometry(stage, pipeline_idx),
    }
    fn = parsers.get(kind)
    if fn is None:
        raise ValueError(f"pipeline[{pipeline_idx}]: unknown kind {kind!r}")
    fn()
