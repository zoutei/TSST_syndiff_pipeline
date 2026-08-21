"""
Per-pipeline-stage parameters parsed from flat YAML keys.

Unknown keys on a stage mapping raise :exc:`ValueError` during validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, FrozenSet, List, Optional, Type, TypeVar, Union


T = TypeVar("T")


def _pick_optional_str(stage: dict, name: str) -> Optional[str]:
    """Pick optional str.
    
    Parameters
    ----------
    stage : dict
    name : str
    
    Returns
    -------
    Optional[str]"""
    if name not in stage:
        return None
    v = stage[name]
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _merge_dataclass(cls: Type[T], stage: dict) -> T:
    """Merge dataclass.
    
    Parameters
    ----------
    stage : dict
    
    Returns
    -------
    T"""
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
    """Validate stage keys.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    kind : str
    allowed : FrozenSet[str]"""
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
        # Legacy aliases for mask_settings.shared.* (explicit stage key only; prefer mask_settings.yaml)
        "gaia_mag_bright",
        "strapsize",
        "ps1_min_hit_count",
        "ref_mag_min",
        "ref_mag_max",
        "ref_isolation_mag",
        "ref_isolation_px",
        "ref_separation_px",
        "mask_settings",
    }
)

ASTROMETRY_ALLOWED = frozenset(
    {
        "kind",
        "sigma_mag_limit",
        "clip_n_sigma",
        "irsa_credentials_file",
        "atlas_credentials_file",
        "atlas_api_config_file",
    }
)

HOTPANTS_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "science",
        "sci_fwhm",
        "hp_sigma_gauss",
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
        "write_kernel_solutions",
        "oversample",
        "use_c_extension",
        "stamp_mode",
        "region_weight",
        "region_max_diameter",
        "region_bisect_on_reject",
        "region_min_npix",
        "region_max_area",
        "region_connectivity",
        "region_rss",
        "region_max_bisects",
        "region_weight_cap",
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
        "min_stars_per_tile",
        "mag_max_rp",
        "mag_min_rp",
        "epsf_maxiters",
        "epsf_recentering_maxiters",
        "extract_size",
        "epsf_n_jobs",
        "epsf_smoothing_kernel",
        "epsf_builder_fit_shape",
        "epsf_recentering_boxsize",
        "epsf_star_box_radius",
        "epsf_use_section_mask",
        "epsf_stamp_border_crop",
    }
)

CENTROIDS_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "mag_max_rp",
        "mag_min_rp",
        "fit_shape",
        "aperture_radius",
        "psf_grouper_min_separation",
        "centroids_n_jobs",
    }
)

PER_FFI_WCS_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "sip_degree",
        "clip_n_sigma",
        "clip_max_iter",
        "min_stars",
        "per_ffi_wcs_n_jobs",
    }
)

TEMPORAL_WCS_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "version",
        "spatial_basis",
        "cheb_degree",
        "temporal_basis",
        "spline_degree",
        "n_interior_knots",
        "edge_densify_knots",
        "edge_fraction",
        "clip_n_sigma",
        "clip_max_iter",
        "min_stars",
        "n_jobs",
        "debug_plots",
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

BACKGROUND_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "recombine_inputs",
        "write_per_frame_fits",
        "write_stack",
        "steps",
    }
)

_METHOD_PSF_KEYS = frozenset(
    {
        "name",
        "type",
        "psf_type",
        "fitter",
        "phot_cutout_size",
        "phot_bkg_poly_order",
        "phot_snap",
        "psf_size",
        "epsf_oversample",
        "tile_nx",
        "tile_ny",
        "fit_shape",
        "aperture_radius",
        "psf_grouper_min_separation",
        "inputs",
        "csv_basename",
    }
)

_METHOD_APERTURE_KEYS = frozenset(
    {
        "name",
        "type",
        "tar_ap",
        "sky_in",
        "sky_out",
        "aperture_cutout_size",
        "subtract_sky",
        "mask_sky_with_shared_mask",
        "csv_basename",
    }
)

FORCED_PHOTOMETRY_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "methods",
        "tile_nx",
        "tile_ny",
    }
)

_KERNEL_HP_KEYS = frozenset(
    {
        "sci_fwhm",
        "hp_sigma_gauss",
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
        "write_kernel_params",
    }
)

KERNEL_FIT_ALLOWED = frozenset(
    {
        "kind",
        "output",
        "weighting_factor",
        "write_debug_fits",
        "tessreduce_smooth_gauss",
        "tessreduce_anomaly_gauss",
        "tessreduce_qe_spline_degree",
        "tessreduce_qe_spline_smooth_mult",
        "tessreduce_boundary_k",
        "tessreduce_boundary_sigma",
        "tessreduce_boundary_rim_width",
    }
    | _KERNEL_HP_KEYS
)

CONVOLVED_TEMPLATES_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
    }
)

KERNEL_SUBTRACT_ALLOWED = frozenset(
    {
        "kind",
        "inputs",
        "output",
        "kernel_subtract_n_jobs",
        "tessreduce_smooth_gauss",
        "tessreduce_anomaly_gauss",
        "tessreduce_qe_spline_degree",
        "tessreduce_qe_spline_smooth_mult",
        "tessreduce_boundary_k",
        "tessreduce_boundary_sigma",
        "tessreduce_boundary_rim_width",
    }
)


@dataclass
class AstrometryParams:
    """AstrometryParams."""
    sigma_mag_limit: float = 0.15
    clip_n_sigma: float = 3.0
    irsa_credentials_file: Optional[str] = None
    atlas_credentials_file: Optional[str] = None
    atlas_api_config_file: Optional[str] = None


@dataclass
class SharedMaskParams:
    """Hotpants ref-star selection + optional mask_settings path.

    Mask geometry/policy (maglims, straps, TNS, asteroids) lives in
    ``mask_settings.yaml``, not here. Legacy stage keys ``gaia_mag_bright``,
    ``strapsize``, and ``ps1_min_hit_count`` are still allowed in YAML and
    applied only when explicitly present (see ``legacy_mask_stage_overrides``).
    """

    ref_mag_min: float = 13.5
    ref_mag_max: float = 14.5
    ref_isolation_mag: float = 13.5
    ref_isolation_px: int = 8
    ref_separation_px: int = 10
    mask_settings: Optional[str] = None


_LEGACY_MASK_STAGE_KEYS = frozenset(
    {"epsf_mag_lim", "gaia_mag_bright", "strapsize", "ps1_min_hit_count"}
)


def legacy_mask_stage_overrides(stage: dict) -> dict[str, Any]:
    """
    Return explicit legacy shared_mask stage overrides for mask_settings.

    Only keys present on the stage dict are returned (dataclass defaults must
    not clobber ``mask_settings.yaml``).
    """
    out: dict[str, Any] = {}
    for key in _LEGACY_MASK_STAGE_KEYS:
        if key in stage and stage[key] is not None:
            out[key] = stage[key]
    return out


@dataclass
class HotpantsParams:
    """HotpantsParams."""
    sci_fwhm: float = 1.88
    hp_sigma_gauss: Optional[list] = None
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
    hp_normalize: str = "t"
    hotpants_n_jobs: Optional[int] = None
    write_convolved: bool = True
    write_bkg: bool = True
    write_stamps: bool = True
    write_kernel_solutions: bool = False
    # Oversampled templates / pure-Python stamp modes (pyhotpants >= 0.2).
    oversample: Optional[int] = None  # None → infer from template vs science shapes
    use_c_extension: Optional[bool] = None  # None → auto (False if OS>1 or connected)
    stamp_mode: str = "grid"  # grid | connected_regions
    region_weight: str = "npix"
    region_max_diameter: float = 40.0
    region_bisect_on_reject: bool = False
    region_min_npix: Optional[int] = None
    region_max_area: int = 0
    region_connectivity: int = 8
    region_rss: Optional[int] = None
    region_max_bisects: int = 100
    region_weight_cap: Optional[tuple] = None


@dataclass
class EpsfParams:
    """EpsfParams."""
    tile_nx: int = 5
    tile_ny: int = 5
    epsf_oversample: int = 4
    psf_size: int = 15
    min_stars_per_tile: int = 5
    mag_max_rp: Optional[float] = 12.95
    mag_min_rp: Optional[float] = None
    epsf_maxiters: int = 15
    epsf_recentering_maxiters: int = 20
    extract_size: Optional[int] = 15
    epsf_n_jobs: Optional[int] = None
    epsf_smoothing_kernel: str = "quadratic"
    epsf_builder_fit_shape: int = 5
    epsf_recentering_boxsize: int = 3
    epsf_star_box_radius: int = 7
    epsf_use_section_mask: bool = True
    epsf_stamp_border_crop: int = 8


@dataclass
class CentroidsParams:
    """CentroidsParams."""
    mag_max_rp: Optional[float] = 12.95
    mag_min_rp: float = 7.5
    fit_shape: int = 11
    aperture_radius: float = 4.0
    psf_grouper_min_separation: float = 10.0
    centroids_n_jobs: Optional[int] = None


@dataclass
class PerFfiWcsParams:
    """Per-FFI Sci2Idl WCS fit parameters."""
    sip_degree: int = 5
    clip_n_sigma: float = 3.0
    clip_max_iter: int = 3
    min_stars: int = 50
    per_ffi_wcs_n_jobs: Optional[int] = None


@dataclass
class TemporalWcsParams:
    """Production temporal Chebyshev WCS fit parameters."""

    version: str = "temporal_cheb5_bspline_v1"
    spatial_basis: str = "chebyshev"
    cheb_degree: int = 5
    temporal_basis: str = "bspline"
    spline_degree: int = 3
    n_interior_knots: int = 20
    edge_densify_knots: bool = True
    edge_fraction: float = 0.12
    clip_n_sigma: float = 3.0
    clip_max_iter: int = 3
    min_stars: int = 50
    n_jobs: Optional[int] = None
    debug_plots: bool = True


@dataclass
class SatTemplateParams:
    """SatTemplateParams."""
    high_res_os: int = 9
    epsf_oversample: int = 2
    psf_size: int = 11
    tile_nx: int = 4
    tile_ny: int = 4



@dataclass
class PsfPhotometryMethodParams:
    """PsfPhotometryMethodParams."""
    name: str
    psf_type: str = "prf"
    fitter: Optional[str] = None  # photutils | tessreduce; None → photutils for epsf
    phot_cutout_size: int = 15
    phot_bkg_poly_order: Optional[int] = 3  # None → no poly surface (flux-only)
    phot_snap: str = "brightest"
    psf_size: int = 11
    epsf_oversample: int = 2
    tile_nx: int = 4
    tile_ny: int = 4
    epsf_workspace: Optional[str] = None
    csv_basename: Optional[str] = None
    fit_shape: int = 11
    aperture_radius: float = 2.0
    # None → grouper=None for single-target forced photometry
    psf_grouper_min_separation: Optional[float] = None


@dataclass
class AperturePhotometryMethodParams:
    """AperturePhotometryMethodParams."""
    name: str
    tar_ap: int = 3
    sky_in: int = 5
    sky_out: int = 9
    aperture_cutout_size: Optional[int] = None
    subtract_sky: bool = True
    mask_sky_with_shared_mask: bool = False
    csv_basename: Optional[str] = None


PhotometryMethodSpec = Union[PsfPhotometryMethodParams, AperturePhotometryMethodParams]


@dataclass
class ForcedPhotometryParams:
    """ForcedPhotometryParams."""
    methods: List[PhotometryMethodSpec] = field(default_factory=list)
    tile_nx: int = 4
    tile_ny: int = 4


_METHOD_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _parse_method_name(raw: object, pipeline_idx: int, method_idx: int) -> str:
    """Parse method name.
    
    Parameters
    ----------
    raw : object
    pipeline_idx : int
    method_idx : int
    
    Returns
    -------
    str"""
    name = str(raw).strip().lower()
    if not name or not _METHOD_NAME_RE.match(name):
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}]: "
            f"'name' must match [a-z0-9_]+, got {raw!r}"
        )
    return name


def _parse_psf_method(
    entry: dict,
    pipeline_idx: int,
    method_idx: int,
    stage_defaults: ForcedPhotometryParams,
) -> PsfPhotometryMethodParams:
    """Parse psf method.
    
    Parameters
    ----------
    entry : dict
    pipeline_idx : int
    method_idx : int
    stage_defaults : ForcedPhotometryParams
    
    Returns
    -------
    PsfPhotometryMethodParams"""
    unknown = set(entry.keys()) - _METHOD_PSF_KEYS
    if unknown:
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}] "
            f"(psf): unknown keys {sorted(unknown)!r}"
        )
    if "psf_type" not in entry:
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}]: "
            "psf_type required for type: psf"
        )
    pt = str(entry["psf_type"]).strip().lower()
    if pt not in ("epsf", "prf"):
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}]: "
            f"psf_type must be 'epsf' or 'prf', got {entry['psf_type']!r}"
        )
    names = {f.name for f in fields(PsfPhotometryMethodParams)} - {"name", "epsf_workspace"}
    kw = {n: entry[n] for n in names if n in entry}
    for n in ("tile_nx", "tile_ny"):
        if n not in kw:
            kw[n] = getattr(stage_defaults, n)
    if "fitter" in kw and kw["fitter"] is not None:
        fitter = str(kw["fitter"]).strip().lower()
        if fitter not in ("photutils", "tessreduce"):
            raise ValueError(
                f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}]: "
                f"fitter must be 'photutils' or 'tessreduce', got {kw['fitter']!r}"
            )
        kw["fitter"] = fitter
    if "phot_bkg_poly_order" in kw and kw["phot_bkg_poly_order"] is not None:
        kw["phot_bkg_poly_order"] = int(kw["phot_bkg_poly_order"])
    if pt == "prf" and kw.get("fitter") is not None:
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}]: "
            "fitter is only valid with psf_type: epsf"
        )
    if pt == "prf" and (entry.get("inputs") or {}).get("epsf"):
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}]: "
            "inputs.epsf is forbidden for psf_type: prf"
        )
    p = PsfPhotometryMethodParams(name=_parse_method_name(entry["name"], pipeline_idx, method_idx), **kw)
    p.psf_type = pt
    inp = entry.get("inputs") or {}
    if isinstance(inp, dict) and inp.get("epsf"):
        p.epsf_workspace = str(inp["epsf"]).strip()
    if entry.get("csv_basename") is not None:
        p.csv_basename = str(entry["csv_basename"]).strip()
    return p


def _parse_aperture_method(
    entry: dict,
    pipeline_idx: int,
    method_idx: int,
) -> AperturePhotometryMethodParams:
    """Parse aperture method.
    
    Parameters
    ----------
    entry : dict
    pipeline_idx : int
    method_idx : int
    
    Returns
    -------
    AperturePhotometryMethodParams"""
    unknown = set(entry.keys()) - _METHOD_APERTURE_KEYS
    if unknown:
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry methods[{method_idx}] "
            f"(aperture): unknown keys {sorted(unknown)!r}"
        )
    names = {f.name for f in fields(AperturePhotometryMethodParams)} - {"name"}
    kw = {n: entry[n] for n in names if n in entry}
    p = AperturePhotometryMethodParams(
        name=_parse_method_name(entry["name"], pipeline_idx, method_idx),
        **kw,
    )
    if entry.get("csv_basename") is not None:
        p.csv_basename = str(entry["csv_basename"]).strip()
    return p


@dataclass
class KernelFitParams:
    """KernelFitParams."""
    weighting_factor: float = 0.5
    write_debug_fits: bool = True
    tessreduce_smooth_gauss: float = 2.0
    tessreduce_anomaly_gauss: float = 2.0
    tessreduce_qe_spline_degree: int = 2
    tessreduce_qe_spline_smooth_mult: float = 10.0
    tessreduce_boundary_k: int = 15
    tessreduce_boundary_sigma: float = 3.0
    tessreduce_boundary_rim_width: int = 1
    sci_fwhm: float = 1.88
    hp_sigma_gauss: Optional[list] = None
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
    hp_normalize: str = "t"
    write_kernel_params: bool = True


@dataclass
class ConvolvedTemplatesParams:
    """ConvolvedTemplatesParams."""
    pass


@dataclass
class KernelSubtractParams:
    """KernelSubtractParams."""
    kernel_subtract_n_jobs: Optional[int] = None
    tessreduce_smooth_gauss: float = 2.0
    tessreduce_anomaly_gauss: float = 2.0
    tessreduce_qe_spline_degree: int = 2
    tessreduce_qe_spline_smooth_mult: float = 10.0
    tessreduce_boundary_k: int = 15
    tessreduce_boundary_sigma: float = 3.0
    tessreduce_boundary_rim_width: int = 1


def _merge_step_params(cls: Type[T], step_dict: dict) -> T:
    """Merge step params.
    
    Parameters
    ----------
    step_dict : dict
    
    Returns
    -------
    T"""
    if not step_dict:
        return cls()
    names = {f.name for f in fields(cls)}
    kw = {n: step_dict[n] for n in names if n in step_dict}
    base = cls()
    return cls(**{**base.__dict__, **kw})  # type: ignore[arg-type]


def parse_astrometry(stage: dict, pipeline_idx: int) -> AstrometryParams:
    """Parse astrometry stage YAML."""
    validate_stage_keys(stage, pipeline_idx, "astrometry", ASTROMETRY_ALLOWED)
    return _merge_dataclass(AstrometryParams, stage)


def parse_shared_mask(stage: dict, pipeline_idx: int) -> SharedMaskParams:
    """Parse shared mask.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    SharedMaskParams"""
    validate_stage_keys(stage, pipeline_idx, "shared_mask", SHARED_MASK_ALLOWED)
    return _merge_dataclass(SharedMaskParams, stage)


def parse_hotpants(stage: dict, pipeline_idx: int) -> HotpantsParams:
    """Parse hotpants.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    HotpantsParams"""
    validate_stage_keys(stage, pipeline_idx, "hotpants", HOTPANTS_ALLOWED)
    hp = _merge_dataclass(HotpantsParams, stage)
    if "hotpants_n_jobs" in stage:
        v = stage["hotpants_n_jobs"]
        hp.hotpants_n_jobs = None if v is None else int(v)
    if "oversample" in stage:
        v = stage["oversample"]
        hp.oversample = None if v is None else int(v)
    if "use_c_extension" in stage:
        v = stage["use_c_extension"]
        hp.use_c_extension = None if v is None else bool(v)
    stamp_mode = str(getattr(hp, "stamp_mode", "grid") or "grid").strip()
    if stamp_mode not in ("grid", "connected_regions"):
        raise ValueError(
            f"pipeline[{pipeline_idx}] hotpants.stamp_mode must be "
            f"'grid' or 'connected_regions', got {stamp_mode!r}"
        )
    hp.stamp_mode = stamp_mode
    if "region_weight_cap" in stage and stage["region_weight_cap"] is not None:
        cap = stage["region_weight_cap"]
        if not (isinstance(cap, (list, tuple)) and len(cap) == 2):
            raise ValueError(
                f"pipeline[{pipeline_idx}] hotpants.region_weight_cap must be "
                f"[lo, hi], got {cap!r}"
            )
        hp.region_weight_cap = (float(cap[0]), float(cap[1]))
    return hp


def parse_epsf(stage: dict, pipeline_idx: int) -> EpsfParams:
    """Parse epsf.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    EpsfParams"""
    validate_stage_keys(stage, pipeline_idx, "epsf", EPSF_ALLOWED)
    params = _merge_dataclass(EpsfParams, stage)
    if params.mag_max_rp is None:
        params = EpsfParams(**{**params.__dict__, "mag_max_rp": 12.95})
    return params


def parse_centroids(stage: dict, pipeline_idx: int) -> CentroidsParams:
    """Parse centroids stage parameters."""
    validate_stage_keys(stage, pipeline_idx, "centroids", CENTROIDS_ALLOWED)
    params = _merge_dataclass(CentroidsParams, stage)
    if params.mag_max_rp is None:
        params = CentroidsParams(**{**params.__dict__, "mag_max_rp": 12.95})
    if "centroids_n_jobs" in stage:
        v = stage["centroids_n_jobs"]
        params.centroids_n_jobs = None if v is None else int(v)
    return params


def parse_per_ffi_wcs(stage: dict, pipeline_idx: int) -> PerFfiWcsParams:
    """Parse per_ffi_wcs stage parameters."""
    validate_stage_keys(stage, pipeline_idx, "per_ffi_wcs", PER_FFI_WCS_ALLOWED)
    params = _merge_dataclass(PerFfiWcsParams, stage)
    if "per_ffi_wcs_n_jobs" in stage:
        v = stage["per_ffi_wcs_n_jobs"]
        params.per_ffi_wcs_n_jobs = None if v is None else int(v)
    return params


def parse_temporal_wcs(stage: dict, pipeline_idx: int) -> TemporalWcsParams:
    """Parse the production ``temporal_wcs`` stage parameters."""
    validate_stage_keys(stage, pipeline_idx, "temporal_wcs", TEMPORAL_WCS_ALLOWED)
    params = _merge_dataclass(TemporalWcsParams, stage)
    if params.version != "temporal_cheb5_bspline_v1":
        raise ValueError(
            f"pipeline[{pipeline_idx}] temporal_wcs: version must be "
            f"'temporal_cheb5_bspline_v1', got {params.version!r}"
        )
    if str(params.spatial_basis).lower() != "chebyshev":
        raise ValueError(
            f"pipeline[{pipeline_idx}] temporal_wcs: spatial_basis must be 'chebyshev'"
        )
    if int(params.cheb_degree) != 5:
        raise ValueError(f"pipeline[{pipeline_idx}] temporal_wcs: cheb_degree must be 5")
    if str(params.temporal_basis).lower() != "bspline":
        raise ValueError(
            f"pipeline[{pipeline_idx}] temporal_wcs: temporal_basis must be 'bspline'"
        )
    if int(params.spline_degree) != 3:
        raise ValueError(f"pipeline[{pipeline_idx}] temporal_wcs: spline_degree must be 3")
    if int(params.n_interior_knots) < 0:
        raise ValueError("temporal_wcs: n_interior_knots must be non-negative")
    if not 0.0 <= float(params.edge_fraction) < 0.5:
        raise ValueError("temporal_wcs: edge_fraction must be in [0, 0.5)")
    if int(params.min_stars) < 1:
        raise ValueError("temporal_wcs: min_stars must be positive")
    if "n_jobs" in stage:
        params.n_jobs = None if stage["n_jobs"] is None else int(stage["n_jobs"])
    return params


def parse_sat_template(stage: dict, pipeline_idx: int) -> SatTemplateParams:
    """Parse sat template.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    SatTemplateParams"""
    validate_stage_keys(stage, pipeline_idx, "sat_template", SAT_TEMPLATE_ALLOWED)
    return _merge_dataclass(SatTemplateParams, stage)


def parse_subtract(stage: dict, pipeline_idx: int) -> None:
    """Parse subtract.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int"""
    validate_stage_keys(stage, pipeline_idx, "subtract", SUBTRACT_ALLOWED)


def parse_background(stage: dict, pipeline_idx: int):
    """Parse background.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int"""
    from syndiff_pipeline.difference_imaging.stages.background.pipeline import (
        BackgroundParams,
        BackgroundStepSpatialParams,
        BackgroundStepStrapParams,
        BackgroundStepTemporalParams,
    )

    validate_stage_keys(stage, pipeline_idx, "background", BACKGROUND_ALLOWED)
    steps = stage.get("steps") or {}
    if not isinstance(steps, dict):
        raise ValueError(
            f"pipeline[{pipeline_idx}] background: 'steps' must be a mapping"
        )

    spatial = _merge_step_params(
        BackgroundStepSpatialParams, steps.get("spatial") or {}
    )
    temporal = _merge_step_params(
        BackgroundStepTemporalParams, steps.get("temporal") or {}
    )
    strap = _merge_step_params(
        BackgroundStepStrapParams, steps.get("strap") or {}
    )
    if temporal.vector_path is None and "vector_path" in (steps.get("temporal") or {}):
        temporal.vector_path = _pick_optional_str(steps.get("temporal") or {}, "vector_path")

    if not (spatial.enabled or temporal.enabled or strap.enabled):
        raise ValueError(
            f"pipeline[{pipeline_idx}] background: at least one step must be enabled"
        )

    label_out = str(stage.get("output", "")).strip()
    for step_name, step in (
        ("spatial", spatial),
        ("temporal", temporal),
        ("strap", strap),
    ):
        save = getattr(step, "save", None)
        if save and str(save).strip() == label_out:
            raise ValueError(
                f"pipeline[{pipeline_idx}] background: steps.{step_name}.save "
                f"must differ from output {label_out!r}"
            )

    recombine = stage.get("recombine_inputs")
    if recombine is None:
        inp = stage.get("inputs") or {}
        recombine = bool(inp.get("bkg"))

    return BackgroundParams(
        recombine_inputs=bool(recombine),
        write_per_frame_fits=bool(stage.get("write_per_frame_fits", True)),
        write_stack=bool(stage.get("write_stack", True)),
        spatial=spatial,
        temporal=temporal,
        strap=strap,
    )


def parse_forced_photometry(stage: dict, pipeline_idx: int) -> ForcedPhotometryParams:
    """Parse forced photometry.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    ForcedPhotometryParams"""
    validate_stage_keys(
        stage, pipeline_idx, "forced_photometry", FORCED_PHOTOMETRY_ALLOWED
    )
    if "psf_type" in stage:
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry: top-level 'psf_type' is no "
            "longer supported; use a 'methods' list with type: psf entries "
            "(see config/README.md)."
        )
    raw_methods = stage.get("methods")
    if not raw_methods or not isinstance(raw_methods, list):
        raise ValueError(
            f"pipeline[{pipeline_idx}] forced_photometry: required non-empty "
            "'methods' list"
        )
    stage_defaults = _merge_dataclass(ForcedPhotometryParams, stage)
    parsed: List[PhotometryMethodSpec] = []
    seen_names: set[str] = set()
    for mi, entry in enumerate(raw_methods):
        if not isinstance(entry, dict):
            raise ValueError(
                f"pipeline[{pipeline_idx}] forced_photometry methods[{mi}]: "
                "must be a mapping"
            )
        if "name" not in entry:
            raise ValueError(
                f"pipeline[{pipeline_idx}] forced_photometry methods[{mi}]: "
                "'name' is required"
            )
        if "type" not in entry:
            raise ValueError(
                f"pipeline[{pipeline_idx}] forced_photometry methods[{mi}]: "
                "'type' is required ('psf' or 'aperture')"
            )
        mtype = str(entry["type"]).strip().lower()
        if mtype == "psf":
            spec = _parse_psf_method(entry, pipeline_idx, mi, stage_defaults)
        elif mtype == "aperture":
            spec = _parse_aperture_method(entry, pipeline_idx, mi)
        else:
            raise ValueError(
                f"pipeline[{pipeline_idx}] forced_photometry methods[{mi}]: "
                f"type must be 'psf' or 'aperture', got {entry['type']!r}"
            )
        if spec.name in seen_names:
            raise ValueError(
                f"pipeline[{pipeline_idx}] forced_photometry: duplicate method "
                f"name {spec.name!r}"
            )
        seen_names.add(spec.name)
        parsed.append(spec)
    stage_defaults.methods = parsed
    return stage_defaults


def kernel_fit_params_to_hotpants(kf: KernelFitParams) -> HotpantsParams:
    """Build :class:`HotpantsParams` from kernel-fit stage settings."""
    return HotpantsParams(
        sci_fwhm=kf.sci_fwhm,
        hp_sigma_gauss=kf.hp_sigma_gauss,
        hp_ko=kf.hp_ko,
        hp_bgo=kf.hp_bgo,
        hp_nstampx=kf.hp_nstampx,
        hp_nstampy=kf.hp_nstampy,
        hp_nss=kf.hp_nss,
        hp_ngauss=kf.hp_ngauss,
        hp_deg_fixe=kf.hp_deg_fixe,
        hp_fitthresh=kf.hp_fitthresh,
        hp_stat_sig=kf.hp_stat_sig,
        hp_kf_spread_mask1=kf.hp_kf_spread_mask1,
        hp_ks=kf.hp_ks,
        hp_kfm=kf.hp_kfm,
        hp_force_convolve=kf.hp_force_convolve,
        hp_normalize=kf.hp_normalize,
        write_convolved=False,
        write_bkg=False,
        write_stamps=False,
    )


def parse_kernel_fit(stage: dict, pipeline_idx: int) -> KernelFitParams:
    """Parse kernel fit.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    KernelFitParams"""
    validate_stage_keys(stage, pipeline_idx, "kernel_fit", KERNEL_FIT_ALLOWED)
    return _merge_dataclass(KernelFitParams, stage)


def parse_convolved_templates(
    stage: dict, pipeline_idx: int
) -> ConvolvedTemplatesParams:
    """Parse convolved templates.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    ConvolvedTemplatesParams"""
    validate_stage_keys(
        stage, pipeline_idx, "convolved_templates", CONVOLVED_TEMPLATES_ALLOWED
    )
    return _merge_dataclass(ConvolvedTemplatesParams, stage)


def parse_kernel_subtract(stage: dict, pipeline_idx: int) -> KernelSubtractParams:
    """Parse kernel subtract.
    
    Parameters
    ----------
    stage : dict
    pipeline_idx : int
    
    Returns
    -------
    KernelSubtractParams"""
    validate_stage_keys(
        stage, pipeline_idx, "kernel_subtract", KERNEL_SUBTRACT_ALLOWED
    )
    ks = _merge_dataclass(KernelSubtractParams, stage)
    if "kernel_subtract_n_jobs" in stage:
        v = stage["kernel_subtract_n_jobs"]
        ks.kernel_subtract_n_jobs = None if v is None else int(v)
    return ks


def upcoming_phot_cutout_size(pipeline: list, pipeline_idx: int) -> int:
    """Max PSF ``phot_cutout_size`` from the next ``forced_photometry`` stage."""
    sizes: list[int] = []
    found = False
    for idx, stage in enumerate(pipeline):
        if not isinstance(stage, dict) or "kind" not in stage:
            continue
        if idx <= pipeline_idx:
            continue
        if stage.get("kind") != "forced_photometry":
            continue
        found = True
        fp = parse_forced_photometry(stage, idx)
        for m in fp.methods:
            if hasattr(m, "phot_cutout_size"):
                sizes.append(int(m.phot_cutout_size))
        break
    if not found or not sizes:
        return 15
    return max(sizes)


PHOTOMETRY_DELEGATOR_ALLOWED = frozenset({"kind", "config"})


def validate_photometry_delegator(stage: dict, pipeline_idx: int) -> None:
    validate_stage_keys(stage, pipeline_idx, "photometry", PHOTOMETRY_DELEGATOR_ALLOWED)
    config_ref = stage.get("config")
    if not config_ref or not str(config_ref).strip():
        raise ValueError(f"pipeline[{pipeline_idx}]: photometry stage requires config:")


def validate_stage_for_kind(stage: dict, pipeline_idx: int, kind: str) -> None:
    """Strict key allow-list for *kind* (no merge). Used from validate_pipeline."""
    parsers = {
        "astrometry": lambda: parse_astrometry(stage, pipeline_idx),
        "shared_mask": lambda: parse_shared_mask(stage, pipeline_idx),
        "hotpants": lambda: parse_hotpants(stage, pipeline_idx),
        "epsf": lambda: parse_epsf(stage, pipeline_idx),
        "centroids": lambda: parse_centroids(stage, pipeline_idx),
        "per_ffi_wcs": lambda: parse_per_ffi_wcs(stage, pipeline_idx),
        "temporal_wcs": lambda: parse_temporal_wcs(stage, pipeline_idx),
        "sat_template": lambda: parse_sat_template(stage, pipeline_idx),
        "subtract": lambda: parse_subtract(stage, pipeline_idx),
        "background": lambda: parse_background(stage, pipeline_idx),
        "forced_photometry": lambda: parse_forced_photometry(stage, pipeline_idx),
        "kernel_fit": lambda: parse_kernel_fit(stage, pipeline_idx),
        "convolved_templates": lambda: parse_convolved_templates(stage, pipeline_idx),
        "kernel_subtract": lambda: parse_kernel_subtract(stage, pipeline_idx),
        "photometry": lambda: validate_photometry_delegator(stage, pipeline_idx),
    }
    fn = parsers.get(kind)
    if fn is None:
        raise ValueError(f"pipeline[{pipeline_idx}]: unknown kind {kind!r}")
    fn()
