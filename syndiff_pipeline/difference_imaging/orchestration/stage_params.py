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
        "ps1_min_hit_count",
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
        "phot_cutout_size",
        "phot_bkg_poly_order",
        "phot_snap",
        "psf_size",
        "epsf_oversample",
        "tile_nx",
        "tile_ny",
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
        "phot_box_size",
        "write_debug_fits",
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
        "phot_box_size",
        "kernel_subtract_n_jobs",
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
    ps1_min_hit_count: int = 5000


@dataclass
class HotpantsParams:
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
class PsfPhotometryMethodParams:
    name: str
    psf_type: str = "prf"
    phot_cutout_size: int = 15
    phot_bkg_poly_order: int = 3
    phot_snap: str = "brightest"
    psf_size: int = 11
    epsf_oversample: int = 2
    tile_nx: int = 4
    tile_ny: int = 4
    epsf_workspace: Optional[str] = None
    csv_basename: Optional[str] = None


@dataclass
class AperturePhotometryMethodParams:
    name: str
    tar_ap: int = 3
    sky_in: int = 5
    sky_out: int = 9
    aperture_cutout_size: Optional[int] = None
    csv_basename: Optional[str] = None


PhotometryMethodSpec = Union[PsfPhotometryMethodParams, AperturePhotometryMethodParams]


@dataclass
class ForcedPhotometryParams:
    methods: List[PhotometryMethodSpec] = field(default_factory=list)
    tile_nx: int = 4
    tile_ny: int = 4


_METHOD_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _parse_method_name(raw: object, pipeline_idx: int, method_idx: int) -> str:
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
    weighting_factor: float = 0.5
    phot_box_size: int = 4
    write_debug_fits: bool = True
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
    pass


@dataclass
class KernelSubtractParams:
    phot_box_size: int = 4
    kernel_subtract_n_jobs: Optional[int] = None


def _merge_step_params(cls: Type[T], step_dict: dict) -> T:
    if not step_dict:
        return cls()
    names = {f.name for f in fields(cls)}
    kw = {n: step_dict[n] for n in names if n in step_dict}
    base = cls()
    return cls(**{**base.__dict__, **kw})  # type: ignore[arg-type]


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


def parse_background(stage: dict, pipeline_idx: int):
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
    validate_stage_keys(stage, pipeline_idx, "kernel_fit", KERNEL_FIT_ALLOWED)
    return _merge_dataclass(KernelFitParams, stage)


def parse_convolved_templates(
    stage: dict, pipeline_idx: int
) -> ConvolvedTemplatesParams:
    validate_stage_keys(
        stage, pipeline_idx, "convolved_templates", CONVOLVED_TEMPLATES_ALLOWED
    )
    return _merge_dataclass(ConvolvedTemplatesParams, stage)


def parse_kernel_subtract(stage: dict, pipeline_idx: int) -> KernelSubtractParams:
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


def validate_stage_for_kind(stage: dict, pipeline_idx: int, kind: str) -> None:
    """Strict key allow-list for *kind* (no merge). Used from validate_pipeline."""
    parsers = {
        "shared_mask": lambda: parse_shared_mask(stage, pipeline_idx),
        "hotpants": lambda: parse_hotpants(stage, pipeline_idx),
        "epsf": lambda: parse_epsf(stage, pipeline_idx),
        "sat_template": lambda: parse_sat_template(stage, pipeline_idx),
        "subtract": lambda: parse_subtract(stage, pipeline_idx),
        "background": lambda: parse_background(stage, pipeline_idx),
        "forced_photometry": lambda: parse_forced_photometry(stage, pipeline_idx),
        "kernel_fit": lambda: parse_kernel_fit(stage, pipeline_idx),
        "convolved_templates": lambda: parse_convolved_templates(stage, pipeline_idx),
        "kernel_subtract": lambda: parse_kernel_subtract(stage, pipeline_idx),
    }
    fn = parsers.get(kind)
    if fn is None:
        raise ValueError(f"pipeline[{pipeline_idx}]: unknown kind {kind!r}")
    fn()
