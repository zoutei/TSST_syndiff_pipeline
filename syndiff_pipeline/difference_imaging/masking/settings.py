"""Mask settings loader and resolution."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

# WIS TNS public objects zip (override in YAML only if WIS moves the file).
DEFAULT_TNS_PUBLIC_ZIP_URL = (
    "https://www.wis-tns.org/system/files/tns_public_objects/"
    "tns_public_objects.csv.zip"
)

# MIT TESS orbit times (override only if MIT moves the file).
DEFAULT_TESS_ORBIT_TIMES_URL = (
    "https://tess.mit.edu/public/files/TESS_orbit_times.csv"
)


@dataclass
class SharedMaskSettings:
    style: str = "empirical"  # or tessreduce
    epsf_mag_lim: float = 7.5
    bright_maglim: float = 13.0
    faint_maglim: float = 18.0
    scale: float = 1.0
    strapsize: int = 6
    include_straps: bool = True
    include_edges: bool = True
    ps1_min_hit_count: int = 5000


@dataclass
class TnsMaskSettings:
    enabled: bool = True
    public_csv: Optional[str] = None
    download_url: str = DEFAULT_TNS_PUBLIC_ZIP_URL
    include_in_static_fits: bool = True
    paint_union_fits: bool = False


@dataclass
class AsteroidMaskSettings:
    enabled: bool = True
    vmag_lim: float = 20.0
    intervals_dir: Optional[str] = None
    # orbit_times_path null → {data_root}/catalogs/TESS_orbit_times.csv
    orbit_times_path: Optional[str] = None
    # orbit_times_url omitted → DEFAULT_TESS_ORBIT_TIMES_URL
    orbit_times_url: str = DEFAULT_TESS_ORBIT_TIMES_URL
    # When candidates.parquet missing, run sbident discover (needs optional deps).
    run_discover: bool = True


@dataclass
class MaskSettings:
    geometry_file: Optional[str] = None
    shared: SharedMaskSettings = field(default_factory=SharedMaskSettings)
    tns: TnsMaskSettings = field(default_factory=TnsMaskSettings)
    asteroids: AsteroidMaskSettings = field(default_factory=AsteroidMaskSettings)


def _merge_dataclass(cls, data: dict | None, *, defaults: Any = None):
    src = defaults if defaults is not None else cls()
    base = {f.name: getattr(src, f.name) for f in fields(cls)}
    if not data:
        return cls(**base)
    known = {f.name for f in fields(cls)}
    for k, v in data.items():
        if k in known and v is not None:
            base[k] = v
    return cls(**base)


def mask_settings_from_dict(raw: dict | None) -> MaskSettings:
    """Build MaskSettings from a YAML dict (missing keys → defaults)."""
    raw = raw or {}
    shared = _merge_dataclass(SharedMaskSettings, raw.get("shared"))
    tns_raw = dict(raw.get("tns") or {})
    # download_url omitted → code default (do not treat missing as None override)
    if "download_url" not in tns_raw or tns_raw.get("download_url") in (None, ""):
        tns_raw.pop("download_url", None)
    tns = _merge_dataclass(TnsMaskSettings, tns_raw)
    ast_raw = dict(raw.get("asteroids") or {})
    if "orbit_times_url" not in ast_raw or ast_raw.get("orbit_times_url") in (None, ""):
        ast_raw.pop("orbit_times_url", None)
    asteroids = _merge_dataclass(AsteroidMaskSettings, ast_raw)
    geo = raw.get("geometry_file")
    style = str(shared.style).strip().lower()
    if style not in ("empirical", "tessreduce"):
        raise ValueError(f"shared.style must be 'empirical' or 'tessreduce', got {shared.style!r}")
    shared.style = style
    if float(shared.epsf_mag_lim) >= float(shared.bright_maglim):
        raise ValueError(
            f"epsf_mag_lim ({shared.epsf_mag_lim}) must be < bright_maglim ({shared.bright_maglim})"
        )
    if float(shared.faint_maglim) < float(shared.bright_maglim):
        raise ValueError(
            f"faint_maglim ({shared.faint_maglim}) must be >= bright_maglim ({shared.bright_maglim})"
        )
    return MaskSettings(
        geometry_file=str(geo) if geo else None,
        shared=shared,
        tns=tns,
        asteroids=asteroids,
    )


def load_mask_settings(path: str | Path | None = None) -> MaskSettings:
    """Load settings YAML, or return packaged defaults when path is None/missing."""
    if path is None:
        return MaskSettings()
    p = Path(path).expanduser()
    if not p.is_file():
        log.warning("mask_settings not found at %s; using defaults", p)
        return MaskSettings()
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"mask_settings must be a mapping: {p}")
    return mask_settings_from_dict(raw)


def resolve_mask_settings(
    *,
    stage_mask_settings: str | None = None,
    site_dir: str | Path | None = None,
    ws_root: str | Path | None = None,
) -> tuple[MaskSettings, Path | None]:
    """
    Resolve order: stage path → ``{ws_root}/mask_settings.yaml`` →
    ``{site}/mask_settings.yaml`` → packaged defaults.

    Returns (settings, path_used_or_None).
    """
    candidates: list[Path] = []
    if stage_mask_settings and str(stage_mask_settings).strip():
        candidates.append(Path(stage_mask_settings).expanduser())
    if ws_root is not None:
        candidates.append(Path(ws_root) / "mask_settings.yaml")
    if site_dir is not None:
        candidates.append(Path(site_dir) / "mask_settings.yaml")

    for cand in candidates:
        if cand.is_file():
            return load_mask_settings(cand), cand.resolve()
    return MaskSettings(), None


def apply_stage_overrides(
    settings: MaskSettings,
    *,
    epsf_mag_lim: float | None = None,
    gaia_mag_bright: float | None = None,
    strapsize: int | None = None,
    ps1_min_hit_count: int | None = None,
) -> MaskSettings:
    """
    Apply explicit shared_mask stage overrides onto resolved mask settings.

    Prefer ``mask_settings.yaml`` for these knobs. Pass ``None`` to leave a
    field unchanged (do not pass SharedMaskParams dataclass defaults).
    """
    shared = SharedMaskSettings(**asdict(settings.shared))
    if epsf_mag_lim is not None:
        shared.epsf_mag_lim = float(epsf_mag_lim)
    if gaia_mag_bright is not None:
        shared.bright_maglim = float(gaia_mag_bright)
    if strapsize is not None:
        shared.strapsize = int(strapsize)
    if ps1_min_hit_count is not None:
        shared.ps1_min_hit_count = int(ps1_min_hit_count)
    return MaskSettings(
        geometry_file=settings.geometry_file,
        shared=shared,
        tns=settings.tns,
        asteroids=settings.asteroids,
    )


def mask_settings_to_dict(settings: MaskSettings) -> dict[str, Any]:
    """Serialize for freeze YAML (omit code-default URLs)."""
    d = asdict(settings)
    if d.get("tns", {}).get("download_url") == DEFAULT_TNS_PUBLIC_ZIP_URL:
        d["tns"].pop("download_url", None)
    if d.get("asteroids", {}).get("orbit_times_url") == DEFAULT_TESS_ORBIT_TIMES_URL:
        d["asteroids"].pop("orbit_times_url", None)
    return d


def write_mask_settings(settings: MaskSettings, path: str | Path) -> Path:
    """Write effective settings YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(mask_settings_to_dict(settings), fh, sort_keys=False, default_flow_style=False)
    return path.resolve()


def default_tns_public_csv(data_root: str | Path) -> Path:
    return Path(data_root) / "catalogs" / "tns" / "tns_public_objects.csv"


def default_asteroid_intervals_dir(
    data_root: str | Path, sector: int, camera: int, ccd: int
) -> Path:
    return (
        Path(data_root)
        / "catalogs"
        / f"sector_{int(sector):04d}"
        / f"camera_{int(camera)}"
        / f"ccd_{int(ccd)}"
        / "asteroids"
    )
