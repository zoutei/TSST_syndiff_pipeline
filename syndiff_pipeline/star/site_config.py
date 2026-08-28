"""Load star site policy, star_targets CSV, and merged per-run configuration."""

from __future__ import annotations

import copy
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from syndiff_pipeline.common.orchestration.condor import parse_condor_policy_block
from syndiff_pipeline.common.orchestration.targets import Target, _parse_bool, find_target
from syndiff_pipeline.difference_imaging.support.paths import normalize_photometry_run_id

log = logging.getLogger(__name__)

STAR_TARGETS_HEADER = frozenset(
    {
        "sector",
        "camera",
        "ccd",
        "target_name",
        "stars_file",
        "baseline_workspace_run_id",
        "baseline_diffs",
        "baseline_convolved",
        "phot_bkg",
        "enabled",
    }
)

VALID_PS1_SOURCES = frozenset({"zarr_local_only", "zarr_download", "stream"})
_LEGACY_PS1_SOURCE = {
    "zarr": "zarr_download",
    "download": "stream",
}


@dataclass
class StarBaselineConfig:
    """Resolved baseline workspace labels for one star run."""

    workspace_run_id: str = "none"
    diffs: str = "hp_d"
    convolved: str | None = "hp_c"
    phot_bkg: str | None = "ks_b_s"


@dataclass
class StarEpsfConfig:
    """Gridded ePSF fitting on baseline diff images before gepsf photometry."""

    enabled: bool = False
    diffs: str | None = None
    output: str = "epsf_r1"
    tile_nx: int = 2
    tile_ny: int = 2
    epsf_oversample: int = 4
    psf_size: int = 11
    extract_size: int | None = 11
    min_stars_per_tile: int = 5
    # Always filters on derived tess_mag, never raw Gaia phot_rp_mean_mag
    # (standing policy, 2026-08-22) -- resolved the same way as the diff
    # stage's EpsfParams.tess_mag_max.
    tess_mag_max: float | None = 12.95
    epsf_maxiters: int = 15
    epsf_recentering_maxiters: int = 20
    epsf_n_jobs: int | None = 8
    # Orbit-binned ePSF (see difference_imaging/stages/gridded_epsf_orbit.py
    # and EpsfParams) -- same knobs, star-branch default matches the diff
    # stage's default so a star epsf block behaves like the production epsf
    # stage unless explicitly overridden.
    epsf_mode: str = "orbit_binned"
    epsf_per_orbit: int = 5
    epsf_frames_per_anchor: int = 20
    epsf_stack_before_fit: bool = True
    epsf_anchor_edge_fraction: float = 0.12
    epsf_anchor_edge_boost: float = 3.0
    epsf_anchor_window_max_expand: int = 80
    epsf_quality_bitmask: int = 583
    epsf_debug_plots: bool = True


@dataclass
class StarCondorConfig:
    """HTCondor resource request for the star stage."""

    request_cpus: int = 8
    request_memory: int = 100_000
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0


@dataclass
class StarSitePolicy:
    """Site-level star policy from ``star_config.yaml``."""

    deployment_file: str = "deployment.yaml"
    defaults: dict = field(default_factory=dict)
    baseline: StarBaselineConfig = field(default_factory=StarBaselineConfig)
    photometry: dict = field(default_factory=dict)
    epsf: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    condor: StarCondorConfig = field(default_factory=StarCondorConfig)
    config_path: str = ""
    ps1_zarr_path: str | None = None


@dataclass(frozen=True)
class StarTargetRow:
    """One enabled row from ``star_targets.csv``."""

    target: Target
    stars_file: str
    baseline_workspace_run_id: str | None = None
    baseline_diffs: str | None = None
    baseline_convolved: str | None = None
    phot_bkg: str | None = None

    def scc_key(self) -> str:
        return self.target.scc_key()


@dataclass
class StarRunConfig:
    """Merged knobs for one SCC star run (row > overrides > defaults)."""

    cutout_size: int = 96
    stamp_size: int = 24
    kernel_margin_px: int = 470
    ps1_source: str = "zarr_download"
    debug_plots: bool = True
    workspace_run_id: str | None = None  # deprecated; unused for writes
    max_ffis: int | None = None
    overwrite: bool = False
    oversampling_factor: int = 1
    # Named templates lane → templates_{NAME}/; None → templates/
    template_store_name: str | None = None
    baseline: StarBaselineConfig = field(default_factory=StarBaselineConfig)
    photometry_methods: list[dict] = field(default_factory=list)
    epsf: StarEpsfConfig | None = None
    stars_file: str = ""
    ps1_zarr_path: str | None = None
    photometry_run_id: str | None = None


def normalize_ps1_source(value: str | None, *, warn_legacy: bool = True) -> str:
    """Normalize ``ps1_source`` including legacy ``zarr`` / ``download`` aliases."""
    raw = str(value or "zarr_download").strip()
    if raw in _LEGACY_PS1_SOURCE:
        normalized = _LEGACY_PS1_SOURCE[raw]
        if warn_legacy:
            log.warning(
                "ps1_source=%r is deprecated; use %r instead",
                raw,
                normalized,
            )
        return normalized
    if raw not in VALID_PS1_SOURCES:
        raise ValueError(
            f"Invalid ps1_source {raw!r}; expected one of: "
            f"{', '.join(sorted(VALID_PS1_SOURCES))}"
        )
    return raw


def _parse_baseline_block(raw: dict | None, *, fallback: StarBaselineConfig) -> StarBaselineConfig:
    raw = raw or {}
    return StarBaselineConfig(
        workspace_run_id=str(
            raw.get("workspace_run_id", fallback.workspace_run_id) or fallback.workspace_run_id
        ).strip()
        or "none",
        diffs=str(raw.get("diffs", fallback.diffs) or fallback.diffs).strip() or "hp_d",
        convolved=_optional_str(raw.get("convolved", fallback.convolved)),
        phot_bkg=_optional_str(raw.get("phot_bkg", fallback.phot_bkg)),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _method_inputs(method: dict) -> dict:
    raw = method.get("inputs") or {}
    return raw if isinstance(raw, dict) else {}


def epsf_workspace_from_method(method: dict) -> str | None:
    """Return the ePSF workspace label from a photometry method ``inputs.epsf``."""
    label = _optional_str(_method_inputs(method).get("epsf"))
    return label


def required_epsf_workspaces(methods: list[dict]) -> list[str]:
    """Unique ePSF workspace labels referenced by ``psf_type: epsf`` methods."""
    seen: set[str] = set()
    out: list[str] = []
    for method in methods:
        mtype = str(method.get("type", "")).strip().lower()
        if mtype not in ("psf", "prf"):
            continue
        if str(method.get("psf_type", "prf")).strip().lower() != "epsf":
            continue
        label = epsf_workspace_from_method(method)
        if not label:
            name = str(method.get("name", "?"))
            raise ValueError(
                f"photometry method {name!r} has psf_type epsf but no inputs.epsf"
            )
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def normalize_photometry_methods(methods: list[dict]) -> list[dict]:
    """Validate photometry methods and normalize ``inputs`` blocks."""
    normalized: list[dict] = []
    for method in methods:
        entry = copy.deepcopy(method)
        mtype = str(entry.get("type", "")).strip().lower()
        if mtype in ("psf", "prf") and str(entry.get("psf_type", "prf")).strip().lower() == "epsf":
            label = epsf_workspace_from_method(entry)
            if not label:
                name = str(entry.get("name", "?"))
                raise ValueError(
                    f"photometry method {name!r} has psf_type epsf but no inputs.epsf"
                )
            entry["epsf_workspace"] = label
        normalized.append(entry)
    return normalized


def _parse_star_epsf(raw: dict | None) -> StarEpsfConfig | None:
    raw = raw or {}
    if not raw:
        return None
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return None
    inputs = raw.get("inputs") or {}
    diffs = _optional_str(inputs.get("diffs")) if isinstance(inputs, dict) else None
    extract_size = raw.get("extract_size")
    return StarEpsfConfig(
        enabled=True,
        diffs=diffs,
        output=str(raw.get("output", "epsf_r1")).strip() or "epsf_r1",
        tile_nx=int(raw.get("tile_nx", 2)),
        tile_ny=int(raw.get("tile_ny", 2)),
        epsf_oversample=int(raw.get("epsf_oversample", 4)),
        psf_size=int(raw.get("psf_size", 11)),
        extract_size=int(extract_size) if extract_size not in (None, "") else None,
        min_stars_per_tile=int(raw.get("min_stars_per_tile", 5)),
        tess_mag_max=float(raw["tess_mag_max"]) if raw.get("tess_mag_max") not in (None, "") else 12.95,
        epsf_maxiters=int(raw.get("epsf_maxiters", 15)),
        epsf_recentering_maxiters=int(raw.get("epsf_recentering_maxiters", 20)),
        epsf_n_jobs=int(raw["epsf_n_jobs"]) if raw.get("epsf_n_jobs") not in (None, "") else None,
        epsf_mode=str(raw.get("epsf_mode", "orbit_binned")).strip() or "orbit_binned",
        epsf_per_orbit=int(raw.get("epsf_per_orbit", 5)),
        epsf_frames_per_anchor=int(raw.get("epsf_frames_per_anchor", 20)),
        epsf_stack_before_fit=bool(raw.get("epsf_stack_before_fit", True)),
        epsf_anchor_edge_fraction=float(raw.get("epsf_anchor_edge_fraction", 0.12)),
        epsf_anchor_edge_boost=float(raw.get("epsf_anchor_edge_boost", 3.0)),
        epsf_anchor_window_max_expand=int(raw.get("epsf_anchor_window_max_expand", 80)),
        epsf_quality_bitmask=int(raw.get("epsf_quality_bitmask", 583)),
        epsf_debug_plots=bool(raw.get("epsf_debug_plots", True)),
    )


def _parse_star_condor(raw: dict | None) -> StarCondorConfig:
    raw = raw or {}
    cpus, mem, min_mem, max_load15 = parse_condor_policy_block(
        raw,
        context="star_config.yaml condor",
        default_cpus=8,
        default_memory=100_000,
    )
    return StarCondorConfig(
        request_cpus=cpus,
        request_memory=mem,
        host_stats_min_mem_mb=min_mem,
        host_stats_max_load15=max_load15,
    )


def load_star_site_policy(path: str | Path) -> StarSitePolicy:
    """Load ``star_config.yaml`` site policy."""
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = dict(raw.get("defaults") or {})
    baseline_raw = raw.get("baseline") or {}
    baseline = _parse_baseline_block(baseline_raw, fallback=StarBaselineConfig())
    photometry = dict(raw.get("photometry") or {})
    epsf = dict(raw.get("epsf") or {})
    overrides = dict(raw.get("overrides") or {})
    ps1_zarr = _optional_str(raw.get("ps1_zarr_path"))
    return StarSitePolicy(
        deployment_file=str(raw.get("deployment_file", "deployment.yaml")).strip()
        or "deployment.yaml",
        defaults=defaults,
        baseline=baseline,
        photometry=photometry,
        epsf=epsf,
        overrides=overrides,
        condor=_parse_star_condor(raw.get("condor")),
        config_path=str(config_path),
        ps1_zarr_path=ps1_zarr,
    )


def _resolve_stars_file(path: str, *, site_dir: Path) -> str:
    text = str(path or "").strip()
    if not text:
        raise ValueError("stars_file is required in star_targets row")
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = (site_dir / p).resolve()
    return str(p)


def _read_star_targets_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = frozenset(f.strip().lower() for f in reader.fieldnames)
        if not STAR_TARGETS_HEADER.issubset(fields):
            missing = sorted(STAR_TARGETS_HEADER - fields)
            raise ValueError(
                f"Unrecognized star_targets CSV header in {path}; missing columns: {missing}"
            )
        return [{k.strip().lower(): v for k, v in row.items()} for row in reader]


def load_star_targets(path: str | Path, *, site_dir: str | Path | None = None) -> list[StarTargetRow]:
    """Load enabled rows from ``star_targets.csv``."""
    csv_path = Path(path).expanduser().resolve()
    site = Path(site_dir).expanduser().resolve() if site_dir else csv_path.parent
    rows = _read_star_targets_csv(csv_path)
    out: list[StarTargetRow] = []
    for row in rows:
        if not _parse_bool(row.get("enabled"), default=True):
            continue
        target = Target(
            sector=int(row["sector"]),
            camera=int(row["camera"]),
            ccd=int(row["ccd"]),
            target_ra=0.0,
            target_dec=0.0,
            target_name=str(row["target_name"]).strip(),
            enabled=True,
        )
        out.append(
            StarTargetRow(
                target=target,
                stars_file=_resolve_stars_file(row.get("stars_file", ""), site_dir=site),
                baseline_workspace_run_id=_optional_str(row.get("baseline_workspace_run_id")),
                baseline_diffs=_optional_str(row.get("baseline_diffs")),
                baseline_convolved=_optional_str(row.get("baseline_convolved")),
                phot_bkg=_optional_str(row.get("phot_bkg")),
            )
        )
    return out


def resolve_star_photometry_run_id(
    *,
    star_defaults: dict,
    site_dir: str | Path,
    photometry_config_path: str | None = None,
    target: Target | None = None,
) -> str | None:
    """Resolve photometry tree id from star defaults, then photometry site policy."""
    raw = star_defaults.get("photometry_run_id")
    if raw is not None:
        rid = normalize_photometry_run_id(str(raw).strip() or None)
        if rid is not None:
            return rid

    phot_path = str(photometry_config_path or "").strip()
    site = Path(site_dir).expanduser().resolve()
    if not phot_path:
        candidate = site / "photometry_config.yaml"
        if candidate.is_file():
            phot_path = str(candidate)
    if not phot_path:
        return None

    from syndiff_pipeline.photometry.site_config import (
        load_photometry_site_policy,
        resolve_photometry_run_config,
    )

    policy = load_photometry_site_policy(phot_path)
    if target is not None:
        return resolve_photometry_run_config(policy, target, site_dir=site).photometry_run_id
    return normalize_photometry_run_id(
        str((policy.defaults or {}).get("photometry_run_id") or "").strip() or None
    )


def find_star_target_row(rows: list[StarTargetRow], scc: str) -> StarTargetRow:
    """Find a star target row by SCC key or full event label."""
    targets = [row.target for row in rows]
    target = find_target(targets, scc)
    for row in rows:
        if row.target.label() == target.label():
            return row
    raise KeyError(f"No star_targets row for {scc!r}")


def resolve_star_run_config(
    policy: StarSitePolicy,
    star_target_row: StarTargetRow,
    *,
    site_dir: str | Path,
    photometry_config_path: str | None = None,
) -> StarRunConfig:
    """Merge policy defaults, SCC overrides, and star_targets row."""
    site = Path(site_dir).expanduser().resolve()
    merged_defaults = copy.deepcopy(policy.defaults)
    override_block = policy.overrides.get(star_target_row.scc_key()) or {}
    if override_block:
        merged_defaults = _deep_merge_dict(merged_defaults, override_block)

    baseline = _parse_baseline_block(
        merged_defaults.pop("baseline", None) or {},
        fallback=policy.baseline,
    )
    if star_target_row.baseline_workspace_run_id:
        baseline.workspace_run_id = star_target_row.baseline_workspace_run_id
    if star_target_row.baseline_diffs:
        baseline.diffs = star_target_row.baseline_diffs
    if star_target_row.baseline_convolved:
        baseline.convolved = star_target_row.baseline_convolved
    if star_target_row.phot_bkg:
        baseline.phot_bkg = star_target_row.phot_bkg

    photometry_block = merged_defaults.pop("photometry", None) or policy.photometry or {}
    epsf_block = merged_defaults.pop("epsf", None) or policy.epsf or {}
    methods = list(photometry_block.get("methods") or [])
    if not methods:
        methods = [
            {"name": "ap3", "type": "aperture", "tar_ap": 3, "sky_in": 5, "sky_out": 9},
        ]
    methods = normalize_photometry_methods(methods)
    epsf_cfg = _parse_star_epsf(epsf_block)
    if epsf_cfg is not None:
        for label in required_epsf_workspaces(methods):
            if label != epsf_cfg.output:
                raise ValueError(
                    f"photometry references inputs.epsf={label!r} but epsf.output "
                    f"is {epsf_cfg.output!r}; add a matching epsf block or align labels"
                )

    ps1_source = normalize_ps1_source(merged_defaults.get("ps1_source", "zarr_download"))
    workspace_run_id = _optional_str(merged_defaults.get("workspace_run_id"))
    if workspace_run_id is not None:
        log.warning(
            "star defaults/overrides workspace_run_id=%r is deprecated and ignored; "
            "star outputs land in phot_{photometry_run_id}/host_star/",
            workspace_run_id,
        )
    photometry_run_id = resolve_star_photometry_run_id(
        star_defaults=merged_defaults,
        site_dir=site,
        photometry_config_path=photometry_config_path,
        target=star_target_row.target,
    )
    max_ffis_raw = merged_defaults.get("max_ffis")
    max_ffis = int(max_ffis_raw) if max_ffis_raw not in (None, "") else None
    oversampling_factor = max(1, int(merged_defaults.get("oversampling_factor", 1) or 1))
    from syndiff_pipeline.common.scc_paths import normalize_store_name

    template_store_name = normalize_store_name(
        merged_defaults.get("template_store_name")
    )

    return StarRunConfig(
        cutout_size=int(merged_defaults.get("cutout_size", 96)),
        stamp_size=int(merged_defaults.get("stamp_size", 24)),
        kernel_margin_px=int(merged_defaults.get("kernel_margin_px", 470)),
        ps1_source=ps1_source,
        debug_plots=bool(merged_defaults.get("debug_plots", True)),
        # Deprecated: kept for config compatibility only; outputs use phot_{run_id}/host_star/.
        workspace_run_id=workspace_run_id,
        max_ffis=max_ffis,
        overwrite=bool(merged_defaults.get("overwrite", False)),
        oversampling_factor=oversampling_factor,
        template_store_name=template_store_name,
        baseline=baseline,
        photometry_methods=methods,
        epsf=epsf_cfg,
        stars_file=star_target_row.stars_file,
        ps1_zarr_path=policy.ps1_zarr_path,
        photometry_run_id=photometry_run_id,
    )


def star_targets_to_orchestrator_targets(rows: list[StarTargetRow]) -> list[Target]:
    """Convert star target rows to orchestrator :class:`Target` objects."""
    return [row.target for row in rows]


def write_frozen_star_config(policy: StarSitePolicy, dest: str | Path) -> Path:
    """Write merged site policy YAML to a frozen run path."""
    dest_path = Path(dest).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "deployment_file": policy.deployment_file,
        "defaults": policy.defaults,
        "baseline": {
            "workspace_run_id": policy.baseline.workspace_run_id,
            "diffs": policy.baseline.diffs,
            "convolved": policy.baseline.convolved,
            "phot_bkg": policy.baseline.phot_bkg,
        },
        "photometry": policy.photometry,
        "epsf": policy.epsf,
        "overrides": policy.overrides,
    }
    if policy.ps1_zarr_path:
        payload["ps1_zarr_path"] = policy.ps1_zarr_path
    payload["condor"] = {
        "request_cpus": policy.condor.request_cpus,
        "request_memory": policy.condor.request_memory,
        "host_stats_min_mem_mb": policy.condor.host_stats_min_mem_mb,
        "host_stats_max_load15": policy.condor.host_stats_max_load15,
    }
    dest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest_path


def resolve_star_config_path(*, meta: dict | None, runner_cfg) -> tuple[Path, Path]:
    """Resolve ``(policy_path, site_dir)`` for the star stage from run metadata.

    ``policy_path`` is where policy *content* (defaults/baseline/photometry/
    epsf/overrides) is read from: the frozen ``runs/{run_id}/star_config.yaml``
    snapshot -- recorded as ``star_config_path`` in run_meta / on
    ``RunnerConfig`` -- which submit always freezes (see
    ``_prepare_run_directory``), so it is authoritative.

    ``site_dir`` is always the directory of the *live* site ``star_config.yaml``
    (``source_star_config_path``'s parent) -- never a run directory. Callers
    must use it, not ``policy_path.parent``, for anything that resolves
    relative to the site config: ``deployment.yaml``, ``stars_file`` (via
    :func:`_resolve_stars_file`), and the ``photometry_config.yaml`` fallback
    probe in :func:`resolve_star_photometry_run_id`. Pointing those at a run
    directory instead -- which has no ``deployment.yaml`` -- breaks every
    execute; see the wave-B-2 brief this fixes.

    When no ``source_star_config_path`` is recorded (e.g. an ad hoc run with
    incomplete run_meta), ``site_dir`` falls back to ``policy_path.parent``.
    """
    meta = meta or {}
    frozen_raw = str(
        meta.get("star_config_path") or getattr(runner_cfg, "star_config_path", "") or ""
    ).strip()
    if not frozen_raw:
        raise ValueError("Star stage requires star_config_path in run_meta or on RunnerConfig")
    policy_path = Path(frozen_raw).expanduser().resolve()
    source_raw = str(meta.get("source_star_config_path") or "").strip()
    site_dir = (
        Path(source_raw).expanduser().resolve().parent if source_raw else policy_path.parent
    )
    return policy_path, site_dir
