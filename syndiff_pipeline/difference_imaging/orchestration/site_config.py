"""Resolve diff site policy + deployment + target row into a frozen SynDiffConfig."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from syndiff_pipeline.common.orchestration.condor import parse_condor_policy_block
from syndiff_pipeline.common.orchestration.deployment import (
    deployment_path_for_config,
    load_deployment,
    load_deployment_file,
    require_deployment_path,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    scc_catalogs_dir,
    scc_ffi_dir,
    scc_templates_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    absolutize_config,
    resolve_config_path,
    save_config,
)

log = logging.getLogger(__name__)

DEFAULT_TEMPLATE_BASE = "shifted_downsampled"
DEFAULT_CATALOG_ROOT = "catalogs"


@dataclass(frozen=True)
class SitePaths:
    """Standard layout under a site directory."""

    site_dir: Path
    template_config: Path
    diff_config: Path
    deployment: Path
    deployment_example: Path

    @classmethod
    def from_site_dir(cls, site_dir: str | Path) -> SitePaths:
        """From site dir.
        
        Parameters
        ----------
        site_dir : str | Path
        
        Returns
        -------
        SitePaths"""
        root = Path(site_dir).expanduser().resolve()
        return cls(
            site_dir=root,
            template_config=root / "pipeline.yaml",
            diff_config=root / "diff_config.yaml",
            deployment=root / "deployment.yaml",
            deployment_example=root / "deployment.yaml.example",
        )


@dataclass
class CondorResources:
    """CondorResources."""
    request_cpus: int = 16
    request_memory: int = 64_000
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0


DIFF_CONDOR_STAGE_NAMES: tuple[str, ...] = ("diff_prep", "background_estimate", "diff")


@dataclass
class DiffSitePolicy:
    """Diff imaging site policy loaded from ``diff_config.yaml``."""

    deployment_file: str = "deployment.yaml"
    pipeline: list = field(default_factory=list)
    defaults: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    additional_forced_targets: list = field(default_factory=list)
    per_event_force_targets: dict = field(default_factory=dict)
    # Back-compat single resource profile (== condor_by_stage["diff"]); prefer
    # condor_by_stage for new code since diff_prep/background_estimate/diff
    # each get their own Condor resource request.
    condor: CondorResources = field(default_factory=CondorResources)
    condor_by_stage: dict = field(default_factory=dict)
    config_path: str = ""


def _parse_deployment_file(raw: dict) -> str:
    """Parse deployment file.
    
    Parameters
    ----------
    raw : dict
    
    Returns
    -------
    str"""
    explicit = str(raw.get("deployment_file", "")).strip()
    if explicit:
        return explicit
    legacy = str((raw.get("notifications") or {}).get("secrets_file", "")).strip()
    if legacy:
        log.warning(
            "notifications.secrets_file is deprecated; use top-level deployment_file instead"
        )
        return legacy
    return "deployment.yaml"


def _parse_condor(raw: dict | None) -> CondorResources:
    """Parse condor.
    
    Parameters
    ----------
    raw : dict | None
    
    Returns
    -------
    CondorResources"""
    cpus, mem, min_mem, max_load15 = parse_condor_policy_block(
        raw,
        context="diff_config.yaml condor",
        default_cpus=16,
        default_memory=64_000,
    )
    return CondorResources(
        request_cpus=cpus,
        request_memory=mem,
        host_stats_min_mem_mb=min_mem,
        host_stats_max_load15=max_load15,
    )


def _parse_condor_by_stage(raw: dict | None) -> dict[str, CondorResources]:
    """Parse the ``condor:`` block into a per-stage resource map.

    Two shapes are accepted:

    - Flat (legacy, pre-diff-split): ``condor: {request_cpus: ..., ...}`` --
      the same resource profile is used for diff_prep/background_estimate/diff.
    - Nested per-stage: ``condor: {diff_prep: {...}, background_estimate:
      {...}, diff: {...}}`` -- all three keys must be present so a config
      author can't accidentally leave a stage on unintended defaults (e.g.
      background_estimate silently getting a small memory request).
    """
    if raw and any(key in DIFF_CONDOR_STAGE_NAMES for key in raw.keys()):
        missing = [name for name in DIFF_CONDOR_STAGE_NAMES if name not in raw]
        if missing:
            raise ValueError(
                "diff_config.yaml condor: block uses the per-stage form but is "
                f"missing entries for {missing}; provide all of "
                f"{list(DIFF_CONDOR_STAGE_NAMES)} explicitly."
            )
        return {name: _parse_condor(raw[name]) for name in DIFF_CONDOR_STAGE_NAMES}
    shared = _parse_condor(raw)
    return {name: shared for name in DIFF_CONDOR_STAGE_NAMES}


def _parse_per_event_force_targets(raw: Any) -> dict[str, list]:
    """Parse per event force targets.
    
    Parameters
    ----------
    raw : Any
    
    Returns
    -------
    dict[str, list]"""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("per_event_force_targets must be a mapping of event label → target list")
    out: dict[str, list] = {}
    for key, val in raw.items():
        label = str(key).strip()
        if not label:
            raise ValueError("per_event_force_targets keys must be non-empty event labels")
        if not isinstance(val, list):
            raise ValueError(
                f"per_event_force_targets[{label!r}] must be a list of target mappings"
            )
        out[label] = copy.deepcopy(val)
    return out


def _per_event_force_targets_for_target(
    policy: DiffSitePolicy, target: Target
) -> list:
    """Look up per-event extras by full label, then bare target_name."""
    by_label = policy.per_event_force_targets.get(target.label())
    if by_label is not None:
        return list(by_label)
    by_name = policy.per_event_force_targets.get(target.target_name)
    if by_name is not None:
        return list(by_name)
    return []


def load_diff_site_policy(config_path: str | Path) -> DiffSitePolicy:
    """Load diff site policy from ``diff_config.yaml``."""
    if not config_path:
        raise ValueError(
            "diff_config_path is empty -- the submitted orchestrator config has "
            "no 'diff_config:' (or 'diff_site_config:'/'diff_config_path:') key "
            "pointing at a diff_config.yaml. Submit 'syndiff diff submit' with "
            "--config pointed at a pipeline.yaml-style file that sets "
            "'diff_config:', not at the diff_config.yaml itself -- otherwise "
            "Condor resource sizing for the diff stages has no policy to load."
        )
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Diff site config must be a YAML mapping: {path}")
    pipeline = raw.get("pipeline")
    if pipeline is None or not isinstance(pipeline, list):
        raise ValueError(f"diff_config.yaml requires a pipeline list: {path}")
    condor_by_stage = _parse_condor_by_stage(raw.get("condor"))
    return DiffSitePolicy(
        deployment_file=_parse_deployment_file(raw),
        pipeline=copy.deepcopy(pipeline),
        defaults=dict(raw.get("defaults") or {}),
        paths=dict(raw.get("paths") or {}),
        overrides=dict(raw.get("overrides") or {}),
        additional_forced_targets=copy.deepcopy(raw.get("additional_forced_targets") or []),
        per_event_force_targets=_parse_per_event_force_targets(
            raw.get("per_event_force_targets")
        ),
        condor=condor_by_stage["diff"],
        condor_by_stage=condor_by_stage,
        config_path=str(path),
    )


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep merge dict.
    
    Parameters
    ----------
    base : dict
    override : dict
    
    Returns
    -------
    dict"""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _target_override(policy: DiffSitePolicy, target: Target) -> dict:
    """Target override.
    
    Parameters
    ----------
    policy : DiffSitePolicy
    target : Target
    
    Returns
    -------
    dict"""
    for key in (target.scc_key(), f"{target.sector}/{target.camera}/{target.ccd}"):
        if key in policy.overrides:
            return policy.overrides[key]
    return {}


def _deployment_paths(
    deployment: dict, *, deployment_path: Path
) -> tuple[str, str, str]:
    """Deployment paths.
    
    Parameters
    ----------
    deployment : dict
    deployment_path : Path
    
    Returns
    -------
    tuple[str, str, str]"""
    workspace_root = require_deployment_path(
        deployment, "workspace_root", deployment_path=deployment_path
    )
    data_root = require_deployment_path(deployment, "data_root", deployment_path=deployment_path)
    ffi_override = str(deployment.get("ffi_dir", "")).strip()
    ffi_dir = (
        str(Path(ffi_override).expanduser().resolve())
        if ffi_override
        else str(Path(data_root) / "tess_ffi")
    )
    return (
        str(Path(workspace_root).expanduser().resolve()),
        str(Path(data_root).expanduser().resolve()),
        ffi_dir,
    )


def _event_dir(workspace_root: str, target: Target) -> Path:
    """Nested event/SCC workspace leaf."""
    return event_scc_leaf(
        workspace_root,
        target.event_name(),
        target.sector,
        target.camera,
        target.ccd,
    )


def resolve_scc_template_dir(
    data_root: str | Path,
    target: Target,
    *,
    oversampling_factor: int = 1,
    store_name: str | None = None,
) -> Path:
    """Absolute SCC templates store for one target."""
    return scc_templates_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        oversampling_factor=int(oversampling_factor),
        store_name=store_name,
    )


def _gaia_catalog_path(
    target: Target,
    *,
    data_root: Path,
    output_store_name: str | None,
) -> Path:
    """Resolve the SCC-scoped Gaia catalog path.

    Keyed by SCC (+ output store lane) only -- never by event label -- so the
    same SCC always resolves the same catalog regardless of which event/target
    row references it.

    Parameters
    ----------
    target : Target
    data_root : Path
    output_store_name : str | None
        Normalized diff output store lane name (``None`` for the default lane).

    Returns
    -------
    Path
        ``{lane_root}/gaia_catalog_pipeline.csv`` if the diff pipeline has
        already cached one for this SCC/lane, else the SCC catalogs-dir
        default.
    """
    from syndiff_pipeline.common.scc_paths import scc_diff_dir
    from syndiff_pipeline.difference_imaging.support.paths import (
        GAIA_CATALOG_PIPELINE_BASENAME,
    )

    s, c, k = target.sector, target.camera, target.ccd
    lane_root = scc_diff_dir(data_root, s, c, k, store_name=output_store_name)
    pipeline_csv = lane_root / GAIA_CATALOG_PIPELINE_BASENAME
    if pipeline_csv.is_file():
        return pipeline_csv.resolve()
    return (
        scc_catalogs_dir(data_root, s, c, k)
        / f"gaia_catalog_s{s:04d}_{c}_{k}.csv"
    )


def event_geometry_mode(event_dir: str | Path) -> str:
    """Read ``geometry_mode`` from the event's ``event_job.json``."""
    from syndiff_pipeline.common.wcs_grouping import _event_job_path

    job = Path(_event_job_path(event_dir))
    if job.is_file():
        try:
            mode = json.loads(job.read_text()).get("geometry_mode")
            if mode:
                return str(mode).lower()
        except Exception:
            pass
    return "linear"


def resolve_event_template_dir(
    event_dir: str | Path,
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int = 1,
    store_name: str | None = None,
) -> str:
    """Resolve SCC templates directory (no workspace symlinks)."""
    store = resolve_scc_template_dir(
        data_root,
        Target(sector, camera, ccd, 0.0, 0.0, "x"),
        oversampling_factor=oversampling_factor,
        store_name=store_name,
    )
    if store.is_dir():
        return str(store.resolve())
    raise FileNotFoundError(
        f"SCC templates store missing: {store}. Run template pipeline templates stage."
    )


def resolve_diff_config(
    target: Target,
    policy: DiffSitePolicy,
    deployment: dict,
    *,
    deployment_path: Path,
    site_config_dir: Path | None = None,
) -> SynDiffConfig:
    """Merge site policy, deployment paths, and target fields into a SynDiffConfig."""
    workspace_root, data_root, _ffi_root = _deployment_paths(
        deployment, deployment_path=deployment_path
    )
    site_dir = site_config_dir or Path(policy.config_path).parent
    override = _target_override(policy, target)
    merged_defaults = _deep_merge_dict(policy.defaults, override.get("defaults", {}))
    merged_paths = _deep_merge_dict(policy.paths, override.get("paths", {}))

    data_root_path = Path(data_root)
    # SynDiffConfig.ffi_dir is the SCC ffi leaf (same convention as template resolve_config).
    ffi_dir = str(
        scc_ffi_dir(data_root_path, target.sector, target.camera, target.ccd)
    )

    from syndiff_pipeline.common.scc_paths import normalize_store_name, scc_diff_dir

    output_store_name = normalize_store_name(merged_paths.get("output_store_name"))
    # output_dir is SCC-scoped (diff is keyed by SCC, not by event) -- the
    # lane root is where diff artifacts, the shared mask, and the frozen
    # diff_config.yaml lock already live.
    output_dir = scc_diff_dir(
        data_root_path,
        target.sector,
        target.camera,
        target.ccd,
        store_name=output_store_name,
    )

    os_factor = int(merged_defaults.get("oversampling_factor", 1) or 1)
    template_dir = merged_paths.get("template_dir")
    if template_dir:
        template_dir = resolve_config_path(str(template_dir), data_root_path)
    else:
        from syndiff_pipeline.common.scc_paths import normalize_store_name

        template_store_name = normalize_store_name(
            merged_paths.get("template_store_name")
        )
        template_dir = str(
            resolve_scc_template_dir(
                data_root_path,
                target,
                oversampling_factor=os_factor,
                store_name=template_store_name,
            )
        )

    gaia_catalog = merged_paths.get("gaia_catalog")
    if gaia_catalog:
        gaia_catalog = resolve_config_path(str(gaia_catalog), data_root_path)
    else:
        gaia_catalog = str(
            _gaia_catalog_path(
                target,
                data_root=data_root_path,
                output_store_name=output_store_name,
            )
        )

    optional_paths = {}
    for key in ("straps_csv", "bsc_catalog", "removed_stars_csv", "manifest"):
        val = merged_paths.get(key) or deployment.get(key)
        if val:
            optional_paths[key] = resolve_config_path(str(val), data_root_path)

    store_names: dict[str, str | None] = {}
    for key in ("template_store_name", "output_store_name", "remap_store_name"):
        raw = merged_paths.get(key)
        if raw is not None:
            store_names[key] = normalize_store_name(str(raw) if str(raw).strip() else None)

    deprecated_median = merged_paths.get("median_mask_path") or deployment.get(
        "median_mask_path"
    )
    if deprecated_median:
        log.warning(
            "median_mask_path is no longer used (TGLC median_mask support removed); "
            "ignoring %r",
            deprecated_median,
        )

    # straps_csv / bsc_catalog unset → packaged defaults at use time (do not
    # inject absolute bundled paths into SynDiffConfig / frozen snapshots).

    pipeline = copy.deepcopy(policy.pipeline)
    if override.get("pipeline"):
        pipeline = copy.deepcopy(override["pipeline"])

    cfg = SynDiffConfig(
        ffi_dir=ffi_dir,
        output_dir=str(output_dir),
        gaia_catalog=gaia_catalog,
        template_dir=template_dir or "",
        pipeline=pipeline,
        target_ra=target.target_ra,
        target_dec=target.target_dec,
        target_name=target.target_name,
        sector=target.sector,
        camera=target.camera,
        ccd=target.ccd,
        data_root=str(data_root),
        site_config_dir=str(Path(site_dir).expanduser().resolve()),
        oversampling_factor=os_factor,
        **optional_paths,
        **store_names,
    )
    for key, val in merged_defaults.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    # Diff execute no longer consumes additional_forced_targets; photometry stage owns them.
    cfg.additional_forced_targets = []

    return absolutize_config(cfg, site_dir)


def freeze_target_diff_config(
    config_path: str | Path,
    target: Target,
    *,
    deployment_path: str | Path | None = None,
) -> SynDiffConfig:
    """Load site policy + deployment and return a frozen per-target SynDiffConfig."""
    policy = load_diff_site_policy(config_path)
    cfg_path = Path(config_path).expanduser().resolve()
    deploy_path = (
        Path(deployment_path).expanduser().resolve()
        if deployment_path is not None
        else deployment_path_for_config(cfg_path, policy.deployment_file)
    )
    deployment = load_deployment_file(deploy_path)
    return resolve_diff_config(
        target,
        policy,
        deployment,
        deployment_path=deploy_path,
        site_config_dir=cfg_path.parent,
    )


def write_frozen_diff_config(cfg: SynDiffConfig, yaml_path: str | Path) -> Path:
    """Write a frozen per-target diff config with absolute paths."""
    path = Path(yaml_path).expanduser().resolve()
    save_config(cfg, str(path))
    return path


def load_deployment_for_diff_config(config_path: str | Path) -> tuple[dict, Path]:
    """Load deployment dict for a diff site config path."""
    cfg_path = Path(config_path).expanduser().resolve()
    policy = load_diff_site_policy(cfg_path)
    deploy_path = deployment_path_for_config(cfg_path, policy.deployment_file)
    return load_deployment(cfg_path, policy.deployment_file), deploy_path
