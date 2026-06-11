"""Resolve diff site policy + deployment + target row into a frozen SynDiffConfig."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from syndiff_pipeline.common.orchestration.deployment import (
    deployment_path_for_config,
    load_deployment,
    load_deployment_file,
    require_deployment_path,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    absolutize_config,
    normalize_additional_forced_targets,
    resolve_config_path,
    save_config,
)
from syndiff_pipeline.common.orchestration.template_handoff import (
    event_templates_symlink_path,
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
    request_cpus: int = 8
    request_memory: int = 64_000
    requirements: str | None = "Memory >= 64000 && LoadAvg < 10"
    rank: str | None = "-LoadAvg"


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
    condor: CondorResources = field(default_factory=CondorResources)
    config_path: str = ""


def _parse_deployment_file(raw: dict) -> str:
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
    raw = raw or {}
    return CondorResources(
        request_cpus=int(raw.get("request_cpus", 8)),
        request_memory=int(raw.get("request_memory", 64_000)),
        requirements=raw.get("requirements"),
        rank=raw.get("rank"),
    )


def _parse_per_event_force_targets(raw: Any) -> dict[str, list]:
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
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Diff site config must be a YAML mapping: {path}")
    pipeline = raw.get("pipeline")
    if pipeline is None or not isinstance(pipeline, list):
        raise ValueError(f"diff_config.yaml requires a pipeline list: {path}")
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
        condor=_parse_condor(raw.get("condor")),
        config_path=str(path),
    )


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _target_override(policy: DiffSitePolicy, target: Target) -> dict:
    for key in (target.scc_key(), f"{target.sector}/{target.camera}/{target.ccd}"):
        if key in policy.overrides:
            return policy.overrides[key]
    return {}


def _deployment_paths(
    deployment: dict, *, deployment_path: Path
) -> tuple[str, str, str]:
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
    return Path(workspace_root) / "events" / target.label()


def _gaia_catalog_path(
    target: Target,
    *,
    data_root: Path,
    event_dir: Path,
    catalog_root: str,
) -> Path:
    pipeline_csv = event_dir / "gaia_catalog_pipeline.csv"
    if pipeline_csv.is_file():
        return pipeline_csv.resolve()
    s, c, k = target.sector, target.camera, target.ccd
    return (
        data_root
        / catalog_root
        / f"sector_{s:04d}"
        / f"camera_{c}"
        / f"ccd_{k}"
        / f"gaia_catalog_s{s:04d}_{c}_{k}.csv"
    )


def resolve_event_template_dir(event_dir: str | Path) -> str:
    """Resolve physical template directory via ``events/{target}/ws/templates`` symlink."""
    link = event_templates_symlink_path(event_dir)
    if link.is_symlink():
        resolved = link.resolve()
        if resolved.is_dir():
            return str(resolved)
    raise FileNotFoundError(
        f"Missing or broken templates symlink {link}. "
        "Run template pipeline downsample to create ws/templates."
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
    workspace_root, data_root, ffi_dir = _deployment_paths(
        deployment, deployment_path=deployment_path
    )
    site_dir = site_config_dir or Path(policy.config_path).parent
    override = _target_override(policy, target)
    merged_defaults = _deep_merge_dict(policy.defaults, override.get("defaults", {}))
    merged_paths = _deep_merge_dict(policy.paths, override.get("paths", {}))

    event_dir = _event_dir(workspace_root, target)
    template_base = str(merged_paths.get("template_base", DEFAULT_TEMPLATE_BASE))
    catalog_root = str(merged_paths.get("catalog_root", DEFAULT_CATALOG_ROOT))
    data_root_path = Path(data_root)

    template_dir = merged_paths.get("template_dir")
    if template_dir:
        template_dir = resolve_config_path(str(template_dir), data_root_path)
    else:
        try:
            template_dir = resolve_event_template_dir(event_dir)
        except FileNotFoundError:
            if template_base and (data_root_path / template_base).is_dir():
                pattern = (
                    f"sector{target.sector:04d}_camera{target.camera}_ccd{target.ccd}"
                )
                matches = sorted(
                    p
                    for p in (data_root_path / template_base).glob(f"{pattern}*")
                    if p.is_dir()
                )
                if len(matches) == 1:
                    template_dir = str(matches[0].resolve())
                else:
                    raise
            else:
                raise

    gaia_catalog = merged_paths.get("gaia_catalog")
    if gaia_catalog:
        gaia_catalog = resolve_config_path(str(gaia_catalog), data_root_path)
    else:
        gaia_catalog = str(
            _gaia_catalog_path(
                target,
                data_root=data_root_path,
                event_dir=event_dir,
                catalog_root=catalog_root,
            )
        )

    optional_paths = {}
    for key in ("median_mask_path", "straps_csv", "removed_stars_csv", "manifest"):
        val = merged_paths.get(key) or deployment.get(key)
        if val:
            optional_paths[key] = resolve_config_path(str(val), data_root_path)

    pipeline = copy.deepcopy(policy.pipeline)
    if override.get("pipeline"):
        pipeline = copy.deepcopy(override["pipeline"])

    cfg = SynDiffConfig(
        ffi_dir=ffi_dir,
        output_dir=str(event_dir),
        gaia_catalog=gaia_catalog,
        template_dir=template_dir or "",
        pipeline=pipeline,
        target_ra=target.target_ra,
        target_dec=target.target_dec,
        sector=target.sector,
        camera=target.camera,
        ccd=target.ccd,
        **optional_paths,
    )
    for key, val in merged_defaults.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    per_event = _per_event_force_targets_for_target(policy, target)
    combined_forced = list(policy.additional_forced_targets) + per_event
    cfg.additional_forced_targets = normalize_additional_forced_targets(combined_forced)

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
