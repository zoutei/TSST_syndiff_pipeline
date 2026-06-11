"""YAML configuration for the template pipeline runner."""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from syndiff_pipeline.common.orchestration.notifications import (
    NotificationConfig,
    parse_notification_config,
)
from syndiff_pipeline.template_creation.orchestration.bundled_assets import skycell_wcs_csv
from syndiff_pipeline.common.orchestration.deployment import (
    deployment_path_for_config,
    load_deployment,
    require_deployment_path,
    warn_legacy_config_paths,
)
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    ResourcePoolParams,
    TemplateStageParams,
    parse_stage_params,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.orchestration.workspace import (
    normalize_workspace_root,
    runs_root as runs_root,
    state_db_path,
)

log = logging.getLogger(__name__)


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(os.path.expanduser(str(value)))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def parse_deployment_file(raw: dict) -> str:
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


@dataclass
class RunnerConfig:
    deployment_file: str = "deployment.yaml"
    data_root: str = ""
    ffi_dir: str = ""
    workspace_root: str = ""
    runs_root: str = ""
    state_db_path: str = ""
    skycell_wcs_csv: str = ""
    diff_config_path: str = ""
    stages: TemplateStageParams = field(default_factory=lambda: parse_stage_params({}))
    resources: Dict[str, ResourcePoolParams] = field(
        default_factory=lambda: _parse_resources({})
    )
    overrides: Dict[str, dict] = field(default_factory=dict)
    scheduler_heartbeat_interval_s: float = 30.0
    verify_max_workers: int = 1
    verify_budget_per_tick: int = 16
    max_stage_attempts: int = 3
    requeue_backoff_s: float = 30.0
    condor_hold_timeout_s: float = 600.0
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    def runs_dir(self) -> str:
        return self.runs_root or str(runs_root(self.workspace_root))

    def stage_executor(self, stage: str) -> str:
        """Return launch executor for a stage: 'local' or 'condor'."""
        from syndiff_pipeline.pipeline_spec import SYNDIFF_PIPELINE

        stage_spec = SYNDIFF_PIPELINE.get(stage)
        if stage_spec is None:
            return "local"
        return stage_spec.resolve_executor(self)


def _parse_resources(raw: dict | None) -> Dict[str, ResourcePoolParams]:
    raw = raw or {}
    out: Dict[str, ResourcePoolParams] = {}
    for name, spec in raw.items():
        spec = spec or {}
        out[name] = ResourcePoolParams(max_concurrent=int(spec.get("max_concurrent", 1)))
    if "network" not in out:
        out["network"] = ResourcePoolParams(max_concurrent=3)
    if "cpu_light" not in out:
        out["cpu_light"] = ResourcePoolParams(max_concurrent=2)
    if "mapping" not in out:
        out["mapping"] = ResourcePoolParams(max_concurrent=6)
    if "ps1_process" not in out:
        out["ps1_process"] = ResourcePoolParams(max_concurrent=4)
    if "diff" not in out:
        out["diff"] = ResourcePoolParams(max_concurrent=2)
    return out


def _paths_from_deployment(
    deployment: dict, *, deployment_path: Path
) -> tuple[str, str, str, str, str, str]:
    handoff = require_deployment_path(deployment, "workspace_root", deployment_path=deployment_path)
    data = require_deployment_path(deployment, "data_root", deployment_path=deployment_path)
    ffi_override = str(deployment.get("ffi_dir", "")).strip()
    ffi_dir = (
        str(Path(ffi_override).expanduser().resolve())
        if ffi_override
        else str(Path(data) / "tess_ffi")
    )
    handoff_path = normalize_workspace_root(handoff)
    db = str(state_db_path(handoff_path))
    runs = str(runs_root(handoff_path))
    wcs = str(skycell_wcs_csv())
    return handoff, data, ffi_dir, db, runs, wcs


def _build_runner_config(raw: dict, *, config_path: Path, base_dir: Path) -> RunnerConfig:
    warn_legacy_config_paths(raw, config_path=config_path)
    deployment_file = parse_deployment_file(raw)
    notifications = parse_notification_config(raw.get("notifications"))
    deployment_path = deployment_path_for_config(config_path, deployment_file)
    deployment = load_deployment(config_path, deployment_file)
    handoff, data, ffi_dir, db, runs, wcs = _paths_from_deployment(
        deployment, deployment_path=deployment_path
    )

    diff_site = str(
        raw.get("diff_config", "")
        or raw.get("diff_site_config", "")
        or raw.get("diff_config_path", "")
    ).strip()
    diff_config_path = ""
    if diff_site:
        diff_config_path = _resolve_path(base_dir, diff_site) or ""

    return RunnerConfig(
        deployment_file=deployment_file,
        data_root=data,
        ffi_dir=ffi_dir,
        workspace_root=handoff,
        runs_root=runs,
        state_db_path=db,
        skycell_wcs_csv=wcs,
        diff_config_path=diff_config_path,
        stages=parse_stage_params(raw.get("stages", {})),
        resources=_parse_resources(raw.get("resources")),
        overrides=dict(raw.get("overrides", {}) or {}),
        scheduler_heartbeat_interval_s=float(
            raw.get("scheduler", {}).get("heartbeat_interval_s", 30.0)
        ),
        verify_max_workers=int(raw.get("scheduler", {}).get("verify_max_workers", 1)),
        verify_budget_per_tick=int(
            raw.get("scheduler", {}).get("verify_budget_per_tick", 16)
        ),
        max_stage_attempts=int(raw.get("scheduler", {}).get("max_stage_attempts", 3)),
        requeue_backoff_s=float(raw.get("scheduler", {}).get("requeue_backoff_s", 30.0)),
        condor_hold_timeout_s=float(
            raw.get("scheduler", {}).get("condor_hold_timeout_s", 600.0)
        ),
        notifications=notifications,
    )


def load_runner_config(yaml_path: str | Path) -> RunnerConfig:
    path = Path(yaml_path).expanduser().resolve()
    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    if _is_materialized_config(raw):
        return load_and_materialize_runner_config(path)
    return _build_runner_config(raw, config_path=path, base_dir=path.parent)


def resolve_workspace_root(config_path: str | Path) -> Path:
    """Resolve handoff workspace from site deployment file."""
    cfg_path = Path(config_path).expanduser().resolve()
    with cfg_path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}
    deployment_file = parse_deployment_file(raw)
    deployment_path = deployment_path_for_config(cfg_path, deployment_file)
    deployment = load_deployment(cfg_path, deployment_file)
    handoff = require_deployment_path(
        deployment, "workspace_root", deployment_path=deployment_path
    )
    return normalize_workspace_root(handoff)


def _normalize_override_paths(overrides: Dict[str, dict], base_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for key, spec in (overrides or {}).items():
        spec = copy.deepcopy(spec or {})
        if spec.get("data_root"):
            spec["data_root"] = _resolve_path(base_dir, spec["data_root"])
        stages = spec.get("stages") or {}
        for stage_name, stage_cfg in stages.items():
            if not isinstance(stage_cfg, dict):
                continue
            for path_key in (
                "bkg_vector_path",
                "local_data_path",
                "catalog_path",
                "mapping_dir",
                "convolved_dir",
                "output_base",
            ):
                if stage_cfg.get(path_key):
                    stage_cfg[path_key] = _resolve_path(base_dir, stage_cfg[path_key])
        out[key] = spec
    return out


def runner_config_to_dict(cfg: RunnerConfig) -> dict:
    """Serialize RunnerConfig to a YAML-ready dict with absolute path fields."""
    data = asdict(cfg)
    data["stages"] = {
        "wcs_grouping": asdict(cfg.stages.wcs_grouping),
        "mapping": asdict(cfg.stages.mapping),
        "ps1_download": asdict(cfg.stages.ps1_download),
        "ps1_process": asdict(cfg.stages.ps1_process),
        "downsample": asdict(cfg.stages.downsample),
        "diff": asdict(cfg.stages.diff),
    }
    if cfg.diff_config_path:
        data["diff_config_path"] = cfg.diff_config_path
    data["resources"] = {name: asdict(pool) for name, pool in cfg.resources.items()}
    data["scheduler"] = {
        "heartbeat_interval_s": cfg.scheduler_heartbeat_interval_s,
        "verify_max_workers": cfg.verify_max_workers,
        "verify_budget_per_tick": cfg.verify_budget_per_tick,
        "max_stage_attempts": cfg.max_stage_attempts,
        "requeue_backoff_s": cfg.requeue_backoff_s,
        "condor_hold_timeout_s": cfg.condor_hold_timeout_s,
    }
    data.pop("scheduler_heartbeat_interval_s", None)
    data.pop("verify_max_workers", None)
    data.pop("verify_budget_per_tick", None)
    data.pop("max_stage_attempts", None)
    data.pop("requeue_backoff_s", None)
    data.pop("condor_hold_timeout_s", None)
    data["deployment_file"] = cfg.deployment_file
    data["notifications"] = {
        "enabled": cfg.notifications.enabled,
        "events": {
            "run_started": cfg.notifications.events.run_started,
            "run_completed": cfg.notifications.events.run_completed,
            "run_failed": cfg.notifications.events.run_failed,
            "run_canceled": cfg.notifications.events.run_canceled,
            "run_retried": cfg.notifications.events.run_retried,
            "run_stalled": cfg.notifications.events.run_stalled,
            "run_resumed": cfg.notifications.events.run_resumed,
            "stage_failed": cfg.notifications.events.stage_failed,
            "stage_completed": cfg.notifications.events.stage_completed,
            "stage_canceled": cfg.notifications.events.stage_canceled,
            "stage_died": cfg.notifications.events.stage_died,
            "daemon_unhealthy": cfg.notifications.events.daemon_unhealthy,
        },
        "bot": {
            "enabled": cfg.notifications.bot.enabled,
            "channel_id": cfg.notifications.bot.channel_id,
        },
    }
    return data


def write_runner_config(cfg: RunnerConfig, yaml_path: str | Path) -> None:
    path = Path(yaml_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runner_config_to_dict(cfg), fh, sort_keys=False, default_flow_style=False)


def load_and_materialize_runner_config(
    source_yaml: str | Path, base_dir: Path | None = None
) -> RunnerConfig:
    """Load config from *source_yaml* and return a RunnerConfig with absolute paths."""
    path = Path(source_yaml).expanduser().resolve()
    base = base_dir or path.parent
    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    if _is_materialized_config(raw):
        cfg = RunnerConfig(
            deployment_file=str(raw.get("deployment_file", "deployment.yaml")),
            data_root=_resolve_path(base, raw.get("data_root", "")) or "",
            ffi_dir=_resolve_path(base, raw.get("ffi_dir", "")) or "",
            workspace_root=_resolve_path(base, raw.get("workspace_root", "")) or "",
            runs_root=_resolve_path(base, raw.get("runs_root")) or "",
            state_db_path=_resolve_path(base, raw.get("state_db_path")) or "",
            skycell_wcs_csv=_resolve_path(base, raw.get("skycell_wcs_csv", "")) or "",
            diff_config_path=_resolve_path(
                base,
                raw.get("diff_config_path")
                or raw.get("diff_config")
                or raw.get("diff_site_config"),
            )
            or "",
            stages=parse_stage_params(raw.get("stages", {})),
            resources=_parse_resources(raw.get("resources")),
            overrides=_normalize_override_paths(dict(raw.get("overrides", {}) or {}), base),
            scheduler_heartbeat_interval_s=float(
                raw.get("scheduler", {}).get("heartbeat_interval_s", 30.0)
            ),
            verify_max_workers=int(raw.get("scheduler", {}).get("verify_max_workers", 1)),
            verify_budget_per_tick=int(
                raw.get("scheduler", {}).get("verify_budget_per_tick", 16)
            ),
            max_stage_attempts=int(raw.get("scheduler", {}).get("max_stage_attempts", 3)),
            requeue_backoff_s=float(raw.get("scheduler", {}).get("requeue_backoff_s", 30.0)),
            condor_hold_timeout_s=float(
                raw.get("scheduler", {}).get("condor_hold_timeout_s", 600.0)
            ),
            notifications=parse_notification_config(raw.get("notifications")),
        )
        if not cfg.ffi_dir and cfg.data_root:
            cfg.ffi_dir = str(Path(cfg.data_root) / "tess_ffi")
        if not cfg.state_db_path and cfg.workspace_root:
            cfg.state_db_path = str(state_db_path(cfg.workspace_root))
        if not cfg.runs_root and cfg.workspace_root:
            cfg.runs_root = str(runs_root(cfg.workspace_root))
        if not cfg.skycell_wcs_csv:
            cfg.skycell_wcs_csv = str(skycell_wcs_csv())
    else:
        cfg = _build_runner_config(raw, config_path=path, base_dir=base)

    _resolve_stage_path_fields(cfg, raw.get("stages", {}) or {}, base)
    return cfg


def _is_materialized_config(raw: dict) -> bool:
    """Frozen run configs embed resolved paths; site configs use deployment.yaml instead."""
    return bool(str(raw.get("workspace_root", "")).strip() and str(raw.get("data_root", "")).strip())


def _resolve_stage_path_fields(cfg: RunnerConfig, stages_raw: dict, base_dir: Path) -> None:
    path_keys_by_stage = {
        "wcs_grouping": ("bkg_vector_path",),
        "ps1_download": ("local_data_path",),
        "ps1_process": ("catalog_path",),
        "downsample": ("mapping_dir", "convolved_dir", "output_base"),
    }
    for stage_name, path_keys in path_keys_by_stage.items():
        stage_obj = getattr(cfg.stages, stage_name)
        stage_cfg = stages_raw.get(stage_name, {}) or {}
        for path_key in path_keys:
            val = stage_cfg.get(path_key)
            if val is None:
                val = getattr(stage_obj, path_key, None)
            if val:
                setattr(stage_obj, path_key, _resolve_path(base_dir, str(val)))


@dataclass
class ResolvedTargetConfig:
    target: Target
    data_root: str
    ffi_dir: str
    event_dir: str
    skycell_wcs_csv: str
    stages: TemplateStageParams
    mapping_root: str
    zarr_dir: str
    template_output_base: str
    config_path: str = ""


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def resolve_config(
    target: Target,
    cfg: RunnerConfig,
    *,
    config_path: str | Path | None = None,
) -> ResolvedTargetConfig:
    merged_stages_raw: dict = {
        "wcs_grouping": cfg.stages.wcs_grouping.__dict__,
        "mapping": cfg.stages.mapping.__dict__,
        "ps1_download": cfg.stages.ps1_download.__dict__,
        "ps1_process": cfg.stages.ps1_process.__dict__,
        "downsample": cfg.stages.downsample.__dict__,
    }
    override = cfg.overrides.get(target.scc_key()) or cfg.overrides.get(
        f"{target.sector}/{target.camera}/{target.ccd}"
    )
    if override:
        merged_stages_raw = _deep_merge_dict(merged_stages_raw, override.get("stages", {}))

    data_root = cfg.data_root
    if override and override.get("data_root"):
        data_root = str(Path(override["data_root"]).expanduser())

    event_dir = str(Path(cfg.workspace_root) / "events" / target.label())
    mapping_root = str(Path(data_root) / "skycell_pixel_mapping")
    zarr_dir = str(Path(data_root) / "ps1_skycells_zarr")
    template_output_base = str(Path(data_root) / "shifted_downsampled")

    return ResolvedTargetConfig(
        target=target,
        data_root=data_root,
        ffi_dir=cfg.ffi_dir,
        event_dir=event_dir,
        skycell_wcs_csv=cfg.skycell_wcs_csv,
        stages=parse_stage_params(merged_stages_raw),
        mapping_root=mapping_root,
        zarr_dir=zarr_dir,
        template_output_base=template_output_base,
        config_path=str(config_path) if config_path else "",
    )


def config_snapshot(resolved: ResolvedTargetConfig) -> Dict[str, Any]:
    t = resolved.target
    return {
        "sector": t.sector,
        "camera": t.camera,
        "ccd": t.ccd,
        "target_name": t.target_name,
        "target_ra": t.target_ra,
        "target_dec": t.target_dec,
        "event_dir": resolved.event_dir,
        "data_root": resolved.data_root,
    }
