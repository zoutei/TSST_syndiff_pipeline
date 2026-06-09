"""YAML configuration for the template pipeline runner."""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from syndiff_pipeline.template_runner.notifications import (
    NotificationConfig,
    parse_notification_config,
)
from syndiff_pipeline.template_runner.stage_params import (
    ResourcePoolParams,
    TemplateStageParams,
    parse_stage_params,
)
from syndiff_pipeline.template_runner.targets import Target

log = logging.getLogger(__name__)


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    import os

    p = Path(os.path.expanduser(str(value)))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


@dataclass
class RunnerConfig:
    data_root: str = ""
    ffi_dir: str = ""
    handoff_root: str = ""
    runs_root: str = ""
    state_db_path: str = ""
    skycell_wcs_csv: str = ""
    gaia_credentials: str | None = None
    stages: TemplateStageParams = field(default_factory=lambda: parse_stage_params({}))
    resources: Dict[str, ResourcePoolParams] = field(default_factory=dict)
    overrides: Dict[str, dict] = field(default_factory=dict)
    scheduler_heartbeat_interval_s: float = 30.0
    verify_max_workers: int = 1
    verify_budget_per_tick: int = 16
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    def runs_dir(self) -> str:
        return self.runs_root or str(Path(self.handoff_root) / "runs")

    def stage_executor(self, stage: str) -> str:
        """Return launch executor for a stage: 'local' or 'condor'."""
        if stage == "ps1_process":
            return self.stages.ps1_process.executor
        if stage == "mapping":
            return self.stages.mapping.executor
        return "local"


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
    return out


def load_runner_config(yaml_path: str | Path) -> RunnerConfig:
    path = Path(yaml_path).expanduser().resolve()
    base_dir = path.parent
    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    data_root = _resolve_path(base_dir, raw.get("data_root", "")) or ""
    ffi_dir = _resolve_path(base_dir, raw.get("ffi_dir", "")) or data_root
    handoff_root = _resolve_path(base_dir, raw.get("handoff_root", "")) or ""
    runs_root = _resolve_path(base_dir, raw.get("runs_root")) or ""
    state_db = _resolve_path(base_dir, raw.get("state_db_path")) or ""

    cfg = RunnerConfig(
        data_root=data_root,
        ffi_dir=ffi_dir,
        handoff_root=handoff_root,
        runs_root=runs_root or "",
        state_db_path=state_db or "",
        skycell_wcs_csv=_resolve_path(base_dir, raw.get("skycell_wcs_csv", "")) or "",
        gaia_credentials=_resolve_path(base_dir, raw.get("gaia_credentials")),
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
        notifications=parse_notification_config(raw.get("notifications")),
    )
    if not cfg.data_root:
        raise ValueError("config.yaml requires data_root")
    if not cfg.handoff_root:
        raise ValueError("config.yaml requires handoff_root")
    if not cfg.skycell_wcs_csv:
        raise ValueError("config.yaml requires skycell_wcs_csv")
    if not cfg.state_db_path:
        cfg.state_db_path = str(Path(cfg.handoff_root) / "pipeline_state.sqlite")
    return cfg


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
    }
    data["resources"] = {name: asdict(pool) for name, pool in cfg.resources.items()}
    data["scheduler"] = {"heartbeat_interval_s": cfg.scheduler_heartbeat_interval_s}
    data.pop("scheduler_heartbeat_interval_s", None)
    data["notifications"] = {
        "enabled": cfg.notifications.enabled,
        "secrets_file": cfg.notifications.secrets_file,
        "events": {
            "run_completed": cfg.notifications.events.run_completed,
            "run_failed": cfg.notifications.events.run_failed,
            "run_canceled": cfg.notifications.events.run_canceled,
            "run_stalled": cfg.notifications.events.run_stalled,
            "run_resumed": cfg.notifications.events.run_resumed,
            "stage_failed": cfg.notifications.events.stage_failed,
            "stage_completed": cfg.notifications.events.stage_completed,
            "stage_canceled": cfg.notifications.events.stage_canceled,
            "stage_died": cfg.notifications.events.stage_died,
            "daemon_unhealthy": cfg.notifications.events.daemon_unhealthy,
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

    data_root = _resolve_path(base, raw.get("data_root", "")) or ""
    ffi_dir = _resolve_path(base, raw.get("ffi_dir", "")) or data_root
    handoff_root = _resolve_path(base, raw.get("handoff_root", "")) or ""
    runs_root = _resolve_path(base, raw.get("runs_root")) or ""
    state_db = _resolve_path(base, raw.get("state_db_path")) or ""

    cfg = RunnerConfig(
        data_root=data_root,
        ffi_dir=ffi_dir,
        handoff_root=handoff_root,
        runs_root=runs_root or "",
        state_db_path=state_db or "",
        skycell_wcs_csv=_resolve_path(base, raw.get("skycell_wcs_csv", "")) or "",
        gaia_credentials=_resolve_path(base, raw.get("gaia_credentials")),
        stages=parse_stage_params(raw.get("stages", {})),
        resources=_parse_resources(raw.get("resources")),
        overrides=_normalize_override_paths(dict(raw.get("overrides", {}) or {}), base),
        scheduler_heartbeat_interval_s=float(raw.get("scheduler", {}).get("heartbeat_interval_s", 30.0)),
        notifications=parse_notification_config(raw.get("notifications")),
    )
    if not cfg.data_root:
        raise ValueError("config.yaml requires data_root")
    if not cfg.handoff_root:
        raise ValueError("config.yaml requires handoff_root")
    if not cfg.skycell_wcs_csv:
        raise ValueError("config.yaml requires skycell_wcs_csv")
    if not cfg.state_db_path:
        cfg.state_db_path = str(Path(cfg.handoff_root) / "pipeline_state.sqlite")

    _resolve_stage_path_fields(cfg, raw.get("stages", {}) or {}, base)
    return cfg


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
    handoff_dir: str
    skycell_wcs_csv: str
    gaia_credentials: str | None
    stages: TemplateStageParams
    mapping_root: str
    zarr_dir: str
    template_output_base: str


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def resolve_config(target: Target, cfg: RunnerConfig) -> ResolvedTargetConfig:
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

    handoff_dir = str(Path(cfg.handoff_root) / target.label())
    mapping_root = str(Path(data_root) / "skycell_pixel_mapping")
    zarr_dir = str(Path(data_root) / "ps1_skycells_zarr")
    template_output_base = str(Path(data_root) / "shifted_downsampled")

    return ResolvedTargetConfig(
        target=target,
        data_root=data_root,
        ffi_dir=cfg.ffi_dir,
        handoff_dir=handoff_dir,
        skycell_wcs_csv=cfg.skycell_wcs_csv,
        gaia_credentials=cfg.gaia_credentials,
        stages=parse_stage_params(merged_stages_raw),
        mapping_root=mapping_root,
        zarr_dir=zarr_dir,
        template_output_base=template_output_base,
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
        "handoff_dir": resolved.handoff_dir,
        "data_root": resolved.data_root,
    }
