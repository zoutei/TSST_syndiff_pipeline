"""YAML configuration for the template pipeline runner."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

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

    def runs_dir(self) -> str:
        return self.runs_root or str(Path(self.handoff_root) / "runs")


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
    if "cpu_heavy" not in out:
        out["cpu_heavy"] = ResourcePoolParams(max_concurrent=1)
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
        scheduler_heartbeat_interval_s=float(raw.get("scheduler", {}).get("heartbeat_interval_s", 30.0)),
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
