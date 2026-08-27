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
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    DiffSitePolicy,
    parse_unified_diff_policy,
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
from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    ps1_skycells_zarr_dir,
    scc_convolved_zarr,
    scc_ffi_dir,
    scc_mapping_dir,
    scc_remap_dir,
    scc_templates_dir,
)
from syndiff_pipeline.common.orchestration.workspace import (
    normalize_workspace_root,
    runs_root as runs_root,
    state_db_path,
)

log = logging.getLogger(__name__)


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    """Resolve path.
    
    Parameters
    ----------
    base_dir : Path
    value : str | None
    
    Returns
    -------
    str | None"""
    if value is None or str(value).strip() == "":
        return None
    p = Path(os.path.expanduser(str(value)))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def parse_deployment_file(raw: dict) -> str:
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


@dataclass
class RunnerConfig:
    """RunnerConfig."""
    deployment_file: str = "deployment.yaml"
    data_root: str = ""
    ffi_dir: str = ""
    workspace_root: str = ""
    runs_root: str = ""
    state_db_path: str = ""
    skycell_wcs_csv: str = ""
    diff_config_path: str = ""
    # Unified (schema v2) diff policy, populated from the embedded ``diff:``
    # block when present. Mutually exclusive with diff_config_path -- exactly
    # one of the two is set when the site has a diff policy at all, both are
    # falsy/None for template-only or photometry-only configs. See
    # _resolve_diff_selection(). diff_config_path is unaffected and keeps
    # working exactly as before: 36 call sites still read it (later wave
    # migrates them to `diff`).
    diff: DiffSitePolicy | None = None
    star_config_path: str = ""
    photometry_config_path: str = ""
    stages: TemplateStageParams = field(default_factory=lambda: parse_stage_params({}))
    resources: Dict[str, ResourcePoolParams] = field(
        default_factory=lambda: _parse_resources({})
    )
    overrides: Dict[str, dict] = field(default_factory=dict)
    scheduler_heartbeat_interval_s: float = 30.0
    verify_max_workers: int = 1
    verify_budget_per_tick: int = 16
    skip_artifact_verify: bool = False
    bookkeeping_trust_index: bool = False
    max_stage_attempts: int = 3
    max_eviction_stage_attempts: int = 20
    requeue_backoff_s: float = 30.0
    condor_hold_timeout_s: float = 600.0
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    def runs_dir(self) -> str:
        """Runs dir.
        
        Returns
        -------
        str"""
        return self.runs_root or str(runs_root(self.workspace_root))

    def stage_executor(self, stage: str) -> str:
        """Return launch executor for a stage: 'local' or 'condor'."""
        from syndiff_pipeline.pipeline_spec import SYNDIFF_PIPELINE

        stage_spec = SYNDIFF_PIPELINE.get(stage)
        if stage_spec is None:
            return "local"
        return stage_spec.resolve_executor(self)


def _parse_resources(raw: dict | None) -> Dict[str, ResourcePoolParams]:
    """Parse resources.
    
    Parameters
    ----------
    raw : dict | None
    
    Returns
    -------
    Dict[str, ResourcePoolParams]"""
    raw = raw or {}
    out: Dict[str, ResourcePoolParams] = {}
    for name, spec in raw.items():
        spec = spec or {}
        out[name] = ResourcePoolParams(max_concurrent=int(spec.get("max_concurrent", 1)))
    if "network" not in out:
        out["network"] = ResourcePoolParams(max_concurrent=3)
    if "downsample" not in out:
        out["downsample"] = ResourcePoolParams(max_concurrent=2)
    if "remap" not in out:
        out["remap"] = ResourcePoolParams(max_concurrent=2)
    if "mapping" not in out:
        out["mapping"] = ResourcePoolParams(max_concurrent=6)
    if "ps1_process" not in out:
        out["ps1_process"] = ResourcePoolParams(max_concurrent=4)
    if "diff_prep" not in out:
        out["diff_prep"] = ResourcePoolParams(max_concurrent=2)
    if "background_estimate" not in out:
        # Conservative default -- this pool contends for the pool's scarce
        # big-RAM boxes; size it explicitly via resources.background_estimate
        # in the site config once the real free-big-RAM-box count is known.
        out["background_estimate"] = ResourcePoolParams(max_concurrent=2)
    if "diff" not in out:
        out["diff"] = ResourcePoolParams(max_concurrent=2)
    if "star" not in out:
        out["star"] = ResourcePoolParams(max_concurrent=4)
    if "photometry" not in out:
        out["photometry"] = ResourcePoolParams(max_concurrent=4)
    return out


def _paths_from_deployment(
    deployment: dict, *, deployment_path: Path
) -> tuple[str, str, str, str, str, str]:
    """Paths from deployment.
    
    Parameters
    ----------
    deployment : dict
    deployment_path : Path
    
    Returns
    -------
    tuple[str, str, str, str, str, str]"""
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


def _parse_bookkeeping_trust_index(raw: dict) -> bool:
    bookkeeping = raw.get("bookkeeping") or {}
    if isinstance(bookkeeping, dict) and "trust_index" in bookkeeping:
        return bool(bookkeeping.get("trust_index"))
    return bool(raw.get("bookkeeping_trust_index", False))


def _resolve_diff_selection(
    *,
    diff_config_pointer: str,
    raw_diff_block: Any,
    base_dir: Path,
    deployment_file: str,
) -> tuple[str, Optional["DiffSitePolicy"]]:
    """Resolve the ``diff_config:`` pointer *or* the unified ``diff:`` block.

    Both forms must coexist for the length of this wave (36 call sites still
    read ``RunnerConfig.diff_config_path``), but a single config must use at
    most one -- mixing them silently produces two policies and would be
    unreviewable. ``diff:`` is optional either way: template-only and
    photometry-only configs legitimately have neither.

    *diff_config_pointer* is the already-precedence-resolved
    ``diff_config``/``diff_site_config``/``diff_config_path`` string (each
    caller applies its own key-precedence order, unchanged from before this
    helper existed, so byte-identical v1 behaviour is preserved exactly).
    *raw_diff_block* is ``raw.get("diff")`` (``None`` when absent).

    Returns
    -------
    tuple[str, DiffSitePolicy | None]
        ``(diff_config_path, diff_policy)`` -- exactly one is
        truthy/non-``None`` when the site has a diff policy, both are
        falsy/``None`` when it has none.
    """
    diff_config_pointer = str(diff_config_pointer or "").strip()

    if diff_config_pointer and raw_diff_block is not None:
        raise ValueError(
            "config sets both a diff_config/diff_site_config pointer and a "
            "unified 'diff:' block -- use exactly one. Fold the pointed-to "
            "diff_config.yaml content under 'diff:' (schema v2) and drop the "
            "pointer key, or drop 'diff:' and keep the pointer (schema v1)."
        )

    diff_config_path = ""
    if diff_config_pointer:
        diff_config_path = _resolve_path(base_dir, diff_config_pointer) or ""

    diff_policy: Optional["DiffSitePolicy"] = None
    if raw_diff_block is not None:
        diff_policy = parse_unified_diff_policy(
            raw_diff_block, source_dir=base_dir, deployment_file=deployment_file
        )

    return diff_config_path, diff_policy


def _build_runner_config(raw: dict, *, config_path: Path, base_dir: Path) -> RunnerConfig:
    """Build runner config.
    
    Parameters
    ----------
    raw : dict
    config_path : Path
    base_dir : Path
    
    Returns
    -------
    RunnerConfig"""
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
    diff_config_path, diff_policy = _resolve_diff_selection(
        diff_config_pointer=diff_site,
        raw_diff_block=raw.get("diff"),
        base_dir=base_dir,
        deployment_file=deployment_file,
    )

    star_site = str(
        raw.get("star_config", "")
        or raw.get("star_site_config", "")
        or raw.get("star_config_path", "")
    ).strip()
    star_config_path = ""
    if star_site:
        star_config_path = _resolve_path(base_dir, star_site) or ""

    phot_site = str(
        raw.get("photometry_config", "")
        or raw.get("photometry_site_config", "")
        or raw.get("photometry_config_path", "")
    ).strip()
    photometry_config_path = ""
    if phot_site:
        photometry_config_path = _resolve_path(base_dir, phot_site) or ""

    return RunnerConfig(
        deployment_file=deployment_file,
        data_root=data,
        ffi_dir=ffi_dir,
        workspace_root=handoff,
        runs_root=runs,
        state_db_path=db,
        skycell_wcs_csv=wcs,
        diff_config_path=diff_config_path,
        diff=diff_policy,
        star_config_path=star_config_path,
        photometry_config_path=photometry_config_path,
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
        skip_artifact_verify=bool(
            raw.get("scheduler", {}).get("skip_artifact_verify", False)
        ),
        bookkeeping_trust_index=_parse_bookkeeping_trust_index(raw),
        max_stage_attempts=int(raw.get("scheduler", {}).get("max_stage_attempts", 3)),
        max_eviction_stage_attempts=int(
            raw.get("scheduler", {}).get("max_eviction_stage_attempts", 20)
        ),
        requeue_backoff_s=float(raw.get("scheduler", {}).get("requeue_backoff_s", 30.0)),
        condor_hold_timeout_s=float(
            raw.get("scheduler", {}).get("condor_hold_timeout_s", 600.0)
        ),
        notifications=notifications,
    )


def load_runner_config(yaml_path: str | Path) -> RunnerConfig:
    """Load runner config.
    
    Parameters
    ----------
    yaml_path : str | Path
    
    Returns
    -------
    RunnerConfig"""
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
    """Normalize override paths.
    
    Parameters
    ----------
    overrides : Dict[str, dict]
    base_dir : Path
    
    Returns
    -------
    Dict[str, dict]"""
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
        "remap": asdict(cfg.stages.remap),
        "downsample": asdict(cfg.stages.downsample),
        "diff": asdict(cfg.stages.diff),
        "star": asdict(cfg.stages.star),
        "photometry": asdict(cfg.stages.photometry),
    }
    if cfg.diff_config_path:
        data["diff_config_path"] = cfg.diff_config_path
    # Freeze the diff: block verbatim -- NOT a re-serialization of the parsed
    # DiffSitePolicy fields (asdict() above would otherwise normalize e.g. a
    # flat `condor:` into condor_by_stage and silently drop unrecognized keys
    # like mask_settings sub-keys nobody's added a field for yet). Verbatim
    # keeps _parse_condor_by_stage's all-three-keys validation as the single
    # gate, and makes hand-editing diff.condor in a frozen
    # runs/{run_id}/config.yaml behave exactly like editing the authored
    # file -- that hand-edit is the intended way to retune a live run's
    # Condor resources.
    if cfg.diff is not None:
        diff_out = copy.deepcopy(cfg.diff.raw)
        diff_out["source_dir"] = cfg.diff.source_dir
        data["diff"] = diff_out
    else:
        data.pop("diff", None)
    if cfg.star_config_path:
        data["star_config_path"] = cfg.star_config_path
    if cfg.photometry_config_path:
        data["photometry_config_path"] = cfg.photometry_config_path
    data["resources"] = {name: asdict(pool) for name, pool in cfg.resources.items()}
    data["bookkeeping"] = {"trust_index": cfg.bookkeeping_trust_index}
    data["scheduler"] = {
        "heartbeat_interval_s": cfg.scheduler_heartbeat_interval_s,
        "verify_max_workers": cfg.verify_max_workers,
        "verify_budget_per_tick": cfg.verify_budget_per_tick,
        "skip_artifact_verify": cfg.skip_artifact_verify,
        "max_stage_attempts": cfg.max_stage_attempts,
        "max_eviction_stage_attempts": cfg.max_eviction_stage_attempts,
        "requeue_backoff_s": cfg.requeue_backoff_s,
        "condor_hold_timeout_s": cfg.condor_hold_timeout_s,
    }
    data.pop("scheduler_heartbeat_interval_s", None)
    data.pop("verify_max_workers", None)
    data.pop("verify_budget_per_tick", None)
    data.pop("skip_artifact_verify", None)
    data.pop("max_stage_attempts", None)
    data.pop("max_eviction_stage_attempts", None)
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
    """Write runner config.
    
    Parameters
    ----------
    cfg : RunnerConfig
    yaml_path : str | Path"""
    path = Path(yaml_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(runner_config_to_dict(cfg), fh, sort_keys=False, default_flow_style=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_and_materialize_runner_config(
    source_yaml: str | Path, base_dir: Path | None = None
) -> RunnerConfig:
    """Load config from *source_yaml* and return a RunnerConfig with absolute paths."""
    path = Path(source_yaml).expanduser().resolve()
    base = base_dir or path.parent
    with path.open(encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    if _is_materialized_config(raw):
        _deployment_file_m = str(raw.get("deployment_file", "deployment.yaml"))
        _diff_site_m = str(
            raw.get("diff_config_path")
            or raw.get("diff_config")
            or raw.get("diff_site_config")
            or ""
        ).strip()
        _diff_config_path_m, _diff_policy_m = _resolve_diff_selection(
            diff_config_pointer=_diff_site_m,
            raw_diff_block=raw.get("diff"),
            base_dir=base,
            deployment_file=_deployment_file_m,
        )
        cfg = RunnerConfig(
            deployment_file=_deployment_file_m,
            data_root=_resolve_path(base, raw.get("data_root", "")) or "",
            ffi_dir=_resolve_path(base, raw.get("ffi_dir", "")) or "",
            workspace_root=_resolve_path(base, raw.get("workspace_root", "")) or "",
            runs_root=_resolve_path(base, raw.get("runs_root")) or "",
            state_db_path=_resolve_path(base, raw.get("state_db_path")) or "",
            skycell_wcs_csv=_resolve_path(base, raw.get("skycell_wcs_csv", "")) or "",
            diff_config_path=_diff_config_path_m,
            diff=_diff_policy_m,
            star_config_path=_resolve_path(
                base,
                raw.get("star_config_path")
                or raw.get("star_config")
                or raw.get("star_site_config"),
            )
            or "",
            photometry_config_path=_resolve_path(
                base,
                raw.get("photometry_config_path")
                or raw.get("photometry_config")
                or raw.get("photometry_site_config"),
            )
            or "",
            # Frozen configs may contain keys from newer feature branches;
            # drop unknowns so progress/status tools stay usable on main.
            stages=parse_stage_params(raw.get("stages", {}), strict=False),
            resources=_parse_resources(raw.get("resources")),
            overrides=_normalize_override_paths(dict(raw.get("overrides", {}) or {}), base),
            scheduler_heartbeat_interval_s=float(
                raw.get("scheduler", {}).get("heartbeat_interval_s", 30.0)
            ),
            verify_max_workers=int(raw.get("scheduler", {}).get("verify_max_workers", 1)),
            verify_budget_per_tick=int(
                raw.get("scheduler", {}).get("verify_budget_per_tick", 16)
            ),
            skip_artifact_verify=bool(
                raw.get("scheduler", {}).get("skip_artifact_verify", False)
            ),
            bookkeeping_trust_index=_parse_bookkeeping_trust_index(raw),
            max_stage_attempts=int(raw.get("scheduler", {}).get("max_stage_attempts", 3)),
            max_eviction_stage_attempts=int(
                raw.get("scheduler", {}).get("max_eviction_stage_attempts", 20)
            ),
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
    """Resolve stage path fields.
    
    Parameters
    ----------
    cfg : RunnerConfig
    stages_raw : dict
    base_dir : Path"""
    path_keys_by_stage = {
        "wcs_grouping": ("bkg_vector_path",),
        "mapping": ("bkg_vector_path",),
        "ps1_download": ("local_data_path",),
        "ps1_process": ("catalog_path",),
        "remap": (),
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
    """ResolvedTargetConfig."""
    target: Target
    data_root: str
    ffi_dir: str
    event_dir: str
    skycell_wcs_csv: str
    stages: TemplateStageParams
    mapping_root: str
    zarr_dir: str
    template_output_base: str
    remap_output_base: str = ""
    # Effective remap lane downsample reads (inherits remap.store_name when unset).
    downsample_remap_store_name: str | None = None
    config_path: str = ""


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


def resolve_config(
    target: Target,
    cfg: RunnerConfig,
    *,
    config_path: str | Path | None = None,
) -> ResolvedTargetConfig:
    """Resolve config.
    
    Parameters
    ----------
    target : Target
    cfg : RunnerConfig
    config_path : str | Path | None, optional, default ``None``
    
    Returns
    -------
    ResolvedTargetConfig"""
    merged_stages_raw: dict = {
        "wcs_grouping": cfg.stages.wcs_grouping.__dict__,
        "mapping": cfg.stages.mapping.__dict__,
        "ps1_download": cfg.stages.ps1_download.__dict__,
        "ps1_process": cfg.stages.ps1_process.__dict__,
        "remap": cfg.stages.remap.__dict__,
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

    t = target
    mapping_os = int(merged_stages_raw.get("mapping", {}).get("oversampling_factor", 1) or 1)
    templates_os = int(
        merged_stages_raw.get("downsample", {}).get("oversampling_factor", 1) or 1
    )
    stages = parse_stage_params(merged_stages_raw)
    # remap oversampling follows mapping (same as dispatch remap stage).
    remap_os = mapping_os
    remap_store_name = stages.remap.store_name
    # Downsample INPUT: explicit remap_store_name, else inherit remap.store_name.
    ds_remap_store = stages.downsample.remap_store_name
    if ds_remap_store is None:
        ds_remap_store = remap_store_name
    tmpl_store = stages.downsample.output_store_name

    event_dir = str(
        event_scc_leaf(cfg.workspace_root, target.event_name(), t.sector, t.camera, t.ccd)
    )
    ffi_dir = str(scc_ffi_dir(data_root, t.sector, t.camera, t.ccd))
    mapping_root = str(
        scc_mapping_dir(
            data_root,
            t.sector,
            t.camera,
            t.ccd,
            oversampling_factor=mapping_os,
            store_name=stages.mapping.store_name,
        )
    )
    zarr_dir = str(ps1_skycells_zarr_dir(data_root))
    template_output_base = str(
        scc_templates_dir(
            data_root,
            t.sector,
            t.camera,
            t.ccd,
            oversampling_factor=templates_os,
            store_name=tmpl_store,
        )
    )
    remap_output_base = str(
        scc_remap_dir(
            data_root,
            t.sector,
            t.camera,
            t.ccd,
            oversampling_factor=remap_os,
            store_name=remap_store_name,
        )
    )

    return ResolvedTargetConfig(
        target=target,
        data_root=data_root,
        ffi_dir=ffi_dir,
        event_dir=event_dir,
        skycell_wcs_csv=cfg.skycell_wcs_csv,
        stages=stages,
        mapping_root=mapping_root,
        zarr_dir=zarr_dir,
        template_output_base=template_output_base,
        remap_output_base=remap_output_base,
        downsample_remap_store_name=ds_remap_store,
        config_path=str(config_path) if config_path else "",
    )


def config_snapshot(resolved: ResolvedTargetConfig) -> Dict[str, Any]:
    """Config snapshot.
    
    Parameters
    ----------
    resolved : ResolvedTargetConfig
    
    Returns
    -------
    Dict[str, Any]"""
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
