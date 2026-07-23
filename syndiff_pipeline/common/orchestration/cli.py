"""Shared orchestration CLI commands (monitoring, control, verify)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.deployment import (
    deployment_path_for_config,
    load_workspace_root_from_deployment,
)
from syndiff_pipeline.common.orchestration.workspace import (
    discover_alive_workspace_handoffs,
    discover_alive_workspace_roots,
    load_recorded_deployment_path,
    record_deployment_path,
    record_handoff_cache,
    resolve_handoff_fast,
    runs_root as runs_root,
    state_db_path,
)
from syndiff_pipeline.common.orchestration.scheduler_control import (
    daemon_is_alive,
    daemon_is_wedged,
    daemon_status,
    ensure_daemon_running,
    stop_daemon,
    warn_if_daemon_host_mismatch,
)
from syndiff_pipeline.common.orchestration.state import (
    STATUS_BLOCKED,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
    PipelineState,
)

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.run_context import RunContext
    from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths

log = logging.getLogger(__name__)

DIFF_STAGE = "diff"
PRESET_NAMES = frozenset({"template", "diff"})


def _template_stage_names() -> list[str]:
    """Template stage names.
    
    Returns
    -------
    list[str]"""
    from syndiff_pipeline.template_creation.orchestration.stages import TEMPLATE_STAGES

    return [spec.name for spec in TEMPLATE_STAGES]


def _stage_names() -> list[str]:
    """Full composed stage list (loads DIFF/STAR specs on first use)."""
    from syndiff_pipeline.pipeline_spec import stage_names

    return list(stage_names())


def preset_stages(preset: str) -> list[str]:
    """Return the stage list for a CLI execution preset."""
    if preset == "template":
        return _template_stage_names()
    if preset == "diff":
        return [DIFF_STAGE]
    raise ValueError(f"Unknown preset: {preset!r}")


def _resolve_single_stage(stage: str) -> str:
    """Resolve single stage.
    
    Parameters
    ----------
    stage : str
    
    Returns
    -------
    str"""
    from syndiff_pipeline.template_creation.orchestration import dispatch

    try:
        return dispatch.resolve_stage_name(stage)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _resolve_stages_arg(args: argparse.Namespace) -> tuple[list[str], str]:
    """Resolve stages arg.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    tuple[list[str], str]"""
    from syndiff_pipeline.template_creation.orchestration import dispatch

    if getattr(args, "stages", None):
        active = dispatch.parse_stage_list(args.stages)
        return active, args.stages
    preset = getattr(args, "preset", None)
    if preset:
        active = preset_stages(preset)
        return active, ",".join(active)
    active = dispatch.parse_stage_list(None)
    return active, ",".join(active)


def _discord_bot_config_path(args: argparse.Namespace, ctx: RunContext | None = None) -> Path | None:
    """Site config path used for Discord bot.enabled / channel overrides."""
    if getattr(args, "config", None):
        return Path(args.config).expanduser().resolve()
    if ctx is not None:
        source = (ctx.meta or {}).get("source_config_path")
        if source:
            return Path(source).expanduser().resolve()
        return logs.run_config_path(ctx.run_dir)
    return None


def _resolve_site_paths(args: argparse.Namespace) -> SitePaths | None:
    """Resolve site paths.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    SitePaths | None"""
    from syndiff_pipeline.difference_imaging.orchestration.site_config import SitePaths

    site = getattr(args, "site", None)
    if site:
        return SitePaths.from_site_dir(site)
    return None


def _resolve_config_from_site(args: argparse.Namespace) -> None:
    """Resolve config from site.
    
    Parameters
    ----------
    args : argparse.Namespace"""
    if getattr(args, "config", None):
        return
    site_paths = _resolve_site_paths(args)
    if site_paths is not None:
        args.config = str(site_paths.template_config)


def _resolve_deployment_from_site(args: argparse.Namespace) -> Path | None:
    """Resolve deployment from site.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    Path | None"""
    if getattr(args, "deployment", None):
        return None
    site_paths = _resolve_site_paths(args)
    if site_paths is None:
        return None
    deploy = site_paths.deployment
    if not deploy.is_file():
        raise SystemExit(f"Site deployment not found: {deploy}")
    path = deploy.resolve()
    handoff = load_workspace_root_from_deployment(path)
    record_deployment_path(handoff, path)
    return path


def _patch_local_diff_executor(run_directory: Path) -> None:
    """Patch local diff executor.
    
    Parameters
    ----------
    run_directory : Path"""
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
        write_runner_config,
    )

    cfg_path = logs.run_config_path(run_directory)
    cfg = load_runner_config(cfg_path)
    cfg.stages.diff.executor = "local"
    write_runner_config(cfg, cfg_path)


def _default_run_id() -> str:
    """Default run id.
    
    Returns
    -------
    str"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_run_id(cfg, run_id: str | None) -> str:
    """Resolve run id.
    
    Parameters
    ----------
    cfg
    run_id : str | None
    
    Returns
    -------
    str"""
    if run_id:
        return run_id
    latest = logs.runs_root(cfg.runs_dir()) / "latest"
    if latest.is_symlink():
        return latest.readlink().name
    runs = sorted(logs.runs_root(cfg.runs_dir()).glob("*"))
    if runs:
        return runs[-1].name
    raise SystemExit("No run_id specified and no runs found.")


def _resolve_latest_run_id_from_handoff(handoff: str | Path) -> str:
    """Resolve latest run id from handoff.
    
    Parameters
    ----------
    handoff : str | Path
    
    Returns
    -------
    str"""
    root = runs_root(handoff)
    latest = root / "latest"
    if latest.is_symlink():
        return latest.readlink().name
    runs = sorted(p for p in root.glob("*") if p.is_dir() and p.name != "latest")
    if runs:
        return runs[-1].name
    raise SystemExit("No runs found in workspace.")


def _resolve_handoff_from_args(args: argparse.Namespace) -> str:
    """Resolve handoff from args.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    str"""
    deployment = getattr(args, "deployment", None)
    if deployment:
        path = Path(deployment).expanduser().resolve()
        handoff = load_workspace_root_from_deployment(path)
        record_deployment_path(handoff, path)
        record_handoff_cache(handoff, path)
        return str(handoff)

    site_deploy = _resolve_deployment_from_site(args)
    if site_deploy is not None:
        handoff = load_workspace_root_from_deployment(site_deploy)
        record_deployment_path(handoff, site_deploy)
        record_handoff_cache(handoff, site_deploy)
        return str(handoff)

    fast = resolve_handoff_fast(require_daemon=True)
    if fast:
        return fast

    discovered = discover_alive_workspace_handoffs()
    if len(discovered) == 1:
        root, deploy = discovered[0]
        record_deployment_path(root, deploy)
        record_handoff_cache(root, deploy)
        return str(root)
    if len(discovered) > 1:
        lines = "\n".join(f"  {p}" for p, _ in discovered)
        raise SystemExit(f"Multiple supervisors running; pass --deployment:\n{lines}")
    raise SystemExit(
        "No supervisor found. Start with: syndiff template submit --site ... "
        "or syndiff daemon start --deployment ..."
    )


def _resolve_deployment_from_args(args: argparse.Namespace) -> Path:
    """Resolve deployment from args.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    Path"""
    deployment = getattr(args, "deployment", None)
    if deployment:
        path = Path(deployment).expanduser().resolve()
        handoff = load_workspace_root_from_deployment(path)
        record_deployment_path(handoff, path)
        return path

    site_deploy = _resolve_deployment_from_site(args)
    if site_deploy is not None:
        return site_deploy

    discovered = discover_alive_workspace_roots()
    if len(discovered) == 1:
        recorded = load_recorded_deployment_path(discovered[0])
        if recorded is not None:
            return recorded
        raise SystemExit(
            f"No deployment.yaml recorded for workspace {discovered[0]}. "
            "Pass --deployment PATH."
        )
    if len(discovered) > 1:
        lines = "\n".join(f"  {p}" for p in discovered)
        raise SystemExit(f"Multiple supervisors running; pass --deployment:\n{lines}")
    raise SystemExit(
        "No supervisor found. Start with: syndiff template submit --site ... "
        "or syndiff daemon start --deployment ..."
    )


def _resolve_run_from_args(args: argparse.Namespace) -> RunContext:
    """Resolve run from args.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    RunContext"""
    from syndiff_pipeline.common.orchestration.run_context import resolve_run_context

    if getattr(args, "run_dir", None):
        return resolve_run_context(
            run_dir=args.run_dir,
            run_id=getattr(args, "run_id", None),
        )

    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise SystemExit("Specify --run-dir, or --run-id with --deployment.")

    deployment = getattr(args, "deployment", None)
    if deployment:
        handoff = load_workspace_root_from_deployment(deployment)
        return resolve_run_context(
            run_id=run_id,
            runs_root=str(runs_root(handoff)),
        )

    handoff = _resolve_handoff_from_args(args)
    return resolve_run_context(run_id=run_id, runs_root=str(runs_root(handoff)))


def _resolve_run_control_from_args(args: argparse.Namespace):
    """Resolve run control context from args.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    RunControlContext"""
    from syndiff_pipeline.common.orchestration.run_context import resolve_run_control_context

    if getattr(args, "run_dir", None):
        return resolve_run_control_context(
            run_dir=args.run_dir,
            run_id=getattr(args, "run_id", None),
        )

    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise SystemExit("Specify --run-dir, or --run-id with --deployment.")

    deployment = getattr(args, "deployment", None)
    if deployment:
        handoff = load_workspace_root_from_deployment(deployment)
        return resolve_run_control_context(
            run_id=run_id,
            runs_root=str(runs_root(handoff)),
        )

    handoff = _resolve_handoff_from_args(args)
    return resolve_run_control_context(run_id=run_id, runs_root=str(runs_root(handoff)))


def _resolve_run_ids_for_monitoring(
    state: PipelineState,
    handoff: str,
    *,
    run_id: str | None = None,
) -> list[str]:
    """Resolve run ids for monitoring.
    
    Parameters
    ----------
    state : PipelineState
    handoff : str
    run_id : str | None, optional, default ``None``
    
    Returns
    -------
    list[str]"""
    if run_id:
        return [run_id]
    active = state.active_runs()
    if active:
        return [row["run_id"] for row in active]
    return [_resolve_latest_run_id_from_handoff(handoff)]


def _monitoring_mode(args: argparse.Namespace) -> bool:
    """Monitoring mode.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    bool"""
    return not getattr(args, "run_dir", None) and not getattr(args, "run_id", None)


def _resolve_template_scope(args: argparse.Namespace):
    """Resolve template SCC input to (optional source path, target list)."""
    from syndiff_pipeline.common.orchestration.targets import load_sccs, scc_from_cli

    has_inline = any(
        getattr(args, name, None) is not None for name in ("sector", "camera", "ccd")
    )
    if getattr(args, "scc", None) and has_inline:
        raise SystemExit("Use either --scc or --sector/--camera/--ccd, not both")
    if getattr(args, "scc", None):
        path = Path(args.scc).expanduser().resolve()
        return str(path), load_sccs(path)
    sector = getattr(args, "sector", None)
    camera = getattr(args, "camera", None)
    ccd = getattr(args, "ccd", None)
    if sector is not None and camera is not None and ccd is not None:
        return None, [scc_from_cli(sector, camera, ccd)]
    if has_inline:
        raise SystemExit("--sector, --camera, and --ccd must all be set together")
    raise SystemExit("Template runs require --scc PATH or --sector, --camera, and --ccd")


def _resolve_diff_scope(args: argparse.Namespace):
    """Resolve diff run targets from --targets, --scc, or inline SCC CLI."""
    from syndiff_pipeline.common.orchestration.targets import load_sccs, load_targets, scc_from_cli

    has_targets = bool(getattr(args, "targets", None))
    has_scc = bool(getattr(args, "scc", None))
    has_inline = any(
        getattr(args, name, None) is not None for name in ("sector", "camera", "ccd")
    )
    if has_targets and (has_scc or has_inline):
        raise SystemExit("Use --targets OR --scc/--sector/--camera/--ccd, not both")
    if has_targets:
        path = str(Path(args.targets).expanduser().resolve())
        return path, load_targets(path)
    if has_scc:
        path = Path(args.scc).expanduser().resolve()
        return str(path), load_sccs(path)
    sector = getattr(args, "sector", None)
    camera = getattr(args, "camera", None)
    ccd = getattr(args, "ccd", None)
    if sector is not None and camera is not None and ccd is not None:
        return None, [scc_from_cli(sector, camera, ccd)]
    if has_inline:
        raise SystemExit("--sector, --camera, and --ccd must all be set together")
    raise SystemExit(
        "Diff runs require --targets PATH or --scc PATH or --sector, --camera, and --ccd"
    )


def _resolve_execution_targets(args: argparse.Namespace):
    """Resolve targets for submit/run from preset-specific CLI args."""
    preset = getattr(args, "preset", None)
    if preset == "template":
        return _resolve_template_scope(args)
    if preset == "diff":
        return _resolve_diff_scope(args)
    if not getattr(args, "targets", None):
        raise SystemExit("Diff runs require --targets")
    path = str(Path(args.targets).expanduser().resolve())
    from syndiff_pipeline.common.orchestration.targets import load_targets

    return path, load_targets(path)


def _patch_skip_artifact_verify(config_path: str) -> None:
    """Set ``scheduler.skip_artifact_verify`` on a frozen run config."""
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
        write_runner_config,
    )

    rcfg = load_runner_config(config_path)
    if rcfg.skip_artifact_verify:
        return
    rcfg.skip_artifact_verify = True
    write_runner_config(rcfg, config_path)


def _prepare_run_directory(
    source_config: str,
    run_id: str,
    runs_root: str,
    *,
    stages: list[str],
    detach: bool,
    force_rerun: bool,
    source_targets: str | None = None,
    source_scc: str | None = None,
    inline_scc_targets: list | None = None,
    source_diff_config_path: str | None = None,
    source_star_config_path: str | None = None,
    source_photometry_config_path: str | None = None,
    workspace_run_id: str | None = None,
    skip_artifact_verify: bool = False,
) -> Path:
    """Prepare run directory.
    
    Parameters
    ----------
    source_config : str
    source_targets : str
    run_id : str
    runs_root : str
    stages : list[str]
    detach : bool
    force_rerun : bool
    source_diff_config_path : str | None, optional, default ``None``
    source_star_config_path : str | None, optional, default ``None``
    workspace_run_id : str | None, optional, default ``None``
    skip_artifact_verify : bool, optional, default ``False``
    
    Returns
    -------
    Path"""
    run_directory = logs.run_dir(runs_root, run_id)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "per_target").mkdir(exist_ok=True)

    if source_targets is not None:
        config_path, targets_path = logs.materialize_run_inputs(
            source_config,
            run_directory,
            source_targets=source_targets,
        )
        meta = {
            "run_id": run_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "source_config_path": str(Path(source_config).resolve()),
            "source_targets_path": str(Path(source_targets).resolve()),
            "config_path": config_path,
            "targets_path": targets_path,
            "stages": stages,
            "detach": detach,
            "force_rerun": force_rerun,
        }
    else:
        from syndiff_pipeline.common.orchestration.targets import (
            load_sccs,
            write_normalized_targets,
        )

        config_path, scc_path = logs.materialize_run_inputs(
            source_config,
            run_directory,
            source_scc=source_scc,
            inline_scc_targets=inline_scc_targets,
        )
        scc_targets = inline_scc_targets or load_sccs(scc_path)
        targets_path = str(logs.run_targets_path(run_directory))
        write_normalized_targets(targets_path, scc_targets)
        meta = {
            "run_id": run_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "source_config_path": str(Path(source_config).resolve()),
            "config_path": config_path,
            "scc_path": scc_path,
            "targets_path": targets_path,
            "stages": stages,
            "detach": detach,
            "force_rerun": force_rerun,
        }
        if source_scc:
            meta["source_scc_path"] = str(Path(source_scc).resolve())
    if skip_artifact_verify:
        _patch_skip_artifact_verify(config_path)
        meta["skip_artifact_verify"] = True
    if source_diff_config_path:
        meta["source_diff_config_path"] = str(Path(source_diff_config_path).resolve())
    if source_star_config_path:
        meta["source_star_config_path"] = str(Path(source_star_config_path).resolve())
        star_dest = run_directory / "star_config.yaml"
        if not star_dest.is_file():
            shutil.copy2(source_star_config_path, star_dest)
        frozen_star = str(star_dest.resolve())
        meta["star_config_path"] = frozen_star
        from syndiff_pipeline.template_creation.orchestration.runner_config import (
            load_runner_config,
            write_runner_config,
        )

        rcfg = load_runner_config(config_path)
        rcfg.star_config_path = frozen_star
        write_runner_config(rcfg, config_path)

    if source_photometry_config_path:
        meta["source_photometry_config_path"] = str(
            Path(source_photometry_config_path).resolve()
        )
        phot_dest = run_directory / "photometry_config.yaml"
        if not phot_dest.is_file():
            shutil.copy2(source_photometry_config_path, phot_dest)
        frozen_phot = str(phot_dest.resolve())
        meta["photometry_config_path"] = frozen_phot
        from syndiff_pipeline.template_creation.orchestration.runner_config import (
            load_runner_config,
            write_runner_config,
        )

        rcfg = load_runner_config(config_path)
        rcfg.photometry_config_path = frozen_phot
        write_runner_config(rcfg, config_path)

    # Freeze site mask_settings.yaml when present (star_config pattern).
    site_mask = None
    if source_diff_config_path:
        cand = Path(source_diff_config_path).expanduser().resolve().parent / "mask_settings.yaml"
        if cand.is_file():
            site_mask = cand
    if site_mask is not None:
        mask_dest = run_directory / "mask_settings.yaml"
        if not mask_dest.is_file():
            shutil.copy2(site_mask, mask_dest)
        meta["mask_settings_path"] = str(mask_dest.resolve())
    if workspace_run_id is not None and str(workspace_run_id).strip():
        meta["workspace_run_id"] = str(workspace_run_id).strip()
    logs.ensure_run_layout(runs_root, run_id, meta)
    logs.update_run_meta(runs_root, run_id, meta)
    return run_directory


def _run_context_from_directory(run_directory: Path, run_id: str) -> RunContext:
    """Run context from directory.
    
    Parameters
    ----------
    run_directory : Path
    run_id : str
    
    Returns
    -------
    RunContext"""
    from syndiff_pipeline.common.orchestration.run_context import resolve_run_context

    return resolve_run_context(run_dir=run_directory, run_id=run_id)


def _reject_duplicate_run_id(state: PipelineState, run_id: str) -> None:
    """Reject duplicate run id.
    
    Parameters
    ----------
    state : PipelineState
    run_id : str"""
    if state.get_run(run_id) is not None:
        raise SystemExit(
            f"Run {run_id!r} already exists. Choose a new --run-id for submit, "
            f"or retry failed stages with: syndiff retry --run-id {run_id}"
        )


def cmd_submit(args: argparse.Namespace) -> int:
    """Cmd submit.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup
    from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config

    cfg = load_runner_config(args.config)
    source_input_path, targets = _resolve_execution_targets(args)
    active, _stages_arg = _resolve_stages_arg(args)
    run_id = args.run_id or _default_run_id()
    runs_root = cfg.runs_dir()

    state = PipelineState(cfg.state_db_path)
    _reject_duplicate_run_id(state, run_id)

    preset = getattr(args, "preset", None)
    if preset == "template":
        run_directory = _prepare_run_directory(
            args.config,
            run_id,
            runs_root,
            stages=active,
            detach=True,
            force_rerun=bool(args.force_rerun),
            source_scc=source_input_path,
            inline_scc_targets=targets if source_input_path is None else None,
            source_diff_config_path=cfg.diff_config_path or None,
            workspace_run_id=getattr(args, "workspace_run_id", None),
            skip_artifact_verify=bool(getattr(args, "skip_artifact_verify", False)),
        )
    else:
        if getattr(args, "targets", None):
            run_directory = _prepare_run_directory(
                args.config,
                run_id,
                runs_root,
                stages=active,
                detach=True,
                force_rerun=bool(args.force_rerun),
                source_targets=source_input_path,
                source_diff_config_path=cfg.diff_config_path or None,
                workspace_run_id=getattr(args, "workspace_run_id", None),
                skip_artifact_verify=bool(getattr(args, "skip_artifact_verify", False)),
            )
        else:
            run_directory = _prepare_run_directory(
                args.config,
                run_id,
                runs_root,
                stages=active,
                detach=True,
                force_rerun=bool(args.force_rerun),
                source_scc=source_input_path,
                inline_scc_targets=targets if source_input_path is None else None,
                source_diff_config_path=cfg.diff_config_path or None,
                workspace_run_id=getattr(args, "workspace_run_id", None),
                skip_artifact_verify=bool(getattr(args, "skip_artifact_verify", False)),
            )

    if getattr(args, "local", False) and preset == "diff":
        _patch_local_diff_executor(run_directory)

    state.create_run(
        run_id,
        str(logs.run_config_path(run_directory)),
        str(logs.run_targets_path(run_directory)),
        runs_root,
        targets,
        active,
        force_rerun=bool(args.force_rerun),
    )
    setup = apply_post_create_run_setup(state, run_id, targets, cfg, active)
    if cfg.stages.ps1_process.ps1_source == "stream" and "ps1_download" in active:
        print(
            "Note: ps1_download ignored for this run (ps1_source=stream); "
            "download happens inside ps1_process."
        )
    if setup.stream_skipped:
        print(f"Marked ps1_download n/a (stream_mode) for {setup.stream_skipped} target(s).")
    if setup.linear_remap_skipped:
        print(
            f"Marked remap n/a (linear_geometry) for {setup.linear_remap_skipped} target(s)."
        )
    if setup.not_selected:
        print(
            f"Marked {setup.not_selected} stage row(s) n/a "
            "(not selected for this run)."
        )
    if setup.superseded:
        print(
            f"Marked {setup.superseded} stage row(s) n/a "
            "(upstream artifacts already satisfied downstream)."
        )

    deploy_path = deployment_path_for_config(args.config, cfg.deployment_file)
    record_deployment_path(cfg.workspace_root, deploy_path)
    from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
        record_discord_bot_site_config,
    )

    record_discord_bot_site_config(cfg.workspace_root, args.config)
    result = ensure_daemon_running(cfg.workspace_root, deployment_path=deploy_path)
    warn_if_daemon_host_mismatch(cfg.workspace_root)
    daemon_log = logs.daemon_log_path(cfg.workspace_root)

    if cfg.notifications.enabled:
        from syndiff_pipeline.common.orchestration.notifications import send_run_started_notification

        send_run_started_notification(
            state,
            cfg.notifications,
            config_path=args.config,
            run_id=run_id,
            run_dir=run_directory,
            target_labels=[t.label() for t in targets if t.enabled],
            stages=active,
            workspace_root=cfg.workspace_root,
            deployment_file=cfg.deployment_file,
            force_rerun=bool(args.force_rerun),
        )

    print(f"Submitted run_id={run_id} supervisor_pid={result.pid}")
    print(f"  daemon log: {daemon_log}")
    print("Monitor: syndiff progress")
    print("         syndiff status --watch")
    print(f"         syndiff progress --run-id {run_id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Cmd run.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.common.orchestration.scheduler import run_scheduler
    from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config

    if sys.stdout.isatty():
        print("Warning: foreground run blocks until complete; use 'submit' for detached runs.")
    run_id = args.run_id or _default_run_id()
    cfg = load_runner_config(args.config)
    source_input_path, targets = _resolve_execution_targets(args)
    active, stages_arg = _resolve_stages_arg(args)
    runs_root = cfg.runs_dir()

    state = PipelineState(cfg.state_db_path)
    _reject_duplicate_run_id(state, run_id)

    preset = getattr(args, "preset", None)
    if preset == "template":
        run_directory = _prepare_run_directory(
            args.config,
            run_id,
            runs_root,
            stages=active,
            detach=False,
            force_rerun=bool(args.force_rerun),
            source_scc=source_input_path,
            inline_scc_targets=targets if source_input_path is None else None,
            source_diff_config_path=cfg.diff_config_path or None,
            workspace_run_id=getattr(args, "workspace_run_id", None),
            skip_artifact_verify=bool(getattr(args, "skip_artifact_verify", False)),
        )
    else:
        if getattr(args, "targets", None):
            run_directory = _prepare_run_directory(
                args.config,
                run_id,
                runs_root,
                stages=active,
                detach=False,
                force_rerun=bool(args.force_rerun),
                source_targets=source_input_path,
                source_diff_config_path=cfg.diff_config_path or None,
                workspace_run_id=getattr(args, "workspace_run_id", None),
                skip_artifact_verify=bool(getattr(args, "skip_artifact_verify", False)),
            )
        else:
            run_directory = _prepare_run_directory(
                args.config,
                run_id,
                runs_root,
                stages=active,
                detach=False,
                force_rerun=bool(args.force_rerun),
                source_scc=source_input_path,
                inline_scc_targets=targets if source_input_path is None else None,
                source_diff_config_path=cfg.diff_config_path or None,
                workspace_run_id=getattr(args, "workspace_run_id", None),
                skip_artifact_verify=bool(getattr(args, "skip_artifact_verify", False)),
            )

    return run_scheduler(
        run_id,
        str(run_directory),
        stages_arg,
        force_rerun=bool(args.force_rerun),
    )


def _print_status_for_run(
    state: PipelineState,
    *,
    run_id: str,
    workspace_root: str,
    multi_run: bool,
) -> None:
    """Print status for run.
    
    Parameters
    ----------
    state : PipelineState
    run_id : str
    workspace_root : str
    multi_run : bool"""
    from syndiff_pipeline.template_creation.orchestration.run_report import format_status_grid
    from syndiff_pipeline.common.orchestration.verify_status import read_verify_run_status

    if multi_run:
        print(f"=== run {run_id} ===")
    run = state.get_run(run_id) or {}
    print(f"Run {run_id} status={run.get('status', '?')}")
    scan_status = read_verify_run_status(workspace_root, run_id)
    scan_queued = int(scan_status.get("scan_queued", 0))
    scan_running = int(scan_status.get("scan_running", 0))
    verify_backlog = scan_queued + scan_running
    if scan_queued:
        print(f"  scan_queued={scan_queued}")
    if scan_running:
        print(f"  scan_running={scan_running}")
    if run.get("status") == "stalled" and run.get("stall_reason") and verify_backlog == 0:
        print(f"  stalled: {run['stall_reason']}")
    for line in format_status_grid(state, run_id, workspace_root=workspace_root):
        print(line)


def cmd_status(args: argparse.Namespace) -> int:
    """Cmd status.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    if _monitoring_mode(args):
        handoff = _resolve_handoff_from_args(args)
        warn_if_daemon_host_mismatch(handoff)
        state = PipelineState(str(state_db_path(handoff)))
        run_ids = _resolve_run_ids_for_monitoring(
            state, handoff, run_id=getattr(args, "run_id", None)
        )

        def _print_once():
            """Print once."""
            multi = len(run_ids) > 1
            for run_id in run_ids:
                _print_status_for_run(
                    state,
                    run_id=run_id,
                    workspace_root=handoff,
                    multi_run=multi,
                )

        if args.watch:
            while True:
                print("\033[2J\033[H", end="")
                _print_once()
                time.sleep(args.interval)
        else:
            _print_once()
            if not daemon_is_alive(handoff):
                print(
                    "WARNING: supervisor daemon is not alive. "
                    "Start with: syndiff template submit --site ... "
                    "or syndiff daemon start --deployment ..."
                )
        return 0

    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    _print_status_for_run(
        state,
        run_id=ctx.run_id,
        workspace_root=ctx.cfg.workspace_root,
        multi_run=False,
    )
    if not args.watch:
        if not daemon_is_alive(ctx.cfg.workspace_root):
            source = (ctx.meta or {}).get("source_config_path")
            hint = source or logs.run_config_path(ctx.run_dir)
            print(
                "WARNING: supervisor daemon is not alive. "
                f"Start with: syndiff daemon start --deployment ... "
                f"(site config: {hint})"
            )
        return 0

    while True:
        print("\033[2J\033[H", end="")
        _print_status_for_run(
            state,
            run_id=ctx.run_id,
            workspace_root=ctx.cfg.workspace_root,
            multi_run=False,
        )
        time.sleep(args.interval)


def cmd_progress(args: argparse.Namespace) -> int:
    """Cmd progress.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.template_creation.orchestration.run_report import (
        format_progress_lines,
        format_run_status_header,
    )

    if _monitoring_mode(args):
        handoff = _resolve_handoff_from_args(args)
        state = PipelineState(str(state_db_path(handoff)))
        run_ids = _resolve_run_ids_for_monitoring(
            state, handoff, run_id=getattr(args, "run_id", None)
        )
        multi = len(run_ids) > 1
        for run_id in run_ids:
            run = state.get_run(run_id) or {}
            if multi:
                print()
            print(format_run_status_header(run_id, run))
            runs_root = run.get("runs_root") or str(runs_root(handoff))
            for line in format_progress_lines(
                state,
                run_id,
                runs_root,
                workspace_root=handoff,
                include_running_detail=not getattr(args, "no_detail", False),
            ):
                print(line)
        return 0

    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    run = state.get_run(ctx.run_id) or {}
    print(format_run_status_header(ctx.run_id, run))
    for line in format_progress_lines(
        state,
        ctx.run_id,
        ctx.cfg.runs_dir(),
        workspace_root=ctx.cfg.workspace_root,
        include_running_detail=not getattr(args, "no_detail", False),
    ):
        print(line)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    """Cmd runs.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    handoff = _resolve_handoff_from_args(args)
    state = PipelineState(str(state_db_path(handoff)))
    alive = daemon_is_alive(handoff)
    for r in state.list_runs(args.limit):
        print(
            f"{r['run_id']}  status={r.get('status')}  "
            f"started={r.get('started_at')}  daemon_alive={alive}"
        )
    return 0


def cmd_notify_test(args: argparse.Namespace) -> int:
    """Cmd notify test.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    from syndiff_pipeline.common.orchestration.notifications import (
        format_preview_message,
        send_preview_notification,
    )

    if getattr(args, "dry_run", False):
        print(
            format_preview_message(
                state,
                ctx.run_id,
                ctx.cfg.runs_dir(),
                workspace_root=ctx.cfg.workspace_root,
            )
        )
        return 0

    message = send_preview_notification(state, ctx)
    print(f"Sent test notification to Discord for run {ctx.run_id}.")
    if getattr(args, "verbose", False):
        print(message)
    return 0


def cmd_active(args: argparse.Namespace) -> int:
    """Cmd active.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    handoff = _resolve_handoff_from_args(args)
    state = PipelineState(str(state_db_path(handoff)))
    found = False
    for r in state.list_runs(50):
        if r.get("status") in ("running", "stalled"):
            print(f"{r['run_id']}  status={r.get('status')}")
            found = True
    if not found:
        print("No active runs.")
    if daemon_is_alive(handoff):
        st = daemon_status(handoff)
        print(f"Supervisor pid={st.pid} heartbeat_age_s={st.heartbeat_age_s}")
    else:
        print("Supervisor daemon is not alive.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Cmd show.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    ctx = _resolve_run_from_args(args)
    meta_path = logs.run_meta_path(ctx.run_dir)
    if meta_path.is_file():
        print(meta_path.read_text(encoding="utf-8"))
    else:
        print(f"No run_meta.json for {ctx.run_id}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    """Cmd logs.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    ctx = _resolve_run_from_args(args)
    if args.target and args.stage:
        stage = _resolve_single_stage(args.stage)
        path = logs.target_log_path(ctx.cfg.runs_dir(), ctx.run_id, args.target, stage)
    else:
        path = logs.daemon_log_path(ctx.cfg.workspace_root)
    if not path.is_file():
        print(f"Log not found: {path}")
        return 1
    if args.follow:
        import subprocess

        return subprocess.call(["tail", "-f", str(path)])
    print(path.read_text(encoding="utf-8", errors="replace"))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Cmd verify.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.common.orchestration.targets import find_target, load_targets
    from syndiff_pipeline.template_creation.orchestration import dispatch
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
        resolve_config,
    )
    from syndiff_pipeline.template_creation.orchestration.verify import persist_completion_manifests, verify_stage

    _resolve_config_from_site(args)
    run_id: str | None = None
    meta: dict | None = None
    if args.run_dir:
        ctx = _resolve_run_from_args(args)
        cfg = ctx.cfg
        targets = ctx.targets
        run_id = ctx.run_id
        meta = ctx.meta
    else:
        if not args.config:
            raise SystemExit("Specify --run-dir, --config, or --site for verify.")
        cfg = load_runner_config(args.config)
        if not args.targets:
            raise SystemExit("--targets required for pre-run verify.")
        targets = load_targets(args.targets)
        if args.run_id:
            run_id = args.run_id

    if args.scc:
        t = find_target(targets, args.scc)
        targets = [t]
    active = dispatch.parse_stage_list(args.stages) if args.stages else _stage_names()
    runs_root = cfg.runs_dir()
    rc = 0
    for target in targets:
        label = target.label()
        resolved = resolve_config(target, cfg)
        for stage in active:
            result = verify_stage(resolved, stage, runner_cfg=cfg, meta=meta)
            mark = "OK" if result.ok else ("UNKNOWN" if result.unknown else "FAIL")
            print(f"[{mark}] {label}/{stage}: {result.message} ({result.path})")
            if result.ok:
                manifest_paths = [logs.stable_stage_manifest_path(runs_root, label, stage)]
                if run_id:
                    manifest_paths.insert(
                        0,
                        logs.stage_manifest_path(runs_root, run_id, label, stage),
                    )
                try:
                    written = persist_completion_manifests(
                        resolved, stage, manifest_paths, runner_cfg=cfg, meta=meta
                    )
                    print(f"[MANIFEST] {label}/{stage} -> {', '.join(written)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] {label}/{stage}: manifest write failed: {exc}")
            elif not result.unknown:
                rc = 1
    return rc


def cmd_reconcile_manifests(args: argparse.Namespace) -> int:
    """Backfill cross-run completion manifests for already-complete targets.

    Scans existing outputs read-only (via the fast on-disk verifiers) and writes
    a stable manifest for every stage that is already complete. Future runs then
    skip the on-disk scan entirely for those stages.
    """
    from syndiff_pipeline.common.orchestration.targets import find_target, load_targets
    from syndiff_pipeline.template_creation.orchestration import dispatch
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
        resolve_config,
    )
    from syndiff_pipeline.template_creation.orchestration.verify import (
        collect_stage_artifacts,
        stage_complete,
        write_manifest,
    )

    _resolve_config_from_site(args)
    meta: dict | None = None
    if args.run_dir:
        ctx = _resolve_run_from_args(args)
        cfg = ctx.cfg
        targets = ctx.targets
        meta = ctx.meta
    else:
        if not args.config:
            raise SystemExit("Specify --run-dir or --config.")
        cfg = load_runner_config(args.config)
        if not args.targets:
            raise SystemExit("--targets required for reconcile-manifests.")
        targets = load_targets(args.targets)

    if args.scc:
        targets = [find_target(targets, args.scc)]
    active = dispatch.parse_stage_list(args.stages) if args.stages else _stage_names()
    runs_root = cfg.runs_dir()

    written = 0
    skipped = 0
    for target in targets:
        label = target.label()
        resolved = resolve_config(target, cfg)
        for stage in active:
            stable_path = str(logs.stable_stage_manifest_path(runs_root, label, stage))
            try:
                complete = stage_complete(
                    resolved, stage, runner_cfg=cfg, meta=meta
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR]   {label}/{stage}: {exc}")
                continue
            if not complete:
                skipped += 1
                if not args.quiet:
                    print(f"[SKIP]  {label}/{stage} not complete")
                continue
            try:
                expected, produced, artifacts = collect_stage_artifacts(
                    resolved, stage, runner_cfg=cfg, meta=meta
                )
                write_manifest(
                    stable_path,
                    resolved,
                    stage,
                    artifacts,
                    expected,
                    produced,
                    runner_cfg=cfg,
                    meta=meta,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR]   {label}/{stage}: manifest write failed: {exc}")
                continue
            written += 1
            print(f"[WROTE] {label}/{stage} ({produced}/{expected}) -> {stable_path}")

    print(f"reconcile-manifests: wrote {written} manifest(s), {skipped} stage(s) not complete")
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    """Cmd retry.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.common.orchestration.targets import find_target_for_run

    ctx = _resolve_run_from_args(args)
    warn_if_daemon_host_mismatch(ctx.cfg.workspace_root)
    state = PipelineState(ctx.cfg.state_db_path)

    if args.scc and args.stage:
        t = find_target_for_run(ctx, state, args.scc)
        stage = _resolve_single_stage(args.stage)
        row = state.get_stage_run(ctx.run_id, t.label(), stage)
        if row is None:
            raise SystemExit(
                f"No stage row for {t.label()} / {stage} in run {ctx.run_id!r}."
            )
        state.insert_command(
            "retry",
            run_id=ctx.run_id,
            args={
                "target_label": t.label(),
                "stage": stage,
                "reset_downstream": not getattr(args, "no_reset_downstream", False),
            },
        )
        print(f"Queued retry for {stage} on {t.label()} in run {ctx.run_id}")
    elif args.scc or args.stage:
        raise SystemExit(
            "Specify both --scc and --stage for a single retry, "
            "or omit both to retry all failed/canceled stages."
        )
    else:
        state.insert_command("retry", run_id=ctx.run_id)
        print(f"Queued bulk retry for run {ctx.run_id}")

    if not getattr(args, "no_start_daemon", False):
        from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
            record_discord_bot_site_config,
        )

        deploy_path = load_recorded_deployment_path(ctx.cfg.workspace_root)
        ensure_daemon_running(
            ctx.cfg.workspace_root,
            deployment_path=deploy_path,
        )
        bot_config = _discord_bot_config_path(args, ctx)
        if bot_config is not None:
            record_discord_bot_site_config(ctx.cfg.workspace_root, bot_config)
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    """Cmd launch.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.common.orchestration.targets import find_target_for_run

    if not args.scc or not args.stage:
        raise SystemExit("Specify both --target (or --scc) and --stage.")

    ctx = _resolve_run_from_args(args)
    warn_if_daemon_host_mismatch(ctx.cfg.workspace_root)
    state = PipelineState(ctx.cfg.state_db_path)
    t = find_target_for_run(ctx, state, args.scc)
    target_label = t.label()
    stage = _resolve_single_stage(args.stage)

    row = state.get_stage_run(ctx.run_id, target_label, stage)
    if row is None:
        raise SystemExit(f"No stage row for {target_label} / {stage} in run {ctx.run_id!r}.")
    if row.status == STATUS_RUNNING:
        raise SystemExit(f"{target_label} / {stage} is already running.")
    if row.status in TERMINAL_STATUSES:
        raise SystemExit(f"{target_label} / {stage} is terminal ({row.status}).")
    if row.status not in (STATUS_READY, STATUS_PENDING, STATUS_BLOCKED):
        print(
            f"Note: {target_label} / {stage} is {row.status}; "
            "force launch will run once the stage is ready."
        )

    state.insert_command(
        "force_launch",
        run_id=ctx.run_id,
        args={"target_label": target_label, "stage": stage},
    )
    print(
        f"Queued force launch for {stage} on {target_label} in run {ctx.run_id} "
        "(bypasses pool max_concurrent)"
    )

    if not getattr(args, "no_start_daemon", False):
        from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
            record_discord_bot_site_config,
        )

        deploy_path = load_recorded_deployment_path(ctx.cfg.workspace_root)
        ensure_daemon_running(
            ctx.cfg.workspace_root,
            deployment_path=deploy_path,
        )
        bot_config = _discord_bot_config_path(args, ctx)
        if bot_config is not None:
            record_discord_bot_site_config(ctx.cfg.workspace_root, bot_config)
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    """Cmd pause.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    ctx = _resolve_run_from_args(args)
    warn_if_daemon_host_mismatch(ctx.cfg.workspace_root)
    PipelineState(ctx.cfg.state_db_path).insert_command("pause", run_id=ctx.run_id)
    print(f"Queued pause for run {ctx.run_id}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Cmd resume.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    ctx = _resolve_run_from_args(args)
    warn_if_daemon_host_mismatch(ctx.cfg.workspace_root)
    state = PipelineState(ctx.cfg.state_db_path)
    state.insert_command("resume", run_id=ctx.run_id)
    print(f"Queued resume for run {ctx.run_id}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """Cmd kill.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    ctx = _resolve_run_control_from_args(args)
    warn_if_daemon_host_mismatch(ctx.cfg.workspace_root)
    state = PipelineState(ctx.cfg.state_db_path)
    state.insert_command("cancel", run_id=ctx.run_id)
    from syndiff_pipeline.common.orchestration import condor

    condor.sweep_run_condor_audit_clusters(ctx.cfg.runs_dir(), ctx.run_id)
    print(f"Queued cancel for run {ctx.run_id}")
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Cmd daemon.
    
    Parameters
    ----------
    args : argparse.Namespace
    
    Returns
    -------
    int"""
    from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
        discord_bot_status_for_handoff,
    )

    handoff = _resolve_handoff_from_args(args)
    deploy_arg = getattr(args, "deployment", None)
    deploy_path = (
        Path(deploy_arg).expanduser().resolve()
        if deploy_arg
        else load_recorded_deployment_path(handoff)
    )
    if args.action == "start":
        try:
            result = ensure_daemon_running(handoff, deployment_path=deploy_path)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        host_text = f" host={result.host!r}" if result.host else ""
        print(f"Supervisor pid={result.pid} spawned={result.spawned}{host_text}")
        if result.spawned:
            print("Discord bot: starts in-process with supervisor when enabled (see daemon.log)")
        return 0
    if args.action == "stop":
        result = stop_daemon(handoff)
        if result.message and not result.stopped:
            print(f"ERROR: {result.message}", file=sys.stderr)
            return 1
        if not result.was_running:
            lock_text = " Stale daemon lock released." if result.lock_reclaimed else ""
            print(f"Supervisor was not running.{lock_text}")
            return 0
        if result.stopped:
            if result.force_killed:
                print(f"Supervisor pid={result.pid} stopped (SIGKILL).")
            else:
                print(f"Supervisor pid={result.pid} stopped.")
            return 0
        print(
            f"ERROR: Supervisor pid={result.pid} is still running "
            "(may be stuck in uninterruptible I/O)."
        )
        return 1
    if args.action == "status":
        st = daemon_status(handoff)
        bot = discord_bot_status_for_handoff(handoff, daemon_alive=st.alive)
        bot_payload = {
            "enabled": bot.enabled,
            "expected_in_process": bot.expected_in_process,
            "skipped_reason": bot.skipped_reason,
        }
        print(
            json.dumps(
                {
                    "alive": st.alive,
                    "wedged": st.wedged or daemon_is_wedged(handoff),
                    "pid": st.pid,
                    "host": st.host,
                    "heartbeat_age_s": st.heartbeat_age_s,
                    "lease_generation": st.lease_generation,
                    "lease_age_s": st.lease_age_s,
                    "stop_pending": st.stop_pending,
                    "lock_held": st.lock_held,
                    "discord_bot": bot_payload,
                },
                indent=2,
            )
        )
        return 0
    raise SystemExit(f"Unknown daemon action: {args.action}")


def _add_site_scope(sp: argparse.ArgumentParser) -> None:
    """Add site scope.
    
    Parameters
    ----------
    sp : argparse.ArgumentParser"""
    sp.add_argument(
        "--site",
        default=None,
        help="Config directory with pipeline.yaml, diff_config.yaml, and deployment.yaml",
    )


def _add_workspace_scope(sp: argparse.ArgumentParser) -> None:
    """Add workspace scope.
    
    Parameters
    ----------
    sp : argparse.ArgumentParser"""
    _add_site_scope(sp)
    sp.add_argument(
        "--deployment",
        default=None,
        help="Path to deployment.yaml (optional; auto-discovers one live supervisor)",
    )


def _add_run_scope(sp: argparse.ArgumentParser) -> None:
    """Add run scope.
    
    Parameters
    ----------
    sp : argparse.ArgumentParser"""
    sp.add_argument("--run-dir", default=None, help="Full run directory path (frozen config/targets)")
    sp.add_argument(
        "--run-id",
        default=None,
        help="Run ID under workspace runs/ (required for run control commands)",
    )
    sp.add_argument(
        "--deployment",
        default=None,
        help="Path to deployment.yaml (with --run-id; optional if one supervisor is running)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build parser.
    
    Returns
    -------
    argparse.ArgumentParser"""
    p = argparse.ArgumentParser(prog="syndiff", description="SynDiff pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("status", help="Show stage status grid (all active runs by default)")
    _add_site_scope(sp)
    _add_run_scope(sp)
    sp.add_argument("--watch", action="store_true")
    sp.add_argument("--interval", type=float, default=10.0)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("progress", help="Summary counts and running-task detail (all active by default)")
    _add_site_scope(sp)
    _add_run_scope(sp)
    sp.add_argument(
        "--no-detail",
        action="store_true",
        help="Print summary counts only (omit running-task log progress)",
    )
    sp.set_defaults(func=cmd_progress)

    sp = sub.add_parser("runs", help="List recent runs")
    _add_workspace_scope(sp)
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("active", help="Show active runs and supervisor")
    _add_workspace_scope(sp)
    sp.set_defaults(func=cmd_active)

    sp = sub.add_parser("show", help="Show run metadata JSON")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("logs", help="Print or follow logs")
    _add_run_scope(sp)
    sp.add_argument("--target", default=None)
    sp.add_argument("--stage", default=None)
    sp.add_argument("--follow", "-f", action="store_true")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("tail", help="Alias for logs --follow")
    _add_run_scope(sp)
    sp.add_argument("--target", default=None)
    sp.add_argument("--stage", default=None)
    sp.set_defaults(func=cmd_logs, follow=True)

    sp = sub.add_parser("verify", help="Verify stage artifacts")
    _add_site_scope(sp)
    sp.add_argument("--run-dir", default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--config", default=None)
    sp.add_argument("--targets", default=None)
    sp.add_argument("--scc", default=None)
    sp.add_argument("--stages", default=None)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser(
        "reconcile-manifests",
        help="Backfill cross-run completion manifests for already-complete targets",
    )
    _add_site_scope(sp)
    sp.add_argument("--run-dir", default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--config", default=None)
    sp.add_argument("--targets", default=None)
    sp.add_argument("--scc", default=None)
    sp.add_argument("--stages", default=None)
    sp.add_argument(
        "--quiet",
        action="store_true",
        help="Only print stages where a manifest was written",
    )
    sp.set_defaults(func=cmd_reconcile_manifests)

    sp = sub.add_parser("retry", help="Retry failed/canceled stage(s)")
    _add_run_scope(sp)
    sp.add_argument(
        "--scc",
        "--target",
        default=None,
        dest="scc",
        help="Target label or SCC key (e.g. 22/3/3 or 2020dgc)",
    )
    sp.add_argument("--stage", default=None)
    sp.add_argument(
        "--no-start-daemon",
        action="store_true",
        help="Queue the intent without ensuring the supervisor daemon is running",
    )
    sp.add_argument(
        "--no-reset-downstream",
        action="store_true",
        help="Only reopen the targeted stage; leave downstream untouched",
    )
    sp.set_defaults(func=cmd_retry)

    sp = sub.add_parser(
        "launch",
        help="Force-launch a ready stage (bypasses pool max_concurrent)",
    )
    _add_site_scope(sp)
    _add_run_scope(sp)
    sp.add_argument(
        "--scc",
        "--target",
        default=None,
        dest="scc",
        help="Target label or SCC key (e.g. 24/1/2 or 2020ghq)",
    )
    sp.add_argument("--stage", required=True)
    sp.add_argument(
        "--no-start-daemon",
        action="store_true",
        help="Queue the intent without ensuring the supervisor daemon is running",
    )
    sp.set_defaults(func=cmd_launch)

    sp = sub.add_parser("pause", help="Pause run dequeuing")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_pause)

    sp = sub.add_parser("resume", help="Resume paused run")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("kill", help="Cancel run (intent to supervisor)")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_kill)

    sp = sub.add_parser("daemon", help="Supervisor daemon control")
    _add_workspace_scope(sp)
    sp.add_argument("action", choices=["start", "stop", "status"])
    sp.set_defaults(func=cmd_daemon)

    sp = sub.add_parser("notify", help="Discord notification utilities")
    notify_sub = sp.add_subparsers(dest="notify_action", required=True)

    sp_test = notify_sub.add_parser(
        "test",
        help="Send a read-only preview (progress + status grid) to Discord",
    )
    _add_run_scope(sp_test)
    sp_test.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message locally instead of posting to Discord",
    )
    sp_test.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the message after sending",
    )
    sp_test.set_defaults(func=cmd_notify_test)

    from syndiff_pipeline.common.provenance.cli import register_bookkeeping_subparser

    register_bookkeeping_subparser(sub)

    return p


def main(argv: list[str] | None = None) -> int:
    """Main.
    
    Parameters
    ----------
    argv : list[str] | None, optional, default ``None``
    
    Returns
    -------
    int"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
