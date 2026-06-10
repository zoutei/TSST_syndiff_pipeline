"""CLI for the template pipeline runner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.template_runner import logs
from syndiff_pipeline.template_runner.run_context import RunContext, resolve_run_context
from syndiff_pipeline.template_runner.deployment import (
    deployment_path_for_config,
    load_handoff_root_from_deployment,
)
from syndiff_pipeline.template_runner.runner_config import load_runner_config
from syndiff_pipeline.template_runner.workspace import (
    discover_alive_handoff_roots,
    load_recorded_deployment_path,
    record_deployment_path,
    runs_root as handoff_runs_root,
    state_db_path,
)
from syndiff_pipeline.template_runner.scheduler_control import (
    daemon_is_alive,
    daemon_is_wedged,
    daemon_status,
    ensure_daemon_running,
    stop_daemon,
)
from syndiff_pipeline.template_runner.state import (
    STAGE_NAMES,
    PipelineState,
)

log = logging.getLogger(__name__)


def _discord_bot_config_path(args: argparse.Namespace, ctx: RunContext | None = None) -> Path | None:
    if getattr(args, "config", None):
        return Path(args.config).expanduser().resolve()
    if ctx is not None:
        source = (ctx.meta or {}).get("source_config_path")
        if source:
            return Path(source).expanduser().resolve()
        return logs.run_config_path(ctx.run_dir)
    return None


def _ensure_discord_bot(
    deployment_path: str | Path,
    *,
    site_config_path: str | Path | None = None,
):
    from syndiff_pipeline.template_runner.discord_bot_control import ensure_discord_bot_running

    return ensure_discord_bot_running(
        deployment_path,
        site_config_path=site_config_path,
    )


def _print_discord_bot_status(
    deployment_path: str | Path,
    result,
    *,
    site_config_path: str | Path | None = None,
) -> None:
    if result is None:
        print("Discord bot: starting with supervisor (see daemon.log)")
        return
    if not result.enabled:
        return
    handoff = str(load_handoff_root_from_deployment(deployment_path))
    bot_log = logs.discord_bot_log_path(handoff)
    if result.pid:
        print(f"Discord bot pid={result.pid} spawned={result.spawned}")
        print(f"  bot log: {bot_log}")
    elif result.skipped_reason:
        print(f"WARNING: Discord bot not running: {result.skipped_reason}")
        print(f"  bot log: {bot_log}")
    else:
        print("WARNING: Discord bot not running (unknown reason)")
        print(f"  bot log: {bot_log}")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_run_id(cfg, run_id: str | None) -> str:
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
    root = handoff_runs_root(handoff)
    latest = root / "latest"
    if latest.is_symlink():
        return latest.readlink().name
    runs = sorted(p for p in root.glob("*") if p.is_dir() and p.name != "latest")
    if runs:
        return runs[-1].name
    raise SystemExit("No runs found in workspace.")


def _resolve_handoff_from_args(args: argparse.Namespace) -> str:
    deployment = getattr(args, "deployment", None)
    if deployment:
        path = Path(deployment).expanduser().resolve()
        handoff = load_handoff_root_from_deployment(path)
        record_deployment_path(handoff, path)
        return str(handoff)

    discovered = discover_alive_handoff_roots()
    if len(discovered) == 1:
        return str(discovered[0])
    if len(discovered) > 1:
        lines = "\n".join(f"  {p}" for p in discovered)
        raise SystemExit(f"Multiple supervisors running; pass --deployment:\n{lines}")
    raise SystemExit(
        "No supervisor found. Start with: syndiff-template submit --config ... "
        "or syndiff-template daemon start --deployment ..."
    )


def _resolve_deployment_from_args(args: argparse.Namespace) -> Path:
    deployment = getattr(args, "deployment", None)
    if deployment:
        path = Path(deployment).expanduser().resolve()
        handoff = load_handoff_root_from_deployment(path)
        record_deployment_path(handoff, path)
        return path

    discovered = discover_alive_handoff_roots()
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
        "No supervisor found. Start with: syndiff-template submit --config ... "
        "or syndiff-template daemon start --deployment ..."
    )


def _resolve_run_from_args(args: argparse.Namespace) -> RunContext:
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
        handoff = load_handoff_root_from_deployment(deployment)
        return resolve_run_context(
            run_id=run_id,
            runs_root=str(handoff_runs_root(handoff)),
        )

    handoff = _resolve_handoff_from_args(args)
    return resolve_run_context(run_id=run_id, runs_root=str(handoff_runs_root(handoff)))


def _resolve_run_ids_for_monitoring(
    state: PipelineState,
    handoff: str,
    *,
    run_id: str | None = None,
) -> list[str]:
    if run_id:
        return [run_id]
    active = state.active_runs()
    if active:
        return [row["run_id"] for row in active]
    return [_resolve_latest_run_id_from_handoff(handoff)]


def _monitoring_mode(args: argparse.Namespace) -> bool:
    return not getattr(args, "run_dir", None) and not getattr(args, "run_id", None)


def _prepare_run_directory(
    source_config: str,
    source_targets: str,
    run_id: str,
    runs_root: str,
    *,
    stages: list[str],
    detach: bool,
    force_rerun: bool,
) -> Path:
    run_directory = logs.run_dir(runs_root, run_id)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "per_target").mkdir(exist_ok=True)

    config_path, targets_path = logs.materialize_run_inputs(
        source_config, source_targets, run_directory
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
    logs.ensure_run_layout(runs_root, run_id, meta)
    logs.update_run_meta(runs_root, run_id, meta)
    return run_directory


def _run_context_from_directory(run_directory: Path, run_id: str) -> RunContext:
    return resolve_run_context(run_dir=run_directory, run_id=run_id)


def cmd_submit(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner import stages
    from syndiff_pipeline.template_runner.targets import load_targets

    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets)
    active = stages.parse_stage_list(args.stages)
    run_id = args.run_id or _default_run_id()
    runs_root = cfg.runs_dir()

    run_directory = _prepare_run_directory(
        args.config,
        args.targets,
        run_id,
        runs_root,
        stages=active,
        detach=True,
        force_rerun=bool(args.force_rerun),
    )

    state = PipelineState(cfg.state_db_path)
    created_new_run = False
    if state.get_run(run_id) is None:
        # New run: stages are materialized pending/external and the force_rerun
        # flag is persisted at creation, before the daemon can schedule it. No
        # post-hoc execution-state mutation is needed (or safe) here.
        state.create_run(
            run_id,
            str(logs.run_config_path(run_directory)),
            str(logs.run_targets_path(run_directory)),
            runs_root,
            targets,
            active,
            force_rerun=bool(args.force_rerun),
        )
        if cfg.stages.ps1_process.ps1_source == "stream":
            if "ps1_download" in active:
                print(
                    "Note: ps1_download ignored for this run (ps1_source=stream); "
                    "download happens inside ps1_process."
                )
            skipped = state.apply_ps1_stream_download_skips(run_id, targets, cfg)
            if skipped:
                print(f"Marked ps1_download n/a (stream_mode) for {skipped} target(s).")
        created_new_run = True
    elif args.force_rerun:
        # Resubmitting force-rerun onto an EXISTING run. The daemon is the sole
        # owner of execution state and may be actively scheduling this run, so
        # the CLI must NOT reset stages directly (that races mid-launch). Emit a
        # force_rerun intent instead; the daemon applies it as the single writer.
        state.insert_command(
            "force_rerun",
            run_id=run_id,
            args={
                "target_labels": [t.label() for t in targets],
                "stages": active,
            },
        )

    deploy_path = deployment_path_for_config(args.config, cfg.deployment_file)
    record_deployment_path(cfg.handoff_root, deploy_path)
    from syndiff_pipeline.template_runner.discord_bot_control import (
        record_discord_bot_site_config,
    )

    record_discord_bot_site_config(cfg.handoff_root, args.config)
    result = ensure_daemon_running(cfg.handoff_root, deployment_path=deploy_path)
    if result.spawned:
        bot_result = None
    else:
        bot_result = _ensure_discord_bot(deploy_path, site_config_path=args.config)
    daemon_log = logs.daemon_log_path(cfg.handoff_root)

    if created_new_run and cfg.notifications.enabled:
        from syndiff_pipeline.template_runner.notifications import send_run_started_notification

        send_run_started_notification(
            state,
            cfg.notifications,
            config_path=args.config,
            run_id=run_id,
            run_dir=run_directory,
            target_labels=[t.label() for t in targets if t.enabled],
            stages=active,
            handoff_root=cfg.handoff_root,
            deployment_file=cfg.deployment_file,
            force_rerun=bool(args.force_rerun),
        )

    print(f"Submitted run_id={run_id} supervisor_pid={result.pid}")
    print(f"  daemon log: {daemon_log}")
    _print_discord_bot_status(deploy_path, bot_result, site_config_path=args.config)
    print("Monitor: syndiff-template progress")
    print("         syndiff-template status --watch")
    print(f"         syndiff-template progress --run-id {run_id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner import stages
    from syndiff_pipeline.template_runner.scheduler import run_scheduler
    from syndiff_pipeline.template_runner.targets import load_targets

    if sys.stdout.isatty():
        print("Warning: foreground run blocks until complete; use 'submit' for detached runs.")
    run_id = args.run_id or _default_run_id()
    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets)
    active = stages.parse_stage_list(args.stages)
    runs_root = cfg.runs_dir()

    run_directory = _prepare_run_directory(
        args.config,
        args.targets,
        run_id,
        runs_root,
        stages=active,
        detach=False,
        force_rerun=bool(args.force_rerun),
    )

    state = PipelineState(cfg.state_db_path)
    if state.get_run(run_id) is None:
        state.create_run(
            run_id,
            str(logs.run_config_path(run_directory)),
            str(logs.run_targets_path(run_directory)),
            runs_root,
            targets,
            active,
            force_rerun=bool(args.force_rerun),
        )

    return run_scheduler(
        run_id,
        str(run_directory),
        args.stages,
        force_rerun=bool(args.force_rerun),
    )


def _print_status_for_run(
    state: PipelineState,
    *,
    run_id: str,
    handoff_root: str,
    multi_run: bool,
) -> None:
    from syndiff_pipeline.template_runner.run_report import format_status_grid
    from syndiff_pipeline.template_runner.verify_status import read_verify_in_flight

    if multi_run:
        print(f"=== run {run_id} ===")
    run = state.get_run(run_id) or {}
    print(f"Run {run_id} status={run.get('status', '?')}")
    in_flight = read_verify_in_flight(handoff_root, run_id)
    if in_flight:
        print(f"  verify_in_flight={in_flight}")
    if run.get("status") == "stalled" and run.get("stall_reason"):
        print(f"  stalled: {run['stall_reason']}")
    for line in format_status_grid(state, run_id):
        print(line)


def cmd_status(args: argparse.Namespace) -> int:
    if _monitoring_mode(args):
        handoff = _resolve_handoff_from_args(args)
        state = PipelineState(str(state_db_path(handoff)))
        run_ids = _resolve_run_ids_for_monitoring(
            state, handoff, run_id=getattr(args, "run_id", None)
        )

        def _print_once():
            multi = len(run_ids) > 1
            for run_id in run_ids:
                _print_status_for_run(
                    state,
                    run_id=run_id,
                    handoff_root=handoff,
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
                    "Start with: syndiff-template submit --config ... "
                    "or syndiff-template daemon start --deployment ..."
                )
        return 0

    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    _print_status_for_run(
        state,
        run_id=ctx.run_id,
        handoff_root=ctx.cfg.handoff_root,
        multi_run=False,
    )
    if not args.watch:
        if not daemon_is_alive(ctx.cfg.handoff_root):
            source = (ctx.meta or {}).get("source_config_path")
            hint = source or logs.run_config_path(ctx.run_dir)
            print(
                "WARNING: supervisor daemon is not alive. "
                f"Start with: syndiff-template daemon start --deployment ... "
                f"(site config: {hint})"
            )
        return 0

    while True:
        print("\033[2J\033[H", end="")
        _print_status_for_run(
            state,
            run_id=ctx.run_id,
            handoff_root=ctx.cfg.handoff_root,
            multi_run=False,
        )
        time.sleep(args.interval)


def cmd_progress(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner.run_report import (
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
            runs_root = run.get("runs_root") or str(handoff_runs_root(handoff))
            for line in format_progress_lines(
                state,
                run_id,
                runs_root,
                handoff_root=handoff,
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
        handoff_root=ctx.cfg.handoff_root,
        include_running_detail=not getattr(args, "no_detail", False),
    ):
        print(line)
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
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
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    from syndiff_pipeline.template_runner.notifications import (
        format_preview_message,
        send_preview_notification,
    )

    if getattr(args, "dry_run", False):
        print(
            format_preview_message(
                state,
                ctx.run_id,
                ctx.cfg.runs_dir(),
                handoff_root=ctx.cfg.handoff_root,
            )
        )
        return 0

    message = send_preview_notification(state, ctx)
    print(f"Sent test notification to Discord for run {ctx.run_id}.")
    if getattr(args, "verbose", False):
        print(message)
    return 0


def cmd_active(args: argparse.Namespace) -> int:
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
    ctx = _resolve_run_from_args(args)
    meta_path = logs.run_meta_path(ctx.run_dir)
    if meta_path.is_file():
        print(meta_path.read_text(encoding="utf-8"))
    else:
        print(f"No run_meta.json for {ctx.run_id}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    if args.target and args.stage:
        path = logs.target_log_path(ctx.cfg.runs_dir(), ctx.run_id, args.target, args.stage)
    else:
        path = logs.daemon_log_path(ctx.cfg.handoff_root)
    if not path.is_file():
        print(f"Log not found: {path}")
        return 1
    if args.follow:
        import subprocess

        return subprocess.call(["tail", "-f", str(path)])
    print(path.read_text(encoding="utf-8", errors="replace"))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner import stages
    from syndiff_pipeline.template_runner.runner_config import resolve_config
    from syndiff_pipeline.template_runner.targets import find_target, load_targets
    from syndiff_pipeline.template_runner.verify import persist_completion_manifests, verify_stage

    run_id: str | None = None
    if args.run_dir:
        ctx = _resolve_run_from_args(args)
        cfg = ctx.cfg
        targets = ctx.targets
        run_id = ctx.run_id
    else:
        if not args.config:
            raise SystemExit("Specify --run-dir or --config for verify.")
        cfg = load_runner_config(args.config)
        if not args.targets:
            raise SystemExit("--targets required for pre-run verify.")
        targets = load_targets(args.targets)
        if args.run_id:
            run_id = args.run_id

    if args.scc:
        t = find_target(targets, args.scc)
        targets = [t]
    active = stages.parse_stage_list(args.stages) if args.stages else list(STAGE_NAMES)
    runs_root = cfg.runs_dir()
    rc = 0
    for target in targets:
        label = target.label()
        resolved = resolve_config(target, cfg)
        for stage in active:
            result = verify_stage(resolved, stage)
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
                    written = persist_completion_manifests(resolved, stage, manifest_paths)
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
    from syndiff_pipeline.template_runner import stages as stages_mod
    from syndiff_pipeline.template_runner.runner_config import resolve_config
    from syndiff_pipeline.template_runner.targets import find_target, load_targets
    from syndiff_pipeline.template_runner.verify import (
        collect_stage_artifacts,
        stage_complete,
        write_manifest,
    )

    if args.run_dir:
        ctx = _resolve_run_from_args(args)
        cfg = ctx.cfg
        targets = ctx.targets
    else:
        if not args.config:
            raise SystemExit("Specify --run-dir or --config.")
        cfg = load_runner_config(args.config)
        if not args.targets:
            raise SystemExit("--targets required for reconcile-manifests.")
        targets = load_targets(args.targets)

    if args.scc:
        targets = [find_target(targets, args.scc)]
    active = stages_mod.parse_stage_list(args.stages) if args.stages else list(STAGE_NAMES)
    runs_root = cfg.runs_dir()

    written = 0
    skipped = 0
    for target in targets:
        label = target.label()
        resolved = resolve_config(target, cfg)
        for stage in active:
            stable_path = str(logs.stable_stage_manifest_path(runs_root, label, stage))
            try:
                complete = stage_complete(resolved, stage)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR]   {label}/{stage}: {exc}")
                continue
            if not complete:
                skipped += 1
                if not args.quiet:
                    print(f"[SKIP]  {label}/{stage} not complete")
                continue
            try:
                expected, produced, artifacts = collect_stage_artifacts(resolved, stage)
                write_manifest(stable_path, resolved, stage, artifacts, expected, produced)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR]   {label}/{stage}: manifest write failed: {exc}")
                continue
            written += 1
            print(f"[WROTE] {label}/{stage} ({produced}/{expected}) -> {stable_path}")

    print(f"reconcile-manifests: wrote {written} manifest(s), {skipped} stage(s) not complete")
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner.targets import find_target_for_run

    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)

    if args.scc and args.stage:
        t = find_target_for_run(ctx, state, args.scc)
        state.insert_command(
            "retry",
            run_id=ctx.run_id,
            args={
                "target_label": t.label(),
                "stage": args.stage,
                "reset_downstream": True,
            },
        )
        print(f"Queued retry for {args.stage} on {t.label()} in run {ctx.run_id}")
    elif args.scc or args.stage:
        raise SystemExit(
            "Specify both --scc and --stage for a single retry, "
            "or omit both to retry all failed/canceled stages."
        )
    else:
        state.insert_command("retry", run_id=ctx.run_id)
        print(f"Queued bulk retry for run {ctx.run_id}")

    if not getattr(args, "no_start_daemon", False):
        from syndiff_pipeline.template_runner.discord_bot_control import (
            record_discord_bot_site_config,
        )

        deploy_path = load_recorded_deployment_path(ctx.cfg.handoff_root)
        result = ensure_daemon_running(
            ctx.cfg.handoff_root,
            deployment_path=deploy_path,
        )
        bot_config = _discord_bot_config_path(args, ctx)
        if bot_config is not None and deploy_path is not None:
            record_discord_bot_site_config(ctx.cfg.handoff_root, bot_config)
            if not result.spawned:
                _ensure_discord_bot(deploy_path, site_config_path=bot_config)
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    PipelineState(ctx.cfg.state_db_path).insert_command("pause", run_id=ctx.run_id)
    print(f"Queued pause for run {ctx.run_id}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    state.insert_command("resume", run_id=ctx.run_id)
    print(f"Queued resume for run {ctx.run_id}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    state.insert_command("cancel", run_id=ctx.run_id)
    from syndiff_pipeline.template_runner import condor

    condor.sweep_run_condor_audit_clusters(ctx.cfg.runs_dir(), ctx.run_id)
    print(f"Queued cancel for run {ctx.run_id}")
    return 0


def cmd_discord_bot(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner.discord_bot import run_discord_bot

    deploy_path = _resolve_deployment_from_args(args)
    run_discord_bot(deploy_path)
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner.discord_bot_control import (
        discord_bot_status_for_handoff,
        ensure_discord_bot_for_handoff_root,
    )

    handoff = _resolve_handoff_from_args(args)
    deploy_arg = getattr(args, "deployment", None)
    deploy_path = (
        Path(deploy_arg).expanduser().resolve()
        if deploy_arg
        else load_recorded_deployment_path(handoff)
    )
    if args.action == "start":
        result = ensure_daemon_running(handoff, deployment_path=deploy_path)
        if result.spawned:
            bot_result = None
        else:
            bot_result = ensure_discord_bot_for_handoff_root(handoff)
        print(f"Supervisor pid={result.pid} spawned={result.spawned}")
        if bot_result is not None:
            if bot_result.enabled and bot_result.pid:
                print(f"Discord bot pid={bot_result.pid} spawned={bot_result.spawned}")
            elif bot_result.skipped_reason:
                print(f"WARNING: Discord bot not running: {bot_result.skipped_reason}")
        elif result.spawned:
            print("Discord bot: starting with supervisor (see daemon.log)")
        else:
            print("Discord bot not started (no recorded site config; submit a run first)")
        return 0
    if args.action == "stop":
        from syndiff_pipeline.template_runner.discord_bot_control import stop_discord_bot

        result = stop_daemon(handoff)
        bot_stopped = stop_discord_bot(handoff)
        if not result.was_running:
            if bot_stopped:
                print("Supervisor was not running. Discord bot stopped.")
            else:
                print("Supervisor was not running.")
            return 0
        if result.stopped:
            if result.force_killed:
                print(f"Supervisor pid={result.pid} stopped (SIGKILL).")
            else:
                print(f"Supervisor pid={result.pid} stopped.")
            if not bot_stopped:
                print("WARNING: Discord bot did not stop cleanly.")
            return 0
        print(
            f"ERROR: Supervisor pid={result.pid} is still running "
            "(may be stuck in uninterruptible I/O)."
        )
        return 1
    if args.action == "status":
        st = daemon_status(handoff)
        bot = discord_bot_status_for_handoff(handoff)
        bot_payload = {
            "enabled": bot.enabled,
            "alive": bot.alive,
            "pid": bot.pid,
            "skipped_reason": bot.skipped_reason,
        }
        print(
            json.dumps(
                {
                    "alive": st.alive,
                    "wedged": daemon_is_wedged(handoff),
                    "pid": st.pid,
                    "heartbeat_age_s": st.heartbeat_age_s,
                    "lock_held": st.lock_held,
                    "discord_bot": bot_payload,
                },
                indent=2,
            )
        )
        return 0
    raise SystemExit(f"Unknown daemon action: {args.action}")


def _add_workspace_scope(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--deployment",
        default=None,
        help="Path to deployment.yaml (optional; auto-discovers one live supervisor)",
    )


def _add_run_scope(sp: argparse.ArgumentParser) -> None:
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
    p = argparse.ArgumentParser(prog="syndiff-template", description="SynDiff template pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("submit", help="Submit detached background run")
    sp.add_argument("--config", required=True)
    sp.add_argument("--targets", required=True)
    sp.add_argument("--stages", default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--force-rerun", action="store_true")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("run", help="Foreground run (debug)")
    sp.add_argument("--config", required=True)
    sp.add_argument("--targets", required=True)
    sp.add_argument("--stages", default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--force-rerun", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="Show stage status grid (all active runs by default)")
    _add_run_scope(sp)
    sp.add_argument("--watch", action="store_true")
    sp.add_argument("--interval", type=float, default=10.0)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("progress", help="Summary counts and running-task detail (all active by default)")
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
    sp.add_argument("--scc", default=None)
    sp.add_argument("--stage", default=None)
    sp.add_argument(
        "--no-start-daemon",
        action="store_true",
        help="Queue the intent without ensuring the supervisor daemon is running",
    )
    sp.set_defaults(func=cmd_retry)

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

    sp = sub.add_parser("discord", help="Discord bot utilities")
    discord_sub = sp.add_subparsers(dest="discord_action", required=True)
    sp_bot = discord_sub.add_parser(
        "bot",
        help="Run bot that replies to channel messages with progress + status",
    )
    _add_workspace_scope(sp_bot)
    sp_bot.set_defaults(func=cmd_discord_bot)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
