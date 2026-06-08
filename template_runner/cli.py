"""CLI for the template pipeline runner."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.run_context import (
    RUNS_ROOT_ENV_VAR,
    RunContext,
    resolve_run_context,
    runs_root_from_env,
)
from syndiff_pipeline.template_runner.runner_config import load_runner_config
from syndiff_pipeline.template_runner.state import PipelineState

log = logging.getLogger(__name__)


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


def _resolve_run_from_args(args: argparse.Namespace) -> RunContext:
    if getattr(args, "run_dir", None):
        return resolve_run_context(
            run_dir=args.run_dir,
            run_id=getattr(args, "run_id", None),
        )

    run_id = getattr(args, "run_id", None)
    env_runs_root = runs_root_from_env()
    if env_runs_root is not None:
        if not run_id:
            raise SystemExit(
                f"--run-id is required when {RUNS_ROOT_ENV_VAR} is set ({env_runs_root})."
            )
        return resolve_run_context(run_id=run_id, runs_root=str(env_runs_root))

    if not getattr(args, "config", None):
        raise SystemExit(
            f"Specify --run-dir, set {RUNS_ROOT_ENV_VAR} with --run-id, "
            "or --config (with optional --run-id)."
        )
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, run_id)
    return resolve_run_context(run_id=run_id, runs_root=cfg.runs_dir())


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
    if state.get_run(run_id) is None:
        state.create_run(
            run_id,
            str(logs.run_config_path(run_directory)),
            str(logs.run_targets_path(run_directory)),
            runs_root,
            targets,
            active,
        )

    sched_log = logs.scheduler_log_path(runs_root, run_id)
    pid = daemon.spawn_detached_scheduler(
        run_id,
        run_directory,
        args.stages,
        sched_log,
        force_rerun=bool(args.force_rerun),
    )
    pid_path = logs.scheduler_pid_path(runs_root, run_id)
    daemon.write_pid(pid_path, pid)
    logs.update_run_meta(runs_root, run_id, {"scheduler_pid": pid, "detach": True})

    print(f"Submitted run_id={run_id} scheduler_pid={pid}")
    print(f"  logs: {sched_log}")
    print(f"Monitor: syndiff-template progress --run-dir {run_directory}")
    print(f"         syndiff-template status --watch --run-dir {run_directory}")
    print(f"  or:    export {RUNS_ROOT_ENV_VAR}={runs_root}")
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
        )

    return run_scheduler(
        run_id,
        str(run_directory),
        args.stages,
        force_rerun=bool(args.force_rerun),
    )


def cmd_status(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)

    def _print_once():
        rows = state.list_stage_runs(ctx.run_id)
        print(f"Run {ctx.run_id}")
        by_target: dict[str, list] = {}
        for r in rows:
            by_target.setdefault(r.target_label, []).append(r)
        for label in sorted(by_target):
            parts = [f"{r.stage[:4]}:{r.status[:4]}" for r in by_target[label]]
            print(f"  {label}: {' | '.join(parts)}")

    if args.watch:
        while True:
            print("\033[2J\033[H", end="")
            _print_once()
            time.sleep(args.interval)
    else:
        _print_once()
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    counts = state.count_by_status(ctx.run_id)
    run = state.get_run(ctx.run_id) or {}
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    print(f"run_id={ctx.run_id} status={run.get('status', '?')} " + " ".join(parts))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    state = PipelineState(cfg.state_db_path)
    for r in state.list_runs(args.limit):
        pid_path = logs.scheduler_pid_path(cfg.runs_dir(), r["run_id"])
        pid = daemon.read_pid(pid_path)
        alive = daemon.is_process_alive(pid) if pid else False
        print(
            f"{r['run_id']}  status={r.get('status')}  "
            f"started={r.get('started_at')}  scheduler_alive={alive}"
        )
    return 0


def cmd_active(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    state = PipelineState(cfg.state_db_path)
    found = False
    for r in state.list_runs(50):
        pid_path = logs.scheduler_pid_path(cfg.runs_dir(), r["run_id"])
        pid = daemon.read_pid(pid_path)
        if pid and daemon.is_process_alive(pid):
            print(f"{r['run_id']}  pid={pid}  status={r.get('status')}")
            found = True
    if not found:
        print("No active scheduler processes.")
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
        path = logs.scheduler_log_path(ctx.cfg.runs_dir(), ctx.run_id)
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
    from syndiff_pipeline.template_runner.state import STAGE_NAMES
    from syndiff_pipeline.template_runner.targets import find_target, load_targets
    from syndiff_pipeline.template_runner.verify import verify_all

    if args.run_dir or runs_root_from_env() is not None:
        ctx = _resolve_run_from_args(args)
        cfg = ctx.cfg
        targets = ctx.targets
    else:
        if not args.config:
            raise SystemExit(
                f"Specify --run-dir, {RUNS_ROOT_ENV_VAR} with --run-id, or --config for verify."
            )
        cfg = load_runner_config(args.config)
        if not args.targets:
            raise SystemExit("--targets required for pre-run verify.")
        targets = load_targets(args.targets)

    if args.scc:
        t = find_target(targets, args.scc)
        targets = [t]
    active = stages.parse_stage_list(args.stages) if args.stages else list(STAGE_NAMES)
    results = verify_all(cfg, targets, active)
    rc = 0
    for r in results:
        mark = "OK" if r.ok else "FAIL"
        print(f"[{mark}] {r.stage}: {r.message} ({r.path})")
        if not r.ok:
            rc = 1
    return rc


def cmd_retry(args: argparse.Namespace) -> int:
    from syndiff_pipeline.template_runner.targets import find_target

    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)

    if args.scc and args.stage:
        t = find_target(ctx.targets, args.scc)
        state.reset_stage_for_retry(ctx.run_id, t.label(), args.stage, reset_downstream=True)
        print(f"Re-queued {args.stage} for {t.label()} in run {ctx.run_id}")
        return 0

    if args.scc or args.stage:
        raise SystemExit(
            "Specify both --scc and --stage for a single retry, "
            "or omit both to retry all failed stages."
        )

    failed = state.list_failed_stage_runs(ctx.run_id)
    if not failed:
        print(f"No failed stages in run {ctx.run_id}")
        return 0

    for row in failed:
        state.reset_stage_for_retry(
            ctx.run_id, row.target_label, row.stage, reset_downstream=True
        )
        print(f"Re-queued {row.stage} for {row.target_label}")
    print(f"Re-queued {len(failed)} failed stage(s) in run {ctx.run_id}")
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    PipelineState(ctx.cfg.state_db_path).set_paused(ctx.run_id, True)
    print(f"Paused run {ctx.run_id}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    PipelineState(ctx.cfg.state_db_path).set_paused(ctx.run_id, False)
    print(f"Resumed run {ctx.run_id}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    ctx = _resolve_run_from_args(args)
    state = PipelineState(ctx.cfg.state_db_path)
    pid_path = logs.scheduler_pid_path(ctx.cfg.runs_dir(), ctx.run_id)
    pid = daemon.read_pid(pid_path)
    if pid:
        daemon.terminate_process_tree(pid)
    running_jobs = state.running_jobs(ctx.run_id)
    condor_removed = 0
    local_terminated = 0
    for job in running_jobs:
        if job.pid is None:
            continue
        if ctx.cfg.stage_executor(job.stage) == "condor":
            from syndiff_pipeline.template_runner import condor

            if condor.remove_cluster(job.pid):
                condor_removed += 1
        else:
            daemon.terminate_process_tree(job.pid)
            local_terminated += 1
    counts = state.finalize_run_killed(ctx.run_id)
    state.set_run_status(ctx.run_id, "killed")
    daemon.remove_pid_file(pid_path)
    print(f"Killed run {ctx.run_id}")
    if condor_removed:
        print(f"  Removed {condor_removed} Condor cluster(s)")
    if local_terminated:
        print(f"  Terminated {local_terminated} local stage job(s)")
    if counts["killed"]:
        print(f"  Marked {counts['killed']} running stage(s) as killed")
    if counts["blocked"]:
        print(f"  Marked {counts['blocked']} pending/ready stage(s) as blocked")
    return 0


def _add_run_scope(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--run-dir",
        default=None,
        help="Full run directory path (frozen config/targets)",
    )
    sp.add_argument(
        "--run-id",
        default=None,
        help=f"Run ID under {RUNS_ROOT_ENV_VAR} or site --config runs_root",
    )
    sp.add_argument(
        "--config",
        default=None,
        help="Site config for runs_root lookup when --run-dir is omitted",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="syndiff-template", description="SynDiff template pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("submit", help="Submit detached background run")
    sp.add_argument("--config", required=True, help="Source config.yaml (copied into run directory)")
    sp.add_argument("--targets", required=True, help="Source targets CSV (copied into run directory)")
    sp.add_argument("--stages", default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run stages even when output artifacts already exist",
    )
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("run", help="Foreground run (debug)")
    sp.add_argument("--config", required=True, help="Source config.yaml (copied into run directory)")
    sp.add_argument("--targets", required=True, help="Source targets CSV (copied into run directory)")
    sp.add_argument("--stages", default=None)
    sp.add_argument("--run-id", default=None)
    sp.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run stages even when output artifacts already exist",
    )
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="Show stage status grid")
    _add_run_scope(sp)
    sp.add_argument("--watch", action="store_true")
    sp.add_argument("--interval", type=float, default=10.0)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("progress", help="Summary counts")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_progress)

    sp = sub.add_parser("runs", help="List recent runs")
    sp.add_argument("--config", required=True)
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("active", help="Show runs with live schedulers")
    sp.add_argument("--config", required=True)
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
    sp.add_argument("--run-dir", default=None, help="Full run directory (frozen config/targets)")
    sp.add_argument(
        "--run-id",
        default=None,
        help=f"Run ID with {RUNS_ROOT_ENV_VAR} or site --config",
    )
    sp.add_argument("--config", default=None, help="Pre-run verify: site config path")
    sp.add_argument("--targets", default=None, help="Pre-run verify: targets CSV")
    sp.add_argument("--scc", default=None, help="sector,camera,ccd")
    sp.add_argument("--stages", default=None)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser(
        "retry",
        help="Retry failed stage(s); omit --scc/--stage to requeue all failed stages",
    )
    _add_run_scope(sp)
    sp.add_argument(
        "--scc",
        default=None,
        help="sector,camera,ccd (with --stage for a single retry)",
    )
    sp.add_argument(
        "--stage",
        default=None,
        help="Stage name (with --scc for a single retry)",
    )
    sp.set_defaults(func=cmd_retry)

    sp = sub.add_parser("pause", help="Pause scheduler dequeuing")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_pause)

    sp = sub.add_parser("resume", help="Resume paused scheduler")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("kill", help="Kill scheduler and running stages")
    _add_run_scope(sp)
    sp.set_defaults(func=cmd_kill)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
