"""CLI for the template pipeline runner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.template_runner import daemon, logs, stages
from syndiff_pipeline.template_runner.runner_config import load_runner_config, resolve_config
from syndiff_pipeline.template_runner.scheduler import run_scheduler
from syndiff_pipeline.template_runner.state import PipelineState, STAGE_NAMES
from syndiff_pipeline.template_runner.targets import find_target, load_targets
from syndiff_pipeline.template_runner.verify import verify_all, verify_target

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


def cmd_submit(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets)
    active = stages.parse_stage_list(args.stages)
    run_id = args.run_id or _default_run_id()
    runs_root = cfg.runs_dir()

    meta = {
        "run_id": run_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "targets_path": str(Path(args.targets).resolve()),
        "stages": active,
        "detach": True,
    }
    logs.ensure_run_layout(runs_root, run_id, meta)

    state = PipelineState(cfg.state_db_path)
    if state.get_run(run_id) is None:
        state.create_run(
            run_id,
            meta["config_path"],
            meta["targets_path"],
            runs_root,
            targets,
            active,
        )

    sched_log = logs.scheduler_log_path(runs_root, run_id)
    pid = daemon.spawn_detached_scheduler(
        run_id, args.config, args.targets, args.stages, sched_log
    )
    pid_path = logs.scheduler_pid_path(runs_root, run_id)
    daemon.write_pid(pid_path, pid)
    logs.update_run_meta(runs_root, run_id, {"scheduler_pid": pid, "detach": True})

    config_abs = str(Path(args.config).resolve())
    print(f"Submitted run_id={run_id} scheduler_pid={pid}")
    print(f"  logs: {sched_log}")
    print(f"Monitor: syndiff-template progress --config {config_abs} --run-id {run_id}")
    print(f"         syndiff-template status --watch --config {config_abs} --run-id {run_id}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if sys.stdout.isatty():
        print("Warning: foreground run blocks until complete; use 'submit' for detached runs.")
    run_id = args.run_id or _default_run_id()
    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets)
    active = stages.parse_stage_list(args.stages)
    meta = {
        "run_id": run_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "targets_path": str(Path(args.targets).resolve()),
        "stages": active,
        "detach": False,
    }
    logs.ensure_run_layout(cfg.runs_dir(), run_id, meta)
    return run_scheduler(run_id, args.config, args.targets, args.stages)


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    state = PipelineState(cfg.state_db_path)

    def _print_once():
        rows = state.list_stage_runs(run_id)
        print(f"Run {run_id}")
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
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    state = PipelineState(cfg.state_db_path)
    counts = state.count_by_status(run_id)
    run = state.get_run(run_id) or {}
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    print(f"run_id={run_id} status={run.get('status', '?')} " + " ".join(parts))
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
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    meta_path = logs.run_dir(cfg.runs_dir(), run_id) / "run_meta.json"
    if meta_path.is_file():
        print(meta_path.read_text(encoding="utf-8"))
    else:
        print(f"No run_meta.json for {run_id}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    if args.target and args.stage:
        path = logs.target_log_path(cfg.runs_dir(), run_id, args.target, args.stage)
    else:
        path = logs.scheduler_log_path(cfg.runs_dir(), run_id)
    if not path.is_file():
        print(f"Log not found: {path}")
        return 1
    if args.follow:
        import subprocess

        return subprocess.call(["tail", "-f", str(path)])
    print(path.read_text(encoding="utf-8", errors="replace"))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    targets = load_targets(args.targets) if args.targets else []
    if args.scc:
        t = find_target(targets, args.scc)
        targets = [t]
    if not targets:
        raise SystemExit("--targets required unless querying a specific run artifact")
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
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    targets = load_targets(args.targets)
    t = find_target(targets, args.scc)
    state = PipelineState(cfg.state_db_path)
    state.reset_stage_for_retry(run_id, t.label(), args.stage, reset_downstream=True)
    print(f"Re-queued {args.stage} for {t.label()} in run {run_id}")
    return 0


def cmd_pause(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    PipelineState(cfg.state_db_path).set_paused(run_id, True)
    print(f"Paused run {run_id}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    PipelineState(cfg.state_db_path).set_paused(run_id, False)
    print(f"Resumed run {run_id}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    cfg = load_runner_config(args.config)
    run_id = _resolve_run_id(cfg, args.run_id)
    state = PipelineState(cfg.state_db_path)
    pid_path = logs.scheduler_pid_path(cfg.runs_dir(), run_id)
    pid = daemon.read_pid(pid_path)
    if pid:
        daemon.terminate_pid(pid)
    for spid in state.running_pids(run_id):
        daemon.terminate_pid(spid)
    state.set_run_status(run_id, "killed")
    print(f"Killed run {run_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="syndiff-template", description="SynDiff template pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--config", required=True, help="Path to config.yaml")

    sp = sub.add_parser("submit", help="Submit detached background run")
    add_common(sp)
    sp.add_argument("--targets", required=True)
    sp.add_argument("--stages", default=None)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("run", help="Foreground run (debug)")
    add_common(sp)
    sp.add_argument("--targets", required=True)
    sp.add_argument("--stages", default=None)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("status", help="Show stage status grid")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--watch", action="store_true")
    sp.add_argument("--interval", type=float, default=10.0)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("progress", help="Summary counts")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_progress)

    sp = sub.add_parser("runs", help="List recent runs")
    add_common(sp)
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_runs)

    sp = sub.add_parser("active", help="Show runs with live schedulers")
    add_common(sp)
    sp.set_defaults(func=cmd_active)

    sp = sub.add_parser("show", help="Show run metadata JSON")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("logs", help="Print or follow logs")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--target", default=None)
    sp.add_argument("--stage", default=None)
    sp.add_argument("--follow", "-f", action="store_true")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("tail", help="Alias for logs --follow")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--target", default=None)
    sp.add_argument("--stage", default=None)
    sp.set_defaults(func=cmd_logs, follow=True)

    sp = sub.add_parser("verify", help="Verify stage artifacts")
    add_common(sp)
    sp.add_argument("--targets", default=None)
    sp.add_argument("--scc", default=None, help="sector,camera,ccd")
    sp.add_argument("--stages", default=None)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("retry", help="Retry a failed stage for one SCC")
    add_common(sp)
    sp.add_argument("--targets", required=True)
    sp.add_argument("--run-id", default=None)
    sp.add_argument("--scc", required=True)
    sp.add_argument("--stage", required=True)
    sp.set_defaults(func=cmd_retry)

    sp = sub.add_parser("pause", help="Pause scheduler dequeuing")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_pause)

    sp = sub.add_parser("resume", help="Resume paused scheduler")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("kill", help="Kill scheduler and running stages")
    add_common(sp)
    sp.add_argument("--run-id", default=None)
    sp.set_defaults(func=cmd_kill)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
