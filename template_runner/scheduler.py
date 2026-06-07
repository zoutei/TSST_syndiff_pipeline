"""Resource-pool scheduler for multi-SCC template pipeline runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from syndiff_pipeline.template_runner import daemon, launcher, logs, stages
from syndiff_pipeline.template_runner.runner_config import load_runner_config, resolve_config
from syndiff_pipeline.template_runner.state import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    PipelineState,
    STAGE_DEPS,
    STAGE_POOL,
    _utc_now,
)
from syndiff_pipeline.template_runner.targets import load_targets
from syndiff_pipeline.template_runner.verify import verify_stage

log = logging.getLogger(__name__)

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Received signal %s — shutting down scheduler gracefully", signum)
    _shutdown = True


def _write_summary(state: PipelineState, run_id: str, runs_root: str) -> None:
    counts = state.count_by_status(run_id)
    summary = {"run_id": run_id, "counts": counts, "updated_at": _utc_now()}
    sp = logs.summary_json_path(runs_root, run_id)
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = logs.summary_csv_path(runs_root, run_id)
    rows = state.list_stage_runs(run_id)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "target_label",
                "stage",
                "status",
                "started_at",
                "finished_at",
                "exit_code",
                "log_path",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "target_label": r.target_label,
                    "stage": r.stage,
                    "status": r.status,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "exit_code": r.exit_code,
                    "log_path": r.log_path,
                }
            )


def _skip_if_artifacts_exist(
    state: PipelineState,
    run_id: str,
    cfg,
    targets,
    active_stages: List[str],
) -> None:
    for t in targets:
        resolved = resolve_config(t, cfg)
        for stage in active_stages:
            sr = state.get_stage_run(run_id, t.label(), stage)
            if sr is None or sr.status != STATUS_PENDING:
                continue
            result = verify_stage(resolved, stage)
            if result.ok:
                log.info("Skipping %s / %s — artifact exists", t.label(), stage)
                state.update_stage_status(
                    run_id,
                    t.label(),
                    stage,
                    STATUS_SKIPPED,
                    finished_at=_utc_now(),
                    exit_code=0,
                )


def _dep_satisfied(
    state: PipelineState,
    run_id: str,
    target_label: str,
    dep: str,
    active_stages: List[str],
    resolved,
) -> bool:
    """True if dependency is satisfied in-run or verified on disk (subset runs)."""
    if dep in active_stages:
        row = state.get_stage_run(run_id, target_label, dep)
        return row is not None and row.status in (STATUS_SUCCESS, STATUS_SKIPPED)
    return verify_stage(resolved, dep).ok


def _promote_ready_stages_subset(
    state: PipelineState,
    run_id: str,
    active_stages: List[str],
    targets,
    cfg,
) -> int:
    promoted = 0
    for t in targets:
        resolved = resolve_config(t, cfg)
        label = t.label()
        for stage in active_stages:
            row = state.get_stage_run(run_id, label, stage)
            if row is None or row.status != STATUS_PENDING:
                continue
            deps = STAGE_DEPS.get(stage, [])
            if all(_dep_satisfied(state, run_id, label, d, active_stages, resolved) for d in deps):
                state.update_stage_status(run_id, label, stage, STATUS_READY)
                promoted += 1
    return promoted


def run_scheduler(
    run_id: str,
    config_path: str,
    targets_path: str,
    stages_arg: str | None = None,
    force_rerun: bool = False,
) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [scheduler] %(message)s",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    cfg = load_runner_config(config_path)
    targets = load_targets(targets_path)
    active_stages = stages.parse_stage_list(stages_arg)
    state = PipelineState(cfg.state_db_path)
    runs_root = cfg.runs_dir()

    logs.ensure_run_layout(
        runs_root,
        run_id,
        {
            "run_id": run_id,
            "config_path": str(Path(config_path).resolve()),
            "targets_path": str(Path(targets_path).resolve()),
            "stages": active_stages,
            "force_rerun": force_rerun,
        },
    )
    pid_path = logs.scheduler_pid_path(runs_root, run_id)
    daemon.write_pid(pid_path, os.getpid())

    run_row = state.get_run(run_id)
    if run_row is None:
        state.create_run(
            run_id,
            str(Path(config_path).resolve()),
            str(Path(targets_path).resolve()),
            runs_root,
            targets,
            active_stages,
        )
    elif force_rerun:
        state.reset_stages_for_force_rerun(
            run_id, [t.label() for t in targets], active_stages
        )
        state.set_run_status(run_id, "running")

    if not force_rerun and run_row is None:
        _skip_if_artifacts_exist(state, run_id, cfg, targets, active_stages)
    elif force_rerun:
        log.info("Force rerun: skipping artifact-exists checks for %s", active_stages)

    _promote_ready_stages_subset(state, run_id, active_stages, targets, cfg)

    running: Dict[str, launcher.StageJobHandle] = {}
    running_meta: Dict[str, tuple[str, str, str]] = {}
    pool_running: Dict[str, int] = defaultdict(int)
    last_heartbeat = 0.0

    try:
        while not _shutdown:
            if state.is_paused(run_id):
                time.sleep(1.0)
                continue

            # Reap finished subprocesses
            done_keys: List[str] = []
            for key, handle in list(running.items()):
                ret = handle.poll()
                if ret is None:
                    continue
                target_label, stage, pool = running_meta[key]
                pool_running[pool] -= 1
                log_path = str(logs.target_log_path(runs_root, run_id, target_label, stage))
                error_tail = logs.read_log_tail(log_path, 20) if ret != 0 else ""
                if ret == 0:
                    state.update_stage_status(
                        run_id,
                        target_label,
                        stage,
                        STATUS_SUCCESS,
                        finished_at=_utc_now(),
                        exit_code=0,
                        log_path=log_path,
                    )
                else:
                    state.update_stage_status(
                        run_id,
                        target_label,
                        stage,
                        STATUS_FAILED,
                        finished_at=_utc_now(),
                        exit_code=ret,
                        log_path=log_path,
                        error_tail=error_tail,
                        pid=None,
                    )
                    state.block_downstream(run_id, target_label, stage)
                done_keys.append(key)
            for key in done_keys:
                running.pop(key, None)
                running_meta.pop(key, None)

            _promote_ready_stages_subset(state, run_id, active_stages, targets, cfg)

            # Launch new work
            for pool_name, pool_cfg in cfg.resources.items():
                capacity = pool_cfg.max_concurrent - pool_running[pool_name]
                if capacity <= 0:
                    continue
                batch = state.fetch_ready_batch(run_id, pool_name, capacity, active_stages)
                for row in batch:
                    cmd = stages.build_stage_command(
                        run_id,
                        row.stage,
                        config_path,
                        targets_path,
                        row.target_label,
                        force_rerun=force_rerun,
                    )
                    executor = cfg.stage_executor(row.stage)
                    log.info(
                        "Launching %s / %s (%s, %s)",
                        row.target_label,
                        row.stage,
                        pool_name,
                        executor,
                    )
                    handle, job_id = launcher.launch_stage(
                        cmd,
                        cfg=cfg,
                        stage=row.stage,
                        runs_root=runs_root,
                        run_id=run_id,
                        target_label=row.target_label,
                    )
                    key = f"{row.target_label}:{row.stage}"
                    running[key] = handle
                    running_meta[key] = (row.target_label, row.stage, pool_name)
                    pool_running[pool_name] += 1
                    log_path = str(
                        logs.target_log_path(runs_root, run_id, row.target_label, row.stage)
                    )
                    state.update_stage_status(
                        run_id,
                        row.target_label,
                        row.stage,
                        STATUS_RUNNING,
                        started_at=_utc_now(),
                        log_path=log_path,
                        pid=job_id,
                    )

            _write_summary(state, run_id, runs_root)
            logs.update_run_meta(
                runs_root,
                run_id,
                {"last_heartbeat": _utc_now(), "scheduler_pid": os.getpid()},
            )

            counts = state.count_by_status(run_id)
            pending_like = sum(
                counts.get(s, 0)
                for s in (STATUS_PENDING, STATUS_READY, STATUS_RUNNING, "queued")
            )
            if not running and pending_like == 0:
                final = "success" if counts.get(STATUS_FAILED, 0) == 0 else "failed"
                state.set_run_status(run_id, final)
                log.info("Run complete: %s", final)
                break

            if time.monotonic() - last_heartbeat > cfg.scheduler_heartbeat_interval_s:
                last_heartbeat = time.monotonic()

            time.sleep(1.0)
    finally:
        for handle in running.values():
            if handle.poll() is None:
                handle.terminate()
        daemon.remove_pid_file(pid_path)
        _write_summary(state, run_id, runs_root)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Template pipeline scheduler")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--stages", default=None)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run stages even when output artifacts already exist",
    )
    args = parser.parse_args(argv)
    return run_scheduler(
        args.run_id,
        args.config,
        args.targets,
        args.stages,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    raise SystemExit(main())
