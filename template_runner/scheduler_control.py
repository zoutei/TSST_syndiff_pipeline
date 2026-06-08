"""Scheduler process lifecycle helpers for submit and retry."""

from __future__ import annotations

from dataclasses import dataclass

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.run_context import RunContext
from syndiff_pipeline.template_runner.state import PipelineState


@dataclass(frozen=True)
class EnsureSchedulerResult:
    spawned: bool
    pid: int | None


def scheduler_is_alive(runs_root: str, run_id: str) -> bool:
    pid_path = logs.scheduler_pid_path(runs_root, run_id)
    pid = daemon.read_pid(pid_path)
    return bool(pid and daemon.is_process_alive(pid))


def _stages_arg(meta: dict) -> str | None:
    stages = meta.get("stages")
    if not stages:
        return None
    if isinstance(stages, str):
        return stages
    return ",".join(stages)


def ensure_scheduler_running(
    ctx: RunContext,
    *,
    force_rerun: bool = False,
) -> EnsureSchedulerResult:
    """Start detached scheduler if not alive; return existing or new PID."""
    runs_root = ctx.cfg.runs_dir()
    run_id = ctx.run_id
    pid_path = logs.scheduler_pid_path(runs_root, run_id)
    existing = daemon.read_pid(pid_path)
    if existing and daemon.is_process_alive(existing):
        return EnsureSchedulerResult(spawned=False, pid=existing)

    sched_log = logs.scheduler_log_path(runs_root, run_id)
    pid = daemon.spawn_detached_scheduler(
        run_id,
        ctx.run_dir,
        _stages_arg(ctx.meta),
        sched_log,
        force_rerun=force_rerun,
    )
    daemon.write_pid(pid_path, pid)
    logs.update_run_meta(runs_root, run_id, {"scheduler_pid": pid, "detach": True})
    PipelineState(ctx.cfg.state_db_path).set_run_status(run_id, "running")
    return EnsureSchedulerResult(spawned=True, pid=pid)
