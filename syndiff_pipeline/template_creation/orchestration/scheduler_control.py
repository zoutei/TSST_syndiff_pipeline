"""Supervisor daemon lifecycle helpers."""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.template_creation.orchestration import daemon, logs
from syndiff_pipeline.template_creation.orchestration.state import PipelineState
from syndiff_pipeline.template_creation.orchestration.workspace import (
    load_recorded_deployment_path,
    state_db_path,
)

DEFAULT_HEARTBEAT_STALE_S = 120.0
DEFAULT_STOP_TERM_TIMEOUT_S = 10.0
DEFAULT_STOP_KILL_WAIT_S = 5.0


@dataclass(frozen=True)
class EnsureDaemonResult:
    spawned: bool
    pid: int | None


@dataclass(frozen=True)
class StopDaemonResult:
    pid: int | None
    was_running: bool
    stopped: bool
    force_killed: bool


def _parse_heartbeat(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _local_heartbeat_age_s(handoff_root: str) -> float | None:
    path = logs.daemon_heartbeat_file(handoff_root)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    try:
        written = float(text)
    except ValueError:
        try:
            written = path.stat().st_mtime
        except OSError:
            return None
    return max(0.0, time.time() - written)


def _db_heartbeat_age_s(handoff_root: str) -> float | None:
    state = PipelineState(str(state_db_path(handoff_root)))
    row = state.get_supervisor_status()
    if not row:
        return None
    heartbeat = _parse_heartbeat(row.get("last_heartbeat"))
    if heartbeat is None:
        return None
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - heartbeat).total_seconds()


def daemon_heartbeat_age_s(handoff_root: str) -> float | None:
    local = _local_heartbeat_age_s(handoff_root)
    if local is not None:
        return local
    return _db_heartbeat_age_s(handoff_root)


def _local_heartbeat_exists(handoff_root: str) -> bool:
    return logs.daemon_heartbeat_file(handoff_root).is_file()


def _clear_daemon_liveness(handoff_root: str) -> None:
    try:
        logs.daemon_heartbeat_file(handoff_root).unlink(missing_ok=True)
    except OSError:
        pass
    try:
        PipelineState(str(state_db_path(handoff_root))).clear_supervisor()
    except OSError:
        pass


def daemon_is_alive(
    handoff_root: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    pid_path = logs.daemon_pid_path(handoff_root)
    pid = daemon.read_pid(pid_path)
    if pid and daemon.is_process_alive(pid):
        age = daemon_heartbeat_age_s(handoff_root)
        if age is None or age <= stale_after_s:
            return True
    if _local_heartbeat_exists(handoff_root):
        return False
    age = _db_heartbeat_age_s(handoff_root)
    if age is not None and age <= stale_after_s:
        return True
    return False


def daemon_is_wedged(
    handoff_root: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    pid = daemon.read_pid(logs.daemon_pid_path(handoff_root))
    if not pid or not daemon.is_process_alive(pid):
        return False
    age = daemon_heartbeat_age_s(handoff_root)
    return age is None or age > stale_after_s


def daemon_status(handoff_root: str) -> daemon.DaemonStatus:
    pid_path = logs.daemon_pid_path(handoff_root)
    pid = daemon.read_pid(pid_path)
    age = daemon_heartbeat_age_s(handoff_root)
    alive = daemon_is_alive(handoff_root)
    lock_held = False
    with daemon.daemon_lock(handoff_root, blocking=False) as fd:
        lock_held = fd is None
    return daemon.DaemonStatus(
        alive=alive,
        pid=pid,
        heartbeat_age_s=age,
        lock_held=lock_held,
    )


def _resolve_deployment_for_spawn(
    handoff_root: str,
    deployment_path: str | Path | None,
) -> Path:
    if deployment_path is not None:
        return Path(deployment_path).expanduser().resolve()
    recorded = load_recorded_deployment_path(handoff_root)
    if recorded is not None:
        return recorded
    raise RuntimeError(
        "Cannot spawn supervisor: no deployment.yaml recorded for this workspace. "
        "Submit a run first or use: syndiff-template daemon start --deployment PATH"
    )


def ensure_daemon_running(
    handoff_root: str,
    *,
    deployment_path: str | Path | None = None,
) -> EnsureDaemonResult:
    """Start detached supervisor daemon if not alive (flock-guarded by the daemon)."""
    if daemon_is_alive(handoff_root):
        pid = daemon.read_pid(logs.daemon_pid_path(handoff_root))
        return EnsureDaemonResult(spawned=False, pid=pid)

    if daemon_is_wedged(handoff_root):
        stop_daemon(handoff_root)

    deploy_path = _resolve_deployment_for_spawn(handoff_root, deployment_path)
    daemon_log = logs.daemon_log_path(handoff_root)
    spawn_pid = daemon.spawn_detached_daemon(deploy_path, daemon_log)
    if daemon.wait_for_daemon(handoff_root):
        owner_pid = daemon.read_pid(logs.daemon_pid_path(handoff_root))
        spawned = owner_pid == spawn_pid
        return EnsureDaemonResult(spawned=spawned, pid=owner_pid or spawn_pid)

    if daemon_is_alive(handoff_root):
        pid = daemon.read_pid(logs.daemon_pid_path(handoff_root))
        return EnsureDaemonResult(spawned=False, pid=pid)
    raise RuntimeError(f"Supervisor daemon pid={spawn_pid} failed to start")


def stop_daemon(
    handoff_root: str,
    *,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> StopDaemonResult:
    """Stop the supervisor daemon, escalating to SIGKILL if SIGTERM is ignored."""
    pid_path = logs.daemon_pid_path(handoff_root)
    pid = daemon.read_pid(pid_path)
    if not pid or not daemon.is_process_alive(pid):
        if pid is not None:
            daemon.remove_pid_file(pid_path)
        _clear_daemon_liveness(handoff_root)
        return StopDaemonResult(
            pid=pid,
            was_running=False,
            stopped=True,
            force_killed=False,
        )

    daemon.terminate_process_tree(pid, signal.SIGTERM)
    force_killed = False
    if not daemon.wait_for_process_exit(pid, timeout_s=term_timeout_s):
        daemon.terminate_process_tree(pid, signal.SIGKILL)
        force_killed = True
        daemon.wait_for_process_exit(pid, timeout_s=kill_wait_s)

    stopped = not daemon.is_process_alive(pid)
    if stopped:
        daemon.remove_pid_file(pid_path)
        _clear_daemon_liveness(handoff_root)
    return StopDaemonResult(
        pid=pid,
        was_running=True,
        stopped=stopped,
        force_killed=force_killed,
    )
