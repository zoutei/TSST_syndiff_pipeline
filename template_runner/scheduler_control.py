"""Supervisor daemon lifecycle helpers."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from datetime import datetime, timezone

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.state import PipelineState

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


def daemon_heartbeat_age_s(state_db_path: str) -> float | None:
    state = PipelineState(state_db_path)
    row = state.get_supervisor_status()
    if not row:
        return None
    heartbeat = _parse_heartbeat(row.get("last_heartbeat"))
    if heartbeat is None:
        return None
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - heartbeat).total_seconds()


def daemon_is_alive(
    state_db_path: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    pid_path = logs.daemon_pid_path(state_db_path)
    pid = daemon.read_pid(pid_path)
    if pid and daemon.is_process_alive(pid):
        age = daemon_heartbeat_age_s(state_db_path)
        if age is None or age <= stale_after_s:
            return True
    age = daemon_heartbeat_age_s(state_db_path)
    if age is not None and age <= stale_after_s:
        return True
    return False


def daemon_status(state_db_path: str) -> daemon.DaemonStatus:
    pid_path = logs.daemon_pid_path(state_db_path)
    pid = daemon.read_pid(pid_path)
    age = daemon_heartbeat_age_s(state_db_path)
    alive = daemon_is_alive(state_db_path)
    lock_held = False
    with daemon.daemon_lock(state_db_path, blocking=False) as fd:
        lock_held = fd is None
    return daemon.DaemonStatus(
        alive=alive,
        pid=pid,
        heartbeat_age_s=age,
        lock_held=lock_held,
    )


def ensure_daemon_running(state_db_path: str) -> EnsureDaemonResult:
    """Start detached supervisor daemon if not alive (flock-guarded by the daemon).

    The single-owner guarantee lives in the daemon itself: it acquires an
    exclusive ``flock`` and exits immediately if another owner holds it. We must
    NOT hold that lock here while spawning, or the child would block on it. If
    two CLIs race to spawn, only one daemon wins the lock; the loser exits.
    """
    if daemon_is_alive(state_db_path):
        pid = daemon.read_pid(logs.daemon_pid_path(state_db_path))
        return EnsureDaemonResult(spawned=False, pid=pid)

    daemon_log = logs.daemon_log_path(state_db_path)
    spawn_pid = daemon.spawn_detached_daemon(state_db_path, daemon_log)
    if daemon.wait_for_daemon(state_db_path):
        # The winning daemon writes its own pid after acquiring the lock.
        owner_pid = daemon.read_pid(logs.daemon_pid_path(state_db_path))
        spawned = owner_pid == spawn_pid
        return EnsureDaemonResult(spawned=spawned, pid=owner_pid or spawn_pid)

    # Our spawn may have lost the lock race to a concurrent starter that is now
    # the live owner; accept that as success.
    if daemon_is_alive(state_db_path):
        pid = daemon.read_pid(logs.daemon_pid_path(state_db_path))
        return EnsureDaemonResult(spawned=False, pid=pid)
    raise RuntimeError(f"Supervisor daemon pid={spawn_pid} failed to start")


def stop_daemon(
    state_db_path: str,
    *,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> StopDaemonResult:
    """Stop the supervisor daemon, escalating to SIGKILL if SIGTERM is ignored."""
    pid_path = logs.daemon_pid_path(state_db_path)
    pid = daemon.read_pid(pid_path)
    if not pid or not daemon.is_process_alive(pid):
        if pid is not None:
            daemon.remove_pid_file(pid_path)
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
    return StopDaemonResult(
        pid=pid,
        was_running=True,
        stopped=stopped,
        force_killed=force_killed,
    )
