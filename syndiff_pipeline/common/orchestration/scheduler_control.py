"""Supervisor daemon lifecycle helpers."""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.state import PipelineState
from syndiff_pipeline.common.orchestration.workspace import (
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
    host: str | None = None


@dataclass(frozen=True)
class StopDaemonResult:
    pid: int | None
    was_running: bool
    stopped: bool
    force_killed: bool
    message: str | None = None


def _parse_heartbeat(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _local_heartbeat_age_s(workspace_root: str) -> float | None:
    path = logs.daemon_heartbeat_file(workspace_root)
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


def _db_heartbeat_age_s(workspace_root: str) -> float | None:
    state = PipelineState(str(state_db_path(workspace_root)))
    row = state.get_supervisor_status()
    if not row:
        return None
    heartbeat = _parse_heartbeat(row.get("last_heartbeat"))
    if heartbeat is None:
        return None
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - heartbeat).total_seconds()


def daemon_heartbeat_age_s(workspace_root: str) -> float | None:
    local = _local_heartbeat_age_s(workspace_root)
    if local is not None:
        return local
    return _db_heartbeat_age_s(workspace_root)


def _local_heartbeat_exists(workspace_root: str) -> bool:
    return logs.daemon_heartbeat_file(workspace_root).is_file()


def _clear_daemon_liveness(workspace_root: str) -> None:
    try:
        logs.daemon_heartbeat_file(workspace_root).unlink(missing_ok=True)
    except OSError:
        pass
    try:
        PipelineState(str(state_db_path(workspace_root))).clear_supervisor()
    except OSError:
        pass


def get_supervisor_host(workspace_root: str) -> str | None:
    """Return the recorded supervisor host (pid file, then SQLite)."""
    pid_path = logs.daemon_pid_path(workspace_root)
    host, _pid = daemon.read_process_identity(pid_path)
    if host:
        return host
    try:
        row = PipelineState(str(state_db_path(workspace_root))).get_supervisor_status()
    except OSError:
        return None
    if not row:
        return None
    db_host = row.get("host")
    return str(db_host) if db_host else None


def _supervisor_pid_identity(workspace_root: str) -> tuple[str | None, int | None]:
    pid_path = logs.daemon_pid_path(workspace_root)
    host, pid = daemon.read_process_identity(pid_path)
    if host or pid:
        return host, pid
    try:
        row = PipelineState(str(state_db_path(workspace_root))).get_supervisor_status()
    except OSError:
        return None, None
    if not row:
        return None, None
    db_host = row.get("host")
    db_pid = row.get("pid")
    return (
        str(db_host) if db_host else None,
        int(db_pid) if db_pid is not None else None,
    )


def daemon_is_alive(
    workspace_root: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    host, pid = _supervisor_pid_identity(workspace_root)
    if daemon.is_local_process_alive(host, pid):
        age = daemon_heartbeat_age_s(workspace_root)
        if age is None or age <= stale_after_s:
            return True
    if _local_heartbeat_exists(workspace_root):
        return False
    age = _db_heartbeat_age_s(workspace_root)
    if age is not None and age <= stale_after_s:
        return True
    if pid and host and not daemon.identity_on_local_host(host):
        return True
    return False


def daemon_is_wedged(
    workspace_root: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    host, pid = daemon.read_process_identity(logs.daemon_pid_path(workspace_root))
    if not daemon.is_local_process_alive(host, pid):
        return False
    age = daemon_heartbeat_age_s(workspace_root)
    return age is None or age > stale_after_s


def _remote_supervisor_running_message(workspace_root: str) -> str:
    host = get_supervisor_host(workspace_root) or "unknown"
    _host, pid = _supervisor_pid_identity(workspace_root)
    local = daemon.local_hostname()
    pid_text = f" (pid={pid})" if pid else ""
    return (
        f"Supervisor already running on host {host!r}{pid_text}. "
        f"This machine is {local!r}. SSH to {host} to manage the daemon, or stop it there first."
    )


def warn_if_daemon_host_mismatch(workspace_root: str) -> None:
    """Warn when the CLI host differs from the supervisor daemon host (SQLite WAL risk)."""
    daemon_host = get_supervisor_host(workspace_root)
    if not daemon_host:
        return
    local = daemon.local_hostname()
    if daemon.hosts_match(local, daemon_host):
        return
    print(
        f"WARNING: supervisor daemon is on {daemon_host!r} but this CLI is on {local!r}. "
        "SQLite WAL mode is unsafe across NFS clients; run CLI commands on the daemon host.",
        file=sys.stderr,
    )


def daemon_status(workspace_root: str) -> daemon.DaemonStatus:
    host, pid = _supervisor_pid_identity(workspace_root)
    age = daemon_heartbeat_age_s(workspace_root)
    alive = daemon_is_alive(workspace_root)
    lock_held = False
    with daemon.daemon_lock(workspace_root, blocking=False) as fd:
        lock_held = fd is None
    return daemon.DaemonStatus(
        alive=alive,
        pid=pid,
        heartbeat_age_s=age,
        lock_held=lock_held,
        host=host or get_supervisor_host(workspace_root),
    )


def _resolve_deployment_for_spawn(
    workspace_root: str,
    deployment_path: str | Path | None,
) -> Path:
    if deployment_path is not None:
        return Path(deployment_path).expanduser().resolve()
    recorded = load_recorded_deployment_path(workspace_root)
    if recorded is not None:
        return recorded
    raise RuntimeError(
        "Cannot spawn supervisor: no deployment.yaml recorded for this workspace. "
        "Submit a run first or use: syndiff daemon start --deployment PATH"
    )


def ensure_daemon_running(
    workspace_root: str,
    *,
    deployment_path: str | Path | None = None,
) -> EnsureDaemonResult:
    """Start detached supervisor daemon if not alive (flock-guarded by the daemon)."""
    if daemon_is_alive(workspace_root):
        host = get_supervisor_host(workspace_root)
        _host, pid = _supervisor_pid_identity(workspace_root)
        if host and not daemon.identity_on_local_host(host):
            raise RuntimeError(_remote_supervisor_running_message(workspace_root))
        return EnsureDaemonResult(spawned=False, pid=pid, host=host)

    if daemon_is_wedged(workspace_root):
        stop_daemon(workspace_root)

    deploy_path = _resolve_deployment_for_spawn(workspace_root, deployment_path)
    daemon_log = logs.daemon_log_path(workspace_root)
    spawn_pid = daemon.spawn_detached_daemon(deploy_path, daemon_log)
    if daemon.wait_for_daemon(workspace_root):
        owner_host, owner_pid = daemon.read_process_identity(logs.daemon_pid_path(workspace_root))
        spawned = owner_pid == spawn_pid
        return EnsureDaemonResult(
            spawned=spawned,
            pid=owner_pid or spawn_pid,
            host=owner_host or daemon.local_hostname(),
        )

    if daemon_is_alive(workspace_root):
        host = get_supervisor_host(workspace_root)
        _host, pid = _supervisor_pid_identity(workspace_root)
        if host and not daemon.identity_on_local_host(host):
            raise RuntimeError(_remote_supervisor_running_message(workspace_root))
        return EnsureDaemonResult(spawned=False, pid=pid, host=host)
    raise RuntimeError(f"Supervisor daemon pid={spawn_pid} failed to start")


def stop_daemon(
    workspace_root: str,
    *,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> StopDaemonResult:
    """Stop the supervisor daemon, escalating to SIGKILL if SIGTERM is ignored."""
    pid_path = logs.daemon_pid_path(workspace_root)
    host, pid = daemon.read_process_identity(pid_path)
    if host and not daemon.identity_on_local_host(host):
        age = _db_heartbeat_age_s(workspace_root)
        if age is not None and age <= DEFAULT_HEARTBEAT_STALE_S:
            return StopDaemonResult(
                pid=pid,
                was_running=True,
                stopped=False,
                force_killed=False,
                message=_remote_supervisor_running_message(workspace_root),
            )
        daemon.remove_pid_file(pid_path)
        _clear_daemon_liveness(workspace_root)
        return StopDaemonResult(
            pid=pid,
            was_running=False,
            stopped=True,
            force_killed=False,
        )

    if not pid or not daemon.is_process_alive(pid):
        if pid is not None:
            daemon.remove_pid_file(pid_path)
        _clear_daemon_liveness(workspace_root)
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
        _clear_daemon_liveness(workspace_root)
    return StopDaemonResult(
        pid=pid,
        was_running=True,
        stopped=stopped,
        force_killed=force_killed,
    )
