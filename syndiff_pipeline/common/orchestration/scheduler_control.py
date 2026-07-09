"""Supervisor daemon lifecycle helpers."""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.common.orchestration import daemon, lease, logs
from syndiff_pipeline.common.orchestration.state import PipelineState
from syndiff_pipeline.common.orchestration.workspace import (
    load_recorded_deployment_path,
    state_db_path,
)

DEFAULT_HEARTBEAT_STALE_S = lease.DEFAULT_LEASE_STALE_S
DEFAULT_STOP_TERM_TIMEOUT_S = 10.0
DEFAULT_STOP_KILL_WAIT_S = 5.0
DEFAULT_REMOTE_STOP_WAIT_S = lease.DEFAULT_REMOTE_STOP_WAIT_S


@dataclass(frozen=True)
class EnsureDaemonResult:
    """EnsureDaemonResult."""

    spawned: bool
    pid: int | None
    host: str | None = None


@dataclass(frozen=True)
class StopDaemonResult:
    """StopDaemonResult."""

    pid: int | None
    was_running: bool
    stopped: bool
    force_killed: bool
    lock_reclaimed: bool = False
    message: str | None = None


def _parse_heartbeat(value: str | None) -> datetime | None:
    """Parse heartbeat."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _local_heartbeat_age_s(workspace_root: str) -> float | None:
    """Local heartbeat age in seconds."""
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
    """DB heartbeat age in seconds (best-effort visibility)."""
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
    """Prefer lease age, then local heartbeat, then DB heartbeat."""
    owned = lease.read_lease(workspace_root)
    if owned is not None:
        age = owned.age_s()
        if age is not None:
            return age
    local = _local_heartbeat_age_s(workspace_root)
    if local is not None:
        return local
    return _db_heartbeat_age_s(workspace_root)


def _local_heartbeat_exists(workspace_root: str) -> bool:
    """True when the host-local heartbeat file exists."""
    return logs.daemon_heartbeat_file(workspace_root).is_file()


def _reclaim_stale_lock_if_local(workspace_root: str) -> bool:
    """Best-effort reclaim of a same-host flock; never overrides lease decisions."""
    owned = lease.read_lease(workspace_root)
    if owned is not None and not daemon.identity_on_local_host(owned.host):
        if owned.is_fresh():
            return False
    host = get_supervisor_host(workspace_root)
    if host and not daemon.identity_on_local_host(host):
        return False
    return daemon.reclaim_stale_daemon_lock(workspace_root)


def _clear_daemon_liveness(workspace_root: str) -> None:
    """Clear local heartbeat and SQLite supervisor row."""
    try:
        logs.daemon_heartbeat_file(workspace_root).unlink(missing_ok=True)
    except OSError:
        pass
    try:
        PipelineState(str(state_db_path(workspace_root))).clear_supervisor()
    except OSError:
        pass


def get_supervisor_host(workspace_root: str) -> str | None:
    """Return the recorded supervisor host (lease, then pid file, then SQLite)."""
    owned = lease.read_lease(workspace_root)
    if owned is not None:
        return owned.host
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
    """Supervisor host/pid from lease, then pid file, then SQLite."""
    owned = lease.read_lease(workspace_root)
    if owned is not None:
        return owned.host, owned.pid
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
    """True when a fresh ownership lease exists (cross-host source of truth).

    Same-host: also require the recorded pid to still be alive when the lease
    is local, so a dead process with a leftover fresh-looking file is not
    treated as alive (fast reclaim path uses lease_is_reclaimable).
    """
    owned = lease.read_lease(workspace_root)
    if owned is None:
        # Legacy fallback while upgrading: local pid + heartbeat.
        host, pid = _supervisor_pid_identity(workspace_root)
        if daemon.is_local_process_alive(host, pid):
            age = _local_heartbeat_age_s(workspace_root)
            if age is None or age <= stale_after_s:
                return True
        return False

    if not owned.is_fresh(stale_after_s=stale_after_s):
        return False

    if daemon.identity_on_local_host(owned.host):
        if not daemon.is_process_alive(owned.pid):
            return False
        # Local process alive but host-local heartbeat missing/stale → wedged,
        # not "alive" for ensure/spawn purposes.
        if _local_heartbeat_exists(workspace_root):
            age = _local_heartbeat_age_s(workspace_root)
            if age is not None and age > stale_after_s:
                return False
        return True

    # Remote fresh lease: trust it (cannot probe remote pid).
    return True


def daemon_is_wedged(
    workspace_root: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    """True when a local supervisor pid is alive but liveness signals are stale."""
    host, pid = _supervisor_pid_identity(workspace_root)
    if not daemon.is_local_process_alive(host, pid):
        return False
    owned = lease.read_lease(workspace_root)
    if owned is not None and owned.is_ours(host=host or daemon.local_hostname(), pid=pid):
        if not owned.is_fresh(stale_after_s=stale_after_s):
            return True
    age = daemon_heartbeat_age_s(workspace_root)
    return age is None or age > stale_after_s


def _remote_supervisor_running_message(workspace_root: str) -> str:
    """Human-readable message when a remote supervisor owns the workspace."""
    host = get_supervisor_host(workspace_root) or "unknown"
    _host, pid = _supervisor_pid_identity(workspace_root)
    local = daemon.local_hostname()
    pid_text = f" (pid={pid})" if pid else ""
    return (
        f"Supervisor already running on host {host!r}{pid_text}. "
        f"This machine is {local!r}. "
        f"Use `syndiff daemon stop` from any machine to request a remote stop, "
        f"then start again on the host you want."
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
    """Daemon status including lease fields."""
    host, pid = _supervisor_pid_identity(workspace_root)
    age = daemon_heartbeat_age_s(workspace_root)
    alive = daemon_is_alive(workspace_root)
    wedged = daemon_is_wedged(workspace_root)
    owned = lease.read_lease(workspace_root)
    stop = lease.read_stop_request(workspace_root)
    lock_held = False
    # Best-effort flock probe; lease is authoritative.
    try:
        with daemon.daemon_lock(workspace_root, blocking=False) as fd:
            lock_held = fd is None
    except OSError:
        lock_held = False
    return daemon.DaemonStatus(
        alive=alive,
        pid=pid,
        heartbeat_age_s=age,
        lock_held=lock_held,
        host=host or get_supervisor_host(workspace_root),
        lease_generation=owned.generation if owned is not None else None,
        lease_age_s=owned.age_s() if owned is not None else None,
        stop_pending=stop is not None,
        wedged=wedged,
    )


def _resolve_deployment_for_spawn(
    workspace_root: str,
    deployment_path: str | Path | None,
) -> Path:
    """Resolve deployment.yaml path for spawning the supervisor."""
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
    """Start detached supervisor daemon if not alive (lease-guarded by the daemon)."""
    if daemon_is_alive(workspace_root):
        host = get_supervisor_host(workspace_root)
        _host, pid = _supervisor_pid_identity(workspace_root)
        if host and not daemon.identity_on_local_host(host):
            raise RuntimeError(_remote_supervisor_running_message(workspace_root))
        return EnsureDaemonResult(spawned=False, pid=pid, host=host)

    stop = lease.read_stop_request(workspace_root)
    owned = lease.read_lease(workspace_root)
    if (
        stop is not None
        and owned is not None
        and owned.is_fresh()
        and lease.stop_targets_owner(stop, owned)
        and daemon_is_alive(workspace_root)
    ):
        raise RuntimeError(
            "Supervisor stop is in progress; wait for the current owner to exit "
            "before starting a new daemon."
        )

    if daemon_is_wedged(workspace_root):
        stop_daemon(workspace_root)

    deploy_path = _resolve_deployment_for_spawn(workspace_root, deployment_path)
    daemon_log = logs.daemon_log_path(workspace_root)
    spawn_pid = daemon.spawn_detached_daemon(deploy_path, daemon_log)
    if daemon.wait_for_daemon(workspace_root):
        owner_host, owner_pid = _supervisor_pid_identity(workspace_root)
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
    remote_wait_s: float = DEFAULT_REMOTE_STOP_WAIT_S,
) -> StopDaemonResult:
    """Stop the supervisor via stop request (+ local signals when on this host)."""
    pid_path = logs.daemon_pid_path(workspace_root)
    owned = lease.read_lease(workspace_root)
    host, pid = _supervisor_pid_identity(workspace_root)
    target_generation = owned.generation if owned is not None else None

    was_running = daemon_is_alive(workspace_root) or (
        owned is not None and owned.is_fresh()
    )

    lease.write_stop_request(workspace_root, target_generation=target_generation)

    if (
        (not host or daemon.identity_on_local_host(host))
        and pid
        and daemon.is_process_alive(pid)
    ):
        daemon.terminate_process_tree(pid, signal.SIGTERM)
        force_killed = False
        if not daemon.wait_for_process_exit(pid, timeout_s=term_timeout_s):
            daemon.terminate_process_tree(pid, signal.SIGKILL)
            force_killed = True
            daemon.wait_for_process_exit(pid, timeout_s=kill_wait_s)

        stopped = not daemon.is_process_alive(pid)
        lock_reclaimed = False
        if stopped:
            daemon.remove_pid_file(pid_path)
            _clear_daemon_liveness(workspace_root)
            lease.release_lease(
                workspace_root,
                host=host,
                pid=pid,
                generation=target_generation,
            )
            lease.clear_stop_request(workspace_root, only_generation=target_generation)
            lock_reclaimed = _reclaim_stale_lock_if_local(workspace_root)
        return StopDaemonResult(
            pid=pid,
            was_running=True,
            stopped=stopped,
            force_killed=force_killed,
            lock_reclaimed=lock_reclaimed,
            message=None if stopped else "Supervisor did not exit after SIGKILL",
        )

    # Remote (or already-dead) owner: wait for lease release / stale.
    released = lease.wait_until_lease_released(
        workspace_root,
        target_generation=target_generation,
        timeout_s=remote_wait_s,
    )
    if released:
        daemon.remove_pid_file(pid_path)
        _clear_daemon_liveness(workspace_root)
        # Clear lease file if still present but reclaimable (stale / dead pid).
        leftover = lease.read_lease(workspace_root)
        if leftover is not None and lease.lease_is_reclaimable(workspace_root):
            lease.ensure_control_files(workspace_root)
            lease._atomic_write_json(
                logs.daemon_lease_path(workspace_root),
                dict(lease._LEASE_RELEASED_PAYLOAD),
            )
        lease.clear_stop_request(workspace_root, only_generation=target_generation)
        lock_reclaimed = _reclaim_stale_lock_if_local(workspace_root)
        return StopDaemonResult(
            pid=pid,
            was_running=was_running,
            stopped=True,
            force_killed=False,
            lock_reclaimed=lock_reclaimed,
        )

    still = lease.read_lease(workspace_root)
    msg = (
        f"Timed out waiting for supervisor on host "
        f"{(still.host if still else host) or 'unknown'!r} "
        f"to honor stop request (generation={target_generation})."
    )
    return StopDaemonResult(
        pid=pid,
        was_running=True,
        stopped=False,
        force_killed=False,
        message=msg,
    )
