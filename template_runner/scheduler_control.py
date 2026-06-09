"""Supervisor daemon lifecycle helpers."""

from __future__ import annotations

import signal
import time
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


def _local_heartbeat_age_s(state_db_path: str) -> float | None:
    """Age of the host-local heartbeat file, or None if missing/unreadable.

    This is the authoritative liveness signal: it lives on local disk and so is
    independent of the (possibly wedged or full) NFS state DB. Reading it never
    opens the SQLite database, keeping ``status``/``start``/``stop`` responsive
    even when NFS is degraded.
    """
    path = logs.daemon_heartbeat_file(state_db_path)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return None
    try:
        written = float(text)
    except ValueError:
        # Fall back to mtime if the contents are unparseable.
        try:
            written = path.stat().st_mtime
        except OSError:
            return None
    return max(0.0, time.time() - written)


def _db_heartbeat_age_s(state_db_path: str) -> float | None:
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


def daemon_heartbeat_age_s(state_db_path: str) -> float | None:
    local = _local_heartbeat_age_s(state_db_path)
    if local is not None:
        return local
    # No local heartbeat (e.g. daemon never ran on this host): fall back to the
    # DB heartbeat for cross-host visibility.
    return _db_heartbeat_age_s(state_db_path)


def _local_heartbeat_exists(state_db_path: str) -> bool:
    return logs.daemon_heartbeat_file(state_db_path).is_file()


def _clear_daemon_liveness(state_db_path: str) -> None:
    """Drop liveness artifacts after an intentional supervisor stop."""
    try:
        logs.daemon_heartbeat_file(state_db_path).unlink(missing_ok=True)
    except OSError:
        pass
    try:
        PipelineState(state_db_path).clear_supervisor()
    except OSError:
        pass


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
    # A host-local heartbeat file means this machine owns (or owned) the
    # supervisor. Without a live pid, stale heartbeats must not read as alive
    # (e.g. after SIGKILL before cleanup, or between stop and heartbeat expiry).
    if _local_heartbeat_exists(state_db_path):
        return False
    # Cross-host status: no local heartbeat here; trust a fresh DB heartbeat.
    age = _db_heartbeat_age_s(state_db_path)
    if age is not None and age <= stale_after_s:
        return True
    return False


def daemon_is_wedged(
    state_db_path: str,
    *,
    stale_after_s: float = DEFAULT_HEARTBEAT_STALE_S,
) -> bool:
    """True when a supervisor process exists but its heartbeat is stale.

    With a background heartbeat thread writing a host-LOCAL file, a merely-busy
    supervisor (slow NFS verification on the main thread) still emits fresh
    heartbeats and is NOT wedged. A stale heartbeat despite a live pid therefore
    means the process is truly hung (e.g. uninterruptible NFS I/O) and should be
    force-replaced rather than left holding the lock while looking dead.
    """
    pid = daemon.read_pid(logs.daemon_pid_path(state_db_path))
    if not pid or not daemon.is_process_alive(pid):
        return False
    age = daemon_heartbeat_age_s(state_db_path)
    return age is None or age > stale_after_s


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

    # A live pid with a stale heartbeat is a truly-hung owner. Force-replace it
    # so a fresh supervisor can take over (reconcile recovers in-flight jobs
    # from durable status files), instead of refusing to start because the lock
    # is held. This is the automatic recovery for the "wedged" failure mode.
    if daemon_is_wedged(state_db_path):
        stop_daemon(state_db_path)

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
        _clear_daemon_liveness(state_db_path)
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
        _clear_daemon_liveness(state_db_path)
    return StopDaemonResult(
        pid=pid,
        was_running=True,
        stopped=stopped,
        force_killed=force_killed,
    )
