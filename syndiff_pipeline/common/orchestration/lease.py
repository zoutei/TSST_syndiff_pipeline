"""NFS-safe workspace ownership lease and cross-host stop requests.

The lease file under ``{workspace_root}/control/`` is the authoritative
cross-host singleton for the supervisor. Flock is best-effort only and must
never override lease decisions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from syndiff_pipeline.common.orchestration import daemon, logs

DEFAULT_LEASE_STALE_S = 120.0
DEFAULT_LEASE_SETTLE_S = 0.75
DEFAULT_REMOTE_STOP_WAIT_S = 150.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _age_seconds(iso_ts: str | None) -> float | None:
    ts = _parse_iso(iso_ts)
    if ts is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via tmp+replace and fsync so NFS clients see content promptly.

    Important for NFS: once created, lease/stop paths should keep existing. Creating
    a brand-new filename is invisible to other clients that already cached ENOENT
    (negative dentry). Prefer ``ensure_control_files`` before first use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(str(tmp), str(path))
        # fsync the directory so the rename is durable / visible on NFS.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = -1
        if dir_fd >= 0:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON with NFS close-to-open semantics (open each call; no is_file probe)."""
    try:
        # O_RDONLY open forces attribute revalidation on many NFS clients.
        fd = os.open(str(path), os.O_RDONLY)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


# Marker written into daemon.lease when no owner holds the workspace. The file
# itself is never unlinked (NFS negative-cache safety).
_LEASE_RELEASED_PAYLOAD: dict[str, Any] = {
    "status": "released",
    "host": "",
    "pid": 0,
    "generation": 0,
    "started_at": "",
    "renewed_at": "",
}

_STOP_CLEARED_PAYLOAD: dict[str, Any] = {
    "status": "cleared",
    "requested_at": "",
    "requested_by_host": "",
    "target_generation": None,
}


def ensure_control_files(workspace_root: str | Path) -> None:
    """Create stable lease/stop files if missing so NFS clients never cache ENOENT."""
    lease_path = logs.daemon_lease_path(workspace_root)
    stop_path = logs.daemon_stop_path(workspace_root)
    if not lease_path.is_file():
        _atomic_write_json(lease_path, dict(_LEASE_RELEASED_PAYLOAD))
    if not stop_path.is_file():
        _atomic_write_json(stop_path, dict(_STOP_CLEARED_PAYLOAD))


def ensure_bot_lease_file(workspace_root: str | Path) -> None:
    """Create a stable Discord bot lease file if missing (NFS ENOENT safety)."""
    path = logs.discord_bot_lease_path(workspace_root)
    if not path.is_file():
        _atomic_write_json(path, dict(_LEASE_RELEASED_PAYLOAD))


def _read_lease_at(lease_path: Path) -> Lease | None:
    """Read a Lease from *lease_path*, or None if missing/released/corrupt."""
    data = _read_json(lease_path)
    if data is None:
        return None
    return Lease.from_dict(data)


def _write_lease_at(lease_path: Path, owned: Lease) -> None:
    """Atomically write *owned* to *lease_path* (file must already exist when possible)."""
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    if not lease_path.is_file():
        _atomic_write_json(lease_path, dict(_LEASE_RELEASED_PAYLOAD))
    _atomic_write_json(lease_path, owned.to_dict())


def _next_generation_at(lease_path: Path) -> int:
    """Return the next lease generation for *lease_path*, including released markers."""
    current = _read_lease_at(lease_path)
    if current is not None:
        return max(1, current.generation + 1)
    data = _read_json(lease_path) or {}
    for key in ("released_generation", "generation"):
        try:
            value = int(data.get(key, 0))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value + 1
    return 1


def _local_owner_pid_dead_for(owned: Lease) -> bool:
    """True when *owned* is for this host and the recorded pid is gone."""
    if not daemon.identity_on_local_host(owned.host):
        return False
    return not daemon.is_process_alive(owned.pid)


@dataclass(frozen=True)
class Lease:
    """Workspace ownership lease."""

    host: str
    pid: int
    generation: int
    started_at: str
    renewed_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lease | None:
        if str(data.get("status", "")).strip().lower() == "released":
            return None
        try:
            host = str(data["host"]).strip()
            pid = int(data["pid"])
            generation = int(data["generation"])
            started_at = str(data["started_at"])
            renewed_at = str(data["renewed_at"])
        except (KeyError, TypeError, ValueError):
            return None
        if not host or pid <= 0 or generation <= 0:
            return None
        return cls(
            host=host,
            pid=pid,
            generation=generation,
            started_at=started_at,
            renewed_at=renewed_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "pid": self.pid,
            "generation": self.generation,
            "started_at": self.started_at,
            "renewed_at": self.renewed_at,
        }

    def age_s(self) -> float | None:
        return _age_seconds(self.renewed_at)

    def is_fresh(self, *, stale_after_s: float = DEFAULT_LEASE_STALE_S) -> bool:
        age = self.age_s()
        return age is not None and age <= stale_after_s

    def matches(self, host: str, pid: int, generation: int) -> bool:
        return (
            daemon.hosts_match(self.host, host)
            and self.pid == pid
            and self.generation == generation
        )

    def is_ours(self, *, host: str | None = None, pid: int | None = None) -> bool:
        local_host = host if host is not None else daemon.local_hostname()
        local_pid = pid if pid is not None else os.getpid()
        return daemon.hosts_match(self.host, local_host) and self.pid == local_pid


@dataclass(frozen=True)
class StopRequest:
    """Cross-host stop request for a lease generation."""

    requested_at: str
    requested_by_host: str
    target_generation: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StopRequest | None:
        if str(data.get("status", "")).strip().lower() == "cleared":
            return None
        try:
            requested_at = str(data["requested_at"])
            requested_by_host = str(data["requested_by_host"]).strip()
        except (KeyError, TypeError, ValueError):
            return None
        if not requested_at or not requested_by_host:
            return None
        target_raw = data.get("target_generation")
        target: int | None
        if target_raw is None:
            target = None
        else:
            try:
                target = int(target_raw)
            except (TypeError, ValueError):
                return None
            if target <= 0:
                return None
        return cls(
            requested_at=requested_at,
            requested_by_host=requested_by_host,
            target_generation=target,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "requested_at": self.requested_at,
            "requested_by_host": self.requested_by_host,
        }
        if self.target_generation is not None:
            payload["target_generation"] = self.target_generation
        return payload


def read_lease(workspace_root: str | Path) -> Lease | None:
    """Read the current daemon lease, or None if missing/corrupt."""
    return _read_lease_at(logs.daemon_lease_path(workspace_root))


def write_lease_atomic(workspace_root: str | Path, lease: Lease) -> None:
    """Atomically write *lease* to the control directory."""
    ensure_control_files(workspace_root)
    _write_lease_at(logs.daemon_lease_path(workspace_root), lease)


def release_lease(
    workspace_root: str | Path,
    *,
    host: str | None = None,
    pid: int | None = None,
    generation: int | None = None,
) -> bool:
    """Mark the lease released without unlinking the file (NFS-safe).

    If *generation* is given, only clear a matching lease. Returns True when
    no live lease remains after the call.
    """
    ensure_control_files(workspace_root)
    current = read_lease(workspace_root)
    if current is None:
        _atomic_write_json(logs.daemon_lease_path(workspace_root), dict(_LEASE_RELEASED_PAYLOAD))
        return True
    local_host = host if host is not None else daemon.local_hostname()
    local_pid = pid if pid is not None else os.getpid()
    if not current.is_ours(host=local_host, pid=local_pid):
        return False
    if generation is not None and current.generation != generation:
        return False
    payload = dict(_LEASE_RELEASED_PAYLOAD)
    payload["released_at"] = _utc_now_iso()
    payload["released_by_host"] = local_host
    payload["released_generation"] = current.generation
    _atomic_write_json(logs.daemon_lease_path(workspace_root), payload)
    return read_lease(workspace_root) is None


def lease_is_fresh(
    workspace_root: str | Path,
    *,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
) -> bool:
    """True when a lease exists and was renewed within *stale_after_s*."""
    owned = read_lease(workspace_root)
    return owned is not None and owned.is_fresh(stale_after_s=stale_after_s)


def _local_owner_pid_dead(lease: Lease) -> bool:
    """True when lease is for this host and the recorded pid is gone."""
    return _local_owner_pid_dead_for(lease)


def lease_is_reclaimable(
    workspace_root: str | Path,
    *,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
) -> bool:
    """True when there is no live foreign owner blocking a new acquire."""
    owned = read_lease(workspace_root)
    if owned is None:
        return True
    if _local_owner_pid_dead(owned):
        return True
    return not owned.is_fresh(stale_after_s=stale_after_s)


def read_stop_request(workspace_root: str | Path) -> StopRequest | None:
    """Read a stop request, or None if missing/corrupt."""
    data = _read_json(logs.daemon_stop_path(workspace_root))
    if data is None:
        return None
    return StopRequest.from_dict(data)


def write_stop_request(
    workspace_root: str | Path,
    *,
    target_generation: int | None = None,
    requested_by_host: str | None = None,
) -> StopRequest:
    """Write a validated stop request for the current (or given) generation."""
    ensure_control_files(workspace_root)
    req = StopRequest(
        requested_at=_utc_now_iso(),
        requested_by_host=requested_by_host or daemon.local_hostname(),
        target_generation=target_generation,
    )
    _atomic_write_json(logs.daemon_stop_path(workspace_root), req.to_dict())
    return req


def clear_stop_request(
    workspace_root: str | Path,
    *,
    only_generation: int | None = None,
) -> bool:
    """Clear the stop request without unlinking the file (NFS-safe)."""
    ensure_control_files(workspace_root)
    if only_generation is not None:
        req = read_stop_request(workspace_root)
        if req is None:
            return True
        if req.target_generation is not None and req.target_generation != only_generation:
            return False
    payload = dict(_STOP_CLEARED_PAYLOAD)
    payload["cleared_at"] = _utc_now_iso()
    _atomic_write_json(logs.daemon_stop_path(workspace_root), payload)
    return read_stop_request(workspace_root) is None


def stop_targets_owner(req: StopRequest | None, lease: Lease | None) -> bool:
    """True when *req* should cause the owner of *lease* to shut down."""
    if req is None or lease is None:
        return False
    if req.target_generation is None:
        return True
    return req.target_generation == lease.generation


def _next_generation(workspace_root: str | Path) -> int:
    """Return the next lease generation, including across released markers."""
    return _next_generation_at(logs.daemon_lease_path(workspace_root))


def try_acquire_lease(
    workspace_root: str | Path,
    *,
    host: str | None = None,
    pid: int | None = None,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
    settle_s: float = DEFAULT_LEASE_SETTLE_S,
) -> Lease | None:
    """Attempt to become the workspace owner.

    Returns the acquired lease on success, or None when another fresh owner
    exists / wins the settle race / a stop is pending against a fresh lease.
    """
    local_host = host if host is not None else daemon.local_hostname()
    local_pid = pid if pid is not None else os.getpid()
    ensure_control_files(workspace_root)

    current = read_lease(workspace_root)
    if current is not None and current.is_ours(host=local_host, pid=local_pid):
        if current.is_fresh(stale_after_s=stale_after_s):
            return current

    if current is not None and current.is_fresh(stale_after_s=stale_after_s):
        if not _local_owner_pid_dead(current):
            return None

    stop = read_stop_request(workspace_root)
    if stop is not None and current is not None and current.is_fresh(stale_after_s=stale_after_s):
        if stop_targets_owner(stop, current) and not _local_owner_pid_dead(current):
            return None

    # Stale stop against a dead/stale lease should not block acquire.
    if stop is not None and (current is None or not current.is_fresh(stale_after_s=stale_after_s) or _local_owner_pid_dead(current)):
        clear_stop_request(workspace_root)

    now = _utc_now_iso()
    next_generation = _next_generation(workspace_root)
    candidate = Lease(
        host=local_host,
        pid=local_pid,
        generation=next_generation,
        started_at=now,
        renewed_at=now,
    )
    write_lease_atomic(workspace_root, candidate)
    if settle_s > 0:
        time.sleep(settle_s)
    confirmed = read_lease(workspace_root)
    if confirmed is None:
        return None
    if not confirmed.matches(local_host, local_pid, candidate.generation):
        return None
    return confirmed


def renew_lease(
    workspace_root: str | Path,
    lease: Lease,
    *,
    host: str | None = None,
    pid: int | None = None,
) -> Lease | None:
    """Renew *lease* without changing generation.

    Returns the renewed lease, or None if ownership was lost (generation/host/pid
    mismatch) or the write failed.
    """
    local_host = host if host is not None else daemon.local_hostname()
    local_pid = pid if pid is not None else os.getpid()
    current = read_lease(workspace_root)
    if current is None:
        return None
    if not current.matches(local_host, local_pid, lease.generation):
        return None
    renewed = Lease(
        host=local_host,
        pid=local_pid,
        generation=lease.generation,
        started_at=current.started_at,
        renewed_at=_utc_now_iso(),
    )
    write_lease_atomic(workspace_root, renewed)
    confirmed = read_lease(workspace_root)
    if confirmed is None:
        return None
    if not confirmed.matches(local_host, local_pid, lease.generation):
        return None
    return confirmed


def wait_until_lease_released(
    workspace_root: str | Path,
    *,
    target_generation: int | None = None,
    timeout_s: float = DEFAULT_REMOTE_STOP_WAIT_S,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
    poll_s: float = 0.5,
) -> bool:
    """Wait until the lease is gone, stale, or no longer the target generation."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        owned = read_lease(workspace_root)
        if owned is None:
            return True
        if target_generation is not None and owned.generation != target_generation:
            return True
        if _local_owner_pid_dead(owned):
            return True
        if not owned.is_fresh(stale_after_s=stale_after_s):
            return True
        time.sleep(poll_s)
    owned = read_lease(workspace_root)
    if owned is None:
        return True
    if target_generation is not None and owned.generation != target_generation:
        return True
    if _local_owner_pid_dead(owned):
        return True
    return not owned.is_fresh(stale_after_s=stale_after_s)


# ---------------------------------------------------------------------------
# Discord bot lease (same Lease struct; no stop-request layer — supervisor owns lifecycle)
# ---------------------------------------------------------------------------


def read_bot_lease(workspace_root: str | Path) -> Lease | None:
    """Read the Discord bot ownership lease, or None if missing/released."""
    return _read_lease_at(logs.discord_bot_lease_path(workspace_root))


def bot_lease_is_fresh(
    workspace_root: str | Path,
    *,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
) -> bool:
    """True when a Discord bot lease exists and was renewed within *stale_after_s*."""
    owned = read_bot_lease(workspace_root)
    return owned is not None and owned.is_fresh(stale_after_s=stale_after_s)


def bot_lease_is_alive(
    workspace_root: str | Path,
    *,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
) -> bool:
    """True when a fresh bot lease exists and the owner pid is live (local) or trusted (remote)."""
    owned = read_bot_lease(workspace_root)
    if owned is None or not owned.is_fresh(stale_after_s=stale_after_s):
        return False
    if daemon.identity_on_local_host(owned.host):
        return daemon.is_process_alive(owned.pid)
    return True


def bot_lease_is_reclaimable(
    workspace_root: str | Path,
    *,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
) -> bool:
    """True when no live foreign Discord bot owner blocks a new acquire."""
    owned = read_bot_lease(workspace_root)
    if owned is None:
        return True
    if _local_owner_pid_dead_for(owned):
        return True
    return not owned.is_fresh(stale_after_s=stale_after_s)


def try_acquire_bot_lease(
    workspace_root: str | Path,
    *,
    host: str | None = None,
    pid: int | None = None,
    stale_after_s: float = DEFAULT_LEASE_STALE_S,
    settle_s: float = DEFAULT_LEASE_SETTLE_S,
) -> Lease | None:
    """Attempt to become the Discord bot owner for *workspace_root*.

    Same race/settle semantics as ``try_acquire_lease``, without stop-request gating.
    """
    local_host = host if host is not None else daemon.local_hostname()
    local_pid = pid if pid is not None else os.getpid()
    ensure_bot_lease_file(workspace_root)
    lease_path = logs.discord_bot_lease_path(workspace_root)

    current = _read_lease_at(lease_path)
    if current is not None and current.is_ours(host=local_host, pid=local_pid):
        if current.is_fresh(stale_after_s=stale_after_s):
            return current

    if current is not None and current.is_fresh(stale_after_s=stale_after_s):
        if not _local_owner_pid_dead_for(current):
            return None

    now = _utc_now_iso()
    candidate = Lease(
        host=local_host,
        pid=local_pid,
        generation=_next_generation_at(lease_path),
        started_at=now,
        renewed_at=now,
    )
    _write_lease_at(lease_path, candidate)
    if settle_s > 0:
        time.sleep(settle_s)
    confirmed = _read_lease_at(lease_path)
    if confirmed is None:
        return None
    if not confirmed.matches(local_host, local_pid, candidate.generation):
        return None
    return confirmed


def renew_bot_lease(
    workspace_root: str | Path,
    owned: Lease,
    *,
    host: str | None = None,
    pid: int | None = None,
) -> Lease | None:
    """Renew the Discord bot lease without changing generation."""
    local_host = host if host is not None else daemon.local_hostname()
    local_pid = pid if pid is not None else os.getpid()
    lease_path = logs.discord_bot_lease_path(workspace_root)
    current = _read_lease_at(lease_path)
    if current is None:
        return None
    if not current.matches(local_host, local_pid, owned.generation):
        return None
    renewed = Lease(
        host=local_host,
        pid=local_pid,
        generation=owned.generation,
        started_at=current.started_at,
        renewed_at=_utc_now_iso(),
    )
    _write_lease_at(lease_path, renewed)
    confirmed = _read_lease_at(lease_path)
    if confirmed is None:
        return None
    if not confirmed.matches(local_host, local_pid, owned.generation):
        return None
    return confirmed


def release_bot_lease(
    workspace_root: str | Path,
    *,
    host: str | None = None,
    pid: int | None = None,
    generation: int | None = None,
) -> bool:
    """Mark the Discord bot lease released without unlinking the file."""
    ensure_bot_lease_file(workspace_root)
    lease_path = logs.discord_bot_lease_path(workspace_root)
    current = _read_lease_at(lease_path)
    if current is None:
        _atomic_write_json(lease_path, dict(_LEASE_RELEASED_PAYLOAD))
        return True
    local_host = host if host is not None else daemon.local_hostname()
    local_pid = pid if pid is not None else os.getpid()
    if not current.is_ours(host=local_host, pid=local_pid):
        return False
    if generation is not None and current.generation != generation:
        return False
    payload = dict(_LEASE_RELEASED_PAYLOAD)
    payload["released_at"] = _utc_now_iso()
    payload["released_by_host"] = local_host
    payload["released_generation"] = current.generation
    _atomic_write_json(lease_path, payload)
    return _read_lease_at(lease_path) is None


def clear_bot_lease_if_reclaimable(workspace_root: str | Path) -> bool:
    """Force-clear a stale/dead Discord bot lease (for stop_all cleanup)."""
    if not bot_lease_is_reclaimable(workspace_root):
        return False
    ensure_bot_lease_file(workspace_root)
    lease_path = logs.discord_bot_lease_path(workspace_root)
    current = _read_lease_at(lease_path)
    payload = dict(_LEASE_RELEASED_PAYLOAD)
    payload["released_at"] = _utc_now_iso()
    payload["released_by_host"] = daemon.local_hostname()
    if current is not None:
        payload["released_generation"] = current.generation
    _atomic_write_json(lease_path, payload)
    return True
