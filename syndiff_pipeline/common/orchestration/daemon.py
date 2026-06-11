"""Global supervisor daemon process management."""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from syndiff_pipeline.common.orchestration import logs

PROCESS_LOG_FORMAT = (
    "%(asctime)s %(levelname)s [%(component)s host=%(host)s pid=%(pid)s] %(message)s"
)


class _ProcessIdentityFilter(logging.Filter):
    def __init__(self, component: str, host: str, pid: int) -> None:
        super().__init__()
        self._component = component
        self._host = host
        self._pid = pid

    def filter(self, record: logging.LogRecord) -> bool:
        record.component = self._component
        record.host = self._host
        record.pid = self._pid
        return True


def configure_process_logging(
    component: str,
    *,
    level: int = logging.INFO,
    host: str | None = None,
    pid: int | None = None,
) -> None:
    """Configure root logging so every line includes *component*, host, and pid."""
    resolved_host = host if host is not None else local_hostname()
    resolved_pid = pid if pid is not None else os.getpid()
    logging.basicConfig(level=level, format=PROCESS_LOG_FORMAT, force=True)
    root = logging.getLogger()
    if not root.handlers:
        return
    identity_filter = _ProcessIdentityFilter(component, resolved_host, resolved_pid)
    for handler in root.handlers:
        handler.addFilter(identity_filter)


@dataclass(frozen=True)
class DaemonStatus:
    alive: bool
    pid: int | None
    heartbeat_age_s: float | None
    lock_held: bool
    host: str | None = None


def local_hostname() -> str:
    return socket.gethostname()


def hosts_match(local: str, remote: str) -> bool:
    if local == remote:
        return True
    return local.startswith(remote) or remote.startswith(local)


def identity_on_local_host(host: str | None) -> bool:
    """True when *host* is unknown (legacy pid file) or matches this machine."""
    if not host:
        return True
    return hosts_match(local_hostname(), host)


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_local_process_alive(host: str | None, pid: int | None) -> bool:
    """Return True only when *pid* is recorded for this host and responds to signal 0."""
    if not pid:
        return False
    if not identity_on_local_host(host):
        return False
    return is_process_alive(pid)


def read_process_identity(pid_path: str | Path) -> tuple[str | None, int | None]:
    """Read ``host`` and ``pid`` from a pid file (two-line or legacy single-pid format)."""
    p = Path(pid_path)
    if not p.is_file():
        return None, None
    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None, None
    if len(lines) == 1:
        try:
            return None, int(lines[0])
        except ValueError:
            return None, None
    try:
        return lines[0], int(lines[1])
    except ValueError:
        return None, None


def write_process_identity(
    pid_path: str | Path,
    pid: int,
    *,
    host: str | None = None,
) -> None:
    recorded_host = host if host is not None else local_hostname()
    Path(pid_path).write_text(f"{recorded_host}\n{pid}\n", encoding="utf-8")


def read_pid(pid_path: str | Path) -> Optional[int]:
    _, pid = read_process_identity(pid_path)
    return pid


def write_pid(pid_path: str | Path, pid: int) -> None:
    write_process_identity(pid_path, pid)


def remove_pid_file(pid_path: str | Path) -> None:
    p = Path(pid_path)
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass


@contextmanager
def file_lock(lock_path: str | Path, *, blocking: bool = False) -> Iterator[int | None]:
    """Exclusive flock on *lock_path*. Yields fd when acquired, else None."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    flags = fcntl.LOCK_EX
    if not blocking:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(fd, flags)
    except BlockingIOError:
        os.close(fd)
        yield None
        return
    try:
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def daemon_lock(workspace_root: str | Path, *, blocking: bool = False) -> Iterator[int | None]:
    """Exclusive flock on the daemon lock file. Yields fd when acquired, else None."""
    with file_lock(logs.daemon_lock_path(workspace_root), blocking=blocking) as fd:
        yield fd


def spawn_detached_daemon(deployment_path: str | Path, daemon_log: str | Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.common.orchestration.scheduler",
        "--daemon",
        "--deployment",
        str(Path(deployment_path).expanduser().resolve()),
    ]
    log_path = Path(daemon_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log_fh.close()
    return proc.pid


def terminate_process_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    if not is_process_alive(pid):
        return
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


terminate_pid = terminate_process_tree


def wait_for_process_exit(
    pid: int,
    *,
    timeout_s: float = 10.0,
    poll_s: float = 0.2,
) -> bool:
    """Return True once *pid* is no longer alive (or was never alive)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return True
        time.sleep(poll_s)
    return not is_process_alive(pid)


def wait_for_daemon(workspace_root: str | Path, *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    pid_path = logs.daemon_pid_path(workspace_root)
    while time.monotonic() < deadline:
        host, pid = read_process_identity(pid_path)
        if is_local_process_alive(host, pid):
            return True
        time.sleep(0.2)
    return False
