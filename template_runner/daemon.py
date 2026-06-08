"""Global supervisor daemon process management."""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from syndiff_pipeline.template_runner import logs


@dataclass(frozen=True)
class DaemonStatus:
    alive: bool
    pid: int | None
    heartbeat_age_s: float | None
    lock_held: bool


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


def read_pid(pid_path: str | Path) -> Optional[int]:
    p = Path(pid_path)
    if not p.is_file():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def write_pid(pid_path: str | Path, pid: int) -> None:
    Path(pid_path).write_text(str(pid), encoding="utf-8")


def remove_pid_file(pid_path: str | Path) -> None:
    p = Path(pid_path)
    if p.is_file():
        try:
            p.unlink()
        except OSError:
            pass


@contextmanager
def daemon_lock(state_db_path: str | Path, *, blocking: bool = False) -> Iterator[int | None]:
    """Exclusive flock on the daemon lock file. Yields fd when acquired, else None."""
    lock_path = logs.daemon_lock_path(state_db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
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


def spawn_detached_daemon(state_db_path: str | Path, daemon_log: str | Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.template_runner.scheduler",
        "--daemon",
        "--state-db",
        str(state_db_path),
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


def wait_for_daemon(state_db_path: str | Path, *, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    pid_path = logs.daemon_pid_path(state_db_path)
    while time.monotonic() < deadline:
        pid = read_pid(pid_path)
        if pid and is_process_alive(pid):
            return True
        time.sleep(0.2)
    return False
