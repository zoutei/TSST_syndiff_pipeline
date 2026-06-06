"""Detached scheduler process management."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional


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
        p.unlink()


def spawn_detached_scheduler(
    run_id: str,
    config_path: str,
    targets_path: str,
    stages: str | None,
    scheduler_log: str | Path,
) -> int:
    """Start scheduler in a new session; return child PID."""
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.template_runner.scheduler",
        "--run-id",
        run_id,
        "--config",
        str(config_path),
        "--targets",
        str(targets_path),
    ]
    if stages:
        cmd.extend(["--stages", stages])
    log_path = Path(scheduler_log)
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


def terminate_pid(pid: int, sig: int = signal.SIGTERM) -> None:
    if is_process_alive(pid):
        os.kill(pid, sig)
