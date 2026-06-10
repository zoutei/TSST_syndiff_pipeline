"""Discord bot process lifecycle (started/stopped with the supervisor daemon)."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from syndiff_pipeline.template_creation.orchestration import daemon, logs
from syndiff_pipeline.template_creation.orchestration.deployment import (
    load_deployment_file,
    load_handoff_root_from_deployment,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config
from syndiff_pipeline.template_creation.orchestration.workspace import (
    load_recorded_deployment_path,
    normalize_handoff_root,
    record_deployment_path,
)

log = logging.getLogger(__name__)

DEFAULT_START_WAIT_S = 5.0
DEFAULT_STOP_TERM_TIMEOUT_S = 10.0
DEFAULT_STOP_KILL_WAIT_S = 5.0


@dataclass(frozen=True)
class EnsureDiscordBotResult:
    enabled: bool
    spawned: bool
    pid: int | None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class DiscordBotStatus:
    enabled: bool
    alive: bool
    pid: int | None
    skipped_reason: str | None = None


def _channel_id_from_deployment(
    deployment: dict,
    *,
    config_channel_id: str = "",
) -> str | None:
    if config_channel_id.strip():
        return config_channel_id.strip()
    channel_id = str(deployment.get("discord_channel_id", "")).strip()
    return channel_id or None


def _bot_configured_from_deployment(
    deployment: dict,
    *,
    config_channel_id: str = "",
) -> tuple[bool, str | None]:
    token = str(deployment.get("discord_bot_token", "")).strip()
    if not token:
        return False, "no bot token configured"
    if not _channel_id_from_deployment(deployment, config_channel_id=config_channel_id):
        return False, "no channel id configured"
    try:
        import discord  # noqa: F401
    except ImportError:
        return False, "discord.py not installed"
    return True, None


@contextmanager
def discord_bot_lock(handoff_root: str | Path, *, blocking: bool = True) -> Iterator[int | None]:
    """Exclusive flock for Discord bot ensure/stop under *handoff_root*."""
    with daemon.file_lock(logs.discord_bot_lock_path(handoff_root), blocking=blocking) as fd:
        yield fd


def discover_discord_bot_pids(handoff_root: str | Path) -> list[int]:
    """Return live detached discord_bot PIDs whose deployment maps to *handoff_root*."""
    target = str(normalize_handoff_root(handoff_root))
    proc = Path("/proc")
    if not proc.is_dir():
        return []

    pids: list[int] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\0") if p]
        if not parts:
            continue
        joined = " ".join(parts)
        if "template_creation.orchestration.discord_bot" not in joined:
            continue
        if "--detached" not in parts:
            continue
        try:
            idx = parts.index("--deployment")
            deploy_path = parts[idx + 1]
        except (ValueError, IndexError):
            continue
        try:
            resolved_handoff = str(load_handoff_root_from_deployment(deploy_path))
        except Exception:
            continue
        if resolved_handoff != target:
            continue
        pid = int(entry.name)
        if daemon.is_process_alive(pid):
            pids.append(pid)

    seen: set[int] = set()
    unique: list[int] = []
    for pid in pids:
        if pid not in seen:
            seen.add(pid)
            unique.append(pid)
    return unique


def discord_bot_is_alive(handoff_root: str | Path) -> bool:
    pid = daemon.read_pid(logs.discord_bot_pid_path(handoff_root))
    return bool(pid and daemon.is_process_alive(pid))


def discord_bot_status(
    deployment_path: str | Path,
    *,
    site_config_path: str | Path | None = None,
) -> DiscordBotStatus:
    deploy_path = Path(deployment_path).expanduser().resolve()
    deployment = load_deployment_file(deploy_path)
    handoff_root = str(load_handoff_root_from_deployment(deploy_path))
    config_channel_id = ""
    enabled = True
    if site_config_path is not None:
        cfg = load_runner_config(site_config_path)
        enabled = cfg.notifications.bot.enabled
        config_channel_id = cfg.notifications.bot.channel_id
    if not enabled:
        return DiscordBotStatus(
            enabled=False,
            alive=False,
            pid=None,
            skipped_reason="disabled",
        )
    configured, reason = _bot_configured_from_deployment(
        deployment,
        config_channel_id=config_channel_id,
    )
    if not configured:
        return DiscordBotStatus(
            enabled=True,
            alive=False,
            pid=None,
            skipped_reason=reason,
        )
    pid = daemon.read_pid(logs.discord_bot_pid_path(handoff_root))
    alive = bool(pid and daemon.is_process_alive(pid))
    return DiscordBotStatus(enabled=True, alive=alive, pid=pid if alive else None)


def discord_bot_status_for_handoff(handoff_root: str | Path) -> DiscordBotStatus:
    """Report Discord bot status using deployment + site config recorded under *handoff_root*."""
    deployment_path = load_recorded_deployment_path(handoff_root)
    site_config = _load_recorded_site_config(handoff_root)
    if deployment_path is not None:
        return discord_bot_status(deployment_path, site_config_path=site_config)
    pids = discover_discord_bot_pids(handoff_root)
    alive = bool(pids)
    return DiscordBotStatus(
        enabled=False,
        alive=alive,
        pid=pids[0] if alive else None,
        skipped_reason=None if alive else "no recorded deployment",
    )


def spawn_detached_discord_bot(
    deployment_path: str | Path,
    bot_log: str | Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.template_creation.orchestration.discord_bot",
        "--deployment",
        str(Path(deployment_path).expanduser().resolve()),
        "--detached",
    ]
    log_path = Path(bot_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log_fh.close()
    return proc.pid


def wait_for_discord_bot(
    handoff_root: str | Path,
    *,
    timeout_s: float = DEFAULT_START_WAIT_S,
) -> bool:
    deadline = time.monotonic() + timeout_s
    pid_path = logs.discord_bot_pid_path(handoff_root)
    while time.monotonic() < deadline:
        pid = daemon.read_pid(pid_path)
        if pid and daemon.is_process_alive(pid):
            return True
        time.sleep(0.2)
    return discord_bot_is_alive(handoff_root)


def record_discord_bot_site_config(handoff_root: str | Path, config_path: str | Path) -> None:
    """Persist site config path so the supervisor can check bot.enabled on auto-start."""
    path = Path(config_path).expanduser().resolve()
    record_path = logs.discord_bot_site_config_path(handoff_root)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(str(path), encoding="utf-8")


def _load_recorded_site_config(handoff_root: str | Path) -> Path | None:
    record_path = logs.discord_bot_site_config_path(handoff_root)
    if not record_path.is_file():
        return None
    text = record_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_file() else None


def _terminate_pids(
    pids: list[int],
    *,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> None:
    for pid in pids:
        if not daemon.is_process_alive(pid):
            continue
        daemon.terminate_process_tree(pid, signal.SIGTERM)
    for pid in pids:
        if not daemon.is_process_alive(pid):
            continue
        if not daemon.wait_for_process_exit(pid, timeout_s=term_timeout_s):
            daemon.terminate_process_tree(pid, signal.SIGKILL)
            daemon.wait_for_process_exit(pid, timeout_s=kill_wait_s)


def _terminate_discord_bots_for_handoff(
    handoff_root: str | Path,
    *,
    exclude: set[int] | None = None,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> None:
    skip = exclude or set()
    pids = [pid for pid in discover_discord_bot_pids(handoff_root) if pid not in skip]
    _terminate_pids(pids, term_timeout_s=term_timeout_s, kill_wait_s=kill_wait_s)


def _ensure_discord_bot_running_locked(
    deployment_path: Path,
    handoff_root: str | Path,
) -> EnsureDiscordBotResult:
    pid_path = logs.discord_bot_pid_path(handoff_root)
    pid = daemon.read_pid(pid_path)
    if pid and daemon.is_process_alive(pid):
        return EnsureDiscordBotResult(enabled=True, spawned=False, pid=pid)

    if pid is not None:
        daemon.remove_pid_file(pid_path)

    _terminate_discord_bots_for_handoff(handoff_root)

    bot_log = logs.discord_bot_log_path(handoff_root)
    spawn_pid = spawn_detached_discord_bot(deployment_path, bot_log)
    if wait_for_discord_bot(handoff_root):
        owner_pid = daemon.read_pid(pid_path) or spawn_pid
        spawned = owner_pid == spawn_pid
        return EnsureDiscordBotResult(enabled=True, spawned=spawned, pid=owner_pid)

    if discord_bot_is_alive(handoff_root):
        owner_pid = daemon.read_pid(pid_path)
        return EnsureDiscordBotResult(enabled=True, spawned=False, pid=owner_pid)

    log.warning("Discord bot pid=%s failed to start (see %s)", spawn_pid, bot_log)
    return EnsureDiscordBotResult(
        enabled=True,
        spawned=False,
        pid=None,
        skipped_reason=f"failed to start (see {bot_log})",
    )


def ensure_discord_bot_running(
    deployment_path: str | Path,
    *,
    site_config_path: str | Path | None = None,
) -> EnsureDiscordBotResult:
    """Start detached Discord bot when enabled and configured (one per handoff_root)."""
    deploy_path = Path(deployment_path).expanduser().resolve()
    deployment = load_deployment_file(deploy_path)
    handoff_root = str(load_handoff_root_from_deployment(deploy_path))
    config_channel_id = ""
    if site_config_path is not None:
        site_path = Path(site_config_path).expanduser().resolve()
        cfg = load_runner_config(site_path)
        if not cfg.notifications.bot.enabled:
            return EnsureDiscordBotResult(enabled=False, spawned=False, pid=None)
        config_channel_id = cfg.notifications.bot.channel_id
        record_discord_bot_site_config(handoff_root, site_path)

    configured, reason = _bot_configured_from_deployment(
        deployment,
        config_channel_id=config_channel_id,
    )
    if not configured:
        log.warning("Discord bot not started: %s", reason)
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=None,
            skipped_reason=reason,
        )

    record_deployment_path(handoff_root, deploy_path)

    with discord_bot_lock(handoff_root, blocking=True) as fd:
        if fd is None:
            log.warning("Discord bot lock unavailable for %s", handoff_root)
            return EnsureDiscordBotResult(
                enabled=True,
                spawned=False,
                pid=None,
                skipped_reason="could not acquire discord bot lock",
            )
        return _ensure_discord_bot_running_locked(deploy_path, handoff_root)


def ensure_discord_bot_for_handoff_root(
    handoff_root: str | Path,
) -> EnsureDiscordBotResult | None:
    """Start the Discord bot when deployment is recorded and it is not alive."""
    deployment_path = load_recorded_deployment_path(handoff_root)
    if deployment_path is None:
        return None
    if discord_bot_is_alive(handoff_root):
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=daemon.read_pid(logs.discord_bot_pid_path(handoff_root)),
        )
    site_config = _load_recorded_site_config(handoff_root)
    return ensure_discord_bot_running(deployment_path, site_config_path=site_config)


def stop_discord_bot(
    handoff_root: str | Path,
    *,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> bool:
    """Stop all Discord bots for *handoff_root*. Returns True when none remain."""
    with discord_bot_lock(handoff_root, blocking=True):
        pid_path = logs.discord_bot_pid_path(handoff_root)
        pid = daemon.read_pid(pid_path)
        targets = set(discover_discord_bot_pids(handoff_root))
        if pid is not None:
            targets.add(pid)

        if not targets:
            daemon.remove_pid_file(pid_path)
            return True

        _terminate_pids(
            sorted(targets),
            term_timeout_s=term_timeout_s,
            kill_wait_s=kill_wait_s,
        )
        remaining = discover_discord_bot_pids(handoff_root)
        if not remaining:
            daemon.remove_pid_file(pid_path)
        return not remaining
