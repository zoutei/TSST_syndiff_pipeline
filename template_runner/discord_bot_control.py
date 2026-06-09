"""Discord bot process lifecycle (started/stopped with the supervisor daemon)."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.notifications import resolve_bot_token, resolve_channel_id
from syndiff_pipeline.template_runner.runner_config import load_runner_config

log = logging.getLogger(__name__)

DEFAULT_START_WAIT_S = 5.0


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


def _bot_configured(config_path: Path, cfg) -> tuple[bool, str | None]:
    if not cfg.notifications.bot.enabled:
        return False, "disabled"
    notif = cfg.notifications
    if not resolve_bot_token(config_path=config_path, deployment_file=cfg.deployment_file):
        return False, "no bot token configured"
    if not resolve_channel_id(
        config_path=config_path,
        deployment_file=cfg.deployment_file,
        config_channel_id=notif.bot.channel_id,
    ):
        return False, "no channel id configured"
    try:
        import discord  # noqa: F401
    except ImportError:
        return False, "discord.py not installed"
    return True, None


def discord_bot_is_alive(handoff_root: str | Path) -> bool:
    pid = daemon.read_pid(logs.discord_bot_pid_path(handoff_root))
    return bool(pid and daemon.is_process_alive(pid))


def discord_bot_status(config_path: str | Path) -> DiscordBotStatus:
    path = Path(config_path).expanduser().resolve()
    cfg = load_runner_config(path)
    configured, reason = _bot_configured(path, cfg)
    if not configured:
        return DiscordBotStatus(
            enabled=cfg.notifications.bot.enabled,
            alive=False,
            pid=None,
            skipped_reason=reason,
        )
    pid = daemon.read_pid(logs.discord_bot_pid_path(cfg.handoff_root))
    alive = bool(pid and daemon.is_process_alive(pid))
    return DiscordBotStatus(enabled=True, alive=alive, pid=pid if alive else None)


def discord_bot_status_for_handoff(handoff_root: str | Path) -> DiscordBotStatus:
    """Report Discord bot status using the site config recorded under *handoff_root*."""
    config_path = _load_recorded_site_config(handoff_root)
    if config_path is not None:
        return discord_bot_status(config_path)
    pid = daemon.read_pid(logs.discord_bot_pid_path(handoff_root))
    alive = bool(pid and daemon.is_process_alive(pid))
    return DiscordBotStatus(
        enabled=False,
        alive=alive,
        pid=pid if alive else None,
        skipped_reason=None if alive else "no recorded site config",
    )


def spawn_detached_discord_bot(
    config_path: str | Path,
    bot_log: str | Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.template_runner.discord_bot",
        "--config",
        str(config_path),
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


def _record_site_config(handoff_root: str | Path, config_path: Path) -> None:
    record_path = logs.discord_bot_site_config_path(handoff_root)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(str(config_path), encoding="utf-8")


def _load_recorded_site_config(handoff_root: str | Path) -> Path | None:
    record_path = logs.discord_bot_site_config_path(handoff_root)
    if not record_path.is_file():
        return None
    text = record_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_file() else None


def ensure_discord_bot_running(config_path: str | Path) -> EnsureDiscordBotResult:
    """Start detached Discord bot when enabled and configured."""
    path = Path(config_path).expanduser().resolve()
    cfg = load_runner_config(path)
    if not cfg.notifications.bot.enabled:
        return EnsureDiscordBotResult(enabled=False, spawned=False, pid=None)

    configured, reason = _bot_configured(path, cfg)
    if not configured:
        log.warning("Discord bot not started: %s", reason)
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=None,
            skipped_reason=reason,
        )

    handoff_root = cfg.handoff_root
    _record_site_config(handoff_root, path)
    pid_path = logs.discord_bot_pid_path(handoff_root)
    pid = daemon.read_pid(pid_path)
    if pid and daemon.is_process_alive(pid):
        return EnsureDiscordBotResult(enabled=True, spawned=False, pid=pid)

    if pid is not None:
        daemon.remove_pid_file(pid_path)

    bot_log = logs.discord_bot_log_path(handoff_root)
    spawn_pid = spawn_detached_discord_bot(path, bot_log)
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


def ensure_discord_bot_for_handoff_root(
    handoff_root: str | Path,
) -> EnsureDiscordBotResult | None:
    """Restart the Discord bot when a recorded site config exists and it is not alive."""
    config_path = _load_recorded_site_config(handoff_root)
    if config_path is None:
        return None
    if discord_bot_is_alive(handoff_root):
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=daemon.read_pid(logs.discord_bot_pid_path(handoff_root)),
        )
    return ensure_discord_bot_running(config_path)


def stop_discord_bot(
    handoff_root: str | Path,
    *,
    term_timeout_s: float = 10.0,
    kill_wait_s: float = 5.0,
) -> bool:
    """Stop the Discord bot if running. Returns True when no live bot remains."""
    pid_path = logs.discord_bot_pid_path(handoff_root)
    pid = daemon.read_pid(pid_path)
    if not pid or not daemon.is_process_alive(pid):
        if pid is not None:
            daemon.remove_pid_file(pid_path)
        return True

    daemon.terminate_process_tree(pid, signal.SIGTERM)
    if not daemon.wait_for_process_exit(pid, timeout_s=term_timeout_s):
        daemon.terminate_process_tree(pid, signal.SIGKILL)
        daemon.wait_for_process_exit(pid, timeout_s=kill_wait_s)

    stopped = not daemon.is_process_alive(pid)
    if stopped:
        daemon.remove_pid_file(pid_path)
    return stopped
