"""Discord bot process lifecycle (started/stopped with the supervisor daemon)."""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.deployment import (
    load_deployment_file,
    load_workspace_root_from_deployment,
)
from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config
from syndiff_pipeline.common.orchestration.workspace import (
    load_recorded_deployment_path,
    normalize_workspace_root,
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
    host: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class DiscordBotStatus:
    enabled: bool
    alive: bool
    pid: int | None
    host: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True)
class DiscordBotLocateResult:
    cli_host: str
    pid_file_host: str | None
    pid_file_pid: int | None
    local_pids: tuple[int, ...]
    log_host: str | None
    log_pid: int | None
    log_recorded_at: str | None
    likely_host: str | None
    likely_pid: int | None
    alive_here: bool
    hints: tuple[str, ...]


_LOG_IDENTITY_RE = re.compile(
    r"host=(['\"]?)(?P<host>[^'\" \]]+)\1 pid=(?P<pid>\d+)"
)
_DAEMON_BOT_PID_RE = re.compile(
    r"Started Discord bot(?: host=(['\"]?)(?P<host>[^'\" \]]+)\1)? pid=(?P<pid>\d+)"
)


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
def discord_bot_lock(workspace_root: str | Path, *, blocking: bool = True) -> Iterator[int | None]:
    """Exclusive flock for Discord bot ensure/stop under *workspace_root*."""
    with daemon.file_lock(logs.discord_bot_lock_path(workspace_root), blocking=blocking) as fd:
        yield fd


def discover_discord_bot_pids(workspace_root: str | Path) -> list[int]:
    """Return live detached discord_bot PIDs whose deployment maps to *workspace_root*."""
    target = str(normalize_workspace_root(workspace_root))
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
            resolved_handoff = str(load_workspace_root_from_deployment(deploy_path))
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


def _discord_bot_pid_identity(workspace_root: str | Path) -> tuple[str | None, int | None]:
    return daemon.read_process_identity(logs.discord_bot_pid_path(workspace_root))


def _log_bot_control(workspace_root: str | Path, message: str) -> None:
    try:
        logs.append_discord_bot_control_log(workspace_root, message)
    except OSError:
        log.warning("Failed to append Discord bot control log", exc_info=True)


def _remote_discord_bot_message(host: str, pid: int | None) -> str:
    pid_text = f" (pid={pid})" if pid else ""
    local = daemon.local_hostname()
    return (
        f"Discord bot running on host {host!r}{pid_text}. "
        f"This machine is {local!r}. SSH to {host} to manage the bot, or stop it there first."
    )


def discord_bot_is_alive(workspace_root: str | Path) -> bool:
    host, pid = _discord_bot_pid_identity(workspace_root)
    if not pid:
        return False
    if host and not daemon.identity_on_local_host(host):
        return True
    return daemon.is_process_alive(pid)


def discord_bot_status(
    deployment_path: str | Path,
    *,
    site_config_path: str | Path | None = None,
) -> DiscordBotStatus:
    deploy_path = Path(deployment_path).expanduser().resolve()
    deployment = load_deployment_file(deploy_path)
    workspace_root = str(load_workspace_root_from_deployment(deploy_path))
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
            host=None,
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
            host=None,
            skipped_reason=reason,
        )
    host, pid = _discord_bot_pid_identity(workspace_root)
    alive = discord_bot_is_alive(workspace_root)
    skipped_reason = None
    if pid and not alive:
        if host and not daemon.identity_on_local_host(host):
            skipped_reason = _remote_discord_bot_message(host, pid)
        else:
            skipped_reason = f"stale pid file (host={host or daemon.local_hostname()!r} pid={pid})"
    return DiscordBotStatus(
        enabled=True,
        alive=alive,
        pid=pid,
        host=host,
        skipped_reason=skipped_reason,
    )


def discord_bot_status_for_handoff(workspace_root: str | Path) -> DiscordBotStatus:
    """Report Discord bot status using deployment + site config recorded under *workspace_root*."""
    deployment_path = load_recorded_deployment_path(workspace_root)
    site_config = _load_recorded_site_config(workspace_root)
    if deployment_path is not None:
        return discord_bot_status(deployment_path, site_config_path=site_config)
    pids = discover_discord_bot_pids(workspace_root)
    alive = bool(pids)
    return DiscordBotStatus(
        enabled=False,
        alive=alive,
        pid=pids[0] if alive else None,
        host=daemon.local_hostname() if alive else None,
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
    workspace_root: str | Path,
    *,
    timeout_s: float = DEFAULT_START_WAIT_S,
) -> bool:
    deadline = time.monotonic() + timeout_s
    pid_path = logs.discord_bot_pid_path(workspace_root)
    while time.monotonic() < deadline:
        host, pid = daemon.read_process_identity(pid_path)
        if daemon.is_local_process_alive(host, pid):
            return True
        time.sleep(0.2)
    return discord_bot_is_alive(workspace_root)


def record_discord_bot_site_config(workspace_root: str | Path, config_path: str | Path) -> None:
    """Persist site config path so the supervisor can check bot.enabled on auto-start."""
    path = Path(config_path).expanduser().resolve()
    record_path = logs.discord_bot_site_config_path(workspace_root)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(str(path), encoding="utf-8")


def _load_recorded_site_config(workspace_root: str | Path) -> Path | None:
    record_path = logs.discord_bot_site_config_path(workspace_root)
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
    workspace_root: str | Path,
    *,
    exclude: set[int] | None = None,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> None:
    skip = exclude or set()
    pids = [pid for pid in discover_discord_bot_pids(workspace_root) if pid not in skip]
    _terminate_pids(pids, term_timeout_s=term_timeout_s, kill_wait_s=kill_wait_s)


def _ensure_discord_bot_running_locked(
    deployment_path: Path,
    workspace_root: str | Path,
) -> EnsureDiscordBotResult:
    pid_path = logs.discord_bot_pid_path(workspace_root)
    host, pid = daemon.read_process_identity(pid_path)
    if pid and host and not daemon.identity_on_local_host(host):
        reason = _remote_discord_bot_message(host, pid)
        _log_bot_control(
            workspace_root,
            f"ensure skipped: remote bot host={host!r} pid={pid} local={daemon.local_hostname()!r}",
        )
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=pid,
            host=host,
            skipped_reason=reason,
        )
    if daemon.is_local_process_alive(host, pid):
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=pid,
            host=host or daemon.local_hostname(),
        )

    if pid is not None:
        daemon.remove_pid_file(pid_path)

    _terminate_discord_bots_for_handoff(workspace_root)

    bot_log = logs.discord_bot_log_path(workspace_root)
    local_host = daemon.local_hostname()
    _log_bot_control(
        workspace_root,
        f"spawning detached bot on host={local_host!r} deployment={deployment_path}",
    )
    spawn_pid = spawn_detached_discord_bot(deployment_path, bot_log)
    if wait_for_discord_bot(workspace_root):
        owner_host, owner_pid = daemon.read_process_identity(pid_path)
        owner_pid = owner_pid or spawn_pid
        spawned = owner_pid == spawn_pid
        owner_host = owner_host or local_host
        _log_bot_control(
            workspace_root,
            f"ensure complete host={owner_host!r} pid={owner_pid} spawned={spawned}",
        )
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=spawned,
            pid=owner_pid,
            host=owner_host,
        )

    if discord_bot_is_alive(workspace_root):
        owner_host, owner_pid = daemon.read_process_identity(pid_path)
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=owner_pid,
            host=owner_host,
        )

    _log_bot_control(
        workspace_root,
        f"ensure failed: spawn_pid={spawn_pid} on host={local_host!r} did not register in pid file",
    )
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
    """Start detached Discord bot when enabled and configured (one per workspace_root)."""
    deploy_path = Path(deployment_path).expanduser().resolve()
    deployment = load_deployment_file(deploy_path)
    workspace_root = str(load_workspace_root_from_deployment(deploy_path))
    config_channel_id = ""
    if site_config_path is not None:
        site_path = Path(site_config_path).expanduser().resolve()
        cfg = load_runner_config(site_path)
        if not cfg.notifications.bot.enabled:
            return EnsureDiscordBotResult(enabled=False, spawned=False, pid=None)
        config_channel_id = cfg.notifications.bot.channel_id
        record_discord_bot_site_config(workspace_root, site_path)

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

    record_deployment_path(workspace_root, deploy_path)

    with discord_bot_lock(workspace_root, blocking=True) as fd:
        if fd is None:
            log.warning("Discord bot lock unavailable for %s", workspace_root)
            return EnsureDiscordBotResult(
                enabled=True,
                spawned=False,
                pid=None,
                skipped_reason="could not acquire discord bot lock",
            )
        return _ensure_discord_bot_running_locked(deploy_path, workspace_root)


def _scan_log_for_identity(
    path: Path,
    *,
    max_lines: int = 3000,
) -> tuple[str | None, int | None, str | None]:
    if not path.is_file():
        return None, None, None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None, None
    for line in reversed(lines[-max_lines:]):
        match = _LOG_IDENTITY_RE.search(line)
        if not match:
            match = _DAEMON_BOT_PID_RE.search(line)
            if not match:
                continue
            host = match.group("host")
            pid = int(match.group("pid"))
            stamp = line[:19] if len(line) >= 19 and line[4] == "-" else None
            return host, pid, stamp
        stamp = line[:19] if len(line) >= 19 and line[4] == "-" else None
        return match.group("host"), int(match.group("pid")), stamp
    return None, None, None


def _last_identity_from_bot_log(
    workspace_root: str | Path,
    *,
    max_lines: int = 3000,
) -> tuple[str | None, int | None, str | None]:
    """Return the most recent (host, pid, timestamp) from bot or daemon logs."""
    bot_host, bot_pid, bot_at = _scan_log_for_identity(
        logs.discord_bot_log_path(workspace_root),
        max_lines=max_lines,
    )
    if bot_host and bot_pid:
        return bot_host, bot_pid, bot_at
    daemon_host, daemon_pid, daemon_at = _scan_log_for_identity(
        logs.daemon_log_path(workspace_root),
        max_lines=max_lines,
    )
    if daemon_host and daemon_pid:
        return daemon_host, daemon_pid, daemon_at
    if bot_pid:
        return bot_host, bot_pid, bot_at
    return daemon_host, daemon_pid, daemon_at


def locate_discord_bot(workspace_root: str | Path) -> DiscordBotLocateResult:
    """Best-effort answer for which host runs the Discord bot for *workspace_root*."""
    cli_host = daemon.local_hostname()
    pid_file_host, pid_file_pid = _discord_bot_pid_identity(workspace_root)
    local_pids = tuple(discover_discord_bot_pids(workspace_root))
    log_host, log_pid, log_at = _last_identity_from_bot_log(workspace_root)
    alive_here = bool(local_pids) or discord_bot_is_alive(workspace_root)

    likely_host: str | None = None
    likely_pid: int | None = None
    hints: list[str] = []

    if local_pids:
        likely_host = cli_host
        likely_pid = local_pids[0]
        hints.append(f"Bot process found on this host (pid={likely_pid}).")
    elif pid_file_host and pid_file_pid:
        likely_host = pid_file_host
        likely_pid = pid_file_pid
        if daemon.identity_on_local_host(pid_file_host):
            if alive_here:
                hints.append("Pid file matches a live process on this host.")
            else:
                hints.append(
                    f"Pid file lists pid={pid_file_pid} on this host but it is not running "
                    "(stale file — safe to remove discord_bot.pid)."
                )
        else:
            hints.append(
                f"Pid file says host={pid_file_host!r} pid={pid_file_pid}. "
                "SSH there and run: syndiff daemon status"
            )
    elif log_host and log_pid:
        likely_host = log_host
        likely_pid = log_pid
        hints.append(
            f"Last log identity at {log_at or 'unknown time'}: host={log_host} pid={log_pid}."
        )

    if not likely_host:
        hints.append(
            "No pid file or log identity found. If Discord still replies, search cluster hosts: "
            "pgrep -af 'orchestration.discord_bot.*--detached'"
        )
    elif not alive_here and likely_host != cli_host:
        hints.append(
            "Discord replies with no local process usually mean the bot is still running on "
            f"{likely_host!r} (or a duplicate with an old token session)."
        )

    bot_log = logs.discord_bot_log_path(workspace_root)
    hints.append(f"Inspect: tail -50 {bot_log}")

    return DiscordBotLocateResult(
        cli_host=cli_host,
        pid_file_host=pid_file_host,
        pid_file_pid=pid_file_pid,
        local_pids=local_pids,
        log_host=log_host,
        log_pid=log_pid,
        log_recorded_at=log_at,
        likely_host=likely_host,
        likely_pid=likely_pid,
        alive_here=alive_here,
        hints=tuple(hints),
    )


def ensure_discord_bot_for_workspace_root(
    workspace_root: str | Path,
) -> EnsureDiscordBotResult | None:
    """Start the Discord bot when deployment is recorded and it is not alive."""
    deployment_path = load_recorded_deployment_path(workspace_root)
    if deployment_path is None:
        return None
    if discord_bot_is_alive(workspace_root):
        host, pid = _discord_bot_pid_identity(workspace_root)
        return EnsureDiscordBotResult(
            enabled=True,
            spawned=False,
            pid=pid,
            host=host,
        )
    site_config = _load_recorded_site_config(workspace_root)
    return ensure_discord_bot_running(deployment_path, site_config_path=site_config)


def stop_discord_bot(
    workspace_root: str | Path,
    *,
    term_timeout_s: float = DEFAULT_STOP_TERM_TIMEOUT_S,
    kill_wait_s: float = DEFAULT_STOP_KILL_WAIT_S,
) -> bool:
    """Stop all Discord bots for *workspace_root*. Returns True when none remain."""
    with discord_bot_lock(workspace_root, blocking=True):
        pid_path = logs.discord_bot_pid_path(workspace_root)
        host, pid = daemon.read_process_identity(pid_path)
        targets = set(discover_discord_bot_pids(workspace_root))
        if pid is not None and daemon.identity_on_local_host(host):
            targets.add(pid)

        if not targets:
            if pid and host and not daemon.identity_on_local_host(host):
                _log_bot_control(
                    workspace_root,
                    f"stop skipped: bot registered on remote host={host!r} pid={pid} "
                    f"(local={daemon.local_hostname()!r})",
                )
                return False
            daemon.remove_pid_file(pid_path)
            return True

        _log_bot_control(
            workspace_root,
            f"stop terminating local pids={sorted(targets)} "
            f"(pid file host={host!r} pid={pid})",
        )
        _terminate_pids(
            sorted(targets),
            term_timeout_s=term_timeout_s,
            kill_wait_s=kill_wait_s,
        )
        remaining = discover_discord_bot_pids(workspace_root)
        if not remaining:
            daemon.remove_pid_file(pid_path)
        return not remaining
