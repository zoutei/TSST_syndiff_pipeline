"""Discord bot configuration helpers (bot runs in-process inside the supervisor)."""

from __future__ import annotations

import logging
import signal
from dataclasses import dataclass
from pathlib import Path

import yaml

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.deployment import (
    load_deployment_file,
    load_workspace_root_from_deployment,
)
from syndiff_pipeline.common.orchestration.workspace import (
    load_recorded_deployment_path,
    normalize_workspace_root,
)

log = logging.getLogger(__name__)


def _bot_overrides_from_site_config(site_config_path: str | Path) -> tuple[bool, str]:
    """Read notifications.bot.enabled / channel_id without loading runner_config."""
    path = Path(site_config_path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except OSError:
        return True, ""
    bot = ((raw.get("notifications") or {}).get("bot") or {})
    return bool(bot.get("enabled", False)), str(bot.get("channel_id", "")).strip()


@dataclass(frozen=True)
class DiscordBotStatus:
    """Discord bot expected status (supervisor-managed child process)."""

    enabled: bool
    expected_in_process: bool
    skipped_reason: str | None = None
    # Legacy fields kept for status JSON compatibility during transition.
    alive: bool = False
    pid: int | None = None
    host: str | None = None


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


def record_discord_bot_site_config(workspace_root: str | Path, config_path: str | Path) -> None:
    """Record the site pipeline.yaml used for bot.enabled / channel overrides."""
    path = Path(config_path).expanduser().resolve()
    logs.discord_bot_site_config_path(workspace_root).write_text(f"{path}\n", encoding="utf-8")


def _load_recorded_site_config(workspace_root: str | Path) -> Path | None:
    path = logs.discord_bot_site_config_path(workspace_root)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    recorded = Path(text).expanduser()
    return recorded if recorded.is_file() else None


def _bot_child_status(
    deployment: dict,
    *,
    daemon_alive: bool,
) -> tuple[bool, int | None, str | None]:
    """Return (alive, pid, host) for the supervisor-managed bot child."""
    if not daemon_alive:
        return False, None, None
    workspace_root = str(deployment.get("workspace_root", "")).strip()
    if not workspace_root:
        return False, None, None
    pid_path = logs.discord_bot_pid_path(workspace_root)
    host, pid = daemon.read_process_identity(pid_path)
    if pid is None:
        return False, None, host
    if host is not None and host != daemon.local_hostname():
        return False, pid, host
    return daemon.is_process_alive(pid), pid, host


def discord_bot_status(
    deployment_path: str | Path,
    *,
    site_config_path: str | Path | None = None,
    daemon_alive: bool = False,
) -> DiscordBotStatus:
    """Report whether the bot is expected to run under the supervisor."""
    deploy_path = Path(deployment_path).expanduser().resolve()
    deployment = load_deployment_file(deploy_path)
    try:
        deployment = {
            **deployment,
            "workspace_root": load_workspace_root_from_deployment(deploy_path),
        }
    except Exception:
        pass
    config_channel_id = ""
    enabled = True
    if site_config_path is not None:
        enabled, config_channel_id = _bot_overrides_from_site_config(site_config_path)
    if not enabled:
        return DiscordBotStatus(
            enabled=False,
            expected_in_process=False,
            skipped_reason="disabled",
        )
    configured, reason = _bot_configured_from_deployment(
        deployment,
        config_channel_id=config_channel_id,
    )
    if not configured:
        return DiscordBotStatus(
            enabled=True,
            expected_in_process=False,
            skipped_reason=reason,
        )
    child_alive, child_pid, child_host = _bot_child_status(
        deployment,
        daemon_alive=daemon_alive,
    )
    return DiscordBotStatus(
        enabled=True,
        expected_in_process=daemon_alive,
        skipped_reason=None if daemon_alive else "supervisor not running",
        alive=child_alive,
        pid=child_pid,
        host=child_host or (daemon.local_hostname() if daemon_alive else None),
    )


def discord_bot_status_for_handoff(
    workspace_root: str | Path,
    *,
    daemon_alive: bool = False,
) -> DiscordBotStatus:
    """Report Discord bot status using deployment + site config under *workspace_root*."""
    deployment_path = load_recorded_deployment_path(workspace_root)
    site_config = _load_recorded_site_config(workspace_root)
    if deployment_path is not None:
        return discord_bot_status(
            deployment_path,
            site_config_path=site_config,
            daemon_alive=daemon_alive,
        )
    return DiscordBotStatus(
        enabled=False,
        expected_in_process=False,
        skipped_reason="no recorded deployment",
    )


def should_start_in_process_bot(workspace_root: str | Path) -> tuple[bool, str | None, Path | None]:
    """Return (should_start, skip_reason, deployment_path) for the in-process bot."""
    deployment_path = load_recorded_deployment_path(workspace_root)
    if deployment_path is None:
        return False, "no recorded deployment", None
    site_config = _load_recorded_site_config(workspace_root)
    config_channel_id = ""
    if site_config is not None:
        try:
            enabled, config_channel_id = _bot_overrides_from_site_config(site_config)
        except Exception as exc:
            return False, f"failed to load site config: {exc}", deployment_path
        if not enabled:
            return False, "disabled", deployment_path
    try:
        deployment = load_deployment_file(deployment_path)
    except Exception as exc:
        return False, f"failed to load deployment: {exc}", deployment_path
    configured, reason = _bot_configured_from_deployment(
        deployment,
        config_channel_id=config_channel_id,
    )
    if not configured:
        return False, reason, deployment_path
    return True, None, deployment_path


def discover_legacy_detached_bot_pids(workspace_root: str | Path) -> list[int]:
    """Return live legacy ``discord_bot --detached`` PIDs for *workspace_root*."""
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


def cleanup_legacy_detached_bots(workspace_root: str | Path) -> int:
    """Terminate leftover detached Discord bot processes from older installs.

    Returns the number of processes signaled.
    """
    pids = discover_legacy_detached_bot_pids(workspace_root)
    for pid in pids:
        try:
            daemon.terminate_process_tree(pid, signal.SIGTERM)
        except Exception:
            log.warning("Failed to terminate legacy Discord bot pid=%s", pid, exc_info=True)
    if pids:
        log.info(
            "Signaled %s legacy detached Discord bot process(es) for cleanup",
            len(pids),
        )
    # Clear stale pid file from older installs.
    try:
        daemon.remove_pid_file(logs.discord_bot_pid_path(workspace_root))
    except Exception:
        pass
    return len(pids)
