"""Discord bot configuration helpers (bot runs as supervisor-managed subprocess)."""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from syndiff_pipeline.common.orchestration import daemon, lease, logs
from syndiff_pipeline.common.orchestration.deployment import (
    load_deployment_file,
    load_workspace_root_from_deployment,
)
from syndiff_pipeline.common.orchestration.workspace import (
    load_recorded_deployment_path,
    normalize_workspace_root,
)

log = logging.getLogger(__name__)

_BOT_MODULE_MARKER = "template_creation.orchestration.discord_bot"
_DEFAULT_STOP_TIMEOUT_S = 5.0


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
    lease_generation: int | None = None
    lease_age_s: float | None = None


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
) -> tuple[bool, int | None, str | None, int | None, float | None]:
    """Return (alive, pid, host, lease_generation, lease_age_s)."""
    workspace_root = str(deployment.get("workspace_root", "")).strip()
    if not workspace_root:
        return False, None, None, None, None
    owned = lease.read_bot_lease(workspace_root)
    lease_gen = owned.generation if owned is not None else None
    lease_age = owned.age_s() if owned is not None else None
    if owned is not None and lease.bot_lease_is_alive(workspace_root):
        return True, owned.pid, owned.host, lease_gen, lease_age
    # Fallback to pid file when lease is missing (upgrade path).
    if not daemon_alive:
        return False, None, None, lease_gen, lease_age
    pid_path = logs.discord_bot_pid_path(workspace_root)
    host, pid = daemon.read_process_identity(pid_path)
    if pid is None:
        return False, None, host, lease_gen, lease_age
    if host is not None and not daemon.identity_on_local_host(host):
        return False, pid, host, lease_gen, lease_age
    return daemon.is_process_alive(pid), pid, host, lease_gen, lease_age


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
    child_alive, child_pid, child_host, lease_gen, lease_age = _bot_child_status(
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
        lease_generation=lease_gen,
        lease_age_s=lease_age,
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


def discover_workspace_bot_pids(workspace_root: str | Path) -> list[int]:
    """Return live Discord bot PIDs for *workspace_root* (any spawn style).

    Matches supervisor-spawned bots and legacy ``--detached`` bots whose
    ``--deployment`` resolves to this workspace.
    """
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
        if _BOT_MODULE_MARKER not in joined:
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
        if _BOT_MODULE_MARKER not in joined:
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


def stop_all_workspace_discord_bots(
    workspace_root: str | Path,
    *,
    timeout_s: float = _DEFAULT_STOP_TIMEOUT_S,
) -> int:
    """Terminate all local Discord bot processes for *workspace_root*.

    Scans ``/proc`` (not only the pid file), waits briefly for exit, clears the
    pid file, and releases a reclaimable bot lease. Returns the number of PIDs
    signaled.
    """
    pids = discover_workspace_bot_pids(workspace_root)
    pid_path = logs.discord_bot_pid_path(workspace_root)
    host, recorded_pid = daemon.read_process_identity(pid_path)
    if (
        recorded_pid is not None
        and daemon.identity_on_local_host(host)
        and daemon.is_process_alive(recorded_pid)
        and recorded_pid not in pids
    ):
        pids.append(recorded_pid)

    for pid in pids:
        try:
            daemon.terminate_process_tree(pid, signal.SIGTERM)
        except Exception:
            log.warning("Failed to terminate Discord bot pid=%s", pid, exc_info=True)

    if pids:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            alive = [p for p in pids if daemon.is_process_alive(p)]
            if not alive:
                break
            time.sleep(0.1)
        still = [p for p in pids if daemon.is_process_alive(p)]
        for pid in still:
            try:
                daemon.terminate_process_tree(pid, signal.SIGKILL)
            except Exception:
                log.warning("Failed to SIGKILL Discord bot pid=%s", pid, exc_info=True)
        log.info(
            "Signaled %s Discord bot process(es) for workspace cleanup",
            len(pids),
        )

    try:
        daemon.remove_pid_file(pid_path)
    except Exception:
        pass

    # If we hold the lease locally, release it; else clear when reclaimable.
    owned = lease.read_bot_lease(workspace_root)
    if owned is not None and owned.is_ours():
        lease.release_bot_lease(
            workspace_root,
            host=owned.host,
            pid=owned.pid,
            generation=owned.generation,
        )
    else:
        lease.clear_bot_lease_if_reclaimable(workspace_root)

    return len(pids)


def cleanup_legacy_detached_bots(workspace_root: str | Path) -> int:
    """Terminate leftover Discord bot processes from older installs.

    Prefer ``stop_all_workspace_discord_bots`` for full singleton cleanup.
    Returns the number of processes signaled.
    """
    return stop_all_workspace_discord_bots(workspace_root)
