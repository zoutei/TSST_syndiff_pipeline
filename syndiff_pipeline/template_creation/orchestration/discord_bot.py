"""Discord bot for on-demand pipeline status replies (supervisor-managed subprocess)."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import getpass
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.deployment import (
    load_deployment_file,
    load_workspace_root_from_deployment,
)
from syndiff_pipeline.common.orchestration.notifications import (
    _DISCORD_PACK_MAX_CHARS,
    format_status_reply_messages,
)
from syndiff_pipeline.common.orchestration.state import PipelineState
from syndiff_pipeline.common.orchestration.workspace import runs_root, state_db_path
from syndiff_pipeline.template_creation.orchestration.run_report import pack_message_lines

log = logging.getLogger(__name__)

_CONDOR_SHELL_COMMANDS = frozenset(
    {"condor_q", "condor_qn", "condor_status", "condor_status -tla"}
)
_CONDOR_SHELL_TIMEOUT_S = 30.0
_CONDOR_SHELL_TIMEOUT_OVERRIDES = {"condor_status -tla": 60.0}
_STATUS_BUILD_TIMEOUT_S = 120.0
_STATUS_EXECUTOR_WORKERS = 2


def _channel_matches(message: Any, channel_id: int) -> bool:
    """True when *message* is in the configured channel or a thread under it."""
    ch = message.channel
    if ch.id == channel_id:
        return True
    parent_id = getattr(ch, "parent_id", None)
    if parent_id == channel_id:
        return True
    parent = getattr(ch, "parent", None)
    if parent is not None and getattr(parent, "id", None) == channel_id:
        return True
    return False


def condor_shell_trigger(message_text: str) -> str | None:
    """Return the Condor shell trigger when *message_text* is an exact command request."""
    key = message_text.strip().lower()
    if key in _CONDOR_SHELL_COMMANDS:
        return key
    return None


def _condor_shell_argv(trigger: str) -> list[str]:
    if trigger == "condor_q":
        return ["condor_q"]
    if trigger == "condor_qn":
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()
        return [
            "condor_q",
            user,
            "-af",
            "ClusterId",
            "ProcId",
            "JobStatus",
            "RemoteHost",
        ]
    if trigger == "condor_status":
        return ["condor_status"]
    if trigger == "condor_status -tla":
        # -af:h hangs with SECMAN errors on some STScI nodes; -af works.
        return ["condor_status", "-af", "Name", "TotalLoadAvg"]
    raise ValueError(f"unknown condor shell trigger: {trigger!r}")


def run_condor_shell_command(trigger: str, *, timeout_s: float | None = None) -> list[str]:
    """Run a whitelisted Condor CLI command and return Discord-sized reply messages."""
    argv = _condor_shell_argv(trigger)
    if timeout_s is None:
        timeout_s = _CONDOR_SHELL_TIMEOUT_OVERRIDES.get(trigger, _CONDOR_SHELL_TIMEOUT_S)
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return [f"`{argv[0]}` not found on this host."]
    except subprocess.TimeoutExpired:
        return [f"`{trigger}` timed out after {timeout_s:.0f}s."]

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    body = stdout.rstrip()
    if proc.returncode != 0:
        err = stderr.rstrip()
        if err and body:
            body = f"{err}\n{body}"
        elif err:
            body = err
        elif not body:
            body = f"(exit {proc.returncode}, no output)"
        else:
            body = f"(exit {proc.returncode})\n{body}"
    elif not body.strip():
        body = stderr.rstrip() or "(no output)"

    if trigger == "condor_status -tla" and body.strip() and proc.returncode == 0:
        if not body.lstrip().startswith("Name"):
            body = f"Name TotalLoadAvg\n{body}"

    max_body = _DISCORD_PACK_MAX_CHARS - len(trigger) - 20
    if len(body) > max_body:
        body = body[: max_body - 20] + "\n… (truncated)"

    header = f"**{trigger}**"
    return pack_message_lines([header, f"```\n{body}\n```"], max_chars=_DISCORD_PACK_MAX_CHARS)


def _require_discord():
    """Import discord.py or raise ImportError."""
    import discord

    return discord


class PipelineDiscordBot:
    """Reply to channel messages with live progress + status grid."""

    def __init__(self, deployment_path: str | Path):
        self._deployment_path = Path(deployment_path).expanduser().resolve()
        deployment = load_deployment_file(self._deployment_path)
        self._workspace_root = str(
            load_workspace_root_from_deployment(self._deployment_path)
        )
        self._runs_dir = str(runs_root(self._workspace_root))
        self._state = PipelineState(str(state_db_path(self._workspace_root)))
        self._token = str(deployment.get("discord_bot_token", "")).strip() or None
        self._channel_id = (
            str(deployment.get("discord_channel_id", "")).strip() or None
        )
        self._client: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_STATUS_EXECUTOR_WORKERS,
            thread_name_prefix="discord-status",
        )

    def _build_status_reply(self, message_text: str) -> list[str]:
        from syndiff_pipeline.common.orchestration.notifications import (
            resolve_run_ids_for_status_request,
        )

        run_ids = resolve_run_ids_for_status_request(self._state, message_text)
        return format_status_reply_messages(
            self._state,
            run_ids,
            self._runs_dir,
            workspace_root=self._workspace_root,
            include_orphan_scan=False,
        )

    def _build_client(self):
        discord = _require_discord()
        if not self._token:
            raise RuntimeError(
                f"No Discord bot token found in {self._deployment_path} "
                "(set discord_bot_token in deployment.yaml)"
            )
        if not self._channel_id:
            raise RuntimeError(
                f"No Discord channel configured in {self._deployment_path}. "
                "Set discord_channel_id in deployment.yaml."
            )

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        channel_id = int(self._channel_id)
        listen_channel_id: int | None = channel_id
        bot_user_id: int | None = None
        executor = self._executor

        @client.event
        async def on_ready():
            nonlocal bot_user_id, listen_channel_id
            bot_user_id = client.user.id if client.user else None
            try:
                ch = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
                ch_name = getattr(ch, "name", "?")
                log.info(
                    "Discord bot connected as %s; listening in #%s (%s)",
                    client.user,
                    ch_name,
                    channel_id,
                )
                guild = getattr(ch, "guild", None)
                if hasattr(ch, "permissions_for") and client.user and guild is not None:
                    member = guild.get_member(client.user.id)
                    if member is None:
                        try:
                            member = await guild.fetch_member(client.user.id)
                        except Exception:
                            member = None
                    if member is not None:
                        perms = ch.permissions_for(member)
                        if not perms.view_channel:
                            log.error("Bot lacks View Channel in #%s", ch_name)
                        if not perms.read_messages:
                            log.error("Bot lacks Read Messages in #%s", ch_name)
                        if not perms.send_messages:
                            log.error("Bot lacks Send Messages in #%s", ch_name)
                    else:
                        log.warning(
                            "Could not resolve guild member for permission check in #%s",
                            ch_name,
                        )
            except Exception as exc:
                listen_channel_id = None
                log.error(
                    "Cannot access configured channel_id=%s (%s). "
                    "Will reply in any channel the bot can read. "
                    "Fix discord_channel_id in deployment.yaml: right-click the target "
                    "channel in Discord → Copy Channel ID (Developer Mode on). "
                    "Ensure the bot is invited to the server with View/Send permissions.",
                    channel_id,
                    exc,
                )

        @client.event
        async def on_message(message):
            if message.author.bot:
                return
            if listen_channel_id is not None and not _channel_matches(message, listen_channel_id):
                log.info(
                    "Ignored message in channel %s (%s); expected %s",
                    getattr(message.channel, "name", "?"),
                    message.channel.id,
                    listen_channel_id,
                )
                return
            if bot_user_id is not None and message.author.id == bot_user_id:
                return
            log.info(
                "Status request from %s in #%s",
                message.author,
                getattr(message.channel, "name", message.channel.id),
            )
            trigger = condor_shell_trigger(message.content)
            try:
                await message.channel.trigger_typing()
            except Exception:
                log.debug("Could not send typing indicator", exc_info=True)
            loop = asyncio.get_running_loop()
            try:
                if trigger is not None:
                    build_coro = loop.run_in_executor(
                        executor, run_condor_shell_command, trigger
                    )
                else:
                    build_coro = loop.run_in_executor(
                        executor, self._build_status_reply, message.content
                    )
                replies = await asyncio.wait_for(
                    build_coro, timeout=_STATUS_BUILD_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                log.warning("Discord status build timed out after %.0fs", _STATUS_BUILD_TIMEOUT_S)
                if trigger is not None:
                    replies = [f"`{trigger}` timed out after {_STATUS_BUILD_TIMEOUT_S:.0f}s."]
                else:
                    replies = [
                        "Pipeline status is taking too long (supervisor may be busy on NFS). "
                        "Try again in a moment or use `syndiff progress`."
                    ]
            except Exception:
                log.exception("Failed to build Discord reply")
                if trigger is not None:
                    replies = [f"Failed to run `{trigger}` (see bot logs)."]
                else:
                    replies = ["Failed to read pipeline status (see bot logs)."]
            try:
                for index, reply in enumerate(replies):
                    if index == 0:
                        await message.reply(reply, mention_author=False)
                    else:
                        await message.channel.send(reply)
            except Exception:
                log.exception("Failed to send Discord reply")
                try:
                    for reply in replies:
                        await message.channel.send(reply)
                except Exception:
                    log.exception("Failed to send Discord reply via channel.send")

        self._client = client
        return client

    async def _async_run(self) -> None:
        client = self._build_client()
        assert self._token is not None
        async with client:
            await client.start(self._token)

    def run_forever(self) -> None:
        """Block the current thread running the Discord client until closed."""
        log.info("Starting Discord bot for %s", self._deployment_path)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_run())
        except Exception:
            log.exception("Discord bot exited with error")
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()
            self._loop = None
            self._client = None
            self._executor.shutdown(wait=False, cancel_futures=True)

    def request_stop(self) -> None:
        """Ask the Discord client to close (thread-safe)."""
        client = self._client
        loop = self._loop
        if client is None or loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(client.close(), loop)
        except Exception:
            log.warning("Failed to schedule Discord client close", exc_info=True)


def spawn_discord_bot_subprocess(
    deployment_path: str | Path,
    *,
    workspace_root: str | Path,
) -> int:
    """Spawn a detached Discord bot child process and return its PID."""
    deploy = Path(deployment_path).expanduser().resolve()
    cmd = [
        sys.executable,
        "-m",
        "syndiff_pipeline.template_creation.orchestration.discord_bot",
        "--deployment",
        str(deploy),
    ]
    log_path = logs.discord_bot_log_path(workspace_root)
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


def stop_discord_bot_subprocess(
    workspace_root: str | Path,
    *,
    timeout_s: float = 5.0,
) -> None:
    """Terminate the supervisor-managed Discord bot child, if recorded locally."""
    pid_path = logs.discord_bot_pid_path(workspace_root)
    host, pid = daemon.read_process_identity(pid_path)
    if pid is None:
        return
    if host is not None and host != daemon.local_hostname():
        log.warning(
            "Discord bot pid file points to host %s (local %s); not signaling pid=%s",
            host,
            daemon.local_hostname(),
            pid,
        )
        return
    if daemon.is_process_alive(pid):
        daemon.terminate_process_tree(pid)
        deadline = time.monotonic() + timeout_s
        while daemon.is_process_alive(pid):
            if time.monotonic() >= deadline:
                log.warning("Discord bot pid=%s did not exit within %.1fs", pid, timeout_s)
                break
            time.sleep(0.1)
    daemon.remove_pid_file(pid_path)


class InProcessDiscordBot:
    """Supervisor-owned Discord bot running in a child process."""

    def __init__(self, deployment_path: str | Path):
        self._deployment_path = Path(deployment_path).expanduser().resolve()
        self._workspace_root = str(
            load_workspace_root_from_deployment(self._deployment_path)
        )
        self._process: subprocess.Popen[Any] | None = None
        self._pid: int | None = None
        self._started = False
        self.skipped_reason: str | None = None

    @property
    def running(self) -> bool:
        if self._process is not None and self._process.poll() is None:
            return True
        if self._pid is not None and daemon.is_process_alive(self._pid):
            return True
        return False

    @property
    def pid(self) -> int | None:
        if self._process is not None and self._process.poll() is None:
            return self._process.pid
        if self._pid is not None and daemon.is_process_alive(self._pid):
            return self._pid
        return None

    def start(self) -> bool:
        """Start the bot subprocess. Returns False when skipped or already running."""
        if self.running:
            return True
        try:
            bot = PipelineDiscordBot(self._deployment_path)
        except Exception as exc:
            self.skipped_reason = str(exc)
            log.warning("Discord bot not started: %s", exc)
            return False
        if not bot._token:
            self.skipped_reason = "no bot token configured"
            return False
        if not bot._channel_id:
            self.skipped_reason = "no channel id configured"
            return False
        try:
            _require_discord()
        except ImportError:
            self.skipped_reason = "discord.py not installed"
            log.warning("Discord bot not started: discord.py not installed")
            return False

        stop_discord_bot_subprocess(self._workspace_root, timeout_s=2.0)
        try:
            pid = spawn_discord_bot_subprocess(
                self._deployment_path,
                workspace_root=self._workspace_root,
            )
        except OSError as exc:
            self.skipped_reason = f"failed to spawn bot subprocess: {exc}"
            log.warning("Discord bot not started: %s", exc)
            return False

        self._pid = pid
        self._process = None
        daemon.write_process_identity(
            logs.discord_bot_pid_path(self._workspace_root),
            pid,
            host=daemon.local_hostname(),
        )
        self._started = True
        self.skipped_reason = None
        log.info(
            "Discord bot subprocess started pid=%s for %s",
            pid,
            self._deployment_path,
        )
        return True

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Stop the bot subprocess and wait briefly for exit."""
        proc = self._process
        if proc is not None and proc.poll() is None:
            daemon.terminate_process_tree(proc.pid)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                log.warning("Discord bot subprocess did not exit within %.1fs", timeout_s)
        elif self._pid is not None:
            stop_discord_bot_subprocess(self._workspace_root, timeout_s=timeout_s)
        self._process = None
        self._pid = None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the supervisor-managed Discord bot subprocess."""
    parser = argparse.ArgumentParser(description="SynDiff Discord status bot")
    parser.add_argument(
        "--deployment",
        required=True,
        help="Path to deployment.yaml (workspace_root + Discord credentials)",
    )
    args = parser.parse_args(argv)
    deploy = Path(args.deployment).expanduser().resolve()
    workspace_root = load_workspace_root_from_deployment(deploy)
    daemon.configure_process_logging("discord-bot")
    try:
        PipelineDiscordBot(deploy).run_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
