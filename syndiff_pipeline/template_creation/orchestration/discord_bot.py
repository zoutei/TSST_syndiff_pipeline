"""Discord bot for on-demand pipeline status replies (runs inside the supervisor)."""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

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
                if trigger is not None:
                    replies = await asyncio.to_thread(run_condor_shell_command, trigger)
                else:
                    replies = await asyncio.to_thread(self._build_status_reply, message.content)
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


class InProcessDiscordBot:
    """Supervisor-owned Discord bot running on a background thread."""

    def __init__(self, deployment_path: str | Path):
        self._deployment_path = Path(deployment_path).expanduser().resolve()
        self._bot: PipelineDiscordBot | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self.skipped_reason: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the bot thread. Returns False when skipped or already running."""
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

        self._bot = bot
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="discord-bot",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        self.skipped_reason = None
        log.info("Discord bot thread started for %s", self._deployment_path)
        return True

    def _run_guarded(self) -> None:
        assert self._bot is not None
        try:
            self._bot.run_forever()
        except Exception:
            log.exception("Discord bot thread crashed")

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Stop the bot thread and wait briefly for exit."""
        if self._bot is not None:
            self._bot.request_stop()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                log.warning("Discord bot thread did not exit within %.1fs", timeout_s)
        self._thread = None
        self._bot = None
