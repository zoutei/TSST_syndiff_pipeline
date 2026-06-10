"""Discord bot for on-demand pipeline status replies."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.deployment import (
    load_deployment_file,
    load_handoff_root_from_deployment,
)
from syndiff_pipeline.template_runner.notifications import format_status_reply_messages
from syndiff_pipeline.template_runner.state import PipelineState
from syndiff_pipeline.template_runner.workspace import runs_root, state_db_path

log = logging.getLogger(__name__)


def _channel_matches(message, channel_id: int) -> bool:
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


def _require_discord():
    try:
        import discord
    except ImportError as exc:
        raise SystemExit(
            "discord.py is required for the Discord bot. Install with:\n"
            "  pip install 'discord.py>=2.3'"
        ) from exc
    return discord


class PipelineDiscordBot:
    """Reply to channel messages with live progress + status grid."""

    def __init__(self, deployment_path: str | Path):
        self._deployment_path = Path(deployment_path).expanduser().resolve()
        deployment = load_deployment_file(self._deployment_path)
        self._handoff_root = str(
            load_handoff_root_from_deployment(self._deployment_path)
        )
        self._runs_dir = str(runs_root(self._handoff_root))
        self._state = PipelineState(str(state_db_path(self._handoff_root)))
        self._token = str(deployment.get("discord_bot_token", "")).strip() or None
        self._channel_id = (
            str(deployment.get("discord_channel_id", "")).strip() or None
        )

    def _build_status_reply(self, message_text: str) -> list[str]:
        from syndiff_pipeline.template_runner.notifications import (
            resolve_run_ids_for_status_request,
        )

        run_ids = resolve_run_ids_for_status_request(self._state, message_text)
        return format_status_reply_messages(
            self._state,
            run_ids,
            self._runs_dir,
            handoff_root=self._handoff_root,
        )

    def run(self) -> None:
        discord = _require_discord()
        if not self._token:
            raise SystemExit(
                f"No Discord bot token found in {self._deployment_path} "
                "(set discord_bot_token in deployment.yaml)"
            )
        if not self._channel_id:
            raise SystemExit(
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
                if hasattr(ch, "permissions_for") and client.user:
                    perms = ch.permissions_for(client.user)
                    if not perms.view_channel:
                        log.error("Bot lacks View Channel in #%s", ch_name)
                    if not perms.read_messages:
                        log.error("Bot lacks Read Messages in #%s", ch_name)
                    if not perms.send_messages:
                        log.error("Bot lacks Send Messages in #%s", ch_name)
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
            try:
                replies = await asyncio.to_thread(self._build_status_reply, message.content)
            except Exception:
                log.exception("Failed to build status reply")
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

        log.info("Starting Discord bot for %s", self._deployment_path)
        client.run(self._token, log_handler=None)


def run_discord_bot(
    deployment_path: str | Path,
    *,
    detached: bool = False,
) -> None:
    path = Path(deployment_path).expanduser().resolve()
    load_deployment_file(path)
    handoff_root = str(load_handoff_root_from_deployment(path))
    pid_path = logs.discord_bot_pid_path(handoff_root)
    if detached:
        daemon.write_pid(pid_path, os.getpid())
        try:
            PipelineDiscordBot(path).run()
        finally:
            daemon.remove_pid_file(pid_path)
    else:
        PipelineDiscordBot(path).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SynDiff pipeline Discord bot")
    parser.add_argument(
        "--deployment",
        required=True,
        help="Path to deployment.yaml",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run as background worker (writes pid file; used by daemon start)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_discord_bot(args.deployment, detached=args.detached)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
