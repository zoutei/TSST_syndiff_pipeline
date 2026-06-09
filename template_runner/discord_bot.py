"""Discord bot for on-demand pipeline status replies."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from syndiff_pipeline.template_runner import daemon, logs
from syndiff_pipeline.template_runner.notifications import (
    format_status_reply_message,
    resolve_bot_token,
    resolve_channel_id,
    resolve_run_ids_for_status_request,
)
from syndiff_pipeline.template_runner.runner_config import RunnerConfig, load_runner_config
from syndiff_pipeline.template_runner.state import PipelineState

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

    def __init__(
        self,
        cfg: RunnerConfig,
        *,
        config_path: str | Path,
    ):
        self._cfg = cfg
        self._config_path = Path(config_path).expanduser().resolve()
        self._state = PipelineState(cfg.state_db_path)
        notif = cfg.notifications
        self._channel_id = resolve_channel_id(
            config_path=self._config_path,
            secrets_file=notif.secrets_file,
            config_channel_id=notif.bot.channel_id,
        )
        self._token = resolve_bot_token(
            config_path=self._config_path,
            secrets_file=notif.secrets_file,
        )

    def _build_status_reply(self, message_text: str) -> str:
        run_ids = resolve_run_ids_for_status_request(self._state, message_text)
        return format_status_reply_message(
            self._state,
            run_ids,
            self._cfg.runs_dir(),
            state_db_path=self._cfg.state_db_path,
        )

    def run(self) -> None:
        discord = _require_discord()
        if not self._token:
            raise SystemExit(
                f"No Discord bot token found (set discord_bot_token in "
                f"{self._cfg.notifications.secrets_file} beside config or DISCORD_BOT_TOKEN)"
            )
        if not self._channel_id:
            raise SystemExit(
                "No Discord channel configured. Set notifications.bot.channel_id in config "
                "or discord_channel_id in secrets.yaml / DISCORD_CHANNEL_ID."
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
                    "Fix discord_channel_id in secrets.yaml: right-click the target "
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
                reply = await asyncio.to_thread(self._build_status_reply, message.content)
            except Exception:
                log.exception("Failed to build status reply")
                reply = "Failed to read pipeline status (see bot logs)."
            try:
                await message.reply(reply, mention_author=False)
            except Exception:
                log.exception("Failed to send Discord reply")
                try:
                    await message.channel.send(reply)
                except Exception:
                    log.exception("Failed to send Discord reply via channel.send")

        log.info("Starting Discord bot for %s", self._config_path)
        client.run(self._token, log_handler=None)


def run_discord_bot(
    config_path: str | Path,
    *,
    detached: bool = False,
    state_db_path: str | Path | None = None,
) -> None:
    path = Path(config_path).expanduser().resolve()
    cfg = load_runner_config(path)
    if not cfg.notifications.bot.enabled:
        raise SystemExit(
            "Discord bot is disabled. Set notifications.bot.enabled: true in config."
        )
    db_path = state_db_path or cfg.state_db_path
    pid_path = logs.discord_bot_pid_path(db_path)
    if detached:
        daemon.write_pid(pid_path, os.getpid())
        try:
            PipelineDiscordBot(cfg, config_path=path).run()
        finally:
            daemon.remove_pid_file(pid_path)
    else:
        PipelineDiscordBot(cfg, config_path=path).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SynDiff pipeline Discord bot")
    parser.add_argument("--config", required=True, help="Site config.yaml path")
    parser.add_argument(
        "--state-db",
        default=None,
        help="State DB path (for pid file location when running detached)",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Run as background worker (writes pid file; used by daemon start)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_discord_bot(
        args.config,
        detached=args.detached,
        state_db_path=args.state_db,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
