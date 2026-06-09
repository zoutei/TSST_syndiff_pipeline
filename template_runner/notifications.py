"""Discord webhook notifications for pipeline run/stage events."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import yaml

from syndiff_pipeline.template_runner.run_report import (
    format_progress_lines,
    format_run_report,
    format_run_report_messages,
    format_target_status_line,
)
from syndiff_pipeline.template_runner.state import STAGE_SHORT_NAMES

if TYPE_CHECKING:
    from syndiff_pipeline.template_runner.runner_config import NotificationConfig
    from syndiff_pipeline.template_runner.state import PipelineState

log = logging.getLogger(__name__)

_DISCORD_MAX_CONTENT = 2000
_WEBHOOK_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class NotificationEvents:
    run_started: bool = True
    run_completed: bool = True
    run_failed: bool = True
    run_canceled: bool = True
    run_stalled: bool = True
    run_resumed: bool = True
    stage_failed: bool = True
    stage_completed: bool = True
    stage_canceled: bool = True
    stage_died: bool = True
    daemon_unhealthy: bool = True


@dataclass(frozen=True)
class DiscordBotConfig:
    enabled: bool = False
    channel_id: str = ""


@dataclass(frozen=True)
class NotificationConfig:
    enabled: bool = False
    secrets_file: str = "secrets.yaml"
    events: NotificationEvents = field(default_factory=NotificationEvents)
    bot: DiscordBotConfig = field(default_factory=DiscordBotConfig)


def parse_notification_config(raw: dict | None) -> NotificationConfig:
    raw = raw or {}
    events_raw = raw.get("events") or {}
    events = NotificationEvents(
        run_started=bool(events_raw.get("run_started", True)),
        run_completed=bool(events_raw.get("run_completed", True)),
        run_failed=bool(events_raw.get("run_failed", True)),
        run_canceled=bool(events_raw.get("run_canceled", True)),
        run_stalled=bool(events_raw.get("run_stalled", True)),
        run_resumed=bool(events_raw.get("run_resumed", True)),
        stage_failed=bool(events_raw.get("stage_failed", True)),
        stage_completed=bool(events_raw.get("stage_completed", True)),
        stage_canceled=bool(events_raw.get("stage_canceled", True)),
        stage_died=bool(events_raw.get("stage_died", True)),
        daemon_unhealthy=bool(events_raw.get("daemon_unhealthy", True)),
    )
    bot_raw = raw.get("bot") or {}
    bot = DiscordBotConfig(
        enabled=bool(bot_raw.get("enabled", False)),
        channel_id=str(bot_raw.get("channel_id", "")).strip(),
    )
    return NotificationConfig(
        enabled=bool(raw.get("enabled", False)),
        secrets_file=str(raw.get("secrets_file", "secrets.yaml")),
        events=events,
        bot=bot,
    )


def _load_secrets(config_path: str | Path, secrets_file: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    secrets_path = path.parent / secrets_file
    if not secrets_path.is_file():
        return {}
    try:
        with secrets_path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except OSError:
        log.warning("Failed to read secrets file %s", secrets_path, exc_info=True)
        return {}


def load_webhook_url(config_path: str | Path, secrets_file: str) -> str | None:
    file_url = str(_load_secrets(config_path, secrets_file).get("discord_webhook_url", "")).strip()
    return file_url or None


def load_bot_token(config_path: str | Path, secrets_file: str) -> str | None:
    token = str(_load_secrets(config_path, secrets_file).get("discord_bot_token", "")).strip()
    return token or None


def load_channel_id(config_path: str | Path, secrets_file: str) -> str | None:
    channel_id = str(_load_secrets(config_path, secrets_file).get("discord_channel_id", "")).strip()
    return channel_id or None


def resolve_webhook_url(
    *,
    config_path: str | Path,
    secrets_file: str,
    source_config_path: str | Path | None = None,
) -> str | None:
    for candidate in (config_path, source_config_path):
        if not candidate:
            continue
        url = load_webhook_url(candidate, secrets_file)
        if url:
            return url
    env_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
    return env_url or None


def resolve_bot_token(
    *,
    config_path: str | Path,
    secrets_file: str,
    source_config_path: str | Path | None = None,
) -> str | None:
    for candidate in (config_path, source_config_path):
        if not candidate:
            continue
        token = load_bot_token(candidate, secrets_file)
        if token:
            return token
    env_token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    return env_token or None


def resolve_channel_id(
    *,
    config_path: str | Path,
    secrets_file: str,
    config_channel_id: str = "",
    source_config_path: str | Path | None = None,
) -> str | None:
    if config_channel_id.strip():
        return config_channel_id.strip()
    for candidate in (config_path, source_config_path):
        if not candidate:
            continue
        channel_id = load_channel_id(candidate, secrets_file)
        if channel_id:
            return channel_id
    env_channel = os.environ.get("DISCORD_CHANNEL_ID", "").strip()
    return env_channel or None


def post_discord_webhook(url: str, content: str) -> None:
    payload = json.dumps({"content": content[: _DISCORD_MAX_CONTENT]}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "syndiff-template-notifications",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_WEBHOOK_TIMEOUT_S) as resp:
        resp.read()


def _utc_header() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class Notifier:
    def __init__(
        self,
        state: PipelineState,
        cfg: NotificationConfig,
        *,
        config_path: str | Path,
        state_db_path: str,
        source_config_path: str | Path | None = None,
    ):
        self._state = state
        self._cfg = cfg
        self._config_path = Path(config_path)
        self._source_config_path = (
            Path(source_config_path).expanduser().resolve()
            if source_config_path
            else None
        )
        self._state_db_path = state_db_path
        self._webhook_url: str | None = None

    def _webhook(self) -> str | None:
        if self._webhook_url is None:
            self._webhook_url = (
                resolve_webhook_url(
                    config_path=self._config_path,
                    secrets_file=self._cfg.secrets_file,
                    source_config_path=self._source_config_path,
                )
                or ""
            )
        return self._webhook_url or None

    def _send(self, run_id: str, event_key: str, content: str | Sequence[str]) -> None:
        if not self._cfg.enabled:
            return
        url = self._webhook()
        if not url:
            log.debug("Notifications enabled but no webhook URL configured")
            return
        if not self._state.try_record_notification(run_id, event_key):
            return
        parts = [content] if isinstance(content, str) else list(content)
        for part in parts:
            try:
                post_discord_webhook(url, part)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                log.warning("Discord notification failed for %s: %s", event_key, exc)
                return

    def notify_run_started(
        self,
        run_id: str,
        *,
        run_dir: str | Path,
        target_labels: list[str],
        stages: list[str],
        force_rerun: bool = False,
    ) -> None:
        if not self._cfg.events.run_started:
            return
        body = format_run_started_message(
            run_id,
            run_dir=run_dir,
            target_labels=target_labels,
            stages=stages,
            force_rerun=force_rerun,
        )
        self._send(run_id, "run:started", body)

    def notify_run_completed(
        self,
        run_id: str,
        runs_root: str,
        *,
        outcome: str,
    ) -> None:
        if outcome == "success" and not self._cfg.events.run_completed:
            return
        if outcome == "failed" and not self._cfg.events.run_failed:
            return
        header = f"[{run_id}] run_{outcome} ({_utc_header()})"
        body = format_run_report_messages(
            self._state,
            run_id,
            runs_root,
            handoff_root=self._handoff_root,
            header=header,
        )
        self._send(run_id, f"run:{outcome}", body)

    def notify_run_stalled(self, run_id: str, runs_root: str, *, stall_reason: str) -> None:
        if not self._cfg.events.run_stalled:
            return
        header = f"[{run_id}] run_stalled ({_utc_header()})\nstall_reason={stall_reason!r}"
        body = format_run_report_messages(
            self._state,
            run_id,
            runs_root,
            handoff_root=self._handoff_root,
            header=header,
        )
        self._send(run_id, f"run:stalled:{_utc_header()}", body)

    def notify_run_resumed(self, run_id: str) -> None:
        if not self._cfg.events.run_resumed:
            return
        header = f"[{run_id}] run_resumed ({_utc_header()})"
        self._send(run_id, f"run:resumed:{_utc_header()}", header)

    def notify_run_canceled(self, run_id: str, runs_root: str) -> None:
        if not self._cfg.events.run_canceled:
            return
        header = f"[{run_id}] run_canceled ({_utc_header()})"
        body = format_run_report_messages(
            self._state,
            run_id,
            runs_root,
            handoff_root=self._handoff_root,
            header=header,
        )
        self._send(run_id, "run:canceled", body)

    def notify_stage_outcome(
        self,
        run_id: str,
        runs_root: str,
        *,
        target_label: str,
        stage: str,
        outcome: str,
        finished_at: str,
        error_tail: str | None = None,
    ) -> None:
        if outcome == "success" and not self._cfg.events.stage_completed:
            return
        if outcome == "failed" and not self._cfg.events.stage_failed:
            return
        if outcome == "canceled" and not self._cfg.events.stage_canceled:
            return
        if outcome == "died" and not self._cfg.events.stage_died:
            return
        if outcome not in ("success", "failed", "canceled", "died"):
            return
        short = STAGE_SHORT_NAMES.get(stage, stage)
        header = f"[{run_id}] stage_{outcome} ({_utc_header()})\n{target_label} / {short}"
        if error_tail:
            header += f"\nerror: {error_tail[:400]}"
        lines = [header, ""]
        lines.extend(
            format_progress_lines(
                self._state,
                run_id,
                runs_root,
                handoff_root=self._handoff_root,
                include_running_detail=True,
            )
        )
        target_line = format_target_status_line(self._state, run_id, target_label)
        if target_line:
            lines.extend(["", target_line])
        self._send(run_id, f"stage:{target_label}:{stage}:{outcome}:{finished_at}", "\n".join(lines))

    def notify_daemon_unhealthy(self, *, detail: str) -> None:
        if not self._cfg.events.daemon_unhealthy:
            return
        header = f"[supervisor] daemon_unhealthy ({_utc_header()})\n{detail}"
        self._send("", f"daemon:unhealthy:{_utc_header()}", header)


def format_run_started_message(
    run_id: str,
    *,
    run_dir: str | Path,
    target_labels: Sequence[str],
    stages: Sequence[str],
    force_rerun: bool = False,
) -> str:
    """Short submit-time announcement (not progress/status grid)."""
    enabled_count = len(target_labels)
    preview = ", ".join(target_labels[:5])
    if len(target_labels) > 5:
        preview += f", … (+{len(target_labels) - 5} more)"
    stage_text = ", ".join(stages)
    lines = [
        f"[{run_id}] run_started ({_utc_header()})",
        f"targets: {enabled_count} ({preview})" if preview else f"targets: {enabled_count}",
        f"stages: {stage_text}",
    ]
    if force_rerun:
        lines.append("force_rerun: true")
    run_path = Path(run_dir).expanduser().resolve()
    lines.extend(
        [
            f"run_dir: {run_path}",
            "",
            "Reply in Discord for live progress/status, or:",
            f"  syndiff-template progress --run-dir {run_path}",
        ]
    )
    return "\n".join(lines)


def resolve_run_ids_for_status_request(
    state: PipelineState,
    message_text: str,
    *,
    limit: int = 3,
) -> list[str]:
    """Pick run_id(s) to report for an on-demand Discord status request."""
    text = message_text.strip()
    known = {row["run_id"] for row in state.list_runs(200)}
    if text:
        for token in text.replace(",", " ").split():
            cleaned = token.strip("`\"'")
            if cleaned in known:
                return [cleaned]
    active = state.active_runs()
    if active:
        return [row["run_id"] for row in active[:limit]]
    recent = state.list_runs(1)
    if recent:
        return [recent[0]["run_id"]]
    return []


def format_status_reply_messages(
    state: PipelineState,
    run_ids: Sequence[str],
    runs_root: str,
    *,
    handoff_root: str | None = None,
) -> list[str]:
    """On-demand progress + status grid; one or more Discord-sized messages."""
    if not run_ids:
        return ["No pipeline runs found."]
    messages: list[str] = []
    for run_id in run_ids:
        run = state.get_run(run_id)
        if run is None:
            messages.append(f"Unknown run_id: {run_id}")
            continue
        root = run.get("runs_root") or runs_root
        header = f"[{run_id}] status ({_utc_header()})"
        messages.extend(
            format_run_report_messages(
                state,
                run_id,
                root,
                handoff_root=handoff_root,
                header=header,
            )
        )
    return messages


def format_status_reply_message(
    state: PipelineState,
    run_ids: Sequence[str],
    runs_root: str,
    *,
    handoff_root: str | None = None,
) -> str:
    """Single-string status reply (joins all parts; prefer format_status_reply_messages)."""
    return "\n\n".join(
        format_status_reply_messages(
            state,
            run_ids,
            runs_root,
            handoff_root=handoff_root,
        )
    )


def format_preview_message(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    handoff_root: str | None = None,
    event_label: str = "notification preview",
) -> str:
    """Read-only snapshot: progress summary + status grid (same shape as daemon alerts)."""
    header = f"[TEST] [{run_id}] {event_label} ({_utc_header()})"
    return format_run_report(
        state,
        run_id,
        runs_root,
        handoff_root=handoff_root,
        header=header,
    )


def send_preview_notification(
    state: PipelineState,
    ctx,
    *,
    event_label: str = "notification preview",
) -> str:
    """Post a test message to Discord without recording notification_events."""
    cfg = getattr(ctx.cfg, "notifications", None)
    if cfg is None:
        raise SystemExit("notifications block missing from config")
    from syndiff_pipeline.template_runner import logs

    config_path = logs.run_config_path(ctx.run_dir)
    source_config_path = (ctx.meta or {}).get("source_config_path")
    url = resolve_webhook_url(
        config_path=config_path,
        secrets_file=cfg.secrets_file,
        source_config_path=source_config_path,
    )
    if not url:
        raise SystemExit(
            f"No Discord webhook URL found (check {cfg.secrets_file} beside config or "
            "DISCORD_WEBHOOK_URL)"
        )
    messages = format_run_report_messages(
        state,
        ctx.run_id,
        ctx.cfg.runs_dir(),
        state_db_path=ctx.cfg.state_db_path,
        header=f"[TEST] [{ctx.run_id}] {event_label} ({_utc_header()})",
    )
    for message in messages:
        post_discord_webhook(url, message)
    return "\n\n".join(messages)


def send_run_started_notification(
    state: PipelineState,
    cfg: NotificationConfig,
    *,
    config_path: str | Path,
    run_id: str,
    run_dir: str | Path,
    target_labels: list[str],
    stages: list[str],
    state_db_path: str,
    force_rerun: bool = False,
) -> None:
    """Post run_started via webhook when notifications are enabled."""
    notifier = Notifier(
        state,
        cfg,
        config_path=config_path,
        state_db_path=state_db_path,
    )
    notifier.notify_run_started(
        run_id,
        run_dir=run_dir,
        target_labels=target_labels,
        stages=stages,
        force_rerun=force_rerun,
    )


def notifier_for_context(state: PipelineState, ctx) -> Notifier | None:
    cfg = getattr(ctx.cfg, "notifications", None)
    if cfg is None:
        return None
    from syndiff_pipeline.template_runner import logs

    config_path = logs.run_config_path(ctx.run_dir)
    source_config_path = (ctx.meta or {}).get("source_config_path")
    return Notifier(
        state,
        cfg,
        config_path=config_path,
        state_db_path=ctx.cfg.state_db_path,
        source_config_path=source_config_path,
    )
