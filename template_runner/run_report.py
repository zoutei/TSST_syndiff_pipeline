"""Format pipeline status/progress reports for CLI and notifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syndiff_pipeline.template_runner.stage_progress import read_log_progress
from syndiff_pipeline.template_runner.state import (
    SKIP_REASON_NOT_SELECTED,
    SKIP_REASON_STREAM,
    SKIP_REASON_SUPERSEDED,
    STAGE_NAMES,
    STAGE_SHORT_NAMES,
    STATUS_EXTERNAL,
    STATUS_PENDING,
    STATUS_SKIPPED,
    artifact_verify_needed,
)

if TYPE_CHECKING:
    from syndiff_pipeline.template_runner.state import PipelineState


def format_run_status_header(
    run_id: str,
    run: dict,
    *,
    timestamp: str | None = None,
) -> str:
    """First-line run summary; run_id is only shown in the brackets."""
    status = run.get("status", "?")
    if timestamp:
        return f"[{run_id}] status = {status} ({timestamp})"
    return f"[{run_id}] status = {status}"


def format_progress_lines(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    handoff_root: str | None = None,
    include_running_detail: bool = True,
) -> list[str]:
    from syndiff_pipeline.template_runner.verify_status import read_verify_run_status

    counts = state.count_by_status(run_id)
    run = state.get_run(run_id) or {}
    count_parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    lines: list[str] = []
    verify_backlog = 0
    if count_parts:
        count_line = " ".join(count_parts)
        if handoff_root:
            verify_status = read_verify_run_status(handoff_root, run_id)
            scan_running = int(verify_status.get("scan_running", 0))
            scan_queued = int(verify_status.get("scan_queued", 0))
            verify_backlog = scan_running + scan_queued
            if scan_queued:
                count_line += f" scan_queued={scan_queued}"
            if scan_running:
                count_line += f" scan_running={scan_running}"
        lines.append(count_line)

    if run.get("status") == "stalled" and run.get("stall_reason") and verify_backlog == 0:
        lines.append(f"stall_reason={run['stall_reason']!r}")

    if not include_running_detail:
        return lines

    running = state.running_stage_runs(run_id)
    if not running:
        lines.append("  (no running tasks)")
        return lines

    lines.append("")
    for row in sorted(running, key=lambda r: (r.target_label, r.stage)):
        from syndiff_pipeline.template_runner import logs

        log_path = row.log_path or str(
            logs.target_log_path(runs_root, run_id, row.target_label, row.stage)
        )
        prog = read_log_progress(log_path, row.stage, started_at=row.started_at)
        short = STAGE_SHORT_NAMES.get(row.stage, row.stage)
        if prog:
            lines.append(f"  {row.target_label} {short}: {prog.text}")
        else:
            lines.append(f"  {row.target_label} {short}: (no log progress yet)")
    return lines


def format_status_grid(
    state: PipelineState,
    run_id: str,
    *,
    handoff_root: str | None = None,
) -> list[str]:
    rows = state.list_stage_runs(run_id)
    by_target: dict[str, list] = {}
    for r in rows:
        by_target.setdefault(r.target_label, []).append(r)
    stage_order = {name: i for i, name in enumerate(STAGE_NAMES)}

    def _stage_sort_key(row) -> int:
        return stage_order.get(row.stage, len(STAGE_NAMES))

    active_stages = state.get_active_stages(run_id)
    verifying_keys: set[tuple[str, str]] = set()
    if handoff_root:
        from syndiff_pipeline.template_runner.verify_status import read_verify_active_keys

        verifying_keys = set(read_verify_active_keys(handoff_root, run_id))

    lines: list[str] = []
    for label in sorted(by_target):
        rows_for_target = sorted(by_target[label], key=_stage_sort_key)
        parts = [
            _format_stage_status_short(
                state,
                run_id,
                r,
                active_stages=active_stages,
                verifying_keys=verifying_keys,
            )
            for r in rows_for_target
        ]
        lines.append(f"  {label}: {' | '.join(parts)}")
    return lines


def _format_stage_status_short(
    state: PipelineState,
    run_id: str,
    row,
    *,
    active_stages: list[str] | None = None,
    verifying_keys: set[tuple[str, str]] | None = None,
) -> str:
    short = STAGE_SHORT_NAMES.get(row.stage, row.stage)
    if row.status == STATUS_SKIPPED:
        reason = state.get_skip_reason(run_id, row.target_label, row.stage)
        if reason in (
            SKIP_REASON_STREAM,
            SKIP_REASON_NOT_SELECTED,
            SKIP_REASON_SUPERSEDED,
        ):
            return f"{short}:n/a"
    stages = active_stages if active_stages is not None else state.get_active_stages(run_id)
    key = (row.target_label, row.stage)
    if verifying_keys and key in verifying_keys:
        return f"{short}:scan"
    needs_verify = False
    if row.status == STATUS_EXTERNAL:
        needs_verify = artifact_verify_needed(
            state, run_id, row.target_label, row.stage, stages
        )
    elif row.status == STATUS_PENDING and row.stage in stages:
        needs_verify = True
    if needs_verify and not state.external_verify_complete(
        run_id, row.target_label, row.stage
    ):
        return f"{short}:sc_q"
    return f"{short}:{row.status[:4]}"


def format_target_status_line(
    state: PipelineState,
    run_id: str,
    target_label: str,
    *,
    handoff_root: str | None = None,
) -> str | None:
    rows = [r for r in state.list_stage_runs(run_id) if r.target_label == target_label]
    if not rows:
        return None
    stage_order = {name: i for i, name in enumerate(STAGE_NAMES)}
    rows_for_target = sorted(rows, key=lambda r: stage_order.get(r.stage, len(STAGE_NAMES)))
    verifying_keys: set[tuple[str, str]] = set()
    if handoff_root:
        from syndiff_pipeline.template_runner.verify_status import read_verify_active_keys

        verifying_keys = set(read_verify_active_keys(handoff_root, run_id))
    active_stages = state.get_active_stages(run_id)
    parts = [
        _format_stage_status_short(
            state,
            run_id,
            r,
            active_stages=active_stages,
            verifying_keys=verifying_keys,
        )
        for r in rows_for_target
    ]
    return f"  {target_label}: {' | '.join(parts)}"


def format_run_report(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    handoff_root: str | None = None,
    header: str,
    include_status_grid: bool = True,
    max_chars: int = 1900,
) -> str:
    """Single-string report; may omit trailing grid rows when over *max_chars*."""
    body_lines = [header]
    body_lines.extend(
        format_progress_lines(
            state,
            run_id,
            runs_root,
            handoff_root=handoff_root,
            include_running_detail=True,
        )
    )
    if include_status_grid:
        body_lines.append("")
        grid = format_status_grid(state, run_id, handoff_root=handoff_root)
        if grid:
            body_lines.extend(
                _truncate_grid(grid, max_chars=max_chars - len("\n".join(body_lines)) - 1)
            )
        else:
            body_lines.append("  (no stage rows)")
    text = "\n".join(body_lines)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n… (truncated)"
    return text


def format_run_report_messages(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    handoff_root: str | None = None,
    header: str,
    include_status_grid: bool = True,
    max_chars: int = 1900,
) -> list[str]:
    """Discord-sized chunks; splits across messages instead of truncating."""
    body_lines = [header]
    body_lines.extend(
        format_progress_lines(
            state,
            run_id,
            runs_root,
            handoff_root=handoff_root,
            include_running_detail=True,
        )
    )
    progress_text = "\n".join(body_lines)

    if not include_status_grid:
        return pack_message_lines(body_lines, max_chars=max_chars)

    grid = format_status_grid(state, run_id, handoff_root=handoff_root)
    if not grid:
        body_lines.append("  (no stage rows)")
        return pack_message_lines(body_lines, max_chars=max_chars)

    combined_lines = body_lines + [""] + grid
    if len("\n".join(combined_lines)) <= max_chars:
        return ["\n".join(combined_lines)]

    messages = pack_message_lines(body_lines, max_chars=max_chars)
    messages.extend(
        pack_message_lines([_continuation_header(header), ""] + grid, max_chars=max_chars)
    )
    return messages


def _continuation_header(header: str) -> str:
    first = header.split("\n", 1)[0]
    if "]" in first:
        return f"{first[: first.index(']') + 1]} status grid (continued)"
    return "(continued)"


def _line_chars(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def pack_message_lines(lines: list[str], *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return []
    messages: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        need = len(line) + (1 if current else 0)
        if current and current_len + need > max_chars:
            messages.append("\n".join(current))
            current = [line]
            current_len = len(line)
        elif need > max_chars:
            if current:
                messages.append("\n".join(current))
                current = []
                current_len = 0
            start = 0
            while start < len(line):
                end = min(start + max_chars, len(line))
                messages.append(line[start:end])
                start = end
        else:
            current.append(line)
            current_len += need
    if current:
        messages.append("\n".join(current))
    return messages


def _truncate_grid(grid: list[str], *, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return ["  … (status grid omitted)"]
    out: list[str] = []
    used = 0
    for line in grid:
        need = len(line) + 1
        if used + need > max_chars:
            remaining = len(grid) - len(out)
            if remaining > 0:
                out.append(f"  … and {remaining} more targets")
            break
        out.append(line)
        used += need
    return out
