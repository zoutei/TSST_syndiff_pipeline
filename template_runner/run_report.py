"""Format pipeline status/progress reports for CLI and notifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syndiff_pipeline.template_runner.stage_progress import read_log_progress
from syndiff_pipeline.template_runner.state import STAGE_NAMES, STAGE_SHORT_NAMES

if TYPE_CHECKING:
    from syndiff_pipeline.template_runner.state import PipelineState


def format_progress_lines(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    state_db_path: str | None = None,
    include_running_detail: bool = True,
) -> list[str]:
    from syndiff_pipeline.template_runner.verify_status import read_verify_in_flight

    counts = state.count_by_status(run_id)
    run = state.get_run(run_id) or {}
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    line = f"run_id={run_id} status={run.get('status', '?')} " + " ".join(parts)
    if state_db_path:
        in_flight = read_verify_in_flight(state_db_path, run_id)
        if in_flight:
            line += f" verify_in_flight={in_flight}"
    if run.get("stall_reason"):
        line += f" stall_reason={run['stall_reason']!r}"
    lines = [line]

    if not include_running_detail:
        return lines

    running = state.running_stage_runs(run_id)
    if not running:
        lines.append("  (no running tasks)")
        return lines

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


def format_status_grid(state: PipelineState, run_id: str) -> list[str]:
    rows = state.list_stage_runs(run_id)
    by_target: dict[str, list] = {}
    for r in rows:
        by_target.setdefault(r.target_label, []).append(r)
    stage_order = {name: i for i, name in enumerate(STAGE_NAMES)}

    def _stage_sort_key(row) -> int:
        return stage_order.get(row.stage, len(STAGE_NAMES))

    lines: list[str] = []
    for label in sorted(by_target):
        rows_for_target = sorted(by_target[label], key=_stage_sort_key)
        parts = [
            f"{STAGE_SHORT_NAMES.get(r.stage, r.stage)}:{r.status[:4]}"
            for r in rows_for_target
        ]
        lines.append(f"  {label}: {' | '.join(parts)}")
    return lines


def format_target_status_line(state: PipelineState, run_id: str, target_label: str) -> str | None:
    rows = [r for r in state.list_stage_runs(run_id) if r.target_label == target_label]
    if not rows:
        return None
    stage_order = {name: i for i, name in enumerate(STAGE_NAMES)}
    rows_for_target = sorted(rows, key=lambda r: stage_order.get(r.stage, len(STAGE_NAMES)))
    parts = [
        f"{STAGE_SHORT_NAMES.get(r.stage, r.stage)}:{r.status[:4]}"
        for r in rows_for_target
    ]
    return f"  {target_label}: {' | '.join(parts)}"


def format_run_report(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    state_db_path: str | None = None,
    header: str,
    include_status_grid: bool = True,
    max_chars: int = 1900,
) -> str:
    body_lines = [header, ""]
    body_lines.extend(
        format_progress_lines(
            state,
            run_id,
            runs_root,
            state_db_path=state_db_path,
            include_running_detail=True,
        )
    )
    if include_status_grid:
        body_lines.append("")
        grid = format_status_grid(state, run_id)
        if grid:
            body_lines.extend(_truncate_grid(grid, max_chars=max_chars - _line_chars(body_lines)))
        else:
            body_lines.append("  (no stage rows)")
    text = "\n".join(body_lines)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n… (truncated)"
    return text


def _line_chars(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


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
