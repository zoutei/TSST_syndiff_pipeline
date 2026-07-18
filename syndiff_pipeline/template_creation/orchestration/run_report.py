"""Format pipeline status/progress reports for CLI and notifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

from syndiff_pipeline.common.orchestration import logs, stage_liveness
from syndiff_pipeline.template_creation.orchestration.stage_progress import (
    PROGRESS_CLI_MAX_TAIL_SCAN_BYTES,
    read_log_progress,
)
from syndiff_pipeline.common.orchestration.state import (
    SKIP_REASON_NOT_SELECTED,
    SKIP_REASON_STREAM,
    SKIP_REASON_SUPERSEDED,
    STATUS_CANCELED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    RunDisplayContext,
    deps_satisfied_from_map,
    stage_needs_artifact_verify_display,
    stage_needs_artifact_verify_display_from_context,
)

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.state import PipelineState, StageRunRow


def _row_executor(row: StageRunRow, cfg) -> str | None:
    """Resolve executor for a stage row (DB value or frozen config default)."""
    if row.executor:
        return row.executor
    if cfg is not None:
        return cfg.stage_executor(row.stage)
    return None


def _condor_status_for_rows(
    running_rows: list[StageRunRow],
    *,
    runs_root: str,
    run_id: str,
    cfg=None,
) -> dict[int, int | None]:
    """Batch-query Condor JobStatus for running Condor stages."""
    from syndiff_pipeline.common.orchestration import condor
    from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config

    resolved_cfg = cfg
    cluster_ids: list[int] = []
    for row in running_rows:
        executor = _row_executor(row, resolved_cfg)
        if executor is None:
            if resolved_cfg is None:
                cfg_path = logs.run_config_path(logs.run_dir(runs_root, run_id))
                if cfg_path.is_file():
                    resolved_cfg = load_runner_config(cfg_path)
            executor = _row_executor(row, resolved_cfg)
        if executor != "condor" or row.native_id is None:
            continue
        cluster_ids.append(int(row.native_id))
    if not cluster_ids:
        return {}
    try:
        queried = condor.query_clusters_display(cluster_ids)
        return {cluster_id: status for cluster_id, (status, _) in queried.items()}
    except Exception:
        return {}


def _orphaned_active_stage_rows(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    cfg=None,
) -> list:
    """Failed/canceled Condor rows whose logs/sidecars still show live work."""
    from syndiff_pipeline.template_creation.orchestration.runner_config import (
        load_runner_config,
    )

    resolved_cfg = cfg
    rows = []
    for row in state.list_stage_runs(run_id):
        if row.status not in (STATUS_FAILED, STATUS_CANCELED):
            continue
        executor = _row_executor(row, resolved_cfg)
        if executor is None:
            if resolved_cfg is None:
                cfg_path = logs.run_config_path(logs.run_dir(runs_root, run_id))
                if cfg_path.is_file():
                    resolved_cfg = load_runner_config(cfg_path)
            executor = _row_executor(row, resolved_cfg)
        if executor != "condor":
            continue
        log_path = row.log_path or str(
            logs.target_log_path(runs_root, run_id, row.target_label, row.stage)
        )
        if not stage_liveness.stage_output_recently_active(log_path, row.stage):
            continue
        rows.append(row)
    return rows


def _progress_detail_lines(
    rows: list,
    *,
    runs_root: str,
    run_id: str,
    cfg,
    cluster_status: dict[int, int | None],
    orphan_note: bool = False,
) -> list[str]:
    from syndiff_pipeline.pipeline_spec import stage_short_names

    short_names = stage_short_names()
    lines: list[str] = []
    for row in sorted(rows, key=lambda r: (r.target_label, r.stage)):
        log_path = row.log_path or str(
            logs.target_log_path(runs_root, run_id, row.target_label, row.stage)
        )
        prog = read_log_progress(
            log_path,
            row.stage,
            started_at=row.started_at,
            max_scan_bytes=PROGRESS_CLI_MAX_TAIL_SCAN_BYTES,
        )
        short = short_names.get(row.stage, row.stage)
        condor_text = _condor_detail_text(row, cluster_status, cfg)
        if not condor_text and row.native_id is None:
            from syndiff_pipeline.common.orchestration import condor

            cluster_id = condor.read_recorded_cluster_id(
                runs_root, run_id, row.target_label, row.stage
            )
            if cluster_id is not None:
                condor_text = condor.format_condor_job_suffix(
                    int(cluster_id), cluster_status.get(int(cluster_id))
                )
        orphan_suffix = " (orphaned bookkeeping)" if orphan_note else ""
        if prog:
            extra = f" {condor_text}" if condor_text else ""
            lines.append(f"  {row.target_label} {short}: {prog.text}{extra}{orphan_suffix}")
        elif condor_text:
            lines.append(
                f"  {row.target_label} {short}: {condor_text} "
                f"(no log progress yet){orphan_suffix}"
            )
        else:
            lines.append(
                f"  {row.target_label} {short}: (no log progress yet){orphan_suffix}"
            )
    return lines


def _condor_detail_text(
    row: StageRunRow,
    cluster_status: dict[int, int | None],
    cfg,
) -> str | None:
    """Condor queue detail for a running stage row, or None when not applicable."""
    from syndiff_pipeline.common.orchestration import condor

    executor = _row_executor(row, cfg)
    if executor != "condor":
        return None
    if row.native_id is None:
        return "condor unsubmitted"
    suffix = condor.format_condor_job_suffix(int(row.native_id), cluster_status.get(int(row.native_id)))
    return suffix or None


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
    workspace_root: str | None = None,
    include_running_detail: bool = True,
) -> list[str]:
    """Format progress lines.
    
    Parameters
    ----------
    state : PipelineState
    run_id : str
    runs_root : str
    workspace_root : str | None, optional, default ``None``
    include_running_detail : bool, optional, default ``True``
    
    Returns
    -------
    list[str]"""
    from syndiff_pipeline.common.orchestration.verify_status import read_verify_run_status

    counts = state.count_by_status(run_id)
    run = state.get_run(run_id) or {}
    count_parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    lines: list[str] = []
    verify_backlog = 0
    if count_parts:
        count_line = " ".join(count_parts)
        if workspace_root:
            verify_status = read_verify_run_status(workspace_root, run_id)
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

    from syndiff_pipeline.template_creation.orchestration.runner_config import load_runner_config

    running = state.running_stage_runs(run_id)
    orphaned = _orphaned_active_stage_rows(state, run_id, runs_root)
    if not running and not orphaned:
        lines.append("  (no running tasks)")
        return lines

    cfg = None
    all_rows = list(running) + list(orphaned)
    if any(row.executor is None for row in all_rows):
        cfg_path = logs.run_config_path(logs.run_dir(runs_root, run_id))
        if cfg_path.is_file():
            cfg = load_runner_config(cfg_path)

    cluster_status = _condor_status_for_rows(
        all_rows, runs_root=runs_root, run_id=run_id, cfg=cfg
    )
    from syndiff_pipeline.common.orchestration import condor

    for row in orphaned:
        if row.native_id is not None:
            continue
        cluster_id = condor.read_recorded_cluster_id(
            runs_root, run_id, row.target_label, row.stage
        )
        if cluster_id is None or cluster_id in cluster_status:
            continue
        try:
            queried = condor.query_clusters_display([int(cluster_id)])
            cluster_status.update(
                {cid: status for cid, (status, _) in queried.items()}
            )
        except Exception:
            pass

    lines.append("")
    if running:
        lines.extend(
            _progress_detail_lines(
                running,
                runs_root=runs_root,
                run_id=run_id,
                cfg=cfg,
                cluster_status=cluster_status,
            )
        )
    if orphaned:
        if running:
            lines.append("")
        lines.extend(
            _progress_detail_lines(
                orphaned,
                runs_root=runs_root,
                run_id=run_id,
                cfg=cfg,
                cluster_status=cluster_status,
                orphan_note=True,
            )
        )
    return lines


def format_status_grid(
    state: PipelineState,
    run_id: str,
    *,
    workspace_root: str | None = None,
) -> list[str]:
    """Format status grid.
    
    Parameters
    ----------
    state : PipelineState
    run_id : str
    workspace_root : str | None, optional, default ``None``
    
    Returns
    -------
    list[str]"""
    display = state.load_run_display_context(run_id)
    by_target: dict[str, list] = {}
    for r in display.rows:
        by_target.setdefault(r.target_label, []).append(r)
    from syndiff_pipeline.pipeline_spec import stage_names

    names = stage_names()
    stage_order = {name: i for i, name in enumerate(names)}

    def _stage_sort_key(row) -> int:
        """Stage sort key.
        
        Parameters
        ----------
        row
        
        Returns
        -------
        int"""
        return stage_order.get(row.stage, len(names))

    verifying_keys: set[tuple[str, str]] = set()
    if workspace_root:
        from syndiff_pipeline.common.orchestration.verify_status import read_verify_active_keys

        verifying_keys = set(read_verify_active_keys(workspace_root, run_id))

    lines: list[str] = []
    for label in sorted(by_target):
        rows_for_target = sorted(by_target[label], key=_stage_sort_key)
        parts = [
            _format_stage_status_short(
                state,
                run_id,
                r,
                active_stages=display.active_stages,
                verifying_keys=verifying_keys,
                display=display,
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
    display: RunDisplayContext | None = None,
) -> str:
    """Format stage status short.
    
    Parameters
    ----------
    state : PipelineState
    run_id : str
    row
    active_stages : list[str] | None, optional, default ``None``
    verifying_keys : set[tuple[str, str]] | None, optional, default ``None``
    display : RunDisplayContext | None, optional, default ``None``
    
    Returns
    -------
    str"""
    from syndiff_pipeline.pipeline_spec import stage_short_names

    short = stage_short_names().get(row.stage, row.stage)
    key = (row.target_label, row.stage)
    if row.status == STATUS_SKIPPED:
        if display is not None:
            reason = display.skip_reasons.get(key)
        else:
            reason = state.get_skip_reason(run_id, row.target_label, row.stage)
        if reason in (
            SKIP_REASON_STREAM,
            SKIP_REASON_NOT_SELECTED,
            SKIP_REASON_SUPERSEDED,
        ):
            return f"{short}:n/a"
    stages = (
        active_stages
        if active_stages is not None
        else (display.active_stages if display is not None else state.get_active_stages(run_id))
    )
    if verifying_keys and key in verifying_keys:
        return f"{short}:scan"
    if display is not None:
        needs_verify = stage_needs_artifact_verify_display_from_context(
            display,
            row.target_label,
            row.stage,
            row.status,
            stages,
            spec=state.pipeline_spec,
        )
        verify_complete = key in display.external_complete
        deps_ok = deps_satisfied_from_map(
            display.status_by_key,
            row.target_label,
            row.stage,
            spec=state.pipeline_spec,
        )
    else:
        needs_verify = stage_needs_artifact_verify_display(
            state, run_id, row.target_label, row.stage, row.status, stages
        )
        verify_complete = state.external_verify_complete(
            run_id, row.target_label, row.stage
        )
        deps_ok = state.deps_satisfied(run_id, row.target_label, row.stage)
    if needs_verify and not verify_complete:
        if row.status == STATUS_PENDING and not deps_ok:
            return f"{short}:pend"
        return f"{short}:sc_q"
    return f"{short}:{row.status[:4]}"


def format_target_status_line(
    state: PipelineState,
    run_id: str,
    target_label: str,
    *,
    workspace_root: str | None = None,
) -> str | None:
    """Format target status line.
    
    Parameters
    ----------
    state : PipelineState
    run_id : str
    target_label : str
    workspace_root : str | None, optional, default ``None``
    
    Returns
    -------
    str | None"""
    display = state.load_run_display_context(run_id)
    rows = [r for r in display.rows if r.target_label == target_label]
    if not rows:
        return None
    from syndiff_pipeline.pipeline_spec import stage_names

    names = stage_names()
    stage_order = {name: i for i, name in enumerate(names)}
    rows_for_target = sorted(rows, key=lambda r: stage_order.get(r.stage, len(names)))
    verifying_keys: set[tuple[str, str]] = set()
    if workspace_root:
        from syndiff_pipeline.common.orchestration.verify_status import read_verify_active_keys

        verifying_keys = set(read_verify_active_keys(workspace_root, run_id))
    parts = [
        _format_stage_status_short(
            state,
            run_id,
            r,
            active_stages=display.active_stages,
            verifying_keys=verifying_keys,
            display=display,
        )
        for r in rows_for_target
    ]
    return f"  {target_label}: {' | '.join(parts)}"


def format_run_report(
    state: PipelineState,
    run_id: str,
    runs_root: str,
    *,
    workspace_root: str | None = None,
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
            workspace_root=workspace_root,
            include_running_detail=True,
        )
    )
    if include_status_grid:
        body_lines.append("")
        grid = format_status_grid(state, run_id, workspace_root=workspace_root)
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
    workspace_root: str | None = None,
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
            workspace_root=workspace_root,
            include_running_detail=True,
        )
    )
    progress_text = "\n".join(body_lines)

    if not include_status_grid:
        return pack_message_lines(body_lines, max_chars=max_chars)

    grid = format_status_grid(state, run_id, workspace_root=workspace_root)
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
    """Continuation header.
    
    Parameters
    ----------
    header : str
    
    Returns
    -------
    str"""
    first = header.split("\n", 1)[0]
    if "]" in first:
        return f"{first[: first.index(']') + 1]} status grid (continued)"
    return "(continued)"


def _line_chars(lines: list[str]) -> int:
    """Line chars.
    
    Parameters
    ----------
    lines : list[str]
    
    Returns
    -------
    int"""
    return sum(len(line) + 1 for line in lines)


def pack_message_lines(lines: list[str], *, max_chars: int) -> list[str]:
    """Pack message lines.
    
    Parameters
    ----------
    lines : list[str]
    max_chars : int
    
    Returns
    -------
    list[str]"""
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
    """Truncate grid.
    
    Parameters
    ----------
    grid : list[str]
    max_chars : int
    
    Returns
    -------
    list[str]"""
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
