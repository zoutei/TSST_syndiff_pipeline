"""Sidecar JSON progress for the convolved_templates build loop.

Mirrors :mod:`hotpants_progress`'s workspace+CLI sidecar pair. Needed because
this stage can run for hours in field mode (one convolved template per
distinct ``group_id``, which can number in the thousands for a drift-tracked
SCC) with no other progress signal -- ``syndiff progress``/Discord otherwise
show "no log progress yet" for the entire duration (see production incident
2026-08-23, S20/C3/K3 tvwcs).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROGRESS_FILENAME = "convolved_templates.progress.json"
CLI_PROGRESS_FILENAME = "diff.convolved_templates.progress.json"


def progress_path_for_meta_workspace(meta_dir: Path | str) -> Path:
    """Canonical sidecar under ``ws/{prefix}_m/`` (mirrors hotpants_progress)."""
    return Path(meta_dir).expanduser().resolve() / PROGRESS_FILENAME


def progress_path_for_convolved_workspace(convolved_dir: Path | str) -> Path:
    """Resolve meta workspace from *convolved_dir* basename (``tmpl_conv`` -> ``tmpl_conv_m``)."""
    from syndiff_pipeline.difference_imaging.support.paths import meta_workspace_label

    conv_path = Path(convolved_dir).expanduser().resolve()
    meta_dir = conv_path.parent / meta_workspace_label(conv_path.name)
    return progress_path_for_meta_workspace(meta_dir)


def progress_path_for_diff_log(log_path: Path | str) -> Path:
    """Active-pass mirror beside ``per_target/<label>/diff.log``."""
    return Path(log_path).expanduser().resolve().parent / CLI_PROGRESS_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_locked(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _update_atomic(path: Path, mutator) -> None:
    state = read_progress(path) or {}
    mutator(state)
    state["updated_at"] = _utc_now_iso()
    _write_locked(path, state)


def init_progress(path: Path | str, *, groups_total: int) -> None:
    """Create or reset a convolved_templates progress sidecar before the build loop."""
    payload = {
        "groups_total": int(groups_total),
        "groups_done": 0,
        "groups_built": 0,
        "phase": "running",
        "updated_at": _utc_now_iso(),
    }
    _write_locked(Path(path), payload)


def mark_group_done(path: Path | str, *, built: bool) -> None:
    """Increment group counters (atomic replace; safe on NFS).

    ``built`` distinguishes a freshly-convolved group from one that was
    already on disk and skipped -- both count toward ``groups_done``.
    """

    def mutator(state: dict[str, Any]) -> None:
        total = int(state.get("groups_total", 0))
        done = int(state.get("groups_done", 0)) + 1
        if total > 0:
            done = min(done, total)
        state["groups_done"] = done
        if built:
            state["groups_built"] = int(state.get("groups_built", 0)) + 1
        state["phase"] = "running"

    _update_atomic(Path(path), mutator)


def set_progress_phase(path: Path | str, phase: str) -> None:
    """Update lifecycle phase (``running`` / ``complete`` / ``failed``)."""
    path = Path(path)

    def mutator(state: dict[str, Any]) -> None:
        state["phase"] = phase

    if path.is_file():
        _update_atomic(path, mutator)
    else:
        _write_locked(path, {"phase": phase, "updated_at": _utc_now_iso()})


def read_progress(path: Path | str) -> dict[str, Any] | None:
    """Load sidecar state, or ``None`` if missing/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def format_progress_text(data: dict[str, Any]) -> Optional[str]:
    """Human-readable progress line for CLI / log parsers."""
    done = int(data.get("groups_done", 0))
    total = int(data.get("groups_total", 0))
    phase = str(data.get("phase", "running"))
    if total <= 0:
        return None
    if phase == "complete":
        return f"convolved_templates complete {done}/{total}"
    return f"convolved_templates {done}/{total} groups"


def init_progress_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    groups_total: int,
) -> None:
    """Initialize workspace and optional CLI mirror sidecars."""
    init_progress(workspace_path, groups_total=groups_total)
    if cli_path is not None:
        init_progress(cli_path, groups_total=groups_total)


def record_group_progress(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    built: bool,
) -> None:
    """Increment counters on workspace and optional CLI mirror sidecars."""
    mark_group_done(workspace_path, built=built)
    if cli_path is not None:
        mark_group_done(cli_path, built=built)


def set_progress_phase_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    phase: str,
) -> None:
    """Set phase on workspace and optional CLI mirror sidecars."""
    set_progress_phase(workspace_path, phase)
    if cli_path is not None:
        set_progress_phase(cli_path, phase)
