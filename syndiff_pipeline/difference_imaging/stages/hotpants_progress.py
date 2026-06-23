"""Sidecar JSON progress for Hotpants frame processing.

Workspace copy lives under the paired meta dir (``ws/hp_m/hotpants.progress.json`` for
``ws/hp_d/``). CLI mirror stays beside ``per_target/<label>/diff.log`` as
``diff.hotpants.progress.json`` (used by ``syndiff progress`` / stage_progress).

Progress counters are updated from the parent process after each frame completes
(see :func:`hotpants_loop` parallel branch). Updates use atomic write via a
temporary file so they remain reliable on NFS mounts without working ``flock``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROGRESS_FILENAME = "hotpants.progress.json"
CLI_PROGRESS_FILENAME = "diff.hotpants.progress.json"


def progress_path_for_meta_workspace(meta_dir: Path | str) -> Path:
    """Canonical per-pass sidecar under ``ws/{prefix}_m/``."""
    return Path(meta_dir).expanduser().resolve() / PROGRESS_FILENAME


def progress_path_for_diffs_workspace(diffs_dir: Path | str) -> Path:
    """Resolve meta workspace from *diffs_dir* basename (``hp_d`` → ``hp_m``)."""
    from syndiff_pipeline.difference_imaging.support.paths import meta_workspace_label

    diffs_path = Path(diffs_dir).expanduser().resolve()
    meta_dir = diffs_path.parent / meta_workspace_label(diffs_path.name)
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
    """Read-modify-write via :func:`_write_locked` (parent process only)."""
    state = read_progress(path) or {}
    mutator(state)
    state["updated_at"] = _utc_now_iso()
    _write_locked(path, state)


def init_progress(
    path: Path | str,
    *,
    diffs_label: str,
    round_id: int,
    science: str,
    frames_total: int,
) -> None:
    """Create or reset a Hotpants progress sidecar before frame processing."""
    payload = {
        "diffs_label": str(diffs_label),
        "round_id": int(round_id),
        "science": str(science),
        "frames_total": int(frames_total),
        "frames_done": 0,
        "frames_ok": 0,
        "phase": "running",
        "updated_at": _utc_now_iso(),
    }
    _write_locked(Path(path), payload)


def mark_frame_done(path: Path | str, *, success: bool) -> None:
    """Increment frame counters (atomic replace; safe on NFS)."""

    def mutator(state: dict[str, Any]) -> None:
        total = int(state.get("frames_total", 0))
        done = int(state.get("frames_done", 0)) + 1
        if total > 0:
            done = min(done, total)
        state["frames_done"] = done
        if success:
            ok = int(state.get("frames_ok", 0)) + 1
            if total > 0:
                ok = min(ok, total)
            state["frames_ok"] = ok
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
    label = str(data.get("diffs_label", "?"))
    done = int(data.get("frames_done", 0))
    total = int(data.get("frames_total", 0))
    phase = str(data.get("phase", "running"))
    if total <= 0:
        return None
    if phase == "complete":
        ok = int(data.get("frames_ok", done))
        return f"hotpants {label} complete {ok}/{total}"
    return f"hotpants {label} {done}/{total}"


def init_progress_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    diffs_label: str,
    round_id: int,
    science: str,
    frames_total: int,
) -> None:
    """Initialize workspace and optional CLI mirror sidecars."""
    kwargs = {
        "diffs_label": diffs_label,
        "round_id": round_id,
        "science": science,
        "frames_total": frames_total,
    }
    init_progress(workspace_path, **kwargs)
    if cli_path is not None:
        init_progress(cli_path, **kwargs)


def record_frame_progress(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    success: bool,
) -> None:
    """Increment counters on workspace and optional CLI mirror sidecars."""
    mark_frame_done(workspace_path, success=success)
    if cli_path is not None:
        mark_frame_done(cli_path, success=success)


def set_progress_phase_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    phase: str,
) -> None:
    """Set phase on workspace and optional CLI mirror sidecars."""
    set_progress_phase(workspace_path, phase)
    if cli_path is not None:
        set_progress_phase(cli_path, phase)
