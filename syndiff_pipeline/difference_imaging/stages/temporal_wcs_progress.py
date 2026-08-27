"""Sidecar progress for the temporal-WCS per-frame fitting loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROGRESS_FILENAME = "temporal_wcs.progress.json"
CLI_PROGRESS_FILENAME = "diff.temporal_wcs.progress.json"


def progress_path_for_output_workspace(output_dir: Path | str) -> Path:
    return Path(output_dir).expanduser().resolve() / PROGRESS_FILENAME


def progress_path_for_diff_log(log_path: Path | str) -> Path:
    return Path(log_path).expanduser().resolve().parent / CLI_PROGRESS_FILENAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _update(path: Path | str, mutator) -> None:
    state = read_progress(path) or {}
    mutator(state)
    state["updated_at"] = _now()
    _write(path, state)


def init_progress(path: Path | str, *, frames_total: int) -> None:
    _write(path, {"frames_total": int(frames_total), "frames_done": 0,
                  "frames_ok": 0, "phase": "fitting", "updated_at": _now()})


def mark_frame_done(path: Path | str, *, success: bool) -> None:
    def mutate(state: dict[str, Any]) -> None:
        total = int(state.get("frames_total", 0))
        state["frames_done"] = min(total, int(state.get("frames_done", 0)) + 1)
        if success:
            state["frames_ok"] = min(total, int(state.get("frames_ok", 0)) + 1)
    _update(path, mutate)


def set_progress_phase(path: Path | str, phase: str) -> None:
    if Path(path).is_file():
        _update(path, lambda state: state.update(phase=str(phase)))
    else:
        _write(path, {"phase": str(phase), "updated_at": _now()})


def read_progress(path: Path | str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def format_progress_text(data: dict[str, Any]) -> Optional[str]:
    done = int(data.get("frames_done", 0))
    total = int(data.get("frames_total", 0))
    if total <= 0:
        return None
    phase = str(data.get("phase", "fitting"))
    if phase == "complete":
        return f"temporal_wcs complete {int(data.get('frames_ok', done))}/{total}"
    if phase == "publishing":
        return f"temporal_wcs publishing ({done}/{total} fits)"
    return f"temporal_wcs {done}/{total}"


def init_progress_pair(workspace_path, cli_path, *, frames_total: int) -> None:
    init_progress(workspace_path, frames_total=frames_total)
    if cli_path is not None:
        init_progress(cli_path, frames_total=frames_total)


def record_frame_progress(workspace_path, cli_path, *, success: bool) -> None:
    mark_frame_done(workspace_path, success=success)
    if cli_path is not None:
        mark_frame_done(cli_path, success=success)


def set_progress_phase_pair(workspace_path, cli_path, phase: str) -> None:
    set_progress_phase(workspace_path, phase)
    if cli_path is not None:
        set_progress_phase(cli_path, phase)
