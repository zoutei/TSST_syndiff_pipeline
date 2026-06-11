"""Sidecar JSON progress for forced-photometry epoch workers."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROGRESS_FILENAME = "photometry.progress.json"
CLI_PROGRESS_FILENAME = "diff.photometry.progress.json"


def progress_path_for_output_workspace(output_dir: Path | str) -> Path:
    """Canonical per-pass sidecar under ``ws/{output_label}/``."""
    return Path(output_dir).expanduser().resolve() / PROGRESS_FILENAME


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


def _update_locked(path: Path, mutator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read()
            if raw.strip():
                state = json.loads(raw)
            else:
                state = {}
            mutator(state)
            state["updated_at"] = _utc_now_iso()
            fh.seek(0)
            fh.truncate(0)
            json.dump(state, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def init_progress(
    path: Path | str,
    *,
    output_label: str,
    diffs_input: str,
    n_sources: int,
    epochs_total: int,
    phase: str = "flux",
) -> None:
    """Create or reset a photometry progress sidecar."""
    payload = {
        "output_label": str(output_label),
        "diffs_input": str(diffs_input),
        "n_sources": int(n_sources),
        "epochs_total": int(epochs_total),
        "epochs_done": 0,
        "phase": phase,
        "updated_at": _utc_now_iso(),
    }
    _write_locked(Path(path), payload)


def mark_epoch_done(path: Path | str) -> None:
    """Increment ``epochs_done`` under an exclusive file lock."""

    def mutator(state: dict[str, Any]) -> None:
        total = int(state.get("epochs_total", 0))
        done = int(state.get("epochs_done", 0)) + 1
        if total > 0:
            done = min(done, total)
        state["epochs_done"] = done

    _update_locked(Path(path), mutator)


def reset_epochs_done(path: Path | str, *, phase: str) -> None:
    """Reset the epoch counter and set lifecycle phase (e.g. cutouts → flux)."""

    def mutator(state: dict[str, Any]) -> None:
        state["epochs_done"] = 0
        state["phase"] = phase

    _update_locked(Path(path), mutator)


def set_progress_phase(path: Path | str, phase: str) -> None:
    """Update lifecycle phase (``cutouts`` / ``flux`` / ``complete`` / ``failed``)."""
    path = Path(path)

    def mutator(state: dict[str, Any]) -> None:
        state["phase"] = phase

    if path.is_file():
        _update_locked(path, mutator)
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
    label = str(data.get("output_label", "?"))
    done = int(data.get("epochs_done", 0))
    total = int(data.get("epochs_total", 0))
    phase = str(data.get("phase", "flux"))
    n_src = int(data.get("n_sources", 1))
    if total <= 0:
        return None
    src_part = f" ({n_src} src)" if n_src > 1 else ""
    if phase == "complete":
        return f"photometry {label}{src_part} complete {done}/{total}"
    if phase == "cutouts":
        return f"photometry {label}{src_part} cutouts {done}/{total}"
    return f"photometry {label}{src_part} {done}/{total}"


def init_progress_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    output_label: str,
    diffs_input: str,
    n_sources: int,
    epochs_total: int,
    phase: str = "flux",
) -> None:
    kwargs = {
        "output_label": output_label,
        "diffs_input": diffs_input,
        "n_sources": n_sources,
        "epochs_total": epochs_total,
        "phase": phase,
    }
    init_progress(workspace_path, **kwargs)
    if cli_path is not None:
        init_progress(cli_path, **kwargs)


def record_epoch_progress(
    workspace_path: Path | str,
    cli_path: Path | str | None,
) -> None:
    mark_epoch_done(workspace_path)
    if cli_path is not None:
        mark_epoch_done(cli_path)


def set_progress_phase_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    phase: str,
) -> None:
    set_progress_phase(workspace_path, phase)
    if cli_path is not None:
        set_progress_phase(cli_path, phase)


def reset_epochs_done_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    phase: str,
) -> None:
    reset_epochs_done(workspace_path, phase=phase)
    if cli_path is not None:
        reset_epochs_done(cli_path, phase=phase)
