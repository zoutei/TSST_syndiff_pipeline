"""Sidecar JSON progress for gridded ePSF frame processing.

Workspace copy lives under the ePSF output dir (``ws/epsf_r1/epsf.progress.json``).
CLI mirror stays beside ``per_target/<label>/diff.log`` as
``diff.epsf.progress.json`` (used by ``syndiff progress`` / stage_progress).

Workers update sidecars after each frame (file-locked). ``syndiff progress`` also
merges live ``*_gridded_epsf.npz`` counts when *output_dir* is recorded.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROGRESS_FILENAME = "epsf.progress.json"
CLI_PROGRESS_FILENAME = "diff.epsf.progress.json"
ARTIFACT_MERGE_STALE_SECONDS = 30.0


def progress_path_for_output_workspace(output_dir: Path | str) -> Path:
    """Canonical per-pass sidecar under ``ws/{epsf_label}/``."""
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
    """Read-modify-write under an exclusive lock (safe for loky workers on NFS)."""
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


def _sidecar_is_stale(data: dict[str, Any]) -> bool:
    """Return True when the sidecar has not been updated recently."""
    updated = data.get("updated_at")
    if not updated:
        return True
    try:
        ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_s = (datetime.now(timezone.utc) - ts).total_seconds()
    return age_s > ARTIFACT_MERGE_STALE_SECONDS


def count_gridded_epsf_artifacts(output_dir: Path | str) -> int:
    """
    Count completed per-frame gridded ePSF models.

    Prefers the workspace's ``gridded_epsf_index.json`` (which may point at
    SCC-lane paths under ``data_root``, not just this workspace directory) and
    falls back to a local glob for older/index-less workspaces.
    """
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        return 0
    from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
        load_gridded_epsf_index,
    )

    index = load_gridded_epsf_index(str(root))
    if index:
        return sum(1 for p in index.values() if os.path.isfile(p))
    return sum(1 for _ in root.glob("*_gridded_epsf.npz"))


def init_progress(
    path: Path | str,
    *,
    epsf_label: str,
    diffs_input: str,
    round_id: int,
    frames_total: int,
    output_dir: str | None = None,
) -> None:
    """Create or reset an ePSF progress sidecar before frame processing."""
    payload = {
        "epsf_label": str(epsf_label),
        "diffs_input": str(diffs_input),
        "round_id": int(round_id),
        "frames_total": int(frames_total),
        "frames_done": 0,
        "frames_ok": 0,
        "phase": "running",
        "updated_at": _utc_now_iso(),
    }
    if output_dir is not None:
        payload["output_dir"] = str(Path(output_dir).expanduser().resolve())
    _write_locked(Path(path), payload)


def mark_frame_done(path: Path | str, *, success: bool) -> None:
    """Increment frame counters under an exclusive file lock."""

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

    _update_locked(Path(path), mutator)


def set_progress_phase(path: Path | str, phase: str) -> None:
    """Update lifecycle phase (``running`` / ``complete`` / ``failed``)."""
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


def merge_progress_with_artifacts(
    data: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Raise ``frames_done`` / ``frames_ok`` to match on-disk npz count.

    Handles stale sidecars when loky workers finish frames before the parent
    process drains the result generator. Skips the NFS glob unless *force* is
    True or the sidecar ``updated_at`` is older than
    :data:`ARTIFACT_MERGE_STALE_SECONDS`.
    """
    out_dir = data.get("output_dir")
    if not out_dir:
        return data
    if not force and not _sidecar_is_stale(data):
        return data
    artifact_n = count_gridded_epsf_artifacts(out_dir)
    if artifact_n <= 0:
        return data
    merged = dict(data)
    total = int(merged.get("frames_total", 0))
    done = max(int(merged.get("frames_done", 0)), artifact_n)
    ok = max(int(merged.get("frames_ok", 0)), artifact_n)
    if total > 0:
        done = min(done, total)
        ok = min(ok, total)
    merged["frames_done"] = done
    merged["frames_ok"] = ok
    return merged


def read_progress_merged(
    path: Path | str,
    *,
    force_artifact_merge: bool = False,
) -> dict[str, Any] | None:
    """Load sidecar and merge artifact counts from *output_dir* when appropriate."""
    data = read_progress(path)
    if not data:
        return None
    return merge_progress_with_artifacts(data, force=force_artifact_merge)


def refresh_progress_pair_from_artifacts(
    workspace_path: Path | str,
    cli_path: Path | str | None,
) -> None:
    """Sync sidecar counters to the npz artifact count (parent, end of run)."""
    ws = Path(workspace_path)
    data = read_progress(ws)
    if not data or not data.get("output_dir"):
        return
    merged = merge_progress_with_artifacts(data, force=True)
    if merged.get("frames_done") == data.get("frames_done") and merged.get(
        "frames_ok"
    ) == data.get("frames_ok"):
        return
    merged["updated_at"] = _utc_now_iso()
    _write_locked(ws, merged)
    if cli_path is not None:
        _write_locked(Path(cli_path), merged)


def format_progress_text(data: dict[str, Any]) -> Optional[str]:
    """Human-readable progress line for CLI / log parsers."""
    data = merge_progress_with_artifacts(data)
    label = str(data.get("epsf_label", "?"))
    done = int(data.get("frames_done", 0))
    total = int(data.get("frames_total", 0))
    phase = str(data.get("phase", "running"))
    if total <= 0:
        return None
    if phase == "complete":
        ok = int(data.get("frames_ok", done))
        return f"epsf {label} complete {ok}/{total}"
    return f"epsf {label} {done}/{total}"


def init_progress_pair(
    workspace_path: Path | str,
    cli_path: Path | str | None,
    *,
    epsf_label: str,
    diffs_input: str,
    round_id: int,
    frames_total: int,
    output_dir: str | None = None,
) -> None:
    """Initialize workspace and optional CLI mirror sidecars."""
    kwargs = {
        "epsf_label": epsf_label,
        "diffs_input": diffs_input,
        "round_id": round_id,
        "frames_total": frames_total,
        "output_dir": output_dir,
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
