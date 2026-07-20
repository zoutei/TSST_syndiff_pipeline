"""Sidecar JSON progress for downsample parallel batch workers.

Batch counters are updated from the **parent** process as Parallel results are
drained (see :func:`run_downsample_pipeline`). Updates use atomic write via a
temporary file so they remain reliable on NFS mounts without working
``flock`` (same pattern as Hotpants / remap).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "downsample.progress.json"


def progress_path_for_log(log_path: Path | str) -> Path:
    """Resolve sidecar path beside ``per_target/<label>/downsample.log``."""
    return Path(log_path).expanduser().resolve().parent / PROGRESS_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds_since_iso(iso_str: str) -> float:
    """Wall-clock seconds since an ISO-8601 UTC timestamp."""
    started = datetime.fromisoformat(iso_str)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - started).total_seconds()


def _transition_phase(state: dict[str, Any], new_phase: str) -> None:
    """Record elapsed time for the outgoing phase and start the new one."""
    old_phase = state.get("phase")
    if old_phase and old_phase != new_phase:
        started = state.get("phase_started_at")
        if isinstance(started, str) and started:
            elapsed = _elapsed_seconds_since_iso(started)
            phase_times = state.get("phase_elapsed_s")
            if not isinstance(phase_times, dict):
                phase_times = {}
            phase_times[old_phase] = round(elapsed, 3)
            state["phase_elapsed_s"] = phase_times
    state["phase"] = new_phase
    state["phase_started_at"] = _utc_now_iso()


def _sum_batch_done(batches: dict[str, Any]) -> int:
    total = 0
    for entry in batches.values():
        if isinstance(entry, dict):
            total += int(entry.get("done", 0))
    return total


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _update_atomic(path: Path, mutator) -> None:
    """Read-modify-write via :func:`_write_atomic` (parent process only)."""
    state = read_progress(path) or {}
    mutator(state)
    state["updated_at"] = _utc_now_iso()
    _write_atomic(path, state)


def init_progress(
    path: Path | str,
    total_skycells: int,
    batch_sizes: list[int],
    *,
    oversampling_factor: int = 1,
) -> None:
    """Create or reset sidecar before parallel batch processing."""
    path = Path(path)
    batches = {
        str(i): {"size": int(size), "done": 0}
        for i, size in enumerate(batch_sizes)
    }

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, "parallel_batches")
        state["total_skycells"] = int(total_skycells)
        state["total_batches"] = len(batch_sizes)
        state["skycells_done"] = 0
        state["batches"] = batches
        state["oversampling_factor"] = int(oversampling_factor)

    if path.is_file():
        _update_atomic(path, mutator)
    else:
        payload: dict[str, Any] = {
            "total_skycells": int(total_skycells),
            "total_batches": len(batch_sizes),
            "skycells_done": 0,
            "batches": batches,
            "oversampling_factor": int(oversampling_factor),
        }
        _transition_phase(payload, "parallel_batches")
        payload["updated_at"] = _utc_now_iso()
        _write_atomic(path, payload)


def set_progress_phase(
    path: Path | str,
    phase: str,
    *,
    total_skycells: int | None = None,
    offsets_done: int | None = None,
    offsets_total: int | None = None,
    oversampling_factor: int | None = None,
) -> None:
    """Update lifecycle phase (and optional shift precompute counters)."""
    path = Path(path)

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, phase)
        if total_skycells is not None:
            state["total_skycells"] = int(total_skycells)
            state["skycells_done"] = int(total_skycells)
        if offsets_done is not None:
            state["offsets_done"] = int(offsets_done)
        if offsets_total is not None:
            state["offsets_total"] = int(offsets_total)
        if oversampling_factor is not None:
            state["oversampling_factor"] = int(oversampling_factor)

    if path.is_file():
        _update_atomic(path, mutator)
    else:
        payload: dict[str, Any] = {
            "skycells_done": 0,
            "total_skycells": int(total_skycells or 0),
        }
        _transition_phase(payload, phase)
        if offsets_done is not None:
            payload["offsets_done"] = int(offsets_done)
        if offsets_total is not None:
            payload["offsets_total"] = int(offsets_total)
        if oversampling_factor is not None:
            payload["oversampling_factor"] = int(oversampling_factor)
        payload["updated_at"] = _utc_now_iso()
        _write_atomic(path, payload)


def mark_skycell_done(path: Path | str, batch_idx: int) -> None:
    """Increment one skycell for *batch_idx* and recompute ``skycells_done``."""
    mark_skycells_done(path, batch_idx, 1)


def mark_skycells_done(path: Path | str, batch_idx: int, n: int) -> None:
    """Increment *n* skycells for *batch_idx* (parent process only)."""
    key = str(batch_idx)
    n = max(0, int(n))

    def mutator(state: dict[str, Any]) -> None:
        batches = state.setdefault("batches", {})
        entry = batches.setdefault(key, {"size": 0, "done": 0})
        size = int(entry.get("size", 0))
        done = int(entry.get("done", 0)) + n
        if size > 0:
            done = min(done, size)
        entry["done"] = done
        state["skycells_done"] = _sum_batch_done(batches)
        state["phase"] = "parallel_batches"

    _update_atomic(Path(path), mutator)


def read_progress(path: Path | str) -> dict[str, Any] | None:
    """Load sidecar state, or ``None`` if missing/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
