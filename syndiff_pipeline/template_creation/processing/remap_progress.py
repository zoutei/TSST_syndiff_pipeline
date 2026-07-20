"""Sidecar JSON progress for field remap (shift schedule, grouping, Exact cache).

Counters are updated from the **parent** process as Parallel results are
drained (see :func:`run_field_remap_scc`). Updates use atomic write via a
temporary file so they remain reliable on NFS mounts without working
``flock`` (same pattern as Hotpants).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "remap.progress.json"


def progress_path_for_log(log_path: Path | str) -> Path:
    """Resolve sidecar path beside ``per_target/<label>/remap.log``."""
    return Path(log_path).expanduser().resolve().parent / PROGRESS_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds_since_iso(iso_str: str) -> float:
    started = datetime.fromisoformat(iso_str)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - started).total_seconds()


def _transition_phase(state: dict[str, Any], new_phase: str) -> None:
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
    *,
    oversampling_factor: int = 1,
) -> None:
    """Create or reset sidecar at remap start (phase ``shift_schedule``)."""

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, "shift_schedule")
        state["oversampling_factor"] = int(oversampling_factor)

    path = Path(path)
    if path.is_file():
        _update_atomic(path, mutator)
    else:
        payload: dict[str, Any] = {"oversampling_factor": int(oversampling_factor)}
        _transition_phase(payload, "shift_schedule")
        payload["updated_at"] = _utc_now_iso()
        _write_atomic(path, payload)


def set_progress_phase(
    path: Path | str,
    phase: str,
    *,
    exact_done: int | None = None,
    exact_total: int | None = None,
    exact_l4a_done: int | None = None,
    exact_l4a_total: int | None = None,
    exact_l4b_done: int | None = None,
    exact_l4b_total: int | None = None,
    oversampling_factor: int | None = None,
) -> None:
    """Update lifecycle phase (and optional exact-cache counters)."""
    path = Path(path)

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, phase)
        if exact_done is not None:
            state["exact_done"] = int(exact_done)
        if exact_total is not None:
            state["exact_total"] = int(exact_total)
        if exact_l4a_done is not None:
            state["exact_l4a_done"] = int(exact_l4a_done)
            state["exact_done"] = int(exact_l4a_done)
        if exact_l4a_total is not None:
            state["exact_l4a_total"] = int(exact_l4a_total)
            state["exact_total"] = int(exact_l4a_total)
        if exact_l4b_done is not None:
            state["exact_l4b_done"] = int(exact_l4b_done)
        if exact_l4b_total is not None:
            state["exact_l4b_total"] = int(exact_l4b_total)
        if oversampling_factor is not None:
            state["oversampling_factor"] = int(oversampling_factor)

    if path.is_file():
        _update_atomic(path, mutator)
    else:
        payload: dict[str, Any] = {}
        _transition_phase(payload, phase)
        if exact_done is not None:
            payload["exact_done"] = int(exact_done)
        if exact_total is not None:
            payload["exact_total"] = int(exact_total)
        if exact_l4a_done is not None:
            payload["exact_l4a_done"] = int(exact_l4a_done)
            payload["exact_done"] = int(exact_l4a_done)
        if exact_l4a_total is not None:
            payload["exact_l4a_total"] = int(exact_l4a_total)
            payload["exact_total"] = int(exact_l4a_total)
        if exact_l4b_done is not None:
            payload["exact_l4b_done"] = int(exact_l4b_done)
        if exact_l4b_total is not None:
            payload["exact_l4b_total"] = int(exact_l4b_total)
        if oversampling_factor is not None:
            payload["oversampling_factor"] = int(oversampling_factor)
        payload["updated_at"] = _utc_now_iso()
        _write_atomic(path, payload)


def init_exact_l4a_cache(path: Path | str, total_keys: int) -> None:
    """Enter ``exact_l4a`` phase with ``exact_l4a_done=0``."""

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, "exact_l4a")
        state["exact_l4a_done"] = 0
        state["exact_l4a_total"] = int(total_keys)
        state["exact_done"] = 0
        state["exact_total"] = int(total_keys)

    _update_atomic(Path(path), mutator)


def init_exact_l4b_cache(path: Path | str, total_keys: int) -> None:
    """Enter ``exact_l4b`` phase with ``exact_l4b_done=0``."""

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, "exact_l4b")
        state["exact_l4b_done"] = 0
        state["exact_l4b_total"] = int(total_keys)

    _update_atomic(Path(path), mutator)


def init_exact_cache(path: Path | str, total_keys: int) -> None:
    """Backward-compatible alias for :func:`init_exact_l4a_cache`."""
    init_exact_l4a_cache(path, total_keys)


def mark_exact_l4a_done(path: Path | str) -> None:
    """Increment ``exact_l4a_done`` (parent process only; NFS-safe tmp+replace)."""

    def mutator(state: dict[str, Any]) -> None:
        total = int(state.get("exact_l4a_total", state.get("exact_total", 0)))
        done = int(state.get("exact_l4a_done", state.get("exact_done", 0))) + 1
        if total > 0:
            done = min(done, total)
        state["exact_l4a_done"] = done
        state["exact_done"] = done
        state["phase"] = "exact_l4a"

    _update_atomic(Path(path), mutator)


def mark_exact_l4b_done(path: Path | str) -> None:
    """Increment ``exact_l4b_done`` (parent process only; NFS-safe tmp+replace)."""

    def mutator(state: dict[str, Any]) -> None:
        total = int(state.get("exact_l4b_total", 0))
        done = int(state.get("exact_l4b_done", 0)) + 1
        if total > 0:
            done = min(done, total)
        state["exact_l4b_done"] = done
        state["phase"] = "exact_l4b"

    _update_atomic(Path(path), mutator)


def mark_exact_done(path: Path | str) -> None:
    """Backward-compatible alias for :func:`mark_exact_l4a_done`."""
    mark_exact_l4a_done(path)


def read_progress(path: Path | str) -> dict[str, Any] | None:
    """Load sidecar state, or ``None`` if missing/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
