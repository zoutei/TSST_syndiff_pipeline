"""Sidecar JSON progress for field-mode downsample (L5 composite-key batches).

Counters are updated from the **parent** process as Parallel results are
drained. Updates use atomic write via a temporary file so they remain reliable
on NFS mounts without working ``flock`` (same pattern as remap / Hotpants).

Uses the same ``downsample.progress.json`` filename as linear mode so
orchestration path resolution stays unchanged.
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
    state = read_progress(path) or {}
    mutator(state)
    state["updated_at"] = _utc_now_iso()
    _write_atomic(path, state)


def read_progress(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def init_field_setup_progress(path: Path | str) -> None:
    """Create sidecar immediately at L5 start (before schedule / staging work)."""
    path = Path(path)
    payload: dict[str, Any] = {
        "geometry_mode": "field",
        "total_skycells": 0,
        "skycells_done": 0,
        "composite_keys_total": 0,
        "composite_keys_done": 0,
        "contrib_keys_total": 0,
        "contrib_writes": 0,
        "contrib_skips": 0,
        "oversampling_factor": 1,
    }
    _transition_phase(payload, "setup")
    payload["updated_at"] = _utc_now_iso()
    _write_atomic(path, payload)


def init_field_progress(
    path: Path | str,
    *,
    n_skycells: int,
    n_composite_keys: int,
    n_contrib_keys: int,
    oversampling_factor: int = 1,
) -> None:
    """Create or reset sidecar before field L5 skycell-batch processing."""
    path = Path(path)

    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, "field_composite_keys")
        state["geometry_mode"] = "field"
        state["total_skycells"] = int(n_skycells)
        state["skycells_done"] = 0
        state["composite_keys_total"] = int(n_composite_keys)
        state["composite_keys_done"] = 0
        state["contrib_keys_total"] = int(n_contrib_keys)
        state["contrib_writes"] = 0
        state["contrib_skips"] = 0
        state["oversampling_factor"] = int(oversampling_factor)

    if path.is_file():
        _update_atomic(path, mutator)
    else:
        payload: dict[str, Any] = {
            "geometry_mode": "field",
            "total_skycells": int(n_skycells),
            "skycells_done": 0,
            "composite_keys_total": int(n_composite_keys),
            "composite_keys_done": 0,
            "contrib_keys_total": int(n_contrib_keys),
            "contrib_writes": 0,
            "contrib_skips": 0,
            "oversampling_factor": int(oversampling_factor),
        }
        _transition_phase(payload, "field_composite_keys")
        payload["updated_at"] = _utc_now_iso()
        _write_atomic(path, payload)


def mark_skycell_batch_done(
    path: Path | str,
    *,
    n_composite_keys: int = 0,
    n_writes: int = 0,
    n_skips: int = 0,
) -> None:
    """Increment counters after one skycell batch completes (parent only)."""

    def mutator(state: dict[str, Any]) -> None:
        state["skycells_done"] = int(state.get("skycells_done", 0)) + 1
        state["composite_keys_done"] = int(state.get("composite_keys_done", 0)) + int(
            n_composite_keys
        )
        state["contrib_writes"] = int(state.get("contrib_writes", 0)) + int(n_writes)
        state["contrib_skips"] = int(state.get("contrib_skips", 0)) + int(n_skips)

    _update_atomic(Path(path), mutator)


def set_perf_metadata(path: Path | str, **kwargs: Any) -> None:
    """Attach timing / IO counters for L5 benchmarks."""

    def mutator(state: dict[str, Any]) -> None:
        perf = state.get("perf")
        if not isinstance(perf, dict):
            perf = {}
        perf.update(kwargs)
        state["perf"] = perf

    _update_atomic(Path(path), mutator)


def set_progress_phase(path: Path | str, phase: str) -> None:
    def mutator(state: dict[str, Any]) -> None:
        _transition_phase(state, phase)

    _update_atomic(Path(path), mutator)
