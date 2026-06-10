"""Sidecar JSON progress for downsample parallel batch workers."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "downsample.progress.json"


def progress_path_for_log(log_path: Path | str) -> Path:
    """Resolve sidecar path beside ``per_target/<label>/downsample.log``."""
    return Path(log_path).expanduser().resolve().parent / PROGRESS_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sum_batch_done(batches: dict[str, Any]) -> int:
    total = 0
    for entry in batches.values():
        if isinstance(entry, dict):
            total += int(entry.get("done", 0))
    return total


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


def init_progress(path: Path | str, total_skycells: int, batch_sizes: list[int]) -> None:
    """Create or reset sidecar before parallel batch processing."""
    path = Path(path)
    batches = {
        str(i): {"size": int(size), "done": 0}
        for i, size in enumerate(batch_sizes)
    }
    payload = {
        "total_skycells": int(total_skycells),
        "total_batches": len(batch_sizes),
        "skycells_done": 0,
        "batches": batches,
        "phase": "parallel_batches",
        "updated_at": _utc_now_iso(),
    }
    _write_locked(path, payload)


def set_progress_phase(
    path: Path | str,
    phase: str,
    *,
    total_skycells: int | None = None,
    offsets_done: int | None = None,
    offsets_total: int | None = None,
) -> None:
    """Update lifecycle phase (and optional shift precompute counters)."""
    path = Path(path)

    def mutator(state: dict[str, Any]) -> None:
        state["phase"] = phase
        if total_skycells is not None:
            state["total_skycells"] = int(total_skycells)
            state["skycells_done"] = int(total_skycells)
        if offsets_done is not None:
            state["offsets_done"] = int(offsets_done)
        if offsets_total is not None:
            state["offsets_total"] = int(offsets_total)

    if path.is_file():
        _update_locked(path, mutator)
    else:
        payload: dict[str, Any] = {
            "phase": phase,
            "skycells_done": 0,
            "total_skycells": int(total_skycells or 0),
            "updated_at": _utc_now_iso(),
        }
        if offsets_done is not None:
            payload["offsets_done"] = int(offsets_done)
        if offsets_total is not None:
            payload["offsets_total"] = int(offsets_total)
        _write_locked(path, payload)


def mark_skycell_done(path: Path | str, batch_idx: int) -> None:
    """Increment one skycell for *batch_idx* and recompute ``skycells_done``."""
    key = str(batch_idx)

    def mutator(state: dict[str, Any]) -> None:
        batches = state.setdefault("batches", {})
        entry = batches.setdefault(key, {"size": 0, "done": 0})
        size = int(entry.get("size", 0))
        done = int(entry.get("done", 0)) + 1
        if size > 0:
            done = min(done, size)
        entry["done"] = done
        state["skycells_done"] = _sum_batch_done(batches)
        state["phase"] = "parallel_batches"

    _update_locked(Path(path), mutator)


def read_progress(path: Path | str) -> dict[str, Any] | None:
    """Load sidecar state, or ``None`` if missing/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
