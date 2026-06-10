"""Host-local artifact-scan counters for lightweight CLI observability.

The supervisor daemon updates these counts each tick. CLI tools read them
without importing the heavy ``verify`` / ``verify_worker`` stack (which pulls
in numpy and template stage modules).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any


def _daemon_local_dir(handoff_root: str | Path) -> Path:
    from syndiff_pipeline.template_creation.orchestration.workspace import state_db_path

    resolved = str(state_db_path(handoff_root))
    key = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "syndiff-daemon" / key


def verify_in_flight_path(handoff_root: str | Path) -> Path:
    return _daemon_local_dir(handoff_root) / "verify_in_flight.json"


def _normalize_run_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        active = entry.get("active")
        if not isinstance(active, list):
            active = []
        scan_running = entry.get("scan_running", entry.get("in_flight", 0))
        scan_queued = entry.get("scan_queued", entry.get("pending", 0))
        return {
            "scan_queued": int(scan_queued),
            "scan_running": int(scan_running),
            "active": active,
        }
    return {"scan_queued": 0, "scan_running": int(entry or 0), "active": []}


def _load_payload(handoff_root: str | Path) -> dict[str, Any]:
    path = verify_in_flight_path(handoff_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return {"by_run": {}}
    by_run = data.get("by_run")
    if not isinstance(by_run, dict):
        return {"by_run": {}}
    return {"by_run": by_run}


def write_verify_in_flight(
    handoff_root: str | Path, by_run: dict[str, int | dict[str, Any]]
) -> None:
    """Persist per-run artifact-scan status (daemon writer).

    Values may be legacy ints (running count only) or dicts with
    ``scan_queued``, ``scan_running``, and ``active`` keys.
    """
    path = verify_in_flight_path(handoff_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        run_id: _normalize_run_entry(entry) for run_id, entry in by_run.items()
    }
    payload = {"updated_at": time.time(), "by_run": normalized}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def read_verify_run_status(
    handoff_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Return artifact-scan observability for one run."""
    by_run = _load_payload(handoff_root).get("by_run", {})
    return _normalize_run_entry(by_run.get(run_id, 0))


def read_verify_in_flight(
    handoff_root: str | Path, run_id: str | None = None
) -> int:
    """Return running artifact-scan count for *run_id*, or total if *run_id* is None."""
    by_run = _load_payload(handoff_root).get("by_run", {})
    if run_id is None:
        return sum(_normalize_run_entry(v)["scan_running"] for v in by_run.values())
    return _normalize_run_entry(by_run.get(run_id, 0))["scan_running"]


def read_verify_pending(handoff_root: str | Path, run_id: str) -> int:
    """Return queued artifact-scan count for *run_id*."""
    return read_verify_run_status(handoff_root, run_id)["scan_queued"]


def read_scan_queued(handoff_root: str | Path, run_id: str) -> int:
    return read_verify_pending(handoff_root, run_id)


def read_scan_running(handoff_root: str | Path, run_id: str) -> int:
    return read_verify_in_flight(handoff_root, run_id)


def read_verify_active_keys(
    handoff_root: str | Path, run_id: str
) -> list[tuple[str, str]]:
    """Return (target_label, stage) pairs currently scanning in the daemon."""
    active = read_verify_run_status(handoff_root, run_id).get("active", [])
    out: list[tuple[str, str]] = []
    for item in active:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
    return out


def clear_verify_in_flight(handoff_root: str | Path) -> None:
    verify_in_flight_path(handoff_root).unlink(missing_ok=True)
