"""Host-local verify-in-flight counters for lightweight CLI observability.

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


def _daemon_local_dir(handoff_root: str | Path) -> Path:
    from syndiff_pipeline.template_runner.workspace import state_db_path

    resolved = str(state_db_path(handoff_root))
    key = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "syndiff-daemon" / key


def verify_in_flight_path(handoff_root: str | Path) -> Path:
    return _daemon_local_dir(handoff_root) / "verify_in_flight.json"


def write_verify_in_flight(handoff_root: str | Path, counts: dict[str, int]) -> None:
    """Persist per-run in-flight verify counts (daemon writer)."""
    path = verify_in_flight_path(handoff_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.time(), "by_run": dict(counts)}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def read_verify_in_flight(
    handoff_root: str | Path, run_id: str | None = None
) -> int:
    """Return in-flight verify count for *run_id*, or total if *run_id* is None."""
    path = verify_in_flight_path(handoff_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return 0
    by_run = data.get("by_run")
    if not isinstance(by_run, dict):
        return 0
    if run_id is None:
        return sum(int(v) for v in by_run.values())
    return int(by_run.get(run_id, 0))


def clear_verify_in_flight(handoff_root: str | Path) -> None:
    verify_in_flight_path(handoff_root).unlink(missing_ok=True)
