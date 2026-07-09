"""Detect whether a stage is still producing output (log/sidecar activity)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

# Stages updating logs/sidecars at least this often are treated as alive even
# when Condor polling briefly misses the job (e.g. after daemon restart).
STAGE_OUTPUT_ACTIVE_S = 600.0

CONDOR_POLL_MISS_FAIL_THRESHOLD = 3


def _path_mtime_age_s(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _iso_age_s(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return time.time() - ts.timestamp()


def _diff_sidecar_recently_active(log_path: Path, *, max_age_s: float) -> bool:
    from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
        progress_path_for_diff_log,
        read_progress_merged,
    )
    from syndiff_pipeline.difference_imaging.stages.hotpants_progress import (
        progress_path_for_diff_log as hotpants_sidecar_path,
        read_progress as read_hotpants_progress,
    )
    from syndiff_pipeline.difference_imaging.stages.photometry_progress import (
        progress_path_for_diff_log as photometry_sidecar_path,
        read_progress as read_photometry_progress,
    )

    for sidecar_path, read_fn in (
        (progress_path_for_diff_log(log_path), read_progress_merged),
        (hotpants_sidecar_path(log_path), read_hotpants_progress),
        (photometry_sidecar_path(log_path), read_photometry_progress),
    ):
        data = read_fn(sidecar_path)
        if not data:
            continue
        age = _iso_age_s(str(data.get("updated_at") or ""))
        if age is not None and age <= max_age_s:
            phase = str(data.get("phase", "running"))
            if phase != "complete":
                return True
    return False


def stage_output_recently_active(
    log_path: Path | str,
    stage: str,
    *,
    max_age_s: float = STAGE_OUTPUT_ACTIVE_S,
) -> bool:
    """Return True when stage logs or known sidecars were updated within *max_age_s*."""
    path = Path(log_path)
    age = _path_mtime_age_s(path)
    if age is not None and age <= max_age_s:
        return True
    if stage == "diff" and path.name != "diff.log":
        path = path.parent / "diff.log"
    if stage == "diff":
        return _diff_sidecar_recently_active(path, max_age_s=max_age_s)
    return False


def read_poll_miss_count(path: Path | str) -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        text = p.read_text(encoding="utf-8").strip()
        return max(0, int(text or "0"))
    except (OSError, ValueError):
        return 0


def record_poll_miss(path: Path | str) -> int:
    p = Path(path)
    count = read_poll_miss_count(p) + 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{count}\n", encoding="utf-8")
    return count


def clear_poll_misses(path: Path | str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
