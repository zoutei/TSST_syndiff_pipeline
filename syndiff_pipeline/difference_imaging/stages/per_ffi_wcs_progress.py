"""Progress sidecar for per_ffi_wcs stage."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_FILENAME = "wcs.progress.json"
CLI_PROGRESS_FILENAME = "diff.wcs.progress.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_path_for_output_workspace(output_dir: Path | str) -> Path:
    return Path(output_dir).expanduser().resolve() / PROGRESS_FILENAME


def progress_path_for_diff_log(log_path: Path | str) -> Path:
    return Path(log_path).expanduser().resolve().parent / CLI_PROGRESS_FILENAME


def init_progress(
    path: Path | str,
    *,
    wcs_label: str,
    centroids_input: str,
    n_frames: int,
    diff_log_path: str | None = None,
) -> None:
    payload = {
        "wcs_label": wcs_label,
        "centroids_input": centroids_input,
        "n_frames": int(n_frames),
        "n_done": 0,
        "updated_at": _utc_now_iso(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if diff_log_path:
        cli = progress_path_for_diff_log(diff_log_path)
        cli.parent.mkdir(parents=True, exist_ok=True)
        cli.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def record_frame_done(path: Path | str, *, stem: str, ok: bool) -> None:
    p = Path(path)
    if not p.is_file():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    data["n_done"] = int(data.get("n_done", 0)) + 1
    data["last_stem"] = stem
    data["last_ok"] = bool(ok)
    data["updated_at"] = _utc_now_iso()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
