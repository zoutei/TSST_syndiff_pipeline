"""Per-run and per-stage log layout."""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from syndiff_pipeline.template_runner.runner_config import (
    load_and_materialize_runner_config,
    write_runner_config,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_root(cfg_runs_root: str) -> Path:
    return Path(cfg_runs_root).expanduser().resolve()


def run_dir(cfg_runs_root: str, run_id: str) -> Path:
    return runs_root(cfg_runs_root) / run_id


def run_config_path(run_directory: str | Path) -> Path:
    return Path(run_directory).expanduser().resolve() / "config.yaml"


def run_targets_path(run_directory: str | Path) -> Path:
    return Path(run_directory).expanduser().resolve() / "targets.csv"


def run_meta_path(run_directory: str | Path) -> Path:
    return Path(run_directory).expanduser().resolve() / "run_meta.json"


def materialize_run_inputs(
    source_config: str | Path,
    source_targets: str | Path,
    run_directory: str | Path,
) -> Tuple[str, str]:
    """Copy config and targets into *run_directory*; return absolute run-local paths."""
    rd = Path(run_directory).expanduser().resolve()
    rd.mkdir(parents=True, exist_ok=True)
    cfg_path = run_config_path(rd)
    targets_path = run_targets_path(rd)

    if not cfg_path.is_file():
        cfg = load_and_materialize_runner_config(source_config)
        write_runner_config(cfg, cfg_path)
    if not targets_path.is_file():
        src_targets = Path(source_targets).expanduser().resolve()
        shutil.copy2(src_targets, targets_path)

    return str(cfg_path), str(targets_path)


def ensure_run_layout(cfg_runs_root: str, run_id: str, meta: dict) -> Path:
    rd = run_dir(cfg_runs_root, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "per_target").mkdir(exist_ok=True)
    meta_path = rd / "run_meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    latest = runs_root(cfg_runs_root) / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_id)
    except OSError:
        pass
    return rd


def update_run_meta(cfg_runs_root: str, run_id: str, patch: dict) -> None:
    rd = run_dir(cfg_runs_root, run_id)
    meta_path = rd / "run_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(patch)
    meta["updated_at"] = _utc_now_iso()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def target_log_path(cfg_runs_root: str, run_id: str, target_label: str, stage: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "per_target" / target_label / f"{stage}.log"


def scheduler_log_path(cfg_runs_root: str, run_id: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "scheduler.log"


def scheduler_pid_path(cfg_runs_root: str, run_id: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "scheduler.pid"


def summary_csv_path(cfg_runs_root: str, run_id: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "summary.csv"


def summary_json_path(cfg_runs_root: str, run_id: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "summary.json"


def _format_header(stage: str, snapshot: Dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        f"STAGE: {stage}",
        f"Started: {_utc_now_iso()}",
    ]
    for k, v in snapshot.items():
        lines.append(f"  {k}: {v}")
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def _format_footer(duration_s: float, exit_code: int, error_tail: str = "") -> str:
    lines = [
        "-" * 72,
        f"Finished: {_utc_now_iso()}",
        f"Duration: {duration_s:.2f}s",
        f"Exit code: {exit_code}",
    ]
    if error_tail:
        lines.append("Error tail:")
        lines.append(error_tail.rstrip())
    lines.append("-" * 72)
    return "\n".join(lines) + "\n"


@contextmanager
def stage_log(cfg_runs_root: str, run_id: str, target_label: str, stage: str, snapshot: dict):
    log_path = target_log_path(cfg_runs_root, run_id, target_label, stage)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    header = _format_header(stage, snapshot)
    chunks: list[str] = [header]
    exit_code = 0
    error_tail = ""
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            fh.flush()

            class Tee:
                def write(self, s: str) -> int:
                    chunks.append(s)
                    fh.write(s)
                    fh.flush()
                    return len(s)

                def flush(self) -> None:
                    fh.flush()

            yield Tee()
    except Exception as exc:
        exit_code = 1
        error_tail = str(exc)
        raise
    finally:
        duration = time.monotonic() - t0
        body = "".join(chunks)
        tail_lines = body.strip().splitlines()[-20:]
        if not error_tail and exit_code != 0:
            error_tail = "\n".join(tail_lines)
        footer = _format_footer(duration, exit_code, error_tail if exit_code else "")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(footer)


def read_log_tail(path: str | Path, n_lines: int = 40) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])
