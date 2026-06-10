"""Per-run and per-stage log layout."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

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
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(meta_path)


def target_log_path(cfg_runs_root: str, run_id: str, target_label: str, stage: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "per_target" / target_label / f"{stage}.log"


def stage_status_path(cfg_runs_root: str, run_id: str, target_label: str, stage: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "per_target" / target_label / f"{stage}.status.json"


def stage_manifest_path(cfg_runs_root: str, run_id: str, target_label: str, stage: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "per_target" / target_label / f"{stage}.manifest.json"


def stable_stage_manifest_path(cfg_runs_root: str, target_label: str, stage: str) -> Path:
    """Cross-run completion manifest, keyed only by target + stage (not run_id).

    Completion is a property of the output data, not of any single run, so a
    stable manifest lets a fresh run skip re-scanning a large already-complete
    store. ``verify.manifest_valid`` still guards staleness via the config
    fingerprint and on-disk artifact existence.
    """
    return runs_root(cfg_runs_root) / ".manifests" / target_label / f"{stage}.manifest.json"


def _handoff_dir(handoff_root: str | Path) -> Path:
    from syndiff_pipeline.template_runner.workspace import normalize_handoff_root

    return normalize_handoff_root(handoff_root)


def daemon_lock_path(handoff_root: str | Path) -> Path:
    return _handoff_dir(handoff_root) / "daemon.lock"


def daemon_pid_path(handoff_root: str | Path) -> Path:
    return _handoff_dir(handoff_root) / "daemon.pid"


def daemon_log_path(handoff_root: str | Path) -> Path:
    return _handoff_dir(handoff_root) / "daemon.log"


def daemon_heartbeat_file(handoff_root: str | Path) -> Path:
    """Host-LOCAL heartbeat file (never on NFS).

    The supervisor's liveness signal must not depend on the same NFS/DB volume
    it may be blocked on. We key the file by a hash of the resolved state-db
    path so distinct pipelines on one host do not collide. Liveness is already
    host-local (pid checks use ``os.kill``), so a local temp path is correct.
    """
    from syndiff_pipeline.template_runner.workspace import state_db_path

    resolved = str(state_db_path(handoff_root))
    key = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "syndiff-daemon" / f"{key}.heartbeat"


def discord_bot_lock_path(handoff_root: str | Path) -> Path:
    return _handoff_dir(handoff_root) / "discord_bot.lock"


def discord_bot_pid_path(handoff_root: str | Path) -> Path:
    return _handoff_dir(handoff_root) / "discord_bot.pid"


def discord_bot_log_path(handoff_root: str | Path) -> Path:
    return _handoff_dir(handoff_root) / "discord_bot.log"


def discord_bot_site_config_path(handoff_root: str | Path) -> Path:
    """Persisted site config used to (re)start the Discord bot."""
    return _handoff_dir(handoff_root) / "discord_bot_config.path"


def workspace_deployment_path(handoff_root: str | Path) -> Path:
    """Persisted deployment.yaml path for this workspace."""
    return _handoff_dir(handoff_root) / "workspace_deployment.path"


def summary_csv_path(cfg_runs_root: str, run_id: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "summary.csv"


def summary_json_path(cfg_runs_root: str, run_id: str) -> Path:
    return run_dir(cfg_runs_root, run_id) / "summary.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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

                def isatty(self) -> bool:
                    return False

                def fileno(self) -> int:
                    return sys.stdout.fileno()

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
