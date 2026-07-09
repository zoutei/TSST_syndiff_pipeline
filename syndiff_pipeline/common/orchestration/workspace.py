"""Workspace paths, deployment recording, and daemon discovery."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.deployment import load_workspace_root_from_deployment

DEFAULT_STATE_DB_NAME = "pipeline_state.sqlite"
CONTROL_DIR_NAME = "control"
DEFAULT_DEPLOYMENT_CANDIDATES = (
    "config/deployment.yaml",
    "./config/deployment.yaml",
)


def normalize_workspace_root(workspace_root: str | Path) -> Path:
    """Normalize workspace root.
    
    Parameters
    ----------
    workspace_root : str | Path
    
    Returns
    -------
    Path"""
    return Path(workspace_root).expanduser().resolve()


def control_root(workspace_root: str | Path) -> Path:
    """Orchestrator state: SQLite, daemon, Discord sidecars."""
    return normalize_workspace_root(workspace_root) / CONTROL_DIR_NAME


def ensure_control_root(workspace_root: str | Path) -> Path:
    """Ensure control root.
    
    Parameters
    ----------
    workspace_root : str | Path
    
    Returns
    -------
    Path"""
    root = control_root(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_db_path(workspace_root: str | Path) -> Path:
    """Fixed SQLite path for a workspace (under ``control/``)."""
    return control_root(workspace_root) / DEFAULT_STATE_DB_NAME


def runs_root(workspace_root: str | Path) -> Path:
    """Runs root.
    
    Parameters
    ----------
    workspace_root : str | Path
    
    Returns
    -------
    Path"""
    return normalize_workspace_root(workspace_root) / "runs"


def record_deployment_path(workspace_root: str | Path, deployment_path: str | Path) -> None:
    """Record deployment path.
    
    Parameters
    ----------
    workspace_root : str | Path
    deployment_path : str | Path"""
    ensure_control_root(workspace_root)
    path = logs.workspace_deployment_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(Path(deployment_path).expanduser().resolve()), encoding="utf-8")


def handoff_cache_path() -> Path:
    """User-local cache of the last resolved workspace + deployment."""
    override = os.environ.get("SYNDIFF_HANDOFF_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "syndiff" / "handoff_cache.json"


def record_handoff_cache(workspace_root: str | Path, deployment_path: str | Path) -> None:
    """Persist workspace/deployment for fast bare CLI resolution."""
    cache_path = handoff_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workspace_root": str(normalize_workspace_root(workspace_root)),
        "deployment_path": str(Path(deployment_path).expanduser().resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_handoff_cache() -> dict | None:
    """Load cached handoff metadata, or ``None`` if missing/unreadable."""
    cache_path = handoff_cache_path()
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not data.get("workspace_root"):
        return None
    return data


def deployment_candidates() -> list[Path]:
    """Deployment paths to try before scanning ``/proc``."""
    candidates: list[Path] = []
    env_deploy = os.environ.get("SYNDIFF_DEPLOYMENT")
    if env_deploy:
        candidates.append(Path(env_deploy).expanduser())
    for rel in DEFAULT_DEPLOYMENT_CANDIDATES:
        path = Path(rel).expanduser()
        if path.is_file():
            candidates.append(path.resolve())
    cached = load_handoff_cache()
    if cached:
        dep = cached.get("deployment_path")
        if dep:
            path = Path(dep).expanduser()
            if path.is_file():
                candidates.append(path.resolve())
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def resolve_handoff_fast(*, require_daemon: bool = True) -> str | None:
    """Resolve workspace root via env/default/cache before ``/proc`` discovery."""
    from syndiff_pipeline.common.orchestration.scheduler_control import daemon_is_alive

    for deploy in deployment_candidates():
        try:
            handoff = load_workspace_root_from_deployment(deploy)
        except (FileNotFoundError, OSError, ValueError):
            continue
        handoff_s = str(normalize_workspace_root(handoff))
        if require_daemon and not daemon_is_alive(handoff_s):
            continue
        record_deployment_path(handoff_s, deploy)
        record_handoff_cache(handoff_s, deploy)
        return handoff_s

    cached = load_handoff_cache()
    if not cached:
        return None
    handoff = cached.get("workspace_root")
    if not handoff:
        return None
    handoff_s = str(normalize_workspace_root(handoff))
    if require_daemon and not daemon_is_alive(handoff_s):
        return None
    dep = cached.get("deployment_path")
    if dep:
        dep_path = Path(dep).expanduser()
        if dep_path.is_file():
            record_deployment_path(handoff_s, dep_path)
    return handoff_s


def load_recorded_deployment_path(workspace_root: str | Path) -> Path | None:
    """Load recorded deployment path.
    
    Parameters
    ----------
    workspace_root : str | Path
    
    Returns
    -------
    Path | None"""
    record_path = logs.workspace_deployment_path(workspace_root)
    if not record_path.is_file():
        return None
    text = record_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_file() else None


def discover_alive_workspace_handoffs() -> list[tuple[Path, Path]]:
    """Return ``(workspace_root, deployment_path)`` for live supervisors on this host."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []

    handoffs: list[tuple[Path, Path]] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        parts = [p.decode("utf-8", errors="replace") for p in raw.split(b"\0") if p]
        if not parts:
            continue
        if "common.orchestration.scheduler" not in " ".join(parts):
            continue
        if "--daemon" not in parts:
            continue
        try:
            idx = parts.index("--deployment")
            deploy = Path(parts[idx + 1]).expanduser().resolve()
            handoff = normalize_workspace_root(load_workspace_root_from_deployment(deploy))
        except (ValueError, IndexError, FileNotFoundError, OSError):
            continue
        pid = int(entry.name)
        if not daemon.is_process_alive(pid):
            continue
        handoffs.append((handoff, deploy))

    seen: set[str] = set()
    unique: list[tuple[Path, Path]] = []
    for root, deploy in handoffs:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append((root, deploy))
    return unique


def discover_alive_workspace_roots() -> list[Path]:
    """Return workspace roots with a live supervisor daemon on this host."""
    return [root for root, _ in discover_alive_workspace_handoffs()]
