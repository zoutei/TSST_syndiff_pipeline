"""Workspace paths, deployment recording, and daemon discovery."""

from __future__ import annotations

from pathlib import Path

from syndiff_pipeline.common.orchestration import daemon, logs
from syndiff_pipeline.common.orchestration.deployment import load_workspace_root_from_deployment

DEFAULT_STATE_DB_NAME = "pipeline_state.sqlite"
CONTROL_DIR_NAME = "control"


def normalize_workspace_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve()


def control_root(workspace_root: str | Path) -> Path:
    """Orchestrator state: SQLite, daemon, Discord sidecars."""
    return normalize_workspace_root(workspace_root) / CONTROL_DIR_NAME


def ensure_control_root(workspace_root: str | Path) -> Path:
    root = control_root(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_db_path(workspace_root: str | Path) -> Path:
    """Fixed SQLite path for a workspace (under ``control/``)."""
    return control_root(workspace_root) / DEFAULT_STATE_DB_NAME


def runs_root(workspace_root: str | Path) -> Path:
    return normalize_workspace_root(workspace_root) / "runs"


def record_deployment_path(workspace_root: str | Path, deployment_path: str | Path) -> None:
    ensure_control_root(workspace_root)
    path = logs.workspace_deployment_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(Path(deployment_path).expanduser().resolve()), encoding="utf-8")


def load_recorded_deployment_path(workspace_root: str | Path) -> Path | None:
    record_path = logs.workspace_deployment_path(workspace_root)
    if not record_path.is_file():
        return None
    text = record_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_file() else None


def discover_alive_workspace_roots() -> list[Path]:
    """Return workspace roots with a live supervisor daemon on this host."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []

    roots: list[Path] = []
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
            deploy = parts[idx + 1]
            handoff = str(load_workspace_root_from_deployment(deploy))
        except (ValueError, IndexError, FileNotFoundError, OSError):
            continue
        pid = int(entry.name)
        if not daemon.is_process_alive(pid):
            continue
        roots.append(normalize_workspace_root(handoff))

    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique
