"""Event-level workspace tree symlinks (``templates``, ``ffis``)."""

from __future__ import annotations

import os
from pathlib import Path

TEMPLATES_WS_LABEL = "templates"
FFIS_WS_LABEL = "ffis"
_WS_SUBDIR = "ws"
_HOTPANTS_STAMPS_WS_SUFFIX = "_stamps"
_SKIP_WS_CHILDREN = frozenset({"master", "templates", FFIS_WS_LABEL})


def _normalize_workspace_run_id(run_id: str | None) -> str | None:
    if run_id is None:
        return None
    s = str(run_id).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def workspace_tree_name(run_id: str | None = None) -> str:
    """Filesystem name for the workspace tree: ``ws`` or ``ws_{run_id}``."""
    rid = _normalize_workspace_run_id(run_id)
    return f"{_WS_SUBDIR}_{rid}" if rid else _WS_SUBDIR


def workspace_tree_path(event_dir: str | Path, *, run_id: str | None = None) -> Path:
    """Absolute path to ``{event_dir}/ws`` or ``{event_dir}/ws_{run_id}``."""
    return Path(event_dir).expanduser().resolve() / workspace_tree_name(run_id)


def _event_ws_symlink_path(
    event_dir: str | Path,
    label: str,
    *,
    run_id: str | None = None,
) -> Path:
    """Absolute path to ``{event_dir}/ws[_{run_id}]/{label}``."""
    return workspace_tree_path(event_dir, run_id=run_id) / label


def event_templates_symlink_path(
    event_dir: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Absolute path to ``{event_dir}/ws[_{run_id}]/templates``."""
    return _event_ws_symlink_path(event_dir, TEMPLATES_WS_LABEL, run_id=run_id)


def event_ffis_symlink_path(
    event_dir: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Absolute path to ``{event_dir}/ws[_{run_id}]/ffis``."""
    return _event_ws_symlink_path(event_dir, FFIS_WS_LABEL, run_id=run_id)


def template_dir_meta_from_event_dir(
    event_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, str] | None:
    """Return manifest audit fields when ``ws/templates`` symlink exists."""
    link_path = event_templates_symlink_path(event_dir, run_id=run_id)
    if not link_path.is_symlink():
        return None
    physical = link_path.resolve()
    return {
        "template_dir_physical": str(physical),
        "template_dir_symlink": str(link_path),
    }


def ffi_dir_meta_from_event_dir(
    event_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, str] | None:
    """Return manifest audit fields when ``ws/ffis`` symlink exists."""
    link_path = event_ffis_symlink_path(event_dir, run_id=run_id)
    if not link_path.is_symlink():
        return None
    physical = link_path.resolve()
    return {
        "ffi_dir_physical": str(physical),
        "ffi_dir_symlink": str(link_path),
    }


def _ensure_workspace_tree_symlink(
    event_dir: str | Path,
    label: str,
    physical_dir: str | Path,
    *,
    kind: str,
    run_id: str | None = None,
) -> Path:
    """
    Create or refresh ``{event_dir}/ws[_{run_id}]/{label}`` → *physical_dir*.

    Uses a relative symlink when possible. Raises if an existing path is a
    symlink pointing elsewhere or a non-symlink file blocks the link.
    """
    event_root = Path(event_dir).expanduser().resolve()
    physical = Path(physical_dir).expanduser().resolve()
    if not physical.is_dir():
        raise FileNotFoundError(
            f"Physical {kind} directory does not exist: {physical}"
        )

    ws_dir = workspace_tree_path(event_root, run_id=run_id)
    ws_dir.mkdir(parents=True, exist_ok=True)
    link_path = ws_dir / label

    try:
        rel_target = os.path.relpath(str(physical), start=str(ws_dir))
    except ValueError:
        rel_target = str(physical)

    if link_path.is_symlink():
        current = link_path.resolve()
        if current == physical:
            return link_path
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(
            f"Cannot create {kind} symlink; path exists and is not a symlink: {link_path}"
        )

    link_path.symlink_to(rel_target)
    return link_path


def ensure_event_templates_symlink(
    event_dir: str | Path,
    physical_template_dir: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Create or refresh templates symlink under the active workspace tree."""
    return _ensure_workspace_tree_symlink(
        event_dir,
        TEMPLATES_WS_LABEL,
        physical_template_dir,
        kind="template",
        run_id=run_id,
    )


def ensure_event_ffis_symlink(
    event_dir: str | Path,
    physical_ffi_dir: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Create or refresh ffis symlink under the active workspace tree."""
    return _ensure_workspace_tree_symlink(
        event_dir,
        FFIS_WS_LABEL,
        physical_ffi_dir,
        kind="FFI",
        run_id=run_id,
    )


def prune_stale_per_workspace_ffis_symlinks(
    event_dir: str | Path,
    *,
    run_id: str | None = None,
) -> int:
    """
    Remove mistaken ``ws/{label}/ffis`` symlinks from workspace subdirectories.

    Returns the number of symlinks removed.
    """
    ws_root = workspace_tree_path(event_dir, run_id=run_id)
    if not ws_root.is_dir():
        return 0

    removed = 0
    for child in ws_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in _SKIP_WS_CHILDREN:
            continue
        if child.name.endswith(_HOTPANTS_STAMPS_WS_SUFFIX):
            continue
        stale = child / FFIS_WS_LABEL
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
            removed += 1
    return removed
