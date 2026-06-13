"""Event-level ws directory symlinks (``templates``, ``ffis``)."""

from __future__ import annotations

import os
from pathlib import Path

TEMPLATES_WS_LABEL = "templates"
FFIS_WS_LABEL = "ffis"
_WS_SUBDIR = "ws"
_HOTPANTS_STAMPS_WS_SUFFIX = "_stamps"
_SKIP_WS_CHILDREN = frozenset({"master", "templates", FFIS_WS_LABEL})


def _event_ws_symlink_path(event_dir: str | Path, label: str) -> Path:
    """Absolute path to ``{event_dir}/ws/{label}``."""
    return Path(event_dir).expanduser().resolve() / _WS_SUBDIR / label


def event_templates_symlink_path(event_dir: str | Path) -> Path:
    """Absolute path to ``{event_dir}/ws/templates``."""
    return _event_ws_symlink_path(event_dir, TEMPLATES_WS_LABEL)


def event_ffis_symlink_path(event_dir: str | Path) -> Path:
    """Absolute path to ``{event_dir}/ws/ffis``."""
    return _event_ws_symlink_path(event_dir, FFIS_WS_LABEL)


def template_dir_meta_from_event_dir(event_dir: str | Path) -> dict[str, str] | None:
    """Return manifest audit fields when ``ws/templates`` symlink exists."""
    link_path = event_templates_symlink_path(event_dir)
    if not link_path.is_symlink():
        return None
    physical = link_path.resolve()
    return {
        "template_dir_physical": str(physical),
        "template_dir_symlink": str(link_path),
    }


def ffi_dir_meta_from_event_dir(event_dir: str | Path) -> dict[str, str] | None:
    """Return manifest audit fields when ``ws/ffis`` symlink exists."""
    link_path = event_ffis_symlink_path(event_dir)
    if not link_path.is_symlink():
        return None
    physical = link_path.resolve()
    return {
        "ffi_dir_physical": str(physical),
        "ffi_dir_symlink": str(link_path),
    }


def _ensure_event_ws_symlink(
    event_dir: str | Path,
    physical_dir: str | Path,
    label: str,
    *,
    kind: str,
) -> Path:
    """
    Create or refresh ``{event_dir}/ws/{label}`` → *physical_dir*.

    Uses a relative symlink when possible. Raises if an existing path is a
    symlink pointing elsewhere or a non-symlink file blocks the link.
    """
    event_root = Path(event_dir).expanduser().resolve()
    physical = Path(physical_dir).expanduser().resolve()
    if not physical.is_dir():
        raise FileNotFoundError(
            f"Physical {kind} directory does not exist: {physical}"
        )

    ws_dir = event_root / _WS_SUBDIR
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
) -> Path:
    """Create or refresh ``{event_dir}/ws/templates`` → *physical_template_dir*."""
    return _ensure_event_ws_symlink(
        event_dir,
        physical_template_dir,
        TEMPLATES_WS_LABEL,
        kind="template",
    )


def ensure_event_ffis_symlink(
    event_dir: str | Path,
    physical_ffi_dir: str | Path,
) -> Path:
    """Create or refresh ``{event_dir}/ws/ffis`` → *physical_ffi_dir*."""
    return _ensure_event_ws_symlink(
        event_dir,
        physical_ffi_dir,
        FFIS_WS_LABEL,
        kind="FFI",
    )


def prune_stale_per_workspace_ffis_symlinks(event_dir: str | Path) -> int:
    """
    Remove mistaken ``ws/{label}/ffis`` symlinks from workspace subdirectories.

    Returns the number of symlinks removed.
    """
    ws_root = Path(event_dir).expanduser().resolve() / _WS_SUBDIR
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
