"""Event-level template directory handoff (symlink under ``events/{target}/ws/templates``)."""

from __future__ import annotations

import os
from pathlib import Path

TEMPLATES_WS_LABEL = "templates"


def event_templates_symlink_path(event_dir: str | Path) -> Path:
    """Absolute path to ``{event_dir}/ws/templates``."""
    return Path(event_dir).expanduser().resolve() / "ws" / TEMPLATES_WS_LABEL


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


def ensure_event_templates_symlink(
    event_dir: str | Path,
    physical_template_dir: str | Path,
) -> Path:
    """
    Create or refresh ``{event_dir}/ws/templates`` → *physical_template_dir*.

    Uses a relative symlink when possible. Raises if an existing path is a
    symlink pointing elsewhere or a non-symlink file blocks the link.
    """
    event_root = Path(event_dir).expanduser().resolve()
    physical = Path(physical_template_dir).expanduser().resolve()
    if not physical.is_dir():
        raise FileNotFoundError(
            f"Physical template directory does not exist: {physical}"
        )

    ws_dir = event_root / "ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    link_path = ws_dir / TEMPLATES_WS_LABEL

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
            f"Cannot create templates symlink; path exists and is not a symlink: {link_path}"
        )

    link_path.symlink_to(rel_target)
    return link_path
