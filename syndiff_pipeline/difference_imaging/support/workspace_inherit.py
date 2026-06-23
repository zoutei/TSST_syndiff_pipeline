"""Bootstrap workspace trees with symlinks to upstream run artifacts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from syndiff_pipeline.difference_imaging.support.paths import workspace_tree_name

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceInheritSpec:
    from_run_id: str
    labels: tuple[str, ...] = ()
    root_artifacts: tuple[str, ...] = ()


def _relative_symlink_target(parent_tree: str, name: str) -> str:
    return os.path.join("..", parent_tree, name)


def _resolved_link_target(link_path: Path) -> Path | None:
    if not link_path.is_symlink():
        return None
    try:
        target = os.readlink(link_path)
    except OSError:
        return None
    if os.path.isabs(target):
        return Path(target).resolve()
    return (link_path.parent / target).resolve()


def _ensure_relative_symlink(link_path: Path, rel_target: str, expected_abs: Path) -> None:
    if link_path.is_symlink():
        resolved = _resolved_link_target(link_path)
        if resolved == expected_abs.resolve():
            return
        raise RuntimeError(
            f"Workspace inherit: {link_path} exists but points to {resolved!r}, "
            f"expected {expected_abs!r}"
        )
    if link_path.exists():
        raise RuntimeError(
            f"Workspace inherit: {link_path} exists and is not a symlink to "
            f"{expected_abs!r}"
        )
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(rel_target, link_path)
    log.info("Workspace inherit: %s -> %s", link_path, rel_target)


def bootstrap_workspace_inherit(
    event_dir: str | Path,
    *,
    run_id: str | None,
    spec: WorkspaceInheritSpec,
) -> None:
    """
    Create symlinks under ``ws_{run_id}/`` into ``ws_{from_run_id}/``.

    Never modifies the parent workspace tree.
    """
    event_root = Path(event_dir).expanduser().resolve()
    parent_tree = workspace_tree_name(spec.from_run_id)
    child_tree = workspace_tree_name(run_id)
    parent_ws = event_root / parent_tree
    child_ws = event_root / child_tree

    if not parent_ws.is_dir():
        raise FileNotFoundError(
            f"Workspace inherit source missing: {parent_ws} (from={spec.from_run_id!r})"
        )

    child_ws.mkdir(parents=True, exist_ok=True)

    for label in spec.labels:
        src = parent_ws / label
        if not src.exists():
            raise FileNotFoundError(
                f"Workspace inherit: parent label missing: {src}"
            )
        link = child_ws / label
        rel = _relative_symlink_target(parent_tree, label)
        _ensure_relative_symlink(link, rel, src)

    for basename in spec.root_artifacts:
        src = parent_ws / basename
        if not src.is_file():
            raise FileNotFoundError(
                f"Workspace inherit: parent artifact missing: {src}"
            )
        link = child_ws / basename
        rel = _relative_symlink_target(parent_tree, basename)
        _ensure_relative_symlink(link, rel, src)
