"""
Workspace and manifest path conventions for the config-driven pipeline.

Workspaces live under ``{output_dir}/ws/{label}/``.
``{output_dir}/master/`` (flat, no subdirectories) contains relative symlinks to
every ``ws/<label>/*.fits`` file; basenames already encode the workspace label
(e.g. ``tess…_hp_d.fits``).
The default per-FFI manifest basename is ``syndiff_ffi_frames.csv`` at ``output_dir``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

WORKSPACE_SUBDIR = "ws"
MASTER_SUBDIR = "master"
DEFAULT_MANIFEST_BASENAME = "syndiff_ffi_frames.csv"

# ``np.savez(..., **{BACKGROUND_STACK_NPZ_ARRAY_KEY: stack})`` for rough/smooth stacks
BACKGROUND_STACK_NPZ_ARRAY_KEY = "stack"

# Basename (no extension) for adaptive background stacks under ``ws/<label>/``
ADAPTIVE_BKG_STACK_BASENAME = "bkg_temp_smooth"


def workspace_dir(output_dir: str, label: str) -> str:
    """Absolute path to the workspace directory for a pipeline label."""
    return os.path.join(os.path.abspath(output_dir), WORKSPACE_SUBDIR, label)


def workspace_root(output_dir: str) -> str:
    """Absolute path of the ``ws/`` root under *output_dir*."""
    return os.path.join(os.path.abspath(output_dir), WORKSPACE_SUBDIR)


def master_root(output_dir: str) -> str:
    """Absolute path of the ``master/`` root under *output_dir*."""
    return os.path.join(os.path.abspath(output_dir), MASTER_SUBDIR)


def _relative_link_target(link_path: str, target_path: str) -> str:
    """Compute the relative symlink target for *target_path* from *link_path*."""
    return os.path.relpath(target_path, start=os.path.dirname(link_path))


def link_workspace_fits_master(output_dir: str) -> int:
    """
    Mirror every ``ws/<label>/*.fits`` file as a relative symlink directly under
    ``master/`` of *output_dir* (no ``master/<label>/`` subdirectories).

    Idempotent: existing correct symlinks are left in place; broken or pointing
    elsewhere ones are replaced. If two workspaces would share the same
    basename (unexpected with current naming), the second is skipped with a
    warning. Returns the number of symlinks created or refreshed in this call.
    """
    ws_root = workspace_root(output_dir)
    if not os.path.isdir(ws_root):
        return 0
    m_root = master_root(output_dir)
    os.makedirs(m_root, exist_ok=True)
    refreshed = 0
    for label in sorted(os.listdir(ws_root)):
        ws_label_dir = os.path.join(ws_root, label)
        if not os.path.isdir(ws_label_dir):
            continue
        for entry in sorted(os.listdir(ws_label_dir)):
            if not entry.lower().endswith(".fits"):
                continue
            target = os.path.join(ws_label_dir, entry)
            if not os.path.isfile(target):
                continue
            link = os.path.join(m_root, entry)
            rel_target = _relative_link_target(link, target)
            if os.path.islink(link):
                try:
                    if os.readlink(link) == rel_target:
                        continue
                except OSError:
                    pass
                try:
                    os.unlink(link)
                except OSError as exc:
                    log.warning("master mirror: unlink %s failed: %s", link, exc)
                    continue
            elif os.path.lexists(link):
                try:
                    existing = os.readlink(link) if os.path.islink(link) else None
                except OSError:
                    existing = None
                if existing == rel_target:
                    continue
                log.warning(
                    "master mirror: %s already exists (not a symlink to %s); "
                    "skipping workspace %r — use unique FITS basenames per workspace.",
                    link,
                    rel_target,
                    label,
                )
                continue
            try:
                os.symlink(rel_target, link)
                refreshed += 1
            except OSError as exc:
                log.warning("master mirror: symlink %s -> %s failed: %s", link, rel_target, exc)
    if refreshed:
        log.info("master mirror: refreshed %d symlink(s) under %s", refreshed, m_root)
    return refreshed


def resolve_manifest_path(output_dir: str, manifest_cfg: Optional[str]) -> str:
    """
    Absolute path to the frame manifest CSV.

    Parameters
    ----------
    output_dir : str
        Pipeline output root.
    manifest_cfg : str or None
        If set, a path (absolute or relative to cwd at runtime — callers should
        resolve via config load). If empty/None, use
        ``{output_dir}/DEFAULT_MANIFEST_BASENAME``.
    """
    root = os.path.abspath(output_dir)
    if manifest_cfg and str(manifest_cfg).strip():
        return os.path.abspath(os.path.expanduser(str(manifest_cfg)))
    return os.path.join(root, DEFAULT_MANIFEST_BASENAME)
