"""
Workspace and manifest path conventions for the config-driven pipeline.

Workspaces live under ``{output_dir}/ws/{label}/``.
``{output_dir}/ws/master/`` contains absolute symlinks to every ``ws/<label>/*.fits``
file (flat basenames), plus flat symlinks for each FFI in the target
sector/camera/CCD leaf directory when configured.
Template FITS for differencing are linked at ``{output_dir}/ws/templates`` (see
``template_handoff``).
The default per-FFI manifest basename is ``syndiff_ffi_frames.csv`` at ``output_dir``.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

WORKSPACE_SUBDIR = "ws"
MASTER_SUBDIR = "master"
DEFAULT_MANIFEST_BASENAME = "syndiff_ffi_frames.csv"
HOTPANTS_SUBSTAMP_STARS_BASENAME = "hotpants_substamp_stars.csv"

MASTER_TESS_FFI_LINK = "tess_ffi"
HOTPANTS_STAMPS_WS_SUFFIX = "_stamps"
HOTPANTS_STAMPS_FITS_SUFFIX = "_stamps.fits"

from syndiff_pipeline.common.orchestration.template_handoff import (  # noqa: E402
    TEMPLATES_WS_LABEL,
    event_templates_symlink_path,
    ensure_event_templates_symlink,
)

__all__ = [
    "TEMPLATES_WS_LABEL",
    "event_templates_symlink_path",
    "ensure_event_templates_symlink",
]

# ``np.savez(..., **{BACKGROUND_STACK_NPZ_ARRAY_KEY: stack})`` for rough/smooth stacks
BACKGROUND_STACK_NPZ_ARRAY_KEY = "stack"

# Basename (no extension) for adaptive background stacks under ``ws/<label>/``
ADAPTIVE_BKG_STACK_BASENAME = "bkg_temp_smooth"

# Union mask (2D): pixels where PRF source-hunt excluded sky in any epoch (output_dir root)
BKG_SOURCE_HUNT_UNION_FITS_BASENAME = "bkg_source_hunt_union.fits"

PIPELINE_PLOTS_SUBDIR = "debug_plots"


def pipeline_plots_root(
    output_dir: str, subdir: str | None = PIPELINE_PLOTS_SUBDIR
) -> str:
    """Return ``output_dir`` or ``output_dir / subdir`` for diagnostic figures."""
    root = os.path.abspath(output_dir)
    if subdir is None:
        return root
    s = str(subdir).strip()
    if not s:
        return root
    return os.path.join(root, s)


def workspace_dir(output_dir: str, label: str) -> str:
    """Absolute path to the workspace directory for a pipeline label."""
    return os.path.join(os.path.abspath(output_dir), WORKSPACE_SUBDIR, label)


def workspace_root(output_dir: str) -> str:
    """Absolute path of the ``ws/`` root under *output_dir*."""
    return os.path.join(os.path.abspath(output_dir), WORKSPACE_SUBDIR)


def clear_diff_workspace(event_dir: Union[str, Path]) -> None:
    """Remove ``ws/`` subtree for force rerun; preserve event_dir root artifacts.

    Root files such as ``gaia_catalog_pipeline.csv`` and
    ``cluster_template_job.json`` are left untouched. The ``ws/templates``
    symlink to physical template output is preserved across clears.
    """
    root = Path(event_dir)
    ws = root / WORKSPACE_SUBDIR
    if not ws.is_dir():
        return
    templates_link = event_templates_symlink_path(root)
    templates_target = None
    if templates_link.is_symlink():
        try:
            templates_target = templates_link.resolve()
        except OSError:
            templates_target = None
    shutil.rmtree(ws)
    log.info("Force rerun: removed diff workspace %s", ws)
    if templates_target is not None and templates_target.is_dir():
        ensure_event_templates_symlink(root, templates_target)
        log.info("Force rerun: restored templates symlink -> %s", templates_target)


def master_root(output_dir: str) -> str:
    """Absolute path of ``ws/master/`` under *output_dir*."""
    return os.path.join(workspace_root(output_dir), MASTER_SUBDIR)


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _resolved_symlink_target(link_path: str) -> Optional[str]:
    """Return the absolute path a symlink points to, or None if not a symlink."""
    if not os.path.islink(link_path):
        return None
    try:
        raw = os.readlink(link_path)
    except OSError:
        return None
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    return os.path.abspath(os.path.join(os.path.dirname(link_path), raw))


def _ensure_abs_symlink(link_path: str, target_path: str) -> bool:
    """
    Create or refresh *link_path* as a symlink to the absolute *target_path*.

    Returns True if the symlink was created or replaced in this call.
    """
    abs_target = _abs_path(target_path)
    if os.path.islink(link_path):
        if _resolved_symlink_target(link_path) == abs_target and os.path.exists(link_path):
            return False
        try:
            os.unlink(link_path)
        except OSError as exc:
            log.warning("master workspace: unlink %s failed: %s", link_path, exc)
            return False
    elif os.path.lexists(link_path):
        log.warning(
            "master workspace: %s already exists and is not a symlink to %s; skipping",
            link_path,
            abs_target,
        )
        return False
    try:
        os.symlink(abs_target, link_path)
        return True
    except OSError as exc:
        log.warning(
            "master workspace: symlink %s -> %s failed: %s",
            link_path,
            abs_target,
            exc,
        )
        return False


def _is_hotpants_stamps_workspace_label(label: str) -> bool:
    return label.endswith(HOTPANTS_STAMPS_WS_SUFFIX)


def _is_hotpants_stamps_fits_basename(name: str) -> bool:
    return name.lower().endswith(HOTPANTS_STAMPS_FITS_SUFFIX)


def _prune_master_stamp_symlinks(m_root: str) -> int:
    """Remove stale Hotpants stamp FITS symlinks from ``ws/master/``."""
    removed = 0
    if not os.path.isdir(m_root):
        return removed
    for entry in os.listdir(m_root):
        if not _is_hotpants_stamps_fits_basename(entry):
            continue
        if _remove_legacy_master_link(os.path.join(m_root, entry)):
            removed += 1
    return removed


def _remove_legacy_master_link(link_path: str) -> bool:
    """Drop a stale ``ws/master/`` entry (e.g. legacy ``tess_ffi`` directory link)."""
    if not os.path.lexists(link_path):
        return False
    try:
        os.unlink(link_path)
        return True
    except OSError as exc:
        log.warning("master workspace: unlink %s failed: %s", link_path, exc)
        return False


def link_master_workspace(
    output_dir: str,
    *,
    ffi_leaf: Optional[str] = None,
) -> int:
    """
    Populate ``ws/master/`` with absolute symlinks for Condor / shared-FS access.

    - Every ``ws/<label>/*.fits`` file is mirrored as a flat basename under
      ``ws/master/`` (skips ``master``, ``templates``, and Hotpants ``*_stamps``
      workspaces).
    - When *ffi_leaf* is set and exists, each ``*.fits`` in that sector/camera/CCD
      directory is mirrored as a flat basename under ``ws/master/``.

    Idempotent: correct symlinks are left in place; broken or stale ones are
    replaced. Returns the number of symlinks created or refreshed in this call.
    """
    ws_root = workspace_root(output_dir)
    if not os.path.isdir(ws_root):
        return 0
    m_root = master_root(output_dir)
    os.makedirs(m_root, exist_ok=True)
    refreshed = _prune_master_stamp_symlinks(m_root)

    for label in sorted(os.listdir(ws_root)):
        if label in (MASTER_SUBDIR, TEMPLATES_WS_LABEL):
            continue
        if _is_hotpants_stamps_workspace_label(label):
            continue
        ws_label_dir = os.path.join(ws_root, label)
        if not os.path.isdir(ws_label_dir):
            continue
        for entry in sorted(os.listdir(ws_label_dir)):
            if not entry.lower().endswith(".fits"):
                continue
            if _is_hotpants_stamps_fits_basename(entry):
                continue
            target = os.path.join(ws_label_dir, entry)
            if not os.path.isfile(target):
                continue
            link = os.path.join(m_root, entry)
            if _ensure_abs_symlink(link, target):
                refreshed += 1

    if ffi_leaf and str(ffi_leaf).strip():
        ffi_leaf_abs = _abs_path(str(ffi_leaf))
        legacy_link = os.path.join(m_root, MASTER_TESS_FFI_LINK)
        if _remove_legacy_master_link(legacy_link):
            refreshed += 1
        if os.path.isdir(ffi_leaf_abs):
            for entry in sorted(os.listdir(ffi_leaf_abs)):
                if not entry.lower().endswith(".fits"):
                    continue
                target = os.path.join(ffi_leaf_abs, entry)
                if not os.path.isfile(target):
                    continue
                link = os.path.join(m_root, entry)
                if _ensure_abs_symlink(link, target):
                    refreshed += 1
        else:
            log.debug("master workspace: skip FFI leaf — not a directory: %s", ffi_leaf_abs)

    if refreshed:
        log.info("master workspace: refreshed %d symlink(s) under %s", refreshed, m_root)
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
