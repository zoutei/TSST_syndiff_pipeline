"""
Workspace and manifest path conventions for the config-driven pipeline.

Workspaces live under ``{output_dir}/ws/{label}/``.
``{output_dir}/ws/master/`` contains absolute symlinks to every ``ws/<label>/*.fits``
file (flat basenames), plus flat symlinks for each FFI in the target
sector/camera/CCD leaf directory when configured.
``{output_dir}/ws/ffis`` symlink points at that same FFI leaf directory.
Template FITS for differencing are linked at ``{output_dir}/ws/templates`` (see
``event_ws_symlinks``).
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
TARGETS_DS9_REGION_BASENAME = "targets.reg"
SHARED_MASK_FITS_BASENAME = "shared_mask.fits"
GAIA_CATALOG_PIPELINE_BASENAME = "gaia_catalog_pipeline.csv"
DIFF_CONFIG_SNAPSHOT_BASENAME = "diff_config.yaml"

WORKSPACE_ROOT_ARTIFACTS = (
    SHARED_MASK_FITS_BASENAME,
    HOTPANTS_SUBSTAMP_STARS_BASENAME,
    GAIA_CATALOG_PIPELINE_BASENAME,
    TARGETS_DS9_REGION_BASENAME,
    DIFF_CONFIG_SNAPSHOT_BASENAME,
)

MASTER_TESS_FFI_LINK = "tess_ffi"
HOTPANTS_STAMPS_WS_SUFFIX = "_stamps"
HOTPANTS_STAMPS_FITS_SUFFIX = "_stamps.fits"

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (  # noqa: E402
    FFIS_WS_LABEL,
    TEMPLATES_WS_LABEL,
    ensure_event_ffis_symlink,
    ensure_event_templates_symlink,
    event_ffis_symlink_path,
    event_templates_symlink_path,
    prune_stale_per_workspace_ffis_symlinks,
)

__all__ = [
    "FFIS_WS_LABEL",
    "TEMPLATES_WS_LABEL",
    "ensure_event_ffis_symlink",
    "ensure_event_templates_symlink",
    "event_ffis_symlink_path",
    "event_templates_symlink_path",
    "prune_stale_per_workspace_ffis_symlinks",
]

# ``np.savez(..., **{BACKGROUND_STACK_NPZ_ARRAY_KEY: stack})`` for rough/smooth stacks
BACKGROUND_STACK_NPZ_ARRAY_KEY = "stack"

# Basename (no extension) for adaptive background stacks under ``ws/<label>/``
ADAPTIVE_BKG_STACK_BASENAME = "bkg_temp_smooth"

# Union mask (2D): pixels where PRF source-hunt excluded sky in any epoch (output_dir root)
BKG_SOURCE_HUNT_UNION_FITS_BASENAME = "bkg_source_hunt_union.fits"

PIPELINE_PLOTS_SUBDIR = "debug_plots"
KERNEL_RECONSTRUCTION_NPZ_BASENAME = "kernel_reconstruction.npz"
PHOT_CALIB_CSV_BASENAME = "phot_calib.csv"


def meta_workspace_label(diffs_label: str) -> str:
    """Meta workspace paired with a diffs label (``hp_d`` → ``hp_m``)."""
    label = str(diffs_label).strip()
    if label.endswith("_d"):
        return label[:-2] + "_m"
    return f"{label}_m"


def meta_workspace_dir_from_diffs_dir(diffs_dir: str) -> str:
    """Absolute path to meta workspace sibling of a diffs workspace directory."""
    d = os.path.abspath(diffs_dir)
    return os.path.join(os.path.dirname(d), meta_workspace_label(os.path.basename(d)))


def normalize_workspace_run_id(run_id: str | None) -> str | None:
    """Return a non-empty run id or ``None`` for canonical ``ws/``."""
    if run_id is None:
        return None
    s = str(run_id).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def workspace_tree_name(run_id: str | None = None) -> str:
    """Filesystem name for the active workspace tree: ``ws`` or ``ws_{run_id}``."""
    rid = normalize_workspace_run_id(run_id)
    return f"{WORKSPACE_SUBDIR}_{rid}" if rid else WORKSPACE_SUBDIR


def pipeline_plots_root(
    output_dir: str,
    subdir: str | None = PIPELINE_PLOTS_SUBDIR,
    *,
    run_id: str | None = None,
) -> str:
    """Return workspace-tree path for diagnostic figures."""
    root = os.path.abspath(workspace_root(output_dir, run_id=run_id))
    if subdir is None:
        return root
    s = str(subdir).strip()
    if not s:
        return root
    return os.path.join(root, s)


def workspace_dir(
    output_dir: str,
    label: str,
    *,
    run_id: str | None = None,
) -> str:
    """Absolute path to the workspace directory for a pipeline label."""
    return os.path.join(workspace_root(output_dir, run_id=run_id), label)


def workspace_root(output_dir: str, *, run_id: str | None = None) -> str:
    """Absolute path of the active workspace tree under *output_dir*."""
    return os.path.join(
        os.path.abspath(output_dir),
        workspace_tree_name(run_id),
    )


def workspace_artifact_path(
    output_dir: str,
    basename: str,
    *,
    run_id: str | None = None,
) -> str:
    """Path to a run-scoped artifact at the workspace tree root."""
    return os.path.join(workspace_root(output_dir, run_id=run_id), basename)


def clear_diff_workspace(
    event_dir: Union[str, Path],
    *,
    run_id: str | None = None,
) -> None:
    """Remove one workspace subtree for force rerun; preserve event_dir handoff files.

    Clears canonical ``ws/`` when *run_id* is unset, else ``ws_{run_id}/``.
    The ``templates`` and ``ffis`` symlinks inside that tree are preserved across clears.
    """
    root = Path(event_dir)
    ws = root / workspace_tree_name(run_id)
    if not ws.is_dir():
        return
    templates_link = ws / TEMPLATES_WS_LABEL
    templates_target = None
    if templates_link.is_symlink():
        try:
            templates_target = templates_link.resolve()
        except OSError:
            templates_target = None
    ffis_link = ws / FFIS_WS_LABEL
    ffis_target = None
    if ffis_link.is_symlink():
        try:
            ffis_target = ffis_link.resolve()
        except OSError:
            ffis_target = None
    shutil.rmtree(ws)
    log.info("Force rerun: removed diff workspace %s", ws)
    if templates_target is not None and templates_target.is_dir():
        ensure_event_templates_symlink(root, templates_target, run_id=run_id)
        log.info("Force rerun: restored templates symlink -> %s", templates_target)
    if ffis_target is not None and ffis_target.is_dir():
        ensure_event_ffis_symlink(root, ffis_target, run_id=run_id)
        log.info("Force rerun: restored ffis symlink -> %s", ffis_target)


def master_root(output_dir: str, *, run_id: str | None = None) -> str:
    """Absolute path of ``ws/master/`` (or debug tree) under *output_dir*."""
    return os.path.join(workspace_root(output_dir, run_id=run_id), MASTER_SUBDIR)


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
    run_id: str | None = None,
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
    ws_root = workspace_root(output_dir, run_id=run_id)
    if not os.path.isdir(ws_root):
        return 0
    m_root = master_root(output_dir, run_id=run_id)
    os.makedirs(m_root, exist_ok=True)
    refreshed = _prune_master_stamp_symlinks(m_root)

    for label in sorted(os.listdir(ws_root)):
        if label in (MASTER_SUBDIR, TEMPLATES_WS_LABEL, FFIS_WS_LABEL):
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
            try:
                ffis_link = event_ffis_symlink_path(output_dir, run_id=run_id)
                existed_ok = False
                if ffis_link.is_symlink():
                    try:
                        existed_ok = ffis_link.resolve() == Path(ffi_leaf_abs)
                    except OSError:
                        existed_ok = False
                ensure_event_ffis_symlink(output_dir, ffi_leaf_abs, run_id=run_id)
                refreshed += int(not existed_ok)
                refreshed += prune_stale_per_workspace_ffis_symlinks(
                    output_dir, run_id=run_id
                )
            except OSError as exc:
                log.warning(
                    "master workspace: ws/ffis symlink failed: %s", exc
                )
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
