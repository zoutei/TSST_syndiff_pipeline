"""
Workspace and manifest path conventions for the config-driven pipeline.

The diff pipeline's ``output_dir`` is the SCC diff lane root directly (no
``ws/`` subtree; see ``workspace_lock.py`` for where the config lock lives).
The ``ws/{label}/`` naming conventions below (``workspace_root``,
``workspace_dir``, ...) remain in use by other, still event-scoped trees
(e.g. photometry's ``pipeline_plots_root``).
The default per-FFI manifest basename is ``syndiff_ffi_frames.csv`` at ``output_dir``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

WORKSPACE_SUBDIR = "ws"
PHOTOMETRY_SUBDIR = "phot"
HOST_STAR_WS_LABEL = "host_star"
DEFAULT_MANIFEST_BASENAME = "frames.csv"
LEGACY_MANIFEST_BASENAME = "syndiff_ffi_frames.csv"
HOTPANTS_SUBSTAMP_STARS_BASENAME = "hotpants_substamp_stars.csv"
TARGETS_DS9_REGION_BASENAME = "targets.reg"
# Canonical on-disk static bitmask (int16, fpacked). Legacy .fits.gz/.fits
# are still resolved by resolve_pipeline_artifact_path at read time.
SHARED_MASK_FITS_BASENAME = "shared_mask.fits.fz"
STATIC_MASK_FITS_BASENAME = SHARED_MASK_FITS_BASENAME  # alias (main naming)
GAIA_CATALOG_PIPELINE_BASENAME = "gaia_catalog_pipeline.csv"
DIFF_CONFIG_SNAPSHOT_BASENAME = "diff_config.yaml"

WORKSPACE_ROOT_ARTIFACTS = (
    SHARED_MASK_FITS_BASENAME,
    HOTPANTS_SUBSTAMP_STARS_BASENAME,
    GAIA_CATALOG_PIPELINE_BASENAME,
    TARGETS_DS9_REGION_BASENAME,
    DIFF_CONFIG_SNAPSHOT_BASENAME,
)

__all__ = [
    "HOST_STAR_WS_LABEL",
    "SHARED_MASK_FITS_BASENAME",
    "STATIC_MASK_FITS_BASENAME",
]

# ``np.savez(..., **{BACKGROUND_STACK_NPZ_ARRAY_KEY: stack})`` for rough/smooth stacks
BACKGROUND_STACK_NPZ_ARRAY_KEY = "stack"

# Basename (no extension) for adaptive background stacks under ``ws/<label>/``
ADAPTIVE_BKG_STACK_BASENAME = "bkg_temp_smooth"

# Union mask (2D): pixels where PRF source-hunt excluded sky in any epoch (output_dir root)
BKG_SOURCE_HUNT_UNION_FITS_BASENAME = "bkg_source_hunt_union.fits.fz"

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


def normalize_photometry_run_id(run_id: str | None) -> str | None:
    """Return a non-empty photometry run id or ``None`` for canonical ``photometry/``."""
    if run_id is None:
        return None
    s = str(run_id).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def photometry_tree_name(run_id: str | None = None) -> str:
    """Filesystem name for the photometry tree: ``phot`` or ``phot_{run_id}``."""
    rid = normalize_photometry_run_id(run_id)
    return f"{PHOTOMETRY_SUBDIR}_{rid}" if rid else PHOTOMETRY_SUBDIR


def photometry_root(event_dir: str, run_id: str | None = None) -> str:
    """Absolute path of the photometry tree under one event directory."""
    return os.path.join(os.path.abspath(event_dir), photometry_tree_name(run_id))


def workspace_artifact_path(
    output_dir: str,
    basename: str,
    *,
    run_id: str | None = None,
) -> str:
    """Path to a run-scoped artifact at the workspace tree root."""
    return os.path.join(workspace_root(output_dir, run_id=run_id), basename)


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
