"""
Workspace and manifest path conventions for the config-driven pipeline.

Workspaces live under ``{output_dir}/ws/{label}/``.
The default per-FFI manifest basename is ``syndiff_ffi_frames.csv`` at ``output_dir``.
"""

from __future__ import annotations

import os
from typing import Optional

WORKSPACE_SUBDIR = "ws"
DEFAULT_MANIFEST_BASENAME = "syndiff_ffi_frames.csv"

# ``np.savez(..., **{BACKGROUND_STACK_NPZ_ARRAY_KEY: stack})`` for rough/smooth stacks
BACKGROUND_STACK_NPZ_ARRAY_KEY = "stack"

# Basename (no extension) for adaptive background stacks under ``ws/<label>/``
ADAPTIVE_BKG_STACK_BASENAME = "bkg_temp_smooth"


def workspace_dir(output_dir: str, label: str) -> str:
    """Absolute path to the workspace directory for a pipeline label."""
    return os.path.join(os.path.abspath(output_dir), WORKSPACE_SUBDIR, label)


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
