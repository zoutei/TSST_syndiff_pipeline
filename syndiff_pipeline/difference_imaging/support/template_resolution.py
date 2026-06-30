"""Resolve per-FFI WCS-group templates from manifest offsets."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    event_templates_symlink_path,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    parse_syndiff_template_filename,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import PIPELINE_FITS_EXT


def _offset_match(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= max(1e-5, tol)


def find_template_by_offset(
    template_dir: str | Path,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    offset_tol: float = 1e-3,
) -> str:
    """Find a syndiff template FITS with the requested (dx, dy) sub-pixel offset."""
    root = Path(template_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {root}")

    matches: list[str] = []
    for full in sorted(root.iterdir()):
        if not full.is_file():
            continue
        parsed = parse_syndiff_template_filename(str(full))
        if parsed is None:
            continue
        if _offset_match(parsed.dx, dx, offset_tol) and _offset_match(
            parsed.dy, dy, offset_tol
        ):
            matches.append(str(full.resolve()))

    if not matches:
        raise FileNotFoundError(
            f"No syndiff_template with dx={dx} dy={dy} under {root}"
        )
    if len(matches) > 1:
        prefer_gz = [p for p in matches if p.lower().endswith(".fits.gz")]
        return prefer_gz[0] if prefer_gz else matches[0]
    return matches[0]


def resolve_template_dir(output_dir: str, *, run_id: str | None = None) -> str:
    link = event_templates_symlink_path(output_dir, run_id=run_id)
    if link.is_symlink() or link.is_dir():
        return str(link.resolve())
    ws_templates = Path(output_dir) / "ws" / "templates"
    if ws_templates.exists():
        return str(ws_templates.resolve())
    event_root = Path(output_dir).expanduser().resolve()
    for cand in sorted(event_root.glob("ws_*/templates")):
        if cand.is_symlink() or cand.is_dir():
            return str(cand.resolve())
    raise FileNotFoundError(
        f"No template directory found under {output_dir} "
        "(expected ws/templates symlink or directory)."
    )


def template_offsets_for_ffi(
    manifest: pd.DataFrame,
    ffi_path: str,
) -> tuple[float, float]:
    """Return ``(group_dx, group_dy)`` for an FFI from the manifest."""
    from syndiff_pipeline.common.wcs_grouping import ref_manifest_row_index

    for col in ("group_dx", "group_dy"):
        if col not in manifest.columns:
            raise KeyError(
                f"manifest missing {col!r}; expected syndiff_ffi_frames.csv columns."
            )

    idx = ref_manifest_row_index(manifest, ffi_path)
    if idx is None:
        raise ValueError(f"No manifest row for FFI {ffi_path!r}")

    row = manifest.iloc[idx]
    gdx = row["group_dx"]
    gdy = row["group_dy"]
    if pd.isna(gdx) or pd.isna(gdy):
        raise ValueError(
            f"Manifest row for {ffi_path!r} has NaN group_dx/group_dy "
            f"(group_id={row.get('group_id', '?')})"
        )
    return float(gdx), float(gdy)


def resolve_template_for_ffi(
    output_dir: str,
    manifest: pd.DataFrame,
    ffi_path: str,
    *,
    template_dir: str | None = None,
) -> tuple[float, float, str]:
    """Return ``(group_dx, group_dy, template_path)`` for one FFI."""
    group_dx, group_dy = template_offsets_for_ffi(manifest, ffi_path)
    tmpl_root = template_dir or resolve_template_dir(output_dir)
    template_path = find_template_by_offset(
        tmpl_root, dx=group_dx, dy=group_dy
    )
    return group_dx, group_dy, template_path


def convolved_template_basename(template_path: str) -> str:
    parsed = parse_syndiff_template_filename(template_path)
    if parsed is None:
        return f"convolved_template{PIPELINE_FITS_EXT}"
    return (
        f"convolved_template_dx{parsed.dx:.3f}_dy{parsed.dy:.3f}{PIPELINE_FITS_EXT}"
    )
