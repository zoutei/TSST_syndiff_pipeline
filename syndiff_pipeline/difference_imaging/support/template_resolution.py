"""Resolve per-FFI WCS-group templates from manifest offsets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from syndiff_pipeline.common.fits_variants import fits_logical_path, prefer_fits_path
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    parse_syndiff_template_filename,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import PIPELINE_FITS_EXT


def _offset_match(a: float, b: float, tol: float = 1e-3) -> bool:
    """Offset match.
    
    Parameters
    ----------
    a : float
    b : float
    tol : float, optional, default ``0.001``
    
    Returns
    -------
    bool"""
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
        preferred = prefer_fits_path(matches)
        return preferred if preferred is not None else matches[0]
    return matches[0]


def resolve_template_dir(
    output_dir: str,
    *,
    run_id: str | None = None,
    data_root: str | None = None,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
    oversampling_factor: int = 1,
    store_name: str | None = None,
) -> str:
    """Resolve SCC templates directory for an event leaf."""
    from syndiff_pipeline.common.scc_paths import scc_templates_dir
    from syndiff_pipeline.common.wcs_grouping import _event_job_path

    if data_root is not None and sector is not None and camera is not None and ccd is not None:
        store = scc_templates_dir(
            data_root,
            sector,
            camera,
            ccd,
            oversampling_factor=oversampling_factor,
            store_name=store_name,
        )
        if store.is_dir():
            return str(store.resolve())

    job = Path(_event_job_path(output_dir))
    if job.is_file():
        try:
            payload = json.loads(job.read_text())
            s, c, k = int(payload["sector"]), int(payload["camera"]), int(payload["ccd"])
            if data_root:
                store = scc_templates_dir(
                    data_root,
                    s,
                    c,
                    k,
                    oversampling_factor=oversampling_factor,
                    store_name=store_name,
                )
                if store.is_dir():
                    return str(store.resolve())
        except Exception:
            pass

    raise FileNotFoundError(
        f"No SCC templates store found for event leaf {output_dir!r}. "
        "Run template pipeline templates stage first."
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


def convolved_template_basename(template_path: str) -> str:
    """Convolved template basename.
    
    Parameters
    ----------
    template_path : str
    
    Returns
    -------
    str"""
    parsed = parse_syndiff_template_filename(template_path)
    if parsed is None:
        return f"convolved_template{PIPELINE_FITS_EXT}"
    return (
        f"convolved_template_dx{parsed.dx:.3f}_dy{parsed.dy:.3f}{PIPELINE_FITS_EXT}"
    )


# ── field-mode resolution (group_id + SCC field store / optional FITS) ─────

_FIELD_GID_FITS_RE = re.compile(
    r"^syndiff_field_s\d+_\d+_\d+(?:_os\d+)?_gid(?P<gid>\d+)\.fits(?:\.fz|\.gz)?$",
    re.IGNORECASE,
)


def is_field_template_store(template_dir: str | Path) -> bool:
    """True if *template_dir* looks like an SCC field_templates root."""
    return (Path(template_dir) / "template_manifest.json").is_file()


def parse_field_gid_from_filename(path_or_basename: str | Path) -> int | None:
    m = _FIELD_GID_FITS_RE.match(Path(path_or_basename).name)
    return int(m.group("gid")) if m else None


def group_id_for_ffi(manifest: pd.DataFrame, ffi_path: str) -> int:
    """Return ``group_id`` for an FFI from the frames CSV."""
    from syndiff_pipeline.common.wcs_grouping import ref_manifest_row_index

    if "group_id" not in manifest.columns:
        raise KeyError("manifest missing 'group_id'")
    idx = ref_manifest_row_index(manifest, ffi_path)
    if idx is None:
        raise ValueError(f"No manifest row for FFI {ffi_path!r}")
    gid = manifest.iloc[idx]["group_id"]
    if pd.isna(gid) or int(gid) < 0:
        raise ValueError(f"Invalid group_id for FFI {ffi_path!r}: {gid!r}")
    return int(gid)


def find_field_fits_by_group_id(template_dir: str | Path, group_id: int) -> str | None:
    """Locate optional materialized ``syndiff_field_*_gid{N}.fits*`` under store/fits/."""
    root = Path(template_dir)
    matches: list[Path] = []
    for d in (root / "fits", root):
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file() and parse_field_gid_from_filename(p) == int(group_id):
                matches.append(p)
    if not matches:
        return None
    preferred = prefer_fits_path(matches)
    chosen = Path(preferred) if preferred is not None else matches[0]
    return str(chosen.resolve())


def resolve_template_for_ffi(
    output_dir: str,
    manifest: pd.DataFrame,
    ffi_path: str,
    *,
    template_dir: str | None = None,
    geometry_mode: str | None = None,
):
    """
    Resolve a template for one FFI.

    Linear: ``(group_dx, group_dy, path)``.
    Field with materialized FITS: ``(group_id, path)``.
    Field without FITS: raises ``FileNotFoundError`` (assemble from contribs).
    """
    tmpl_root = template_dir or resolve_template_dir(output_dir)
    mode = (geometry_mode or "").lower()
    if mode == "linear":
        use_field = False
    elif mode == "field":
        use_field = True
    else:
        # Unset: detect from store (legacy linear events omit geometry_mode).
        use_field = is_field_template_store(tmpl_root)
    if use_field:
        gid = group_id_for_ffi(manifest, ffi_path)
        path = find_field_fits_by_group_id(tmpl_root, gid)
        if path is None:
            raise FileNotFoundError(
                f"No materialized field FITS for group_id={gid} under {tmpl_root}; "
                "assemble from SCC contribs (geometry_mode=field, materialize_fits=false)."
            )
        return gid, path
    group_dx, group_dy = template_offsets_for_ffi(manifest, ffi_path)
    template_path = find_template_by_offset(tmpl_root, dx=group_dx, dy=group_dy)
    return group_dx, group_dy, template_path


# ── on-demand field assemble loader ───────────────────────────────────────

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class FieldModeTemplateContext:
    """Picklable context for assembling field templates from SCC contribs."""

    store_root: str
    shifts_df: Any  # pd.DataFrame
    base_tess_shape: tuple[int, int]
    template_roi_bounds: tuple[int, int, int, int]
    oversampling_factor: int = 1
    mapping_grid: Any = None  # MappingGrid | None for schema v3


def build_field_mode_template_loader(
    ctx: FieldModeTemplateContext,
    diff_crop_bounds: dict,
    *,
    planes: str = "flux",
    crop_to_science: bool = True,
) -> Callable[[int], Any]:
    """Return ``group_id -> cropped flux array`` (float64).

    *diff_crop_bounds* are native FFI coordinates. When
    ``ctx.oversampling_factor`` > 1, the assembled array is oversampled and the
    crop window into that array is scaled accordingly.
    """
    import numpy as np

    from syndiff_pipeline.template_creation.processing.field_downsample import (
        assemble_field_group_flux,
    )

    if planes not in ("flux", "all"):
        raise ValueError(f"planes must be flux|all, got {planes!r}")

    os_factor = max(1, int(getattr(ctx, "oversampling_factor", 1) or 1))
    grid = getattr(ctx, "mapping_grid", None)
    if grid is not None and not crop_to_science:
        if os_factor > 1:
            h, w = grid.array_shape_os()
        else:
            h, w = grid.array_shape_native()
        lx0, ly0, lx1, ly1 = 0, 0, int(w), int(h)
        x_min, y_min, x_max, y_max = ctx.template_roi_bounds
    elif grid is not None:
        lx0 = int(round((int(diff_crop_bounds["x_min"]) - grid.ffi_xmin) * os_factor))
        lx1 = int(round((int(diff_crop_bounds["x_max"]) - grid.ffi_xmin) * os_factor))
        ly0 = int(round((int(diff_crop_bounds["y_min"]) - grid.ffi_ymin) * os_factor))
        ly1 = int(round((int(diff_crop_bounds["y_max"]) - grid.ffi_ymin) * os_factor))
        x_min, y_min, x_max, y_max = ctx.template_roi_bounds
    else:
        x_min, y_min, x_max, y_max = ctx.template_roi_bounds
        lx0 = int(diff_crop_bounds["x_min"]) * os_factor - int(x_min)
        lx1 = int(diff_crop_bounds["x_max"]) * os_factor - int(x_min)
        ly0 = int(diff_crop_bounds["y_min"]) * os_factor - int(y_min)
        ly1 = int(diff_crop_bounds["y_max"]) * os_factor - int(y_min)

    def _load(group_id: int):
        crop = (int(x_min), int(x_max), int(y_min), int(y_max))
        flux = assemble_field_group_flux(
            ctx.store_root,
            ctx.shifts_df,
            int(group_id),
            shape=ctx.base_tess_shape,
            crop=crop,
        )
        out = flux[ly0:ly1, lx0:lx1].astype(np.float32).astype(np.float64)
        if planes == "all":
            # COUNT/MASK not reconstructed here; stack flux thrice for API parity callers
            return np.stack([out, np.ones_like(out), np.zeros_like(out)], axis=0)
        return out

    return _load


def build_field_mode_count_loader(
    ctx: FieldModeTemplateContext,
    diff_crop_bounds: dict,
) -> Callable[[int], Any]:
    """Return ``group_id -> cropped PS1 hit-COUNT array`` (float64).

    Same crop geometry as :func:`build_field_mode_template_loader`, but assembles
    the COUNT plane — used by ``shared_mask``'s PS1-coverage mask in field mode
    (equivalent to a linear template's COUNT extension).
    """
    import numpy as np

    from syndiff_pipeline.template_creation.processing.field_downsample import (
        assemble_field_group_count,
    )

    os_factor = max(1, int(getattr(ctx, "oversampling_factor", 1) or 1))
    grid = getattr(ctx, "mapping_grid", None)
    if grid is not None:
        lx0 = int(round((int(diff_crop_bounds["x_min"]) - grid.ffi_xmin) * os_factor))
        lx1 = int(round((int(diff_crop_bounds["x_max"]) - grid.ffi_xmin) * os_factor))
        ly0 = int(round((int(diff_crop_bounds["y_min"]) - grid.ffi_ymin) * os_factor))
        ly1 = int(round((int(diff_crop_bounds["y_max"]) - grid.ffi_ymin) * os_factor))
        x_min, y_min, x_max, y_max = ctx.template_roi_bounds
    else:
        x_min, y_min, x_max, y_max = ctx.template_roi_bounds
        lx0 = int(diff_crop_bounds["x_min"]) * os_factor - int(x_min)
        lx1 = int(diff_crop_bounds["x_max"]) * os_factor - int(x_min)
        ly0 = int(diff_crop_bounds["y_min"]) * os_factor - int(y_min)
        ly1 = int(diff_crop_bounds["y_max"]) * os_factor - int(y_min)

    def _load(group_id: int):
        crop = (int(x_min), int(x_max), int(y_min), int(y_max))
        count = assemble_field_group_count(
            ctx.store_root,
            ctx.shifts_df,
            int(group_id),
            shape=ctx.base_tess_shape,
            crop=crop,
        )
        count_hr = count[ly0:ly1, lx0:lx1]
        if os_factor > 1:
            from syndiff_pipeline.common.template_coverage import (
                block_sum_oversampled_to_native,
            )

            count_hr = block_sum_oversampled_to_native(count_hr, os_factor)
        return count_hr.astype(np.float64)

    return _load


def _group_id_for_ffi_name(manifest, ffi_name: str) -> int:
    """Resolve group_id from an FFI *name* (basename) or full path.

    Prefers the canonical full-path match (:func:`group_id_for_ffi`); falls back
    to a basename match against the manifest ``filename``/``path`` columns,
    treating ``.fits`` / ``.fits.gz`` / ``.fits.fz`` as the same file.
    """
    from syndiff_pipeline.common.wcs_grouping import ref_manifest_row_index

    if ref_manifest_row_index(manifest, ffi_name) is not None:
        return group_id_for_ffi(manifest, ffi_name)

    def _stem(name: str) -> str:
        return Path(fits_logical_path(name)).name

    target = _stem(ffi_name)
    for col in ("filename", "path"):
        if col not in manifest.columns:
            continue
        matches = manifest.index[manifest[col].map(_stem) == target].tolist()
        if matches:
            gid = manifest.loc[matches[0], "group_id"]
            if pd.isna(gid) or int(gid) < 0:
                raise ValueError(f"Invalid group_id for FFI {ffi_name!r}: {gid!r}")
            return int(gid)
    raise ValueError(f"No manifest row for FFI {ffi_name!r}")


def assemble_field_template_for_ffi(
    ctx: "FieldModeTemplateContext",
    manifest,
    ffi_path: str,
    *,
    crop: tuple[int, int, int, int] | None = None,
    plane: str = "flux",
):
    """Assemble the field template for the group of a single FFI, by FFI name.

    Resolves the FFI's ``group_id`` from *manifest* (the frames CSV), then
    assembles that group's template from the SCC field store. With ``crop=None``
    (default) this returns the **full-FFI** ("big") template at
    ``ctx.base_tess_shape``; pass ``crop=(x_min, x_max, y_min, y_max)`` in
    full-FFI pixels for a window. Works against a crop-only store — skycells
    without contribs stay zero (``present_only=True``).

    Parameters
    ----------
    ctx : FieldModeTemplateContext
    manifest : pandas.DataFrame
        The event frames manifest (``syndiff_ffi_frames.csv``) with ``group_id``.
    ffi_path : str
        FFI filename or path (matched via the manifest's canonical path key).
    crop : tuple, optional
        ``(x_min, x_max, y_min, y_max)`` half-open full-FFI window; ``None`` → full FFI.
    plane : str
        ``"flux"`` (mean flux, default) or ``"count"`` (PS1 hit count).

    Returns
    -------
    numpy.ndarray (float64)
    """
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        assemble_field_group_count,
        assemble_field_group_flux,
    )

    if plane not in ("flux", "count"):
        raise ValueError(f"plane must be flux|count, got {plane!r}")
    gid = _group_id_for_ffi_name(manifest, ffi_path)
    assembler = assemble_field_group_flux if plane == "flux" else assemble_field_group_count
    return assembler(
        ctx.store_root,
        ctx.shifts_df,
        int(gid),
        shape=ctx.base_tess_shape,
        crop=crop,
        present_only=True,
    )


def _resolve_template_group_shifts_path(
    store_root: Path,
    side: Mapping[str, Any],
) -> Path:
    """Locate ``template_group_shifts.parquet`` for an SCC field store."""
    candidates: list[Path] = [store_root / "template_group_shifts.parquet"]
    remap_root = side.get("remap_root")
    if remap_root:
        candidates.append(Path(str(remap_root)) / "template_group_shifts.parquet")
    for path in candidates:
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No template_group_shifts.parquet found for field store {store_root} "
        f"(tried: {tried})"
    )


def _field_mode_context_from_sidecar(
    store_root: Path,
    side: Mapping[str, Any],
    *,
    shifts_df: pd.DataFrame,
) -> FieldModeTemplateContext:
    from syndiff_pipeline.common.mapping_grid import MappingGrid, MappingGridError

    schema_version = int(side.get("schema_version", 0))
    if schema_version < 3 or "mapping_grid" not in side:
        raise MappingGridError(
            f"field template store {store_root} is v1/v2 field mode "
            f"(field_mode_assembly schema_version={schema_version}; need >=3 with mapping_grid). "
            "Rebuild mapping (MAPGRID>=2) and field downsample before using field-mode templates."
        )
    os_factor = max(1, int(side.get("oversampling_factor", 1) or 1))
    mapping_grid = MappingGrid.from_sidecar(side)
    if os_factor > 1:
        h, w = mapping_grid.array_shape_os()
    else:
        h, w = mapping_grid.array_shape_native()
    template_roi_bounds = (0, 0, int(w), int(h))
    return FieldModeTemplateContext(
        store_root=str(side.get("store_root") or store_root),
        shifts_df=shifts_df,
        base_tess_shape=tuple(side["base_tess_shape"]),
        template_roi_bounds=template_roi_bounds,
        oversampling_factor=os_factor,
        mapping_grid=mapping_grid,
    )


def load_field_mode_template_context_from_store(
    store_root: str | Path,
) -> FieldModeTemplateContext:
    """Load field assemble context from an SCC template store sidecar."""
    root = Path(str(store_root)).expanduser().resolve()
    sidecar = root / "field_mode_assembly.json"
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing field mode sidecar: {sidecar}")
    side = json.loads(sidecar.read_text(encoding="utf-8"))
    shifts_path = _resolve_template_group_shifts_path(root, side)
    shifts_df = pd.read_parquet(shifts_path)
    return _field_mode_context_from_sidecar(root, side, shifts_df=shifts_df)


def write_field_template_fits_for_ffi(
    ctx: FieldModeTemplateContext,
    manifest: pd.DataFrame,
    ffi_path: str,
    out_path: str | Path,
    *,
    sector: int,
    camera: int,
    ccd: int,
    crop: tuple[int, int, int, int] | None = None,
    provenance: Mapping[str, Any] | None = None,
    extra_header: Mapping[str, Any] | None = None,
) -> str:
    """Assemble and write one FFI's field template FITS (flux + COUNT)."""
    from syndiff_pipeline.template_creation.processing.field_templates import (
        build_field_fits_header,
        write_field_group_fits,
    )

    gid = _group_id_for_ffi_name(manifest, ffi_path)
    flux = assemble_field_template_for_ffi(
        ctx, manifest, ffi_path, crop=crop, plane="flux"
    )
    count = assemble_field_template_for_ffi(
        ctx, manifest, ffi_path, crop=crop, plane="count"
    )
    roi_bounds = None
    if crop is not None:
        x_min, x_max, y_min, y_max = (int(v) for v in crop)
        roi_bounds = (x_min, y_min, x_max, y_max)
    elif ctx.template_roi_bounds:
        roi_bounds = tuple(int(v) for v in ctx.template_roi_bounds)

    header = build_field_fits_header(
        sector=int(sector),
        camera=int(camera),
        ccd=int(ccd),
        group_id=int(gid),
        oversampling_factor=int(ctx.oversampling_factor),
        roi_bounds=roi_bounds,
        provenance=provenance,
        mapping_grid=ctx.mapping_grid,
    )
    ffi_file = Path(fits_logical_path(ffi_path)).name
    header["FFI_FILE"] = (ffi_file, "Source FFI logical filename")
    if extra_header:
        for key, value in extra_header.items():
            header[str(key)] = value

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return write_field_group_fits(out, flux, count, header=header)


def maybe_load_field_mode_template_context(
    template_dir: str | Path | None,
    event_dir: str | Path,
) -> Optional[FieldModeTemplateContext]:
    """Load field assemble context from SCC store sidecar, or None."""
    import json
    import logging

    log = logging.getLogger(__name__)
    if template_dir is None:
        return None
    root = Path(str(template_dir)).expanduser()
    if not root.is_dir():
        # try event ws/field_templates
        link = Path(event_dir) / "ws" / "field_templates"
        if link.exists():
            root = link.resolve()
        else:
            return None
    sidecar = root / "field_mode_assembly.json"
    if not sidecar.is_file():
        return None
    try:
        side = json.loads(sidecar.read_text())
    except Exception as exc:
        log.warning("field mode context load failed: %s", exc)
        return None
    remap_root = side.get("remap_root")
    shifts_path = None
    if remap_root:
        remap_shifts = Path(str(remap_root)) / "template_group_shifts.parquet"
        if remap_shifts.is_file():
            shifts_path = remap_shifts
    if shifts_path is None:
        event_shifts = Path(event_dir) / "template_group_shifts.parquet"
        if event_shifts.is_file():
            shifts_path = event_shifts
    if shifts_path is None:
        return None
    try:
        shifts_df = pd.read_parquet(shifts_path)
    except Exception as exc:
        log.warning("field mode context load failed: %s", exc)
        return None
    return _field_mode_context_from_sidecar(root, side, shifts_df=shifts_df)

