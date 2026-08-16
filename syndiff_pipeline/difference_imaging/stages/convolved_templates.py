"""Convolve syndiff templates with the fixed min-background kernel."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    _load_template_cropped,
    _write_image_fits,
    build_hotpants_config,
    parse_syndiff_template_filename,
)
from syndiff_pipeline.difference_imaging.stages.kernel import (
    CONVOLVED_TEMPLATES_CSV_BASENAME,
    convolve_template_with_kernel_solution,
)
from syndiff_pipeline.difference_imaging.stages.kernel_fit import (
    kernel_r2_npz_path,
    load_kernel_fit_meta,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    PIPELINE_FITS_EXT,
    resolve_pipeline_fits_path,
    strip_fits_suffix,
)
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    convolved_template_basename,
)

log = logging.getLogger(__name__)


def convolved_templates_csv_path(ws_dir: str) -> str:
    """Convolved templates csv path.
    
    Parameters
    ----------
    ws_dir : str
    
    Returns
    -------
    str"""
    return os.path.join(ws_dir, CONVOLVED_TEMPLATES_CSV_BASENAME)


def load_convolved_templates_table(ws_dir: str) -> pd.DataFrame:
    """Load convolved templates table.
    
    Parameters
    ----------
    ws_dir : str
    
    Returns
    -------
    pd.DataFrame"""
    path = convolved_templates_csv_path(ws_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing convolved templates manifest: {path}")
    return pd.read_csv(path)


def _unique_template_entries(template_paths: dict[int, str]) -> list[dict]:
    """Unique template entries.
    
    Parameters
    ----------
    template_paths : dict[int, str]
    
    Returns
    -------
    list[dict]"""
    seen: set[tuple[float, float]] = set()
    rows: list[dict] = []
    for group_id, tmpl_path in sorted(template_paths.items()):
        parsed = parse_syndiff_template_filename(tmpl_path)
        if parsed is None:
            log.warning("Skipping unparseable template path: %s", tmpl_path)
            continue
        key = (round(parsed.dx, 6), round(parsed.dy, 6))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "group_id": int(group_id),
                "group_dx": float(parsed.dx),
                "group_dy": float(parsed.dy),
                "template_path": os.path.abspath(tmpl_path),
            }
        )
    return rows


def run_convolved_templates(
    *,
    kernel_fit_dir: str,
    crop_bounds: dict,
    template_paths: dict[int, str],
    hp: HotpantsParams,
    convolved_ws_dir: str,
    skip_existing: bool = True,
    field_ctx=None,
    manifest=None,
) -> pd.DataFrame:
    """
    Convolve each unique WCS-group template with the kernel from ``kernel_r2.npz``.

    Field mode (``field_ctx`` set): instead of parsing linear ``dx/dy`` template
    FITS, assemble each distinct ``group_id`` (from *manifest*) from the SCC field
    store, convolve it, and key the convolved product by ``group_id``.
    """
    os.makedirs(convolved_ws_dir, exist_ok=True)
    csv_path = convolved_templates_csv_path(convolved_ws_dir)
    if skip_existing and os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        if len(existing) and all(
            os.path.isfile(str(p))
            for p in existing["convolved_path"].astype(str)
        ):
            log.info("Using cached convolved templates manifest %s", csv_path)
            return existing

    meta = load_kernel_fit_meta(kernel_fit_dir)
    npz_path = kernel_r2_npz_path(kernel_fit_dir)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing kernel NPZ: {npz_path}")

    data = dict(np.load(npz_path, allow_pickle=False))
    kernel_solution = np.asarray(data["kernel_solution"], dtype=np.float64).ravel()

    hp_fit = replace(hp, hp_bgo=0)
    work = os.path.join(convolved_ws_dir, "_kernel_conv_tmp")
    os.makedirs(work, exist_ok=True)
    mapping_grid = (
        getattr(field_ctx, "mapping_grid", None) if field_ctx is not None else None
    )
    if mapping_grid is not None:
        sci_shape = tuple(mapping_grid.template_ffi_bounds()["shape"])
        science_shape = (
            int(mapping_grid.science_ymax - mapping_grid.science_ymin),
            int(mapping_grid.science_xmax - mapping_grid.science_xmin),
        )
    else:
        sci_shape = tuple(crop_bounds.get("shape") or ())
        science_shape = sci_shape
    if len(sci_shape) != 2:
        sci_shape = (
            int(crop_bounds["y_max"]) - int(crop_bounds["y_min"]),
            int(crop_bounds["x_max"]) - int(crop_bounds["x_min"]),
        )
        science_shape = sci_shape
    hp_config = build_hotpants_config(
        hp_fit,
        work,
        work,
        "kernel_conv_stub",
        write_stamps=False,
        sci_shape=sci_shape,
    )

    def _convolve_crop(template_crop: np.ndarray) -> np.ndarray:
        from syndiff_pipeline.difference_imaging.stages.hotpants import (
            resolve_hotpants_oversample,
        )
        from syndiff_pipeline.common.grid_pairing import trim_padded_products

        tmpl = np.asarray(template_crop)
        if mapping_grid is None:
            raise ValueError("convolved template output requires MAPGRID=3 geometry")
        science_shape_local = tuple(mapping_grid.science_ffi_bounds()["shape"])
        pad_rows = int(mapping_grid.conv_pad_native)

        if mapping_grid is not None:
            expected_template_shape = tuple(mapping_grid.template_ffi_bounds()["shape"])
            if tuple(tmpl.shape) != expected_template_shape:
                raise ValueError(
                    "convolution template shape does not match MAPGRID template support: "
                    f"{tmpl.shape} != {expected_template_shape}"
                )

        factor = resolve_hotpants_oversample(
            sci_shape,
            tmpl.shape,
            getattr(hp, "oversample", None),
        )
        convolved = convolve_template_with_kernel_solution(
            tmpl,
            kernel_solution,
            hp_config,
            oversample=factor,
            science_shape=sci_shape if factor > 1 else None,
        )
        if mapping_grid is not None:
            convolved = trim_padded_products(convolved, grid=mapping_grid)
        return convolved

    os.makedirs(convolved_ws_dir, exist_ok=True)

    ref_header = None
    try:
        ref_ffi = meta.get("min_bg_ffi_path")
        if ref_ffi and wcs_grouping.fits_path_exists(ref_ffi):
            ref_header = wcs_grouping.crop_ffi_header(str(ref_ffi), crop_bounds)
    except Exception as exc:
        log.warning("Could not build WCS header for convolved templates: %s", exc)

    rows: list[dict] = []
    if field_ctx is not None:
        # Field mode: assemble + convolve one template per distinct group_id.
        from syndiff_pipeline.difference_imaging.support.template_resolution import (
            build_field_mode_template_loader,
        )

        if manifest is None or "group_id" not in getattr(manifest, "columns", []):
            raise RuntimeError(
                "convolved_templates field mode requires a manifest with group_id"
            )
        loader = build_field_mode_template_loader(
            field_ctx,
            crop_bounds,
            crop_to_science=False,
        )
        gids = sorted(
            {
                int(g)
                for g in manifest["group_id"].tolist()
                if pd.notna(g) and int(g) >= 0
            }
        )
        if not gids:
            raise RuntimeError("No valid group_id in manifest for convolved_templates")
        for gid in gids:
            out_name = f"convolved_template_gid{gid}{PIPELINE_FITS_EXT}"
            out_path = os.path.join(convolved_ws_dir, out_name)
            existing = resolve_pipeline_fits_path(
                convolved_ws_dir, strip_fits_suffix(out_name)
            )
            entry = {"group_id": int(gid), "group_dx": float("nan"), "group_dy": float("nan")}
            if skip_existing and existing is not None:
                rows.append({**entry, "convolved_path": existing})
                continue
            template_crop = loader(int(gid))
            convolved = _convolve_crop(template_crop)
            _write_image_fits(out_path, convolved, header=ref_header)
            rows.append({**entry, "convolved_path": out_path})
            log.info("Convolved field template group_id=%d -> %s", gid, out_path)
    else:
        entries = _unique_template_entries(template_paths)
        if not entries:
            raise RuntimeError("No syndiff templates found to convolve")
        for entry in entries:
            tmpl_path = entry["template_path"]
            out_name = convolved_template_basename(tmpl_path)
            out_path = os.path.join(convolved_ws_dir, out_name)
            existing = resolve_pipeline_fits_path(
                convolved_ws_dir, strip_fits_suffix(out_name)
            )
            if skip_existing and existing is not None:
                rows.append({**entry, "convolved_path": existing})
                continue

            template_crop = _load_template_cropped(tmpl_path, crop_bounds)
            convolved = _convolve_crop(template_crop)
            _write_image_fits(out_path, convolved, header=ref_header)
            rows.append({**entry, "convolved_path": out_path})
            log.info(
                "Convolved template dx=%.3f dy=%.3f -> %s",
                entry["group_dx"],
                entry["group_dy"],
                out_path,
            )

    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)
    log.info("Wrote convolved templates manifest: %s", csv_path)
    return table


def lookup_convolved_path(
    table: pd.DataFrame,
    group_dx: float,
    group_dy: float,
    *,
    tol: float = 1e-3,
) -> str:
    """Return convolved template path for manifest group offsets."""
    for _, row in table.iterrows():
        if abs(float(row["group_dx"]) - group_dx) <= tol and abs(
            float(row["group_dy"]) - group_dy
        ) <= tol:
            return str(row["convolved_path"])
    raise FileNotFoundError(
        f"No convolved template for group_dx={group_dx} group_dy={group_dy}"
    )


def lookup_convolved_path_by_group_id(table: pd.DataFrame, group_id: int) -> str:
    """Return the convolved template path for a ``group_id`` (field mode)."""
    if "group_id" not in table.columns:
        raise FileNotFoundError("convolved templates table has no group_id column")
    hit = table.loc[table["group_id"].astype("Int64") == int(group_id)]
    if hit.empty:
        raise FileNotFoundError(f"No convolved template for group_id={group_id}")
    return str(hit.iloc[0]["convolved_path"])
