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
    resolve_pipeline_fits_path,
    strip_fits_suffix,
)
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    convolved_template_basename,
)

log = logging.getLogger(__name__)


def convolved_templates_csv_path(ws_dir: str) -> str:
    return os.path.join(ws_dir, CONVOLVED_TEMPLATES_CSV_BASENAME)


def load_convolved_templates_table(ws_dir: str) -> pd.DataFrame:
    path = convolved_templates_csv_path(ws_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing convolved templates manifest: {path}")
    return pd.read_csv(path)


def _unique_template_entries(template_paths: dict[int, str]) -> list[dict]:
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
) -> pd.DataFrame:
    """
    Convolve each unique WCS-group template with the kernel from ``kernel_r2.npz``.
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
    hp_config = build_hotpants_config(
        hp_fit,
        work,
        work,
        "kernel_conv_stub",
        write_stamps=False,
    )

    os.makedirs(convolved_ws_dir, exist_ok=True)
    entries = _unique_template_entries(template_paths)
    if not entries:
        raise RuntimeError("No syndiff templates found to convolve")

    ref_header = None
    try:
        ref_ffi = meta.get("min_bg_ffi_path")
        if ref_ffi and wcs_grouping.fits_path_exists(ref_ffi):
            ref_header = wcs_grouping.crop_ffi_header(str(ref_ffi), crop_bounds)
    except Exception as exc:
        log.warning("Could not build WCS header for convolved templates: %s", exc)

    rows: list[dict] = []
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
        convolved = convolve_template_with_kernel_solution(
            template_crop, kernel_solution, hp_config
        )
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
