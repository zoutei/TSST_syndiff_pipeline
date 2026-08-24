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
from syndiff_pipeline.difference_imaging.stages.convolved_templates_progress import (
    init_progress_pair as _init_progress_pair,
    progress_path_for_convolved_workspace as _progress_path_for_convolved_workspace,
    progress_path_for_diff_log as _progress_path_for_diff_log,
    record_group_progress as _record_group_progress,
    set_progress_phase_pair as _set_progress_phase_pair,
)

log = logging.getLogger(__name__)


def _progress_paths(convolved_ws_dir: str, diff_log_path: str | None):
    """Return (workspace_sidecar, cli_mirror_or_none) progress paths."""
    ws_path = str(_progress_path_for_convolved_workspace(convolved_ws_dir))
    cli_path = str(_progress_path_for_diff_log(diff_log_path)) if diff_log_path else None
    return ws_path, cli_path


def _mark_progress_complete(convolved_ws_dir: str, diff_log_path: str | None, *, total: int) -> None:
    """Best-effort: stamp the progress sidecar complete on a cache-hit return."""
    ws_path, cli_path = _progress_paths(convolved_ws_dir, diff_log_path)
    _init_progress_pair(ws_path, cli_path, groups_total=total)
    _set_progress_phase_pair(ws_path, cli_path, "complete")


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


def crop_hr_from_template_roi_bounds(
    template_roi_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """``FieldModeTemplateContext.template_roi_bounds`` -> the ``(x0, x1,
    y0, y1)`` crop convention ``build_group_convolved_template`` expects.

    ``template_roi_bounds`` is stored as ``(x_min, y_min, x_max, y_max)``
    (see ``FieldModeTemplateContext`` / ``build_field_mode_template_loader``,
    which reorders it into ``(x0, x1, y0, y1)`` before using it as a crop
    tuple) -- a *different* field order than the ``(x0, x1, y0, y1)`` crop
    convention used everywhere else (``assemble_group_from_contribs``,
    ``build_group_convolved_template``). Passing ``template_roi_bounds``
    straight through without this reorder silently computes
    ``nx_hr``/``ny_hr`` as ``y_min - x_min`` / ``y_max - x_max`` instead of
    the real width/height -- usually wrong, occasionally negative (caught
    on a real S20/C3/K3 run: ``ValueError: negative dimensions are not
    allowed`` in ``scatter_add_patch_valid_maps``).
    """
    x_min, y_min, x_max, y_max = (int(v) for v in template_roi_bounds)
    return (x_min, x_max, y_min, y_max)


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
    diff_log_path: str | None = None,
    use_patch_cache: bool = False,
) -> pd.DataFrame:
    """
    Convolve each unique WCS-group template with the kernel from ``kernel_r2.npz``.

    Field mode (``field_ctx`` set): instead of parsing linear ``dx/dy`` template
    FITS, assemble each distinct ``group_id`` (from *manifest*) from the SCC field
    store, convolve it, and key the convolved product by ``group_id``.
    """
    os.makedirs(convolved_ws_dir, exist_ok=True)
    csv_path = convolved_templates_csv_path(convolved_ws_dir)

    # Field mode's manifest is keyed by group_id, and group_id coverage grows
    # as more of the SCC's frame list is included (e.g. a smoke-test-scale
    # run's cached manifest only lists a handful of groups). A per-file
    # existence check alone can't detect that the cache predates the current
    # (larger) run's required group set, so compute it up front and require
    # the cache to be a superset before trusting it -- otherwise a stale,
    # partial manifest silently short-circuits the whole build loop below
    # and every frame outside its groups fails downstream with "No
    # convolved template" forever (found in production, 2026-08-23: a
    # 9-group manifest from an old 10-frame smoke test capped a live
    # 1183-frame S20/C3/K3 run at 453 completed hotpants frames).
    required_gids: set[int] | None = None
    if field_ctx is not None:
        if manifest is None or "group_id" not in getattr(manifest, "columns", []):
            raise RuntimeError(
                "convolved_templates field mode requires a manifest with group_id"
            )
        required_gids = {
            int(g) for g in manifest["group_id"].tolist() if pd.notna(g) and int(g) >= 0
        }

    if skip_existing and os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        covers_required = required_gids is None or set(
            existing["group_id"].astype(int)
        ) >= required_gids
        if (
            len(existing)
            and covers_required
            and all(
                os.path.isfile(str(p))
                for p in existing["convolved_path"].astype(str)
            )
        ):
            log.info("Using cached convolved templates manifest %s", csv_path)
            _mark_progress_complete(convolved_ws_dir, diff_log_path, total=len(existing))
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
    linear_pad = 0
    if mapping_grid is None and field_ctx is None:
        # Linear mode has no live field-context grid, but its templates carry
        # their own frozen support-plane header (see linear_downsample.py) --
        # validate it the same way kernel_fit/hotpants do, from whichever
        # template we have on hand (every group_dx/dy template for one SCC
        # shares the identical support geometry). Raises if a template
        # doesn't actually cover crop_bounds with padding; see
        # hotpants._resolve_linear_template_pad.
        probe_entries = _unique_template_entries(template_paths) if template_paths else []
        if probe_entries:
            from syndiff_pipeline.difference_imaging.stages.hotpants import (
                _resolve_linear_template_pad,
            )

            linear_pad = (
                _resolve_linear_template_pad(
                    probe_entries[0]["template_path"], crop_bounds
                )
                or 0
            )

    if mapping_grid is not None:
        sci_shape = tuple(mapping_grid.template_ffi_bounds()["shape"])
        science_shape = (
            int(mapping_grid.science_ymax - mapping_grid.science_ymin),
            int(mapping_grid.science_xmax - mapping_grid.science_xmin),
        )
    else:
        science_shape = tuple(crop_bounds.get("shape") or ())
        if len(science_shape) != 2:
            science_shape = (
                int(crop_bounds["y_max"]) - int(crop_bounds["y_min"]),
                int(crop_bounds["x_max"]) - int(crop_bounds["x_min"]),
            )
        sci_shape = (
            (science_shape[0] + 2 * linear_pad, science_shape[1] + 2 * linear_pad)
            if linear_pad
            else science_shape
        )
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
            _trim_linear_pad,
            resolve_hotpants_oversample,
        )
        from syndiff_pipeline.common.grid_pairing import trim_padded_products

        tmpl = np.asarray(template_crop)
        if mapping_grid is not None:
            science_shape_local = tuple(mapping_grid.science_ffi_bounds()["shape"])
            pad_rows = int(mapping_grid.conv_pad_native)
            # The template is assembled at the field store's own oversampling
            # (1x or Nx), not always native -- template_ffi_bounds() only
            # returns the native-resolution support shape. Accept either the
            # native shape or the Nx-oversampled shape; resolve_hotpants_oversample
            # below re-derives the actual factor from the sci/tmpl ratio.
            native_template_shape = tuple(mapping_grid.template_ffi_bounds()["shape"])
            os_template_shape = tuple(mapping_grid.array_shape_os())
            if tuple(tmpl.shape) not in (native_template_shape, os_template_shape):
                raise ValueError(
                    "convolution template shape does not match MAPGRID template support: "
                    f"{tmpl.shape} not in {{{native_template_shape}, {os_template_shape}}}"
                )
        elif linear_pad:
            science_shape_local = science_shape
            pad_rows = linear_pad
            expected_template_shape = (
                science_shape[0] + 2 * linear_pad,
                science_shape[1] + 2 * linear_pad,
            )
            if tuple(tmpl.shape) != expected_template_shape:
                raise ValueError(
                    "convolution template shape does not match crop_bounds "
                    f"+/- linear padding: {tmpl.shape} != {expected_template_shape}"
                )
        else:
            # No support-plane header on this template at all (fully
            # legacy/synthetic template, or run_convolved_templates was
            # called with an empty template_paths) -- no padding contract is
            # being claimed, so pass the crop through unpadded.
            science_shape_local = sci_shape
            pad_rows = 0

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
        elif linear_pad:
            convolved = _trim_linear_pad(convolved, linear_pad)
        return convolved

    def _convolve_group_via_patch_cache(gid: int) -> np.ndarray:
        """H.2: convolve one field-mode group entirely from cached
        per-skycell basis convolutions, never materializing the dense
        group template. Mirrors ``_convolve_crop``'s field-mode
        (mapping_grid) branch exactly (same basis/kc_step/oversample
        derivation, same post-convolution trim) -- only the convolution
        itself is replaced.
        """
        from hotpants.pure.kernel import calculate_kernel_basis
        from hotpants.pure.utils import downsample_image
        from syndiff_pipeline.common.grid_pairing import trim_padded_products
        from syndiff_pipeline.difference_imaging.stages.convolved_templates_patch_cache import (
            build_group_convolved_template,
        )
        from syndiff_pipeline.difference_imaging.stages.hotpants import (
            resolve_hotpants_oversample,
        )

        if mapping_grid is None:
            raise RuntimeError("use_patch_cache requires field mode with a mapping_grid")
        os_template_shape = tuple(mapping_grid.array_shape_os())
        factor = resolve_hotpants_oversample(
            sci_shape, os_template_shape, getattr(hp, "oversample", None)
        )
        if factor <= 1:
            raise RuntimeError(
                "use_patch_cache currently targets the oversample>1 field-mode "
                f"path only (resolved factor={factor})"
            )
        hr_rkernel = int(getattr(hp_config, "rkernel", 2)) * factor
        k_size = 2 * hr_rkernel + 1
        scaled_sigmas = [
            float(s) / (factor ** 2)
            for s in getattr(hp_config, "sigma_gauss", [0.7, 1.5, 3.0])
        ]
        basis_funcs = np.asarray(
            calculate_kernel_basis(
                (k_size, k_size), scaled_sigmas, list(getattr(hp_config, "deg_fixe", [6, 4, 2]))
            ),
            dtype=np.float64,
        )
        n_comp_ker = basis_funcs.shape[0]
        ker_order = int(getattr(hp_config, "ko"))
        fw_kernel_lr = 2 * int(getattr(hp_config, "rkernel", 2)) + 1
        kc_step_lr = int(getattr(hp_config, "kc_step", fw_kernel_lr) or fw_kernel_lr)
        kc_step_hr = kc_step_lr * factor

        from syndiff_pipeline.template_creation.processing.field_downsample import (
            _group_shifts_present,
        )

        shifts = _group_shifts_present(
            field_ctx.store_root, field_ctx.shifts_df, int(gid), present_only=True
        )
        crop_hr = crop_hr_from_template_roi_bounds(field_ctx.template_roi_bounds)
        conv_hr = build_group_convolved_template(
            field_ctx.store_root,
            shifts,
            group_id=int(gid),
            base_tess_shape=tuple(field_ctx.base_tess_shape),
            crop_hr=crop_hr,
            basis_funcs=basis_funcs,
            kernel_solution=kernel_solution,
            hw_kernel=hr_rkernel,
            kc_step=kc_step_hr,
            n_comp_ker=n_comp_ker,
            ker_order=ker_order,
            oversample=factor,
        )
        conv_lr = downsample_image(conv_hr, factor)
        return trim_padded_products(conv_lr, grid=mapping_grid)

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
        ws_prog, cli_prog = _progress_paths(convolved_ws_dir, diff_log_path)
        _init_progress_pair(ws_prog, cli_prog, groups_total=len(gids))
        for gid in gids:
            out_name = f"convolved_template_gid{gid}{PIPELINE_FITS_EXT}"
            out_path = os.path.join(convolved_ws_dir, out_name)
            existing = resolve_pipeline_fits_path(
                convolved_ws_dir, strip_fits_suffix(out_name)
            )
            entry = {"group_id": int(gid), "group_dx": float("nan"), "group_dy": float("nan")}
            if skip_existing and existing is not None:
                rows.append({**entry, "convolved_path": existing})
                _record_group_progress(ws_prog, cli_prog, built=False)
                continue
            if use_patch_cache:
                convolved = _convolve_group_via_patch_cache(int(gid))
            else:
                template_crop = loader(int(gid))
                convolved = _convolve_crop(template_crop)
            _write_image_fits(out_path, convolved, header=ref_header)
            rows.append({**entry, "convolved_path": out_path})
            _record_group_progress(ws_prog, cli_prog, built=True)
            log.info("Convolved field template group_id=%d -> %s", gid, out_path)
        _set_progress_phase_pair(ws_prog, cli_prog, "complete")
    else:
        entries = _unique_template_entries(template_paths)
        if not entries:
            raise RuntimeError("No syndiff templates found to convolve")
        ws_prog, cli_prog = _progress_paths(convolved_ws_dir, diff_log_path)
        _init_progress_pair(ws_prog, cli_prog, groups_total=len(entries))
        for entry in entries:
            tmpl_path = entry["template_path"]
            out_name = convolved_template_basename(tmpl_path)
            out_path = os.path.join(convolved_ws_dir, out_name)
            existing = resolve_pipeline_fits_path(
                convolved_ws_dir, strip_fits_suffix(out_name)
            )
            if skip_existing and existing is not None:
                rows.append({**entry, "convolved_path": existing})
                _record_group_progress(ws_prog, cli_prog, built=False)
                continue

            if linear_pad:
                from syndiff_pipeline.difference_imaging.stages.hotpants import (
                    _padded_crop_bounds,
                )

                template_crop = _load_template_cropped(
                    tmpl_path, _padded_crop_bounds(crop_bounds, linear_pad)
                )
            else:
                template_crop = _load_template_cropped(tmpl_path, crop_bounds)
            convolved = _convolve_crop(template_crop)
            _write_image_fits(out_path, convolved, header=ref_header)
            rows.append({**entry, "convolved_path": out_path})
            _record_group_progress(ws_prog, cli_prog, built=True)
            log.info(
                "Convolved template dx=%.3f dy=%.3f -> %s",
                entry["group_dx"],
                entry["group_dy"],
                out_path,
            )
        _set_progress_phase_pair(ws_prog, cli_prog, "complete")

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
