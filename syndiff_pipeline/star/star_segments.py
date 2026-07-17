"""PS1 skycell selection, band combination, and host-star SEP isolation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel
from syndiff_pipeline.star.context import (
    DEFAULT_MANIFEST_BASENAME,
    StarEventContext,
    full_ffi_to_crop_local,
    resolve_host_full_ffi_xy,
)
from syndiff_pipeline.star.identifiers import ResolvedHost
from syndiff_pipeline.star.mini_downsample import (
    build_field_star_shifts,
    convolve_star_only_cutout,
    downsample_star_arrays,
    write_star_mini_templates,
)
from syndiff_pipeline.star.plots import (
    write_mini_template_downsample_png,
    write_ps1_segment_overlay_png,
)
from syndiff_pipeline.template_creation.processing.band_utils import (
    build_sep_background_segmentation,
    catalog_segment_assignments,
    filled_segment_map,
    process_skycell_bands,
)
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    RELEVANT_WCS_KEYS,
    build_ps1_wcs,
    load_tess_wcs,
)
from syndiff_pipeline.template_creation.processing.downsample import (
    offsets_from_cluster_job_payload,
    precompute_shifts_for_offsets,
)
from syndiff_pipeline.template_creation.processing.ps1_process import project_gaia_to_skycell
from syndiff_pipeline.star.ps1_cache import load_skycell_bands_for_source

logger = logging.getLogger(__name__)

PS1_REMOVED_STARS_CSV = "ps1_removed_stars.csv"
_DEFAULT_SEP_SIGMA = 2.5
_DEFAULT_SEP_SIGMA_MASK = 50.0


@dataclass
class HostSegmentResult:
    target_seg_id: int
    star_only_image: np.ndarray
    background_suppressed: np.ndarray
    filled_seg_map: np.ndarray
    blend_flag: bool
    blended_catalog_rows: pd.DataFrame


def _load_skycell_csv(ctx: StarEventContext) -> pd.DataFrame:
    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    return pd.read_csv(ctx.mapping_csv, usecols=usecols)


def _reg_file_for_skycell(ctx: StarEventContext, skycell_name: str) -> str | None:
    rel = f"sector_{ctx.sector:04d}/camera_{ctx.camera}/ccd_{ctx.ccd}"
    pattern = str(Path(ctx.mapping_dir) / rel / f"*{skycell_name}*.fits*")
    matches = [
        path
        for path in sorted(glob(pattern))
        if "master_pixels2skycells" not in Path(path).name
    ]
    if not matches:
        return None
    return matches[0]


def _skycell_row_for_host(
    ctx: StarEventContext,
    host: ResolvedHost,
    skycell_id: int,
) -> dict:
    skycell_df = _load_skycell_csv(ctx)
    row = skycell_df.iloc[int(skycell_id)]
    skycell_name = str(row["NAME"])
    ps1_wcs, ps1_shape = build_ps1_wcs(row)
    px, py = world_ra_dec_to_pixel(ps1_wcs, host.ra, host.dec)
    w, h = ps1_shape
    host_in_cell = (
        np.isfinite(px)
        and np.isfinite(py)
        and 0 <= px < w
        and 0 <= py < h
    )
    if not host_in_cell:
        logger.warning(
            "Canonical skycell %s for gaia_source_id=%s does not contain host "
            "in PS1 WCS footprint (px=%.2f, py=%.2f); using master map assignment",
            skycell_name,
            host.gaia_source_id,
            px,
            py,
        )
    return {
        "skycell_name": skycell_name,
        "host_pixel_x": float(px),
        "host_pixel_y": float(py),
        "host_in_cell": bool(host_in_cell),
        "reg_file": _reg_file_for_skycell(ctx, skycell_name),
    }


def find_owning_skycell_for_host(
    ctx: StarEventContext,
    host: ResolvedHost,
) -> pd.DataFrame:
    """Return the single PS1 skycell that owns the host's exact TESS pixel."""
    empty_columns = [
        "skycell_name",
        "host_pixel_x",
        "host_pixel_y",
        "host_in_cell",
        "reg_file",
    ]
    host_xy = resolve_host_full_ffi_xy(ctx, host)
    x_pix = int(round(host_xy[0]))
    y_pix = int(round(host_xy[1]))

    with fits.open(ctx.master_mapping_fits) as hdul:
        hdu_idx = 1 if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else 0
        tess_data = hdul[hdu_idx].data

    if (
        y_pix < 0
        or x_pix < 0
        or y_pix >= tess_data.shape[0]
        or x_pix >= tess_data.shape[1]
    ):
        logger.warning(
            "Host gaia_source_id=%s at full-FFI pixel (%d, %d) is outside "
            "master_pixels2skycells shape %s",
            host.gaia_source_id,
            x_pix,
            y_pix,
            tess_data.shape,
        )
        return pd.DataFrame(columns=empty_columns)

    skycell_id = int(tess_data[y_pix, x_pix])
    if skycell_id < 0:
        logger.warning(
            "Host gaia_source_id=%s at full-FFI pixel (%d, %d) is unmapped "
            "in master_pixels2skycells",
            host.gaia_source_id,
            x_pix,
            y_pix,
        )
        return pd.DataFrame(columns=empty_columns)

    skycell_df = _load_skycell_csv(ctx)
    if skycell_id >= len(skycell_df):
        logger.warning(
            "Host gaia_source_id=%s maps to skycell id %d outside "
            "master_skycells_list length %d",
            host.gaia_source_id,
            skycell_id,
            len(skycell_df),
        )
        return pd.DataFrame(columns=empty_columns)

    return pd.DataFrame([_skycell_row_for_host(ctx, host, skycell_id)])


def load_and_combine_skycell(
    skycell_name: str,
    ctx: StarEventContext,
    *,
    ps1_source: str = "zarr_download",
    ps1_zarr_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WCS]:
    """Load r/i/z/y PS1 bands for one skycell and combine them."""
    bands, masks, weights, headers, headers_weight = load_skycell_bands_for_source(
        skycell_name,
        ctx,
        ps1_source=ps1_source,
        zarr_path_override=ps1_zarr_path,
    )

    combined_image, combined_mask, combined_uncert = process_skycell_bands(
        bands,
        masks,
        weights,
        headers,
        headers_weight,
    )
    header_str = next(iter(headers.values()))
    wcs = WCS(fits.Header.fromstring(header_str))
    return combined_image, combined_mask, combined_uncert, wcs


def _segment_overlay_cutout_bounds(
    image_shape: tuple[int, int],
    host_pixel_xy: tuple[float, float],
    target_segment_mask: np.ndarray | None,
    *,
    margin_px: int = 200,
    min_size: int = 400,
) -> tuple[int, int, int, int]:
    h, w = image_shape
    hx, hy = host_pixel_xy

    if target_segment_mask is not None and np.any(target_segment_mask):
        ys, xs = np.where(target_segment_mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0 = max(0, x0 - margin_px)
        y0 = max(0, y0 - margin_px)
        x1 = min(w, x1 + margin_px)
        y1 = min(h, y1 + margin_px)
    else:
        x0 = max(0, int(round(hx)) - min_size // 2)
        y0 = max(0, int(round(hy)) - min_size // 2)
        x1 = min(w, x0 + min_size)
        y1 = min(h, y0 + min_size)

    width = x1 - x0
    height = y1 - y0
    if width < min_size:
        pad = (min_size - width) // 2
        x0 = max(0, x0 - pad)
        x1 = min(w, max(x1, x0 + min_size))
        x0 = max(0, x1 - min_size)
    if height < min_size:
        pad = (min_size - height) // 2
        y0 = max(0, y0 - pad)
        y1 = min(h, max(y1, y0 + min_size))
        y0 = max(0, y1 - min_size)

    return x0, y0, x1, y1


def _load_gaia_catalog(ctx: StarEventContext) -> pd.DataFrame:
    return pd.read_csv(ctx.gaia_catalog_path)


def _host_already_removed(ctx: StarEventContext, host: ResolvedHost) -> bool:
    removed_path = Path(ctx.event_dir) / PS1_REMOVED_STARS_CSV
    if not removed_path.is_file():
        return False
    removed = pd.read_csv(removed_path)
    if "source_id" not in removed.columns:
        return False
    source_ids = pd.to_numeric(removed["source_id"], errors="coerce").astype("Int64")
    return bool((source_ids == int(host.gaia_source_id)).any())


def isolate_host_segment(
    combined_image: np.ndarray,
    combined_uncert: np.ndarray,
    combined_mask: np.ndarray,
    gaia_catalog_pixels: pd.DataFrame,
    host: ResolvedHost,
    host_pixel_xy: tuple[float, float],
    *,
    sigma: float = _DEFAULT_SEP_SIGMA,
    sigma_mask: float = _DEFAULT_SEP_SIGMA_MASK,
    close_bright_mask: bool = False,
) -> HostSegmentResult:
    """Isolate the SEP segment containing the host star."""
    data = np.array(combined_image, dtype=np.float32, copy=True)
    uncert = np.asarray(combined_uncert, dtype=np.float32)
    mask = np.asarray(combined_mask)

    sep_result = build_sep_background_segmentation(
        data,
        uncert,
        sigma=sigma,
        sigma_mask=sigma_mask,
        close_bright_mask=close_bright_mask,
    )
    segmap = sep_result.segmap
    mask_bright_stars = sep_result.mask_bright_stars

    background_suppressed = data.copy()
    background_suppressed[np.logical_and(segmap == 0, ~mask_bright_stars)] = 0

    filled_seg_map = filled_segment_map(segmap)
    has_id = segmap > 0
    filled_seg_map_cat = np.where(has_id | mask_bright_stars, filled_seg_map, 0)

    cat_df = catalog_segment_assignments(
        gaia_catalog_pixels,
        filled_seg_map,
        mask_bright_stars,
        segmap=segmap,
    )

    px = int(np.clip(np.round(host_pixel_xy[0]), 0, segmap.shape[1] - 1))
    py = int(np.clip(np.round(host_pixel_xy[1]), 0, segmap.shape[0] - 1))
    target_seg_id = int(filled_seg_map_cat[py, px])

    if target_seg_id == 0:
        logger.warning(
            "No SEP segment found for host gaia_source_id=%s at skycell pixel (%s, %s)",
            host.gaia_source_id,
            host_pixel_xy[0],
            host_pixel_xy[1],
        )
        return HostSegmentResult(
            target_seg_id=0,
            star_only_image=np.zeros_like(background_suppressed, dtype=np.float32),
            background_suppressed=background_suppressed,
            filled_seg_map=filled_seg_map,
            blend_flag=False,
            blended_catalog_rows=pd.DataFrame(),
        )

    segment_mask = filled_seg_map == target_seg_id
    star_only_image = background_suppressed * segment_mask.astype(np.float32)

    host_rows = cat_df
    if "source_id" in cat_df.columns:
        host_rows = cat_df[cat_df["source_id"] == host.gaia_source_id]
    if len(host_rows) == 0:
        host_seg_ids = {target_seg_id}
    else:
        host_seg_ids = set(host_rows["seg_id_cat"].astype(int).tolist())

    blended = cat_df[
        cat_df["seg_id_cat"].isin(host_seg_ids)
        & (cat_df.get("source_id", pd.Series(dtype="Int64")) != host.gaia_source_id)
    ].copy()
    blend_flag = len(blended) > 0

    return HostSegmentResult(
        target_seg_id=target_seg_id,
        star_only_image=star_only_image,
        background_suppressed=background_suppressed,
        filled_seg_map=filled_seg_map,
        blend_flag=blend_flag,
        blended_catalog_rows=blended,
    )


def _full_ffi_mapping_shape(ctx: StarEventContext) -> tuple[int, int]:
    with fits.open(ctx.master_mapping_fits) as hdul:
        hdu_idx = 1 if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else 0
        data = hdul[hdu_idx].data
    return int(data.shape[0]), int(data.shape[1])


def _mini_roi_full_ffi(
    ctx: StarEventContext,
    host_xy_full: tuple[float, float],
    cutout_size: int,
) -> tuple[int, int, int, int]:
    """Mini-template ROI in full-FFI pixel coordinates (for downsample binning)."""
    x_ref, y_ref = host_xy_full
    half = cutout_size // 2
    x_min_crop = int(ctx.crop_bounds["x_min"])
    y_min_crop = int(ctx.crop_bounds["y_min"])
    x_max_crop = int(ctx.crop_bounds["x_max"])
    y_max_crop = int(ctx.crop_bounds["y_max"])

    x0 = max(x_min_crop, int(round(x_ref)) - half)
    y0 = max(y_min_crop, int(round(y_ref)) - half)
    x1 = min(x_max_crop, x0 + cutout_size)
    y1 = min(y_max_crop, y0 + cutout_size)
    x0 = max(x_min_crop, x1 - cutout_size)
    y0 = max(y_min_crop, y1 - cutout_size)
    return x0, y0, x1, y1


def _mini_roi_crop_local(
    ctx: StarEventContext,
    host_xy_full: tuple[float, float],
    cutout_size: int,
) -> tuple[int, int, int, int]:
    x_local, y_local = full_ffi_to_crop_local(ctx, *host_xy_full)
    shape = ctx.crop_bounds.get("shape")
    if shape is not None:
        crop_h, crop_w = int(shape[0]), int(shape[1])
    else:
        crop_w = int(ctx.crop_bounds["x_max"] - ctx.crop_bounds["x_min"])
        crop_h = int(ctx.crop_bounds["y_max"] - ctx.crop_bounds["y_min"])

    half = cutout_size // 2
    x0 = max(0, int(round(x_local)) - half)
    y0 = max(0, int(round(y_local)) - half)
    x1 = min(crop_w, x0 + cutout_size)
    y1 = min(crop_h, y0 + cutout_size)
    x0 = max(0, x1 - cutout_size)
    y0 = max(0, y1 - cutout_size)
    return x0, y0, x1, y1


def _embed_convolved_cutout(
    full_shape: tuple[int, int],
    cutout: np.ndarray,
    origin: tuple[int, int],
) -> np.ndarray:
    canvas = np.zeros(full_shape, dtype=np.float32)
    y0, x0 = origin
    y1 = y0 + cutout.shape[0]
    x1 = x0 + cutout.shape[1]
    canvas[y0:y1, x0:x1] = cutout
    return canvas


def isolate_and_write_mini_templates(
    ctx: StarEventContext,
    host: ResolvedHost,
    *,
    cutout_size: int = 96,
    psf_sigma: float = 60.0,
    ps1_source: str = "zarr_download",
    ps1_zarr_path: str | None = None,
    output_dir: str,
    write_debug_plots: bool = True,
) -> dict:
    """Run skycell lookup, SEP isolation, convolution, and mini-template write."""
    if _host_already_removed(ctx, host):
        logger.info(
            "Host gaia_source_id=%s already in %s; skipping isolation",
            host.gaia_source_id,
            PS1_REMOVED_STARS_CSV,
        )
        return {
            "already_removed": True,
            "host_gaia_source_id": int(host.gaia_source_id),
            "skycells": {},
            "mini_template_paths": [],
            "plot_paths": [],
        }

    host_xy_full = resolve_host_full_ffi_xy(ctx, host)
    skycell_table = find_owning_skycell_for_host(ctx, host)
    if skycell_table.empty:
        logger.warning(
            "Host gaia_source_id=%s is outside PS1 skycell coverage",
            host.gaia_source_id,
        )
        return {
            "already_removed": False,
            "host_gaia_source_id": int(host.gaia_source_id),
            "skycells": {},
            "mini_template_paths": [],
            "plot_paths": [],
            "error": "host_outside_ps1_skycell_coverage",
        }

    gaia_catalog = _load_gaia_catalog(ctx)
    plots_dir = Path(output_dir).parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    skycell_summaries: dict[str, dict] = {}
    convolved_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    plot_paths: list[str] = []

    for row in skycell_table.itertuples(index=False):
        skycell_name = row.skycell_name
        host_px = float(row.host_pixel_x)
        host_py = float(row.host_pixel_y)
        try:
            combined_image, combined_mask, combined_uncert, wcs = load_and_combine_skycell(
                skycell_name,
                ctx,
                ps1_source=ps1_source,
                ps1_zarr_path=ps1_zarr_path,
            )
        except Exception as exc:
            logger.error("Failed to load skycell %s: %s", skycell_name, exc)
            skycell_summaries[skycell_name] = {
                "target_seg_id": 0,
                "blend_flag": False,
                "host_in_cell": bool(row.host_in_cell),
                "error": str(exc),
            }
            continue

        gaia_pixels = project_gaia_to_skycell(gaia_catalog, wcs, combined_image.shape)
        segment = isolate_host_segment(
            combined_image,
            combined_uncert,
            combined_mask,
            gaia_pixels,
            host,
            (host_px, host_py),
        )

        target_mask = segment.filled_seg_map == segment.target_seg_id
        cutout_bounds = _segment_overlay_cutout_bounds(
            combined_image.shape,
            (host_px, host_py),
            target_mask if segment.target_seg_id > 0 else None,
        )
        plot_path = None
        if write_debug_plots:
            plot_path = write_ps1_segment_overlay_png(
                plots_dir / f"ps1_segment_{skycell_name}.png",
                original_image=combined_image,
                data_wo_bkg_sat=segment.star_only_image,
                host_pixel_xy=(host_px, host_py),
                skycell_name=skycell_name,
                host=host,
                blend_flag=segment.blend_flag,
                cutout_bounds=cutout_bounds,
                target_seg_id=segment.target_seg_id,
            )
            plot_paths.append(plot_path)

        skycell_summaries[skycell_name] = {
            "target_seg_id": int(segment.target_seg_id),
            "blend_flag": bool(segment.blend_flag),
            "host_in_cell": bool(row.host_in_cell),
            "blended_source_ids": segment.blended_catalog_rows.get(
                "source_id", pd.Series(dtype="Int64")
            ).tolist(),
            "segment_plot": plot_path,
        }

        if segment.target_seg_id == 0:
            continue

        convolved_cutout, origin = convolve_star_only_cutout(
            segment.star_only_image,
            psf_sigma=psf_sigma,
        )
        if convolved_cutout.size == 0:
            continue
        full_convolved = _embed_convolved_cutout(
            segment.star_only_image.shape,
            convolved_cutout,
            origin,
        )
        mask = np.zeros(segment.star_only_image.shape, dtype=np.uint32)
        convolved_arrays[skycell_name] = (full_convolved, mask)

    if not convolved_arrays:
        return {
            "already_removed": False,
            "host_gaia_source_id": int(host.gaia_source_id),
            "skycells": skycell_summaries,
            "mini_template_paths": [],
            "plot_paths": plot_paths,
            "error": "no_valid_segments",
        }

    involved_names = list(convolved_arrays.keys())
    field_mode = str(ctx.cluster_job.get("geometry_mode") or "linear").lower() == "field"
    group_to_index: dict[int, int] | None = None
    if field_mode:
        # Use the new field mapping: per-skycell integer shifts per group_id from
        # template_group_shifts drive the SAME star-only binning, deduped to the
        # distinct local shift signatures over the star's few skycells.
        gsf = pd.read_parquet(Path(ctx.event_dir) / "template_group_shifts.parquet")
        frames = pd.read_csv(Path(ctx.event_dir) / DEFAULT_MANIFEST_BASENAME)
        group_ids = sorted(
            {int(g) for g in frames["group_id"].tolist() if pd.notna(g) and int(g) >= 0}
        )
        offsets, shifts_dict, group_to_index = build_field_star_shifts(
            gsf, group_ids, involved_names
        )
    else:
        offsets = offsets_from_cluster_job_payload(ctx.cluster_job)
        tess_wcs, _ = load_tess_wcs(Path(ctx.master_mapping_fits))
        skycell_df = _load_skycell_csv(ctx)
        skycell_df = skycell_df[skycell_df["NAME"].isin(involved_names)].reset_index(
            drop=True
        )
        shifts_dict = precompute_shifts_for_offsets(tess_wcs, skycell_df, offsets)

    reg_files = []
    skycell_names = []
    for name in involved_names:
        reg_file = _reg_file_for_skycell(ctx, name)
        if reg_file is None:
            logger.warning("Missing registration file for skycell %s", name)
            continue
        reg_files.append(reg_file)
        skycell_names.append(name)

    if not reg_files:
        return {
            "already_removed": False,
            "host_gaia_source_id": int(host.gaia_source_id),
            "skycells": skycell_summaries,
            "mini_template_paths": [],
            "plot_paths": plot_paths,
            "error": "missing_reg_files",
        }

    roi_bounds_full = _mini_roi_full_ffi(ctx, host_xy_full, cutout_size)
    base_tess_shape = _full_ffi_mapping_shape(ctx)

    arrays_by_offset = downsample_star_arrays(
        arrays=convolved_arrays,
        reg_files=reg_files,
        skycell_names=skycell_names,
        offsets=offsets,
        shifts_dict=shifts_dict,
        base_tess_shape=base_tess_shape,
        roi_bounds=roi_bounds_full,
    )

    x_min_crop = int(ctx.crop_bounds["x_min"])
    y_min_crop = int(ctx.crop_bounds["y_min"])
    roi_origin = (roi_bounds_full[0] - x_min_crop, roi_bounds_full[1] - y_min_crop)

    host_metadata = {
        "gaia_source_id": int(host.gaia_source_id),
        "tic_id": host.tic_id,
        "label": host.label,
        "sector": ctx.sector,
        "camera": ctx.camera,
        "ccd": ctx.ccd,
    }
    mini_paths = write_star_mini_templates(
        output_dir,
        arrays_by_offset,
        offsets=offsets,
        roi_origin=roi_origin,
        host_identifier_metadata=host_metadata,
    )

    # Field mode: persist group_id -> mini-template path (via the deduped
    # signature index) so the star diff runner can look up a frame's mini
    # template by its group_id instead of a linear (dx, dy) offset.
    field_group_to_template: dict[int, str] | None = None
    if field_mode and group_to_index is not None:
        import json as _json

        field_group_to_template = {
            int(gid): mini_paths[idx]
            for gid, idx in group_to_index.items()
            if 0 <= idx < len(mini_paths)
        }
        (Path(output_dir) / "star_mini_template_index.json").write_text(
            _json.dumps(
                {
                    "schema_version": 1,
                    "geometry_mode": "field",
                    "group_to_template": {
                        str(k): v for k, v in field_group_to_template.items()
                    },
                }
            )
            + "\n"
        )

    x_local, y_local = full_ffi_to_crop_local(ctx, *host_xy_full)
    roi_bounds_crop = (
        roi_bounds_full[0] - x_min_crop,
        roi_bounds_full[1] - y_min_crop,
        roi_bounds_full[2] - x_min_crop,
        roi_bounds_full[3] - y_min_crop,
    )
    host_local_xy = (x_local - roi_bounds_crop[0], y_local - roi_bounds_crop[1])
    dx, dy = float(offsets[0, 0]), float(offsets[0, 1])
    if write_debug_plots:
        downsample_plot = write_mini_template_downsample_png(
            plots_dir / f"mini_template_downsampled_dx{dx:.3f}_dy{dy:.3f}.png",
            mini_flux_sum=arrays_by_offset[0, 0],
            host_local_xy=host_local_xy,
            dx=dx,
            dy=dy,
            host=host,
            roi_bounds=roi_bounds_crop,
        )
        plot_paths.append(downsample_plot)

    return {
        "already_removed": False,
        "host_gaia_source_id": int(host.gaia_source_id),
        "skycells": skycell_summaries,
        "mini_template_paths": mini_paths,
        "field_group_to_template": field_group_to_template,
        "plot_paths": plot_paths,
        "roi_bounds": roi_bounds_crop,
        "debug_offset": (dx, dy),
    }
