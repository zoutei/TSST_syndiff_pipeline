"""Per-frame star-only difference stamps for host light curves."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.difference_imaging.stages.hotpants import _write_image_fits
from syndiff_pipeline.difference_imaging.stages.kernel import (
    convolve_template_with_kernel_solution,
    load_frame_kernel,
)
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    resolve_pipeline_fits_path,
    workspace_frame_stem,
    workspace_label_from_dir,
)
from syndiff_pipeline.difference_imaging.support.manifest import row_ffi_product_id_series
from syndiff_pipeline.difference_imaging.support.paths import DEFAULT_MANIFEST_BASENAME
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    template_offsets_for_ffi,
)
from syndiff_pipeline.star.context import StarEventContext


def load_frame_kernel_for_diff(
    kernels_dir: str,
    product_id: str,
) -> tuple[np.ndarray, Any]:
    """Load a persisted per-frame kernel and rebuild the Hotpants config for convolution."""
    from hotpants import HotpantsConfig

    kernel_solution, hp_config_fields = load_frame_kernel(kernels_dir, product_id)
    hp_config = HotpantsConfig(**hp_config_fields)
    return kernel_solution, hp_config


def read_window(
    fits_path: str,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Memmap-read a 2D window from a pipeline or raw FFI FITS."""
    with wcs_grouping.open_fits_memmap(fits_path) as hdul:
        data = hdul[0].data
        if data is None or getattr(data, "ndim", 0) != 2:
            data = hdul[1].data
        return np.asarray(data[y0:y1, x0:x1], dtype=np.float64)


def _read_science_window(
    fits_path: str,
    crop_bounds: dict,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Read a crop-local window from a full-chip science FFI (HDU 1)."""
    ox = int(crop_bounds["x_min"])
    oy = int(crop_bounds["y_min"])
    with wcs_grouping.open_fits_memmap(fits_path) as hdul:
        data = hdul[1].data
        return np.asarray(
            data[oy + y0 : oy + y1, ox + x0 : ox + x1],
            dtype=np.float64,
        )


def place_mini_template_in_window(
    mini_flux_sum: np.ndarray,
    *,
    mini_xmin: int,
    mini_ymin: int,
    window_x0: int,
    window_y0: int,
    window_shape: tuple[int, int],
) -> np.ndarray:
    """
    Embed a mini star-template ``FLUX_SUM`` plane into a zero array of *window_shape*.

    Coordinates are crop-local. Partial overlap is clipped; zero overlap yields zeros.
    """
    mini = np.asarray(mini_flux_sum, dtype=np.float64)
    out = np.zeros(window_shape, dtype=np.float64)
    if mini.size == 0:
        return out

    mini_h, mini_w = mini.shape
    win_h, win_w = int(window_shape[0]), int(window_shape[1])

    overlap_x0 = max(window_x0, mini_xmin)
    overlap_y0 = max(window_y0, mini_ymin)
    overlap_x1 = min(window_x0 + win_w, mini_xmin + mini_w)
    overlap_y1 = min(window_y0 + win_h, mini_ymin + mini_h)
    if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
        return out

    out_y0 = overlap_y0 - window_y0
    out_x0 = overlap_x0 - window_x0
    mini_y0 = overlap_y0 - mini_ymin
    mini_x0 = overlap_x0 - mini_xmin
    out_h = overlap_y1 - overlap_y0
    out_w = overlap_x1 - overlap_x0
    out[out_y0 : out_y0 + out_h, out_x0 : out_x0 + out_w] = mini[
        mini_y0 : mini_y0 + out_h, mini_x0 : mini_x0 + out_w
    ]
    return out


def load_mini_template_flux_sum(mini_template_fits_path: str) -> tuple[np.ndarray, int, int]:
    """Return ``(flux_sum, xmin, ymin)`` from a star mini-template FITS."""
    with fits.open(mini_template_fits_path, memmap=True) as hdul:
        header = hdul[0].header
        xmin = int(header["XMIN"])
        ymin = int(header["YMIN"])
        flux_hdu = hdul["FLUX_SUM"] if "FLUX_SUM" in hdul else hdul[1]
        flux_sum = np.asarray(flux_hdu.data, dtype=np.float64)
    return flux_sum, xmin, ymin


def compute_star_only_stamp(
    *,
    ffi_window: np.ndarray,
    conv_temp_window: np.ndarray,
    background_window: np.ndarray,
    mini_star_template_window: np.ndarray,
    kernel_solution: np.ndarray,
    hp_config,
    convolve_shape: tuple[int, int],
    stamp_offset_in_conv: tuple[int, int],
) -> np.ndarray:
    """
    Compute ``ffi - (conv_temp - S_conv) - phot_bkg`` on the final stamp window.

    *background_window* must be the photutils map (``ks_b`` / ``ks_b_s``), not Hotpants
    ``hp_b``.

    *mini_star_template_window* must already be embedded in an array whose shape is
    *convolve_shape*, large enough to hold the stamp plus the Hotpants kernel spatial
    extent (``~2 * rkernel + 1`` pixels per side). Convolution runs on that larger
    canvas; the result is cropped with origin *stamp_offset_in_conv* ``(y0, x0)`` so it
    aligns with *ffi_window* before subtraction. Edge artifacts from truncating the
    kernel therefore fall outside the stamp.
    """
    ffi = np.asarray(ffi_window, dtype=np.float64)
    conv_temp = np.asarray(conv_temp_window, dtype=np.float64)
    background = np.asarray(background_window, dtype=np.float64)
    mini_embedded = np.asarray(mini_star_template_window, dtype=np.float64)

    if ffi.shape != conv_temp.shape or ffi.shape != background.shape:
        raise ValueError(
            f"stamp windows must match: ffi {ffi.shape}, conv_temp {conv_temp.shape}, "
            f"background {background.shape}"
        )
    if mini_embedded.shape != convolve_shape:
        raise ValueError(
            f"mini_star_template_window shape {mini_embedded.shape} != "
            f"convolve_shape {convolve_shape}"
        )

    stamp_h, stamp_w = ffi.shape
    conv_h, conv_w = mini_embedded.shape
    crop_y0, crop_x0 = int(stamp_offset_in_conv[0]), int(stamp_offset_in_conv[1])
    if crop_y0 < 0 or crop_x0 < 0 or crop_y0 + stamp_h > conv_h or crop_x0 + stamp_w > conv_w:
        raise ValueError(
            f"stamp_offset_in_conv {(crop_y0, crop_x0)} invalid for "
            f"convolve_shape {convolve_shape} and stamp {ffi.shape}"
        )

    s_conv_full = convolve_template_with_kernel_solution(
        mini_embedded, kernel_solution, hp_config
    )
    s_conv = s_conv_full[crop_y0 : crop_y0 + stamp_h, crop_x0 : crop_x0 + stamp_w]
    return ffi - (conv_temp - s_conv) - background


def _centered_window_bounds(
    host_x: float,
    host_y: float,
    size: int,
    *,
    max_x: int,
    max_y: int,
) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` for a square window centered on the host."""
    half = size // 2
    cx = int(round(host_x))
    cy = int(round(host_y))
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > max_x:
        shift = x1 - max_x
        x0 = max(0, x0 - shift)
        x1 = max_x
    if y1 > max_y:
        shift = y1 - max_y
        y0 = max(0, y0 - shift)
        y1 = max_y
    return x0, y0, x1, y1


def _expanded_window_bounds(
    stamp_bounds: tuple[int, int, int, int],
    *,
    margin_px: int,
    max_x: int,
    max_y: int,
) -> tuple[int, int, int, int]:
    """Expand a stamp window by *margin_px* on each side, clipped to the crop."""
    x0, y0, x1, y1 = stamp_bounds
    return (
        max(0, x0 - margin_px),
        max(0, y0 - margin_px),
        min(max_x, x1 + margin_px),
        min(max_y, y1 + margin_px),
    )


def _manifest_path_for_ctx(ctx: StarEventContext) -> str:
    return os.path.join(ctx.event_dir, DEFAULT_MANIFEST_BASENAME)


def _ffi_path_for_product_id(manifest: pd.DataFrame, product_id: str) -> str:
    pids = row_ffi_product_id_series(manifest)
    matches = manifest.loc[pids == product_id]
    if matches.empty:
        raise ValueError(f"No manifest row for product_id={product_id!r}")
    row = matches.iloc[0]
    for col in ("path", "filename"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    raise ValueError(f"Manifest row for {product_id!r} has no path/filename")


def _resolve_frame_fits_path(directory: str, product_id: str) -> str:
    if not directory:
        raise FileNotFoundError(f"workspace directory unset for {product_id}")
    label = workspace_label_from_dir(directory)
    stem = workspace_frame_stem(product_id, label)
    path = resolve_pipeline_fits_path(directory, stem)
    if path is None:
        raise FileNotFoundError(
            f"Missing {stem} under {directory}"
        )
    return path


def compute_star_only_stamp_for_frame(
    *,
    ctx: StarEventContext,
    product_id: str,
    host_local_xy: tuple[float, float],
    mini_template_fits_paths: dict[tuple[float, float], str],
    stamp_size: int = 24,
    kernel_margin_px: int = 470,
    science_fits_path: Optional[str] = None,
    field_group_to_template: Optional[dict[int, str]] = None,
) -> tuple[np.ndarray, dict]:
    """
    Build one star-only diff stamp for a frame/host.

    Science arrays are read from the raw FFI path in ``syndiff_ffi_frames.csv`` unless
    *science_fits_path* is supplied (useful for tests or pre-cropped science workspaces).
    Convolved template, photutils background (``ks_b`` / ``ks_b_s``), and kernel come
    from the baseline workspace paths in *ctx*. Hotpants ``hp_b`` is not used.
    """
    manifest_path = _manifest_path_for_ctx(ctx)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    ffi_path = science_fits_path or _ffi_path_for_product_id(manifest, product_id)

    if field_group_to_template is not None:
        # Field mode: look up the mini template by the frame's group_id.
        from syndiff_pipeline.difference_imaging.support.template_resolution import (
            group_id_for_ffi,
        )

        gid = group_id_for_ffi(manifest, ffi_path)
        mini_path = field_group_to_template.get(int(gid))
        if mini_path is None:
            raise FileNotFoundError(
                f"No field mini template for group_id={gid}; "
                f"available groups: {sorted(field_group_to_template)}"
            )
    else:
        group_dx, group_dy = template_offsets_for_ffi(manifest, ffi_path)
        offset_key = (float(group_dx), float(group_dy))
        mini_path = None
        for key, path in mini_template_fits_paths.items():
            if abs(float(key[0]) - offset_key[0]) <= 1e-3 and abs(float(key[1]) - offset_key[1]) <= 1e-3:
                mini_path = path
                break
        if mini_path is None:
            raise FileNotFoundError(
                f"No mini template for offset dx={group_dx} dy={group_dy}; "
                f"available keys: {sorted(mini_template_fits_paths)}"
            )

    kernel_solution, hp_config = load_frame_kernel_for_diff(
        ctx.baseline_kernels_dir,
        product_id,
    )

    crop_h, crop_w = int(ctx.crop_bounds["shape"][0]), int(ctx.crop_bounds["shape"][1])
    host_x, host_y = float(host_local_xy[0]), float(host_local_xy[1])
    stamp_x0, stamp_y0, stamp_x1, stamp_y1 = _centered_window_bounds(
        host_x,
        host_y,
        stamp_size,
        max_x=crop_w,
        max_y=crop_h,
    )
    conv_x0, conv_y0, conv_x1, conv_y1 = _expanded_window_bounds(
        (stamp_x0, stamp_y0, stamp_x1, stamp_y1),
        margin_px=kernel_margin_px,
        max_x=crop_w,
        max_y=crop_h,
    )
    conv_shape = (conv_y1 - conv_y0, conv_x1 - conv_x0)

    conv_path = _resolve_frame_fits_path(ctx.baseline_convolved_dir, product_id)
    bkg_path = _resolve_frame_fits_path(ctx.baseline_phot_bkg_dir, product_id)

    conv_temp_window = read_window(
        conv_path, y0=stamp_y0, y1=stamp_y1, x0=stamp_x0, x1=stamp_x1
    )
    background_window = read_window(
        bkg_path, y0=stamp_y0, y1=stamp_y1, x0=stamp_x0, x1=stamp_x1
    )
    if science_fits_path is not None and _is_crop_sized_science(science_fits_path, ctx.crop_bounds):
        ffi_window = read_window(
            science_fits_path,
            y0=stamp_y0,
            y1=stamp_y1,
            x0=stamp_x0,
            x1=stamp_x1,
        )
    else:
        ffi_window = _read_science_window(
            ffi_path,
            ctx.crop_bounds,
            y0=stamp_y0,
            y1=stamp_y1,
            x0=stamp_x0,
            x1=stamp_x1,
        )

    mini_flux_sum, mini_xmin, mini_ymin = load_mini_template_flux_sum(mini_path)
    mini_star_template_window = place_mini_template_in_window(
        mini_flux_sum,
        mini_xmin=mini_xmin,
        mini_ymin=mini_ymin,
        window_x0=conv_x0,
        window_y0=conv_y0,
        window_shape=conv_shape,
    )

    stamp = compute_star_only_stamp(
        ffi_window=ffi_window,
        conv_temp_window=conv_temp_window,
        background_window=background_window,
        mini_star_template_window=mini_star_template_window,
        kernel_solution=kernel_solution,
        hp_config=hp_config,
        convolve_shape=conv_shape,
        stamp_offset_in_conv=(stamp_y0 - conv_y0, stamp_x0 - conv_x0),
    )

    metadata = {
        "window_x0": stamp_x0,
        "window_y0": stamp_y0,
        "host_local_x": host_x,
        "host_local_y": host_y,
        "product_id": product_id,
        "group_dx": group_dx,
        "group_dy": group_dy,
        "stamp_shape": stamp.shape,
        "conv_window_x0": conv_x0,
        "conv_window_y0": conv_y0,
    }
    return stamp, metadata


def _is_crop_sized_science(fits_path: str, crop_bounds: dict) -> bool:
    with wcs_grouping.open_fits_memmap(fits_path) as hdul:
        data = hdul[0].data
        if data is None or getattr(data, "ndim", 0) != 2:
            return False
        return tuple(data.shape) == tuple(crop_bounds["shape"])


def write_star_diff_stamp(
    path: str,
    stamp: np.ndarray,
    *,
    window_origin: tuple[int, int],
    host_local_xy: tuple[float, float],
    header: Optional[fits.Header] = None,
) -> str:
    """Write a star-only diff stamp FITS with crop-local placement metadata."""
    hdr = fits.Header(header) if header is not None else fits.Header()
    wx, wy = int(window_origin[0]), int(window_origin[1])
    host_x, host_y = float(host_local_xy[0]), float(host_local_xy[1])
    hdr["XMIN"] = (wx, "Stamp xmin in crop-local pixels")
    hdr["YMIN"] = (wy, "Stamp ymin in crop-local pixels")
    hdr["HOSTX"] = (host_x - wx, "Host x within stamp (crop-local minus XMIN)")
    hdr["HOSTY"] = (host_y - wy, "Host y within stamp (crop-local minus YMIN)")
    out_path = str(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _write_image_fits(out_path, stamp, header=hdr)
    return out_path
