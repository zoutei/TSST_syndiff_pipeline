"""Star-only mini-template convolution and downsampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.fits_io import write_hdul_fits
from syndiff_pipeline.difference_imaging.support.ffi_naming import PIPELINE_FITS_EXT
from syndiff_pipeline.template_creation.processing.convolution_utils import (
    apply_gaussian_convolution,
)
from syndiff_pipeline.template_creation.processing.downsample import (
    combine_sparse_downsample_results,
    process_skycell_batch_from_arrays,
)


def convolve_star_only_cutout(
    star_only_image: np.ndarray,
    *,
    psf_sigma: float,
    margin_px: int = 470,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Convolve a star-only skycell image on a tight window around nonzero flux.

    Returns the convolved cutout and its ``(y0, x0)`` origin in the full skycell
    canvas.
    """
    image = np.asarray(star_only_image, dtype=np.float32)
    nonzero = np.argwhere(image != 0)
    if len(nonzero) == 0:
        return np.zeros((0, 0), dtype=np.float32), (0, 0)

    y0, x0 = nonzero.min(axis=0)
    y1, x1 = nonzero.max(axis=0) + 1
    height, width = image.shape
    y0 = max(0, int(y0) - margin_px)
    x0 = max(0, int(x0) - margin_px)
    y1 = min(height, int(y1) + margin_px)
    x1 = min(width, int(x1) + margin_px)

    cutout = image[y0:y1, x0:x1]
    convolved = apply_gaussian_convolution(
        cutout,
        sigma=psf_sigma,
        radius=margin_px,
        cval=0.0,
    )
    return np.asarray(convolved, dtype=np.float32), (y0, x0)


def build_field_star_shifts(
    group_shifts_df: pd.DataFrame,
    group_ids: list[int],
    involved_skycells: list[str],
) -> tuple[np.ndarray, dict[tuple[float, float], pd.DataFrame], dict[int, int]]:
    """Build star-binning ``(offsets, shifts_dict, group_to_index)`` from the
    field ``template_group_shifts`` (columns ``group_id, skycell, sx_int, sy_int``).

    "Use the new mapping": instead of the linear target-anchored offset model,
    each group's per-skycell integer shifts drive the same star-only binning.
    Because the star ROI covers only a few skycells, groups whose shifts over
    ``involved_skycells`` are identical collapse to one mini-template — so the
    star still produces a handful of templates, not one per global group.

    Returns
    -------
    offsets : (n_sig, 2) float pseudo-keys (index encoded in column 0)
    shifts_dict : {(idx, 0.0) -> DataFrame(NAME, shift_x, shift_y)}
    group_to_index : {group_id -> offset row index}
    """
    involved = [str(s) for s in involved_skycells]
    gs = group_shifts_df[group_shifts_df["skycell"].astype(str).isin(involved)]

    sig_to_index: dict[tuple, int] = {}
    offset_rows: list[list[float]] = []
    shift_frames: list[pd.DataFrame] = []
    group_to_index: dict[int, int] = {}

    for gid in group_ids:
        sub = gs[gs["group_id"] == int(gid)]
        shift_map = {
            str(r.skycell): (int(r.sx_int), int(r.sy_int))
            for r in sub.itertuples(index=False)
        }
        sig = tuple((s, shift_map.get(s, (0, 0))) for s in involved)
        if sig not in sig_to_index:
            idx = len(offset_rows)
            sig_to_index[sig] = idx
            offset_rows.append([float(idx), 0.0])
            shift_frames.append(
                pd.DataFrame(
                    {
                        "NAME": involved,
                        "shift_x": [shift_map.get(s, (0, 0))[0] for s in involved],
                        "shift_y": [shift_map.get(s, (0, 0))[1] for s in involved],
                    }
                )
            )
        group_to_index[int(gid)] = sig_to_index[sig]

    offsets = (
        np.array(offset_rows, dtype=float)
        if offset_rows
        else np.zeros((0, 2), dtype=float)
    )
    shifts_dict = {
        (float(offsets[i, 0]), float(offsets[i, 1])): shift_frames[i]
        for i in range(len(offset_rows))
    }
    return offsets, shifts_dict, group_to_index


def downsample_star_arrays(
    *,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    reg_files: list[str],
    skycell_names: list[str],
    offsets: np.ndarray,
    shifts_dict: dict[tuple[float, float], pd.DataFrame],
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int = 1,
    ignore_mask_bits: list[int] | None = None,
) -> np.ndarray:
    """
    Downsample in-memory star-only skycell arrays to TESS-resolution planes.

    Returns ``(num_offsets, 3, h, w)`` with planes ``FLUX_SUM``, ``COUNT``,
    ``MASK`` in the same order as production templates.
    """
    sparse = process_skycell_batch_from_arrays(
        0,
        reg_files,
        skycell_names,
        arrays,
        offsets,
        shifts_dict,
        base_tess_shape,
        roi_bounds,
        oversampling_factor=oversampling_factor,
        ignore_mask_bits=ignore_mask_bits,
    )
    return combine_sparse_downsample_results(
        [sparse],
        offsets,
        base_tess_shape,
        roi_bounds,
        oversampling_factor=oversampling_factor,
    )


def _host_tag_from_metadata(host_identifier_metadata: dict) -> str:
    for key in ("gaia_source_id", "tic_id", "label"):
        value = host_identifier_metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def write_star_mini_templates(
    output_dir: str | Path,
    arrays_by_offset: np.ndarray,
    *,
    offsets: np.ndarray,
    roi_origin: tuple[int, int],
    host_identifier_metadata: dict,
    oversampling_factor: int = 1,
) -> list[str]:
    """
    Write per-offset mini star-template FITS cutouts.

    ``arrays_by_offset`` has shape ``(num_offsets, 3, h, w)`` with planes
    ``FLUX_SUM``, ``COUNT``, ``MASK``. ``roi_origin`` is ``(x_min, y_min)``
    of the mini ROI in crop-local **native** template coordinates. When
    ``oversampling_factor`` > 1 the array planes are high-resolution
    (``native * F``) and ``OVERSAMP`` is written; ``XMIN``/``XMAX`` stay native.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os_factor = max(1, int(oversampling_factor))
    x_min, y_min = roi_origin
    arr_h = int(arrays_by_offset.shape[2])
    arr_w = int(arrays_by_offset.shape[3])
    if os_factor > 1:
        if arr_h % os_factor != 0 or arr_w % os_factor != 0:
            raise ValueError(
                f"Mini template shape {(arr_h, arr_w)} not divisible by "
                f"oversampling_factor={os_factor}"
            )
        native_h, native_w = arr_h // os_factor, arr_w // os_factor
    else:
        native_h, native_w = arr_h, arr_w
    x_max = x_min + native_w
    y_max = y_min + native_h

    host_tag = _host_tag_from_metadata(host_identifier_metadata)
    sector = int(host_identifier_metadata.get("sector", 0))
    camera = int(host_identifier_metadata.get("camera", 0))
    ccd = int(host_identifier_metadata.get("ccd", 0))

    roi_part = ""
    if not (x_min == 0 and y_min == 0):
        roi_part = f"_x{x_min}-{x_max}_y{y_min}-{y_max}"

    written_paths: list[str] = []
    for idx, (dx, dy) in enumerate(offsets):
        header = fits.Header()
        header["SYNDIFF"] = (True, "Syndiff star mini template")
        header["XMIN"] = (x_min, "Mini ROI xmin in crop-local native pixels")
        header["YMIN"] = (y_min, "Mini ROI ymin in crop-local native pixels")
        header["XMAX"] = (x_max, "Mini ROI xmax (exclusive) in crop-local native pixels")
        header["YMAX"] = (y_max, "Mini ROI ymax (exclusive) in crop-local native pixels")
        header["DX_SHIFT"] = (float(dx), "TESS pixel x shift")
        header["DY_SHIFT"] = (float(dy), "TESS pixel y shift")
        if os_factor > 1:
            header["OVERSAMP"] = (os_factor, "Oversampling factor")
        if sector:
            header["SECTOR"] = (sector, "TESS sector")
        if camera:
            header["CAMERA"] = (camera, "TESS camera")
        if ccd:
            header["CCD"] = (ccd, "TESS CCD")
        for key, value in host_identifier_metadata.items():
            if key in ("sector", "camera", "ccd"):
                continue
            header[key.upper()] = value

        primary_hdu = fits.PrimaryHDU(header=header)
        hdu1 = fits.ImageHDU(
            data=arrays_by_offset[idx, 0].astype(np.float32),
            header=header,
            name="FLUX_SUM",
        )
        hdu2 = fits.ImageHDU(
            data=arrays_by_offset[idx, 1].astype(np.int32),
            header=header,
            name="COUNT",
        )
        hdu3 = fits.ImageHDU(
            data=arrays_by_offset[idx, 2].astype(np.int32),
            header=header,
            name="MASK",
        )
        hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])

        output_filename = (
            output_dir
            / f"star_template_{host_tag}_s{sector:04d}_{camera}_{ccd}"
            f"{roi_part}_dx{dx:.3f}_dy{dy:.3f}{PIPELINE_FITS_EXT}"
        )
        written_paths.append(write_hdul_fits(output_filename, hdu_list))

    return written_paths
