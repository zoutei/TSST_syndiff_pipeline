#!/usr/bin/env python3
"""
Multi-Offset Downsampling Script

This script generates multiple downsampled images, each with a different
pixel offset. It handles mask bits and produces FITS output with proper headers.

Updated to use Zarr data from the convolved_results directory structure:
- data/convolved_results/sector_{sector:04d}/camera_{camera}/ccd_{ccd}/convolved_images.zarr
- data/convolved_results/sector_{sector:04d}/camera_{camera}/ccd_{ccd}/cell_metadata.json

The script loads PS1 convolved image data from Zarr stores instead of individual
FITS files, providing better performance and organization.

When ``event_dir`` and ``cluster_job_json_path`` are set (orchestrator path or
``--job-json`` CLI), by default the script also writes
``{event_dir}/ps1_removed_stars.csv`` — crop-local Gaia rows for PS1
``removed_stars``, using paths and WCS from the cluster job JSON. Use
``write_ps1_removed_stars_csv=False`` (or ``--skip-ps1-removed-star-gaia-csv`` on
the CLI) to disable. This extra step does not run for ``--single-offset`` or when
no cluster job / event dir is provided.
"""

import errno
import json
import os
import shutil
import tempfile
import time
import warnings
from glob import glob
from pathlib import Path
import re

import numpy as np
import pandas as pd
import zarr
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from joblib import Parallel, delayed
from tqdm import tqdm

# Import from existing script
from syndiff_pipeline.common.wcs_grouping import open_fits_memmap, resolve_existing_fits_path
from syndiff_pipeline.difference_imaging.support.ffi_naming import strip_fits_suffix
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import RELEVANT_WCS_KEYS, build_ps1_wcs, compute_ps1_shift_for_skycell, load_tess_wcs
from syndiff_pipeline.template_creation.processing.downsample_progress import (
    init_progress as init_downsample_progress,
    mark_skycell_done as mark_downsample_skycell_done,
    set_progress_phase as set_downsample_progress_phase,
)


def load_cluster_template_job_payload(path: str | Path) -> dict:
    """Load ``cluster_template_job.json`` and validate fields needed for offsets."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cluster job JSON not found: {path}")
    with open(path) as fh:
        payload = json.load(fh)
    if "schema_version" not in payload:
        raise ValueError(f"{path}: missing schema_version")
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) == 0:
        raise ValueError(f"{path}: missing or empty 'groups'")
    for g in groups:
        if not isinstance(g, dict) or "group_dx" not in g or "group_dy" not in g:
            raise ValueError(f"{path}: each group must be a dict with group_dx and group_dy")
    return payload


def offsets_from_cluster_job_payload(payload: dict) -> np.ndarray:
    """Unique (dx, dy) pairs from ``groups``, preserving first-seen order."""
    rows: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for g in payload["groups"]:
        dx = float(g["group_dx"])
        dy = float(g["group_dy"])
        key = (round(dx, 12), round(dy, 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append([dx, dy])
    if not rows:
        raise ValueError("No unique offsets after deduplicating cluster_template_job groups")
    return np.asarray(rows, dtype=np.float64)


def roi_tuple_from_cluster_job_payload(payload: dict) -> tuple[int, int, int, int]:
    """ROI (x_min, y_min, x_max, y_max) in base TESS pixels, [min, max)."""
    required = ("x_min", "x_max", "y_min", "y_max")
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"cluster_template_job.json missing keys: {missing}")
    return int(payload["x_min"]), int(payload["y_min"]), int(payload["x_max"]), int(payload["y_max"])


def instrument_tuple_from_cluster_job_payload(payload: dict) -> tuple[int, int, int]:
    """sector, camera, ccd from cluster handoff JSON."""
    missing = [k for k in ("sector", "camera", "ccd") if k not in payload]
    if missing:
        raise KeyError(f"cluster_template_job.json missing keys: {missing}")
    return int(payload["sector"]), int(payload["camera"]), int(payload["ccd"])


def read_removed_stars_csv(path: str | Path) -> pd.DataFrame:
    """Load PS1 ``*_removed_stars.csv`` keeping Gaia ``source_id`` as nullable integer."""
    df = pd.read_csv(path)
    if "source_id" in df.columns:
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
    return df


PS1_REMOVED_STARS_CSV_FILENAME = "ps1_removed_stars.csv"


def default_ps1_process_removed_stars_csv_path(
    convolved_dir: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Default PS1 pipeline ``*_removed_stars.csv`` beside the convolved zarr."""
    return Path(convolved_dir) / (
        f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}_removed_stars.csv"
    )


def write_ps1_removed_star_gaia_csv(
    *,
    job_json_path: str | Path,
    removed_stars_csv: str | Path,
    event_dir: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    roi_bounds: tuple[int, int, int, int],
) -> Path:
    """
    Dedup PS1 removed-star rows by Gaia ``source_id``, project to crop-local ``x``/``y``
    using ``reference_ffi_path`` from ``cluster_template_job.json`` (HDU 1 WCS).

    Returns path written, or None if no rows after filtering.
    """
    payload = load_cluster_template_job_payload(job_json_path)
    ref_ffi = payload.get("reference_ffi_path")
    if not ref_ffi or not str(ref_ffi).strip():
        raise KeyError("cluster_template_job.json missing reference_ffi_path")
    ref_ffi = str(ref_ffi).strip()

    x_min, y_min, x_max, y_max = roi_bounds
    crop_bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    ps1_df = read_removed_stars_csv(removed_stars_csv)
    if "source_id" not in ps1_df.columns:
        raise ValueError(f"removed_stars CSV missing source_id: {removed_stars_csv}")

    unique_df = ps1_df.drop_duplicates(subset="source_id").copy()
    ok_id = unique_df["source_id"].notna() & (unique_df["source_id"] != -1)
    unique_df = unique_df[ok_id].copy()

    keep_cols = ["source_id", "ra", "dec", "tess_mag"]
    for col in ("phot_rp_mean_mag", "phot_g_mean_mag", "phot_bp_mean_mag"):
        if col in unique_df.columns:
            keep_cols.append(col)
    unique_df = unique_df[keep_cols].reset_index(drop=True)

    # Support ref_ffi with or without .gz at the end
    ref_ffi_path = resolve_existing_fits_path(ref_ffi)

    with open_fits_memmap(ref_ffi_path) as hdul:
        ref_header = hdul[1].header
        nx = int(ref_header["NAXIS1"])
        ny = int(ref_header["NAXIS2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(ref_header)

    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

    coords = SkyCoord(
        ra=unique_df["ra"].values,
        dec=unique_df["dec"].values,
        unit="deg",
    )
    x_pix, y_pix = world_ra_dec_to_pixel(wcs, coords.ra.deg, coords.dec.deg)
    unique_df["x_ffi"] = x_pix
    unique_df["y_ffi"] = y_pix

    on_chip = (
        (unique_df["x_ffi"] >= 0) & (unique_df["x_ffi"] < nx) &
        (unique_df["y_ffi"] >= 0) & (unique_df["y_ffi"] < ny)
    )
    unique_df = unique_df[on_chip].copy()

    cx0, cy0, cx1, cy1 = (
        crop_bounds["x_min"],
        crop_bounds["y_min"],
        crop_bounds["x_max"],
        crop_bounds["y_max"],
    )
    in_crop = (
        (unique_df["x_ffi"] >= cx0) & (unique_df["x_ffi"] < cx1) &
        (unique_df["y_ffi"] >= cy0) & (unique_df["y_ffi"] < cy1)
    )
    cropped_df = unique_df[in_crop].copy()
    cropped_df["x"] = cropped_df["x_ffi"] - cx0
    cropped_df["y"] = cropped_df["y_ffi"] - cy0
    cropped_df = cropped_df.drop(columns=["x_ffi", "y_ffi"]).reset_index(drop=True)

    # Integer source_id for CSV (no float rounding)
    cropped_df["source_id"] = cropped_df["source_id"].astype("int64")

    out_path = Path(event_dir) / PS1_REMOVED_STARS_CSV_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cropped_df.to_csv(out_path, index=False)
    print(
        f"Wrote crop-local Gaia catalog for PS1 removed stars: {out_path} "
        f"({len(cropped_df)} rows)"
    )
    return out_path


def resolve_downsample_scratch_dir() -> Path:
    """Local scratch for staging regmaps (Condor ``_CONDOR_SCRATCH_DIR`` or ``TMPDIR``)."""
    for env_var in ("_CONDOR_SCRATCH_DIR", "TMPDIR"):
        val = os.environ.get(env_var)
        if val:
            return Path(val)
    return Path(tempfile.gettempdir())


def resolve_stage_regmaps_to_scratch(stage_regmaps_to_scratch: bool | None) -> bool:
    """Auto-enable staging on Condor when ``stage_regmaps_to_scratch`` is None."""
    if stage_regmaps_to_scratch is not None:
        return stage_regmaps_to_scratch
    return "_CONDOR_SCRATCH_DIR" in os.environ or "CONDOR_JOB_AD" in os.environ


def stage_regmap_files_to_scratch(
    reg_files: list[str],
    *,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int,
    scratch_base: Path | None = None,
) -> tuple[list[str], Path, int, float]:
    """
    Copy ROI regmap FITS onto local scratch **as-is** (keep ``.fits.gz`` compressed).

    Gunzipping full-chip osN regmaps would require hundreds of GB; ``fits.open``
    reads ``.fits.gz`` directly. Convolved Zarr stays on shared NFS.

    Returns
    -------
    local_paths, scratch_dir, n_newly_staged, elapsed_s
    """
    t0 = time.perf_counter()
    scratch_root = scratch_base or resolve_downsample_scratch_dir()
    os_suffix = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""
    scratch_dir = (
        scratch_root
        / f"syndiff_downsample_regmaps_{sector:04d}_{camera}_{ccd}{os_suffix}"
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[str] = []
    n_staged = 0
    for src_str in reg_files:
        src = Path(src_str)
        dest = scratch_dir / src.name
        if not dest.is_file():
            shutil.copy2(src, dest)
            n_staged += 1
        local_paths.append(str(dest))

    return local_paths, scratch_dir, n_staged, time.perf_counter() - t0


def checkpoint_dir_for_output(output_dir: str | Path) -> Path:
    """Sparse per-skycell NPZ checkpoint directory under the sector output tree."""
    return Path(output_dir) / "_partial"


def checkpoint_npz_path(checkpoint_dir: Path, skycell_name: str) -> Path:
    """Safe on-disk path for one skycell's sparse contribution arrays."""
    safe = skycell_name.replace("/", "_")
    return checkpoint_dir / f"{safe}.npz"


def is_valid_skycell_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as data:
            for key in ("indices", "sums", "counts", "mask_counts"):
                if key not in data:
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def load_skycell_checkpoint(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["indices"]),
            np.asarray(data["sums"]),
            np.asarray(data["counts"]),
            np.asarray(data["mask_counts"]),
        )


def save_skycell_checkpoint(
    path: Path,
    indices: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
    mask_counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = path.parent / f"{path.stem}.tmp"
    np.savez_compressed(
        tmp_stem,
        indices=indices,
        sums=sums,
        counts=counts,
        mask_counts=mask_counts,
    )
    tmp_path = Path(f"{tmp_stem}.npz")
    tmp_path.replace(path)


def scan_completed_skycell_checkpoints(checkpoint_dir: Path) -> set[str]:
    """Skycell names with a readable checkpoint NPZ in ``checkpoint_dir``."""
    if not checkpoint_dir.is_dir():
        return set()
    completed: set[str] = set()
    for path in checkpoint_dir.glob("*.npz"):
        if path.name.endswith(".tmp.npz"):
            continue
        if not is_valid_skycell_checkpoint(path):
            continue
        completed.add(path.stem)
    return completed


def load_all_checkpoint_contributions(
    checkpoint_dir: Path,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load every valid skycell checkpoint for combine after a mid-run restart."""
    if not checkpoint_dir.is_dir():
        return []
    contributions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in sorted(checkpoint_dir.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        if not is_valid_skycell_checkpoint(path):
            continue
        contributions.append(load_skycell_checkpoint(path))
    return contributions


def extract_skycell_name_from_reg_file(reg_file: str) -> str | None:
    """Extract skycell.<proj>.<cell> from a registration filename."""
    fname = Path(reg_file).name
    match = re.search(r"(skycell\.\d+\.\d+)", fname)
    if match:
        return match.group(1)
    return None


def load_zarr_metadata(sector: int, camera: int, ccd: int, convolved_data_path: Path) -> tuple[dict, Path]:
    """
    Load Zarr metadata once to avoid repeated file access.

    Returns:
        Tuple of (metadata_dict, zarr_path)
    """
    zarr_path = convolved_data_path / f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    # metadata_path = convolved_data_path / f"sector_{sector:04d}" / f"camera_{camera}" / f"ccd_{ccd}" / "cell_metadata.json"

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    # if not metadata_path.exists():
    #     raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # # Load metadata once
    # with open(metadata_path) as f:
    #     metadata = json.load(f)

    return zarr_path


def _count_non_empty_convolved_data_arrays(zarr_path: Path) -> tuple[int, list[str]]:
    """Return (non-empty *_data array count, all *_data array names) in a convolved zarr."""
    root = zarr.open(str(zarr_path), mode="r")
    data_keys = [str(k) for k in root.array_keys() if str(k).endswith("_data")]
    non_empty = sum(1 for key in data_keys if int(root[key].size) > 0)
    return non_empty, data_keys


def require_convolved_zarr_data(zarr_path: Path) -> None:
    """Raise if the convolved zarr store has no usable PS1 skycell arrays."""
    saved, data_keys = _count_non_empty_convolved_data_arrays(zarr_path)
    if saved > 0:
        return
    if data_keys:
        raise RuntimeError(
            f"Convolved zarr has {len(data_keys)} *_data arrays but all are empty: {zarr_path}"
        )
    raise RuntimeError(f"Convolved zarr store is empty (no *_data arrays): {zarr_path}")


def load_zarr_data_for_skycell(skycell_name: str, zarr_store) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PS1 convolved image and mask data from Zarr store for a specific skycell.

    Args:
        skycell_name: Name of the skycell (e.g., "rings.v3.skycell.1234.567")
        metadata: Pre-loaded metadata dictionary
        zarr_path: Path to the Zarr store

    Returns:
        Tuple of (image_data, mask_data) as numpy arrays
    """
    # Find the index for this skycell from pre-loaded metadata

    # Load the Zarr store (this is cached by Zarr internally)

    if skycell_name.startswith("skycell."):
        skycell_key = skycell_name
    else:
        skycell_key = f"skycell.{skycell_name}"

    image_data = zarr_store[f"{skycell_key}_data"]
    mask_data = zarr_store[f"{skycell_key}_mask"]

    return np.array(image_data).astype(np.float32), np.array(mask_data).astype(np.uint32)


def precompute_shifts_for_offsets(
    tess_wcs: WCS,
    skycell_df: pd.DataFrame,
    offsets: np.ndarray,
    progress_path: str | Path | None = None,
) -> dict[tuple[float, float], pd.DataFrame]:
    """
    Precompute all PS1 shifts for each offset pair and skycell

    Returns:
        Dictionary mapping (dx, dy) to DataFrame with NAME, shift_x, shift_y
    """
    shift_results = {}
    offsets_total = len(offsets)
    if progress_path is not None:
        set_downsample_progress_phase(
            progress_path,
            "precomputing_shifts",
            offsets_done=0,
            offsets_total=offsets_total,
        )

    for offset_idx, (dx, dy) in enumerate(tqdm(offsets, desc="Computing shifts")):
        shift_x_list = []
        shift_y_list = []

        for _, row in skycell_df.iterrows():
            ps1_wcs, _ = build_ps1_wcs(row)
            sx, sy = compute_ps1_shift_for_skycell(
                tess_wcs,
                dx,
                dy,
                float(row["RA"]),
                float(row["DEC"]),
                ps1_wcs,
            )
            # Round to nearest integer (no interpolation)
            sx_int = int(round(sx))
            sy_int = int(round(sy))
            shift_x_list.append(sx_int)
            shift_y_list.append(sy_int)

        shift_df = pd.DataFrame(
            {
                "NAME": skycell_df["NAME"],
                "shift_x": shift_x_list,
                "shift_y": shift_y_list,
            }
        )

        shift_results[(dx, dy)] = shift_df
        if progress_path is not None:
            set_downsample_progress_phase(
                progress_path,
                "precomputing_shifts",
                offsets_done=offset_idx + 1,
                offsets_total=offsets_total,
            )

    return shift_results


def _ignore_mask_from_bits(ignore_mask_bits: list[int] | None) -> int:
    if ignore_mask_bits is None:
        ignore_mask_bits = []
    ignore_mask = 0
    for bit in ignore_mask_bits:
        ignore_mask |= 1 << bit
    return ignore_mask


def _shifts_for_skycell(
    shifts_dict: dict[tuple[float, float], pd.DataFrame],
    offsets: np.ndarray,
    skycell_name: str,
) -> list[tuple[int, int] | None]:
    """Per-offset (shift_x, shift_y) for ``skycell_name``, or None when absent."""
    result: list[tuple[int, int] | None] = []
    for dx, dy in offsets:
        shift_df = shifts_dict[(float(dx), float(dy))]
        name_to_shift = dict(
            zip(
                shift_df["NAME"],
                zip(
                    shift_df["shift_x"].astype(int, copy=False),
                    shift_df["shift_y"].astype(int, copy=False),
                ),
            )
        )
        result.append(name_to_shift.get(skycell_name))
    return result


def _aggregate_sorted_groups(
    ps1_rav: np.ndarray,
    ps1_mask_rav: np.ndarray,
    group_starts: np.ndarray,
    ignore_mask: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized per-group counts, nansum (excluding ignored), and mask counts."""
    n_groups = len(group_starts)
    group_ends = np.concatenate([group_starts[1:], [len(ps1_rav)]])
    group_sizes = group_ends - group_starts
    group_ids = np.repeat(np.arange(n_groups, dtype=np.intp), group_sizes)

    slice_start = int(group_starts[0])
    ps1_grouped = ps1_rav[slice_start:]
    mask_grouped = ps1_mask_rav[slice_start:]

    ignored = (mask_grouped & ignore_mask) > 0
    sum_weights = np.where(ignored, 0.0, ps1_grouped).astype(np.float64, copy=False)
    sum_weights = np.where(np.isnan(sum_weights), 0.0, sum_weights)

    counts = np.bincount(group_ids, minlength=n_groups).astype(np.int32, copy=False)
    sums = np.bincount(group_ids, weights=sum_weights, minlength=n_groups).astype(
        np.float32, copy=False
    )
    mask_weights = (mask_grouped != 0).astype(np.int32, copy=False)
    mask_counts = np.bincount(
        group_ids, weights=mask_weights, minlength=n_groups
    ).astype(np.int32, copy=False)
    return sums, counts, mask_counts


def _decode_sparse_indices_to_roi(
    combined_indices: np.ndarray,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (valid_mask, out_y, out_x) for sparse linearized TESS indices."""
    x_min, y_min, x_max, y_max = roi_bounds
    _, t_x = base_tess_shape

    if oversampling_factor > 1:
        os_width = t_x * oversampling_factor
        y_os = combined_indices // os_width
        x_os = combined_indices % os_width
        y_base = y_os // oversampling_factor
        x_base = x_os // oversampling_factor
        sub_y = y_os % oversampling_factor
        sub_x = x_os % oversampling_factor
        out_y = (y_base - y_min) * oversampling_factor + sub_y
        out_x = (x_base - x_min) * oversampling_factor + sub_x
    else:
        y_base = combined_indices // t_x
        x_base = combined_indices % t_x
        out_y = y_base - y_min
        out_x = x_base - x_min

    valid = (
        (x_min <= x_base)
        & (x_base < x_max)
        & (y_min <= y_base)
        & (y_base < y_max)
        & (out_y >= 0)
        & (out_y < out_h)
        & (out_x >= 0)
        & (out_x < out_w)
    )
    return (
        valid,
        out_y[valid].astype(np.int32, copy=False),
        out_x[valid].astype(np.int32, copy=False),
    )


def _process_skycell_registration_binning(
    *,
    ps1_assignment: np.ndarray,
    ps1_data: np.ndarray,
    ps1_mask: np.ndarray,
    skycell_name: str,
    offsets: np.ndarray,
    shifts_dict: dict[tuple[float, float], pd.DataFrame],
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int,
    ignore_mask: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Bin one skycell's in-memory (data, mask) into sparse TESS-pixel contributions."""
    t_y, t_x = base_tess_shape
    num_offsets = len(offsets)
    x_min, y_min, x_max, y_max = roi_bounds

    pind = ps1_assignment.ravel()
    sort_ind = np.argsort(pind)

    tess_pixels = np.unique(pind[np.isfinite(pind)]).astype(int)
    tess_pixels = tess_pixels[tess_pixels >= 0]

    if len(tess_pixels) == 0:
        return None

    breaks = np.where(np.diff(pind[sort_ind]) > 0)[0] + 1
    breaks = np.append(breaks, len(sort_ind))
    # ``breaks[i]:breaks[i+1]`` slices match the legacy loop (skips unmapped prefix).
    group_starts = breaks[:-1]

    ps1_base = ps1_data
    ps1_mask_base = ps1_mask

    pixel_sums = np.zeros((len(tess_pixels), num_offsets), dtype=np.float32)
    pixel_counts = np.zeros((len(tess_pixels), num_offsets), dtype=np.int32)
    pixel_mask_counts = np.zeros((len(tess_pixels), num_offsets), dtype=np.int32)

    offset_shifts = _shifts_for_skycell(shifts_dict, offsets, skycell_name)

    for offset_idx, shift in enumerate(offset_shifts):
        if shift is None:
            continue

        sx, sy = shift
        ps1_shifted = np.roll(ps1_base, (sy, sx), axis=(0, 1))
        ps1_mask_shifted = np.roll(ps1_mask_base, (sy, sx), axis=(0, 1))

        ps1_rav = ps1_shifted.ravel()[sort_ind]
        ps1_mask_rav = ps1_mask_shifted.ravel()[sort_ind]

        sums, counts, mask_counts = _aggregate_sorted_groups(
            ps1_rav,
            ps1_mask_rav,
            group_starts,
            ignore_mask,
        )

        pixel_sums[:, offset_idx] = sums
        pixel_counts[:, offset_idx] = counts.astype(np.int32, copy=False)
        pixel_mask_counts[:, offset_idx] = mask_counts.astype(np.int32, copy=False)

    if oversampling_factor > 1:
        os_width = t_x * oversampling_factor
        y_os = tess_pixels // os_width
        x_os = tess_pixels % os_width
        y_base = y_os // oversampling_factor
        x_base = x_os // oversampling_factor
    else:
        y_base = tess_pixels // t_x
        x_base = tess_pixels % t_x

    valid_mask = (
        (0 <= y_base)
        & (y_base < t_y)
        & (0 <= x_base)
        & (x_base < t_x)
        & (x_base >= x_min)
        & (x_base < x_max)
        & (y_base >= y_min)
        & (y_base < y_max)
    )

    if not np.any(valid_mask):
        return None

    return (
        tess_pixels[valid_mask],
        pixel_sums[valid_mask],
        pixel_counts[valid_mask],
        pixel_mask_counts[valid_mask],
    )


def _finalize_skycell_batch_results(
    all_indices: list[np.ndarray],
    all_sums: list[np.ndarray],
    all_counts: list[np.ndarray],
    all_mask_counts: list[np.ndarray],
    num_offsets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if all_indices:
        indices = np.concatenate(all_indices)
        sums = np.vstack(all_sums)
        counts = np.vstack(all_counts)
        mask_counts = np.vstack(all_mask_counts)
    else:
        indices = np.array([], dtype=int)
        sums = np.zeros((0, num_offsets), dtype=np.float32)
        counts = np.zeros((0, num_offsets), dtype=np.int32)
        mask_counts = np.zeros((0, num_offsets), dtype=np.int32)
    return indices, sums, counts, mask_counts


def _run_skycell_batch_core(
    batch_idx: int,
    reg_files: list[str],
    skycell_names: list[str],
    offsets: np.ndarray,
    shifts_dict: dict[tuple[float, float], pd.DataFrame],
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int,
    ignore_mask_bits: list[int] | None,
    progress_path: str | Path | None,
    load_ps1_arrays,
    *,
    log_level: str = "INFO",
    total_batches: int | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_skycells: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_offsets = len(offsets)
    ignore_mask = _ignore_mask_from_bits(ignore_mask_bits)
    debug = log_level.upper() == "DEBUG"
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None

    all_indices: list[np.ndarray] = []
    all_sums: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_mask_counts: list[np.ndarray] = []

    batch_t0 = time.perf_counter()
    regmap_time = 0.0
    zarr_time = 0.0
    binning_time = 0.0

    for reg_file, skycell_name in zip(reg_files, skycell_names):
        skycell_regmap = 0.0
        skycell_zarr = 0.0
        skycell_binning = 0.0
        ckpt_path = (
            checkpoint_npz_path(ckpt_dir, skycell_name)
            if checkpoint_skycells and ckpt_dir is not None
            else None
        )
        try:
            if ckpt_path is not None and is_valid_skycell_checkpoint(ckpt_path):
                contribution = load_skycell_checkpoint(ckpt_path)
                if contribution[0].size > 0:
                    all_indices.append(contribution[0])
                    all_sums.append(contribution[1])
                    all_counts.append(contribution[2])
                    all_mask_counts.append(contribution[3])
                if debug:
                    print(f"[downsample] skycell {skycell_name} loaded from checkpoint")
                continue

            t_reg = time.perf_counter()
            with fits.open(reg_file) as hdul:
                ps1_assignment = hdul[1].data.astype(int)
            skycell_regmap = time.perf_counter() - t_reg
            regmap_time += skycell_regmap

            try:
                t_zarr = time.perf_counter()
                ps1_data, ps1_mask = load_ps1_arrays(skycell_name)
                skycell_zarr = time.perf_counter() - t_zarr
                zarr_time += skycell_zarr

                t_bin = time.perf_counter()
                contribution = _process_skycell_registration_binning(
                    ps1_assignment=ps1_assignment,
                    ps1_data=ps1_data,
                    ps1_mask=ps1_mask,
                    skycell_name=skycell_name,
                    offsets=offsets,
                    shifts_dict=shifts_dict,
                    base_tess_shape=base_tess_shape,
                    roi_bounds=roi_bounds,
                    oversampling_factor=oversampling_factor,
                    ignore_mask=ignore_mask,
                )
                skycell_binning = time.perf_counter() - t_bin
                binning_time += skycell_binning
                if contribution is not None:
                    all_indices.append(contribution[0])
                    all_sums.append(contribution[1])
                    all_counts.append(contribution[2])
                    all_mask_counts.append(contribution[3])
                    if ckpt_path is not None:
                        save_skycell_checkpoint(
                            ckpt_path,
                            contribution[0],
                            contribution[1],
                            contribution[2],
                            contribution[3],
                        )
            except Exception as e:
                print(f"Error processing PS1 data for skycell {skycell_name}: {e}")
                continue

        except Exception as e:
            print(f"Error processing registration for skycell {skycell_name}: {e}")
        finally:
            if progress_path is not None:
                mark_downsample_skycell_done(progress_path, batch_idx)
            if debug:
                print(
                    f"[downsample] skycell {skycell_name} regmap={skycell_regmap:.1f}s "
                    f"zarr={skycell_zarr:.1f}s binning={skycell_binning:.1f}s"
                )

    batch_elapsed = time.perf_counter() - batch_t0
    n_skycells = len(reg_files)
    if total_batches is not None and total_batches > 0:
        batch_label = f"{batch_idx + 1}/{total_batches}"
    else:
        batch_label = str(batch_idx + 1)
    print(
        f"[downsample] batch {batch_label} done skycells={n_skycells} "
        f"elapsed={batch_elapsed:.1f}s zarr={zarr_time:.1f}s regmap={regmap_time:.1f}s "
        f"binning={binning_time:.1f}s"
    )
    return _finalize_skycell_batch_results(
        all_indices, all_sums, all_counts, all_mask_counts, num_offsets
    )


def combine_sparse_downsample_results(
    batch_results: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    offsets: np.ndarray,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int = 1,
) -> np.ndarray:
    """Merge sparse skycell-batch outputs into dense (num_offsets, 3, h, w) planes."""
    x_min, y_min, x_max, y_max = roi_bounds
    all_indices: list[np.ndarray] = []
    all_sums: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_mask_counts: list[np.ndarray] = []

    for indices, sums, counts, mask_counts in batch_results:
        if len(indices) > 0:
            all_indices.append(indices)
            all_sums.append(sums)
            all_counts.append(counts)
            all_mask_counts.append(mask_counts)

    if not all_indices:
        roi_h = y_max - y_min
        roi_w = x_max - x_min
        out_h = roi_h * oversampling_factor
        out_w = roi_w * oversampling_factor
        return np.zeros((len(offsets), 3, out_h, out_w), dtype=np.float32)

    combined_indices = np.concatenate(all_indices)
    combined_sums = np.vstack(all_sums)
    combined_counts = np.vstack(all_counts)
    combined_mask_counts = np.vstack(all_mask_counts)

    if len(combined_indices) > len(np.unique(combined_indices)):
        unique_indices, inverse_indices = np.unique(combined_indices, return_inverse=True)

        unique_sums = np.zeros((len(unique_indices), len(offsets)), dtype=np.float32)
        unique_counts = np.zeros((len(unique_indices), len(offsets)), dtype=np.int32)
        unique_mask_counts = np.zeros((len(unique_indices), len(offsets)), dtype=np.int32)

        np.add.at(unique_sums, inverse_indices, combined_sums)
        np.add.at(unique_counts, inverse_indices, combined_counts)
        np.add.at(unique_mask_counts, inverse_indices, combined_mask_counts)

        combined_indices = unique_indices
        combined_sums = unique_sums
        combined_counts = unique_counts
        combined_mask_counts = unique_mask_counts

    roi_h = y_max - y_min
    roi_w = x_max - x_min
    out_h = roi_h * oversampling_factor
    out_w = roi_w * oversampling_factor
    combined_results = np.zeros((len(offsets), 3, out_h, out_w), dtype=np.float32)

    valid, out_y, out_x = _decode_sparse_indices_to_roi(
        combined_indices,
        base_tess_shape,
        roi_bounds,
        oversampling_factor,
        out_h,
        out_w,
    )
    if not np.any(valid):
        return combined_results

    sums_valid = combined_sums[valid]
    counts_valid = combined_counts[valid]
    mask_counts_valid = combined_mask_counts[valid]

    combined_results[:, 0, out_y, out_x] = sums_valid.T
    combined_results[:, 1, out_y, out_x] = counts_valid.T
    combined_results[:, 2, out_y, out_x] = mask_counts_valid.T

    return combined_results


def process_skycell_batch(
    batch_idx: int,
    reg_files: list[str],
    skycell_names: list[str],
    offsets: np.ndarray,
    shifts_dict: dict[tuple[float, float], pd.DataFrame],
    base_tess_shape: tuple[int, int],
    zarr_path: Path,
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int = 1,
    ignore_mask_bits: list[int] | None = None,
    progress_path: str | Path | None = None,
    log_level: str = "INFO",
    total_batches: int | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_skycells: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a batch of skycells using sparse arrays for memory efficiency

    Returns:
        Tuple of (indices, sums, counts, mask_counts) where:
        - indices: Array of TESS pixel indices (1D linearized from y,x)
        - sums: Array of shape (len(indices), num_offsets) with sum values
        - counts: Array of shape (len(indices), num_offsets) with count values
        - mask_counts: Array of shape (len(indices), num_offsets) with mask count values
    """
    zarr_store = zarr.open(zarr_path, mode="r")

    def load_ps1_arrays(skycell_name: str) -> tuple[np.ndarray, np.ndarray]:
        return load_zarr_data_for_skycell(skycell_name, zarr_store)

    return _run_skycell_batch_core(
        batch_idx,
        reg_files,
        skycell_names,
        offsets,
        shifts_dict,
        base_tess_shape,
        roi_bounds,
        oversampling_factor,
        ignore_mask_bits,
        progress_path,
        load_ps1_arrays,
        log_level=log_level,
        total_batches=total_batches,
        checkpoint_dir=checkpoint_dir,
        checkpoint_skycells=checkpoint_skycells,
    )


def process_skycell_batch_from_arrays(
    batch_idx: int,
    reg_files: list[str],
    skycell_names: list[str],
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    offsets: np.ndarray,
    shifts_dict: dict[tuple[float, float], pd.DataFrame],
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int = 1,
    ignore_mask_bits: list[int] | None = None,
    progress_path: str | Path | None = None,
    log_level: str = "INFO",
    total_batches: int | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_skycells: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a batch of skycells from in-memory (data, mask) arrays.

    ``arrays`` maps skycell name (e.g. ``skycell.1234.567``) to
    ``(data, mask)`` float32/uint32 arrays matching the Zarr layout.
    """

    def load_ps1_arrays(skycell_name: str) -> tuple[np.ndarray, np.ndarray]:
        key = skycell_name if skycell_name.startswith("skycell.") else f"skycell.{skycell_name}"
        if key not in arrays:
            raise KeyError(f"Missing in-memory arrays for skycell {skycell_name!r}")
        data, mask = arrays[key]
        return np.asarray(data, dtype=np.float32), np.asarray(mask, dtype=np.uint32)

    return _run_skycell_batch_core(
        batch_idx,
        reg_files,
        skycell_names,
        offsets,
        shifts_dict,
        base_tess_shape,
        roi_bounds,
        oversampling_factor,
        ignore_mask_bits,
        progress_path,
        load_ps1_arrays,
        log_level=log_level,
        total_batches=total_batches,
        checkpoint_dir=checkpoint_dir,
        checkpoint_skycells=checkpoint_skycells,
    )


def create_syndiff_header(
    tess_header,
    roi_bounds: tuple[int, int, int, int] | None = None,
    oversampling_factor: int = 1,
    sector: int | None = None,
):
    """
    Create a header for the syndiff output based on the TESS header.
    """
    # Instrument provenance keywords (SECTOR before CAMERA/CCD).
    syndiff_header = fits.Header()
    for key in ("TELESCOP", "INSTRUME"):
        if key in tess_header:
            syndiff_header.set(key, tess_header[key], tess_header.comments[key])

    if sector is not None:
        syndiff_header.set("SECTOR", sector, "TESS sector")
    elif "SECTOR" in tess_header:
        syndiff_header.set("SECTOR", tess_header["SECTOR"], tess_header.comments["SECTOR"])

    for key in ("CAMERA", "CCD"):
        if key in tess_header:
            syndiff_header.set(key, tess_header[key], tess_header.comments[key])

    if "TESS_FFI" in tess_header:
        syndiff_header.set(
            "TESS_REFERENCE_FFI",
            tess_header["TESS_FFI"],
            "TESS reference FFI filename",
        )

    # Set PS1 date information
    syndiff_header.set("MJD-OBS", "55197.00000", "TSTART of PS1")
    syndiff_header.set("DATE-OBS", "2010-01-01T00:00:00.000", "TSTART of PS1")
    syndiff_header.set("DATE-END", "2015-01-01T00:00:00.000", "TSTOP of PS1")

    # Copy WCS and quality information
    keys_to_copy = ["RADESYS", "EQUINOX", "WCSAXES", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "DQUALITY", "IMAGTYPE"]

    for key in tess_header:
        if key.startswith(("A_", "B_", "AP_", "BP_", "RA_", "DEC_", "ROLL_")) or key in keys_to_copy:
            syndiff_header.set(key, tess_header[key], tess_header.comments[key])

    # Add syndiff tag
    syndiff_header.set("SYNDIFF", True, "Syndiff template")

    # Apply oversampling WCS scaling if needed (smaller pixel scale).
    if oversampling_factor > 1:
        for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"]:
            if key in syndiff_header:
                syndiff_header[key] = syndiff_header[key] / oversampling_factor
        syndiff_header.set("OVERSAMP", oversampling_factor, "Oversampling factor")

    # Apply ROI crop metadata and CRPIX shift.
    if roi_bounds is not None:
        x_min, y_min, x_max, y_max = roi_bounds
        shift_x = x_min * oversampling_factor
        shift_y = y_min * oversampling_factor

        if "CRPIX1" in syndiff_header:
            syndiff_header["CRPIX1"] = syndiff_header["CRPIX1"] - shift_x
        if "CRPIX2" in syndiff_header:
            syndiff_header["CRPIX2"] = syndiff_header["CRPIX2"] - shift_y

        syndiff_header.set("XMIN", x_min, "ROI xmin in base TESS pixels")
        syndiff_header.set("XMAX", x_max, "ROI xmax (exclusive) in base TESS pixels")
        syndiff_header.set("YMIN", y_min, "ROI ymin in base TESS pixels")
        syndiff_header.set("YMAX", y_max, "ROI ymax (exclusive) in base TESS pixels")
        syndiff_header.set("ROIW", x_max - x_min, "ROI width in base TESS pixels")
        syndiff_header.set("ROIH", y_max - y_min, "ROI height in base TESS pixels")

    return syndiff_header


def save_fits_outputs(
    output_dir: Path,
    sector: int,
    camera: int,
    ccd: int,
    results: np.ndarray,
    offsets: np.ndarray,
    tess_header: fits.Header,
    roi_bounds: tuple[int, int, int, int] | None = None,
    oversampling_factor: int = 1,
) -> list[str]:
    """
    Save the results as FITS files.

    Args:
        output_dir: Directory to save outputs
        results: Array of shape (num_offsets, 3, ny, nx) with:
            [0] = sum of PS1 pixel values
            [1] = count of PS1 pixels
            [2] = count of masked PS1 pixels
        offsets: Array of (dx, dy) pairs
        tess_header: Header from TESS file to use as a base
        save_extensions: Whether to save data, count and mask as HDU extensions

    Returns:
        List of written FITS file paths (one per offset), in offset order.
    """
    # Create syndiff header based on TESS header
    syndiff_header = create_syndiff_header(
        tess_header,
        roi_bounds=roi_bounds,
        oversampling_factor=oversampling_factor,
        sector=sector,
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # # Save a CSV with the offsets
    # offset_df = pd.DataFrame(offsets, columns=['dx', 'dy'])
    # offset_df.to_csv(output_dir / "offsets.csv", index=False)

    written_paths: list[str] = []
    # Save each offset result as a FITS file
    for idx, (dx, dy) in enumerate(offsets):
        # Update header with offset information
        offset_header = syndiff_header.copy()
        offset_header["DX_SHIFT"] = (dx, "TESS pixel x shift")
        offset_header["DY_SHIFT"] = (dy, "TESS pixel y shift")

        # File with data, count, and mask as extensions
        primary_hdu = fits.PrimaryHDU(header=offset_header)
        # FLUX extension = SUM per TESS pixel
        hdu1 = fits.ImageHDU(data=results[idx, 0].astype(np.float32), header=offset_header, name="FLUX_SUM")
        # Optionally add average if desired:
        # hdu_avg = fits.ImageHDU(data=avg_image, header=offset_header, name="FLUX_AVG")
        hdu2 = fits.ImageHDU(data=results[idx, 1].astype(np.int32), header=offset_header, name="COUNT")
        hdu3 = fits.ImageHDU(data=results[idx, 2].astype(np.int32), header=offset_header, name="MASK")

        hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
        # Build filename including ROI and oversampling when applicable
        roi_part = ""
        if roi_bounds is not None:
            rx0, ry0, rx1, ry1 = roi_bounds
            if not (rx0 == 0 and ry0 == 0):
                roi_part = f"_x{rx0}-{rx1}_y{ry0}-{ry1}"
        os_part = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""

        output_filename = output_dir / f"syndiff_template_s{sector:04d}_{camera}_{ccd}{roi_part}{os_part}_dx{dx:.3f}_dy{dy:.3f}.fits.gz"
        hdu_list.writeto(output_filename, overwrite=True)
        written_paths.append(str(output_filename))

    return written_paths


def main(
    sector: int = 20,
    camera: int = 3,
    ccd: int = 3,
    offsets: np.ndarray = np.array([[0.0, 0.0]]),
    ignore_mask_bits: list[int] = [12],
    data_root: str | Path = "data",
    mapping_dir: str | Path | None = None,
    convolved_dir: str | Path | None = None,
    output_base: str | Path | None = None,
    x_min: int | None = None,
    y_min: int | None = None,
    x_max: int | None = None,
    y_max: int | None = None,
    oversampling_factor: int = 1,
    reference_ffi_basename_expected: str | None = None,
    cluster_job_json_path: str | None = None,
    allow_reference_ffi_mismatch: bool = False,
    progress_path: str | Path | None = None,
    n_jobs: int = 16,
    skycells_per_batch: int = 20,
    event_dir: str | Path | None = None,
    write_ps1_removed_stars_csv: bool = True,
    removed_stars_csv: str | Path | None = None,
    log_level: str = "INFO",
    stage_regmaps_to_scratch: bool | None = None,
    checkpoint_skycells: bool = False,
) -> dict:
    # Resolve base paths (allow overrides)
    """Main.
    
    Parameters
    ----------
    sector : int, optional, default ``20``
    camera : int, optional, default ``3``
    ccd : int, optional, default ``3``
    offsets : np.ndarray, optional, default ``np.array([[0.0, 0.0]])``
    ignore_mask_bits : list[int], optional, default ``[12]``
    data_root : str | Path, optional, default ``'data'``
    mapping_dir : str | Path | None, optional, default ``None``
    convolved_dir : str | Path | None, optional, default ``None``
    output_base : str | Path | None, optional, default ``None``
    x_min : int | None, optional, default ``None``
    y_min : int | None, optional, default ``None``
    x_max : int | None, optional, default ``None``
    y_max : int | None, optional, default ``None``
    oversampling_factor : int, optional, default ``1``
    reference_ffi_basename_expected : str | None, optional, default ``None``
    cluster_job_json_path : str | None, optional, default ``None``
    allow_reference_ffi_mismatch : bool, optional, default ``False``
    progress_path : str | Path | None, optional, default ``None``
    n_jobs : int, optional, default ``16``
    skycells_per_batch : int, optional, default ``20``
    event_dir : str | Path | None, optional, default ``None``
    write_ps1_removed_stars_csv : bool, optional, default ``True``
    removed_stars_csv : str | Path | None, optional, default ``None``
    log_level : str, optional, default ``'INFO'``
    stage_regmaps_to_scratch : bool | None, optional, default ``None``
        When None, auto-enable on Condor (``_CONDOR_SCRATCH_DIR`` in env).
    checkpoint_skycells : bool, optional, default ``False``
        Write per-skycell sparse NPZ checkpoints under ``{output_dir}/_partial/``.
    
    Returns
    -------
    dict"""
    data_root = Path(data_root)
    log_level = (log_level or "INFO").upper()
    if log_level not in ("INFO", "DEBUG"):
        raise ValueError(f"log_level must be INFO or DEBUG, got {log_level!r}")
    run_t0 = time.perf_counter()
    if mapping_dir is None:
        mapping_root = data_root / "skycell_pixel_mapping"
    else:
        mapping_root = Path(mapping_dir)
    if convolved_dir is None:
        convolved_dir = data_root / "convolved_results"
    else:
        convolved_dir = Path(convolved_dir)
    if output_base is None:
        output_base = data_root / "shifted_downsampled"
    else:
        output_base = Path(output_base)

    # Generate paths based on parameters
    # mapping_root from resolve_config is already the SCC oversampling leaf.
    suffix = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""
    csv_flat = mapping_root / f"tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list{suffix}.csv"
    csv_nested = mapping_root / f"sector_{sector:04d}/camera_{camera}/ccd_{ccd}/tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list{suffix}.csv"
    if oversampling_factor > 1 and not csv_flat.is_file() and not csv_nested.is_file():
        # Legacy: oversampling nest outside sector tree
        alt_root = mapping_root / f"oversampling_{oversampling_factor}"
        csv_flat = alt_root / f"tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list{suffix}.csv"
        csv_nested = alt_root / f"sector_{sector:04d}/camera_{camera}/ccd_{ccd}/tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list{suffix}.csv"
        if csv_flat.is_file() or csv_nested.is_file():
            mapping_root = alt_root
    SKYCELL_CSV_PATH = csv_flat if csv_flat.is_file() or not csv_nested.is_file() else csv_nested
    CONVOLVED_DATA_PATH = Path(convolved_dir)
    leaf = SKYCELL_CSV_PATH.parent
    REG_FILES_PATTERN = str(leaf / "*.fits.gz")
    REG_MASTER_FILES_PATH = str(
        leaf / f"tess_s{sector:04d}_{camera}_{ccd}_master_pixels2skycells{suffix}.fits.gz"
    )
    OUTPUT_DIR = output_base / f"sector{sector:04d}_camera{camera}_ccd{ccd}"

    # Processing parameters - lower n_jobs / skycells_per_batch for full-FFI runs
    N_JOBS = max(1, int(n_jobs))
    SKYCELLS_PER_BATCH = max(1, int(skycells_per_batch))
    print(f"Parallel workers: n_jobs={N_JOBS}, skycells_per_batch={SKYCELLS_PER_BATCH}")

    # Load TESS data and WCS
    print("Loading TESS data and WCS...")
    start_time = time.time()
    tess_wcs, tess_dims = load_tess_wcs(REG_MASTER_FILES_PATH)
    with fits.open(REG_MASTER_FILES_PATH) as hdul:
        # Find HDU with data
        hdu_idx = 1 if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else 0
        tess_data = hdul[hdu_idx].data.astype(np.float32)
        tess_header = hdul[hdu_idx].header

    if reference_ffi_basename_expected:
        expected_raw = os.path.basename(str(reference_ffi_basename_expected).strip())
        tess_ffi_raw = tess_header.get("TESS_FFI")
        actual_raw = (
            os.path.basename(str(tess_ffi_raw).strip())
            if tess_ffi_raw not in (None, "")
            else ""
        )
        expected_bn = strip_fits_suffix(expected_raw)
        actual_bn = strip_fits_suffix(actual_raw) if actual_raw else ""
        json_note = f" ({cluster_job_json_path})" if cluster_job_json_path else ""
        if not actual_bn or expected_bn != actual_bn:
            actual_display = repr(actual_raw) if actual_raw else "(missing)"
            msg = (
                "Reference FFI basename from cluster job JSON does not match master mapping TESS_FFI. "
                f"expected_basename={expected_raw!r} (from job JSON{json_note}), "
                f"mapping TESS_FFI basename={actual_display} "
                f"(mapping file: {REG_MASTER_FILES_PATH})"
            )
            if allow_reference_ffi_mismatch:
                warnings.warn(f"{msg} Continuing because allow_reference_ffi_mismatch=True.", UserWarning, stacklevel=1)
            else:
                raise ValueError(msg)

    if oversampling_factor < 1:
        raise ValueError("oversampling_factor must be >= 1")

    if oversampling_factor > 1:
        if tess_data.shape[0] % oversampling_factor != 0 or tess_data.shape[1] % oversampling_factor != 0:
            raise ValueError("Oversampled mapping dimensions are not divisible by oversampling_factor")
        base_shape = (tess_data.shape[0] // oversampling_factor, tess_data.shape[1] // oversampling_factor)
    else:
        base_shape = tess_data.shape

    # ROI is always in base TESS pixel coordinates [min, max)
    roi_values = [x_min, y_min, x_max, y_max]
    if any(value is not None for value in roi_values) and not all(value is not None for value in roi_values):
        raise ValueError("Provide all ROI bounds together: x_min, y_min, x_max, y_max")

    if all(value is None for value in roi_values):
        x_min, y_min, x_max, y_max = 0, 0, base_shape[1], base_shape[0]
    else:
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

    if not (0 <= x_min < x_max <= base_shape[1] and 0 <= y_min < y_max <= base_shape[0]):
        raise ValueError(f"Invalid ROI bounds for base shape {base_shape}: " f"x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    roi_bounds = (x_min, y_min, x_max, y_max)
    print(f"Using ROI (base TESS scale): x=[{x_min},{x_max}), y=[{y_min},{y_max})")

    # Build an output directory name that includes ROI bounds and oversampling
    roi_suffix = ""
    if not (x_min == 0 and y_min == 0 and x_max == base_shape[1] and y_max == base_shape[0]):
        roi_suffix = f"_x{x_min}-{x_max}_y{y_min}-{y_max}"
    os_suffix = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""

    OUTPUT_DIR = output_base / f"sector{sector:04d}_camera{camera}_ccd{ccd}{roi_suffix}{os_suffix}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load skycell info
    print("Loading skycell info...")
    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    skycell_df = pd.read_csv(SKYCELL_CSV_PATH, usecols=usecols)

    # Prefilter skycells to only those present in ROI, using master mapping IDs.
    if oversampling_factor > 1:
        roi_mapping = tess_data[y_min * oversampling_factor : y_max * oversampling_factor, x_min * oversampling_factor : x_max * oversampling_factor]
    else:
        roi_mapping = tess_data[y_min:y_max, x_min:x_max]

    roi_ids = np.unique(roi_mapping.astype(np.int64))
    roi_ids = roi_ids[roi_ids >= 0]

    if len(roi_ids) > 0:
        roi_ids = roi_ids[roi_ids < len(skycell_df)]
        roi_names = set(skycell_df.iloc[roi_ids]["NAME"].tolist())
        skycell_df = skycell_df[skycell_df["NAME"].isin(roi_names)].reset_index(drop=True)
        print(f"Prefiltered to {len(skycell_df)} ROI-intersecting skycells")
    else:
        print("No mapped skycells found in ROI; output will be empty.")
        skycell_df = skycell_df.iloc[0:0].copy()

    # Load Zarr metadata once for efficient access
    print("Loading Zarr metadata...")
    zarr_path = load_zarr_metadata(sector, camera, ccd, CONVOLVED_DATA_PATH)
    require_convolved_zarr_data(zarr_path)
    # print(f"Found {len(zarr_metadata['cells'])} cells in Zarr store")

    # Precompute shifts for all offsets
    print("Precomputing shifts for all offsets...")
    t_shifts = time.perf_counter()
    if progress_path is not None:
        set_downsample_progress_phase(
            progress_path,
            "precomputing_shifts",
            oversampling_factor=oversampling_factor,
        )
    shifts_dict = precompute_shifts_for_offsets(
        tess_wcs, skycell_df, offsets, progress_path=progress_path
    )
    shifts_elapsed = time.perf_counter() - t_shifts

    # Get registration files
    print("Getting registration files...")
    t_batches = time.perf_counter()
    reg_files_all = sorted(glob(REG_FILES_PATTERN))
    reg_files = [f for f in reg_files_all if "master_pixels2skycells" not in Path(f).name]
    skycell_names = [extract_skycell_name_from_reg_file(f) for f in reg_files]

    # Keep only registration files for ROI-intersecting skycells.
    allowed_names = set(skycell_df["NAME"].tolist())
    filtered_pairs = [(rf, sn) for rf, sn in zip(reg_files, skycell_names) if sn is not None and sn in allowed_names]
    reg_files = [rf for rf, _ in filtered_pairs]
    skycell_names = [sn for _, sn in filtered_pairs]

    # Split into batches
    num_batches = (len(reg_files) + SKYCELLS_PER_BATCH - 1) // SKYCELLS_PER_BATCH if len(reg_files) > 0 else 0
    print(f"Processing {len(reg_files)} skycells in {num_batches} batches...")

    if num_batches > 0:
        reg_batches = np.array_split(reg_files, num_batches)
        name_batches = np.array_split(skycell_names, num_batches)
    else:
        reg_batches = []
        name_batches = []

    total_skycells = len(reg_files)
    if progress_path is not None and num_batches > 0:
        batch_sizes = [len(batch) for batch in reg_batches]
        init_downsample_progress(
            progress_path,
            total_skycells,
            batch_sizes,
            oversampling_factor=oversampling_factor,
        )

    do_stage_regmaps = resolve_stage_regmaps_to_scratch(stage_regmaps_to_scratch)
    if do_stage_regmaps and reg_files:
        try:
            staged_paths, scratch_dir, n_staged, stage_elapsed = stage_regmap_files_to_scratch(
                reg_files,
                sector=sector,
                camera=camera,
                ccd=ccd,
                oversampling_factor=oversampling_factor,
            )
            reg_files = staged_paths
            if num_batches > 0:
                reg_batches = np.array_split(reg_files, num_batches)
            print(
                f"[downsample] staged {n_staged}/{len(staged_paths)} regmaps to scratch "
                f"{scratch_dir} in {stage_elapsed:.1f}s (zarr stays on NFS)"
            )
        except OSError as exc:
            if getattr(exc, "errno", None) != errno.ENOSPC:
                raise
            # Undersized Condor scratch: drop partial copies and continue on NFS.
            os_suffix = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""
            scratch_dir = (
                resolve_downsample_scratch_dir()
                / f"syndiff_downsample_regmaps_{sector:04d}_{camera}_{ccd}{os_suffix}"
            )
            if scratch_dir.is_dir():
                shutil.rmtree(scratch_dir, ignore_errors=True)
            warnings.warn(
                f"[downsample] scratch staging hit ENOSPC ({exc}); "
                "continuing with NFS regmap paths",
                UserWarning,
                stacklevel=1,
            )
            print(
                "[downsample] scratch staging disabled after ENOSPC; "
                "using NFS regmap paths"
            )

    partial_dir = checkpoint_dir_for_output(OUTPUT_DIR) if checkpoint_skycells else None
    if partial_dir is not None:
        completed = scan_completed_skycell_checkpoints(partial_dir)
        if completed:
            print(
                f"[downsample] resume: {len(completed)} skycell checkpoints in {partial_dir}"
            )

    # Process batches in parallel
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_skycell_batch)(
            i,
            reg_batch,
            name_batch,
            offsets,
            shifts_dict,
            base_shape,
            zarr_path,
            roi_bounds,
            oversampling_factor=oversampling_factor,
            ignore_mask_bits=ignore_mask_bits,
            progress_path=progress_path,
            log_level=log_level,
            total_batches=num_batches,
            checkpoint_dir=partial_dir,
            checkpoint_skycells=checkpoint_skycells,
        )
        for i, (reg_batch, name_batch) in enumerate(zip(reg_batches, name_batches))
    )
    batches_elapsed = time.perf_counter() - t_batches

    # Combine results using the sparse array approach
    t_combine = time.perf_counter()
    if progress_path is not None and total_skycells > 0:
        set_downsample_progress_phase(
            progress_path, "combining", total_skycells=total_skycells
        )
    print("Combining results...")
    combined_results = combine_sparse_downsample_results(
        results,
        offsets,
        base_shape,
        roi_bounds,
        oversampling_factor=oversampling_factor,
    )
    if not np.any(combined_results):
        raise RuntimeError("No PS1 convolved data loaded for any skycell")
    combine_elapsed = time.perf_counter() - t_combine

    # Save outputs as FITS files
    t_save = time.perf_counter()
    if progress_path is not None and total_skycells > 0:
        set_downsample_progress_phase(
            progress_path, "saving", total_skycells=total_skycells
        )
    print("Saving outputs...")
    written_paths = save_fits_outputs(output_dir=OUTPUT_DIR, sector=sector, camera=camera, ccd=ccd, results=combined_results, offsets=offsets, tess_header=tess_header, roi_bounds=roi_bounds, oversampling_factor=oversampling_factor)
    save_elapsed = time.perf_counter() - t_save
    total_elapsed = time.perf_counter() - run_t0
    print(
        f"[downsample] timing: shifts={shifts_elapsed:.1f}s "
        f"batches={batches_elapsed:.1f}s combine={combine_elapsed:.1f}s "
        f"save={save_elapsed:.1f}s total={total_elapsed:.1f}s"
    )

    # Record processing time
    total_time = time.time() - start_time
    print(f"Done! Total processing time: {total_time / 60:.2f} minutes")

    # Print summary information
    print(f"Processing completed at: {time.ctime()}")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Processed {len(reg_files)} skycells in {num_batches} batches")
    print(f"Generated {len(offsets)} shifted images")
    print(f"Ignored mask bits: {ignore_mask_bits}")
    print("Shifts processed:")
    for dx, dy in offsets:
        print(f"  dx={dx:.3f}, dy={dy:.3f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    if progress_path is not None and total_skycells > 0:
        set_downsample_progress_phase(
            progress_path, "complete", total_skycells=total_skycells
        )

    artifacts = [str(p) for p in written_paths]
    expected_count = len(offsets)
    produced_count = len(written_paths)

    if event_dir and cluster_job_json_path and write_ps1_removed_stars_csv:
        event_dir_p = Path(event_dir)
        event_dir_p.mkdir(parents=True, exist_ok=True)
        removed_path = (
            Path(removed_stars_csv).expanduser().resolve()
            if removed_stars_csv
            else default_ps1_process_removed_stars_csv_path(
                convolved_dir, sector, camera, ccd
            ).resolve()
        )
        csv_out = event_dir_p / PS1_REMOVED_STARS_CSV_FILENAME
        if not removed_path.is_file():
            warnings.warn(
                f"PS1 removed-star Gaia CSV skipped: file not found: {removed_path}",
                UserWarning,
                stacklevel=1,
            )
        else:
            write_ps1_removed_star_gaia_csv(
                job_json_path=cluster_job_json_path,
                removed_stars_csv=removed_path,
                event_dir=event_dir_p,
                sector=sector,
                camera=camera,
                ccd=ccd,
                roi_bounds=roi_bounds,
            )
            artifacts.append(str(csv_out))
            expected_count += 1
            produced_count += 1

    return {
        "output_dir": str(OUTPUT_DIR),
        "artifacts": artifacts,
        "expected_count": expected_count,
        "produced_count": produced_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-offset downsampling runner")
    parser.add_argument(
        "sector",
        nargs="?",
        type=int,
        default=20,
        help="TESS sector (default: 20; with --job-json, JSON wins—see warning if this differs)",
    )
    parser.add_argument(
        "camera",
        nargs="?",
        type=int,
        default=3,
        help="Camera (default: 3; with --job-json, JSON wins—see warning if this differs)",
    )
    parser.add_argument(
        "ccd",
        nargs="?",
        type=int,
        default=3,
        help="CCD (default: 3; with --job-json, JSON wins—see warning if this differs)",
    )
    parser.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parent / "data"), help="Root data directory")
    parser.add_argument("--mapping-dir", type=str, default=None, help="Skycell pixel mapping directory (default: data-root/skycell_pixel_mapping)")
    parser.add_argument("--convolved-dir", type=str, default=None, help="Convolved results directory (overrides data-root/convolved_results)")
    parser.add_argument("--output-base", type=str, default=None, help="Base output directory (overrides data-root/shifted_downsamples)")
    parser.add_argument("--x-min", type=int, default=None, help="ROI xmin in base TESS pixels (inclusive)")
    parser.add_argument("--y-min", type=int, default=None, help="ROI ymin in base TESS pixels (inclusive)")
    parser.add_argument("--x-max", type=int, default=None, help="ROI xmax in base TESS pixels (exclusive)")
    parser.add_argument("--y-max", type=int, default=None, help="ROI ymax in base TESS pixels (exclusive)")
    parser.add_argument("--oversampling-factor", type=int, default=1, help="Oversampling factor for reading mapping files and index decoding")
    parser.add_argument(
        "--job-json",
        type=str,
        default=None,
        help="Path to cluster_template_job.json (offsets, sector/camera/ccd, and ROI when bounds omitted)",
    )
    parser.add_argument(
        "--removed-stars-csv",
        type=str,
        default=None,
        help=(
            "PS1 pipeline removed_stars CSV (default: "
            "{data-root}/convolved_results/sector_{s:04d}_camera_{c}_ccd_{k}_removed_stars.csv). "
            "Only used when --job-json is set and --skip-ps1-removed-star-gaia-csv is not set."
        ),
    )
    parser.add_argument(
        "--skip-ps1-removed-star-gaia-csv",
        action="store_true",
        help=(
            "Skip writing event_dir/ps1_removed_stars.csv (crop-local Gaia for PS1 removed stars). "
            "By default this step runs only when --job-json is provided (not with --single-offset)."
        ),
    )
    parser.add_argument(
        "--single-offset",
        action="store_true",
        help="Use only [0.0, 0.0] for fast testing; ignores --job-json offsets",
    )
    parser.add_argument(
        "--progress-path",
        type=str,
        default=None,
        help="Write downsample.progress.json sidecar for syndiff progress monitoring",
    )
    args = parser.parse_args()

    roi_cli = [args.x_min, args.y_min, args.x_max, args.y_max]
    if any(v is not None for v in roi_cli) and not all(v is not None for v in roi_cli):
        parser.error("Provide all four ROI bounds together: --x-min, --y-min, --x-max, --y-max")

    x_min, y_min, x_max, y_max = args.x_min, args.y_min, args.x_max, args.y_max
    sector, camera, ccd = args.sector, args.camera, args.ccd

    reference_ffi_basename_expected: str | None = None
    cluster_job_json_path: str | None = None

    if args.single_offset:
        offsets = np.array([[0.0, 0.0]], dtype=np.float64)
    elif args.job_json:
        cluster_job_json_path = str(Path(args.job_json).resolve())
        payload = load_cluster_template_job_payload(args.job_json)
        js_sec, js_cam, js_ccd = instrument_tuple_from_cluster_job_payload(payload)
        if (args.sector, args.camera, args.ccd) != (js_sec, js_cam, js_ccd):
            warnings.warn(
                f"Using sector/camera/ccd from --job-json ({js_sec}, {js_cam}, {js_ccd}); "
                f"CLI positionals ({args.sector}, {args.camera}, {args.ccd}) are ignored.",
                UserWarning,
                stacklevel=1,
            )
        sector, camera, ccd = js_sec, js_cam, js_ccd
        offsets = offsets_from_cluster_job_payload(payload)
        if all(v is None for v in roi_cli):
            x_min, y_min, x_max, y_max = roi_tuple_from_cluster_job_payload(payload)
        _ref = payload.get("reference_ffi_basename")
        if isinstance(_ref, str) and _ref.strip():
            reference_ffi_basename_expected = _ref.strip()
    else:
        parser.error("Provide --job-json or --single-offset")

    # Set mask bits to ignore (0-indexed)
    ignore_mask_bits = [12]

    event_dir = (
        str(Path(cluster_job_json_path).resolve().parent)
        if cluster_job_json_path
        else None
    )

    main(
        sector=sector,
        camera=camera,
        ccd=ccd,
        offsets=offsets,
        ignore_mask_bits=ignore_mask_bits,
        data_root=args.data_root,
        mapping_dir=args.mapping_dir,
        convolved_dir=args.convolved_dir,
        output_base=args.output_base,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        oversampling_factor=args.oversampling_factor,
        reference_ffi_basename_expected=reference_ffi_basename_expected,
        cluster_job_json_path=cluster_job_json_path,
        event_dir=event_dir,
        write_ps1_removed_stars_csv=not args.skip_ps1_removed_star_gaia_csv,
        removed_stars_csv=args.removed_stars_csv,
        progress_path=args.progress_path,
    )
