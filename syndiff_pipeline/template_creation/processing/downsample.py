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

When ``event_dir`` and ``cluster_job_json_path`` are set, helpers such as
``write_ps1_removed_star_gaia_csv`` can still write
``{event_dir}/ps1_removed_stars.csv``. The linear/event ``main()`` entry point
has been removed; use orchestrated field downsample
(``run_field_downsample_scc`` / ``geometry_mode=field``).
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
from joblib import delayed
from tqdm import tqdm

# Import from existing script
from syndiff_pipeline.common.fits_io import write_hdul_fits
from syndiff_pipeline.common.fits_variants import (
    is_fits_storage_filename,
    iter_fits_variant_globs,
    storage_suffix_rank,
    strip_fits_storage_suffix,
)
from syndiff_pipeline.common.wcs_grouping import open_fits_memmap, resolve_existing_fits_path
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    PIPELINE_FITS_EXT,
    strip_fits_suffix,
)
from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import RELEVANT_WCS_KEYS, build_ps1_wcs, compute_ps1_shift_for_skycell, load_tess_wcs
from syndiff_pipeline.template_creation.processing.downsample_progress import (
    init_progress as init_downsample_progress,
    mark_skycells_done as mark_downsample_skycells_done,
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
    Copy ROI regmap FITS onto local scratch **as-is** (keep ``.fits.fz`` /
    ``.fits.gz`` compressed).

    Gunzipping full-chip osN regmaps would require hundreds of GB; ``fits.open``
    reads compressed FITS directly. Convolved Zarr stays on shared NFS.

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

    # mask dtype on disk may be narrower (e.g. uint8) than the ignore_mask bit
    # position being tested (e.g. bit 12 = 4096); upcast before the bitwise AND
    # so numpy's strict same-dtype casting doesn't reject the Python int.
    ignored = (mask_grouped.astype(np.int64, copy=False) & int(ignore_mask)) > 0
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

    # Decode sparse linearized TESS indices into ROI-local (out_y, out_x).
    # Legacy event-ROI linear path only; field mode uses MappingGrid contribs.
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
    if not np.any(valid):
        return combined_results

    out_y = out_y[valid].astype(np.int32, copy=False)
    out_x = out_x[valid].astype(np.int32, copy=False)
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

        output_filename = (
            output_dir
            / f"syndiff_template_s{sector:04d}_{camera}_{ccd}{roi_part}{os_part}"
            f"_dx{dx:.3f}_dy{dy:.3f}{PIPELINE_FITS_EXT}"
        )
        written_paths.append(write_hdul_fits(output_filename, hdu_list))

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
    """Linear/event ROI downsample entry point — removed.

    Orchestrated field downsample is the only supported path
    (``run_field_downsample_scc`` via ``syndiff template`` /
    ``geometry_mode=field``).
    """
    raise NotImplementedError(
        "downsample.main() linear/event ROI path was removed; use orchestrated "
        "field downsample (geometry_mode='field' / run_field_downsample_scc)"
    )


if __name__ == "__main__":
    raise SystemExit(
        "downsample.py CLI linear/event ROI path was removed; use "
        "`syndiff template` with geometry_mode=field / run_field_downsample_scc"
    )
