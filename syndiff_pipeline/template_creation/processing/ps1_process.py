"""
Modern Sliding Window Pipeline Implementation

Refactored to use a high-throughput, memory-efficient, four-stage pipeline.
This version uses parallel upstream workers for data ingestion and preprocessing,
feeding a single, sequential assembler that processes one projection at a time
to correctly manage the sliding window state.
"""

import atexit
import faulthandler
import gc
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import queue as _thread_queue
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty, Full
from typing import Any, Optional
import warnings

faulthandler.enable()

import numpy as np
import pandas as pd
import zarr
from astropy.wcs import FITSFixedWarning

# Import existing utilities
from syndiff_pipeline.template_creation.processing import band_utils, convolution_utils, zarr_utils
from syndiff_pipeline.template_creation.processing.band_utils import compute_tess_mag, process_skycell_bands, remove_background
from syndiff_pipeline.template_creation.processing.correct_saturation import apply_saturation_to_row
from syndiff_pipeline.template_creation.processing.cross_projection_padding import apply_cross_projection_padding, identify_all_padding_sources
from syndiff_pipeline.template_creation.processing.csv_utils import find_csv_file, get_projections_from_csv, load_csv_data
from syndiff_pipeline.template_creation.processing.ps1_download import fetch_skycell_bands_masks_and_headers
from syndiff_pipeline.template_creation.processing.zarr_utils import load_skycell_bands_masks_and_headers


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FITSFixedWarning)

_child_processes = []
_active_executor: Optional[ProcessPoolExecutor] = None


def _cleanup_child_processes():
    """atexit handler: kill any surviving ProcessPoolExecutor workers and tracked
    child processes so they don't linger as orphans after the main process exits."""
    global _active_executor
    if _active_executor is not None:
        try:
            _active_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # Forcefully terminate worker processes via the executor's internal list
        for proc in (getattr(_active_executor, "_processes", None) or {}).values():
            try:
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
        _active_executor = None
    for p in _child_processes:
        try:
            if p.is_alive():
                p.kill()
        except Exception:
            pass
    _child_processes.clear()


atexit.register(_cleanup_child_processes)


# Key constants from the specification
CELL_OVERLAP = 480
EDGE_EXCLUSION = 10
EFFECTIVE_OVERLAP = CELL_OVERLAP - EDGE_EXCLUSION
PAD_SIZE = 480

GATHER_TIMEOUT_SECONDS = 180
MAX_ACTIVE_TASKS = 35  # Maximum concurrent preprocessing tasks
MAX_TOTAL_PENDING_WORK = 30  # Maximum total pending work (queue + buffer + active tasks)
MIN_AVAILABLE_MEMORY_FRACTION = 0.15  # Pause task submission when available RAM drops below this fraction of total

# Per-cell shared convolved-store canonical-skip (see
# ``ps1_process_percell_skip_plan.md``). Below this fraction of a row's
# cells actually needing (re)convolution, prefer local ±radius windowed
# Gaussian convolution per contiguous run of missing cells over falling back
# to the old whole-row convolution. Not a config knob -- the skip itself is
# unconditional/always-on; this constant only controls the crossover point
# between "many small local windows" and "one full-row pass", chosen from
# the FLOP-vs-call-overhead analysis in the plan doc (local windows stay
# FLOP-cheaper than a full row up to ~85-90% missing; below that, the
# dominant cost at low missing-fractions is call/I-O overhead, not FLOPs).
ROW_FALLBACK_THRESHOLD = 0.2

def calculate_total_buffer_size(cell_buffer: dict) -> int:
    """Calculate total number of cells in the buffer across all projection/row combinations."""
    return sum(len(cells) for cells in cell_buffer.values())


def load_gaia_catalog(data_root: str, sector: int, camera: int, ccd: int, catalog_path: Optional[str] = None) -> pd.DataFrame:
    """Load Gaia catalog for the specified sector/camera/ccd."""
    if catalog_path is None:
        from syndiff_pipeline.common.scc_paths import default_gaia_catalog_path

        catalog_path = str(default_gaia_catalog_path(data_root, sector, camera, ccd))

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    logger.info(f"Loading catalog from {catalog_path}")
    catalog = pd.read_csv(catalog_path)

    if "source_id" in catalog.columns:
        catalog["source_id"] = pd.to_numeric(
            catalog["source_id"], errors="coerce"
        ).astype("Int64")

    # Validate required columns (G, BP, RP all needed for tess_mag computation).
    required_cols = ["ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
    missing_cols = [col for col in required_cols if col not in catalog.columns]
    if missing_cols:
        raise ValueError(f"Catalog missing required columns: {missing_cols}")

    if "source_id" not in catalog.columns:
        logger.warning(
            f"[Pipeline] Catalog at {catalog_path} has no 'source_id' column; "
            f"source_id will be NaN in removed-star records."
        )

    logger.info(f"Loaded {len(catalog)} stars from catalog")
    return catalog


# --- Data Structures (Retained) ---
@dataclass
class MasterArrayConfig:
    """Configuration for master arrays."""

    width: int
    height: int
    cell_width: int
    cell_height: int
    max_cells: int
    starting_x: int
    cell_dimensions: dict


@dataclass
class ProcessingState:
    """Current processing state for sliding window."""

    current_array: Optional[np.ndarray] = None
    next_array: Optional[np.ndarray] = None
    current_masks: dict[str, np.ndarray] = None
    next_masks: dict[str, np.ndarray] = None
    current_row_id: Optional[int] = None
    next_row_id: Optional[int] = None
    cell_locations: dict[str, tuple[int, int, int, int]] = None
    next_cell_locations: dict[str, tuple[int, int, int, int]] = None
    # Per-cell lineage metadata ({"headers_data": ..., "removed_stars": ...})
    # carried alongside cell_locations/current_masks so the shared
    # convolved-store canonical snapshot (plan §13, decision #3) can publish
    # with the exact headers/removed-star records the cell's combined-store
    # publish used, without re-reading anything from disk. Swapped in lockstep
    # with cell_locations/current_masks in advance_sliding_window.
    cell_metadata: dict[str, dict] = None
    next_cell_metadata: dict[str, dict] = None

    def __post_init__(self):
        """Post init."""
        if self.current_masks is None:
            self.current_masks = {}
        if self.next_masks is None:
            self.next_masks = {}
        if self.cell_metadata is None:
            self.cell_metadata = {}
        if self.next_cell_metadata is None:
            self.next_cell_metadata = {}
        if self.cell_locations is None:
            self.cell_locations = {}
        if self.next_cell_locations is None:
            self.next_cell_locations = {}


# --- Core Logic Functions (Retained and Modified) ---


def initialize_processing_state(config: MasterArrayConfig) -> ProcessingState:
    """Initialize processing state with empty master arrays."""
    current_array = np.full((config.height, config.width), np.nan, dtype=np.float32)
    next_array = np.full((config.height, config.width), np.nan, dtype=np.float32)
    return ProcessingState(current_array=current_array, next_array=next_array)


def advance_sliding_window(state: ProcessingState) -> None:
    """Advance the sliding window (next becomes current)."""
    state.current_array, state.next_array = state.next_array, state.current_array
    state.next_array.fill(np.nan)
    state.current_masks, state.next_masks = state.next_masks, {}
    state.current_row_id = state.next_row_id
    state.next_row_id = None
    state.cell_locations.clear()
    state.cell_locations.update(state.next_cell_locations)
    state.next_cell_locations.clear()
    state.cell_metadata.clear()
    state.cell_metadata.update(state.next_cell_metadata)
    state.next_cell_metadata.clear()
    gc.collect()


def apply_cross_row_padding(state: ProcessingState, config: MasterArrayConfig) -> None:
    """Apply cross-row padding between current and next arrays with correct overlap handling."""
    if state.next_array is None:
        return

    # Source region from next row
    next_source_y_start = PAD_SIZE + CELL_OVERLAP - EDGE_EXCLUSION
    next_source_y_end = PAD_SIZE * 2 + CELL_OVERLAP

    # Target region in current row (top padding area)
    current_target_y_start = config.cell_height - EDGE_EXCLUSION + PAD_SIZE
    state.current_array[current_target_y_start:, :] = state.next_array[next_source_y_start:next_source_y_end, :]

    # Opposite direction: from current to next
    current_source_y_start = config.cell_height - CELL_OVERLAP
    current_source_y_end = PAD_SIZE + config.cell_height - CELL_OVERLAP + EDGE_EXCLUSION
    next_target_y_end = PAD_SIZE + EDGE_EXCLUSION
    state.next_array[:next_target_y_end, :] = state.current_array[current_source_y_start:current_source_y_end, :]


def extract_cell_results(convolved_array: np.ndarray, cell_positions: dict) -> dict[str, np.ndarray]:
    """Extract individual cell results from convolved array."""
    results = {}
    for cell_name, (x_start, x_end, y_start, y_end) in cell_positions.items():
        results[cell_name] = convolved_array[y_start:y_end, x_start:x_end].copy()
    return results


def extract_projection_metadata(df: pd.DataFrame, projection: str) -> dict:
    """Extract projection metadata from a pre-loaded DataFrame."""
    proj_df = df[df["projection"].astype(str) == projection]
    if proj_df.empty:
        raise ValueError(f"No data found for projection {projection}")

    rows = {}
    cell_dimensions = {}
    starting_x = 10
    for _, row in proj_df.iterrows():
        row_id = int(row["y"])
        cell_name = row["NAME"]
        x_coord = int(row["x"])
        starting_x = x_coord if x_coord < starting_x else starting_x
        cell_width = int(row.get("NAXIS1"))
        cell_height = int(row.get("NAXIS2"))
        cell_dimensions[cell_name] = (cell_width, cell_height)
        if row_id not in rows:
            rows[row_id] = []
        rows[row_id].append((cell_name, x_coord))

    for row_id in rows:
        rows[row_id].sort(key=lambda x: x[1])

    all_dims = list(cell_dimensions.values())
    if len(set(all_dims)) > 1:
        logger.warning(f"[Metadata] Inconsistent cell dimensions found: {set(all_dims)}")
    typical_width, typical_height = max(all_dims, key=lambda item: item[0] * item[1]) if all_dims else (0, 0)
    max_cells_per_row = max(len(cells) for cells in rows.values()) if rows else 0

    return {
        "projection": projection,
        "rows": rows,
        "cell_width": typical_width,
        "cell_height": typical_height,
        "max_cells_per_row": max_cells_per_row,
        "starting_x": starting_x,
        "cell_dimensions": cell_dimensions,
        "dataframe": proj_df,
    }


def create_master_array_config(metadata: dict) -> MasterArrayConfig:
    """Create master array configuration from metadata."""
    cell_width = metadata["cell_width"]
    cell_height = metadata["cell_height"]
    max_cells = metadata["max_cells_per_row"]
    starting_x = metadata["starting_x"]
    master_width = PAD_SIZE + (max_cells * (cell_width - CELL_OVERLAP)) + CELL_OVERLAP + PAD_SIZE
    master_height = cell_height + (2 * PAD_SIZE)
    return MasterArrayConfig(width=master_width, height=master_height, cell_width=cell_width, cell_height=cell_height, max_cells=max_cells, starting_x=starting_x, cell_dimensions=metadata["cell_dimensions"])


def create_master_task_list(df: pd.DataFrame, projection: str) -> tuple[dict, list[tuple[str, str, int]]]:
    """Generate the task list for a single projection from a pre-loaded DataFrame."""
    metadata = extract_projection_metadata(df, projection)
    task_list = []
    for row_id in sorted(metadata["rows"].keys()):
        for skycell_id in metadata["rows"][row_id]:
            task_list.append((skycell_id, projection, row_id))
    return metadata, task_list


def expected_convolved_skycells(
    data_root: str,
    sector: int,
    camera: int,
    ccd: int,
    *,
    projections_limit: Optional[int] = None,
    oversampling_factor: int = 1,
    mapping_csv_path: str | None = None,
) -> list[str]:
    """Skycell names ``ps1_process`` must write for the given SCC/config.

    The mapping CSV is the authoritative inventory.  In particular, an OS4
    MAPGRID=3 run must not fall back to the native (OS1) CSV: doing so silently
    drops support cells at the template edges and lets L5 manufacture zeros.
    ``mapping_csv_path`` is accepted for dispatchers that already resolved the
    exact P2 artifact; otherwise the canonical SCC path is selected from
    ``oversampling_factor``.
    """
    if mapping_csv_path is None:
        from syndiff_pipeline.common.scc_paths import scc_mapping_master_skycells_csv

        mapping_csv_path = str(
            scc_mapping_master_skycells_csv(
                data_root, sector, camera, ccd,
                oversampling_factor=int(oversampling_factor),
            )
        )
    csv_path = str(mapping_csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"master skycell inventory missing: {csv_path}")
    projections = get_projections_from_csv(csv_path)
    if projections_limit:
        projections = projections[: int(projections_limit)]
    df = load_csv_data(csv_path)
    required = {"NAME", "projection"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"master skycell inventory missing columns: {sorted(missing)}")
    # MAPGRID=3 carries the geometry contract on every CSV row.  Refuse mixed
    # or partially annotated inventories before any PS1 work is dispatched.
    # pancakes.py now writes MAPGRID/GEOMFP/COORDFRM unconditionally (not just
    # for OS>1), so enforce this check whenever the columns are present,
    # regardless of oversampling_factor.  OS>1 still hard-requires the
    # columns to exist at all; OS1 stays permissive on missing columns only
    # for backward compatibility with pre-MAPGRID3 OS1 inventories elsewhere
    # in the data root that predate this contract.
    geometry_columns = ("MAPGRID", "GEOMFP", "COORDFRM")
    has_geometry_columns = all(column in df.columns for column in geometry_columns)
    if int(oversampling_factor) > 1 and not has_geometry_columns:
        missing_columns = [column for column in geometry_columns if column not in df.columns]
        raise ValueError(
            f"OS{int(oversampling_factor)} master inventory lacks {missing_columns}; "
            "rebuild P2 mapping with MAPGRID=3"
        )
    if has_geometry_columns:
        if set(df["MAPGRID"].astype(int)) != {3}:
            raise ValueError("master inventory must be MAPGRID=3; refusing PS1 processing")
        if set(df["COORDFRM"].astype(str)) != {"full_ffi"}:
            raise ValueError("master inventory is not in the full_ffi coordinate frame")
        if df["GEOMFP"].astype(str).nunique() != 1:
            raise ValueError("master inventory has mixed geometry fingerprints")
    # NAME is the published P2 identity column.  Do not reconstruct identities
    # through row geometry here: MAPGRID=3 edge rows may intentionally have a
    # different support shape, while their skycell names remain authoritative.
    return sorted(
        set(df.loc[df["projection"].astype(str).isin(set(projections)), "NAME"].astype(str))
    )


def master_skycell_inventory(
    data_root: str,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    mapping_csv_path: str | None = None,
) -> dict[str, Any]:
    """Return the immutable P2 skycell identity/provenance inventory.

    This small, side-effect-free API is shared by PS1 processing and stage
    verifiers.  ``names`` is the exact set that must be present in the
    convolved store; ``geometry_fingerprint`` is the P2 contract that callers
    must carry into their stage provenance.
    """
    if mapping_csv_path is None:
        from syndiff_pipeline.common.scc_paths import scc_mapping_master_skycells_csv

        mapping_csv_path = str(
            scc_mapping_master_skycells_csv(
                data_root, sector, camera, ccd,
                oversampling_factor=int(oversampling_factor),
            )
        )
    df = load_csv_data(str(mapping_csv_path))
    names = {str(v) for v in df["NAME"].dropna().astype(str)}
    if not names:
        raise ValueError(f"master skycell inventory is empty: {mapping_csv_path}")
    result: dict[str, Any] = {
        "names": sorted(names),
        "count": len(names),
        "mapping_csv": str(mapping_csv_path),
    }
    for column, key in (("MAPGRID", "mapgrid_version"), ("GEOMFP", "geometry_fingerprint"),
                        ("COORDFRM", "coordinate_frame")):
        if column in df.columns:
            values = sorted({str(v) for v in df[column].dropna()})
            if len(values) != 1:
                raise ValueError(f"master inventory has mixed {column}: {values[:5]}")
            result[key] = int(values[0]) if column == "MAPGRID" else values[0]
    if int(oversampling_factor) > 1 and result.get("mapgrid_version") != 3:
        raise ValueError("OS master inventory must be MAPGRID=3")
    return result


# --- NEW Pipeline Worker Functions ---


def _array_to_shm(arr: np.ndarray, prefix: str) -> dict:
    """Write a NumPy array into a new shared memory block.
    Returns a lightweight descriptor dict (name, shape, dtype) that can be
    cheaply pickled across the ProcessPoolExecutor pipe instead of the full array."""
    shm = SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    desc = {"shm_name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
    shm.close()
    return desc


def _shm_to_array(desc: dict) -> np.ndarray:
    """Reconstruct a NumPy array from a shared memory descriptor and free the block."""
    shm = SharedMemory(name=desc["shm_name"], create=False)
    arr = np.ndarray(desc["shape"], dtype=np.dtype(desc["dtype"]), buffer=shm.buf).copy()
    shm.close()
    shm.unlink()
    return arr


def project_gaia_to_skycell(
    gaia_catalog: pd.DataFrame,
    wcs,
    cell_shape: tuple,
) -> pd.DataFrame:
    """Project a Gaia catalog onto skycell pixel coordinates.

    Converts RA/Dec to pixel (x, y) using the provided astropy WCS, then
    filters to stars whose projected position falls within the skycell
    footprint.  Adds columns ``pixel_x``, ``pixel_y``, and ``tess_mag``.

    Args:
        gaia_catalog: Full or pre-filtered Gaia catalog DataFrame.  Must
            contain ``ra``, ``dec``, ``phot_g_mean_mag``.  ``phot_bp_mean_mag``
            and ``phot_rp_mean_mag`` are used for the colour term when present.
            ``source_id`` is passed through when present.
        wcs: Astropy WCS object for the skycell.
        cell_shape: (height, width) tuple of the skycell image.

    Returns:
        Filtered DataFrame with extra columns: pixel_x, pixel_y, tess_mag.
        Returns an empty DataFrame if projection fails or no stars are in footprint.
    """
    if gaia_catalog is None or len(gaia_catalog) == 0:
        return pd.DataFrame()
    try:
        H, W = cell_shape
        from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

        ra_dec = gaia_catalog[["ra", "dec"]].values
        pixel_x, pixel_y = world_ra_dec_to_pixel(wcs, ra_dec[:, 0], ra_dec[:, 1])

        in_footprint = (
            (pixel_x >= 0) & (pixel_x < W)
            & (pixel_y >= 0) & (pixel_y < H)
        )
        result = gaia_catalog[in_footprint].copy().reset_index(drop=True)
        result["pixel_x"] = pixel_x[in_footprint]
        result["pixel_y"] = pixel_y[in_footprint]

        g  = result["phot_g_mean_mag"].values
        bp = result["phot_bp_mean_mag"].values if "phot_bp_mean_mag" in result.columns else np.full(len(result), np.nan)
        rp = result["phot_rp_mean_mag"].values if "phot_rp_mean_mag" in result.columns else np.full(len(result), np.nan)
        result["tess_mag"] = compute_tess_mag(g, bp, rp)

        logger.debug(
            f"[project_gaia_to_skycell] {len(result)}/{len(gaia_catalog)} stars in footprint"
        )
        return result
    except Exception as e:
        logger.warning(f"[project_gaia_to_skycell] Projection failed: {e}")
        return pd.DataFrame()


def process_single_cell(bundle: dict) -> dict:
    """Run SEP source extraction on a pre-combined cell in a subprocess.

    Expects a bundle with pre-combined arrays (combined_image, combined_mask,
    combined_uncert) produced by band_combiner_worker.  Only the slow
    remove_background / SEP step runs here; band combination already happened
    in-process in the band combiner thread.

    When ``gaia_catalog`` is present in the bundle and ``remove_saturated_stars``
    is True, the Gaia catalog is projected to pixel coordinates before being
    passed to remove_background for catalog-based segment removal.

    Returns results via shared memory descriptors to avoid pickle-pipe deadlocks.
    """
    try:
        import logging

        from astropy.io import fits as afits
        from astropy.wcs import WCS
        from syndiff_pipeline.template_creation.processing.band_utils import remove_background

        logger = logging.getLogger(__name__)
        skycell_id = bundle["skycell_id"]
        remove_saturated_stars = bundle.get("remove_saturated_stars", True)
        bright_star_mag_threshold = bundle.get("bright_star_mag_threshold", 13.0)

        logger.info(f"[PreProcessor] Starting Source Extractor {skycell_id}")

        # Project Gaia catalog to pixel coordinates for this skycell.
        # This must happen before remove_background so the projected positions
        # can be used for catalog-based segment identification.
        gaia_catalog_pixels = None
        wcs = None
        if remove_saturated_stars and bundle.get("gaia_catalog") is not None:
            try:
                header_str = next(iter(bundle["headers_data"].values()))
                wcs = WCS(afits.Header.fromstring(header_str))
                gaia_catalog_pixels = project_gaia_to_skycell(
                    bundle["gaia_catalog"], wcs, bundle["combined_image"].shape
                )
                logger.info(
                    f"[PreProcessor] {len(gaia_catalog_pixels)} Gaia stars projected "
                    f"into footprint of {skycell_id}"
                )
            except Exception as proj_err:
                logger.warning(
                    f"[PreProcessor] Gaia projection failed for {skycell_id}: {proj_err}"
                )

        combined_image, removed_stars_list = remove_background(
            bundle["combined_image"],
            bundle["combined_uncert"],
            mask=bundle["combined_mask"],
            remove_saturated_stars=remove_saturated_stars,
            gaia_catalog_pixels=gaia_catalog_pixels,
            bright_star_mag_threshold=bright_star_mag_threshold,
        )

        # Records from catalog passes already carry Gaia RA/Dec.
        # Just stamp skycell_id on every record.
        for star in removed_stars_list:
            star["skycell_id"] = skycell_id

        image_desc = _array_to_shm(combined_image, f"img_{skycell_id}")
        mask_desc = _array_to_shm(bundle["combined_mask"], f"msk_{skycell_id}")

        result = {
            "skycell_id": skycell_id,
            "projection": bundle["projection"],
            "row_id": bundle["row_id"],
            "x_coord": bundle["x_coord"],
            "combined_image_shm": image_desc,
            "combined_mask_shm": mask_desc,
            "headers_data": bundle["headers_data"],
            "removed_stars": removed_stars_list,
        }

        logger.info(f"[PreProcessor] Processed {skycell_id}")
        return result

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"[PreProcessor] Failed for {bundle.get('skycell_id', '?')}: {e}", exc_info=True)
        return None


def ingest_worker(
    task_queue: Queue,
    raw_cell_queue: Queue,
    *,
    ps1_source: str = "zarr",
    zarr_store=None,
    use_local_files: bool = False,
    local_data_path: str | None = None,
    remove_saturated_stars: bool = True,
    band_cache: dict | None = None,
    combined_store_data_root: str | None = None,
    combined_store_recipe: Any = None,
):
    """Stage 1: Load raw cell data from Zarr (zarr mode) or HTTP (stream mode)."""
    from pathlib import Path

    ingest_label = "Ingest" if ps1_source == "stream" else "Reader"
    local_path = Path(local_data_path) if local_data_path else None

    while True:
        task = task_queue.get()
        if task is None:
            break

        if len(task) == 5:
            skycell_id, projection, row_id, x_coord, task_type = task
        else:
            skycell_id, projection, row_id, x_coord = task
            task_type = "regular"

        if task_type == "regular" and band_cache is not None and skycell_id in band_cache:
            raw_bundle = {
                "skycell_id": skycell_id,
                "projection": projection,
                "row_id": row_id,
                "x_coord": x_coord,
                "task_type": "regular_cache_hit",
            }
            raw_cell_queue.put(raw_bundle)
            logger.info(f"[{ingest_label}] Cache hit for {skycell_id}, skipping load")
            continue

        # Tier-2: lazy, single-cell combined-store lookup (replaces the old
        # eager bulk preload of all regular cells -- see
        # doc/ps1_process_tiered_ingest_architecture_plan.md Part A). Only
        # called for the one cell currently being ingested, so it's bounded
        # by the worker pool size instead of the whole SCC's cell count.
        if (
            task_type == "regular"
            and combined_store_data_root is not None
            and combined_store_recipe is not None
        ):
            from syndiff_pipeline.template_creation.processing.combined_store import (
                seed_band_cache_from_combined_store,
            )

            try:
                hits = seed_band_cache_from_combined_store(
                    combined_store_data_root, [skycell_id], combined_store_recipe
                )
            except Exception:
                hits = {}
                logger.warning(
                    "[%s] Combined-store lookup failed for %s (falling back to raw fetch)",
                    ingest_label,
                    skycell_id,
                    exc_info=True,
                )
            if skycell_id in hits:
                if band_cache is not None:
                    band_cache[skycell_id] = hits[skycell_id]
                raw_bundle = {
                    "skycell_id": skycell_id,
                    "projection": projection,
                    "row_id": row_id,
                    "x_coord": x_coord,
                    "task_type": "regular_cache_hit",
                }
                raw_cell_queue.put(raw_bundle)
                logger.info(f"[{ingest_label}] Combined-store hit for {skycell_id}, skipping raw fetch")
                continue

        try:
            if ps1_source == "stream":
                bands, masks, weights, headers, headers_weight = fetch_skycell_bands_masks_and_headers(
                    skycell_id,
                    use_local_files=use_local_files,
                    local_data_path=local_path,
                )
            else:
                bands, masks, weights, headers, headers_weight = zarr_utils.load_skycell_bands_masks_and_headers(
                    zarr_store, projection, skycell_id
                )
            if not bands:
                logger.warning(f"[{ingest_label}] No band data for {skycell_id}, skipping.")
                continue
            raw_bundle = {
                "skycell_id": skycell_id,
                "projection": projection,
                "row_id": row_id,
                "x_coord": x_coord,
                "task_type": task_type,
                "bands_data": bands,
                "masks_data": masks,
                "headers_data": headers,
                "weights_data": weights,
                "headers_weight_data": headers_weight,
                "remove_saturated_stars": remove_saturated_stars,
            }
            raw_cell_queue.put(raw_bundle)
            verb = "Fetched" if ps1_source == "stream" else "Loaded"
            logger.info(f"[{ingest_label}] {verb} {skycell_id} (type={task_type})")
        except Exception as e:
            logger.error(f"[{ingest_label}] Failed to load {skycell_id}: {e}", exc_info=True)


def reader_worker(task_queue: Queue, raw_cell_queue: Queue, zarr_store, remove_saturated_stars: bool = True, band_cache: dict = None):
    """Backward-compatible alias for zarr-mode ingest."""
    ingest_worker(
        task_queue,
        raw_cell_queue,
        ps1_source="zarr",
        zarr_store=zarr_store,
        remove_saturated_stars=remove_saturated_stars,
        band_cache=band_cache,
    )


def _load_skycell_raw_bands(
    skycell_name: str,
    projection: str,
    ingest_config: dict,
) -> tuple[dict, dict, dict, dict, dict]:
    """Load raw bands/masks/weights using zarr or stream ingest."""
    from pathlib import Path

    ps1_source = ingest_config.get("ps1_source", "zarr")
    if ps1_source == "stream":
        return fetch_skycell_bands_masks_and_headers(
            skycell_name,
            use_local_files=ingest_config.get("use_local_files", False),
            local_data_path=Path(ingest_config["local_data_path"])
            if ingest_config.get("local_data_path")
            else None,
        )

    zarr_path = ingest_config.get("zarr_path")
    zarr_store = zarr.open(zarr_path, mode="r")
    try:
        return load_skycell_bands_masks_and_headers(zarr_store, projection, skycell_name)
    except Exception:
        short_id = skycell_name.split(".")[-1]
        return load_skycell_bands_masks_and_headers(zarr_store, projection, short_id)


def band_combiner_worker(raw_cell_queue: _thread_queue.Queue, combined_raw_queue: _thread_queue.Queue):
    """Stage 1.5: Combines raw bands into a single image+mask+uncert in-process.

    Runs as a thread. Reads large raw bundles (~1.6 GB each with 4 bands × data/mask/weight),
    runs the fast process_skycell_bands (~4s), and outputs a much smaller combined bundle
    (~400 MB: combined_image + combined_mask + combined_uncert) to the next queue.

    Passthrough tasks (regular_cache_hit, padding_source with no band data) are forwarded
    unchanged since they don't carry raw bands.
    """
    while True:
        raw_bundle = raw_cell_queue.get()
        if raw_bundle is None:
            combined_raw_queue.put(None)
            break

        task_type = raw_bundle.get("task_type", "regular")
        skycell_id = raw_bundle["skycell_id"]

        if "bands_data" not in raw_bundle:
            combined_raw_queue.put(raw_bundle)
            continue

        try:
            logger.info(f"[BandCombiner] Combining bands for {skycell_id}")
            combined_image, combined_mask, combined_uncert = process_skycell_bands(
                bands_data=raw_bundle["bands_data"],
                masks_data=raw_bundle["masks_data"],
                weights_data=raw_bundle["weights_data"],
                headers_data=raw_bundle["headers_data"],
                headers_weight_data=raw_bundle["headers_weight_data"],
            )

            reduced_bundle = {
                "skycell_id": skycell_id,
                "projection": raw_bundle["projection"],
                "row_id": raw_bundle["row_id"],
                "x_coord": raw_bundle["x_coord"],
                "task_type": task_type,
                "combined_image": combined_image,
                "combined_mask": combined_mask,
                "combined_uncert": combined_uncert,
                "headers_data": raw_bundle["headers_data"],
                "remove_saturated_stars": raw_bundle.get("remove_saturated_stars", True),
            }

            del raw_bundle
            combined_raw_queue.put(reduced_bundle)
            logger.info(f"[BandCombiner] Combined {skycell_id}")
        except Exception as e:
            logger.error(f"[BandCombiner] Failed for {skycell_id}: {e}", exc_info=True)


def _materialize_shm_result(result: dict) -> dict:
    """Convert a subprocess result that carries shared-memory descriptors into a
    regular result dict with materialised NumPy arrays. If the result already has
    plain arrays (e.g. from a cache-hit fast-path), return it unchanged."""
    if "combined_image_shm" in result:
        result["combined_image"] = _shm_to_array(result.pop("combined_image_shm"))
        result["combined_mask"] = _shm_to_array(result.pop("combined_mask_shm"))
    return result


def process_coordinator(
    combined_raw_queue: _thread_queue.Queue,
    combined_cell_queue: _thread_queue.Queue,
    cell_buffer: dict,
    num_workers: int = 4,
    band_cache: dict = None,
    band_cache_uses: dict = None,
    pipeline_paused_event: threading.Event = None,
    gaia_catalog: Optional[pd.DataFrame] = None,
    bright_star_mag_threshold: float = 13.0,
    combined_store_data_root: Optional[str] = None,
    combined_store_recipe=None,
):
    """Coordinates between the band-combiner output queue and ProcessPoolExecutor
    for source extraction (SEP).

    Reads pre-combined bundles from combined_raw_queue (produced by
    band_combiner_worker threads). Each bundle already contains combined_image,
    combined_mask, combined_uncert — only the slow SEP step runs in subprocess
    workers. This dramatically reduces the memory footprint in the subprocess
    pool because the raw 4-band data (~1.6 GB) has already been compressed to
    ~0.4 GB by the time it reaches here.

    When ``combined_store_data_root``/``combined_store_recipe`` are set, freshly
    computed regular (non padding-role) results are published to the shared
    combined-skycell store (Phase 1, ``combined_store.py``). Best-effort only.

    Does NOT publish to the shared convolved-skycell store. An earlier
    version of this function convolved each isolated regular cell here (zero
    row/neighbor context) and published that as the "same_projection_only"
    canonical cell -- that was architecturally wrong (see plan §13, decision
    #3): the real canonical convolution needs the same-projection-padded ROW
    master array, which only exists in ``process_row_step_from_queue``
    (a different function, running on the main thread, not this
    background-thread coordinator). The correct publish now happens there,
    from a snapshot of ``state.current_array`` taken right after
    ``apply_cross_row_padding`` and before ``apply_cross_projection_padding``.
    See ``_publish_canonical_convolved_snapshot`` below.
    """
    _publish_combined = None
    if combined_store_data_root is not None and combined_store_recipe is not None:
        from syndiff_pipeline.template_creation.processing import combined_store
        from syndiff_pipeline.template_creation.processing.combined_store import (
            _projection_and_cell,
            publish_combined_cell,
            raw_skycell_input_fingerprint,
        )

        def _publish_combined(result: dict) -> Optional[dict]:
            parsed = _projection_and_cell(result["skycell_id"])
            if parsed is None:
                return None
            projection, cell = parsed
            # Same helper, same (data_root, projection, cell) shape as the
            # seed-lookup path in seed_band_cache_from_combined_store, so the
            # two sides stay symmetric (a lookup only ever hits a fingerprint
            # this call site actually published under).
            raw_fp = raw_skycell_input_fingerprint(combined_store_data_root, projection, cell)
            info = publish_combined_cell(
                combined_store_data_root,
                projection,
                cell,
                combined_image=result["combined_image"],
                combined_mask=result["combined_mask"],
                headers_data=result.get("headers_data"),
                removed_stars=result.get("removed_stars"),
                recipe=combined_store_recipe,
                input_fingerprints=[raw_fp],
                producer="ps1_process",
            )
            if info is not None and info.get("fingerprint"):
                # Defense-in-depth (plan Phase 1): select this run's own
                # recipe as "current" for this cell. Readers that check the
                # pointer (e.g. padding_correction, when no recipe context of
                # their own is available) then never fall back to
                # newest-mtime among possibly-unrelated published recipes.
                # This never overrides recipe-matched resolution -- it is
                # only ever a secondary/fallback selection mechanism.
                combined_store.update_current_pointer(
                    combined_store_data_root, projection, cell, info["fingerprint"],
                )
            return info
    logger.info(f"[ProcessCoordinator] Starting with {num_workers} process workers")
    if band_cache is None:
        band_cache = {}
    if band_cache_uses is None:
        band_cache_uses = {}

    pending_results: deque = deque()
    last_coordinator_log_time: float = 0.0
    last_mem_warn_time: float = 0.0

    import psutil
    _min_available_bytes = int(MIN_AVAILABLE_MEMORY_FRACTION * psutil.virtual_memory().total)
    logger.info(
        f"[ProcessCoordinator] Memory guard: will pause submissions when available RAM < "
        f"{_min_available_bytes / (1024**3):.1f} GB ({MIN_AVAILABLE_MEMORY_FRACTION:.0%} of total)"
    )

    global _active_executor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        _active_executor = executor
        futures: dict = {}
        active_tasks: set = set()

        while True:
            try:
                while pending_results:
                    try:
                        combined_cell_queue.put_nowait(pending_results[0])
                        pending_results.popleft()
                    except Full:
                        break

                completed = set()
                for future in list(active_tasks):
                    if future.done():
                        completed.add(future)
                        skycell_id, task_type = futures.pop(future, ("unknown", "regular"))
                        try:
                            result = future.result()
                            if result is not None:
                                result = _materialize_shm_result(result)
                                if task_type == "padding_source":
                                    band_cache[result["skycell_id"]] = {
                                        "combined_image": result["combined_image"],
                                        "combined_mask": result["combined_mask"],
                                        "headers_data": result["headers_data"],
                                        "removed_stars": result.get("removed_stars", []),
                                    }
                                    logger.info(f"[ProcessCoordinator] Cached padding source {result['skycell_id']}")
                                    # Also publish to the shared combined store (best-effort,
                                    # same as the "regular" branch below): a padding_source
                                    # cell is either dual-role (its own "regular" task will
                                    # arrive later and reuse this via regular_cache_hit) or
                                    # genuinely padding-only (e.g. under projections_limit),
                                    # in which case this combined-store record is the only
                                    # thing padding_correction's cross-projection neighbor
                                    # lookup needs -- never a convolved-store entry.
                                    if _publish_combined is not None:
                                        try:
                                            _publish_combined(result)
                                        except Exception:
                                            logger.warning(
                                                "[ProcessCoordinator] Combined-store publish "
                                                "failed for padding source %s (non-fatal)",
                                                result["skycell_id"],
                                                exc_info=True,
                                            )
                                else:
                                    if _publish_combined is not None:
                                        try:
                                            _publish_combined(result)
                                        except Exception:
                                            logger.warning(
                                                "[ProcessCoordinator] Combined-store publish "
                                                "failed for %s (non-fatal)",
                                                result["skycell_id"],
                                                exc_info=True,
                                            )
                                    pending_results.append(result)
                            else:
                                logger.warning(f"[ProcessCoordinator] Got None result for {skycell_id}")
                        except Exception as e:
                            logger.error(f"[ProcessCoordinator] Process failed for {skycell_id}: {e}")
                active_tasks -= completed

                _now = time.time()
                if _now - last_coordinator_log_time >= 30.0:
                    current_queue_size = combined_cell_queue.qsize()
                    current_buffer_size = calculate_total_buffer_size(cell_buffer)
                    total_pending = len(active_tasks) + current_queue_size
                    _mem = psutil.virtual_memory()
                    logger.info(
                        f"[ProcessCoordinator] Queue States - CombinedRaw: {combined_raw_queue.qsize()}, "
                        f"CombinedOut: {current_queue_size}, Buffer: {current_buffer_size}, "
                        f"Active: {len(active_tasks)}, PendingResults: {len(pending_results)}, "
                        f"BandCache: {len(band_cache)}, TotalPending: {total_pending}, "
                        f"AvailMem: {_mem.available / (1024**3):.1f}GB ({_mem.percent:.0f}% used)"
                    )
                    last_coordinator_log_time = _now

                total_pending = len(active_tasks) + combined_cell_queue.qsize()
                if total_pending >= MAX_TOTAL_PENDING_WORK:
                    time.sleep(0.1)
                    continue

                _mem_now = psutil.virtual_memory()
                if _mem_now.available < _min_available_bytes:
                    if _now - last_mem_warn_time >= 30.0:
                        logger.warning(
                            f"[ProcessCoordinator] Memory pressure: {_mem_now.available / (1024**3):.1f}GB available "
                            f"(threshold: {_min_available_bytes / (1024**3):.1f}GB). "
                            f"Pausing new submissions until memory recovers."
                        )
                        last_mem_warn_time = _now
                    time.sleep(0.5)
                    continue

                if pipeline_paused_event is not None and pipeline_paused_event.is_set():
                    time.sleep(0.1)
                    continue

                try:
                    bundle = combined_raw_queue.get(timeout=0.1)
                except Empty:
                    continue
                except Exception:
                    continue

                if bundle is None:
                    logger.info("[ProcessCoordinator] Received shutdown signal")
                    break

                task_type = bundle.get("task_type", "regular")
                skycell_id = bundle["skycell_id"]

                if task_type == "regular_cache_hit":
                    cached = band_cache.get(skycell_id)
                    if cached:
                        result = {
                            "skycell_id": skycell_id,
                            "projection": bundle["projection"],
                            "row_id": bundle["row_id"],
                            "x_coord": bundle["x_coord"],
                            "combined_image": cached["combined_image"],
                            "combined_mask": cached["combined_mask"],
                            "headers_data": cached["headers_data"],
                            "removed_stars": cached.get("removed_stars", []),
                        }
                        # Dual-role cell: this "regular" task reuses a
                        # padding_source fetch cached earlier this run. That
                        # earlier fetch may not have published a combined-store
                        # record for it (e.g. if this coordinator's
                        # padding_source publish above failed, or in older
                        # runs before that publish existed) -- publish it here
                        # too so _publish_canonical_convolved_snapshot's own
                        # fingerprint lookup (downstream, in the row-processing
                        # thread) finds arrays.npz and doesn't skip this cell.
                        # publish_combined_cell is idempotent, so this is safe
                        # even if it was already published.
                        if _publish_combined is not None:
                            try:
                                _publish_combined(result)
                            except Exception:
                                logger.warning(
                                    "[ProcessCoordinator] Combined-store publish "
                                    "failed for cache-hit %s (non-fatal)",
                                    skycell_id,
                                    exc_info=True,
                                )
                        pending_results.append(result)
                        logger.info(f"[ProcessCoordinator] Cache hit fast-path for {skycell_id}")
                    else:
                        logger.warning(f"[ProcessCoordinator] Expected cache hit for {skycell_id} not found; dropping")
                    continue

                # Inject catalog reference into bundle so process_single_cell
                # can project Gaia stars to this skycell's pixel frame.
                if gaia_catalog is not None:
                    bundle["gaia_catalog"] = gaia_catalog
                bundle["bright_star_mag_threshold"] = bright_star_mag_threshold

                future = executor.submit(process_single_cell, bundle)
                futures[future] = (skycell_id, task_type)
                active_tasks.add(future)

            except Exception as e:
                logger.error(f"[ProcessCoordinator] Error in coordination loop: {e}", exc_info=True)
                break

        logger.info(f"[ProcessCoordinator] Waiting for {len(active_tasks)} remaining tasks")
        for future in list(active_tasks):
            skycell_id, task_type = futures.get(future, ("unknown", "regular"))
            try:
                result = future.result(timeout=60)
                if result is not None:
                    result = _materialize_shm_result(result)
                    if task_type == "padding_source":
                        band_cache[result["skycell_id"]] = {
                            "combined_image": result["combined_image"],
                            "combined_mask": result["combined_mask"],
                            "headers_data": result["headers_data"],
                            "removed_stars": result.get("removed_stars", []),
                        }
                        if _publish_combined is not None:
                            try:
                                _publish_combined(result)
                            except Exception:
                                logger.warning(
                                    "[ProcessCoordinator] Combined-store publish "
                                    "failed for padding source %s (non-fatal)",
                                    result["skycell_id"],
                                    exc_info=True,
                                )
                    else:
                        if _publish_combined is not None:
                            try:
                                _publish_combined(result)
                            except Exception:
                                logger.warning(
                                    "[ProcessCoordinator] Combined-store publish "
                                    "failed for %s (non-fatal)",
                                    result["skycell_id"],
                                    exc_info=True,
                                )
                        pending_results.append(result)
            except Exception as e:
                logger.error(f"[ProcessCoordinator] Final task failed for {skycell_id}: {e}")

        while pending_results:
            try:
                combined_cell_queue.put(pending_results.popleft(), timeout=5)
            except Exception:
                break

    _active_executor = None
    logger.info("[ProcessCoordinator] Finished")


def saver_worker(results_queue: _thread_queue.Queue, output_path: str, *, enabled: bool = True):
    """Stage 4: Saves final results to an output Zarr store."""
    if not enabled:
        while True:
            processed_bundle = results_queue.get()
            if processed_bundle is None:
                break
        logger.info("[Saver] Per-SCC convolved.zarr writes disabled; drained results queue.")
        return
    try:
        output_store = zarr.open(output_path, mode="a")
        logger.info(f"[Saver] Opened output store {output_path}")
        while True:
            processed_bundle = results_queue.get()
            if processed_bundle is None:
                break
            try:
                zarr_utils.save_convolved_results(output_store, processed_bundle["projection"], processed_bundle["row_id"], processed_bundle["results_data"], processed_bundle["results_masks"])
                logger.info(f"[Saver] Saved row {processed_bundle['row_id']} for projection {processed_bundle['projection']}")
            except Exception as e:
                logger.error(f"[Saver] Failed saving row {processed_bundle['row_id']}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[Saver] Failed opening output store at {output_path}: {e}", exc_info=True)


# --- NEW: Sequential Assembler & Convolution (Stage 3) ---


def assemble_row_from_bundles(target_array: np.ndarray, cell_bundles: list[dict], config: MasterArrayConfig) -> tuple[dict, dict]:
    """
    SPEC: Assembles a master row image from a pre-gathered list of cell bundles.
    This function is purely for image assembly and does not interact with queues.
    """
    target_array.fill(np.nan)
    cell_positions = {}
    cell_masks = {}
    # A master array represents one row, not the whole projection.  A row
    # near a tessellation edge may begin far to the right of the projection's
    # global minimum x; anchoring to ``config.starting_x`` then drops or
    # wrongly trims its first cell.  Sort also makes the producer's
    # left-to-right overlap ownership deterministic even if queue arrival was
    # not ordered.
    ordered_bundles = sorted(cell_bundles, key=lambda bundle: int(bundle["x_coord"]))
    first_x_coord = int(ordered_bundles[0]["x_coord"]) if ordered_bundles else 0

    for bundle in ordered_bundles:
        cell_name = bundle["skycell_id"]
        image = bundle["combined_image"]
        mask = bundle["combined_mask"]
        x_coord = bundle["x_coord"]
        cell_index = x_coord - first_x_coord

        target_y_start = PAD_SIZE
        target_x_start_full = PAD_SIZE + cell_index * (config.cell_width - CELL_OVERLAP)

        if cell_index == 0:
            source_x_start = 0
            target_x_start = target_x_start_full
        else:
            source_x_start = EFFECTIVE_OVERLAP
            target_x_start = target_x_start_full + EFFECTIVE_OVERLAP

        source_height, source_width = image.shape
        place_width = source_width - source_x_start
        place_height = source_height

        target_x_end = target_x_start + place_width
        target_y_end = target_y_start + place_height

        if target_x_end <= target_array.shape[1] and target_y_end <= target_array.shape[0]:
            target_array[target_y_start:target_y_end, target_x_start:target_x_end] = image[:, source_x_start:]
            cell_masks[cell_name] = mask
            cell_positions[cell_name] = (target_x_start_full, target_x_start_full + config.cell_width, PAD_SIZE, PAD_SIZE + config.cell_height)
        else:
            logger.warning(f"[Assembler] Cell {cell_name} out of bounds for master array. Skipping placement.")

    logger.info(f"[Assembler] Assembled row with {len(cell_bundles)} cells.")
    return cell_positions, cell_masks


def _manually_process_cell(
    skycell_name: str,
    projection: str,
    ingest_config: dict,
    remove_saturated_stars: bool = True,
    gaia_catalog: Optional[pd.DataFrame] = None,
    bright_star_mag_threshold: float = 13.0,
) -> Optional[dict]:
    """
    Manually load and fully process a single skycell directly in the calling thread.
    This is the shared fallback used by both regular-cell gather and padding-cell gather
    when the pipeline has not delivered a result within the timeout period.

    Processing mirrors process_single_cell:
      zarr load -> process_skycell_bands -> remove_background (catalog + flag passes)

    Returns a result dict on success:
      {combined_image, combined_mask, headers_data, removed_stars}
    Returns None on failure.
    """
    manual_start = time.time()
    logger.info(f"[ManualLoader] Starting {skycell_name} (proj={projection})")
    try:
        bands, masks, weights, headers, headers_weight = _load_skycell_raw_bands(
            skycell_name, projection, ingest_config
        )

        if not bands:
            logger.error(f"[ManualLoader] No band data found for {skycell_name}")
            return None

        combined_image, combined_mask, combined_uncert = process_skycell_bands(
            bands, masks, weights, headers, headers_weight
        )

        # Project Gaia catalog to this skycell's pixel frame (mirrors process_single_cell).
        gaia_catalog_pixels = None
        if remove_saturated_stars and gaia_catalog is not None:
            try:
                from astropy.io import fits as afits
                from astropy.wcs import WCS

                header_str = next(iter(headers.values()))
                wcs = WCS(afits.Header.fromstring(header_str))
                gaia_catalog_pixels = project_gaia_to_skycell(
                    gaia_catalog, wcs, combined_image.shape
                )
                logger.info(
                    f"[ManualLoader] {len(gaia_catalog_pixels)} Gaia stars projected "
                    f"into footprint of {skycell_name}"
                )
            except Exception as proj_err:
                logger.warning(
                    f"[ManualLoader] Gaia projection failed for {skycell_name}: {proj_err}"
                )

        combined_image, removed_stars_list = remove_background(
            combined_image, combined_uncert,
            mask=combined_mask,
            remove_saturated_stars=remove_saturated_stars,
            gaia_catalog_pixels=gaia_catalog_pixels,
            bright_star_mag_threshold=bright_star_mag_threshold,
        )

        # Stamp skycell_id on every record (RA/Dec already in catalog records).
        for star in removed_stars_list:
            star["skycell_id"] = skycell_name

        logger.info(f"[ManualLoader] Finished {skycell_name} in {time.time() - manual_start:.1f}s")
        return {
            "combined_image": combined_image,
            "combined_mask": combined_mask,
            "headers_data": headers,
            "removed_stars": removed_stars_list,
        }
    except Exception as e:
        logger.error(f"[ManualLoader] Failed for {skycell_name}: {e}", exc_info=True)
        return None


def _gather_cells_for_row(
    projection: str,
    row_id: int,
    metadata: dict,
    combined_cell_queue: Queue,
    cell_buffer: dict,
    ingest_config: dict,
    remove_saturated_stars: bool = True,
    gaia_catalog: Optional[pd.DataFrame] = None,
    bright_star_mag_threshold: float = 13.0,
) -> list[dict]:
    """
    Gathers all necessary cell bundles for a given row from the queue.
    Handles out-of-order arrivals using a buffer and has a timeout mechanism
    to manually load/process cells if they don't arrive in time.

    Timeout is measured from the last arrival of a cell for the same projection/row.
    """
    expected_cells = metadata["rows"].get(row_id, [])
    num_to_expect = len(expected_cells)
    if num_to_expect == 0:
        return []

    # Check buffer first for any cells that have already arrived
    key = (projection, row_id)
    gathered_bundles = cell_buffer.pop(key, [])

    logger.info(f"[Gather] Gathering {num_to_expect} cells for P:{projection} R:{row_id}")
    logger.info(f"[Gather] Expected cells: {[cell[0] for cell in expected_cells]}")
    logger.info(f"[Gather] Queue size at start: {combined_cell_queue.qsize()}")
    logger.info(f"[Gather] Already buffered: {len(gathered_bundles)} cells")

    # Track both function start time and last relevant cell arrival time
    function_start_time = time.time()
    last_relevant_arrival_time = time.time() if len(gathered_bundles) > 0 else None
    last_log_time = time.time()

    while len(gathered_bundles) < num_to_expect:
        elapsed_total = time.time() - function_start_time

        # Calculate timeout from last relevant arrival, or from function start if none yet
        if last_relevant_arrival_time is not None:
            time_since_last_relevant = time.time() - last_relevant_arrival_time
            remaining_time = GATHER_TIMEOUT_SECONDS - time_since_last_relevant
        else:
            time_since_function_start = time.time() - function_start_time
            remaining_time = GATHER_TIMEOUT_SECONDS - time_since_function_start

        # Log a progress message at most once per 30 seconds
        now = time.time()
        if now - last_log_time >= 30.0:
            logger.info(
                f"[Gather] Still gathering P:{projection} R:{row_id}, "
                f"have {len(gathered_bundles)}/{num_to_expect}, "
                f"queue_size: {combined_cell_queue.qsize()}, "
                f"elapsed: {elapsed_total:.0f}s, "
                f"remaining_timeout: {remaining_time:.1f}s"
            )
            last_log_time = now

        if remaining_time <= 0:
            received_names = {b["skycell_id"] for b in gathered_bundles}
            expected_names = {name for name, _ in expected_cells}
            still_missing = expected_names - received_names
            logger.warning(
                f"[Gather] Timeout for P:{projection} R:{row_id} after {elapsed_total:.0f}s. "
                f"Received {len(gathered_bundles)}/{num_to_expect}: {received_names}. "
                f"Missing: {still_missing}. Triggering manual fallback."
            )
            break

        try:
            bundle = combined_cell_queue.get(timeout=min(remaining_time, 5.0))
            if bundle is None:
                logger.warning("[Gather] Shutdown signal received while gathering cells.")
                break

            bundle_key = (bundle["projection"], bundle["row_id"])

            if bundle_key == key:
                gathered_bundles.append(bundle)
                last_relevant_arrival_time = time.time()
                logger.debug(f"[Gather] Received relevant cell {bundle['skycell_id']} for P:{projection} R:{row_id}")
            else:
                cell_buffer.setdefault(bundle_key, []).append(bundle)
                logger.debug(f"[Gather] Buffered cell {bundle['skycell_id']} for P:{bundle['projection']} R:{bundle['row_id']}")

        except Empty:
            continue

    # Fix 4 — before manual fallback, do a non-blocking sweep to recover late-arriving cells
    if len(gathered_bundles) < num_to_expect:
        logger.info(f"[Gather] Sweeping combined_cell_queue for late arrivals before manual fallback...")
        while True:
            try:
                bundle = combined_cell_queue.get_nowait()
                if bundle is None:
                    break
                bundle_key = (bundle["projection"], bundle["row_id"])
                if bundle_key == key:
                    gathered_bundles.append(bundle)
                    logger.info(f"[Gather] Late arrival recovered: {bundle['skycell_id']}")
                else:
                    cell_buffer.setdefault(bundle_key, []).append(bundle)
            except Empty:
                break

    # If, after waiting and sweeping, cells are still missing, load them manually
    if len(gathered_bundles) < num_to_expect:
        received_cell_names = {b["skycell_id"] for b in gathered_bundles}
        expected_cell_info = {name: x for name, x in expected_cells}
        missing_cell_names = set(expected_cell_info.keys()) - received_cell_names

        logger.warning(f"[Gather] {len(missing_cell_names)} cells still missing after sweep. Manual load: {missing_cell_names}")
        for cell_name in missing_cell_names:
            result = _manually_process_cell(
                cell_name, projection, ingest_config, remove_saturated_stars,
                gaia_catalog, bright_star_mag_threshold,
            )
            if result is not None:
                manual_bundle = {
                    "skycell_id": cell_name,
                    "projection": projection,
                    "row_id": row_id,
                    "x_coord": expected_cell_info[cell_name],
                    **result,
                }
                gathered_bundles.append(manual_bundle)

    logger.info(f"[Gather] Successfully gathered {len(gathered_bundles)}/{num_to_expect} cells for P:{projection} R:{row_id}")
    return gathered_bundles




def _wait_for_padding_cells(
    needed: set,
    band_cache: dict,
    ingest_config: dict,
    remove_saturated_stars: bool = True,
    timeout: float = 180.0,
    combined_cell_queue: Queue = None,
    cell_buffer: dict = None,
    gaia_catalog: Optional[pd.DataFrame] = None,
    bright_star_mag_threshold: float = 13.0,
) -> None:
    """
    Wait for padding source cells to appear in band_cache.
    Mirrors _gather_cells_for_row: logs progress, tracks time since last arrival,
    falls back to manual full-processing load on timeout.

    While waiting, actively drains combined_cell_queue into cell_buffer. Without
    this drain the queue fills to maxsize while the main thread is blocked here,
    which causes the multiprocessing.Queue feeder thread to hold an internal lock
    that deadlocks the coordinator thread (which also tries to access the queue).
    """
    if not needed:
        return

    remaining = set(needed) - set(band_cache.keys())
    if not remaining:
        logger.info(f"[PaddingGather] All {len(needed)} padding cells already in cache, no wait needed.")
        return

    logger.info(
        f"[PaddingGather] Waiting for {len(remaining)}/{len(needed)} padding cells: {remaining}"
    )
    function_start = time.time()
    last_arrival_time = time.time()
    last_log_time = time.time()

    while remaining:
        elapsed = time.time() - function_start
        time_since_last = time.time() - last_arrival_time
        remaining_timeout = timeout - time_since_last

        # Log a progress message at most once per 30 seconds
        now = time.time()
        if now - last_log_time >= 30.0:
            logger.info(
                f"[PaddingGather] Still waiting for {len(remaining)} padding cells: {remaining}, "
                f"elapsed: {elapsed:.0f}s, timeout_remaining: {remaining_timeout:.1f}s"
            )
            last_log_time = now

        if remaining_timeout <= 0:
            logger.warning(
                f"[PaddingGather] Timeout ({timeout:.0f}s since last arrival). "
                f"Manually loading {len(remaining)} padding cells: {remaining}"
            )
            break

        # Check which cells have arrived in band_cache since last poll
        newly_arrived = {name for name in remaining if name in band_cache}
        if newly_arrived:
            for name in newly_arrived:
                logger.info(f"[PaddingGather] Padding cell arrived in cache: {name}")
            remaining -= newly_arrived
            last_arrival_time = time.time()

        # Drain combined_cell_queue into cell_buffer while we wait.
        # Without this, the queue fills to maxsize (the main thread is not consuming it
        # during this wait), causing the multiprocessing feeder thread to block on pipe
        # writes, which deadlocks the coordinator thread.
        if combined_cell_queue is not None and cell_buffer is not None:
            while True:
                try:
                    bundle = combined_cell_queue.get_nowait()
                    if bundle is None:
                        break
                    bkey = (bundle["projection"], bundle["row_id"])
                    cell_buffer.setdefault(bkey, []).append(bundle)
                except Empty:
                    break

        if remaining:
            time.sleep(0.5)

    # Manual fallback for cells that didn't arrive in time
    for skycell_name in list(remaining):
        parts = skycell_name.split(".")
        source_proj = parts[1] if len(parts) >= 3 else None
        if source_proj is None:
            logger.error(f"[PaddingGather] Cannot derive projection from name: {skycell_name}, skipping")
            continue
        result = _manually_process_cell(
            skycell_name, source_proj, ingest_config, remove_saturated_stars,
            gaia_catalog, bright_star_mag_threshold,
        )
        if result is not None:
            band_cache[skycell_name] = result

    logger.info(
        f"[PaddingGather] All padding cells ready after {time.time() - function_start:.1f}s total."
    )


def _evict_band_cache_for_step(
    band_cache: dict,
    band_cache_uses: dict,
    metadata: dict,
    projection: str,
    current_row_id: int,
    next_row_id: Optional[int],
    row_padding_map: dict,
) -> None:
    """Decrement use counts for cells consumed in this row step and evict exhausted entries.

    Note: ``next_row_id`` is retained in the signature for call-site symmetry but is
    intentionally not used for regular-cell decrements (see comment below).
    """
    cells_to_decrement: set = set()

    # Regular cells assembled this step. Only count current_row membership:
    # every row is the "current row" in exactly one step, so each regular cell
    # is decremented exactly once over the projection's lifetime, matching the
    # +1 dual-role budget granted in run_modern_sliding_window_pipeline. Counting
    # next_row too would decrement dual-role cells twice (a row is visited once as
    # "next" and once as "current"), evicting them one use early and starving
    # later cross-projection padding steps (which then fall back to ManualLoader).
    for cell_name, _ in metadata["rows"].get(current_row_id, []):
        cells_to_decrement.add(cell_name)

    # Padding source cells consumed in this step
    cells_to_decrement.update(row_padding_map.get((str(projection), current_row_id), set()))

    for name in cells_to_decrement:
        if name in band_cache_uses:
            band_cache_uses[name] -= 1
            if band_cache_uses[name] <= 0:
                band_cache.pop(name, None)
                band_cache_uses.pop(name, None)
                logger.debug(f"[BandCache] Evicted {name}")


def _convolve_whole_row_snapshot(
    state: ProcessingState, psf_sigma: float, projection: str, radius: int
) -> Optional[dict[str, np.ndarray]]:
    """Old (pre-percell-skip) behavior: convolve the entire same-projection
    master row array once and extract every cell. Used as the fallback path
    when >= ``ROW_FALLBACK_THRESHOLD`` of this row's cells actually need
    (re)convolution -- at that point call/I-O overhead from many small
    local-window convolutions outweighs the FLOPs saved by not touching
    already-canonical cells.
    """
    try:
        snapshot = state.current_array.copy()
        nan_mask = np.isnan(snapshot)
        snapshot[nan_mask] = 0.0
        convolved_snapshot = convolution_utils.apply_gaussian_convolution(snapshot, sigma=psf_sigma, radius=radius)
        convolved_snapshot[nan_mask] = np.nan
        return extract_cell_results(convolved_snapshot, state.cell_locations)
    except Exception:
        logger.warning(
            "[SequentialProcessor] Canonical same-projection convolution snapshot "
            "failed for projection %s (shared convolved-store publish skipped for "
            "this row, non-fatal)",
            projection,
            exc_info=True,
        )
        return None


def _convolve_local_windows_for_missing_cells(
    state: ProcessingState, missing_entries: list[dict], psf_sigma: float, radius: int
) -> dict[str, np.ndarray]:
    """Convolve only the cells in ``missing_entries``, in merged contiguous
    windows rather than one call per cell (plan doc "contiguous-run
    batching").

    Each window is the union of one or more adjacent-in-x missing cells'
    bounding boxes, expanded by ``radius`` on every side and clipped to
    ``state.current_array``'s own bounds. Because the kernel is truncated at
    ``radius`` (``truncate = radius / sigma`` in
    ``convolution_utils.apply_gaussian_convolution``), every output pixel
    inside a cell's own bounds depends only on input pixels within
    ``radius`` of that cell -- all of which lie inside this window by
    construction -- so cropping the window's cell region after convolving
    reproduces exactly what convolving the full row array and cropping would
    have produced. The only place this could differ is where the window
    clips at the *true* array edge; that clip coincides exactly with the
    same true edge the whole-row path would also clip at, using the same
    ``mode="constant"`` boundary convention, so the two paths still agree
    there too.
    """
    array = state.current_array
    array_h, array_w = array.shape

    ordered = sorted(missing_entries, key=lambda e: e["x_start"])
    runs: list[list[dict]] = []
    for entry in ordered:
        if runs and entry["x_start"] <= runs[-1][-1]["x_end"] + 2 * radius:
            # Windows for these two cells would overlap (or nearly so)
            # anyway -- merge into a single convolution call instead of two
            # redundant overlapping ones.
            runs[-1].append(entry)
        else:
            runs.append([entry])

    results: dict[str, np.ndarray] = {}
    for run in runs:
        x_start = min(e["x_start"] for e in run)
        x_end = max(e["x_end"] for e in run)
        y_start = min(e["y_start"] for e in run)
        y_end = max(e["y_end"] for e in run)

        win_x0, win_x1 = max(0, x_start - radius), min(array_w, x_end + radius)
        win_y0, win_y1 = max(0, y_start - radius), min(array_h, y_end + radius)

        window = array[win_y0:win_y1, win_x0:win_x1].copy()
        nan_mask = np.isnan(window)
        window[nan_mask] = 0.0
        convolved_window = convolution_utils.apply_gaussian_convolution(window, sigma=psf_sigma, radius=radius)
        convolved_window[nan_mask] = np.nan

        for entry in run:
            cx0, cx1 = entry["x_start"] - win_x0, entry["x_end"] - win_x0
            cy0, cy1 = entry["y_start"] - win_y0, entry["y_end"] - win_y0
            results[entry["cell_name"]] = convolved_window[cy0:cy1, cx0:cx1].copy()
    return results


def _lazy_fetch_combined_cell(
    cell_name: str,
    band_cache: Optional[dict],
    combined_store_data_root: Optional[str],
    combined_store_recipe: Any,
) -> Optional[dict]:
    """Tier-1 (``band_cache``) then tier-2 (combined store) lookup for a
    single cell, synchronous, no worker/queue involvement -- used by the
    per-skycell path (Part B step 3a). Never raw-fetches or combines; a
    miss here means the caller should fall back to the dense/full-loop path
    for the enclosing projection.
    """
    if band_cache is not None and cell_name in band_cache:
        return band_cache[cell_name]
    if combined_store_data_root is None or combined_store_recipe is None:
        return None
    from syndiff_pipeline.template_creation.processing.combined_store import (
        seed_band_cache_from_combined_store,
    )

    hits = seed_band_cache_from_combined_store(
        combined_store_data_root, [cell_name], combined_store_recipe
    )
    if cell_name in hits:
        if band_cache is not None:
            band_cache[cell_name] = hits[cell_name]
        return hits[cell_name]
    return None


def _find_projection_neighbors(
    metadata: dict, cell_name: str, row_id: int, x_coord: int
) -> list[tuple[str, int, int]]:
    """Up to 8 neighbors of ``cell_name`` in the same projection: same-row
    left/right (edges) and the up-to-3 cells in each of row-1/row+1 whose
    grid-column index (``x_coord``) is within 1 of this cell's (edges +
    corners). ``x_coord`` is the same grid-column index
    ``assemble_row_from_bundles`` uses (``cell_index = x_coord -
    first_x_coord``), not a physical pixel offset, so exact-index proximity
    is the correct adjacency test, not a pixel-range overlap.

    A projection/grid edge (no row above/below, or no cell at that column in
    a neighbor row) simply yields fewer neighbors on that side -- the
    missing border stays NaN in the padded array, matching how
    ``assemble_row_from_bundles``/cross-row padding already treat true grid
    edges. True cross-projection edges are a separate mechanism entirely
    (``identify_all_padding_sources``) and are not handled here.
    """
    neighbors: list[tuple[str, int, int]] = []
    for row_offset in (-1, 0, 1):
        row_cells = metadata["rows"].get(row_id + row_offset)
        if not row_cells:
            continue
        for name, x in row_cells:
            x_offset = x - x_coord
            if row_offset == 0 and x_offset == 0:
                continue
            if abs(x_offset) <= 1:
                neighbors.append((name, row_offset, x_offset))
    return neighbors


def convolve_single_skycell(
    cell_name: str,
    row_id: int,
    x_coord: int,
    projection: str,
    metadata: dict,
    fetch_cell,
    psf_sigma: float,
    radius: int,
) -> Optional[dict]:
    """Convolve exactly one skycell using only the ``radius``-wide border
    strips of its up-to-8 same-projection neighbors -- never full neighbor
    cells, never a multi-cell mosaic (see
    doc/ps1_process_tiered_ingest_architecture_plan.md, Part B step 3a).

    ``fetch_cell(name) -> Optional[dict]`` resolves a cell to a pre-combined,
    star-removed bundle (``combined_image``/``combined_mask``/...) via the
    existing tier-1/tier-2 lookup -- never raw-fetches. Returns ``None`` if
    the center cell or any neighbor isn't already available this way, so
    the caller can fall back to the dense/full-loop path for the whole
    projection rather than reimplementing raw-fetch+combine+star-removal
    here.

    Because the Gaussian kernel is truncated at ``radius`` (``truncate =
    radius / sigma``), every output pixel inside the center cell's own
    bounds depends only on input pixels within ``radius`` -- exactly what
    this small ``(cell_height + 2*radius) x (cell_width + 2*radius)`` array
    contains, so this reproduces exactly what the whole-row convolution
    would have produced for this one cell.
    """
    center = fetch_cell(cell_name)
    if center is None:
        return None

    image = center["combined_image"]
    cell_h, cell_w = image.shape[:2]

    padded_h, padded_w = cell_h + 2 * radius, cell_w + 2 * radius
    padded = np.full((padded_h, padded_w), np.nan, dtype=np.float32)
    padded[radius:radius + cell_h, radius:radius + cell_w] = image

    for name, row_offset, x_offset in _find_projection_neighbors(metadata, cell_name, row_id, x_coord):
        neighbor = fetch_cell(name)
        if neighbor is None:
            # No border context from this side -- stays NaN, same convention
            # as a true grid edge (assemble_row_from_bundles/cross-row
            # padding already do this).
            continue
        n_image = neighbor["combined_image"]
        n_h, n_w = n_image.shape[:2]

        if x_offset < 0:
            src_x = slice(max(0, n_w - radius), n_w)
        elif x_offset > 0:
            src_x = slice(0, min(radius, n_w))
        else:
            src_x = slice(0, n_w)

        if row_offset < 0:
            src_y = slice(max(0, n_h - radius), n_h)
        elif row_offset > 0:
            src_y = slice(0, min(radius, n_h))
        else:
            src_y = slice(0, n_h)

        strip = n_image[src_y, src_x]
        strip_h, strip_w = strip.shape

        dst_x_start = 0 if x_offset < 0 else (radius + cell_w if x_offset > 0 else radius)
        dst_y_start = 0 if row_offset < 0 else (radius + cell_h if row_offset > 0 else radius)
        dst_x_end = min(padded_w, dst_x_start + strip_w)
        dst_y_end = min(padded_h, dst_y_start + strip_h)
        if dst_x_end <= dst_x_start or dst_y_end <= dst_y_start:
            continue
        padded[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = strip[
            : dst_y_end - dst_y_start, : dst_x_end - dst_x_start
        ]

    nan_mask = np.isnan(padded)
    padded[nan_mask] = 0.0
    convolved = convolution_utils.apply_gaussian_convolution(padded, sigma=psf_sigma, radius=radius)
    convolved[nan_mask] = np.nan
    result_image = convolved[radius:radius + cell_h, radius:radius + cell_w].copy()

    return {
        "skycell_id": cell_name,
        "projection": projection,
        "combined_image": result_image,
        "combined_mask": center.get("combined_mask"),
        "headers_data": center.get("headers_data") or {},
        "removed_stars": center.get("removed_stars", []),
    }


def _publish_single_convolved_cell(
    data_root: str,
    cell_projection: str,
    cell: str,
    convolved_image: np.ndarray,
    convolved_mask,
    headers_data: dict,
    removed_stars: list,
    combined_store_recipe,
    convolved_store_recipe,
) -> bool:
    """Publish one cell to the shared convolved store, mirroring exactly how
    ``_publish_canonical_convolved_snapshot`` publishes each cell -- same
    fingerprint chain, same idempotent publish + pointer update. Returns
    ``True`` on success (never raises past its own boundary)."""
    from syndiff_pipeline.template_creation.processing.combined_store import (
        combined_fingerprint as _combined_fingerprint,
        combined_recipe_id,
        raw_skycell_input_fingerprint,
    )
    from syndiff_pipeline.template_creation.processing.convolved_store import (
        publish_convolved_cell,
        update_current_pointer,
    )

    if convolved_mask is None:
        logger.debug(
            "[SparseProjection] No mask available for %s; skipping shared "
            "convolved-store publish.",
            cell,
        )
        return False
    try:
        rid = combined_recipe_id(combined_store_recipe)
        raw_fp = raw_skycell_input_fingerprint(data_root, cell_projection, cell)
        fp = _combined_fingerprint(cell_projection, cell, rid, [raw_fp])
        convolved_info = publish_convolved_cell(
            data_root,
            cell_projection,
            cell,
            convolved_image=convolved_image,
            convolved_mask=convolved_mask,
            headers_data=headers_data or {},
            removed_stars=removed_stars or [],
            recipe=convolved_store_recipe,
            combined_fingerprint=fp,
        )
        if convolved_info is not None and convolved_info.get("fingerprint"):
            update_current_pointer(data_root, cell_projection, cell, convolved_info["fingerprint"])
        return True
    except Exception:
        logger.warning(
            "[SparseProjection] Shared convolved-store publish failed for %s (non-fatal)",
            cell,
            exc_info=True,
        )
        return False


def process_sparse_projection(
    projection: str,
    metadata: dict,
    missing_cells: set,
    band_cache: Optional[dict],
    data_root: str,
    combined_store_recipe,
    convolved_store_recipe,
    psf_sigma: float,
) -> Optional[set]:
    """Per-skycell path for a projection with few (``<= MISSING_CELL_THRESHOLD``)
    missing cells (Part B step 3a) -- bypasses the sliding-row worker
    pipeline entirely. All-or-nothing per projection: if any required cell
    (a missing cell or one of its neighbors) isn't already combined
    (tier-1/tier-2), returns ``None`` so the caller falls back to dispatching
    this projection through the dense/full-loop path (Part B step 3b, which
    can do a real raw-fetch+combine+star-removal via the worker pipeline).

    Returns the set of cell names successfully published on success.
    """
    from syndiff_pipeline.template_creation.processing.convolved_store import (
        _projection_and_cell,
    )

    try:
        radius = int((convolved_store_recipe or {}).get("radius", 470))
    except Exception:
        radius = 470

    cell_to_rowx: dict[str, tuple[int, int]] = {}
    for row_id, cells in metadata["rows"].items():
        for name, x in cells:
            cell_to_rowx[name] = (row_id, x)

    def fetch_cell(name: str) -> Optional[dict]:
        return _lazy_fetch_combined_cell(name, band_cache, data_root, combined_store_recipe)

    computed: dict[str, dict] = {}
    for cell_name in missing_cells:
        if cell_name not in cell_to_rowx:
            logger.warning(
                "[SparseProjection] %s not found in projection %s metadata; "
                "falling back to dense/full-loop path.",
                cell_name,
                projection,
            )
            return None
        row_id, x_coord = cell_to_rowx[cell_name]
        out = convolve_single_skycell(
            cell_name, row_id, x_coord, projection, metadata, fetch_cell, psf_sigma, radius,
        )
        if out is None:
            logger.info(
                "[SparseProjection] %s (or a neighbor) requires a cold raw "
                "fetch; falling back to dense/full-loop path for projection %s.",
                cell_name,
                projection,
            )
            return None
        computed[cell_name] = out

    published: set = set()
    for cell_name, out in computed.items():
        parsed = _projection_and_cell(cell_name)
        if parsed is None:
            continue
        cell_projection, cell = parsed
        ok = _publish_single_convolved_cell(
            data_root,
            cell_projection,
            cell,
            out["combined_image"],
            out["combined_mask"],
            out["headers_data"],
            out["removed_stars"],
            combined_store_recipe,
            convolved_store_recipe,
        )
        if ok:
            published.add(cell_name)
    return published


def _publish_canonical_convolved_snapshot(
    state: ProcessingState,
    projection: str,
    psf_sigma: float,
    convolved_store_data_root: str,
    combined_store_recipe,
    convolved_store_recipe,
) -> None:
    """Publish the same-projection-only canonical convolved cell for every
    cell currently placed in ``state.current_array`` that isn't already
    canonical under the caller's exact recipe chain (plan §13, decision #3;
    per-cell skip per ``ps1_process_percell_skip_plan.md``).

    MUST be called by the caller exactly between ``apply_cross_row_padding``
    and ``apply_cross_projection_padding`` -- see the call site in
    ``process_row_step_from_queue`` for why that's the only correct point.

    Never mutates ``state.current_array``; the live array used by the
    unchanged main convolution path (``process_row_step_from_queue`` step 5)
    is untouched, and this function never raises past its own boundary
    (best-effort, mirrors every other shared-store publish call site in this
    module).

    **Per-cell canonical skip (unconditional, no opt-out):** for every cell
    in this row, ``convolved_store.skycell_already_canonical`` is checked
    *before* any convolution happens. That check recomputes the upstream
    ``combined_skycell`` fingerprint the same deterministic way
    ``combined_store.py``'s own seed-lookup path does
    (``raw_skycell_input_fingerprint`` + ``combined_fingerprint``) from the
    caller's own ``combined_store_recipe``, then resolves whether a
    ``convolved_skycell`` payload already exists for that exact fingerprint
    chain under the caller's own ``convolved_store_recipe`` -- never trusted
    blindly, never inspecting mtimes. Two cells can only both resolve
    "canonical" here if *every* combined-recipe parameter (band weights,
    ``apply_flux_conv``, saturated-star removal params, ``gaia_version``)
    *and* every convolution parameter (``psf_sigma``, ``radius``, ``mode``)
    match, because ``convolved_fingerprint`` always Merkles in the resolved
    ``combined_fingerprint`` as one of its inputs.

    - If every cell in the row is already canonical: no convolution runs at
      all for this row.
    - If the fraction of non-canonical cells is below
      ``ROW_FALLBACK_THRESHOLD``: only those cells are (re)convolved, in
      merged local ±radius windows (see
      ``_convolve_local_windows_for_missing_cells``).
    - Otherwise: falls back to convolving the whole row once (see
      ``_convolve_whole_row_snapshot``), same as before this change, but
      still only *publishes* the non-canonical cells.

    Cells whose combined-store record isn't actually present on disk (e.g. a
    dual-role cell that was only ever cached as a padding source and never
    published this run) are skipped for this publish -- never fabricated.

    headers_data/removed_stars come from ``state.cell_metadata`` (populated
    in lockstep with ``state.cell_locations``/``state.current_masks`` at row
    load time from the exact same bundle dict ``_publish_combined`` used) --
    the in-memory record, not a re-read from disk, since it is exactly what
    a fresh combined-store publish for this cell would have written.
    """
    from syndiff_pipeline.template_creation.processing.combined_store import (
        _projection_and_cell,
        combined_cell_dir,
        combined_fingerprint as _combined_fingerprint,
        combined_recipe_id,
        raw_skycell_input_fingerprint,
    )
    from syndiff_pipeline.template_creation.processing.convolved_store import (
        publish_convolved_cell,
        resolve_convolved_fingerprint_for_recipe,
    )
    from syndiff_pipeline.template_creation.processing import convolved_store

    try:
        rid = combined_recipe_id(combined_store_recipe)
    except Exception:
        logger.warning(
            "[SequentialProcessor] Could not compute combined_recipe_id for shared "
            "convolved-store publish (skipping this row, non-fatal)",
            exc_info=True,
        )
        return

    try:
        radius = int((convolved_store_recipe or {}).get("radius", 470))
    except Exception:
        radius = 470

    # Pass 1 (cheap, read-only): resolve each cell's combined_fingerprint and
    # its canonical status, without convolving anything yet.
    cell_entries: list[dict] = []
    for cell_name, (x_start, x_end, y_start, y_end) in state.cell_locations.items():
        parsed = _projection_and_cell(cell_name)
        if parsed is None:
            continue
        cell_projection, cell = parsed
        try:
            raw_fp = raw_skycell_input_fingerprint(convolved_store_data_root, cell_projection, cell)
            fp = _combined_fingerprint(cell_projection, cell, rid, [raw_fp])
            cell_dir = combined_cell_dir(convolved_store_data_root, cell_projection, cell, fp)
            has_combined_record = (cell_dir / "arrays.npz").is_file()
        except Exception:
            logger.warning(
                "[SequentialProcessor] Could not resolve combined_fingerprint for "
                "%s (skipping shared convolved-store publish for this cell, "
                "non-fatal)",
                cell_name,
                exc_info=True,
            )
            continue
        if not has_combined_record:
            # No actually-published combined_skycell record under this
            # fingerprint (e.g. a dual-role cell that was only ever cached
            # as a padding source and never published this run). Never
            # fabricate a combined_fingerprint edge to nothing.
            logger.debug(
                "[SequentialProcessor] No published combined_skycell record "
                "for %s (fp=%s); skipping shared convolved-store publish for "
                "this cell.",
                cell_name,
                fp,
            )
            continue

        already_canonical = (
            resolve_convolved_fingerprint_for_recipe(
                convolved_store_data_root, cell_projection, cell, convolved_store_recipe, fp,
            )
            is not None
        )
        cell_entries.append(
            {
                "cell_name": cell_name,
                "cell_projection": cell_projection,
                "cell": cell,
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end,
                "combined_fp": fp,
                "already_canonical": already_canonical,
            }
        )

    if not cell_entries:
        return

    missing_entries = [entry for entry in cell_entries if not entry["already_canonical"]]
    if not missing_entries:
        logger.debug(
            "[SequentialProcessor] Row for projection %s already fully canonical "
            "in the shared convolved store; skipping convolution entirely.",
            projection,
        )
        return

    missing_fraction = len(missing_entries) / len(cell_entries)
    if missing_fraction >= ROW_FALLBACK_THRESHOLD:
        canonical_results = _convolve_whole_row_snapshot(state, psf_sigma, projection, radius)
        if canonical_results is None:
            return
        targets = [
            (entry, canonical_results[entry["cell_name"]])
            for entry in missing_entries
            if entry["cell_name"] in canonical_results
        ]
    else:
        local_results = _convolve_local_windows_for_missing_cells(state, missing_entries, psf_sigma, radius)
        targets = [
            (entry, local_results[entry["cell_name"]])
            for entry in missing_entries
            if entry["cell_name"] in local_results
        ]

    for entry, convolved_image in targets:
        cell_name = entry["cell_name"]
        cell_projection = entry["cell_projection"]
        cell = entry["cell"]
        fp = entry["combined_fp"]
        try:
            mask = state.current_masks.get(cell_name)
            if mask is None:
                logger.debug(
                    "[SequentialProcessor] No mask available for %s; skipping "
                    "shared convolved-store publish for this cell.",
                    cell_name,
                )
                continue

            meta = state.cell_metadata.get(cell_name, {})
            convolved_info = publish_convolved_cell(
                convolved_store_data_root,
                cell_projection,
                cell,
                convolved_image=convolved_image,
                convolved_mask=mask,
                headers_data=meta.get("headers_data") or {},
                removed_stars=meta.get("removed_stars", []),
                recipe=convolved_store_recipe,
                combined_fingerprint=fp,
            )
            if convolved_info is not None and convolved_info.get("fingerprint"):
                # Defense-in-depth (plan Phase 1), mirroring the combined-store
                # pointer update: see ``_publish_combined`` above.
                convolved_store.update_current_pointer(
                    convolved_store_data_root, cell_projection, cell,
                    convolved_info["fingerprint"],
                )
        except Exception:
            logger.warning(
                "[SequentialProcessor] Shared convolved-store publish failed for "
                "%s (non-fatal)",
                cell_name,
                exc_info=True,
            )


def process_row_step_from_queue(
    state: ProcessingState,
    config: MasterArrayConfig,
    metadata: dict,
    current_row_id: int,
    next_row_id: Optional[int],
    combined_cell_queue: Queue,
    cell_buffer: dict,
    psf_sigma: float,
    ingest_config: dict,
    projection: str,
    catalog: Optional[pd.DataFrame] = None,
    enable_saturation_correction: bool = False,
    remove_saturated_stars: bool = True,
    csv_path: Optional[str] = None,
    pipeline_paused_event: threading.Event = None,
    band_cache: dict = None,
    band_cache_uses: dict = None,
    row_padding_map: dict = None,
    bright_star_mag_threshold: float = 13.0,
    convolved_store_data_root: Optional[str] = None,
    combined_store_recipe=None,
    convolved_store_recipe=None,
    write_per_scc_convolved_zarr: bool = True,
) -> tuple[dict, dict, list[dict]]:
    """
    Encapsulates the logic for processing a single row step in the sliding window.
    It loads necessary data, applies padding, performs convolution, and extracts results.

    When ``convolved_store_recipe``/``convolved_store_data_root``/
    ``combined_store_recipe`` are all set, also publishes each cell's
    same-projection-only canonical convolved result to the shared convolved
    store (plan §13, decision #3) via ``_publish_canonical_convolved_snapshot``
    -- see that function's docstring for exactly where/why. Best-effort only;
    never affects the return values below or the unchanged main convolution
    path (step 5).

    ``write_per_scc_convolved_zarr=False`` skips step 5's (fully cross-
    projection-padded) Gaussian convolution entirely: its only consumer is
    ``saver_worker``, which -- when the legacy per-SCC ``convolved.zarr``
    writer is disabled -- just drains ``results_queue`` and discards every
    ``results_data``/``results_masks`` payload without ever reading its
    content (see ``saver_worker(..., enabled=False)``). Step 5's own cost
    (a full-row Gaussian convolution, independent of step 3b's shared-store
    snapshot convolution and of any per-cell combined-store caching) is
    therefore pure wasted compute whenever the per-SCC store is disabled
    (e.g. ``use_shared_convolved_store: true`` site configs). Skipping it
    changes no observable output: the placeholder ``results_data`` values are
    never read by anything (only their *keys*, for ``produced_skycells``
    bookkeeping/logging, are used).

    Returns:
        (results_data, results_masks, row_removed_stars) where row_removed_stars is
        a flat list of removed-star records collected from all bundles in this step.
    """
    row_removed_stars: list[dict] = []

    # Determine which padding cells are needed for this row step (for _wait_for_padding_cells).
    # The tasks were already dispatched upfront in the interleaved task list.
    needed_padding_cells: set = row_padding_map.get((str(projection), current_row_id), set()) if row_padding_map else set()

    # 1. Load the Current Row (Only If Necessary)
    if state.current_row_id != current_row_id:
        logger.info(f"[SequentialProcessor] Loading initial current row ID {current_row_id}")
        current_row_bundles = _gather_cells_for_row(
            projection,
            current_row_id,
            metadata,
            combined_cell_queue,
            cell_buffer,
            ingest_config,
            remove_saturated_stars,
            gaia_catalog=catalog,
            bright_star_mag_threshold=bright_star_mag_threshold,
        )
        for bundle in current_row_bundles:
            row_removed_stars.extend(bundle.get("removed_stars", []))
        positions, masks = assemble_row_from_bundles(state.current_array, current_row_bundles, config)
        state.cell_locations.update(positions)
        state.current_masks.update(masks)
        state.cell_metadata.update(
            {
                bundle["skycell_id"]: {
                    "headers_data": bundle.get("headers_data") or {},
                    "removed_stars": bundle.get("removed_stars", []),
                }
                for bundle in current_row_bundles
                if bundle["skycell_id"] in positions
            }
        )
        state.current_row_id = current_row_id
        logger.info(f"[SequentialProcessor] Built current row ID {current_row_id} with {len(positions)} cells.")

        # Apply Saturation Correction
        if enable_saturation_correction and (not remove_saturated_stars) and catalog is not None:
            logger.info(f"[SequentialProcessor] Applying parallel saturation correction for current row {current_row_id}...")
            start_sat = time.time()
            apply_saturation_to_row(state.current_array, state.current_masks, state.cell_locations, current_row_bundles, catalog)
            logger.info(f"[SequentialProcessor] Saturation correction finished in {time.time() - start_sat:.2f}s")

    # 2. Load the Next Row (Always)
    if next_row_id is not None:
        logger.info(f"[SequentialProcessor] Preparing next row ID {next_row_id}")
        next_row_bundles = _gather_cells_for_row(
            projection,
            next_row_id,
            metadata,
            combined_cell_queue,
            cell_buffer,
            ingest_config,
            remove_saturated_stars,
            gaia_catalog=catalog,
            bright_star_mag_threshold=bright_star_mag_threshold,
        )
        for bundle in next_row_bundles:
            row_removed_stars.extend(bundle.get("removed_stars", []))
        positions, masks = assemble_row_from_bundles(state.next_array, next_row_bundles, config)
        state.next_cell_locations.update(positions)
        state.next_masks.update(masks)
        state.next_cell_metadata.update(
            {
                bundle["skycell_id"]: {
                    "headers_data": bundle.get("headers_data") or {},
                    "removed_stars": bundle.get("removed_stars", []),
                }
                for bundle in next_row_bundles
                if bundle["skycell_id"] in positions
            }
        )
        state.next_row_id = next_row_id
        logger.info(f"[SequentialProcessor] Prepared next row ID {next_row_id} with {len(state.next_cell_locations)} cells.")

        # Apply Saturation Correction
        if enable_saturation_correction and (not remove_saturated_stars) and catalog is not None:
            logger.info(f"[SequentialProcessor] Applying parallel saturation correction for next row {next_row_id}...")
            start_sat = time.time()
            apply_saturation_to_row(state.next_array, state.next_masks, state.next_cell_locations, next_row_bundles, catalog)
            logger.info(f"[SequentialProcessor] Saturation correction finished in {time.time() - start_sat:.2f}s")

    else:
        # Clear next state if there is no next row
        state.next_array.fill(np.nan)
        state.next_cell_locations.clear()
        state.next_masks.clear()
        state.next_cell_metadata.clear()
        state.next_row_id = None
        logger.info("[SequentialProcessor] No next row to prepare.")

    # 3. Apply Cross-Row Padding
    apply_cross_row_padding(state, config)

    # np.savez(f"debug_cross_proj_row_{current_row_id}.npz", state=state, config=config, metadata=metadata, current_row_id=current_row_id, next_row_id=next_row_id, zarr_path=zarr_path, csv_path=csv_path)
    # raise RuntimeError("Debug stop")

    # 3b. Shared convolved-store canonical snapshot (plan §13, decision #3).
    # Must run exactly here: AFTER cross-row (same-projection) padding has
    # been baked into state.current_array, but BEFORE cross-projection
    # padding (step 4, below) touches it -- this is the only point in the
    # row's lifecycle where state.current_array holds the same-projection-
    # only master array the canonical cell definition requires. Operates on
    # an independent copy; never mutates state.current_array, so the
    # unchanged main path (step 5) and its output are unaffected regardless
    # of whether this runs. Fully gated off (zero work, zero import) unless
    # all three of convolved_store_recipe/convolved_store_data_root/
    # combined_store_recipe are set, i.e. only when the caller opted into
    # use_shared_convolved_store=True.
    if (
        convolved_store_recipe is not None
        and convolved_store_data_root
        and combined_store_recipe is not None
    ):
        _publish_canonical_convolved_snapshot(
            state,
            projection,
            psf_sigma,
            convolved_store_data_root,
            combined_store_recipe,
            convolved_store_recipe,
        )

    # 4. Apply Cross-Projection Padding (if applicable)
    if csv_path:
        # Wait for precomputed-interleaved padding cells to be ready in band_cache.
        # They were dispatched upfront in the task list; the coordinator is processing
        # them concurrently. If any don't arrive in time, manually load+process them.
        if needed_padding_cells and band_cache is not None:
            _wait_for_padding_cells(
                needed_padding_cells, band_cache, ingest_config,
                remove_saturated_stars=remove_saturated_stars,
                combined_cell_queue=combined_cell_queue,
                cell_buffer=cell_buffer,
                gaia_catalog=catalog,
                bright_star_mag_threshold=bright_star_mag_threshold,
            )

        logger.info(f"[SequentialProcessor] Applying parallel cross-projection padding for row {current_row_id}...")
        start_cp = time.time()
        # Fix 3 — pause coordinator new-submissions so reproject_interp gets full CPU
        if pipeline_paused_event is not None:
            pipeline_paused_event.set()
        try:
            apply_cross_projection_padding(
                state, config, metadata, current_row_id, next_row_id, ingest_config, csv_path,
                band_cache=band_cache,
                remove_saturated_stars=remove_saturated_stars,
                gaia_catalog=catalog,
                bright_star_mag_threshold=bright_star_mag_threshold,
            )
        finally:
            if pipeline_paused_event is not None:
                pipeline_paused_event.clear()
        logger.info(f"[SequentialProcessor] Cross-projection padding finished in {time.time() - start_cp:.2f}s")
        # Fix 7 — evict band_cache entries no longer needed after this step
        if band_cache is not None and band_cache_uses is not None and row_padding_map is not None:
            _evict_band_cache_for_step(
                band_cache, band_cache_uses, metadata, projection,
                current_row_id, next_row_id, row_padding_map,
            )

    # 5. Perform Convolution
    # Skipped entirely when the legacy per-SCC convolved.zarr writer is
    # disabled -- its output is never consumed in that case (see docstring
    # above and saver_worker(..., enabled=False)).
    if write_per_scc_convolved_zarr:
        nan_mask = np.isnan(state.current_array)
        state.current_array[nan_mask] = 0.0
        logger.info(f"[SequentialProcessor] Applying convolution for row ID {current_row_id}")
        convolved_array = convolution_utils.apply_gaussian_convolution(state.current_array, sigma=psf_sigma)
        # Restore NaNs on the result, not the state array which will be replaced
        convolved_array[nan_mask] = np.nan

        # 6. Extract and Return Results
        results_data = extract_cell_results(convolved_array, state.cell_locations)
        results_masks = {name: mask for name, mask in state.current_masks.items() if name in state.cell_locations}
    else:
        logger.debug(
            f"[SequentialProcessor] Skipping step-5 convolution for row ID "
            f"{current_row_id} (per-SCC convolved.zarr writer disabled; "
            f"shared-store publish already handled in step 3b)."
        )
        results_data = {name: None for name in state.cell_locations}
        results_masks = {name: mask for name, mask in state.current_masks.items() if name in state.cell_locations}

    return results_data, results_masks, row_removed_stars


def sequential_processor(
    projections: list[str],
    df: pd.DataFrame,
    combined_cell_queue: Queue,
    results_queue: Queue,
    psf_sigma: float,
    ingest_config: dict,
    cell_buffer: dict,
    catalog: Optional[pd.DataFrame] = None,
    enable_saturation_correction: bool = False,
    remove_saturated_stars: bool = True,
    csv_path: Optional[str] = None,
    pipeline_paused_event: threading.Event = None,
    band_cache: dict = None,
    band_cache_uses: dict = None,
    row_padding_map: dict = None,
    bright_star_mag_threshold: float = 13.0,
    convolved_store_data_root: Optional[str] = None,
    combined_store_recipe=None,
    convolved_store_recipe=None,
    write_per_scc_convolved_zarr: bool = True,
):
    """
    SPEC: This is Stage 3. It iterates through projections sequentially,
    calling a helper function to process each row, queues results, and manages
    the sliding window state.

    Returns:
        Tuple of (all_removed_stars, produced_skycells) where ``all_removed_stars``
        is the flat list of removed-star records accumulated across every
        projection and row, and ``produced_skycells`` is the sorted list of skycell
        names actually written to the output store (the saver creates one
        ``<skycell>_data``/``_mask`` per name).
    """
    all_removed_stars: list[dict] = []
    produced_skycells: set[str] = set()
    last_progress_log = time.monotonic()
    total_projections = len(projections)

    for proj_idx, projection in enumerate(projections):
        logger.info(f"[SequentialProcessor] --- Starting sequential processing for projection: {projection} ---")
        try:
            metadata = extract_projection_metadata(df, projection)
            config = create_master_array_config(metadata)
            state = initialize_processing_state(config)
            row_ids = sorted(metadata["rows"].keys())
        except Exception as e:
            logger.error(f"[SequentialProcessor] Failed to initialize projection {projection}: {e}. Skipping.")
            # Attempt to drain the queue of cells for this failed projection
            num_cells_to_skip = sum(len(cells) for _, cells in metadata.get("rows", {}).items())
            for _ in range(num_cells_to_skip):
                try:
                    combined_cell_queue.get(timeout=1)
                except Empty:
                    break
            continue

        logger.info(
            f"[Pipeline] Progress: projection {proj_idx}/{total_projections} "
            f"row 0/{len(row_ids)}"
        )

        # Inner Loop: Process each row
        for i, current_row_id in enumerate(row_ids):
            logger.info(f"[SequentialProcessor] --- Processing step for row {i + 1}/{len(row_ids)}: ROW ID {current_row_id} ---")
            logger.info(f"[SequentialProcessor] Combined queue size: {combined_cell_queue.qsize()}")

            # Determine Next Row
            next_row_id = row_ids[i + 1] if i + 1 < len(row_ids) else None

            try:
                # Call the Helper Function to do the heavy lifting
                results_data, results_masks, row_removed_stars = process_row_step_from_queue(
                    state,
                    config,
                    metadata,
                    current_row_id,
                    next_row_id,
                    combined_cell_queue,
                    cell_buffer,
                    psf_sigma,
                    ingest_config,
                    projection,
                    catalog,
                    enable_saturation_correction,
                    remove_saturated_stars,
                    csv_path,
                    pipeline_paused_event=pipeline_paused_event,
                    band_cache=band_cache,
                    band_cache_uses=band_cache_uses,
                    row_padding_map=row_padding_map,
                    bright_star_mag_threshold=bright_star_mag_threshold,
                    convolved_store_data_root=convolved_store_data_root,
                    combined_store_recipe=combined_store_recipe,
                    convolved_store_recipe=convolved_store_recipe,
                    write_per_scc_convolved_zarr=write_per_scc_convolved_zarr,
                )

                all_removed_stars.extend(row_removed_stars)

                # Track the exact skycells the saver will write for this row.
                produced_skycells.update(str(name) for name in results_data.keys())

                now = time.monotonic()
                if now - last_progress_log >= 30.0:
                    logger.info(
                        f"[Pipeline] Progress: projection {proj_idx}/{total_projections} "
                        f"row {i + 1}/{len(row_ids)}"
                    )
                    last_progress_log = now

                # Queue the Results
                processed_bundle = {"projection": projection, "row_id": current_row_id, "results_data": results_data, "results_masks": results_masks}
                results_queue.put(processed_bundle)
                logger.info(f"[SequentialProcessor] Finished processing and queued results for row {current_row_id}")

                # Advance the Window if not the last row
                if next_row_id is not None:
                    advance_sliding_window(state)
            except Exception:
                logger.exception(f"[SequentialProcessor] Critical failure processing row {current_row_id} for projection {projection}")
                # If a row fails, the sliding window state for this projection is likely corrupted.
                # Skip the rest of this projection.
                break

        logger.info(f"[SequentialProcessor] --- Finished sequential processing for projection: {projection} ---")
        logger.info(
            f"[Pipeline] Progress: projection {proj_idx + 1}/{total_projections} "
            f"row {len(row_ids)}/{len(row_ids)}"
        )

    # Shutdown Signal for the saver
    results_queue.put(None)

    return all_removed_stars, sorted(produced_skycells)


# --- Main Orchestrator ---


def run_modern_sliding_window_pipeline(
    sector: int,
    camera: int,
    ccd: int,
    data_root: str = "data",
    projections_limit: Optional[int] = None,
    psf_sigma: float = 40.0,
    ps1_source: str = "zarr",
    num_ingest_workers: int = 16,
    use_local_files: bool = False,
    local_data_path: str | None = None,
    enable_saturation_correction: bool = False,
    remove_saturated_stars: bool = True,
    catalog_path: Optional[str] = None,
    bright_star_mag_threshold: float = 13.0,
    use_shared_convolved_store: bool = False,
    write_per_scc_convolved_zarr: bool = True,
    oversampling_factor: int = 1,
    mapping_csv_path: str | None = None,
):
    """The top-level master orchestrator for the entire pipeline.

    ``oversampling_factor``/``mapping_csv_path`` select which mapping master
    skycells CSV to load. Callers running an OS4/MAPGRID=3 (e.g. tvwcs field
    geometry) SCC MUST pass these through -- otherwise this function silently
    falls back to the native OS1 CSV (via ``find_csv_file``), which drops the
    OS4-only support/edge cells that ``downsample``'s shared convolved-store
    completeness gate (``expected_convolved_skycells``) requires. See
    ``docs/markdown/stages/ps1_process_technical.md`` and the mapping/remap/
    downsample stages in ``dispatch.py``, which already resolve the correct
    CSV via ``resolved.mapping_root`` -- this must match.
    """
    global _child_processes
    _child_processes.clear()
    signal.signal(signal.SIGINT, shutdown_handler)

    logger.info(
        f"[Pipeline] Starting pipeline for sector {sector}, camera {camera}, ccd {ccd} "
        f"(ps1_source={ps1_source})"
    )
    from syndiff_pipeline.common.scc_paths import ps1_skycells_zarr_path

    zarr_path = str(ps1_skycells_zarr_path(data_root))
    ingest_config = {
        "ps1_source": ps1_source,
        "zarr_path": zarr_path,
        "use_local_files": use_local_files,
        "local_data_path": local_data_path,
    }
    output_path = f"{data_root}/convolved_results/sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        if mapping_csv_path is not None:
            csv_path = str(mapping_csv_path)
        elif int(oversampling_factor) > 1:
            from syndiff_pipeline.common.scc_paths import scc_mapping_master_skycells_csv

            csv_path = str(
                scc_mapping_master_skycells_csv(
                    data_root, sector, camera, ccd,
                    oversampling_factor=int(oversampling_factor),
                )
            )
        else:
            csv_path = find_csv_file(data_root, sector, camera, ccd)
        projections = get_projections_from_csv(csv_path)
        if projections_limit:
            projections = projections[:projections_limit]
        df = load_csv_data(csv_path)
        expected_skycells = expected_convolved_skycells(
            data_root, sector, camera, ccd,
            projections_limit=projections_limit,
            oversampling_factor=oversampling_factor,
            mapping_csv_path=mapping_csv_path,
        )
        logger.info(f"[Pipeline] Processing {len(projections)} projections")
    except Exception as e:
        logger.error(f"[Pipeline] Failed to load configuration: {e}")
        return {"error": str(e)}

    if remove_saturated_stars and enable_saturation_correction:
        logger.info("[Pipeline] remove_saturated_stars enabled; skipping catalog-based saturation correction.")

    # Load Gaia catalog.  Used for:
    #   - apply_saturation_to_row (when enable_saturation_correction and not remove_saturated_stars)
    #   - catalog-based segment removal in remove_background (when remove_saturated_stars)
    #
    # A failure here must abort the run rather than silently degrade to
    # catalog=None: remove_background()'s primary (TESS-mag) removal pass is
    # a no-op without a catalog, so a swallowed failure here means every
    # skycell this run publishes only gets the magnitude-blind PS1-mask
    # fallback pass -- and because the combined-skycell store is a shared,
    # content-addressed, cross-sector cache keyed by recipe (not by which
    # run produced it or whether its catalog load actually succeeded), that
    # catalog-less build then gets silently reused by every other SCC that
    # ever resolves the same nominal recipe. See docs/markdown/bookkeeping.md
    # and storage_layout.md for the shared-store fingerprint contract.
    catalog = None
    if enable_saturation_correction or remove_saturated_stars:
        try:
            catalog = load_gaia_catalog(data_root, sector, camera, ccd, catalog_path)
            logger.info(f"[Pipeline] Loaded {len(catalog)} stars from catalog")
        except Exception as e:
            logger.error(f"[Pipeline] Failed to load Gaia catalog (required for "
                         f"remove_saturated_stars/enable_saturation_correction): {e}")
            return {"error": f"Gaia catalog load failed: {e}"}

    # --- Setup ---
    zarr_store = None
    if ps1_source == "zarr":
        zarr_store = zarr.open(zarr_path, mode="r")
    num_ingest_workers = max(1, int(num_ingest_workers))
    num_band_combiners = 4

    # Scale source-extractor subprocess count to available memory.
    # Each forked worker inherits a CoW snapshot; under load each can dirty ~4 GB
    # (reduced from ~10 GB since raw bands are no longer passed to subprocesses).
    # Default: min(ncpus // 2, available_gb // 4), clamped to [2, ncpus // 2].
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        ncpus = os.cpu_count() or 8
        mem_limit = max(2, int(available_gb // 4))
        num_source_extractors = max(2, min(ncpus // 2, mem_limit))
        logger.info(
            f"[Pipeline] Workers: ingest={num_ingest_workers}, band_combiners={num_band_combiners}, "
            f"source_extractors={num_source_extractors} "
            f"(available RAM: {available_gb:.1f} GB, cpus: {ncpus})"
        )
    except ImportError:
        num_source_extractors = 8
        logger.info(f"[Pipeline] Using {num_source_extractors} source extractors (psutil unavailable)")

    task_queue = _thread_queue.Queue()
    raw_cell_queue = _thread_queue.Queue(maxsize=max(6, num_ingest_workers * 2))
    combined_raw_queue = _thread_queue.Queue(maxsize=12)
    combined_cell_queue = _thread_queue.Queue(maxsize=30)
    # results_queue was previously a multiprocessing.Queue (crossing a process boundary
    # to saver_proc). That caused POSIX semaphore corruption: ProcessPoolExecutor workers
    # are forked after the queue is created and inherit its semaphore; when workers exit
    # their finalizers can call acquire() without a release(), draining the semaphore to
    # zero and causing results_queue.put() to block indefinitely. Fix: run the saver as a
    # daemon thread instead of a process so results_queue can be a plain thread queue.
    results_queue = _thread_queue.Queue(maxsize=3)
    cell_buffer = {}  # Shared buffer for all projections

    # Fix 7 — Build band_cache structures and identify all padding source cells.
    # We do NOT dispatch padding tasks upfront; instead they are dispatched JIT
    # (just-in-time) inside sequential_processor at the start of each row step.
    band_cache: dict = {}
    band_cache_uses: dict = {}
    row_padding_map: dict = {}
    padding_sources: dict = {}  # skycell_name -> source_projection
    all_regular_cell_names: set = set()
    for projection in projections:
        try:
            meta_tmp = extract_projection_metadata(df, projection)
            for cells in meta_tmp["rows"].values():
                for cell_name, _ in cells:
                    all_regular_cell_names.add(cell_name)
        except Exception:
            pass

    if csv_path:
        try:
            logger.info("[Pipeline] Identifying all cross-projection padding source cells...")
            padding_sources, padding_uses, row_padding_map = identify_all_padding_sources(
                projections, df, csv_path
            )
            # Merge use counts (padding uses only; regular uses added below)
            for skycell_name, uses in padding_uses.items():
                band_cache_uses[skycell_name] = band_cache_uses.get(skycell_name, 0) + uses

            # Add +1 use for dual-role cells (will also be consumed as regular cells)
            for skycell_name in padding_sources:
                if skycell_name in all_regular_cell_names:
                    band_cache_uses[skycell_name] = band_cache_uses.get(skycell_name, 0) + 1

            logger.info(
                f"[Pipeline] Found {len(padding_sources)} unique padding source cells "
                f"({len([s for s in padding_sources if s in all_regular_cell_names])} dual-role). "
                f"Will be dispatched JIT as each row step begins."
            )
        except Exception as e:
            logger.warning(f"[Pipeline] Failed to identify padding sources: {e}. Continuing without cache.")

    # Shared combined-skycell store: compute this run's recipe once. The
    # actual per-cell lookup happens lazily inside ingest_worker (tier-2
    # check) instead of an eager bulk preload here -- see
    # doc/ps1_process_tiered_ingest_architecture_plan.md Part A. The eager
    # preload used to np.load() every regular cell's combined arrays into
    # ``band_cache`` upfront, unconditionally, which OOM'd on SCCs with
    # hundreds of already-combined cells (measured ~248MB/cell).
    combined_store_recipe = None
    try:
        from syndiff_pipeline.template_creation.processing.combined_store import (
            production_combined_recipe,
        )

        combined_store_recipe = production_combined_recipe(
            {
                "enable_saturation_correction": enable_saturation_correction,
                "remove_saturated_stars": remove_saturated_stars,
                "bright_star_mag_threshold": bright_star_mag_threshold,
                "catalog_path": catalog_path,
            },
            data_root=data_root,
            sector=sector,
            camera=camera,
            ccd=ccd,
        )
    except Exception as e:
        logger.warning(
            "[Pipeline] Failed to compute combined-store recipe (continuing without "
            "shared-store cache): %s",
            e,
            exc_info=True,
        )

    convolved_store_recipe = None
    if use_shared_convolved_store:
        try:
            from syndiff_pipeline.template_creation.processing.convolved_store import (
                convolved_recipe,
            )

            convolved_store_recipe = convolved_recipe(psf_sigma=psf_sigma)
            logger.info(
                "[Pipeline] Shared convolved store enabled (canonical same-projection cells)."
            )
        except Exception as e:
            logger.warning(
                "[Pipeline] Convolved-store recipe setup failed (continuing without "
                "shared convolved publish): %s",
                e,
                exc_info=True,
            )

    # Fix 3 — event to pause coordinator new-submissions during cross-projection padding
    pipeline_paused_event = threading.Event()

    # --- Start Persistent Workers (Stages 1, 1.5, 2, 4) ---
    saver_thread = threading.Thread(
        target=saver_worker,
        args=(results_queue, output_path),
        kwargs={"enabled": bool(write_per_scc_convolved_zarr)},
        daemon=True,
    )
    saver_thread.start()

    band_combiner_threads = []
    for _ in range(num_band_combiners):
        t = threading.Thread(target=band_combiner_worker, args=(raw_cell_queue, combined_raw_queue), daemon=True)
        t.start()
        band_combiner_threads.append(t)

    process_coordinator_thread = threading.Thread(
        target=process_coordinator,
        args=(combined_raw_queue, combined_cell_queue, cell_buffer, num_source_extractors,
              band_cache, band_cache_uses, pipeline_paused_event,
              catalog if remove_saturated_stars else None,
              bright_star_mag_threshold),
        kwargs={
            "combined_store_data_root": data_root,
            "combined_store_recipe": combined_store_recipe,
        },
        daemon=True,
    )
    process_coordinator_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=num_ingest_workers) as ingest_executor:
            for _ in range(num_ingest_workers):
                ingest_executor.submit(
                    ingest_worker,
                    task_queue,
                    raw_cell_queue,
                    ps1_source=ps1_source,
                    zarr_store=zarr_store,
                    use_local_files=use_local_files,
                    local_data_path=local_data_path,
                    remove_saturated_stars=remove_saturated_stars,
                    band_cache=band_cache,
                    combined_store_data_root=data_root,
                    combined_store_recipe=combined_store_recipe,
                )

            # --- Part B: per-projection classification (tiered ingest plan) ---
            # Cheap upfront classification (fingerprint checks only, no pixel
            # loads) partitions projections into:
            #   - already fully canonical: nothing to do at all
            #   - sparse (<= MISSING_CELL_THRESHOLD missing): per-skycell path,
            #     bypasses the worker pipeline entirely (falls back to dense
            #     if any required cell needs a cold raw fetch)
            #   - dense (> MISSING_CELL_THRESHOLD missing, or classification
            #     unavailable/disabled): unchanged sliding-row worker pipeline
            # See doc/ps1_process_tiered_ingest_architecture_plan.md, Part B.
            MISSING_CELL_THRESHOLD = 5
            dense_projections: list = []
            already_canonical_cells: set = set()
            sparse_published: set = set()

            classify_enabled = (
                use_shared_convolved_store
                and combined_store_recipe is not None
                and convolved_store_recipe is not None
            )
            if not classify_enabled:
                dense_projections = list(projections)
            else:
                from syndiff_pipeline.template_creation.processing.convolved_store import (
                    classify_projection_missing_cells,
                )

                for projection in projections:
                    try:
                        metadata = extract_projection_metadata(df, projection)
                    except Exception as e:
                        logger.error(
                            f"[Pipeline] Failed to load metadata for projection "
                            f"{projection} during classification: {e}. Routing to "
                            f"dense/full-loop path."
                        )
                        dense_projections.append(projection)
                        continue

                    all_cells_in_proj = [
                        name for cells in metadata["rows"].values() for name, _ in cells
                    ]
                    missing_cells = classify_projection_missing_cells(
                        data_root, projection, all_cells_in_proj,
                        combined_store_recipe, convolved_store_recipe,
                    )
                    canonical_cells = set(all_cells_in_proj) - missing_cells

                    if not missing_cells:
                        logger.info(
                            f"[Pipeline] Projection {projection} already fully canonical "
                            f"({len(canonical_cells)} cells) -- nothing to do."
                        )
                        already_canonical_cells.update(canonical_cells)
                        continue

                    if len(missing_cells) <= MISSING_CELL_THRESHOLD:
                        logger.info(
                            f"[Pipeline] Projection {projection}: {len(missing_cells)} "
                            f"missing cell(s) -- trying per-skycell path."
                        )
                        published = process_sparse_projection(
                            projection, metadata, missing_cells, band_cache,
                            data_root, combined_store_recipe, convolved_store_recipe,
                            psf_sigma,
                        )
                        if published is None:
                            logger.info(
                                f"[Pipeline] Projection {projection}: per-skycell path "
                                f"declined (cold cell present) -- falling back to "
                                f"dense/full-loop path."
                            )
                            dense_projections.append(projection)
                        else:
                            already_canonical_cells.update(canonical_cells)
                            sparse_published.update(published)
                    else:
                        logger.info(
                            f"[Pipeline] Projection {projection}: {len(missing_cells)} "
                            f"missing cells (> {MISSING_CELL_THRESHOLD}) -- dense/"
                            f"full-loop path."
                        )
                        dense_projections.append(projection)

            # --- Build Interleaved Task List ---
            logger.info(f"[Pipeline] Building interleaved task list for {len(dense_projections)} projections.")

            master_task_list = []
            num_regular_tasks = 0
            num_padding_tasks = 0

            for projection in dense_projections:
                try:
                    metadata = extract_projection_metadata(df, projection)
                    row_ids = sorted(metadata["rows"].keys())
                    already_dispatched_padding: set = set()

                    for i, row_id in enumerate(row_ids):
                        for cell_name, x_coord in metadata["rows"][row_id]:
                            master_task_list.append((cell_name, projection, row_id, x_coord))
                            num_regular_tasks += 1

                        if i >= 1:
                            step_row = row_ids[i - 1]
                            for skycell_name in row_padding_map.get((str(projection), step_row), set()):
                                if skycell_name not in already_dispatched_padding:
                                    source_proj = padding_sources.get(skycell_name)
                                    if source_proj:
                                        master_task_list.append((skycell_name, source_proj, -1, 0, "padding_source"))
                                        already_dispatched_padding.add(skycell_name)
                                        num_padding_tasks += 1

                    last_row = row_ids[-1]
                    for skycell_name in row_padding_map.get((str(projection), last_row), set()):
                        if skycell_name not in already_dispatched_padding:
                            source_proj = padding_sources.get(skycell_name)
                            if source_proj:
                                master_task_list.append((skycell_name, source_proj, -1, 0, "padding_source"))
                                already_dispatched_padding.add(skycell_name)
                                num_padding_tasks += 1

                except Exception as e:
                    logger.error(f"[Pipeline] Failed to create tasks for projection {projection}: {e}")

            for task in master_task_list:
                task_queue.put(task)
            logger.info(
                f"[Pipeline] Dispatched {num_regular_tasks} regular tasks and "
                f"{num_padding_tasks} padding source tasks (interleaved). "
                f"Sending reader shutdown signals now."
            )

            for _ in range(num_ingest_workers):
                task_queue.put(None)

            # --- Run Sequential Processor (Stage 3) in Main Thread ---
            all_removed_stars, produced_skycells_list = sequential_processor(
                dense_projections,
                df,
                combined_cell_queue,
                results_queue,
                psf_sigma,
                ingest_config,
                cell_buffer,
                catalog,
                enable_saturation_correction,
                remove_saturated_stars,
                csv_path,
                pipeline_paused_event=pipeline_paused_event,
                band_cache=band_cache,
                band_cache_uses=band_cache_uses,
                row_padding_map=row_padding_map,
                bright_star_mag_threshold=bright_star_mag_threshold,
                convolved_store_data_root=data_root if use_shared_convolved_store else None,
                combined_store_recipe=combined_store_recipe,
                convolved_store_recipe=convolved_store_recipe,
                write_per_scc_convolved_zarr=write_per_scc_convolved_zarr,
            )
            # Fast-path manifest accounting: union in cells that were never
            # touched this run because they were already canonical (whole
            # projections skipped entirely, or canonical neighbors within a
            # sparse projection), plus cells freshly published by the
            # per-skycell path. Not a correctness requirement -- verify_ps1_process
            # independently rescans the shared convolved store regardless
            # (see doc/ps1_process_tiered_ingest_architecture_plan.md) -- but
            # keeps the scheduler's fast-path manifest check accurate too.
            produced_skycells = sorted(
                set(produced_skycells_list) | already_canonical_cells | sparse_published
            )

            # --- Write Removed Stars CSV ---
            removed_stars_path = output_path.replace(".zarr", "_removed_stars.csv")
            if all_removed_stars:
                pd.DataFrame(all_removed_stars).to_csv(removed_stars_path, index=False)
                logger.info(f"[Pipeline] Wrote {len(all_removed_stars)} removed-star records to {removed_stars_path}")
            else:
                logger.info("[Pipeline] No removed stars to save.")

            # --- Final Shutdown Sequence ---
            # Send one None per band combiner thread; each forwards its None to
            # combined_raw_queue, so the coordinator gets exactly one None too.
            for _ in range(num_band_combiners):
                raw_cell_queue.put(None)

            for t in band_combiner_threads:
                t.join(timeout=60)
            logger.info("[Pipeline] All band combiner threads finished.")

            if process_coordinator_thread.is_alive():
                logger.info("[Pipeline] Waiting for process coordinator to finish...")
                process_coordinator_thread.join(timeout=30)

        saver_thread.join()
        logger.info("[Pipeline] Pipeline completed successfully!")

        # Produced inventory: exact skycells written this run plus the planned
        # (expected) set, so a caller can write a completion manifest without
        # re-deriving what the pipeline did.
        try:
            expected_skycells = expected_convolved_skycells(
                data_root, sector, camera, ccd,
                projections_limit=projections_limit,
                oversampling_factor=oversampling_factor,
                mapping_csv_path=mapping_csv_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[Pipeline] Could not derive expected skycell inventory: %s", exc)
            expected_skycells = produced_skycells
        return {
            "status": "success",
            "output_path": output_path,
            "produced_skycells": produced_skycells,
            "produced_count": len(produced_skycells),
            "expected_skycells": expected_skycells,
            "expected_count": len(expected_skycells),
            "artifacts": [f"{output_path}/{name}_data" for name in produced_skycells],
        }

    except Exception:
        logger.exception("[Pipeline] Unhandled exception — cleaning up child processes")
        raise
    finally:
        _cleanup_child_processes()


def shutdown_handler(signum, frame):
    """
    Handles SIGINT (Ctrl+C) to ensure all child processes are terminated.
    """
    logger.warning("[Pipeline] Ctrl+C detected! Initiating graceful shutdown...")
    _cleanup_child_processes()
    logger.info("[Pipeline] All child processes terminated. Exiting.")
    sys.exit(1)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Syndiff Template PS1 Processing Pipeline")
    parser.add_argument("sector", type=int, help="TESS sector number")
    parser.add_argument("camera", type=int, help="TESS camera number")
    parser.add_argument("ccd", type=int, help="TESS CCD number")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--limit", type=int, help="Limit projections for testing")
    parser.add_argument("--psf-sigma", type=float, default=40.0, help="PSF sigma for convolution")
    parser.add_argument("--enable-saturation-correction", action="store_true", default=False, help="Enable saturation correction")
    parser.add_argument("--remove-saturated-stars", action="store_true", default=False, help="Remove saturated stars during background removal")
    parser.add_argument("--catalog-path", help="Path to Gaia catalog CSV file")
    parser.add_argument(
        "--bright-star-mag-threshold", type=float, default=13.0,
        help="Gaia TESS-equivalent magnitude threshold for catalog-based segment removal (default: 13.0)",
    )
    args = parser.parse_args()
    results = run_modern_sliding_window_pipeline(
        args.sector,
        args.camera,
        args.ccd,
        data_root=args.data_root,
        projections_limit=args.limit,
        psf_sigma=args.psf_sigma,
        enable_saturation_correction=args.enable_saturation_correction,
        remove_saturated_stars=args.remove_saturated_stars,
        catalog_path=args.catalog_path,
        bright_star_mag_threshold=args.bright_star_mag_threshold,
    )

    if results.get("status") == "success":
        print("\n✅ Syndiff Template PS1 processing pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
