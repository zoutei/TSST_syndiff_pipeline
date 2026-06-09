"""
Downloads PS1 skycell data and stores it in a single Zarr array.

This script combines the download and data conversion steps into a single,
efficient pipeline. It downloads skycell FITS files, decompresses them in
memory, and writes the data directly to a single Zarr store.

Key Features:
- Parallel Downloads: Uses Dask for distributed proce    # Download and process the band data
    result = download_and_process_band(skycell_name_parts, band, data_type, use_local_files, local_data_path)
    if result:
        # Store data in zarr
        store_data_in_zarr(root=root, projection_id=projection_id, skycell_name=skycell_name, band=band, data=result["data"], header=result["header"], array_name=array_name, lock_file=lock_file)
        return Trueof skycells.
- Direct-to-Zarr: Avoids saving intermediate FITS files to disk, reducing
  I/O and speeding up the process significantly.
- Single Zarr Store: Organizes all data into one Zarr store for easier management
  and querying.
- Thread-Safe Writing: Uses filelock for thread-safe operations.
- Efficient Organization: Data is organized by projection, skycell, band, and type
  (image, mask, weight) in a hierarchical structure.
- Resumable: The script is idempotent. If stopped, it can be restarted and
  will skip any skycells that have already been downloaded and stored.
- Local Files Support: Can use locally saved FITS files instead of downloading
  when available (useful for faster processing if data is already downloaded).
- Callable Function: Can be used both as a command-line script and as a Python
  function imported from other scripts.

Usage:
    Command line:
        python download_and_store_zarr.py                     # Download mode (ERROR logging by default)
        python download_and_store_zarr.py --log-level INFO    # Show informational messages
        python download_and_store_zarr.py --log-level DEBUG   # Show debug messages
        python download_and_store_zarr.py --use-local-files   # Use local files with default path
        python download_and_store_zarr.py --use-local-files --local-data-path "data/my_ps1_data"
        python download_and_store_zarr.py --sector 21 --camera 2 --ccd 1 --num-workers 16

    From Python script:
        from download_and_store_zarr import download_and_store_ps1_data

        # Basic usage (ERROR logging by default)
        result = download_and_store_ps1_data()

        # With custom parameters including log level
        result = download_and_store_ps1_data(
            sector=21, camera=2, ccd=1,
            use_local_files=True,
            local_data_path="data/my_ps1_data",
            log_level="INFO"  # Show informational messages
        )

        # Check result
        if result['status'] == 'completed':
            print(f"Processing successful! Zarr saved to: {result['zarr_path']}")
        else:
            print(f"Processing failed: {result['message']}")
"""

import argparse
import concurrent.futures
import io
import logging
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import dask.bag as db
import numpy as np
import pandas as pd
import requests
import zarr
from astropy.io import fits
from dask.diagnostics import Callback, ProgressBar
from filelock import FileLock

from syndiff_pipeline.template import csv_utils

# --- Configuration ---
# Default logging level is set in main() based on command line arguments
# Global variables for signal handling
shutdown_requested = False
active_executors = []
active_dask_computations = []
active_writers: list["ZarrWriter"] = []

_WRITE_QUEUE_MAXSIZE = 4
_WRITE_SENTINEL = object()

logger = logging.getLogger(__name__)

_PS1_BANDS = ("r", "i", "z", "y")
_ARRAYS_PER_SKYCELL = len(_PS1_BANDS) * 3


def expected_array_names() -> list[str]:
    names: list[str] = []
    for band in _PS1_BANDS:
        names.extend([band, f"{band}_mask", f"{band}_wt"])
    return names


@dataclass
class ArrayWriteItem:
    projection_id: str
    skycell_name: str
    array_name: str
    band: str
    data: np.ndarray
    header: str


@dataclass
class _WriteBatch:
    items: list[ArrayWriteItem]
    done: threading.Event = field(default_factory=threading.Event)
    written: bool = False


class SkycellProgress:
    """Thread-safe counter for skycell start/finish progress lines."""

    def __init__(self, total: int) -> None:
        self.total = total
        self._lock = threading.Lock()
        self._started = 0
        self._finished = 0

    def mark_started(self) -> tuple[int, int]:
        with self._lock:
            self._started += 1
            return self._started, self.total

    def mark_finished(self) -> tuple[int, int]:
        with self._lock:
            self._finished += 1
            return self._finished, self.total


class LogProgressCallback(Callback):
    """Log aggregate Dask progress when stdout is not a TTY (pipeline log files)."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self._start_time = time.monotonic()

    def _posttask(self, key, result, dsk, state, id):  # noqa: ARG002
        self.completed += 1
        elapsed = time.monotonic() - self._start_time
        logging.info(
            "Dask progress: %d/%d skycells finished (elapsed %.0fs)",
            self.completed,
            self.total,
            elapsed,
        )


def _array_complete_unlocked(
    root: zarr.Group,
    projection_id: str,
    skycell_name: str,
    array_name: str,
) -> bool:
    """Check array completeness; caller must hold *lock_file*."""
    if projection_id not in root or skycell_name not in root[projection_id]:
        return False
    if array_name not in root[projection_id][skycell_name]:
        return False

    array = root[projection_id][skycell_name][array_name]
    if array.shape == (0,) or array.size == 0:
        return False

    try:
        _ = array[0:1, 0:1]
        return True
    except Exception:
        logging.warning(
            "Found corrupted array %s for %s, will re-download",
            array_name,
            skycell_name,
        )
        return False


def skycell_array_status(
    root: zarr.Group,
    projection_id: str,
    skycell_name: str,
    lock_file: Path,
    overwrite: bool = False,
) -> dict[str, bool]:
    """Return completion status for all expected arrays under one lock."""
    if overwrite:
        return {name: False for name in expected_array_names()}

    try:
        with FileLock(lock_file):
            return {
                array_name: _array_complete_unlocked(root, projection_id, skycell_name, array_name)
                for array_name in expected_array_names()
            }
    except Exception:
        return {name: False for name in expected_array_names()}


def count_complete_arrays(
    root: zarr.Group,
    projection_id: str,
    skycell_name: str,
    lock_file: Path,
    overwrite: bool = False,
) -> int:
    return sum(
        1 for complete in skycell_array_status(root, projection_id, skycell_name, lock_file, overwrite).values() if complete
    )


def signal_handler(signum, frame):
    """
    Handle Ctrl+C (SIGINT) and other termination signals.
    Gracefully shutdown all active processes and threads.
    """
    global shutdown_requested
    logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True

    # Shutdown all active thread pool executors
    for executor in active_executors:
        try:
            logging.info("Shutting down thread pool executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logging.warning(f"Error shutting down executor: {e}")

    # Cancel any active Dask computations
    for computation in active_dask_computations:
        try:
            logging.info("Cancelling Dask computation...")
            computation.cancel()
        except Exception as e:
            logging.warning(f"Error cancelling Dask computation: {e}")

    for writer in active_writers:
        try:
            logging.info("Closing Zarr writer...")
            writer.close(drain=False)
        except Exception as e:
            logging.warning(f"Error closing Zarr writer: {e}")

    logging.info("Graceful shutdown initiated. Exiting...")
    # 128+signal is the conventional exit code for signal termination (e.g. 143
    # for SIGTERM). Exit 0 would make Condor and the scheduler treat a kill as
    # success when this module is imported by unrelated stages (e.g. mapping).
    raise SystemExit(128 + signum)


def get_projection_from_name(skycell_name: str) -> Optional[str]:
    """Extracts the projection ID from a skycell name."""
    try:
        return skycell_name.split(".")[1]
    except IndexError:
        logging.warning(f"Could not parse projection from skycell name: {skycell_name}")
        return None


def load_local_fits_file(skycell_name_parts: list, band: str, data_type: str, local_data_path: Path) -> Optional[dict[str, Any]]:
    """
    Load a FITS file from local storage if it exists.

    Args:
        skycell_name_parts: List of skycell name components
        band: Band name (r, i, z, y)
        data_type: Type of data ("image", "mask", "weight")
        local_data_path: Base path to local PS1 data

    Returns:
        dict with 'data' and 'header' keys, or None if file not found
    """
    projection, cell = skycell_name_parts[1], skycell_name_parts[2]

    # Construct local file path
    filename = f"rings.v3.skycell.{projection}.{cell}.stk.{band}.unconv.fits"
    if data_type == "mask":
        filename = filename.replace(".fits", ".mask.fits")
    elif data_type == "weight":
        filename = filename.replace(".fits", ".wt.fits")

    local_file_path = local_data_path / projection / cell / filename

    if not local_file_path.exists():
        logging.debug(f"Local file not found: {local_file_path}")
        return None

    try:
        logging.info(f"Loading local file: {local_file_path}")
        with fits.open(local_file_path) as hdul:
            hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            if data_type == "mask":
                data = hdu.data.astype(np.uint16)
            else:  # image or weight
                data = hdu.data.astype(np.float32)
            header = hdu.header.tostring()
            return {"data": data, "header": header}

    except Exception as e:
        logging.error(f"Error loading local file {local_file_path}: {e}")
        return None


def download_and_process_band(
    skycell_name_parts: list,
    band: str,
    data_type: str,
    use_local_files: bool = False,
    local_data_path: Optional[Path] = None,
    skycell_name: str = "",
) -> Optional[dict[str, Any]]:
    """
    Downloads a single FITS file (.fz) and processes it directly in memory,
    avoiding temporary disk I/O operations for better performance.

    If use_local_files is True, will first try to load from local storage before downloading.
    """
    # Try to load from local files first if enabled
    if use_local_files and local_data_path:
        local_result = load_local_fits_file(skycell_name_parts, band, data_type, local_data_path)
        if local_result is not None:
            if skycell_name:
                logging.info("Downloaded and converted %s %s %s", skycell_name, band, data_type)
            return local_result
        # If local file not found, fall back to downloading
        logging.info(f"Local file not found for {skycell_name_parts[1]}.{skycell_name_parts[2]} {band}_{data_type}, downloading...")

    projection, cell = skycell_name_parts[1], skycell_name_parts[2]

    file_path_in_repo = f"{projection}/{cell}/rings.v3.skycell.{projection}.{cell}.stk.{band}.unconv.fits"
    if data_type == "mask":
        file_path_in_repo = file_path_in_repo.replace(".fits", ".mask.fits")
    elif data_type == "weight":
        file_path_in_repo = file_path_in_repo.replace(".fits", ".wt.fits")

    url = f"http://ps1images.stsci.edu/rings.v3.skycell/{file_path_in_repo}"

    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            logging.warning(f"Failed to download {url}, status: {response.status_code}")
            return None

        # Process the FITS data directly in memory
        try:
            # First try direct in-memory decompression with astropy
            with fits.open(io.BytesIO(response.content)) as hdul:
                hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                if data_type == "mask":
                    data = hdu.data.astype(np.uint16)
                else:  # image or weight
                    data = hdu.data.astype(np.float32)
                header = hdu.header.tostring()
                logging.info("Downloaded and converted %s %s %s", skycell_name, band, data_type)
                return {"data": data, "header": header}
        except Exception as fits_error:
            # If direct decompression fails, fall back to funpack (should be rare)
            logging.warning(f"In-memory decompression failed for {url}, falling back to funpack: {fits_error}")
            with tempfile.TemporaryDirectory() as tmpdir:
                fz_path = Path(tmpdir) / "temp.fits.fz"
                fits_path = Path(tmpdir) / "temp.fits"

                with open(fz_path, "wb") as f:
                    f.write(response.content)

                # Use funpack to decompress the file
                subprocess.run(["funpack", "-D", str(fz_path)], check=True, capture_output=True)

                if not fits_path.exists():
                    logging.error(f"funpack failed to create {fits_path}")
                    return None

                with fits.open(fits_path) as hdul:
                    hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                    if data_type == "mask":
                        data = hdu.data.astype(np.uint16)
                    else:  # image or weight
                        data = hdu.data.astype(np.float32)
                    header = hdu.header.tostring()
                    logging.info("Downloaded and converted %s %s %s", skycell_name, band, data_type)
                    return {"data": data, "header": header}

    except (requests.exceptions.RequestException, subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.error(f"Error processing {url}: {e}")
        return None


def initialize_zarr_store(zarr_path: Path) -> zarr.Group:
    """
    Initializes a single Zarr store for all projections and skycells.
    Uses a flat structure: projection_id > skycell > bands/masks
    """
    # Create lock file path
    lock_file = zarr_path.parent / f"{zarr_path.name}.lock"

    # Use filelock for safe initialization
    with FileLock(lock_file):
        # Check if store exists, create if not
        if not zarr_path.exists():
            root = zarr.open(str(zarr_path), mode="w")
            logging.info(f"Created new zarr store at {zarr_path}")
        else:
            root = zarr.open(str(zarr_path), mode="a")

    return root


def _store_arrays_unlocked(
    root: zarr.Group,
    projection_id: str,
    skycell_name: str,
    items: list[ArrayWriteItem],
) -> None:
    """Write multiple arrays; caller must hold *lock_file*."""
    if projection_id not in root:
        root.create_group(projection_id)
    if skycell_name not in root[projection_id]:
        root[projection_id].create_group(skycell_name)

    skycell_group = root[projection_id][skycell_name]
    for item in items:
        chunks = (min(1024, item.data.shape[0]), min(1024, item.data.shape[1]))
        compressor = {"name": "zstd", "configuration": {"level": 3}}
        fill_value = 0 if "mask" in item.array_name else np.nan

        if item.array_name in skycell_group:
            del skycell_group[item.array_name]

        array = skycell_group.create_array(
            name=item.array_name,
            data=item.data,
            chunks=chunks,
            compressors=[compressor],
            fill_value=fill_value,
        )
        array.attrs["header"] = item.header


def store_skycell_batch(
    root: zarr.Group,
    projection_id: str,
    skycell_name: str,
    items: list[ArrayWriteItem],
    lock_file: Path,
) -> None:
    """Persist all *items* for one skycell under a single file lock."""
    if not items:
        return

    with FileLock(lock_file):
        _store_arrays_unlocked(root, projection_id, skycell_name, items)

    logging.info("Stored %d arrays for %s in zarr", len(items), skycell_name)


def store_data_in_zarr(
    root: zarr.Group,
    projection_id: str,
    skycell_name: str,
    band: str,
    data: np.ndarray,
    header: str,
    array_name: str,
    lock_file: Path,
) -> None:
    """Stores data for a single band/mask/weight in the zarr store."""
    store_skycell_batch(
        root,
        projection_id,
        skycell_name,
        [
            ArrayWriteItem(
                projection_id=projection_id,
                skycell_name=skycell_name,
                array_name=array_name,
                band=band,
                data=data,
                header=header,
            )
        ],
        lock_file,
    )
    logging.info("Stored %s %s in zarr", skycell_name, array_name)


class ZarrWriter:
    """Background consumer that serializes batched Zarr writes under one lock."""

    def __init__(self, root: zarr.Group, lock_file: Path) -> None:
        self._root = root
        self._lock_file = lock_file
        self._queue: queue.Queue[_WriteBatch | object] = queue.Queue(maxsize=_WRITE_QUEUE_MAXSIZE)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="ZarrWriter", daemon=True)
        self._thread.start()

    def submit_batch(self, items: list[ArrayWriteItem]) -> bool:
        """Enqueue a skycell batch and block until the writer persists it."""
        if not items:
            return True
        if shutdown_requested or self._closed:
            return False

        batch = _WriteBatch(items=items)
        self._queue.put(batch)
        batch.done.wait()
        return batch.written

    def close(self, drain: bool = True) -> None:
        if self._closed:
            return
        self._closed = True

        if not drain:
            while True:
                try:
                    pending = self._queue.get_nowait()
                except queue.Empty:
                    break
                if pending is not _WRITE_SENTINEL:
                    pending.written = False
                    pending.done.set()

        self._queue.put(_WRITE_SENTINEL)
        self._thread.join()

    def _run(self) -> None:
        while True:
            batch = self._queue.get()
            if batch is _WRITE_SENTINEL:
                break

            first = batch.items[0]
            try:
                store_skycell_batch(
                    self._root,
                    first.projection_id,
                    first.skycell_name,
                    batch.items,
                    self._lock_file,
                )
                batch.written = True
            except Exception as exc:
                logging.error(
                    "Failed to write batch for %s: %s",
                    first.skycell_name,
                    exc,
                )
                batch.written = False
            finally:
                batch.done.set()


def is_array_complete(root: zarr.Group, projection_id: str, skycell_name: str, array_name: str, lock_file: Path, overwrite: bool = False) -> bool:
    """
    Check if an array is complete and not corrupted.
    This helps detect arrays that were interrupted during writing.
    """
    if overwrite:
        return False

    try:
        with FileLock(lock_file):
            return _array_complete_unlocked(root, projection_id, skycell_name, array_name)
    except Exception:
        return False


def process_skycell_band(
    root: zarr.Group,
    skycell_name: str,
    skycell_name_parts: list,
    band: str,
    data_type: str,
    projection_id: str,
    lock_file: Path,
    use_local_files: bool = False,
    local_data_path: Optional[Path] = None,
    overwrite: bool = False,
    skip_if_complete: bool = True,
) -> bool:
    """
    Process a single band/data_type for a skycell and store it in the zarr store.
    """
    if data_type == "image":
        array_name = band
    elif data_type == "mask":
        array_name = f"{band}_mask"
    elif data_type == "weight":
        array_name = f"{band}_wt"
    else:
        logging.error(f"Unknown data_type: {data_type}")
        return False

    if skip_if_complete and is_array_complete(
        root, projection_id, skycell_name, array_name, lock_file, overwrite=overwrite
    ):
        logging.debug(f"Skipping existing complete {array_name} for {skycell_name}")
        return True

    # Download and process the band data
    result = download_and_process_band(
        skycell_name_parts,
        band,
        data_type,
        use_local_files,
        local_data_path,
        skycell_name=skycell_name,
    )
    if result:
        # Store data in zarr
        store_data_in_zarr(root=root, projection_id=projection_id, skycell_name=skycell_name, band=band, data=result["data"], header=result["header"], array_name=array_name, lock_file=lock_file)
        return True

    return False


def download_and_store_skycell(
    root: zarr.Group,
    skycell_name: str,
    lock_file: Path,
    writer: ZarrWriter,
    use_local_files: bool = False,
    local_data_path: Optional[Path] = None,
    overwrite: bool = False,
    progress: Optional[SkycellProgress] = None,
) -> None:
    """
    Manages the download and storage of a single skycell into the
    single Zarr store. Downloads run in parallel; writes go through *writer*.
    """
    projection_id = get_projection_from_name(skycell_name)
    if not projection_id:
        logging.warning(f"Could not parse projection from skycell name: {skycell_name}")
        return

    bands = list(_PS1_BANDS)
    expected_arrays = expected_array_names()
    array_status = skycell_array_status(root, projection_id, skycell_name, lock_file, overwrite=overwrite)
    complete_count = sum(1 for complete in array_status.values() if complete)

    if complete_count == len(expected_arrays):
        logging.info(
            "Skipping skycell %s (%d/%d arrays complete in zarr)",
            skycell_name,
            complete_count,
            len(expected_arrays),
        )
        if progress is not None:
            finished, total = progress.mark_finished()
            logging.info("Finished skycell %s (%d/%d)", skycell_name, finished, total)
        return

    if progress is not None:
        started, total = progress.mark_started()
        logging.info(
            "Processing skycell %s (%d/%d; %d/%d arrays already in zarr)",
            skycell_name,
            started,
            total,
            complete_count,
            len(expected_arrays),
        )
    else:
        logging.info(
            "Processing skycell %s (%d/%d arrays already in zarr)",
            skycell_name,
            complete_count,
            len(expected_arrays),
        )

    skycell_name_parts = skycell_name.split(".")
    pending_tasks: list[tuple[str, str, str]] = []
    for band in bands:
        for data_type, array_name in (
            ("image", band),
            ("mask", f"{band}_mask"),
            ("weight", f"{band}_wt"),
        ):
            if not array_status.get(array_name, False):
                pending_tasks.append((band, data_type, array_name))

    if not pending_tasks:
        if progress is not None:
            finished, total = progress.mark_finished()
            logging.info("Finished skycell %s (%d/%d)", skycell_name, finished, total)
        return

    if shutdown_requested:
        logging.info("Shutdown requested, stopping skycell processing")
        return

    def _download_one(task_band: str, data_type: str, array_name: str) -> Optional[ArrayWriteItem]:
        result = download_and_process_band(
            skycell_name_parts,
            task_band,
            data_type,
            use_local_files,
            local_data_path,
            skycell_name=skycell_name,
        )
        if result is None:
            return None
        return ArrayWriteItem(
            projection_id=projection_id,
            skycell_name=skycell_name,
            array_name=array_name,
            band=task_band,
            data=result["data"],
            header=result["header"],
        )

    items: list[ArrayWriteItem] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(pending_tasks))) as executor:
        active_executors.append(executor)
        try:
            futures = {
                executor.submit(_download_one, task_band, data_type, array_name): (task_band, data_type)
                for task_band, data_type, array_name in pending_tasks
            }
            for future in concurrent.futures.as_completed(futures):
                task_band, data_type = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    logging.warning(
                        "Failed to process %s %s for %s: %s",
                        task_band,
                        data_type,
                        skycell_name,
                        exc,
                    )
                    continue
                if item is None:
                    logging.warning(
                        "Failed to process %s %s for %s",
                        task_band,
                        data_type,
                        skycell_name,
                    )
                else:
                    items.append(item)
        finally:
            if executor in active_executors:
                active_executors.remove(executor)

    if items:
        if not writer.submit_batch(items):
            logging.warning("Batch write skipped for %s (shutdown)", skycell_name)

    if progress is not None:
        finished, total = progress.mark_finished()
        logging.info("Finished skycell %s (%d/%d)", skycell_name, finished, total)


def _process_one_skycell(
    skycell_name: str,
    root: zarr.Group,
    lock_file: Path,
    progress: SkycellProgress,
    writer: ZarrWriter,
    use_local_files: bool,
    local_data_path: Optional[Path],
    overwrite: bool,
) -> int:
    if shutdown_requested:
        logging.info("Shutdown requested, stopping skycell processing")
        return 0
    download_and_store_skycell(
        root,
        skycell_name,
        lock_file,
        writer,
        use_local_files=use_local_files,
        local_data_path=local_data_path,
        overwrite=overwrite,
        progress=progress,
    )
    return 1


def process_skycells_with_dask(root: zarr.Group, skycells: list, lock_file: Path, batch_size: int = 10, num_workers: int = 8, use_local_files: bool = False, local_data_path: Optional[Path] = None, overwrite: bool = False):
    """
    Process skycells using Dask for parallel orchestration (threaded scheduler).
    """
    progress = SkycellProgress(len(skycells))
    computation = None
    writer = ZarrWriter(root, lock_file)
    active_writers.append(writer)
    try:
        skycells_bag = db.from_sequence(skycells)

        def process_skycell(skycell_name: str) -> int:
            return _process_one_skycell(
                skycell_name,
                root,
                lock_file,
                progress,
                writer,
                use_local_files,
                local_data_path,
                overwrite,
            )

        computation = skycells_bag.map(process_skycell)
        active_dask_computations.append(computation)

        compute_kwargs = {"scheduler": "threads", "num_workers": num_workers}
        if sys.stdout.isatty():
            with ProgressBar():
                results = computation.compute(**compute_kwargs)
        else:
            logging.info("Processing %d skycells with Dask (%d workers)...", len(skycells), num_workers)
            with LogProgressCallback(total=len(skycells)):
                results = computation.compute(**compute_kwargs)

        logging.info("Processed %d skycells", sum(results))

    except KeyboardInterrupt:
        logging.info("Dask computation interrupted by user")
        return
    finally:
        writer.close()
        if writer in active_writers:
            active_writers.remove(writer)
        if "computation" in locals() and computation in active_dask_computations:
            active_dask_computations.remove(computation)


def download_and_store_ps1_data(sector=20, camera=3, ccd=3, num_workers=8, zarr_output_dir="data/ps1_skycells_zarr", use_local_files=False, local_data_path="data/ps1_skycells", log_level="ERROR", overwrite: bool = False):
    """
    Download PS1 skycell data and store it in a single Zarr array.

    This function can be called both from the command line and from other Python scripts.

    Parameters:
    -----------
    sector : int, default 20
        TESS sector number.
    camera : int, default 3
        TESS camera number.
    ccd : int, default 3
        TESS CCD number.
    num_workers : int, default 8
        Number of parallel workers for Dask.
    zarr_output_dir : str, default "data/ps1_skycells_zarr"
        Directory for Zarr output.
    use_local_files : bool, default False
        Whether to use local FITS files instead of downloading when available.
    local_data_path : str, default "data/ps1_skycells"
        Path to local FITS files directory.
    log_level : str, default "ERROR"
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns:
    --------
    dict
        Summary statistics about the processing.
    """
    # --- Configuration ---
    # Set up logging with the specified level
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}

    numeric_level = log_level_map.get(log_level.upper(), logging.ERROR)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,  # This ensures the level is updated even if basicConfig was called before
    )

    zarr_output_dir = Path(zarr_output_dir)
    zarr_output_file = zarr_output_dir / "ps1_skycells.zarr"
    lock_file = zarr_output_dir / "ps1_skycells.zarr.lock"

    # Local files configuration
    local_data_path = Path(local_data_path) if use_local_files else None

    if use_local_files:
        if not local_data_path.exists():
            logging.warning(f"Local data path does not exist: {local_data_path}")
            logging.info("Continuing with download-only mode...")
            use_local_files = False
            local_data_path = None
        else:
            logging.info(f"Using local files from: {local_data_path}")

    # Construct path to the skycell list CSV
    sector_str = f"sector_{sector:04d}"
    data_root = Path(zarr_output_dir).parent
    skycell_list_csv = (
        data_root
        / "skycell_pixel_mapping"
        / sector_str
        / f"camera_{camera}"
        / f"ccd_{ccd}"
        / f"tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list.csv"
    )

    if not skycell_list_csv.exists():
        error_msg = f"Skycell list not found: {skycell_list_csv}"
        logging.error(error_msg)
        return {"status": "error", "message": error_msg, "skycells_found": 0, "unique_images": 0}

    logging.info(f"Reading skycell list from {skycell_list_csv}")
    skycells_df = pd.read_csv(skycell_list_csv)

    # Get unique PS1 images
    unique_ps1_images = set(skycells_df["NAME"].unique())
    logger.info(f"Found {len(unique_ps1_images)} main skycells")

    # Get padding cells
    try:
        padding_map = csv_utils.get_all_padding_cells(str(skycell_list_csv), list(unique_ps1_images))
        padding_cells = set()
        for cells in padding_map.values():
            padding_cells.update(cells)
        
        num_padding = len(padding_cells)
        logger.info(f"Found {num_padding} additional padding skycells")
        
        # Merge padding cells
        unique_ps1_images.update(padding_cells)
        
    except Exception as e:
        logger.error(f"Failed to get padding cells: {e}")
        # Continue with just main cells if padding fails? 
        # Better to warn but continue, or fail? The user wants padding, so this is important.
        # But if the CSV doesn't support it or something, maybe we shouldn't crash the whole download.
        logger.warning("Continuing with main skycells only.")

    # Convert back to list for processing
    unique_ps1_images = sorted(list(unique_ps1_images))
    logger.info(f"Found {len(unique_ps1_images)} total skycells to process")

    # Create output directory
    zarr_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the single zarr store
    root = initialize_zarr_store(zarr_output_file)

    logging.info(f"Found {len(unique_ps1_images)} total skycells to process")

    # Check if shutdown was requested before starting main processing
    if shutdown_requested:
        logging.info("Shutdown requested, exiting before processing")
        return {"status": "interrupted", "message": "Processing interrupted by shutdown request", "skycells_found": len(skycells_df), "unique_images": len(unique_ps1_images)}

    # Process skycells using Dask
    process_skycells_with_dask(root, list(unique_ps1_images), lock_file, num_workers=num_workers, use_local_files=use_local_files, local_data_path=local_data_path, overwrite=overwrite)

    logging.info("Download and Zarr storage process completed.")

    # Produced inventory: skycells whose full set of 12 arrays is complete in the
    # store, plus their zarr group paths. Lets a caller write a completion manifest
    # reflecting exactly what is present without re-deriving it. Best-effort: never
    # fail the stage over an inventory scan.
    produced_skycells: list[str] = []
    produced_paths: list[str] = []
    try:
        for skycell_name in unique_ps1_images:
            projection_id = get_projection_from_name(skycell_name)
            if not projection_id:
                continue
            if count_complete_arrays(root, projection_id, skycell_name, lock_file) == _ARRAYS_PER_SKYCELL:
                produced_skycells.append(skycell_name)
                produced_paths.append(str(zarr_output_file / projection_id / skycell_name))
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Could not build produced skycell inventory: %s", exc)

    return {
        "status": "completed",
        "message": "Successfully processed all skycells",
        "skycells_found": len(skycells_df),
        "unique_images": len(unique_ps1_images),
        "zarr_path": str(zarr_output_file),
        "produced_skycells": produced_skycells,
        "produced_count": len(produced_skycells),
        "expected_count": len(unique_ps1_images),
        "artifacts": produced_paths,
    }


def main():
    """Main function to run the parallel download and store process."""
    parser = argparse.ArgumentParser(description="Download PS1 skycell data and store in single Zarr array")

    parser.add_argument("sector", type=int, help="TESS sector number")
    parser.add_argument("camera", type=int, help="TESS camera number")
    parser.add_argument("ccd", type=int, help="TESS CCD number")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of parallel workers for Dask")
    parser.add_argument("--zarr-output-dir", type=str, default="data/ps1_skycells_zarr", help="Directory for Zarr output")

    parser.add_argument("--use-local-files", action="store_true", help="Use locally saved FITS files instead of downloading when available")
    parser.add_argument("--local-data-path", type=str, default="data/ps1_skycells", help="Path to local PS1 skycell data directory")
    parser.add_argument("--log-level", type=str, default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level (default: WARNING)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing arrays in the Zarr store (default: skip existing)")

    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Call the main processing function
    result = download_and_store_ps1_data(sector=args.sector, camera=args.camera, ccd=args.ccd, num_workers=args.num_workers, zarr_output_dir=args.zarr_output_dir, use_local_files=args.use_local_files, local_data_path=args.local_data_path, log_level=args.log_level, overwrite=args.overwrite)

    # Print result summary
    print(f"\nProcessing completed with status: {result['status']}")
    print(f"Message: {result['message']}")
    if result["status"] == "completed":
        print(f"Zarr store saved to: {result['zarr_path']}")
    elif result["status"] == "error":
        print(f"Error occurred: {result['message']}")
        exit(1)


if __name__ == "__main__":
    main()
