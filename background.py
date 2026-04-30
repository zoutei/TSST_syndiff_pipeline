"""
background.py
=============
Steps 7 & 11 of the SynDiff pipeline:

  7. Mask difference image, apply TESSreduce-style smooth background.
  11. Combine round-1 and round-2 smoothed backgrounds; final adaptive temporal smooth
      is applied in ``temporal_smooth.compute_final_background``.

``Smooth_bkg`` and related helpers are vendored from the **TESSreduce** toolkit
(smooth background estimation on masked difference images).
"""

import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from skimage.restoration import inpaint

from .paths import BACKGROUND_STACK_NPZ_ARRAY_KEY

warnings.filterwarnings("ignore", category=RuntimeWarning)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Vendored from TESSreduce helpers module ───────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def strip_units(data):
    """Remove astropy units from data if not a plain ndarray."""
    if type(data) != np.ndarray:
        return data.value
    return data


def Smooth_bkg(data, gauss_smooth=2, interpolate=False, extrapolate=True):
    """
    Estimate background by interpolating over masked (NaN) pixels then
    applying a Gaussian smooth.

    Vendored from TESSreduce helpers.

    Parameters
    ----------
    data : 2D ndarray with NaN at masked positions
    gauss_smooth : float  (Gaussian sigma in pixels)
    interpolate  : bool   (if True: use griddata; if False: use skimage inpaint)
    extrapolate  : bool   (fill extrapolated NaNs with nearest neighbor)

    Returns
    -------
    2D ndarray — smoothed background estimate
    """
    if (~np.isnan(data)).any():
        x = np.arange(0, data.shape[1])
        y = np.arange(0, data.shape[0])
        arr = np.ma.masked_invalid(deepcopy(data))
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~arr.mask]
        y1 = yy[~arr.mask]
        newarr = arr[~arr.mask]
        if (len(x1) > 10) & (len(y1) > 10):
            if interpolate:
                estimate = griddata((x1, y1), newarr.ravel(), (xx, yy), method="linear")
                nearest  = griddata((x1, y1), newarr.ravel(), (xx, yy), method="nearest")
                if extrapolate:
                    estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
                estimate = gaussian_filter(estimate, gauss_smooth)
            else:
                mask = deepcopy(arr.mask).astype(bool)
                estimate = inpaint.inpaint_biharmonic(data, mask)
                if (np.nanmedian(estimate) < 100) & (np.nanstd(estimate) < 3):
                    gauss_smooth = gauss_smooth * 3
                estimate = gaussian_filter(estimate, gauss_smooth)
        else:
            estimate = np.zeros_like(data) * np.nan
    else:
        estimate = np.zeros_like(data)

    return estimate


def parallel_bkg3(data, mask):
    """
    Simple inpaint-based background (no Gaussian smooth).
    Vendored from TESSreduce helpers.
    """
    data = deepcopy(data)
    data[mask] = np.nan
    estimate = inpaint.inpaint_biharmonic(data, mask)
    return estimate


# ═══════════════════════════════════════════════════════════════════════════════
# ── New pipeline functions ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_frame_background(diff_image: np.ndarray,
                               hotpants_bkg: np.ndarray,
                               mask: np.ndarray,
                               gauss_smooth: float = 2.0,
                               recombine_hotpants: bool = False) -> tuple:
    """
    Estimate the per-frame smoothed residual and the stack slice for ``background_rough``.

    ``recombine_hotpants`` only changes what is fed to ``Smooth_bkg``:

    - False (default): ``to_smooth = diff_image`` (Hotpants diff, background already removed).
    - True: ``to_smooth = diff_image + hotpants_bkg`` (undo Hotpants polynomial in the diff
      domain before smoothing).

    The value persisted in ``rough_bkg_r*`` stacks is **only** the Gaussian-smoothed
    inpainted residual (``smooth_bkg``), **not** ``hotpants_bkg``. The pipeline adds
    ``hotpants_bkg`` back in memory in ``background_adaptive`` before temporal smoothing.

    Returns
    -------
    (rough_stack_slice, smooth_bkg)
        ``rough_stack_slice`` equals ``smooth_bkg`` (same array); kept as a pair for
        call-site clarity.
    """
    if recombine_hotpants:
        to_smooth = diff_image + hotpants_bkg
    else:
        to_smooth = diff_image

    mask_bool = (mask > 0)
    arr_masked = to_smooth.copy().astype(np.float64)
    arr_masked[mask_bool] = np.nan

    smooth_bkg = Smooth_bkg(arr_masked, gauss_smooth=gauss_smooth)
    rough_stack_slice = smooth_bkg

    return rough_stack_slice, smooth_bkg


def load_hotpants_row_from_disk(
    stem: str,
    diff_dir: str,
    bkg_dir: Optional[str],
    group_id: int = 0,
) -> dict:
    """
    Load one frame's Hotpants diff (and optional bkg) FITS from disk.

    Returns a dict compatible with :func:`_background_frame_worker` /
    :func:`background_loop` (same keys as pipeline hotpants result rows).
    """
    dp = os.path.join(diff_dir, f"{stem}.fits")
    if not os.path.isfile(dp):
        return {
            "stem": stem,
            "success": False,
            "diff": None,
            "bkg": None,
            "group_id": int(group_id),
        }
    diff_data = fits.getdata(dp).astype(np.float64)
    bkg_data = None
    if bkg_dir:
        bp = os.path.join(bkg_dir, f"{stem}.fits")
        if os.path.isfile(bp):
            bkg_data = fits.getdata(bp).astype(np.float64)
    return {
        "stem": stem,
        "success": True,
        "diff": diff_data,
        "bkg": bkg_data,
        "group_id": int(group_id),
        "path": dp,
    }


def _background_frame_worker(
    task: Tuple[int, dict, np.ndarray, float, bool],
) -> Tuple[int, Optional[np.ndarray]]:
    """
    One frame of rough background stack slice for :func:`background_loop`.

    Returns (frame_index, smooth_residual_float32) or (index, None) if skipped.
    """
    i, result, mask, gauss_smooth, recombine_hotpants = task
    if not result.get("success") or result.get("diff") is None:
        log.debug("  Frame %s: hotpants failed — background set to zero.", i)
        return i, None

    diff_img = result["diff"].astype(np.float64)
    hp_bkg = (
        result["bkg"].astype(np.float64)
        if result.get("bkg") is not None
        else np.zeros_like(diff_img)
    )

    rough_stack_slice, _ = estimate_frame_background(
        diff_img,
        hp_bkg,
        mask,
        gauss_smooth,
        recombine_hotpants=recombine_hotpants,
    )
    return i, rough_stack_slice.astype(np.float32)


def _load_and_rough_stream_worker(
    packed: Tuple[int, str, str, Optional[str], int, np.ndarray, float, bool],
) -> Tuple[int, Optional[np.ndarray]]:
    """Load one stem from disk then run :func:`_background_frame_worker` (for joblib)."""
    i, stem, diff_dir, bkg_dir, group_id, mask, gauss_smooth, recombine = packed
    row = load_hotpants_row_from_disk(stem, diff_dir, bkg_dir, group_id)
    return _background_frame_worker((i, row, mask, gauss_smooth, recombine))


def _parallel_map_with_optional_tqdm(
    delayed_calls,
    n_tasks: int,
    desc: str,
    n_jobs_eff: int,
):
    """
    Run joblib Parallel over *delayed_calls*; when tqdm is installed, show a bar
    using ``return_as='generator'`` (joblib >= 1.3).
    """
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return Parallel(n_jobs=n_jobs_eff, backend="loky")(delayed_calls)
    try:
        gen = Parallel(n_jobs=n_jobs_eff, backend="loky", return_as="generator")(
            delayed_calls
        )
        return list(tqdm(gen, total=n_tasks, desc=desc, unit="frame"))
    except TypeError:
        log.debug("joblib Parallel(return_as=...) unavailable; running without tqdm bar.")
        return Parallel(n_jobs=n_jobs_eff, backend="loky")(delayed_calls)


def _tqdm_iter(tasks: list, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(tasks, desc=desc, unit="frame")
    except ImportError:
        return tasks


def background_loop_streaming(
    ffi_paths: Sequence[Union[str, Path]],
    diff_dir: str,
    bkg_dir: Optional[str],
    path_to_group: Dict[str, int],
    mask: np.ndarray,
    output_dir: Optional[str] = None,
    round_id: int = 1,
    gauss_smooth: float = 2.0,
    recombine_hotpants: bool = False,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Per-frame load (diff/bkg FITS) + rough background estimate with optional parallelism.

    Unlike :func:`background_loop`, this never holds the full Hotpants diff/bkg cube
    in memory—each worker loads one frame, estimates, returns a 2D array. Peak RAM is
    roughly ``O(n_jobs * frame)`` plus the output stack.

    Parameters
    ----------
    ffi_paths
        Ordered list of FFI paths (only ``Path(ffi_path).stem`` is used).
    diff_dir, bkg_dir
        Hotpants diff / bkg FITS directories (same layout as the pipeline).
    path_to_group
        Mapping stem → ``group_id`` from the WCS table.
    mask, output_dir, round_id, gauss_smooth, recombine_hotpants, n_jobs
        Same meaning as in :func:`background_loop`.

    Returns
    -------
    ndarray (n_frames, ny, nx)
    """
    ffi_paths = list(ffi_paths)
    n_frames = len(ffi_paths)
    if n_frames == 0:
        raise RuntimeError("background_loop_streaming: empty ffi_paths.")

    shape = None
    for ffi_path in ffi_paths:
        stem = Path(ffi_path).stem
        dp = os.path.join(diff_dir, f"{stem}.fits")
        if os.path.isfile(dp):
            shape = fits.getdata(dp).shape
            break
    if shape is None:
        raise RuntimeError(
            "background_loop_streaming: no diff FITS found under diff_dir for any stem."
        )

    rough_bkg_stack = np.zeros((n_frames, *shape), dtype=np.float32)
    tasks = [
        (
            i,
            Path(ffi_path).stem,
            diff_dir,
            bkg_dir,
            int(path_to_group.get(Path(ffi_path).stem, 0)),
            mask,
            gauss_smooth,
            recombine_hotpants,
        )
        for i, ffi_path in enumerate(ffi_paths)
    ]
    n_jobs_eff = max(1, int(n_jobs or 1))
    parallel = n_jobs_eff != 1 and n_frames > 1

    if parallel:
        log.info(
            "  background_loop_streaming: per-frame load+rough bkg n_jobs=%s (loky), %d frames",
            n_jobs_eff,
            n_frames,
        )
        frame_results = _parallel_map_with_optional_tqdm(
            (delayed(_load_and_rough_stream_worker)(t) for t in tasks),
            n_frames,
            "Rough bkg (load+est)",
            n_jobs_eff,
        )
    else:
        frame_results = [
            _load_and_rough_stream_worker(t) for t in _tqdm_iter(tasks, "Rough bkg (load+est)")
        ]

    for i, rough_bkg in frame_results:
        if rough_bkg is not None:
            rough_bkg_stack[i] = rough_bkg

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"rough_bkg_r{round_id}")
        npz_path = f"{base}.npz"
        npy_path = f"{base}.npy"
        np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: rough_bkg_stack})
        np.save(npy_path, rough_bkg_stack)
        log.info(
            "  Rough background stack (round %s) saved to %s and %s",
            round_id,
            npz_path,
            npy_path,
        )

    return rough_bkg_stack


def background_loop(hotpants_results: list,
                    mask: np.ndarray,
                    output_dir: str = None,
                    round_id: int = 1,
                    gauss_smooth: float = 2.0,
                    recombine_hotpants: bool = False,
                    n_jobs: int = 1) -> np.ndarray:
    """
    Run estimate_frame_background for every frame and stack results.

    Parameters
    ----------
    hotpants_results : list of dicts from hotpants_runner.hotpants_loop
        Each dict must have keys: 'diff' (2D ndarray), 'bkg' (2D ndarray or None),
        'group_id' (int), 'success' (bool).
    mask             : 2D int ndarray (shared bitmask)
    output_dir       : str, optional — saves ``rough_bkg_rN.npz`` and ``rough_bkg_rN.npy``
    round_id         : int
    gauss_smooth     : float
    recombine_hotpants : bool
        Passed to :func:`estimate_frame_background`.
    n_jobs           : int
        When > 1 and more than one frame, run per-frame rough estimates with joblib **loky**.

    Returns
    -------
    ndarray (n_frames, ny, nx) — rough background per frame
    """
    n_frames = len(hotpants_results)
    # Determine shape from first successful frame
    shape = None
    for r in hotpants_results:
        if r.get("success") and r.get("diff") is not None:
            shape = r["diff"].shape
            break
    if shape is None:
        raise RuntimeError("No successful hotpants frames found — cannot compute background.")

    rough_bkg_stack = np.zeros((n_frames, *shape), dtype=np.float32)

    tasks = [
        (i, r, mask, gauss_smooth, recombine_hotpants)
        for i, r in enumerate(hotpants_results)
    ]
    n_jobs_eff = max(1, int(n_jobs or 1))
    parallel = n_jobs_eff != 1 and n_frames > 1

    if parallel:
        log.info(
            "  background_loop: per-frame rough bkg n_jobs=%s (loky), %d frames",
            n_jobs_eff,
            n_frames,
        )
        frame_results = _parallel_map_with_optional_tqdm(
            (delayed(_background_frame_worker)(t) for t in tasks),
            n_frames,
            "Rough bkg (estimate)",
            n_jobs_eff,
        )
    else:
        frame_results = [
            _background_frame_worker(t) for t in _tqdm_iter(tasks, "Rough bkg (estimate)")
        ]

    for i, rough_bkg in frame_results:
        if rough_bkg is not None:
            rough_bkg_stack[i] = rough_bkg

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"rough_bkg_r{round_id}")
        npz_path = f"{base}.npz"
        npy_path = f"{base}.npy"
        np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: rough_bkg_stack})
        np.save(npy_path, rough_bkg_stack)
        log.info(
            "  Rough background stack (round %s) saved to %s and %s",
            round_id,
            npz_path,
            npy_path,
        )

    return rough_bkg_stack


def load_background_stack(path: str) -> np.ndarray:
    """Load a background stack from ``.npz`` (``stack`` array) or ``.npy``."""
    if path.endswith(".npz"):
        z = np.load(path)
        if BACKGROUND_STACK_NPZ_ARRAY_KEY not in z.files:
            raise KeyError(
                f"{path!r} missing array {BACKGROUND_STACK_NPZ_ARRAY_KEY!r}; "
                f"have {list(z.files)}"
            )
        return np.asarray(z[BACKGROUND_STACK_NPZ_ARRAY_KEY])
    return np.load(path)


def save_background_stack(bkg: np.ndarray, path: str) -> None:
    """Save a background stack to ``.npz`` and ``.npy`` (same basename, no extension in ``path``)."""
    path = os.path.abspath(path)
    root, ext = os.path.splitext(path)
    if ext.lower() not in (".npy", ".npz"):
        root = path
    d = os.path.dirname(root)
    if d:
        os.makedirs(d, exist_ok=True)
    npz_path = f"{root}.npz"
    npy_path = f"{root}.npy"
    arr = np.asarray(bkg, dtype=np.float32)
    np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: arr})
    np.save(npy_path, arr)
    log.info("Background stack saved to %s and %s", npz_path, npy_path)
