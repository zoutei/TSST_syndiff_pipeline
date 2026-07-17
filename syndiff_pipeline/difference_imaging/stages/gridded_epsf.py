"""
Per-frame gridded empirical PSF fitting with photutils.

Builds a :class:`~photutils.psf.GriddedPSFModel` per difference image from
Gaia stars on a tile_ny × tile_nx grid (see starpositioningscript reference).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from joblib import delayed
from photutils.psf import EPSFBuilder, GriddedPSFModel, extract_stars

from syndiff_pipeline.common.joblib_progress import (
    parallel_map_with_optional_tqdm,
    tqdm_iter,
)
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    parse_workspace_frame_stem,
    strip_fits_suffix,
    tess_product_id_from_ffi_path,
)

log = logging.getLogger(__name__)

# photutils logs per-star exclusion at WARNING on crowded fields; not actionable.
_PHOTUTILS_EPSF_LOGGER = "photutils.psf.epsf_builder"


def _suppress_photutils_epsf_noise() -> None:
    """Silence photutils per-star ePSF exclusion chatter (uses logging, not warnings)."""
    logging.getLogger(_PHOTUTILS_EPSF_LOGGER).setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*has been excluded from ePSF fitting.*",
    )


_suppress_photutils_epsf_noise()

GRIDDED_EPSF_INDEX_BASENAME = "gridded_epsf_index.json"
GRIDDED_EPSF_NPZ_SUFFIX = "_gridded_epsf.npz"

# Per-process state for loky workers (initialized once per worker, not per frame).
_WORKER_CTX: dict[str, Any] = {}


def _init_gridded_epsf_worker(
    gaia_df: pd.DataFrame,
    epsf_params,
    output_dir: str,
    mask_2d: np.ndarray | None,
    skip_existing: bool = True,
    mask_catalog=None,
    btjd_by_stem: dict | None = None,
) -> None:
    """Load shared ePSF inputs once per loky worker (see starpositioningscript)."""
    _suppress_photutils_epsf_noise()
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "gaia_df": gaia_df,
            "epsf_params": epsf_params,
            "output_dir": output_dir,
            "mask_2d": mask_2d,
            "skip_existing": bool(skip_existing),
            "mask_catalog": mask_catalog,
            "btjd_by_stem": btjd_by_stem or {},
        }
    )


def _resolve_epsf_frame_mask(ctx: dict, ffi_stem: str) -> np.ndarray | None:
    """Per-FFI ePSF reject mask from MaskCatalog (preferred) or static fallback."""
    catalog = ctx.get("mask_catalog")
    if catalog is not None:
        from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask

        btjd = ctx.get("btjd_by_stem", {}).get(ffi_stem)
        return epsf_reject_mask(catalog.mask_at(btjd, which="full"))
    return ctx.get("mask_2d")


def _configure_blas_threads(n_workers: int) -> None:
    """Match starpositioningscript: let BLAS use cores not claimed by other workers."""
    cpu_cap = os.cpu_count() or 1
    per_worker = max(1, cpu_cap // max(1, n_workers))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(per_worker)


def gridded_epsf_npz_path(output_dir: str, ffi_stem: str) -> str:
    """Path for one frame's gridded PSF archive."""
    return os.path.join(output_dir, f"{ffi_stem}{GRIDDED_EPSF_NPZ_SUFFIX}")


def _filter_gaia_for_epsf(
    gaia_df: pd.DataFrame,
    *,
    mag_max_rp: float | None,
) -> pd.DataFrame:
    """
    Brightness pre-filter for ePSF star catalogs.

    Mirrors ``starpositioningscript.py``::

        combined_filter = (df['phot_rp_mean_mag'] < 12.95) & in_crop

    Expects crop-local ``x``/``y`` (from :func:`ensure_gaia_crop_xy`).
    """
    df = gaia_df
    if mag_max_rp is not None and "phot_rp_mean_mag" in df.columns:
        rp = pd.to_numeric(df["phot_rp_mean_mag"], errors="coerce")
        df = df.loc[rp < float(mag_max_rp)].copy()
    return df.reset_index(drop=True)


def _resolve_mag_max_rp(epsf_params) -> float:
    """
    Brightness cut for ePSF star selection.

    ``starpositioningscript.py`` always uses ``phot_rp_mean_mag < 12.95``.
    Legacy frozen configs wrote ``mag_max_rp: null`` for experiment B (no narrow
    mag window); treat that as the reference default, not "use all Gaia".
    """
    mag_max = getattr(epsf_params, "mag_max_rp", 12.95)
    if mag_max is None:
        return 12.95
    return float(mag_max)


def prepare_gaia_for_gridded_epsf(
    gaia_df: pd.DataFrame,
    epsf_params,
) -> pd.DataFrame:
    """
    One-time Gaia table for the frame-parallel ePSF loop.

    Call once in the parent process before ``Parallel`` (see
    ``starpositioningscript.py`` main block). Per-frame workers receive this
    pre-filtered table; section loops only apply spatial cuts.
    """
    mag_max = _resolve_mag_max_rp(epsf_params)
    out = _filter_gaia_for_epsf(gaia_df, mag_max_rp=mag_max)
    if "x" not in out.columns or "y" not in out.columns:
        raise ValueError("Gaia catalog for ePSF requires crop-local columns x, y")
    n = len(out)
    log.info(
        "ePSF Gaia catalog: %d stars after phot_rp_mean_mag < %s pre-filter",
        n,
        mag_max,
    )
    return out


def _section_bounds(
    ny: int,
    nx: int,
    tile_ny: int,
    tile_nx: int,
    i: int,
    j: int,
) -> tuple[int, int, int, int]:
    """Section pixel bounds matching starpositioningscript grid layout."""
    step_x = nx // tile_nx
    step_y = ny // tile_ny
    x_min = j * step_x
    x_max = (j + 1) * step_x
    y_min = i * step_y
    y_max = (i + 1) * step_y
    return x_min, x_max, y_min, y_max


def _stars_in_section(
    gaia_df: pd.DataFrame,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    margin: float = 0.0,
) -> pd.DataFrame:
    """
    Gaia rows in a grid section.

    Base cut matches ``starpositioningscript.py`` (``>= x_min``, ``< x_max``).
    Optional *margin* excludes edge stars (experiment B: ``extract_size//2 + 2``).
    """
    if margin <= 0.0:
        in_sec = (
            (gaia_df["x"] >= x_min)
            & (gaia_df["x"] < x_max)
            & (gaia_df["y"] >= y_min)
            & (gaia_df["y"] < y_max)
        )
        return gaia_df.loc[in_sec].copy()

    x0 = x_min + margin
    x1 = x_max - margin
    y0 = y_min + margin
    y1 = y_max - margin
    if x1 <= x0 or y1 <= y0:
        return gaia_df.iloc[0:0].copy()
    in_sec = (
        (gaia_df["x"] >= x0)
        & (gaia_df["x"] < x1)
        & (gaia_df["y"] >= y0)
        & (gaia_df["y"] < y1)
    )
    return gaia_df.loc[in_sec].copy()


def _filter_stars_off_mask(
    stars_df: pd.DataFrame,
    mask_2d: np.ndarray,
    *,
    ny: int,
    nx: int,
) -> pd.DataFrame:
    """Drop Gaia rows whose integer pixel positions fall on masked pixels."""
    if len(stars_df) == 0:
        return stars_df
    xi = np.clip(np.round(stars_df["x"].values).astype(int), 0, nx - 1)
    yi = np.clip(np.round(stars_df["y"].values).astype(int), 0, ny - 1)
    good = ~mask_2d[yi, xi]
    return stars_df.loc[good].copy()


def fit_epsf_section(
    section_data: np.ndarray,
    stars_tbl: Table,
    *,
    extract_size: int,
    oversampling: int,
    maxiters: int,
    recentering_maxiters: int = 20,
    section_mask: np.ndarray | None = None,
    use_mask: bool = False,
) -> np.ndarray | None:
    """
    Fit one grid-section ePSF stamp with photutils.

    Returns oversampled 2D stamp array or None on failure.
    """
    if len(stars_tbl) == 0:
        return None
    data = np.asarray(section_data, dtype=np.float64)
    mask = None
    if use_mask and section_mask is not None:
        mask = np.asarray(section_mask, dtype=bool)
        if mask.shape != data.shape:
            mask = None
    nddata = NDData(data=data, mask=mask)
    try:
        _suppress_photutils_epsf_noise()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extracted = extract_stars(nddata, stars_tbl, size=int(extract_size))
            if extracted is None or len(extracted) == 0:
                return None
            builder = EPSFBuilder(
                oversampling=int(oversampling),
                maxiters=int(maxiters),
                recentering_maxiters=int(recentering_maxiters),
                progress_bar=False,
            )
            epsf, _fitted = builder(extracted)
        stamp = np.asarray(epsf.data, dtype=np.float64)
        if not np.all(np.isfinite(stamp)):
            return None
        return stamp
    except Exception as exc:
        log.debug("fit_epsf_section failed: %s", exc)
        return None


def _tile_centers_from_shape(
    ny: int, nx: int, tile_ny: int, tile_nx: int
) -> list[tuple[float, float]]:
    """Grid-node centers in crop-local pixels (matches section layout)."""
    centers: list[tuple[float, float]] = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            x_center = j * (nx / tile_nx) + (nx / (2 * tile_nx))
            y_center = i * (ny / tile_ny) + (ny / (2 * tile_ny))
            centers.append((float(x_center), float(y_center)))
    return centers


def build_gridded_psf_for_frame(
    diff_image: np.ndarray,
    gaia_df: pd.DataFrame,
    epsf_params,
    *,
    mask_2d: np.ndarray | None = None,
    frame_label: str = "",
) -> tuple[GriddedPSFModel | None, list[tuple[float, float]], np.ndarray | None]:
    """
    Build a spatially varying PSF model for one difference image.

    Algorithm follows ``starpositioningscript.py`` ``processing()``:
    section grid → per-section ``extract_stars`` + ``EPSFBuilder`` → mean
    fallback for failed sections → ``GriddedPSFModel``.

    *gaia_df* must already be prepared via :func:`prepare_gaia_for_gridded_epsf`.
    """
    ny, nx = diff_image.shape
    tile_ny = int(epsf_params.tile_ny)
    tile_nx = int(epsf_params.tile_nx)
    oversampling = int(epsf_params.epsf_oversample)
    min_stars = int(getattr(epsf_params, "min_stars_per_tile", 5))
    maxiters = int(getattr(epsf_params, "epsf_maxiters", 15))
    recentering_maxiters = int(getattr(epsf_params, "epsf_recentering_maxiters", 20))
    extract_size = int(
        getattr(epsf_params, "extract_size", None) or epsf_params.psf_size
    )
    # experiment B section edge margin: extract_size // 2 + 2
    star_margin = float(extract_size) / 2.0 + 2.0

    step_x = nx // tile_nx
    step_y = ny // tile_ny
    if step_x < extract_size or step_y < extract_size:
        log.warning(
            "ePSF grid cells smaller than extract_size=%s for %s",
            extract_size,
            frame_label or "frame",
        )

    epsf_grid: dict[tuple[int, int], np.ndarray | str] = {}
    grid_xypos: list[tuple[float, float]] = []

    for i in range(tile_ny):
        for j in range(tile_nx):
            x_min, x_max, y_min, y_max = _section_bounds(ny, nx, tile_ny, tile_nx, i, j)

            x_center = j * (nx / tile_nx) + (nx / (2 * tile_nx))
            y_center = i * (ny / tile_ny) + (ny / (2 * tile_ny))
            grid_xypos.append((float(x_center), float(y_center)))

            section = diff_image[y_min:y_max, x_min:x_max]
            sec_stars = _stars_in_section(
                gaia_df, x_min, x_max, y_min, y_max, margin=star_margin
            )
            if mask_2d is not None:
                sec_stars = _filter_stars_off_mask(sec_stars, mask_2d, ny=ny, nx=nx)
            if len(sec_stars) < min_stars:
                epsf_grid[(i, j)] = "too_few"
                continue

            stars_tbl = Table()
            stars_tbl["x"] = np.asarray(sec_stars["x"].values - x_min, dtype=float)
            stars_tbl["y"] = np.asarray(sec_stars["y"].values - y_min, dtype=float)

            stamp = fit_epsf_section(
                np.asarray(section, dtype=np.float64),
                stars_tbl,
                extract_size=extract_size,
                oversampling=oversampling,
                maxiters=maxiters,
                recentering_maxiters=recentering_maxiters,
                use_mask=False,
            )
            if stamp is None:
                epsf_grid[(i, j)] = "fit_failed"
            else:
                epsf_grid[(i, j)] = stamp

    valid = [v for v in epsf_grid.values() if isinstance(v, np.ndarray)]
    if not valid:
        suffix = f" ({frame_label})" if frame_label else ""
        log.warning("ePSF: all grid sections failed%s", suffix)
        return None, grid_xypos, None

    fallback = np.mean(valid, axis=0)
    psf_list: list[np.ndarray] = []
    for i in range(tile_ny):
        for j in range(tile_nx):
            result = epsf_grid.get((i, j), "too_few")
            if isinstance(result, np.ndarray):
                psf_list.append(result)
            else:
                psf_list.append(fallback)

    stack = np.array(psf_list, dtype=np.float64)
    meta = {"grid_xypos": grid_xypos, "oversampling": oversampling}
    nddata_grid = NDData(data=stack, meta=meta)
    model = GriddedPSFModel(nddata_grid)
    return model, grid_xypos, stack


def save_gridded_epsf_npz(
    path: str,
    stack: np.ndarray,
    grid_xypos: list[tuple[float, float]],
    oversampling: int,
) -> str:
    """Write one frame's gridded PSF cube."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    xy = np.asarray(grid_xypos, dtype=np.float64)
    np.savez_compressed(
        path,
        data=np.asarray(stack, dtype=np.float64),
        grid_xypos=xy,
        oversampling=int(oversampling),
    )
    return path


def load_gridded_psf_model(path: str) -> GriddedPSFModel:
    """Load a per-frame :class:`GriddedPSFModel` from npz."""
    z = np.load(path, allow_pickle=False)
    try:
        stack = np.asarray(z["data"], dtype=np.float64)
        grid_xypos = [tuple(row) for row in np.asarray(z["grid_xypos"])]
        oversampling = int(z["oversampling"])
    finally:
        z.close()
    meta = {"grid_xypos": grid_xypos, "oversampling": oversampling}
    return GriddedPSFModel(NDData(data=stack, meta=meta))


def save_gridded_epsf_index(output_dir: str, index: dict[str, str]) -> str:
    """Persist ffi_stem → npz path mapping."""
    path = os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    return path


def register_gridded_epsf_index_entry(output_dir: str, ffi_stem: str) -> None:
    """Append one frame to ``gridded_epsf_index.json`` (worker-safe, file-locked)."""
    import fcntl

    index_path = os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME)
    os.makedirs(output_dir, exist_ok=True)
    npz_path = gridded_epsf_npz_path(output_dir, ffi_stem)
    with open(index_path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read()
            if raw.strip():
                index = json.loads(raw)
            else:
                index = {}
            index[str(ffi_stem)] = npz_path
            fh.seek(0)
            fh.truncate(0)
            json.dump(index, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_gridded_epsf_index(output_dir: str) -> dict[str, str]:
    """Load ffi_stem → npz path mapping; empty dict if missing."""
    path = os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k): str(v) for k, v in raw.items()}


def workspace_has_gridded_epsf(output_dir: str) -> bool:
    """True when the workspace contains a gridded ePSF index."""
    return os.path.isfile(os.path.join(output_dir, GRIDDED_EPSF_INDEX_BASENAME))


@dataclass(frozen=True)
class GriddedEpsfCatalog:
    """Per-frame gridded PSF lookup for forced photometry."""

    workspace_dir: str
    index: dict[str, str]

    def path_for_stem(self, ffi_stem: str) -> str | None:
        """Return npz path for a tess product stem, if present."""
        p = self.index.get(ffi_stem)
        if p and os.path.isfile(p):
            return p
        alt = gridded_epsf_npz_path(self.workspace_dir, ffi_stem)
        if os.path.isfile(alt):
            return alt
        return None

    def load_model(self, ffi_stem: str) -> GriddedPSFModel | None:
        """Load gridded model for one frame."""
        path = self.path_for_stem(ffi_stem)
        if path is None:
            return None
        return load_gridded_psf_model(path)


def catalog_from_workspace(output_dir: str) -> GriddedEpsfCatalog | None:
    """Build catalog from workspace dir; None if no gridded index."""
    if not workspace_has_gridded_epsf(output_dir):
        return None
    return GriddedEpsfCatalog(
        workspace_dir=output_dir,
        index=load_gridded_epsf_index(output_dir),
    )


def stack_from_gridded_cube(stack: np.ndarray) -> np.ndarray:
    """Flatten (n_grid, ny, nx) cube to (n_grid, ny*nx) for legacy tile stacks."""
    n_grid = stack.shape[0]
    return stack.reshape(n_grid, -1)


def _diff_path_to_stem(diff_path: str) -> str:
    stem = strip_fits_suffix(Path(str(diff_path)).name)
    parsed = parse_workspace_frame_stem(stem)
    if parsed is not None:
        return parsed[0]
    return tess_product_id_from_ffi_path(stem) or stem


def _is_valid_gridded_epsf_npz(path: str) -> bool:
    """True when *path* is a readable gridded ePSF per-frame archive."""
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(k in z.files for k in ("data", "grid_xypos", "oversampling"))
    except (OSError, ValueError):
        return False


def _load_gridded_epsf_stack(path: str) -> np.ndarray | None:
    """Load the PSF cube from a per-frame npz, or ``None`` if unreadable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return np.asarray(z["data"], dtype=np.float64)
    except (OSError, ValueError):
        return None


def _fit_one_frame_task(
    frame_idx: int,
    diff_path: str,
) -> tuple[int, str, bool, list[tuple[float, float]] | None, np.ndarray | None, bool]:
    """Worker: fit one frame and write npz.

    Shared inputs come from :func:`_init_gridded_epsf_worker` (not pickled per frame).

    Returns
    -------
    frame_idx, ffi_stem, ok, grid_xypos, stack, skipped_existing
    """
    ctx = _WORKER_CTX
    gaia_df = ctx["gaia_df"]
    epsf_params = ctx["epsf_params"]
    output_dir = ctx["output_dir"]

    ffi_stem = _diff_path_to_stem(diff_path) if diff_path else f"frame_{frame_idx}"
    mask_2d = _resolve_epsf_frame_mask(ctx, ffi_stem)
    out_path = gridded_epsf_npz_path(output_dir, ffi_stem)
    if ctx.get("skip_existing", True) and _is_valid_gridded_epsf_npz(out_path):
        return frame_idx, ffi_stem, True, None, None, True
    if diff_path is None or not os.path.exists(diff_path):
        log.warning("  diff frame missing: %s", diff_path)
        return frame_idx, ffi_stem, False, None, None, False
    try:
        diff_img = fits.getdata(diff_path).astype(np.float64)
    except Exception as exc:
        log.warning("  Cannot load %s: %s", diff_path, exc)
        return frame_idx, ffi_stem, False, None, None, False

    model, grid_xypos, stack = build_gridded_psf_for_frame(
        diff_img,
        gaia_df,
        epsf_params,
        mask_2d=mask_2d,
        frame_label=os.path.basename(diff_path),
    )
    if model is None or stack is None:
        return frame_idx, ffi_stem, False, grid_xypos, None, False

    save_gridded_epsf_npz(
        out_path,
        stack,
        grid_xypos,
        int(epsf_params.epsf_oversample),
    )
    return frame_idx, ffi_stem, True, grid_xypos, stack, False


def fit_gridded_epsf_all_frames(
    diff_paths: list[str],
    gaia_df: pd.DataFrame,
    cfg,
    epsf_params,
    output_dir: str,
    *,
    mask_2d: np.ndarray | None = None,
    mask_catalog=None,
    btjd_by_stem: dict | None = None,
    round_id: int = 1,
    diff_log_path: str | None = None,
    epsf_label: str | None = None,
    diffs_input: str | None = None,
    skip_existing: bool = True,
) -> tuple[np.ndarray, list[tuple[float, float]], list[str], list[bool]]:
    """
    Fit gridded ePSF on every difference image (thread-parallel over frames).

    When ``mask_catalog`` is provided, each frame resolves an ePSF reject mask
    via in-memory ``mask_at(btjd)`` (no static-mask FITS I/O per frame).
    ``mask_2d`` remains a static fallback when no catalog is available.

    Returns
    -------
    epsf_stack : ndarray (n_frames, n_tiles, n_pix) — legacy flat tile layout
    tile_centers : list of (cx, cy)
    ffi_stems : list of str
    epsf_ok : list of bool
    """
    n_frames = len(diff_paths)
    epsf_n_jobs = getattr(epsf_params, "epsf_n_jobs", None)
    n_workers = resolve_effective_n_jobs(
        int(getattr(cfg, "n_jobs", 1) or 1),
        stage_n_jobs=epsf_n_jobs,
    )
    _configure_blas_threads(n_workers)

    from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
        init_progress_pair,
        progress_path_for_diff_log,
        progress_path_for_output_workspace,
        refresh_progress_pair_from_artifacts,
        set_progress_phase_pair,
    )

    gaia_df = prepare_gaia_for_gridded_epsf(gaia_df, epsf_params)
    os.makedirs(output_dir, exist_ok=True)

    track_progress = epsf_label is not None
    cli_progress_path = (
        str(progress_path_for_diff_log(diff_log_path))
        if track_progress and diff_log_path is not None
        else None
    )
    workspace_progress_path: str | None = None
    if track_progress:
        workspace_progress_path = str(progress_path_for_output_workspace(output_dir))
        init_progress_pair(
            workspace_progress_path,
            cli_progress_path,
            epsf_label=str(epsf_label),
            diffs_input=str(diffs_input or "?"),
            round_id=round_id,
            frames_total=n_frames,
            output_dir=output_dir,
        )
        refresh_progress_pair_from_artifacts(
            workspace_progress_path, cli_progress_path
        )

    tasks = [(i, p) for i, p in enumerate(diff_paths)]

    tqdm_desc = f"epsf {epsf_label}" if track_progress else "ePSF gridded"

    def _on_frame_done(result: tuple) -> None:
        if not track_progress or workspace_progress_path is None:
            return
        if result[5]:
            return
        from syndiff_pipeline.difference_imaging.stages.epsf_progress import (
            record_frame_progress,
        )

        _ok = bool(result[2])
        record_frame_progress(
            workspace_progress_path, cli_progress_path, success=_ok
        )

    worker_initargs = (
        gaia_df,
        epsf_params,
        output_dir,
        mask_2d,
        skip_existing,
        mask_catalog,
        btjd_by_stem or {},
    )
    results: list[tuple] = []
    if n_workers <= 1 or n_frames <= 1:
        _init_gridded_epsf_worker(*worker_initargs)
        log.info(
            "ePSF [%s] round %s: starting %d frames (n_jobs=1)",
            epsf_label or "?",
            round_id,
            n_frames,
        )
        for t in tqdm_iter(tasks, desc=tqdm_desc):
            result = _fit_one_frame_task(*t)
            _on_frame_done(result)
            results.append(result)
    else:
        log.info(
            "ePSF [%s] round %s: starting %d frames (n_jobs=%s, backend=loky)",
            epsf_label or "?",
            round_id,
            n_frames,
            n_workers,
        )
        delayed_calls = [delayed(_fit_one_frame_task)(i, p) for i, p in tasks]
        # Parent updates progress when each result is yielded (no worker NFS locks).
        results = parallel_map_with_optional_tqdm(
            delayed_calls,
            n_tasks=n_frames,
            desc=tqdm_desc,
            n_jobs_eff=n_workers,
            initializer=_init_gridded_epsf_worker,
            initargs=worker_initargs,
            on_result=_on_frame_done,
        )

    if track_progress and workspace_progress_path is not None:
        refresh_progress_pair_from_artifacts(
            workspace_progress_path, cli_progress_path
        )
        set_progress_phase_pair(workspace_progress_path, cli_progress_path, "complete")

    results.sort(key=lambda r: r[0])
    ffi_stems = [r[1] for r in results]
    epsf_ok = [r[2] for r in results]
    tile_centers: list[tuple[float, float]] | None = None
    index: dict[str, str] = {}

    stacks: list[np.ndarray | None] = []
    for _idx, stem, ok, centers, stack, _skipped in results:
        if ok:
            path = gridded_epsf_npz_path(output_dir, stem)
            index[stem] = path
            if stack is None:
                stack = _load_gridded_epsf_stack(path)
        if ok and stack is not None:
            stacks.append(stack)
            if tile_centers is None and centers is not None:
                tile_centers = centers
        else:
            stacks.append(None)

    if tile_centers is None:
        first_path = next((p for p in diff_paths if p and os.path.exists(p)), None)
        if mask_2d is not None:
            ny, nx = mask_2d.shape
        elif first_path is not None:
            ny, nx = fits.getdata(first_path).shape
        else:
            ny, nx = 1024, 1024
        tile_centers = _tile_centers_from_shape(
            ny, nx, int(epsf_params.tile_ny), int(epsf_params.tile_nx)
        )

    save_gridded_epsf_index(output_dir, index)

    n_tiles = epsf_params.tile_ny * epsf_params.tile_nx
    n_pix = 0
    for s in stacks:
        if s is not None:
            n_pix = s.reshape(s.shape[0], -1).shape[1]
            break
    if n_pix == 0:
        n_pix = (2 * epsf_params.psf_size + 1) ** 2

    epsf_stack = np.full((n_frames, n_tiles, n_pix), np.nan)
    good_rows = [stack_from_gridded_cube(s) for s in stacks if s is not None]
    med_row = np.nanmedian(np.stack(good_rows), axis=0) if good_rows else None

    for i, stack in enumerate(stacks):
        if stack is not None:
            epsf_stack[i] = stack_from_gridded_cube(stack)
        elif med_row is not None:
            epsf_stack[i] = med_row

    n_ok = sum(epsf_ok)
    n_skipped = sum(1 for r in results if r[5])
    if epsf_label:
        if n_skipped:
            log.info(
                "ePSF [%s] round %s: %d/%d frames succeeded (%d skipped existing)",
                epsf_label,
                round_id,
                n_ok,
                n_frames,
                n_skipped,
            )
        else:
            log.info(
                "ePSF [%s] round %s: %d/%d frames succeeded",
                epsf_label,
                round_id,
                n_ok,
                n_frames,
            )
    elif n_skipped:
        log.info(
            "  ePSF round %s: %d/%d frames succeeded (%d skipped existing)",
            round_id,
            n_ok,
            n_frames,
            n_skipped,
        )
    else:
        log.info("  ePSF round %s: %d/%d frames succeeded", round_id, n_ok, n_frames)
    return epsf_stack, tile_centers, ffi_stems, epsf_ok


def compute_group_epsf_gridded(
    output_dir: str,
    group_ids: np.ndarray,
    ffi_stems: list[str],
    epsf_ok: list[bool],
    *,
    group_subdir: str = "group_epsf",
) -> dict[int, np.ndarray]:
    """
    Median gridded PSF cube per template group across frames.

    Saves ``group_epsf_{gid}.npz`` with keys data, grid_xypos, oversampling.
    """
    index = load_gridded_epsf_index(output_dir)
    group_ids = np.asarray(group_ids)
    unique_groups = [g for g in sorted(set(group_ids.tolist())) if g >= 0]
    group_epsf: dict[int, np.ndarray] = {}

    out_subdir = os.path.join(output_dir, group_subdir)
    os.makedirs(out_subdir, exist_ok=True)

    for gid in unique_groups:
        cubes: list[np.ndarray] = []
        grid_xypos = None
        oversampling = 2
        for stem, frame_gid, ok in zip(ffi_stems, group_ids, epsf_ok):
            if not ok or int(frame_gid) != int(gid):
                continue
            path = index.get(stem) or gridded_epsf_npz_path(output_dir, stem)
            if not os.path.isfile(path):
                continue
            z = np.load(path, allow_pickle=False)
            try:
                cubes.append(np.asarray(z["data"], dtype=np.float64))
                if grid_xypos is None:
                    grid_xypos = np.asarray(z["grid_xypos"])
                    oversampling = int(z["oversampling"])
            finally:
                z.close()
        if not cubes:
            continue
        med = np.nanmedian(np.stack(cubes, axis=0), axis=0)
        group_epsf[gid] = med
        out_path = os.path.join(out_subdir, f"group_epsf_{gid}.npz")
        np.savez_compressed(
            out_path,
            data=med,
            grid_xypos=grid_xypos,
            oversampling=oversampling,
        )
        log.info("  Group %s: %d frames → gridded ePSF cube %s", gid, len(cubes), med.shape)

    return group_epsf
