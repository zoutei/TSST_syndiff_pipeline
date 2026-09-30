# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Build 6x6 gridded ePSF from pooled multi-frame star cutouts."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from joblib import delayed
from photutils.psf import EPSFBuilder, EPSFStars, extract_stars

from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.common.wcs_grouping import gaia_science_xy_for_frame
from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask
from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
    _filter_stars_geometric_mask,
    _filter_stars_off_mask,
    _section_bounds,
    _stars_in_section,
    _suppress_photutils_epsf_noise,
    save_gridded_epsf_npz,
)

from syndiff_pipeline.forward_model._vendor.ref_epsf_photometry.stamp_norm import (
    apply_border_crop,
    renormalize_epsf_stamp_after_crop,
)

log = logging.getLogger(__name__)


def _configure_blas_threads(n_workers: int) -> None:
    cpu_cap = os.cpu_count() or 1
    per_worker = max(1, cpu_cap // max(1, n_workers))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(per_worker)


@dataclass
class EpsfBuildParams:
    tile_nx: int = 6
    tile_ny: int = 6
    epsf_oversample: int = 4
    psf_size: int = 15
    min_stars_per_tile: int = 5
    mag_max_tess: float = 12.95
    mag_min_tess: float | None = None
    epsf_maxiters: int = 15
    epsf_recentering_maxiters: int = 20
    extract_size: int = 15
    epsf_smoothing_kernel: str = "quadratic"
    epsf_builder_fit_shape: int = 5
    epsf_recentering_boxsize: int = 3
    epsf_star_box_radius: int = 7
    epsf_use_section_mask: bool = True
    epsf_stamp_border_crop: int = 8
    n_jobs: int | None = None
    # photutils EPSFBuilder iteration bar (maxiters). Keep False for shared
    # ref_epsf paths; init_study enables True and redirects each tile to a log.
    progress_bar: bool = False


@dataclass
class _PreparedFrame:
    stem: str
    diff_img: np.ndarray
    gaia_frame: pd.DataFrame
    full_mask: np.ndarray | None


@dataclass(frozen=True)
class _TileResult:
    i: int
    j: int
    x_center: float
    y_center: float
    stamp: np.ndarray | None
    status: str
    n_cutouts: int


def prepare_gaia_for_pooled_epsf(
    gaia_df: pd.DataFrame,
    params: EpsfBuildParams,
) -> pd.DataFrame:
    """Brightness pre-filter using ``tess_mag`` (not Gaia RP)."""
    if "tess_mag" not in gaia_df.columns:
        raise ValueError("Gaia catalog for pooled ePSF requires tess_mag column")
    mag = pd.to_numeric(gaia_df["tess_mag"], errors="coerce")
    keep = pd.Series(True, index=gaia_df.index)
    if params.mag_max_tess is not None:
        keep &= mag < float(params.mag_max_tess)
    if params.mag_min_tess is not None:
        keep &= mag > float(params.mag_min_tess)
    out = gaia_df.loc[keep].copy().reset_index(drop=True)
    if "ra" not in out.columns or "dec" not in out.columns:
        raise ValueError("Gaia catalog for ePSF requires ra, dec columns")
    log.info(
        "pooled ePSF Gaia catalog: %d stars after %s < tess_mag < %s pre-filter",
        len(out),
        params.mag_min_tess,
        params.mag_max_tess,
    )
    return out


def _merge_epsf_stars(extracted_list: list) -> EPSFStars | None:
    merged: list = []
    for ex in extracted_list:
        if ex is None:
            continue
        for star in ex:
            merged.append(star)
    if not merged:
        return None
    return EPSFStars(merged)


def _build_stamp_from_extracted(
    extracted: EPSFStars | None,
    params: EpsfBuildParams,
) -> np.ndarray | None:
    if extracted is None or len(extracted) == 0:
        return None
    _suppress_photutils_epsf_noise()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        builder = EPSFBuilder(
            oversampling=int(params.epsf_oversample),
            maxiters=int(params.epsf_maxiters),
            recentering_maxiters=int(params.epsf_recentering_maxiters),
            smoothing_kernel=str(params.epsf_smoothing_kernel),
            fit_shape=int(params.epsf_builder_fit_shape),
            recentering_boxsize=int(params.epsf_recentering_boxsize),
            progress_bar=bool(params.progress_bar),
        )
        epsf, _ = builder(extracted)
    stamp = np.asarray(epsf.data, dtype=np.float64)
    if not np.all(np.isfinite(stamp)):
        return None
    return stamp


def _extract_tile_stars(
    section_data: np.ndarray,
    stars_tbl: Table,
    section_mask: np.ndarray | None,
    params: EpsfBuildParams,
):
    stars = stars_tbl.copy()
    if section_mask is not None and params.epsf_star_box_radius > 0:
        stars = _filter_stars_geometric_mask(
            stars, section_mask, int(params.epsf_star_box_radius)
        )
    if len(stars) == 0:
        return None
    mask = None
    if params.epsf_use_section_mask and section_mask is not None:
        mask = np.asarray(section_mask, dtype=bool)
    nddata = NDData(data=np.asarray(section_data, dtype=np.float64), mask=mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return extract_stars(nddata, stars, size=int(params.extract_size))


def _prepare_reference_frames(
    reference_stems: list[str],
    *,
    hp_d_by_stem: dict[str, Path],
    ffi_path_by_stem: dict[str, str],
    gaia_filtered: pd.DataFrame,
    ffi_list_df: pd.DataFrame,
    science_bounds: dict,
    mask_catalog,
    btjd_by_stem: dict[str, float],
) -> list[_PreparedFrame]:
    prepared: list[_PreparedFrame] = []
    for stem in reference_stems:
        hp_path = hp_d_by_stem.get(stem)
        ffi_path = ffi_path_by_stem.get(stem)
        if hp_path is None or not hp_path.is_file() or ffi_path is None:
            continue
        try:
            diff_img = fits.getdata(hp_path).astype(np.float64)
        except Exception as exc:
            log.warning("skip %s: %s", stem, exc)
            continue
        try:
            gaia_frame = gaia_science_xy_for_frame(
                gaia_filtered, ffi_path, ffi_list_df, science_bounds
            )
        except Exception as exc:
            log.warning("gaia xy failed %s: %s", stem, exc)
            continue
        full_mask = None
        if mask_catalog is not None:
            btjd = btjd_by_stem.get(stem)
            full_mask = np.asarray(
                epsf_reject_mask(mask_catalog.mask_at(btjd, which="full")),
                dtype=bool,
            )
        prepared.append(
            _PreparedFrame(
                stem=stem,
                diff_img=diff_img,
                gaia_frame=gaia_frame,
                full_mask=full_mask,
            )
        )
    return prepared


def _fit_pooled_tile(
    i: int,
    j: int,
    *,
    prepared_frames: list[_PreparedFrame],
    ny: int,
    nx: int,
    params: EpsfBuildParams,
) -> _TileResult:
    tile_ny = int(params.tile_ny)
    tile_nx = int(params.tile_nx)
    oversampling = int(params.epsf_oversample)
    min_stars = int(params.min_stars_per_tile)
    extract_size = int(params.extract_size)
    star_margin = float(extract_size) / 2.0 + 2.0
    border_crop = int(params.epsf_stamp_border_crop)

    x_min, x_max, y_min, y_max = _section_bounds(ny, nx, tile_ny, tile_nx, i, j)
    x_center = j * (nx / tile_nx) + (nx / (2 * tile_nx))
    y_center = i * (ny / tile_ny) + (ny / (2 * tile_ny))

    extracted_frames: list = []
    for frame in prepared_frames:
        sec_stars = _stars_in_section(
            frame.gaia_frame, x_min, x_max, y_min, y_max, margin=star_margin
        )
        if frame.full_mask is not None:
            sec_stars = _filter_stars_off_mask(
                sec_stars, frame.full_mask, ny=ny, nx=nx
            )
        if len(sec_stars) < min_stars:
            continue

        section = frame.diff_img[y_min:y_max, x_min:x_max]
        section_mask = None
        if frame.full_mask is not None:
            section_mask = np.asarray(
                frame.full_mask[y_min:y_max, x_min:x_max], dtype=bool
            )

        stars_tbl = Table()
        stars_tbl["x"] = np.asarray(sec_stars["x"].values - x_min, dtype=float)
        stars_tbl["y"] = np.asarray(sec_stars["y"].values - y_min, dtype=float)

        ex = _extract_tile_stars(section, stars_tbl, section_mask, params)
        if ex is not None and len(ex) > 0:
            extracted_frames.append(ex)

    merged = _merge_epsf_stars(extracted_frames)
    n_cutouts = len(merged) if merged is not None else 0
    if merged is None or n_cutouts < min_stars:
        return _TileResult(i, j, float(x_center), float(y_center), None, "too_few", n_cutouts)

    stamp = _build_stamp_from_extracted(merged, params)
    if stamp is None:
        return _TileResult(i, j, float(x_center), float(y_center), None, "fit_failed", n_cutouts)

    stamp = apply_border_crop(stamp, border_crop)
    stamp = renormalize_epsf_stamp_after_crop(stamp, oversampling)
    return _TileResult(i, j, float(x_center), float(y_center), stamp, "ok", n_cutouts)


def build_pooled_gridded_epsf(
    reference_stems: list[str],
    *,
    hp_d_by_stem: dict[str, Path],
    ffi_path_by_stem: dict[str, str],
    gaia_base: pd.DataFrame,
    ffi_list_df: pd.DataFrame,
    science_bounds: dict,
    mask_catalog,
    btjd_by_stem: dict[str, float],
    params: EpsfBuildParams,
) -> tuple[np.ndarray, list[tuple[float, float]], dict[str, Any]]:
    """
    Pool star cutouts across reference frames; return (stack, grid_xypos, stats).

    Parallelizes over grid tiles (thread pool). Each reference FFI is loaded and
    Gaia-projected once per window, not once per tile.
    """
    if not reference_stems:
        raise ValueError("empty reference_stems")

    first_stem = reference_stems[0]
    first_path = hp_d_by_stem.get(first_stem)
    if first_path is None or not first_path.is_file():
        raise FileNotFoundError(f"missing hp_d for {first_stem}")
    ny, nx = fits.getdata(first_path).shape

    tile_ny = int(params.tile_ny)
    tile_nx = int(params.tile_nx)
    n_tiles = tile_nx * tile_ny
    n_workers = resolve_effective_n_jobs(
        n_tiles,
        stage_n_jobs=params.n_jobs,
    )
    _configure_blas_threads(n_workers)

    gaia_filtered = prepare_gaia_for_pooled_epsf(gaia_base, params)
    prepared_frames = _prepare_reference_frames(
        reference_stems,
        hp_d_by_stem=hp_d_by_stem,
        ffi_path_by_stem=ffi_path_by_stem,
        gaia_filtered=gaia_filtered,
        ffi_list_df=ffi_list_df,
        science_bounds=science_bounds,
        mask_catalog=mask_catalog,
        btjd_by_stem=btjd_by_stem,
    )
    if not prepared_frames:
        raise RuntimeError("no reference frames could be prepared")

    log.info(
        "pooled ePSF: %d/%d reference frames, %d tiles, n_jobs=%d",
        len(prepared_frames),
        len(reference_stems),
        n_tiles,
        n_workers,
    )

    tile_tasks = [(i, j) for i in range(tile_ny) for j in range(tile_nx)]
    delayed_calls = [
        delayed(_fit_pooled_tile)(
            i,
            j,
            prepared_frames=prepared_frames,
            ny=ny,
            nx=nx,
            params=params,
        )
        for i, j in tile_tasks
    ]
    tile_results: list[_TileResult] = parallel_map_with_optional_tqdm(
        delayed_calls,
        n_tasks=len(tile_tasks),
        desc="pooled ePSF tiles",
        n_jobs_eff=n_workers,
        prefer="threads",
    )

    epsf_grid: dict[tuple[int, int], np.ndarray | str] = {}
    tile_cutout_counts: dict[tuple[int, int], int] = {}
    grid_xypos: list[tuple[float, float]] = []
    for result in sorted(tile_results, key=lambda r: (r.i, r.j)):
        grid_xypos.append((result.x_center, result.y_center))
        epsf_grid[(result.i, result.j)] = (
            result.stamp if result.stamp is not None else result.status
        )
        tile_cutout_counts[(result.i, result.j)] = result.n_cutouts

    valid = [v for v in epsf_grid.values() if isinstance(v, np.ndarray)]
    if not valid:
        raise RuntimeError("all grid tiles failed")

    fallback = np.mean(valid, axis=0)
    psf_list: list[np.ndarray] = []
    n_ok = 0
    for i in range(tile_ny):
        for j in range(tile_nx):
            result = epsf_grid.get((i, j), "too_few")
            if isinstance(result, np.ndarray):
                psf_list.append(result)
                n_ok += 1
            else:
                psf_list.append(fallback)

    stack = np.array(psf_list, dtype=np.float64)
    stats = {
        "n_tiles_ok": n_ok,
        "n_tiles_total": tile_nx * tile_ny,
        "tile_cutout_counts": {
            f"{i}_{j}": tile_cutout_counts.get((i, j), 0)
            for i in range(tile_ny)
            for j in range(tile_nx)
        },
        "n_reference_stems": len(reference_stems),
        "n_prepared_frames": len(prepared_frames),
        "n_jobs": n_workers,
    }
    return stack, grid_xypos, stats


def save_window_npz(
    path: Path,
    stack: np.ndarray,
    grid_xypos: list[tuple[float, float]],
    params: EpsfBuildParams,
    *,
    orbit_idx: int,
    window: str,
    reference_stems: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    save_gridded_epsf_npz(
        str(path),
        stack,
        grid_xypos,
        int(params.epsf_oversample),
    )
    # Attach metadata via numpy save append — rewrite with extra arrays
    z = np.load(path, allow_pickle=True)
    meta = {
        "orbit_idx": int(orbit_idx),
        "window": str(window),
        "tile_nx": int(params.tile_nx),
        "tile_ny": int(params.tile_ny),
        "border_crop": int(params.epsf_stamp_border_crop),
        "renormalized": True,
        "reference_stems": np.asarray(reference_stems, dtype=object),
    }
    if extra_meta:
        meta.update(extra_meta)
    np.savez_compressed(
        path,
        data=np.asarray(z["data"], dtype=np.float64),
        grid_xypos=np.asarray(z["grid_xypos"], dtype=np.float64),
        oversampling=int(np.asarray(z["oversampling"])),
        meta_json=np.asarray([meta], dtype=object),
    )
    z.close()
    return path
