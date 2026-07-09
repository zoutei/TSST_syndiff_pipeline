"""
Gaia-star PSF photometry on difference images using per-frame gridded ePSF models.

Mirrors ``starpositioningscript.py``: filter Gaia to ``phot_rp_mean_mag < 12.95``,
run ``PSFPhotometry`` on all selected stars per FFI, and save ``*_photresults.ecsv``.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from joblib import delayed

from syndiff_pipeline.common.joblib_progress import (
    parallel_map_with_optional_tqdm,
    tqdm_iter,
)
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
    GriddedEpsfCatalog,
    _diff_path_to_stem,
    prepare_gaia_for_gridded_epsf,
)

log = logging.getLogger(__name__)

PHOTRESULTS_ECSV_SUFFIX = "_photresults.ecsv"
CENTROIDS_INDEX_BASENAME = "centroids_index.json"

_GAIA_META_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "phot_rp_mean_mag",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "tess_mag",
    "tess_flux",
    "x",
    "y",
)

_WORKER_CTX: dict[str, Any] = {}


def _configure_blas_threads(n_workers: int) -> None:
    """Cap BLAS threads per worker when running frame-parallel photometry."""
    cpu_cap = os.cpu_count() or 1
    per_worker = max(1, cpu_cap // max(1, n_workers))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(per_worker)


def photresults_ecsv_path(output_dir: str, ffi_stem: str) -> str:
    """Path for one frame's centroid / PSF photometry table."""
    return os.path.join(output_dir, f"{ffi_stem}{PHOTRESULTS_ECSV_SUFFIX}")


def save_centroids_index(output_dir: str, index: dict[str, str]) -> str:
    """Persist ffi_stem → photresults.ecsv path mapping."""
    path = os.path.join(output_dir, CENTROIDS_INDEX_BASENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    return path


def load_centroids_index(output_dir: str) -> dict[str, str]:
    """Load ffi_stem → photresults.ecsv path mapping; empty dict if missing."""
    path = os.path.join(output_dir, CENTROIDS_INDEX_BASENAME)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k): str(v) for k, v in raw.items()}


def workspace_has_centroids(output_dir: str) -> bool:
    """True when the workspace contains centroid outputs."""
    if os.path.isfile(os.path.join(output_dir, CENTROIDS_INDEX_BASENAME)):
        return True
    root = Path(output_dir)
    return any(root.glob(f"*{PHOTRESULTS_ECSV_SUFFIX}"))


def _init_centroids_worker(
    gaia_df: pd.DataFrame,
    epsf_catalog: GriddedEpsfCatalog,
    params,
    output_dir: str,
) -> None:
    """Load shared centroid inputs once per loky worker."""
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "gaia_df": gaia_df,
            "epsf_catalog": epsf_catalog,
            "params": params,
            "output_dir": output_dir,
        }
    )


def _build_init_params(gaia_df: pd.DataFrame) -> Table:
    """Build photutils init_params with Gaia metadata carried through."""
    init_params = Table()
    init_params["x_init"] = np.asarray(gaia_df["x"].values, dtype=np.float64)
    init_params["y_init"] = np.asarray(gaia_df["y"].values, dtype=np.float64)
    for col in _GAIA_META_COLUMNS:
        if col in gaia_df.columns and col not in ("x", "y"):
            init_params[col] = np.asarray(gaia_df[col].values)
    return init_params


def _photometry_one_frame(
    diff_img: np.ndarray,
    gridded_model,
    gaia_df: pd.DataFrame,
    params,
) -> Table | None:
    """Run multi-star PSFPhotometry on one difference image."""
    from photutils.psf import PSFPhotometry, SourceGrouper

    if gaia_df.empty:
        log.warning("  centroids: no Gaia stars after magnitude filter")
        return None

    init_params = _build_init_params(gaia_df)
    fit_shape = int(getattr(params, "fit_shape", 11) or 11)
    aperture_radius = float(getattr(params, "aperture_radius", 2.0) or 2.0)
    grouper_sep = float(getattr(params, "psf_grouper_min_separation", 8.0) or 8.0)

    psf_phot = PSFPhotometry(
        gridded_model,
        fit_shape=fit_shape,
        aperture_radius=aperture_radius,
        grouper=SourceGrouper(min_separation=grouper_sep),
        local_bkg_estimator=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = psf_phot(
            np.asarray(diff_img, dtype=np.float64),
            init_params=init_params,
        )
    return result if hasattr(result, "colnames") else result.to_table()


def _centroids_one_frame_task(
    frame_idx: int,
    diff_path: str,
) -> tuple[int, str, bool]:
    """Worker: PSF photometry on one frame and write ECSV."""
    ctx = _WORKER_CTX
    gaia_df: pd.DataFrame = ctx["gaia_df"]
    epsf_catalog: GriddedEpsfCatalog = ctx["epsf_catalog"]
    params = ctx["params"]
    output_dir: str = ctx["output_dir"]

    ffi_stem = _diff_path_to_stem(diff_path) if diff_path else f"frame_{frame_idx}"
    if diff_path is None or not os.path.exists(diff_path):
        log.warning("  centroids: diff frame missing: %s", diff_path)
        return frame_idx, ffi_stem, False

    model = epsf_catalog.load_model(ffi_stem)
    if model is None:
        log.warning("  centroids: no gridded ePSF for %s", ffi_stem)
        return frame_idx, ffi_stem, False

    try:
        diff_img = fits.getdata(diff_path).astype(np.float64)
    except Exception as exc:
        log.warning("  centroids: cannot load %s: %s", diff_path, exc)
        return frame_idx, ffi_stem, False

    try:
        phot_results = _photometry_one_frame(diff_img, model, gaia_df, params)
    except Exception as exc:
        log.warning("  centroids: PSF photometry failed for %s: %s", ffi_stem, exc)
        return frame_idx, ffi_stem, False

    if phot_results is None or len(phot_results) == 0:
        log.warning("  centroids: empty photometry result for %s", ffi_stem)
        return frame_idx, ffi_stem, False

    out_path = photresults_ecsv_path(output_dir, ffi_stem)
    os.makedirs(output_dir, exist_ok=True)
    phot_results.write(out_path, format="ascii.ecsv", overwrite=True)
    return frame_idx, ffi_stem, True


def run_centroids_all_frames(
    diff_paths: list[str],
    gaia_df: pd.DataFrame,
    epsf_catalog: GriddedEpsfCatalog,
    cfg,
    params,
    output_dir: str,
    *,
    centroids_label: str | None = None,
    diffs_input: str | None = None,
    diff_log_path: str | None = None,
) -> tuple[list[str], list[bool]]:
    """
    PSF photometry on every difference image (thread-parallel over frames).

    Returns
    -------
    ffi_stems : list of str
    centroids_ok : list of bool
    """
    n_frames = len(diff_paths)
    n_workers = resolve_effective_n_jobs(
        int(getattr(cfg, "n_jobs", 1) or 1),
        stage_n_jobs=getattr(params, "centroids_n_jobs", None),
    )
    _configure_blas_threads(n_workers)

    # Reuse the same Gaia brightness filter as the ePSF stage.
    class _MagParams:
        mag_max_rp = getattr(params, "mag_max_rp", 12.95)

    gaia_filtered = prepare_gaia_for_gridded_epsf(gaia_df, _MagParams())
    os.makedirs(output_dir, exist_ok=True)

    from syndiff_pipeline.difference_imaging.stages.centroids_progress import (
        init_progress_pair,
        progress_path_for_diff_log,
        progress_path_for_output_workspace,
        record_frame_progress,
        refresh_progress_pair_from_artifacts,
        set_progress_phase_pair,
    )

    track_progress = centroids_label is not None
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
            centroids_label=str(centroids_label),
            diffs_input=str(diffs_input or "?"),
            frames_total=n_frames,
            output_dir=output_dir,
        )
        refresh_progress_pair_from_artifacts(
            workspace_progress_path, cli_progress_path
        )

    tasks = [(i, p) for i, p in enumerate(diff_paths)]
    tqdm_desc = f"centroids {centroids_label}" if track_progress else "centroids"

    def _on_frame_done(result: tuple[int, str, bool]) -> None:
        if not track_progress or workspace_progress_path is None:
            return
        record_frame_progress(
            workspace_progress_path,
            cli_progress_path,
            success=bool(result[2]),
        )

    worker_initargs = (gaia_filtered, epsf_catalog, params, output_dir)
    results: list[tuple[int, str, bool]] = []

    if n_workers <= 1 or n_frames <= 1:
        _init_centroids_worker(*worker_initargs)
        log.info(
            "centroids [%s]: starting %d frames (n_jobs=1)",
            centroids_label or "?",
            n_frames,
        )
        for task in tqdm_iter(tasks, desc=tqdm_desc):
            result = _centroids_one_frame_task(*task)
            _on_frame_done(result)
            results.append(result)
    else:
        log.info(
            "centroids [%s]: starting %d frames (n_jobs=%s, backend=loky)",
            centroids_label or "?",
            n_frames,
            n_workers,
        )
        delayed_calls = [delayed(_centroids_one_frame_task)(i, p) for i, p in tasks]
        results = parallel_map_with_optional_tqdm(
            delayed_calls,
            n_tasks=n_frames,
            desc=tqdm_desc,
            n_jobs_eff=n_workers,
            initializer=_init_centroids_worker,
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
    centroids_ok = [r[2] for r in results]

    index: dict[str, str] = {}
    for stem, ok in zip(ffi_stems, centroids_ok):
        if ok:
            index[stem] = photresults_ecsv_path(output_dir, stem)
    save_centroids_index(output_dir, index)

    n_ok = sum(centroids_ok)
    log.info(
        "centroids [%s]: %d/%d frames wrote %s",
        centroids_label or "?",
        n_ok,
        n_frames,
        PHOTRESULTS_ECSV_SUFFIX,
    )
    return ffi_stems, centroids_ok
