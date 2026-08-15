"""Per-FFI Sci2Idl WCS fit on centroids (parallel over frames)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from joblib import delayed

from syndiff_pipeline.common.fits_variants import try_resolve_fits_variant
from syndiff_pipeline.common.joblib_progress import (
    parallel_map_with_optional_tqdm,
    tqdm_iter,
)
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.difference_imaging.stages.centroids import (
    load_centroids_index,
)
from syndiff_pipeline.difference_imaging.stages.per_ffi_wcs_progress import (
    init_progress,
    record_frame_done,
)
from syndiff_pipeline.difference_imaging.wcs.io import (
    FrameRecord,
    btjd_from_header,
)
from syndiff_pipeline.difference_imaging.wcs.audit_matrix import (
    AUDIT_NPZ,
    StarAuditMatrixWriter,
    build_star_index,
)
from syndiff_pipeline.difference_imaging.wcs.metrics import compute_frame_qc_metrics
from syndiff_pipeline.difference_imaging.wcs.sci2idl import sci2idl_du_dv_px
from syndiff_pipeline.difference_imaging.wcs.reference import reference_wcs_from_tesswcs
from syndiff_pipeline.difference_imaging.wcs.sci2idl import (
    StarSelectionConfig,
    build_frame_stars,
    crop_bounds_from_header,
    join_stars,
    select_good_stars,
    warmstart_frame,
    warmstart_table_row,
)

log = logging.getLogger(__name__)

COEFFS_CSV = "per_ffi_coeffs.csv"
METRICS_CSV = "frame_qc_metrics.csv"
META_JSON = "fit_meta.json"
WCS_INDEX_JSON = "wcs_index.json"

_WORKER_CTX: dict[str, Any] = {}


def _init_per_ffi_wcs_worker(
    frames_by_stem: dict[str, FrameRecord],
    gaia_df: pd.DataFrame,
    reference_wcs: WCS,
    star_cfg: StarSelectionConfig,
    sip_degree: int,
    skip_stems: set[str] | None = None,
) -> None:
    """Load shared per-FFI WCS inputs once per loky worker."""
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "frames_by_stem": frames_by_stem,
            "gaia_df": gaia_df,
            "reference_wcs": reference_wcs,
            "star_cfg": star_cfg,
            "sip_degree": sip_degree,
            "skip_stems": set(skip_stems or ()),
        }
    )


def _configure_blas_threads(n_workers: int) -> None:
    cpu_cap = os.cpu_count() or 1
    per_worker = max(1, cpu_cap // max(1, n_workers))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(per_worker)


def list_frames_for_lane(
    lane_root: Path,
    *,
    centroids_label: str,
    hp_d_label: str,
) -> list[FrameRecord]:
    centroids_dir = lane_root / centroids_label
    hp_d_dir = lane_root / hp_d_label
    index = load_centroids_index(str(centroids_dir))
    frames: list[FrameRecord] = []
    for stem, phot_rel in sorted(index.items()):
        phot_path = Path(phot_rel)
        if not phot_path.is_file():
            phot_path = centroids_dir / Path(phot_rel).name
        hp_path = try_resolve_fits_variant(hp_d_dir / f"{stem}_hp_d.fits")
        if hp_path is None:
            candidates = sorted(hp_d_dir.glob(f"{stem}*.fits*"))
            if not candidates:
                continue
            hp_path = str(candidates[0])
        hdr = fits.getheader(hp_path, ext=1)
        ny, nx = crop_bounds_from_header(hdr)["shape"]
        frames.append(
            FrameRecord(
                stem=stem,
                btjd=btjd_from_header(hdr),
                hp_d_path=Path(hp_path),
                phot_path=phot_path,
                crop_shape=(ny, nx),
            )
        )
    frames.sort(key=lambda f: f.btjd)
    return frames


def _fit_one_frame_task(frame_idx: int, stem: str) -> dict[str, Any]:
    ctx = _WORKER_CTX
    fr_by_stem: dict[str, FrameRecord] = ctx["frames_by_stem"]
    fr = fr_by_stem[stem]
    star_cfg: StarSelectionConfig = ctx["star_cfg"]
    sip_degree: int = ctx["sip_degree"]
    reference_wcs: WCS = ctx["reference_wcs"]
    gaia: pd.DataFrame = ctx["gaia_df"]
    skip_existing: set[str] = ctx.get("skip_stems", set())

    empty = {
        "frame_idx": frame_idx,
        "stem": stem,
        "ok": False,
        "skipped": False,
        "coeff_row": None,
        "metrics_row": None,
        "audit_frame": None,
        "log": f"{stem}: missing frame record",
    }
    if stem in skip_existing:
        return {**empty, "ok": True, "skipped": True, "log": f"{stem}: skipped (existing)"}

    if not fr.phot_path.is_file():
        return {**empty, "log": f"{stem}: missing photresults {fr.phot_path}"}

    phot = Table.read(fr.phot_path, format="ascii.ecsv")
    merged = join_stars(phot, gaia)
    qc = select_good_stars(merged, star_cfg)
    n_qc = len(qc)
    if n_qc < star_cfg.min_stars:
        coeff_row = warmstart_table_row(
            fr.stem,
            fr.btjd,
            _empty_result(reference_wcs, sip_degree),
            fit_ok=False,
            n_stars_qc=n_qc,
            message=f"only {n_qc} QC stars",
        )
        metrics_row = compute_frame_qc_metrics(
            _empty_result(reference_wcs, sip_degree),
            qc if n_qc else merged.head(0),
            stem=fr.stem,
            btjd=fr.btjd,
            fit_ok=False,
            message=coeff_row["message"],
        )
        return {
            "frame_idx": frame_idx,
            "stem": stem,
            "ok": False,
            "skipped": False,
            "coeff_row": coeff_row,
            "metrics_row": metrics_row,
            "audit_frame": None,
            "log": f"{stem}: QC={n_qc} insufficient",
        }

    stars = build_frame_stars(qc, reference_wcs, stem=fr.stem, btjd=fr.btjd)
    result = warmstart_frame(stars, reference_wcs, sip_degree=sip_degree, star_cfg=star_cfg)
    coeff_row = warmstart_table_row(fr.stem, fr.btjd, result, fit_ok=True, n_stars_qc=n_qc)
    metrics_row = compute_frame_qc_metrics(
        result, stars, stem=fr.stem, btjd=fr.btjd, fit_ok=True
    )
    du, dv = sci2idl_du_dv_px(result, stars)
    log_line = (
        f"{fr.stem}: QC={n_qc} keep={int(result.keep_mask.sum())} "
        f"med_hypot={metrics_row.get('med_hypot', float('nan')):.4f}"
    )
    return {
        "frame_idx": frame_idx,
        "stem": stem,
        "ok": True,
        "skipped": False,
        "coeff_row": coeff_row,
        "metrics_row": metrics_row,
        "audit_frame": {
            "frame_idx": frame_idx,
            "source_id": qc["source_id"].to_numpy(np.int64),
            "du": du.astype(np.float32),
            "dv": dv.astype(np.float32),
            "keep_mask": result.keep_mask.astype(bool),
        },
        "log": log_line,
    }


def _empty_result(reference_wcs: WCS, sip_degree: int):
    from syndiff_pipeline.difference_imaging.wcs.sci2idl import Sci2IdlFitResult
    from syndiff_pipeline.difference_imaging.wcs.sip_poly_fit import n_sci2idl_terms

    n = n_sci2idl_terms(sip_degree)
    cx = [0.0, 1.0, 0.0] + [0.0] * max(0, n - 3)
    cy = [0.0, 0.0, 1.0] + [0.0] * max(0, n - 3)
    return Sci2IdlFitResult(
        linear_wcs=reference_wcs,
        coeff_x=cx,
        coeff_y=cy,
        poly_degree=sip_degree,
        rotation_fit_x=True,
        rotation_fit_y=True,
        keep_mask=np.zeros(0, dtype=bool),
    )


def run_per_ffi_wcs_all_frames(
    lane_root: str,
    gaia_df: pd.DataFrame,
    cfg,
    params,
    output_dir: str,
    *,
    centroids_label: str,
    hp_d_label: str,
    wcs_label: str | None = None,
    diff_log_path: str | None = None,
    force_rerun: bool = False,
    sector: int,
    camera: int,
    ccd: int,
) -> tuple[int, int]:
    """Fit per-FFI Sci2Idl coefficients for every centroids frame."""
    lane_path = Path(lane_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = list_frames_for_lane(
        lane_path, centroids_label=centroids_label, hp_d_label=hp_d_label
    )
    if not frames:
        raise RuntimeError(f"No centroid frames under {lane_path / centroids_label}")

    crop_bounds = crop_bounds_from_header(fits.getheader(frames[0].hp_d_path, ext=1))
    reference_wcs = reference_wcs_from_tesswcs(sector, camera, ccd, crop_bounds)

    star_cfg = StarSelectionConfig(
        clip_n_sigma=float(params.clip_n_sigma),
        clip_max_iter=int(params.clip_max_iter),
        min_stars=int(params.min_stars),
    )
    sip_degree = int(params.sip_degree)

    skip_stems: set[str] = set()
    coeffs_path = out / COEFFS_CSV
    if coeffs_path.is_file() and not force_rerun:
        existing = pd.read_csv(coeffs_path)
        if "stem" in existing.columns:
            skip_stems = set(existing["stem"].astype(str))

    n_workers = resolve_effective_n_jobs(
        int(getattr(cfg, "n_jobs", 1) or 1),
        stage_n_jobs=getattr(params, "per_ffi_wcs_n_jobs", None),
    )
    _configure_blas_threads(n_workers)

    global _WORKER_CTX
    frames_by_stem = {f.stem: f for f in frames}
    worker_initargs = (
        frames_by_stem,
        gaia_df,
        reference_wcs,
        star_cfg,
        sip_degree,
        skip_stems,
    )
    _WORKER_CTX = {
        "frames_by_stem": frames_by_stem,
        "gaia_df": gaia_df,
        "reference_wcs": reference_wcs,
        "star_cfg": star_cfg,
        "sip_degree": sip_degree,
        "skip_stems": skip_stems,
    }

    stems = [f.stem for f in frames]
    btjd_values = [f.btjd for f in frames]
    n_frames = len(stems)
    progress_path = out / "wcs.progress.json"
    init_progress(
        progress_path,
        wcs_label=wcs_label or "wcs",
        centroids_input=centroids_label,
        n_frames=n_frames,
        diff_log_path=diff_log_path,
    )

    star_index = build_star_index(
        gaia_df,
        phot_paths=[f.phot_path for f in frames],
    )
    audit_writer = StarAuditMatrixWriter(
        star_index,
        stems,
        btjd_values,
        out,
    )

    def _on_frame_done(res: dict[str, Any]) -> None:
        record_frame_done(progress_path, stem=res["stem"], ok=bool(res.get("ok")))
        audit_writer.write_audit_frame(res.get("audit_frame"))

    if n_workers <= 1 or n_frames <= 1:
        _init_per_ffi_wcs_worker(*worker_initargs)
        results = []
        for i, stem in enumerate(tqdm_iter(stems, desc="per_ffi_wcs")):
            res = _fit_one_frame_task(i, stem)
            _on_frame_done(res)
            results.append(res)
    else:
        delayed_calls = [delayed(_fit_one_frame_task)(i, stem) for i, stem in enumerate(stems)]
        results = parallel_map_with_optional_tqdm(
            delayed_calls,
            n_tasks=n_frames,
            desc="per_ffi_wcs",
            n_jobs_eff=n_workers,
            initializer=_init_per_ffi_wcs_worker,
            initargs=worker_initargs,
            on_result=_on_frame_done,
        )

    coeff_rows: list[dict] = []
    metrics_rows: list[dict] = []
    index: dict[str, str] = {}
    n_ok = 0
    n_skipped = 0

    for res in results:
        log.info("  %s", res.get("log", res["stem"]))
        if res.get("skipped"):
            n_skipped += 1
            continue
        if res.get("coeff_row") is not None:
            coeff_rows.append(res["coeff_row"])
        if res.get("metrics_row") is not None:
            metrics_rows.append(res["metrics_row"])
        if res.get("ok"):
            n_ok += 1
            index[res["stem"]] = "ok"
        else:
            index[res["stem"]] = res.get("log", "failed")

    if skip_stems and coeffs_path.is_file() and not force_rerun:
        prev = pd.read_csv(coeffs_path)
        new_df = pd.DataFrame(coeff_rows)
        coeff_df = (
            pd.concat([prev, new_df], ignore_index=True)
            .drop_duplicates(subset=["stem"], keep="last")
            .sort_values("btjd")
        )
    else:
        coeff_df = pd.DataFrame(coeff_rows).sort_values("btjd")

    coeff_df.to_csv(out / COEFFS_CSV, index=False)
    pd.DataFrame(metrics_rows).sort_values("btjd").to_csv(out / METRICS_CSV, index=False)
    audit_writer.finalize()

    with open(out / WCS_INDEX_JSON, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    meta = {
        "lane_root": str(lane_path),
        "sector": sector,
        "camera": camera,
        "ccd": ccd,
        "reference_wcs": "tesswcs.from_sector + crop",
        "sip_degree": sip_degree,
        "clip_n_sigma": star_cfg.clip_n_sigma,
        "n_frames": len(frames),
        "n_fit_ok": int(coeff_df["fit_ok"].sum()) if "fit_ok" in coeff_df.columns else n_ok,
        "n_skipped_existing": n_skipped,
        "centroids_label": centroids_label,
        "audit_artifact": AUDIT_NPZ,
        "n_audit_stars": int(len(star_index.source_id)),
    }
    with open(out / META_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    return n_ok, len(frames)
