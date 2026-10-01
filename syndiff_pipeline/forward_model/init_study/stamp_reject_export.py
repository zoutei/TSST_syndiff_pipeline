# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Export-time Hotpants stamp QA mask for forward_epsf_wcs bundles."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model.init_study.build_epsf_prior import StudyConfig, _load_config
from syndiff_pipeline.forward_model.init_study.lc_baselines import (
    load_cheb_static,
    load_wcs_artifacts,
    predict_xy_warmstart,
)
from syndiff_pipeline.forward_model.init_study.stamp_qa import (
    StampQaConfig,
    apply_hotpants_stamp_qa,
    load_gridded_model_from_pooled_npz,
    run_gridded_photometry,
)
from syndiff_pipeline.forward_model.init_study.wcs_warmstart_viz import _select_fit_frames
from syndiff_pipeline.forward_model._vendor.sat_star_lc.rgi_epsf_phot import load_fits_image_data, resolve_hp_d

log = logging.getLogger(__name__)

CHECKPOINT_EVERY = 10
DEFAULT_WORKERS = max(1, min(16, (os.cpu_count() or 4) // 2))
EXPORT_KEEP_NAME = "export_stamp_keep.parquet"
EXPORT_CKPT_NAME = "export_stamp_keep.checkpoint.parquet"
EXPORT_META_NAME = "export_stamp_keep_meta.json"
FIT_STARS_CACHE = "_fit_stars_export.parquet"


def _export_paths(study_dir: Path) -> tuple[Path, Path, Path]:
    d = study_dir / "stamp_qa"
    d.mkdir(parents=True, exist_ok=True)
    return d / EXPORT_KEEP_NAME, d / EXPORT_CKPT_NAME, d / EXPORT_META_NAME


def _load_epsf_model(study_dir: Path):
    epsf_dir = study_dir / "epsf_prior"
    meta = json.loads((epsf_dir / "build_meta.json").read_text(encoding="utf-8"))
    lane_npz = Path(meta["lane_npz"])
    if not lane_npz.is_file():
        oi = int(meta.get("orbit_index_1based", 1))
        alt = epsf_dir / f"orbit{oi}_mid20_epsf_2x2_mag813.npz"
        lane_npz = alt if alt.is_file() else lane_npz
    crop_origin = tuple(meta["crop_origin"])
    return load_gridded_model_from_pooled_npz(lane_npz, crop_origin=crop_origin)


def _stamp_qa_one_frame(
    *,
    stem: str,
    btjd: float,
    frame_index: int,
    wcs_fi: int,
    study_dir: str,
    workspace: str,
    stars_path: str,
    qa_cfg: dict,
) -> list[dict]:
    """Picklable per-frame worker (loky process). Loads hp_d + ePSF locally."""
    study = Path(study_dir)
    stars = pd.read_parquet(stars_path)
    wcs = load_wcs_artifacts(study)
    cheb_static = load_cheb_static(_load_config(study / "config.yaml"))
    epsf_model = _load_epsf_model(study)
    cfg = StampQaConfig(**qa_cfg)

    hp = resolve_hp_d(Path(workspace), stem)
    if hp is None:
        return []
    try:
        image = load_fits_image_data(Path(hp))
    except Exception:
        return []

    x_init, y_init = predict_xy_warmstart(
        stars["ra"].to_numpy(dtype=float),
        stars["dec"].to_numpy(dtype=float),
        wcs_fi,
        wcs=wcs,
        cheb_static=cheb_static,
    )
    frame_stars = stars.copy()
    frame_stars["x"] = x_init
    frame_stars["y"] = y_init

    phot, _ = run_gridded_photometry(
        image,
        epsf_model,
        frame_stars,
        cfg=cfg,
        fix_xy=False,
        compute_stamp_chi2=True,
    )
    qa = apply_hotpants_stamp_qa(phot, cfg=cfg)
    rows: list[dict] = []
    for _, row in qa.table.iterrows():
        rows.append(
            {
                "stem": stem,
                "btjd": float(btjd),
                "frame_index": int(frame_index),
                "source_id": int(row["source_id"]),
                "keep": bool(row["keep"]),
                "pass_phase1": bool(row["pass_phase1"]),
                "pass_phase2": bool(row["pass_phase2"]),
                "stamp_chi2": float(row["stamp_chi2"]),
            }
        )
    return rows


def _write_checkpoint(ckpt_path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(ckpt_path, index=False)


def build_stamp_keep_table(
    cfg: StudyConfig,
    frames: list,
    fit_stars: pd.DataFrame,
    *,
    force: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
    n_workers: int = DEFAULT_WORKERS,
    qa_cfg: StampQaConfig | None = None,
) -> pd.DataFrame:
    """Run Hotpants stamp QA on all fit-window frames for primary stars."""
    from joblib import delayed

    from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm

    out_path, ckpt_path, meta_path = _export_paths(cfg.study_dir)
    fit_fingerprint = {
        "n_fit_stars": int(len(fit_stars)),
        "source_ids": sorted(int(s) for s in fit_stars["source_id"].astype("int64")),
    }
    if ckpt_path.is_file() and not force:
        invalidate_ckpt = False
        if meta_path.is_file():
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved_meta.get("fit_fingerprint") != fit_fingerprint:
                invalidate_ckpt = True
        else:
            ckpt = pd.read_parquet(ckpt_path)
            if int(ckpt["source_id"].nunique()) != len(fit_stars):
                invalidate_ckpt = True
        if invalidate_ckpt:
            log.warning(
                "stamp QA checkpoint does not match fit_stars (n=%d); restarting",
                len(fit_stars),
            )
            ckpt_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
    if out_path.is_file() and not force:
        df = pd.read_parquet(out_path)
        log.info("using cached stamp keep table %s (%d rows)", out_path, len(df))
        return df

    qa_cfg = qa_cfg or StampQaConfig()
    qa_cfg_dict = asdict(qa_cfg)
    wcs = load_wcs_artifacts(cfg.study_dir)
    stems_wcs = [str(s) for s in wcs.stems]

    stars_cache = cfg.study_dir / "stamp_qa" / FIT_STARS_CACHE
    stars_cache.parent.mkdir(parents=True, exist_ok=True)
    fit_stars.to_parquet(stars_cache, index=False)

    all_rows: list[dict] = []
    done_stems: set[str] = set()
    if ckpt_path.is_file() and not force:
        ckpt = pd.read_parquet(ckpt_path)
        all_rows = ckpt.to_dict(orient="records")
        done_stems = set(ckpt["stem"].astype(str))
        log.info(
            "resuming stamp QA checkpoint: %d/%d frames, %d rows",
            len(done_stems),
            len(frames),
            len(all_rows),
        )

    if "source_id" not in fit_stars.columns:
        raise ValueError("fit_stars must include source_id")

    pending: list[tuple] = []
    for fi, frame in enumerate(frames):
        stem = frame.stem
        if stem in done_stems:
            continue
        if stem not in stems_wcs:
            log.warning("skip stamp QA frame %s: not in wcs_coeff stems", stem)
            continue
        wcs_fi = stems_wcs.index(stem)
        pending.append((stem, float(frame.btjd), int(fi), int(wcs_fi)))

    if not pending:
        out_df = pd.DataFrame(all_rows)
    else:
        n_workers_eff = max(1, min(int(n_workers), len(pending)))
        log.info(
            "stamp QA parallel: %d pending frames, %d primaries, workers=%d",
            len(pending),
            len(fit_stars),
            n_workers_eff,
        )
        jobs = [
            delayed(_stamp_qa_one_frame)(
                stem=stem,
                btjd=btjd,
                frame_index=fi,
                wcs_fi=wcs_fi,
                study_dir=str(cfg.study_dir),
                workspace=str(cfg.workspace),
                stars_path=str(stars_cache),
                qa_cfg=qa_cfg_dict,
            )
            for stem, btjd, fi, wcs_fi in pending
        ]

        n_done = len(done_stems)

        def _on_result(frame_rows: list[dict]) -> None:
            nonlocal n_done
            if not frame_rows:
                return
            all_rows.extend(frame_rows)
            done_stems.add(str(frame_rows[0]["stem"]))
            n_done += 1
            if n_done % checkpoint_every == 0 or n_done == len(frames):
                _write_checkpoint(ckpt_path, all_rows)
                keep_frac = float(np.mean([r["keep"] for r in all_rows])) if all_rows else 0.0
                log.info(
                    "stamp QA checkpoint %d/%d frames (%d rows, keep=%.1f%%)",
                    n_done,
                    len(frames),
                    len(all_rows),
                    100.0 * keep_frac,
                )

        parallel_map_with_optional_tqdm(
            jobs,
            n_tasks=len(jobs),
            desc="stamp QA frames",
            n_jobs_eff=n_workers_eff,
            on_result=_on_result,
        )
        out_df = pd.DataFrame(all_rows)

    out_df.to_parquet(out_path, index=False)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    meta = {
        "n_frames": int(out_df["stem"].nunique()) if len(out_df) else 0,
        "n_primaries": int(out_df["source_id"].nunique()) if len(out_df) else 0,
        "n_rows": int(len(out_df)),
        "frac_keep": float(out_df["keep"].mean()) if len(out_df) else 0.0,
        "ker_sig_reject": float(qa_cfg.ker_sig_reject),
        "n_workers": int(n_workers),
        "fit_fingerprint": fit_fingerprint,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows, keep=%.1f%%)", out_path, len(out_df), 100.0 * meta["frac_keep"])
    return out_df


def load_or_build_stamp_keep(
    cfg: StudyConfig,
    frames: list,
    fit_stars: pd.DataFrame,
    *,
    force: bool = False,
    n_workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    return build_stamp_keep_table(
        cfg, frames, fit_stars, force=force, n_workers=n_workers,
    )


def apply_stamp_keep_mask(
    mask_active: np.ndarray,
    keep_df: pd.DataFrame,
    *,
    groups,
    expanded_stars: pd.DataFrame,
    primary_index_set: set[int],
    frames: list,
) -> tuple[np.ndarray, dict]:
    """AND init-study stamp QA keep into (n_groups, n_frames) mask_active."""
    mask = np.asarray(mask_active, dtype=np.float32).copy()
    if keep_df.empty:
        return mask, {"enabled": False, "reason": "empty_keep_table"}

    keep_map = {
        (str(row.stem), int(row.source_id)): bool(row.keep)
        for row in keep_df.itertuples(index=False)
    }
    source_ids = expanded_stars["source_id"].astype("int64").to_numpy()
    mags = expanded_stars["tess_mag"].to_numpy(dtype=float)
    stems = [f.stem for f in frames]

    n_before = int(mask.sum())
    n_clipped = 0
    for gi in range(groups.n_groups):
        idx = groups.members[gi][groups.valid[gi]]
        prim = [int(i) for i in idx if int(i) in primary_index_set]
        if not prim:
            continue
        pi = int(prim[int(np.argmin(mags[prim]))])
        sid = int(source_ids[pi])
        for ti, stem in enumerate(stems):
            keep = keep_map.get((stem, sid), True)
            if not keep and mask[gi, ti] > 0:
                mask[gi, ti] = 0.0
                n_clipped += 1

    stats = {
        "enabled": True,
        "n_cells_clipped": int(n_clipped),
        "n_cells_before": int(n_before),
        "n_cells_after": int(mask.sum()),
        "frac_keep_after": float(mask.mean()) if mask.size else 0.0,
    }
    return mask, stats


def resolve_study_config(path: Path) -> StudyConfig:
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return _load_config(path.resolve())
    if path.is_dir() and (path / "config.yaml").is_file():
        return _load_config((path / "config.yaml").resolve())
    raise ValueError(f"expected init study config.yaml or study dir, got {path}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args(argv)
    cfg = resolve_study_config(args.config)
    import yaml

    from syndiff_pipeline.forward_model import cheb_wcs as CW
    from syndiff_pipeline.forward_model import gaia_pm as GP
    from syndiff_pipeline.forward_model.data import filter_stars_by_xy, primary_candidates_by_mag
    from syndiff_pipeline.forward_model.shared_wcs import fit_region_shared_wcs
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_gaia_catalog

    _, frames, _ = _select_fit_frames(cfg)
    raw = yaml.safe_load((cfg.study_dir / "config.yaml").read_text(encoding="utf-8"))

    mag = raw["bundle"]["tess_mag"]
    mag_lo, mag_hi = float(mag[0]), float(mag[1])

    gaia_full = load_gaia_catalog(cfg.workspace)
    target_btjd = float(np.median([f.btjd for f in frames]))
    gaia_full, _ = GP.apply_pm_to_dataframe(gaia_full, target_btjd)
    wcs, _ = fit_region_shared_wcs(
        frames[0], gaia_full, cfg.region, margin_px=float(cfg.region_margin_px),
    )
    cheb_static = CW.ChebWcsStatic.from_wcs(
        wcs, cfg.region, poly_degree=int(raw["wcs"]["cheb_degree"]),
    )
    wcs_art = load_wcs_artifacts(cfg.study_dir)

    primary_cand = primary_candidates_by_mag(gaia_full, tess_mag_range=(mag_lo, mag_hi))
    ra_pc = primary_cand["ra"].to_numpy(dtype=float)
    dec_pc = primary_cand["dec"].to_numpy(dtype=float)
    ref_fi = len(frames) // 2
    x_ws, y_ws = CW.eval_positions_at_frame_index(
        ra_pc, dec_pc, wcs_art.wcs_coeff, cheb_static, wcs_art.wcs_frame_basis, ref_fi,
    )
    gaia_fit, x_ws_fit, y_ws_fit = filter_stars_by_xy(
        primary_cand, x_ws, y_ws, cfg.region, margin_px=float(cfg.region_margin_px),
    )
    fit_stars = gaia_fit.loc[~gaia_fit.too_bright & ~gaia_fit.too_faint].reset_index(drop=True)
    log.info("building stamp keep for %d primaries x %d frames", len(fit_stars), len(frames))

    out = build_stamp_keep_table(
        cfg, frames, fit_stars, force=args.force, n_workers=int(args.workers),
    )
    print(json.dumps({"ok": True, "n_rows": len(out), "frac_keep": float(out["keep"].mean())}, indent=2))


if __name__ == "__main__":
    main()
