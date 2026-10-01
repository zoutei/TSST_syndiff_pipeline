# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Export per-FFI centroids + flux from built gridded ePSF (free-xy photometry).

Replaces ``centroids_r1`` for init studies that fit WCS from the pooled ePSF
prior. Parallel over frames (loky); checkpointed parquet output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import gaia_pm as GP
from syndiff_pipeline.forward_model.data import RegionSpec, xy_in_region_mask
from syndiff_pipeline.forward_model.init_study.build_epsf_prior import StudyConfig, _load_config
from syndiff_pipeline.forward_model.init_study.lc_baselines import _load_epsf_model
from syndiff_pipeline.forward_model.init_study.stamp_qa import StampQaConfig, run_gridded_photometry
from syndiff_pipeline.forward_model.init_study.wcs_warmstart_viz import _select_fit_frames
from syndiff_pipeline.forward_model.shared_wcs import fit_region_shared_wcs
from syndiff_pipeline.forward_model._vendor.sat_star_lc.rgi_epsf_phot import load_fits_image_data, resolve_hp_d

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_gaia_catalog  # noqa: E402

log = logging.getLogger(__name__)

CHECKPOINT_EVERY = 10
DEFAULT_WORKERS = max(1, min(16, (os.cpu_count() or 4) // 2))
OUT_NAME = "photometry.parquet"
CKPT_NAME = "photometry.checkpoint.parquet"
META_NAME = "export_meta.json"
STAR_POOL_NAME = "star_pool.parquet"


def _centroids_dir(study_dir: Path) -> Path:
    d = study_dir / "epsf_centroids"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _study_mag_range(cfg: StudyConfig) -> tuple[float, float]:
    raw = yaml.safe_load(cfg.study_dir.joinpath("config.yaml").read_text(encoding="utf-8"))
    mag = raw.get("bundle", {}).get("tess_mag") or raw["epsf_pool"]["tess_mag"]
    return float(mag[0]), float(mag[1])


def build_star_pool(cfg: StudyConfig, *, force: bool = False) -> pd.DataFrame:
    """Gaia stars in region with linear-WCS init positions (ref frame)."""
    out_dir = _centroids_dir(cfg.study_dir)
    pool_path = out_dir / STAR_POOL_NAME
    if pool_path.is_file() and not force:
        return pd.read_parquet(pool_path)

    mag_lo, mag_hi = _study_mag_range(cfg)
    _, frames, _ = _select_fit_frames(cfg)
    gaia_full = load_gaia_catalog(cfg.workspace)
    target_btjd = float(np.median([f.btjd for f in frames]))
    gaia_full, _ = GP.apply_pm_to_dataframe(gaia_full, target_btjd)

    margin = float(cfg.region_margin_px)
    wcs, _ = fit_region_shared_wcs(frames[0], gaia_full, cfg.region, margin_px=margin)
    cheb_static = CW.ChebWcsStatic.from_wcs(wcs, cfg.region, poly_degree=4)

    mag = pd.to_numeric(gaia_full["tess_mag"], errors="coerce")
    sub = gaia_full.loc[np.isfinite(mag) & (mag >= mag_lo) & (mag <= mag_hi)].copy()
    sub["tess_mag"] = mag.loc[sub.index].to_numpy(dtype=float)
    ra = sub["ra"].to_numpy(dtype=float)
    dec = sub["dec"].to_numpy(dtype=float)
    x_lin, y_lin = map(np.asarray, CW.linear_predict(ra, dec, cheb_static))
    mask = xy_in_region_mask(x_lin, y_lin, cfg.region, margin_px=margin)
    pool = sub.loc[mask].copy().reset_index(drop=True)
    pool["x_init"] = x_lin[mask]
    pool["y_init"] = y_lin[mask]

    pool.to_parquet(pool_path, index=False)
    log.info("wrote star pool %s (%d stars, mag %.1f-%.1f)", pool_path, len(pool), mag_lo, mag_hi)
    return pool


def _phot_one_frame(
    *,
    stem: str,
    btjd: float,
    frame_index: int,
    study_dir: str,
    workspace: str,
    stars_path: str,
    phot_cfg: dict,
) -> list[dict]:
    """Picklable per-frame worker."""
    study = Path(study_dir)
    stars = pd.read_parquet(stars_path)
    epsf_model = _load_epsf_model(_load_config(study / "config.yaml"))
    cfg = StampQaConfig(**phot_cfg)

    hp = resolve_hp_d(Path(workspace), stem)
    if hp is None:
        return []
    try:
        image = load_fits_image_data(Path(hp))
    except Exception:
        return []

    frame_stars = stars.copy()
    phot_df, _ = run_gridded_photometry(
        image,
        epsf_model,
        frame_stars,
        cfg=cfg,
        fix_xy=False,
        compute_stamp_chi2=False,
    )
    if phot_df.empty:
        return []

    rows: list[dict] = []
    for _, row in phot_df.iterrows():
        flux = float(row.get("flux_fit", np.nan))
        if not np.isfinite(flux):
            continue
        x_fit = float(row.get("x_fit", np.nan))
        y_fit = float(row.get("y_fit", np.nan))
        if not (np.isfinite(x_fit) and np.isfinite(y_fit)):
            continue
        sid = int(row["source_id"]) if "source_id" in row and pd.notna(row["source_id"]) else -1
        rows.append(
            {
                "stem": stem,
                "btjd": float(btjd),
                "frame_index": int(frame_index),
                "source_id": sid,
                "ra": float(row["ra"]) if "ra" in row and pd.notna(row["ra"]) else float("nan"),
                "dec": float(row["dec"]) if "dec" in row and pd.notna(row["dec"]) else float("nan"),
                "tess_mag": float(row["tess_mag"]) if "tess_mag" in row and pd.notna(row["tess_mag"]) else float("nan"),
                "x_init": float(row.get("x_init", np.nan)),
                "y_init": float(row.get("y_init", np.nan)),
                "x_fit": x_fit,
                "y_fit": y_fit,
                "flux_fit": flux,
                "flux_err": float(row.get("flux_err", np.nan)),
            }
        )
    return rows


def export_epsf_centroids(
    cfg: StudyConfig,
    *,
    force: bool = False,
    n_workers: int = DEFAULT_WORKERS,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """Run free-xy gridded photometry on all fit-window frames."""
    from joblib import delayed

    from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm

    out_dir = _centroids_dir(cfg.study_dir)
    out_path = out_dir / OUT_NAME
    ckpt_path = out_dir / CKPT_NAME
    meta_path = out_dir / META_NAME

    if out_path.is_file() and not force:
        df = pd.read_parquet(out_path)
        log.info("using cached %s (%d rows)", out_path, len(df))
        return df

    if not (cfg.study_dir / "epsf_prior" / "epsf_base_init.npz").is_file():
        raise FileNotFoundError(
            f"missing ePSF prior; run build_epsf_prior first: {cfg.study_dir / 'epsf_prior'}"
        )

    stars = build_star_pool(cfg, force=force)
    stars_cache = out_dir / "_star_pool_worker.parquet"
    stars.to_parquet(stars_cache, index=False)

    _, frames, _ = _select_fit_frames(cfg)
    phot_cfg = StampQaConfig()
    phot_cfg_dict = {
        "fit_shape": phot_cfg.fit_shape,
        "aperture_radius": phot_cfg.aperture_radius,
        "grouper_min_separation": phot_cfg.grouper_min_separation,
        "ker_sig_reject": phot_cfg.ker_sig_reject,
        "sigma_clip_sigma": phot_cfg.sigma_clip_sigma,
        "sigma_clip_maxiters": phot_cfg.sigma_clip_maxiters,
        "noise_floor": phot_cfg.noise_floor,
    }

    all_rows: list[dict] = []
    done_stems: set[str] = set()
    if ckpt_path.is_file() and not force:
        ckpt = pd.read_parquet(ckpt_path)
        all_rows = ckpt.to_dict(orient="records")
        done_stems = set(ckpt["stem"].astype(str))
        log.info("resume checkpoint: %d/%d frames, %d rows", len(done_stems), len(frames), len(all_rows))

    pending = [
        (frame.stem, float(frame.btjd), int(fi))
        for fi, frame in enumerate(frames)
        if frame.stem not in done_stems
    ]
    if not pending:
        out_df = pd.DataFrame(all_rows)
    else:
        n_workers_eff = max(1, min(int(n_workers), len(pending)))
        log.info(
            "epsf centroids: %d pending frames, %d stars, workers=%d",
            len(pending),
            len(stars),
            n_workers_eff,
        )
        jobs = [
            delayed(_phot_one_frame)(
                stem=stem,
                btjd=btjd,
                frame_index=fi,
                study_dir=str(cfg.study_dir),
                workspace=str(cfg.workspace),
                stars_path=str(stars_cache),
                phot_cfg=phot_cfg_dict,
            )
            for stem, btjd, fi in pending
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
                pd.DataFrame(all_rows).to_parquet(ckpt_path, index=False)
                log.info(
                    "checkpoint %d/%d frames (%d rows)",
                    n_done,
                    len(frames),
                    len(all_rows),
                )

        parallel_map_with_optional_tqdm(
            jobs,
            n_tasks=len(jobs),
            desc="epsf centroid frames",
            n_jobs_eff=n_workers_eff,
            on_result=_on_result,
        )
        out_df = pd.DataFrame(all_rows)

    out_df.to_parquet(out_path, index=False)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    meta = {
        "n_frames": int(out_df["stem"].nunique()) if len(out_df) else 0,
        "n_stars_pool": int(len(stars)),
        "n_rows": int(len(out_df)),
        "n_workers": int(n_workers),
        "source": "epsf_gridded_free_xy",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows, %d frames)", out_path, len(out_df), meta["n_frames"])
    return out_df


def merged_by_stem_from_parquet(parquet_path: Path) -> dict[str, pd.DataFrame]:
    """Build ``merged_by_stem`` dict for WCS warmstart (x_fit/y_fit columns)."""
    df = pd.read_parquet(parquet_path)
    out: dict[str, pd.DataFrame] = {}
    for stem, sub in df.groupby("stem", sort=False):
        merged = sub.rename(columns={"x_fit": "x_fit", "y_fit": "y_fit"}).copy()
        merged["flags"] = 0
        merged["qfit"] = 1.0
        merged["x_err"] = 0.01
        merged["y_err"] = 0.01
        merged["cfit"] = 0.0
        out[str(stem)] = merged.reset_index(drop=True)
    return out


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "dev/forward_epsf_wcs/init_study/orbit2/config.yaml",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = p.parse_args(argv)
    cfg = _load_config(args.config.resolve())
    export_epsf_centroids(cfg, force=args.force, n_workers=args.workers)
    print(json.dumps({"ok": True, "study": str(cfg.study_dir)}, indent=2))


if __name__ == "__main__":
    main()
