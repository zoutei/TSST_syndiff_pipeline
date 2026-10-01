# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Fixed vs free GriddedPSF light-curve baselines for init studies."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import gaia_pm as GP
from syndiff_pipeline.forward_model.init_study.build_epsf_prior import StudyConfig, _load_config
from syndiff_pipeline.forward_model.init_study.stamp_qa import (
    StampQaConfig,
    load_gridded_model_from_pooled_npz,
    run_gridded_photometry,
)
from syndiff_pipeline.forward_model.init_study.wcs_warmstart_viz import _select_fit_frames
from syndiff_pipeline.forward_model.shared_wcs import fit_region_shared_wcs
from syndiff_pipeline.forward_model._vendor.sat_star_lc.rgi_epsf_phot import load_fits_image_data, resolve_hp_d

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_gaia_catalog  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_PER_BIN = 25
MAG_BIN_WIDTH = 0.5
MAG_LO = 8.0
MAG_HI = 13.0
CHECKPOINT_EVERY = 25


@dataclass
class WcsArtifacts:
    wcs_coeff: np.ndarray
    wcs_frame_basis: np.ndarray
    btjd: np.ndarray
    stems: np.ndarray
    n_terms: int


def _mag_bin_edges(lo: float = MAG_LO, hi: float = MAG_HI, width: float = MAG_BIN_WIDTH) -> np.ndarray:
    return np.arange(lo, hi + width * 0.5, width, dtype=float)


def _mag_bin_label(mag: float, edges: np.ndarray) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= mag < edges[i + 1]:
            return f"{edges[i]:.1f}-{edges[i + 1]:.1f}"
    return f"{edges[-2]:.1f}-{edges[-1]:.1f}"


def load_wcs_artifacts(study_dir: Path, *, wcs_subdir: str = "wcs_warmstart") -> WcsArtifacts:
    z = np.load(study_dir / wcs_subdir / "wcs_coeff.npz", allow_pickle=True)
    return WcsArtifacts(
        wcs_coeff=np.asarray(z["wcs_coeff"], dtype=np.float64),
        wcs_frame_basis=np.asarray(z["wcs_frame_basis"], dtype=np.float64),
        btjd=np.asarray(z["btjd"], dtype=np.float64),
        stems=np.asarray(z["stems"], dtype=object),
        n_terms=int(np.asarray(z["n_terms"])),
    )


def load_cheb_static(cfg: StudyConfig) -> CW.ChebWcsStatic:
    """Rebuild region Cheb static WCS (same path as wcs_warmstart_viz)."""
    raw = yaml.safe_load(cfg.study_dir.joinpath("config.yaml").read_text(encoding="utf-8"))
    cheb_degree = int(raw["wcs"]["cheb_degree"])
    _, frames, _ = _select_fit_frames(cfg)
    gaia_full = load_gaia_catalog(cfg.workspace)
    target_btjd = float(np.median([f.btjd for f in frames]))
    gaia_full, _ = GP.apply_pm_to_dataframe(gaia_full, target_btjd)
    wcs, _ = fit_region_shared_wcs(frames[0], gaia_full, cfg.region, margin_px=float(cfg.region_margin_px))
    return CW.ChebWcsStatic.from_wcs(wcs, cfg.region, poly_degree=cheb_degree)


def predict_xy_warmstart(
    ra: np.ndarray,
    dec: np.ndarray,
    frame_index: int,
    *,
    wcs: WcsArtifacts,
    cheb_static: CW.ChebWcsStatic,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame warmstart xy from saved wcs_coeff + Cheb static."""
    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.cheb_poly_fit import cheb_design_matrix  # noqa: E402

    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    x_lin, y_lin = map(np.asarray, CW.linear_predict(ra, dec, cheb_static))
    xhat = (x_lin - cheb_static.center[0]) / cheb_static.half_extents[0]
    yhat = (y_lin - cheb_static.center[1]) / cheb_static.half_extents[1]
    basis = cheb_design_matrix(xhat, yhat, cheb_static.poly_degree)
    frame_coeff = wcs.wcs_coeff @ wcs.wcs_frame_basis[int(frame_index)]
    n_terms = int(wcs.n_terms)
    x_pred = x_lin + basis @ frame_coeff[:n_terms]
    y_pred = y_lin + basis @ frame_coeff[n_terms:]
    return np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float)


def _stamp_keep_lookup(study_dir: Path) -> dict[int, bool]:
    stamp_dir = study_dir / "stamp_qa"
    if not stamp_dir.is_dir():
        return {}
    out: dict[int, bool] = {}
    for path in sorted(stamp_dir.glob("*_stamp_qa.csv")):
        df = pd.read_csv(path, usecols=["source_id", "keep"])
        for sid, keep in zip(df["source_id"], df["keep"]):
            out[int(sid)] = bool(keep)
    return out


def select_stars_half_mag(
    cfg: StudyConfig,
    *,
    per_bin: int = DEFAULT_PER_BIN,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> pd.DataFrame:
    """Create-or-load frozen half-mag star sample."""
    out_dir = cfg.study_dir / "lc_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_path = out_dir / "star_selection.csv"
    if sel_path.is_file() and not force:
        return pd.read_csv(sel_path)

    gaia = pd.read_csv(cfg.workspace / "gaia_catalog_pipeline.csv")
    mag = pd.to_numeric(gaia["tess_mag"], errors="coerce")
    x = pd.to_numeric(gaia["x"], errors="coerce")
    y = pd.to_numeric(gaia["y"], errors="coerce")
    x0, y0, x1, y1 = cfg.region.x_min, cfg.region.y_min, cfg.region.x_max, cfg.region.y_max
    pool = gaia.loc[
        np.isfinite(mag)
        & (mag >= MAG_LO)
        & (mag <= MAG_HI)
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
    ].copy()
    pool["tess_mag"] = mag.loc[pool.index].to_numpy(dtype=float)

    edges = _mag_bin_edges()
    keep_map = _stamp_keep_lookup(cfg.study_dir)
    rng = np.random.default_rng(int(seed))
    rows: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sub = pool.loc[(pool["tess_mag"] >= lo) & (pool["tess_mag"] < hi)]
        if sub.empty:
            continue
        n_take = min(int(per_bin), len(sub))
        pick = rng.choice(sub.index.to_numpy(), size=n_take, replace=False)
        for idx in pick:
            row = sub.loc[idx]
            sid = int(row["source_id"])
            rows.append(
                {
                    "source_id": sid,
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "tess_mag": float(row["tess_mag"]),
                    "mag_bin": f"{lo:.1f}-{hi:.1f}",
                    "stamp_keep": bool(keep_map.get(sid, False)),
                    "seed": int(seed),
                }
            )

    sel = pd.DataFrame(rows)
    sel.to_csv(sel_path, index=False)
    log.info("wrote %d stars to %s", len(sel), sel_path)
    return sel


def _load_epsf_model(cfg: StudyConfig):
    epsf_dir = cfg.study_dir / "epsf_prior"
    meta_path = epsf_dir / "build_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lane_npz = Path(meta["lane_npz"])
    if not lane_npz.is_file():
        oi = int(meta.get("orbit_index_1based", 1))
        alt = epsf_dir / f"orbit{oi}_mid20_epsf_2x2_mag813.npz"
        lane_npz = alt if alt.is_file() else lane_npz
    crop_origin = tuple(meta["crop_origin"])
    return load_gridded_model_from_pooled_npz(lane_npz, crop_origin=crop_origin)


def _phot_rows_from_table(
    phot_df: pd.DataFrame,
    *,
    mode: str,
    stem: str,
    btjd: float,
    stars: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    if phot_df.empty:
        return rows
    for _, prow in phot_df.iterrows():
        sid = int(prow["source_id"])
        star = stars.loc[stars["source_id"].astype("int64") == sid]
        mag_bin = str(star.iloc[0]["mag_bin"]) if not star.empty else ""
        tess_mag = float(star.iloc[0]["tess_mag"]) if not star.empty else float("nan")
        flux = float(prow["flux_fit"])
        flux_err = float(prow["flux_err"]) if "flux_err" in prow and np.isfinite(prow["flux_err"]) else np.nan
        base = {
            "stem": stem,
            "btjd": float(btjd),
            "source_id": sid,
            "tess_mag": tess_mag,
            "mag_bin": mag_bin,
            "x_init": float(prow["x_init"]),
            "y_init": float(prow["y_init"]),
        }
        if mode == "fixed":
            base.update(
                {
                    "flux_fixed": flux,
                    "flux_err_fixed": flux_err,
                    "ok_fixed": bool(np.isfinite(flux)),
                }
            )
        else:
            base.update(
                {
                    "flux_free": flux,
                    "flux_err_free": flux_err,
                    "x_fit_free": float(prow["x_fit"]),
                    "y_fit_free": float(prow["y_fit"]),
                    "ok_free": bool(np.isfinite(flux)),
                }
            )
        rows.append(base)
    return rows


def run_lc_baselines(
    cfg: StudyConfig,
    *,
    max_frames: int | None = None,
    force: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> Path:
    """Photometer selected stars on all fit-window frames (fixed + free xy)."""
    out_dir = cfg.study_dir / "lc_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "photometry_fixed_free.parquet"
    ckpt_path = out_dir / "photometry_fixed_free.checkpoint.parquet"

    if out_path.is_file() and not force:
        log.info("using existing %s", out_path)
        return out_path

    stars = select_stars_half_mag(cfg, force=force)
    wcs = load_wcs_artifacts(cfg.study_dir)
    cheb_static = load_cheb_static(cfg)
    epsf_model = _load_epsf_model(cfg)
    phot_cfg = StampQaConfig()

    n_frames = len(wcs.stems)
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    done_stems: set[str] = set()
    all_rows: list[dict] = []
    if ckpt_path.is_file() and not force:
        ckpt = pd.read_parquet(ckpt_path)
        all_rows = ckpt.to_dict(orient="records")
        done_stems = set(ckpt["stem"].astype(str))
        log.info("resuming from checkpoint: %d frames, %d rows", len(done_stems), len(all_rows))

    for fi in range(n_frames):
        stem = str(wcs.stems[fi])
        if stem in done_stems:
            continue
        btjd = float(wcs.btjd[fi])
        hp = resolve_hp_d(cfg.workspace, stem)
        if hp is None:
            log.warning("skip frame %s: no hp_d", stem)
            continue
        try:
            image = load_fits_image_data(Path(hp))
        except Exception as exc:
            log.warning("skip frame %s: load failed (%s)", stem, exc)
            continue

        x_init, y_init = predict_xy_warmstart(
            stars["ra"].to_numpy(dtype=float),
            stars["dec"].to_numpy(dtype=float),
            fi,
            wcs=wcs,
            cheb_static=cheb_static,
        )
        frame_stars = stars.copy()
        frame_stars["x_init"] = x_init
        frame_stars["y_init"] = y_init

        fixed_df, _ = run_gridded_photometry(
            image,
            epsf_model,
            frame_stars,
            cfg=phot_cfg,
            fix_xy=True,
            compute_stamp_chi2=False,
        )
        free_df, _ = run_gridded_photometry(
            image,
            epsf_model,
            frame_stars,
            cfg=phot_cfg,
            fix_xy=False,
            compute_stamp_chi2=False,
        )

        fixed_rows = {r["source_id"]: r for r in _phot_rows_from_table(
            fixed_df, mode="fixed", stem=stem, btjd=btjd, stars=frame_stars
        )}
        free_rows = {r["source_id"]: r for r in _phot_rows_from_table(
            free_df, mode="free", stem=stem, btjd=btjd, stars=frame_stars
        )}
        for sid in frame_stars["source_id"].astype("int64"):
            sid = int(sid)
            row = {"stem": stem, "btjd": btjd}
            row.update(fixed_rows.get(sid, {}))
            row.update(free_rows.get(sid, {}))
            if "source_id" not in row:
                star = frame_stars.loc[frame_stars["source_id"].astype("int64") == sid].iloc[0]
                row.update(
                    {
                        "source_id": sid,
                        "tess_mag": float(star["tess_mag"]),
                        "mag_bin": str(star["mag_bin"]),
                        "x_init": float(star["x_init"]),
                        "y_init": float(star["y_init"]),
                        "flux_fixed": np.nan,
                        "flux_err_fixed": np.nan,
                        "ok_fixed": False,
                        "flux_free": np.nan,
                        "flux_err_free": np.nan,
                        "x_fit_free": np.nan,
                        "y_fit_free": np.nan,
                        "ok_free": False,
                    }
                )
            all_rows.append(row)
        done_stems.add(stem)

        if (fi + 1) % checkpoint_every == 0 or fi == n_frames - 1:
            pd.DataFrame(all_rows).to_parquet(ckpt_path, index=False)
            log.info("checkpoint frame %d/%d (%d rows)", fi + 1, n_frames, len(all_rows))

    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(out_path, index=False)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    log.info("wrote %s (%d rows)", out_path, len(out_df))
    return out_path


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "dev/forward_epsf_wcs/init_study/orbit1_midhalf_mag813_center1k/config.yaml",
    )
    p.add_argument("--max-frames", type=int, default=None, help="smoke test: only first N frames")
    p.add_argument("--force", action="store_true", help="rebuild selection and photometry")
    p.add_argument("--select-only", action="store_true", help="only write star_selection.csv")
    args = p.parse_args(argv)
    cfg = _load_config(args.config.resolve())

    if args.select_only:
        sel = select_stars_half_mag(cfg, force=args.force)
        print(json.dumps({"ok": True, "n_stars": len(sel)}, indent=2))
        return

    out = run_lc_baselines(cfg, max_frames=args.max_frames, force=args.force)
    print(json.dumps({"ok": True, "parquet": str(out)}, indent=2))


if __name__ == "__main__":
    main()
