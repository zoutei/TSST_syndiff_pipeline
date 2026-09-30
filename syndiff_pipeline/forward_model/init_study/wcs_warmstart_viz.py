# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""WCS warmstart residual QA for forward_epsf_wcs init studies."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Avoid JAX prealloc / multi-process fights with concurrent ePSF loky workers.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model import cheb_wcs as CW
from syndiff_pipeline.forward_model import fit as FIT
from syndiff_pipeline.forward_model import gaia_pm as GP
from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model.data import RegionSpec, list_orbit_frames, select_middle_frames
from syndiff_pipeline.forward_model.init_study.build_epsf_prior import StudyConfig, _load_config
from syndiff_pipeline.forward_model.shared_wcs import fit_region_shared_wcs

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_gaia_catalog, load_merged_stars, select_qc_stars  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class WarmstartResult:
    per_frame_coeffs: pd.DataFrame
    frame_summary: pd.DataFrame
    star_panels: dict[str, pd.DataFrame]
    wcs_coeff: np.ndarray
    cheb_static: CW.ChebWcsStatic
    wcs_tb: T.TemporalBasis
    centroids_cache: dict[str, np.ndarray]
    panel_stems: list[str]


def _select_fit_frames(cfg: StudyConfig):
    frames_all, (btjd0, btjd1, orbit_num) = list_orbit_frames(
        cfg.workspace,
        sector=cfg.sector,
        orbit_index=cfg.orbit_index,
    )
    raw = yaml.safe_load(cfg.study_dir.joinpath("config.yaml").read_text(encoding="utf-8"))
    fw = raw["fit_window"]
    n_raw = fw.get("n_frames")
    offset = str(fw.get("frame_offset", "middle"))
    if n_raw in (None, "all", "full"):
        frames = list(frames_all)
    else:
        n_frames = int(n_raw)
        if offset == "middle":
            frames = select_middle_frames(frames_all, n_frames)
        else:
            frames = frames_all[:n_frames]
    return frames_all, frames, (btjd0, btjd1, orbit_num)


def run_wcs_warmstart_qa(cfg: StudyConfig) -> WarmstartResult:
    raw = yaml.safe_load(cfg.study_dir.joinpath("config.yaml").read_text(encoding="utf-8"))
    wcs_cfg = raw["wcs"]
    cheb_degree = int(wcs_cfg["cheb_degree"])
    wcs_n_interior = int(wcs_cfg["wcs_n_interior_knots"])
    edge_densify = bool(wcs_cfg.get("edge_densify_knots", True))
    knots_anchor = str(wcs_cfg.get("knots_anchor", "full-orbit"))
    edge_frac = float(wcs_cfg.get("edge_frac", 0.12))
    wcs_edge_split_raw = wcs_cfg.get("wcs_edge_interior_split")
    wcs_edge_split: tuple[int, int, int] | None = None
    if wcs_edge_split_raw is not None:
        parts = [int(v) for v in wcs_edge_split_raw]
        if len(parts) != 3 or any(p < 1 for p in parts):
            raise ValueError("wcs.wcs_edge_interior_split must be three positive integers")
        if sum(parts) != wcs_n_interior:
            raise ValueError(
                f"wcs.wcs_edge_interior_split {parts} must sum to wcs_n_interior_knots={wcs_n_interior}"
            )
        wcs_edge_split = (parts[0], parts[1], parts[2])
    margin = float(cfg.region_margin_px)

    frames_all, frames, (btjd0, btjd1, orbit_num) = _select_fit_frames(cfg)
    btjd = np.array([f.btjd for f in frames], dtype=float)
    log.info(
        "orbit_index=%d (1-based): %d/%d frames, btjd [%.3f, %.3f]",
        cfg.orbit_index,
        len(frames),
        len(frames_all),
        float(btjd[0]),
        float(btjd[-1]),
    )

    gaia_full = load_gaia_catalog(cfg.workspace)
    target_btjd = float(np.median(btjd))
    gaia_full, pm_stats = GP.apply_pm_to_dataframe(gaia_full, target_btjd)
    log.info("Gaia PM applied=%s", pm_stats.get("applied"))

    wcs, _region_qc = fit_region_shared_wcs(frames[0], gaia_full, cfg.region, margin_px=margin)
    cheb_static = CW.ChebWcsStatic.from_wcs(wcs, cfg.region, poly_degree=cheb_degree)

    btjd_full = np.array([f.btjd for f in frames_all], dtype=float)
    if knots_anchor == "full-orbit":
        # orbit_fraction assumes fit frames are an orbit *prefix* (tau from 0)
        # truncated at tau_cut. Middle-half windows sit at tau≈[0.25,0.75] and
        # get clipped to tau_cut≈0.5 → flat B-spline for the second half.
        tau_fit = (btjd - float(btjd_full[0])) / float(btjd_full[-1] - btjd_full[0])
        if float(tau_fit[0]) > 1e-3:
            log.warning(
                "knots_anchor=full-orbit but fit window starts at orbit "
                "fraction tau=%.3f (not a prefix); falling back to fit-window "
                "uniform knots to avoid tau_cut flatline",
                float(tau_fit[0]),
            )
            wcs_tb = T.build_temporal_basis(
                btjd,
                n_interior=wcs_n_interior,
                uniform_knots=True,
            )
        else:
            wcs_tb = T.build_temporal_basis_orbit_fraction(
                btjd,
                btjd_full,
                n_interior=wcs_n_interior,
                edge_frac=edge_frac,
                edge_interior_split=wcs_edge_split,
                tau_cut=float(len(frames) / len(frames_all)),
            )
    else:
        wcs_tb = T.build_temporal_basis(
            btjd,
            n_interior=wcs_n_interior,
            uniform_knots=not edge_densify,
        )

    from syndiff_pipeline.forward_model.data import preload_merged_stars

    centroid_src = str(raw.get("centroids", {}).get("source", "centroids_r1"))
    if centroid_src == "epsf":
        from syndiff_pipeline.forward_model.init_study.epsf_centroids_export import (
            merged_by_stem_from_parquet,
        )

        parquet = cfg.study_dir / "epsf_centroids" / "photometry.parquet"
        if not parquet.is_file():
            raise FileNotFoundError(
                f"centroids.source=epsf but missing {parquet}; "
                "run epsf_centroids_export first"
            )
        log.info("loading ePSF centroids from %s...", parquet)
        merged_by_stem = merged_by_stem_from_parquet(parquet)
        log.info("  %d frames in centroid table", len(merged_by_stem))
    else:
        log.info("preloading centroids_r1 (n_workers=1)...")
        merged_by_stem = preload_merged_stars(frames, gaia_full, n_workers=1)
    log.info("warmstart_wcs_coeff (%d frames)...", len(frames))
    wcs_coeff = np.asarray(
        FIT.warmstart_wcs_coeff(
            frames,
            gaia_full,
            cheb_static,
            np.asarray(wcs_tb.frame_basis),
            merged_by_stem=merged_by_stem,
        ),
        dtype=np.float64,
    )
    log.info("warmstart done; evaluating residuals...")

    n_terms = cheb_static.n_terms
    frame_basis_np = np.asarray(wcs_tb.frame_basis, dtype=np.float64)
    coeff_rows: list[dict] = []
    frame_rows: list[dict] = []
    panels: dict[str, pd.DataFrame] = {}
    pick = [0, len(frames) // 2, len(frames) - 1]
    panel_stems = [frames[i].stem for i in pick]

    # Packed QC-centroid cache so notebooks need not re-read centroids_r1.
    cache_frame_index: list[int] = []
    cache_btjd: list[float] = []
    cache_sid: list[np.int64] = []
    cache_ra: list[float] = []
    cache_dec: list[float] = []
    cache_x_obs: list[float] = []
    cache_y_obs: list[float] = []
    cache_x_lin: list[float] = []
    cache_y_lin: list[float] = []
    cache_dx_single: list[float] = []
    cache_dy_single: list[float] = []
    cache_dx_temp: list[float] = []
    cache_dy_temp: list[float] = []

    from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.cheb_poly_fit import cheb_design_matrix  # noqa: E402

    for fi, frame in enumerate(frames):
        if fi % 50 == 0:
            log.info("  residual frame %d/%d", fi, len(frames))
        if centroid_src == "epsf":
            if frame.stem not in merged_by_stem:
                log.warning("skip frame %s: no ePSF centroids", frame.stem)
                continue
            merged = merged_by_stem[frame.stem]
        elif frame.stem in merged_by_stem:
            merged = merged_by_stem[frame.stem]
        else:
            merged = load_merged_stars(frame, gaia_full)
        qc = select_qc_stars(merged) if centroid_src != "epsf" else merged.loc[
            np.isfinite(merged["x_fit"]) & np.isfinite(merged["y_fit"])
        ].copy()
        ra = qc["ra"].to_numpy(dtype=float)
        dec = qc["dec"].to_numpy(dtype=float)
        x_lin, y_lin = map(np.asarray, CW.linear_predict(ra, dec, cheb_static))
        x_obs = qc["x_fit"].to_numpy(dtype=float)
        y_obs = qc["y_fit"].to_numpy(dtype=float)
        sid = (
            qc["source_id"].to_numpy(dtype=np.int64)
            if "source_id" in qc.columns
            else np.full(len(qc), -1, dtype=np.int64)
        )

        cx, cy, mask = CW.fit_frame_cheb_warmstart(x_lin, y_lin, x_obs, y_obs, cheb_static)
        row = {"stem": frame.stem, "btjd": frame.btjd, "frame_index": fi, "n_qc": int(mask.sum())}
        for ti, val in enumerate(cx):
            row[f"cx_{ti}"] = float(val)
        for ti, val in enumerate(cy):
            row[f"cy_{ti}"] = float(val)
        coeff_rows.append(row)

        xhat = (x_lin - cheb_static.center[0]) / cheb_static.half_extents[0]
        yhat = (y_lin - cheb_static.center[1]) / cheb_static.half_extents[1]
        basis = cheb_design_matrix(xhat, yhat, cheb_static.poly_degree)
        frame_coeff = wcs_coeff @ frame_basis_np[fi]
        x_pred = x_lin + basis @ frame_coeff[:n_terms]
        y_pred = y_lin + basis @ frame_coeff[n_terms:]
        dx_single = x_obs - (x_lin + basis @ cx)
        dy_single = y_obs - (y_lin + basis @ cy)
        dx_temp = x_obs - x_pred
        dy_temp = y_obs - y_pred

        m = np.asarray(mask, dtype=bool)
        n_m = int(m.sum())
        cache_frame_index.append(np.full(n_m, fi, dtype=np.int32))
        cache_btjd.append(np.full(n_m, float(frame.btjd), dtype=np.float64))
        cache_sid.append(sid[m])
        cache_ra.append(ra[m])
        cache_dec.append(dec[m])
        cache_x_obs.append(x_obs[m])
        cache_y_obs.append(y_obs[m])
        cache_x_lin.append(x_lin[m])
        cache_y_lin.append(y_lin[m])
        cache_dx_single.append(dx_single[m])
        cache_dy_single.append(dy_single[m])
        cache_dx_temp.append(dx_temp[m])
        cache_dy_temp.append(dy_temp[m])

        frame_rows.append(
            {
                "stem": frame.stem,
                "btjd": frame.btjd,
                "med_abs_dx_single": float(np.median(np.abs(dx_single[m]))),
                "med_abs_dy_single": float(np.median(np.abs(dy_single[m]))),
                "med_abs_dx_temporal": float(np.median(np.abs(dx_temp[m]))),
                "med_abs_dy_temporal": float(np.median(np.abs(dy_temp[m]))),
                "n_qc": int(m.sum()),
            }
        )

        if fi in pick:
            panels[frame.stem] = pd.DataFrame(
                {
                    "source_id": sid[m],
                    "x_obs": x_obs[m],
                    "y_obs": y_obs[m],
                    "x_lin": x_lin[m],
                    "y_lin": y_lin[m],
                    "dx_single": dx_single[m],
                    "dy_single": dy_single[m],
                    "dx_temporal": dx_temp[m],
                    "dy_temporal": dy_temp[m],
                    "r_single": np.hypot(dx_single[m], dy_single[m]),
                    "r_temporal": np.hypot(dx_temp[m], dy_temp[m]),
                }
            )

    centroids_cache = {
        "stems": np.asarray([f.stem for f in frames], dtype=object),
        "btjd": np.asarray([f.btjd for f in frames], dtype=np.float64),
        "frame_index": np.concatenate(cache_frame_index),
        "star_btjd": np.concatenate(cache_btjd),
        "source_id": np.concatenate(cache_sid),
        "ra": np.concatenate(cache_ra),
        "dec": np.concatenate(cache_dec),
        "x_obs": np.concatenate(cache_x_obs),
        "y_obs": np.concatenate(cache_y_obs),
        "x_lin": np.concatenate(cache_x_lin),
        "y_lin": np.concatenate(cache_y_lin),
        "dx_single": np.concatenate(cache_dx_single),
        "dy_single": np.concatenate(cache_dy_single),
        "dx_temporal": np.concatenate(cache_dx_temp),
        "dy_temporal": np.concatenate(cache_dy_temp),
        "panel_stems": np.asarray(panel_stems, dtype=object),
        "wcs_frame_basis": frame_basis_np.astype(np.float32),
        "cheb_center": np.asarray(cheb_static.center, dtype=np.float64),
        "cheb_half_extents": np.asarray(cheb_static.half_extents, dtype=np.float64),
        "cheb_poly_degree": np.asarray(cheb_static.poly_degree, dtype=np.int32),
        "region": np.asarray(
            [cfg.region.x_min, cfg.region.y_min, cfg.region.x_max, cfg.region.y_max],
            dtype=np.int32,
        ),
    }

    per_frame_coeffs = pd.DataFrame(coeff_rows)
    frame_summary = pd.DataFrame(frame_rows)
    return WarmstartResult(
        per_frame_coeffs=per_frame_coeffs,
        frame_summary=frame_summary,
        star_panels=panels,
        wcs_coeff=wcs_coeff,
        cheb_static=cheb_static,
        wcs_tb=wcs_tb,
        centroids_cache=centroids_cache,
        panel_stems=panel_stems,
    )


def _plot_residual_timeseries(
    frame_summary: pd.DataFrame,
    out_path: Path,
    *,
    cheb_degree: int,
) -> None:
    sm = frame_summary.sort_values("btjd")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sm["btjd"], sm["med_abs_dx_single"], label="per-FFI |dx|", alpha=0.8)
    ax.plot(sm["btjd"], sm["med_abs_dy_single"], label="per-FFI |dy|", alpha=0.8)
    ax.plot(sm["btjd"], sm["med_abs_dx_temporal"], label="temporal |dx|", lw=2)
    ax.plot(sm["btjd"], sm["med_abs_dy_temporal"], label="temporal |dy|", lw=2)
    ax.set_xlabel("BTJD")
    ax.set_ylabel("median |residual| [px]")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"WCS warmstart residuals (Cheb d={int(cheb_degree)} + B-spline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_star_panel(df: pd.DataFrame, title: str, out_path: Path, *, ylim: float = 0.15) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, col_x, col_y, label in (
        (axes[0], "dx_single", "dy_single", "per-FFI Cheb"),
        (axes[1], "dx_temporal", "dy_temporal", "temporal B-spline"),
    ):
        ax.scatter(df[col_x], df[col_y], s=6, alpha=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-ylim, ylim)
        ax.set_ylim(-ylim, ylim)
        ax.set_xlabel("dx [px]")
        ax.set_ylabel("dy [px]")
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_spatial_residual_maps(
    panels: dict[str, pd.DataFrame],
    panel_stems: list[str],
    out_path: Path,
    *,
    region: list[int] | None = None,
    vmax: float = 0.08,
) -> None:
    """2D CCD maps: each star colored by residual magnitude (first/mid/last)."""
    labels = ("first", "mid", "last")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for col, stem in enumerate(panel_stems):
        df = panels[stem]
        for row, (rcol, title) in enumerate(
            (("r_single", "per-FFI Cheb"), ("r_temporal", "temporal B-spline"))
        ):
            ax = axes[row, col]
            sc = ax.scatter(
                df["x_obs"],
                df["y_obs"],
                c=df[rcol],
                s=6,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_aspect("equal")
            if region is not None:
                ax.set_xlim(region[0], region[2])
                ax.set_ylim(region[1], region[3])
            ax.set_title(f"{labels[col]} — {title}\n{stem}")
            if row == 1:
                ax.set_xlabel("x [px]")
            if col == 0:
                ax.set_ylabel("y [px]")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="|resid| [px]")
    fig.suptitle("Spatial WCS residuals (color = hypot(dx, dy))")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_warmstart_outputs(cfg: StudyConfig, result: WarmstartResult) -> Path:
    out_dir = cfg.study_dir / "wcs_warmstart"
    panel_dir = out_dir / "residual_panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    result.per_frame_coeffs.to_csv(out_dir / "per_frame_coeffs.csv", index=False)
    result.frame_summary.to_csv(out_dir / "warmstart_summary.csv", index=False)

    np.savez_compressed(
        out_dir / "wcs_coeff.npz",
        wcs_coeff=result.wcs_coeff.astype(np.float32),
        wcs_frame_basis=np.asarray(result.wcs_tb.frame_basis, dtype=np.float32),
        btjd=result.frame_summary["btjd"].to_numpy(dtype=np.float64),
        stems=result.frame_summary["stem"].to_numpy(dtype=object),
        cheb_poly_degree=np.int32(result.cheb_static.poly_degree),
        cheb_center=np.asarray(result.cheb_static.center, dtype=np.float64),
        cheb_half_extents=np.asarray(result.cheb_static.half_extents, dtype=np.float64),
        n_terms=np.int32(result.cheb_static.n_terms),
    )
    np.savez_compressed(out_dir / "centroids_cache.npz", **result.centroids_cache)

    labels = ("first", "mid", "last")
    for lab, stem in zip(labels, result.panel_stems):
        df = result.star_panels[stem]
        df.to_parquet(panel_dir / f"panel_{lab}.parquet", index=False)
        _plot_star_panel(df, f"{lab}: {stem}", panel_dir / f"{lab}_dxdy.png")

    region = [
        cfg.region.x_min,
        cfg.region.y_min,
        cfg.region.x_max,
        cfg.region.y_max,
    ]
    _plot_residual_timeseries(
        result.frame_summary,
        out_dir / "residual_vs_btjd.png",
        cheb_degree=int(result.cheb_static.poly_degree),
    )
    _plot_spatial_residual_maps(
        result.star_panels,
        result.panel_stems,
        out_dir / "spatial_residuals_first_mid_last.png",
        region=region,
    )

    meta = {
        "orbit_index_1based": cfg.orbit_index,
        "n_frames": int(len(result.frame_summary)),
        "cheb_degree": int(result.cheb_static.poly_degree),
        "n_terms": int(result.cheb_static.n_terms),
        "wcs_n_basis": int(result.wcs_tb.n_basis),
        "panel_stems": list(result.panel_stems),
        "region": region,
        "med_abs_dx_temporal_median": float(result.frame_summary["med_abs_dx_temporal"].median()),
        "med_abs_dy_temporal_median": float(result.frame_summary["med_abs_dy_temporal"].median()),
        "centroids_cache": str(out_dir / "centroids_cache.npz"),
        "n_star_rows_cached": int(len(result.centroids_cache["frame_index"])),
    }
    (out_dir / "warmstart_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("wrote WCS warmstart QA to %s", out_dir)
    return out_dir


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_REPO / "dev/forward_epsf_wcs/init_study/orbit1_midhalf_mag813_center1k/config.yaml",
    )
    args = p.parse_args(argv)
    cfg = _load_config(args.config.resolve())
    result = run_wcs_warmstart_qa(cfg)
    out_dir = write_warmstart_outputs(cfg, result)
    print(json.dumps({"ok": True, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
