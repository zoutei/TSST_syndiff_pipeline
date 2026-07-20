"""
Debug figures for the per-skycell PS1 shift schedule (remap L2).

Written next to ``shift_schedule.npz`` under
``{data_root}/s…/c…/k…/remap/oversampling_{N}/``:

- ``skycell_shift_grid_debug.png`` — 3×3 FoV grid, SG + quantized absolute dx/dy
- ``skycell_shift_relative_center_debug.png`` — same grid; off-center = skycell −
  FoV-center SG; center panel absolute (no quantized)

Layout: tess_x=0, tess_y=0 at bottom-left. X-axis: BTJD. Orbit segments are
plotted as disconnected series (same colors). Empty-WCS frames are gray guides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec

from syndiff_pipeline.template_creation.processing.shift_schedule import (
    FRAME_ORIGIN_MEASURED,
    FRAME_ORIGIN_SYNTH_MISSING_WCS,
    FRAME_ORIGIN_SYNTH_SIGMA_CLIPPED,
    ShiftSchedule,
)

log = logging.getLogger(__name__)

SKYCELL_SHIFT_GRID_DEBUG_FILENAME = "skycell_shift_grid_debug.png"
SKYCELL_SHIFT_RELATIVE_CENTER_DEBUG_FILENAME = "skycell_shift_relative_center_debug.png"

_GRID_FRACS = (0.15, 0.50, 0.85)
_N_GRID = 3
_COLOR_SG = "#1f77b4"
_COLOR_Q = "#ff7f0e"


def _skycell_centroids_on_master(
    master: np.ndarray, names: np.ndarray, name_to_master_id: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    n_id = int(master.max()) + 1
    flat = master.ravel()
    yy = np.repeat(np.arange(master.shape[0]), master.shape[1])
    xx = np.tile(np.arange(master.shape[1]), master.shape[0])
    w = flat >= 0
    cnt = np.bincount(flat[w], minlength=n_id).astype(np.float64)
    sx_c = np.bincount(flat[w], weights=xx[w].astype(np.float64), minlength=n_id)
    sy_c = np.bincount(flat[w], weights=yy[w].astype(np.float64), minlength=n_id)
    centers = np.column_stack([sx_c / np.maximum(cnt, 1), sy_c / np.maximum(cnt, 1)])
    cx = np.full(len(names), np.nan, dtype=np.float64)
    cy = np.full(len(names), np.nan, dtype=np.float64)
    for j, nm in enumerate(names):
        mid = name_to_master_id.get(str(nm))
        if mid is not None and mid < len(centers):
            cx[j], cy[j] = centers[mid]
    return cx, cy


def _pick_grid_skycells(
    cx: np.ndarray, cy: np.ndarray, names: np.ndarray, fracs=_GRID_FRACS
) -> list[dict[str, Any]]:
    """Nearest skycell to each 3×3 FoV target; plot_row flips so low tess_y is bottom."""
    ok = np.isfinite(cx) & np.isfinite(cy)
    if not ok.any():
        raise RuntimeError("no skycell centroids on master for shift debug plot")
    x_min, x_max = float(np.nanmin(cx[ok])), float(np.nanmax(cx[ok]))
    y_min, y_max = float(np.nanmin(cy[ok])), float(np.nanmax(cy[ok]))
    picks: list[dict[str, Any]] = []
    used: set[int] = set()
    n = len(fracs)
    for yi, fy in enumerate(fracs):
        for xi, fx in enumerate(fracs):
            tx = x_min + fx * (x_max - x_min)
            ty = y_min + fy * (y_max - y_min)
            d = np.hypot(cx - tx, cy - ty)
            d[~ok] = np.inf
            for j in np.argsort(d):
                ji = int(j)
                if ji in used:
                    continue
                used.add(ji)
                picks.append(
                    {
                        "plot_row": (n - 1) - yi,
                        "plot_col": xi,
                        "skycell": str(names[ji]),
                        "col_idx": ji,
                        "is_center": (yi == n // 2 and xi == n // 2),
                    }
                )
                break
    return picks


def _orbit_segments(n_frames: int, orbit_bounds: list) -> list[tuple[int, int]]:
    if not orbit_bounds:
        return [(0, n_frames)]
    segs: list[tuple[int, int]] = []
    for a, b in orbit_bounds:
        a_i, b_i = int(a), int(b)
        if b_i > a_i:
            segs.append((a_i, min(b_i, n_frames)))
    return segs or [(0, n_frames)]


def _plot_by_orbit(
    ax,
    btjd: np.ndarray,
    y: np.ndarray,
    orbit_bounds: list,
    *,
    color: str,
    lw: float,
    label: str | None = None,
    drawstyle: str | None = None,
) -> None:
    segs = _orbit_segments(len(btjd), orbit_bounds)
    for i, (a, b) in enumerate(segs):
        kw: dict[str, Any] = {"color": color, "lw": lw}
        if drawstyle is not None:
            kw["drawstyle"] = drawstyle
        if label is not None and i == 0:
            kw["label"] = label
        ax.plot(btjd[a:b], y[a:b], **kw)


def _empty_wcs_guides(
    ax, btjd: np.ndarray, empty_idx: np.ndarray, clip_idx: np.ndarray
) -> None:
    for f in empty_idx:
        if 0 <= f < len(btjd):
            ax.axvline(btjd[f], color="0.6", lw=0.4, alpha=0.5, zorder=0)
    for f in clip_idx:
        if 0 <= f < len(btjd):
            ax.axvline(btjd[f], color="crimson", lw=0.5, alpha=0.6, zorder=0)


def _plot_panel_pair(
    fig,
    outer: GridSpec,
    plot_row: int,
    plot_col: int,
    btjd: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    title: str,
    ylabel_dx: str,
    ylabel_dy: str,
    empty_idx: np.ndarray,
    clip_idx: np.ndarray,
    orbit_bounds: list,
    show_quantized: bool = False,
    dx_q: np.ndarray | None = None,
    dy_q: np.ndarray | None = None,
    legend: bool = False,
) -> None:
    inner = outer[plot_row, plot_col].subgridspec(2, 1, hspace=0.08)
    ax_dx = fig.add_subplot(inner[0])
    ax_dy = fig.add_subplot(inner[1], sharex=ax_dx)

    _plot_by_orbit(
        ax_dx, btjd, dx, orbit_bounds, color=_COLOR_SG, lw=1.0, label="SG (orbit-split)"
    )
    _plot_by_orbit(ax_dy, btjd, dy, orbit_bounds, color=_COLOR_SG, lw=1.0)
    if show_quantized and dx_q is not None and dy_q is not None:
        _plot_by_orbit(
            ax_dx,
            btjd,
            dx_q,
            orbit_bounds,
            color=_COLOR_Q,
            lw=0.9,
            label="quantized",
            drawstyle="steps-post",
        )
        _plot_by_orbit(
            ax_dy,
            btjd,
            dy_q,
            orbit_bounds,
            color=_COLOR_Q,
            lw=0.9,
            drawstyle="steps-post",
        )

    for ax in (ax_dx, ax_dy):
        _empty_wcs_guides(ax, btjd, empty_idx, clip_idx)

    ax_dx.set_ylabel(ylabel_dx, fontsize=7)
    ax_dy.set_ylabel(ylabel_dy, fontsize=7)
    ax_dy.set_xlabel("BTJD", fontsize=7)
    ax_dx.tick_params(labelbottom=False, labelsize=6)
    ax_dy.tick_params(labelsize=6)
    ax_dx.set_title(title, fontsize=8)
    if legend:
        ax_dx.legend(fontsize=6, loc="upper right", framealpha=0.85)


def _ensure_btjd(btjd: np.ndarray | None, n_frames: int) -> np.ndarray:
    if btjd is None:
        return np.arange(n_frames, dtype=np.float64)
    out = np.asarray(btjd, dtype=np.float64).reshape(-1)
    if out.shape[0] != n_frames:
        raise ValueError(
            f"btjd length ({out.shape[0]}) does not match schedule frames ({n_frames})"
        )
    if not np.isfinite(out).all():
        good = np.isfinite(out)
        if good.any():
            out = out.copy()
            out[~good] = np.interp(
                np.flatnonzero(~good), np.flatnonzero(good), out[good]
            )
        else:
            out = np.arange(n_frames, dtype=np.float64)
    return out


def write_skycell_shift_debug_plots(
    schedule: ShiftSchedule,
    *,
    out_dir: str | Path,
    btjd: np.ndarray | None,
    ref_wcs: WCS,
    skycell_df: pd.DataFrame,
    master_path: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    dpi: int = 140,
) -> dict[str, Path]:
    """
    Write the two skycell shift debug PNGs under ``out_dir``.

    Returns ``{"grid": Path, "relative_center": Path}``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from syndiff_pipeline.template_creation.processing.pancakes import (
        calculate_radec_center,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = Path(master_path)

    names = np.asarray(schedule.skycell_names).astype(str)
    n_frames = int(schedule.sx_float.shape[0])
    btjd_arr = _ensure_btjd(btjd, n_frames)

    origin = np.asarray(
        schedule.frame_origin
        if schedule.frame_origin is not None
        else np.zeros(n_frames, dtype=np.int8)
    )
    counts = schedule.meta.get("frame_origin_counts") or {
        "measured": int((origin == FRAME_ORIGIN_MEASURED).sum()),
        "synth_missing_wcs": int((origin == FRAME_ORIGIN_SYNTH_MISSING_WCS).sum()),
        "synth_sigma_clipped": int((origin == FRAME_ORIGIN_SYNTH_SIGMA_CLIPPED).sum()),
    }
    n_empty = int(counts.get("synth_missing_wcs", 0))
    n_measured = int(counts.get("measured", 0))
    orbit_bounds = list(schedule.meta.get("orbit_segment_bounds") or [])
    empty_idx = np.flatnonzero(origin == FRAME_ORIGIN_SYNTH_MISSING_WCS)
    clip_idx = np.flatnonzero(origin == FRAME_ORIGIN_SYNTH_SIGMA_CLIPPED)

    sk = skycell_df.copy()
    if "RA" not in sk.columns or "DEC" not in sk.columns:
        centers = calculate_radec_center(sk.reset_index(drop=True))
        sk["RA"] = centers[:, 0]
        sk["DEC"] = centers[:, 1]
    name_col = "NAME" if "NAME" in sk.columns else sk.columns[0]
    name_to_sk = {str(r[name_col]): i for i, r in sk.iterrows()}

    with fits.open(master_path) as hdul:
        master = np.asarray(hdul[1].data).astype(np.int32)
        tab = hdul[2].data
        name_to_master_id = {
            str(n).strip(): int(i) for n, i in zip(tab["SKYCELL"], tab["SKYCIND"])
        }
    cx, cy = _skycell_centroids_on_master(master, names, name_to_master_id)
    picks = _pick_grid_skycells(cx, cy, names)

    for p in picks:
        row = sk.iloc[name_to_sk[p["skycell"]]]
        tx, ty = ref_wcs.world_to_pixel_values(float(row["RA"]), float(row["DEC"]))
        p["tess_x"] = float(tx)
        p["tess_y"] = float(ty)

    center = next(p for p in picks if p["is_center"])
    ccol = center["col_idx"]
    cx_s = schedule.sx_float[:, ccol]
    cy_s = schedule.sy_float[:, ccol]
    ref_label = str(schedule.meta.get("reference_ffi") or "")
    ref_name = Path(ref_label).name if ref_label else "(reference)"
    guide = "gray = empty-WCS frames  |  origin BL (tess_x=0, tess_y=0)"
    scc = f"s{int(sector):04d}/c{int(camera)}/k{int(ccd)}"

    # Absolute SG + quantized
    fig1 = plt.figure(figsize=(14, 12))
    outer1 = GridSpec(_N_GRID, _N_GRID, figure=fig1, hspace=0.45, wspace=0.30)
    for p in picks:
        col = p["col_idx"]
        _plot_panel_pair(
            fig1,
            outer1,
            p["plot_row"],
            p["plot_col"],
            btjd_arr,
            schedule.sx_float[:, col],
            schedule.sy_float[:, col],
            title=f"{p['skycell']}\ntess_x={p['tess_x']:.1f}, tess_y={p['tess_y']:.1f}",
            ylabel_dx="dx (PS1 px)",
            ylabel_dy="dy (PS1 px)",
            empty_idx=empty_idx,
            clip_idx=clip_idx,
            orbit_bounds=orbit_bounds,
            show_quantized=True,
            dx_q=schedule.sx_int[:, col].astype(np.float64),
            dy_q=schedule.sy_int[:, col].astype(np.float64),
            legend=(p["plot_row"] == 0 and p["plot_col"] == 0),
        )
    fig1.suptitle(
        f"{scc} — skycell shift schedule (SG + quantized) vs reference\n"
        f"measured={n_measured}  synth_missing_wcs={n_empty}  "
        f"synth_sigma={counts.get('synth_sigma_clipped', 0)}  |  "
        f"ref: {ref_name}\n{guide}",
        fontsize=10,
        y=0.995,
    )
    out_grid = out_dir / SKYCELL_SHIFT_GRID_DEBUG_FILENAME
    fig1.savefig(out_grid, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)

    # Relative to FoV center
    fig2 = plt.figure(figsize=(14, 12))
    outer2 = GridSpec(_N_GRID, _N_GRID, figure=fig2, hspace=0.45, wspace=0.30)
    for p in picks:
        col = p["col_idx"]
        if p["is_center"]:
            dx = schedule.sx_float[:, col]
            dy = schedule.sy_float[:, col]
            ylab_dx, ylab_dy = "dx (PS1 px)", "dy (PS1 px)"
            tag = "CENTER (absolute)"
        else:
            dx = schedule.sx_float[:, col] - cx_s
            dy = schedule.sy_float[:, col] - cy_s
            ylab_dx, ylab_dy = "Δdx vs center", "Δdy vs center"
            tag = "− center"
        _plot_panel_pair(
            fig2,
            outer2,
            p["plot_row"],
            p["plot_col"],
            btjd_arr,
            dx,
            dy,
            title=(
                f"{p['skycell']}  [{tag}]\n"
                f"tess_x={p['tess_x']:.1f}, tess_y={p['tess_y']:.1f}"
            ),
            ylabel_dx=ylab_dx,
            ylabel_dy=ylab_dy,
            empty_idx=empty_idx,
            clip_idx=clip_idx,
            orbit_bounds=orbit_bounds,
            show_quantized=False,
            legend=bool(p["is_center"]),
        )
    fig2.suptitle(
        f"{scc} — skycell shifts relative to FoV center "
        f"({center['skycell']}; center panel = absolute)\n"
        f"off-center: Δ = skycell − center  |  SG only  |  "
        f"measured={n_measured}  synth_missing_wcs={n_empty}\n"
        f"ref: {ref_name}  |  {guide}",
        fontsize=10,
        y=0.995,
    )
    out_rel = out_dir / SKYCELL_SHIFT_RELATIVE_CENTER_DEBUG_FILENAME
    fig2.savefig(out_rel, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)

    log.info(
        "Skycell shift debug plots: %s , %s",
        out_grid.name,
        out_rel.name,
    )
    return {"grid": out_grid, "relative_center": out_rel}
