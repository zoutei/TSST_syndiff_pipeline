"""QA plots for shared-mask bits under ``diff/debug_plots/masks/``.

Includes TSST-style TNS location and asteroid track/interval overlays
(adapted from ``TSST_Syndiff/development/mask_tns`` and ``mask_astroids``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking import bits
from syndiff_pipeline.difference_imaging.masking.bits import epsf_reject_mask, full_mask_bool, hotpants_mask_bool
from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog

log = logging.getLogger(__name__)

# Full-chip axis limits matching TSST_Syndiff development debug plots.
_FFI_XLIM = (44, 2091)
_FFI_YLIM = (0, 2047)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_bit_png(path: Path, layer: np.ndarray, title: str, dpi: int = 120) -> None:
    try:
        plt = _pyplot()
    except ImportError:
        log.debug("matplotlib missing; skip mask plot %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(layer.astype(float), origin="lower", cmap="gray_r", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x (crop)")
    ax.set_ylabel("y (crop)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info("  mask plot → %s", path)


def plot_tns_locations(
    table: pd.DataFrame,
    *,
    out_path: Path | None = None,
    title: str | None = None,
    dpi: int = 150,
):
    """
    TNS SN mask centers + names (and radius circles) on the full CCD.

    Port of ``TSST_Syndiff/development/mask_tns/debug_plots.plot_sn_locations``.
    Expects full-FFI 0-based ``x``/``y`` (as in ``transient_fixed``).
    """
    plt = _pyplot()
    if table is None or table.empty:
        raise ValueError("TNS table is empty")
    if "x" not in table.columns or "y" not in table.columns:
        raise ValueError("TNS table needs x,y columns")

    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in table.iterrows():
        x, y = float(row["x"]), float(row["y"])
        r = float(row.get("radius_px", 2) or 2)
        ax.add_patch(
            plt.Circle((x, y), r, fill=False, color="C0", lw=1.0, alpha=0.8)
        )
        ax.plot(x, y, "o", ms=4, color="C0")
        sid = str(row.get("source_id", "?"))
        mag = row.get("mag_mask")
        if mag is not None and pd.notna(mag) and np.isfinite(float(mag)):
            label = f"{sid}\n{float(mag):.2f}"
        else:
            label = sid
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlim(*_FFI_XLIM)
    ax.set_ylim(*_FFI_YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("TESS column")
    ax.set_ylabel("TESS row")
    ax.set_title(title or "TNS events on CCD")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        log.info("  mask plot → %s", out_path)
    return fig


def plot_asteroid_tracks_by_epoch(
    tracks: pd.DataFrame,
    *,
    out_path: Path | None = None,
    title: str | None = None,
    time_col: str = "time_jd",
    x_col: str = "column",
    y_col: str = "row",
    id_col: str = "target_id",
    xy_are_one_based: bool = True,
    dpi: int = 150,
):
    """
    Asteroid paths on the CCD, colored by epoch (colorbar).

    Port of ``TSST_Syndiff/development/mask_astroids/debug_plots.plot_asteroid_tracks_by_epoch``.
    When *xy_are_one_based* is True (tess-ephem), subtract 1 for 0-based plotting.
    """
    plt = _pyplot()
    if tracks is None or tracks.empty:
        raise ValueError("tracks is empty")

    fig, ax = plt.subplots(figsize=(8, 7))
    t = tracks[time_col].to_numpy(dtype=float)
    t_rel = t - float(np.nanmin(t))
    x = tracks[x_col].to_numpy(dtype=float)
    y = tracks[y_col].to_numpy(dtype=float)
    if xy_are_one_based:
        x = x - 1.0
        y = y - 1.0

    sc = ax.scatter(
        x,
        y,
        c=t_rel,
        s=8,
        cmap="viridis",
        linewidths=0,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Days / cadence units since first on-chip epoch")

    if id_col in tracks.columns:
        for tid, g in tracks.groupby(id_col):
            g = g.sort_values(time_col)
            gx = g[x_col].to_numpy(dtype=float)
            gy = g[y_col].to_numpy(dtype=float)
            if xy_are_one_based:
                gx = gx - 1.0
                gy = gy - 1.0
            ax.plot(gx, gy, "-", color="0.35", lw=0.6, alpha=0.5, zorder=1)
            mid_i = len(g) // 2
            ax.annotate(
                str(tid),
                (gx[mid_i], gy[mid_i]),
                fontsize=8,
                color="k",
                ha="left",
                va="bottom",
            )

    ax.set_xlim(*_FFI_XLIM)
    ax.set_ylim(*_FFI_YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("TESS column")
    ax.set_ylabel("TESS row")
    ax.set_title(title or "Asteroid tracks (color = epoch)")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        log.info("  mask plot → %s", out_path)
    return fig


def _intervals_to_track_like(
    intervals: pd.DataFrame,
    *,
    crop_bounds: dict | None,
) -> pd.DataFrame:
    """
    Collapse crop-local pixel intervals into a track-like table for plotting.

    Uses mean (x,y) per (target_id, cadence_lo) and mid-cadence as the epoch.
    Converts to full-FFI 0-based pixels when *crop_bounds* is set.
    """
    iv = intervals.copy()
    if "x" in iv.columns and "y" in iv.columns:
        x = iv["x"].to_numpy(float)
        y = iv["y"].to_numpy(float)
        if crop_bounds is not None:
            x = x + float(crop_bounds["x_min"])
            y = y + float(crop_bounds["y_min"])
    elif "col" in iv.columns and "row" in iv.columns:
        # 1-based full-FFI → 0-based
        x = iv["col"].to_numpy(float) - 1.0
        y = iv["row"].to_numpy(float) - 1.0
    else:
        raise ValueError("intervals need x/y or col/row")

    iv = iv.assign(_x=x, _y=y)
    if "target_id" not in iv.columns:
        iv["target_id"] = "ast"
    gcols = ["target_id", "cadence_lo"]
    if "cadence_hi" not in iv.columns:
        iv["cadence_hi"] = iv["cadence_lo"]
    agg = (
        iv.groupby(gcols, as_index=False)
        .agg(_x=("_x", "mean"), _y=("_y", "mean"), cadence_hi=("cadence_hi", "first"))
    )
    agg["epoch"] = 0.5 * (agg["cadence_lo"].to_numpy(float) + agg["cadence_hi"].to_numpy(float))
    return agg.rename(columns={"_x": "x", "_y": "y"})


def write_tns_asteroid_overlay_plots(
    catalog: MaskCatalog,
    plot_dir: str | Path,
    *,
    tracks: pd.DataFrame | None = None,
    dpi: int = 150,
) -> list[Path]:
    """
    Write TSST-style TNS location + asteroid track overlays under *plot_dir*.

    - ``tns_locations.png`` from ``catalog.tns_table``
    - ``asteroid_tracks_by_epoch.png`` from *tracks* (tess-ephem style) when given,
      else from crop-local ``catalog.asteroid_intervals`` collapsed to track-like points
    """
    plt = _pyplot()
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if catalog.tns_table is not None and not catalog.tns_table.empty:
        out = plot_dir / "tns_locations.png"
        fig = plot_tns_locations(
            catalog.tns_table,
            out_path=out,
            title="TNS events on CCD (mask bit 64)",
            dpi=dpi,
        )
        plt.close(fig)
        written.append(out)
    else:
        log.info("  skip tns_locations.png (no tns_table)")

    if tracks is not None and not tracks.empty:
        out = plot_dir / "asteroid_tracks_by_epoch.png"
        # Prefer tess-ephem column names when present
        time_col = "time_jd" if "time_jd" in tracks.columns else (
            "btjd" if "btjd" in tracks.columns else None
        )
        x_col = "column" if "column" in tracks.columns else (
            "col" if "col" in tracks.columns else None
        )
        y_col = "row" if "row" in tracks.columns else None
        if time_col and x_col and y_col:
            one_based = x_col in ("column", "col") and "x" not in tracks.columns
            fig = plot_asteroid_tracks_by_epoch(
                tracks,
                out_path=out,
                title="Asteroid tracks (color = epoch)",
                time_col=time_col,
                x_col=x_col,
                y_col=y_col,
                xy_are_one_based=one_based,
                dpi=dpi,
            )
            plt.close(fig)
            written.append(out)
        else:
            log.warning("  tracks missing time/x/y columns; skip asteroid track plot")
    elif catalog.has_temporal():
        track_like = _intervals_to_track_like(
            catalog.asteroid_intervals,  # type: ignore[arg-type]
            crop_bounds=catalog.crop_bounds,
        )
        out = plot_dir / "asteroid_tracks_by_epoch.png"
        fig = plot_asteroid_tracks_by_epoch(
            track_like,
            out_path=out,
            title="Asteroid intervals (color = cadence; from pixel_intervals)",
            time_col="epoch",
            x_col="x",
            y_col="y",
            id_col="target_id",
            xy_are_one_based=False,
            dpi=dpi,
        )
        plt.close(fig)
        written.append(out)
    else:
        log.info("  skip asteroid_tracks_by_epoch.png (no intervals/tracks)")

    return written


def write_mask_debug_plots(
    catalog: MaskCatalog,
    plot_dir: str | Path,
    *,
    sample_cadence: int | None = None,
    dpi: int = 120,
    tracks: pd.DataFrame | None = None,
) -> None:
    """Write bit-plane, consumer-predicate, and TNS/asteroid overlay PNGs."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    static = catalog.static

    planes = [
        (bits.BRIGHT_CAT, "bit1_very_bright.png", "bit 1 very bright (T<7.5 + BSC)"),
        (bits.SAT_CROSS, "bit2_mid_bright.png", "bit 2 mid bright (7.5≤T<13)"),
        (bits.FAINT_CAT, "bit32_faint_cat.png", "bit 32 FAINT_CAT"),
        (bits.TNS, "bit64_tns.png", "bit 64 TNS"),
    ]
    for bit, name, title in planes:
        _save_bit_png(plot_dir / name, (static.astype(np.int64) & bit) != 0, title, dpi=dpi)

    _save_bit_png(
        plot_dir / "epsf_reject_mask.png",
        epsf_reject_mask(static),
        "ePSF reject (ignore bits 1|2|32)",
        dpi=dpi,
    )

    cad = sample_cadence
    if cad is None and catalog.asteroid_times is not None and not catalog.asteroid_times.empty:
        cad = int(catalog.asteroid_times["cadence"].iloc[len(catalog.asteroid_times) // 2])
    if cad is not None and catalog.has_temporal():
        temporal = catalog.mask_at(cad, which="temporal")
        _save_bit_png(
            plot_dir / "bit128_asteroid_sample.png",
            (temporal.astype(np.int64) & bits.ASTEROID) != 0,
            f"bit 128 ASTEROID cadence={cad}",
            dpi=dpi,
        )
        full = catalog.mask_at(cad, which="full")
        _save_bit_png(
            plot_dir / "hotpants_full_sample.png",
            hotpants_mask_bool(full),
            f"Hotpants mask (ignore bit 32) cadence={cad}",
            dpi=dpi,
        )
    else:
        _save_bit_png(
            plot_dir / "hotpants_full_sample.png",
            hotpants_mask_bool(static),
            "Hotpants mask (ignore bit 32; static)",
            dpi=dpi,
        )

    try:
        write_tns_asteroid_overlay_plots(
            catalog, plot_dir, tracks=tracks, dpi=max(dpi, 150)
        )
    except Exception as exc:
        log.warning("TNS/asteroid overlay plots failed: %s", exc)
