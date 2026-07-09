"""
Diagnostic plots when ``SynDiffConfig.pipeline_plots`` is True.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


def _safe_plot_token(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name))


def gridded_epsf_frame_plot_path(
    plot_dir: str,
    epsf_label: str,
    ffi_stem: str,
) -> str:
    """PNG path for one frame's gridded ePSF tile montage."""
    return os.path.join(
        plot_dir,
        f"epsf_{_safe_plot_token(epsf_label)}_{_safe_plot_token(ffi_stem)}.png",
    )


def gridded_epsf_group_plot_path(plot_dir: str, epsf_label: str, group_id: int) -> str:
    """PNG path for a template-group median gridded ePSF."""
    return os.path.join(
        plot_dir,
        f"epsf_{_safe_plot_token(epsf_label)}_group_{int(group_id)}.png",
    )


def lightcurve_plot_path_from_csv(
    plot_dir: str,
    lc_label: str,
    csv_basename: str,
) -> str:
    """PNG path mirroring a ``lightcurve_*.csv`` basename."""
    stem = os.path.splitext(csv_basename)[0]
    return os.path.join(plot_dir, f"{stem}.png")


def _normalize_stamp(stamp: np.ndarray) -> np.ndarray:
    """Peak-normalize a 2D ePSF stamp for display."""
    arr = np.asarray(stamp, dtype=np.float64)
    peak = float(np.nanmax(arr))
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(arr)
    out = arr / peak
    out[~np.isfinite(out)] = 0.0
    return out


def write_gridded_epsf_frame_plot(
    npz_path: str,
    png_path: str,
    *,
    dpi: int = 150,
    title: str = "",
    crop_shape: tuple[int, int] | None = None,
) -> Optional[str]:
    """
    Plot the gridded ePSF tiles from one per-frame ``*_gridded_epsf.npz``.

    Layout is a square grid of normalized stamp images with grid-node
  positions annotated.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning(
            "pipeline_plots: matplotlib is not installed; skipping ePSF frame plot."
        )
        return None

    z = np.load(npz_path, allow_pickle=False)
    try:
        stack = np.asarray(z["data"], dtype=np.float64)
        grid_xypos = np.asarray(z["grid_xypos"], dtype=np.float64)
        oversampling = int(z["oversampling"])
    finally:
        z.close()

    n_grid = stack.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_grid)))
    n_rows = int(np.ceil(n_grid / n_cols))

    fig = plt.figure(figsize=(2.2 * n_cols + 1.5, 2.2 * n_rows + 1.8), layout="constrained")
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1.0] * n_cols + [1.1])

    for k in range(n_grid):
        row, col = divmod(k, n_cols)
        ax = fig.add_subplot(gs[row, col])
        stamp = _normalize_stamp(stack[k])
        ax.imshow(stamp, origin="lower", cmap="viridis", interpolation="nearest")
        if k < len(grid_xypos):
            gx, gy = grid_xypos[k]
            ax.set_title(f"node {k}\n({gx:.0f}, {gy:.0f})", fontsize=8)
        else:
            ax.set_title(f"node {k}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    ax_map = fig.add_subplot(gs[:, -1])
    if crop_shape is not None:
        ny, nx = crop_shape
        ax_map.set_xlim(0, nx)
        ax_map.set_ylim(0, ny)
        ax_map.set_aspect("equal")
        ax_map.invert_yaxis()
    for k, (gx, gy) in enumerate(grid_xypos):
        ax_map.plot(gx, gy, "o", ms=8, color=f"C{k % 10}")
        ax_map.annotate(str(k), (gx, gy), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax_map.set_title("grid nodes\n(crop-local px)", fontsize=9)
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.grid(True, alpha=0.25)

    head = title or os.path.basename(npz_path)
    fig.suptitle(f"Gridded ePSF · {head} · oversample={oversampling}", fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    log.info("  pipeline_plots: ePSF frame figure %s", png_path)
    return png_path


def write_gridded_epsf_workspace_plots(
    epsf_workspace_dir: str,
    plot_dir: str,
    *,
    epsf_label: str = "epsf_r1",
    dpi: int = 150,
    max_frames: int = 3,
    crop_shape: tuple[int, int] | None = None,
) -> list[str]:
    """
    Write diagnostic PNGs for representative per-frame gridded ePSFs and any
    group-level median cubes under ``group_epsf/``.
    """
    from syndiff_pipeline.difference_imaging.stages import gridded_epsf

    written: list[str] = []
    index = gridded_epsf.load_gridded_epsf_index(epsf_workspace_dir)
    if not index:
        for npz_path in sorted(
            glob.glob(os.path.join(epsf_workspace_dir, f"*{gridded_epsf.GRIDDED_EPSF_NPZ_SUFFIX}"))
        ):
            stem = os.path.basename(npz_path).replace(gridded_epsf.GRIDDED_EPSF_NPZ_SUFFIX, "")
            if stem:
                index[stem] = npz_path
    if not index:
        log.warning(
            "pipeline_plots: no gridded ePSF index in %s; skip ePSF plots.",
            epsf_workspace_dir,
        )
        return written

    stems = sorted(index.keys())
    if max_frames > 0 and len(stems) > max_frames:
        pick_idx = np.linspace(0, len(stems) - 1, num=max_frames, dtype=int)
        stems = [stems[i] for i in np.unique(pick_idx)]

    os.makedirs(plot_dir, exist_ok=True)
    for stem in stems:
        npz_path = index.get(stem) or gridded_epsf.gridded_epsf_npz_path(
            epsf_workspace_dir, stem
        )
        if not os.path.isfile(npz_path):
            continue
        png_path = gridded_epsf_frame_plot_path(plot_dir, epsf_label, stem)
        out = write_gridded_epsf_frame_plot(
            npz_path,
            png_path,
            dpi=dpi,
            title=stem,
            crop_shape=crop_shape,
        )
        if out:
            written.append(out)

    group_dir = os.path.join(epsf_workspace_dir, "group_epsf")
    for npz_path in sorted(glob.glob(os.path.join(group_dir, "group_epsf_*.npz"))):
        gid = int(
            os.path.basename(npz_path)
            .replace("group_epsf_", "")
            .replace(".npz", "")
        )
        png_path = gridded_epsf_group_plot_path(plot_dir, epsf_label, gid)
        out = write_gridded_epsf_frame_plot(
            npz_path,
            png_path,
            dpi=dpi,
            title=f"group {gid} median",
            crop_shape=crop_shape,
        )
        if out:
            written.append(out)

    summary_path = os.path.join(plot_dir, f"epsf_{_safe_plot_token(epsf_label)}_index.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "epsf_workspace": epsf_workspace_dir,
                "frames_plotted": stems,
                "plot_paths": written,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    written.append(summary_path)
    return written


def write_lightcurve_diagnostics_from_workspace(
    lc_workspace_dir: str,
    plot_dir: str,
    *,
    lc_label: str = "lc",
    dpi: int = 150,
    flux_column: str = "flux",
) -> list[str]:
    """Plot every ``lightcurve_*.csv`` under a forced-photometry workspace."""
    import pandas as pd

    from syndiff_pipeline.difference_imaging.stages.photometry import (
        write_lightcurve_diagnostic_plot,
    )

    written: list[str] = []
    if not os.path.isdir(lc_workspace_dir):
        return written

    os.makedirs(plot_dir, exist_ok=True)
    for csv_path in sorted(glob.glob(os.path.join(lc_workspace_dir, "lightcurve_*.csv"))):
        base = os.path.basename(csv_path)
        png_path = lightcurve_plot_path_from_csv(plot_dir, lc_label, base)
        try:
            lc_df = pd.read_csv(csv_path)
        except Exception as exc:
            log.warning("pipeline_plots: cannot read %s: %s", csv_path, exc)
            continue
        title = f"{lc_label} · {base}"
        out = write_lightcurve_diagnostic_plot(
            lc_df,
            lc_workspace_dir,
            dpi=dpi,
            title_line=title,
            png_path=png_path,
            flux_column=flux_column,
        )
        if out:
            written.append(out)
    return written


def write_background_removal_animation(
    bkg_smooth_r1: np.ndarray,
    wcs_table,
    hotpants_results: List[dict],
    output_dir: str,
    *,
    dpi: int = 150,
    fps: float = 3.0,
    max_frames: int = 150,
    filename: str = "bkg_smooth_r1_removed_background.gif",
    cbar_label: str = "Estimated background",
) -> Optional[str]:
    """
    Write an animated GIF of a per-frame background cube (e.g. rough stack or
    adaptively smoothed ``bkg_smooth``).

    Uses a fixed colour scale (1–99 percentile over all animated frames) so
    temporal changes are visible.

    Parameters
    ----------
    bkg_smooth_r1 : ndarray, shape (n_frames, ny, nx)
        Background estimate per epoch (rough stack or temporally smoothed bkg).
    wcs_table : DataFrame with BTJD / path or filename (optional alignment)
    hotpants_results : list aligned with axis 0 (for BTJD via stems)
    output_dir : str
    dpi : int — passed to the GIF writer (affects pixel size)
    fps : float — frames per second for the GIF
    max_frames : int — if n_frames exceeds this, subsample evenly for file size
    filename : str — output basename under ``output_dir``
    cbar_label : str — colorbar label in the figure

    Returns
    -------
    str or None — path to GIF, or None if matplotlib/Pillow missing or no data
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        log.warning(
            "pipeline_plots: matplotlib not available; skip background removal animation."
        )
        return None

    if bkg_smooth_r1 is None or bkg_smooth_r1.size == 0:
        log.warning("pipeline_plots: no bkg_smooth_r1; skip background animation.")
        return None

    cube = np.asarray(bkg_smooth_r1, dtype=float)
    n = cube.shape[0]
    if n == 0:
        return None

    idx = np.arange(n)
    if n > max_frames:
        idx = np.unique(
            np.linspace(0, n - 1, num=max_frames, dtype=np.int64)
        )
        cube = cube[idx]
        n = cube.shape[0]

    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        log.warning("pipeline_plots: bkg_smooth_r1 all non-finite; skip animation.")
        return None
    vmin = float(np.nanpercentile(finite, 1))
    vmax = float(np.nanpercentile(finite, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0

    Full_n = bkg_smooth_r1.shape[0]
    btjd = None
    try:
        from syndiff_pipeline.difference_imaging.stages.background import btjd_for_hotpants_order

        if wcs_table is not None and len(hotpants_results):
            btjd = btjd_for_hotpants_order(wcs_table, hotpants_results)
            if btjd.shape[0] != Full_n:
                btjd = None
            elif n < Full_n:
                btjd = btjd[idx]
    except Exception as exc:
        log.debug("pipeline_plots: BTJD labels for animation: %s", exc)
        btjd = None

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    im = ax.imshow(
        cube[0],
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    def _title(fi: int) -> str:
        """Title.
        
        Parameters
        ----------
        fi : int
        
        Returns
        -------
        str"""
        subsampled = Full_n > n
        parts = [
            f"Frame {fi + 1}/{n} (subsampled from {Full_n})"
            if subsampled
            else f"Frame {fi + 1}/{n}"
        ]
        if btjd is not None and fi < len(btjd) and np.isfinite(btjd[fi]):
            parts.append(f"BTJD {btjd[fi]:.4f}")
        return " · ".join(parts)

    ax.set_title(_title(0))
    ax.set_xlabel("x (crop px)")
    ax.set_ylabel("y (crop px)")

    def _update(fi: int):
        """Update.
        
        Parameters
        ----------
        fi : int"""
        im.set_data(cube[fi])
        ax.set_title(_title(fi))
        return (im,)

    anim = FuncAnimation(fig, _update, frames=n, blit=False)
    try:
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer, dpi=dpi)
    except Exception as exc:
        log.warning(
            "pipeline_plots: could not save background animation (%s). "
            "Install pillow if missing.",
            exc,
        )
        plt.close(fig)
        return None
    plt.close(fig)
    log.info("  pipeline_plots: background removal animation %s", out_path)
    return out_path
