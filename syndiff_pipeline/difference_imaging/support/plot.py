"""
Diagnostic plots when ``SynDiffConfig.pipeline_plots`` is True.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


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
