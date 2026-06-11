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


def _stack_cutouts(cutouts: list, size: int) -> np.ndarray:
    """List of optional (size, size) stamps → (n_epochs, size, size) float cube."""
    n = len(cutouts)
    cube = np.full((n, size, size), np.nan, dtype=float)
    for i, c in enumerate(cutouts):
        if c is not None:
            cube[i] = np.asarray(c, dtype=float)
    return cube


def _fixed_scale_limits(cube: np.ndarray, scale_mode: str) -> tuple[float, float]:
    """Fixed vmin/vmax over all frames (``symmetric`` or ``percentile``)."""
    finite = cube[np.isfinite(cube)]
    if finite.size == 0:
        return 0.0, 1.0
    if scale_mode == "symmetric":
        v = float(np.nanpercentile(np.abs(finite), 99))
        if not np.isfinite(v) or v <= 0.0:
            v = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(v) or v <= 0.0:
            v = 1.0
        return -v, v
    vmin = float(np.nanpercentile(finite, 1))
    vmax = float(np.nanpercentile(finite, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def _subsample_cube_and_btjd(
    cube: np.ndarray,
    btjd: Optional[np.ndarray],
    max_frames: int,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, int]:
    """Evenly subsample axis 0 when ``max_frames`` exceeded; return idx used."""
    n = cube.shape[0]
    full_n = n
    idx = np.arange(n, dtype=np.int64)
    if n > max_frames:
        idx = np.unique(np.linspace(0, n - 1, num=max_frames, dtype=np.int64))
        cube = cube[idx]
        n = cube.shape[0]
    if btjd is not None and len(btjd) == full_n and n < full_n:
        btjd = btjd[idx]
    return cube, btjd, idx, full_n


def _draw_photometry_marker(
    ax,
    marker_xy: Optional[tuple[float, float]],
    *,
    half_len: float = 3.0,
    color: str = "yellow",
) -> None:
    """Small ``+`` at the PSF-flux anchor in stamp pixel coordinates."""
    if marker_xy is None:
        return
    x, y = marker_xy
    if not (np.isfinite(x) and np.isfinite(y)):
        return
    ax.plot(
        [x - half_len, x + half_len],
        [y, y],
        color=color,
        lw=1.4,
        alpha=0.95,
        solid_capstyle="round",
    )
    ax.plot(
        [x, x],
        [y - half_len, y + half_len],
        color=color,
        lw=1.4,
        alpha=0.95,
        solid_capstyle="round",
    )


def _frame_title(fi: int, n: int, full_n: int, btjd: Optional[np.ndarray]) -> str:
    subsampled = full_n > n
    parts = [
        f"Frame {fi + 1}/{n} (subsampled from {full_n})"
        if subsampled
        else f"Frame {fi + 1}/{n}"
    ]
    if btjd is not None and fi < len(btjd) and np.isfinite(btjd[fi]):
        parts.append(f"BTJD {btjd[fi]:.4f}")
    return " · ".join(parts)


def _import_animation_backend():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        return plt, FuncAnimation, PillowWriter
    except ImportError:
        log.warning(
            "pipeline_plots: matplotlib not available; skip stamp animation."
        )
        return None, None, None


def write_stamp_animation(
    cutouts: list,
    output_path: str,
    *,
    btjd: Optional[np.ndarray] = None,
    stamp_size: int = 15,
    cmap: str = "RdBu_r",
    scale_mode: str = "symmetric",
    cbar_label: str = "Diff stamp",
    dpi: int = 150,
    fps: float = 3.0,
    max_frames: int = 150,
    marker_xy: Optional[tuple[float, float]] = None,
) -> Optional[str]:
    """
    Animated GIF of per-epoch square stamps with a fixed colour scale.

    ``scale_mode`` is ``symmetric`` (diff; ±99th |value|) or ``percentile`` (science).
    ``marker_xy`` marks the forced-photometry anchor in stamp pixel coordinates.
    """
    plt, FuncAnimation, PillowWriter = _import_animation_backend()
    if plt is None:
        return None

    cube = _stack_cutouts(cutouts, stamp_size)
    if cube.size == 0 or not np.isfinite(cube).any():
        log.warning("pipeline_plots: no finite stamp data; skip %s", output_path)
        return None

    cube, btjd, _, full_n = _subsample_cube_and_btjd(cube, btjd, max_frames)
    n = cube.shape[0]
    vmin, vmax = _fixed_scale_limits(cube, scale_mode)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
    im = ax.imshow(
        cube[0],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    _draw_photometry_marker(ax, marker_xy)
    ax.set_title(_frame_title(0, n, full_n, btjd))
    ax.set_xticks([])
    ax.set_yticks([])

    def _update(fi: int):
        im.set_data(cube[fi])
        ax.set_title(_frame_title(fi, n, full_n, btjd))
        return (im,)

    anim = FuncAnimation(fig, _update, frames=n, blit=False)
    try:
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        log.warning(
            "pipeline_plots: could not save stamp animation (%s). "
            "Install pillow if missing.",
            exc,
        )
        plt.close(fig)
        return None
    plt.close(fig)
    log.info("  pipeline_plots: stamp animation %s", output_path)
    return output_path


def write_dual_stamp_animation(
    diff_cutouts: list,
    science_cutouts: list,
    output_path: str,
    *,
    btjd: Optional[np.ndarray] = None,
    stamp_size: int = 15,
    dpi: int = 150,
    fps: float = 3.0,
    max_frames: int = 150,
    marker_xy: Optional[tuple[float, float]] = None,
) -> Optional[str]:
    """Side-by-side diff + science stamp GIF with independent fixed scales."""
    plt, FuncAnimation, PillowWriter = _import_animation_backend()
    if plt is None:
        return None

    diff_cube = _stack_cutouts(diff_cutouts, stamp_size)
    sci_cube = _stack_cutouts(science_cutouts, stamp_size)
    if diff_cube.shape[0] != sci_cube.shape[0]:
        log.warning(
            "pipeline_plots: diff/science cutout length mismatch; skip %s",
            output_path,
        )
        return None
    if not np.isfinite(diff_cube).any() and not np.isfinite(sci_cube).any():
        log.warning("pipeline_plots: no finite dual-stamp data; skip %s", output_path)
        return None

    full_n = diff_cube.shape[0]
    idx = np.arange(full_n, dtype=np.int64)
    if full_n > max_frames:
        idx = np.unique(np.linspace(0, full_n - 1, num=max_frames, dtype=np.int64))
    diff_cube = diff_cube[idx]
    sci_cube = sci_cube[idx]
    if btjd is not None and len(btjd) == full_n:
        btjd = np.asarray(btjd, dtype=float)[idx]
    n = diff_cube.shape[0]
    diff_vmin, diff_vmax = _fixed_scale_limits(diff_cube, "symmetric")
    sci_vmin, sci_vmax = _fixed_scale_limits(sci_cube, "percentile")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    im_d = ax_d.imshow(
        diff_cube[0],
        origin="lower",
        cmap="RdBu_r",
        vmin=diff_vmin,
        vmax=diff_vmax,
        interpolation="nearest",
    )
    im_s = ax_s.imshow(
        sci_cube[0],
        origin="lower",
        cmap="viridis",
        vmin=sci_vmin,
        vmax=sci_vmax,
        interpolation="nearest",
    )
    fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04).set_label("Diff stamp")
    fig.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04).set_label("Science stamp")
    for ax in (ax_d, ax_s):
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_photometry_marker(ax, marker_xy)
    ax_d.set_title("Diff")
    ax_s.set_title("Science")
    fig.suptitle(_frame_title(0, n, full_n, btjd))

    def _update(fi: int):
        im_d.set_data(diff_cube[fi])
        im_s.set_data(sci_cube[fi])
        fig.suptitle(_frame_title(fi, n, full_n, btjd))
        return (im_d, im_s)

    anim = FuncAnimation(fig, _update, frames=n, blit=False)
    try:
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        log.warning(
            "pipeline_plots: could not save dual stamp animation (%s). "
            "Install pillow if missing.",
            exc,
        )
        plt.close(fig)
        return None
    plt.close(fig)
    log.info("  pipeline_plots: dual stamp animation %s", output_path)
    return output_path
