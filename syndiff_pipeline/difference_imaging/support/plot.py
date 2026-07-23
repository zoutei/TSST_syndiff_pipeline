"""
Diagnostic plots when ``SynDiffConfig.pipeline_plots`` is True.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

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
        f"{_safe_plot_token(epsf_label)}_{_safe_plot_token(ffi_stem)}.png",
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


def _epsf_asinh_norm(stamps: np.ndarray) -> "matplotlib.colors.AsinhNorm":
    """
    AsinhNorm for peak-normalized ePSF tiles.

    vmin is pinned at 0 (negative residuals are display noise). vmax is 1 after
    per-tile peak normalization. linear_width sets the linear cutoff in the same
    units: wings sit near ~0.001–0.01, so ~5× wing scatter is a good default.
    """
    from matplotlib.colors import AsinhNorm

    norm_stamps = np.stack([_normalize_stamp(s) for s in stamps], axis=0)
    finite = norm_stamps[np.isfinite(norm_stamps)]
    if finite.size == 0:
        return AsinhNorm(vmin=0.0, vmax=1.0, linear_width=0.01)

    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    if not np.isfinite(mad) or mad <= 0:
        mad = float(np.nanstd(finite))
    if not np.isfinite(mad) or mad <= 0:
        mad = 0.01

    linear_width = float(np.clip(5.0 * mad, 0.005, 0.05))
    return AsinhNorm(vmin=0.0, vmax=1.0, linear_width=linear_width)


_EPSF_COLORBAR_TICKS = (0.0, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0)


def _btjd_for_stem(stem: str, btjd_by_stem: dict[str, float]) -> float:
    from syndiff_pipeline.difference_imaging.support.ffi_naming import (
        tess_product_id_from_ffi_path,
    )

    t = btjd_by_stem.get(stem)
    if t is not None and np.isfinite(t):
        return float(t)
    product_id = tess_product_id_from_ffi_path(stem) or ""
    if product_id:
        t = btjd_by_stem.get(product_id)
        if t is not None and np.isfinite(t):
            return float(t)
    return float("nan")


def select_evenly_spaced_stems(
    stems: list[str],
    *,
    wcs_table: "pd.DataFrame | None" = None,
    max_frames: int = 10,
) -> list[str]:
    """Pick up to *max_frames* stems evenly spaced in BTJD (or sorted stem order)."""
    if max_frames <= 0 or len(stems) <= max_frames:
        return list(stems)

    btjd_map: dict[str, float] = {}
    if wcs_table is not None:
        from syndiff_pipeline.difference_imaging.stages.epsf import (
            btjd_by_stem_from_manifest,
        )

        btjd_map = btjd_by_stem_from_manifest(wcs_table)

    if btjd_map:
        ordered = sorted(stems, key=lambda s: (_btjd_for_stem(s, btjd_map), s))
    else:
        ordered = sorted(stems)

    n = len(ordered)
    pick_idx = np.linspace(0, n - 1, num=max_frames, dtype=int)
    return [ordered[int(i)] for i in np.unique(pick_idx)]


def spatial_tile_subplot_grid(
    grid_xypos: np.ndarray,
) -> tuple[int, int, list[tuple[int, int, int]]]:
    """
    Return ``(n_rows, n_cols, placements)`` for a spatial tile montage.

    Each placement is ``(node_index, matplotlib_row, col)`` with the
    bottom-left subplot at the lowest ``(x, y)`` and top-right at the highest.
    """
    grid_xypos = np.asarray(grid_xypos, dtype=np.float64)
    n_grid = int(grid_xypos.shape[0])
    xs = [float(grid_xypos[k, 0]) for k in range(n_grid)]
    ys = [float(grid_xypos[k, 1]) for k in range(n_grid)]
    unique_xs = sorted(set(xs))
    unique_ys = sorted(set(ys))
    n_rows = max(1, len(unique_ys))
    n_cols = max(1, len(unique_xs))
    x_rank = {x: i for i, x in enumerate(unique_xs)}
    y_rank = {y: i for i, y in enumerate(unique_ys)}
    placements: list[tuple[int, int, int]] = []
    for k in range(n_grid):
        col = x_rank[xs[k]]
        row_from_bottom = y_rank[ys[k]]
        mrow = n_rows - 1 - row_from_bottom
        placements.append((k, mrow, col))
    return n_rows, n_cols, placements


def write_gridded_epsf_frame_plot(
    npz_path: str,
    png_path: str,
    *,
    dpi: int = 150,
    title: str = "",
    cmap: str = "viridis",
) -> Optional[str]:
    """
    Plot the gridded ePSF tiles from one per-frame ``*_gridded_epsf.npz``.

    Tiles are laid out by crop-local position: bottom-left is lowest ``(x, y)``,
    top-right is highest ``(x, y)``. Uses peak-normalized stamps with a shared
    AsinhNorm colorbar so PSF wings remain visible across the montage.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
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

    n_rows, n_cols, placements = spatial_tile_subplot_grid(grid_xypos)
    norm = _epsf_asinh_norm(stack)

    fig = plt.figure(
        figsize=(2.0 * n_cols + 1.2, 2.0 * n_rows + 1.0),
        layout="constrained",
    )
    gs = fig.add_gridspec(n_rows, n_cols)

    for k, row, col in placements:
        ax = fig.add_subplot(gs[row, col])
        stamp = _normalize_stamp(stack[k])
        ax.imshow(
            stamp,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        if k < len(grid_xypos):
            gx, gy = grid_xypos[k]
            ax.set_title(f"node {k}\n({gx:.0f}, {gy:.0f})", fontsize=8)
        else:
            ax.set_title(f"node {k}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    head = title or os.path.basename(npz_path)
    lw = getattr(norm, "linear_width", None)
    fig.suptitle(
        f"Gridded ePSF · {head} · oversample={oversampling} · "
        f"Asinh linear_width={lw:.4g}",
        fontsize=11,
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=fig.axes,
        location="right",
        shrink=0.85,
        pad=0.02,
        fraction=0.035,
    )
    cbar.set_label("peak-normalized flux")
    cbar.set_ticks(list(_EPSF_COLORBAR_TICKS))
    cbar.set_ticklabels([f"{v:g}" for v in _EPSF_COLORBAR_TICKS])
    cbar.minorticks_off()

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
    max_frames: int = 10,
    wcs_table: "pd.DataFrame | None" = None,
) -> list[str]:
    """Write diagnostic PNGs for representative per-frame gridded ePSFs."""
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

    stems = select_evenly_spaced_stems(
        list(index.keys()),
        wcs_table=wcs_table,
        max_frames=max_frames,
    )

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
        )
        if out:
            written.append(out)

    summary_path = os.path.join(plot_dir, f"{_safe_plot_token(epsf_label)}_index.json")
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


def centroids_residual_fits_path(
    plot_dir: str,
    centroids_label: str,
    ffi_stem: str,
) -> str:
    """FITS path for one frame's PSF-subtraction residual image."""
    return os.path.join(
        plot_dir,
        f"{_safe_plot_token(centroids_label)}_{_safe_plot_token(ffi_stem)}_epsf_photometry_residual.fits",
    )


def select_pipeline_debug_stems(
    available_stems: list[str],
    *,
    reference_plot_dir: str | None = None,
    reference_label: str | None = None,
    wcs_table: "pd.DataFrame | None" = None,
    max_frames: int = 10,
) -> list[str]:
    """
    Pick debug frames, preferring an existing reference plot index (e.g. ePSF).

    When ``reference_plot_dir`` contains ``{reference_label}_index.json`` with a
    ``frames_plotted`` list, those stems are used (intersected with
    *available_stems*). Otherwise falls back to :func:`select_evenly_spaced_stems`.
    """
    available = list(available_stems)
    if reference_plot_dir and reference_label:
        summary_path = os.path.join(
            reference_plot_dir,
            f"{_safe_plot_token(reference_label)}_index.json",
        )
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                frames = [str(s) for s in (data.get("frames_plotted") or [])]
                picked = [s for s in frames if s in available]
                if picked:
                    return picked[:max_frames]
            except Exception as exc:
                log.warning(
                    "pipeline_plots: cannot read reference index %s: %s",
                    summary_path,
                    exc,
                )
    return select_evenly_spaced_stems(
        available,
        wcs_table=wcs_table,
        max_frames=max_frames,
    )


def write_centroids_workspace_plots(
    centroids_workspace_dir: str,
    plot_dir: str,
    *,
    centroids_label: str = "centroids_r1",
    diff_paths_by_stem: dict[str, str],
    gaia_df: "pd.DataFrame",
    epsf_catalog,
    params,
    ffi_list_df: "pd.DataFrame",
    science_bounds: dict,
    ffi_path_by_stem: dict[str, str],
    max_frames: int = 10,
    wcs_table: "pd.DataFrame | None" = None,
    reference_plot_dir: str | None = None,
    reference_label: str | None = None,
) -> list[str]:
    """Write PSF-subtraction residual FITS for representative centroid frames."""
    from syndiff_pipeline.difference_imaging.stages import centroids as centroids_stage

    written: list[str] = []
    index = centroids_stage.load_centroids_index(centroids_workspace_dir)
    available = [s for s in index if s in diff_paths_by_stem]
    if not available:
        available = sorted(diff_paths_by_stem)
    if not available:
        log.warning(
            "pipeline_plots: no centroid frames available in %s; skip centroids debug FITS.",
            centroids_workspace_dir,
        )
        return written

    stems = select_pipeline_debug_stems(
        available,
        reference_plot_dir=reference_plot_dir,
        reference_label=reference_label,
        wcs_table=wcs_table,
        max_frames=max_frames,
    )

    from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
        prepare_gaia_for_gridded_epsf,
    )

    class _MagParams:
        mag_max_rp = getattr(params, "mag_max_rp", 12.95)

    gaia_filtered = prepare_gaia_for_gridded_epsf(gaia_df, _MagParams())

    os.makedirs(plot_dir, exist_ok=True)
    for stem in stems:
        diff_path = diff_paths_by_stem.get(stem)
        if not diff_path:
            continue
        ffi_path = ffi_path_by_stem.get(stem)
        if not ffi_path:
            from syndiff_pipeline.difference_imaging.support.ffi_naming import (
                tess_product_id_from_ffi_path,
            )

            product_id = tess_product_id_from_ffi_path(stem) or ""
            ffi_path = ffi_path_by_stem.get(product_id)
        if not ffi_path:
            log.warning("pipeline_plots: no FFI path for centroid debug stem %s", stem)
            continue

        residual_path = centroids_residual_fits_path(plot_dir, centroids_label, stem)
        if not os.path.isfile(residual_path):
            centroids_stage.write_frame_residual_fits(
                diff_path,
                stem,
                gaia_filtered,
                epsf_catalog,
                params,
                residual_path,
                ffi_path=ffi_path,
                ffi_list_df=ffi_list_df,
                science_bounds=science_bounds,
            )
        if os.path.isfile(residual_path):
            written.append(residual_path)
            log.info("  pipeline_plots: centroids residual FITS %s", residual_path)

    summary_path = os.path.join(
        plot_dir, f"{_safe_plot_token(centroids_label)}_index.json"
    )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "centroids_workspace": centroids_workspace_dir,
                "frames_plotted": stems,
                "reference_plot_dir": reference_plot_dir,
                "reference_label": reference_label,
                "residual_fits_paths": list(written),
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
