"""Debug plots for star-host segmentation, mini-template downsampling, and light curves."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.star.identifiers import ResolvedHost


def _asinh_norm(data: np.ndarray, sigma: float | None = None):
    from matplotlib.colors import AsinhNorm

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        floor = 1.0
    else:
        if sigma is None:
            sigma = float(np.nanmedian(np.abs(finite - np.nanmedian(finite))))
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = float(np.nanstd(finite))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        floor = max(float(np.nanmin(finite)), sigma * 1e-3, 1e-12)
    vmax = float(np.nanpercentile(finite, 99.5)) if finite.size else floor + 1.0
    if not np.isfinite(vmax) or vmax <= floor:
        vmax = floor + 1.0
    return AsinhNorm(vmin=floor, vmax=vmax)


def write_ps1_segment_overlay_png(
    path,
    *,
    original_image: np.ndarray,
    data_wo_bkg_sat: np.ndarray,
    host_pixel_xy: tuple[float, float],
    skycell_name: str,
    host: ResolvedHost,
    blend_flag: bool,
    cutout_bounds: tuple[int, int, int, int],
    target_seg_id: int = 0,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    x0, y0, x1, y1 = cutout_bounds
    image = np.asarray(original_image, dtype=np.float64)[y0:y1, x0:x1]
    overlay = np.asarray(data_wo_bkg_sat, dtype=np.float64)[y0:y1, x0:x1]
    host_x = float(host_pixel_xy[0]) - x0
    host_y = float(host_pixel_xy[1]) - y0

    finite = image[np.isfinite(image) & (image > 0)]
    if finite.size:
        vmin = max(float(np.nanpercentile(finite, 5)), 1e-6)
        vmax = max(float(np.nanpercentile(finite, 99.5)), vmin * 1.01)
    else:
        vmin, vmax = 1e-6, 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, origin="lower", cmap="gray_r", norm=norm, interpolation="nearest")

    if target_seg_id > 0 and np.any(overlay > 0):
        ax.imshow(
            overlay,
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=0.01,
            alpha=0.7,
            interpolation="nearest",
        )

    ax.plot(host_x, host_y, "+", color="red", ms=12, mew=2)

    title = (
        f"{skycell_name} | Gaia {host.gaia_source_id} | "
        f"seg {target_seg_id} | blend={blend_flag}"
    )
    ax.set_title(title, fontsize=10)
    if target_seg_id == 0:
        ax.text(
            0.5,
            0.02,
            "no segment found",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="yellow",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
        )

    ax.set_xlabel("PS1 x")
    ax.set_ylabel("PS1 y")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


_LC_METHOD_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def _lc_flux_series(df: "pd.DataFrame") -> "pd.Series":
    """Prefer the background-subtracted aperture flux when both are present."""
    return df["flux_wo_sky"] if "flux_wo_sky" in df.columns else df["flux"]


def write_lightcurve_debug_png(
    path,
    *,
    lightcurves: dict[str, "pd.DataFrame"],
    host: ResolvedHost,
) -> str:
    """
    Debug plot of one or more windowed-photometry light curves for one host.

    Parameters
    ----------
    path : output PNG path
    lightcurves : mapping of method name (e.g. ``"ap3"``, ``"prf"``) to the
        DataFrame written by :func:`~syndiff_pipeline.star.windowed_photometry.run_windowed_forced_photometry`
    host : the resolved host these light curves belong to
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method_names = [name for name, df in lightcurves.items() if not df.empty]
    if not method_names:
        raise ValueError("lightcurves must contain at least one non-empty DataFrame")

    fig, ax_main = plt.subplots(figsize=(9, 4))
    axes = [ax_main] + [ax_main.twinx() for _ in method_names[1:]]

    has_time = any(
        lightcurves[name]["btjd"].notna().any() for name in method_names
    )

    for i, (name, ax, color) in enumerate(zip(method_names, axes, _LC_METHOD_COLORS)):
        df = lightcurves[name]
        flux = _lc_flux_series(df)
        eflux = df["eflux"] if "eflux" in df.columns else None
        if has_time and df["btjd"].notna().any():
            x = df["btjd"].to_numpy()
            xlabel = "BTJD"
        else:
            x = np.arange(len(df))
            xlabel = "frame index"
        ax.errorbar(
            x, flux.to_numpy(),
            yerr=eflux.to_numpy() if eflux is not None else None,
            fmt="o-", color=color, label=f"{name} flux", ms=4,
        )
        ax.set_ylabel(f"{name} flux (ADU)", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        if i > 0:
            ax.spines["right"].set_position(("outward", 55 * (i - 1)))

    ax_main.set_xlabel(xlabel)
    ax_main.set_title(
        f"Host-star light curve | Gaia {host.gaia_source_id}"
        + (f" | TIC {host.tic_id}" if host.tic_id else ""),
        fontsize=10,
    )
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def write_mini_template_downsample_png(
    path,
    *,
    mini_flux_sum: np.ndarray,
    host_local_xy: tuple[float, float],
    dx: float,
    dy: float,
    host: ResolvedHost,
    production_template_slice: np.ndarray | None = None,
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flux = np.asarray(mini_flux_sum, dtype=np.float64)
    norm = _asinh_norm(flux)
    host_x, host_y = host_local_xy

    if production_template_slice is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax_main, ax_inset = axes
    else:
        fig, ax_main = plt.subplots(figsize=(6, 6))
        ax_inset = None

    im = ax_main.imshow(
        flux,
        origin="lower",
        cmap="magma",
        norm=norm,
        interpolation="nearest",
    )
    ax_main.plot(host_x, host_y, "+", color="red", ms=12, mew=2)
    fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)

    integrated = float(np.nansum(flux))
    roi_text = ""
    if roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        roi_text = f" ROI [{x0},{x1})×[{y0},{y1})"
    ax_main.set_title(
        f"Mini star template dx={dx:.3f} dy={dy:.3f} | Gaia {host.gaia_source_id}"
        f"{roi_text}\nintegrated flux={integrated:.4g}",
        fontsize=10,
    )
    ax_main.set_xlabel("TESS x (mini ROI local)")
    ax_main.set_ylabel("TESS y (mini ROI local)")

    if ax_inset is not None and production_template_slice is not None:
        inset = np.asarray(production_template_slice, dtype=np.float64)
        inset_norm = _asinh_norm(inset)
        ax_inset.imshow(
            inset,
            origin="lower",
            cmap="magma",
            norm=inset_norm,
            interpolation="nearest",
        )
        cx = min(max(host_x, 0), inset.shape[1] - 1)
        cy = min(max(host_y, 0), inset.shape[0] - 1)
        ax_inset.plot(cx, cy, "+", color="red", ms=10, mew=2)
        ax_inset.set_title("Production template (inset)", fontsize=9)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)
