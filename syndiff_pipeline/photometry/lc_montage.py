"""Compact 10x10 diagnostic light-curve montages for forced photometry."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_lc_montage(
    *,
    manifest_path: str | Path,
    lc_dir: str | Path,
    method: str,
    output_path: str | Path,
    grid_size: int = 10,
    field_label: str = "",
) -> Path:
    manifest = pd.read_csv(manifest_path)
    lc_dir = Path(lc_dir)
    output_path = Path(output_path)
    names = manifest["target_name"].astype(str).tolist()
    if len(names) != grid_size * grid_size:
        raise ValueError(f"expected {grid_size * grid_size} targets, got {len(names)}")

    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(20, 18),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(grid_size, grid_size)

    records = []
    for name, row in zip(names, manifest.itertuples(index=False)):
        path = lc_dir / f"lightcurve_{method}_{name}.csv"
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        btjd = pd.to_numeric(table["btjd"], errors="coerce").to_numpy(float)
        flux = pd.to_numeric(table["flux"], errors="coerce").to_numpy(float)
        x_fit = pd.to_numeric(table["x_fit"], errors="coerce").to_numpy(float)
        y_fit = pd.to_numeric(table["y_fit"], errors="coerce").to_numpy(float)
        finite = np.isfinite(btjd) & np.isfinite(flux) & (flux > 0)
        xy_finite = np.isfinite(x_fit) & np.isfinite(y_fit)
        if finite.sum() < 2 or xy_finite.sum() < 1:
            continue
        median_flux = float(np.nanmedian(flux[finite]))
        if not np.isfinite(median_flux) or median_flux <= 0:
            continue
        records.append(
            {
                "name": name,
                "tess_mag": float(row.tess_mag),
                "btjd": btjd,
                "flux": flux,
                "finite": finite,
                "median_flux": median_flux,
                "x": float(np.nanmedian(x_fit[xy_finite])),
                "y": float(np.nanmedian(y_fit[xy_finite])),
            }
        )

    # Spatial ordering: y high at the top, y low at the bottom; x low at the
    # left, x high at the right. Split the y-sorted list into spatial rows,
    # then sort each row by x.
    records.sort(key=lambda r: (-r["y"], r["x"]))
    spatial_records = []
    for start in range(0, len(records), grid_size):
        spatial_records.extend(
            sorted(records[start : start + grid_size], key=lambda r: r["x"])
        )

    plotted = 0
    secondary_axes = []
    for i, rec in enumerate(spatial_records[: grid_size * grid_size]):
        ax = axes.flat[i]
        btjd = rec["btjd"]
        flux = rec["flux"]
        finite = rec["finite"]
        median_flux = rec["median_flux"]
        rel = flux / median_flux - 1.0
        ax.plot(
            btjd[finite],
            flux[finite],
            ".-",
            ms=1.0,
            lw=0.35,
            color="C0",
            alpha=0.8,
        )
        lo, hi = np.nanpercentile(flux[finite], [0.5, 99.5])
        span = max(float(hi - lo), abs(float(median_flux)) * 1e-3)
        pad = 0.08 * span
        ax.set_ylim(float(lo) - pad, float(hi) + pad)
        ax.axhline(median_flux, color="k", lw=0.35, alpha=0.35)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=5, length=2)
        ax.set_title(
            f"{rec['name'].removeprefix('gaia_')}  T={rec['tess_mag']:.2f}\n"
            f"x={rec['x']:.1f}, y={rec['y']:.1f}",
            fontsize=5.5,
        )
        sec = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y, m=median_flux: y / m,
                lambda y, m=median_flux: y * m,
            ),
        )
        sec.tick_params(labelsize=4, length=1)
        secondary_axes.append(sec)
        plotted += 1

    for ax in axes[-1, :]:
        ax.set_xlabel("BTJD", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("flux", fontsize=6)
    if secondary_axes:
        secondary_axes[0].set_ylabel(r"$F/F_{\rm med}$", fontsize=7)
    fig.suptitle(
        f"SN2022jhq · {field_label} · {method} · 100 Gaia stars · n={plotted}\n"
        r"left: raw flux  ·  right: $F/F_{\rm med}$  ·  spatial order: x low→high, y high→low",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lc-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--field-label", default="")
    args = parser.parse_args()
    for method in ("ffiwcs", "temporalwcs"):
        write_lc_montage(
            manifest_path=args.manifest,
            lc_dir=args.lc_dir,
            method=method,
            output_path=args.output_dir / f"lightcurve_montage_{method}.png",
            field_label=args.field_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
