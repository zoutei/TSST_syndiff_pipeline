# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""RGI-cubic ePSF interpolation + fixed-position / no-background PSF photometry."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from joblib import delayed
from scipy.interpolate import RegularGridInterpolator

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.common.fits_variants import try_resolve_fits_variant
from syndiff_pipeline.common.joblib_progress import parallel_map_with_optional_tqdm
from syndiff_pipeline.difference_imaging.stages.per_ffi_wcs import list_frames_for_lane
from syndiff_pipeline.difference_imaging.wcs.io import load_gaia_catalog

from syndiff_pipeline.forward_model._vendor.sat_star_lc.radec_to_xy_temporal import (
    load_s20_scaled_catalog,
    radec_to_crop_xy_series,
)

DEFAULT_LANE = Path("/astro/armin/koji/syndiff/data/s0020/c3/k3/diff_linear")
DEFAULT_SOURCE_ID = 1713781222000483456
DEFAULT_FIT_SHAPE = 11
DEFAULT_APERTURE_RADIUS = 4.0
DEFAULT_N_WORKERS = 12


def load_epsf_archive(npz_path: Path) -> SimpleNamespace:
    """Per-frame ePSF NPZ (data, grid_xypos, oversampling) without GriddedPSFModel."""
    z = np.load(npz_path, allow_pickle=True)
    return SimpleNamespace(
        data=np.asarray(z["data"], dtype=np.float64),
        grid_xypos=np.asarray(z["grid_xypos"], dtype=np.float64),
        oversampling=int(np.asarray(z["oversampling"])),
    )


def build_rgi_cubic(archive: SimpleNamespace) -> RegularGridInterpolator:
    """RegularGridInterpolator over the 5×5 ePSF grid; values shape (ny, nx, h, w)."""
    grid = np.asarray(archive.grid_xypos, dtype=np.float64)
    data = np.asarray(archive.data, dtype=np.float64)
    xs = np.unique(np.round(grid[:, 0], decimals=6))
    ys = np.unique(np.round(grid[:, 1], decimals=6))
    xs = np.sort(xs)
    ys = np.sort(ys)
    nx, ny = len(xs), len(ys)
    if nx * ny != data.shape[0]:
        raise ValueError(
            f"grid is not a full rectilinear lattice: {nx}×{ny} vs {data.shape[0]} stamps"
        )
    h, w = data.shape[1], data.shape[2]
    values = np.empty((ny, nx, h, w), dtype=np.float64)
    x_index = {float(x): i for i, x in enumerate(xs)}
    y_index = {float(y): j for j, y in enumerate(ys)}
    for k, (gx, gy) in enumerate(grid):
        ix = x_index[float(np.round(gx, decimals=6))]
        iy = y_index[float(np.round(gy, decimals=6))]
        values[iy, ix] = data[k]
    return RegularGridInterpolator(
        (ys, xs),
        values,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )


def interpolate_epsf_os(
    archive: SimpleNamespace,
    x: float,
    y: float,
    *,
    rgi: RegularGridInterpolator | None = None,
) -> np.ndarray:
    """Oversampled ePSF at crop-local (x, y), sum-normalized to 1."""
    interp = rgi if rgi is not None else build_rgi_cubic(archive)
    stamp = np.asarray(interp((float(y), float(x))), dtype=np.float64)
    s = float(np.nansum(stamp))
    if s > 0:
        stamp = stamp / s
    return stamp


def downsample_blocksum(epsf_os: np.ndarray, os_factor: int) -> np.ndarray:
    """Block-sum oversampled PSF to native pixels (zero fractional pre-shift)."""
    os_ = int(os_factor)
    arr = np.asarray(epsf_os, dtype=np.float64)
    over_size = arr.shape[0]
    h_trim = (over_size // os_) * os_
    w_trim = (arr.shape[1] // os_) * os_
    trimmed = arr[:h_trim, :w_trim]
    native = trimmed.reshape(h_trim // os_, os_, w_trim // os_, os_).sum(axis=(1, 3))
    s = float(native.sum())
    return native / s if s > 0 else native


def native_psf_at(archive: SimpleNamespace, x: float, y: float) -> np.ndarray:
    """RGI-cubic oversampled stamp → native sum-normalized PSF."""
    epsf_os = interpolate_epsf_os(archive, x, y)
    return downsample_blocksum(epsf_os, archive.oversampling)


def forced_photutils_flux_fixed(
    image: np.ndarray,
    native_psf: np.ndarray,
    x: float,
    y: float,
    *,
    fit_shape: int = DEFAULT_FIT_SHAPE,
    aperture_radius: float = DEFAULT_APERTURE_RADIUS,
) -> tuple[float, float, float, float]:
    """Flux-only PSFPhotometry at fixed (x, y); no local background."""
    from photutils.psf import ImagePSF, PSFPhotometry

    if not np.isfinite(x) or not np.isfinite(y):
        return np.nan, np.nan, np.nan, np.nan

    stamp = np.asarray(native_psf, dtype=np.float64)
    model = ImagePSF(stamp)
    model.x_0.fixed = True
    model.y_0.fixed = True
    fit_shape_use = min(int(fit_shape), int(stamp.shape[0]), int(stamp.shape[1]))
    if fit_shape_use % 2 == 0:
        fit_shape_use = max(3, fit_shape_use - 1)

    phot = PSFPhotometry(
        model,
        fit_shape=fit_shape_use,
        aperture_radius=float(aperture_radius),
        local_bkg_estimator=None,
    )
    init = Table()
    init["x_init"] = [float(x)]
    init["y_init"] = [float(y)]
    ix, iy = int(round(x)), int(round(y))
    half = fit_shape_use // 2
    ny, nx = image.shape
    r0, r1 = max(0, iy - half), min(ny, iy + half + 1)
    c0, c1 = max(0, ix - half), min(nx, ix + half + 1)
    cut = image[r0:r1, c0:c1]
    flux_guess = float(np.nansum(cut)) if cut.size else 0.0
    init["flux_init"] = [flux_guess if np.isfinite(flux_guess) else 0.0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = phot(np.asarray(image, dtype=np.float64), init_params=init)
        tab = result if hasattr(result, "colnames") else result.to_table()
        flux = float(tab["flux_fit"][0])
        eflux = float(tab["flux_err"][0]) if "flux_err" in tab.colnames else np.nan
        x_fit = float(tab["x_fit"][0]) if "x_fit" in tab.colnames else float(x)
        y_fit = float(tab["y_fit"][0]) if "y_fit" in tab.colnames else float(y)
        return flux, eflux, x_fit, y_fit
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def resolve_epsf_npz(lane: Path, stem: str, epsf_label: str = "epsf_r1") -> Path | None:
    path = lane / epsf_label / f"{stem}_epsf_r1.npz"
    if path.is_file():
        return path
    matches = sorted((lane / epsf_label).glob(f"{stem}*_epsf_r1.npz"))
    return matches[0] if matches else None


def resolve_hp_d(lane: Path, stem: str, hp_d_label: str = "hp_d") -> Path | None:
    frame_dir = lane / hp_d_label
    resolved = try_resolve_fits_variant(frame_dir / f"{stem}_hp_d.fits")
    if resolved is not None:
        return Path(resolved)
    candidates = sorted(frame_dir.glob(f"{stem}*.fits*"))
    return candidates[0] if candidates else None


def load_fits_image_data(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[1].data, dtype=np.float64)


def process_frame_forced_psf(
    stem: str,
    btjd: float,
    hp_d_path: str,
    epsf_path: str,
    x: float,
    y: float,
    fit_shape: int = DEFAULT_FIT_SHAPE,
    aperture_radius: float = DEFAULT_APERTURE_RADIUS,
) -> dict:
    """Picklable per-frame worker: load hp_d + ePSF → RGI → fixed-xy flux-only phot."""
    out = {
        "btjd": float(btjd),
        "ffi_stem": stem,
        "x": float(x),
        "y": float(y),
        "flux": np.nan,
        "flux_err": np.nan,
        "x_fit": np.nan,
        "y_fit": np.nan,
        "ok": False,
    }
    try:
        image = load_fits_image_data(Path(hp_d_path))
        archive = load_epsf_archive(Path(epsf_path))
        native = native_psf_at(archive, x, y)
        flux, eflux, x_fit, y_fit = forced_photutils_flux_fixed(
            image,
            native,
            x,
            y,
            fit_shape=fit_shape,
            aperture_radius=aperture_radius,
        )
        out.update(
            {
                "flux": flux,
                "flux_err": eflux,
                "x_fit": x_fit,
                "y_fit": y_fit,
                "ok": bool(np.isfinite(flux)),
            }
        )
    except Exception as exc:
        out["error"] = str(exc)
    return out


def build_frame_jobs(
    lane: Path,
    ra: float,
    dec: float,
    *,
    centroids_label: str = "centroids_r1",
    hp_d_label: str = "hp_d",
    epsf_label: str = "epsf_r1",
    catalog=None,
) -> list[dict]:
    """Precompute temporal (x, y) and resolve paths for every indexed frame."""
    if catalog is None:
        catalog = load_s20_scaled_catalog()
    frames = list_frames_for_lane(
        lane,
        centroids_label=centroids_label,
        hp_d_label=hp_d_label,
    )
    if not frames:
        raise RuntimeError(f"no frames found under {lane}")

    windows = [(lo, hi) for lo, hi, _ in catalog.runs]
    in_window_frames = [
        fr for fr in frames if any(lo <= float(fr.btjd) <= hi for lo, hi in windows)
    ]
    skipped_btjd = len(frames) - len(in_window_frames)
    if skipped_btjd:
        warnings.warn(
            f"skipped {skipped_btjd} frames outside temporal WCS windows",
            stacklevel=2,
        )

    btjds = [float(fr.btjd) for fr in in_window_frames]
    xy = radec_to_crop_xy_series(ra, dec, btjds, catalog)
    xy_by_btjd = {float(r.btjd): (float(r.x), float(r.y)) for _, r in xy.iterrows()}

    jobs: list[dict] = []
    skipped_paths = 0
    for fr in in_window_frames:
        btjd = float(fr.btjd)
        x, y = xy_by_btjd[btjd]
        hp = resolve_hp_d(lane, fr.stem, hp_d_label=hp_d_label)
        epsf = resolve_epsf_npz(lane, fr.stem, epsf_label=epsf_label)
        if hp is None or epsf is None:
            skipped_paths += 1
            continue
        jobs.append(
            {
                "stem": fr.stem,
                "btjd": btjd,
                "hp_d_path": str(hp),
                "epsf_path": str(epsf),
                "x": x,
                "y": y,
            }
        )
    if skipped_paths:
        warnings.warn(f"skipped {skipped_paths} frames missing hp_d or epsf", stacklevel=2)
    return jobs


def run_forced_psf_parallel(
    jobs: list[dict],
    *,
    n_workers: int = DEFAULT_N_WORKERS,
    fit_shape: int = DEFAULT_FIT_SHAPE,
    aperture_radius: float = DEFAULT_APERTURE_RADIUS,
) -> pd.DataFrame:
    """Run forced PSF photometry over *jobs* with loky parallelism."""
    if not jobs:
        return pd.DataFrame(
            columns=["btjd", "ffi_stem", "x", "y", "flux", "flux_err", "x_fit", "y_fit", "ok"]
        )
    delayed_calls = [
        delayed(process_frame_forced_psf)(
            j["stem"],
            j["btjd"],
            j["hp_d_path"],
            j["epsf_path"],
            j["x"],
            j["y"],
            fit_shape,
            aperture_radius,
        )
        for j in jobs
    ]
    rows = parallel_map_with_optional_tqdm(
        delayed_calls,
        n_tasks=len(delayed_calls),
        desc="forced_psf",
        n_jobs_eff=int(n_workers),
    )
    df = pd.DataFrame(rows).sort_values("btjd").reset_index(drop=True)
    return df


def gaia_radec(lane: Path, source_id: int) -> tuple[float, float]:
    gaia = load_gaia_catalog(lane)
    row = gaia.loc[gaia["source_id"].astype("int64") == int(source_id)].iloc[0]
    return float(row.ra), float(row.dec)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel RGI-cubic forced PSF photometry.")
    p.add_argument("--lane", type=Path, default=DEFAULT_LANE)
    p.add_argument("--source-id", type=int, default=DEFAULT_SOURCE_ID)
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--max-frames", type=int, default=None, help="Smoke/limit number of frames.")
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "dev" / "sat_star_lc" / "forced_photometry" / "outputs" / "forced_psf_lc.csv",
    )
    p.add_argument("--xy-out", type=Path, default=None, help="Optional xy table CSV path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ra, dec = gaia_radec(args.lane, args.source_id)
    print(f"source_id={args.source_id} ra={ra:.6f} dec={dec:.6f}")
    catalog = load_s20_scaled_catalog()
    print(f"temporal WCS windows: {[(lo, hi) for lo, hi, _ in catalog.runs]}")
    jobs = build_frame_jobs(args.lane, ra, dec, catalog=catalog)
    if args.max_frames is not None:
        jobs = jobs[: int(args.max_frames)]
    print(f"n_jobs={len(jobs)} n_workers={args.n_workers}")

    if args.xy_out is not None:
        xy_df = pd.DataFrame(
            [{"btjd": j["btjd"], "ffi_stem": j["stem"], "x": j["x"], "y": j["y"]} for j in jobs]
        )
        args.xy_out.parent.mkdir(parents=True, exist_ok=True)
        xy_df.to_csv(args.xy_out, index=False)
        print(f"wrote {args.xy_out}")

    df = run_forced_psf_parallel(jobs, n_workers=args.n_workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_ok = int(df["ok"].sum()) if "ok" in df.columns else 0
    print(f"wrote {args.out}  n={len(df)} ok={n_ok}")
    if len(df):
        print(df[["btjd", "x", "y", "flux", "flux_err"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
