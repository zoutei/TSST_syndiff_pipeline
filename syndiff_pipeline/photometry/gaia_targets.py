"""Select common Gaia stars and prepare SN event photometry inputs.

This module deliberately uses the production SCC Gaia catalogs and hp_d WCS
headers.  It does not download Gaia data or modify any diff artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

from syndiff_pipeline.common.orchestration.targets import load_targets
from syndiff_pipeline.common.scc_paths import default_gaia_catalog_path, scc_diff_label_dir
from syndiff_pipeline.difference_imaging.stages.epsf import tess_mag_from_gaia_phot
from syndiff_pipeline.difference_imaging.support.ffi_naming import is_pipeline_fits_filename


def _column(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _normalise_catalog(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source = _column(out, "source_id", "SOURCE_ID")
    ra = _column(out, "ra", "RA")
    dec = _column(out, "dec", "DEC")
    tess = _column(out, "tess_mag", "TESS_MAG")
    derived = tess_mag_from_gaia_phot(
        _column(out, "phot_g_mean_mag").to_numpy(float),
        _column(out, "phot_bp_mean_mag").to_numpy(float),
        _column(out, "phot_rp_mean_mag").to_numpy(float),
    )
    tess = tess.to_numpy(float)
    tess[~np.isfinite(tess)] = derived[~np.isfinite(tess)]
    result = pd.DataFrame(
        {"source_id": source.astype("Int64"), "ra": ra, "dec": dec, "tess_mag": tess}
    )
    result = result.dropna(subset=["source_id", "ra", "dec", "tess_mag"])
    result["source_id"] = result["source_id"].astype(np.int64)
    return result.drop_duplicates("source_id", keep="first")


def _representative_hp_d(data_root: str | Path, scc: tuple[int, int, int], label: str) -> Path:
    lane = scc_diff_label_dir(data_root, *scc, store_name="linear", label=label)
    files = sorted(p for p in lane.rglob("*") if p.is_file() and is_pipeline_fits_filename(p.name))
    if not files:
        raise FileNotFoundError(f"No {label} FITS found for SCC {scc} under {lane}")
    return files[0]


def _in_hp_d(path: Path, ra: np.ndarray, dec: np.ndarray, margin: float = 12.0) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        header = hdu.header
        shape = np.asarray(hdu.data).shape[-2:]
        wcs = WCS(header).celestial
    x, y = wcs.world_to_pixel_values(ra, dec)
    ny, nx = shape
    return np.isfinite(x) & np.isfinite(y) & (x >= margin) & (x < nx - margin) & (y >= margin) & (y < ny - margin)


def select_common_targets(
    *,
    data_root: str | Path,
    targets_csv: str | Path,
    output_manifest: str | Path,
    count: int = 100,
    tess_mag_limit: float = 13.0,
    diffs_label: str = "hp_d",
) -> pd.DataFrame:
    """Select the nearest bright Gaia sources valid in every target SCC."""
    event_targets = load_targets(targets_csv)
    if not event_targets:
        raise ValueError("targets CSV contains no enabled SCCs")
    sccs = [(t.sector, t.camera, t.ccd) for t in event_targets]
    catalogs: list[pd.DataFrame] = []
    for scc in sccs:
        path = default_gaia_catalog_path(data_root, *scc)
        if not path.is_file():
            raise FileNotFoundError(f"Gaia catalog missing for {scc}: {path}")
        df = _normalise_catalog(pd.read_csv(path))
        df = df[df["tess_mag"] < tess_mag_limit].copy()
        df = df[_in_hp_d(_representative_hp_d(data_root, scc, diffs_label), df.ra.to_numpy(), df.dec.to_numpy())]
        catalogs.append(df[["source_id", "ra", "dec", "tess_mag"]])

    common = catalogs[0].rename(columns={"ra": "ra_0", "dec": "dec_0", "tess_mag": "tess_mag_0"})
    for i, df in enumerate(catalogs[1:], start=1):
        right = df.rename(columns={"ra": f"ra_{i}", "dec": f"dec_{i}", "tess_mag": f"tess_mag_{i}"})
        common = common.merge(right, on="source_id", how="inner")
    if len(common) < count:
        raise RuntimeError(f"Only {len(common)} common Gaia stars satisfy tess_mag < {tess_mag_limit}; need {count}")

    sn = SkyCoord(event_targets[0].target_ra * u.deg, event_targets[0].target_dec * u.deg)
    stars = SkyCoord(common.ra_0.to_numpy() * u.deg, common.dec_0.to_numpy() * u.deg)
    common["separation_arcsec"] = sn.separation(stars).arcsec
    common["tess_mag"] = common[[f"tess_mag_{i}" for i in range(len(sccs))]].median(axis=1)
    common = common.sort_values(["separation_arcsec", "tess_mag", "source_id"], kind="mergesort").head(count).copy()
    result = common[["source_id", "ra_0", "dec_0", "tess_mag", "separation_arcsec"]].rename(columns={"ra_0": "ra", "dec_0": "dec"})
    result.insert(0, "target_name", "gaia_" + result.source_id.astype(str))
    result["n_scc"] = len(sccs)
    out = Path(output_manifest).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({"count": len(result), "sccs": sccs, "tess_mag_limit": tess_mag_limit}, indent=2) + "\n", encoding="utf-8")
    return result


def write_photometry_config(base_config: str | Path, manifest: str | Path, output_config: str | Path) -> Path:
    """Create a star-only fixed-WCS config from the existing SN policy."""
    cfg = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    deployment = cfg.get("deployment_file")
    if deployment and not Path(str(deployment)).is_absolute():
        cfg["deployment_file"] = str((Path(base_config).expanduser().resolve().parent / str(deployment)).resolve())
    stars = pd.read_csv(manifest)
    cfg.setdefault("defaults", {})["photometry_run_id"] = "sn2022jhq_gaia13_common100"
    cfg["pipeline"] = [{
        "kind": "forced_photometry",
        "inputs": {"diffs": "hp_d", "epsf": "epsf_r1"},
        "output": "lc_all",
        "include_primary_target": False,
        "methods": [
            {"name": "ffiwcs", "type": "psf", "psf_type": "epsf", "fit_shape": 11, "aperture_radius": 2, "psf_grouper_min_separation": 10, "position_source": "native_wcs"},
            {"name": "temporalwcs", "type": "psf", "psf_type": "epsf", "fit_shape": 11, "aperture_radius": 2, "psf_grouper_min_separation": 10, "position_source": "temporal_wcs", "temporal_wcs_version": "temporal_cheb5_bspline_v1"},
        ],
    }]
    cfg["additional_forced_targets"] = [
        {"name": str(row.target_name), "ra": float(row.ra), "dec": float(row.dec)}
        for row in stars.itertuples()
    ]
    out = Path(output_config).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def write_common_target_montages(
    *, data_root: str | Path, manifest: str | Path, sccs: list[tuple[int, int, int]], output_dir: str | Path, diffs_label: str = "hp_d", stamp_size: int = 31,
) -> list[Path]:
    """Write one 10x10 north-up WCS montage for each SCC."""
    stars = pd.read_csv(manifest)
    if len(stars) != 100:
        raise ValueError(f"Expected exactly 100 targets for a 10x10 montage, got {len(stars)}")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    from astropy.visualization import ImageNormalize, AsinhStretch, ZScaleInterval
    from astropy.wcs.utils import proj_plane_pixel_scales
    import matplotlib.pyplot as plt
    from reproject import reproject_interp
    written: list[Path] = []
    for scc in sccs:
        image_path = _representative_hp_d(data_root, scc, diffs_label)
        with fits.open(image_path, memmap=False) as hdul:
            hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            image = np.asarray(hdu.data, dtype=float)
            wcs = WCS(hdu.header).celestial
        fig, axes = plt.subplots(10, 10, figsize=(20, 20), constrained_layout=True)
        norm = ImageNormalize(image, interval=ZScaleInterval(), stretch=AsinhStretch())
        for ax, row in zip(axes.flat, stars.itertuples()):
            try:
                position = SkyCoord(float(row.ra) * u.deg, float(row.dec) * u.deg)
                scale = float(np.nanmean(proj_plane_pixel_scales(wcs)))
                north_up = WCS(naxis=2)
                north_up.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                north_up.wcs.crval = [float(row.ra), float(row.dec)]
                north_up.wcs.crpix = [(stamp_size + 1) / 2.0] * 2
                north_up.wcs.cdelt = [-scale, scale]
                cut, _ = reproject_interp((image, wcs), north_up, shape_out=(stamp_size, stamp_size), order="bilinear")
                ax.imshow(cut, origin="lower", cmap="gray_r", norm=norm, interpolation="nearest")
                ax.set_title(f"{row.target_name}\nT={row.tess_mag:.2f}", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as exc:
                ax.text(0.5, 0.5, f"{row.target_name}\n{type(exc).__name__}", ha="center", va="center", fontsize=6)
                ax.set_axis_off()
        fig.suptitle(f"SN2022jhq common Gaia stars · S{scc[0]} C{scc[1]} K{scc[2]} · north-up WCS")
        path = out_dir / f"gaia_tess13_common100_s{scc[0]:04d}_c{scc[1]}_k{scc[2]}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-photometry-config")
    parser.add_argument("--output-photometry-config")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--tess-mag-limit", type=float, default=13.0)
    parser.add_argument("--diffs-label", default="hp_d")
    args = parser.parse_args()
    select_common_targets(data_root=args.data_root, targets_csv=args.targets, output_manifest=args.manifest, count=args.count, tess_mag_limit=args.tess_mag_limit, diffs_label=args.diffs_label)
    if bool(args.base_photometry_config) != bool(args.output_photometry_config):
        parser.error("--base-photometry-config and --output-photometry-config must be supplied together")
    if args.base_photometry_config:
        write_photometry_config(args.base_photometry_config, args.manifest, args.output_photometry_config)


if __name__ == "__main__":
    _cli()
