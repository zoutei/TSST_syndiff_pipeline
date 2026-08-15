"""Load centroids frames and Gaia for per-FFI WCS fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.time import Time

from syndiff_pipeline.common.fits_variants import try_resolve_fits_variant
from syndiff_pipeline.difference_imaging.stages.centroids import load_centroids_index
from syndiff_pipeline.difference_imaging.wcs.sci2idl import GAIA_CATALOG_BASENAME


@dataclass
class FrameRecord:
    stem: str
    btjd: float
    hp_d_path: Path
    phot_path: Path
    crop_shape: tuple[int, int]


def btjd_from_header(header: fits.Header) -> float:
    date = header.get("DATE-OBS")
    if not date:
        return float("nan")
    return float(Time(date, format="isot", scale="utc").jd - 2457000.0)


def load_gaia_catalog(lane_root: Path) -> pd.DataFrame:
    for rel in (GAIA_CATALOG_BASENAME, f"../{GAIA_CATALOG_BASENAME}"):
        path = lane_root / rel
        if path.is_file():
            return pd.read_csv(path)
    scc_root = lane_root.parent.parent
    path = scc_root / GAIA_CATALOG_BASENAME
    if path.is_file():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Gaia catalog not found near {lane_root}")


def list_centroid_frames(lane_root: Path) -> list[FrameRecord]:
    centroids_dir = lane_root / "centroids_r1"
    hp_d_dir = lane_root / "hp_d"
    index = load_centroids_index(str(centroids_dir))
    frames: list[FrameRecord] = []
    for stem, phot_rel in sorted(index.items()):
        phot_path = Path(phot_rel)
        if not phot_path.is_file():
            phot_path = centroids_dir / Path(phot_rel).name
        hp_candidates = sorted(hp_d_dir.glob(f"{stem}*.fits*"))
        if not hp_candidates:
            hp_path = try_resolve_fits_variant(hp_d_dir / f"{stem}_hp_d.fits")
            if hp_path is None:
                continue
            hp_candidates = [Path(hp_path)]
        hdr = fits.getheader(hp_candidates[0], ext=1)
        from syndiff_pipeline.difference_imaging.wcs.sci2idl import crop_bounds_from_header

        ny, nx = crop_bounds_from_header(hdr)["shape"]
        frames.append(
            FrameRecord(
                stem=stem,
                btjd=btjd_from_header(hdr),
                hp_d_path=hp_candidates[0],
                phot_path=phot_path,
                crop_shape=(ny, nx),
            )
        )
    frames.sort(key=lambda f: f.btjd)
    return frames
