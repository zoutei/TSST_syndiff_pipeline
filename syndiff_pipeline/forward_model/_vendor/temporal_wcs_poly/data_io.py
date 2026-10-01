# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Load centroids, Gaia, shared linear WCS, and MIT orbit windows."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.stages.centroids import load_centroids_index
from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
  ensure_tess_orbit_times_csv,
)

_DEV_WCS = Path(__file__).resolve().parents[1] / "wcs_fit_from_centroids"
if str(_DEV_WCS) not in sys.path:
  pass  # migration: dev sys.path wiring removed

from syndiff_pipeline.forward_model._vendor.wcs_fit_from_centroids.fit_wcs_from_centroids import (  # noqa: E402
  FitConfig,
  StarSelectionConfig,
  crop_bounds_from_header,
  fit_crop_wcs,
  join_stars,
  linear_wcs_param_table,
  select_good_stars,
  _uvprime_from_linear_wcs,
)

GAIA_CATALOG_BASENAME = "gaia_catalog_pipeline.csv"


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


def load_gaia_catalog(workspace: Path) -> pd.DataFrame:
  for rel in (GAIA_CATALOG_BASENAME, f"../{GAIA_CATALOG_BASENAME}"):
    path = workspace / rel
    if path.is_file():
      return pd.read_csv(path)
  raise FileNotFoundError(f"Gaia catalog not found under {workspace}")


def load_merged_stars(
    frame: FrameRecord,
    gaia: pd.DataFrame,
) -> pd.DataFrame:
  phot = Table.read(frame.phot_path, format="ascii.ecsv")
  return join_stars(phot, gaia)


def select_qc_stars(
    merged: pd.DataFrame,
    star_cfg: StarSelectionConfig | None = None,
) -> pd.DataFrame:
  cfg = star_cfg or StarSelectionConfig()
  return select_good_stars(merged, cfg)


def fit_shared_linear_wcs(
    stars: pd.DataFrame,
    crop_shape: tuple[int, int],
) -> WCS:
  wcs_linear, _, err = fit_crop_wcs(
    stars, crop_shape, FitConfig(sip_degree=0, sip_fallback=())
  )
  if wcs_linear is None:
    raise RuntimeError(f"linear WCS fit failed: {err}")
  return wcs_linear


def linear_wcs_summary(wcs_linear: WCS) -> dict[str, Any]:
  row = linear_wcs_param_table(wcs_linear).iloc[0].to_dict()
  return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in row.items()}


def list_centroid_frames(workspace: Path) -> list[FrameRecord]:
  centroids_dir = workspace / "centroids_r1"
  hp_d_dir = workspace / "hp_d"
  index = load_centroids_index(str(centroids_dir))
  frames: list[FrameRecord] = []
  for stem, phot_rel in sorted(index.items()):
    phot_path = Path(phot_rel)
    if not phot_path.is_file():
      phot_path = centroids_dir / Path(phot_rel).name
    # Match only the final, complete filenames -- not the diff pipeline's
    # atomic-write temp files (e.g. "{stem}_hp_d.48u3i8jj.fits.part"), which
    # can be left behind, uncleaned, by an interrupted/evicted write and
    # otherwise sort ahead of the real file (ASCII '.' + digit < '.' + "fits").
    # See dev/forward_epsf_wcs/data.py's list_orbit_frames_from_temporal_wcs
    # for the confirmed root-cause incident (s51 orbit1, 2026-08-27).
    hp_candidates = sorted(
      p for p in hp_d_dir.glob(f"{stem}*")
      if p.name in {f"{stem}_hp_d.fits", f"{stem}_hp_d.fits.gz", f"{stem}_hp_d.fits.fz"}
    )
    if not hp_candidates:
      continue
    hdr = fits.getheader(hp_candidates[0], ext=1)
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


def sector_orbit_window(sector: int, orbit_index: int = 1) -> tuple[float, float, int]:
  """BTJD window for orbit_index (1-based) within sector rows in MIT orbit table."""
  csv_path = ensure_tess_orbit_times_csv()
  orbits = pd.read_csv(csv_path, skipfooter=1, engine="python")
  sec_rows = orbits.loc[orbits["Sector"].astype(str) == str(int(sector))].reset_index(drop=True)
  if sec_rows.empty:
    raise RuntimeError(f"No orbit rows for sector {sector} in {csv_path}")
  idx = int(orbit_index) - 1
  if idx < 0 or idx >= len(sec_rows):
    raise RuntimeError(
      f"orbit_index={orbit_index} out of range for sector {sector} "
      f"({len(sec_rows)} orbits in table)"
    )
  row = sec_rows.iloc[idx]
  t0 = Time(str(row["Start of Orbit"]).replace(" ", "T"), format="isot", scale="utc").jd - 2457000.0
  t1 = Time(str(row["End of Orbit"]).replace(" ", "T"), format="isot", scale="utc").jd - 2457000.0
  return float(t0), float(t1), int(row["Orbit"])


def first_orbit_window(sector: int, frames: list[FrameRecord]) -> tuple[float, float, int]:
  del frames  # kept for call-site compatibility
  return sector_orbit_window(sector, orbit_index=1)


def list_frames_in_orbit(
    frames: list[FrameRecord],
    btjd_start: float,
    btjd_end: float,
) -> list[FrameRecord]:
  return [f for f in frames if btjd_start <= f.btjd <= btjd_end]


def select_fit_frames(
    frames: list[FrameRecord],
    *,
    n_frames: int | None,
    all_orbit: bool,
    btjd_start: float,
    btjd_end: float,
) -> list[FrameRecord]:
  orbit_frames = list_frames_in_orbit(frames, btjd_start, btjd_end)
  if all_orbit:
    return orbit_frames
  n = n_frames if n_frames is not None else 3
  return orbit_frames[:n]


def build_frame_stars(
    stars_qc: pd.DataFrame,
    linear_wcs: WCS,
    *,
    stem: str,
    btjd: float,
) -> pd.DataFrame:
  if stars_qc.empty:
    return stars_qc

  x_fit = stars_qc["x_fit"].to_numpy(dtype=float)
  y_fit = stars_qc["y_fit"].to_numpy(dtype=float)
  ra = stars_qc["ra"].to_numpy(dtype=float)
  dec = stars_qc["dec"].to_numpy(dtype=float)

  crpix1 = float(linear_wcs.wcs.crpix[0])
  crpix2 = float(linear_wcs.wcs.crpix[1])
  stars = stars_qc.copy()
  stars["xprime"] = x_fit - (crpix1 - 1.0)
  stars["yprime"] = y_fit - (crpix2 - 1.0)
  u, v = _uvprime_from_linear_wcs(linear_wcs, ra, dec)
  stars["u"] = u
  stars["v"] = v
  stars["stem"] = stem
  stars["btjd"] = btjd
  return stars


def subsample_stars_per_frame(
    stacked: pd.DataFrame,
    *,
    max_per_frame: int,
    seed: int = 0,
) -> pd.DataFrame:
  if max_per_frame <= 0:
    return stacked
  rng = np.random.default_rng(seed)
  parts: list[pd.DataFrame] = []
  for stem in stacked["stem"].unique():
    sub = stacked.loc[stacked["stem"] == stem]
    if len(sub) <= max_per_frame:
      parts.append(sub)
    else:
      idx = rng.choice(len(sub), size=max_per_frame, replace=False)
      parts.append(sub.iloc[idx])
  return pd.concat(parts, ignore_index=True)


def stack_frame_stars(frame_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
  parts = [df for df in frame_tables.values() if len(df)]
  if not parts:
    return pd.DataFrame()
  return pd.concat(parts, ignore_index=True)


def write_fit_meta(path: Path, meta: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(meta, indent=2) + "\n")
