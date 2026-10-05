# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Frame listing, region/magnitude selection, hp_d + NOISE loading.

Data source is a completed ``diff_linear`` workspace (e.g.
``data/data/s0020/c3/k3/diff_linear``): per-FFI ``hp_d`` difference images
(background- and faint-star-free above ~tess_mag 13, see plan §Data), the
pipeline Gaia catalog (crop-local x,y), and ``centroids_r1`` photometry used
only for the shared-WCS fit and warm-start, never as pixel-fit input.

The hp_d MASK extension is **not** used — it is not the correct mask for this
prototype. Stamp weights are all-ones; variance comes from the NOISE plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from . import _bootstrap  # noqa: F401  (sys.path wiring, must run before dev imports)

from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import (  # noqa: E402
    FrameRecord,
    crop_bounds_from_header,
    list_centroid_frames,
    list_frames_in_orbit,
    load_gaia_catalog,
    sector_orbit_window,
)

@dataclass(frozen=True)
class RegionSpec:
    """Crop-local pixel region, half-open [x_min, x_max) x [y_min, y_max)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def shape(self) -> tuple[int, int]:
        """(ny, nx)."""
        return (self.y_max - self.y_min, self.x_max - self.x_min)

    @property
    def center(self) -> tuple[float, float]:
        """(x_center, y_center) in crop-local pixels."""
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def half_extents(self) -> tuple[float, float]:
        """(sx, sy) so (x - x_center)/sx, (y - y_center)/sy in [-1, 1]."""
        return ((self.x_max - self.x_min) / 2.0, (self.y_max - self.y_min) / 2.0)

    @classmethod
    def parse(cls, spec: str) -> "RegionSpec":
        """Parse ``"x_min,y_min,x_max,y_max"``."""
        parts = [int(p.strip()) for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"region spec must be 'x_min,y_min,x_max,y_max', got {spec!r}")
        return cls(*parts)


@dataclass
class FrameImage:
    stem: str
    btjd: float
    cal: np.ndarray  # (ny, nx) electrons/s, region crop
    noise: np.ndarray  # (ny, nx) electrons/s, region crop
    bad: np.ndarray  # (ny, nx) bool, True = fatal mask bit set
    exposure_days: float  # header EXPOSURE, [d]


def list_orbit_frames(
    workspace: Path,
    *,
    sector: int,
    orbit_index: int = 1,
) -> tuple[list[FrameRecord], tuple[float, float, int]]:
    """All centroid-indexed frames within one MIT orbit window, BTJD-sorted."""
    all_frames = list_centroid_frames(workspace)
    btjd0, btjd1, orbit_num = sector_orbit_window(sector, orbit_index=orbit_index)
    frames = list_frames_in_orbit(all_frames, btjd0, btjd1)
    return frames, (btjd0, btjd1, orbit_num)


def list_orbit_frames_from_temporal_wcs(
    workspace: Path,
    temporal_wcs_root: Path,
    *,
    sector: int,
    orbit_index: int = 1,
) -> tuple[list[FrameRecord], tuple[float, float, int]]:
    """List hp_d frames from the published temporal-WCS cadence table.

    This is the centroid-free prep path.  The temporal store, rather than
    ``centroids_r1/centroids_index.json``, establishes the usable cadence set.
    ``FrameRecord.phot_path`` is retained only for API compatibility and is
    never read by this mode.
    """
    from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcsStore

    store = TemporalChebWcsStore(temporal_wcs_root)
    btjd0, btjd1, orbit_num = sector_orbit_window(sector, orbit_index=orbit_index)
    table = store.frames
    rows = table.loc[(table["btjd"] >= btjd0) & (table["btjd"] <= btjd1)].sort_values("btjd")
    hp_d_dir = Path(workspace) / "hp_d"
    frames: list[FrameRecord] = []
    valid_suffixes = ("_hp_d.fits", "_hp_d.fits.gz", "_hp_d.fits.fz")
    for row in rows.itertuples(index=False):
        stem = str(row.stem)
        # Match only the final, complete filenames above -- not the diff
        # pipeline's atomic-write temp files (e.g. "{stem}_hp_d.48u3i8jj.fits.part"),
        # which can be left behind, uncleaned, by an interrupted/evicted write and
        # otherwise sort ahead of the real file (ASCII '.' + digit < '.' + "fits").
        # Confirmed root cause of a `TypeError: buffer is too small for requested
        # array` mmap failure on s51 orbit1 (2026-08-27): a stale .part file from
        # 2026-08-26 was picked over the valid, freshly-rewritten .fits.fz sibling.
        candidates = sorted(
            p for p in hp_d_dir.glob(f"{stem}*")
            if p.name in {f"{stem}{suffix}" for suffix in valid_suffixes}
        )
        if not candidates:
            continue
        hp_path = candidates[0]
        hdr = fits.getheader(hp_path, ext=1)
        ny, nx = crop_bounds_from_header(hdr)["shape"]
        frames.append(
            FrameRecord(
                stem=stem,
                btjd=float(row.btjd),
                hp_d_path=hp_path,
                phot_path=Path(),
                crop_shape=(ny, nx),
            )
        )
    if not frames:
        raise RuntimeError(
            f"no hp_d frames in temporal-WCS orbit {orbit_index} under {workspace}"
        )
    return frames, (btjd0, btjd1, orbit_num)


def select_middle_frames(frames: list[FrameRecord], n_frames: int) -> list[FrameRecord]:
    """Contiguous block of ``n_frames`` centred in the (BTJD-sorted) list."""
    n = len(frames)
    if n_frames >= n:
        return list(frames)
    start = (n - n_frames) // 2
    return frames[start : start + n_frames]


def xy_in_region_mask(
    x: np.ndarray,
    y: np.ndarray,
    region: RegionSpec,
    *,
    margin_px: float = 8.0,
) -> np.ndarray:
    """Boolean mask: ``(x, y)`` inside ``region`` expanded by ``margin_px`` (half-open)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_lo, x_hi = region.x_min - margin_px, region.x_max + margin_px
    y_lo, y_hi = region.y_min - margin_px, region.y_max + margin_px
    return (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)


def filter_stars_by_xy(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    region: RegionSpec,
    *,
    margin_px: float = 8.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Keep rows whose provided ``(x, y)`` fall in ``region±margin`` (not catalog columns)."""
    if len(df) != len(x) or len(df) != len(y):
        raise ValueError(
            f"df/x/y length mismatch: len(df)={len(df)}, len(x)={len(x)}, len(y)={len(y)}"
        )
    mask = xy_in_region_mask(x, y, region, margin_px=margin_px)
    return df.loc[mask].reset_index(drop=True), np.asarray(x)[mask], np.asarray(y)[mask]


def primary_candidates_by_mag(
    gaia: pd.DataFrame,
    *,
    tess_mag_range: tuple[float, float] = (7.0, 12.0),
) -> pd.DataFrame:
    """Mag-filter Gaia for primary candidates (no spatial cut); set too_bright/too_faint."""
    sub = gaia.loc[np.isfinite(gaia["tess_mag"])].copy()
    lo, hi = tess_mag_range
    sub["too_bright"] = sub["tess_mag"] < lo
    sub["too_faint"] = sub["tess_mag"] > hi
    return sub.reset_index(drop=True)


def companion_candidates_by_mag(
    gaia: pd.DataFrame,
    *,
    tess_mag_max: float = 13.0,
) -> pd.DataFrame:
    """Mag-filter Gaia for companion pool (no spatial cut)."""
    # Difference-image fit processing is deliberately capped at the inclusive
    # catalog limit: a Tmag=13.0 faint anchor is valid, fainter sources are not.
    mask = np.isfinite(gaia["tess_mag"]) & (gaia["tess_mag"] <= tess_mag_max)
    return gaia.loc[mask].reset_index(drop=True)


def load_gaia_region(
    workspace: Path,
    region: RegionSpec,
    *,
    tess_mag_range: tuple[float, float] = (7.0, 12.0),
    margin_px: float = 8.0,
) -> pd.DataFrame:
    """Gaia rows with catalog crop-local (x, y) inside the region (legacy helper).

    Prefer warmstart ``filter_stars_by_xy`` for membership in ``run_fit``.
    Rows brighter than ``tess_mag_range[0]`` inside the region are also returned
    (flagged ``too_bright``) so callers can mask/drop them explicitly rather than
    silently missing that they exist.
    """
    gaia = load_gaia_catalog(workspace)
    sub = primary_candidates_by_mag(gaia, tess_mag_range=tess_mag_range)
    sub, _, _ = filter_stars_by_xy(
        sub, sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float),
        region, margin_px=margin_px,
    )
    return sub


def load_gaia_companion_pool(
    workspace: Path,
    region: RegionSpec,
    *,
    tess_mag_max: float = 13.0,
    margin_px: float = 8.0,
) -> pd.DataFrame:
    """Gaia rows with catalog (x, y) in the region (legacy); prefer warmstart filter."""
    gaia = load_gaia_catalog(workspace)
    sub = companion_candidates_by_mag(gaia, tess_mag_max=tess_mag_max)
    sub, _, _ = filter_stars_by_xy(
        sub, sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float),
        region, margin_px=margin_px,
    )
    return sub


def merge_star_tables(primary: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Union primary and companion pool rows, preserving primary order on duplicates."""
    merged = pd.concat([primary, pool], ignore_index=True)
    if "source_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["source_id"], keep="first")
    else:
        merged = merged.drop_duplicates(subset=["ra", "dec"], keep="first")
    return merged.reset_index(drop=True)


def primary_to_expanded_index_map(primary: pd.DataFrame, expanded: pd.DataFrame) -> np.ndarray:
    """Map each primary row index to its index in ``expanded``."""
    if "source_id" in primary.columns and "source_id" in expanded.columns:
        sid_to_idx = dict(zip(expanded["source_id"].astype(np.int64), expanded.index))
        return np.array([sid_to_idx[int(s)] for s in primary["source_id"].astype(np.int64)], dtype=int)
    ra_dec_to_idx = {
        (float(r), float(d)): int(i)
        for i, (r, d) in enumerate(zip(expanded["ra"], expanded["dec"]))
    }
    return np.array(
        [ra_dec_to_idx[(float(r), float(d))] for r, d in zip(primary["ra"], primary["dec"])],
        dtype=int,
    )


def load_frame_region(frame: FrameRecord, region: RegionSpec) -> FrameImage:
    """Load one hp_d frame's cal + NOISE planes, cropped to ``region``.

    The MASK extension is ignored (not the correct mask for this fit).
    ``bad`` is all-False so stamp weights are ones.
    """
    with fits.open(frame.hp_d_path) as hdul:
        cal = np.asarray(hdul[1].data, dtype=np.float32)
        noise = np.asarray(hdul[2].data, dtype=np.float32)
        exposure_days = float(hdul[1].header.get("EXPOSURE", float("nan")))

    sl = (slice(region.y_min, region.y_max), slice(region.x_min, region.x_max))
    cal_c = cal[sl].copy()
    noise_c = noise[sl].copy()
    bad = np.zeros(cal_c.shape, dtype=bool)
    return FrameImage(
        stem=frame.stem,
        btjd=frame.btjd,
        cal=cal_c,
        noise=noise_c,
        bad=bad,
        exposure_days=exposure_days,
    )


def load_region_stack(
    frames: list[FrameRecord],
    region: RegionSpec,
    *,
    n_workers: int | None = None,
) -> list[FrameImage]:
    """Load hp_d crops for all frames (threaded I/O by default)."""
    if not frames:
        return []
    n = n_workers if n_workers is not None else min(16, max(1, len(frames)))
    if n <= 1 or len(frames) == 1:
        return [load_frame_region(f, region) for f in frames]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda f: load_frame_region(f, region), frames))


def preload_merged_stars(
    frames: list[FrameRecord],
    gaia_full: pd.DataFrame,
    *,
    n_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all centroids_r1 merges once; keyed by frame stem."""
    from syndiff_pipeline.forward_model._vendor.temporal_wcs_poly.data_io import load_merged_stars  # local import; wired via _bootstrap

    if not frames:
        return {}
    n = n_workers if n_workers is not None else min(16, max(1, len(frames)))

    def _one(frame: FrameRecord) -> tuple[str, pd.DataFrame]:
        return frame.stem, load_merged_stars(frame, gaia_full)

    if n <= 1 or len(frames) == 1:
        return dict(_one(f) for f in frames)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n) as pool:
        return dict(pool.map(_one, frames))
