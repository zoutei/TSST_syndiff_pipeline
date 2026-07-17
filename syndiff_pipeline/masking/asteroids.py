"""Asteroid pixel-interval masks (bit 128) — load / optional generate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from syndiff_pipeline.masking.geometry import load_geometry, radius_from_mag
from syndiff_pipeline.masking.settings import default_asteroid_intervals_dir

log = logging.getLogger(__name__)

PIXEL_INTERVALS_BASENAME = "pixel_intervals.parquet"
ASTEROID_FFI_TIMES_BASENAME = "asteroid_ffi_times.parquet"

# Woods et al. 2021 / tess-asteroids; also in packaged mask_geometry.yaml
V_TO_T = 0.671

SCI_ROW_LO, SCI_ROW_HI = 1, 2048
SCI_COL_LO, SCI_COL_HI = 45, 2092
FFI_NROWS, FFI_NCOLS = 2048, 2136


def radius_from_vmag(vmag: float, scale: float = 1.0) -> int:
    """Map visual mag → empirical circle radius via V→T then radius_from_mag."""
    if not np.isfinite(vmag):
        return int(load_geometry().get("faint_radius", 2))
    tmag = float(vmag) - float(load_geometry().get("v_to_t", V_TO_T))
    return radius_from_mag(tmag, scale=scale)


def motion_pad_radius(pixels_per_hour: float, exposure_hours: float) -> int:
    if not np.isfinite(pixels_per_hour):
        return 0
    return int(np.ceil(0.5 * abs(pixels_per_hour) * exposure_hours))


def sector_ffi_cadence_hours(sector: int) -> float:
    return 0.5 if int(sector) < 27 else 10.0 / 60.0


def _circle_pixels(col: float, row: float, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """1-based integer pixel centers inside circle."""
    r = int(max(radius, 0))
    c0 = int(np.floor(col))
    r0 = int(np.floor(row))
    rr = np.arange(r0 - r, r0 + r + 1, dtype=int)
    cc = np.arange(c0 - r, c0 + r + 1, dtype=int)
    yy, xx = np.meshgrid(rr, cc, indexing="ij")
    dist2 = (xx + 0.5 - col) ** 2 + (yy + 0.5 - row) ** 2
    m = dist2 <= (r + 0.5) ** 2
    rows = yy[m]
    cols = xx[m]
    ok = (
        (rows >= SCI_ROW_LO)
        & (rows <= SCI_ROW_HI)
        & (cols >= SCI_COL_LO)
        & (cols <= SCI_COL_HI)
    )
    return rows[ok], cols[ok]


def rasterize_track_to_visits(track: pd.DataFrame) -> pd.DataFrame:
    records_r: list = []
    records_c: list = []
    records_cad: list = []
    for _, row in track.iterrows():
        rad = int(row.get("radius_px", 2))
        pr, pc = _circle_pixels(float(row["column"]), float(row["row"]), rad)
        if len(pr) == 0:
            continue
        cad = int(row["cadence"])
        records_r.append(pr)
        records_c.append(pc)
        records_cad.append(np.full(len(pr), cad, dtype=int))
    if not records_r:
        return pd.DataFrame(columns=["row", "col", "cadence"])
    return pd.DataFrame(
        {
            "row": np.concatenate(records_r),
            "col": np.concatenate(records_c),
            "cadence": np.concatenate(records_cad),
        }
    )


def visits_to_intervals(visits: pd.DataFrame, target_id: str) -> pd.DataFrame:
    if visits.empty:
        return pd.DataFrame(
            columns=["target_id", "row", "col", "cadence_lo", "cadence_hi"]
        )
    visits = visits.sort_values(["row", "col", "cadence"])
    intervals: list[dict] = []
    for (r, c), g in visits.groupby(["row", "col"], sort=False):
        cads = np.unique(g["cadence"].to_numpy(dtype=int))
        lo = cads[0]
        prev = cads[0]
        for cad in cads[1:]:
            if cad == prev + 1:
                prev = cad
                continue
            intervals.append(
                {
                    "target_id": target_id,
                    "row": int(r),
                    "col": int(c),
                    "cadence_lo": int(lo),
                    "cadence_hi": int(prev),
                }
            )
            lo = prev = cad
        intervals.append(
            {
                "target_id": target_id,
                "row": int(r),
                "col": int(c),
                "cadence_lo": int(lo),
                "cadence_hi": int(prev),
            }
        )
    return pd.DataFrame(intervals)


def build_pixel_intervals(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(
            columns=["target_id", "row", "col", "cadence_lo", "cadence_hi"]
        )
    parts: list[pd.DataFrame] = []
    for tid, g in tracks.groupby("target_id"):
        visits = rasterize_track_to_visits(g)
        parts.append(visits_to_intervals(visits, str(tid)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def convert_intervals_to_crop_local(
    intervals: pd.DataFrame,
    crop_bounds: dict,
    shape: tuple[int, int],
) -> pd.DataFrame:
    """
    Convert 1-based full-FFI row/col → crop-local 0-based y,x; drop OOB.
    """
    if intervals is None or intervals.empty:
        return pd.DataFrame(
            columns=["target_id", "y", "x", "cadence_lo", "cadence_hi"]
        )
    x_min = int(crop_bounds["x_min"])
    y_min = int(crop_bounds["y_min"])
    ny, nx = shape
    # 1-based FFI → 0-based FFI → crop-local
    x = intervals["col"].to_numpy(int) - 1 - x_min
    y = intervals["row"].to_numpy(int) - 1 - y_min
    ok = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    out = intervals.loc[ok].copy()
    out["x"] = x[ok]
    out["y"] = y[ok]
    return out[["target_id", "y", "x", "cadence_lo", "cadence_hi"]].reset_index(
        drop=True
    )


def build_ffi_times_from_manifest(wcs_table: pd.DataFrame) -> pd.DataFrame:
    """Prefer event manifest BTJDs; cadence = 0-based row order after time sort."""
    if wcs_table is None or len(wcs_table) == 0:
        return pd.DataFrame(columns=["cadence", "btjd"])
    df = wcs_table.copy()
    btjd_col = None
    for c in ("btjd", "BTJD", "tjd", "TJD", "jd", "JD"):
        if c in df.columns:
            btjd_col = c
            break
    if btjd_col is None:
        log.warning(
            "Manifest has no btjd/jd column; asteroid cadence grid unavailable from manifest"
        )
        return pd.DataFrame(columns=["cadence", "btjd"])
    df = df.sort_values(btjd_col).reset_index(drop=True)
    return pd.DataFrame(
        {
            "cadence": np.arange(len(df), dtype=np.int64),
            "btjd": pd.to_numeric(df[btjd_col], errors="coerce").to_numpy(float),
        }
    )


def remap_track_points_to_manifest_cadence(
    track_points: pd.DataFrame,
    ffi_times: pd.DataFrame,
    *,
    jd_col: str = "time_jd",
    btjd_offset: float = 2457000.0,
) -> pd.DataFrame:
    """
    Re-index track points onto an event ``asteroid_ffi_times`` grid.

    Development track products may use a pointing-grid cadence index that does
    not match the event manifest. Map each point's JD/BTJD to the nearest
    manifest cadence so bit-128 lines up with Hotpants / sample FITS epochs.
    """
    if track_points is None or track_points.empty:
        return track_points
    if ffi_times is None or ffi_times.empty:
        raise ValueError("ffi_times required to remap track cadences")
    times = ffi_times.sort_values("cadence").reset_index(drop=True)
    jds = times["btjd"].to_numpy(float)
    cads = times["cadence"].to_numpy(int)

    out = track_points.copy()
    if jd_col in out.columns:
        t = pd.to_numeric(out[jd_col], errors="coerce").to_numpy(float) - float(
            btjd_offset
        )
    elif "btjd" in out.columns:
        t = pd.to_numeric(out["btjd"], errors="coerce").to_numpy(float)
    else:
        raise ValueError(f"track_points need {jd_col!r} or 'btjd'")

    # nearest cadence
    idx = np.searchsorted(jds, t)
    idx = np.clip(idx, 0, len(jds) - 1)
    for i, tj in enumerate(t):
        if not np.isfinite(tj):
            idx[i] = 0
            continue
        j = idx[i]
        if j > 0 and abs(jds[j - 1] - tj) < abs(jds[j] - tj):
            idx[i] = j - 1
    out["cadence"] = cads[idx]
    # normalize column naming for rasterize_track_to_visits
    if "column" not in out.columns and "col" in out.columns:
        out["column"] = out["col"]
    if "row" not in out.columns and "Row" in out.columns:
        out["row"] = out["Row"]
    if "radius_px" not in out.columns:
        out["radius_px"] = 2
    if "target_id" not in out.columns and "horizons_id" in out.columns:
        out["target_id"] = out["horizons_id"].astype(str)
    return out


def intervals_from_track_points_on_manifest(
    track_points: pd.DataFrame,
    wcs_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remap tracks → pixel_intervals + asteroid_ffi_times on the event grid."""
    ffi_times = build_ffi_times_from_manifest(wcs_table)
    # drop non-finite btjd rows from times
    ffi_times = ffi_times.loc[np.isfinite(ffi_times["btjd"])].reset_index(drop=True)
    ffi_times["cadence"] = np.arange(len(ffi_times), dtype=np.int64)
    remapped = remap_track_points_to_manifest_cadence(track_points, ffi_times)
    intervals = build_pixel_intervals(remapped)
    return intervals, ffi_times



def resolve_cadence_from_btjd(
    times: pd.DataFrame,
    btjd: float,
    *,
    tol: float | None = None,
) -> int | None:
    """Nearest cadence within tolerance (default half median Δbtjd)."""
    if times is None or times.empty or not np.isfinite(btjd):
        return None
    jds = times["btjd"].to_numpy(float)
    cads = times["cadence"].to_numpy(int)
    diffs = np.abs(jds - float(btjd))
    i = int(np.argmin(diffs))
    if tol is None:
        if len(jds) > 1:
            tol = 0.5 * float(np.nanmedian(np.diff(np.sort(jds))))
        else:
            tol = 0.01
    if diffs[i] > tol:
        return None
    return int(cads[i])


def load_asteroid_products(
    intervals_dir: str | Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load SCC pixel_intervals + asteroid_ffi_times if present."""
    root = Path(intervals_dir)
    iv_path = root / PIXEL_INTERVALS_BASENAME
    tm_path = root / ASTEROID_FFI_TIMES_BASENAME
    intervals = pd.read_parquet(iv_path) if iv_path.is_file() else None
    times = pd.read_parquet(tm_path) if tm_path.is_file() else None
    return intervals, times


def _load_or_discover_candidates(
    *,
    intervals_dir: Path,
    sector: int,
    camera: int,
    ccd: int,
    vmag_lim: float,
    data_root: str | Path | None,
    orbit_times_path: str | Path | None,
    orbit_times_url: str | None,
    run_discover: bool,
) -> pd.DataFrame | None:
    """Load candidates.parquet/csv, or run sbident discover when enabled."""
    cand_parquet = intervals_dir / "candidates.parquet"
    cand_csv = intervals_dir / "candidates.csv"
    if cand_parquet.is_file():
        return pd.read_parquet(cand_parquet)
    if cand_csv.is_file():
        return pd.read_csv(cand_csv)

    if not run_discover:
        log.warning(
            "Asteroid generate: no candidates at %s and discover disabled; omit bit 128",
            intervals_dir,
        )
        return None

    try:
        import sbident  # noqa: F401
    except Exception as exc:
        log.warning(
            "Asteroid discover skipped (sbident not importable): %s. "
            "Install with: pip install git+https://github.com/bengebre/sbident",
            exc,
        )
        return None

    from syndiff_pipeline.masking.asteroid_discover import discover_candidates

    log.info(
        "Running asteroid discover (sbident) for s%s c%s ccd%s → %s",
        sector,
        camera,
        ccd,
        intervals_dir,
    )
    try:
        return discover_candidates(
            sector,
            camera,
            ccd,
            maglim=float(vmag_lim),
            data_root=data_root,
            orbit_times_path=orbit_times_path,
            orbit_times_url=orbit_times_url,
            out_dir=intervals_dir,
            cache_dir=intervals_dir / "sb_ident_cache",
        )
    except Exception as exc:
        log.warning("Asteroid discover failed (%s); omit bit 128", exc)
        return None


def try_generate_asteroid_products(
    *,
    intervals_dir: Path,
    sector: int,
    camera: int,
    ccd: int,
    vmag_lim: float,
    ffi_times: pd.DataFrame | None = None,
    data_root: str | Path | None = None,
    orbit_times_path: str | Path | None = None,
    orbit_times_url: str | None = None,
    run_discover: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Attempt generate when tess-ephem / sbident deps present.

    Missing ``candidates.parquet`` → live discover via ``sbident`` (optional).
    Orbit epochs use MIT ``TESS_orbit_times.csv`` (auto-downloaded under
    ``{data_root}/catalogs/``). Returns ``(intervals, times)`` or ``(None, None)``.
    """
    try:
        # Optional heavy deps — keep imports local
        from astropy.time import Time
    except Exception as exc:
        log.warning("Asteroid generate skipped (astropy.time): %s", exc)
        return None, None

    try:
        from tess_ephem import TessEphem  # noqa: F401
    except Exception as exc:
        log.warning(
            "Asteroid generate skipped (tess-ephem not importable): %s. "
            "Install with: pip install tess-ephem",
            exc,
        )
        return None, None

    intervals_dir = Path(intervals_dir)
    intervals_dir.mkdir(parents=True, exist_ok=True)

    if ffi_times is None or ffi_times.empty:
        log.warning(
            "Asteroid generate: no FFI time grid (prefer manifest BTJDs); "
            "falling back to pointing + nominal cadence"
        )
        try:
            from tesswcs import pointings

            row = pointings[pointings["Sector"] == int(sector)][0]
            t0 = Time(float(row["Start"]), format="jd", scale="tdb")
            t1 = Time(float(row["End"]), format="jd", scale="tdb")
            dt_day = sector_ffi_cadence_hours(sector) / 24.0
            n = int(np.floor((t1.jd - t0.jd) / dt_day)) + 1
            jds = t0.jd + dt_day * np.arange(n)
            jds = jds[jds < t1.jd]
            ffi_times = pd.DataFrame(
                {"cadence": np.arange(len(jds), dtype=int), "btjd": jds}
            )
        except Exception as exc:
            log.warning("Asteroid pointing-grid fallback failed: %s", exc)
            return None, None

    candidates = _load_or_discover_candidates(
        intervals_dir=intervals_dir,
        sector=sector,
        camera=camera,
        ccd=ccd,
        vmag_lim=vmag_lim,
        data_root=data_root,
        orbit_times_path=orbit_times_path,
        orbit_times_url=orbit_times_url,
        run_discover=run_discover,
    )
    if candidates is None:
        return None, None
    if candidates.empty:
        empty = pd.DataFrame(
            columns=["target_id", "row", "col", "cadence_lo", "cadence_hi"]
        )
        empty.to_parquet(intervals_dir / PIXEL_INTERVALS_BASENAME, index=False)
        ffi_times.to_parquet(intervals_dir / ASTEROID_FFI_TIMES_BASENAME, index=False)
        return empty, ffi_times

    # Track build using tess-ephem
    from tess_ephem import TessEphem

    times_jd = ffi_times["btjd"].to_numpy(float)
    time_obj = Time(times_jd, format="jd", scale="tdb")
    exp_h = sector_ffi_cadence_hours(sector)
    pieces: list[pd.DataFrame] = []
    ids = candidates["horizons_id"].astype(str).tolist()
    for hid in ids:
        try:
            te = TessEphem.from_sector(hid, sector=sector)
            df = te.predict(time=time_obj)
        except Exception as exc:
            log.warning("ephem failed for %s: %s", hid, exc)
            continue
        if df is None or len(df) == 0:
            continue
        track = df[(df["camera"] == camera) & (df["ccd"] == ccd)].copy()
        if track.empty:
            continue
        track = track.reset_index()
        time_col = "time" if "time" in track.columns else track.columns[0]
        t_jd = Time(track[time_col]).jd
        cadence = np.searchsorted(times_jd, t_jd)
        cadence = np.clip(cadence, 0, len(times_jd) - 1)
        for i, tj in enumerate(t_jd):
            c = cadence[i]
            if c > 0 and abs(times_jd[c - 1] - tj) < abs(times_jd[c] - tj):
                cadence[i] = c - 1
        track["cadence"] = cadence
        track["target_id"] = hid
        vmags = (
            track["vmag"].to_numpy(float)
            if "vmag" in track.columns
            else np.full(len(track), np.nan)
        )
        pph = (
            track["pixels_per_hour"].to_numpy(float)
            if "pixels_per_hour" in track.columns
            else np.zeros(len(track))
        )
        # Keep only V brighter than limit
        keep = (~np.isfinite(vmags)) | (vmags <= float(vmag_lim))
        track = track.loc[keep]
        if track.empty:
            continue
        vmags = vmags[keep]
        pph = pph[keep]
        base = np.array([radius_from_vmag(v) for v in vmags], dtype=int)
        pad = np.array([motion_pad_radius(p, exp_h) for p in pph], dtype=int)
        track["radius_px"] = base + pad
        # tess-ephem uses row/column (1-based science) when present
        if "row" not in track.columns and "Row" in track.columns:
            track["row"] = track["Row"]
        if "column" not in track.columns and "Column" in track.columns:
            track["column"] = track["Column"]
        pieces.append(track)

    if not pieces:
        intervals = pd.DataFrame(
            columns=["target_id", "row", "col", "cadence_lo", "cadence_hi"]
        )
    else:
        tracks = pd.concat(pieces, ignore_index=True)
        intervals = build_pixel_intervals(tracks)

    intervals_dir.mkdir(parents=True, exist_ok=True)
    intervals.to_parquet(intervals_dir / PIXEL_INTERVALS_BASENAME, index=False)
    ffi_times.to_parquet(intervals_dir / ASTEROID_FFI_TIMES_BASENAME, index=False)
    log.info(
        "Wrote asteroid intervals (%d) + times (%d) → %s",
        len(intervals),
        len(ffi_times),
        intervals_dir,
    )
    return intervals, ffi_times


def ensure_asteroid_products(
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    intervals_dir: str | Path | None = None,
    vmag_lim: float = 20.0,
    wcs_table: pd.DataFrame | None = None,
    enabled: bool = True,
    orbit_times_path: str | Path | None = None,
    orbit_times_url: str | None = None,
    run_discover: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load SCC parquet; else try generate (sbident discover + tess-ephem tracks);
    else warn and return (None, None).
    """
    if not enabled:
        return None, None

    root = (
        Path(intervals_dir)
        if intervals_dir
        else default_asteroid_intervals_dir(data_root, sector, camera, ccd)
    )
    intervals, times = load_asteroid_products(root)
    if intervals is not None:
        if times is None and wcs_table is not None:
            times = build_ffi_times_from_manifest(wcs_table)
            if not times.empty:
                root.mkdir(parents=True, exist_ok=True)
                times.to_parquet(root / ASTEROID_FFI_TIMES_BASENAME, index=False)
        return intervals, times

    ffi_times = build_ffi_times_from_manifest(wcs_table) if wcs_table is not None else None
    try:
        return try_generate_asteroid_products(
            intervals_dir=root,
            sector=sector,
            camera=camera,
            ccd=ccd,
            vmag_lim=vmag_lim,
            ffi_times=ffi_times if ffi_times is not None and not ffi_times.empty else None,
            data_root=data_root,
            orbit_times_path=orbit_times_path,
            orbit_times_url=orbit_times_url,
            run_discover=run_discover,
        )
    except Exception as exc:
        log.warning("Asteroid generate failed (%s); omit bit 128", exc)
        return None, None
