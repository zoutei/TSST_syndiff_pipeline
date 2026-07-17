"""TNS public catalog download, transient_fixed table, and bit-64 paint."""

from __future__ import annotations

import logging
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import numpy as np
import pandas as pd

from syndiff_pipeline.masking import bits
from syndiff_pipeline.masking.geometry import paint_circles, radius_from_mag, size_limit
from syndiff_pipeline.masking.settings import DEFAULT_TNS_PUBLIC_ZIP_URL

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TNS_PUBLIC_ZIP_URL",
    "TRANSIENT_FIXED_BASENAME",
    "ensure_tns_public_csv",
    "select_from_public_with_tesswcs",
    "build_transient_fixed",
    "paint_tns_bit",
    "load_or_build_transient_fixed",
]

SCI_COL_LO, SCI_COL_HI = 45, 2092
SCI_ROW_LO, SCI_ROW_HI = 1, 2048

TRANSIENT_FIXED_BASENAME = "transient_fixed.parquet"


def ensure_tns_public_csv(
    sector: int,
    dest: str | Path,
    url: str | None = None,
) -> Path:
    """
    Ensure TNS public CSV exists and is fresher than sector pointing end.

    Downloads the WIS zip when missing or ``mtime < tesswcs.pointings[sector].End``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    download_url = url or DEFAULT_TNS_PUBLIC_ZIP_URL

    need_download = not dest.is_file()
    if dest.is_file():
        try:
            from tesswcs import pointings

            row = pointings[pointings["Sector"] == int(sector)]
            if len(row):
                end_jd = float(row[0]["End"])
                # Compare mtime (unix) vs JD end → convert JD to approximate unix
                # JD 2440587.5 = 1970-01-01
                end_unix = (end_jd - 2440587.5) * 86400.0
                if dest.stat().st_mtime < end_unix:
                    need_download = True
                    log.info(
                        "TNS public CSV older than sector %s pointing end; re-downloading",
                        sector,
                    )
        except Exception as exc:
            log.debug("Could not check TNS freshness vs pointings: %s", exc)

    if need_download:
        log.info("Downloading TNS public objects from %s → %s", download_url, dest)
        try:
            with urlopen(download_url, timeout=120) as resp:
                payload = resp.read()
            with zipfile.ZipFile(BytesIO(payload)) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not names:
                    raise RuntimeError(f"No CSV in TNS zip from {download_url}")
                with zf.open(names[0]) as src, open(dest, "wb") as out:
                    out.write(src.read())
        except Exception as exc:
            if dest.is_file():
                log.warning("TNS download failed (%s); using existing %s", exc, dest)
            else:
                raise RuntimeError(
                    f"Failed to download TNS public CSV to {dest}: {exc}"
                ) from exc
    return dest


def select_from_public_with_tesswcs(
    sector: int,
    camera: int,
    ccd: int,
    public_csv: Path | str,
) -> pd.DataFrame:
    """Project TNS public catalog with tesswcs onto one chip."""
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]

    from astropy.coordinates import SkyCoord
    from tesswcs.locate import get_pixel_locations

    path = Path(public_csv)
    df = pd.read_csv(path, header=1)
    if "name_prefix" in df.columns and "name" in df.columns:
        df["source_id"] = (
            df["name_prefix"].astype(str).str.strip()
            + " "
            + df["name"].astype(str).str.strip()
        )
    elif "name" in df.columns:
        df["source_id"] = df["name"].astype(str).str.strip()
    else:
        raise ValueError(f"Cannot find name columns in {path}")

    ra_col = "ra" if "ra" in df.columns else "RA"
    dec_col = "declination" if "declination" in df.columns else "dec"
    df["ra"] = pd.to_numeric(df[ra_col], errors="coerce")
    df["dec"] = pd.to_numeric(df[dec_col], errors="coerce")
    df["mag_tns"] = pd.to_numeric(df.get("discoverymag", df.get("mag")), errors="coerce")
    df = df.dropna(subset=["ra", "dec"])

    if "type" in df.columns:
        bad = df["type"].astype(str).str.lower().isin(
            {"varstar", "cv", "asteroid", "comet", "nan", ""}
        )
        df = df.loc[~bad]

    rows: list[dict] = []
    chunk = 500
    for i in range(0, len(df), chunk):
        sub = df.iloc[i : i + chunk]
        crd = SkyCoord(sub["ra"].to_numpy(), sub["dec"].to_numpy(), unit="deg")
        try:
            hits = get_pixel_locations(crd, sector=sector).to_pandas()
        except Exception as exc:
            log.warning("locate chunk %d failed: %s", i, exc)
            continue
        if hits.empty:
            continue
        cam_c = "Camera" if "Camera" in hits.columns else "camera"
        ccd_c = "CCD" if "CCD" in hits.columns else "ccd"
        idx_c = "Target Index" if "Target Index" in hits.columns else None
        keep = hits[(hits[cam_c] == camera) & (hits[ccd_c] == ccd)]
        if keep.empty:
            continue
        for _, h in keep.iterrows():
            ti = int(h[idx_c]) if idx_c else 0
            if ti < 0 or ti >= len(sub):
                continue
            src = sub.iloc[ti]
            rcol = "Row" if "Row" in h.index else "row"
            ccol = "Column" if "Column" in h.index else "column"
            row_1 = float(h[rcol])
            col_1 = float(h[ccol])
            if not (
                SCI_COL_LO <= col_1 <= SCI_COL_HI and SCI_ROW_LO <= row_1 <= SCI_ROW_HI
            ):
                continue
            rows.append(
                {
                    "source_id": str(src["source_id"]),
                    "ra": float(src["ra"]),
                    "dec": float(src["dec"]),
                    "mag_tns": float(src["mag_tns"])
                    if pd.notna(src["mag_tns"])
                    else np.nan,
                    "x_tesspoint_1based": col_1,
                    "y_tesspoint_1based": row_1,
                    "Sector": sector,
                    "Camera": camera,
                    "CCD": ccd,
                }
            )

    out = pd.DataFrame(rows).drop_duplicates(subset=["source_id"], keep="first")
    log.info("tesswcs TNS → %d SNe on %s/%s/%s", len(out), sector, camera, ccd)
    return out.reset_index(drop=True)


def project_to_ffi(
    ra: np.ndarray,
    dec: np.ndarray,
    sector: int,
    camera: int,
    ccd: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 0-based full-FFI (x, y) using tesswcs WCS."""
    from astropy.coordinates import SkyCoord
    from tesswcs import WCS

    wcs = WCS.from_sector(sector, camera, ccd)
    crd = SkyCoord(ra, dec, unit="deg")
    x, y = wcs.world_to_pixel(crd)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def build_transient_fixed(
    seeds: pd.DataFrame,
    sector: int,
    camera: int,
    ccd: int,
    *,
    n_frames: int | None = None,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Build transient_fixed table (0-based full-FFI x,y) from TNS seeds."""
    if seeds.empty:
        return pd.DataFrame()

    # Prefer refined coords if present; else tesspoint → 0-based; else project
    if "x" in seeds.columns and "y" in seeds.columns:
        x = seeds["x"].to_numpy(float)
        y = seeds["y"].to_numpy(float)
    elif "x_tesspoint_1based" in seeds.columns:
        x = seeds["x_tesspoint_1based"].to_numpy(float) - 1.0
        y = seeds["y_tesspoint_1based"].to_numpy(float) - 1.0
    else:
        x, y = project_to_ffi(
            seeds["ra"].to_numpy(float),
            seeds["dec"].to_numpy(float),
            sector,
            camera,
            ccd,
        )

    x1 = x + 1.0
    y1 = y + 1.0
    on = (
        (x1 >= SCI_COL_LO)
        & (x1 <= SCI_COL_HI)
        & (y1 >= SCI_ROW_LO)
        & (y1 <= SCI_ROW_HI)
    )

    rows = []
    records = seeds.reset_index(drop=True)
    for j, row in records.iterrows():
        if not bool(on[j]):
            continue
        mag = float(row.get("mag_mask", row.get("mag_tns", np.nan)))
        if not np.isfinite(mag):
            mag = float("nan")
        rad = radius_from_mag(mag, scale=scale)
        rows.append(
            {
                "source_type": "sn",
                "source_id": row["source_id"],
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
                "x": float(x[j]),
                "y": float(y[j]),
                "mag_mask": mag,
                "mag_source": row.get("mag_source", "tns"),
                "radius_px": int(rad),
                "refined": bool(row.get("refined", False)),
                "frame_index_lo": 0,
                "frame_index_hi": int(n_frames - 1) if n_frames else -1,
                "sector": sector,
                "camera": camera,
                "ccd": ccd,
            }
        )
    return pd.DataFrame(rows)


def paint_tns_bit(
    mask: np.ndarray,
    tns_table: pd.DataFrame,
    crop_bounds: dict,
) -> np.ndarray:
    """
    Paint bit 64 from transient_fixed (0-based full-FFI x,y → crop-local).
    """
    out = np.asarray(mask, dtype=np.int16).copy()
    if tns_table is None or tns_table.empty:
        return out

    x_min = int(crop_bounds["x_min"])
    y_min = int(crop_bounds["y_min"])
    xs = np.round(tns_table["x"].to_numpy(float) - x_min, 0).astype(np.int64)
    ys = np.round(tns_table["y"].to_numpy(float) - y_min, 0).astype(np.int64)
    radii = tns_table["radius_px"].to_numpy(int).astype(np.int64)
    ind = size_limit(xs, ys, out)
    xs, ys, radii = xs[ind], ys[ind], radii[ind]
    if len(xs) == 0:
        return out

    layer = np.zeros(out.shape, dtype=np.uint8)
    paint_circles(layer, xs, ys, radii)
    out = out | (layer.astype(np.int16) * bits.TNS)
    log.info("  TNS: painted bit 64 on %d sources (%d px)", len(xs), int(layer.sum()))
    return out


def load_or_build_transient_fixed(
    *,
    ws_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    public_csv: Path,
    scale: float = 1.0,
    force: bool = False,
) -> pd.DataFrame:
    """Load ``ws/transient_fixed.parquet`` or build from public CSV + tesswcs."""
    ws_root = Path(ws_root)
    path = ws_root / TRANSIENT_FIXED_BASENAME
    if path.is_file() and not force:
        return pd.read_parquet(path)
    try:
        seeds = select_from_public_with_tesswcs(sector, camera, ccd, public_csv)
    except Exception as exc:
        log.warning("TNS select failed (%s); continuing without bit 64", exc)
        return pd.DataFrame()
    table = build_transient_fixed(seeds, sector, camera, ccd, scale=scale)
    if not table.empty:
        ws_root.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        log.info("Wrote %d TNS rows → %s", len(table), path)
    return table
