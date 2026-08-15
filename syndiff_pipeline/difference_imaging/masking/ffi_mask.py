"""Helpers to resolve an FFI id → per-epoch mask FITS."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    tess_product_id_from_ffi_path,
)
from syndiff_pipeline.difference_imaging.masking.catalog import MaskCatalog

log = logging.getLogger(__name__)

_DIGITS_ONLY = re.compile(r"^(\d{10,})$")
_TESS_DIGITS = re.compile(r"^(?:tess)?(\d{10,})$", re.IGNORECASE)


def normalize_ffi_product_id(ffi_id: str | int) -> str:
    """
    Normalize an FFI identifier to ``tess<digits>``.

    Accepts a product id (``tess2020019142923``), bare digits, or any path /
    basename that starts with the SPOC product id.
    """
    if isinstance(ffi_id, (int, np.integer)):
        return f"tess{int(ffi_id)}"
    s = str(ffi_id).strip()
    pid = tess_product_id_from_ffi_path(s)
    if pid:
        return pid
    m = _TESS_DIGITS.match(s)
    if m:
        return f"tess{m.group(1)}"
    raise ValueError(f"Cannot parse FFI product id from {ffi_id!r}")


def _manifest_product_id_series(wcs_table: pd.DataFrame) -> pd.Series:
    if "product_id" in wcs_table.columns:
        return wcs_table["product_id"].astype(str)
    col = "filename" if "filename" in wcs_table.columns else "path"
    if col not in wcs_table.columns:
        raise ValueError("wcs_table needs product_id, filename, or path")
    return wcs_table[col].map(lambda x: tess_product_id_from_ffi_path(str(x)) or "")


def lookup_ffi_row(wcs_table: pd.DataFrame, ffi_id: str | int) -> pd.Series:
    """Return the manifest row for *ffi_id* (raises if missing)."""
    pid = normalize_ffi_product_id(ffi_id)
    pids = _manifest_product_id_series(wcs_table)
    hit = wcs_table.loc[pids == pid]
    if hit.empty:
        # also try bare digits match against product ids
        digits = pid.replace("tess", "")
        hit = wcs_table.loc[pids.str.replace("tess", "", regex=False) == digits]
    if hit.empty:
        raise KeyError(f"FFI id {ffi_id!r} ({pid}) not found in manifest")
    return hit.iloc[0]


def btjd_for_ffi_id(wcs_table: pd.DataFrame, ffi_id: str | int) -> float:
    """Look up BTJD for an FFI id from the frame manifest."""
    row = lookup_ffi_row(wcs_table, ffi_id)
    if "btjd" not in row.index or not np.isfinite(float(row["btjd"])):
        raise ValueError(f"No finite btjd for FFI {ffi_id!r}")
    return float(row["btjd"])


def mask_array_for_ffi_id(
    catalog: MaskCatalog,
    ffi_id: str | int,
    *,
    wcs_table: pd.DataFrame | None = None,
    which: str = "full",
    as_bool: bool = False,
) -> np.ndarray:
    """
    Return the mask array for one FFI.

    *ffi_id* may be a product id, bare digits, path, **or** an integer cadence
    when *wcs_table* is omitted (then *ffi_id* is treated as cadence).
    """
    if wcs_table is None:
        # cadence or btjd passthrough
        return catalog.mask_at(ffi_id, which=which, as_bool=as_bool)  # type: ignore[arg-type]
    btjd = btjd_for_ffi_id(wcs_table, ffi_id)
    return catalog.mask_at(btjd, which=which, as_bool=as_bool)


def write_mask_fits_for_ffi(
    catalog: MaskCatalog,
    ffi_id: str | int,
    out_path: str | Path,
    *,
    wcs_table: pd.DataFrame | None = None,
    which: str = "full",
    overwrite: bool = True,
) -> Path:
    """
    Write an int16 mask FITS for one FFI id.

    Header keywords: ``FFI_ID``, ``BTJD`` (when known), ``MASKWHCH``, ``CADENCE``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdr = fits.Header()
    btjd = None
    cadence = None
    pid = None
    if wcs_table is not None:
        pid = normalize_ffi_product_id(ffi_id)
        btjd = btjd_for_ffi_id(wcs_table, ffi_id)
        cadence = catalog.resolve_cadence(btjd)
        mask = catalog.mask_at(btjd, which=which)
        hdr["FFI_ID"] = (pid, "TESS FFI product id")
        hdr["BTJD"] = (float(btjd), "Barycentric TESS JD")
    else:
        mask = catalog.mask_at(ffi_id, which=which)  # type: ignore[arg-type]
        if isinstance(ffi_id, (int, np.integer)) or (
            isinstance(ffi_id, str) and _DIGITS_ONLY.match(str(ffi_id))
        ):
            cadence = int(ffi_id)
            hdr["FFI_ID"] = (str(ffi_id), "cadence or raw id")
        else:
            try:
                pid = normalize_ffi_product_id(ffi_id)
                hdr["FFI_ID"] = (pid, "TESS FFI product id")
            except ValueError:
                hdr["FFI_ID"] = (str(ffi_id), "raw id")

    if cadence is not None:
        hdr["CADENCE"] = (int(cadence), "0-based asteroid cadence index")
    hdr["MASKWHCH"] = (which, "MaskCatalog.mask_at which=")
    hdr["BITPIX"] = 16

    hdu = fits.PrimaryHDU(np.asarray(mask, dtype=np.int16), header=hdr)
    hdu.writeto(out_path, overwrite=overwrite)
    log.info("Wrote mask FITS → %s (ffi=%s cadence=%s)", out_path, pid or ffi_id, cadence)
    return out_path.resolve()


def select_begin_mid_end_ffi_ids(
    wcs_table: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Pick three FFI product ids at the beginning, middle, and end of the sector.

    Rows with non-finite ``btjd`` are dropped; remaining rows are sorted by BTJD.
    """
    df = wcs_table.copy()
    if "btjd" not in df.columns:
        raise ValueError("wcs_table must have a btjd column")
    df = df.loc[np.isfinite(pd.to_numeric(df["btjd"], errors="coerce"))].copy()
    if df.empty:
        raise ValueError("No rows with finite btjd in wcs_table")
    df["_pid"] = _manifest_product_id_series(df)
    df = df.loc[df["_pid"].astype(bool)].sort_values("btjd").reset_index(drop=True)
    n = len(df)
    idxs = (0, n // 2, n - 1)
    return tuple(str(df.iloc[i]["_pid"]) for i in idxs)  # type: ignore[return-value]


def select_asteroid_active_ffi_ids(
    catalog: MaskCatalog,
    wcs_table: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Pick three FFI ids at early / mid / late cadences that have bit-128 pixels.

    Useful when sector begin/mid/end fall outside the asteroid window for the crop.
    """
    if not catalog.has_temporal() or catalog.asteroid_times is None:
        raise ValueError("catalog has no asteroid intervals/times")
    iv = catalog.asteroid_intervals
    assert iv is not None
    active: set[int] = set()
    for lo, hi in zip(iv["cadence_lo"].to_numpy(int), iv["cadence_hi"].to_numpy(int)):
        active.update(range(int(lo), int(hi) + 1))
    if not active:
        raise ValueError("No active asteroid cadences in crop")
    cads = sorted(active)
    picks = (cads[0], cads[len(cads) // 2], cads[-1])
    times = catalog.asteroid_times.set_index("cadence")["btjd"]
    pids: list[str] = []
    df = wcs_table.loc[
        np.isfinite(pd.to_numeric(wcs_table["btjd"], errors="coerce"))
    ].copy()
    df["_pid"] = _manifest_product_id_series(df)
    btjd = pd.to_numeric(df["btjd"], errors="coerce").to_numpy(float)
    for c in picks:
        t = float(times.loc[c])
        j = int(np.argmin(np.abs(btjd - t)))
        pids.append(str(df.iloc[j]["_pid"]))
    return pids[0], pids[1], pids[2]


def write_sector_sample_mask_fits(
    catalog: MaskCatalog,
    wcs_table: pd.DataFrame,
    out_dir: str | Path,
    *,
    which: str = "full",
    prefix: str = "mask",
) -> list[Path]:
    """
    Write three mask FITS files for begin / mid / end FFIs of the sector.

    Filenames: ``{prefix}_{label}_{product_id}.fits`` with
    ``label`` in ``begin``, ``mid``, ``end``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    begin, mid, end = select_begin_mid_end_ffi_ids(wcs_table)
    paths: list[Path] = []
    for label, pid in (("begin", begin), ("mid", mid), ("end", end)):
        dest = out_dir / f"{prefix}_{label}_{pid}.fits"
        paths.append(
            write_mask_fits_for_ffi(
                catalog, pid, dest, wcs_table=wcs_table, which=which
            )
        )
    return paths


def load_catalog_for_event(
    ws_root: str | Path,
    *,
    crop_bounds: dict | None = None,
    data_root: str | Path | None = None,
    sector: int | None = None,
    camera: int | None = None,
    ccd: int | None = None,
) -> MaskCatalog:
    """Load MaskCatalog from an event workspace (+ optional SCC asteroids)."""
    from syndiff_pipeline.difference_imaging.masking.asteroids import load_asteroid_products
    from syndiff_pipeline.difference_imaging.masking.settings import default_asteroid_intervals_dir

    cat = MaskCatalog.from_workspace(ws_root, crop_bounds=crop_bounds)
    if cat.has_temporal() or data_root is None or sector is None:
        return cat
    root = default_asteroid_intervals_dir(data_root, sector, camera or 1, ccd or 1)
    iv, tm = load_asteroid_products(root)
    if iv is None:
        return cat
    return MaskCatalog.from_arrays(
        cat.static,
        tns_table=cat.tns_table,
        asteroid_intervals_ffi=iv,
        asteroid_times=tm,
        crop_bounds=crop_bounds,
    )


def _find_mapping_master_fits(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> Path:
    """Locate mapping master_pixels2skycells FITS for crop bounds."""
    from syndiff_pipeline.common.scc_paths import scc_mapping_dir

    data_root = Path(data_root)
    for factor in (1, 2, 4, 8):
        base = scc_mapping_dir(data_root, sector, camera, ccd, oversampling_factor=factor)
        if not base.is_dir():
            continue
        stem = f"tess_s{int(sector):04d}_{int(camera)}_{int(ccd)}_master_pixels2skycells"
        if factor > 1:
            stem = f"{stem}_os{factor}"
        for name in (f"{stem}.fits.fz", f"{stem}.fits.gz", f"{stem}.fits"):
            path = base / name
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"No mapping master_pixels2skycells FITS under "
        f"{scc_mapping_dir(data_root, sector, camera, ccd, oversampling_factor=1).parent}"
    )


def load_catalog_for_scc_lane(
    lane_root: str | Path,
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> MaskCatalog:
    """
    Load MaskCatalog from an SCC diff lane (static shared mask + asteroid sidecars).

  *lane_root* is the diff lane directory (e.g. ``…/diff_linear``) containing
    ``shared_mask.fits.fz``. Crop bounds come from the SCC mapping master FITS.
    """
    from syndiff_pipeline.common.mapping_grid import load_mapping_grid_from_master
    from syndiff_pipeline.difference_imaging.masking.asteroids import load_asteroid_products
    from syndiff_pipeline.difference_imaging.masking.settings import default_asteroid_intervals_dir
    from syndiff_pipeline.difference_imaging.support.ffi_naming import (
        resolve_pipeline_artifact_path,
    )
    from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME

    lane_root = Path(lane_root)
    sm_path = resolve_pipeline_artifact_path(str(lane_root), SHARED_MASK_FITS_BASENAME)
    if not sm_path:
        alt = lane_root / "shared_mask.fits"
        if alt.is_file():
            sm_path = str(alt)
        else:
            raise FileNotFoundError(
                f"shared_mask not found under {lane_root} "
                f"(expected {SHARED_MASK_FITS_BASENAME})"
            )
    static = np.asarray(fits.getdata(sm_path), dtype=np.int16)

    master = _find_mapping_master_fits(data_root, sector, camera, ccd)
    grid = load_mapping_grid_from_master(master)
    crop_bounds = grid.science_ffi_bounds()

    iv, tm = load_asteroid_products(
        default_asteroid_intervals_dir(data_root, sector, camera, ccd)
    )
    return MaskCatalog.from_arrays(
        static,
        asteroid_intervals_ffi=iv,
        asteroid_times=tm,
        crop_bounds=crop_bounds,
    )


def load_ffi_times_table_for_lane(
    lane_root: str | Path,
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> pd.DataFrame:
    """
    Build an FFI timing table (``filename`` + ``btjd``) for one SCC diff lane.

    Priority: lane ``syndiff_ffi_frames.csv`` → ``wcs/per_ffi_coeffs.csv`` →
    FFI header ``DATE-OBS``.
    """
    from astropy.time import Time

    from syndiff_pipeline.common.scc_paths import scc_ffi_dir
    from syndiff_pipeline.difference_imaging.support.manifest import (
        DEFAULT_MANIFEST_BASENAME,
        load_frame_manifest,
    )

    lane_root = Path(lane_root)

    manifest_path = lane_root / DEFAULT_MANIFEST_BASENAME
    if manifest_path.is_file():
        df = load_frame_manifest(str(lane_root))
        if "btjd" in df.columns and pd.to_numeric(df["btjd"], errors="coerce").notna().any():
            return _ensure_manifest_filename_column(df)

    wcs_csv = lane_root / "wcs" / "per_ffi_coeffs.csv"
    if wcs_csv.is_file():
        df = pd.read_csv(wcs_csv)
        if "stem" in df.columns and "btjd" in df.columns:
            out = df.rename(columns={"stem": "filename"}).copy()
            return _ensure_manifest_filename_column(out)

    ffi_dir = scc_ffi_dir(data_root, sector, camera, ccd)
    if not ffi_dir.is_dir():
        raise FileNotFoundError(f"No FFI timing table and no FFI dir at {ffi_dir}")

    rows: list[dict] = []
    for path in sorted(ffi_dir.glob("tess*.fits*")):
        hdr = fits.getheader(path, ext=1)
        date_obs = hdr.get("DATE-OBS")
        if not date_obs:
            continue
        t = Time(date_obs, format="isot", scale="utc")
        try:
            btjd = float(t.btjd)
        except AttributeError:
            btjd = float(t.jd) - 2457000.0
        rows.append({"filename": path.name, "path": str(path), "btjd": btjd})
    if not rows:
        raise FileNotFoundError(f"No FFI files with DATE-OBS under {ffi_dir}")
    return pd.DataFrame(rows).sort_values("btjd").reset_index(drop=True)


def _ensure_manifest_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "filename" not in out.columns:
        if "path" in out.columns:
            out["filename"] = out["path"].map(lambda p: Path(str(p)).name)
        elif "product_id" in out.columns:
            out["filename"] = out["product_id"].astype(str)
        else:
            raise ValueError("manifest needs filename, path, or product_id")
    return out


def mask_bit_summary(arr: np.ndarray) -> tuple[list[int], int]:
    """Return (active bit flags, count of bit-128 pixels)."""
    bits_present = sorted(
        {
            b
            for v in np.unique(arr)
            for b in (1, 2, 4, 8, 16, 32, 64, 128)
            if int(v) & b
        }
    )
    b128 = int((np.asarray(arr, dtype=np.int16) & 128).astype(bool).sum())
    return bits_present, b128
