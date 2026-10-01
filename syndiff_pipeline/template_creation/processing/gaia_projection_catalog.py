"""Per-projection Gaia catalogues shared by every PS1 skycell of a projection.

Bright-star removal needs the stars just *outside* a cell, so the catalogue
cannot be tied to one TESS SCC footprint.  Instead one catalogue per PS1
projection (all cells of a projection share CRVAL/CDELT/PC) covers the union
of its cells plus a 600 px margin, and is independent of any SCC.

Layout: ``{data_root}/catalogs/gaia_projections/{GAIA_PROJECTION_SCHEME}/proj_{PPPP}.parquet``
with a ``.meta.json`` beside it.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GAIA_PROJECTION_SCHEME = "gaia_dr3_projection_rp18_v1"
GAIA_RELEASE = "gaiadr3"
GAIA_MAGNITUDE_LIMIT = 18.0
# Margin around the projection's cells. Must exceed the largest padded-window
# reach of band_utils.select_catalog_for_cell (R(T) + 10 <= 490 px), so every
# star a cell can select is in its own projection's file whichever other
# projections a run loads.
DEFAULT_MARGIN_PX = 600

_REQUIRED_COLUMNS = ("ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag")

# downloader(ra_coords, dec_coords, magnitude_limit) -> DataFrame of Gaia rows
Downloader = Callable[[np.ndarray, np.ndarray, float], pd.DataFrame]


def projection_id(name_or_projection) -> str:
    """Normalise ``skycell.2486.085`` / ``skycell.2486`` / ``2486`` to ``"2486"``."""
    s = str(name_or_projection).strip()
    m = re.fullmatch(r"(?:skycell\.)?(\d+)(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Cannot parse PS1 projection from {name_or_projection!r}")
    return m.group(1).zfill(4)


def projection_catalog_path(data_root, projection) -> Path:
    """Parquet path of the catalogue for ``projection`` under ``data_root``."""
    return (
        Path(data_root) / "catalogs" / "gaia_projections" / GAIA_PROJECTION_SCHEME
        / f"proj_{projection_id(projection)}.parquet"
    )


def _meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.name + ".meta.json")


def _projection_cells(projection: str) -> pd.DataFrame:
    from syndiff_pipeline.template_creation.orchestration.bundled_assets import skycell_wcs_csv

    table = pd.read_csv(skycell_wcs_csv())
    pid = table["NAME"].astype(str).str.extract(r"skycell\.(\d+)\.")[0].str.zfill(4)
    cells = table[pid == projection]
    if cells.empty:
        raise ValueError(f"Projection {projection} not found in the bundled skycell table")
    return cells


def projection_footprint_polygon(
    projection,
    *,
    margin_px: float = DEFAULT_MARGIN_PX,
    edge_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky polygon (ra, dec in degrees) around all cells of a projection.

    Bounding box, in the projection's tangent plane, of every cell's pixel
    extent, expanded by ``margin_px``; its edges are sampled and mapped to
    RA/Dec.  Depends only on the PS1 tessellation, never on an SCC.
    """
    from astropy.wcs import WCS

    pid = projection_id(projection)
    cells = _projection_cells(pid)
    crval1 = float(cells["CRVAL1"].iloc[0])
    crval2 = float(cells["CRVAL2"].iloc[0])

    us: list[float] = []
    vs: list[float] = []
    for c in cells.itertuples(index=False):
        # FITS 1-based pixel edges of the cell: 0.5 .. NAXIS + 0.5.
        px = np.array([0.5, c.NAXIS1 + 0.5])
        py = np.array([0.5, c.NAXIS2 + 0.5])
        gx, gy = np.meshgrid(px - c.CRPIX1, py - c.CRPIX2)
        u = c.CDELT1 * (c.PC1_1 * gx + c.PC1_2 * gy)
        v = c.CDELT2 * (c.PC2_1 * gx + c.PC2_2 * gy)
        us.extend(u.ravel().tolist())
        vs.extend(v.ravel().tolist())
    cdelt = float(abs(cells["CDELT1"].iloc[0]))
    pad = margin_px * cdelt
    u0, u1 = min(us) - pad, max(us) + pad
    v0, v1 = min(vs) - pad, max(vs) + pad

    t = np.linspace(0.0, 1.0, int(edge_samples), endpoint=False)
    uu = np.concatenate([u0 + (u1 - u0) * t, np.full_like(t, u1), u1 - (u1 - u0) * t, np.full_like(t, u0)])
    vv = np.concatenate([np.full_like(t, v0), v0 + (v1 - v0) * t, np.full_like(t, v1), v1 - (v1 - v0) * t])

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [crval1, crval2]
    w.wcs.crpix = [0.0, 0.0]
    w.wcs.cdelt = [1.0, 1.0]
    ra, dec = w.wcs_pix2world(uu, vv, 1)
    return np.mod(np.asarray(ra, dtype=float), 360.0), np.asarray(dec, dtype=float)


def _default_downloader(backend: str, gaia_credentials_file: Optional[str], parquet_path: Path) -> Downloader:
    """Download via the pancakes flathub / TAP backends (same as SCC catalogues)."""
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "flathub", "tap"}:
        raise ValueError(f"backend must be 'auto', 'flathub', or 'tap'; got {backend!r}")

    def _download(ra, dec, mag_limit):
        from syndiff_pipeline.template_creation.processing import pancakes

        df = None
        if backend in {"auto", "flathub"}:
            try:
                df = pancakes._download_gaia_catalog_flathub(ra, dec, mag_limit)
            except Exception as exc:
                if backend == "flathub":
                    raise
                logger.warning(f"[GaiaProj] flathub download failed ({exc}); falling back to TAP")
        if df is None:
            df = pancakes._download_gaia_catalog_tap(
                ra, dec, mag_limit, str(parquet_path),
                gaia_credentials_file=gaia_credentials_file,
            )
        return df

    return _download


def content_sha256(df: pd.DataFrame) -> str:
    """SHA-256 of the catalogue *content*: rows sorted by source_id, the
    required columns as little-endian float64 (source_id as int64). Unlike the
    parquet bytes it does not depend on the writer library version, so two
    data roots that built the same projection get the same fingerprint."""
    h = hashlib.sha256()
    df = df.sort_values("source_id", kind="mergesort").reset_index(drop=True) if "source_id" in df.columns else df
    if "source_id" in df.columns:
        h.update(np.asarray(df["source_id"].fillna(-1).astype("int64"), dtype="<i8").tobytes())
    for col in _REQUIRED_COLUMNS:
        h.update(col.encode())
        h.update(np.asarray(df[col], dtype="<f8").tobytes())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_projection_catalog(
    data_root,
    projection,
    *,
    backend: str = "auto",
    gaia_credentials_file: Optional[str] = None,
    downloader: Optional[Downloader] = None,
    margin_px: float = DEFAULT_MARGIN_PX,
) -> Path:
    """Return the projection catalogue path, downloading it once if absent.

    An existing parquet + meta pair is never rewritten.  Writes are atomic
    and serialised with an ``fcntl`` lock so concurrent runs download once.
    """
    pid = projection_id(projection)
    path = projection_catalog_path(data_root, pid)
    meta = _meta_path(path)
    if path.is_file() and meta.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            if path.is_file() and meta.is_file():  # another process won the race
                return path

            ra, dec = projection_footprint_polygon(pid, margin_px=margin_px)
            dl = downloader or _default_downloader(backend, gaia_credentials_file, path)
            logger.info(f"[GaiaProj] Downloading Gaia catalogue for projection {pid}")
            df = dl(ra, dec, GAIA_MAGNITUDE_LIMIT)

            from syndiff_pipeline.template_creation.processing import pancakes

            df = pancakes.filter_gaia_dataframe_to_polygon(df, ra, dec)
            from syndiff_pipeline.template_creation.processing.pancakes import GAIA_CATALOG_COLUMNS

            cols = [c for c in GAIA_CATALOG_COLUMNS if c in df.columns]
            df = df[cols].copy()
            if "source_id" in df.columns:
                df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
                df = df.sort_values("source_id", kind="mergesort")
            df = df.reset_index(drop=True)

            tmp = path.with_name(path.name + f".tmp{os.getpid()}")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)

            meta_doc = {
                "scheme": GAIA_PROJECTION_SCHEME,
                "projection": pid,
                "release": GAIA_RELEASE,
                "magnitude_limit": GAIA_MAGNITUDE_LIMIT,
                "margin_px": float(margin_px),
                "polygon_ra": [float(v) for v in ra],
                "polygon_dec": [float(v) for v in dec],
                "n_rows": int(len(df)),
                "content_sha256": content_sha256(df),
                "file_sha256": _sha256_file(path),
            }
            meta_tmp = meta.with_name(meta.name + f".tmp{os.getpid()}")
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(meta_doc, fh, indent=2, sort_keys=True)
            os.replace(meta_tmp, meta)
            logger.info(f"[GaiaProj] Projection {pid}: {len(df)} stars -> {path}")
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    return path


def load_projection_catalog(data_root, projection) -> pd.DataFrame:
    """Load a projection catalogue; raises ``FileNotFoundError`` if absent."""
    path = projection_catalog_path(data_root, projection)
    if not path.is_file():
        raise FileNotFoundError(f"Gaia projection catalogue not found: {path}")
    df = pd.read_parquet(path)
    if "source_id" in df.columns:
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Catalog {path} missing required columns: {missing}")
    return df


def projection_catalog_fingerprint(data_root, projection) -> Optional[str]:
    """``"{scheme}:{content_sha256[:24]}"`` from the meta file, or ``None`` if absent."""
    meta = _meta_path(projection_catalog_path(data_root, projection))
    try:
        with open(meta, encoding="utf-8") as fh:
            doc = json.load(fh)
        return f"{GAIA_PROJECTION_SCHEME}:{str(doc['content_sha256'])[:24]}"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_catalog_for_projections(data_root, projections: Iterable) -> pd.DataFrame:
    """Concatenate several projection catalogues, de-duplicated on ``source_id``."""
    frames = [load_projection_catalog(data_root, p) for p in dict.fromkeys(projection_id(p) for p in projections)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "source_id" in df.columns:
        has_id = df["source_id"].notna()
        df = pd.concat([df[has_id].drop_duplicates("source_id", keep="first"), df[~has_id]])
        df = df.sort_values("source_id", kind="mergesort", na_position="last")
    return df.reset_index(drop=True)
