"""Per-projection Gaia catalogues shared by every PS1 skycell of a projection.

Bright-star removal needs the stars just *outside* a cell, so the catalogue
cannot be tied to one TESS SCC footprint.  Instead one catalogue per PS1
projection (all cells of a projection share CRVAL/CDELT/PC) covers the union
of its cells plus a 600 px margin, and is independent of any SCC.

Layout: ``{data_root}/catalogs/gaia_projections/{GAIA_PROJECTION_STORE}/proj_{PPPP}.parquet``
with a ``.meta.json`` beside it.  The store holds the **uncut** catalogue;
``removal_subset`` recovers exactly the RP < 18 rows of the legacy
``gaia_dr3_projection_rp18_v1`` store, whose fingerprints are inherited when the
content matches.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GAIA_PROJECTION_SCHEME = "gaia_dr3_projection_rp18_v1"
GAIA_PROJECTION_STORE = "gaia_dr3_projection_all_v1"
GAIA_REMOVAL_RP_LIMIT = 18.0
GAIA_RELEASE = "gaiadr3"
GAIA_MAGNITUDE_LIMIT = 18.0
# Margin around the projection's cells. Must exceed the largest padded-window
# reach of band_utils.select_catalog_for_cell (R(T) + 10 <= 490 px), so every
# star a cell can select is in its own projection's file whichever other
# projections a run loads.
DEFAULT_MARGIN_PX = 600

_REQUIRED_COLUMNS = ("ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag")

# downloader(ra_coords, dec_coords, magnitude_limit) -> DataFrame of Gaia rows
# (magnitude_limit is None for the uncut store)
Downloader = Callable[[np.ndarray, np.ndarray, Optional[float]], pd.DataFrame]


def projection_id(name_or_projection) -> str:
    """Normalise ``skycell.2486.085`` / ``skycell.2486`` / ``2486`` to ``"2486"``."""
    s = str(name_or_projection).strip()
    m = re.fullmatch(r"(?:skycell\.)?(\d+)(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"Cannot parse PS1 projection from {name_or_projection!r}")
    return m.group(1).zfill(4)


def projection_catalog_path(data_root, projection) -> Path:
    """Parquet path of the (uncut) catalogue for ``projection`` under ``data_root``."""
    return (
        Path(data_root) / "catalogs" / "gaia_projections" / GAIA_PROJECTION_STORE
        / f"proj_{projection_id(projection)}.parquet"
    )


def legacy_projection_catalog_path(data_root, projection) -> Path:
    """Parquet path of the legacy RP<18 catalogue (``GAIA_PROJECTION_SCHEME`` store)."""
    return (
        Path(data_root) / "catalogs" / "gaia_projections" / GAIA_PROJECTION_SCHEME
        / f"proj_{projection_id(projection)}.parquet"
    )


def removal_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with ``phot_rp_mean_mag < 18`` (NaN excluded), sorted by source_id."""
    out = df[df["phot_rp_mean_mag"].to_numpy(dtype=float) < GAIA_REMOVAL_RP_LIMIT]
    if "source_id" in out.columns:
        out = out.sort_values("source_id", kind="mergesort")
    return out.reset_index(drop=True)


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


def _read_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "source_id" in df.columns:
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Catalog {path} missing required columns: {missing}")
    return df


def _legacy_inheritance(data_root, pid: str, removal_hash: str) -> Optional[tuple[str, Path]]:
    """Legacy meta hash to inherit, if the legacy file's removal content matches."""
    legacy = legacy_projection_catalog_path(data_root, pid)
    legacy_meta = _meta_path(legacy)
    if not (legacy.is_file() and legacy_meta.is_file()):
        return None
    try:
        with open(legacy_meta, encoding="utf-8") as fh:
            legacy_hash = str(json.load(fh)["content_sha256"])
        legacy_df = _read_catalog(legacy)
        if content_sha256(removal_subset(legacy_df)) != removal_hash:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return legacy_hash, legacy


def ensure_projection_catalog(
    data_root,
    projection,
    *,
    backend: str = "auto",
    gaia_credentials_file: Optional[str] = None,
    downloader: Optional[Downloader] = None,
    margin_px: float = DEFAULT_MARGIN_PX,
) -> Path:
    """Return the uncut projection catalogue path, downloading it once if absent.

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
            logger.info(f"[GaiaProj] Downloading uncut Gaia catalogue for projection {pid}")
            df = dl(ra, dec, None)

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

            # Hashes come from the file as read back, never the in-memory frame.
            back = _read_catalog(path)
            full_hash = content_sha256(back)
            removal_hash = content_sha256(removal_subset(back))
            inherited = _legacy_inheritance(data_root, pid, removal_hash)

            meta_doc = {
                "scheme": GAIA_PROJECTION_SCHEME,
                "scheme_store": GAIA_PROJECTION_STORE,
                "projection": pid,
                "release": GAIA_RELEASE,
                "magnitude_limit": None,
                "margin_px": float(margin_px),
                "polygon_ra": [float(v) for v in ra],
                "polygon_dec": [float(v) for v in dec],
                "n_rows": int(len(back)),
                "n_removal_rows": int(len(removal_subset(back))),
                "content_sha256": full_hash,
                "full_content_sha256": full_hash,
                "removal_content_sha256": removal_hash,
                "removal_fingerprint_sha256": inherited[0] if inherited else removal_hash,
                "file_sha256": _sha256_file(path),
            }
            if inherited:
                meta_doc["fingerprint_inherited_from"] = str(inherited[1])
            meta_tmp = meta.with_name(meta.name + f".tmp{os.getpid()}")
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(meta_doc, fh, indent=2, sort_keys=True)
            os.replace(meta_tmp, meta)
            logger.info(f"[GaiaProj] Projection {pid}: {len(back)} stars -> {path}")
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    return path


def load_projection_catalog(data_root, projection, *, subset: str = "removal") -> pd.DataFrame:
    """Load a projection catalogue (``subset`` = ``"removal"`` RP<18 or ``"all"``).

    Raises ``FileNotFoundError`` if absent.
    """
    if subset not in ("removal", "all"):
        raise ValueError(f"subset must be 'removal' or 'all'; got {subset!r}")
    path = projection_catalog_path(data_root, projection)
    if not path.is_file():
        raise FileNotFoundError(f"Gaia projection catalogue not found: {path}")
    df = _read_catalog(path)
    return removal_subset(df) if subset == "removal" else df


def projection_catalog_fingerprint(data_root, projection) -> Optional[str]:
    """``"{scheme}:{removal_fingerprint_sha256[:24]}"`` from the new-store meta, or
    ``None`` if absent (no fallback to the legacy store)."""
    meta = _meta_path(projection_catalog_path(data_root, projection))
    try:
        with open(meta, encoding="utf-8") as fh:
            doc = json.load(fh)
        h = doc.get("removal_fingerprint_sha256") or doc["content_sha256"]
        return f"{GAIA_PROJECTION_SCHEME}:{str(h)[:24]}"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_catalog_for_projections(data_root, projections: Iterable, *, subset: str = "removal") -> pd.DataFrame:
    """Concatenate several projection catalogues, de-duplicated on ``source_id``."""
    frames = [load_projection_catalog(data_root, p, subset=subset) for p in dict.fromkeys(projection_id(p) for p in projections)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "source_id" in df.columns:
        has_id = df["source_id"].notna()
        df = pd.concat([df[has_id].drop_duplicates("source_id", keep="first"), df[~has_id]])
        df = df.sort_values("source_id", kind="mergesort", na_position="last")
    return df.reset_index(drop=True)



class ProjectionCatalogPrefetcher:
    """Ensure projection catalogues on one background thread, in ``order``.

    ``wait(p)`` blocks until ``p`` is ensured (re-raising its error); a
    projection not yet queued is put at the *front* of the queue.  Errors are
    per projection.
    """

    def __init__(self, data_root, order: Iterable, *, ensure: Callable = None, **ensure_kwargs):
        self.data_root = data_root
        self._ensure = ensure or ensure_projection_catalog
        self._kwargs = ensure_kwargs
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: deque = deque(dict.fromkeys(projection_id(p) for p in order))
        self._events: dict[str, threading.Event] = {p: threading.Event() for p in self._pending}
        self._results: dict[str, Path] = {}
        self._errors: dict[str, BaseException] = {}
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._closed = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "ProjectionCatalogPrefetcher":
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="gaia-prefetch", daemon=True)
                self._thread.start()
        return self

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                pid = self._pending.popleft()
            try:
                result = self._ensure(self.data_root, pid, **self._kwargs)
                with self._lock:
                    self._results[pid] = Path(result)
            except BaseException as exc:  # noqa: BLE001 - reported per projection
                logger.warning(f"[GaiaProj] prefetch of projection {pid} failed: {exc}")
                with self._lock:
                    self._errors[pid] = exc
            self._events[pid].set()

    def wait(self, projection, timeout: Optional[float] = None) -> Path:
        pid = projection_id(projection)
        with self._cv:
            if pid not in self._events:
                if self._closed:
                    raise RuntimeError("prefetcher is closed")
                self._events[pid] = threading.Event()
                self._pending.appendleft(pid)
                self._cv.notify_all()
            ev = self._events[pid]
        if self._thread is None:
            self.start()
        if not ev.wait(timeout):
            raise TimeoutError(f"Gaia catalogue for projection {pid} not ready after {timeout}s")
        with self._lock:
            if pid in self._errors:
                raise self._errors[pid]
            return self._results[pid]

    def catalog(self, projection, subset: str = "removal") -> pd.DataFrame:
        pid = projection_id(projection)
        self.wait(pid)
        key = (pid, subset)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = load_projection_catalog(self.data_root, pid, subset=subset)
            return self._cache[key]

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=30)
        # Release anything still waiting.
        with self._lock:
            for pid, ev in self._events.items():
                if not ev.is_set():
                    self._errors.setdefault(pid, RuntimeError("prefetcher closed before projection was ensured"))
                    ev.set()


# ---------------------------------------------------------------------------
# Gaia DR3 -> PS1 best-neighbour magnitudes
# ---------------------------------------------------------------------------

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
PS1_MAGS_VERSION = "v1"
PS1_MAGS_CHUNK = 500
_PS1_BANDS = ("g", "r", "i", "z", "y")
PS1_MAGS_COLUMNS = (
    "source_id", "ps1_obj_id", "angular_distance", "number_of_neighbours", "number_of_mates",
    *(f"{b}_mean_psf_mag" for b in _PS1_BANDS),
    *(f"{b}_mean_psf_mag_error" for b in _PS1_BANDS),
    "obj_info_flag", "quality_flag", "ps1_match",
)
PS1_MAGS_QUERY_TEMPLATE = (
    "SELECT b.source_id, b.original_ext_source_id AS ps1_obj_id, b.angular_distance, "
    "b.number_of_neighbours, b.number_of_mates, "
    + ", ".join(f"p.{b}_mean_psf_mag, p.{b}_mean_psf_mag_error" for b in _PS1_BANDS)
    + ", p.obj_info_flag, p.quality_flag "
    "FROM gaiadr3.panstarrs1_best_neighbour AS b "
    "JOIN gaiadr2.panstarrs1_original_valid AS p ON p.obj_id = b.original_ext_source_id "
    "WHERE b.source_id IN ({ids})"
)


def ps1_mags_path(data_root, projection) -> Path:
    return (
        Path(data_root) / "catalogs" / "gaia_ps1_best_neighbour" / PS1_MAGS_VERSION
        / f"proj_{projection_id(projection)}.parquet"
    )


def _tap_fetch(ids: list) -> pd.DataFrame:
    """Query the Gaia archive TAP service for one chunk of source_ids."""
    import pyvo

    query = PS1_MAGS_QUERY_TEMPLATE.format(ids=",".join(str(int(i)) for i in ids))
    svc = pyvo.dal.TAPService(GAIA_TAP_URL)
    try:
        res = svc.run_sync(query)
    except Exception as exc:
        logger.warning(f"[GaiaPS1] run_sync failed ({exc}); trying run_async")
        res = svc.run_async(query)
    return res.to_table().to_pandas()


def _normalise_ps1_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(df)))
    for col in PS1_MAGS_COLUMNS:
        if col == "ps1_match":
            continue
        if col in df.columns:
            out[col] = df[col].reset_index(drop=True)
        else:
            out[col] = np.nan
    for col in ("source_id", "ps1_obj_id", "obj_info_flag", "quality_flag", "number_of_neighbours", "number_of_mates"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["ps1_match"] = out["ps1_obj_id"].notna()
    return out[list(PS1_MAGS_COLUMNS)]


def ensure_gaia_ps1_mags(data_root, projection, source_ids, *, fetch: Optional[Callable] = None) -> pd.DataFrame:
    """PS1 best-neighbour magnitudes for ``source_ids``, cached per projection.

    Only IDs not yet cached are fetched (chunks of 500); IDs with no match are
    cached with null PS1 columns and ``ps1_match=False``.  Returns the rows for
    the requested IDs.  Source IDs stay int64 throughout.
    """
    pid = projection_id(projection)
    ids = np.unique(np.asarray([int(i) for i in source_ids], dtype=np.int64))
    path = ps1_mags_path(data_root, pid)
    meta = _meta_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch = fetch or _tap_fetch

    def _cached() -> Optional[pd.DataFrame]:
        if not path.is_file():
            return None
        d = pd.read_parquet(path)
        d["source_id"] = pd.to_numeric(d["source_id"], errors="coerce").astype("Int64")
        return d

    cached = _cached()
    if cached is not None and set(ids.tolist()) <= set(cached["source_id"].astype("int64").tolist()):
        return cached[cached["source_id"].isin(ids.tolist())].reset_index(drop=True)

    with open(path.with_name(path.name + ".lock"), "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            cached = _cached()  # re-read under the lock
            have = set(cached["source_id"].astype("int64").tolist()) if cached is not None else set()
            todo = [int(i) for i in ids if int(i) not in have]
            frames = [cached] if cached is not None else []
            if todo:
                for k in range(0, len(todo), PS1_MAGS_CHUNK):
                    chunk = todo[k:k + PS1_MAGS_CHUNK]
                    got = _normalise_ps1_frame(pd.DataFrame(fetch(chunk)))
                    got = got.drop_duplicates("source_id", keep="first")
                    got = got[got["source_id"].isin(chunk)]
                    missing = sorted(set(chunk) - set(got["source_id"].astype("int64").tolist()))
                    if missing:
                        nm = _normalise_ps1_frame(pd.DataFrame({"source_id": pd.array(missing, dtype="Int64")}))
                        got = pd.concat([got, nm], ignore_index=True)
                    frames.append(got)
                full = pd.concat(frames, ignore_index=True)
                full = full.drop_duplicates("source_id", keep="last").sort_values("source_id", kind="mergesort")
                full = full.reset_index(drop=True)
                tmp = path.with_name(path.name + f".tmp{os.getpid()}")
                full.to_parquet(tmp, index=False)
                os.replace(tmp, path)
                meta_doc = {
                    "projection": pid,
                    "version": PS1_MAGS_VERSION,
                    "n_rows": int(len(full)),
                    "n_matched": int(full["ps1_match"].sum()),
                    "query_template": PS1_MAGS_QUERY_TEMPLATE,
                    "service_url": GAIA_TAP_URL,
                    "chunk_size": PS1_MAGS_CHUNK,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                mtmp = meta.with_name(meta.name + f".tmp{os.getpid()}")
                with open(mtmp, "w", encoding="utf-8") as fh:
                    json.dump(meta_doc, fh, indent=2, sort_keys=True)
                os.replace(mtmp, meta)
                cached = full
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    return cached[cached["source_id"].isin(ids.tolist())].reset_index(drop=True)
