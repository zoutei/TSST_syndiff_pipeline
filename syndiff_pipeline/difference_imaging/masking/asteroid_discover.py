"""
Discover asteroids in a TESS sector/camera/CCD as seen from TESS (not Earth).

Uses Horizons spacecraft -95 state (xobs-hel) + sbident SB Identification with a
FOV larger than the FFI, samples multiple epochs per orbit, then hard-filters
candidates onto the target chip with tess-ephem.

Orbit epoch sampling reads MIT ``TESS_orbit_times.csv`` (auto-downloaded when
missing or when the requested sector is absent).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.masking.settings import DEFAULT_TESS_ORBIT_TIMES_URL

log = logging.getLogger(__name__)

ORBIT_TIMES_BASENAME = "TESS_orbit_times.csv"

FOV_PAD_DEG = 2.0
VMAG_LIM = 20.0
EPOCH_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

_NAME_NUM = re.compile(r"^(\d+)\s")
_NAME_DESIG = re.compile(r"\(([0-9]{4}\s+[A-Z]{1,2}[0-9]*)\)")
_NAME_PROV = re.compile(r"\(([0-9]{4}\s+[A-Z]{2}[0-9]+)\)")


def default_orbit_times_path(data_root: str | Path) -> Path:
    return Path(data_root) / "catalogs" / ORBIT_TIMES_BASENAME


def _download_orbit_times(dest: Path, url: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading TESS orbit times from %s → %s", url, dest)
    with urlopen(url, timeout=120) as resp:
        payload = resp.read()
    dest.write_bytes(payload)


def _sector_in_orbit_csv(path: Path, sector: int) -> bool:
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path, comment="#")
    except Exception as exc:
        log.warning("Could not read orbit times %s: %s", path, exc)
        return False
    if "Sector" not in df.columns:
        return False
    sectors = df["Sector"].dropna().astype(str).str.strip()
    return str(int(sector)) in set(sectors)


def ensure_tess_orbit_times(
    sector: int,
    dest: str | Path,
    *,
    url: str | None = None,
    force: bool = False,
) -> Path:
    """
    Ensure MIT ``TESS_orbit_times.csv`` exists and lists ``sector``.

    Downloads when the file is missing, ``force`` is set, or the sector is
    absent (re-fetch in case MIT published an update). Returns the path even if
    the sector is still missing after download — callers should fall back.
    """
    dest = Path(dest)
    download_url = url or DEFAULT_TESS_ORBIT_TIMES_URL

    need = force or (not dest.is_file())
    if not need and not _sector_in_orbit_csv(dest, sector):
        log.info(
            "Sector %s not in %s; re-downloading orbit times",
            sector,
            dest,
        )
        need = True

    if need:
        try:
            _download_orbit_times(dest, download_url)
        except Exception as exc:
            log.warning("TESS orbit times download failed (%s)", exc)
            return dest

    if dest.is_file() and not _sector_in_orbit_csv(dest, sector):
        log.warning(
            "Sector %s still absent from %s after download; "
            "discover will fall back to tesswcs.pointings",
            sector,
            dest,
        )
    return dest


def load_orbit_times(sector: int, csv_path: Path) -> pd.DataFrame:
    """Return orbit start/end rows for one sector from TESS_orbit_times.csv."""
    from astropy.time import Time

    df = pd.read_csv(csv_path, comment="#")
    df = df.dropna(subset=["Sector", "Start of Orbit", "End of Orbit"])
    df["Sector"] = df["Sector"].astype(str).str.strip()
    out = df[df["Sector"] == str(int(sector))].copy()
    if out.empty:
        raise ValueError(f"No orbits for sector {sector} in {csv_path}")
    out["start"] = Time(out["Start of Orbit"].astype(str).tolist())
    out["end"] = Time(out["End of Orbit"].astype(str).tolist())
    return out.reset_index(drop=True)


def sample_epochs_from_pointings(sector: int) -> list:
    """Fallback: 5 epochs across tesswcs.pointings Start–End for the sector."""
    from astropy.time import Time
    from tesswcs import pointings

    rows = pointings[pointings["Sector"] == int(sector)]
    if len(rows) == 0:
        raise ValueError(f"Sector {sector} not in tesswcs.pointings")
    t0 = Time(float(rows[0]["Start"]), format="jd", scale="tdb")
    t1 = Time(float(rows[0]["End"]), format="jd", scale="tdb")
    span = (t1 - t0).jd
    return [Time(t0.jd + f * span, format="jd", scale="tdb") for f in EPOCH_FRACTIONS]


def sample_epochs_for_sector(
    sector: int,
    *,
    data_root: str | Path | None = None,
    orbit_times_path: str | Path | None = None,
    orbit_times_url: str | None = None,
) -> list:
    """
    5 epochs per orbit from MIT orbit-times CSV (auto-downloaded as needed).

    If the sector is still missing after download, fall back to pointings.
    """
    from astropy.time import Time

    if orbit_times_path is not None:
        csv_path = Path(orbit_times_path)
    elif data_root is not None:
        csv_path = default_orbit_times_path(data_root)
    else:
        csv_path = Path(ORBIT_TIMES_BASENAME)

    ensure_tess_orbit_times(sector, csv_path, url=orbit_times_url)

    try:
        orbits = load_orbit_times(sector, csv_path)
    except Exception as exc:
        log.warning(
            "Orbit-times CSV unusable for sector %s (%s); using pointings fallback",
            sector,
            exc,
        )
        return sample_epochs_from_pointings(sector)

    times: list = []
    for _, row in orbits.iterrows():
        t0 = row["start"]
        t1 = row["end"]
        span = (t1 - t0).jd
        for f in EPOCH_FRACTIONS:
            times.append(Time(t0.jd + f * span, format="jd", scale="utc"))
    uniq: list = []
    for t in times:
        if not uniq or abs(t.jd - uniq[-1].jd) > 1e-4:
            uniq.append(t)
    return uniq


def tess_xobs_hel(time) -> str:
    """Heliocentric ICRF state of TESS (Horizons -95) for sb_ident xobs-hel."""
    from astroquery.jplhorizons import Horizons

    vec = Horizons(id="-95", location="@sun", epochs=float(time.jd)).vectors()
    return ",".join(
        f"{float(vec[c][0]):.16e}" for c in ("x", "y", "z", "vx", "vy", "vz")
    )


def ffi_fov_center_hwidth(
    sector: int,
    camera: int,
    ccd: int,
    pad_deg: float = FOV_PAD_DEG,
    *,
    crop_bounds: dict | None = None,
):
    """CCD sky center and isotropic half-width (deg) = FFI half-diagonal + pad.

    Science corners come from :func:`science_bounds_1based` (*crop_bounds* when
    provided; otherwise legacy full-chip defaults), converted to 0-based pixels
    for ``tesswcs`` ``pixel_to_world``.
    """
    from syndiff_pipeline.difference_imaging.masking.bounds import science_bounds_1based

    # NumPy ≥2.4 removed np.in1d; tesswcs still calls it.
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]

    from tesswcs import WCS

    bounds = science_bounds_1based(crop_bounds)
    col0 = float(bounds["col_lo"] - 1)
    col1 = float(bounds["col_hi"] - 1)
    row0 = float(bounds["row_lo"] - 1)
    row1 = float(bounds["row_hi"] - 1)

    wcs = WCS.from_sector(sector, camera, ccd)
    corners = wcs.pixel_to_world(
        np.array([col0, col1, col1, col0]),
        np.array([row0, row0, row1, row1]),
    )
    center = wcs.pixel_to_world(
        0.5 * (col0 + col1),
        0.5 * (row0 + row1),
    )
    seps = center.separation(corners).deg
    r_ffi = float(np.max(seps))
    hwidth = r_ffi + float(pad_deg)
    log.info(
        "FOV sector=%s cam=%s ccd=%s: r_ffi=%.3f deg, pad=%.3f → hwidth=%.3f",
        sector,
        camera,
        ccd,
        r_ffi,
        pad_deg,
        hwidth,
    )
    return center, hwidth


def _cache_path(
    cache_dir: Path,
    sector: int,
    camera: int,
    ccd: int,
    time,
    maglim: float,
) -> Path:
    tag = time.utc.isot.replace(":", "").replace("-", "")
    return cache_dir / f"s{sector}_c{camera}_ccd{ccd}_V{maglim:g}_{tag}.json"


def query_sb_ident(
    time,
    center,
    hwidth: float,
    *,
    sector: int,
    camera: int,
    ccd: int,
    maglim: float = VMAG_LIM,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run sbident with TESS xobs-hel; return results as DataFrame."""
    from sbident import SBIdent

    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = _cache_path(cache_dir, sector, camera, ccd, time, maglim)
        if use_cache and cache.exists():
            log.info("Loading cached SB Ident: %s", cache.name)
            payload = json.loads(cache.read_text())
            field = payload.get("data_field")
            if field and field in payload.get("json", {}):
                cols = payload["json"].get(
                    "fields_second" if "second" in field else "fields_first", []
                )
                rows = payload["json"][field]
                return pd.DataFrame(rows, columns=cols)
            if "table" in payload:
                return pd.DataFrame(payload["table"])

    xobs_hel = tess_xobs_hel(time)
    log.info("SB Ident at %s (TESS xobs-hel, V<=%s)", time.utc.isot, maglim)
    sbid = SBIdent(
        location={"xobs-hel": xobs_hel},
        obstime=time,
        fov=center,
        hwidth=hwidth,
        maglim=maglim,
        precision="high",
        filters={"sb-kind": "a"},
    )
    if cache is not None:
        cache.write_text(
            json.dumps(
                {
                    "data_field": sbid.data_field,
                    "json": sbid.json,
                    "uri": sbid.uri,
                    "table": sbid.results.to_pandas().to_dict(orient="list")
                    if sbid.results is not None and len(sbid.results)
                    else {},
                },
                default=str,
            )
        )
    if sbid.results is None or len(sbid.results) == 0:
        return pd.DataFrame()
    return sbid.results.to_pandas()


def horizons_id_from_object_name(name: str) -> str:
    """Map SB Ident 'Object name' to a Horizons small-body id string."""
    name = str(name).strip()
    m_desig = _NAME_DESIG.search(name) or _NAME_PROV.search(name)
    m_num = _NAME_NUM.match(name)
    # Prefer IAU number when present, but not a leading 4-digit year
    # (provisional designations like "2019 AB (2019 AB1)").
    if m_num:
        num = m_num.group(1)
        if not (len(num) == 4 and num.startswith(("19", "20")) and m_desig):
            return num
    if m_desig:
        return m_desig.group(1)
    if "(" in name:
        return name.split("(")[0].strip() or name
    return name


def _parse_vmag(row: pd.Series) -> float:
    for key in ("Visual magnitude (V)", "V", "vmag"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return np.nan


def confirm_on_ccd_with_ephem(
    horizons_ids: list[str],
    sector: int,
    camera: int,
    ccd: int,
    *,
    time_step: float = 0.5,
) -> set[str]:
    """Keep targets that tess-ephem places on (sector, camera, ccd) at least once."""
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]

    from tess_ephem import ephem

    keep: set[str] = set()
    for hid in horizons_ids:
        try:
            df = ephem(str(hid), sector=sector, time_step=time_step)
        except Exception as exc:
            log.warning("ephem confirm failed for %s: %s", hid, exc)
            continue
        if df is None or len(df) == 0:
            continue
        sub = df[(df["camera"] == camera) & (df["ccd"] == ccd)]
        if len(sub):
            keep.add(str(hid))
    return keep


def discover_candidates(
    sector: int,
    camera: int,
    ccd: int,
    *,
    pad_deg: float = FOV_PAD_DEG,
    maglim: float = VMAG_LIM,
    use_cache: bool = True,
    confirm_with_ephem: bool = True,
    data_root: str | Path | None = None,
    orbit_times_path: str | Path | None = None,
    orbit_times_url: str | None = None,
    cache_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    crop_bounds: dict | None = None,
) -> pd.DataFrame:
    """
    Full discovery: multi-epoch SB Ident (TESS observer, padded FOV) → union →
    optional tess-ephem confirmation onto the target camera/CCD.
    """
    if not hasattr(np, "in1d"):
        np.in1d = np.isin  # type: ignore[attr-defined]

    from tesswcs import pointings

    if sector not in set(pointings["Sector"]):
        raise ValueError(f"Sector {sector} not in tesswcs.pointings")

    center, hwidth = ffi_fov_center_hwidth(
        sector, camera, ccd, pad_deg=pad_deg, crop_bounds=crop_bounds
    )
    epochs = sample_epochs_for_sector(
        sector,
        data_root=data_root,
        orbit_times_path=orbit_times_path,
        orbit_times_url=orbit_times_url,
    )
    log.info("Discovery: %d epochs for sector %s", len(epochs), sector)

    if cache_dir is None and out_dir is not None:
        cache_dir = Path(out_dir) / "sb_ident_cache"
    cache_path = Path(cache_dir) if cache_dir else None

    frames: list[pd.DataFrame] = []
    for t in epochs:
        raw = query_sb_ident(
            t,
            center,
            hwidth,
            sector=sector,
            camera=camera,
            ccd=ccd,
            maglim=maglim,
            cache_dir=cache_path,
            use_cache=use_cache,
        )
        if raw.empty:
            log.warning("No SB Ident results at %s", t.utc.isot)
            continue
        raw = raw.copy()
        raw["sample_time"] = t.utc.isot
        name_col = "Object name" if "Object name" in raw.columns else raw.columns[0]
        raw["object_name"] = raw[name_col]
        raw["horizons_id"] = raw["object_name"].map(horizons_id_from_object_name)
        raw["vmag"] = raw.apply(_parse_vmag, axis=1)
        frames.append(raw)

    if not frames:
        return pd.DataFrame(
            columns=["horizons_id", "object_name", "vmag", "n_samples_seen", "sample_time"]
        )

    all_hits = pd.concat(frames, ignore_index=True)
    all_hits = all_hits.sort_values("vmag", ascending=True, na_position="last")
    uniq = all_hits.drop_duplicates(subset=["horizons_id"], keep="first").copy()
    counts = all_hits.groupby("horizons_id")["sample_time"].nunique()
    uniq["n_samples_seen"] = uniq["horizons_id"].map(counts)

    if confirm_with_ephem:
        keep = confirm_on_ccd_with_ephem(
            uniq["horizons_id"].astype(str).tolist(),
            sector,
            camera,
            ccd,
        )
        before = len(uniq)
        uniq = uniq[uniq["horizons_id"].astype(str).isin(keep)].copy()
        log.info(
            "Ephem CCD filter: %d → %d on cam%s/ccd%s",
            before,
            len(uniq),
            camera,
            ccd,
        )

    cols = [
        c
        for c in (
            "horizons_id",
            "object_name",
            "vmag",
            "n_samples_seen",
            "sample_time",
            "hit_row",
            "hit_column",
        )
        if c in uniq.columns
    ]
    out = uniq[cols].copy() if cols else uniq

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        parquet = out_path / "candidates.parquet"
        csv_path = out_path / "candidates.csv"
        out.to_parquet(parquet, index=False)
        out.to_csv(csv_path, index=False)
        log.info("Wrote %d candidates → %s", len(out), parquet)

    return out
