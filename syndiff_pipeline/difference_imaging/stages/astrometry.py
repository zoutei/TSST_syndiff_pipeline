"""
Multi-survey supernova astrometry for the diff pipeline.

Resolves TNS names, fetches ZTF IRSA LC + ATLAS transient LC + Gaia alerts,
then refines coordinates with survey_ivw mixing (per-survey clip → IVW combine).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig

import numpy as np
import pandas as pd
import requests
import yaml
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from bs4 import BeautifulSoup

MAS_PER_DEG = 3.6e6
ARCSEC_PER_DEG = 3600.0
MAS_PER_ARCSEC = 1000.0
SNR_FROM_SIGMA_MAG = 1.0857

FINK_RESOLVER_URL = "https://api.ztf.fink-portal.org/api/v1/resolver"
FINK_OBJECTS_URL = "https://api.ztf.fink-portal.org/api/v1/objects"
FINK_CONESEARCH_URL = "https://api.ztf.fink-portal.org/api/v1/conesearch"
IRSA_ZTF_LC_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
IRSA_ZTF_BAD_CATFLAGS_MASK = 32768
IRSA_ZTF_COLLECTION = os.environ.get("IRSA_ZTF_COLLECTION", "ztf_dr24")
IRSA_ZTF_OBJECTS_CATALOG = os.environ.get("IRSA_ZTF_OBJECTS_CATALOG", "ztf_objects_dr24")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_DIR = _REPO_ROOT / "dev"
IRSA_CONFIG_DIR = Path(os.environ.get("IRSA_CONFIG_DIR", _DEV_DIR / "irsa_config"))
IRSA_CREDENTIALS_FILE = Path(
    os.environ.get("IRSA_CREDENTIALS_FILE", IRSA_CONFIG_DIR / "credentials.yaml")
)
TNS_OBJECT_URL = "https://www.wis-tns.org/api/get/object"
TNS_OBJECT_WEB_URL = "https://www.wis-tns.org/object/{objname}"
ATLAS_TRANSIENT_API_URL = "https://psweb.mp.qub.ac.uk/sne/atlas4/api/"
GAIA_ALERT_URL = "https://gsaweb.ast.cam.ac.uk/alerts/alert/{alert_id}/"
GAIA_ALERTS_CSV = Path(
    os.environ.get(
        "GAIA_ALERTS_CSV",
        Path(__file__).resolve().parents[1]
        / "syndiff_pipeline"
        / "resources"
        / "gaia_alerts.csv",
    )
)

ATLAS_CONFIG_DIR = Path(os.environ.get("ATLAS_CONFIG_DIR", _DEV_DIR / "atlas_config"))
ATLAS_API_CONFIG_FILE = Path(
    os.environ.get("ATLAS_API_CONFIG", ATLAS_CONFIG_DIR / "api_config_MINE.yaml")
)
ATLAS_CREDENTIALS_FILE = Path(
    os.environ.get("ATLAS_CREDENTIALS_FILE", ATLAS_CONFIG_DIR / "credentials.yaml")
)

log = logging.getLogger(__name__)

ASTROMETRY_RESULT_FILENAME = "astrometry_result.json"
ASTROMETRY_MIX_PLOT_FILENAME = "astrometry_mix.png"
DEFAULT_SIGMA_MAG_LIMIT = 0.15
DEFAULT_CLIP_N_SIGMA = 3.0

SURVEY_SIGMA_MAG_DEFAULT = {
    "ZTF": 0.12,
    "Gaia": 0.03,
    "ATLAS": 0.06,
    "YSE": 0.08,
}
ASTROMETRY_SOURCES = set(SURVEY_SIGMA_MAG_DEFAULT)

# Fixed astrometric uncertainty for Gaia alert positions [mas].
GAIA_ASTROMETRY_SIGMA_MAS = 50.0


@dataclass
class ResolvedTarget:
    tns_name: str
    seed_ra_deg: float
    seed_dec_deg: float
    redshift: float | None = None
    internal_names: list[str] = field(default_factory=list)
    ztf_object_id: str | None = None
    ztf_irsa_oid: str | None = None
    atlas_id: str | None = None
    atlas_object_id: str | None = None
    gaia_alert_id: str | None = None
    yse_id: str | None = None
    tns_ra_deg: float | None = None
    tns_dec_deg: float | None = None
    resolution_notes: list[str] = field(default_factory=list)


@dataclass
class AstrometryResult:
    ra_deg: float
    dec_deg: float
    cov_mas2: np.ndarray
    n_input: int
    n_after_mag_filter: int
    n_after_clip: int
    clip_history: list[dict[str, Any]]
    ellipse_1sigma: tuple[float, float, float]
    ellipse_3sigma: tuple[float, float, float]
    survey_summaries: list[dict[str, Any]] = field(default_factory=list)


def normalize_tns_name(name: str) -> str:
    name = name.strip()
    for prefix in ("SN ", "AT ", "sn ", "at "):
        if name.lower().startswith(prefix.lower()):
            return name[len(prefix) :].strip()
    return name


def display_tns_name(name: str) -> str:
    base = normalize_tns_name(name)
    return f"SN {base}"


def tns_name_variants(name: str) -> list[str]:
    """TNS/Fink accept different prefixes; AT is often required pre-classification."""
    base = normalize_tns_name(name)
    return [f"AT {base}", f"SN {base}", base]


def parse_tns_ra(ra_str: str) -> float:
    ra_str = str(ra_str).strip()
    if ":" in ra_str:
        return float(Angle(ra_str, unit=u.hourangle).deg)
    return float(ra_str)


def parse_tns_dec(dec_str: str) -> float:
    dec_str = str(dec_str).strip().replace("−", "-")
    if ":" in dec_str:
        sign = -1.0 if dec_str.startswith("-") else 1.0
        parts = dec_str.lstrip("+-").split(":")
        if len(parts) == 3:
            d, m, s = parts
            return sign * float(Angle(f"{d}d{m}m{s}s").deg)
    return float(dec_str)


def normalize_reporting_group(group: str) -> str:
    g = str(group).strip()
    mapping = {
        "GaiaAlerts": "Gaia",
        "Gaia": "Gaia",
        "ATLAS": "ATLAS",
        "ZTF": "ZTF",
        "YSE": "YSE",
        "Pan-STARRS": "YSE",
    }
    return mapping.get(g, g)


def assign_internal_names(target: ResolvedTarget, names: list[str]) -> None:
    for name in names:
        clean = str(name).strip()
        if not clean or clean in target.internal_names:
            continue
        target.internal_names.append(clean)
        upper = clean.upper()
        if upper.startswith("ZTF"):
            target.ztf_object_id = target.ztf_object_id or clean
        elif upper.startswith("ATLAS"):
            target.atlas_id = target.atlas_id or clean
        elif upper.startswith("GAIA"):
            target.gaia_alert_id = target.gaia_alert_id or clean
        elif upper.startswith("PS") and target.yse_id is None:
            target.yse_id = clean


def discovery_sigma_mag(survey: str, discovery_mag: float | None) -> float:
    base = SURVEY_SIGMA_MAG_DEFAULT.get(survey, 0.1)
    if discovery_mag is None or not np.isfinite(discovery_mag):
        return base
    return float(np.clip(base + 0.008 * (discovery_mag - 18.0), 0.03, 0.35))


def angular_offset_arcsec(
    ra1_deg: np.ndarray | float,
    dec1_deg: np.ndarray | float,
    ra0_deg: float,
    dec0_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    ra1 = np.asarray(ra1_deg, dtype=float)
    dec1 = np.asarray(dec1_deg, dtype=float)
    cos_dec = np.cos(np.deg2rad(dec0_deg))
    dra = (ra1 - ra0_deg) * cos_dec * 3600.0
    ddec = (dec1 - dec0_deg) * 3600.0
    return dra, ddec


def resolve_via_fink(tns_name: str, seed_ra: float, seed_dec: float) -> ResolvedTarget:
    """Resolve survey designations via Fink, trying AT/SN/bare TNS name variants."""
    base = normalize_tns_name(tns_name)
    target = ResolvedTarget(
        tns_name=base,
        seed_ra_deg=seed_ra,
        seed_dec_deg=seed_dec,
    )

    frames: list[pd.DataFrame] = []
    for variant in tns_name_variants(base):
        response = requests.post(
            FINK_RESOLVER_URL,
            json={"resolver": "tns", "name": variant},
            timeout=60,
        )
        response.raise_for_status()
        if not response.content:
            continue
        df = pd.read_json(io.BytesIO(response.content))
        if not df.empty:
            frames.append(df)
            target.resolution_notes.append(f"Fink resolver matched '{variant}' ({len(df)} rows).")

    if not frames:
        target.resolution_notes.append("Fink resolver returned no cross-matches for AT/SN/bare name.")
        return target

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["d:internalname", "d:ra", "d:declination"])
    dra, ddec = angular_offset_arcsec(df["d:ra"], df["d:declination"], seed_ra, seed_dec)
    df = df.copy()
    df["sep_arcsec"] = np.hypot(dra, ddec)
    near = df.loc[df["sep_arcsec"] <= 5.0].copy()
    if near.empty:
        near = df.sort_values("sep_arcsec").head(4)

    row = near.sort_values("sep_arcsec").iloc[0]
    target.tns_ra_deg = float(row["d:ra"])
    target.tns_dec_deg = float(row["d:declination"])
    if "d:redshift" in row and pd.notna(row["d:redshift"]):
        target.redshift = float(row["d:redshift"])

    names = sorted({str(v).strip() for v in near["d:internalname"] if str(v).strip()})
    assign_internal_names(target, names)
    target.resolution_notes.append(
        f"Fink internal names within 5 arcsec: {', '.join(names) if names else '(none)'}"
    )
    return target


def fetch_tns_discovery_reports(tns_name: str) -> pd.DataFrame:
    """
    Parse per-survey discovery coordinates from the public TNS object page.

    Returns one row per AT report with columns used by the unified detection table.
    """
    base = normalize_tns_name(tns_name)
    url = TNS_OBJECT_WEB_URL.format(objname=base)
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "syndiff_pipeline_dev/1.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    rows: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if "RA" not in headers or "DEC" not in headers or "Reporting group" not in headers:
            continue

        col = {h: i for i, h in enumerate(headers)}
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < len(headers):
                continue
            try:
                group = normalize_reporting_group(cells[col["Reporting group"]])
                if group not in ASTROMETRY_SOURCES:
                    continue
                ra_deg = parse_tns_ra(cells[col["RA"]])
                dec_deg = parse_tns_dec(cells[col["DEC"]])
                internal = cells[col["Internal name"]] if "Internal name" in col else ""
                disc_mag = float(cells[col["Discovery Mag."]]) if cells[col["Discovery Mag."]] else np.nan
                at_id = cells[col["ID"]] if "ID" in col else ""
            except (KeyError, ValueError, IndexError):
                continue

            sigma_mag = discovery_sigma_mag(group, disc_mag)
            rows.append(
                {
                    "source": group,
                    "survey_id": internal or group,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "sigma_mag": sigma_mag,
                    "mag": disc_mag,
                    "jd": np.nan,
                    "filter_id": -1,
                    "candid": f"tns_at{at_id}_{group.lower()}",
                    "snr_proxy": SNR_FROM_SIGMA_MAG / sigma_mag,
                    "epoch_type": "tns_discovery",
                }
            )

    return pd.DataFrame(rows)


def resolve_via_tns_web(tns_name: str, seed_ra: float, seed_dec: float) -> ResolvedTarget | None:
    """Read the public TNS object page for coordinates and discovery-report names."""
    base = normalize_tns_name(tns_name)
    try:
        reports = fetch_tns_discovery_reports(base)
    except requests.RequestException as exc:
        return ResolvedTarget(
            tns_name=base,
            seed_ra_deg=seed_ra,
            seed_dec_deg=seed_dec,
            resolution_notes=[f"TNS web page fetch failed: {exc}"],
        )

    if reports.empty:
        return None

    target = ResolvedTarget(tns_name=base, seed_ra_deg=seed_ra, seed_dec_deg=seed_dec)
    assign_internal_names(target, reports["survey_id"].astype(str).tolist())

    # Prefer Gaia-updated TNS coordinates when present on the page.
    gaia_rows = reports.loc[reports["source"] == "Gaia"]
    if not gaia_rows.empty:
        target.tns_ra_deg = float(gaia_rows.iloc[0]["ra_deg"])
        target.tns_dec_deg = float(gaia_rows.iloc[0]["dec_deg"])
    else:
        target.tns_ra_deg = float(reports.iloc[0]["ra_deg"])
        target.tns_dec_deg = float(reports.iloc[0]["dec_deg"])

    target.resolution_notes.append(
        f"TNS web page: {len(reports)} discovery-report positions "
        f"({', '.join(sorted(reports['source'].unique()))})."
    )
    return target


def resolve_via_tns_api(
    tns_name: str,
    seed_ra: float,
    seed_dec: float,
    *,
    api_key: str | None = None,
    user_agent: str | None = None,
) -> ResolvedTarget | None:
    """Optional TNS API lookup (requires registered API key)."""
    api_key = api_key or os.environ.get("TNS_API_KEY") or os.environ.get("TNS_BOT_API_KEY")
    if not api_key:
        return None

    base = normalize_tns_name(tns_name)
    headers = {
        "User-Agent": user_agent
        or os.environ.get("TNS_USER_AGENT", "syndiff_pipeline_dev (astrometry@localhost)")
    }
    payload = {"api_key": api_key, "data": json.dumps({"objname": base})}
    response = requests.post(TNS_OBJECT_URL, data=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        return None

    reply = response.json().get("data", {}).get("reply", {})
    if not reply:
        return None

    target = ResolvedTarget(
        tns_name=base,
        seed_ra_deg=seed_ra,
        seed_dec_deg=seed_dec,
        tns_ra_deg=float(reply.get("ra")) if reply.get("ra") is not None else None,
        tns_dec_deg=float(reply.get("declination")) if reply.get("declination") is not None else None,
    )
    if reply.get("redshift") not in (None, ""):
        try:
            target.redshift = float(reply["redshift"])
        except (TypeError, ValueError):
            pass

    internal = reply.get("internal_names") or ""
    names = [n.strip() for n in str(internal).split(",") if n.strip()]
    assign_internal_names(target, names)

    target.resolution_notes.append("TNS API object lookup succeeded.")
    return target


def resolve_target(
    tns_name: str,
    seed_ra: float,
    seed_dec: float,
    *,
    tns_api_key: str | None = None,
) -> ResolvedTarget:
    """Resolve via TNS (API/web) and Fink; conesearch is last-resort for ZTF only."""
    base = normalize_tns_name(tns_name)
    target = ResolvedTarget(tns_name=base, seed_ra_deg=seed_ra, seed_dec_deg=seed_dec)

    tns_api = resolve_via_tns_api(tns_name, seed_ra, seed_dec, api_key=tns_api_key)
    tns_web = resolve_via_tns_web(tns_name, seed_ra, seed_dec)
    fink = resolve_via_fink(tns_name, seed_ra, seed_dec)

    for piece in (tns_api, tns_web, fink):
        if piece is None:
            continue
        if piece.tns_ra_deg is not None and target.tns_ra_deg is None:
            target.tns_ra_deg = piece.tns_ra_deg
            target.tns_dec_deg = piece.tns_dec_deg
        if piece.redshift is not None and target.redshift is None:
            target.redshift = piece.redshift
        assign_internal_names(target, piece.internal_names)
        target.resolution_notes.extend(piece.resolution_notes)

    if target.ztf_object_id is None:
        cone = fetch_ztf_conesearch(seed_ra, seed_dec, radius_arcsec=120.0)
        if not cone.empty:
            sep_arcsec = float(cone["v:separation_degree"].iloc[0]) * 3600.0
            if sep_arcsec <= 5.0:
                target.ztf_object_id = str(cone.iloc[0]["i:objectId"])
                target.resolution_notes.append(
                    f"ZTF ID last-resort conesearch: {target.ztf_object_id} "
                    f"({sep_arcsec:.2f} arcsec from seed)"
                )
            else:
                target.resolution_notes.append(
                    f"Fink conesearch nearest match is {sep_arcsec:.1f} arcsec away; not adopted."
                )

    if target.tns_ra_deg is None:
        target.tns_ra_deg = seed_ra
        target.tns_dec_deg = seed_dec
        target.resolution_notes.append("Using seed coordinates as TNS reference.")

    return target


def _post_fink_json(url: str, payload: dict[str, Any]) -> pd.DataFrame:
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    if not response.content:
        return pd.DataFrame()
    return pd.read_json(io.BytesIO(response.content))


def fetch_ztf_fink_alerts(object_id: str) -> pd.DataFrame:
    df = _post_fink_json(
        FINK_OBJECTS_URL,
        {"objectId": object_id, "output-format": "json"},
    )
    if df.empty:
        return df

    out = pd.DataFrame(
        {
            "source": "ZTF",
            "survey_id": object_id,
            "ra_deg": df["i:ra"].astype(float),
            "dec_deg": df["i:dec"].astype(float),
            "sigma_mag": df["i:sigmapsf"].astype(float),
            "mag": df["i:magpsf"].astype(float) if "i:magpsf" in df else np.nan,
            "jd": df["i:jd"].astype(float) if "i:jd" in df else np.nan,
            "filter_id": df["i:fid"].astype(int) if "i:fid" in df else -1,
            "candid": df["i:candid"].astype(str) if "i:candid" in df else "",
        }
    )
    out["snr_proxy"] = SNR_FROM_SIGMA_MAG / out["sigma_mag"].clip(lower=1e-6)
    return out


fetch_ztf_object_alerts = fetch_ztf_fink_alerts


def _gaia_alerts_csv_path() -> Path:
    env = os.environ.get("GAIA_ALERTS_CSV")
    if env:
        return Path(env)
    try:
        from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
            gaia_alerts_csv,
        )

        return gaia_alerts_csv()
    except (ImportError, FileNotFoundError):
        return GAIA_ALERTS_CSV


@lru_cache(maxsize=1)
def _load_gaia_alerts_table() -> pd.DataFrame:
    """Load bundled Gaia Photometric Science Alerts CSV (gsaweb export)."""
    path = _gaia_alerts_csv_path()
    if not path.is_file():
        return pd.DataFrame()

    with path.open(encoding="utf-8") as handle:
        header = handle.readline().lstrip("#").strip()
    columns = [col.strip() for col in header.split(",")]
    df = pd.read_csv(path, skiprows=1, names=columns)
    df["Name"] = df["Name"].astype(str).str.strip()
    return df


def lookup_gaia_alert_csv(alert_id: str) -> pd.DataFrame:
    """
    Look up one Gaia alert in the bundled CSV.

    Columns: Name, Date, RaDeg, DecDeg, AlertMag, HistoricMag, HistoricStdDev,
    Class, Published, Comment, TNSid. No per-coordinate astrometric errors are
    published; ``HistoricStdDev`` is historic G-band scatter (mag), not σ_RA/σ_Dec.
    """
    table = _load_gaia_alerts_table()
    if table.empty:
        return pd.DataFrame()

    key = str(alert_id).strip()
    row = table.loc[table["Name"].str.lower() == key.lower()]
    if row.empty:
        return pd.DataFrame()

    rec = row.iloc[0]
    ra_deg = float(rec["RaDeg"])
    dec_deg = float(rec["DecDeg"])
    alert_mag = pd.to_numeric(rec.get("AlertMag"), errors="coerce")
    hist_std = pd.to_numeric(rec.get("HistoricStdDev"), errors="coerce")
    if np.isfinite(hist_std) and hist_std > 0:
        sigma_mag = float(hist_std)
    else:
        sigma_mag = discovery_sigma_mag(
            "Gaia",
            float(alert_mag) if np.isfinite(alert_mag) else None,
        )

    jd = np.nan
    try:
        from astropy.time import Time

        jd = float(Time(str(rec["Date"]).strip()).jd)
    except Exception:
        pass

    out = pd.DataFrame(
        [
            {
                "source": "Gaia",
                "survey_id": key,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "sigma_mag": sigma_mag,
                "mag": float(alert_mag) if np.isfinite(alert_mag) else np.nan,
                "jd": jd,
                "filter_id": -1,
                "candid": key,
                "epoch_type": "gaia_alert_csv",
                "snr_proxy": SNR_FROM_SIGMA_MAG / max(sigma_mag, 1e-6),
            }
        ]
    )
    return out


def _fetch_gaia_alert_html(alert_id: str) -> pd.DataFrame:
    """Fallback: scrape gsaweb HTML when the alert is missing from the bundled CSV."""
    url = GAIA_ALERT_URL.format(alert_id=alert_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    ra_deg = dec_deg = None
    for dt in soup.find_all("dt"):
        label = dt.get_text(" ", strip=True).lower()
        dd = dt.find_next_sibling("dd")
        if dd is None:
            continue
        if "ra" in label and "dec" in label:
            nums = re.findall(r"[-+]?\d*\.?\d+", dd.get_text(" ", strip=True))
            if len(nums) >= 2:
                ra_deg, dec_deg = float(nums[0]), float(nums[1])
            break

    if ra_deg is None:
        m = re.search(
            r"target:\s*\"([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\"",
            response.text,
        )
        if m:
            ra_deg, dec_deg = float(m.group(1)), float(m.group(2))

    rows: list[dict[str, Any]] = []
    if ra_deg is not None:
        rows.append(
            {
                "source": "Gaia",
                "survey_id": alert_id,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "sigma_mag": discovery_sigma_mag("Gaia", None),
                "mag": np.nan,
                "jd": np.nan,
                "filter_id": -1,
                "candid": alert_id,
                "epoch_type": "gaia_alert_page",
                "snr_proxy": SNR_FROM_SIGMA_MAG / discovery_sigma_mag("Gaia", None),
            }
        )

    table = soup.find("table")
    if table is not None and ra_deg is not None:
        for tr in table.find_all("tr")[1:]:
            cells = [c.get_text(strip=True) for c in tr.find_all("td")]
            if len(cells) < 3:
                continue
            try:
                jd = float(cells[1])
                mag = float(cells[2])
            except ValueError:
                continue
            rows.append(
                {
                    "source": "Gaia_alert_phot",
                    "survey_id": alert_id,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "sigma_mag": 0.05,
                    "mag": mag,
                    "jd": jd,
                    "filter_id": -1,
                    "candid": f"{alert_id}_{jd:.2f}",
                    "epoch_type": "gaia_alert_phot",
                    "snr_proxy": SNR_FROM_SIGMA_MAG / 0.05,
                }
            )

    return pd.DataFrame(rows)


def fetch_gaia_alert(alert_id: str) -> pd.DataFrame:
    """Fetch Gaia alert astrometry from bundled CSV, with gsaweb HTML fallback."""
    csv_hit = lookup_gaia_alert_csv(alert_id)
    if not csv_hit.empty:
        return csv_hit
    return _fetch_gaia_alert_html(alert_id)


def fetch_ztf_conesearch(ra_deg: float, dec_deg: float, radius_arcsec: float = 120.0) -> pd.DataFrame:
    """Cone search on Fink; API ``radius`` is in degrees. Tries expanding radii."""
    min_radius_deg = max(radius_arcsec / 3600.0, 1.0 / 60.0)
    radii_deg = []
    for r in (min_radius_deg, 0.1, 0.25, 0.5):
        if not radii_deg or r > radii_deg[-1]:
            radii_deg.append(r)

    best = pd.DataFrame()
    for radius_deg in radii_deg:
        df = _post_fink_json(
            FINK_CONESEARCH_URL,
            {"ra": ra_deg, "dec": dec_deg, "radius": radius_deg, "output-format": "json"},
        )
        if df.empty:
            continue
        if "v:separation_degree" in df.columns:
            df = df.sort_values("v:separation_degree")
        best = df
        nearest_arcsec = float(df["v:separation_degree"].iloc[0]) * 3600.0
        if nearest_arcsec <= max(radius_arcsec, 5.0):
            break
    return best


def _load_irsa_credentials(credentials_file: Path | None = None) -> tuple[str | None, str | None]:
    path = credentials_file or IRSA_CREDENTIALS_FILE
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        user = data.get("email") or data.get("username")
        return user, data.get("password")

    user = os.environ.get("IRSA_USER") or os.environ.get("IRSA_EMAIL")
    return user, os.environ.get("IRSA_PASSWORD")


def _irsa_ztf_auth(credentials_file: Path | None = None) -> tuple[str, str] | None:
    user, password = _load_irsa_credentials(credentials_file)
    if user and password:
        return user, password
    return None


def _irsa_ztf_lightcurve_url(**params: str) -> str:
    """Build IRSA ZTF-LC-API URL (space-separated multi-value params)."""
    parts = [IRSA_ZTF_LC_URL + "?"]
    for key, value in params.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={requests.utils.quote(str(value), safe=' ')}")
        parts.append("&")
    return "".join(parts).rstrip("&")


def _fetch_irsa_ztf_table(
    params: dict[str, str],
    *,
    timeout: int = 180,
    credentials_file: Path | None = None,
) -> pd.DataFrame:
    url = _irsa_ztf_lightcurve_url(**params)
    response = requests.get(url, auth=_irsa_ztf_auth(credentials_file), timeout=timeout)
    response.raise_for_status()
    text = response.text.strip()
    if not text or text.startswith("<?xml"):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def _ztf_filtercode_to_fid(filtercode: str) -> int:
    code = str(filtercode).strip().lower()
    return {"zg": 1, "zr": 2, "zi": 3, "g": 1, "r": 2, "i": 3}.get(code, -1)


def angular_separation_arcsec(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    dra, ddec = angular_offset_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg)
    return float(np.hypot(dra, ddec))


def _ztf_object_name(
    resolved: ResolvedTarget,
    tns_reports: pd.DataFrame,
) -> str | None:
    """Return the ZTF object name (e.g. ZTF20aaeuxqk) when known."""
    if resolved.ztf_object_id:
        return resolved.ztf_object_id
    if not tns_reports.empty:
        ztf_disc = tns_reports.loc[
            (tns_reports["source"] == "ZTF") & (tns_reports["epoch_type"] == "tns_discovery")
        ]
        if not ztf_disc.empty:
            return str(ztf_disc.iloc[0]["survey_id"])
    return None


def resolve_ztf_irsa_oid_from_name(
    ztf_name: str,
    *,
    search_radius_arcsec: float = 5.0,
    catalog: str | None = None,
    sep_scale_arcsec: float = 2.0,
    match_center: tuple[float, float] | None = None,
) -> tuple[str | None, float]:
    """
    Map a ZTF object name to an IRSA numeric oid via the objects catalog.

    Uses ``astroquery.ipac.irsa.Irsa.query_region`` on ``ztf_objects_dr24``
    (name resolver → 5″ cone). Candidates are ranked by
    ``nobs * exp(-(sep / sep_scale)^2)`` relative to ``match_center`` or the
    name-resolved position.
    """
    import astropy.units as au
    from astroquery.ipac.irsa import Irsa

    catalog = catalog or IRSA_ZTF_OBJECTS_CATALOG
    table = Irsa.query_region(
        coordinates=ztf_name,
        catalog=catalog,
        spatial="Cone",
        radius=search_radius_arcsec * au.arcsec,
    )
    if len(table) == 0:
        return None, float("inf")

    if match_center is not None:
        center = SkyCoord(match_center[0], match_center[1], unit="deg")
    else:
        center = SkyCoord.from_name(ztf_name)

    best_oid: str | None = None
    best_sep = float("inf")
    best_score = -1.0
    for row in table:
        ra = float(row["ra"])
        dec = float(row["dec"])
        sep = float(center.separation(SkyCoord(ra, dec, unit="deg")).arcsec)
        score = int(row["nobs"]) * np.exp(-((sep / sep_scale_arcsec) ** 2))
        if score > best_score:
            best_score = score
            best_sep = sep
            best_oid = str(int(row["oid"]))

    return best_oid, best_sep


def resolve_ztf_irsa_oid_tap(
    oid: int | str,
    *,
    catalog: str | None = None,
) -> tuple[str | None, float]:
    """Look up a numeric IRSA oid directly via TAP on the objects table."""
    from astroquery.ipac.irsa import Irsa

    catalog = catalog or IRSA_ZTF_OBJECTS_CATALOG
    adql = f"SELECT oid, ra, dec, nobs FROM {catalog} WHERE oid = {int(oid)}"
    table = Irsa.query_tap(adql).to_table()
    if len(table) == 0:
        return None, float("inf")
    return str(int(table["oid"][0])), 0.0


def fetch_ztf_irsa_lightcurve(
    oid: str,
    *,
    bad_catflags_mask: int = IRSA_ZTF_BAD_CATFLAGS_MASK,
    credentials_file: Path | None = None,
) -> pd.DataFrame:
    """Fetch full ZTF lightcurve for one IRSA object id (per-epoch ra/dec/mag)."""
    params: dict[str, str] = {
        "ID": str(oid),
        "FORMAT": "csv",
        "BAD_CATFLAGS_MASK": str(bad_catflags_mask),
        "COLLECTION": IRSA_ZTF_COLLECTION,
    }

    df = _fetch_irsa_ztf_table(params, timeout=180, credentials_file=credentials_file)
    if df.empty:
        return df

    required = {"ra", "dec", "magerr"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    mag = df["mag"].astype(float) if "mag" in df.columns else np.nan
    valid = (
        df["ra"].notna()
        & df["dec"].notna()
        & df["magerr"].astype(float).gt(0)
    )
    if isinstance(mag, pd.Series):
        valid &= mag.notna() & np.isfinite(mag)
    df = df.loc[valid].copy()
    if df.empty:
        return df

    mjd = df["mjd"].astype(float) if "mjd" in df.columns else np.nan
    jd = mjd + 2400000.5 if isinstance(mjd, pd.Series) else np.nan
    filtercode = df["filtercode"].astype(str) if "filtercode" in df.columns else ""
    expid = df["expid"].astype(str) if "expid" in df.columns else ""

    out = pd.DataFrame(
        {
            "source": "ZTF",
            "survey_id": str(oid),
            "ra_deg": df["ra"].astype(float),
            "dec_deg": df["dec"].astype(float),
            "sigma_mag": df["magerr"].astype(float),
            "mag": mag,
            "jd": jd,
            "filter_id": filtercode.map(_ztf_filtercode_to_fid),
            "candid": [
                f"{oid}_{e}" if e else f"{oid}_{i}"
                for i, e in enumerate(expid.tolist())
            ],
        }
    )
    out["snr_proxy"] = SNR_FROM_SIGMA_MAG / out["sigma_mag"].clip(lower=1e-6)
    return out


def _load_atlas_credentials(credentials_file: Path | None = None) -> tuple[str | None, str | None]:
    path = credentials_file or ATLAS_CREDENTIALS_FILE
    if path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("username"), data.get("password")

    return os.environ.get("ATLAS_USERNAME"), os.environ.get("ATLAS_PASSWORD")


def _refresh_atlas_api_token(
    config_path: Path | None = None,
    *,
    credentials_file: Path | None = None,
) -> str:
    """Refresh ATLAS Transient API token and persist to api_config_MINE.yaml."""
    config_path = config_path or ATLAS_API_CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"ATLAS API config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_url = cfg.get("base_url") or ATLAS_TRANSIENT_API_URL
    if not base_url.endswith("/"):
        base_url += "/"

    username, password = _load_atlas_credentials(credentials_file)
    if not username or not password:
        raise RuntimeError(
            f"ATLAS credentials missing. Set {ATLAS_CREDENTIALS_FILE} or "
            "ATLAS_USERNAME/ATLAS_PASSWORD."
        )

    response = requests.post(
        f"{base_url}auth-token/",
        data={"username": username, "password": password},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ATLAS auth-token HTTP {response.status_code}: {response.text[:200]}")

    token = response.json().get("token")
    if not token:
        raise RuntimeError("ATLAS auth-token response missing token")

    cfg["token"] = token
    cfg["base_url"] = base_url
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return token


def _atlas_api_config_path() -> Path | None:
    if ATLAS_API_CONFIG_FILE.is_file():
        return ATLAS_API_CONFIG_FILE
    return None


def resolve_atlas_object_id(
    resolved: ResolvedTarget,
    *,
    api_config_file: str | None = None,
    radius_arcsec: float = 30.0,
) -> tuple[str | None, str]:
    """
    Resolve the 19-digit ATLAS object id via ConeSearch on the Transient Server.

    Prefers a match to ``resolved.atlas_id`` (e.g. ATLAS20jul) when present.
    """
    config_path = api_config_file or (_atlas_api_config_path() and str(ATLAS_API_CONFIG_FILE))
    if not config_path:
        return None, "ATLAS API config file missing"

    from atlasapiclient import client as atlasapi

    try:
        cone = atlasapi.ConeSearch(
            payload={
                "ra": resolved.seed_ra_deg,
                "dec": resolved.seed_dec_deg,
                "radius": radius_arcsec,
                "requestType": "nearest",
            },
            get_response=True,
            api_config_file=config_path,
            auto_refresh_fl=False,
        )
    except Exception as exc:
        msg = str(exc)
        if "expired" in msg.lower() or "invalid token" in msg.lower():
            _refresh_atlas_api_token(Path(config_path))
            cone = atlasapi.ConeSearch(
                payload={
                    "ra": resolved.seed_ra_deg,
                    "dec": resolved.seed_dec_deg,
                    "radius": radius_arcsec,
                    "requestType": "nearest",
                },
                get_response=True,
                api_config_file=config_path,
                auto_refresh_fl=False,
            )
        else:
            return None, f"ATLAS cone search failed: {exc}"

    data = cone.response_data
    if not isinstance(data, dict) or "object" not in data:
        return None, "ATLAS cone search returned no object"

    atlas_numeric = str(int(data["object"]))
    objectname = str(data.get("objectname", ""))

    if resolved.atlas_id and objectname and objectname.upper() != resolved.atlas_id.upper():
        return (
            None,
            f"ATLAS cone nearest is {objectname}, expected {resolved.atlas_id}",
        )

    return (
        atlas_numeric,
        f"ATLAS cone: {objectname or atlas_numeric} "
        f"({float(data.get('separation', 0)):.2f} arcsec)",
    )


def _atlas_lc_to_detections(
    lc_rows: list[dict[str, Any]],
    *,
    atlas_object_id: str,
    atlas_designation: str | None,
) -> pd.DataFrame:
    """Convert ATLAS transient-server light-curve rows to detection epochs."""
    if not lc_rows:
        return pd.DataFrame()

    df = pd.DataFrame(lc_rows)
    if df.empty:
        return df

    mag = pd.to_numeric(df.get("mag"), errors="coerce")
    magerr = pd.to_numeric(df.get("magerr"), errors="coerce")
    valid = mag.notna() & magerr.notna() & (mag > 0) & (magerr > 0)
    df = df.loc[valid].copy()
    if df.empty:
        return pd.DataFrame()

    survey_id = atlas_designation or atlas_object_id
    out = pd.DataFrame(
        {
            "source": "ATLAS",
            "survey_id": survey_id,
            "ra_deg": pd.to_numeric(df["ra"], errors="coerce"),
            "dec_deg": pd.to_numeric(df["dec"], errors="coerce"),
            "sigma_mag": pd.to_numeric(df["magerr"], errors="coerce").clip(lower=1e-3),
            "mag": pd.to_numeric(df["mag"], errors="coerce"),
            "jd": pd.to_numeric(df["mjd"], errors="coerce") + 2400000.5,
            "filter_id": -1,
            "candid": df["id"].astype(str) + "_" + df["mjd"].astype(str),
            "epoch_type": "atlas_transient_lc",
        }
    )
    out["snr_proxy"] = SNR_FROM_SIGMA_MAG / out["sigma_mag"].clip(lower=1e-6)
    return out


def fetch_atlas_transient_detections(
    resolved: ResolvedTarget,
    *,
    mjd_min: float = 58000.0,
    api_config_file: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch per-detection RA/Dec centroids from the ATLAS Transient Server via atlasapiclient.

    Uses ``RequestSingleSourceData`` light-curve rows (fields ``ra``, ``dec``, ``mag``,
    ``magerr``, ``mjd``). Config: ``dev/atlas_config/api_config_MINE.yaml``.
    """
    config_path = api_config_file or (_atlas_api_config_path() and str(ATLAS_API_CONFIG_FILE))
    if not config_path:
        return pd.DataFrame(), "ATLAS transient API skipped (no api_config_MINE.yaml)."

    atlas_object_id = resolved.atlas_object_id
    status_parts: list[str] = []

    if not atlas_object_id:
        atlas_object_id, cone_status = resolve_atlas_object_id(
            resolved,
            api_config_file=config_path,
        )
        status_parts.append(cone_status)
        if not atlas_object_id:
            return pd.DataFrame(), "; ".join(status_parts)

    from atlasapiclient import client as atlasapi

    def _fetch() -> atlasapi.RequestSingleSourceData:
        return atlasapi.RequestSingleSourceData(
            atlas_id=atlas_object_id,
            mjdthreshold=mjd_min,
            get_response=True,
            api_config_file=config_path,
            auto_refresh_fl=False,
        )

    try:
        client = _fetch()
    except Exception as exc:
        msg = str(exc)
        if "expired" in msg.lower() or "invalid token" in msg.lower():
            _refresh_atlas_api_token(Path(config_path))
            try:
                client = _fetch()
            except Exception as retry_exc:
                return pd.DataFrame(), f"ATLAS fetch failed after token refresh: {retry_exc}"
        else:
            return pd.DataFrame(), f"ATLAS fetch failed: {exc}"

    if not client.response_data:
        return pd.DataFrame(), "; ".join(status_parts + ["ATLAS returned empty response"])

    payload = client.response_data[0]
    designation = None
    if isinstance(payload.get("object"), dict):
        designation = payload["object"].get("atlas_designation")

    out = _atlas_lc_to_detections(
        payload.get("lc", []),
        atlas_object_id=atlas_object_id,
        atlas_designation=designation or resolved.atlas_id,
    )
    if out.empty:
        return out, "; ".join(status_parts + ["ATLAS light curve had no valid detections"])

    resolved.atlas_object_id = atlas_object_id
    status_parts.append(
        f"ATLAS {designation or atlas_object_id}: {len(out)} transient-server centroid epochs"
    )
    return out, "; ".join(status_parts)


def astrometry_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only epochs that contribute independent astrometric measurements."""
    if df.empty:
        return df
    return df.loc[df["source"].isin(ASTROMETRY_SOURCES)].copy()


def compile_detection_table(
    resolved: ResolvedTarget,
    *,
    irsa_credentials_file: Path | None = None,
    atlas_api_config_file: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch survey epochs (ZTF IRSA LC, ATLAS LC, Gaia) into one detection table."""
    statuses: list[str] = []
    parts: list[pd.DataFrame] = []

    try:
        tns_reports = fetch_tns_discovery_reports(resolved.tns_name)
    except requests.RequestException as exc:
        tns_reports = pd.DataFrame()
        statuses.append(f"TNS discovery reports unavailable: {exc}")

    if not tns_reports.empty:
        parts.append(tns_reports)
        statuses.append(
            "TNS discovery reports: "
            + ", ".join(
                f"{row['source']}={row['survey_id']}"
                for _, row in tns_reports.iterrows()
            )
        )

    if resolved.tns_ra_deg is not None:
        parts.append(
            pd.DataFrame(
                [
                    {
                        "source": "TNS",
                        "survey_id": display_tns_name(resolved.tns_name),
                        "ra_deg": resolved.tns_ra_deg,
                        "dec_deg": resolved.tns_dec_deg,
                        "sigma_mag": 0.5,
                        "mag": np.nan,
                        "jd": np.nan,
                        "filter_id": -1,
                        "candid": "tns_current",
                        "snr_proxy": SNR_FROM_SIGMA_MAG / 0.5,
                        "epoch_type": "tns_current",
                    }
                ]
            )
        )
        statuses.append("Added current TNS object coordinate as reference.")

    if ztf_name := _ztf_object_name(resolved, tns_reports):
        try:
            match_center: tuple[float, float] | None = None
            fink_probe = fetch_ztf_fink_alerts(ztf_name)
            if not fink_probe.empty:
                match_center = (
                    float(fink_probe["ra_deg"].median()),
                    float(fink_probe["dec_deg"].median()),
                )
            oid, sep = resolve_ztf_irsa_oid_from_name(ztf_name, match_center=match_center)
        except Exception as exc:
            oid, sep = None, float("inf")
            statuses.append(f"ZTF IRSA oid lookup failed ({exc}).")
        if oid:
            resolved.ztf_irsa_oid = oid
            try:
                ztf = fetch_ztf_irsa_lightcurve(oid, credentials_file=irsa_credentials_file)
            except requests.RequestException as exc:
                ztf = pd.DataFrame()
                statuses.append(f"ZTF IRSA oid {oid}: fetch failed ({exc}).")
            if not ztf.empty:
                ztf = ztf.copy()
                ztf["epoch_type"] = "ztf_irsa_lc"
                parts.append(ztf)
                statuses.append(
                    f"ZTF IRSA {ztf_name} → oid {oid}: {len(ztf)} LC epochs "
                    f"(catalog match {sep:.2f} arcsec)."
                )
    else:
        statuses.append("ZTF IRSA: no ZTF object name available.")

    if resolved.gaia_alert_id:
        gaia_fetch_failed = False
        try:
            gaia = fetch_gaia_alert(resolved.gaia_alert_id)
        except requests.RequestException as exc:
            gaia = pd.DataFrame()
            gaia_fetch_failed = True
            statuses.append(f"Gaia alert {resolved.gaia_alert_id}: fetch failed ({exc})")
        if gaia.empty:
            if not gaia_fetch_failed:
                statuses.append(f"Gaia alert {resolved.gaia_alert_id}: parse failed.")
        else:
            gaia_astrom = gaia[gaia["source"] == "Gaia"].copy()
            if not gaia_astrom.empty:
                parts.append(gaia_astrom)
            source = (
                "bundled CSV"
                if not gaia_astrom.empty
                and (gaia_astrom["epoch_type"] == "gaia_alert_csv").any()
                else "gsaweb"
            )
            statuses.append(
                f"Gaia alert {resolved.gaia_alert_id} ({source}): {len(gaia)} rows."
            )

    atlas_df, atlas_status = fetch_atlas_transient_detections(
        resolved,
        api_config_file=atlas_api_config_file,
    )
    statuses.append(atlas_status)
    if not atlas_df.empty:
        parts.append(atlas_df)

    if not parts:
        return pd.DataFrame(), statuses

    detections = pd.concat(parts, ignore_index=True)
    if "epoch_type" not in detections.columns:
        detections["epoch_type"] = "unknown"
    dra, ddec = angular_offset_arcsec(
        detections["ra_deg"],
        detections["dec_deg"],
        resolved.seed_ra_deg,
        resolved.seed_dec_deg,
    )
    detections["dra_arcsec"] = dra
    detections["ddec_arcsec"] = ddec
    detections["sep_arcsec"] = np.hypot(dra, ddec)
    detections["weight"] = 1.0 / np.square(detections["sigma_mag"].clip(lower=1e-6))
    return detections, statuses


def _select_atlas_lc_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all ATLAS transient-server LC epochs (drop TNS discovery when LC exists)."""
    atlas = df.loc[df["source"] == "ATLAS"]
    if atlas.empty:
        return df.copy()
    lc = atlas.loc[atlas["epoch_type"] == "atlas_transient_lc"]
    non_atlas = df.loc[df["source"] != "ATLAS"]
    if lc.empty:
        tns = atlas.loc[atlas["epoch_type"] == "tns_discovery"]
        return pd.concat([non_atlas, tns], ignore_index=True)
    return pd.concat([non_atlas, lc], ignore_index=True)


def exclude_tns_astrometry_when_lc_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop TNS discovery coordinates when a survey has real light-curve astrometry.

    TNS reports are used for name resolution; LC epochs supersede TNS RA/Dec.
    """
    if df.empty:
        return df

    out = df.copy()
    drop_idx: list[Any] = []

    atlas_lc = out.loc[
        (out["source"] == "ATLAS") & (out["epoch_type"] == "atlas_transient_lc")
    ]
    if not atlas_lc.empty:
        drop_idx.extend(
            out.index[
                (out["source"] == "ATLAS") & (out["epoch_type"] == "tns_discovery")
            ].tolist()
        )

    ztf_lc = out.loc[
        (out["source"] == "ZTF")
        & out["epoch_type"].isin(["ztf_irsa_lc", "ztf_alert", "ztf_best_irsa_lc"])
    ]
    if not ztf_lc.empty:
        drop_idx.extend(
            out.index[
                (out["source"] == "ZTF") & (out["epoch_type"] == "tns_discovery")
            ].tolist()
        )

    gaia_alert = out.loc[
        (out["source"] == "Gaia")
        & out["epoch_type"].isin(["gaia_alert_csv", "gaia_alert_page"])
    ]
    if not gaia_alert.empty:
        drop_idx.extend(
            out.index[
                (out["source"] == "Gaia") & (out["epoch_type"] == "tns_discovery")
            ].tolist()
        )

    if drop_idx:
        out = out.drop(index=sorted(set(drop_idx)))
    return out.reset_index(drop=True)


def prepare_work_epochs(raw: pd.DataFrame) -> pd.DataFrame:
    work = astrometry_epochs(raw)
    work = _select_atlas_lc_epochs(work)
    return exclude_tns_astrometry_when_lc_available(work)


def collapse_survey_position(
    df: pd.DataFrame,
    source: str,
) -> dict[str, Any] | None:
    """
    Collapse one survey's epochs to a single position and astrometric error.

    Per-epoch weights use inverse photometric variance (1/σ_mag²) as an SNR
    proxy. Gaia uses a fixed 50 mas uncertainty. All other surveys use the
    weighted RMS scatter of RA/Dec around the weighted mean (no survey floors).
    """
    grp = df.loc[df["source"] == source]
    if grp.empty:
        return None

    sigma_mag = grp["sigma_mag"].astype(float).clip(lower=1e-6)
    weights = 1.0 / np.square(sigma_mag)
    wsum = float(weights.sum())
    if wsum <= 0:
        return None

    ra = grp["ra_deg"].astype(float).to_numpy()
    dec = grp["dec_deg"].astype(float).to_numpy()
    ra_bar = float(np.sum(weights * ra) / wsum)
    dec_bar = float(np.sum(weights * dec) / wsum)
    cos_dec = np.cos(np.deg2rad(dec_bar))

    dra_mas = (ra - ra_bar) * MAS_PER_DEG * cos_dec
    ddec_mas = (dec - dec_bar) * MAS_PER_DEG
    w_norm = weights / wsum

    var_ra = float(np.sum(w_norm * dra_mas**2))
    var_dec = float(np.sum(w_norm * ddec_mas**2))
    cov_rd = float(np.sum(w_norm * dra_mas * ddec_mas))
    n_eff = float(wsum**2 / np.sum(np.square(weights)))

    scatter_ra = float(np.sqrt(max(var_ra, 0.0)))
    scatter_dec = float(np.sqrt(max(var_dec, 0.0)))

    if source == "Gaia":
        sigma_ra = sigma_dec = GAIA_ASTROMETRY_SIGMA_MAS
    else:
        sigma_ra = scatter_ra
        sigma_dec = scatter_dec

    return {
        "source": source,
        "n_epochs": int(len(grp)),
        "n_eff": n_eff,
        "ra_deg": ra_bar,
        "dec_deg": dec_bar,
        "sigma_ra_mas": sigma_ra,
        "sigma_dec_mas": sigma_dec,
        "scatter_ra_mas": scatter_ra,
        "scatter_dec_mas": scatter_dec,
        "cov_mas2": np.array(
            [[sigma_ra**2, cov_rd], [cov_rd, sigma_dec**2]],
            dtype=float,
        ),
    }


def collapse_all_surveys(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Collapse each astrometry survey to one error-weighted position."""
    summaries: list[dict[str, Any]] = []
    for source in sorted(df["source"].unique()):
        if source not in ASTROMETRY_SOURCES:
            continue
        row = collapse_survey_position(df, source)
        if row is not None:
            summaries.append(row)
    return summaries


def mix_surveys_by_error(
    summaries: list[dict[str, Any]],
) -> tuple[float, float, np.ndarray]:
    """Inverse-variance combine per-survey positions (diagonal uncertainties)."""
    usable = [
        s
        for s in summaries
        if s["sigma_ra_mas"] > 0 and s["sigma_dec_mas"] > 0
    ]
    if not usable:
        raise RuntimeError("No survey summaries with positive astrometric scatter.")

    ra_num = dec_num = 0.0
    w_ra = w_dec = 0.0
    for summary in usable:
        w_ra_i = 1.0 / summary["sigma_ra_mas"] ** 2
        w_dec_i = 1.0 / summary["sigma_dec_mas"] ** 2
        ra_num += w_ra_i * summary["ra_deg"]
        dec_num += w_dec_i * summary["dec_deg"]
        w_ra += w_ra_i
        w_dec += w_dec_i

    ra_bar = ra_num / w_ra
    dec_bar = dec_num / w_dec
    cov_mas2 = np.array([[1.0 / w_ra, 0.0], [0.0, 1.0 / w_dec]], dtype=float)
    return ra_bar, dec_bar, cov_mas2


def plot_survey_position_mix(
    epochs: pd.DataFrame,
    survey_summaries: list[dict[str, Any]],
    *,
    ref_ra_deg: float,
    ref_dec_deg: float,
    final_ra_deg: float | None = None,
    final_dec_deg: float | None = None,
    target_name: str | None = None,
    lc_sources: tuple[str, ...] = ("ATLAS", "ZTF"),
    ax: Any | None = None,
) -> Any:
    """
    One-panel sky plot: faintness-weighted LC epochs plus collapsed survey means.

    Individual ATLAS/ZTF points are sized and faded by 1/σ_mag² (brighter epochs
    with smaller mag uncertainty get larger, more opaque markers). Gaia is shown
    only as a large mean point with error bars; ATLAS/ZTF means are mid-sized.
    """
    import matplotlib.pyplot as plt

    survey_style = {
        "ATLAS": {"color": "C3", "lc_label": "ATLAS LC epochs", "mean_label": "ATLAS weighted mean"},
        "ZTF": {"color": "C0", "lc_label": "ZTF LC epochs", "mean_label": "ZTF weighted mean"},
        "Gaia": {"color": "C2", "mean_label": "Gaia alert"},
    }
    lc_epoch_types = {
        "ATLAS": {"atlas_transient_lc"},
        "ZTF": {"ztf_irsa_lc", "ztf_alert", "ztf_best_irsa_lc"},
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure

    cos_ref = np.cos(np.deg2rad(ref_dec_deg))
    summary_by_source = {s["source"]: s for s in survey_summaries}

    def _dra_arcsec(ra_deg: float | np.ndarray) -> float | np.ndarray:
        return (ra_deg - ref_ra_deg) * ARCSEC_PER_DEG

    def _ddec_arcsec(dec_deg: float | np.ndarray) -> float | np.ndarray:
        return (dec_deg - ref_dec_deg) * ARCSEC_PER_DEG

    def _sigma_ra_arcsec(sigma_ra_mas: float) -> float:
        # Stored RA sigmas are in ΔRA cos(dec) [mas]; convert to raw ΔRA [arcsec].
        return float(sigma_ra_mas) / MAS_PER_ARCSEC / max(cos_ref, 1e-12)

    for source in lc_sources:
        style = survey_style.get(source)
        if style is None:
            continue
        types = lc_epoch_types.get(source, set())
        lc = epochs.loc[(epochs["source"] == source) & epochs["epoch_type"].isin(types)]
        if lc.empty:
            continue

        sigma_mag = lc["sigma_mag"].astype(float).clip(lower=1e-6)
        weights = 1.0 / np.square(sigma_mag)
        w_norm = weights / weights.max()
        sizes = 8.0 + 60.0 * w_norm
        alphas = 0.12 + 0.78 * w_norm
        dra_arcsec = _dra_arcsec(lc["ra_deg"])
        ddec_arcsec = _ddec_arcsec(lc["dec_deg"])
        ax.scatter(
            dra_arcsec,
            ddec_arcsec,
            s=sizes,
            c=style["color"],
            alpha=alphas,
            edgecolors="none",
            label=style["lc_label"],
            zorder=1,
        )

    for source, summary in summary_by_source.items():
        style = survey_style.get(source)
        if style is None:
            continue
        dra_arcsec = _dra_arcsec(summary["ra_deg"])
        ddec_arcsec = _ddec_arcsec(summary["dec_deg"])
        if source == "Gaia":
            marker_size, cap, zorder = 14, 5, 6
        else:
            marker_size, cap, zorder = 9, 4, 5
        ax.errorbar(
            dra_arcsec,
            ddec_arcsec,
            xerr=_sigma_ra_arcsec(summary["sigma_ra_mas"]),
            yerr=summary["sigma_dec_mas"] / MAS_PER_ARCSEC,
            fmt="o",
            color=style["color"],
            markersize=marker_size,
            capsize=cap,
            elinewidth=2.0,
            markeredgecolor="k",
            markeredgewidth=0.7,
            label=f"{style['mean_label']} (n={summary['n_epochs']})",
            zorder=zorder,
        )

    if final_ra_deg is not None and final_dec_deg is not None:
        dra_f = _dra_arcsec(final_ra_deg)
        ddec_f = _ddec_arcsec(final_dec_deg)
        ax.scatter(
            dra_f,
            ddec_f,
            c="red",
            marker="*",
            s=320,
            label="mix",
            zorder=7,
            edgecolors="k",
            linewidths=0.5,
        )

    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("ΔRA [arcsec]")
    ax.set_ylabel("ΔDec [arcsec]")
    if final_ra_deg is not None and final_dec_deg is not None:
        name = str(target_name or "").strip() or "transient"
        ax.set_title(
            f"{name} — mix RA={final_ra_deg:.8f}°, Dec={final_dec_deg:.8f}°"
        )
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    return ax


def refine_astrometry(
    work: pd.DataFrame,
    *,
    sigma_mag_limit: float = DEFAULT_SIGMA_MAG_LIMIT,
    clip_n_sigma: float = DEFAULT_CLIP_N_SIGMA,
) -> tuple[pd.DataFrame, pd.DataFrame, AstrometryResult]:
    """Filter → per-survey clip → survey_ivw mix."""
    if work.empty:
        raise RuntimeError("No astrometry epochs to refine.")

    filtered, _ = filter_by_sigma_mag(work, sigma_mag_limit)
    if filtered.empty:
        raise RuntimeError(
            f"All epochs rejected by sigma_mag <= {sigma_mag_limit}."
        )

    per_survey_parts: list[pd.DataFrame] = []
    history: list[dict[str, Any]] = []
    for source in sorted(filtered["source"].unique()):
        if source not in ASTROMETRY_SOURCES:
            continue
        grp = filtered.loc[filtered["source"] == source]
        if len(grp) < 3:
            per_survey_parts.append(grp)
            continue
        sub, sub_hist = iterative_sigma_clip(grp, n_sigma=clip_n_sigma)
        per_survey_parts.append(sub if not sub.empty else grp)
        history.extend(sub_hist)
    clipped = (
        pd.concat(per_survey_parts, ignore_index=True)
        if per_survey_parts
        else filtered
    )
    if clipped.empty:
        raise RuntimeError("survey_ivw: all epochs rejected by per-survey clipping.")
    survey_summaries = collapse_all_surveys(clipped)
    if not survey_summaries:
        raise RuntimeError("survey_ivw: no per-survey summaries after clipping.")
    ra_bar, dec_bar, cov = mix_surveys_by_error(survey_summaries)

    a1, b1, pa = covariance_to_ellipse(cov)
    result = AstrometryResult(
        ra_deg=ra_bar,
        dec_deg=dec_bar,
        cov_mas2=cov,
        n_input=len(work),
        n_after_mag_filter=len(filtered),
        n_after_clip=len(clipped),
        clip_history=history,
        ellipse_1sigma=(a1, b1, pa),
        ellipse_3sigma=(3 * a1, 3 * b1, pa),
        survey_summaries=survey_summaries,
    )
    return filtered, clipped, result


def filter_by_sigma_mag(df: pd.DataFrame, sigma_limit: float) -> pd.DataFrame:
    mask = df["sigma_mag"] <= sigma_limit
    return df.loc[mask].copy(), mask


def iterative_sigma_clip(
    df: pd.DataFrame,
    *,
    n_sigma: float = 3.0,
    max_iter: int = 5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    active = df.copy()
    history: list[dict[str, Any]] = []

    for iteration in range(max_iter):
        mu_ra = active["ra_deg"].mean()
        mu_dec = active["dec_deg"].mean()
        sigma_ra = active["ra_deg"].std(ddof=1)
        sigma_dec = active["dec_deg"].std(ddof=1)

        if len(active) < 3 or sigma_ra == 0 or sigma_dec == 0:
            history.append(
                {
                    "iteration": iteration,
                    "mu_ra": mu_ra,
                    "mu_dec": mu_dec,
                    "sigma_ra": sigma_ra,
                    "sigma_dec": sigma_dec,
                    "outlier_idx": [],
                    "n_active": len(active),
                }
            )
            break

        dev_ra = (active["ra_deg"] - mu_ra).abs() / sigma_ra
        dev_dec = (active["dec_deg"] - mu_dec).abs() / sigma_dec
        outlier_mask = (dev_ra > n_sigma) | (dev_dec > n_sigma)
        outlier_idx = active.index[outlier_mask].tolist()

        history.append(
            {
                "iteration": iteration,
                "mu_ra": mu_ra,
                "mu_dec": mu_dec,
                "sigma_ra": sigma_ra,
                "sigma_dec": sigma_dec,
                "outlier_idx": outlier_idx,
                "n_active": len(active),
                "dev_ra": dev_ra.to_dict(),
                "dev_dec": dev_dec.to_dict(),
            }
        )

        if not outlier_idx:
            break
        active = active.loc[~outlier_mask].copy()

    return active, history


def covariance_to_ellipse(cov_mas2: np.ndarray) -> tuple[float, float, float]:
    tr = float(np.trace(cov_mas2))
    det = float(np.linalg.det(cov_mas2))
    disc = max(tr * tr - 4.0 * det, 0.0)
    lam1 = 0.5 * (tr + np.sqrt(disc))
    lam2 = 0.5 * (tr - np.sqrt(disc))
    a = float(np.sqrt(max(lam1, 0.0)))
    b = float(np.sqrt(max(lam2, 0.0)))
    if cov_mas2[0, 0] == cov_mas2[1, 1]:
        pa = 0.0
    else:
        pa = float(0.5 * np.degrees(np.arctan2(2 * cov_mas2[0, 1], cov_mas2[0, 0] - cov_mas2[1, 1])))
    return a, b, pa


def _seed_coords(
    target_name: str,
    seed_ra: float | None,
    seed_dec: float | None,
) -> tuple[float, float]:
    if seed_ra is not None and seed_dec is not None and np.isfinite(seed_ra) and np.isfinite(seed_dec):
        return float(seed_ra), float(seed_dec)
    resolved = resolve_target(target_name, 0.0, 0.0)
    if resolved.tns_ra_deg is not None and resolved.tns_dec_deg is not None:
        return float(resolved.tns_ra_deg), float(resolved.tns_dec_deg)
    raise RuntimeError(
        f"Cannot determine seed coordinates for {target_name!r}; "
        "provide target_ra/target_dec in targets CSV or ensure TNS resolves."
    )


def _summary_for_json(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in summaries:
        row = {k: v for k, v in s.items() if k != "cov_mas2"}
        if "cov_mas2" in s:
            row["cov_mas2"] = np.asarray(s["cov_mas2"], dtype=float).tolist()
        out.append(row)
    return out


def astrometry_result_path(ws_root: str | Path) -> Path:
    return Path(ws_root) / ASTROMETRY_RESULT_FILENAME


def load_astrometry_coords(ws_root: str | Path) -> tuple[float, float] | None:
    path = astrometry_result_path(ws_root)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ra = data.get("ra_deg")
    dec = data.get("dec_deg")
    if ra is None or dec is None:
        return None
    return float(ra), float(dec)


def write_astrometry_result(
    ws_root: str | Path,
    *,
    target_name: str,
    seed_ra_deg: float | None,
    seed_dec_deg: float | None,
    result: AstrometryResult,
    statuses: list[str],
) -> Path:
    path = astrometry_result_path(ws_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    a1, b1, pa = result.ellipse_1sigma
    payload = {
        "target_name": target_name,
        "seed_ra_deg": seed_ra_deg,
        "seed_dec_deg": seed_dec_deg,
        "ra_deg": result.ra_deg,
        "dec_deg": result.dec_deg,
        "n_input": result.n_input,
        "n_after_mag_filter": result.n_after_mag_filter,
        "n_after_clip": result.n_after_clip,
        "cov_mas2": np.asarray(result.cov_mas2, dtype=float).tolist(),
        "ellipse_1sigma_mas": [a1, b1, pa],
        "survey_summaries": _summary_for_json(result.survey_summaries),
        "statuses": statuses,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def run_astrometry(
    target_name: str,
    seed_ra: float | None,
    seed_dec: float | None,
    *,
    sigma_mag_limit: float = DEFAULT_SIGMA_MAG_LIMIT,
    clip_n_sigma: float = DEFAULT_CLIP_N_SIGMA,
    irsa_credentials_file: Path | None = None,
    atlas_credentials_file: Path | None = None,
    atlas_api_config_file: str | None = None,
) -> tuple[ResolvedTarget, pd.DataFrame, AstrometryResult, list[str]]:
    if atlas_credentials_file is not None:
        global ATLAS_CREDENTIALS_FILE
        ATLAS_CREDENTIALS_FILE = Path(atlas_credentials_file)
    seed_ra_f, seed_dec_f = _seed_coords(target_name, seed_ra, seed_dec)
    resolved = resolve_target(target_name, seed_ra_f, seed_dec_f)
    raw, statuses = compile_detection_table(
        resolved,
        irsa_credentials_file=irsa_credentials_file,
        atlas_api_config_file=atlas_api_config_file,
    )
    if raw.empty:
        raise RuntimeError("No astrometric detections were returned from any survey.")
    work = prepare_work_epochs(raw)
    if work.empty:
        raise RuntimeError("No astrometry epochs after preparing work table.")
    _, _, result = refine_astrometry(
        work,
        sigma_mag_limit=sigma_mag_limit,
        clip_n_sigma=clip_n_sigma,
    )
    return resolved, work, result, statuses


def run_astrometry_stage(
    cfg: "SynDiffConfig",
    stage: dict[str, Any],
    ws_root: str,
    *,
    force_rerun: bool = False,
) -> None:
    """Diff pipeline entry: fetch surveys, write JSON, optional debug plot."""
    from syndiff_pipeline.difference_imaging.support.paths import pipeline_plots_root
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import parse_astrometry

    params = parse_astrometry(stage, 0)
    result_path = astrometry_result_path(ws_root)
    if result_path.is_file() and not force_rerun:
        log.info("Using existing %s", result_path)
        coords = load_astrometry_coords(ws_root)
        if coords is not None:
            cfg.target_ra, cfg.target_dec = coords
        if getattr(cfg, "pipeline_plots", False):
            plot_dir = pipeline_plots_root(
                cfg.output_dir,
                getattr(cfg, "pipeline_plots_dir", "debug_plots"),
                run_id=getattr(cfg, "workspace_run_id", None),
            )
            plot_path = os.path.join(plot_dir, ASTROMETRY_MIX_PLOT_FILENAME)
            if not os.path.isfile(plot_path):
                log.info("Regenerating astrometry plot (JSON exists, plot missing)")
            else:
                return
        else:
            return

    target_name = str(getattr(cfg, "target_name", "") or "").strip()
    if not target_name:
        raise RuntimeError("astrometry stage requires cfg.target_name")

    seed_ra = getattr(cfg, "target_ra", None)
    seed_dec = getattr(cfg, "target_dec", None)
    if seed_ra is not None and not np.isfinite(float(seed_ra)):
        seed_ra = None
    if seed_dec is not None and not np.isfinite(float(seed_dec)):
        seed_dec = None

    irsa_cred = Path(params.irsa_credentials_file) if params.irsa_credentials_file else None
    atlas_cred = Path(params.atlas_credentials_file) if params.atlas_credentials_file else None

    resolved, clipped, result, statuses = run_astrometry(
        target_name,
        seed_ra,
        seed_dec,
        sigma_mag_limit=params.sigma_mag_limit,
        clip_n_sigma=params.clip_n_sigma,
        irsa_credentials_file=irsa_cred,
        atlas_credentials_file=atlas_cred,
        atlas_api_config_file=params.atlas_api_config_file,
    )
    for note in statuses:
        log.info("  astrometry: %s", note)

    write_astrometry_result(
        ws_root,
        target_name=target_name,
        seed_ra_deg=seed_ra,
        seed_dec_deg=seed_dec,
        result=result,
        statuses=statuses,
    )
    cfg.target_ra = float(result.ra_deg)
    cfg.target_dec = float(result.dec_deg)
    log.info(
        "Astrometry mix: RA=%.8f Dec=%.8f (%d epochs after clip)",
        result.ra_deg,
        result.dec_deg,
        result.n_after_clip,
    )

    if not getattr(cfg, "pipeline_plots", False):
        return

    plot_dir = pipeline_plots_root(
        cfg.output_dir,
        getattr(cfg, "pipeline_plots_dir", "debug_plots"),
        run_id=getattr(cfg, "workspace_run_id", None),
    )
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, ASTROMETRY_MIX_PLOT_FILENAME)

    import matplotlib.pyplot as plt

    plot_survey_position_mix(
        clipped,
        result.survey_summaries,
        ref_ra_deg=float(result.ra_deg),
        ref_dec_deg=float(result.dec_deg),
        final_ra_deg=result.ra_deg,
        final_dec_deg=result.dec_deg,
        target_name=target_name,
    )
    plt.savefig(plot_path, dpi=int(getattr(cfg, "pipeline_plot_dpi", 150)))
    plt.close("all")
    log.info("  astrometry plot: %s", plot_path)


def pipeline_needs_template_handoff(pipeline: list[dict[str, Any]]) -> bool:
    """True when the pipeline needs template handoff artifacts.

    The per-FFI and temporal WCS stages are intentionally runnable as a
    standalone fitting/publication lane.  They consume their declared
    centroid/difference workspaces and do not consume template geometry.  A
    pipeline containing only those WCS stages (plus preamble entries) must
    therefore not load the SCC template handoff.  Any mixed pipeline remains
    handoff-bound so that template, mapping, and difference stages retain
    their strict MAPGRID=3 validation.
    """
    kinds = [s.get("kind") for s in pipeline if isinstance(s, dict) and s.get("kind")]
    if not kinds:
        return False
    if kinds == ["astrometry"]:
        return False
    if all(k in {"per_ffi_wcs", "temporal_wcs"} for k in kinds):
        return False
    return any(k != "astrometry" for k in kinds)
