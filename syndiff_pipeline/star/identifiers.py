"""Resolve TIC/Gaia host identifiers to sky positions and persist lookup records."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from syndiff_pipeline.star.hosts import StarHostRequest

_GAIA_CATALOG_COLUMNS = (
    "source_id",
    "ra",
    "ra_error",
    "dec",
    "dec_error",
    "parallax",
    "parallax_error",
    "pm",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
)
_TIC_GAIA_COLUMNS = ("GAIA", "gaia", "Gaia", "gaia_dr3_id", "GAIA_DR3_ID")
_GAIA_MIN_DIGITS = 15
_LOCAL_MATCH_RADIUS_ARCSEC = 1.0
_AMBIGUOUS_SEP_ARCSEC = 0.1


@dataclass(frozen=True)
class ResolvedHost:
    input_kind: Literal["tic", "gaia"]
    input_value: int
    tic_id: Optional[int]
    gaia_source_id: int
    ra: float
    dec: float
    phot_g_mean_mag: Optional[float]
    phot_bp_mean_mag: Optional[float]
    phot_rp_mean_mag: Optional[float]
    resolution_method: str
    label: Optional[str]


def _validate_gaia_source_id(source_id: int) -> None:
    digits = len(str(abs(int(source_id))))
    if digits < _GAIA_MIN_DIGITS:
        raise ValueError(
            f"Gaia source_id {source_id} has only {digits} digit(s); real Gaia DR3 "
            f"source_id values have at least {_GAIA_MIN_DIGITS} digits. "
            "If you meant a TIC id, put it in the tic_id column instead."
        )


def _optional_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _row_to_resolved_host(
    row: pd.Series,
    *,
    input_kind: Literal["tic", "gaia"],
    input_value: int,
    tic_id: Optional[int],
    resolution_method: str,
    label: Optional[str],
) -> ResolvedHost:
    source_id = int(row["source_id"])
    return ResolvedHost(
        input_kind=input_kind,
        input_value=input_value,
        tic_id=tic_id,
        gaia_source_id=source_id,
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        phot_g_mean_mag=_optional_float(row.get("phot_g_mean_mag")),
        phot_bp_mean_mag=_optional_float(row.get("phot_bp_mean_mag")),
        phot_rp_mean_mag=_optional_float(row.get("phot_rp_mean_mag")),
        resolution_method=resolution_method,
        label=label,
    )


def _load_gaia_catalog(gaia_catalog_path: str | Path) -> pd.DataFrame:
    path = Path(gaia_catalog_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Gaia catalog not found: {path}")
    return pd.read_csv(path)


def _lookup_gaia_local(
    gaia_source_id: int,
    gaia_catalog_path: str,
) -> pd.Series | None:
    df = _load_gaia_catalog(gaia_catalog_path)
    if "source_id" not in df.columns:
        raise ValueError(
            f"Gaia catalog {gaia_catalog_path} missing required column 'source_id'"
        )
    matches = df[df["source_id"].astype(np.int64) == int(gaia_source_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _lookup_gaia_remote(gaia_source_id: int) -> pd.Series:
    try:
        from astroquery.gaia import Gaia
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for remote Gaia lookups but is not installed. "
            "Install astroquery or set allow_remote=False and ensure the source_id "
            "is present in the local Gaia catalog."
        ) from exc

    job = Gaia.launch_job(
        f"SELECT source_id,ra,dec,ra_error,dec_error,parallax,parallax_error,"
        f"pm,pmra,pmra_error,pmdec,pmdec_error,"
        f"phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag "
        f"FROM gaia_dr3.gaia_source WHERE source_id = {int(gaia_source_id)}"
    )
    table = job.get_results()
    if len(table) == 0:
        raise ValueError(
            f"Gaia source_id {gaia_source_id} not found in Gaia DR3 (remote query)"
        )
    row = table[0]
    return pd.Series(
        {
            "source_id": int(row["source_id"]),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "ra_error": float(row["ra_error"]) if row["ra_error"] is not None else np.nan,
            "dec_error": float(row["dec_error"]) if row["dec_error"] is not None else np.nan,
            "parallax": float(row["parallax"]) if row["parallax"] is not None else np.nan,
            "parallax_error": (
                float(row["parallax_error"]) if row["parallax_error"] is not None else np.nan
            ),
            "pm": float(row["pm"]) if row["pm"] is not None else np.nan,
            "pmra": float(row["pmra"]) if row["pmra"] is not None else np.nan,
            "pmra_error": (
                float(row["pmra_error"]) if row["pmra_error"] is not None else np.nan
            ),
            "pmdec": float(row["pmdec"]) if row["pmdec"] is not None else np.nan,
            "pmdec_error": (
                float(row["pmdec_error"]) if row["pmdec_error"] is not None else np.nan
            ),
            "phot_g_mean_mag": (
                float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] is not None else np.nan
            ),
            "phot_bp_mean_mag": (
                float(row["phot_bp_mean_mag"])
                if row["phot_bp_mean_mag"] is not None
                else np.nan
            ),
            "phot_rp_mean_mag": (
                float(row["phot_rp_mean_mag"])
                if row["phot_rp_mean_mag"] is not None
                else np.nan
            ),
        }
    )


def _resolve_gaia_source_id(
    gaia_source_id: int,
    *,
    gaia_catalog_path: str,
    allow_remote: bool,
    input_kind: Literal["tic", "gaia"],
    input_value: int,
    tic_id: Optional[int],
    label: Optional[str],
) -> ResolvedHost:
    _validate_gaia_source_id(gaia_source_id)
    row = _lookup_gaia_local(gaia_source_id, gaia_catalog_path)
    method = "local_catalog"
    if row is None:
        if not allow_remote:
            raise ValueError(
                f"Gaia source_id {gaia_source_id} not found in local catalog "
                f"{gaia_catalog_path} and allow_remote=False"
            )
        row = _lookup_gaia_remote(gaia_source_id)
        method = "gaia_remote"
    return _row_to_resolved_host(
        row,
        input_kind=input_kind,
        input_value=input_value,
        tic_id=tic_id,
        resolution_method=method,
        label=label,
    )


def _query_tic(tic_id: int):
    try:
        from astroquery.mast import Catalogs
    except ImportError as exc:
        raise ImportError(
            "astroquery is required for TIC lookups but is not installed."
        ) from exc
    return Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))


def _tic_gaia_source_id(table) -> int | None:
    if table is None or len(table) == 0:
        return None
    row = table[0]
    for col in _TIC_GAIA_COLUMNS:
        if col not in row.colnames:
            continue
        val = row[col]
        if val is None or (isinstance(val, np.ma.MaskedArray) and val.mask):
            continue
        text = str(val).strip()
        if not text or text.lower() in {"null", "nan", "none", "0"}:
            continue
        try:
            gid = int(float(text))
        except (TypeError, ValueError):
            continue
        if gid > 0:
            return gid
    return None


def _tic_radec(table) -> tuple[float, float]:
    if table is None or len(table) == 0:
        raise ValueError("TIC query returned no rows")
    row = table[0]
    ra_col = next((c for c in ("ra", "RA") if c in row.colnames), None)
    dec_col = next((c for c in ("dec", "DEC") if c in row.colnames), None)
    if ra_col is None or dec_col is None:
        raise ValueError("TIC row missing ra/dec columns")
    return float(row[ra_col]), float(row[dec_col])


def _nearest_gaia_match(
    ra: float,
    dec: float,
    gaia_catalog_path: str,
) -> pd.Series:
    df = _load_gaia_catalog(gaia_catalog_path)
    if "ra" not in df.columns or "dec" not in df.columns:
        raise ValueError(
            f"Gaia catalog {gaia_catalog_path} missing ra/dec columns "
            "required for nearest-neighbor matching"
        )
    target = SkyCoord(ra=ra, dec=dec, unit="deg")
    catalog = SkyCoord(
        ra=df["ra"].astype(float).values,
        dec=df["dec"].astype(float).values,
        unit="deg",
    )
    sep = target.separation(catalog).arcsec
    within = np.where(sep <= _LOCAL_MATCH_RADIUS_ARCSEC)[0]
    if len(within) == 0:
        raise ValueError(
            f"No Gaia catalog match within {_LOCAL_MATCH_RADIUS_ARCSEC}\" of "
            f"TIC position (ra={ra:.6f}, dec={dec:.6f}) in {gaia_catalog_path}"
        )
    order = within[np.argsort(sep[within])]
    best_idx = int(order[0])
    if len(order) > 1:
        second_idx = int(order[1])
        if sep[second_idx] - sep[best_idx] <= _AMBIGUOUS_SEP_ARCSEC:
            candidates = []
            for idx in order:
                row = df.iloc[int(idx)]
                candidates.append(
                    f"source_id={int(row['source_id'])} "
                    f"sep={sep[idx]:.3f}\" ra={float(row['ra']):.6f} "
                    f"dec={float(row['dec']):.6f}"
                )
            raise ValueError(
                "Multiple comparably-close Gaia matches within "
                f"{_LOCAL_MATCH_RADIUS_ARCSEC}\" of TIC position; candidates: "
                + "; ".join(candidates)
            )
    return df.iloc[best_idx]


def resolve_host(
    request: StarHostRequest,
    *,
    gaia_catalog_path: str,
    allow_remote: bool = True,
) -> ResolvedHost:
    """Resolve a stars-file row to a Gaia-backed :class:`ResolvedHost`."""
    if request.gaia_source_id is not None:
        return _resolve_gaia_source_id(
            request.gaia_source_id,
            gaia_catalog_path=gaia_catalog_path,
            allow_remote=allow_remote,
            input_kind="gaia",
            input_value=request.gaia_source_id,
            tic_id=None,
            label=request.label,
        )

    assert request.tic_id is not None
    tic_table = _query_tic(request.tic_id)
    if tic_table is None or len(tic_table) == 0:
        raise ValueError(f"TIC id {request.tic_id} not found in MAST TIC catalog")

    gaia_id = _tic_gaia_source_id(tic_table)
    if gaia_id is not None:
        host = _resolve_gaia_source_id(
            gaia_id,
            gaia_catalog_path=gaia_catalog_path,
            allow_remote=allow_remote,
            input_kind="tic",
            input_value=request.tic_id,
            tic_id=request.tic_id,
            label=request.label,
        )
        if host.resolution_method == "local_catalog":
            method = "tic_local_match"
        elif host.resolution_method == "gaia_remote":
            method = "tic_remote"
        else:
            method = host.resolution_method
        return ResolvedHost(
            input_kind=host.input_kind,
            input_value=host.input_value,
            tic_id=host.tic_id,
            gaia_source_id=host.gaia_source_id,
            ra=host.ra,
            dec=host.dec,
            phot_g_mean_mag=host.phot_g_mean_mag,
            phot_bp_mean_mag=host.phot_bp_mean_mag,
            phot_rp_mean_mag=host.phot_rp_mean_mag,
            resolution_method=method,
            label=host.label,
        )

    tic_ra, tic_dec = _tic_radec(tic_table)
    row = _nearest_gaia_match(tic_ra, tic_dec, gaia_catalog_path)
    return _row_to_resolved_host(
        row,
        input_kind="tic",
        input_value=request.tic_id,
        tic_id=request.tic_id,
        resolution_method="tic_local_match",
        label=request.label,
    )


def write_identifier_json(host: ResolvedHost, path: str) -> None:
    """Write ``identifier.json`` for one resolved host."""
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(host)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_host_gaia_row_csv(host: ResolvedHost, path: str) -> None:
    """Write a one-row Gaia-catalog-style CSV for one resolved host."""
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source_id": host.gaia_source_id,
        "ra": host.ra,
        "ra_error": "",
        "dec": host.dec,
        "dec_error": "",
        "parallax": "",
        "parallax_error": "",
        "pm": "",
        "pmra": "",
        "pmra_error": "",
        "pmdec": "",
        "pmdec_error": "",
        "phot_g_mean_mag": host.phot_g_mean_mag if host.phot_g_mean_mag is not None else "",
        "phot_bp_mean_mag": (
            host.phot_bp_mean_mag if host.phot_bp_mean_mag is not None else ""
        ),
        "phot_rp_mean_mag": (
            host.phot_rp_mean_mag if host.phot_rp_mean_mag is not None else ""
        ),
    }
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_GAIA_CATALOG_COLUMNS))
        writer.writeheader()
        writer.writerow(row)
