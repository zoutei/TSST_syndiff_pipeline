"""
wcs_grouping.py
===============
``wcs_grouping`` pipeline stage: WCS extraction and template grouping. The crop-local
Gaia catalog is supplied externally:

  1. Extract WCS from every FFI header.
  2. Compute the pixel drift of the science target over time; optionally smooth
     ``delta_x``/``delta_y`` with a Savitzky–Golay filter (time-ordered valid frames).
  3. Group frames by template-offset bin (offset_threshold).
  4. Validate and return the image crop bounds.

The file ``unique_gaia_stars_for_cropped_template.csv`` (crop-local ``x``, ``y``)
is produced by the template-creation (cluster) pipeline and loaded via
``SynDiffConfig.gaia_catalog``.  Catalogs with only ``ra``/``dec`` (and photometry)
are projected with :func:`ensure_gaia_crop_xy` using the reference-ffi WCS and
``crop_bounds`` from this stage.  The helper ``build_unique_gaia_catalog``
remains for tests or one-off builds from ``removed_stars`` CSV.

Algorithmically this mirrors common pixel-shift and template-grouping workflows
(WCS drift relative to a reference frame).
"""

import json
import logging
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS, FITSFixedWarning

try:
    from syndiff_pipeline.difference_imaging.stages.adaptive_background import get_tessvectors
except ImportError:
    from adaptive_background import get_tessvectors

warnings.filterwarnings("ignore", category=FITSFixedWarning)
warnings.filterwarnings("ignore", category=AstropyWarning)

log = logging.getLogger(__name__)

CLUSTER_TEMPLATE_JOB_FILENAME = "cluster_template_job.json"
WCS_DRIFT_TEMPLATE_DEBUG_FILENAME = "wcs_drift_template_debug.png"

_VALID_CROP_MODES = frozenset({"full", "tl", "tr", "bl", "br"})

# WCS header keywords needed to build an astropy WCS
_WCS_KEYS = [
    "NAXIS", "NAXIS1", "NAXIS2",
    "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
]
_SIP_KEY_PREFIXES = ("A_", "B_", "AP_", "BP_", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wcs_header_complete(header) -> bool:
    """Return True if the header has the minimum keys to build a WCS."""
    for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD2_2"):
        if key not in header:
            return False
    return True


def _header_to_wcs(header) -> WCS:
    """Build an astropy WCS from a FITS header object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(header)
    return wcs


def world_ra_dec_to_pixel(
    wcs: WCS,
    ra_deg,
    dec_deg,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Map RA/Dec (degrees) to 0-based pixel ``(x, y)`` via ``WCS.world_to_pixel_values``.

    Avoids iterative ``all_world2pix`` convergence warnings on TESS SIP WCS.
    """
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    scalar = ra.ndim == 0
    x, y = wcs.world_to_pixel_values(ra, dec)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if scalar:
        return float(x), float(y)
    return x, y


# ── Public functions ──────────────────────────────────────────────────────────

def extract_wcs_from_ffi(ffi_path: str) -> dict:
    """
    Open an FFI FITS file (header-only) and return a dict of WCS keywords
    plus DATE-OBS and NAXIS1/NAXIS2.

    Parameters
    ----------
    ffi_path : str

    Returns
    -------
    dict with keys: 'filename', 'path', 'header', 'wcs_ok', 'DATE-OBS',
                    'NAXIS1', 'NAXIS2'
    """
    result = {
        "filename": os.path.basename(ffi_path),
        "path": ffi_path,
        "header": None,
        "wcs_ok": False,
        "DATE-OBS": None,
        "NAXIS1": None,
        "NAXIS2": None,
    }
    try:
        with fits.open(ffi_path, memmap=True) as hdul:
            hdr = hdul[1].header
            result["header"] = hdr
            result["wcs_ok"] = _wcs_header_complete(hdr)
            result["DATE-OBS"] = hdr.get("DATE-OBS", None)
            result["NAXIS1"] = hdr.get("NAXIS1", None)
            result["NAXIS2"] = hdr.get("NAXIS2", None)
    except Exception as exc:
        log.warning(f"Could not open {ffi_path}: {exc}")
    return result


def _ffi_usable_for_target_pixel(ffi_path: str, target_coord: SkyCoord) -> bool:
    """True if ``extract_wcs_from_ffi`` WCS maps ``target_coord`` to finite pixels."""
    info = extract_wcs_from_ffi(ffi_path)
    if not info["wcs_ok"]:
        return False
    try:
        wcs = _header_to_wcs(info["header"])
        x, y = world_ra_dec_to_pixel(wcs, target_coord.ra.deg, target_coord.dec.deg)
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
    except Exception:
        return False
    return True


def select_ffis_with_valid_target_wcs(
    sorted_ffi_paths: list,
    target_ra: Optional[float],
    target_dec: Optional[float],
    *,
    max_ffis: Optional[int] = None,
) -> list:
    """
    Choose FFI paths in sort order, skipping files with unusable WCS for the target.

    When ``max_ffis`` is set, keep scanning until that many usable paths are
    found (or raise if the pool is exhausted). When ``max_ffis`` is ``None``,
    return all ``sorted_ffi_paths`` without pre-filtering (same as before).

    Parameters
    ----------
    sorted_ffi_paths : list of str
        Typically time-sorted local FFI paths.
    target_ra, target_dec : float or None
        Target sky position in degrees; required when ``max_ffis`` is set.
    max_ffis : int or None
        Cap on the number of FFIs to return, counting only WCS-valid frames.

    Returns
    -------
    list of str
    """
    if not sorted_ffi_paths:
        return []
    if max_ffis is None:
        return list(sorted_ffi_paths)

    if target_ra is None or target_dec is None:
        raise ValueError(
            "target_ra and target_dec are required when max_ffis is set "
            "(needed to skip FFIs without usable WCS for the science target)."
        )
    cap = int(max_ffis)
    if cap < 1:
        return list(sorted_ffi_paths)

    target_coord = SkyCoord(ra=target_ra, dec=target_dec, unit="deg")
    selected = []
    skipped = 0
    log_first = 0
    for ffi_path in sorted_ffi_paths:
        if _ffi_usable_for_target_pixel(ffi_path, target_coord):
            selected.append(ffi_path)
            if len(selected) >= cap:
                break
        else:
            skipped += 1
            if log_first < 3:
                log.info(
                    "  Skipping %s (no usable WCS for target).",
                    os.path.basename(ffi_path),
                )
                log_first += 1
    if skipped > 3:
        log.info(
            "  ... and %d more FFI(s) skipped (no usable WCS for target).",
            skipped - 3,
        )

    if len(selected) < cap:
        raise RuntimeError(
            f"Only {len(selected)} FFI(s) have usable WCS for the target among "
            f"{len(sorted_ffi_paths)} on disk (need {cap} for max_ffis={cap}). "
            "Check target_ra/dec, sector/camera/ccd, or FFI products; or lower max_ffis."
        )

    log.info(
        "  Using %d FFI(s) with valid target WCS (max_ffis=%d; skipped %d unusable).",
        len(selected),
        cap,
        skipped,
    )
    return selected


def build_wcs_table(ffi_paths: list, target_ra: float,
                    target_dec: float) -> pd.DataFrame:
    """
    For each FFI, build a WCS and compute the pixel position of the science
    target.  Initial ``delta_x``/``delta_y`` are measured relative to the first
    valid frame; :func:`reanchor_wcs_drift_to_reference` re-bases them to the
    chosen reference FFI before template grouping.

    Parameters
    ----------
    ffi_paths : list of str
        Paths to all FFI FITS files, sorted by time.
    target_ra, target_dec : float
        RA/Dec (degrees, J2000) of the science target.

    Returns
    -------
    pd.DataFrame
        Columns: filename, path, delta_x, delta_y, btjd, x_pix, y_pix, wcs_ok
    """
    target_coord = SkyCoord(ra=target_ra, dec=target_dec, unit="deg")
    rows = []
    x0, y0 = None, None

    log.info(f"Building WCS table for {len(ffi_paths)} FFIs ...")

    for ffi_path in ffi_paths:
        info = extract_wcs_from_ffi(ffi_path)
        row = {
            "filename": info["filename"],
            "path": info["path"],
            "wcs_ok": info["wcs_ok"],
            "DATE-OBS": info["DATE-OBS"],
            "delta_x": np.nan,
            "delta_y": np.nan,
            "x_pix": np.nan,
            "y_pix": np.nan,
            "btjd": np.nan,
        }

        if info["wcs_ok"]:
            try:
                wcs = _header_to_wcs(info["header"])
                x, y = world_ra_dec_to_pixel(wcs, target_coord.ra.deg, target_coord.dec.deg)
                x, y = float(x), float(y)
                row["x_pix"] = x
                row["y_pix"] = y

                if x0 is None:
                    x0, y0 = x, y

                row["delta_x"] = x - x0
                row["delta_y"] = y - y0

                if info["DATE-OBS"]:
                    t = Time(info["DATE-OBS"], format="isot", scale="utc")
                    try:
                        row["btjd"] = float(t.btjd)
                    except AttributeError:
                        # Older astropy: BTJD = BJD - 2457000.0
                        row["btjd"] = float(t.jd) - 2457000.0
            except Exception as exc:
                log.warning(f"WCS computation failed for {info['filename']}: {exc}")
                row["wcs_ok"] = False

        rows.append(row)

    df = pd.DataFrame(rows)
    n_ok = df["wcs_ok"].sum()
    log.info(f"WCS table built: {n_ok}/{len(df)} frames have valid WCS.")
    return df


def smooth_wcs_drift_savgol(
    wcs_table: pd.DataFrame,
    *,
    window_length: Optional[int] = 11,
    polyorder: int = 2,
    time_col: str = "btjd",
) -> pd.DataFrame:
    """
    Smooth ``delta_x`` and ``delta_y`` with a Savitzky–Golay filter along the
    sequence of frames with valid WCS and finite drifts, ordered by ``time_col``
    (secondary key: row index for stable ordering).

    Raw values are copied to ``delta_x_raw`` / ``delta_y_raw`` when smoothing
    runs. Rows outside the valid mask are unchanged. If there are too few
    points or ``window_length`` is ``None`` or < 3, returns the table
    unmodified (no raw columns added).

    Parameters
    ----------
    wcs_table : pd.DataFrame
        Output of :func:`build_wcs_table`.
    window_length : int or None
        SG window length (odd). Even values are increased by 1. Capped to the
        number of valid frames; if the cap makes smoothing impossible, skipped.
    polyorder : int
        Polynomial order (clamped to ``window_length - 1``).
    time_col : str
        Column used to sort valid frames before filtering (e.g. ``btjd``).

    Returns
    -------
    pd.DataFrame
        Copy of ``wcs_table`` with smoothed ``delta_x`` / ``delta_y`` (and
        optionally raw columns).
    """
    if window_length is None or int(window_length) < 3:
        return wcs_table

    base = wcs_table["wcs_ok"] & wcs_table["delta_x"].notna() & wcs_table["delta_y"].notna()
    n = int(base.sum())
    if n < 3:
        log.info(
            "WCS drift SG smooth skipped: only %d valid drift samples (need >= 3).",
            n,
        )
        return wcs_table

    if time_col not in wcs_table.columns:
        log.warning(
            "WCS drift SG smooth skipped: time column %r missing.", time_col
        )
        return wcs_table

    valid_idx = np.flatnonzero(base.to_numpy())
    tkey = wcs_table.loc[valid_idx, time_col].to_numpy(dtype=float)
    tkey = np.where(np.isfinite(tkey), tkey, np.inf)
    order = np.lexsort((valid_idx, tkey))
    sorted_idx = valid_idx[order]

    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, n)
    if wl % 2 == 0:
        wl -= 1
    if wl < 3:
        log.info(
            "WCS drift SG smooth skipped: effective window %d for %d samples.",
            wl,
            n,
        )
        return wcs_table

    po = min(max(int(polyorder), 0), wl - 1)

    df = wcs_table.copy()
    dx = df.loc[sorted_idx, "delta_x"].to_numpy(dtype=float)
    dy = df.loc[sorted_idx, "delta_y"].to_numpy(dtype=float)

    df["delta_x_raw"] = df["delta_x"]
    df["delta_y_raw"] = df["delta_y"]
    sx = savgol_filter(dx, wl, po, mode="interp")
    sy = savgol_filter(dy, wl, po, mode="interp")
    df.loc[sorted_idx, "delta_x"] = sx
    df.loc[sorted_idx, "delta_y"] = sy

    log.info(
        "WCS drift Savitzky–Golay smooth: %d samples, window=%d, polyorder=%d "
        "(time-sorted on %r).",
        n,
        wl,
        po,
        time_col,
    )
    return df


def _wcs_ok_mask(wok: pd.Series) -> pd.Series:
    return wok.apply(lambda x: x is True or str(x).lower() in ("true", "1"))


def attach_tessvector_earth_moon_angles(
    wcs_table: pd.DataFrame,
    *,
    sector: int,
    camera: int,
    tessvectors_data_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add ``earth_deg`` and ``moon_deg`` (camera–body angles, degrees) by
    interpolating TESSVectors FFI CSV onto each row's ``btjd``.
    """
    out = wcs_table.copy()
    out["earth_deg"] = np.nan
    out["moon_deg"] = np.nan
    if "btjd" not in out.columns:
        log.warning("wcs_table has no btjd; cannot attach TESSVectors angles.")
        return out

    df = get_tessvectors(int(sector), int(camera), data_path=tessvectors_data_path)
    if df is None or df.empty:
        log.warning(
            "TESSVectors unavailable for sector=%s camera=%s; earth_deg/moon_deg left NaN.",
            sector,
            camera,
        )
        return out

    vec_t = np.asarray(df["MidTime"].values, dtype=float)
    earth_v = np.asarray(df["Earth_Camera_Angle"].values, dtype=float)
    moon_v = np.asarray(df["Moon_Camera_Angle"].values, dtype=float)
    btjd = pd.to_numeric(out["btjd"], errors="coerce")
    m = btjd.notna()
    if not m.any():
        return out
    bi = btjd.loc[m].to_numpy(dtype=float)
    out.loc[m, "earth_deg"] = np.interp(bi, vec_t, earth_v)
    out.loc[m, "moon_deg"] = np.interp(bi, vec_t, moon_v)
    return out


def _pick_closest_to_median_smoothed(
    sub: pd.DataFrame, median_dx: float, median_dy: float
) -> str:
    """Return ``path`` of row minimizing squared distance to median smoothed drift."""
    dx = pd.to_numeric(sub["delta_x"], errors="coerce").to_numpy(dtype=float)
    dy = pd.to_numeric(sub["delta_y"], errors="coerce").to_numpy(dtype=float)
    d2 = (dx - median_dx) ** 2 + (dy - median_dy) ** 2
    pos = int(np.nanargmin(d2))
    return str(sub.iloc[pos]["path"])


def choose_reference_ffi_path(
    wcs_table: pd.DataFrame,
    *,
    earth_deg_min: float = 45.0,
    moon_deg_min: float = 25.0,
    max_smoothed_residual: float = 0.05,
) -> str:
    """
    Pick a reference FFI path after WCS drift smoothing.

    Uses smoothed ``delta_x``/``delta_y`` medians as the target pointing, prefers
    frames with small raw–smooth residual (when ``delta_x_raw`` exist), TESSVectors
    Earth/Moon angle cuts when ``earth_deg``/``moon_deg`` are present on the table,
    and otherwise falls back with logged warnings.

    Call :func:`attach_tessvector_earth_moon_angles` first so angle columns are
    populated (unless intentionally omitting scatter-light screening).
    """
    dx = pd.to_numeric(wcs_table["delta_x"], errors="coerce")
    dy = pd.to_numeric(wcs_table["delta_y"], errors="coerce")
    ok = dx.notna() & dy.notna()
    if "wcs_ok" in wcs_table.columns:
        ok = ok & _wcs_ok_mask(wcs_table["wcs_ok"])
    if not ok.any():
        raise RuntimeError("No FFI with usable WCS offsets for ref frame.")

    S = ok
    sub_S = wcs_table.loc[S]
    median_dx = float(np.nanmedian(sub_S["delta_x"].astype(float)))
    median_dy = float(np.nanmedian(sub_S["delta_y"].astype(float)))

    has_raw = {"delta_x_raw", "delta_y_raw"}.issubset(wcs_table.columns)
    has_angles = {"earth_deg", "moon_deg"}.issubset(wcs_table.columns)

    if has_raw:
        dx_s = pd.to_numeric(wcs_table["delta_x"], errors="coerce")
        dy_s = pd.to_numeric(wcs_table["delta_y"], errors="coerce")
        dx0 = pd.to_numeric(wcs_table["delta_x_raw"], errors="coerce")
        dy0 = pd.to_numeric(wcs_table["delta_y_raw"], errors="coerce")
        r = np.hypot(dx0 - dx_s, dy0 - dy_s)
        residual_ok = dx0.notna() & dy0.notna() & (r <= float(max_smoothed_residual))
    else:
        residual_ok = pd.Series(True, index=wcs_table.index)

    if has_angles:
        ed = pd.to_numeric(wcs_table["earth_deg"], errors="coerce")
        md = pd.to_numeric(wcs_table["moon_deg"], errors="coerce")
        angle_ok = (
            ed.notna()
            & md.notna()
            & (ed >= float(earth_deg_min))
            & (md >= float(moon_deg_min))
        )
    else:
        angle_ok = pd.Series(True, index=wcs_table.index)

    trials = [
        ("residual+Earth/Moon angle cuts", S & residual_ok & angle_ok),
        ("Earth/Moon angle cuts only (no residual gate)", S & angle_ok),
        ("residual gate only (no angle cuts)", S & residual_ok),
        ("all usable WCS rows", S),
    ]

    for label, mask in trials:
        if not mask.any():
            continue
        sub = wcs_table.loc[mask]
        path = _pick_closest_to_median_smoothed(sub, median_dx, median_dy)
        if label != trials[0][0]:
            log.warning(
                "Reference FFI: using fallback selection (%s); chosen path may have "
                "worse scatter-light geometry or larger raw–smooth drift.",
                label,
            )
        log.info(
            "Reference FFI selected (%s): median smoothed drift (%.4f, %.4f) px; %d candidates.",
            label,
            median_dx,
            median_dy,
            len(sub),
        )
        return path

    first = str(wcs_table.loc[S].iloc[0]["path"])
    log.warning(
        "Reference FFI: unexpected empty trial masks; using first usable row: %s",
        first,
    )
    return first


def _row_index_for_path(wcs_table: pd.DataFrame, ref_ffi_path: str) -> Optional[int]:
    """Return row index matching ``ref_ffi_path``, or ``None``."""
    if not ref_ffi_path or not str(ref_ffi_path).strip():
        return None
    path_col = "path" if "path" in wcs_table.columns else "filename"
    if path_col not in wcs_table.columns:
        return None
    try:
        ref_r = Path(ref_ffi_path).resolve()
    except Exception:
        ref_r = Path(os.path.expanduser(ref_ffi_path))
    ref_abs = os.path.abspath(os.path.expanduser(str(ref_ffi_path)))
    for i in range(len(wcs_table)):
        p = wcs_table.iloc[i].get(path_col)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        ps = str(p).strip()
        if not ps:
            continue
        try:
            match = Path(ps).resolve() == ref_r
        except Exception:
            match = os.path.abspath(os.path.expanduser(ps)) == ref_abs
        if match:
            return i
    return None


def reanchor_wcs_drift_to_reference(
    wcs_table: pd.DataFrame, ref_ffi_path: str
) -> pd.DataFrame:
    """
    Re-base drifts to the SG-smoothed reference origin.

    When Savitzky–Golay raw columns exist, subtract the reference row's
    **smoothed** offsets from both the smoothed and raw columns. This keeps the
    debug plot in one coordinate frame so the raw points show their residuals
    around the SG-smoothed reference origin. Otherwise recompute
    ``delta_x``/``delta_y`` from ``x_pix``/``y_pix`` relative to the reference
    row.
    """
    idx = _row_index_for_path(wcs_table, ref_ffi_path)
    if idx is None:
        raise ValueError(f"Reference FFI not found in WCS table: {ref_ffi_path!r}")
    ref_row = wcs_table.iloc[idx]
    if not bool(ref_row.get("wcs_ok", False)):
        raise ValueError(f"Reference FFI has invalid WCS: {ref_ffi_path!r}")

    df = wcs_table.copy()
    ok = df["wcs_ok"] & df["delta_x"].notna() & df["delta_y"].notna()
    has_raw = {"delta_x_raw", "delta_y_raw"}.issubset(df.columns)

    if has_raw:
        ref_dx = float(ref_row["delta_x"])
        ref_dy = float(ref_row["delta_y"])
        if not all(np.isfinite(v) for v in (ref_dx, ref_dy)):
            raise ValueError(
                f"Reference FFI has non-finite drift values: {ref_ffi_path!r}"
            )
        df.loc[ok, "delta_x_raw"] = df.loc[ok, "delta_x_raw"].astype(float) - ref_dx
        df.loc[ok, "delta_y_raw"] = df.loc[ok, "delta_y_raw"].astype(float) - ref_dy
        df.loc[ok, "delta_x"] = df.loc[ok, "delta_x"].astype(float) - ref_dx
        df.loc[ok, "delta_y"] = df.loc[ok, "delta_y"].astype(float) - ref_dy
    else:
        x_ref = float(ref_row["x_pix"])
        y_ref = float(ref_row["y_pix"])
        if not (np.isfinite(x_ref) and np.isfinite(y_ref)):
            raise ValueError(
                f"Reference FFI has non-finite target pixel position: {ref_ffi_path!r}"
            )
        pix_ok = ok & df["x_pix"].notna() & df["y_pix"].notna()
        df.loc[pix_ok, "delta_x"] = df.loc[pix_ok, "x_pix"].astype(float) - x_ref
        df.loc[pix_ok, "delta_y"] = df.loc[pix_ok, "y_pix"].astype(float) - y_ref

    df.loc[~ok, "delta_x"] = np.nan
    df.loc[~ok, "delta_y"] = np.nan
    if has_raw:
        df.loc[~ok, "delta_x_raw"] = np.nan
        df.loc[~ok, "delta_y_raw"] = np.nan

    log.info(
        "WCS drift re-anchored to SG-smoothed reference FFI origin: %s",
        ref_ffi_path,
    )
    return df


def finalize_wcs_table_with_reference_anchor(
    wcs_table: pd.DataFrame,
    *,
    offset_threshold: float,
    ref_ffi_path: Optional[str] = None,
    ref_earth_deg_min: float = 45.0,
    ref_moon_deg_min: float = 25.0,
    ref_max_smoothed_residual: float = 0.05,
) -> tuple[pd.DataFrame, str]:
    """
    Pick (or accept) a reference FFI, re-anchor drifts to it, and assign
    template groups.

    Expects *wcs_table* to already have a Savitzky–Golay smooth (for reference
    selection and grouping) and optional TESSVectors columns from
    :func:`attach_tessvector_earth_moon_angles`.
    """
    if ref_ffi_path and os.path.exists(ref_ffi_path):
        chosen_ref = ref_ffi_path
    else:
        chosen_ref = choose_reference_ffi_path(
            wcs_table,
            earth_deg_min=ref_earth_deg_min,
            moon_deg_min=ref_moon_deg_min,
            max_smoothed_residual=ref_max_smoothed_residual,
        )
    wcs_table = reanchor_wcs_drift_to_reference(wcs_table, chosen_ref)
    wcs_table = assign_template_groups(wcs_table, offset_threshold)
    return wcs_table, chosen_ref


def _ref_ffi_btjd(wcs_table: pd.DataFrame, ref_ffi_path: Optional[str]) -> float:
    """BTJD of the manifest row matching ``ref_ffi_path``, or NaN."""
    if not ref_ffi_path or ref_ffi_path.strip() == "" or "btjd" not in wcs_table.columns:
        return float("nan")
    idx = _row_index_for_path(wcs_table, ref_ffi_path)
    if idx is None:
        return float("nan")
    t = wcs_table.iloc[idx].get("btjd")
    try:
        return float(t)
    except (TypeError, ValueError):
        return float("nan")


def summarize_template_groups(wcs_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build the per-group summary table (``group_id``, ``group_dx``, ``group_dy``,
    ``n_frames``) from a ``wcs_table`` that already has group columns assigned.
    """
    if "group_id" not in wcs_table.columns:
        raise ValueError("wcs_table must have assign_template_groups run first.")
    v = wcs_table[wcs_table["group_id"] >= 0]
    if v.empty:
        return pd.DataFrame(
            columns=["group_id", "group_dx", "group_dy", "n_frames"]
        )
    g = (
        v.groupby("group_id", sort=True)
        .agg(
            group_dx=("group_dx", "first"),
            group_dy=("group_dy", "first"),
            n_frames=("group_id", "size"),
        )
        .reset_index()
    )
    return g


def load_cluster_template_job(output_dir: str) -> dict:
    """Load ``cluster_template_job.json`` from ``output_dir``."""
    path = os.path.join(output_dir, CLUSTER_TEMPLATE_JOB_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing cluster handoff: {path}")
    with open(path) as fh:
        return json.load(fh)


def crop_bounds_from_cluster_payload(payload: dict) -> dict:
    """
    Build a ``crop_bounds`` dict (including ``shape`` as ``(ny, nx)`` tuples)
    from a loaded cluster-template JSON payload.
    """
    required = ("x_min", "x_max", "y_min", "y_max")
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(
            f"cluster_template_job.json missing {missing}; re-run wcs_grouping."
        )
    out = {k: int(payload[k]) for k in required}
    if "shape" in payload:
        sh = payload["shape"]
        if isinstance(sh, (list, tuple)) and len(sh) == 2:
            out["shape"] = (int(sh[0]), int(sh[1]))
        else:
            out["shape"] = (
                out["y_max"] - out["y_min"],
                out["x_max"] - out["x_min"],
            )
    else:
        out["shape"] = (
            out["y_max"] - out["y_min"],
            out["x_max"] - out["x_min"],
        )
    return out


def load_reference_ffi_path(
    output_dir: str, fallback: Optional[str] = None
) -> Optional[str]:
    """
    Reference FFI absolute path from ``cluster_template_job.json``, or legacy
    ``ref_ffi_path.txt``, or ``fallback`` (e.g. ``cfg.ref_ffi_path``).
    """
    job = os.path.join(output_dir, CLUSTER_TEMPLATE_JOB_FILENAME)
    if os.path.isfile(job):
        with open(job) as fh:
            return str(json.load(fh)["reference_ffi_path"])
    txt = os.path.join(output_dir, "ref_ffi_path.txt")
    if os.path.isfile(txt):
        with open(txt) as fh:
            return fh.read().strip()
    return str(fallback) if fallback else None


def write_cluster_template_job_json(
    summary_df: pd.DataFrame,
    ref_ffi_path: str,
    sector: int,
    camera: int,
    ccd: int,
    offset_threshold: float,
    output_dir: str,
    crop_bounds: Optional[dict] = None,
    crop_mode: str | None = None,
    crop_box_size: int | None = None,
) -> str:
    """
    Write a JSON bundle for the cluster template job: reference FFI name/path,
    instrument IDs, threshold, optional FFI crop (``x_min`` … ``y_max``), and the
    per-group table formerly in ``group_offsets.csv`` (now only in this JSON).

    Parameters
    ----------
    crop_bounds : dict, optional
        From ``get_crop_bounds``. When given, writes ``x_min`` … ``y_max`` and
        ``shape`` ``[ny, nx]`` for downstream reload without separate crop JSON.
    """
    def _json_val(x: Any) -> Union[int, float, str]:
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        return str(x)

    ref_abs = os.path.abspath(os.path.expanduser(ref_ffi_path))
    groups = []
    for _, row in summary_df.iterrows():
        groups.append(
            {
                "group_id": _json_val(row["group_id"]),
                "group_dx": _json_val(row["group_dx"]),
                "group_dy": _json_val(row["group_dy"]),
                "n_frames": _json_val(row["n_frames"]),
            }
        )
    payload = {
        "schema_version": 1,
        "reference_ffi_basename": os.path.basename(ref_abs),
        "reference_ffi_path": ref_abs,
        "sector": int(sector),
        "camera": int(camera),
        "ccd": int(ccd),
        "offset_threshold": float(offset_threshold),
        "groups": groups,
    }
    if crop_bounds is not None:
        for key in ("x_min", "x_max", "y_min", "y_max"):
            if key in crop_bounds:
                payload[key] = int(crop_bounds[key])
        if "shape" in crop_bounds:
            sh = crop_bounds["shape"]
            payload["shape"] = [int(sh[0]), int(sh[1])]
    if crop_mode:
        payload["crop_mode"] = str(crop_mode)
    if crop_box_size is not None:
        payload["crop_box_size"] = int(crop_box_size)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, CLUSTER_TEMPLATE_JOB_FILENAME)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(f"Cluster template job JSON written to {out_path}")
    return out_path


def plot_wcs_drift_and_template_assignment(
    wcs_table: pd.DataFrame,
    output_path: str,
    time_col: str = "btjd",
    *,
    ref_ffi_path: Optional[str] = None,
    ref_earth_deg_min: float = 45.0,
    ref_moon_deg_min: float = 25.0,
    sector: Optional[int] = None,
    camera: Optional[int] = None,
    ccd: Optional[int] = None,
    target_name: Optional[str] = None,
) -> Optional[str]:
    """
    Four stacked panels: ``delta_x``, ``delta_y``, ``group_id``, and Earth/Moon
    camera angles (TESSVectors) vs time.

    ``delta_x``/``delta_y`` are expected to be reference-FFI-relative (see
    :func:`reanchor_wcs_drift_to_reference`). When ``delta_x_raw`` /
    ``delta_y_raw`` are present (after Savitzky–Golay smoothing), the first two
    panels overlay **original** scatter points and a **smoothed** polyline in
    time order. Optional ``ref_ffi_path`` draws a vertical reference line on all
    panels at that FFI's ``btjd``.
    """
    need = {"delta_x", "delta_y", "group_id"}
    if not need.issubset(wcs_table.columns):
        log.warning("wcs_table missing columns for drift plot; skipping.")
        return None
    if time_col not in wcs_table.columns:
        log.warning(f"wcs_table missing {time_col!r}; skipping drift plot.")
        return None

    t = pd.to_numeric(wcs_table[time_col], errors="coerce")
    dx = pd.to_numeric(wcs_table["delta_x"], errors="coerce")
    dy = pd.to_numeric(wcs_table["delta_y"], errors="coerce")
    gid = wcs_table["group_id"]

    valid_xy = t.notna() & dx.notna() & dy.notna() & (gid >= 0)
    valid_t = t.notna()

    gids_pos = gid[gid >= 0].unique()
    gids_sorted = sorted(int(x) for x in gids_pos)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(gids_sorted), 1)))[: len(gids_sorted)]
    color_by_gid = {g: cmap[i % len(cmap)] for i, g in enumerate(gids_sorted)}

    t_ref = _ref_ffi_btjd(wcs_table, ref_ffi_path)
    show_vline = np.isfinite(t_ref)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, layout="constrained")

    ax0, ax1, ax2, ax3 = axes
    has_raw = {"delta_x_raw", "delta_y_raw"}.issubset(wcs_table.columns)
    if has_raw:
        dx0 = pd.to_numeric(wcs_table["delta_x_raw"], errors="coerce")
        dy0 = pd.to_numeric(wcs_table["delta_y_raw"], errors="coerce")
        m0 = t.notna() & dx0.notna() & dy0.notna() & (gid >= 0)
        ax0.scatter(
            t[m0],
            dx0[m0],
            s=22,
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.9,
            alpha=0.9,
            label="original",
            zorder=2,
        )
        ax1.scatter(
            t[m0],
            dy0[m0],
            s=22,
            facecolors="none",
            edgecolors="0.35",
            linewidths=0.9,
            alpha=0.9,
            label="original",
            zorder=2,
        )

        idx_xy = np.flatnonzero(valid_xy.to_numpy())
        ts = t.to_numpy(dtype=float)[idx_xy]
        order = np.argsort(ts, kind="mergesort")
        idx_s = idx_xy[order]
        ax0.plot(
            t.iloc[idx_s],
            dx.iloc[idx_s],
            color="C0",
            linewidth=1.85,
            alpha=0.95,
            solid_capstyle="round",
            label="smoothed",
            zorder=3,
        )
        ax1.plot(
            t.iloc[idx_s],
            dy.iloc[idx_s],
            color="C1",
            linewidth=1.85,
            alpha=0.95,
            solid_capstyle="round",
            label="smoothed",
            zorder=3,
        )
    else:
        ax0.scatter(t[valid_xy], dx[valid_xy], s=8, alpha=0.6, c="C0", label=r"$\delta x$")
        ax1.scatter(t[valid_xy], dy[valid_xy], s=8, alpha=0.6, c="C1", label=r"$\delta y$")

    ax0.set_ylabel(r"$\delta x$ (pix)")
    ax0.grid(True, alpha=0.3)

    ax1.set_ylabel(r"$\delta y$ (pix)")
    ax1.grid(True, alpha=0.3)
    h1, lab1 = ax1.get_legend_handles_labels()
    if h1:
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for g in gids_sorted:
        m = valid_t & (gid == g)
        if not m.any():
            continue
        color = color_by_gid[g]
        ax2.scatter(t[m], np.full(m.sum(), g), s=12, alpha=0.7, color=color, label=f"g{g}")

    unassigned = valid_t & (gid < 0)
    if unassigned.any():
        ax2.scatter(
            t[unassigned],
            np.full(unassigned.sum(), -1.0),
            s=6,
            alpha=0.4,
            c="0.5",
            label="unassigned",
        )

    ax2.set_ylabel("group_id")
    yticks = list(gids_sorted)
    if unassigned.any():
        yticks = [-1] + yticks
    if not yticks:
        yticks = [0]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(["—" if y == -1 else str(y) for y in yticks])
    ax2.grid(True, alpha=0.3)
    if gids_sorted:
        ax2.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)

    if {"earth_deg", "moon_deg"}.issubset(wcs_table.columns):
        earth = pd.to_numeric(wcs_table["earth_deg"], errors="coerce")
        moon = pd.to_numeric(wcs_table["moon_deg"], errors="coerce")
        ma = t.notna() & earth.notna() & moon.notna()
        if ma.any():
            ax3.scatter(
                t[ma], earth[ma], s=10, alpha=0.65, c="C2", label="Earth–camera (deg)"
            )
            ax3.scatter(
                t[ma], moon[ma], s=10, alpha=0.65, c="C3", label="Moon–camera (deg)"
            )
        ax3.axhline(
            ref_earth_deg_min,
            color="C2",
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
            label=f"Earth min ({ref_earth_deg_min:g}°)",
        )
        ax3.axhline(
            ref_moon_deg_min,
            color="C3",
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
            label=f"Moon min ({ref_moon_deg_min:g}°)",
        )
        ax3.set_ylabel("angle (deg)")
        ax3.grid(True, alpha=0.3)
        h3, _ = ax3.get_legend_handles_labels()
        if h3:
            ax3.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9)
    else:
        ax3.text(0.5, 0.5, "no earth_deg/moon_deg (run TESSVectors attach)", ha="center", va="center", transform=ax3.transAxes, fontsize=9)
        ax3.set_ylabel("angle (deg)")

    if show_vline:
        ax0.axvline(
            t_ref,
            color="0.25",
            linestyle="--",
            linewidth=1.2,
            zorder=5,
            label="reference FFI",
        )
        for ax in (ax1, ax2, ax3):
            ax.axvline(
                t_ref,
                color="0.25",
                linestyle="--",
                linewidth=1.2,
                zorder=4,
            )

    h0, _ = ax0.get_legend_handles_labels()
    if h0:
        ax0.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax3.set_xlabel(f"{time_col} (TESS BTJD)" if time_col == "btjd" else time_col)
    if (
        sector is not None
        and camera is not None
        and ccd is not None
        and target_name
    ):
        title = (
            f"Sector {sector}, Camera {camera}, CCD {ccd}, {target_name} — "
            "WCS Drift, Template Groups"
        )
    else:
        title = "WCS Drift, Template Groups"
    fig.suptitle(title)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"WCS drift / template debug figure saved to {output_path}")
    return output_path


def assign_template_groups(wcs_table: pd.DataFrame,
                            offset_threshold: float = 0.01,
                            output_dir: str = None) -> pd.DataFrame:
    """
    Round (delta_x, delta_y) to the nearest offset_threshold grid and assign
    a unique integer group_id to each distinct rounded value.

    Parameters
    ----------
    wcs_table : pd.DataFrame
        Output of :func:`build_wcs_table` (optionally after other edits).
    offset_threshold : float
        Grid spacing in TESS pixels.
    output_dir : str, optional
        Deprecated, ignored. Group summary is written only to
        ``cluster_template_job.json`` from the ``wcs_grouping`` stage.

    Returns
    -------
    pd.DataFrame
        wcs_table augmented with columns: group_id, group_dx, group_dy.
    """
    df = wcs_table.copy()
    df["group_id"] = -1
    df["group_dx"] = np.nan
    df["group_dy"] = np.nan

    valid = df["wcs_ok"] & df["delta_x"].notna() & df["delta_y"].notna()

    if valid.sum() == 0:
        log.error("No valid WCS frames found; cannot assign template groups.")
        return df

    # Round offsets to grid
    dx_rounded = (df.loc[valid, "delta_x"] / offset_threshold).round() * offset_threshold
    dy_rounded = (df.loc[valid, "delta_y"] / offset_threshold).round() * offset_threshold

    # Map rounded pairs to group_id (in order of first occurrence)
    group_map = {}
    group_id_col = []
    for dx, dy in zip(dx_rounded, dy_rounded):
        key = (round(dx, 6), round(dy, 6))
        if key not in group_map:
            group_map[key] = len(group_map)
        group_id_col.append(group_map[key])

    df.loc[valid, "group_id"] = group_id_col
    df.loc[valid, "group_dx"] = dx_rounded.values
    df.loc[valid, "group_dy"] = dy_rounded.values

    summary_df = summarize_template_groups(df)

    log.info(f"\nTemplate groups ({len(group_map)} total, threshold={offset_threshold} px):")
    log.info("\n" + summary_df.to_string(index=False))
    print(f"\nTemplate groups ({len(group_map)} total, threshold={offset_threshold} px):")
    print(summary_df.to_string(index=False))
    print(
        "\n→ Handoff for the cluster: cluster_template_job.json (written by wcs_grouping "
        "after the reference FFI and crop are set)."
    )
    print("  Then fill cfg.template_paths = {group_id: path} in your YAML and run hotpants.")

    return df


def get_crop_bounds(
    ffi_header,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    crop_mode: str = "full",
    x_left_dead: int = 44,
    x_right_dead: int = 44,
    y_edge_strip: int = 30,
) -> dict:
    """
    Compute and validate the crop region for the pipeline.

    **Explicit mode** — if any of ``x_min``, ``x_max``, ``y_min``, ``y_max`` is
    not ``None``: those values define the box. Any edge left ``None`` is set to
    the corresponding corner of the **usable** rectangle (dead strips removed);
    then all edges are clamped to valid FFI indices ``[0, nx]`` / ``[0, ny]``.

    **Preset mode** — if all four coords are ``None``: use ``crop_mode``.
    For ``'tl'``/``'tr'``/``'bl'``/``'br'``, subdivide the **usable** rectangle
    (dead strips removed) using chip midlines ``nx // 2`` and ``ny // 2``.
    Usable area is ``x ∈ [x_left_dead, nx - x_right_dead)``,
    ``y ∈ [0, ny - y_edge_strip)``.
    ``'full'`` selects the entire FFI array ``[0, nx) × [0, ny)`` including dead
    columns/rows. ``'target_box'`` is handled by :func:`resolve_crop_bounds_from_params`.

    Parameters
    ----------
    ffi_header : astropy.io.fits.Header
    x_min, x_max, y_min, y_max : int or None
    crop_mode : str
        ``'full'`` | ``'tl'`` | ``'tr'`` | ``'bl'`` | ``'br'`` (preset mode only).
    x_left_dead, x_right_dead : int
        Dead columns on left and right (usable x excludes them).
    y_edge_strip : int
        Dead rows on the **top** only; usable y is ``[0, ny - y_edge_strip)``.

    Returns
    -------
    dict with keys: x_min, x_max, y_min, y_max, shape (ny_crop, nx_crop)
    """
    nx = int(ffi_header["NAXIS1"])
    ny = int(ffi_header["NAXIS2"])

    x_usable_lo = int(x_left_dead)
    x_usable_hi = nx - int(x_right_dead)
    y_usable_lo = 0
    y_usable_hi = ny - int(y_edge_strip)

    if x_usable_lo >= x_usable_hi or y_usable_lo >= y_usable_hi:
        raise ValueError(
            f"Usable area is empty after dead strips: x=[{x_usable_lo},{x_usable_hi}), "
            f"y=[{y_usable_lo},{y_usable_hi}), FFI shape {ny}×{nx}."
        )

    mode = str(crop_mode).strip().lower()
    if mode not in _VALID_CROP_MODES:
        raise ValueError(
            f"crop_mode must be one of {sorted(_VALID_CROP_MODES)}, got {crop_mode!r}"
        )

    explicit = any(v is not None for v in (x_min, x_max, y_min, y_max))

    if explicit:
        xm = int(x_min) if x_min is not None else x_usable_lo
        xM = int(x_max) if x_max is not None else x_usable_hi
        ym = int(y_min) if y_min is not None else y_usable_lo
        yM = int(y_max) if y_max is not None else y_usable_hi
    else:
        if mode == "full":
            xm, xM, ym, yM = 0, nx, 0, ny
        else:
            x_mid = nx // 2
            y_mid = ny // 2
            if mode == "tr":
                xm = max(x_usable_lo, x_mid)
                xM = x_usable_hi
                ym = max(y_usable_lo, y_mid)
                yM = y_usable_hi
            elif mode == "tl":
                xm = x_usable_lo
                xM = min(x_usable_hi, x_mid)
                ym = max(y_usable_lo, y_mid)
                yM = y_usable_hi
            elif mode == "br":
                xm = max(x_usable_lo, x_mid)
                xM = x_usable_hi
                ym = y_usable_lo
                yM = min(y_usable_hi, y_mid)
            else:  # bl
                xm = x_usable_lo
                xM = min(x_usable_hi, x_mid)
                ym = y_usable_lo
                yM = min(y_usable_hi, y_mid)

    xm = max(0, min(xm, nx - 1))
    xM = max(1, min(xM, nx))
    ym = max(0, min(ym, ny - 1))
    yM = max(1, min(yM, ny))

    if xm >= xM or ym >= yM:
        raise ValueError(
            f"Invalid crop bounds: x=[{xm}, {xM}), y=[{ym}, {yM}). "
            f"FFI shape: {ny}×{nx}."
        )

    bounds = {
        "x_min": xm,
        "x_max": xM,
        "y_min": ym,
        "y_max": yM,
        "shape": (yM - ym, xM - xm),
    }
    log.info(
        f"Crop bounds: x=[{xm}, {xM}), y=[{ym}, {yM}), "
        f"shape={bounds['shape']} (ny×nx)"
    )
    return bounds


def get_target_box_crop_bounds(
    ffi_header,
    target_ra: float,
    target_dec: float,
    *,
    box_size: int = 1024,
) -> dict:
    """
    Square crop centered on ``(target_ra, target_dec)``, edge-clamped to the FFI.

    Returns the same dict shape as :func:`get_crop_bounds`.
    """
    nx = int(ffi_header["NAXIS1"])
    ny = int(ffi_header["NAXIS2"])
    box_size = int(box_size)
    if box_size < 1:
        raise ValueError(f"box_size must be positive, got {box_size}")

    wcs = WCS(ffi_header)
    coord = SkyCoord(ra=float(target_ra), dec=float(target_dec), unit="deg")
    tx, ty = world_ra_dec_to_pixel(wcs, coord.ra.deg, coord.dec.deg)
    tx, ty = float(tx), float(ty)
    if not (0 <= tx < nx and 0 <= ty < ny):
        raise ValueError(
            f"Target ({target_ra}, {target_dec}) projects to ({tx:.2f}, {ty:.2f}) "
            f"outside FFI [0, {nx}) × [0, {ny})."
        )

    half = box_size // 2
    cx = int(round(tx))
    cy = int(round(ty))
    xm = cx - half
    ym = cy - half
    xM = xm + box_size
    yM = ym + box_size

    if xm < 0:
        xM -= xm
        xm = 0
    if ym < 0:
        yM -= ym
        ym = 0
    if xM > nx:
        shift = xM - nx
        xm -= shift
        xM = nx
    if yM > ny:
        shift = yM - ny
        ym -= shift
        yM = ny

    xm = max(0, xm)
    ym = max(0, ym)
    xM = min(nx, max(xM, xm + 1))
    yM = min(ny, max(yM, ym + 1))

    bounds = {
        "x_min": xm,
        "x_max": xM,
        "y_min": ym,
        "y_max": yM,
        "shape": (yM - ym, xM - xm),
    }
    log.info(
        "Target-box crop: x=[%d, %d), y=[%d, %d), shape=%s",
        xm,
        xM,
        ym,
        yM,
        bounds["shape"],
    )
    return bounds


def resolve_crop_bounds_from_params(
    ffi_header,
    *,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    crop_mode: str | None = "full",
    crop_box_size: int = 1024,
    target_ra: float | None = None,
    target_dec: float | None = None,
    x_left_dead: int = 44,
    x_right_dead: int = 44,
    y_edge_strip: int = 30,
) -> dict:
    """Shared crop resolver for template wcs_grouping and diff bootstrap."""
    explicit = any(v is not None for v in (x_min, x_max, y_min, y_max))
    if explicit:
        return get_crop_bounds(
            ffi_header,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            crop_mode="full",
            x_left_dead=x_left_dead,
            x_right_dead=x_right_dead,
            y_edge_strip=y_edge_strip,
        )
    mode = str(crop_mode or "full").strip().lower()
    if mode == "target_box":
        if target_ra is None or target_dec is None:
            raise ValueError(
                "target_ra and target_dec are required when crop_mode is 'target_box'"
            )
        return get_target_box_crop_bounds(
            ffi_header,
            target_ra,
            target_dec,
            box_size=crop_box_size,
        )
    return get_crop_bounds(
        ffi_header,
        crop_mode=mode,
        x_left_dead=x_left_dead,
        x_right_dead=x_right_dead,
        y_edge_strip=y_edge_strip,
    )


def diff_crop_explicitly_configured(cfg) -> bool:
    """True when diff_config crop fields should override cluster JSON."""
    if any(
        getattr(cfg, k, None) is not None
        for k in ("x_min", "x_max", "y_min", "y_max")
    ):
        return True
    mode = str(getattr(cfg, "crop_mode", None) or "").strip().lower()
    if mode == "target_box":
        return True
    return mode not in ("", "full")


def resolve_diff_crop_bounds(cfg, event_dir: str) -> dict:
    """Diff bootstrap crop: ``diff_config`` override or cluster JSON default."""
    from astropy.io import fits

    ref_path = load_reference_ffi_path(event_dir, getattr(cfg, "ref_ffi_path", None))
    with fits.open(ref_path, memmap=True) as hdul:
        ref_header = hdul[1].header

    if diff_crop_explicitly_configured(cfg):
        bounds = resolve_crop_bounds_from_params(
            ref_header,
            x_min=getattr(cfg, "x_min", None),
            x_max=getattr(cfg, "x_max", None),
            y_min=getattr(cfg, "y_min", None),
            y_max=getattr(cfg, "y_max", None),
            crop_mode=getattr(cfg, "crop_mode", None),
            crop_box_size=int(getattr(cfg, "crop_box_size", 1024)),
            target_ra=getattr(cfg, "target_ra", None),
            target_dec=getattr(cfg, "target_dec", None),
            x_left_dead=int(getattr(cfg, "x_left_dead", 44)),
            x_right_dead=int(getattr(cfg, "x_right_dead", 44)),
            y_edge_strip=int(getattr(cfg, "y_edge_strip", 30)),
        )
        log.info("Diff crop from diff_config override")
        return bounds

    bounds = load_crop_bounds(event_dir)
    log.info("Diff crop inherited from cluster_template_job.json")
    return bounds


def crop_image(data: np.ndarray, bounds: dict) -> np.ndarray:
    """Apply crop bounds dict to a 2D array."""
    return data[bounds["y_min"]:bounds["y_max"], bounds["x_min"]:bounds["x_max"]]


def crop_ffi_header(ffi_path: str, crop_bounds: dict) -> fits.Header:
    """
    Return a SIP-safe cropped copy of the science FFI HDU1 header.

    For a rectangular subimage crop (no resampling), shift ``CRPIX`` and update
    ``NAXIS``; all SIP polynomial keys (``A_*``, ``B_*``, etc.) are preserved
    unchanged from the full FFI header.
    """
    x_min = int(crop_bounds["x_min"])
    x_max = int(crop_bounds["x_max"])
    y_min = int(crop_bounds["y_min"])
    y_max = int(crop_bounds["y_max"])
    ny_crop, nx_crop = (int(crop_bounds["shape"][0]), int(crop_bounds["shape"][1]))

    with fits.open(ffi_path, memmap=True) as hdul:
        hdr = deepcopy(hdul[1].header)

    if "CRPIX1" in hdr:
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) - x_min
    if "CRPIX2" in hdr:
        hdr["CRPIX2"] = float(hdr["CRPIX2"]) - y_min

    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx_crop
    hdr["NAXIS2"] = ny_crop

    hdr.set("XMIN", x_min, "Crop xmin in full FFI pixels")
    hdr.set("XMAX", x_max, "Crop xmax (exclusive) in full FFI pixels")
    hdr.set("YMIN", y_min, "Crop ymin in full FFI pixels")
    hdr.set("YMAX", y_max, "Crop ymax (exclusive) in full FFI pixels")
    hdr.set("ROIW", x_max - x_min, "Crop width in full FFI pixels")
    hdr.set("ROIH", y_max - y_min, "Crop height in full FFI pixels")

    return hdr


def log_gaia_crop_alignment(gaia_df: pd.DataFrame, crop_bounds: dict) -> None:
    """
    Log catalog x,y range and the fraction of rows inside the crop shape.

    Call before ePSF fitting to spot FFI-vs-crop-local mix-ups early.
    """
    if gaia_df is None or len(gaia_df) == 0:
        log.info("Gaia vs crop: catalog is empty.")
        return
    if "x" not in gaia_df.columns or "y" not in gaia_df.columns:
        log.info(
            "Gaia vs crop: no x,y columns (sky-only or not yet projected)."
        )
        return
    ny, nx = crop_bounds["shape"]
    x = pd.to_numeric(gaia_df["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(gaia_df["y"], errors="coerce").to_numpy(dtype=float)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x < nx)
        & (y >= 0)
        & (y < ny)
    )
    frac = float(np.mean(inside)) if x.size else 0.0
    log.info(
        "Gaia vs crop: N=%d, x [%.2f, %.2f], y [%.2f, %.2f]; "
        "crop (ny,nx)=%s; fraction inside crop=%.1f%%",
        len(gaia_df),
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
        (ny, nx),
        100.0 * frac,
    )


def ensure_gaia_crop_xy(
    gaia_df: pd.DataFrame,
    ref_ffi_path: str,
    crop_bounds: dict,
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
    xy_in_crop_fraction_min: float = 0.5,
) -> pd.DataFrame:
    """
    Ensure crop-local ``x``, ``y`` columns suitable for masking and ePSF.

    * If ``x`` and ``y`` are already present **and** at least a fraction
      ``xy_in_crop_fraction_min`` of rows lie inside ``[0, nx) × [0, ny)``
      (``crop_bounds["shape"]`` is ``(ny, nx)``), return a copy unchanged.
    * If ``x``/``y`` look inconsistent with the crop (common when catalog
      pixels are full-chip but labeled as crop-local), they are dropped and
      ``x_ffi``/``y_ffi`` or ``ra``/``dec`` are used when available.
    * Else if ``x_ffi`` / ``y_ffi`` are present, rebase to crop-local coords and
      keep only rows inside the crop (same as ``build_unique_gaia_catalog``).
    * Else if ``ra`` / ``dec`` are present, project with the reference FFI WCS
      (HDU 1), keep on-chip sources inside ``crop_bounds``, then set
      ``x = x_ffi - x_min``, ``y = y_ffi - y_min``.

    Parameters
    ----------
    gaia_df : pd.DataFrame
    ref_ffi_path : str
        Path to reference TESS FFI (WCS read from extension 1).
    crop_bounds : dict
        ``x_min``, ``x_max``, ``y_min``, ``y_max`` from :func:`get_crop_bounds`.
    ra_col, dec_col : str
        Sky coordinate column names for the WCS branch.
    xy_in_crop_fraction_min : float
        Minimum fraction of finite ``x``, ``y`` rows that must fall inside the
        crop to trust pre-existing pixel columns (default ``0.5``).

    Returns
    -------
    pd.DataFrame
    """
    df = gaia_df.copy()
    ny, nx = crop_bounds["shape"]

    if ra_col in df.columns and dec_col in df.columns:
        df = df.drop(columns=["x", "y"], errors="ignore")

    if "x" in df.columns and "y" in df.columns:
        if len(df) == 0:
            return df.reset_index(drop=True)
        xv = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        yv = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        inside = (
            np.isfinite(xv)
            & np.isfinite(yv)
            & (xv >= 0)
            & (xv < nx)
            & (yv >= 0)
            & (yv < ny)
        )
        frac = float(np.mean(inside))
        if frac >= xy_in_crop_fraction_min:
            return df.reset_index(drop=True)
        log.warning(
            "Gaia catalog: only %.1f%% of rows have x,y inside the image crop "
            "[0,%d)×[0,%d); dropping x,y and re-deriving positions if possible.",
            100.0 * frac,
            nx,
            ny,
        )
        df = df.drop(columns=["x", "y"], errors="ignore")

    x_min = crop_bounds["x_min"]
    y_min = crop_bounds["y_min"]
    x_max = crop_bounds["x_max"]
    y_max = crop_bounds["y_max"]

    if "x_ffi" in df.columns and "y_ffi" in df.columns:
        in_crop = (
            (df["x_ffi"] >= x_min)
            & (df["x_ffi"] < x_max)
            & (df["y_ffi"] >= y_min)
            & (df["y_ffi"] < y_max)
        )
        out = df[in_crop].copy()
        out["x"] = out["x_ffi"] - x_min
        out["y"] = out["y_ffi"] - y_min
        out = out.drop(columns=[c for c in ("x_ffi", "y_ffi") if c in out.columns])
        n = len(out)
        log.info(
            f"Gaia catalog: rebased x_ffi/y_ffi to crop-local x,y; {n} stars in crop"
        )
        return out.reset_index(drop=True)

    if ra_col not in df.columns or dec_col not in df.columns:
        raise ValueError(
            "Gaia catalog needs crop-local 'x'/'y', or 'x_ffi'/'y_ffi', or "
            f"'{ra_col}'/'{dec_col}' for WCS projection (got columns: "
            f"{list(df.columns)!r})"
        )

    with fits.open(ref_ffi_path, memmap=True) as hdul:
        ref_header = hdul[1].header
        nx = int(ref_header["NAXIS1"])
        ny = int(ref_header["NAXIS2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(ref_header)

    coords = SkyCoord(
        ra=df[ra_col].values,
        dec=df[dec_col].values,
        unit="deg",
    )
    x_pix, y_pix = world_ra_dec_to_pixel(wcs, coords.ra.deg, coords.dec.deg)
    df = df.copy()
    df["x_ffi"] = x_pix
    df["y_ffi"] = y_pix

    on_chip = (
        (df["x_ffi"] >= 0)
        & (df["x_ffi"] < nx)
        & (df["y_ffi"] >= 0)
        & (df["y_ffi"] < ny)
    )
    n_all = len(df)
    n_on = int(on_chip.sum())
    df = df[on_chip].copy()
    log.info(f"Gaia catalog: {n_on} / {n_all} rows on FFI chip after WCS")

    in_crop = (
        (df["x_ffi"] >= x_min)
        & (df["x_ffi"] < x_max)
        & (df["y_ffi"] >= y_min)
        & (df["y_ffi"] < y_max)
    )
    cropped = df[in_crop].copy()
    log.info(f"Gaia catalog: {len(cropped)} stars within crop after WCS projection")

    cropped["x"] = cropped["x_ffi"] - x_min
    cropped["y"] = cropped["y_ffi"] - y_min
    cropped = cropped.drop(columns=["x_ffi", "y_ffi"]).reset_index(drop=True)
    return cropped


def build_unique_gaia_catalog(removed_stars_csv: str,
                               ref_ffi_path: str,
                               crop_bounds: dict,
                               output_dir: str) -> pd.DataFrame:
    """
    Create the Gaia catalog used by epsf_fitting and sat_template, derived
    from the removed_stars CSV produced by the PS1 pipeline.

    Steps:
      1. Load removed_stars_csv; deduplicate by source_id; drop unmatched
         rows (source_id == -1).
      2. Keep diagnostic columns: source_id, ra, dec, tess_mag,
         phot_rp_mean_mag, phot_g_mean_mag, phot_bp_mean_mag.
      3. Open ref_ffi_path (HDU 1); build WCS; project (ra, dec) → (x, y).
      4. Filter: keep only stars on chip (0 ≤ x < nx, 0 ≤ y < ny).
      5. Further filter: x ≥ x_min and y ≥ y_min (within the crop quadrant).
      6. Rebase: x -= x_min;  y -= y_min  (crop-local coordinates).
      7. Save output_dir/unique_gaia_stars_for_cropped_template.csv.

    Parameters
    ----------
    removed_stars_csv : str
    ref_ffi_path : str
    crop_bounds : dict  (from get_crop_bounds)
    output_dir : str

    Returns
    -------
    pd.DataFrame with columns: source_id, ra, dec, tess_mag, phot_*, x, y
    """
    log.info(f"Building unique Gaia catalog from {removed_stars_csv} ...")

    ps1_df = pd.read_csv(removed_stars_csv)

    # Deduplicate by Gaia source_id; drop unmatched entries
    unique_df = ps1_df.drop_duplicates(subset="source_id").copy()
    unique_df = unique_df[unique_df["source_id"] != -1].copy()
    log.info(f"  {len(ps1_df)} rows → {len(unique_df)} unique Gaia matches after dedup")

    # Keep only required columns (gracefully handle missing optional columns)
    keep_cols = ["source_id", "ra", "dec", "tess_mag"]
    for col in ("phot_rp_mean_mag", "phot_g_mean_mag", "phot_bp_mean_mag"):
        if col in unique_df.columns:
            keep_cols.append(col)
    unique_df = unique_df[keep_cols].reset_index(drop=True)

    # Project sky coords to FFI pixel coords using reference FFI WCS
    with fits.open(ref_ffi_path, memmap=True) as hdul:
        ref_header = hdul[1].header
        nx = int(ref_header["NAXIS1"])
        ny = int(ref_header["NAXIS2"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(ref_header)

    coords = SkyCoord(
        ra=unique_df["ra"].values,
        dec=unique_df["dec"].values,
        unit="deg",
    )
    x_pix, y_pix = world_ra_dec_to_pixel(wcs, coords.ra.deg, coords.dec.deg)
    unique_df["x_ffi"] = x_pix
    unique_df["y_ffi"] = y_pix

    # Filter: on chip
    on_chip = (
        (unique_df["x_ffi"] >= 0) & (unique_df["x_ffi"] < nx) &
        (unique_df["y_ffi"] >= 0) & (unique_df["y_ffi"] < ny)
    )
    unique_df = unique_df[on_chip].copy()
    log.info(f"  {on_chip.sum()} stars on chip")

    # Filter: within crop quadrant
    x_min = crop_bounds["x_min"]
    y_min = crop_bounds["y_min"]
    x_max = crop_bounds["x_max"]
    y_max = crop_bounds["y_max"]
    in_crop = (
        (unique_df["x_ffi"] >= x_min) & (unique_df["x_ffi"] < x_max) &
        (unique_df["y_ffi"] >= y_min) & (unique_df["y_ffi"] < y_max)
    )
    cropped_df = unique_df[in_crop].copy()
    log.info(f"  {len(cropped_df)} stars within crop bounds")

    # Rebase to crop-local coordinates
    cropped_df["x"] = cropped_df["x_ffi"] - x_min
    cropped_df["y"] = cropped_df["y_ffi"] - y_min
    cropped_df = cropped_df.drop(columns=["x_ffi", "y_ffi"]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "unique_gaia_stars_for_cropped_template.csv")
    cropped_df.to_csv(out_path, index=False)
    log.info(f"  Unique Gaia catalog saved to {out_path}")

    return cropped_df


def save_crop_bounds(bounds: dict, output_dir: str) -> None:
    """Persist crop bounds to ``crop_bounds.json`` (legacy debugging; prefer cluster JSON)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "crop_bounds.json")
    with open(path, "w") as fh:
        json.dump(bounds, fh, indent=2)
    log.info(f"Crop bounds saved to {path}")


def load_crop_bounds(output_dir: str) -> dict:
    """
    Load crop bounds from ``cluster_template_job.json`` (preferred) or legacy
    ``crop_bounds.json``.
    """
    job_path = os.path.join(output_dir, CLUSTER_TEMPLATE_JOB_FILENAME)
    if os.path.isfile(job_path):
        with open(job_path) as fh:
            return crop_bounds_from_cluster_payload(json.load(fh))
    legacy = os.path.join(output_dir, "crop_bounds.json")
    if os.path.isfile(legacy):
        with open(legacy) as fh:
            d = json.load(fh)
        d["shape"] = tuple(d["shape"])
        return d
    raise FileNotFoundError(
        f"No crop bounds: expected {job_path} or {legacy}"
    )
