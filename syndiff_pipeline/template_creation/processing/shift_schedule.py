"""
shift_schedule.py
==================
Per-skycell PS1-pixel shift schedule and signature-based template grouping
for distortion-aware templates: measures real per-frame WCS drift directly
at each skycell center (no interpolation -- every skycell's own sky position
is round-tripped through that frame's own WCS), Savitzky-Golay smooths each
skycell's time series per **TESS orbit segment**, converts the smoothed
TESS-pixel drift to a hysteresis-rounded PS1-pixel shift schedule, and turns
the schedule into signature-based template groups plus the frozen
``template_group_shifts.parquet`` / ``template_groups.json`` handoff.

Non-measurable frames (missing WCS or pre-SG 5σ MAD outliers) are **not
dropped**: their shifts are synthesized so every FFI still participates in
remap and template (L5).  Provenance is stored in ``frame_origin`` (NPZ) and
``frames_missing_wcs`` / ``frames_sigma_clipped`` (JSON sidecar).

Synthesis policy (v1):
  - Interior gaps: hold last measurable quantized ``(sx_int, sy_int)``;
    floats = ``float(int)``.
  - Leading / trailing gaps: flat-line extrapolation (constant = first /
    last measurable float and int values).

Orbit segments prefer the MIT ``TESS_orbit_times.csv`` (via
``ensure_tess_orbit_times_csv``); ``sector`` + ``frame_times`` are required
(no btjd-gap fallback).

Ports validated prototype logic into the package (per project policy, this
module does not import from the scratch dirs):

- vectorized TESS-drift -> PS1-shift conversion, the Savitzky-Golay-aware
  fill pattern, hysteresis rounding, and RLE compression:
  ``scripts/verify_seam_remap/01_shift_schedule.py``
  (``compute_ps1_shift_vectorized``, ``hysteresis_round_series``,
  ``build_rle_dataframe``).
- quantized-shift cache keying:
  ``scripts/verify_seam_remap/03_seam_remap.py``
  (``quantize_shift``, ``encode_quantized_shift``).
- per-orbit-segment Savitzky-Golay smoothing (``_split_orbit_segments``,
  ``_sg_smooth_series``): relocated from the now-deleted
  ``common/wcs_drift_field.py``.

See ``docs/markdown/field_geometry.md`` for grouping/cache quantum details
and the synthesis / ``frame_origin`` bookkeeping policy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from scipy.signal import savgol_filter

from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
    build_ps1_wcs,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Per-frame provenance in shift_schedule.npz ``frame_origin`` (int8).
FRAME_ORIGIN_MEASURED = 0
FRAME_ORIGIN_SYNTH_MISSING_WCS = 1
FRAME_ORIGIN_SYNTH_SIGMA_CLIPPED = 2

SYNTHESIS_POLICY = "interior_hold_quantized_edges_flat"


# ── Savitzky-Golay smoothing (relocated from common/wcs_drift_field.py) ────

def _sg_smooth_series(
    data: np.ndarray, valid: np.ndarray, window: int, polyorder: int
) -> np.ndarray:
    """
    Savitzky-Golay smooth ``data`` along axis 0 at ``valid & isfinite`` entries;
    other entries are left untouched (NaN stays NaN). Window is capped to the
    number of valid samples and forced odd; returns ``data`` unchanged if fewer
    than 3 valid samples remain after capping.
    """
    out = data.copy()
    valid_idx = np.flatnonzero(valid & np.isfinite(data))
    if len(valid_idx) < 3:
        return out
    wl = min(int(window), len(valid_idx) - (1 - len(valid_idx) % 2))
    if wl < 3:
        return out
    if wl % 2 == 0:
        wl -= 1
    po = min(int(polyorder), wl - 1)
    out[valid_idx] = savgol_filter(data[valid_idx], wl, po, mode="interp")
    return out


def _split_orbit_segments(
    btjd: np.ndarray | None, n_frames: int, gap_threshold_days: float
) -> np.ndarray:
    """
    Split ``[0, n_frames)`` into ``[start, end)`` segments wherever the btjd
    gap between consecutive frames exceeds ``gap_threshold_days``. Frames are
    assumed already in time order (manifest row order). One segment when
    ``btjd`` is ``None`` or gaps can't be measured (NaN on either side never
    forces a split).
    """
    if btjd is None or n_frames == 0:
        return np.array([[0, n_frames]], dtype=np.int64)

    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, n_frames):
        b0, b1 = btjd[i - 1], btjd[i]
        if np.isfinite(b0) and np.isfinite(b1) and (b1 - b0) > gap_threshold_days:
            bounds.append((start, i))
            start = i
    bounds.append((start, n_frames))
    return np.array(bounds, dtype=np.int64)


def _split_orbit_segments_from_csv(
    sector: int,
    frame_times: Sequence[str],
    csv_path: str | Path,
) -> np.ndarray:
    """
    Split ``[0, n_frames)`` into ``[start, end)`` segments using MIT
    ``TESS_orbit_times.csv`` windows for ``sector``.

    Each frame is assigned the orbit row whose ``[Start of Orbit, End of Orbit]``
    contains its ``date_obs``. Boundaries are placed wherever the assigned
    orbit index changes between consecutive frames. Frames outside all
    windows get orbit id ``-1`` (still produce a segment if they cluster).
    """
    from astropy.time import Time

    n_frames = len(frame_times)
    if n_frames == 0:
        return np.array([[0, 0]], dtype=np.int64)

    orbit_df = pd.read_csv(csv_path, skipfooter=1, engine="python")
    sector_str = str(int(sector))
    rows = orbit_df[orbit_df["Sector"].astype(str) == sector_str].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No orbit rows for sector {sector_str} in {csv_path}")

    starts = [
        Time(str(v).replace(" ", "T"), format="isot", scale="utc").mjd
        for v in rows["Start of Orbit"]
    ]
    ends = [
        Time(str(v).replace(" ", "T"), format="isot", scale="utc").mjd
        for v in rows["End of Orbit"]
    ]

    orbit_ids = np.full(n_frames, -1, dtype=np.int32)
    for i, ts in enumerate(frame_times):
        if ts is None or (isinstance(ts, float) and not np.isfinite(ts)):
            continue
        try:
            mjd = Time(str(ts).replace(" ", "T"), format="isot", scale="utc").mjd
        except Exception:
            continue
        for oi, (s, e) in enumerate(zip(starts, ends)):
            if s <= mjd <= e:
                orbit_ids[i] = oi
                break

    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, n_frames):
        if orbit_ids[i] != orbit_ids[i - 1]:
            bounds.append((start, i))
            start = i
    bounds.append((start, n_frames))
    return np.array(bounds, dtype=np.int64)


def _reject_raw_drift_outliers(
    drift_raw: np.ndarray,
    measurable: np.ndarray,
    segment_bounds: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, list[dict]]:
    """
    Per orbit segment, MAD-based rejection on frame-level median |tess drift|.

    Modifies ``measurable`` in place (sets outliers False). Returns the
    newly rejected frame indices and audit rows for the JSON sidecar.
    """
    if sigma is None or not np.isfinite(sigma) or float(sigma) <= 0:
        return np.array([], dtype=np.int64), []

    sigma = float(sigma)
    mag = np.hypot(drift_raw[:, :, 0], drift_raw[:, :, 1])
    with np.errstate(all="ignore"):
        # NaN frames (non-measurable) yield All-NaN rows; ignore the warning.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame_med = np.nanmedian(mag, axis=1)

    newly: list[int] = []
    audit: list[dict] = []
    for seg_start, seg_end in segment_bounds:
        seg = slice(int(seg_start), int(seg_end))
        cand = measurable[seg] & np.isfinite(frame_med[seg])
        if int(cand.sum()) < 3:
            continue
        vals = frame_med[seg][cand]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        # Floor the scale so a single huge spike among identical quiet frames
        # is still rejected (MAD would otherwise be exactly 0).
        scale = max(mad, 1e-3)
        thresh = sigma * 1.4826 * scale
        local_idx = np.flatnonzero(cand)
        for li in local_idx:
            f = int(seg_start) + int(li)
            if abs(float(frame_med[f]) - med) > thresh:
                measurable[f] = False
                newly.append(f)
                audit.append(
                    {
                        "frame": f,
                        "median_tess_drift_px": float(frame_med[f]),
                        "segment": [int(seg_start), int(seg_end)],
                    }
                )
    return np.asarray(newly, dtype=np.int64), audit


def _synthesize_shift_gaps(
    sx_float: np.ndarray,
    sy_float: np.ndarray,
    sx_int: np.ndarray,
    sy_int: np.ndarray,
    measurable: np.ndarray,
) -> None:
    """
    Fill non-measurable frames in place.

    Interior: hold last measurable quantized ints (floats = float(int)).
    Leading / trailing: flat-line from first / last measurable float+int.
    """
    n_frames, n_cells = sx_float.shape
    meas_idx = np.flatnonzero(measurable)
    if meas_idx.size == 0:
        log.warning("Shift schedule: no measurable frames; cannot synthesize gaps")
        return

    first = int(meas_idx[0])
    last = int(meas_idx[-1])

    for c in range(n_cells):
        if first > 0:
            sx_float[:first, c] = sx_float[first, c]
            sy_float[:first, c] = sy_float[first, c]
            sx_int[:first, c] = sx_int[first, c]
            sy_int[:first, c] = sy_int[first, c]

        if last < n_frames - 1:
            sx_float[last + 1 :, c] = sx_float[last, c]
            sy_float[last + 1 :, c] = sy_float[last, c]
            sx_int[last + 1 :, c] = sx_int[last, c]
            sy_int[last + 1 :, c] = sy_int[last, c]

        prev = first
        for f in range(first + 1, last + 1):
            if measurable[f]:
                prev = f
                continue
            sx_int[f, c] = sx_int[prev, c]
            sy_int[f, c] = sy_int[prev, c]
            sx_float[f, c] = float(sx_int[prev, c])
            sy_float[f, c] = float(sy_int[prev, c])


# ── quantization / cache-key encoding (ported from 03_seam_remap.py) ───────

def quantize_shift(v, quantum: float) -> float:
    """
    Round ``v`` to the nearest multiple of ``quantum``: ``round(v/quantum)*quantum``.

    ``v`` may be a scalar or an array; scalar input returns a python
    ``float``, array input returns an ``ndarray`` (used internally to
    quantize a whole shift column at once).
    """
    arr = np.asarray(v, dtype=np.float64)
    q = np.round(arr / quantum) * quantum
    if arr.ndim == 0:
        return float(q)
    return q


def encode_quantized_shift(q: float) -> str:
    """Filename-safe encoding of a quantized shift: ``+1.25 -> 'p1p25'``, ``-0.5 -> 'n0p50'``."""
    sign = "p" if q >= 0 else "n"
    mag = abs(float(q))
    body = f"{mag:.2f}".replace(".", "p")
    return f"{sign}{body}"


def cache_key_for(qx: float, qy: float) -> str:
    """Cache key string for a quantized (qx, qy) pair: ``qx{enc}_qy{enc}``."""
    return f"qx{encode_quantized_shift(qx)}_qy{encode_quantized_shift(qy)}"


# ── TESS-drift -> PS1-shift (ported from 01_shift_schedule.py) ─────────────

def compute_ps1_shift_vectorized(
    tess_wcs: WCS,
    dx_tess: np.ndarray,
    dy_tess: np.ndarray,
    sky_ra_deg: float,
    sky_dec_deg: float,
    ps1_wcs: WCS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized per-frame TESS-pixel drift -> PS1-pixel shift at one sky
    point: map the point to its TESS pixel position, perturb it by
    ``(dx_tess, dy_tess)`` (arrays over frames), and round-trip both the
    unperturbed and perturbed positions through ``ps1_wcs`` to get the
    resulting PS1-pixel displacement.

    Ported from ``scripts/verify_seam_remap/01_shift_schedule.py``.
    """
    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

    x_tess, y_tess = world_ra_dec_to_pixel(tess_wcs, sky_ra_deg, sky_dec_deg)
    ra1, dec1 = tess_wcs.pixel_to_world_values(x_tess, y_tess)
    ra2, dec2 = tess_wcs.pixel_to_world_values(x_tess + dx_tess, y_tess + dy_tess)
    u1, v1 = world_ra_dec_to_pixel(ps1_wcs, ra1, dec1)
    u2, v2 = world_ra_dec_to_pixel(ps1_wcs, ra2, dec2)
    return u2 - u1, v2 - v1


def hysteresis_round_series(frac: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """
    Stateful hysteresis rounding along time (axis 0): stays on the current
    integer bin until the value strays more than ``0.5 + margin`` away from
    it, which prevents single-frame flapping at bin edges.

    Ported from ``scripts/verify_seam_remap/01_shift_schedule.py``.
    """
    n = len(frac)
    # int32: PS1 shifts can exceed int16 after fpack/WCS cache rebuilds or
    # large skycell offsets (int16 overflowed at e.g. -69523).
    out = np.empty(n, dtype=np.int32)
    if n == 0:
        return out
    current = int(round(float(frac[0])))
    out[0] = current
    threshold = 0.5 + margin
    for t in range(1, n):
        f = float(frac[t])
        candidate = int(round(f))
        if candidate != current and abs(f - current) > threshold:
            current = candidate
        out[t] = current
    return out


def _hysteresis_round_measurable(
    values: np.ndarray, measurable: np.ndarray, margin: float
) -> np.ndarray:
    """
    Hysteresis-round **measurable** frames only.

    Non-measurable slots are left as ``0`` and filled later by
    :func:`_synthesize_shift_gaps`. Measurable frames are hysteresis-rounded
    as a time-ordered subsequence (gaps are skipped — no linear interpolation
    into the hysteresis state).
    """
    n = len(values)
    out = np.zeros(n, dtype=np.int32)
    meas_idx = np.flatnonzero(np.asarray(measurable, dtype=bool))
    if meas_idx.size == 0:
        return out
    meas_vals = np.asarray(values, dtype=np.float64)[meas_idx]
    out[meas_idx] = hysteresis_round_series(meas_vals, margin)
    return out


def build_rle_dataframe(
    skycell_names: np.ndarray,
    sx_int: np.ndarray,
    sy_int: np.ndarray,
) -> pd.DataFrame:
    """
    Run-length encode the integer shift schedule per skycell: one row per
    maximal run of constant ``(sx_int, sy_int)``.

    Ported from ``scripts/verify_seam_remap/01_shift_schedule.py``.
    """
    n_frames, n_cells = sx_int.shape
    rows: list[dict] = []
    for c in range(n_cells):
        sx_col = sx_int[:, c]
        sy_col = sy_int[:, c]
        start = 0
        while start < n_frames:
            end = start
            while (
                end + 1 < n_frames
                and sx_col[end + 1] == sx_col[start]
                and sy_col[end + 1] == sy_col[start]
            ):
                end += 1
            rows.append(
                {
                    "skycell": str(skycell_names[c]),
                    "seg_start_frame_idx": int(start),
                    "seg_end_frame_idx": int(end),
                    "sx_int": int(sx_col[start]),
                    "sy_int": int(sy_col[start]),
                    "n_frames": int(end - start + 1),
                }
            )
            start = end + 1
    return pd.DataFrame(rows)


# ── ShiftSchedule ────────────────────────────────────────────────────────

@dataclass
class ShiftSchedule:
    """Per-skycell PS1-pixel shift time series for one event (frame axis = manifest row order)."""

    skycell_names: np.ndarray   # (C,)
    sx_float: np.ndarray        # (F, C) f4 — drift-field-derived shift, smoothed
    sy_float: np.ndarray        # (F, C) f4
    sx_int: np.ndarray          # (F, C) i4 — hysteresis-rounded
    sy_int: np.ndarray          # (F, C) i4
    frame_valid: np.ndarray     # (F,) bool — True for all frames after synthesis
    meta: dict
    frame_origin: np.ndarray | None = None  # (F,) i1 — see FRAME_ORIGIN_* constants

    def to_rle_dataframe(self) -> pd.DataFrame:
        """Run-length-encoded integer shift segments; see :func:`build_rle_dataframe`."""
        return build_rle_dataframe(self.skycell_names, self.sx_int, self.sy_int)

    def save(self, npz_path: str | Path) -> None:
        """Write the NPZ arrays plus a JSON sidecar (same stem, ``.json``)."""
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        origin = self.frame_origin
        if origin is None:
            origin = np.zeros(self.frame_valid.shape[0], dtype=np.int8)
        np.savez(
            npz_path,
            skycell_names=self.skycell_names,
            sx_float=self.sx_float,
            sy_float=self.sy_float,
            sx_int=self.sx_int,
            sy_int=self.sy_int,
            frame_valid=self.frame_valid,
            frame_origin=np.asarray(origin, dtype=np.int8),
        )
        json_path = npz_path.with_suffix(".json")
        with open(json_path, "w") as fh:
            json.dump(self.meta, fh, indent=2)
        log.info("Shift schedule written to %s (+ %s)", npz_path, json_path.name)

    @classmethod
    def load(cls, npz_path: str | Path) -> "ShiftSchedule":
        """Load a :class:`ShiftSchedule` from its NPZ + JSON sidecar."""
        npz_path = Path(npz_path)
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}

        json_path = npz_path.with_suffix(".json")
        meta: dict = {}
        if json_path.is_file():
            with open(json_path) as fh:
                meta = json.load(fh)
        else:
            log.warning("Shift schedule sidecar JSON missing: %s", json_path)

        frame_valid = arrays["frame_valid"]
        if "frame_origin" in arrays:
            frame_origin = np.asarray(arrays["frame_origin"], dtype=np.int8)
        else:
            frame_origin = np.zeros(len(frame_valid), dtype=np.int8)

        return cls(
            skycell_names=arrays["skycell_names"],
            sx_float=arrays["sx_float"],
            sy_float=arrays["sy_float"],
            sx_int=arrays["sx_int"],
            sy_int=arrays["sy_int"],
            frame_valid=frame_valid,
            meta=meta,
            frame_origin=frame_origin,
        )


def build_skycell_shift_schedule(
    frames: Sequence[tuple[str, Optional[WCS]]],
    skycell_df: pd.DataFrame,
    ref_wcs: WCS,
    *,
    btjd: np.ndarray | None = None,
    frame_times: Sequence[str] | None = None,
    sector: int | None = None,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    orbit_gap_threshold_days: float = 0.5,
    hysteresis_margin: float = 0.1,
    raw_drift_outlier_sigma: float | None = 5.0,
    orbit_csv_path: str | Path | None = None,
) -> ShiftSchedule:
    """
    Measure real per-frame WCS drift directly at every skycell center and
    convert it to a hysteresis-rounded PS1-pixel shift schedule.

    ``frames`` is ``(filename, WCS-or-None)`` in manifest row order. Frames
    with ``WCS is None`` (missing / ``wcs_ok=False``) are non-measurable:
    they do not enter Savitzky–Golay, and their shifts are synthesized
    after the measurable path (see module docstring).

    Orbit segments: **requires** ``sector`` and ``frame_times`` and splits with
    MIT ``TESS_orbit_times.csv`` (bundled via ``ensure_tess_orbit_times_csv``
    unless ``orbit_csv_path`` is given). There is no btjd-gap fallback.

    Pre-SG outlier gate: per orbit segment, frames whose median |TESS
    drift| exceeds ``raw_drift_outlier_sigma``×MAD are marked non-measurable
    (``None`` disables). After synthesis every frame has ``frame_valid=True``.
    """
    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel

    if sector is None or frame_times is None:
        raise ValueError(
            "build_skycell_shift_schedule requires sector and frame_times "
            "(MIT TESS_orbit_times.csv; no btjd-gap fallback)"
        )

    skycell_names = np.array([str(name) for name in skycell_df["NAME"]])
    n_cells = len(skycell_df)
    n_frames = len(frames)
    filenames = [str(fn) for fn, _ in frames]

    if len(frame_times) != n_frames:
        raise ValueError(
            f"frame_times length ({len(frame_times)}) does not match frames length ({n_frames})"
        )

    ra = skycell_df["RA"].to_numpy(dtype=np.float64)
    dec = skycell_df["DEC"].to_numpy(dtype=np.float64)
    x_ref, y_ref = world_ra_dec_to_pixel(ref_wcs, ra, dec)

    frame_origin = np.full(n_frames, FRAME_ORIGIN_SYNTH_MISSING_WCS, dtype=np.int8)
    frame_measurable = np.zeros(n_frames, dtype=bool)
    drift_raw = np.full((n_frames, n_cells, 2), np.nan, dtype=np.float64)
    for i, (_, wcs_f) in enumerate(frames):
        if wcs_f is None:
            continue
        try:
            fx, fy = wcs_f.world_to_pixel_values(ra, dec)
        except Exception as exc:
            log.warning("Shift schedule: WCS evaluation failed for frame index %d: %s", i, exc)
            continue
        drift_raw[i, :, 0] = np.asarray(fx, dtype=np.float64) - x_ref
        drift_raw[i, :, 1] = np.asarray(fy, dtype=np.float64) - y_ref
        if np.isfinite(drift_raw[i]).all():
            frame_measurable[i] = True
            frame_origin[i] = FRAME_ORIGIN_MEASURED
        else:
            frame_origin[i] = FRAME_ORIGIN_SYNTH_MISSING_WCS

    if btjd is not None and len(btjd) != n_frames:
        raise ValueError(f"btjd length ({len(btjd)}) does not match frames length ({n_frames})")

    if orbit_csv_path is None:
        from syndiff_pipeline.template_creation.orchestration.bundled_assets import (
            ensure_tess_orbit_times_csv,
        )

        csv_path = ensure_tess_orbit_times_csv()
    else:
        csv_path = Path(orbit_csv_path)
    segment_bounds = _split_orbit_segments_from_csv(int(sector), frame_times, csv_path)
    orbit_segment_source = "tess_orbit_times_csv"

    sigma_audit: list[dict] = []
    if raw_drift_outlier_sigma is not None:
        newly, sigma_audit = _reject_raw_drift_outliers(
            drift_raw, frame_measurable, segment_bounds, float(raw_drift_outlier_sigma)
        )
        for f in newly:
            frame_origin[int(f)] = FRAME_ORIGIN_SYNTH_SIGMA_CLIPPED
        for row in sigma_audit:
            f = int(row["frame"])
            row["filename"] = filenames[f] if f < len(filenames) else ""

    drift_smooth = drift_raw.copy()
    for seg_start, seg_end in segment_bounds:
        seg = slice(int(seg_start), int(seg_end))
        seg_valid = frame_measurable[seg]
        if int(seg_valid.sum()) < 3:
            continue
        for c in range(n_cells):
            for comp in range(2):
                drift_smooth[seg, c, comp] = _sg_smooth_series(
                    drift_raw[seg, c, comp], seg_valid, savgol_window, savgol_polyorder
                )

    sx_float = np.full((n_frames, n_cells), np.nan, dtype=np.float32)
    sy_float = np.full((n_frames, n_cells), np.nan, dtype=np.float32)
    sx_int = np.zeros((n_frames, n_cells), dtype=np.int32)
    sy_int = np.zeros((n_frames, n_cells), dtype=np.int32)

    for c in range(n_cells):
        row = skycell_df.iloc[c]
        ps1_wcs, _ = build_ps1_wcs(row)
        with np.errstate(invalid="ignore"):
            sx, sy = compute_ps1_shift_vectorized(
                ref_wcs, drift_smooth[:, c, 0], drift_smooth[:, c, 1],
                float(ra[c]), float(dec[c]), ps1_wcs
            )
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        sx_out = np.full(n_frames, np.nan, dtype=np.float64)
        sy_out = np.full(n_frames, np.nan, dtype=np.float64)
        sx_out[frame_measurable] = sx[frame_measurable]
        sy_out[frame_measurable] = sy[frame_measurable]
        sx_float[:, c] = sx_out.astype(np.float32)
        sy_float[:, c] = sy_out.astype(np.float32)

        if not frame_measurable.any():
            continue
        sx_int[:, c] = _hysteresis_round_measurable(
            sx_out, frame_measurable, hysteresis_margin
        )
        sy_int[:, c] = _hysteresis_round_measurable(
            sy_out, frame_measurable, hysteresis_margin
        )

    _synthesize_shift_gaps(sx_float, sy_float, sx_int, sy_int, frame_measurable)
    frame_valid = np.ones(n_frames, dtype=bool)

    frames_missing_wcs = [
        {"frame": int(f), "filename": filenames[int(f)]}
        for f in np.flatnonzero(frame_origin == FRAME_ORIGIN_SYNTH_MISSING_WCS)
    ]
    frames_sigma_clipped = list(sigma_audit)
    for row in frames_sigma_clipped:
        if "filename" not in row:
            f = int(row["frame"])
            row["filename"] = filenames[f] if f < len(filenames) else ""

    n_measured = int((frame_origin == FRAME_ORIGIN_MEASURED).sum())
    n_missing = int((frame_origin == FRAME_ORIGIN_SYNTH_MISSING_WCS).sum())
    n_clipped = int((frame_origin == FRAME_ORIGIN_SYNTH_SIGMA_CLIPPED).sum())

    meta = {
        "schema_version": SCHEMA_VERSION,
        "n_frames": int(n_frames),
        "n_skycells": int(n_cells),
        "savgol_window": int(savgol_window),
        "savgol_polyorder": int(savgol_polyorder),
        "orbit_gap_threshold_days": float(orbit_gap_threshold_days),
        "hysteresis_margin": float(hysteresis_margin),
        "orbit_segment_source": orbit_segment_source,
        "orbit_segment_bounds": [[int(a), int(b)] for a, b in segment_bounds],
        "raw_drift_outlier_sigma": (
            None if raw_drift_outlier_sigma is None else float(raw_drift_outlier_sigma)
        ),
        "synthesis_policy": SYNTHESIS_POLICY,
        "frame_origin_schema_version": 1,
        "frame_origin_counts": {
            "measured": n_measured,
            "synth_missing_wcs": n_missing,
            "synth_sigma_clipped": n_clipped,
        },
        "frames_missing_wcs": frames_missing_wcs,
        "frames_sigma_clipped": frames_sigma_clipped,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if sector is not None:
        meta["sector"] = int(sector)

    return ShiftSchedule(
        skycell_names=skycell_names,
        sx_float=sx_float,
        sy_float=sy_float,
        sx_int=sx_int,
        sy_int=sy_int,
        frame_valid=frame_valid,
        meta=meta,
        frame_origin=frame_origin,
    )


# ── Signature-based grouping ────────────────────────────────────────────

@dataclass
class GroupAssignment:
    """Signature-based template grouping derived from a :class:`ShiftSchedule`."""

    group_id_per_frame: np.ndarray   # (F,) i4, -1 for invalid frames
    groups: list[dict]                # [{group_id, n_frames, signature_hash}, ...]
    shifts_df: pd.DataFrame           # template_group_shifts schema (§1.2)


def assign_groups_from_schedule(
    schedule: ShiftSchedule,
    *,
    grouping_quantum_ps1_px: float,
    cache_quantum_ps1_px: float,
    keying: str,
) -> GroupAssignment:
    """
    Assign signature-based template groups from a shift schedule.

    Per-frame signature = the vector of per-skycell shifts quantized at
    ``grouping_quantum_ps1_px``: the existing hysteresis-rounded int arrays
    when the grouping quantum is exactly 1.0 PS1 px (they already embody
    hysteresis, which naive re-quantization of the float arrays would not),
    otherwise the float arrays freshly quantized at the given quantum.
    ``group_id`` is the first-appearance dense rank of each distinct
    signature among valid frames (0, 1, 2, ... in the order the signature
    is first seen); invalid frames get ``-1``.

    Each group's representative frame (its first member) supplies one
    ``template_group_shifts`` row per skycell: ``sx_int``/``sy_int`` as
    scheduled at that frame, plus ``qx``/``qy`` quantized at
    ``cache_quantum_ps1_px`` under ``keying``:

    - ``"absolute"``: quantizes the full float shift.
    - ``"phase"``: quantizes only the fractional part (``sx_float - sx_int``,
      nominally in ``[-0.5, 0.5)``); the integer part is applied as a roll
      elsewhere (downstream cache/downsample layers).

    ``signature_hash`` is the sha1 hexdigest of the group's sorted
    ``(skycell, sx_int, sy_int, cache_key)`` tuples.
    """
    if keying not in ("phase", "absolute"):
        raise ValueError(f"keying must be 'phase' or 'absolute', got {keying!r}")

    n_frames, n_cells = schedule.sx_int.shape

    if np.isclose(float(grouping_quantum_ps1_px), 1.0):
        gx = schedule.sx_int
        gy = schedule.sy_int
    else:
        gx = quantize_shift(schedule.sx_float.astype(np.float64), grouping_quantum_ps1_px)
        gy = quantize_shift(schedule.sy_float.astype(np.float64), grouping_quantum_ps1_px)

    group_id_per_frame = np.full(n_frames, -1, dtype=np.int32)
    signature_to_group: dict[tuple, int] = {}
    representative_frames: list[int] = []

    for f in range(n_frames):
        if not schedule.frame_valid[f]:
            continue
        sig = tuple(zip(gx[f].tolist(), gy[f].tolist()))
        gid = signature_to_group.get(sig)
        if gid is None:
            gid = len(representative_frames)
            signature_to_group[sig] = gid
            representative_frames.append(f)
        group_id_per_frame[f] = gid

    rows: list[dict] = []
    groups: list[dict] = []
    for gid, rep_f in enumerate(representative_frames):
        n_frames_in_group = int(np.sum(group_id_per_frame == gid))
        sig_tuples: list[tuple] = []
        for c in range(n_cells):
            sx_i = int(schedule.sx_int[rep_f, c])
            sy_i = int(schedule.sy_int[rep_f, c])
            sx_f = float(schedule.sx_float[rep_f, c])
            sy_f = float(schedule.sy_float[rep_f, c])
            if keying == "absolute":
                qx = quantize_shift(sx_f, cache_quantum_ps1_px)
                qy = quantize_shift(sy_f, cache_quantum_ps1_px)
            else:
                qx = quantize_shift(sx_f - sx_i, cache_quantum_ps1_px)
                qy = quantize_shift(sy_f - sy_i, cache_quantum_ps1_px)
            cache_key = cache_key_for(qx, qy)
            skycell = str(schedule.skycell_names[c])
            rows.append(
                {
                    "group_id": gid,
                    "skycell": skycell,
                    "sx_int": sx_i,
                    "sy_int": sy_i,
                    "qx": qx,
                    "qy": qy,
                    "cache_key": cache_key,
                }
            )
            sig_tuples.append((skycell, sx_i, sy_i, cache_key))

        sig_tuples.sort()
        signature_hash = hashlib.sha1(
            json.dumps(sig_tuples, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        groups.append(
            {
                "group_id": gid,
                "n_frames": n_frames_in_group,
                "signature_hash": signature_hash,
            }
        )

    shifts_df = _build_shifts_dataframe(rows)

    return GroupAssignment(
        group_id_per_frame=group_id_per_frame,
        groups=groups,
        shifts_df=shifts_df,
    )


def _build_shifts_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build the ``template_group_shifts`` DataFrame with its frozen §1.2 dtypes."""
    dtypes = {
        "group_id": "int32",
        "skycell": "object",
        "sx_int": "int32",
        "sy_int": "int32",
        "qx": "float32",
        "qy": "float32",
        "cache_key": "object",
    }
    if not rows:
        return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})
    df = pd.DataFrame(rows)
    return df.astype(dtypes)


def write_group_artifacts(
    assignment: GroupAssignment,
    event_dir: str | Path,
    *,
    geometry_mode: str,
    grouping_quantum_ps1_px: float,
    cache_quantum_ps1_px: float,
) -> tuple[Path, Path]:
    """
    Write ``template_group_shifts.parquet`` and ``template_groups.json`` into
    ``event_dir`` (schema_version 1; see ``docs/markdown/field_geometry.md``
    Storage / Cache keys and reuse).
    """
    event_dir = Path(event_dir)
    event_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = event_dir / "template_group_shifts.parquet"
    assignment.shifts_df.to_parquet(parquet_path, index=False)

    json_path = event_dir / "template_groups.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "geometry_mode": geometry_mode,
        "grouping_quantum_ps1_px": float(grouping_quantum_ps1_px),
        "cache_quantum_ps1_px": float(cache_quantum_ps1_px),
        "n_groups": len(assignment.groups),
        "groups": assignment.groups,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    log.info(
        "Wrote %s (%d rows) and %s (%d groups)",
        parquet_path.name,
        len(assignment.shifts_df),
        json_path.name,
        len(assignment.groups),
    )
    return parquet_path, json_path
