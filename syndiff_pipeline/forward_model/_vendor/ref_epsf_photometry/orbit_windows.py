# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Orbit segmentation and reference-window stem selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BTJD_TO_MJD_OFFSET = 2457000.0 - 2400000.5
_JD_SCALE_BTJD_THRESHOLD = 1e6


def _frame_epoch_to_mjd(epoch: str | float | None) -> float | None:
    """Map DATE-OBS ISO string or BTJD float to MJD for orbit CSV lookup."""
    from astropy.time import Time

    if epoch is None:
        return None
    if isinstance(epoch, (int, float, np.floating)):
        v = float(epoch)
        if not np.isfinite(v):
            return None
        if v > _JD_SCALE_BTJD_THRESHOLD:
            return v
        return v + BTJD_TO_MJD_OFFSET
    text = str(epoch).strip()
    if not text:
        return None
    try:
        return float(Time(text.replace(" ", "T"), format="isot", scale="utc").mjd)
    except Exception:
        return None


def _orbit_ids_from_csv(
    sector: int,
    frame_epochs: list[str | float],
    csv_path: Path,
) -> np.ndarray:
    from astropy.time import Time

    n_frames = len(frame_epochs)
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
    for i, epoch in enumerate(frame_epochs):
        mjd = _frame_epoch_to_mjd(epoch)
        if mjd is None:
            continue
        for oi, (s, e) in enumerate(zip(starts, ends)):
            if s <= mjd <= e:
                orbit_ids[i] = oi
                break
    return orbit_ids


def _central_slice(n: int, width: int) -> slice:
    w = int(width)
    if n < w:
        raise ValueError(f"segment length {n} < window width {w}")
    start = (n - w) // 2
    return slice(start, start + w)


def select_orbit_windows(
    ordered_stems: list[str],
    frame_epochs: list[str | float],
    sector: int,
    orbit_csv: Path,
    *,
    n_mid: int = 20,
    n_edge: int = 10,
) -> dict[str, dict[str, list[str]]]:
    """
    Return ``{"orbit0": {"mid": [...], "begin": [...], "end": [...]}, ...}``.

    Uses the two most-populated non-negative orbit ids in the sector.
    """
    if len(ordered_stems) != len(frame_epochs):
        raise ValueError("ordered_stems and frame_epochs length mismatch")

    orbit_ids = _orbit_ids_from_csv(sector, frame_epochs, orbit_csv)
    valid_ids = [int(oid) for oid in np.unique(orbit_ids) if int(oid) >= 0]
    if not valid_ids:
        raise RuntimeError("no frames assigned to TESS orbits")

    counts = {oid: int(np.sum(orbit_ids == oid)) for oid in valid_ids}
    top_two = sorted(counts, key=lambda k: (-counts[k], k))[:2]
    top_two = sorted(top_two)

    out: dict[str, dict[str, list[str]]] = {}
    for oid in top_two:
        idx = np.where(orbit_ids == oid)[0]
        stems = [ordered_stems[int(i)] for i in idx]
        n = len(stems)
        if n < n_mid:
            raise ValueError(f"orbit {oid} has only {n} frames (< {n_mid})")
        if n < n_edge:
            raise ValueError(f"orbit {oid} has only {n} frames (< {n_edge})")
        mid_sl = _central_slice(n, n_mid)
        out[f"orbit{oid}"] = {
            "mid": stems[mid_sl],
            "begin": stems[: int(n_edge)],
            "end": stems[-int(n_edge) :],
        }
    return out


def select_rolling_orbit_windows(
    ordered_stems: list[str],
    frame_epochs: list[str | float],
    sector: int,
    orbit_csv: Path,
    *,
    orbit_idx: int = 0,
    window: int = 20,
    stride: int = 10,
) -> list[dict[str, Any]]:
    """
    Rolling FFI windows within one orbit.

    Returns a list of dicts ordered by start index::

        {
          "roll_idx": int,
          "orbit_idx": int,
          "start": int,          # inclusive index into orbit stem list
          "stop": int,           # exclusive
          "stems": list[str],
          "window": str,         # label e.g. "roll000_i0000"
        }

    With ``window=20`` and ``stride=10``, consecutive rolls overlap by 10 FFIs.
    """
    if len(ordered_stems) != len(frame_epochs):
        raise ValueError("ordered_stems and frame_epochs length mismatch")
    w = int(window)
    s = int(stride)
    if w < 1 or s < 1:
        raise ValueError("window and stride must be >= 1")

    orbit_ids = _orbit_ids_from_csv(sector, frame_epochs, orbit_csv)
    idx = np.where(orbit_ids == int(orbit_idx))[0]
    stems = [ordered_stems[int(i)] for i in idx]
    n = len(stems)
    if n < w:
        raise ValueError(
            f"orbit {orbit_idx} has only {n} frames (< window={w})"
        )

    out: list[dict[str, Any]] = []
    roll_idx = 0
    for start in range(0, n - w + 1, s):
        stop = start + w
        chunk = stems[start:stop]
        label = f"roll{roll_idx:03d}_i{start:04d}"
        out.append(
            {
                "roll_idx": roll_idx,
                "orbit_idx": int(orbit_idx),
                "start": int(start),
                "stop": int(stop),
                "stems": chunk,
                "window": label,
            }
        )
        roll_idx += 1
    return out


def windows_to_manifest_entries(
    windows: dict[str, dict[str, list[str]]],
    btjd_by_stem: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit_key, wmap in windows.items():
        orbit_idx = int(orbit_key.replace("orbit", ""))
        for window, stems in wmap.items():
            btjds = [float(btjd_by_stem[s]) for s in stems if s in btjd_by_stem]
            rows.append(
                {
                    "orbit_idx": orbit_idx,
                    "window": window,
                    "n_ffis": len(stems),
                    "reference_stems": stems,
                    "btjd_min": float(min(btjds)) if btjds else None,
                    "btjd_max": float(max(btjds)) if btjds else None,
                }
            )
    return rows
