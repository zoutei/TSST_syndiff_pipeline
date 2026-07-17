"""MaskCatalog: static FITS + optional TNS / asteroid sidecars; mask_at API."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.masking import bits
from syndiff_pipeline.masking.asteroids import (
    ASTEROID_FFI_TIMES_BASENAME,
    convert_intervals_to_crop_local,
    resolve_cadence_from_btjd,
)
from syndiff_pipeline.masking.bits import full_mask_bool
from syndiff_pipeline.masking.tns import TRANSIENT_FIXED_BASENAME

log = logging.getLogger(__name__)

WhichKind = Literal["full", "static", "temporal"]


@dataclass
class MaskCatalog:
    """In-memory shared mask with optional temporal asteroid layer."""

    static: np.ndarray  # int16, crop-shaped
    tns_table: pd.DataFrame | None = None
    asteroid_intervals: pd.DataFrame | None = None  # crop-local y,x + cadence_lo/hi
    asteroid_times: pd.DataFrame | None = None  # cadence, btjd
    crop_bounds: dict | None = None
    _buf: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.static.shape)  # type: ignore[return-value]

    def has_temporal(self) -> bool:
        return (
            self.asteroid_intervals is not None
            and not self.asteroid_intervals.empty
        )

    def resolve_cadence(self, time) -> int | None:
        """Resolve *time* to cadence int (direct int or btjd→nearest)."""
        if time is None:
            return None
        if isinstance(time, (int, np.integer)) and not isinstance(time, bool):
            return int(time)
        btjd = float(time)
        return resolve_cadence_from_btjd(self.asteroid_times, btjd)

    def _asteroid_bool_at(self, cadence: int, out: np.ndarray | None = None) -> np.ndarray:
        ny, nx = self.static.shape
        if out is None:
            out = np.zeros((ny, nx), dtype=bool)
        else:
            out[...] = False
        iv = self.asteroid_intervals
        if iv is None or iv.empty:
            return out
        active = iv[(iv["cadence_lo"] <= cadence) & (iv["cadence_hi"] >= cadence)]
        if active.empty:
            return out
        yy = active["y"].to_numpy(int)
        xx = active["x"].to_numpy(int)
        out[yy, xx] = True
        return out

    def mask_at(
        self,
        time=None,
        *,
        which: WhichKind = "full",
        out: np.ndarray | None = None,
        as_bool: bool = False,
    ) -> np.ndarray:
        """
        Return mask for one epoch.

        Parameters
        ----------
        time : int cadence, float btjd, or None
            Required for temporal / full when asteroids present (None → static only).
        which : ``full`` | ``static`` | ``temporal``
        out : optional preallocated array (int16 or bool matching as_bool)
        as_bool : if True, return != 0 bool mask
        """
        ny, nx = self.static.shape
        want_bool = bool(as_bool)

        if which == "static":
            if want_bool:
                if out is not None:
                    out[...] = full_mask_bool(self.static)
                    return out
                return full_mask_bool(self.static)
            if out is not None:
                out[...] = self.static
                return out
            return self.static.copy()

        cadence = self.resolve_cadence(time) if time is not None else None

        if which == "temporal":
            layer = np.zeros((ny, nx), dtype=np.int16)
            if cadence is not None:
                ast = self._asteroid_bool_at(cadence)
                layer[ast] = bits.ASTEROID
            if want_bool:
                if out is not None:
                    out[...] = layer != 0
                    return out
                return layer != 0
            if out is not None:
                out[...] = layer
                return out
            return layer

        # full
        if out is not None and not want_bool:
            buf = out
            buf[...] = self.static
        else:
            if self._buf is None or self._buf.shape != self.static.shape:
                self._buf = np.empty_like(self.static)
            buf = self._buf
            buf[...] = self.static

        if cadence is not None and self.has_temporal():
            ast = self._asteroid_bool_at(cadence)
            buf[ast] = buf[ast] | np.int16(bits.ASTEROID)

        if want_bool:
            if out is not None:
                out[...] = buf != 0
                return out
            return buf != 0
        if out is not None and out is not buf:
            out[...] = buf
            return out
        # return copy so callers mutating don't poison buffer
        return buf.copy() if out is None else out

    @classmethod
    def from_arrays(
        cls,
        static: np.ndarray,
        *,
        tns_table: pd.DataFrame | None = None,
        asteroid_intervals_ffi: pd.DataFrame | None = None,
        asteroid_times: pd.DataFrame | None = None,
        crop_bounds: dict | None = None,
    ) -> "MaskCatalog":
        static = np.asarray(static, dtype=np.int16)
        crop_iv = None
        if (
            asteroid_intervals_ffi is not None
            and not asteroid_intervals_ffi.empty
            and crop_bounds is not None
        ):
            crop_iv = convert_intervals_to_crop_local(
                asteroid_intervals_ffi, crop_bounds, static.shape
            )
        return cls(
            static=static,
            tns_table=tns_table,
            asteroid_intervals=crop_iv,
            asteroid_times=asteroid_times,
            crop_bounds=crop_bounds,
        )

    @classmethod
    def from_workspace(
        cls,
        ws_root: str | Path,
        *,
        crop_bounds: dict | None = None,
        asteroid_intervals: pd.DataFrame | None = None,
        asteroid_times: pd.DataFrame | None = None,
        load_tns: bool = True,
    ) -> "MaskCatalog":
        """Load static FITS + optional sidecars from event workspace."""
        ws_root = Path(ws_root)
        sm_path = ws_root / SHARED_MASK_FITS_BASENAME
        if not sm_path.is_file():
            # legacy uncompressed
            alt = ws_root / "shared_mask.fits"
            if alt.is_file():
                sm_path = alt
            else:
                raise FileNotFoundError(f"shared_mask not found under {ws_root}")
        static = np.asarray(fits.getdata(sm_path), dtype=np.int16)

        tns_table = None
        if load_tns:
            tns_path = ws_root / TRANSIENT_FIXED_BASENAME
            if tns_path.is_file():
                tns_table = pd.read_parquet(tns_path)

        crop_iv = None
        if asteroid_intervals is not None and crop_bounds is not None:
            # Detect whether already crop-local (has y,x) or FFI (row,col)
            if "y" in asteroid_intervals.columns and "x" in asteroid_intervals.columns:
                crop_iv = asteroid_intervals
            else:
                crop_iv = convert_intervals_to_crop_local(
                    asteroid_intervals, crop_bounds, static.shape
                )

        return cls(
            static=static,
            tns_table=tns_table,
            asteroid_intervals=crop_iv,
            asteroid_times=asteroid_times,
            crop_bounds=crop_bounds,
        )
