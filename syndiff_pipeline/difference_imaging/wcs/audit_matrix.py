"""Compact star×frame audit matrices for per-FFI WCS fits."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table

from syndiff_pipeline.difference_imaging.wcs.sci2idl import join_stars

AUDIT_NPZ = "stars_fit_audit.npz"

STATUS_MISSING = np.uint8(0)
STATUS_CLIPPED = np.uint8(1)
STATUS_USED = np.uint8(2)


@dataclass(frozen=True)
class StarIndex:
    source_id: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    row_lookup: dict[int, int]


def _xy_to_source_id_map(gaia_df: pd.DataFrame) -> dict[tuple[float, float], int]:
    if not {"x", "y", "source_id"}.issubset(gaia_df.columns):
        return {}
    out: dict[tuple[float, float], int] = {}
    for x, y, sid in zip(
        gaia_df["x"].to_numpy(dtype=float),
        gaia_df["y"].to_numpy(dtype=float),
        gaia_df["source_id"].to_numpy(dtype=np.int64),
    ):
        if np.isfinite(x) and np.isfinite(y):
            out[(float(x), float(y))] = int(sid)
    return out


def collect_source_ids_from_phot_paths(
    phot_paths: list[Path],
    gaia_df: pd.DataFrame,
) -> set[int]:
    """Union of Gaia ``source_id`` values reachable from photresults init positions."""
    xy_to_sid = _xy_to_source_id_map(gaia_df)
    if not xy_to_sid:
        return set()

    source_ids: set[int] = set()
    for path in phot_paths:
        if not path.is_file():
            continue
        phot = Table.read(path, format="ascii.ecsv")
        for x_init, y_init in zip(phot["x_init"], phot["y_init"]):
            sid = xy_to_sid.get((float(x_init), float(y_init)))
            if sid is not None:
                source_ids.add(sid)
    return source_ids


def build_star_index(
    gaia_df: pd.DataFrame,
    *,
    phot_paths: list[Path] | None = None,
) -> StarIndex:
    """Build a stable row index keyed by Gaia ``source_id``."""
    if "source_id" not in gaia_df.columns:
        raise ValueError("gaia_df requires source_id for audit matrix indexing")

    g = gaia_df.copy()
    for col in ("ra", "dec"):
        if col not in g.columns:
            g[col] = np.nan

    if phot_paths:
        source_ids = collect_source_ids_from_phot_paths(phot_paths, gaia_df)
        if not source_ids:
            raise ValueError("No source_id values found from photresults paths")
        g = g[g["source_id"].astype(np.int64).isin(source_ids)]

    g = g.dropna(subset=["source_id", "ra", "dec"])
    g = g.drop_duplicates(subset=["source_id"], keep="first")
    g = g.sort_values("source_id", kind="mergesort").reset_index(drop=True)

    source_id = g["source_id"].to_numpy(dtype=np.int64)
    ra = g["ra"].to_numpy(dtype=np.float64)
    dec = g["dec"].to_numpy(dtype=np.float64)
    row_lookup = {int(sid): i for i, sid in enumerate(source_id)}
    return StarIndex(source_id=source_id, ra=ra, dec=dec, row_lookup=row_lookup)


class StarAuditMatrixWriter:
    """Stream per-frame audit columns into memmapped arrays, then NPZ."""

    def __init__(
        self,
        star_index: StarIndex,
        stems: list[str],
        btjd: list[float],
        out_dir: str | Path,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.star_index = star_index
        self.n_stars = len(star_index.source_id)
        self.n_frames = len(stems)
        self.stems = np.asarray(stems, dtype=str)
        self.btjd = np.asarray(btjd, dtype=np.float64)

        self._tmp_dir = self.out_dir / "_audit_tmp"
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        shape = (self.n_stars, self.n_frames)
        self._du = np.memmap(
            self._tmp_dir / "du.npy", dtype=np.float32, mode="w+", shape=shape
        )
        self._dv = np.memmap(
            self._tmp_dir / "dv.npy", dtype=np.float32, mode="w+", shape=shape
        )
        self._status = np.memmap(
            self._tmp_dir / "status.npy", dtype=np.uint8, mode="w+", shape=shape
        )
        self._du[:] = np.nan
        self._dv[:] = np.nan
        self._status[:] = STATUS_MISSING

    def write_frame(
        self,
        frame_idx: int,
        source_ids: np.ndarray,
        du: np.ndarray,
        dv: np.ndarray,
        keep_mask: np.ndarray,
    ) -> None:
        """Scatter QC-star residuals into column ``frame_idx``."""
        col = int(frame_idx)
        if col < 0 or col >= self.n_frames:
            raise IndexError(f"frame_idx {frame_idx} out of range for {self.n_frames} frames")

        lookup = self.star_index.row_lookup
        rows: list[int] = []
        du_vals: list[float] = []
        dv_vals: list[float] = []
        status_vals: list[np.uint8] = []

        for sid, du_i, dv_i, keep in zip(source_ids, du, dv, keep_mask, strict=True):
            row = lookup.get(int(sid))
            if row is None:
                continue
            rows.append(row)
            du_vals.append(float(du_i))
            dv_vals.append(float(dv_i))
            status_vals.append(STATUS_USED if bool(keep) else STATUS_CLIPPED)

        if not rows:
            return

        row_idx = np.asarray(rows, dtype=np.int64)
        self._du[row_idx, col] = np.asarray(du_vals, dtype=np.float32)
        self._dv[row_idx, col] = np.asarray(dv_vals, dtype=np.float32)
        self._status[row_idx, col] = np.asarray(status_vals, dtype=np.uint8)

    def write_audit_frame(self, audit_frame: dict[str, Any] | None) -> None:
        if not audit_frame:
            return
        self.write_frame(
            int(audit_frame["frame_idx"]),
            audit_frame["source_id"],
            audit_frame["du"],
            audit_frame["dv"],
            audit_frame["keep_mask"],
        )

    def finalize(self) -> Path:
        """Flush memmaps and write compressed NPZ; remove temp files."""
        out_path = self.out_dir / AUDIT_NPZ
        np.savez_compressed(
            out_path,
            source_id=self.star_index.source_id,
            ra=self.star_index.ra,
            dec=self.star_index.dec,
            stems=self.stems,
            btjd=self.btjd,
            du=np.asarray(self._du),
            dv=np.asarray(self._dv),
            status=np.asarray(self._status),
        )
        del self._du
        del self._dv
        del self._status
        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        return out_path


def load_stars_fit_audit(path: str | Path) -> dict[str, np.ndarray]:
    """Load ``stars_fit_audit.npz`` arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def to_long_dataframe(audit: dict[str, np.ndarray]) -> pd.DataFrame:
    """Expand a star×frame audit matrix into a long stacked table."""
    du = audit["du"]
    dv = audit["dv"]
    status = audit["status"]
    stems = audit["stems"]
    btjd = audit["btjd"]
    source_id = audit["source_id"]

    rows, cols = np.nonzero(status)
    if rows.size == 0:
        return pd.DataFrame(
            columns=["source_id", "stem", "btjd", "du", "dv", "status", "hypot_resid"]
        )

    du_vals = du[rows, cols]
    dv_vals = dv[rows, cols]
    return pd.DataFrame(
        {
            "source_id": source_id[rows],
            "stem": stems[cols],
            "btjd": btjd[cols],
            "du": du_vals,
            "dv": dv_vals,
            "status": status[rows, cols],
            "hypot_resid": np.hypot(du_vals, dv_vals),
        }
    )
