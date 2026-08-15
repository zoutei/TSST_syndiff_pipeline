"""Build hash-bucket Parquet light-curve store from per-FFI centroids."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table
from joblib import delayed

from syndiff_pipeline.common.download import list_local_ffis
from syndiff_pipeline.common.joblib_progress import (
    parallel_map_with_optional_tqdm,
    tqdm_iter,
)
from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs
from syndiff_pipeline.common.scc_paths import scc_ffi_dir, scc_ffi_list_parquet
from syndiff_pipeline.common.wcs_grouping import gaia_science_xy_for_frame
from syndiff_pipeline.common.wcs_header_cache import load_ffi_list
from syndiff_pipeline.difference_imaging.stages.centroids import (
    CENTROIDS_INDEX_BASENAME,
    _filter_gaia_for_centroids,
    load_centroids_index,
)
from syndiff_pipeline.difference_imaging.stages.per_ffi_wcs import list_frames_for_lane
from syndiff_pipeline.difference_imaging.support.ffi_naming import ffi_frame_stem_from_path
from syndiff_pipeline.difference_imaging.wcs.io import load_gaia_catalog
from syndiff_pipeline.difference_imaging.wcs.sci2idl import crop_bounds_from_header, join_stars

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
N_BUCKETS_DEFAULT = 64
PROGRESS_BASENAME = "pivot.progress.json"
META_BASENAME = "meta.json"
STAR_INDEX_BASENAME = "star_index.parquet"
TMP_SHARDS_DIRNAME = "_tmp_shards"

SLIM_COLUMNS = [
    "source_id",
    "btjd",
    "ffi_stem",
    "flux_fit",
    "flux_err",
    "x_fit",
    "y_fit",
    "x_err",
    "y_err",
    "flags",
    "qfit",
]

def source_id_bucket(source_id: int, n_buckets: int) -> int:
    """Hash Gaia ``source_id`` into a bucket (low 6 bits are always zero in DR3)."""
    return (int(source_id) >> 16) % int(n_buckets)


_WORKER_CTX: dict[str, Any] = {}


@dataclass(frozen=True)
class LaneContext:
    lane_root: Path
    data_root: Path
    sector: int
    camera: int
    ccd: int
    diff_lane_name: str
    centroids_label: str
    hp_d_label: str
    n_buckets: int

    @property
    def scc_label(self) -> str:
        return f"s{self.sector:04d}_c{self.camera}_k{self.ccd}"

    @property
    def output_dir(self) -> Path:
        return self.lane_root / f"{self.centroids_label}_lc"


def parse_lane_path(lane: Path) -> tuple[Path, int, int, int, str]:
    """Return (data_root, sector, camera, ccd, diff_lane_name) from a lane path."""
    lane = lane.resolve()
    if not lane.is_dir():
        raise FileNotFoundError(f"Lane directory not found: {lane}")
    diff_lane_name = lane.name
    k_part = lane.parent.name
    c_part = lane.parent.parent.name
    s_part = lane.parent.parent.parent.name
    if not (s_part.startswith("s") and c_part.startswith("c") and k_part.startswith("k")):
        raise ValueError(f"Cannot parse SCC from lane path: {lane}")
    data_root = lane.parent.parent.parent.parent
    return (
        data_root,
        int(s_part[1:]),
        int(c_part[1:]),
        int(k_part[1:]),
        diff_lane_name,
    )


def _centroids_params_defaults() -> SimpleNamespace:
    return SimpleNamespace(mag_max_rp=12.95, mag_min_rp=7.5)


def _load_progress(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return set(raw.get("completed_stems", []))


def _save_progress(path: Path, completed: set[str], *, n_frames: int) -> None:
    payload = {
        "completed_stems": sorted(completed),
        "n_completed": len(completed),
        "n_frames": n_frames,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _slim_frame_table(merged: pd.DataFrame, *, stem: str, btjd: float) -> pd.DataFrame:
    if merged.empty or "source_id" not in merged.columns:
        return pd.DataFrame(columns=SLIM_COLUMNS)
    out = pd.DataFrame(
        {
            "source_id": pd.to_numeric(merged["source_id"], errors="coerce").astype("Int64"),
            "btjd": float(btjd),
            "ffi_stem": stem,
            "flux_fit": pd.to_numeric(merged.get("flux_fit"), errors="coerce"),
            "flux_err": pd.to_numeric(merged.get("flux_err"), errors="coerce"),
            "x_fit": pd.to_numeric(merged.get("x_fit"), errors="coerce"),
            "y_fit": pd.to_numeric(merged.get("y_fit"), errors="coerce"),
            "x_err": pd.to_numeric(merged.get("x_err"), errors="coerce"),
            "y_err": pd.to_numeric(merged.get("y_err"), errors="coerce"),
            "flags": pd.to_numeric(merged.get("flags"), errors="coerce"),
            "qfit": pd.to_numeric(merged.get("qfit"), errors="coerce"),
        }
    )
    out = out.dropna(subset=["source_id"])
    out["source_id"] = out["source_id"].astype(np.int64)
    return out[SLIM_COLUMNS]


def _ffi_path_by_stem(ctx: LaneContext) -> dict[str, str]:
    ffi_dir = scc_ffi_dir(ctx.data_root, ctx.sector, ctx.camera, ctx.ccd)
    out: dict[str, str] = {}
    for path in list_local_ffis(str(ffi_dir), ctx.sector, ctx.camera, ctx.ccd):
        try:
            stem = ffi_frame_stem_from_path(path)
        except ValueError:
            continue
        out[stem] = path
    return out


def _init_worker(ctx: LaneContext, gaia_base: pd.DataFrame, ffi_list_df: pd.DataFrame) -> None:
    frames = list_frames_for_lane(
        ctx.lane_root,
        centroids_label=ctx.centroids_label,
        hp_d_label=ctx.hp_d_label,
    )
    if not frames:
        raise RuntimeError(f"No centroid frames under {ctx.lane_root / ctx.centroids_label}")
    science_bounds = crop_bounds_from_header(fits.getheader(frames[0].hp_d_path, ext=1))
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "ctx": ctx,
            "gaia_base": gaia_base,
            "ffi_list_df": ffi_list_df,
            "science_bounds": science_bounds,
            "ffi_path_by_stem": _ffi_path_by_stem(ctx),
            "frames_by_stem": {f.stem: f for f in frames},
            "params": _centroids_params_defaults(),
        }
    )


def _process_one_frame_task(frame_idx: int, stem: str) -> tuple[str, int, bool, str]:
    del frame_idx
    ctx: LaneContext = _WORKER_CTX["ctx"]
    fr = _WORKER_CTX["frames_by_stem"].get(stem)
    if fr is None:
        return stem, 0, False, f"{stem}: missing frame record"

    shard_path = ctx.output_dir / TMP_SHARDS_DIRNAME / f"{stem}.parquet"
    if shard_path.is_file():
        try:
            n = len(pq.read_table(shard_path))
            return stem, n, True, f"{stem}: skipped existing shard ({n} rows)"
        except Exception:
            pass

    if not fr.phot_path.is_file():
        return stem, 0, False, f"{stem}: missing photresults {fr.phot_path}"

    ffi_path = _WORKER_CTX["ffi_path_by_stem"].get(stem)
    if ffi_path is None:
        return stem, 0, False, f"{stem}: missing FFI path"

    try:
        phot = Table.read(fr.phot_path, format="ascii.ecsv")
        gaia_frame = gaia_science_xy_for_frame(
            _WORKER_CTX["gaia_base"],
            ffi_path,
            _WORKER_CTX["ffi_list_df"],
            _WORKER_CTX["science_bounds"],
        )
        gaia_frame = _filter_gaia_for_centroids(gaia_frame, _WORKER_CTX["params"])
        merged = join_stars(phot, gaia_frame)
        slim = _slim_frame_table(merged, stem=stem, btjd=fr.btjd)
    except Exception as exc:
        return stem, 0, False, f"{stem}: {exc}"

    if slim.empty:
        return stem, 0, False, f"{stem}: no joined rows"

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(shard_path, compression="zstd", index=False)
    return stem, len(slim), True, f"{stem}: {len(slim)} rows"


def _reduce_shards(ctx: LaneContext) -> None:
    shard_dir = ctx.output_dir / TMP_SHARDS_DIRNAME
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"No shard parquet files under {shard_dir}")

    log.info("Reducing %d frame shards into %d buckets", len(shards), ctx.n_buckets)
    bucket_parts: list[list[pd.DataFrame]] = [[] for _ in range(ctx.n_buckets)]
    for shard_path in tqdm_iter(shards, desc="reduce_shards"):
        df = pd.read_parquet(shard_path)
        if df.empty:
            continue
        buckets = np.fromiter(
            (source_id_bucket(int(sid), ctx.n_buckets) for sid in df["source_id"]),
            dtype=np.int64,
            count=len(df),
        )
        for bucket in range(ctx.n_buckets):
            mask = buckets == bucket
            if not np.any(mask):
                continue
            bucket_parts[bucket].append(df.loc[mask])

    for bucket in range(ctx.n_buckets):
        parts = bucket_parts[bucket]
        bucket_parts[bucket] = []
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(["source_id", "btjd"], kind="mergesort")
        out_dir = ctx.output_dir / f"bucket={bucket:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_dir / "data.parquet", compression="zstd", index=False)
        log.info("  bucket=%02d: %d rows", bucket, len(combined))


def _build_star_index(ctx: LaneContext) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bucket in range(ctx.n_buckets):
        path = ctx.output_dir / f"bucket={bucket:02d}" / "data.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path, columns=["source_id", "btjd"]))

    if not frames:
        return pd.DataFrame(
            columns=[
                "source_id",
                "ra",
                "dec",
                "bucket",
                "n_epochs",
                "btjd_min",
                "btjd_max",
            ]
        )

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["source_id"] = all_rows["source_id"].astype(np.int64)
    gaia = load_gaia_catalog(ctx.lane_root)
    gaia_ids = gaia[["source_id", "ra", "dec"]].drop_duplicates(subset=["source_id"])
    gaia_ids["source_id"] = pd.to_numeric(gaia_ids["source_id"], errors="coerce").astype(
        np.int64
    )
    grouped = (
        all_rows.groupby("source_id", as_index=False)
        .agg(n_epochs=("btjd", "size"), btjd_min=("btjd", "min"), btjd_max=("btjd", "max"))
        .merge(gaia_ids, on="source_id", how="left")
    )
    grouped["source_id"] = grouped["source_id"].astype(np.int64)
    grouped["bucket"] = grouped["source_id"].map(
        lambda sid: source_id_bucket(int(sid), ctx.n_buckets)
    )
    return grouped[
        ["source_id", "ra", "dec", "bucket", "n_epochs", "btjd_min", "btjd_max"]
    ].sort_values("source_id")


def _write_meta(ctx: LaneContext) -> None:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "centroids_input": ctx.centroids_label,
        "diffs_input": ctx.hp_d_label,
        "lane": ctx.diff_lane_name,
        "scc": ctx.scc_label,
        "n_buckets": ctx.n_buckets,
        "bucket_fn": f"(source_id >> 16) % {ctx.n_buckets}",
        "columns": SLIM_COLUMNS,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with (ctx.output_dir / META_BASENAME).open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)


def load_centroid_lc(
    lane_root: Path | str,
    source_id: int,
    *,
    centroids_label: str = "centroids_r1",
    n_buckets: int = N_BUCKETS_DEFAULT,
) -> pd.DataFrame:
    """Load one star light curve from a built ``{centroids_label}_lc`` store."""
    lane_root = Path(lane_root)
    bucket = source_id_bucket(int(source_id), n_buckets)
    path = (
        lane_root
        / f"{centroids_label}_lc"
        / f"bucket={bucket:02d}"
        / "data.parquet"
    )
    return (
        pd.read_parquet(path, filters=[("source_id", "=", int(source_id))])
        .sort_values("btjd")
        .reset_index(drop=True)
    )


def build_centroids_lc_buckets(
    ctx: LaneContext,
    *,
    n_jobs: int = 1,
    resume: bool = True,
    keep_shards: bool = False,
) -> None:
    centroids_dir = ctx.lane_root / ctx.centroids_label
    index = load_centroids_index(str(centroids_dir))
    if not index:
        raise RuntimeError(f"Missing or empty {centroids_dir / CENTROIDS_INDEX_BASENAME}")

    ffi_list_path = scc_ffi_list_parquet(ctx.data_root, ctx.sector, ctx.camera, ctx.ccd)
    if not ffi_list_path.is_file():
        raise FileNotFoundError(f"Missing ffi_list.parquet: {ffi_list_path}")
    ffi_list_df = load_ffi_list(str(ffi_list_path))
    gaia_base = load_gaia_catalog(ctx.lane_root)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = ctx.output_dir / PROGRESS_BASENAME
    completed = _load_progress(progress_path) if resume else set()

    stems = sorted(index.keys())
    if resume and completed:
        stems = [s for s in stems if s not in completed]

    n_workers = resolve_effective_n_jobs(n_jobs)
    worker_initargs = (ctx, gaia_base, ffi_list_df)

    log.info(
        "Building %s from %s (%d frames, %d workers, resume=%s)",
        ctx.output_dir.name,
        ctx.centroids_label,
        len(stems),
        n_workers,
        resume,
    )

    def _record(stem: str, ok: bool) -> None:
        if ok:
            completed.add(stem)
            _save_progress(progress_path, completed, n_frames=len(index))

    if n_workers <= 1 or len(stems) <= 1:
        _init_worker(*worker_initargs)
        for i, stem in enumerate(tqdm_iter(stems, desc="pivot_frames")):
            res_stem, _n, ok, msg = _process_one_frame_task(i, stem)
            log.info("  %s", msg)
            _record(res_stem, ok)
    else:
        delayed_calls = [delayed(_process_one_frame_task)(i, stem) for i, stem in enumerate(stems)]

        def _on_result(res: tuple[str, int, bool, str]) -> None:
            log.info("  %s", res[3])
            _record(res[0], res[2])

        parallel_map_with_optional_tqdm(
            delayed_calls,
            n_tasks=len(stems),
            desc="pivot_frames",
            n_jobs_eff=n_workers,
            initializer=_init_worker,
            initargs=worker_initargs,
            on_result=_on_result,
        )

    _reduce_shards(ctx)
    star_index = _build_star_index(ctx)
    star_index.to_parquet(ctx.output_dir / STAR_INDEX_BASENAME, compression="zstd", index=False)
    _write_meta(ctx)

    if not keep_shards:
        shutil.rmtree(ctx.output_dir / TMP_SHARDS_DIRNAME, ignore_errors=True)

    log.info("Done: %d stars -> %s", len(star_index), ctx.output_dir)
