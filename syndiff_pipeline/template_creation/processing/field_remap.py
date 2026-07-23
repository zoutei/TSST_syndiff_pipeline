"""Field-mode remap: SCC shift schedule, group artifacts, and Exact caches.

Owns L2–L4 under ``{data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/``.
Downsample (L5) reads these artifacts and bins sparse contribs under
``templates/oversampling_{N}/``.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.fits_variants import (
    FITS_STORAGE_SUFFIXES,
    is_fits_storage_filename,
    prefer_fits_path,
    try_resolve_fits_variant,
)
from syndiff_pipeline.common.scc_paths import scc_remap_dir
from syndiff_pipeline.common.wcs_grouping import _frames_csv_path
from syndiff_pipeline.difference_imaging.support.ffi_naming import PIPELINE_FITS_EXT
from syndiff_pipeline.template_creation.processing.field_templates import (
    contrib_basename,
    field_store_lock,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    assign_groups_from_schedule,
    build_pair_epochs,
    build_shift_epochs,
    write_group_artifacts,
)

log = logging.getLogger(__name__)

REMAP_MANIFEST_NAME = "remap_manifest.json"
EXACT_CACHE_L4A_DIRNAME = "exact_cache_l4a"
EXACT_CACHE_L4B_DIRNAME = "exact_cache_l4b"
# Legacy monolithic tree (L4b-lite pollution); never read for hybrid L4a.
EXACT_CACHE_LEGACY_DIRNAME = "exact_cache"
EXACT_CACHE_LEGACY_POLLUTED_DIRNAME = "exact_cache_legacy_polluted"
REMAP_SCHEMA_VERSION = 3
SHIFT_EPOCHS_PARQUET = "shift_epochs.parquet"
PAIR_EPOCHS_PARQUET = "pair_epochs.parquet"
EPOCH_GROUP_MEMBERS_PARQUET = "epoch_group_members.parquet"
GID_EPOCH_INDEX_NPZ = "gid_epoch_index.npz"
GROUP_ID_PER_FRAME_NPY = "group_id_per_frame.npy"

def remap_root(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
    store_name: str | None = None,
) -> Path:
    """Return the SCC remap store directory (does not create it)."""
    return scc_remap_dir(
        data_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
        store_name=store_name,
    )


def resolve_remap_read_root(
    remap_store: str | Path,
    templates_store: str | Path,
) -> tuple[Path, bool]:
    """Resolve where to read remap artifacts (dual-read migration).

    Returns ``(read_root, legacy_colocated)``. When ``remap_manifest.json`` is
    missing, falls back to schedule/group artifacts colocated under the legacy
    ``templates/`` store with a warning (not polluted ``exact_cache/``).
    """
    remap_path = Path(remap_store)
    templates_path = Path(templates_store)
    manifest = remap_path / REMAP_MANIFEST_NAME
    if manifest.is_file():
        return remap_path, False
    legacy_schedule = templates_path / "shift_schedule.npz"
    if legacy_schedule.is_file():
        log.warning(
            "remap_manifest missing at %s; reading schedule/groups from legacy "
            "colocated templates store %s (exact_cache_l4a not used from legacy)",
            remap_path,
            templates_path,
        )
        return templates_path, True
    raise FileNotFoundError(
        f"remap artifacts missing: no {REMAP_MANIFEST_NAME} under {remap_path} "
        f"and no shift_schedule.npz under {templates_path}; run field_remap first"
    )


def load_remap_shifts_df(read_root: str | Path) -> pd.DataFrame:
    """Load ``template_group_shifts.parquet`` from a remap read root."""
    path = Path(read_root) / "template_group_shifts.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"template_group_shifts missing: {path}")
    return pd.read_parquet(path)


def exact_cache_l4a_dir_for_read_root(read_root: str | Path) -> Path:
    """Return the pure L4a Exact cache directory (no legacy ``exact_cache/`` fallback)."""
    return Path(read_root) / EXACT_CACHE_L4A_DIRNAME


def exact_cache_l4b_dir_for_read_root(read_root: str | Path) -> Path:
    """Return the L4b F2 rim Exact cache directory."""
    return Path(read_root) / EXACT_CACHE_L4B_DIRNAME


def exact_cache_dir_for_read_root(read_root: str | Path) -> Path:
    """Alias for :func:`exact_cache_l4a_dir_for_read_root`."""
    return exact_cache_l4a_dir_for_read_root(read_root)


def _load_or_copy_shift_schedule(
    event_dir: Path,
    store_root: Path,
) -> ShiftSchedule | None:
    """Prefer event ``shift_schedule.npz``; fall back to store copy. None if missing."""
    event_npz = event_dir / "shift_schedule.npz"
    store_npz = store_root / "shift_schedule.npz"
    src = event_npz if event_npz.is_file() else store_npz
    if not src.is_file():
        return None
    if src != store_npz:
        with field_store_lock(store_root):
            shutil.copy2(src, store_npz)
            sidecar = src.with_suffix(".json")
            if sidecar.is_file():
                shutil.copy2(sidecar, store_npz.with_suffix(".json"))
    return ShiftSchedule.load(store_npz)


def _mapping_scc_dir(
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Resolve the SCC mapping directory for the requested oversampling."""
    root = Path(mapping_root)
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    flat_stem = f"tess_s{int(sector):04d}_{camera}_{ccd}_master_pixels2skycells{suffix}"
    if any((root / f"{flat_stem}{sfx}").is_file() for sfx in FITS_STORAGE_SUFFIXES) or any(
        is_fits_storage_filename(p.name)
        for p in root.glob("tess_s*_master_pixels2skycells*")
        if p.is_file()
    ):
        return root
    scc_tail = Path(f"sector_{int(sector):04d}") / f"camera_{int(camera)}" / f"ccd_{int(ccd)}"
    if root.name == f"ccd_{int(ccd)}" and any(
        (root / f"tess_s{int(sector):04d}_{camera}_{ccd}_master_pixels2skycells{sfx}").is_file()
        for sfx in FITS_STORAGE_SUFFIXES
    ):
        return root
    if root.name == f"ccd_{int(ccd)}" and any(
        is_fits_storage_filename(p.name)
        for p in root.glob("tess_s*_master_pixels2skycells*")
        if p.is_file()
    ):
        return root
    if int(oversampling_factor) > 1 and root.name != f"oversampling_{int(oversampling_factor)}":
        root = root / f"oversampling_{int(oversampling_factor)}"
    nested = root / scc_tail
    if nested.is_dir():
        return nested
    return root


def _skycell_csv_path(
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    scc = _mapping_scc_dir(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    return scc / f"tess_s{int(sector):04d}_{camera}_{ccd}_master_skycells_list{suffix}.csv"


def _build_shift_schedule_for_scc(
    *,
    store_root: Path,
    data_root: Path,
    mapping_root: Path,
    ffi_dir: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    ref_ffi_path: str | Path,
    oversampling_factor: int = 1,
    raw_drift_outlier_sigma: float | None = 5.0,
    drift_source: str = "per_skycell",
    target_drift: np.ndarray | None = None,
) -> ShiftSchedule:
    """Build ``shift_schedule.npz`` from all SCC FFIs (no event handoff)."""
    from syndiff_pipeline.common.download import list_local_ffis, manifest_basename_from_local
    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap
    from syndiff_pipeline.common.wcs_header_cache import (
        ensure_scc_ffi_list,
        ffi_list_is_complete,
        load_ffi_list,
        wcs_from_cached_row,
    )
    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        RELEVANT_WCS_KEYS,
        load_tess_wcs,
    )
    from syndiff_pipeline.template_creation.processing.shift_schedule import (
        build_skycell_shift_schedule,
    )

    ref_path = Path(ref_ffi_path).expanduser()
    resolved_ref = try_resolve_fits_variant(ref_path)
    if resolved_ref is None:
        raise FileNotFoundError(f"reference_ffi_path missing: {ref_path}")
    ref_path = resolved_ref
    ref_wcs, _ = load_tess_wcs(ref_path)

    csv_path = _skycell_csv_path(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"skycell catalog missing: {csv_path}")
    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    skycell_df = pd.read_csv(csv_path, usecols=usecols)

    paths = sorted(list_local_ffis(str(ffi_dir), sector, camera, ccd))
    if not paths:
        raise FileNotFoundError(f"No FFIs under {ffi_dir!r}")

    ffi_list_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
    ffi_list_df = load_ffi_list(ffi_list_path)
    if not ffi_list_is_complete(paths, ffi_list_df):
        ffi_list_df = ensure_scc_ffi_list(
            data_root,
            sector,
            camera,
            ccd,
            paths,
            open_fits=open_fits_memmap,
        )
    frame_wcs: list[tuple[str, Any]] = []
    btjd_list: list[float] = []
    frame_times: list[str | None] = []
    for p in paths:
        logical = manifest_basename_from_local(p)
        row = ffi_list_df.loc[logical] if logical in ffi_list_df.index else None
        if row is not None and bool(row.get("wcs_ok", False)):
            try:
                frame_wcs.append((logical, wcs_from_cached_row(row)))
            except Exception as exc:
                log.warning("WCS reconstruct failed for %s: %s", logical, exc)
                frame_wcs.append((logical, None))
        else:
            if row is not None and not bool(row.get("wcs_ok", False)):
                log.info("Shift schedule: skipping WCS for %s (wcs_ok=False)", logical)
            frame_wcs.append((logical, None))
        if row is not None and pd.notna(row.get("date_obs")):
            date_obs = str(row["date_obs"])
            frame_times.append(date_obs)
            try:
                from astropy.time import Time

                t = Time(date_obs, format="isot", scale="utc")
                # BTJD ≈ JD − 2457000.0 (TESS convention).
                btjd_list.append(float(t.jd) - 2457000.0)
            except Exception:
                btjd_list.append(float("nan"))
        else:
            frame_times.append(None)
            btjd_list.append(float("nan"))

    btjd = np.asarray(btjd_list, dtype=np.float64)
    schedule = build_skycell_shift_schedule(
        frame_wcs,
        skycell_df,
        ref_wcs,
        btjd=btjd,
        frame_times=frame_times,
        sector=int(sector),
        raw_drift_outlier_sigma=raw_drift_outlier_sigma,
        drift_source=drift_source,
        target_drift=target_drift,
    )
    schedule.meta = dict(schedule.meta or {})
    schedule.meta["source"] = "built_from_scc_ffis"
    schedule.meta["reference_ffi"] = str(ref_path.resolve())
    schedule.meta["drift_source"] = str(drift_source)
    schedule.meta["frame_filenames"] = [manifest_basename_from_local(p) for p in paths]
    with field_store_lock(store_root):
        schedule.save(store_root / "shift_schedule.npz")
        if str(drift_source or "per_skycell").strip().lower() != "point":
            _try_write_skycell_shift_debug_plots(
                schedule,
                store_root=store_root,
                mapping_root=mapping_root,
                btjd=btjd,
                ref_wcs=ref_wcs,
                skycell_df=skycell_df,
                sector=sector,
                camera=camera,
                ccd=ccd,
                oversampling_factor=oversampling_factor,
                data_root=data_root,
            )
    log.info(
        "Built SCC shift_schedule.npz (%d frames × %d skycells)",
        schedule.sx_int.shape[0],
        schedule.sx_int.shape[1],
    )
    return schedule


def _build_shift_schedule_for_event(
    *,
    event_dir: Path,
    store_root: Path,
    data_root: Path,
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int = 1,
    raw_drift_outlier_sigma: float | None = 5.0,
) -> ShiftSchedule:
    """Build ``shift_schedule.npz`` from WCS cache + skycell catalog when missing."""
    from syndiff_pipeline.common.download import manifest_basename_from_local
    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap
    from syndiff_pipeline.common.wcs_header_cache import (
        ensure_scc_ffi_list,
        ffi_list_is_complete,
        load_ffi_list,
        wcs_from_cached_row,
    )
    from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        RELEVANT_WCS_KEYS,
        load_tess_wcs,
    )
    from syndiff_pipeline.template_creation.processing.shift_schedule import (
        build_skycell_shift_schedule,
    )

    frames_path = Path(_frames_csv_path(event_dir))
    if not frames_path.is_file():
        raise FileNotFoundError(f"Cannot build shift schedule without {frames_path}")
    frames_df = pd.read_csv(frames_path)
    if "path" not in frames_df.columns:
        raise KeyError(f"{frames_path} missing 'path' column")

    from syndiff_pipeline.common.wcs_grouping import _event_job_path

    job_path = Path(_event_job_path(event_dir))
    if not job_path.is_file():
        raise FileNotFoundError(f"Cannot build shift schedule without {job_path}")
    job = json.loads(job_path.read_text())
    ref_path = Path(str(job.get("reference_ffi_path") or "")).expanduser()
    resolved_ref = try_resolve_fits_variant(ref_path)
    if resolved_ref is None:
        raise FileNotFoundError(
            f"reference_ffi_path missing or not a file: {ref_path} (from {job_path})"
        )
    ref_path = resolved_ref
    ref_wcs, _ = load_tess_wcs(ref_path)

    csv_path = _skycell_csv_path(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"skycell catalog missing: {csv_path}")
    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    skycell_df = pd.read_csv(csv_path, usecols=usecols)

    paths = [Path(str(p)) for p in frames_df["path"].tolist()]
    ffi_list_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
    ffi_list_df = load_ffi_list(ffi_list_path)
    if not ffi_list_is_complete(paths, ffi_list_df):
        ffi_list_df = ensure_scc_ffi_list(
            data_root,
            sector,
            camera,
            ccd,
            paths,
            open_fits=open_fits_memmap,
        )
    frame_wcs: list[tuple[str, Any]] = []
    frame_times: list[str | None] = []
    for p in paths:
        logical = manifest_basename_from_local(p)
        row = ffi_list_df.loc[logical] if logical in ffi_list_df.index else None
        if row is not None and bool(row.get("wcs_ok", False)):
            try:
                frame_wcs.append((logical, wcs_from_cached_row(row)))
            except Exception as exc:
                log.warning("WCS reconstruct failed for %s: %s", logical, exc)
                frame_wcs.append((logical, None))
        else:
            if row is not None and not bool(row.get("wcs_ok", False)):
                log.info("Shift schedule: skipping WCS for %s (wcs_ok=False)", logical)
            frame_wcs.append((logical, None))
        if row is not None and pd.notna(row.get("date_obs")):
            frame_times.append(str(row["date_obs"]))
        else:
            frame_times.append(None)

    btjd = None
    if "btjd" in frames_df.columns:
        btjd = frames_df["btjd"].to_numpy(dtype=np.float64)

    schedule = build_skycell_shift_schedule(
        frame_wcs,
        skycell_df,
        ref_wcs,
        btjd=btjd,
        frame_times=frame_times,
        sector=int(sector),
        raw_drift_outlier_sigma=raw_drift_outlier_sigma,
    )
    schedule.meta = dict(schedule.meta or {})
    schedule.meta["source"] = "built_from_ffi_list"
    schedule.meta["reference_ffi"] = str(ref_path)
    schedule.meta["frame_filenames"] = [manifest_basename_from_local(p) for p in paths]
    with field_store_lock(store_root):
        schedule.save(store_root / "shift_schedule.npz")
        shutil.copy2(store_root / "shift_schedule.npz", event_dir / "shift_schedule.npz")
        store_json = store_root / "shift_schedule.json"
        if store_json.is_file():
            shutil.copy2(store_json, event_dir / "shift_schedule.json")
        _try_write_skycell_shift_debug_plots(
            schedule,
            store_root=store_root,
            mapping_root=mapping_root,
            btjd=btjd,
            ref_wcs=ref_wcs,
            skycell_df=skycell_df,
            sector=sector,
            camera=camera,
            ccd=ccd,
            oversampling_factor=oversampling_factor,
            data_root=data_root,
        )
    log.info(
        "Built shift_schedule.npz (%d frames × %d skycells) for field remap",
        schedule.sx_int.shape[0],
        schedule.sx_int.shape[1],
    )
    return schedule


def _ensure_shift_schedule(
    *,
    event_dir: Path,
    store_root: Path,
    data_root: Path,
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int = 1,
    ffi_dir: str | Path | None = None,
    ref_ffi_path: str | Path | None = None,
    scc_only: bool = False,
    raw_drift_outlier_sigma: float | None = 5.0,
    drift_source: str = "per_skycell",
    target_drift: np.ndarray | None = None,
) -> ShiftSchedule:
    existing = _load_or_copy_shift_schedule(event_dir, store_root)
    if existing is not None:
        return existing
    if scc_only or ref_ffi_path is not None:
        if ref_ffi_path is None or ffi_dir is None:
            raise ValueError("SCC shift schedule requires ref_ffi_path and ffi_dir")
        return _build_shift_schedule_for_scc(
            store_root=store_root,
            data_root=data_root,
            mapping_root=mapping_root,
            ffi_dir=ffi_dir or "",
            sector=sector,
            camera=camera,
            ccd=ccd,
            ref_ffi_path=ref_ffi_path,
            oversampling_factor=oversampling_factor,
            raw_drift_outlier_sigma=raw_drift_outlier_sigma,
            drift_source=drift_source,
            target_drift=target_drift,
        )
    return _build_shift_schedule_for_event(
        event_dir=event_dir,
        store_root=store_root,
        data_root=data_root,
        mapping_root=mapping_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
        oversampling_factor=oversampling_factor,
        raw_drift_outlier_sigma=raw_drift_outlier_sigma,
    )


def _try_write_skycell_shift_debug_plots(
    schedule: ShiftSchedule,
    *,
    store_root: Path,
    mapping_root: Path,
    btjd: np.ndarray | None,
    ref_wcs: Any,
    skycell_df: pd.DataFrame,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int = 1,
    data_root: str | Path | None = None,
    store_name: str | None = None,
) -> None:
    """Best-effort remap debug PNGs under SCC ``debug_plots/`` (never fail remap)."""
    try:
        from syndiff_pipeline.common.scc_paths import (
            REMAP_SUBDIR,
            normalize_store_name,
            scc_debug_plots_dir,
        )
        from syndiff_pipeline.template_creation.processing.shift_schedule_plots import (
            write_skycell_shift_debug_plots,
        )

        name = normalize_store_name(store_name)
        if name is None and store_root is not None:
            # Infer from …/remap[_NAME]/oversampling_N
            parent = Path(store_root).name
            if parent.startswith("oversampling_"):
                parent = Path(store_root).parent.name
            prefix = f"{REMAP_SUBDIR}_"
            if parent.startswith(prefix):
                name = parent[len(prefix) :]
            elif parent != REMAP_SUBDIR:
                name = None

        if data_root is not None:
            out_dir = scc_debug_plots_dir(data_root, sector, camera, ccd)
        else:
            # Fallback: sibling debug_plots next to remap/ under the SCC root.
            out_dir = Path(store_root).parent.parent / "debug_plots"

        master_path = _master_pixels2skycells_path(
            mapping_root,
            sector,
            camera,
            ccd,
            oversampling_factor=oversampling_factor,
        )
        write_skycell_shift_debug_plots(
            schedule,
            out_dir=out_dir,
            btjd=btjd,
            ref_wcs=ref_wcs,
            skycell_df=skycell_df,
            master_path=master_path,
            sector=int(sector),
            camera=int(camera),
            ccd=int(ccd),
            store_name=name,
        )
    except Exception as exc:
        log.warning("Skycell shift debug plots skipped: %s", exc)


def _master_pixels2skycells_path(
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    scc = _mapping_scc_dir(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    canonical = (
        scc
        / f"tess_s{int(sector):04d}_{camera}_{ccd}_master_pixels2skycells{suffix}{PIPELINE_FITS_EXT}"
    )
    found = try_resolve_fits_variant(canonical)
    if found is not None:
        return found
    matches = [
        p
        for p in scc.glob("tess_*_master_pixels2skycells*")
        if p.is_file() and is_fits_storage_filename(p.name)
    ]
    preferred = prefer_fits_path(matches)
    if preferred is not None:
        return Path(preferred)
    raise FileNotFoundError(f"master pixels2skycells not found under {scc}")


def _master_skycell_id_map(
    master_path: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    with fits.open(master_path) as hdul:
        master = np.asarray(hdul[1].data)
        name_to_id: dict[str, int] = {}
        if len(hdul) > 2:
            tab = hdul[2].data
            name_to_id = {
                str(n).strip(): int(i) for n, i in zip(tab["SKYCELL"], tab["SKYCIND"])
            }
    return master, name_to_id


def _frame_index_for_shift(
    schedule: ShiftSchedule,
    skycell: str,
    sx_int: int,
    sy_int: int,
) -> int | None:
    names = np.asarray(schedule.skycell_names).astype(str)
    hits = np.where(names == str(skycell))[0]
    if hits.size == 0:
        return None
    c = int(hits[0])
    valid = np.asarray(schedule.frame_valid).astype(bool)
    match = (
        valid
        & (schedule.sx_int[:, c] == int(sx_int))
        & (schedule.sy_int[:, c] == int(sy_int))
    )
    idxs = np.flatnonzero(match)
    if idxs.size == 0:
        return None
    return int(idxs[0])


def _load_frame_wcs(frames_df: pd.DataFrame, frame_index: int) -> Any:
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        load_tess_wcs,
    )

    path = Path(str(frames_df.iloc[int(frame_index)]["path"]))
    wcs, _ = load_tess_wcs(path)
    return wcs


def _resolve_frame_filenames(
    schedule: ShiftSchedule,
    ffi_list_df: pd.DataFrame,
) -> list[str]:
    meta = schedule.meta or {}
    names = meta.get("frame_filenames")
    if names:
        return [str(n) for n in names]
    n_frames = int(schedule.sx_int.shape[0])
    candidates = sorted(str(n) for n in ffi_list_df.index)
    if len(candidates) != n_frames:
        raise RuntimeError(
            f"shift schedule has {n_frames} frames but ffi_list has "
            f"{len(candidates)} entries and schedule.meta lacks frame_filenames"
        )
    return candidates


def _load_frame_wcs_from_cache(
    ffi_list_df: pd.DataFrame,
    frame_filenames: list[str],
    frame_index: int,
) -> Any:
    from syndiff_pipeline.common.wcs_header_cache import wcs_from_cached_row

    name = frame_filenames[int(frame_index)]
    if name not in ffi_list_df.index:
        raise KeyError(f"frame {name!r} missing from ffi_list")
    return wcs_from_cached_row(ffi_list_df.loc[name])


def _skycell_catalog_row(
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    skycell: str,
    *,
    oversampling_factor: int = 1,
) -> pd.Series:
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        RELEVANT_WCS_KEYS,
    )

    csv_path = _skycell_csv_path(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    usecols = ["NAME", "RA", "DEC"] + RELEVANT_WCS_KEYS
    df = pd.read_csv(csv_path, usecols=usecols)
    rows = df[df["NAME"].astype(str) == str(skycell)]
    if rows.empty:
        raise KeyError(f"skycell {skycell} not in {csv_path}")
    return rows.iloc[0]


def _frame_index_for_pair_state(
    schedule: ShiftSchedule,
    skycell_a: str,
    skycell_b: str,
    sx_a: int,
    sy_a: int,
    sx_b: int,
    sy_b: int,
) -> int | None:
    names = np.asarray(schedule.skycell_names).astype(str)
    hits_a = np.where(names == str(skycell_a))[0]
    hits_b = np.where(names == str(skycell_b))[0]
    if hits_a.size == 0 or hits_b.size == 0:
        return None
    ca = int(hits_a[0])
    cb = int(hits_b[0])
    valid = np.asarray(schedule.frame_valid).astype(bool)
    match = (
        valid
        & (schedule.sx_int[:, ca] == int(sx_a))
        & (schedule.sy_int[:, ca] == int(sy_a))
        & (schedule.sx_int[:, cb] == int(sx_b))
        & (schedule.sy_int[:, cb] == int(sy_b))
    )
    idxs = np.flatnonzero(match)
    if idxs.size == 0:
        return None
    return int(idxs[0])


def _effective_parallel_jobs(n_jobs: int, n_tasks: int) -> int:
    import os as _os

    n_jobs_eff = max(1, min(int(n_jobs), int(n_tasks)))
    hybrid_cap = int(_os.environ.get("SYNDIFF_HYBRID_MAX_JOBS", "24"))
    avail = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (
        _os.cpu_count() or hybrid_cap
    )
    return min(n_jobs_eff, max(1, hybrid_cap), max(1, avail))


# Per-process caches for remap Exact workers (loky reuses processes).
_REMAP_WORKER: dict[str, Any] = {}


def _build_remap_tpix(
    *,
    master_path: str | Path | None,
    base_tess_shape: tuple[int, int],
    oversampling_factor: int,
) -> np.ndarray:
    """Build FFI-pixel ``tpix`` for remap workers (v2 grid-aware)."""
    from syndiff_pipeline.common.coordinate_preflight import assert_wcs_uses_ffi_coords
    from syndiff_pipeline.common.mapping_grid import (
        MappingGridError,
        create_coords_for_grid,
        load_mapping_grid_from_master,
    )

    os_factor = max(1, int(oversampling_factor))
    if not master_path:
        raise MappingGridError(
            "remap requires master_path with MAPGRID>=2; legacy shrunk-chip coords are banned"
        )
    grid = load_mapping_grid_from_master(master_path)
    tpix, _ = create_coords_for_grid(grid, os_factor)
    assert_wcs_uses_ffi_coords(tpix, grid, oversampling_factor=os_factor)
    return tpix


def _init_remap_worker(payload: dict[str, Any]) -> None:
    """Initialize one loky worker: hoist TESS coords once per process."""
    global _REMAP_WORKER

    _REMAP_WORKER = dict(payload)
    master_path = _REMAP_WORKER.get("master_path")
    if master_path and _REMAP_WORKER.get("master") is None:
        master, name_to_id = _master_skycell_id_map(Path(master_path))
        _REMAP_WORKER["master"] = master
        if not _REMAP_WORKER.get("idx_to_name"):
            _REMAP_WORKER["idx_to_name"] = {
                int(v): str(k) for k, v in name_to_id.items()
            }
    tpix = _build_remap_tpix(
        master_path=master_path,
        base_tess_shape=tuple(_REMAP_WORKER["base_tess_shape"]),
        oversampling_factor=int(_REMAP_WORKER["oversampling_factor"]),
    )
    _REMAP_WORKER["_tpix"] = tpix


def _ensure_remap_worker(payload: dict[str, Any]) -> None:
    """Initialize worker caches in the parent (serial path) or after fork."""
    if not _REMAP_WORKER:
        _init_remap_worker(payload)


def _reset_remap_worker() -> None:
    global _REMAP_WORKER
    _REMAP_WORKER = {}


def _ensure_worker_master_tables() -> dict[int, str]:
    """Return ``idx_to_name``, loading ``master`` from FITS at most once per worker."""
    idx_to_name: dict[int, str] = dict(_REMAP_WORKER.get("idx_to_name") or {})
    if _REMAP_WORKER.get("master") is not None:
        return idx_to_name
    master_path = _REMAP_WORKER.get("master_path")
    if not master_path:
        return idx_to_name
    master, name_to_id = _master_skycell_id_map(Path(master_path))
    _REMAP_WORKER["master"] = master
    if not idx_to_name:
        idx_to_name = {int(v): str(k) for k, v in name_to_id.items()}
        _REMAP_WORKER["idx_to_name"] = idx_to_name
    return idx_to_name


def _reset_joblib_executor_args() -> None:
    """Clear joblib's cached Parallel initargs so a second pool can start.

    L4a then L4b both pass ``ShiftSchedule`` (ndarray fields) in initargs.
    joblib compares cached vs new initargs with ``==``, which raises
    ``ValueError: ambiguous truth value`` on numpy arrays. Clearing the cache
    forces a fresh executor for L4b (smoke_4 crash).
    """
    try:
        import joblib.executor as _joblib_executor

        _joblib_executor._executor_args = None
    except Exception:
        pass


def _worker_frame_wcs(frame_index: int) -> Any:
    mode = _REMAP_WORKER["wcs_mode"]
    if mode == "scc":
        return _load_frame_wcs_from_cache(
            _REMAP_WORKER["ffi_list_df"],
            _REMAP_WORKER["frame_filenames"],
            frame_index,
        )
    return _load_frame_wcs(_REMAP_WORKER["frames_df"], frame_index)


def _read_regmap_assignment(skycell: str) -> np.ndarray:
    """Load one skycell TESS_PIXEL_MAP (scratch first, then mapping_root)."""
    regmap_path = _REMAP_WORKER["scratch_regmaps"].get(skycell)
    if regmap_path is None:
        regmap_path = str(
            _find_regmap(
                Path(_REMAP_WORKER["mapping_root"]),
                int(_REMAP_WORKER["sector"]),
                int(_REMAP_WORKER["camera"]),
                int(_REMAP_WORKER["ccd"]),
                skycell,
                oversampling_factor=int(_REMAP_WORKER["oversampling_factor"]),
            )
        )
    with fits.open(regmap_path) as hdul:
        if "TESS_PIXEL_MAP" in hdul:
            return np.asarray(hdul["TESS_PIXEL_MAP"].data)
        return np.asarray(hdul[1].data)


def _group_l4a_epochs_by_skycell(
    shift_epochs: pd.DataFrame,
) -> list[tuple[str, list[tuple[int, int, int, int]]]]:
    """Group L4a epochs by skycell: ``(epoch_id, sx, sy, rep_frame_index)``."""
    from collections import defaultdict

    by_skycell: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    if shift_epochs is None or len(shift_epochs) == 0:
        return []
    for row in shift_epochs.itertuples(index=False):
        by_skycell[str(row.skycell)].append(
            (
                int(row.epoch_id),
                int(row.sx_int),
                int(row.sy_int),
                int(row.rep_frame_index),
            )
        )
    return sorted(by_skycell.items())


def _group_l4b_epochs_by_pair(
    pair_epochs: pd.DataFrame,
) -> list[tuple[tuple[int, int], list[tuple[int, int, int, int, int, int]]]]:
    """Group L4b pair-epochs by abutting pair ``(id_lo, id_hi)``.

    Each epoch tuple is
    ``(pair_epoch_id, sx_lo, sy_lo, sx_hi, sy_hi, rep_frame_index)``.
    """
    from collections import defaultdict

    by_pair: dict[tuple[int, int], list[tuple[int, int, int, int, int, int]]] = (
        defaultdict(list)
    )
    if pair_epochs is None or len(pair_epochs) == 0:
        return []
    for row in pair_epochs.itertuples(index=False):
        by_pair[(int(row.id_lo), int(row.id_hi))].append(
            (
                int(row.pair_epoch_id),
                int(row.sx_lo),
                int(row.sy_lo),
                int(row.sx_hi),
                int(row.sy_hi),
                int(row.rep_frame_index),
            )
        )
    return sorted(by_pair.items())


def _worker_ps1_info(skycell: str) -> pd.Series:
    rows = _REMAP_WORKER["skycell_rows"]
    if skycell not in rows:
        rows[skycell] = _skycell_catalog_row(
            Path(_REMAP_WORKER["mapping_root"]),
            int(_REMAP_WORKER["sector"]),
            int(_REMAP_WORKER["camera"]),
            int(_REMAP_WORKER["ccd"]),
            skycell,
            oversampling_factor=int(_REMAP_WORKER["oversampling_factor"]),
        )
    return rows[skycell]


def _worker_exact_regmap(
    tess_wcs: Any,
    skycell: str,
    tess_ids: np.ndarray,
) -> np.ndarray:
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        exact_regmap_for_tess_ids,
    )

    row = _worker_ps1_info(skycell)
    return exact_regmap_for_tess_ids(
        tess_wcs,
        row,
        tess_ids,
        data_shape=tuple(_REMAP_WORKER["base_tess_shape"]),
        oversampling_factor=int(_REMAP_WORKER["oversampling_factor"]),
        tpix_coord_input=_REMAP_WORKER["_tpix"],
    )


def _l4a_exact_one_shift(
    skycell: str,
    epoch_id: int,
    sx_i: int,
    sy_i: int,
    rep_frame_index: int,
    assignment_map: np.ndarray,
) -> str:
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        l4a_exact_path,
    )
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        candidate_tess_ids_for_l4a,
    )

    exact_l4a_dir = Path(_REMAP_WORKER["exact_l4a_dir"])
    rebuild = bool(_REMAP_WORKER["rebuild_remap_cache"])
    cache_path = l4a_exact_path(exact_l4a_dir, skycell, epoch_id, sx_i, sy_i)
    if cache_path.is_file() and not rebuild:
        return "skip"
    if cache_path.is_file() and rebuild:
        cache_path.unlink()
    frame_i = int(rep_frame_index)
    try:
        tess_wcs = _worker_frame_wcs(frame_i)
        tids, mask = candidate_tess_ids_for_l4a(
            assignment_map,
            sx_i,
            sy_i,
            hybrid_R=int(_REMAP_WORKER["intra_skycell_R"]),
        )
        if tids.size == 0 or int(mask.sum()) == 0:
            return "skip"
        exact = _worker_exact_regmap(tess_wcs, skycell, tids)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            exact_tid=exact.astype(np.int32),
            epoch_id=np.int32(epoch_id),
            rep_frame_index=np.int32(frame_i),
            sx_int=np.int32(sx_i),
            sy_int=np.int32(sy_i),
        )
        return "write"
    except Exception as exc:
        log.warning(
            "Exact cache failed for %s epoch=%d sx=%+d sy=%+d (%s)",
            skycell,
            epoch_id,
            sx_i,
            sy_i,
            exc,
        )
        return "fail"


def _l4a_exact_skycell_batch(
    skycell: str,
    epochs: list[tuple[int, int, int, int]],
) -> list[str]:
    """Process L4a epochs for one skycell: ``(epoch_id, sx, sy, rep_frame_index)``."""
    if not epochs:
        return []
    assignment_map = _read_regmap_assignment(skycell)
    return [
        _l4a_exact_one_shift(skycell, epoch_id, sx_i, sy_i, rep_f, assignment_map)
        for epoch_id, sx_i, sy_i, rep_f in epochs
    ]


def _l4b_rim_one_epoch(
    id_lo: int,
    id_hi: int,
    pair_epoch_id: int,
    sx_lo: int,
    sy_lo: int,
    sx_hi: int,
    sy_hi: int,
    rep_frame_index: int,
    *,
    name_lo: str,
    name_hi: str,
    ids_lo: np.ndarray,
    ids_hi: np.ndarray,
) -> str:
    """Write or skip one L4b rim NPZ; border tess ids are provided by the batch."""
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        l4b_rim_path,
        write_l4b_rim_cache,
    )

    exact_l4b_dir = Path(_REMAP_WORKER["exact_l4b_dir"])
    rebuild = bool(_REMAP_WORKER["rebuild_inter_skycell_cache"])
    cache_path = l4b_rim_path(
        exact_l4b_dir,
        id_lo,
        id_hi,
        pair_epoch_id,
        sx_lo,
        sy_lo,
        sx_hi,
        sy_hi,
    )
    if cache_path.is_file() and not rebuild:
        return "skip"
    if cache_path.is_file() and rebuild:
        cache_path.unlink()

    if ids_lo.size == 0 and ids_hi.size == 0:
        return "skip"

    frame_i = int(rep_frame_index)
    try:
        rep_wcs = _worker_frame_wcs(frame_i)
        exact_tid_lo = (
            _worker_exact_regmap(rep_wcs, name_lo, ids_lo)
            if ids_lo.size
            else np.array([], dtype=np.int32)
        )
        exact_tid_hi = (
            _worker_exact_regmap(rep_wcs, name_hi, ids_hi)
            if ids_hi.size
            else np.array([], dtype=np.int32)
        )

        # Sparse (v2) layout + atomic replace. Rim caches are ~1% valid, so the
        # dense form spent ~315 MB of decompression per read at L5; it also
        # wrote straight to the final path, which could leave a truncated NPZ
        # that a later is_file() skip check would treat as complete.
        write_l4b_rim_cache(
            cache_path,
            exact_tid_lo=exact_tid_lo,
            exact_tid_hi=exact_tid_hi,
            id_lo=int(id_lo),
            id_hi=int(id_hi),
            sx_lo=int(sx_lo),
            sy_lo=int(sy_lo),
            sx_hi=int(sx_hi),
            sy_hi=int(sy_hi),
            pair_epoch_id=int(pair_epoch_id),
            rep_frame_index=frame_i,
        )
        return "write"
    except Exception as exc:
        log.warning(
            "L4b rim cache failed for pair %d|%d epoch=%d (%+d,%+d)|(%+d,%+d): %s",
            id_lo,
            id_hi,
            pair_epoch_id,
            sx_lo,
            sy_lo,
            sx_hi,
            sy_hi,
            exc,
        )
        return "fail"


def _l4b_rim_pair_batch(
    id_lo: int,
    id_hi: int,
    epochs: list[tuple[int, int, int, int, int, int]],
) -> list[str]:
    """Process L4b pair-epochs for one abutting border; hoist border tess ids once."""
    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        shared_abutting_border_tess_ids,
    )

    if not epochs:
        return []

    idx_to_name = _ensure_worker_master_tables()
    name_lo = idx_to_name.get(int(id_lo))
    name_hi = idx_to_name.get(int(id_hi))
    if name_lo is None or name_hi is None:
        log.warning(
            "L4b pair-state ids %d|%d missing from master table; skipping batch",
            id_lo,
            id_hi,
        )
        return ["skip"] * len(epochs)

    master = _REMAP_WORKER["master"]
    if master is None:
        log.warning(
            "L4b pair-state ids %d|%d missing master map; skipping batch",
            id_lo,
            id_hi,
        )
        return ["skip"] * len(epochs)
    ids_lo, ids_hi = shared_abutting_border_tess_ids(master, id_lo, id_hi)
    # Warm PS1 catalog rows once per border (shared across epochs).
    _worker_ps1_info(name_lo)
    _worker_ps1_info(name_hi)

    return [
        _l4b_rim_one_epoch(
            id_lo,
            id_hi,
            pair_epoch_id,
            sx_lo,
            sy_lo,
            sx_hi,
            sy_hi,
            rep_frame_index,
            name_lo=name_lo,
            name_hi=name_hi,
            ids_lo=ids_lo,
            ids_hi=ids_hi,
        )
        for pair_epoch_id, sx_lo, sy_lo, sx_hi, sy_hi, rep_frame_index in epochs
    ]


def _build_remap_worker_payload(
    *,
    schedule: ShiftSchedule,
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    base_tess_shape: tuple[int, int],
    oversampling_factor: int,
    intra_skycell_R: int,
    exact_l4a_dir: Path,
    exact_l4b_dir: Path,
    rebuild_remap_cache: bool,
    rebuild_inter_skycell_cache: bool,
    scratch_regmaps: dict[str, str],
    wcs_mode: str,
    ffi_list_df: pd.DataFrame | None = None,
    frame_filenames: list[str] | None = None,
    frames_df: pd.DataFrame | None = None,
    master: np.ndarray | None = None,
    master_path: str | Path | None = None,
    idx_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    # Prefer master_path over master ndarray so joblib initargs equality does
    # not compare large numpy arrays when spinning up the L4b pool after L4a.
    payload: dict[str, Any] = {
        "schedule": schedule,
        "mapping_root": str(mapping_root),
        "sector": int(sector),
        "camera": int(camera),
        "ccd": int(ccd),
        "base_tess_shape": tuple(int(x) for x in base_tess_shape),
        "oversampling_factor": int(oversampling_factor),
        "intra_skycell_R": int(intra_skycell_R),
        "exact_l4a_dir": str(exact_l4a_dir),
        "exact_l4b_dir": str(exact_l4b_dir),
        "rebuild_remap_cache": bool(rebuild_remap_cache),
        "rebuild_inter_skycell_cache": bool(rebuild_inter_skycell_cache),
        "scratch_regmaps": dict(scratch_regmaps),
        "skycell_rows": {},
        "wcs_mode": wcs_mode,
        "ffi_list_df": ffi_list_df,
        "frame_filenames": frame_filenames,
        "frames_df": frames_df,
        "master": None if master_path is not None else master,
        "master_path": str(master_path) if master_path is not None else None,
        "idx_to_name": idx_to_name or {},
    }
    return payload


def _stage_remap_regmaps_to_scratch(
    *,
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int,
    skycells: set[str],
    stage_regmaps_to_scratch: bool | None = None,
) -> tuple[dict[str, str], int, float]:
    """Copy unique skycell regmaps to local scratch; return skycell→path map."""
    from syndiff_pipeline.template_creation.processing.downsample import (
        resolve_stage_regmaps_to_scratch,
        stage_regmap_files_to_scratch,
    )

    if not skycells or not resolve_stage_regmaps_to_scratch(stage_regmaps_to_scratch):
        return {}, 0, 0.0

    sky_reg: list[tuple[str, str]] = []
    for sc in sorted(skycells):
        try:
            sky_reg.append(
                (
                    sc,
                    str(
                        _find_regmap(
                            mapping_root,
                            sector,
                            camera,
                            ccd,
                            sc,
                            oversampling_factor=oversampling_factor,
                        )
                    ),
                )
            )
        except FileNotFoundError:
            continue
    if not sky_reg:
        return {}, 0, 0.0

    local_paths, scratch_dir, n_staged, elapsed = stage_regmap_files_to_scratch(
        [p for _, p in sky_reg],
        sector=sector,
        camera=camera,
        ccd=ccd,
        oversampling_factor=oversampling_factor,
    )
    scratch_regmaps = {sc: lp for (sc, _), lp in zip(sky_reg, local_paths)}
    log.info(
        "Staged %d/%d remap regmaps to scratch %s in %.1fs",
        n_staged,
        len(sky_reg),
        scratch_dir,
        elapsed,
    )
    return scratch_regmaps, n_staged, elapsed


def load_gid_epoch_index(
    path: str | Path,
    *,
    include_inter: bool = True,
) -> dict[str, Any]:
    """Load ``gid_epoch_index.npz`` into dicts for O(1) compose lookup.

    Arrays are materialized once before the Python dict loops. Walking an
    open ``NpzFile`` element-by-element (especially object ``skycell``
    strings) is pathologically slow on large indexes.

    When ``include_inter`` is False, skip the L4b half entirely (intra-only
    downsample).
    """
    data = np.load(Path(path), allow_pickle=True)
    # Materialize once — do not index into NpzFile inside the loops.
    l4a_skycell = np.asarray(data["l4a_skycell"])
    l4a_gid = np.asarray(data["l4a_gid"], dtype=np.int32)
    l4a_sx = np.asarray(data["l4a_sx"], dtype=np.int32)
    l4a_sy = np.asarray(data["l4a_sy"], dtype=np.int32)
    l4a_epoch_id = np.asarray(data["l4a_epoch_id"], dtype=np.int32)
    l4a: dict[tuple[str, int, int, int], int] = {
        (str(sk), int(gid), int(sx), int(sy)): int(eid)
        for sk, gid, sx, sy, eid in zip(
            l4a_skycell, l4a_gid, l4a_sx, l4a_sy, l4a_epoch_id, strict=True
        )
    }
    l4b: dict[tuple[int, int, int, int, int, int, int], int] = {}
    if include_inter and "l4b_gid" in data.files:
        l4b_gid = np.asarray(data["l4b_gid"], dtype=np.int32)
        if len(l4b_gid):
            l4b_pair_lo = np.asarray(data["l4b_pair_lo"], dtype=np.int32)
            l4b_pair_hi = np.asarray(data["l4b_pair_hi"], dtype=np.int32)
            l4b_sx_lo = np.asarray(data["l4b_sx_lo"], dtype=np.int32)
            l4b_sy_lo = np.asarray(data["l4b_sy_lo"], dtype=np.int32)
            l4b_sx_hi = np.asarray(data["l4b_sx_hi"], dtype=np.int32)
            l4b_sy_hi = np.asarray(data["l4b_sy_hi"], dtype=np.int32)
            l4b_pair_epoch_id = np.asarray(data["l4b_pair_epoch_id"], dtype=np.int32)
            l4b = {
                (
                    int(plo),
                    int(phi),
                    int(gid),
                    int(sx_lo),
                    int(sy_lo),
                    int(sx_hi),
                    int(sy_hi),
                ): int(peid)
                for plo, phi, gid, sx_lo, sy_lo, sx_hi, sy_hi, peid in zip(
                    l4b_pair_lo,
                    l4b_pair_hi,
                    l4b_gid,
                    l4b_sx_lo,
                    l4b_sy_lo,
                    l4b_sx_hi,
                    l4b_sy_hi,
                    l4b_pair_epoch_id,
                    strict=True,
                )
            }
    return {"l4a": l4a, "l4b": l4b}


def resolve_l4a_epoch_id(
    epoch_index: Mapping[str, Any],
    *,
    skycell: str,
    group_id: int,
    sx_int: int,
    sy_int: int,
) -> int:
    key = (str(skycell), int(group_id), int(sx_int), int(sy_int))
    l4a = epoch_index["l4a"]
    if key not in l4a:
        raise KeyError(f"No L4a epoch for {key}")
    return int(l4a[key])


def resolve_l4b_pair_epoch_id(
    epoch_index: Mapping[str, Any],
    *,
    id_lo: int,
    id_hi: int,
    group_id: int,
    sx_lo: int,
    sy_lo: int,
    sx_hi: int,
    sy_hi: int,
) -> int:
    lo, hi = (int(id_lo), int(id_hi)) if int(id_lo) <= int(id_hi) else (int(id_hi), int(id_lo))
    if (int(id_lo), int(id_hi)) != (lo, hi):
        sx_lo, sy_lo, sx_hi, sy_hi = int(sx_hi), int(sy_hi), int(sx_lo), int(sy_lo)
    key = (lo, hi, int(group_id), int(sx_lo), int(sy_lo), int(sx_hi), int(sy_hi))
    l4b = epoch_index["l4b"]
    if key not in l4b:
        raise KeyError(f"No L4b pair-epoch for {key}")
    return int(l4b[key])


def _wipe_exact_cache_tree(cache_dir: Path) -> None:
    """Remove an Exact cache directory tree and recreate an empty root."""
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _write_gid_epoch_index(
    path: Path,
    *,
    shift_epochs: pd.DataFrame,
    pair_epochs: pd.DataFrame,
    members: pd.DataFrame,
) -> None:
    """Persist O(1) compose lookup arrays for L4a/L4b epochs."""
    l4a_m = members[members["kind"] == "l4a"] if len(members) else members
    l4b_m = members[members["kind"] == "l4b"] if len(members) else members

    if len(l4a_m) and len(shift_epochs):
        l4a_join = l4a_m.merge(
            shift_epochs[["skycell", "epoch_id", "sx_int", "sy_int"]],
            left_on=["scope_key", "epoch_id"],
            right_on=["skycell", "epoch_id"],
            how="inner",
        )
        l4a_skycell = l4a_join["scope_key"].astype(str).to_numpy(dtype=object)
        l4a_gid = l4a_join["group_id"].to_numpy(dtype=np.int32)
        l4a_sx = l4a_join["sx_int"].to_numpy(dtype=np.int32)
        l4a_sy = l4a_join["sy_int"].to_numpy(dtype=np.int32)
        l4a_epoch_id = l4a_join["epoch_id"].to_numpy(dtype=np.int32)
    else:
        l4a_skycell = np.asarray([], dtype=object)
        l4a_gid = np.asarray([], dtype=np.int32)
        l4a_sx = np.asarray([], dtype=np.int32)
        l4a_sy = np.asarray([], dtype=np.int32)
        l4a_epoch_id = np.asarray([], dtype=np.int32)

    if len(l4b_m) and len(pair_epochs):
        l4b_prep = l4b_m.copy()
        parts = l4b_prep["scope_key"].astype(str).str.split("__", expand=True)
        l4b_prep["id_lo"] = parts[0].str.removeprefix("pair_").astype(np.int32)
        l4b_prep["id_hi"] = parts[1].astype(np.int32)
        l4b_join = l4b_prep.merge(
            pair_epochs[
                [
                    "id_lo",
                    "id_hi",
                    "pair_epoch_id",
                    "sx_lo",
                    "sy_lo",
                    "sx_hi",
                    "sy_hi",
                ]
            ],
            left_on=["id_lo", "id_hi", "epoch_id"],
            right_on=["id_lo", "id_hi", "pair_epoch_id"],
            how="inner",
        )
        l4b_pair_lo = l4b_join["id_lo"].to_numpy(dtype=np.int32)
        l4b_pair_hi = l4b_join["id_hi"].to_numpy(dtype=np.int32)
        l4b_gid = l4b_join["group_id"].to_numpy(dtype=np.int32)
        l4b_sx_lo = l4b_join["sx_lo"].to_numpy(dtype=np.int32)
        l4b_sy_lo = l4b_join["sy_lo"].to_numpy(dtype=np.int32)
        l4b_sx_hi = l4b_join["sx_hi"].to_numpy(dtype=np.int32)
        l4b_sy_hi = l4b_join["sy_hi"].to_numpy(dtype=np.int32)
        l4b_pair_epoch_id = l4b_join["epoch_id"].to_numpy(dtype=np.int32)
    else:
        l4b_pair_lo = np.asarray([], dtype=np.int32)
        l4b_pair_hi = np.asarray([], dtype=np.int32)
        l4b_gid = np.asarray([], dtype=np.int32)
        l4b_sx_lo = np.asarray([], dtype=np.int32)
        l4b_sy_lo = np.asarray([], dtype=np.int32)
        l4b_sx_hi = np.asarray([], dtype=np.int32)
        l4b_sy_hi = np.asarray([], dtype=np.int32)
        l4b_pair_epoch_id = np.asarray([], dtype=np.int32)

    np.savez_compressed(
        path,
        l4a_skycell=l4a_skycell,
        l4a_gid=l4a_gid,
        l4a_sx=l4a_sx,
        l4a_sy=l4a_sy,
        l4a_epoch_id=l4a_epoch_id,
        l4b_pair_lo=l4b_pair_lo,
        l4b_pair_hi=l4b_pair_hi,
        l4b_gid=l4b_gid,
        l4b_sx_lo=l4b_sx_lo,
        l4b_sy_lo=l4b_sy_lo,
        l4b_sx_hi=l4b_sx_hi,
        l4b_sy_hi=l4b_sy_hi,
        l4b_pair_epoch_id=l4b_pair_epoch_id,
    )


def _write_epoch_artifacts(
    store: Path,
    *,
    schedule: ShiftSchedule,
    group_id_per_frame: np.ndarray,
    pair_ids: np.ndarray,
    pair_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write shift/pair epoch tables + membership + gid index under *store*."""
    shift_epochs, l4a_members = build_shift_epochs(schedule, group_id_per_frame)
    pair_epochs, l4b_members = build_pair_epochs(
        schedule,
        group_id_per_frame,
        pair_ids=pair_ids,
        pair_idx=pair_idx,
    )
    members = pd.concat([l4a_members, l4b_members], ignore_index=True)
    shift_epochs.to_parquet(store / SHIFT_EPOCHS_PARQUET, index=False)
    pair_epochs.to_parquet(store / PAIR_EPOCHS_PARQUET, index=False)
    members.to_parquet(store / EPOCH_GROUP_MEMBERS_PARQUET, index=False)
    np.save(store / GROUP_ID_PER_FRAME_NPY, np.asarray(group_id_per_frame, dtype=np.int32))
    _write_gid_epoch_index(
        store / GID_EPOCH_INDEX_NPZ,
        shift_epochs=shift_epochs,
        pair_epochs=pair_epochs,
        members=members,
    )
    return shift_epochs, pair_epochs, members


def _write_remap_manifest(
    store: Path,
    *,
    oversampling_factor: int,
    intra_skycell_R: int,
    cache_quantum_ps1_px: float,
    keying: str,
    n_intra_skycell_keys: int,
    n_groups: int,
    reference_ffi: str | None,
    n_inter_skycell_pair_states: int = 0,
    n_shift_epochs: int = 0,
    n_pair_epochs: int = 0,
    rebuild_inter_skycell_cache: bool = False,
    shift_schedule_frame_origin_counts: dict[str, int] | None = None,
) -> None:
    payload = {
        "schema_version": REMAP_SCHEMA_VERSION,
        "geometry_mode": "field",
        "oversampling_factor": int(oversampling_factor),
        "intra_skycell_R": int(intra_skycell_R),
        "cache_quantum_ps1_px": float(cache_quantum_ps1_px),
        "keying": str(keying),
        "n_shift_epochs": int(n_shift_epochs),
        "n_pair_epochs": int(n_pair_epochs),
        # Legacy aliases (same counts as epochs under schema v3)
        "n_intra_skycell_keys": int(n_shift_epochs if n_shift_epochs else n_intra_skycell_keys),
        "n_inter_skycell_pair_states": int(
            n_pair_epochs if n_pair_epochs else n_inter_skycell_pair_states
        ),
        "exact_cache_l4a": EXACT_CACHE_L4A_DIRNAME,
        "n_groups": int(n_groups),
        "exact_cache_l4b": EXACT_CACHE_L4B_DIRNAME,
        "rebuild_inter_skycell_cache": bool(rebuild_inter_skycell_cache),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    if shift_schedule_frame_origin_counts is not None:
        payload["shift_schedule_frame_origin_counts"] = dict(
            shift_schedule_frame_origin_counts
        )
    if reference_ffi:
        payload["reference_ffi"] = str(reference_ffi)
    (store / REMAP_MANIFEST_NAME).write_text(json.dumps(payload, indent=2) + "\n")


def run_field_remap_scc(
    *,
    sector: int,
    camera: int,
    ccd: int,
    data_root: str | Path,
    event_dir: str | Path,
    mapping_root: str | Path,
    base_tess_shape: tuple[int, int],
    oversampling_factor: int = 1,
    grouping_quantum_ps1_px: float = 1.0,
    cache_quantum_ps1_px: float = 1.0,
    keying: str = "absolute",
    intra_skycell_R: int = 1,
    rebuild_remap_cache: bool = False,
    rebuild_inter_skycell_cache: bool = False,
    store_root: str | Path | None = None,
    scc_only: bool = False,
    ffi_dir: str | Path | None = None,
    ref_ffi_path: str | Path | None = None,
    n_jobs: int = 1,
    progress_path: str | Path | None = None,
    raw_drift_outlier_sigma: float | None = 5.0,
    stage_regmaps_to_scratch: bool | None = None,
    drift_source: str = "per_skycell",
    target_drift: np.ndarray | None = None,
    apply_intra_skycell: bool = True,
    apply_inter_skycell: bool = True,
) -> dict[str, Any]:
    """Build or reuse SCC remap artifacts (L2–L4).

    Writes under ``remap/oversampling_{N}/``: shift schedule, group artifacts,
    ``exact_cache_l4a/`` (intra-skycell, when ``apply_intra_skycell``),
    ``exact_cache_l4b/`` (inter-skycell, when ``apply_inter_skycell``),
    and ``remap_manifest.json``.
    """
    from joblib import delayed

    from syndiff_pipeline.template_creation.processing import remap_progress

    progress_file = Path(progress_path) if progress_path is not None else None
    if progress_file is not None:
        remap_progress.init_progress(
            progress_file,
            oversampling_factor=int(oversampling_factor),
        )

    event_dir = Path(event_dir)
    data_root = Path(data_root)
    mapping_root = Path(mapping_root)
    store = Path(store_root) if store_root is not None else remap_root(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    store.mkdir(parents=True, exist_ok=True)
    exact_l4a_dir = store / EXACT_CACHE_L4A_DIRNAME
    exact_l4a_dir.mkdir(exist_ok=True)
    exact_l4b_dir = store / EXACT_CACHE_L4B_DIRNAME
    exact_l4b_dir.mkdir(exist_ok=True)

    schedule = _ensure_shift_schedule(
        event_dir=event_dir,
        store_root=store,
        data_root=data_root,
        mapping_root=mapping_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
        oversampling_factor=oversampling_factor,
        ffi_dir=ffi_dir,
        ref_ffi_path=ref_ffi_path,
        scc_only=scc_only,
        raw_drift_outlier_sigma=raw_drift_outlier_sigma,
        drift_source=drift_source,
        target_drift=target_drift,
    )
    if progress_file is not None:
        remap_progress.set_progress_phase(progress_file, "grouping")
    assignment = assign_groups_from_schedule(
        schedule,
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=cache_quantum_ps1_px,
        keying=keying,
    )
    write_group_artifacts(
        assignment,
        store,
        geometry_mode="field",
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=cache_quantum_ps1_px,
    )
    if not scc_only:
        write_group_artifacts(
            assignment,
            event_dir,
            geometry_mode="field",
            grouping_quantum_ps1_px=grouping_quantum_ps1_px,
            cache_quantum_ps1_px=cache_quantum_ps1_px,
        )

    # Contiguous group islands are already assigned. Build epoch tables next
    # (needed before Exact enumeration) and persist group_id_per_frame.
    from syndiff_pipeline.template_creation.processing.field_abutting import (
        abutting_undirected_pairs,
        build_col_of_name,
        pair_column_indices,
    )

    master_path = _master_pixels2skycells_path(
        mapping_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
    )
    master, name_to_id = _master_skycell_id_map(master_path)
    idx_to_name = {int(v): str(k) for k, v in name_to_id.items()}
    pair_ids = abutting_undirected_pairs(master)
    col_of_name = build_col_of_name(schedule.skycell_names)
    pair_ids, pair_idx = pair_column_indices(
        pair_ids,
        name_to_id=name_to_id,
        col_of_name=col_of_name,
        idx_to_name=idx_to_name,
    )
    shift_epochs, pair_epochs, _members = _write_epoch_artifacts(
        store,
        schedule=schedule,
        group_id_per_frame=assignment.group_id_per_frame,
        pair_ids=pair_ids,
        pair_idx=pair_idx,
    )
    if not scc_only:
        _write_epoch_artifacts(
            event_dir,
            schedule=schedule,
            group_id_per_frame=assignment.group_id_per_frame,
            pair_ids=pair_ids,
            pair_idx=pair_idx,
        )

    n_intra_skycell_keys = int(len(shift_epochs))
    n_inter_skycell_pair_states = int(len(pair_epochs))
    n_intra_skycell_written = 0
    n_inter_skycell_written = 0
    run_intra = bool(n_intra_skycell_keys) and bool(apply_intra_skycell)
    run_inter = bool(apply_inter_skycell)

    if rebuild_remap_cache:
        _wipe_exact_cache_tree(exact_l4a_dir)
    if rebuild_inter_skycell_cache:
        _wipe_exact_cache_tree(exact_l4b_dir)

    if progress_file is not None:
        if run_intra:
            remap_progress.init_exact_l4a_cache(progress_file, n_intra_skycell_keys)
        else:
            remap_progress.set_progress_phase(progress_file, "complete")

    frame_wcs_loader: Any | None = None
    wcs_mode = "event"
    ffi_list_df: pd.DataFrame | None = None
    frame_filenames: list[str] | None = None
    frames_df: pd.DataFrame | None = None
    if run_intra or run_inter:
        if scc_only:
            from syndiff_pipeline.common.wcs_header_cache import load_ffi_list
            from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet

            wcs_mode = "scc"
            ffi_list_path = scc_ffi_list_parquet(data_root, sector, camera, ccd)
            if not ffi_list_path.is_file():
                raise FileNotFoundError(f"Missing ffi_list at {ffi_list_path}")
            ffi_list_df = load_ffi_list(ffi_list_path)
            frame_filenames = _resolve_frame_filenames(schedule, ffi_list_df)

            def _frame_wcs_at(frame_index: int) -> Any:
                return _load_frame_wcs_from_cache(ffi_list_df, frame_filenames, frame_index)
        else:
            frames_path = Path(_frames_csv_path(event_dir))
            if not frames_path.is_file():
                raise FileNotFoundError(f"Missing frames CSV at {frames_path}")
            frames_df = pd.read_csv(frames_path)

            def _frame_wcs_at(frame_index: int) -> Any:
                return _load_frame_wcs(frames_df, frame_index)

        frame_wcs_loader = _frame_wcs_at

    scratch_regmaps: dict[str, str] = {}
    regmaps_staged = 0
    scratch_elapsed_s = 0.0
    if run_intra or run_inter:
        skycells_for_stage = (
            set(str(s) for s in shift_epochs["skycell"].unique())
            if len(shift_epochs)
            else set()
        )
        for id_a, id_b in pair_ids:
            for sid in (id_a, id_b):
                name = idx_to_name.get(int(sid))
                if name:
                    skycells_for_stage.add(str(name))
        scratch_regmaps, regmaps_staged, scratch_elapsed_s = _stage_remap_regmaps_to_scratch(
            mapping_root=mapping_root,
            sector=sector,
            camera=camera,
            ccd=ccd,
            oversampling_factor=oversampling_factor,
            skycells=skycells_for_stage,
            stage_regmaps_to_scratch=stage_regmaps_to_scratch,
        )

    if run_intra:
        assert frame_wcs_loader is not None

        skycell_batches = _group_l4a_epochs_by_skycell(shift_epochs)
        n_jobs_eff = _effective_parallel_jobs(n_jobs, len(skycell_batches))
        worker_payload = _build_remap_worker_payload(
            schedule=schedule,
            mapping_root=mapping_root,
            sector=sector,
            camera=camera,
            ccd=ccd,
            base_tess_shape=base_tess_shape,
            oversampling_factor=oversampling_factor,
            intra_skycell_R=intra_skycell_R,
            exact_l4a_dir=exact_l4a_dir,
            exact_l4b_dir=exact_l4b_dir,
            rebuild_remap_cache=rebuild_remap_cache,
            rebuild_inter_skycell_cache=rebuild_inter_skycell_cache,
            scratch_regmaps=scratch_regmaps,
            wcs_mode=wcs_mode,
            ffi_list_df=ffi_list_df,
            frame_filenames=frame_filenames,
            frames_df=frames_df,
            master_path=master_path,
            idx_to_name=idx_to_name,
        )
        if progress_file is not None:
            import os as _os

            perf_meta = {
                "n_jobs_requested": int(n_jobs),
                "n_jobs_eff": int(n_jobs_eff),
                "regmaps_staged": int(regmaps_staged),
                "scratch_elapsed_s": round(scratch_elapsed_s, 3),
                "worker_cache": "tpix_skycell_batch",
            }
            tag = _os.environ.get("SYNDIFF_REMAP_BENCHMARK_TAG")
            if tag:
                perf_meta["benchmark_tag"] = tag
            remap_progress.set_perf_metadata(progress_file, **perf_meta)

        def _on_l4a_done(_status: str) -> None:
            if progress_file is not None:
                remap_progress.mark_exact_l4a_done(progress_file)

        def _on_l4a_batch_done(batch_statuses: list[str]) -> None:
            for status in batch_statuses:
                _on_l4a_done(status)

        l4a_t0 = time.perf_counter()
        if n_jobs_eff == 1 or len(skycell_batches) <= 1:
            _reset_remap_worker()
            _ensure_remap_worker(worker_payload)
            exact_statuses = []
            for skycell, epochs in skycell_batches:
                batch_statuses = _l4a_exact_skycell_batch(skycell, epochs)
                for status in batch_statuses:
                    _on_l4a_done(status)
                exact_statuses.extend(batch_statuses)
        else:
            from syndiff_pipeline.common.joblib_progress import (
                parallel_map_with_optional_tqdm,
            )

            batch_results = parallel_map_with_optional_tqdm(
                (
                    delayed(_l4a_exact_skycell_batch)(sc, epochs)
                    for sc, epochs in skycell_batches
                ),
                n_tasks=len(skycell_batches),
                desc="remap L4a exact",
                n_jobs_eff=n_jobs_eff,
                prefer="processes",
                initializer=_init_remap_worker,
                initargs=(worker_payload,),
                on_result=_on_l4a_batch_done,
            )
            exact_statuses = [s for batch in batch_results for s in batch]
        l4a_elapsed_s = time.perf_counter() - l4a_t0
        n_intra_skycell_written = sum(1 for s in exact_statuses if s == "write")
        l4a_rate = (
            n_intra_skycell_keys / l4a_elapsed_s if l4a_elapsed_s > 0 else 0.0
        )
        log.info(
            "Intra-skycell exact cache: %d epochs (%d skycells), %d written in %.1fs (%.2f task/s)",
            n_intra_skycell_keys,
            len(skycell_batches),
            n_intra_skycell_written,
            l4a_elapsed_s,
            l4a_rate,
        )
        if progress_file is not None:
            remap_progress.set_perf_metadata(
                progress_file,
                l4a_elapsed_s=round(l4a_elapsed_s, 3),
                l4a_task_rate_per_s=round(l4a_rate, 3),
                l4a_keys_processed=n_intra_skycell_keys,
                l4a_skycell_batches=len(skycell_batches),
            )

    if run_inter:
        assert frame_wcs_loader is not None

        if progress_file is not None:
            if n_inter_skycell_pair_states:
                remap_progress.init_exact_l4b_cache(
                    progress_file, n_inter_skycell_pair_states
                )
            elif run_intra:
                remap_progress.set_progress_phase(
                    progress_file,
                    "complete",
                    exact_l4a_done=n_intra_skycell_keys,
                    exact_l4a_total=n_intra_skycell_keys,
                )
            else:
                remap_progress.set_progress_phase(progress_file, "complete")

        if n_inter_skycell_pair_states:
            pair_batches = _group_l4b_epochs_by_pair(pair_epochs)
            n_jobs_l4b = _effective_parallel_jobs(n_jobs, len(pair_batches))
            worker_payload = _build_remap_worker_payload(
                schedule=schedule,
                mapping_root=mapping_root,
                sector=sector,
                camera=camera,
                ccd=ccd,
                base_tess_shape=base_tess_shape,
                oversampling_factor=oversampling_factor,
                intra_skycell_R=intra_skycell_R,
                exact_l4a_dir=exact_l4a_dir,
                exact_l4b_dir=exact_l4b_dir,
                rebuild_remap_cache=rebuild_remap_cache,
                rebuild_inter_skycell_cache=rebuild_inter_skycell_cache,
                scratch_regmaps=scratch_regmaps,
                wcs_mode=wcs_mode,
                ffi_list_df=ffi_list_df,
                frame_filenames=frame_filenames,
                frames_df=frames_df,
                master_path=master_path,
                idx_to_name=idx_to_name,
            )
            if progress_file is not None:
                remap_progress.set_perf_metadata(
                    progress_file,
                    l4b_n_jobs_eff=int(n_jobs_l4b),
                    l4b_pair_batches=len(pair_batches),
                    worker_cache="tpix_border_batch",
                )

            def _on_l4b_done(_status: str) -> None:
                if progress_file is not None:
                    remap_progress.mark_exact_l4b_done(progress_file)

            def _on_l4b_batch_done(batch_statuses: list[str]) -> None:
                for status in batch_statuses:
                    _on_l4b_done(status)

            # Avoid joblib initargs == crash after L4a pool (ShiftSchedule ndarrays).
            _reset_joblib_executor_args()
            _reset_remap_worker()

            l4b_t0 = time.perf_counter()
            if n_jobs_l4b == 1 or len(pair_batches) <= 1:
                _reset_remap_worker()
                _ensure_remap_worker(worker_payload)
                l4b_statuses: list[str] = []
                for (id_lo, id_hi), epochs in pair_batches:
                    batch_statuses = _l4b_rim_pair_batch(id_lo, id_hi, epochs)
                    for status in batch_statuses:
                        _on_l4b_done(status)
                    l4b_statuses.extend(batch_statuses)
            else:
                from syndiff_pipeline.common.joblib_progress import (
                    parallel_map_with_optional_tqdm,
                )

                batch_results = parallel_map_with_optional_tqdm(
                    (
                        delayed(_l4b_rim_pair_batch)(id_lo, id_hi, epochs)
                        for (id_lo, id_hi), epochs in pair_batches
                    ),
                    n_tasks=len(pair_batches),
                    desc="remap L4b rim",
                    n_jobs_eff=n_jobs_l4b,
                    prefer="processes",
                    initializer=_init_remap_worker,
                    initargs=(worker_payload,),
                    on_result=_on_l4b_batch_done,
                )
                l4b_statuses = [s for batch in batch_results for s in batch]
            l4b_elapsed_s = time.perf_counter() - l4b_t0
            n_inter_skycell_written = sum(1 for s in l4b_statuses if s == "write")
            l4b_rate = (
                n_inter_skycell_pair_states / l4b_elapsed_s if l4b_elapsed_s > 0 else 0.0
            )
            log.info(
                "Inter-skycell rim cache: %d pair-epochs in %d pair batches, "
                "%d written in %.1fs (%.2f epoch/s)",
                n_inter_skycell_pair_states,
                len(pair_batches),
                n_inter_skycell_written,
                l4b_elapsed_s,
                l4b_rate,
            )
            if (
                n_inter_skycell_pair_states > 0
                and n_inter_skycell_written == 0
                and not any(exact_l4b_dir.rglob("*_rim.npz"))
            ):
                raise RuntimeError(
                    f"L4b rim cache wrote 0 of {n_inter_skycell_pair_states} "
                    "pair-epochs and exact_cache_l4b is empty; check remap.log "
                    "for skipped batches"
                )
            if progress_file is not None:
                remap_progress.set_perf_metadata(
                    progress_file,
                    l4b_elapsed_s=round(l4b_elapsed_s, 3),
                    l4b_task_rate_per_s=round(l4b_rate, 3),
                    l4b_keys_processed=n_inter_skycell_pair_states,
                    l4b_pair_batches=len(pair_batches),
                )

        if progress_file is not None:
            remap_progress.set_progress_phase(
                progress_file,
                "complete",
                exact_l4a_done=n_intra_skycell_keys if run_intra else None,
                exact_l4a_total=n_intra_skycell_keys if run_intra else None,
                exact_l4b_done=(
                    n_inter_skycell_pair_states
                    if run_inter and n_inter_skycell_pair_states
                    else None
                ),
                exact_l4b_total=(
                    n_inter_skycell_pair_states
                    if run_inter and n_inter_skycell_pair_states
                    else None
                ),
            )

    ref_ffi = None
    if schedule.meta:
        ref_ffi = schedule.meta.get("reference_ffi")
    origin_counts = None
    if schedule.meta:
        origin_counts = schedule.meta.get("frame_origin_counts")
    _write_remap_manifest(
        store,
        oversampling_factor=oversampling_factor,
        intra_skycell_R=intra_skycell_R,
        cache_quantum_ps1_px=cache_quantum_ps1_px,
        keying=keying,
        n_intra_skycell_keys=n_intra_skycell_keys,
        n_groups=len(assignment.groups),
        reference_ffi=str(ref_ffi) if ref_ffi else None,
        n_inter_skycell_pair_states=n_inter_skycell_pair_states,
        n_shift_epochs=n_intra_skycell_keys,
        n_pair_epochs=n_inter_skycell_pair_states,
        rebuild_inter_skycell_cache=bool(rebuild_inter_skycell_cache),
        shift_schedule_frame_origin_counts=origin_counts,
    )

    return {
        "output_dir": str(store),
        "n_groups": len(assignment.groups),
        "n_intra_skycell_keys": n_intra_skycell_keys,
        "n_intra_skycell_written": n_intra_skycell_written,
        "n_inter_skycell_pair_states": n_inter_skycell_pair_states,
        "n_inter_skycell_written": n_inter_skycell_written,
        "n_shift_epochs": n_intra_skycell_keys,
        "n_pair_epochs": n_inter_skycell_pair_states,
        "geometry_mode": "field",
        "intra_skycell_R": int(intra_skycell_R),
        "rebuild_remap_cache": bool(rebuild_remap_cache),
        "rebuild_inter_skycell_cache": bool(rebuild_inter_skycell_cache),
        "shift_schedule_frame_origin_counts": origin_counts,
    }


def _find_regmap(
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    skycell: str,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Locate one skycell regmap; never cross oversampling trees."""
    scc = _mapping_scc_dir(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    name = str(skycell).strip()
    if not name.startswith("skycell."):
        name = f"skycell.{name}"
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    candidates: list[Path] = []
    for sfx in FITS_STORAGE_SUFFIXES:
        candidates.append(scc / f"tess_s{int(sector)}_{camera}_{ccd}_{name}{suffix}{sfx}")
        candidates.append(
            scc / f"tess_s{int(sector):04d}_{camera}_{ccd}_{name}{suffix}{sfx}"
        )
    found = prefer_fits_path(candidates)
    if found is not None:
        return Path(found)
    matches = sorted(
        p
        for p in scc.glob(f"*_{name}{suffix}.fits*")
        if p.is_file() and is_fits_storage_filename(p.name)
    )
    preferred = prefer_fits_path(matches)
    if preferred is not None:
        return Path(preferred)
    raise FileNotFoundError(
        f"No regmap for {name} under {scc} (oversampling_factor={oversampling_factor})"
    )
