"""Field-mode remap: SCC shift schedule, group artifacts, and Exact caches.

Owns L2–L4 under ``{data_root}/s{SSSS}/c{C}/k{K}/remap/oversampling_{N}/``.
Downsample (L5) reads these artifacts and bins sparse contribs under
``templates/oversampling_{N}/``.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    write_group_artifacts,
)

log = logging.getLogger(__name__)

REMAP_MANIFEST_NAME = "remap_manifest.json"
EXACT_CACHE_DIRNAME = "exact_cache"
REMAP_SCHEMA_VERSION = 1


def remap_root(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Return the SCC remap store directory (does not create it)."""
    return scc_remap_dir(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )


def resolve_remap_read_root(
    remap_store: str | Path,
    templates_store: str | Path,
) -> tuple[Path, bool]:
    """Resolve where to read remap artifacts (dual-read migration).

    Returns ``(read_root, legacy_colocated)``. When ``remap_manifest.json`` is
    missing, falls back to schedule/group/exact_cache colocated under the legacy
    ``templates/`` store with a warning.
    """
    remap_path = Path(remap_store)
    templates_path = Path(templates_store)
    manifest = remap_path / REMAP_MANIFEST_NAME
    if manifest.is_file():
        return remap_path, False
    legacy_schedule = templates_path / "shift_schedule.npz"
    if legacy_schedule.is_file():
        log.warning(
            "remap_manifest missing at %s; reading schedule/groups/exact_cache "
            "from legacy colocated templates store %s",
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


def exact_cache_dir_for_read_root(read_root: str | Path) -> Path:
    return Path(read_root) / EXACT_CACHE_DIRNAME


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
    if not ref_path.is_file():
        raise FileNotFoundError(f"reference_ffi_path missing: {ref_path}")
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
    for p in paths:
        logical = manifest_basename_from_local(p)
        if logical in ffi_list_df.index:
            try:
                frame_wcs.append((logical, wcs_from_cached_row(ffi_list_df.loc[logical])))
            except Exception as exc:
                log.warning("WCS reconstruct failed for %s: %s", logical, exc)
                frame_wcs.append((logical, None))
        else:
            frame_wcs.append((logical, None))
        row = ffi_list_df.loc[logical] if logical in ffi_list_df.index else None
        if row is not None and pd.notna(row.get("date_obs")):
            try:
                from astropy.time import Time

                btjd_list.append(float(Time(str(row["date_obs"]), format="isot", scale="utc").btjd))
            except Exception:
                btjd_list.append(float("nan"))
        else:
            btjd_list.append(float("nan"))

    btjd = np.asarray(btjd_list, dtype=np.float64)
    schedule = build_skycell_shift_schedule(
        frame_wcs, skycell_df, ref_wcs, btjd=btjd
    )
    schedule.meta = dict(schedule.meta or {})
    schedule.meta["source"] = "built_from_scc_ffis"
    schedule.meta["reference_ffi"] = str(ref_path.resolve())
    schedule.meta["frame_filenames"] = [manifest_basename_from_local(p) for p in paths]
    with field_store_lock(store_root):
        schedule.save(store_root / "shift_schedule.npz")
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
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"reference_ffi_path missing or not a file: {ref_path} (from {job_path})"
        )
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
    for p in paths:
        logical = manifest_basename_from_local(p)
        if logical in ffi_list_df.index:
            try:
                frame_wcs.append((logical, wcs_from_cached_row(ffi_list_df.loc[logical])))
            except Exception as exc:
                log.warning("WCS reconstruct failed for %s: %s", logical, exc)
                frame_wcs.append((logical, None))
        else:
            frame_wcs.append((logical, None))

    btjd = None
    if "btjd" in frames_df.columns:
        btjd = frames_df["btjd"].to_numpy(dtype=np.float64)

    schedule = build_skycell_shift_schedule(
        frame_wcs, skycell_df, ref_wcs, btjd=btjd
    )
    schedule.meta = dict(schedule.meta or {})
    schedule.meta["source"] = "built_from_ffi_list"
    schedule.meta["reference_ffi"] = str(ref_path)
    with field_store_lock(store_root):
        schedule.save(store_root / "shift_schedule.npz")
        shutil.copy2(store_root / "shift_schedule.npz", event_dir / "shift_schedule.npz")
        store_json = store_root / "shift_schedule.json"
        if store_json.is_file():
            shutil.copy2(store_json, event_dir / "shift_schedule.json")
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
    )


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


def _write_remap_manifest(
    store: Path,
    *,
    oversampling_factor: int,
    apply_hybrid_exact: bool,
    hybrid_R: int,
    include_abutting_border_exact: bool,
    cache_quantum_ps1_px: float,
    keying: str,
    n_exact_keys: int,
    n_groups: int,
    reference_ffi: str | None,
) -> None:
    payload = {
        "schema_version": REMAP_SCHEMA_VERSION,
        "geometry_mode": "field",
        "oversampling_factor": int(oversampling_factor),
        "apply_hybrid_exact": bool(apply_hybrid_exact),
        "hybrid_R": int(hybrid_R),
        "include_abutting_border_exact": bool(include_abutting_border_exact),
        "cache_quantum_ps1_px": float(cache_quantum_ps1_px),
        "keying": str(keying),
        "n_exact_keys": int(n_exact_keys),
        "n_groups": int(n_groups),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
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
    apply_hybrid_exact: bool = True,
    hybrid_R: int = 1,
    include_abutting_border_exact: bool = True,
    rebuild_remap_cache: bool = False,
    store_root: str | Path | None = None,
    scc_only: bool = False,
    ffi_dir: str | Path | None = None,
    ref_ffi_path: str | Path | None = None,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Build or reuse SCC remap artifacts (L2–L4).

    Writes under ``remap/oversampling_{N}/``: shift schedule, group artifacts,
    optional ``exact_cache/``, and ``remap_manifest.json``.
    """
    import os as _os
    from joblib import Parallel, delayed

    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        abutting_border_tess_ids,
        candidate_tess_ids_for_l4a,
        exact_regmap_for_tess_ids,
    )

    event_dir = Path(event_dir)
    data_root = Path(data_root)
    mapping_root = Path(mapping_root)
    store = Path(store_root) if store_root is not None else remap_root(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    store.mkdir(parents=True, exist_ok=True)
    exact_cache_dir = store / EXACT_CACHE_DIRNAME
    if apply_hybrid_exact:
        exact_cache_dir.mkdir(exist_ok=True)

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
    )
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

    keys = {
        (str(r.skycell), int(r.sx_int), int(r.sy_int))
        for r in assignment.shifts_df.itertuples(index=False)
    }
    n_exact_keys = sum(
        1 for s, x, y in keys if apply_hybrid_exact and not (x == 0 and y == 0)
    )
    n_exact_written = 0
    if apply_hybrid_exact and n_exact_keys:
        master_arr = None
        name_to_id: dict[str, int] = {}
        if include_abutting_border_exact:
            master_path = _master_pixels2skycells_path(
                mapping_root,
                sector,
                camera,
                ccd,
                oversampling_factor=oversampling_factor,
            )
            master_arr, name_to_id = _master_skycell_id_map(master_path)

        if scc_only:
            from syndiff_pipeline.common.wcs_header_cache import load_ffi_list
            from syndiff_pipeline.common.scc_paths import scc_ffi_list_parquet

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

        skycell_row_cache: dict[str, pd.Series] = {}

        def _skycell_row(skycell: str) -> pd.Series:
            if skycell not in skycell_row_cache:
                skycell_row_cache[skycell] = _skycell_catalog_row(
                    mapping_root,
                    sector,
                    camera,
                    ccd,
                    skycell,
                    oversampling_factor=oversampling_factor,
                )
            return skycell_row_cache[skycell]

        def _one_exact(skycell: str, sx_i: int, sy_i: int) -> str:
            if int(sx_i) == 0 and int(sy_i) == 0:
                return "skip"
            cache_name = contrib_basename(skycell, sx_i, sy_i).replace(".npz", "_exact.npz")
            cache_path = exact_cache_dir / cache_name
            if cache_path.is_file() and not rebuild_remap_cache:
                return "skip"
            if cache_path.is_file() and rebuild_remap_cache:
                cache_path.unlink()
            frame_i = _frame_index_for_shift(schedule, skycell, sx_i, sy_i)
            if frame_i is None:
                log.warning(
                    "No frame WCS for exact cache %s sx=%+d sy=%+d; skipping",
                    skycell,
                    sx_i,
                    sy_i,
                )
                return "skip"
            try:
                tess_wcs = _frame_wcs_at(frame_i)
                extra = None
                if include_abutting_border_exact and master_arr is not None:
                    sid = name_to_id.get(str(skycell))
                    if sid is not None:
                        extra = abutting_border_tess_ids(master_arr, sid)
                with fits.open(
                    _find_regmap(
                        mapping_root,
                        sector,
                        camera,
                        ccd,
                        skycell,
                        oversampling_factor=oversampling_factor,
                    )
                ) as hdul:
                    if "TESS_PIXEL_MAP" in hdul:
                        assignment_map = np.asarray(hdul["TESS_PIXEL_MAP"].data)
                    else:
                        assignment_map = np.asarray(hdul[1].data)
                tids, mask = candidate_tess_ids_for_l4a(
                    assignment_map,
                    sx_i,
                    sy_i,
                    hybrid_R=int(hybrid_R),
                    extra_tess_ids=extra,
                )
                if tids.size == 0 or int(mask.sum()) == 0:
                    return "skip"
                exact = exact_regmap_for_tess_ids(
                    tess_wcs,
                    _skycell_row(skycell),
                    tids,
                    data_shape=base_tess_shape,
                    oversampling_factor=oversampling_factor,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, exact_tid=exact.astype(np.int32))
                return "write"
            except Exception as exc:
                log.warning(
                    "Exact cache failed for %s sx=%+d sy=%+d (%s)",
                    skycell,
                    sx_i,
                    sy_i,
                    exc,
                )
                return "fail"

        key_list = sorted(keys)
        n_jobs_eff = max(1, min(int(n_jobs), n_exact_keys))
        hybrid_cap = int(_os.environ.get("SYNDIFF_HYBRID_MAX_JOBS", "24"))
        avail = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (
            _os.cpu_count() or hybrid_cap
        )
        n_jobs_eff = min(n_jobs_eff, max(1, hybrid_cap), max(1, avail))
        if n_jobs_eff == 1 or n_exact_keys <= 1:
            exact_statuses = [_one_exact(s, x, y) for s, x, y in key_list]
        else:
            exact_statuses = Parallel(n_jobs=n_jobs_eff, prefer="processes")(
                delayed(_one_exact)(s, x, y) for s, x, y in key_list
            )
        n_exact_written = sum(1 for s in exact_statuses if s == "write")
        log.info(
            "Exact cache: %d keys, %d written",
            n_exact_keys,
            n_exact_written,
        )

    ref_ffi = None
    if schedule.meta:
        ref_ffi = schedule.meta.get("reference_ffi")
    _write_remap_manifest(
        store,
        oversampling_factor=oversampling_factor,
        apply_hybrid_exact=apply_hybrid_exact,
        hybrid_R=hybrid_R,
        include_abutting_border_exact=include_abutting_border_exact,
        cache_quantum_ps1_px=cache_quantum_ps1_px,
        keying=keying,
        n_exact_keys=n_exact_keys,
        n_groups=len(assignment.groups),
        reference_ffi=str(ref_ffi) if ref_ffi else None,
    )

    return {
        "output_dir": str(store),
        "n_groups": len(assignment.groups),
        "n_exact_keys": n_exact_keys,
        "n_exact_written": n_exact_written,
        "geometry_mode": "field",
        "apply_hybrid_exact": bool(apply_hybrid_exact),
        "hybrid_R": int(hybrid_R),
        "rebuild_remap_cache": bool(rebuild_remap_cache),
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
