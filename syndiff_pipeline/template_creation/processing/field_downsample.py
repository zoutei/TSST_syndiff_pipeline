"""Field-mode downsample: SCC ``field_templates/`` schedule + sparse contribs.

Builds (or reuses) a per-SCC shift schedule, writes ``template_group_shifts``,
bins unique ``(skycell, sx, sy)`` contribs with hybrid Exact L4a (R=1) plus
optional abutting-border Exact (L4b-lite), and updates the event frames CSV
``group_id`` from the signature schedule. Set ``apply_hybrid_exact=False`` to
fall back to frozen-regmap + integer PS1 data roll.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    assemble_group_from_contribs,
    contrib_path,
    field_store_lock,
    field_templates_root,
    verify_field_store,
    write_contrib,
    write_template_manifest,
)
from syndiff_pipeline.template_creation.processing.shift_schedule import (
    ShiftSchedule,
    assign_groups_from_schedule,
    write_group_artifacts,
)

log = logging.getLogger(__name__)


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
    from syndiff_pipeline.common.wcs_grouping import open_fits_memmap
    from syndiff_pipeline.common.wcs_header_cache import (
        load_or_build_wcs_cache,
        wcs_cache_path,
        wcs_from_cached_row,
    )
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        RELEVANT_WCS_KEYS,
        load_tess_wcs,
    )
    from syndiff_pipeline.template_creation.processing.shift_schedule import (
        build_skycell_shift_schedule,
    )

    frames_path = event_dir / "syndiff_ffi_frames.csv"
    if not frames_path.is_file():
        raise FileNotFoundError(f"Cannot build shift schedule without {frames_path}")
    frames_df = pd.read_csv(frames_path)
    if "path" not in frames_df.columns:
        raise KeyError(f"{frames_path} missing 'path' column")

    job_path = event_dir / "cluster_template_job.json"
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
    cache = load_or_build_wcs_cache(
        paths,
        wcs_cache_path(data_root, sector, camera, ccd),
        open_fits=open_fits_memmap,
    )
    frame_wcs: list[tuple[str, Any]] = []
    for p in paths:
        name = p.name
        if name in cache.index:
            try:
                frame_wcs.append((name, wcs_from_cached_row(cache.loc[name])))
            except Exception as exc:
                log.warning("WCS reconstruct failed for %s: %s", name, exc)
                frame_wcs.append((name, None))
        else:
            frame_wcs.append((name, None))

    btjd = None
    if "btjd" in frames_df.columns:
        btjd = frames_df["btjd"].to_numpy(dtype=np.float64)

    schedule = build_skycell_shift_schedule(
        frame_wcs, skycell_df, ref_wcs, btjd=btjd
    )
    schedule.meta = dict(schedule.meta or {})
    schedule.meta["source"] = "built_from_wcs_cache"
    schedule.meta["reference_ffi"] = str(ref_path)
    with field_store_lock(store_root):
        schedule.save(store_root / "shift_schedule.npz")
        shutil.copy2(store_root / "shift_schedule.npz", event_dir / "shift_schedule.npz")
        store_json = store_root / "shift_schedule.json"
        if store_json.is_file():
            shutil.copy2(store_json, event_dir / "shift_schedule.json")
    log.info(
        "Built shift_schedule.npz (%d frames × %d skycells) for field downsample",
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
) -> ShiftSchedule:
    existing = _load_or_copy_shift_schedule(event_dir, store_root)
    if existing is not None:
        return existing
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


def _update_frames_group_ids(event_dir: Path, group_id_per_frame: np.ndarray) -> None:
    frames_path = event_dir / "syndiff_ffi_frames.csv"
    frames = pd.read_csv(frames_path)
    n = min(len(frames), len(group_id_per_frame))
    if "group_id" not in frames.columns:
        frames["group_id"] = -1
    # Positional assignment (label-based .loc slicing is fragile if the CSV
    # ever carries a non-default index or n == 0).
    col = frames.columns.get_loc("group_id")
    if n:
        frames.iloc[:n, col] = np.asarray(group_id_per_frame[:n], dtype=np.int64)
    frames.to_csv(frames_path, index=False)


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
    scc_tail = Path(f"sector_{int(sector):04d}") / f"camera_{int(camera)}" / f"ccd_{int(ccd)}"
    # Caller may already pass the SCC directory.
    if root.name == f"ccd_{int(ccd)}" and (root / f"tess_s{int(sector):04d}_{camera}_{ccd}_master_pixels2skycells.fits.gz").is_file():
        return root
    if root.name == f"ccd_{int(ccd)}" and any(root.glob("tess_s*_master_pixels2skycells*.fits.gz")):
        return root
    if int(oversampling_factor) > 1:
        root = root / f"oversampling_{int(oversampling_factor)}"
    return root / scc_tail


def _find_regmap(
    mapping_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    skycell: str,
    *,
    oversampling_factor: int = 1,
) -> Path:
    """Locate one skycell regmap; never cross oversampling trees.

    Per-skycell files use unpadded sector (``tess_s20_...``), matching
    ``pancakes.py``. Master maps use zero-padded sector.
    """
    scc = _mapping_scc_dir(
        mapping_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    name = str(skycell).strip()
    if not name.startswith("skycell."):
        name = f"skycell.{name}"
    suffix = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    candidates = [
        scc / f"tess_s{int(sector)}_{camera}_{ccd}_{name}{suffix}.fits.gz",
        scc / f"tess_s{int(sector):04d}_{camera}_{ccd}_{name}{suffix}.fits.gz",
        scc / f"tess_s{int(sector)}_{camera}_{ccd}_{name}{suffix}.fits",
        scc / f"tess_s{int(sector):04d}_{camera}_{ccd}_{name}{suffix}.fits",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(p for p in scc.glob(f"*_{name}{suffix}.fits*") if p.is_file())
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No regmap for {name} under {scc} (oversampling_factor={oversampling_factor})"
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
    path = scc / f"tess_s{int(sector):04d}_{camera}_{ccd}_master_pixels2skycells{suffix}.fits.gz"
    if path.is_file():
        return path
    matches = sorted(scc.glob("tess_*_master_pixels2skycells*.fits.gz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"master pixels2skycells not found under {scc}")


def _bin_skycell_contrib(
    *,
    assignment: np.ndarray,
    ps1_data: np.ndarray,
    ps1_mask: np.ndarray,
    sx_int: int,
    sy_int: int,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    ignore_mask: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Sparse bin one skycell contribution.

    Production linear convention: ``assignment`` is frozen and PS1 data/mask are
    rolled by ``(+sy, +sx)``. Hybrid Exact convention: pass an already-shifted
    hybrid ``assignment`` with ``sx_int=sy_int=0`` (unshifted PS1 data).
    """
    from syndiff_pipeline.template_creation.processing.downsample import (
        _aggregate_sorted_groups,
    )

    if assignment.shape != ps1_data.shape:
        raise ValueError(
            f"regmap assignment shape {assignment.shape} != PS1 data shape {ps1_data.shape}"
        )
    t_y, t_x = base_tess_shape
    x_min, y_min, x_max, y_max = roi_bounds
    pind = assignment.ravel()
    sort_ind = np.argsort(pind)
    tess_pixels = np.unique(pind[np.isfinite(pind)]).astype(int)
    tess_pixels = tess_pixels[tess_pixels >= 0]
    if len(tess_pixels) == 0:
        return None
    breaks = np.where(np.diff(pind[sort_ind]) > 0)[0] + 1
    breaks = np.append(breaks, len(sort_ind))
    group_starts = breaks[:-1]

    if int(sx_int) == 0 and int(sy_int) == 0:
        ps1_shifted = ps1_data
        mask_shifted = ps1_mask
    else:
        ps1_shifted = np.roll(ps1_data, (int(sy_int), int(sx_int)), axis=(0, 1))
        mask_shifted = np.roll(ps1_mask, (int(sy_int), int(sx_int)), axis=(0, 1))
    ps1_rav = np.asarray(ps1_shifted).ravel()[sort_ind]
    mask_rav = np.asarray(mask_shifted).ravel()[sort_ind]
    sums, counts, mask_counts = _aggregate_sorted_groups(
        ps1_rav, mask_rav, group_starts, ignore_mask
    )
    y_base = tess_pixels // t_x
    x_base = tess_pixels % t_x
    valid = (
        (0 <= y_base)
        & (y_base < t_y)
        & (0 <= x_base)
        & (x_base < t_x)
        & (x_base >= x_min)
        & (x_base < x_max)
        & (y_base >= y_min)
        & (y_base < y_max)
    )
    if not np.any(valid):
        return None
    return (
        tess_pixels[valid].astype(np.int64),
        sums[valid].astype(np.float64),
        counts[valid].astype(np.float64),
        mask_counts[valid].astype(np.float64),
    )


def _frame_index_for_shift(
    schedule: ShiftSchedule,
    skycell: str,
    sx_int: int,
    sy_int: int,
) -> int | None:
    """First valid frame whose schedule matches ``(skycell, sx, sy)``."""
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


def _load_zarr_skycell(zstore, skycell: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(zstore[f"{skycell}_data"][:], dtype=np.float32)
    try:
        mask = np.asarray(zstore[f"{skycell}_mask"][:])
    except Exception:
        mask = np.zeros(data.shape, dtype=np.int32)
    return data, mask


def _skycells_in_crop(
    master_path: Path,
    roi_bounds: tuple[int, int, int, int],
    name_by_idx: dict[int, str] | None = None,
) -> set[str]:
    """Skycell names whose exclusive TESS pixels intersect the crop ROI."""
    x0, y0, x1, y1 = (int(v) for v in roi_bounds)
    with fits.open(master_path) as hdul:
        master = np.asarray(hdul[1].data)
        if name_by_idx is None and len(hdul) > 2:
            tab = hdul[2].data
            name_by_idx = {
                int(i): str(n).strip() for n, i in zip(tab["SKYCELL"], tab["SKYCIND"])
            }
    crop = master[y0:y1, x0:x1]
    ids = np.unique(crop)
    ids = ids[ids >= 0]
    if not name_by_idx:
        return {str(i) for i in ids}
    return {name_by_idx[int(i)] for i in ids if int(i) in name_by_idx}


def run_field_downsample_scc(
    *,
    sector: int,
    camera: int,
    ccd: int,
    data_root: str | Path,
    event_dir: str | Path,
    mapping_root: str | Path,
    convolved_dir: str | Path,
    roi_bounds: tuple[int, int, int, int],
    base_tess_shape: tuple[int, int],
    oversampling_factor: int = 1,
    ignore_mask_bits: list[int] | None = None,
    grouping_quantum_ps1_px: float = 1.0,
    materialize_fits: bool = False,
    n_jobs: int = 1,
    update_frames_csv: bool = True,
    crop_filter_skycells: bool = True,
    store_root: str | Path | None = None,
    apply_hybrid_exact: bool = True,
    hybrid_R: int = 1,
    include_abutting_border_exact: bool = True,
    rebuild_field_store: bool = False,
) -> dict[str, Any]:
    """
    Build/reuse the SCC field store and point the event at it.

    Returns a result dict with ``output_dir`` = SCC store root.

    Parameters
    ----------
    crop_filter_skycells
        If True (default), only build contribs for skycells intersecting
        ``roi_bounds``.
    store_root
        Optional override for the field store path (tests/smoke).
    apply_hybrid_exact
        If True (default), Exact-patch the L4a R=1 seam/rim under a frame WCS
        that realizes each ``(skycell, sx, sy)``, then bin with hybrid
        assignment (unshifted PS1). If False, use frozen map + data roll.
    hybrid_R
        Dilation radius for the L4a recompute mask (design default 1).
    include_abutting_border_exact
        Expand Exact TESS-id set with this skycell's master abutting-border
        pixels (L4b-lite / neighbour-rim under the Type-I realizing frame WCS).
    rebuild_field_store
        If True, overwrite existing contrib NPZs (and Exact caches for those
        keys). Default False preserves the shared SCC store across events /
        force-reruns.
    """
    import zarr
    from joblib import Parallel, delayed

    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        abutting_border_tess_ids,
        build_hybrid_assignment_with_exact,
    )
    from syndiff_pipeline.template_creation.processing.field_templates import (
        contrib_basename,
    )

    event_dir = Path(event_dir)
    data_root = Path(data_root)
    mapping_root = Path(mapping_root)
    store = Path(store_root) if store_root is not None else field_templates_root(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    store.mkdir(parents=True, exist_ok=True)
    (store / "contribs").mkdir(exist_ok=True)
    exact_cache_dir = store / "exact_cache"
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
    )
    assignment = assign_groups_from_schedule(
        schedule,
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=1.0,
        keying="absolute",
    )
    write_group_artifacts(
        assignment,
        store,
        geometry_mode="field",
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=1.0,
    )
    write_group_artifacts(
        assignment,
        event_dir,
        geometry_mode="field",
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=1.0,
    )
    if update_frames_csv:
        _update_frames_group_ids(event_dir, assignment.group_id_per_frame)

    ignore_mask = 0
    for bit in ignore_mask_bits or [12]:
        ignore_mask |= 1 << int(bit)

    zarr_path = Path(convolved_dir) / f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    if not zarr_path.exists():
        alt = list(Path(convolved_dir).glob(f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}*.zarr"))
        if not alt:
            raise FileNotFoundError(f"convolved zarr not found under {convolved_dir}")
        zarr_path = alt[0]
    zarr.open(str(zarr_path), mode="r")

    keys = {
        (str(r.skycell), int(r.sx_int), int(r.sy_int))
        for r in assignment.shifts_df.itertuples(index=False)
    }
    master_path = _master_pixels2skycells_path(
        mapping_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
    )
    master_arr = None
    name_to_id: dict[str, int] = {}
    if crop_filter_skycells or (apply_hybrid_exact and include_abutting_border_exact):
        master_arr, name_to_id = _master_skycell_id_map(master_path)
    if crop_filter_skycells:
        allowed = _skycells_in_crop(master_path, roi_bounds)
        before = len(keys)
        keys = {k for k in keys if k[0] in allowed}
        log.info(
            "Crop-filter skycells: %d -> %d contrib keys (%d skycells in ROI)",
            before,
            len(keys),
            len(allowed),
        )

    frames_df = pd.read_csv(event_dir / "syndiff_ffi_frames.csv")
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

    def _one(skycell: str, sx_i: int, sy_i: int) -> str:
        out = contrib_path(store, skycell, sx_i, sy_i)
        if out.is_file() and not rebuild_field_store:
            return "skip"
        if out.is_file() and rebuild_field_store:
            out.unlink()
        reg = _find_regmap(
            mapping_root,
            sector,
            camera,
            ccd,
            skycell,
            oversampling_factor=oversampling_factor,
        )
        with fits.open(reg) as hdul:
            if "TESS_PIXEL_MAP" in hdul:
                assignment_map = np.asarray(hdul["TESS_PIXEL_MAP"].data)
            else:
                assignment_map = np.asarray(hdul[1].data)
        zs = zarr.open(str(zarr_path), mode="r")
        ps1_data, ps1_mask = _load_zarr_skycell(zs, skycell)

        use_hybrid = bool(apply_hybrid_exact) and not (int(sx_i) == 0 and int(sy_i) == 0)
        binned = None
        if use_hybrid:
            frame_i = _frame_index_for_shift(schedule, skycell, sx_i, sy_i)
            if frame_i is None:
                log.warning(
                    "No frame WCS for %s sx=%+d sy=%+d; falling back to data-roll",
                    skycell,
                    sx_i,
                    sy_i,
                )
                use_hybrid = False
            else:
                try:
                    tess_wcs = _load_frame_wcs(frames_df, frame_i)
                    extra = None
                    if include_abutting_border_exact and master_arr is not None:
                        sid = name_to_id.get(str(skycell))
                        if sid is not None:
                            extra = abutting_border_tess_ids(master_arr, sid)
                    cache_name = contrib_basename(skycell, sx_i, sy_i).replace(
                        ".npz", "_exact.npz"
                    )
                    cache_path = exact_cache_dir / cache_name
                    if rebuild_field_store and cache_path.is_file():
                        cache_path.unlink()
                    hybrid_map, meta = build_hybrid_assignment_with_exact(
                        assignment_map,
                        sx_i,
                        sy_i,
                        tess_wcs,
                        _skycell_row(skycell),
                        data_shape=base_tess_shape,
                        hybrid_R=int(hybrid_R),
                        oversampling_factor=oversampling_factor,
                        extra_tess_ids=extra,
                        exact_cache_path=cache_path,
                    )
                    log.debug(
                        "L4a hybrid %s sx=%+d sy=%+d mask=%s tids=%s cache=%s",
                        skycell,
                        sx_i,
                        sy_i,
                        meta.get("n_mask"),
                        meta.get("n_tess_ids"),
                        meta.get("cache_hit"),
                    )
                    binned = _bin_skycell_contrib(
                        assignment=hybrid_map,
                        ps1_data=ps1_data,
                        ps1_mask=ps1_mask,
                        sx_int=0,
                        sy_int=0,
                        base_tess_shape=base_tess_shape,
                        roi_bounds=roi_bounds,
                        ignore_mask=ignore_mask,
                    )
                except Exception as exc:
                    log.warning(
                        "L4a Exact failed for %s sx=%+d sy=%+d (%s); data-roll fallback",
                        skycell,
                        sx_i,
                        sy_i,
                        exc,
                    )
                    use_hybrid = False

        if not use_hybrid:
            binned = _bin_skycell_contrib(
                assignment=assignment_map,
                ps1_data=ps1_data,
                ps1_mask=ps1_mask,
                sx_int=sx_i,
                sy_int=sy_i,
                base_tess_shape=base_tess_shape,
                roi_bounds=roi_bounds,
                ignore_mask=ignore_mask,
            )
        if binned is None:
            write_contrib(
                store,
                skycell,
                sx_i,
                sy_i,
                indices=np.array([], dtype=np.int64),
                flux_sum=np.array([], dtype=np.float64),
                count=np.array([], dtype=np.float64),
                mask_count=np.array([], dtype=np.float64),
            )
        else:
            idx, sums, counts, mcounts = binned
            write_contrib(
                store,
                skycell,
                sx_i,
                sy_i,
                indices=idx,
                flux_sum=sums,
                count=counts,
                mask_count=mcounts,
            )
        return "write"

    key_list = sorted(keys)
    n_jobs_eff = max(1, min(int(n_jobs), len(key_list) or 1))
    # Hybrid Exact workers each hold ~2 GB (frozen regmap + full-chip tpix +
    # process_skycell_pixel_mapping scratch). Cap parallelism to keep memory
    # bounded, but 4 badly under-utilizes large multi-core hosts; allow up to
    # HYBRID_MAX_JOBS (override with SYNDIFF_HYBRID_MAX_JOBS) so a 1024-box crop
    # (~4k keys) finishes in tens of minutes instead of hours.
    if apply_hybrid_exact:
        import os as _os

        hybrid_cap = int(_os.environ.get("SYNDIFF_HYBRID_MAX_JOBS", "24"))
        n_jobs_eff = min(n_jobs_eff, max(1, hybrid_cap))
    if n_jobs_eff == 1 or len(key_list) <= 1:
        statuses = [_one(s, x, y) for s, x, y in key_list]
    else:
        statuses = Parallel(n_jobs=n_jobs_eff, prefer="processes")(
            delayed(_one)(s, x, y) for s, x, y in key_list
        )
    n_written = sum(1 for s in statuses if s == "write")
    n_skipped = sum(1 for s in statuses if s == "skip")

    write_template_manifest(
        store,
        FieldManifest(
            geometry_mode="field",
            scope="scc",
            assembly="sparse_sum",
            materialize_fits=bool(materialize_fits),
            sector=int(sector),
            camera=int(camera),
            ccd=int(ccd),
            contribs_dir="contribs",
            groups=list(assignment.groups),
        ),
    )
    sidecar = {
        "schema_version": 1,
        "store_root": str(store),
        "zarr_path": str(zarr_path),
        "base_tess_shape": list(base_tess_shape),
        "roi_bounds": list(roi_bounds),
        "oversampling_factor": int(oversampling_factor),
        "ignore_mask": int(ignore_mask),
        "apply_hybrid_exact": bool(apply_hybrid_exact),
        "hybrid_R": int(hybrid_R),
        "include_abutting_border_exact": bool(include_abutting_border_exact),
        "l4b_policy": "abutting_under_type1_wcs",
        "flux_note": (
            "Field contribs are in convolved/PS1 flux units; Hotpants may need "
            "a per-event flux scale vs linear ADU templates (~1e3–1e4)."
        ),
    }
    (store / "field_mode_assembly.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    v = verify_field_store(
        store,
        required_keys=list(keys),
        require_nonempty=False,
    )
    if not v["ok"]:
        raise RuntimeError(f"field store incomplete: {v['reasons']}")
    # Content gate: at least one required key must carry flux (crop-edge keys
    # may legitimately be empty NPZs).
    nonempty = 0
    for skycell, sx_i, sy_i in keys:
        p = contrib_path(store, skycell, sx_i, sy_i)
        if not p.is_file():
            continue
        from syndiff_pipeline.template_creation.processing.field_templates import (
            load_contrib,
        )

        data = load_contrib(p)
        if len(np.asarray(data["indices"])) > 0:
            nonempty += 1
    if nonempty == 0 and len(keys) > 0:
        raise RuntimeError(
            f"field store has {len(keys)} contrib keys but all are empty "
            f"(regmap/ROI mismatch?)"
        )

    # Per-event completeness marker: the SCC store is shared across events, so
    # each event records exactly the crop-filtered keys IT required. Written
    # only after the store verify passes, so its presence means this event's
    # field downsample completed. verify_downsample_field_mode checks against
    # this (not the full-chip template_group_shifts, which would false-fail a
    # cropped run). See verify.py::verify_downsample_field_mode.
    (event_dir / "field_contrib_keys.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "store_root": str(store),
                "n_contrib_keys": len(keys),
                "keys": [[str(s), int(x), int(y)] for s, x, y in key_list],
            }
        )
        + "\n"
    )

    return {
        "output_dir": str(store),
        "n_groups": len(assignment.groups),
        "n_contrib_keys": len(keys),
        "n_contribs_written": n_written,
        "n_contribs_skipped": n_skipped,
        "geometry_mode": "field",
        "apply_hybrid_exact": bool(apply_hybrid_exact),
        "hybrid_R": int(hybrid_R),
        "rebuild_field_store": bool(rebuild_field_store),
    }



def _group_shifts_present(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    group_id: int,
    *,
    present_only: bool,
) -> list[tuple[str, int, int]]:
    """The group's ``(skycell, sx, sy)`` shifts, optionally restricted to keys
    whose contrib NPZ exists (cropped stores only materialize their ROI skycells)."""
    rows = shifts_df.loc[shifts_df["group_id"] == int(group_id)]
    if rows.empty:
        raise KeyError(f"group_id={group_id} not in template_group_shifts")
    shifts = [
        (str(r.skycell), int(r.sx_int), int(r.sy_int))
        for r in rows.itertuples(index=False)
    ]
    if present_only:
        shifts = [
            (s, x, y)
            for (s, x, y) in shifts
            if contrib_path(store_root, s, x, y).is_file()
        ]
        if not shifts:
            raise FileNotFoundError(
                f"No materialized contribs for group_id={group_id} under {store_root}"
            )
    return shifts


def assemble_field_group_flux(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    group_id: int,
    *,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    present_only: bool | None = None,
) -> np.ndarray:
    """Assemble mean flux for one group_id (optionally cropped).

    ``present_only`` restricts to keys whose contrib NPZ exists; it defaults to
    ``crop is not None`` so a cropped assemble tolerates a crop-only store, while
    a full-FFI assemble (``crop=None``) requires every key by default. Pass
    ``present_only=True`` to assemble a full-FFI template from a cropped store
    (missing skycells simply stay zero).
    """
    if present_only is None:
        present_only = crop is not None
    shifts = _group_shifts_present(
        store_root, shifts_df, group_id, present_only=present_only
    )
    out = assemble_group_from_contribs(store_root, shifts, shape=shape, crop=crop)
    flux = out["flux_sum"]
    count = out["count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, flux / count, 0.0)
    return mean.astype(np.float64)


def assemble_field_group_count(
    store_root: str | Path,
    shifts_df: pd.DataFrame,
    group_id: int,
    *,
    shape: tuple[int, int],
    crop: tuple[int, int, int, int] | None = None,
    present_only: bool | None = None,
) -> np.ndarray:
    """Assemble the per-TESS-pixel PS1 hit COUNT for one group_id (optionally cropped).

    This is the same COUNT plane a linear template FITS carries, used by the
    ``shared_mask`` PS1-coverage mask (``COUNT < ps1_min_hit_count``).
    """
    if present_only is None:
        present_only = crop is not None
    shifts = _group_shifts_present(
        store_root, shifts_df, group_id, present_only=present_only
    )
    out = assemble_group_from_contribs(store_root, shifts, shape=shape, crop=crop)
    return np.asarray(out["count"], dtype=np.float64)
