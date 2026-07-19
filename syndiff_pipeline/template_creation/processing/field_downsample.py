"""Field-mode downsample (L5): bin sparse contribs from remap artifacts.

Reads shift schedule, group artifacts, and optional ``exact_cache/`` from
``remap/oversampling_{N}/`` (or legacy colocated ``templates/`` during
migration) and writes ``contribs/`` under ``templates/oversampling_{N}/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.wcs_grouping import _frames_csv_path
from syndiff_pipeline.template_creation.processing.field_remap import (
    REMAP_MANIFEST_NAME,
    _find_regmap,
    _mapping_scc_dir,
    _master_pixels2skycells_path,
    exact_cache_dir_for_read_root,
    load_remap_shifts_df,
    remap_root,
    resolve_remap_read_root,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    assemble_group_from_contribs,
    contrib_basename,
    contrib_path,
    templates_root,
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

# Re-export mapping helpers for existing tests/callers.
__all__ = [
    "_find_regmap",
    "_mapping_scc_dir",
    "_master_pixels2skycells_path",
    "assemble_field_group_count",
    "assemble_field_group_flux",
    "run_field_downsample_scc",
]


def _load_remap_manifest(read_root: Path) -> dict[str, Any]:
    path = read_root / REMAP_MANIFEST_NAME
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _update_frames_group_ids(event_dir: Path, group_id_per_frame: np.ndarray) -> None:
    frames_path = Path(_frames_csv_path(event_dir))
    frames = pd.read_csv(frames_path)
    n = min(len(frames), len(group_id_per_frame))
    if "group_id" not in frames.columns:
        frames["group_id"] = -1
    col = frames.columns.get_loc("group_id")
    if n:
        frames.iloc[:n, col] = np.asarray(group_id_per_frame[:n], dtype=np.int64)
    frames.to_csv(frames_path, index=False)


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
    """Sparse bin one skycell contribution."""
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
    cache_quantum_ps1_px: float = 1.0,
    keying: str = "absolute",
    materialize_fits: bool = False,
    n_jobs: int = 1,
    update_frames_csv: bool = True,
    crop_filter_skycells: bool = True,
    store_root: str | Path | None = None,
    remap_store_root: str | Path | None = None,
    apply_hybrid_exact: bool = True,
    hybrid_R: int = 1,
    include_abutting_border_exact: bool = True,
    rebuild_field_store: bool = False,
    stage_regmaps_to_scratch: bool | None = None,
    scc_only: bool = False,
    ffi_dir: str | Path | None = None,
    ref_ffi_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Bin sparse contribs into the SCC templates store (L5 only).

    Requires remap artifacts under ``remap/oversampling_{N}/`` (or legacy
    colocated schedule/cache under ``templates/``). Does not compute Exact
    remaps; reads ``exact_cache/`` and merges when ``apply_hybrid_exact`` is set.

    Parameters ``ffi_dir`` and ``ref_ffi_path`` are accepted for dispatch
    compatibility but ignored (remap must run separately).
    """
    import os as _os
    import zarr
    from joblib import Parallel, delayed

    from syndiff_pipeline.template_creation.processing.field_hybrid_exact import (
        hybrid_assignment_from_exact_cache,
    )

    del ffi_dir, ref_ffi_path  # remap stage owns schedule build inputs

    event_dir = Path(event_dir)
    data_root = Path(data_root)
    mapping_root = Path(mapping_root)
    store = Path(store_root) if store_root is not None else templates_root(
        data_root, sector, camera, ccd, oversampling_factor=oversampling_factor
    )
    remap_store = (
        Path(remap_store_root)
        if remap_store_root is not None
        else remap_root(data_root, sector, camera, ccd, oversampling_factor=oversampling_factor)
    )
    store.mkdir(parents=True, exist_ok=True)
    (store / "contribs").mkdir(exist_ok=True)

    remap_read, _legacy = resolve_remap_read_root(remap_store, store)
    remap_manifest = _load_remap_manifest(remap_read)
    if remap_manifest:
        apply_hybrid_exact = bool(
            remap_manifest.get("apply_hybrid_exact", apply_hybrid_exact)
        )
        hybrid_R = int(remap_manifest.get("hybrid_R", hybrid_R))
        include_abutting_border_exact = bool(
            remap_manifest.get("include_abutting_border_exact", include_abutting_border_exact)
        )
        cache_quantum_ps1_px = float(
            remap_manifest.get("cache_quantum_ps1_px", cache_quantum_ps1_px)
        )
        keying = str(remap_manifest.get("keying", keying))

    schedule_path = remap_read / "shift_schedule.npz"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"shift schedule missing under remap read root {remap_read}")
    schedule = ShiftSchedule.load(schedule_path)
    shifts_df = load_remap_shifts_df(remap_read)
    assignment = assign_groups_from_schedule(
        schedule,
        grouping_quantum_ps1_px=grouping_quantum_ps1_px,
        cache_quantum_ps1_px=cache_quantum_ps1_px,
        keying=keying,
    )
    if not scc_only:
        write_group_artifacts(
            assignment,
            event_dir,
            geometry_mode="field",
            grouping_quantum_ps1_px=grouping_quantum_ps1_px,
            cache_quantum_ps1_px=cache_quantum_ps1_px,
        )
    if update_frames_csv and not scc_only:
        _update_frames_group_ids(event_dir, assignment.group_id_per_frame)

    exact_cache_dir = exact_cache_dir_for_read_root(remap_read)

    ignore_mask = 0
    for bit in ignore_mask_bits or [12]:
        ignore_mask |= 1 << int(bit)

    zarr_path = Path(convolved_dir)
    if zarr_path.suffix != ".zarr" or not zarr_path.name.endswith(".zarr"):
        zarr_path = zarr_path / f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    if not zarr_path.exists():
        alt = list(Path(convolved_dir).glob(f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}*.zarr"))
        if not alt:
            from syndiff_pipeline.common.scc_paths import scc_convolved_zarr

            zarr_path = scc_convolved_zarr(data_root, sector, camera, ccd)
            if not zarr_path.exists():
                raise FileNotFoundError(f"convolved zarr not found: {zarr_path}")
        else:
            zarr_path = alt[0]
    zarr.open(str(zarr_path), mode="r")

    keys = {
        (str(r.skycell), int(r.sx_int), int(r.sy_int))
        for r in shifts_df.itertuples(index=False)
    }
    master_path = _master_pixels2skycells_path(
        mapping_root,
        sector,
        camera,
        ccd,
        oversampling_factor=oversampling_factor,
    )
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

    from syndiff_pipeline.template_creation.processing.downsample import (
        resolve_stage_regmaps_to_scratch,
        stage_regmap_files_to_scratch,
    )

    scratch_regmaps: dict[str, str] = {}
    if resolve_stage_regmaps_to_scratch(stage_regmaps_to_scratch):
        sky_reg: list[tuple[str, str]] = []
        for sc in sorted({k[0] for k in keys}):
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
        if sky_reg:
            local_paths, scratch_dir, n_staged, elapsed = stage_regmap_files_to_scratch(
                [p for _, p in sky_reg],
                sector=sector,
                camera=camera,
                ccd=ccd,
                oversampling_factor=oversampling_factor,
            )
            scratch_regmaps = {sc: lp for (sc, _), lp in zip(sky_reg, local_paths)}
            log.info(
                "Staged %d/%d ROI regmaps to scratch %s in %.1fs",
                n_staged,
                len(sky_reg),
                scratch_dir,
                elapsed,
            )

    def _one(skycell: str, sx_i: int, sy_i: int) -> str:
        out = contrib_path(store, skycell, sx_i, sy_i)
        if out.is_file() and not rebuild_field_store:
            return "skip"
        if out.is_file() and rebuild_field_store:
            out.unlink()
        reg = scratch_regmaps.get(skycell) or _find_regmap(
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
            cache_name = contrib_basename(skycell, sx_i, sy_i).replace(".npz", "_exact.npz")
            cache_path = exact_cache_dir / cache_name
            try:
                hybrid_map, meta = hybrid_assignment_from_exact_cache(
                    assignment_map,
                    sx_i,
                    sy_i,
                    cache_path,
                    hybrid_R=int(hybrid_R),
                )
                log.debug(
                    "L5 hybrid %s sx=%+d sy=%+d cache=%s",
                    skycell,
                    sx_i,
                    sy_i,
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
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning(
                    "L5 exact-cache miss for %s sx=%+d sy=%+d (%s); data-roll fallback",
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
    if apply_hybrid_exact:
        hybrid_cap = int(_os.environ.get("SYNDIFF_HYBRID_MAX_JOBS", "24"))
        avail = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") else (
            _os.cpu_count() or hybrid_cap
        )
        n_jobs_eff = min(n_jobs_eff, max(1, hybrid_cap), max(1, avail))
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
        "remap_root": str(remap_store),
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

    if not scc_only:
        event_dir.mkdir(parents=True, exist_ok=True)
        (event_dir / "field_contrib_keys.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "store_root": str(store),
                    "remap_root": str(remap_store),
                    "n_contrib_keys": len(keys),
                    "keys": [[str(s), int(x), int(y)] for s, x, y in key_list],
                }
            )
            + "\n"
        )

    return {
        "output_dir": str(store),
        "remap_root": str(remap_store),
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
    if present_only is None:
        present_only = crop is not None
    shifts = _group_shifts_present(
        store_root, shifts_df, group_id, present_only=present_only
    )
    out = assemble_group_from_contribs(store_root, shifts, shape=shape, crop=crop)
    return np.asarray(out["count"], dtype=np.float64)
