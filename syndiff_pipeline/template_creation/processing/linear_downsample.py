"""True ``geometry_mode: linear`` template downsample.

Restores the pre-``distortion_aware_templates`` (commit ``78147a3``) linear
bootstrap algorithm on current SCC-scoped storage: frame-level offset groups
(~19 per SCC, from the SCC point-drift table -- see
``scc_reference_ffi.resolve_scc_point_drift_table``), one integer PS1-pixel
roll per skycell per group (no L4a/L4b intra/inter-skycell correction), binned
directly into one whole-SCC mosaic FITS per group. No remap stage and no
hierarchical contribs -- both are field-mode-only concerns: field mode needs
per-skycell-signature grouping (~10^2-10^3 groups) and L4 correction; linear
groups are shared identically across every skycell, so each skycell's
contribution is scattered straight into its group's final mosaic array.

Reuses field mode's already-tested per-skycell binning primitive
(``field_downsample._bin_skycell_contrib``) and shared/legacy convolved-store
loaders -- only the group source (SCC point-drift table, not signature
islands) and the output layout (flat ``syndiff_template_*_dx*_dy*.fits.gz``,
not hierarchical contribs) are new.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from astropy.io import fits

log = logging.getLogger(__name__)

LINEAR_ASSEMBLY_BASENAME = "linear_mode_assembly.json"


def _linear_template_basename(
    sector: int,
    camera: int,
    ccd: int,
    dx: float,
    dy: float,
    *,
    oversampling_factor: int = 1,
) -> str:
    os_part = f"_os{int(oversampling_factor)}" if int(oversampling_factor) > 1 else ""
    return (
        f"syndiff_template_s{int(sector):04d}_{int(camera)}_{int(ccd)}"
        f"{os_part}_dx{float(dx):.3f}_dy{float(dy):.3f}.fits.gz"
    )


_PAD_COLUMNS = (
    "pad_skycell_top", "pad_skycell_right", "pad_skycell_top_right",
    "pad_skycell_bottom", "pad_skycell_left", "pad_skycell_bottom_left",
    "pad_skycell_bottom_right", "pad_skycell_top_left",
)


def _skycells_needing_cross_projection_padding(skycell_df: pd.DataFrame) -> list[str]:
    """Skycell NAMEs whose mapping requires a *cross-projection* padding neighbor.

    These are the skycells for which
    ``padding_correction.load_padding_aware_convolved_cell`` applies the
    standalone additive seam correction (see
    ``doc/shared_convolved_cross_projection_simple_fix_plan.md``) before this
    module bins the shared canonical convolved cell. Recorded here (not
    silently dropped) purely for audit/telemetry -- e.g. to spot-check the
    corrected cells against a legacy per-SCC ``convolved.zarr`` crop.
    """
    if not any(c in skycell_df.columns for c in _PAD_COLUMNS):
        return []
    flagged: list[str] = []
    for name, row in skycell_df.iterrows():
        proj = str(row.get("projection", ""))
        for col in _PAD_COLUMNS:
            val = row.get(col)
            if pd.isna(val) or not str(val).strip():
                continue
            for cell in str(val).split("/"):
                cell = cell.strip()
                if not cell:
                    continue
                parts = cell.split(".")
                if len(parts) >= 2 and parts[1] != proj:
                    flagged.append(str(name))
                    break
            else:
                continue
            break
    return flagged


def _ignore_mask_from_bits(ignore_mask_bits: list[int] | None) -> int:
    ignore_mask = 0
    for bit in ignore_mask_bits or []:
        ignore_mask |= 1 << int(bit)
    return ignore_mask


def _resolve_convolved_source(
    convolved_dir: str | Path,
    *,
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
) -> tuple[Path | None, bool, Path | None]:
    """Mirror ``field_downsample``'s convolved-store resolution.

    Returns ``(zarr_path, shared_convolved_store, legacy_zarr_path)``. When
    the shared store is in use, ``zarr_path`` is the shared-store directory
    (kept for parity/logging) and per-cell reads go through
    ``field_downsample._try_load_shared_convolved_arrays``.
    """
    import zarr

    from syndiff_pipeline.common.scc_paths import scc_convolved_zarr
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _is_shared_convolved_store_path,
    )

    zarr_path = Path(convolved_dir)
    shared_convolved_store = _is_shared_convolved_store_path(zarr_path)
    legacy_zarr_path: Path | None = None
    if shared_convolved_store:
        if not zarr_path.is_dir():
            raise FileNotFoundError(f"shared convolved store not found: {zarr_path}")
        legacy_candidate = scc_convolved_zarr(data_root, sector, camera, ccd)
        if legacy_candidate.is_dir() or legacy_candidate.exists():
            try:
                zarr.open(str(legacy_candidate), mode="r")
                legacy_zarr_path = legacy_candidate
            except Exception:
                legacy_zarr_path = None
        return zarr_path, True, legacy_zarr_path

    if zarr_path.suffix != ".zarr" or not zarr_path.name.endswith(".zarr"):
        zarr_path = zarr_path / f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}.zarr"
    if not zarr_path.exists():
        alt = list(
            Path(convolved_dir).glob(f"sector_{sector:04d}_camera_{camera}_ccd_{ccd}*.zarr")
        )
        zarr_path = alt[0] if alt else scc_convolved_zarr(data_root, sector, camera, ccd)
        if not zarr_path.exists():
            raise FileNotFoundError(f"convolved zarr not found: {zarr_path}")
    zarr.open(str(zarr_path), mode="r")
    return zarr_path, False, None


def _load_ps1_skycell(
    skycell: str,
    *,
    data_root: str | Path,
    shared_convolved_store: bool,
    legacy_zarr_path: Path | None,
    zstore_cache: dict[str, Any],
    skycell_df: pd.DataFrame | None = None,
    psf_sigma: float | None = None,
    combined_recipe: Mapping | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one skycell's convolved (image, mask); shared store first, legacy zarr fallback.

    When the shared store is in use and ``skycell_df``/``psf_sigma`` are
    given, the canonical (same-projection-only) cell is additively
    seam-corrected for cross-projection padding via
    ``padding_correction.load_padding_aware_convolved_cell`` -- see that
    module's docstring. Falls back to the legacy per-SCC zarr path unchanged.

    ``combined_recipe`` is threaded through for deterministic recipe-matched
    fingerprint discovery in the shared, cross-sector combined/convolved
    stores (see ``field_downsample._discover_shared_convolved_fp``).
    """
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _load_zarr_skycell,
        _try_load_shared_convolved_arrays,
    )

    if shared_convolved_store:
        if skycell_df is not None and psf_sigma is not None:
            from syndiff_pipeline.template_creation.processing.padding_correction import (
                load_padding_aware_convolved_cell,
            )

            got = load_padding_aware_convolved_cell(
                data_root, skycell, skycell_df=skycell_df, psf_sigma=psf_sigma,
                combined_recipe=combined_recipe,
            )
        else:
            got = _try_load_shared_convolved_arrays(
                data_root, skycell, psf_sigma=psf_sigma, combined_recipe=combined_recipe,
            )
        if got is not None:
            return got
        if legacy_zarr_path is None:
            return None
        zstore = zstore_cache.get("legacy")
        if zstore is None:
            import zarr

            zstore = zarr.open(str(legacy_zarr_path), mode="r")
            zstore_cache["legacy"] = zstore
        try:
            return _load_zarr_skycell(zstore, skycell)
        except KeyError:
            return None

    zstore = zstore_cache.get("primary")
    if zstore is None:
        return None
    try:
        return _load_zarr_skycell(zstore, skycell)
    except KeyError:
        return None


def _bin_one_skycell_all_groups(
    skycell: str,
    *,
    mapping_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    oversampling_factor: int,
    tess_wcs,
    ps1_row: pd.Series,
    groups: list[tuple[int, float, float]],
    data_root: str | Path,
    shared_convolved_store: bool,
    legacy_zarr_path: Path | None,
    zstore_cache: dict[str, Any],
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    ignore_mask: int,
    mapping_grid,
    skycell_df: pd.DataFrame | None = None,
    psf_sigma: float | None = None,
    combined_recipe: Mapping | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """For one skycell, bin its contribution into every group. Returns
    ``{group_id: (tess_pixel_indices, sums, counts, mask_counts)}`` (groups
    with no valid contribution for this skycell are omitted)."""
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        build_ps1_wcs,
        compute_ps1_shift_for_skycell,
    )
    from syndiff_pipeline.template_creation.processing.field_downsample import (
        _bin_skycell_contrib,
    )
    from syndiff_pipeline.template_creation.processing.field_remap import _find_regmap

    loaded = _load_ps1_skycell(
        skycell,
        data_root=data_root,
        shared_convolved_store=shared_convolved_store,
        legacy_zarr_path=legacy_zarr_path,
        zstore_cache=zstore_cache,
        skycell_df=skycell_df,
        psf_sigma=psf_sigma,
        combined_recipe=combined_recipe,
    )
    if loaded is None:
        return {}
    ps1_data, ps1_mask = loaded

    regmap_path = _find_regmap(
        Path(mapping_root), sector, camera, ccd, skycell,
        oversampling_factor=oversampling_factor,
    )
    with fits.open(regmap_path) as hdul:
        assignment = np.asarray(
            hdul["TESS_PIXEL_MAP"].data if "TESS_PIXEL_MAP" in hdul else hdul[1].data
        )

    if assignment.shape != ps1_data.shape:
        log.warning(
            "skycell %s: regmap shape %s != convolved shape %s; skipping",
            skycell, assignment.shape, ps1_data.shape,
        )
        return {}

    ps1_wcs, _ = build_ps1_wcs(ps1_row)
    ra = float(ps1_row["RA"])
    dec = float(ps1_row["DEC"])

    # Groups are quantized in 0.01-TESS-px steps (~1 PS1 px at this plate
    # scale, see CLAUDE.md invariant #1) -- many groups round to the *same*
    # integer PS1-pixel shift for a given skycell. np.roll on this skycell's
    # full native-resolution array (tens of millions of pixels) dominates the
    # cost of `_bin_skycell_contrib`, so binning once per unique shift instead
    # of once per group is an exact (not approximate) and large speedup:
    # groups sharing a shift get byte-identical contributions by construction.
    shift_to_groups: dict[tuple[int, int], list[int]] = {}
    for group_id, dx, dy in groups:
        sx, sy = compute_ps1_shift_for_skycell(tess_wcs, dx, dy, ra, dec, ps1_wcs)
        shift_to_groups.setdefault((int(round(sx)), int(round(sy))), []).append(group_id)

    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for (sx_int, sy_int), group_ids in shift_to_groups.items():
        result = _bin_skycell_contrib(
            assignment=assignment,
            ps1_data=ps1_data,
            ps1_mask=ps1_mask,
            sx_int=sx_int,
            sy_int=sy_int,
            base_tess_shape=base_tess_shape,
            roi_bounds=roi_bounds,
            ignore_mask=ignore_mask,
            mapping_grid=mapping_grid,
        )
        if result is not None:
            for group_id in group_ids:
                out[group_id] = result
    return out


_LINEAR_WORKER: dict[str, Any] = {}


def _init_linear_worker(payload: dict[str, Any]) -> None:
    """Per-process joblib initializer (``backend='loky'``).

    Mirrors ``field_downsample._init_l5_worker``: this per-skycell binning is
    CPU-bound numpy work (roll + bincount over tens-of-millions-of-pixel
    arrays) that does not release the GIL enough for ``prefer="threads"`` to
    give real parallelism -- field mode's own L5 worker pool already made
    this choice (``prefer="processes"`` in field_downsample.py). Real
    zarr/file handles can't cross a process boundary via pickling, so each
    worker process reopens its own here, once, instead of inheriting one.
    """
    global _LINEAR_WORKER
    _LINEAR_WORKER = dict(payload)
    if not _LINEAR_WORKER.get("shared_convolved_store") and _LINEAR_WORKER.get("zarr_path"):
        import zarr

        _LINEAR_WORKER["zstore_cache"] = {
            "primary": zarr.open(str(_LINEAR_WORKER["zarr_path"]), mode="r")
        }
    else:
        _LINEAR_WORKER["zstore_cache"] = {}


def _linear_worker_process_skycell(skycell: str):
    p = _LINEAR_WORKER
    return skycell, _bin_one_skycell_all_groups(
        skycell,
        mapping_root=p["mapping_root"], sector=p["sector"], camera=p["camera"], ccd=p["ccd"],
        oversampling_factor=p["oversampling_factor"], tess_wcs=p["tess_wcs"],
        ps1_row=p["skycell_df"].loc[skycell], groups=p["groups"],
        data_root=p["data_root"], shared_convolved_store=p["shared_convolved_store"],
        legacy_zarr_path=p["legacy_zarr_path"], zstore_cache=p["zstore_cache"],
        base_tess_shape=p["base_tess_shape"], roi_bounds=p["roi_bounds"],
        ignore_mask=p["ignore_mask"], mapping_grid=p["mapping_grid"],
        skycell_df=p["skycell_df"], psf_sigma=p.get("psf_sigma"),
        combined_recipe=p.get("combined_recipe"),
    )


def run_linear_downsample_scc(
    resolved,
    *,
    convolved_dir: str | Path,
    mapping_root: str | Path,
    mapping_grid,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    store_root: str | Path,
    ref_ffi_path: str,
    oversampling_factor: int = 1,
    ignore_mask_bits: list[int] | None = None,
    n_jobs: int = 1,
    rebuild_field_store: bool = False,
    force_rerun: bool = False,
    progress_path: str | None = None,
) -> dict:
    """Build ~19 frame-level offset-group templates for one SCC (linear mode).

    Writes ``syndiff_template_s{S}_{C}_{K}[_osN]_dx{dx:.3f}_dy{dy:.3f}.fits.gz``
    directly under ``store_root`` -- one whole-SCC mosaic per group, matching
    the pre-fork convention ``difference_imaging.support.template_resolution``
    already knows how to read (``find_template_by_offset``). Also writes
    ``linear_mode_assembly.json`` (expected groups + produced files) so
    ``verify_downsample_linear_mode`` doesn't need to touch the legacy
    event-scoped ``cluster_template_job.json`` path.
    """
    import json
    import time

    from syndiff_pipeline.common.scc_paths import scc_mapping_master_skycells_csv
    from syndiff_pipeline.template_creation.processing.compute_ps1_skycell_shifts import (
        load_tess_wcs,
    )
    from syndiff_pipeline.template_creation.processing.field_remap import (
        _master_pixels2skycells_path,
        _master_skycell_id_map,
    )
    from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
        resolve_scc_point_drift_table,
        write_scc_wcs_drift_debug_plot,
    )

    t = resolved.target
    sector, camera, ccd = int(t.sector), int(t.camera), int(t.ccd)
    store = Path(store_root)
    store.mkdir(parents=True, exist_ok=True)

    if rebuild_field_store:
        for existing in store.glob("syndiff_template_*"):
            existing.unlink()

    # Point-drift groups: the modern (~19-row) equivalent of the old
    # cluster_template_job.json ``groups`` list -- computed here (not by
    # remap, which SKIP_REASON_LINEAR_GEOMETRY skips entirely for linear mode).
    from syndiff_pipeline.common.scc_paths import scc_remap_dir

    point_drift_store = scc_remap_dir(
        resolved.data_root, sector, camera, ccd,
        oversampling_factor=oversampling_factor, store_name="linear",
    )
    wcs_table, _drift = resolve_scc_point_drift_table(
        resolved, ref_ffi_path=ref_ffi_path, store_root=point_drift_store,
        force_rerun=force_rerun,
    )
    # Point-drift owns wcs_drift_linear_template.png (ref-FFI-center groups).
    write_scc_wcs_drift_debug_plot(
        resolved,
        ref_ffi_path,
        wcs_table=wcs_table,
        force_rerun=force_rerun,
    )
    from syndiff_pipeline.common import wcs_grouping

    summary = wcs_grouping.summarize_template_groups(wcs_table)
    groups: list[tuple[int, float, float]] = [
        (int(r.group_id), float(r.group_dx), float(r.group_dy))
        for r in summary.itertuples()
    ]
    if not groups:
        raise RuntimeError(f"No template groups resolved for s{sector:04d}_{camera}_{ccd}")

    master_csv_path = scc_mapping_master_skycells_csv(
        resolved.data_root, sector, camera, ccd, oversampling_factor=oversampling_factor,
    )
    skycell_df = pd.read_csv(master_csv_path).set_index("NAME", drop=False)

    master_path = _master_pixels2skycells_path(
        Path(mapping_root), sector, camera, ccd, oversampling_factor=oversampling_factor,
    )
    _master_map, name_to_id = _master_skycell_id_map(master_path)
    skycell_names = [n for n in name_to_id if n in skycell_df.index]

    zarr_path, shared_convolved_store, legacy_zarr_path = _resolve_convolved_source(
        convolved_dir, data_root=resolved.data_root, sector=sector, camera=camera, ccd=ccd,
    )
    log.info(
        "linear downsample s%04d_%d_%d: %d groups, %d skycells (%s)",
        sector, camera, ccd, len(groups), len(skycell_names),
        "shared convolved store" if shared_convolved_store else str(zarr_path),
    )

    cross_proj_corrected: list[str] = []
    if shared_convolved_store:
        cross_proj_corrected = _skycells_needing_cross_projection_padding(skycell_df)
        if cross_proj_corrected:
            log.info(
                "linear downsample s%04d_%d_%d: %d/%d skycells needed and received the "
                "cross-projection seam correction (padding_correction.load_padding_aware_"
                "convolved_cell); recorded in %s.",
                sector, camera, ccd, len(cross_proj_corrected), len(skycell_names),
                LINEAR_ASSEMBLY_BASENAME,
            )

    tess_wcs, _shape = load_tess_wcs(Path(ref_ffi_path))
    ignore_mask = _ignore_mask_from_bits(ignore_mask_bits)

    t_y, t_x = base_tess_shape
    accum_sums: dict[int, np.ndarray] = {
        gid: np.zeros((t_y, t_x), dtype=np.float32) for gid, _, _ in groups
    }
    accum_counts: dict[int, np.ndarray] = {
        gid: np.zeros((t_y, t_x), dtype=np.int32) for gid, _, _ in groups
    }
    accum_mask_counts: dict[int, np.ndarray] = {
        gid: np.zeros((t_y, t_x), dtype=np.int32) for gid, _, _ in groups
    }

    t0 = time.perf_counter()
    n_done = 0
    n_jobs_eff = max(1, int(n_jobs))
    zstore_cache: dict[str, Any] = {}
    if not shared_convolved_store:
        import zarr

        zstore_cache["primary"] = zarr.open(str(zarr_path), mode="r")

    psf_sigma = float(getattr(resolved.stages.ps1_process, "psf_sigma", 40.0))
    from syndiff_pipeline.template_creation.processing.combined_store import (
        production_combined_recipe,
    )

    combined_recipe = production_combined_recipe(
        resolved.stages.ps1_process,
        data_root=resolved.data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
    )

    def _process(skycell: str):
        return skycell, _bin_one_skycell_all_groups(
            skycell,
            mapping_root=mapping_root, sector=sector, camera=camera, ccd=ccd,
            oversampling_factor=oversampling_factor, tess_wcs=tess_wcs,
            ps1_row=skycell_df.loc[skycell], groups=groups,
            data_root=resolved.data_root, shared_convolved_store=shared_convolved_store,
            legacy_zarr_path=legacy_zarr_path, zstore_cache=zstore_cache,
            base_tess_shape=base_tess_shape, roi_bounds=roi_bounds,
            ignore_mask=ignore_mask, mapping_grid=mapping_grid,
            skycell_df=skycell_df, psf_sigma=psf_sigma, combined_recipe=combined_recipe,
        )

    from syndiff_pipeline.template_creation.processing import downsample_progress

    if progress_path is not None:
        downsample_progress.init_progress(
            progress_path, total_skycells=len(skycell_names),
            batch_sizes=[len(skycell_names)], oversampling_factor=oversampling_factor,
        )

    if n_jobs_eff <= 1:
        results_iter = (_process(sc) for sc in skycell_names)
    else:
        from joblib import Parallel, delayed

        worker_payload = {
            "mapping_root": mapping_root, "sector": sector, "camera": camera, "ccd": ccd,
            "oversampling_factor": oversampling_factor, "tess_wcs": tess_wcs,
            "skycell_df": skycell_df, "groups": groups,
            "data_root": resolved.data_root, "shared_convolved_store": shared_convolved_store,
            "legacy_zarr_path": legacy_zarr_path, "zarr_path": zarr_path,
            "base_tess_shape": base_tess_shape, "roi_bounds": roi_bounds,
            "ignore_mask": ignore_mask, "mapping_grid": mapping_grid,
            "psf_sigma": psf_sigma, "combined_recipe": combined_recipe,
        }
        # ``return_as="generator"`` (joblib >= 1.3) streams each skycell's
        # result back to the parent as soon as it completes, instead of
        # blocking until the entire batch finishes -- this is what lets the
        # loop below drain results (and update the progress sidecar/log)
        # incrementally rather than going silent for the whole run.
        results_iter = Parallel(
            n_jobs=n_jobs_eff,
            backend="loky",
            return_as="generator",
            initializer=_init_linear_worker,
            initargs=(worker_payload,),
        )(delayed(_linear_worker_process_skycell)(sc) for sc in skycell_names)

    for skycell, per_group in results_iter:
        n_done += 1
        for gid, (idx, sums, counts, mask_counts) in per_group.items():
            flat_sums = accum_sums[gid].reshape(-1)
            flat_counts = accum_counts[gid].reshape(-1)
            flat_mask = accum_mask_counts[gid].reshape(-1)
            # _bin_skycell_contrib returns float64 for all three planes (see
            # its docstring) even though counts/mask_counts are integer-
            # valued; the int32 accumulators need an explicit cast since
            # float64 -> int32 in-place add is a cross-kind cast numpy
            # refuses under the default 'same_kind' casting rule.
            flat_sums[idx] += sums.reshape(-1)
            flat_counts[idx] += counts.reshape(-1).astype(np.int32)
            flat_mask[idx] += mask_counts.reshape(-1).astype(np.int32)
        if progress_path is not None:
            downsample_progress.mark_skycell_done(progress_path, 0)
        if n_done % 100 == 0 or n_done == len(skycell_names):
            log.info(
                "linear downsample s%04d_%d_%d: %d/%d skycells binned (%.1fs)",
                sector, camera, ccd, n_done, len(skycell_names),
                time.perf_counter() - t0,
            )

    if progress_path is not None:
        downsample_progress.set_progress_phase(progress_path, "writing_outputs")

    with fits.open(ref_ffi_path) as hdul:
        tess_header = hdul[1].header

    from syndiff_pipeline.template_creation.processing.downsample import (
        create_syndiff_header,
    )

    syndiff_header = create_syndiff_header(
        tess_header, roi_bounds=roi_bounds, oversampling_factor=oversampling_factor,
        sector=sector,
    )
    # linear-mode templates share the same MAPGRID=3 paired-padding contract
    # as field-mode templates (mapping_grid is already computed above and
    # written into the assembly sidecar) -- stamp the same geometry keys
    # into every per-group FITS header so downstream MAPGRID=3 consumers
    # (template_coverage.py, hotpants.py) can resolve template bounds
    # directly from the FITS file instead of falling back to a full-chip
    # guess.
    for key, value in mapping_grid.to_fits_header_updates().items():
        serialized = value if isinstance(value, (str, bool)) else int(value)
        syndiff_header[key] = (serialized, f"MappingGrid {key}")

    artifacts: list[str] = []
    for gid, dx, dy in groups:
        basename = _linear_template_basename(
            sector, camera, ccd, dx, dy, oversampling_factor=oversampling_factor,
        )
        out_path = store / basename
        offset_header = syndiff_header.copy()
        offset_header["DX_SHIFT"] = (dx, "TESS pixel x shift")
        offset_header["DY_SHIFT"] = (dy, "TESS pixel y shift")
        offset_header["GROUP_ID"] = (gid, "SCC point-drift group id")
        primary_hdu = fits.PrimaryHDU(header=offset_header)
        hdu1 = fits.ImageHDU(data=accum_sums[gid], header=offset_header, name="FLUX_SUM")
        hdu2 = fits.ImageHDU(data=accum_counts[gid], header=offset_header, name="COUNT")
        hdu3 = fits.ImageHDU(data=accum_mask_counts[gid], header=offset_header, name="MASK")
        fits.HDUList([primary_hdu, hdu1, hdu2, hdu3]).writeto(out_path, overwrite=True)
        artifacts.append(str(out_path))

    assembly_path = store / LINEAR_ASSEMBLY_BASENAME
    assembly_payload = {
        "schema_version": 3,
        "geometry_mode": "linear",
        "sector": sector, "camera": camera, "ccd": ccd,
        "oversampling_factor": oversampling_factor,
        "mapping_grid": mapping_grid.to_mapping_dict(),
        "reference_ffi_path": str(ref_ffi_path),
        "n_groups": len(groups),
        "groups": [{"group_id": gid, "group_dx": dx, "group_dy": dy} for gid, dx, dy in groups],
        "n_skycells": len(skycell_names),
        "artifacts": [Path(p).name for p in artifacts],
        "cross_projection_padding_corrected": cross_proj_corrected,
    }
    assembly_path.write_text(json.dumps(assembly_payload, indent=2) + "\n")

    if progress_path is not None:
        downsample_progress.set_progress_phase(
            progress_path, "complete", total_skycells=len(skycell_names),
        )

    return {
        "output_dir": str(store),
        "n_groups": len(groups),
        "n_skycells": len(skycell_names),
        "geometry_mode": "linear",
        "artifacts": artifacts + [str(assembly_path)],
        "expected_count": len(groups) + 1,
        "produced_count": sum(1 for p in artifacts if Path(p).is_file()) + int(assembly_path.is_file()),
    }
