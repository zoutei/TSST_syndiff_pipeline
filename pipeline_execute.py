"""
Config-driven pipeline execution (YAML ``pipeline`` list).
"""

from __future__ import annotations

import glob
import gc
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from . import (
    background,
    epsf_fitting,
    final_diff,
    hotpants_runner,
    masking,
    photometry,
    sat_template,
    temporal_smooth,
    wcs_grouping,
)
from .config import SynDiffConfig
from .download import list_local_ffis, nested_ffi_dir, _ffi_filename_pattern
from .frame_manifest import (
    apply_epsf_status,
    apply_hotpants_workspace_results,
    group_ids_from_ffi_stems,
    load_frame_manifest,
    manifest_csv_exists,
    manifest_path_from_output_dir,
    ordered_diff_paths_for_workspace,
    save_frame_manifest,
)
from .hotpants_runner import HotpantsWorkspaceDirs
from .paths import ADAPTIVE_BKG_STACK_BASENAME, BACKGROUND_STACK_NPZ_ARRAY_KEY
from .pipeline_context import PipelineInvocationContext
from .pipeline_validate import validate_pipeline
from .subtract_expr import parse_subtract_expression

log = logging.getLogger(__name__)


def _load_stack_npz_or_npy(dir_path: str, basename: str) -> np.ndarray:
    """
    Load ``{dir_path}/{basename}.npz`` (array ``BACKGROUND_STACK_NPZ_ARRAY_KEY``)
    or ``{basename}.npy``. Uses memory mapping when loading from disk.
    """
    npz_path = os.path.join(dir_path, f"{basename}.npz")
    npy_path = os.path.join(dir_path, f"{basename}.npy")
    if os.path.isfile(npz_path):
        z = np.load(npz_path, mmap_mode="r")
        if BACKGROUND_STACK_NPZ_ARRAY_KEY not in z.files:
            raise KeyError(
                f"{npz_path!r} missing {BACKGROUND_STACK_NPZ_ARRAY_KEY!r}; "
                f"have {list(z.files)}"
            )
        return z[BACKGROUND_STACK_NPZ_ARRAY_KEY]
    if os.path.isfile(npy_path):
        return np.load(npy_path, mmap_mode="r")
    raise FileNotFoundError(
        f"missing stack under {dir_path!r}: expected {basename}.npz or {basename}.npy"
    )


def _write_per_frame_background_fits(
    out_ws: str,
    stack: np.ndarray,
    stem_rows: list,
    filename_fmt: str,
    *,
    row_ok: Optional[Callable[[dict], bool]] = None,
) -> None:
    """Write one float32 FITS per row using ``filename_fmt.format(stem=...)``."""
    for i, row in enumerate(stem_rows):
        if i >= stack.shape[0]:
            break
        if row_ok is not None and not row_ok(row):
            continue
        stem = row["stem"]
        fn = filename_fmt.format(stem=stem)
        fits.writeto(
            os.path.join(out_ws, fn),
            np.asarray(stack[i], dtype=np.float32),
            overwrite=True,
        )


def _pipeline_plots_root(output_dir: str, cfg: SynDiffConfig) -> str:
    """``output_dir`` or ``output_dir / pipeline_plots_dir`` for diagnostic figures."""
    sub = getattr(cfg, "pipeline_plots_dir", "debug_plots")
    if sub is None:
        return output_dir
    s = str(sub).strip()
    if not s:
        return output_dir
    return os.path.join(output_dir, s)


def _maybe_write_background_gif(
    cfg: SynDiffConfig,
    output_dir: str,
    cube: np.ndarray,
    wcs_table: Optional[pd.DataFrame],
    stem_rows: list,
    *,
    filename: str,
    cbar_label: str,
) -> None:
    """Animated GIF of a (T, ny, nx) background cube when ``pipeline_plots`` is True."""
    if not getattr(cfg, "pipeline_plots", False):
        return
    if wcs_table is None:
        log.debug("pipeline_plots: skip background GIF %s (no wcs_table)", filename)
        return
    from . import plot_pipeline

    plot_dir = _pipeline_plots_root(output_dir, cfg)
    os.makedirs(plot_dir, exist_ok=True)
    plot_pipeline.write_background_removal_animation(
        cube,
        wcs_table,
        stem_rows,
        plot_dir,
        filename=filename,
        cbar_label=cbar_label,
    )


def _subtract_load_plane(
    ws_root: str,
    stem: str,
    frame_index: int,
    npy_stack_by_ws: dict[str, np.ndarray | None],
) -> np.ndarray | None:
    """
    One per-frame 2D array from a workspace: ``bkg_temp_smooth.npz`` (array ``stack``) /
    ``bkg_temp_smooth.npy`` row *i* (if present) or ``<stem>.fits``.
    """
    if ws_root not in npy_stack_by_ws:
        stack = None
        npz_path = os.path.join(ws_root, f"{ADAPTIVE_BKG_STACK_BASENAME}.npz")
        npy_path = os.path.join(ws_root, f"{ADAPTIVE_BKG_STACK_BASENAME}.npy")
        if os.path.isfile(npz_path):
            z = np.load(npz_path, mmap_mode="r")
            if BACKGROUND_STACK_NPZ_ARRAY_KEY in z.files:
                stack = z[BACKGROUND_STACK_NPZ_ARRAY_KEY]
        elif os.path.isfile(npy_path):
            stack = np.load(npy_path, mmap_mode="r")
        npy_stack_by_ws[ws_root] = stack
    stack = npy_stack_by_ws[ws_root]
    if stack is not None and frame_index < len(stack):
        return stack[frame_index].astype(np.float64)
    fp = os.path.join(ws_root, f"{stem}.fits")
    if not os.path.isfile(fp):
        return None
    return fits.getdata(fp).astype(np.float64)


def _bootstrap_state_skip_wcs_grouping(
    cfg: SynDiffConfig, out: str, manifest_path: str | None
) -> tuple[Optional[pd.DataFrame], Optional[dict], Optional[str]]:
    """
    When ``wcs_grouping`` is not in the pipeline, load manifest + crop + ref FFI
    from *out* so downstream stages (e.g. forced photometry) can run alone.
    """
    kinds = {s.get("kind") for s in (cfg.pipeline or [])}
    if "wcs_grouping" in kinds:
        return None, None, None
    wcs_table: Optional[pd.DataFrame] = None
    crop_bounds: Optional[dict] = None
    ref_ffi_path: Optional[str] = None
    if manifest_csv_exists(out, manifest_path):
        wcs_table = load_frame_manifest(out, manifest_path)
        log.info(
            "  Loaded frame manifest for resume (wcs_grouping not in pipeline): %s",
            manifest_path_from_output_dir(out, manifest_path),
        )
    try:
        crop_bounds = wcs_grouping.load_crop_bounds(out)
    except FileNotFoundError:
        pass
    ref_ffi_path = wcs_grouping.load_reference_ffi_path(out, cfg.ref_ffi_path)
    return wcs_table, crop_bounds, ref_ffi_path


def _cfg_ffi_leaf(cfg: SynDiffConfig) -> str:
    return nested_ffi_dir(cfg.sector, cfg.camera, cfg.ccd, root=cfg.ffi_dir)


def _sorted_local_ffis(cfg: SynDiffConfig) -> list:
    return sorted(list_local_ffis(_cfg_ffi_leaf(cfg), cfg.sector, cfg.camera, cfg.ccd))


def _ffi_paths_for_processing(cfg: SynDiffConfig) -> list:
    all_sorted = _sorted_local_ffis(cfg)
    return wcs_grouping.select_ffis_with_valid_target_wcs(
        all_sorted,
        cfg.target_ra,
        cfg.target_dec,
        max_ffis=cfg.max_ffis,
    )


def _load_gaia_catalog(cfg: SynDiffConfig, output_dir: str) -> Optional[pd.DataFrame]:
    if cfg.gaia_catalog and os.path.isfile(cfg.gaia_catalog):
        return pd.read_csv(cfg.gaia_catalog)
    legacy = os.path.join(output_dir, "unique_gaia_stars_for_cropped_template.csv")
    if os.path.isfile(legacy):
        log.warning("Loading Gaia catalog from output_dir (legacy path).")
        return pd.read_csv(legacy)
    return None


def _ensure_gaia_crop(
    gaia_df: pd.DataFrame,
    ref_ffi_path: str,
    crop_bounds: dict,
) -> pd.DataFrame:
    return wcs_grouping.ensure_gaia_crop_xy(gaia_df, ref_ffi_path, crop_bounds)


def _load_tile_centers_json(output_dir: str) -> Optional[list]:
    path = os.path.join(output_dir, "tile_centers.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        raw = json.load(fh)
    return [tuple(c) for c in raw]


def _save_tile_centers(tile_centers: list, output_dir: str) -> None:
    path = os.path.join(output_dir, "tile_centers.json")
    with open(path, "w") as fh:
        json.dump(tile_centers, fh)


def _path_to_group_from_wcs(wcs_table: Optional[pd.DataFrame]) -> dict[str, int]:
    """Map FFI stem → group_id from ``wcs_table`` (``filename`` or ``path`` column)."""
    path_to_group: dict[str, int] = {}
    if wcs_table is None:
        return path_to_group
    if "filename" in wcs_table.columns:
        for _, row in wcs_table.iterrows():
            stem = Path(str(row["filename"])).stem
            path_to_group[stem] = int(row.get("group_id", 0))
    elif "path" in wcs_table.columns:
        for _, row in wcs_table.iterrows():
            stem = Path(str(row["path"])).stem
            path_to_group[stem] = int(row.get("group_id", 0))
    return path_to_group


def _tqdm_ffi_paths(ffi_paths: list, desc: str):
    """Iterate FFI paths with a tqdm bar when tqdm is installed."""
    try:
        from tqdm import tqdm

        return tqdm(ffi_paths, desc=desc, unit="frame")
    except ImportError:
        log.debug("tqdm not installed; skipping FITS load progress bar.")
        return ffi_paths


def _hotpants_results_from_dirs(
    ffi_paths: list,
    wcs_table: pd.DataFrame,
    diff_dir: str,
    bkg_dir: Optional[str],
) -> list:
    path_to_group = _path_to_group_from_wcs(wcs_table)
    results = []
    for ffi_path in _tqdm_ffi_paths(ffi_paths, "Loading diff/bkg FITS"):
        stem = Path(ffi_path).stem
        group_id = path_to_group.get(stem, 0)
        results.append(
            background.load_hotpants_row_from_disk(stem, diff_dir, bkg_dir, group_id)
        )
    return results


def _hotpants_result_stems_for_ordering(
    ffi_paths: list,
    wcs_table: pd.DataFrame,
    diff_dir: str,
) -> list:
    """
    Same stem / success / group_id order as :func:`_hotpants_results_from_dirs`,
    without loading diff or bkg arrays (for BTJD alignment in ``background_adaptive``).
    """
    path_to_group = _path_to_group_from_wcs(wcs_table)

    results = []
    for ffi_path in ffi_paths:
        stem = Path(ffi_path).stem
        dp = os.path.join(diff_dir, f"{stem}.fits")
        group_id = path_to_group.get(stem, 0)
        ok = os.path.isfile(dp)
        results.append(
            {
                "stem": stem,
                "success": ok,
                "group_id": group_id,
            }
        )
    return results


def _ensure_shared_mask_loaded(
    out: str,
    shared_mask: Optional[np.ndarray],
) -> np.ndarray:
    if shared_mask is not None:
        return shared_mask
    sm_path = os.path.join(out, "shared_mask.fits")
    if os.path.isfile(sm_path):
        mask = np.asarray(fits.getdata(sm_path), dtype=np.int16)
        log.info("  Loaded shared_mask from prior run (%s)", sm_path)
        return mask
    raise RuntimeError(
        "Background stages need shared_mask in memory (run shared_mask first) "
        f"or an existing {sm_path!r} from a prior run."
    )


def _strip_hp_heavy_arrays(hp_results: list) -> None:
    """Drop diff/bkg FITS arrays from in-memory hotpants dicts to free RAM."""
    for r in hp_results:
        r.pop("diff", None)
        r.pop("bkg", None)
        r.pop("path", None)


def _add_hotpants_bkg_to_stack_inplace(
    stack: np.ndarray,
    stem_rows: list,
    bkg_dir: str,
) -> None:
    """
    Add Hotpants polynomial background FITS per frame into ``stack`` (axis 0 = time).

    Only rows with ``success`` True are updated; missing ``{stem}.fits`` under ``bkg_dir``
    contributes nothing (same as a zero ``hp_b`` plane).
    """
    for i, row in enumerate(stem_rows):
        if i >= stack.shape[0]:
            break
        if not row.get("success"):
            continue
        stem = row["stem"]
        bp = os.path.join(bkg_dir, f"{stem}.fits")
        if not os.path.isfile(bp):
            continue
        stack[i] = stack[i] + fits.getdata(bp).astype(np.float32, copy=False)


def _adaptive_smooth_bkg_stack(
    bkg_stack: np.ndarray,
    wcs_table: pd.DataFrame,
    hotpants_results: list,
    cfg: SynDiffConfig,
) -> np.ndarray:
    log.info(
        "_adaptive_smooth_bkg_stack: BTJD order for %d hotpants result(s), bkg_stack %s",
        len(hotpants_results),
        getattr(bkg_stack, "shape", None),
    )
    time_btjd = temporal_smooth.btjd_for_hotpants_order(wcs_table, hotpants_results)
    vpath = cfg.bkg_vector_path or None
    if vpath == "":
        vpath = None
    return temporal_smooth.adaptive_smooth_background(
        bkg_stack,
        time_btjd,
        cfg.sector,
        cfg.camera,
        vector_path=vpath,
        method=cfg.bkg_adaptive_method,
        savgol_window=cfg.bkg_adaptive_savgol_window,
        savgol_polyorder=cfg.bkg_adaptive_savgol_polyorder,
        w_min=cfg.bkg_adaptive_w_min,
        w_max=cfg.bkg_adaptive_w_max,
        block_size=cfg.bkg_adaptive_block_size,
        n_jobs=cfg.n_jobs,
    )


def run_config_pipeline(cfg: SynDiffConfig, *, validate_only: bool = False) -> None:
    validate_pipeline(cfg)
    if validate_only:
        log.info("Pipeline configuration is valid.")
        return

    ctx = PipelineInvocationContext.from_config(cfg)
    out = ctx.cfg.output_dir
    manifest_path = ctx.manifest_path
    os.makedirs(out, exist_ok=True)

    wcs_table: Optional[pd.DataFrame] = None
    crop_bounds = None
    ref_ffi_path: Optional[str] = None
    shared_mask = None
    ref_stars: Optional[pd.DataFrame] = None
    gaia_df: Optional[pd.DataFrame] = None
    tile_centers = None
    processing_ffi_paths: list = []

    _bw, _bc, _br = _bootstrap_state_skip_wcs_grouping(cfg, out, manifest_path)
    if _bw is not None:
        wcs_table = _bw
    if _bc is not None:
        crop_bounds = _bc
    if _br is not None:
        ref_ffi_path = _br

    for stage in cfg.pipeline:
        kind = stage["kind"]
        log.info("=" * 70)
        log.info("Stage: %s", kind)

        if kind == "wcs_grouping":
            ffi_leaf = _cfg_ffi_leaf(cfg)
            all_sorted = _sorted_local_ffis(cfg)
            if not all_sorted:
                glob_pat = os.path.join(
                    ffi_leaf,
                    _ffi_filename_pattern(cfg.sector, cfg.camera, cfg.ccd),
                )
                raise FileNotFoundError(f"No FFI files matching {glob_pat!r}")
            ffi_paths = wcs_grouping.select_ffis_with_valid_target_wcs(
                all_sorted,
                cfg.target_ra,
                cfg.target_dec,
                max_ffis=cfg.max_ffis,
            )
            log.info("  FFIs on disk: %d; processing: %d", len(all_sorted), len(ffi_paths))

            wcs_table = wcs_grouping.build_wcs_table(
                ffi_paths, cfg.target_ra, cfg.target_dec
            )
            wcs_table = wcs_grouping.assign_template_groups(
                wcs_table, cfg.offset_threshold
            )
            save_frame_manifest(wcs_table, out, manifest_path)

            if cfg.ref_ffi_path and os.path.exists(cfg.ref_ffi_path):
                ref_ffi_path = cfg.ref_ffi_path
            else:
                dx = pd.to_numeric(wcs_table["delta_x"], errors="coerce")
                dy = pd.to_numeric(wcs_table["delta_y"], errors="coerce")
                ok = dx.notna() & dy.notna()
                if "wcs_ok" in wcs_table.columns:
                    wok = wcs_table["wcs_ok"]
                    ok = ok & wok.apply(
                        lambda x: x is True or str(x).lower() in ("true", "1")
                    )
                if not ok.any():
                    raise RuntimeError("No FFI with usable WCS offsets for ref frame.")
                ref_ffi_path = str(wcs_table[ok].iloc[0]["path"])
            log.info("  Reference FFI: %s", ref_ffi_path)

            with fits.open(ref_ffi_path, memmap=True) as hdul:
                ref_header = hdul[1].header
            crop_bounds = wcs_grouping.get_crop_bounds(
                ref_header,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
                y_min=cfg.y_min,
                y_max=cfg.y_max,
                crop_quadrant=cfg.crop_quadrant,
                x_left_dead=cfg.x_left_dead,
                x_right_dead=cfg.x_right_dead,
                y_edge_strip=cfg.y_edge_strip,
            )

            summary_df = wcs_grouping.summarize_template_groups(wcs_table)
            wcs_grouping.write_cluster_template_job_json(
                summary_df,
                ref_ffi_path,
                cfg.sector,
                cfg.camera,
                cfg.ccd,
                cfg.offset_threshold,
                out,
                crop_bounds=crop_bounds,
            )
            if getattr(cfg, "pipeline_plots", False):
                plot_dir = _pipeline_plots_root(out, cfg)
                os.makedirs(plot_dir, exist_ok=True)
                wcs_grouping.plot_wcs_drift_and_template_assignment(
                    wcs_table,
                    os.path.join(plot_dir, "wcs_drift_template_debug.png"),
                )

        elif kind == "shared_mask":
            if wcs_table is None or crop_bounds is None or ref_ffi_path is None:
                raise RuntimeError("shared_mask requires wcs_grouping first.")
            gaia_df = _load_gaia_catalog(cfg, out)
            if gaia_df is None:
                raise RuntimeError("gaia_catalog required for shared_mask.")
            gaia_df = _ensure_gaia_crop(gaia_df, ref_ffi_path, crop_bounds)

            with fits.open(ref_ffi_path, memmap=True) as hdul:
                ref_data = hdul[1].data.astype(np.float64)
            ref_crop = wcs_grouping.crop_image(ref_data, crop_bounds)

            gaia_mask_df = epsf_fitting.add_tess_flux_ratio(gaia_df.copy())
            gaia_mask_df["mag"] = gaia_mask_df["tess_mag"]

            shared_mask = masking.make_shared_mask(
                ref_image=ref_crop,
                gaia_df=gaia_mask_df,
                crop_bounds=crop_bounds,
                straps_csv=cfg.straps_csv,
                maglim=cfg.gaia_mag_bright,
                strapsize=cfg.strapsize,
                output_dir=out,
            )
            ref_stars = masking.select_hotpants_ref_stars(
                gaia_df=gaia_mask_df,
                crop_bounds=crop_bounds,
                mag_min=cfg.ref_mag_min,
                mag_max=cfg.ref_mag_max,
                isolation_mag=cfg.ref_isolation_mag,
                isolation_radius_px=cfg.ref_isolation_px,
                separation_px=cfg.ref_separation_px,
                output_dir=out,
            )
            pipe_csv = os.path.join(out, "gaia_catalog_pipeline.csv")
            gaia_mask_df.to_csv(pipe_csv, index=False)
            gaia_df = gaia_mask_df.drop(columns=["mag"], errors="ignore")

        elif kind == "hotpants":
            if wcs_table is None or crop_bounds is None or ref_ffi_path is None:
                raise RuntimeError("hotpants requires wcs_grouping first.")
            if shared_mask is None or ref_stars is None:
                raise RuntimeError("hotpants requires shared_mask first.")
            try:
                hotpants_runner.ensure_template_paths_from_syndiff_or_group_dirs(
                    cfg, wcs_table, crop_bounds
                )
            except hotpants_runner.SyndiffTemplateDiscoveryError as e:
                raise RuntimeError(str(e)) from e
            if not cfg.template_paths:
                raise RuntimeError(
                    "template_paths empty; set template_dir or template_paths after WCS grouping."
                )

            inp = stage.get("inputs") or {}
            o = stage["output"]
            diffs_l = o["diffs"]
            conv_l = o["convolved"]
            bkg_l = o.get("bkg")

            diff_dir = ctx.workspace(diffs_l)
            conv_dir = ctx.workspace(conv_l)
            bkg_dir = ctx.workspace(bkg_l) if bkg_l else None

            dirs = HotpantsWorkspaceDirs(
                diffs=diff_dir,
                convolved=conv_dir,
                bkg=bkg_dir,
            )

            processing_ffi_paths = _ffi_paths_for_processing(cfg)
            sci_bkg_stack = None
            if inp.get("bkg"):
                bkg_ws = ctx.workspace(inp["bkg"])
                arr = []
                for p in processing_ffi_paths:
                    st = Path(p).stem
                    bp = os.path.join(bkg_ws, f"{st}.fits")
                    if os.path.isfile(bp):
                        arr.append(fits.getdata(bp).astype(np.float64))
                    else:
                        arr.append(np.zeros((1, 1)))  # placeholder; hotpants may fail
                ny, nx = crop_bounds["shape"]
                sci_bkg_stack = np.stack(
                    [
                        a if a.shape == (ny, nx) else np.zeros((ny, nx), dtype=np.float64)
                        for a in arr
                    ]
                )

            round_id = 2 if inp.get("bkg") else 1
            results = hotpants_runner.hotpants_loop(
                ffi_paths=processing_ffi_paths,
                wcs_table=wcs_table,
                template_path_map={int(k): v for k, v in cfg.template_paths.items()},
                mask=shared_mask,
                crop_bounds=crop_bounds,
                cfg=cfg,
                output_dir=out,
                ref_stars_df=ref_stars,
                round_id=round_id,
                sci_bkg_stack=sci_bkg_stack,
                workspace_dirs=dirs,
            )
            wcs_table = apply_hotpants_workspace_results(
                wcs_table, processing_ffi_paths, results, diffs_l
            )
            save_frame_manifest(wcs_table, out, manifest_path)

            if inp.get("convolved"):
                log.warning(
                    "hotpants inputs.convolved=%r ignored in this version (convolved products are always written to output.convolved).",
                    inp["convolved"],
                )

        elif kind == "epsf":
            inp = stage["inputs"]
            label_out = stage["output"]
            diff_paths = ordered_diff_paths_for_workspace(
                wcs_table, out, inp["diffs"], manifest_path
            )
            if gaia_df is None:
                gaia_df = _load_gaia_catalog(cfg, out)
            if gaia_df is None:
                raise RuntimeError("epsf requires gaia_catalog.")
            gaia_df = _ensure_gaia_crop(gaia_df, ref_ffi_path, crop_bounds)
            gaia_df_flux = epsf_fitting.add_tess_flux_ratio(gaia_df)
            col_corr_2d = epsf_fitting.build_median_mask_correction(
                cfg.median_mask_path, cfg.camera, cfg.ccd, crop_bounds
            )
            ws_out = ctx.workspace(label_out)
            os.makedirs(ws_out, exist_ok=True)
            epsf_stack, tile_centers_new, ffi_stems, epsf_ok = (
                epsf_fitting.fit_epsf_all_frames(
                    diff_paths,
                    gaia_df_flux,
                    col_corr_2d,
                    cfg,
                    ws_out,
                    round_id=1,
                )
            )
            if tile_centers_new is not None:
                tile_centers = tile_centers_new
            wcs_table = apply_epsf_status(wcs_table, ffi_stems, epsf_ok, round_id=1)
            save_frame_manifest(wcs_table, out, manifest_path)

            if cfg.epsf_temporal_smooth:
                epsf_smooth = temporal_smooth.temporal_smooth_epsf(
                    epsf_stack, cfg.temporal_smooth_window
                )
            else:
                epsf_smooth = temporal_smooth.prepare_epsf_stack_no_time_filter(
                    epsf_stack
                )
            temporal_smooth.save_epsf_smooth(
                epsf_smooth, ws_out, round_id=1, ffi_stem=ffi_stems
            )
            group_ids = group_ids_from_ffi_stems(wcs_table, ffi_stems)
            temporal_smooth.compute_group_epsf(
                epsf_smooth, group_ids, output_dir=ws_out
            )

            if tile_centers is not None:
                _save_tile_centers(tile_centers, out)

        elif kind == "sat_template":
            inp = stage["inputs"]
            label_out = stage["output"]
            ws_epsf = ctx.workspace(inp["epsf"])
            epsf_smooth, _ = temporal_smooth.load_epsf_smooth(ws_epsf, 1)
            group_epsf = _load_group_epsf_from_dir(ws_epsf, "group_epsf")

            tile_centers = _load_tile_centers_json(out)
            if tile_centers is None and crop_bounds is not None:
                from .epsf_fitting import _make_tile_grid

                ny, nx = crop_bounds["shape"]
                tiles = _make_tile_grid(ny, nx, cfg.tile_ny, cfg.tile_nx)
                tile_centers = [
                    (c0 + ts / 2, r0 + ts / 2) for (r0, c0, ts) in tiles
                ]
                _save_tile_centers(tile_centers, out)

            removed_df = _load_removed_stars_in_crop(
                cfg.removed_stars_csv, crop_bounds, gaia_df
            )
            ws_sat = ctx.workspace(label_out)
            os.makedirs(ws_sat, exist_ok=True)
            sat_native, sat_hr = sat_template.build_all_group_templates(
                removed_df, group_epsf, tile_centers, crop_bounds, cfg
            )
            sat_template.save_group_templates(sat_native, sat_hr, ws_sat, round_id=1)

        elif kind == "subtract":
            inp = stage["inputs"]
            label_out = stage["output"]
            out_ws = ctx.workspace(label_out)
            os.makedirs(out_ws, exist_ok=True)

            expr = inp.get("expression")
            if isinstance(expr, str) and expr.strip():
                terms = parse_subtract_expression(expr)
            else:
                terms = [
                    (1, inp["science"]),
                    (-1, inp["template"]),
                ]

            stems = (
                wcs_table["filename"].map(lambda f: Path(str(f)).stem)
                if "filename" in wcs_table.columns
                else wcs_table["path"].map(lambda p: Path(str(p)).stem)
            )
            npy_cache: dict[str, np.ndarray | None] = {}

            for i, stem in enumerate(stems):
                acc: np.ndarray | None = None
                skip = False
                for sign, lab in terms:
                    plane = _subtract_load_plane(
                        ctx.workspace(lab), stem, i, npy_cache
                    )
                    if plane is None:
                        skip = True
                        break
                    if acc is None:
                        acc = sign * plane
                    else:
                        if plane.shape != acc.shape:
                            raise RuntimeError(
                                "subtract: shape mismatch for "
                                f"{stem!r} between workspaces ({acc.shape} vs {plane.shape})"
                            )
                        acc = acc + sign * plane
                if skip or acc is None:
                    continue
                fits.writeto(
                    os.path.join(out_ws, f"{stem}.fits"),
                    acc.astype(np.float32),
                    overwrite=True,
                )

        elif kind == "background_rough":
            inp = stage["inputs"]
            label_out = stage["output"]
            diff_dir = ctx.workspace(inp["diffs"])
            hp_bkg_dir = ctx.workspace(inp["bkg"])
            out_ws = ctx.workspace(label_out)
            os.makedirs(out_ws, exist_ok=True)
            shared_mask = _ensure_shared_mask_loaded(out, shared_mask)
            if not processing_ffi_paths:
                processing_ffi_paths = _ffi_paths_for_processing(cfg)
            round_id = int(stage.get("round_id", 1))
            stream = bool(stage.get("stream_load_rough", False))
            if stream:
                log.info(
                    "  background_rough: stream_load_rough — load+estimate per frame in parallel "
                    "(%d FFIs; avoids holding the full diff/bkg stack in RAM)",
                    len(processing_ffi_paths),
                )
                path_to_group = _path_to_group_from_wcs(wcs_table)
                rough = background.background_loop_streaming(
                    processing_ffi_paths,
                    diff_dir,
                    hp_bkg_dir,
                    path_to_group,
                    shared_mask,
                    output_dir=out_ws,
                    round_id=round_id,
                    recombine_hotpants=cfg.bkg_r1_recombine_hotpants,
                    n_jobs=cfg.n_jobs,
                )
                hp_results = _hotpants_result_stems_for_ordering(
                    processing_ffi_paths, wcs_table, diff_dir
                )
                for _r in hp_results:
                    _r["diff"] = True if _r.get("success") else None
            else:
                log.info(
                    "  background_rough: loading diff/bkg FITS into memory (%d FFIs) ...",
                    len(processing_ffi_paths),
                )
                hp_results = _hotpants_results_from_dirs(
                    processing_ffi_paths, wcs_table, diff_dir, hp_bkg_dir
                )
                log.info(
                    "  background_rough: FITS load complete; starting per-frame rough stack"
                )
                rough = background.background_loop(
                    hotpants_results=hp_results,
                    mask=shared_mask,
                    output_dir=out_ws,
                    round_id=round_id,
                    recombine_hotpants=cfg.bkg_r1_recombine_hotpants,
                    n_jobs=cfg.n_jobs,
                )
            if bool(stage.get("write_per_frame_fits", True)):
                _write_per_frame_background_fits(
                    out_ws,
                    rough,
                    hp_results,
                    "{stem}_rough_bkg.fits",
                    row_ok=lambda r: bool(r.get("success")) and r.get("diff") is not None,
                )
                log.info(
                    "  background_rough: wrote per-frame *_rough_bkg.fits under %s",
                    out_ws,
                )
            _maybe_write_background_gif(
                cfg,
                out,
                rough,
                wcs_table,
                hp_results,
                filename=f"rough_bkg_r{round_id}_animation.gif",
                cbar_label="Rough background (per frame)",
            )
            _strip_hp_heavy_arrays(hp_results)
            del rough
            gc.collect()
            log.info(
                "  background_rough: stack saved as rough_bkg_r%s.npz and .npy under %s",
                round_id,
                out_ws,
            )

        elif kind == "background_adaptive":
            inp = stage["inputs"]
            label_out = stage["output"]
            rough_ws = ctx.workspace(inp["rough"])
            diff_dir = ctx.workspace(inp["diffs"])
            hp_bkg_dir = ctx.workspace(inp["bkg"])
            out_ws = ctx.workspace(label_out)
            os.makedirs(out_ws, exist_ok=True)
            round_id = int(stage.get("round_id", 1))
            rough_base = f"rough_bkg_r{round_id}"
            try:
                rough = _load_stack_npz_or_npy(rough_ws, rough_base)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"background_adaptive: missing {rough_base}.npz or {rough_base}.npy "
                    f"under {rough_ws!r}"
                ) from e
            rough_path = os.path.join(rough_ws, rough_base + ".npz")
            if not os.path.isfile(rough_path):
                rough_path = os.path.join(rough_ws, rough_base + ".npy")
            if not processing_ffi_paths:
                processing_ffi_paths = _ffi_paths_for_processing(cfg)
            hp_order = _hotpants_result_stems_for_ordering(
                processing_ffi_paths, wcs_table, diff_dir
            )
            if len(hp_order) != rough.shape[0]:
                raise RuntimeError(
                    "background_adaptive: length mismatch "
                    f"(rough shape[0]={rough.shape[0]} vs ffi stem list len={len(hp_order)}). "
                    "Use the same max_ffis / manifest as when building the rough stack."
                )
            log.info(
                "  background_adaptive: loaded %s; adding hp_b per frame then adaptive smooth ...",
                rough_path,
            )
            bkg_arr = np.array(rough, dtype=np.float32, copy=True)
            _add_hotpants_bkg_to_stack_inplace(bkg_arr, hp_order, hp_bkg_dir)
            temp_smooth = _adaptive_smooth_bkg_stack(bkg_arr, wcs_table, hp_order, cfg)
            log.info(
                "  background_adaptive: adaptive smooth returned; writing %s.npz and .npy",
                ADAPTIVE_BKG_STACK_BASENAME,
            )
            background.save_background_stack(
                temp_smooth,
                os.path.join(out_ws, f"{ADAPTIVE_BKG_STACK_BASENAME}.npy"),
            )
            fits_fmt = f"{{stem}}_{ADAPTIVE_BKG_STACK_BASENAME}.fits"
            if bool(stage.get("write_per_frame_fits", True)):
                _write_per_frame_background_fits(
                    out_ws, temp_smooth, hp_order, fits_fmt
                )
                log.info(
                    "  background_adaptive: wrote per-frame *_%s.fits under %s",
                    ADAPTIVE_BKG_STACK_BASENAME,
                    out_ws,
                )
            _maybe_write_background_gif(
                cfg,
                out,
                temp_smooth,
                wcs_table,
                hp_order,
                filename=f"{ADAPTIVE_BKG_STACK_BASENAME}_adaptive_animation.gif",
                cbar_label="Adaptive smoothed background",
            )

        elif kind == "background_estimate":
            inp = stage["inputs"]
            label_out = stage["output"]
            diff_dir = ctx.workspace(inp["diffs"])
            hp_bkg_dir = ctx.workspace(inp["bkg"])
            out_ws = ctx.workspace(label_out)
            os.makedirs(out_ws, exist_ok=True)
            shared_mask = _ensure_shared_mask_loaded(out, shared_mask)
            if not processing_ffi_paths:
                processing_ffi_paths = _ffi_paths_for_processing(cfg)
            round_id = int(stage.get("round_id", 1))
            stream = bool(stage.get("stream_load_rough", False))
            if stream:
                log.info(
                    "  background_estimate: stream_load_rough — load+estimate per frame "
                    "(%d FFIs; avoids holding the full diff/bkg stack in RAM)",
                    len(processing_ffi_paths),
                )
                path_to_group = _path_to_group_from_wcs(wcs_table)
                rough = background.background_loop_streaming(
                    processing_ffi_paths,
                    diff_dir,
                    hp_bkg_dir,
                    path_to_group,
                    shared_mask,
                    output_dir=out_ws,
                    round_id=round_id,
                    recombine_hotpants=cfg.bkg_r1_recombine_hotpants,
                    n_jobs=cfg.n_jobs,
                )
                hp_results = _hotpants_result_stems_for_ordering(
                    processing_ffi_paths, wcs_table, diff_dir
                )
                for _r in hp_results:
                    _r["diff"] = True if _r.get("success") else None
            else:
                log.info(
                    "  background_estimate: loading diff/bkg FITS into memory (%d FFIs) ...",
                    len(processing_ffi_paths),
                )
                hp_results = _hotpants_results_from_dirs(
                    processing_ffi_paths, wcs_table, diff_dir, hp_bkg_dir
                )
                log.info(
                    "  background_estimate: FITS load complete; starting rough per-frame stack"
                )
                rough = background.background_loop(
                    hotpants_results=hp_results,
                    mask=shared_mask,
                    output_dir=out_ws,
                    round_id=round_id,
                    recombine_hotpants=cfg.bkg_r1_recombine_hotpants,
                    n_jobs=cfg.n_jobs,
                )
            if bool(stage.get("write_per_frame_fits", True)):
                _write_per_frame_background_fits(
                    out_ws,
                    rough,
                    hp_results,
                    "{stem}_rough_bkg.fits",
                    row_ok=lambda r: bool(r.get("success")) and r.get("diff") is not None,
                )
                log.info(
                    "  background_estimate: wrote per-frame *_rough_bkg.fits under %s",
                    out_ws,
                )
            _maybe_write_background_gif(
                cfg,
                out,
                rough,
                wcs_table,
                hp_results,
                filename=f"rough_bkg_r{round_id}_animation.gif",
                cbar_label="Rough background (per frame)",
            )
            log.info(
                "  background_estimate: rough stack done shape=%s; adaptive temporal smooth ...",
                getattr(rough, "shape", None),
            )
            _strip_hp_heavy_arrays(hp_results)
            gc.collect()
            bkg_arr = np.array(rough, dtype=np.float32, copy=True)
            _add_hotpants_bkg_to_stack_inplace(bkg_arr, hp_results, hp_bkg_dir)
            temp_smooth = _adaptive_smooth_bkg_stack(bkg_arr, wcs_table, hp_results, cfg)
            log.info(
                "  background_estimate: adaptive smooth returned; writing %s.npz and .npy",
                ADAPTIVE_BKG_STACK_BASENAME,
            )
            background.save_background_stack(
                temp_smooth,
                os.path.join(out_ws, f"{ADAPTIVE_BKG_STACK_BASENAME}.npy"),
            )
            fits_fmt = f"{{stem}}_{ADAPTIVE_BKG_STACK_BASENAME}.fits"
            if bool(stage.get("write_per_frame_fits", True)):
                _write_per_frame_background_fits(
                    out_ws, temp_smooth, hp_results, fits_fmt
                )
                log.info(
                    "  background_estimate: wrote per-frame *_%s.fits under %s",
                    ADAPTIVE_BKG_STACK_BASENAME,
                    out_ws,
                )
            _maybe_write_background_gif(
                cfg,
                out,
                temp_smooth,
                wcs_table,
                hp_results,
                filename=f"{ADAPTIVE_BKG_STACK_BASENAME}_adaptive_animation.gif",
                cbar_label="Adaptive smoothed background",
            )

        elif kind == "forced_photometry":
            inp = stage["inputs"]
            label_out = stage["output"]
            phot_out = ctx.workspace(label_out)
            os.makedirs(phot_out, exist_ok=True)

            if wcs_table is None or crop_bounds is None or not ref_ffi_path:
                raise RuntimeError(
                    "forced_photometry needs WCS/crop state: run wcs_grouping first, "
                    "or keep output_dir's syndiff_ffi_frames.csv and "
                    "cluster_template_job.json from a prior run when omitting "
                    "wcs_grouping from pipeline:."
                )

            if cfg.target_ra is None or cfg.target_dec is None:
                log.warning("target_ra/target_dec not set; skipping forced_photometry.")
                continue

            with fits.open(ref_ffi_path, memmap=True) as hdul:
                ref_wcs = WCS(hdul[1].header)
            target_x, target_y = photometry.get_target_pixel_crop(
                ref_wcs, cfg.target_ra, cfg.target_dec, crop_bounds
            )

            diff_label = inp["diffs"]
            paths_for_phot = ordered_diff_paths_for_workspace(
                wcs_table, out, diff_label, manifest_path
            )

            use_prf = str(stage.get("psf", "")).lower() == "prf"
            if use_prf:
                over_size = 2 * cfg.psf_size + 1
                n_tiles = cfg.tile_ny * cfg.tile_nx
                epsf_for_phot = np.zeros((n_tiles, over_size**2))
            else:
                epsf_ws = ctx.workspace(inp["epsf"])
                epsf_for_phot, _ = temporal_smooth.load_epsf_smooth(epsf_ws, 1)
                if epsf_for_phot.ndim == 3:
                    epsf_for_phot = np.nanmedian(epsf_for_phot, axis=0)

            # Stage ``psf: prf`` only supplies a dummy zero ePSF array; ``build_psf_kernel``
            # must see ``psf_type == "prf"`` or it would wrap zeros in EpsfLocator (NaN PSF,
            # zero flux).
            cfg_phot = replace(cfg, psf_type="prf") if use_prf else cfg

            if tile_centers is None:
                tile_centers = _load_tile_centers_json(out)
            if tile_centers is None and crop_bounds is not None:
                from .epsf_fitting import _make_tile_grid

                ny, nx = crop_bounds["shape"]
                tiles = _make_tile_grid(ny, nx, cfg.tile_ny, cfg.tile_nx)
                tile_centers = [
                    (c0 + ts / 2, r0 + ts / 2) for (r0, c0, ts) in tiles
                ]

            if getattr(cfg, "pipeline_plots", False):
                pdir = _pipeline_plots_root(out, cfg)
                os.makedirs(pdir, exist_ok=True)
                lc_plot_path = os.path.join(
                    pdir, f"lightcurve_{label_out}.png"
                )
            else:
                lc_plot_path = None

            photometry.run_forced_photometry(
                diff_final_paths=paths_for_phot,
                target_x=target_x,
                target_y=target_y,
                epsf_r2_smooth=epsf_for_phot,
                tile_centers=tile_centers,
                wcs_table=wcs_table,
                crop_bounds=crop_bounds,
                cfg=cfg_phot,
                output_dir=phot_out,
                lightcurve_plot_path=lc_plot_path,
                plot_title_suffix=label_out,
            )

        elif kind == "diff_final":
            inp = stage["inputs"]
            label_out = stage["output"]
            diff_ws = ctx.workspace(inp["diffs"])
            hp_results = _hotpants_results_from_dirs(
                processing_ffi_paths, wcs_table, diff_ws, None
            )
            bkg_path = os.path.join(ctx.workspace(inp["bkg_final"]), "bkg_final.npy")
            if os.path.isfile(bkg_path):
                bkg_final = np.load(bkg_path)
            else:
                log.warning("diff_final: no bkg_final.npy at %s; using None", bkg_path)
                bkg_final = None

            sat_ws = ctx.workspace(inp["sat_hr"])
            _, sat_hr_map = sat_template.load_group_templates(sat_ws, round_id=1)

            epsf_ws = ctx.workspace(inp["epsf"])
            epsf_smooth, ffi_stems_epsf = temporal_smooth.load_epsf_smooth(epsf_ws, 1)

            out_img = ctx.workspace(label_out)
            os.makedirs(out_img, exist_ok=True)

            final_diff.final_diff_loop(
                hotpants_results_r2=hp_results,
                bkg_final=bkg_final,
                sat_template_hr_map=sat_hr_map,
                epsf_r2_smooth=epsf_smooth,
                tile_centers=tile_centers or _load_tile_centers_json(out),
                wcs_table=wcs_table,
                mask=shared_mask,
                crop_bounds=crop_bounds,
                cfg=cfg,
                output_dir=out,
                ffi_stems_epsf=ffi_stems_epsf,
                output_images_dir=out_img,
            )

        else:
            raise RuntimeError(f"Unhandled stage kind {kind!r}")

    log.info("=" * 70)
    log.info("Config pipeline complete. Outputs: %s", out)


def _load_group_epsf_from_dir(output_dir: str, subdir: str = "group_epsf") -> dict:
    d = {}
    sub = os.path.join(output_dir, subdir)
    if not os.path.isdir(sub):
        return d
    for path in sorted(glob.glob(os.path.join(sub, "group_epsf_*.npy"))):
        gid = int(os.path.basename(path).replace("group_epsf_", "").replace(".npy", ""))
        d[gid] = np.load(path)
    return d


def _load_removed_stars_in_crop(
    removed_stars_csv: str,
    crop_bounds: dict,
    gaia_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if gaia_df is not None and "x" in gaia_df.columns and "y" in gaia_df.columns:
        return gaia_df.copy()
    if not removed_stars_csv or not os.path.isfile(removed_stars_csv):
        log.warning("removed_stars_csv missing; empty DataFrame for sat templates.")
        return pd.DataFrame()
    df = pd.read_csv(removed_stars_csv)
    df = df.drop_duplicates(subset="source_id")
    df = df[df["source_id"] != -1].copy()
    if "x" not in df.columns:
        return pd.DataFrame()
    ny, nx = crop_bounds["shape"]
    in_crop = (
        (df["x"] >= 0)
        & (df["x"] < nx)
        & (df["y"] >= 0)
        & (df["y"] < ny)
    )
    return df[in_crop].copy().reset_index(drop=True)
