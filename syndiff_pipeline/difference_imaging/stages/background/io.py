"""Load/save background stacks and flux cubes for the unified background stage."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.fits_io import write_image_fits
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    parse_workspace_frame_stem,
    resolve_pipeline_fits_path,
    tess_product_id_from_ffi_path,
    workspace_frame_fits_basename,
    workspace_frame_stem,
    workspace_label_from_dir,
)
from syndiff_pipeline.difference_imaging.support.paths import DIFF_CONFIG_SNAPSHOT_BASENAME
from syndiff_pipeline.difference_imaging.support.paths import BACKGROUND_STACK_NPZ_ARRAY_KEY

log = logging.getLogger(__name__)

STACK_BASENAME = "stack"


@dataclass(frozen=True)
class FrameRecord:
    """FrameRecord."""
    index: int
    product_id: str
    stem: str
    diff_path: str
    bkg_path: Optional[str] = None
    success: bool = True


def load_stack(dir_path: str, *, basename: str = STACK_BASENAME) -> np.ndarray:
    """Load stack.
    
    Parameters
    ----------
    dir_path : str
    basename : str, optional, default ``STACK_BASENAME``
    
    Returns
    -------
    np.ndarray"""
    npz_path = os.path.join(dir_path, f"{basename}.npz")
    npy_path = os.path.join(dir_path, f"{basename}.npy")
    if os.path.isfile(npz_path):
        z = np.load(npz_path, mmap_mode="r")
        if BACKGROUND_STACK_NPZ_ARRAY_KEY not in z.files:
            raise KeyError(
                f"{npz_path!r} missing {BACKGROUND_STACK_NPZ_ARRAY_KEY!r}; "
                f"have {list(z.files)}"
            )
        return np.asarray(z[BACKGROUND_STACK_NPZ_ARRAY_KEY])
    if os.path.isfile(npy_path):
        return np.asarray(np.load(npy_path, mmap_mode="r"))
    raise FileNotFoundError(
        f"missing stack under {dir_path!r}: expected {basename}.npz or {basename}.npy"
    )


def save_stack(stack: np.ndarray, dir_path: str, *, basename: str = STACK_BASENAME) -> None:
    """Save stack.
    
    Parameters
    ----------
    stack : np.ndarray
    dir_path : str
    basename : str, optional, default ``STACK_BASENAME``"""
    os.makedirs(dir_path, exist_ok=True)
    arr = np.asarray(stack, dtype=np.float32)
    npz_path = os.path.join(dir_path, f"{basename}.npz")
    npy_path = os.path.join(dir_path, f"{basename}.npy")
    np.savez(npz_path, **{BACKGROUND_STACK_NPZ_ARRAY_KEY: arr})
    np.save(npy_path, arr)
    log.info("Background stack saved to %s and %s", npz_path, npy_path)


def stack_from_bkg_records(records: List[FrameRecord]) -> np.ndarray:
    """Build (T, ny, nx) cube from per-frame background FITS in *records*."""
    shape = None
    for rec in records:
        path = rec.bkg_path or rec.diff_path
        if rec.success and path and os.path.isfile(path):
            shape = fits.getdata(path, memmap=True).shape
            break
    if shape is None:
        raise RuntimeError("stack_from_bkg_records: no readable FITS in records.")
    stack = np.zeros((len(records), *shape), dtype=np.float32)
    for i, rec in enumerate(records):
        path = rec.bkg_path or rec.diff_path
        if not rec.success or not path or not os.path.isfile(path):
            continue
        stack[i] = fits.getdata(path).astype(np.float32)
    return stack


def load_stack_or_fits(
    dir_path: str,
    records: List[FrameRecord],
    *,
    basename: str = STACK_BASENAME,
) -> np.ndarray:
    """Load ``stack.npz`` / ``stack.npy`` or stack per-frame FITS under *dir_path*."""
    try:
        return load_stack(dir_path, basename=basename)
    except FileNotFoundError:
        return stack_from_bkg_records(records)


def _row_from_paths(
    product_id: str,
    diff_dir: str,
    diff_label: str,
    bkg_dir: Optional[str],
    bkg_label: Optional[str],
) -> FrameRecord:
    """Row from paths.
    
    Parameters
    ----------
    product_id : str
    diff_dir : str
    diff_label : str
    bkg_dir : Optional[str]
    bkg_label : Optional[str]
    
    Returns
    -------
    FrameRecord"""
    diff_stem = workspace_frame_stem(product_id, diff_label)
    diff_path = resolve_pipeline_fits_path(diff_dir, diff_stem)
    ok = diff_path is not None
    bkg_path = None
    if bkg_dir and bkg_label:
        bkg_stem = workspace_frame_stem(product_id, bkg_label)
        bkg_path = resolve_pipeline_fits_path(bkg_dir, bkg_stem)
    return FrameRecord(
        index=0,
        product_id=product_id,
        stem=diff_stem,
        diff_path=diff_path or "",
        bkg_path=bkg_path,
        success=ok,
    )


def build_frame_records(
    ffi_paths: List[str],
    wcs_table: pd.DataFrame,
    diff_dir: str,
    bkg_dir: Optional[str] = None,
) -> List[FrameRecord]:
    """Build frame records.
    
    Parameters
    ----------
    ffi_paths : List[str]
    wcs_table : pd.DataFrame
    diff_dir : str
    bkg_dir : Optional[str], optional, default ``None``
    
    Returns
    -------
    List[FrameRecord]"""
    path_to_group = {}
    if wcs_table is not None:
        col = "path" if "path" in wcs_table.columns else "filename"
        if col in wcs_table.columns:
            for _, row in wcs_table.iterrows():
                pid = tess_product_id_from_ffi_path(str(row[col]))
                if pid:
                    path_to_group[pid] = int(row.get("group_id", 0))

    diff_label = workspace_label_from_dir(diff_dir)
    bkg_label = workspace_label_from_dir(bkg_dir) if bkg_dir else None
    records: List[FrameRecord] = []
    for ffi_path in ffi_paths:
        pid = tess_product_id_from_ffi_path(ffi_path)
        if not pid:
            continue
        rec = _row_from_paths(pid, diff_dir, diff_label, bkg_dir, bkg_label)
        records.append(
            FrameRecord(
                index=len(records),
                product_id=rec.product_id,
                stem=rec.stem,
                diff_path=rec.diff_path,
                bkg_path=rec.bkg_path,
                success=rec.success,
            )
        )
    if not records:
        raise RuntimeError(f"No diff FITS found under {diff_dir!r}")
    return records


def build_frame_records_from_stack_ws(
    ffi_paths: List[str],
    stack_ws_dir: str,
) -> List[FrameRecord]:
    """Order frame records to match ``ffi_paths`` using FITS under a background workspace."""
    label = workspace_label_from_dir(stack_ws_dir)
    records: List[FrameRecord] = []
    for ffi_path in ffi_paths:
        pid = tess_product_id_from_ffi_path(ffi_path)
        if not pid:
            continue
        stem = workspace_frame_stem(pid, label)
        fp = resolve_pipeline_fits_path(stack_ws_dir, stem)
        records.append(
            FrameRecord(
                index=len(records),
                product_id=pid,
                stem=stem,
                diff_path="",
                bkg_path=fp,
                success=fp is not None,
            )
        )
    if not records:
        raise RuntimeError(f"No FITS found under background workspace {stack_ws_dir!r}")
    return records


def load_flux_cubes(
    records: List[FrameRecord],
    *,
    recombine_inputs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(fit_flux, strap_flux)`` cubes shaped (T, ny, nx).

    ``fit_flux`` may include Hotpants bkg when ``recombine_inputs`` is True.
    ``strap_flux`` is always raw diff only (for calc_qe).
    """
    shape = None
    for rec in records:
        if rec.success and rec.diff_path and os.path.isfile(rec.diff_path):
            shape = fits.getdata(rec.diff_path, memmap=True).shape
            break
    if shape is None:
        raise RuntimeError("load_flux_cubes: no readable diff FITS in records.")

    t = len(records)
    fit_cube = np.zeros((t, *shape), dtype=np.float64)
    strap_cube = np.zeros((t, *shape), dtype=np.float64)

    for i, rec in enumerate(records):
        if not rec.success or not rec.diff_path or not os.path.isfile(rec.diff_path):
            continue
        diff = fits.getdata(rec.diff_path).astype(np.float64)
        strap_cube[i] = diff
        fit = diff.copy()
        if recombine_inputs and rec.bkg_path and os.path.isfile(rec.bkg_path):
            fit += fits.getdata(rec.bkg_path).astype(np.float64)
        fit_cube[i] = fit

    return fit_cube, strap_cube


def _event_dirs_from_out_ws(out_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Return ``(event_dir, ws_tree)`` from a label workspace path, or ``(None, None)``."""
    ws_label_dir = os.path.abspath(out_dir)
    ws_tree = os.path.dirname(ws_label_dir)
    if not os.path.basename(ws_tree).startswith("ws"):
        return None, None
    return os.path.dirname(ws_tree), ws_tree


def _diff_image_fingerprint_from_store(
    *,
    sck: tuple[int, int, int],
    product_id: str,
    diff_label: str,
    data_root: str,
) -> Optional[str]:
    """Indexed ``diff_image`` fingerprint for one frame, or ``None`` when ambiguous."""
    store = provenance_glue.open_store(str(data_root))
    if store is None:
        return None
    spatial = provenance_glue.ffi_spatial_key(
        sck[0], sck[1], sck[2], product_id, diff_label
    )
    rows = store.artifacts_by_kind_spatial("diff_image", spatial)
    complete = [r for r in rows if getattr(r, "state", "complete") == "complete"]
    if len(complete) == 1:
        return str(complete[0].fingerprint)
    if len(complete) > 1:
        log.debug(
            "background provenance: %d complete diff_image rows for %s/%s; skip emit",
            len(complete),
            product_id,
            diff_label,
        )
    return None


def _diff_image_fingerprint_from_frozen_config(
    *,
    rec: FrameRecord,
    sck: tuple[int, int, int],
    diff_label: str,
    out_dir: str,
    data_root: Optional[str],
) -> Optional[str]:
    """Reconstruct ``diff_image`` fp from frozen ``ws/diff_config.yaml`` (no ``loc:``)."""
    event_dir, ws_tree = _event_dirs_from_out_ws(out_dir)
    if not event_dir or not ws_tree:
        return None
    cfg_path = os.path.join(ws_tree, DIFF_CONFIG_SNAPSHOT_BASENAME)
    if not os.path.isfile(cfg_path):
        return None
    try:
        from syndiff_pipeline.difference_imaging.orchestration.config import load_config
        from syndiff_pipeline.difference_imaging.orchestration import diff_verify as dv

        cfg = load_config(cfg_path)
        if data_root and not getattr(cfg, "data_root", ""):
            cfg.data_root = str(data_root)
        frames_df = dv.load_diff_frames_for_verify(cfg, event_dir)
        diff_stage = dv._upstream_diff_image_stage(cfg, {"inputs": {"diffs": diff_label}})
        if diff_stage is None:
            return None
        downsample_fp = provenance_glue.resolve_downsample_fingerprint_from_cfg(cfg)
        return dv._diff_image_fingerprint_for_product(
            cfg,
            frames_df,
            rec.product_id,
            diff_stage,
            downsample_fp=downsample_fp,
        )
    except Exception:
        log.debug(
            "background provenance: diff_image reconstruct failed for %s",
            rec.product_id,
            exc_info=True,
        )
        return None


def _resolve_diff_image_fingerprint_for_background(
    *,
    rec: FrameRecord,
    sck: tuple[int, int, int],
    data_root: Optional[str],
    out_dir: str,
) -> Optional[str]:
    """Real upstream ``diff_image`` fingerprint for standalone background emit."""
    if not rec.diff_path:
        return None
    parsed = parse_workspace_frame_stem(rec.stem)
    if parsed:
        _product_id, diff_label = parsed
    else:
        diff_label = workspace_label_from_dir(os.path.dirname(rec.diff_path))
        if not diff_label:
            return None
    fp = _diff_image_fingerprint_from_frozen_config(
        rec=rec,
        sck=sck,
        diff_label=diff_label,
        out_dir=out_dir,
        data_root=data_root,
    )
    if fp is not None:
        return fp
    if data_root:
        return _diff_image_fingerprint_from_store(
            sck=sck,
            product_id=rec.product_id,
            diff_label=diff_label,
            data_root=str(data_root),
        )
    return None


def write_per_frame_fits(
    out_dir: str,
    stack: np.ndarray,
    records: List[FrameRecord],
    *,
    sck: Optional[tuple] = None,
    data_root: Optional[str] = None,
    background_params: Optional[Any] = None,
    workspace_root: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Write per frame fits.

    Parameters
    ----------
    out_dir : str
    stack : np.ndarray
    records : List[FrameRecord]
    sck : Optional[tuple], optional
        ``(sector, camera, ccd)``; when set (with *data_root* and
        *background_params*), emits a best-effort ``diff_background``
        provenance sidecar per frame (PR-D1, never changes what/where is
        written; see ``orchestration/provenance_glue.py``).
    data_root : Optional[str], optional
    background_params : Optional[Any], optional
        The stage's ``BackgroundParams`` (recipe source for the sidecar).
    run_id : Optional[str], optional
        Orchestrator run id, stamped into the emitted artifact's ``meta``
        when non-empty. Not yet fed by any caller (``SynDiffConfig.run_id``
        lands in a later wave); accepted now so the wiring is a no-op change
        once it does."""
    os.makedirs(out_dir, exist_ok=True)
    out_label = workspace_label_from_dir(out_dir)
    for i, rec in enumerate(records):
        if i >= stack.shape[0] or not rec.success:
            continue
        stem = workspace_frame_stem(rec.product_id, out_label)
        fn = workspace_frame_fits_basename(stem)
        out_path = os.path.join(out_dir, fn)
        write_image_fits(
            out_path,
            np.asarray(stack[i], dtype=np.float32),
        )
        if sck is not None and background_params is not None:
            try:
                # Standalone background edges on the upstream diff_image node
                # (hotpants-internal bkg uses ffi instead — see
                # diff_background_input_fingerprints).
                diff_image_fp = _resolve_diff_image_fingerprint_for_background(
                    rec=rec,
                    sck=sck,
                    data_root=data_root,
                    out_dir=out_dir,
                )
                inputs = provenance_glue.required_input_fingerprints(diff_image_fp)
                if inputs is None:
                    continue
                meta = {"producer": "background_temporal_smoothing"}
                if run_id:
                    meta["run_id"] = run_id
                provenance_glue.emit_diff_artifact(
                    kind="diff_background",
                    sector=sck[0],
                    camera=sck[1],
                    ccd=sck[2],
                    product_id=rec.product_id,
                    label=out_label,
                    params=background_params,
                    location=out_path,
                    input_fingerprints=inputs,
                    data_root=data_root,
                    meta=meta,
                    workspace_root=workspace_root,
                )
            except Exception:
                log.debug(
                    "provenance emit (diff_background/background_temporal_smoothing) failed for %s",
                    rec.product_id,
                    exc_info=True,
                )
