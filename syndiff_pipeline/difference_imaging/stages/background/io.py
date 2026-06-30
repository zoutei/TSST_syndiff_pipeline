"""Load/save background stacks and flux cubes for the unified background stage."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    resolve_pipeline_fits_path,
    tess_product_id_from_ffi_path,
    workspace_frame_fits_basename,
    workspace_frame_stem,
    workspace_label_from_dir,
)
from syndiff_pipeline.difference_imaging.support.paths import BACKGROUND_STACK_NPZ_ARRAY_KEY

log = logging.getLogger(__name__)

STACK_BASENAME = "stack"


@dataclass(frozen=True)
class FrameRecord:
    index: int
    product_id: str
    stem: str
    diff_path: str
    bkg_path: Optional[str] = None
    success: bool = True


def load_stack(dir_path: str, *, basename: str = STACK_BASENAME) -> np.ndarray:
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


def write_per_frame_fits(
    out_dir: str,
    stack: np.ndarray,
    records: List[FrameRecord],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_label = workspace_label_from_dir(out_dir)
    for i, rec in enumerate(records):
        if i >= stack.shape[0] or not rec.success:
            continue
        stem = workspace_frame_stem(rec.product_id, out_label)
        fn = workspace_frame_fits_basename(stem)
        fits.writeto(
            os.path.join(out_dir, fn),
            np.asarray(stack[i], dtype=np.float32),
            overwrite=True,
        )
