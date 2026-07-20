"""PR5 numeric gate: shared vs per-SCC legacy convolved stores (report-only).

Plan §13 ("blocking numeric-equivalence gate") and §17 (testing note): a real
SCC assembled from shared convolved cells must match today's per-SCC
``convolved.zarr`` within tolerance before the shared store can be trusted in
production. This module performs that comparison for real, on-disk data --
it is the human-in-the-loop, real-data companion to the synthetic proof in
``tests/test_convolved_store_padding_decouple.py`` (which proves the snapshot
*logic* is correct under a controlled precondition but uses no real PS1
pixels).

Eligibility (which cells can be compared at all)
-------------------------------------------------
The shared store's canonical cell is defined as same-projection-only
(``convolved_store.PADDING_MODE = "same_projection_only"``); the legacy
per-SCC ``convolved.zarr`` cell is fully padded (same-projection *and*
cross-projection). The two are only expected to match where cross-projection
padding contributed nothing, i.e. cells belonging to a **projection** that
has zero cross-projection padding requirements anywhere in its row set.

This is intentionally **projection-level**, not per-cell: Gaussian
convolution here uses a ~470px truncation radius (``convolved_store.
DEFAULT_RADIUS``), roughly one skycell's ``CELL_OVERLAP``. A cross-projection
padding edit applied to one cell's row-master-array region can blur into a
*neighboring* cell's own crop even when that neighbor has no direct padding
requirement of its own. Filtering only cells that individually lack a
requirement would let that neighbor-bleed contamination silently produce a
false PASS. Filtering at the projection level sidesteps this: if literally no
row in a projection ever needed a cross-projection neighbor,
``apply_cross_projection_padding`` is a no-op for every row in it, so that
projection's row master arrays are byte-identical whether the step "ran" or
not -- same-projection-only canonical and legacy fully-padded are the exact
same computation for every cell in it.

If an SCC has no projection meeting that bar, no valid comparison is
possible and the report says so explicitly with ``pass: False`` -- it does
NOT silently report a stub pass on zero real comparisons (that would
recreate the exact problem this rewrite fixes).

Tolerance
---------
Both sides run the identical ``convolution_utils.apply_gaussian_convolution``
call over identical-projection row master arrays. The one legitimate,
non-buggy source of numeric difference is dtype: ``convolved_store.
publish_convolved_cell`` always downcasts to float32 on save, while the
legacy path's ``convolved_array`` (whatever ``apply_gaussian_convolution``
returns for a float32 input) is saved as-is by ``zarr_utils.
save_convolved_results``. Default tolerance
(``rtol=1e-5``, ``atol=1e-3``) is float32 machine epsilon (~1.2e-7) scaled up
roughly two orders of magnitude to cover accumulated rounding across a
~940px-diameter kernel, plus a small absolute floor for near-zero background
pixels where a pure relative tolerance is too strict. Both are exposed as
parameters so an operator can tighten/loosen them against a specific SCC's
real flux scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def _cross_projection_padding_free_cells(
    data_root: str | Path,
    sector: int,
    camera: int,
    ccd: int,
    *,
    projections_limit: Optional[int] = None,
) -> tuple[set[str], dict]:
    """Skycell names (``skycell.PROJ.CELL``) whose **projection** has zero
    cross-projection padding requirements anywhere in its row set, for this
    SCC. See the module docstring for why this is projection-level, not
    per-cell.

    Returns ``(eligible_cells, diagnostics)`` where diagnostics lists which
    projections were judged clean vs dirty (useful for a human operator
    auditing *why* a given cell was or wasn't compared).
    """
    from syndiff_pipeline.template_creation.processing.csv_utils import (
        find_csv_file,
        get_projections_from_csv,
        load_csv_data,
    )
    from syndiff_pipeline.template_creation.processing.cross_projection_padding import (
        identify_all_padding_sources,
    )
    from syndiff_pipeline.template_creation.processing.ps1_process import (
        extract_projection_metadata,
    )

    csv_path = find_csv_file(str(data_root), sector, camera, ccd)
    projections = get_projections_from_csv(csv_path)
    if projections_limit:
        projections = projections[: int(projections_limit)]
    df = load_csv_data(csv_path)

    _source_cells, _uses, row_padding_map = identify_all_padding_sources(projections, df, csv_path)

    clean_projections: list[str] = []
    dirty_projections: list[str] = []
    eligible: set[str] = set()

    for projection in projections:
        metadata = extract_projection_metadata(df, projection)
        row_ids = list(metadata["rows"].keys())
        has_padding = any(
            row_padding_map.get((str(projection), row_id)) for row_id in row_ids
        )
        if has_padding:
            dirty_projections.append(projection)
            continue
        clean_projections.append(projection)
        for cells in metadata["rows"].values():
            for cell_name, _x_coord in cells:
                eligible.add(cell_name)

    diagnostics = {
        "clean_projections": sorted(clean_projections),
        "dirty_projections": sorted(dirty_projections),
    }
    return eligible, diagnostics


def convolved_gate_report(
    data_root: str | Path,
    *,
    sector: int = 20,
    camera: int = 3,
    ccd: int = 3,
    sample_cells: int = 5,
    projections_limit: Optional[int] = None,
    rtol: float = 1e-5,
    atol: float = 1e-3,
) -> dict:
    data_root = Path(data_root).expanduser()
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path, scc_convolved_zarr
    from syndiff_pipeline.template_creation.processing.combined_store import _projection_and_cell

    legacy_path = scc_convolved_zarr(data_root, sector, camera, ccd)
    shared_root = ps1_convolved_zarr_path(data_root)

    report: dict = {
        "sector": sector,
        "camera": camera,
        "ccd": ccd,
        "legacy_path": str(legacy_path),
        "shared_root": str(shared_root),
        "legacy_exists": legacy_path.exists(),
        "shared_exists": shared_root.is_dir(),
        "tolerance": {"rtol": rtol, "atol": atol},
        "compared": [],
        "failures": [],
        "pass": False,
    }

    if not legacy_path.exists():
        report["error"] = "legacy convolved.zarr missing"
        return report
    if not shared_root.is_dir():
        report["error"] = "shared ps1_convolved.zarr missing"
        return report

    try:
        eligible_cells, diagnostics = _cross_projection_padding_free_cells(
            data_root, sector, camera, ccd, projections_limit=projections_limit
        )
    except Exception as exc:
        report["error"] = f"failed to compute cross-projection-padding-free cell set: {exc}"
        return report

    report["cross_projection_padding_diagnostics"] = diagnostics
    report["eligible_cell_count"] = len(eligible_cells)

    if not eligible_cells:
        report["note"] = (
            "No projection in this SCC is free of cross-projection padding "
            "requirements (every projection has at least one row that needs a "
            "cross-projection neighbor); a same-projection-only vs fully-padded "
            "numeric comparison is not meaningful for any cell here without "
            "applying the seam correction first (plan §13, decision #4). "
            "pass=False rather than a stub pass on zero real comparisons."
        )
        return report

    try:
        import zarr

        legacy_store = zarr.open(str(legacy_path), mode="r")
    except Exception as exc:
        report["error"] = f"failed to open legacy convolved.zarr: {exc}"
        return report

    matched: list[tuple[str, str, str, Path]] = []
    for full_name in sorted(eligible_cells):
        parsed = _projection_and_cell(full_name)
        if parsed is None:
            continue
        projection, cell = parsed
        cell_root = shared_root / projection / cell
        if not cell_root.is_dir():
            continue
        legacy_key = f"{full_name}_data"
        if legacy_key not in legacy_store:
            continue
        # A cell may have >1 published fingerprint dir across recipe/param
        # changes over time; compare against every one still on disk.
        for fp_dir in sorted(d for d in cell_root.iterdir() if d.is_dir()):
            matched.append((full_name, projection, cell, fp_dir))

    report["matched_cell_count"] = len(matched)
    if not matched:
        report["note"] = (
            "No cross-projection-padding-free skycell in this SCC has both a "
            "legacy convolved.zarr entry and a shared-store entry -- nothing to "
            "compare. pass=False (not a silent stub pass)."
        )
        return report

    sample = matched[: max(1, int(sample_cells))]
    ok = True

    for full_name, projection, cell, fp_dir in sample:
        arrays_path = fp_dir / "arrays.npz"
        try:
            with np.load(arrays_path) as z:
                shared_arr = np.asarray(z["convolved_image"], dtype=np.float64)
        except Exception as exc:
            entry = {"cell": full_name, "fp_dir": str(fp_dir), "error": f"failed to load shared array: {exc}"}
            report["failures"].append(entry)
            ok = False
            continue

        try:
            legacy_arr = np.asarray(legacy_store[f"{full_name}_data"][:], dtype=np.float64)
        except Exception as exc:
            entry = {"cell": full_name, "fp_dir": str(fp_dir), "error": f"failed to load legacy array: {exc}"}
            report["failures"].append(entry)
            ok = False
            continue

        if legacy_arr.shape != shared_arr.shape:
            entry = {
                "cell": full_name,
                "fp_dir": str(fp_dir),
                "error": f"shape mismatch: legacy {legacy_arr.shape} vs shared {shared_arr.shape}",
            }
            report["failures"].append(entry)
            ok = False
            continue

        legacy_nan = np.isnan(legacy_arr)
        shared_nan = np.isnan(shared_arr)
        nan_mismatch = int(np.count_nonzero(legacy_nan != shared_nan))

        valid = ~legacy_nan & ~shared_nan
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            entry = {
                "cell": full_name,
                "fp_dir": str(fp_dir),
                "error": "no overlapping valid (non-NaN) pixels to compare",
            }
            report["failures"].append(entry)
            ok = False
            continue

        diff = np.abs(legacy_arr[valid] - shared_arr[valid])
        max_abs_diff = float(np.max(diff))
        close = bool(np.allclose(legacy_arr[valid], shared_arr[valid], rtol=rtol, atol=atol))

        entry = {
            "cell": full_name,
            "fp_dir": str(fp_dir),
            "shape": list(shared_arr.shape),
            "n_compared_pixels": n_valid,
            "n_nan_mask_mismatch": nan_mismatch,
            "max_abs_diff": max_abs_diff,
            "close": close,
        }
        report["compared"].append(entry)

        if not close or nan_mismatch > 0:
            ok = False
            failure = dict(entry)
            if nan_mismatch > 0:
                failure["error"] = (
                    f"{nan_mismatch} pixel(s) disagree on NaN-ness between legacy "
                    f"and shared arrays"
                )
            report["failures"].append(failure)

    report["pass"] = ok and bool(report["compared"])
    report["note"] = (
        f"Real numeric comparison over {len(report['compared'])} "
        f"cross-projection-padding-free cell(s) sampled from "
        f"{len(matched)} matched (legacy+shared both present), out of "
        f"{len(eligible_cells)} eligible cells in this SCC's padding-free "
        f"projections. See 'cross_projection_padding_diagnostics' for which "
        f"projections were excluded and why."
    )
    return report


def format_convolved_gate(data_root: str | Path, **kwargs) -> str:
    return json.dumps(convolved_gate_report(data_root, **kwargs), indent=2)
