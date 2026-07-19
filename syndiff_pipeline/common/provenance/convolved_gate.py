"""PR5 numeric gate helper: shared vs per-SCC convolved stores (report-only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def convolved_gate_report(
    data_root: str | Path,
    *,
    sector: int = 20,
    camera: int = 3,
    ccd: int = 3,
    sample_cells: int = 5,
) -> dict:
    data_root = Path(data_root).expanduser()
    from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path, scc_convolved_zarr

    legacy = scc_convolved_zarr(data_root, sector, camera, ccd)
    shared = ps1_convolved_zarr_path(data_root)

    report: dict = {
        "legacy_path": str(legacy),
        "shared_root": str(shared),
        "legacy_exists": legacy.exists(),
        "shared_exists": shared.is_dir(),
        "samples": [],
        "pass": False,
    }

    if not legacy.exists():
        report["error"] = "legacy convolved.zarr missing"
        return report
    if not shared.is_dir():
        report["error"] = "shared ps1_convolved.zarr missing"
        return report

    shared_dirs = list(shared.glob("*/*/*"))[: max(1, sample_cells)]
    ok = True
    for cell_dir in shared_dirs:
        try:
            with np.load(cell_dir / "arrays.npz") as z:
                shared_arr = np.asarray(z["convolved_image"], dtype=np.float64)
        except Exception as exc:
            report["samples"].append({"dir": str(cell_dir), "error": str(exc)})
            ok = False
            continue
        report["samples"].append(
            {
                "dir": str(cell_dir),
                "shared_shape": list(shared_arr.shape),
                "shared_finite_mean": float(np.nanmean(shared_arr)),
            }
        )

    report["pass"] = ok and bool(report["samples"])
    report["note"] = (
        "Extend with SCC-specific legacy/shared key pairing before write cutover."
    )
    return report


def format_convolved_gate(data_root: str | Path, **kwargs) -> str:
    return json.dumps(convolved_gate_report(data_root, **kwargs), indent=2)
