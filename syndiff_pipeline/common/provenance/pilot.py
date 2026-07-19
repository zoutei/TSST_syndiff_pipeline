"""Phase-5 dual-write pilot checklist (report-only)."""

from __future__ import annotations

import json
from pathlib import Path


def pilot_checklist_report(data_root: str | Path) -> dict:
    data_root = Path(data_root).expanduser()
    checks: dict[str, dict] = {}
    ok = True

    from syndiff_pipeline.common.provenance.store import ProvenanceStore
    from syndiff_pipeline.common.scc_paths import provenance_db_path, provenance_spool_dir

    db_path = provenance_db_path(data_root)
    spool_dir = provenance_spool_dir(data_root)
    checks["provenance_db_exists"] = {"pass": db_path.is_file(), "detail": str(db_path)}
    if not db_path.is_file():
        ok = False

    if db_path.is_file():
        store = ProvenanceStore(db_path, read_only=True)
        stats = store.stats()
        checks["db_row_counts"] = {"pass": True, "detail": stats}

    spool_files = list(spool_dir.glob("*.jsonl")) if spool_dir.is_dir() else []
    spool_bytes = sum(p.stat().st_size for p in spool_files)
    checks["spool_not_backing_up"] = {
        "pass": spool_bytes < 50_000_000,
        "detail": f"{len(spool_files)} spool files, {spool_bytes} bytes under {spool_dir}",
    }
    if spool_bytes >= 50_000_000:
        ok = False

    combined_root = data_root / "ps1_skycells_zarr" / "ps1_combined.zarr"
    checks["combined_store_present"] = {
        "pass": combined_root.is_dir(),
        "detail": str(combined_root),
    }

    pilot_scc = data_root / "s0020" / "c3" / "k3"
    checks["pilot_scc_tree"] = {"pass": pilot_scc.is_dir(), "detail": str(pilot_scc)}

    return {
        "data_root": str(data_root),
        "go": ok,
        "checks": checks,
        "manual_ops": [
            "Confirm ps1_process verify hits checkpoint path on a completed SCC",
            "Confirm diff per-FFI rows appear after a small finalized diff run",
            "Enable publish_scc on one SCC only after D2 confidence",
            "Run syndiff bookkeeping convolved-gate before PR5 write cutover",
        ],
    }


def format_pilot_checklist(data_root: str | Path) -> str:
    return json.dumps(pilot_checklist_report(data_root), indent=2)
