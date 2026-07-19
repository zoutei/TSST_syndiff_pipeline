"""Rebuild ``provenance.db`` from the self-describing on-disk stores.

The database is a *derived* index: every fact in it can be reconstructed by
walking the fingerprinted stores, because each artifact directory carries its
own ``_provenance.json`` (written by :func:`publish.publish_record`). This makes
DB loss a non-event and gives us a one-time offline bootstrap for products that
already exist on disk.

Reconciliation policy (locked decision #7): an artifact directory *with* a
``_provenance.json`` is ingested verbatim as ``complete``. A fingerprinted
directory *without* one (e.g. produced before publish plumbing landed) is
registered as ``legacy_unverified`` — recognized so it is not blindly rebuilt,
but flagged so it can be rebuilt lazily on first real use rather than trusted.

Dependency-free (stdlib) so ``reindex`` can run from the CLI without importing
zarr/astropy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from syndiff_pipeline.common.provenance.publish import (
    STORE_PROVENANCE_BASENAME,
    read_store_provenance,
)
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import (
    ps1_combined_zarr_path,
    ps1_convolved_zarr_path,
    ps1_skycells_zarr_path,
)

logger = logging.getLogger(__name__)

LEGACY_STATE = "legacy_unverified"


def _legacy_record(kind: str, projection: str, skycell: str, fp: str, location: Path) -> dict:
    """Build a ``legacy_unverified`` record for a fingerprinted dir lacking sidecar."""
    return {
        "fingerprint": fp,
        "kind": kind,
        "spatial_key": {"projection": projection, "skycell": skycell},
        "recipe_id": f"legacy:{kind}",
        "recipe": {
            "kind": kind,
            "params": {"legacy_unverified": True},
            "code_version": None,
            "git_sha": None,
        },
        "inputs": [],
        "location": str(location),
        "state": LEGACY_STATE,
        "bytes": None,
        "wall_time_s": None,
        "produced_by": "reindex",
    }


def _iter_fp_dirs(store_path: Path):
    """Yield (projection, skycell, fp_dir) for a ``{proj}/{skycell}/{fp}`` store."""
    if not store_path.is_dir():
        return
    for proj_dir in sorted(p for p in store_path.iterdir() if p.is_dir()):
        for skycell_dir in sorted(p for p in proj_dir.iterdir() if p.is_dir()):
            for fp_dir in sorted(p for p in skycell_dir.iterdir() if p.is_dir()):
                yield proj_dir.name, skycell_dir.name, fp_dir


def reindex_shared_store(
    store: ProvenanceStore, store_path: str | Path, kind: str, *, batch_size: int = 500
) -> Dict[str, int]:
    """Reindex a fingerprint-keyed shared skycell store into *store*.

    Returns counts ``{"sidecar": n, "legacy": n}``. Ingests in batches so a
    very large store does not build one giant transaction.
    """
    store_path = Path(store_path)
    counts = {"sidecar": 0, "legacy": 0}
    batch: List[dict] = []

    def _flush() -> None:
        if batch:
            store.ingest_records(batch)
            batch.clear()

    for projection, skycell, fp_dir in _iter_fp_dirs(store_path):
        record = read_store_provenance(fp_dir)
        if record is not None and record.get("fingerprint"):
            # Trust the self-describing sidecar, but the on-disk directory is the
            # authoritative location (the sidecar may carry a null/placeholder).
            if not record.get("location"):
                record["location"] = str(fp_dir)
            batch.append(record)
            counts["sidecar"] += 1
        else:
            batch.append(_legacy_record(kind, projection, skycell, fp_dir.name, fp_dir))
            counts["legacy"] += 1
        if len(batch) >= batch_size:
            _flush()
    _flush()
    return counts


def reindex_raw_skycells(
    store: ProvenanceStore, zarr_path: str | Path
) -> int:
    """Record present raw skycells as input files (``kind='raw_skycell'``).

    Layout: ``{zarr}/{projection}/{skycell}``. Records presence only (no byte
    sizing — that is an optional, costlier pass). Returns the count recorded.
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.is_dir():
        return 0
    count = 0
    for proj_dir in sorted(p for p in zarr_path.iterdir() if p.is_dir()):
        for skycell_dir in sorted(p for p in proj_dir.iterdir() if p.is_dir()):
            store.upsert_input_file(
                "raw_skycell",
                f"{proj_dir.name}.{skycell_dir.name}",
                {"projection": proj_dir.name, "skycell": skycell_dir.name},
                source="zarr",
            )
            count += 1
    return count


def reindex_data_root(store: ProvenanceStore, data_root: str | Path) -> Dict[str, Dict[str, int]]:
    """Reindex every known store under *data_root*. Returns per-source counts.

    Safe to run repeatedly (ingest is idempotent). Missing stores contribute
    zero — Phase-1/2 stores simply return empties until they exist.
    """
    result: Dict[str, Dict[str, int]] = {}
    result["combined_skycell"] = reindex_shared_store(
        store, ps1_combined_zarr_path(data_root), "combined_skycell"
    )
    result["convolved_skycell"] = reindex_shared_store(
        store, ps1_convolved_zarr_path(data_root), "convolved_skycell"
    )
    result["raw_skycell"] = {"input_files": reindex_raw_skycells(
        store, ps1_skycells_zarr_path(data_root)
    )}
    logger.info("reindex complete under %s: %s", data_root, result)
    return result


__all__ = [
    "LEGACY_STATE",
    "STORE_PROVENANCE_BASENAME",
    "reindex_data_root",
    "reindex_raw_skycells",
    "reindex_shared_store",
]
