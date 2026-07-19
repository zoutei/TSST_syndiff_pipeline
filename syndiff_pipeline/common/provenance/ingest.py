"""Supervisor-side ingest: drain worker sidecar spool files into the index.

Workers publish artifacts atomically to fingerprinted keys and append one JSON
line per artifact to a per-host spool file (see ``publish.py``). The supervisor
is the **sole writer** of ``provenance.db``; it periodically drains the spool
into the store via :func:`drain_spool`.

Design (see ``doc/template_bookkeeping_plan.md`` §10):

* **Rotate-then-read.** Each spool file is atomically renamed to a unique
  ``*.draining-<ts>`` name before it is read, so a worker that appends again
  after the rotate creates a fresh spool file (open/append/close per record in
  ``publish.py``) and never races the drain.
* **Lenient parsing.** A malformed trailing line (e.g. a partial append over
  NFS) is skipped with a warning, not fatal. Content authority means a lost
  sidecar only delays the index; ``reindex`` reconciles it, and the artifact
  bytes already exist at their fingerprinted key.
* **Dependency-free.** stdlib only; imports the stdlib-only ``store`` and the
  pathlib-only ``scc_paths``. Safe to run inside the daemon loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List

from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_spool_dir

logger = logging.getLogger(__name__)

# Only rotate spool files that look like live worker logs.
_SPOOL_GLOB = "*.jsonl"
_DRAINING_SUFFIX = ".draining"


def _rotate(spool_file: Path) -> Path | None:
    """Atomically rename *spool_file* aside for draining; None if it vanished.

    A unique timestamp+pid suffix avoids collisions when multiple drains or
    hosts touch the directory. The rename is atomic on the same filesystem, so
    a worker appending concurrently either wrote before the rename (captured
    here) or recreates the original path on its next open/append/close.
    """
    target = spool_file.with_name(
        f"{spool_file.name}{_DRAINING_SUFFIX}-{int(time.time()*1000)}.{os.getpid()}"
    )
    try:
        os.replace(spool_file, target)
    except FileNotFoundError:
        return None
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("provenance ingest: could not rotate %s: %s", spool_file, exc)
        return None
    return target


def _parse_records(path: Path) -> tuple[List[dict], int]:
    """Return (valid records, skipped-line count) from a rotated spool file.

    Each non-blank line is one JSON record. Malformed lines are skipped and
    counted, never raised — a partial NFS append must not stall the daemon.
    """
    records: List[dict] = []
    skipped = 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("provenance ingest: could not read %s: %s", path, exc)
        return records, skipped
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(rec, dict) and rec.get("fingerprint") and rec.get("kind"):
            records.append(rec)
        else:
            skipped += 1
    return records, skipped


def drain_spool(store: ProvenanceStore, spool_dir: str | Path) -> int:
    """Drain every spool file under *spool_dir* into *store*; return records ingested.

    Called by the supervisor. Rotates each ``*.jsonl`` file aside, ingests its
    records in one batch, then removes the rotated file. Idempotent at the
    record level (``ingest_records`` upserts), so a crash between ingest and
    unlink at worst re-ingests identical rows on the next drain.
    """
    spool_path = Path(spool_dir)
    if not spool_path.is_dir():
        return 0
    total = 0
    for spool_file in sorted(spool_path.glob(_SPOOL_GLOB)):
        rotated = _rotate(spool_file)
        if rotated is None:
            continue
        records, skipped = _parse_records(rotated)
        if skipped:
            logger.warning(
                "provenance ingest: skipped %d malformed line(s) in %s",
                skipped,
                spool_file.name,
            )
        if records:
            try:
                total += store.ingest_records(records)
            except Exception as exc:  # pragma: no cover - defensive
                # Keep the rotated file for a later retry rather than losing it.
                logger.error(
                    "provenance ingest: ingest failed for %s (kept for retry): %s",
                    rotated.name,
                    exc,
                )
                continue
        try:
            rotated.unlink()
        except OSError:  # pragma: no cover - defensive
            pass
    return total


def drain_data_root(store: ProvenanceStore, data_root: str | Path) -> int:
    """Convenience: drain the canonical spool dir under *data_root*."""
    return drain_spool(store, provenance_spool_dir(data_root))
