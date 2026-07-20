"""
ingest.py
=========
Supervisor-side spool drain (§10). The supervisor is the **sole writer** of
``provenance.db``; every producer only ever appends to its own lock-free
spool file (:mod:`publish`). ``drain_spool`` is called once per supervisor
loop pass (alongside ``write_verify_in_flight``, per §10/§15) to fold spool
records into the DB.

Rotation protocol per file: rename ``{host}.{pid}.jsonl`` ->
``{host}.{pid}.jsonl.draining`` (an atomic, in-place rename -- the producer
that owns that pid may still be appending to a *new* file of the same base
name concurrently, which is fine: the rename detaches the bytes already
written from any future append), read + parse the rotated file, ingest all
records in **one transaction**, then delete the rotated file. If the process
crashes between rotate and delete, the next drain pass finds the
``.draining`` file (not a fresh ``.jsonl``) and resumes from it -- ingestion
is idempotent (``INSERT OR REPLACE`` artifacts, ``INSERT OR IGNORE``
recipes/edges), so re-ingesting a partially-processed file is always safe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from syndiff_pipeline.common.provenance.store import ProvenanceStore

log = logging.getLogger(__name__)

__all__ = ["DrainResult", "rotate_spool_files", "drain_spool"]

DRAINING_SUFFIX = ".draining"


class DrainResult:
    """Summary of one :func:`drain_spool` call."""

    def __init__(self) -> None:
        self.files_drained: int = 0
        self.records_ingested: int = 0
        self.records_skipped: int = 0
        self.errors: list[str] = []

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return (
            f"DrainResult(files_drained={self.files_drained}, "
            f"records_ingested={self.records_ingested}, "
            f"records_skipped={self.records_skipped}, "
            f"errors={len(self.errors)})"
        )


def rotate_spool_files(spool_dir: str | Path) -> list[Path]:
    """
    Rename every live ``*.jsonl`` spool file to ``*.jsonl.draining``.

    Returns the list of rotated (``.draining``) paths, including any that
    were already ``.draining`` from a previous, interrupted drain pass.
    Safe to call with no live producers (returns existing ``.draining``
    files only).
    """
    spool_dir = Path(spool_dir)
    if not spool_dir.is_dir():
        return []
    rotated: list[Path] = []
    for path in sorted(spool_dir.glob("*.jsonl")):
        draining = path.with_name(path.name + DRAINING_SUFFIX)
        try:
            path.rename(draining)
        except FileNotFoundError:  # pragma: no cover - raced with itself, ignore
            continue
        rotated.append(draining)
    # Pick up any left over from a prior crash between rotate and delete.
    for path in sorted(spool_dir.glob(f"*.jsonl{DRAINING_SUFFIX}")):
        if path not in rotated:
            rotated.append(path)
    return rotated


def _iter_records(path: Path) -> Iterable[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("provenance ingest: could not read %s: %s", path, exc)
        return
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning(
                "provenance ingest: skipping malformed spool line %s:%d: %s",
                path,
                lineno,
                exc,
            )


def drain_spool(store: ProvenanceStore, spool_dir: str | Path) -> DrainResult:
    """
    Rotate + ingest every spool file into *store* in one transaction per file,
    then delete the rotated file. Idempotent: safe to call repeatedly, safe
    to re-run after a crash mid-drain.
    """
    result = DrainResult()
    rotated = rotate_spool_files(spool_dir)
    for path in rotated:
        records = list(_iter_records(path))
        try:
            with store._lock, store._conn() as conn:  # noqa: SLF001 - ingest owns the writer lock
                for record in records:
                    ok = _ingest_record_with_conn(conn, record)
                    if ok:
                        result.records_ingested += 1
                    else:
                        result.records_skipped += 1
        except Exception as exc:  # noqa: BLE001
            msg = f"provenance ingest: failed on {path}: {exc}"
            log.error(msg, exc_info=True)
            result.errors.append(msg)
            # Leave the .draining file in place for the next pass -- do not
            # delete on partial/failed ingest.
            continue
        try:
            path.unlink()
        except OSError as exc:  # pragma: no cover - best-effort
            log.warning("provenance ingest: could not remove drained %s: %s", path, exc)
        result.files_drained += 1
    return result


def _ingest_record_with_conn(conn, record: dict) -> bool:
    """Same logic as :func:`_ingest_record` but against an already-open
    connection, so a whole file's records share one transaction."""
    import json as _json

    required = ("fingerprint", "kind", "spatial_key", "recipe_id", "location")
    if not all(k in record for k in required):
        log.warning("provenance ingest: dropping record missing required keys: %r", record)
        return False

    recipe_params = record.get("recipe_params")
    if recipe_params is not None:
        conn.execute(
            "INSERT OR IGNORE INTO recipes"
            "(recipe_id, kind, params_json, code_version, git_sha, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (
                record["recipe_id"],
                record["kind"],
                _json.dumps(dict(recipe_params), sort_keys=True),
                int(record.get("code_version", 0)),
                record.get("git_sha"),
                record.get("created_at") or "",
            ),
        )

    conn.execute(
        "INSERT OR REPLACE INTO artifacts"
        "(fingerprint, kind, spatial_key, recipe_id, location, state,"
        " bytes, wall_time_s, produced_by, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            record["fingerprint"],
            record["kind"],
            _json.dumps(dict(record["spatial_key"]), sort_keys=True),
            record["recipe_id"],
            record["location"],
            record.get("state", "complete"),
            record.get("bytes"),
            record.get("wall_time_s"),
            record.get("produced_by"),
            record.get("created_at") or "",
        ),
    )

    for input_fp in record.get("inputs", ()):
        conn.execute(
            "INSERT OR IGNORE INTO artifact_inputs(fingerprint, input_fingerprint) VALUES (?,?)",
            (record["fingerprint"], input_fp),
        )

    return True
