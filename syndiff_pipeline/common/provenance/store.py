"""Provenance SQLite store — the derived, rebuildable artifact index.

This is the queryable index for the content-addressed provenance graph
described in ``doc/template_bookkeeping_plan.md`` (§8 schema, §10 protocol,
§11 the query that replaces the O(cells) NFS verify scan).

Design constraints (deliberately narrow):

* **Derived, not authoritative.** The finalized bytes at an artifact's
  fingerprinted key are the truth; this DB can always be rebuilt by
  ``reindex``. No correctness decision depends on a DB write succeeding.
* **Single writer.** ``ingest_records`` is the *sole* mutation entry point
  for the artifact graph and is called only by the supervisor draining
  worker sidecars. Everything else is a read. This mirrors the NFS-safe
  single-writer model already proven by the run-state DB
  (``common/orchestration/state.py``).
* **Dependency-free.** stdlib ``sqlite3`` only. This module never imports
  zarr/astropy/numpy or the sibling provenance submodules, so schedulers
  and daemons can open it without pulling heavy dependencies.

Connection hardening (WAL set once, per-connection ``busy_timeout`` /
``synchronous=NORMAL``, additive ``_ensure_column`` schema evolution) is
modeled directly on ``PipelineState`` in the run-state DB.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

# SQLite compiles with a default host-parameter cap (SQLITE_MAX_VARIABLE_NUMBER,
# historically 999). Chunk any IN (...) list below it with headroom to spare.
_SQL_VAR_CHUNK = 900


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp (matches the run-state DB convention)."""
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators.

    Used for ``spatial_key`` and recipe ``params`` so identical logical
    content always serializes to the same bytes (stable equality / dedup).
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _dedup_preserve(fingerprints: Iterable[str]) -> List[str]:
    """Unique fingerprints in first-seen order (dict preserves insertion order)."""
    return list(dict.fromkeys(fingerprints))


def _chunks(seq: Sequence[str], size: int = _SQL_VAR_CHUNK) -> Iterator[Sequence[str]]:
    """Yield ``seq`` in slices no larger than *size* (SQLite variable-limit safe)."""
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


class ProvenanceStore:
    """Read-many / write-one index over the artifact provenance graph.

    Open freely for reads from any process; only the supervisor should call
    :meth:`ingest_records`. Schema init is idempotent, so concurrent opens
    against an existing DB are safe.
    """

    def __init__(self, db_path: str | Path):
        """Resolve *db_path*, create its parent, and initialize the schema.

        Parameters
        ----------
        db_path : str | Path
            Location of ``provenance.db``. ``~`` is expanded and the path is
            resolved to an absolute path.
        """
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a hardened connection; commit on clean exit, always close.

        ``journal_mode=WAL`` is persisted in the DB header (set once in
        :meth:`_init_schema`), so it is intentionally *not* re-issued here —
        re-checking the ``-wal``/``-shm`` files on every connect is expensive
        and fragile over NFS. Only the per-connection pragmas are set each
        time.
        """
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create tables/indexes if absent. Idempotent; safe to call repeatedly."""
        with self._conn() as conn:
            # Durable journal mode, set exactly once; later connections inherit it.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    fingerprint  TEXT PRIMARY KEY,
                    kind         TEXT NOT NULL,
                    spatial_key  TEXT NOT NULL,
                    recipe_id    TEXT NOT NULL,
                    location     TEXT,
                    state        TEXT NOT NULL,
                    bytes        INTEGER,
                    wall_time_s  REAL,
                    produced_by  TEXT,
                    created_at   TEXT
                );
                CREATE TABLE IF NOT EXISTS recipes (
                    recipe_id    TEXT PRIMARY KEY,
                    kind         TEXT NOT NULL,
                    params_json  TEXT NOT NULL,
                    code_version TEXT,
                    git_sha      TEXT,
                    created_at   TEXT
                );
                CREATE TABLE IF NOT EXISTS artifact_inputs (
                    fingerprint       TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    PRIMARY KEY (fingerprint, input_fingerprint)
                );
                CREATE TABLE IF NOT EXISTS input_files (
                    kind         TEXT NOT NULL,
                    key          TEXT NOT NULL,
                    spatial_key  TEXT NOT NULL,
                    bytes        INTEGER,
                    mtime        TEXT,
                    checksum     TEXT,
                    source       TEXT,
                    batch_id     TEXT,
                    recorded_at  TEXT,
                    PRIMARY KEY (kind, key)
                );

                CREATE INDEX IF NOT EXISTS ix_art_kind_spatial
                    ON artifacts(kind, spatial_key);
                CREATE INDEX IF NOT EXISTS ix_art_recipe
                    ON artifacts(recipe_id);
                CREATE INDEX IF NOT EXISTS ix_art_state
                    ON artifacts(kind, state);
                """
            )

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection, table: str, column: str, decl: str
    ) -> None:
        """Additively add *column* to *table* if missing (schema evolution).

        Mirrors the run-state DB pattern; kept available for future additive
        columns without rewriting the schema or breaking older databases.
        """
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # ------------------------------------------------------------------
    # Row helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _artifact_row(row: sqlite3.Row) -> Dict[str, Any]:
        """Materialize an artifacts row, decoding ``spatial_key`` JSON to a dict."""
        d = dict(row)
        raw = d.get("spatial_key")
        d["spatial_key"] = json.loads(raw) if raw is not None else None
        return d

    @staticmethod
    def _recipe_row(row: sqlite3.Row) -> Dict[str, Any]:
        """Materialize a recipes row, decoding ``params_json`` back to ``params``."""
        d = dict(row)
        raw = d.get("params_json")
        d["params"] = json.loads(raw) if raw is not None else None
        return d

    # ------------------------------------------------------------------
    # Sole writer: ingest
    # ------------------------------------------------------------------
    def ingest_records(self, records: List[dict]) -> int:
        """Drain a batch of sidecar records into the index. SOLE writer entry point.

        Called by the supervisor with records produced by workers at publish
        time. The whole batch runs in one transaction and is fully idempotent:
        re-ingesting the same record leaves exactly one artifact row, the
        recipe deduped, and the edge set replaced (not duplicated).

        Each record is a dict with the documented producer contract::

            {"fingerprint": str, "kind": str, "spatial_key": dict,
             "recipe_id": str,
             "recipe": {"kind": str, "params": dict,
                        "code_version": str, "git_sha": str|None},
             "inputs": [fingerprint, ...], "location": str|None,
             "state": "complete"|"building"|"failed", "bytes": int|None,
             "wall_time_s": float|None, "produced_by": str|None,
             "created_at": iso|None}

        Returns
        -------
        int
            Number of records ingested (== ``len(records)``).
        """
        if not records:
            return 0
        with self._conn() as conn:
            for rec in records:
                fingerprint = rec["fingerprint"]
                recipe_id = rec["recipe_id"]
                created_at = rec.get("created_at") or _utc_now()

                recipe = rec.get("recipe") or {}
                # recipes: dedup by recipe_id; first writer wins (identical
                # params_json for a given id by construction).
                conn.execute(
                    "INSERT OR IGNORE INTO recipes "
                    "(recipe_id, kind, params_json, code_version, git_sha, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        recipe_id,
                        recipe.get("kind", rec["kind"]),
                        _canonical_json(recipe.get("params", {})),
                        recipe.get("code_version"),
                        recipe.get("git_sha"),
                        created_at,
                    ),
                )

                # artifacts: latest write wins (state transitions building ->
                # complete, re-publish with new location/bytes, etc.).
                conn.execute(
                    "INSERT OR REPLACE INTO artifacts "
                    "(fingerprint, kind, spatial_key, recipe_id, location, state, "
                    " bytes, wall_time_s, produced_by, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fingerprint,
                        rec["kind"],
                        _canonical_json(rec.get("spatial_key") or {}),
                        recipe_id,
                        rec.get("location"),
                        rec["state"],
                        rec.get("bytes"),
                        rec.get("wall_time_s"),
                        rec.get("produced_by"),
                        created_at,
                    ),
                )

                # edges: replace the child's full input set (idempotent).
                conn.execute(
                    "DELETE FROM artifact_inputs WHERE fingerprint = ?",
                    (fingerprint,),
                )
                inputs = _dedup_preserve(rec.get("inputs") or [])
                if inputs:
                    conn.executemany(
                        "INSERT OR IGNORE INTO artifact_inputs "
                        "(fingerprint, input_fingerprint) VALUES (?, ?)",
                        [(fingerprint, ip) for ip in inputs],
                    )
        return len(records)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_artifact(self, fingerprint: str) -> Dict[str, Any] | None:
        """Return the artifact row for *fingerprint* (spatial_key decoded), or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            return self._artifact_row(row) if row else None

    def artifact_inputs(self, fingerprint: str) -> List[str]:
        """Return the input (parent) fingerprints of *fingerprint*, sorted."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT input_fingerprint FROM artifact_inputs "
                "WHERE fingerprint = ? ORDER BY input_fingerprint",
                (fingerprint,),
            ).fetchall()
            return [r["input_fingerprint"] for r in rows]

    def scc_stage_complete(self, required_fingerprints: Iterable[str]) -> bool:
        """True iff every required fingerprint has a ``state='complete'`` row.

        This is the hot-path query that replaces the O(cells) NFS verify
        scan (plan §11). It touches only the index — never the filesystem —
        and issues one indexed lookup per chunk (chunked to stay under
        SQLite's host-parameter limit for large required sets).
        """
        required = _dedup_preserve(required_fingerprints)
        if not required:
            return True
        found = 0
        with self._conn() as conn:
            for chunk in _chunks(required):
                placeholders = ",".join("?" for _ in chunk)
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM artifacts "
                    f"WHERE state = 'complete' AND fingerprint IN ({placeholders})",
                    tuple(chunk),
                ).fetchone()
                found += row["n"]
                # Early out: a shortfall in any chunk means incomplete.
                if found < 0:  # pragma: no cover - defensive
                    return False
        return found == len(required)

    def missing_fingerprints(self, required_fingerprints: Iterable[str]) -> List[str]:
        """Return the required fingerprints lacking a ``complete`` row (order-preserving).

        Complement of :meth:`scc_stage_complete`: these are the work units a
        run still needs to produce. No filesystem access.
        """
        required = _dedup_preserve(required_fingerprints)
        if not required:
            return []
        complete: set[str] = set()
        with self._conn() as conn:
            for chunk in _chunks(required):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT fingerprint FROM artifacts "
                    f"WHERE state = 'complete' AND fingerprint IN ({placeholders})",
                    tuple(chunk),
                ).fetchall()
                complete.update(r["fingerprint"] for r in rows)
        return [fp for fp in required if fp not in complete]

    def artifacts_by_recipe(self, recipe_id: str) -> List[Dict[str, Any]]:
        """Return all artifacts produced by *recipe_id* (uses ``ix_art_recipe``)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE recipe_id = ? ORDER BY fingerprint",
                (recipe_id,),
            ).fetchall()
            return [self._artifact_row(r) for r in rows]

    def get_recipe(self, recipe_id: str) -> Dict[str, Any] | None:
        """Return the recipe row for *recipe_id* (``params`` decoded), or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recipes WHERE recipe_id = ?", (recipe_id,)
            ).fetchone()
            return self._recipe_row(row) if row else None

    # ------------------------------------------------------------------
    # Input-file tracking (FFIs & raw skycells)
    # ------------------------------------------------------------------
    def upsert_input_file(
        self,
        kind: str,
        key: str,
        spatial_key: dict,
        *,
        bytes: int | None = None,
        mtime: str | None = None,
        checksum: str | None = None,
        source: str | None = None,
        batch_id: str | None = None,
    ) -> None:
        """Insert or replace an input-file record keyed by ``(kind, key)``.

        Tracks tracked pipeline inputs (``ffi`` / ``raw_skycell``). Upsert
        semantics: a second call with the same ``(kind, key)`` overwrites the
        prior row (e.g. a re-download updating size/mtime/batch).
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO input_files "
                "(kind, key, spatial_key, bytes, mtime, checksum, source, "
                " batch_id, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind,
                    key,
                    _canonical_json(spatial_key or {}),
                    bytes,
                    mtime,
                    checksum,
                    source,
                    batch_id,
                    _utc_now(),
                ),
            )

    def list_input_files(self, kind: str) -> List[Dict[str, Any]]:
        """Return all input-file rows of *kind* (``spatial_key`` decoded)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM input_files WHERE kind = ? ORDER BY key", (kind,)
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                raw = d.get("spatial_key")
                d["spatial_key"] = json.loads(raw) if raw is not None else None
                out.append(d)
            return out

    # ------------------------------------------------------------------
    # Bulk read (reindex / gc)
    # ------------------------------------------------------------------
    def iter_artifacts(self, kind: str | None = None) -> Iterator[Dict[str, Any]]:
        """Iterate artifact rows (optionally filtered by *kind*) for reindex/gc.

        Rows are fully read into memory before yielding so the underlying
        connection is not held open across consumer work.
        """
        with self._conn() as conn:
            if kind is None:
                rows = conn.execute(
                    "SELECT * FROM artifacts ORDER BY fingerprint"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM artifacts WHERE kind = ? ORDER BY fingerprint",
                    (kind,),
                ).fetchall()
        for row in rows:
            yield self._artifact_row(row)
