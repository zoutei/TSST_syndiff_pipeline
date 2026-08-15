"""
store.py
========
``ProvenanceStore``: SQLite-backed read/write access to
``bookkeeping/provenance.db`` (§8 schema, WAL, single writer -- the
supervisor via :mod:`ingest`).

Read-side queries (``scc_stage_complete``, ``missing_fingerprints``,
``artifact``, stats/query helpers) are the only thing meant to sit on any
hot path -- they are indexed ``SELECT``s, never directory walks. Writers
(``upsert_artifact``, ``upsert_recipe``, ``add_edge``, ``upsert_input_file``)
are used by :mod:`ingest` (draining the sidecar spool) and :mod:`reindex`
(rebuilding from disk); ordinary producers never call them directly -- they
go through :mod:`publish`, which only ever *writes files*, never touches the
DB.

Fault-injection support: pass a ``fs_probe`` callable to the constructor (or
override :meth:`_stat_missing`) to make the "authoritative fallback on index
lag" path (§10: ``stat`` only the missing fingerprinted keys) observable/
testable without a real filesystem, and to let a test assert it is never
invoked on the indexed hot path.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactRow",
    "RecipeRow",
    "ProvenanceStore",
    "NoDirectoryWalkError",
]

SCHEMA_VERSION = 1

STATE_BUILDING = "building"
STATE_COMPLETE = "complete"
STATE_FAILED = "failed"
VALID_STATES = frozenset({STATE_BUILDING, STATE_COMPLETE, STATE_FAILED})


class NoDirectoryWalkError(RuntimeError):
    """Raised by fault-injection stores when code attempts a directory walk."""


@dataclass(frozen=True)
class RecipeRow:
    recipe_id: str
    kind: str
    params_json: str
    code_version: int
    git_sha: Optional[str]
    created_at: str

    @property
    def params(self) -> dict:
        return json.loads(self.params_json)


@dataclass(frozen=True)
class ArtifactRow:
    fingerprint: str
    kind: str
    spatial_key_json: str
    recipe_id: str
    location: str
    state: str
    bytes: Optional[int]
    wall_time_s: Optional[float]
    produced_by: Optional[str]
    created_at: str

    @property
    def spatial_key(self) -> dict:
        return json.loads(self.spatial_key_json)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS artifacts (
    fingerprint     TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    spatial_key     TEXT NOT NULL,
    recipe_id       TEXT NOT NULL,
    location        TEXT NOT NULL,
    state           TEXT NOT NULL,
    bytes           INTEGER,
    wall_time_s     REAL,
    produced_by     TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
);

CREATE TABLE IF NOT EXISTS recipes (
    recipe_id       TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    code_version    INTEGER NOT NULL,
    git_sha         TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_inputs (
    fingerprint         TEXT NOT NULL,
    input_fingerprint    TEXT NOT NULL,
    PRIMARY KEY (fingerprint, input_fingerprint)
);

CREATE TABLE IF NOT EXISTS input_files (
    kind            TEXT NOT NULL,
    key             TEXT NOT NULL,
    spatial_key     TEXT NOT NULL,
    bytes           INTEGER,
    mtime           REAL,
    checksum        TEXT,
    source          TEXT,
    batch_id        TEXT,
    PRIMARY KEY (kind, key)
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_kind_spatial ON artifacts(kind, spatial_key);
CREATE INDEX IF NOT EXISTS idx_artifacts_recipe ON artifacts(recipe_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind_state ON artifacts(kind, state);
"""


class ProvenanceStore:
    """
    SQLite-backed provenance index.

    Content authority lives on disk (§5 invariants) -- this store is a
    derived, rebuildable cache. It is safe to delete ``provenance.db`` and
    run ``reindex`` at any time.

    Parameters
    ----------
    db_path : str | Path
        Path to ``provenance.db``. Parent directory is created if missing.
    fs_probe : Callable[[str], bool] | None
        Injectable filesystem existence check used only by the authoritative
        fallback (§10) when the index is missing a fingerprint that might
        still be freshly published but not yet ingested. Defaults to
        ``os.path.exists``. Tests pass a probe that raises
        :class:`NoDirectoryWalkError` to prove no scan happens on the
        indexed hot path (``scc_stage_complete``/``missing_fingerprints``
        never call it unless a key is actually missing from the index).
    read_only : bool
        When True, refuses any write method (defense in depth for reader
        call sites -- "readers never write", §10).
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        fs_probe: Optional[Callable[[str], bool]] = None,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fs_probe = fs_probe if fs_probe is not None else _default_fs_probe
        self.read_only = read_only
        self._lock = threading.Lock()
        self._init_schema()

    # -- connection -----------------------------------------------------

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            # WAL mode relies on shared-memory (mmap) coordination between
            # readers/writers that network filesystems don't properly
            # support; this DB lives on NFS and is opened from multiple
            # hosts (supervisor lease handover migrates between machines),
            # which repeatedly corrupted it ("database disk image is
            # malformed"). DELETE is SQLite's documented-safe rollback
            # journal mode for network filesystems.
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.executescript(_SCHEMA_SQL)
            conn.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _require_write(self) -> None:
        if self.read_only:
            raise PermissionError("ProvenanceStore opened read_only=True; refusing write")

    # -- writes (used by ingest.py / reindex.py only) --------------------

    def upsert_recipe(
        self,
        recipe_id: str,
        kind: str,
        params: Mapping[str, Any],
        code_version: int,
        *,
        git_sha: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """``INSERT OR IGNORE`` a recipe (recipes are immutable once written)."""
        self._require_write()
        created = created_at or _utcnow_iso()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO recipes"
                "(recipe_id, kind, params_json, code_version, git_sha, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    recipe_id,
                    kind,
                    json.dumps(dict(params), sort_keys=True),
                    int(code_version),
                    git_sha,
                    created,
                ),
            )

    def upsert_artifact(
        self,
        fingerprint: str,
        kind: str,
        spatial_key: Mapping[str, Any],
        recipe_id: str,
        location: str,
        state: str = STATE_COMPLETE,
        *,
        bytes_: Optional[int] = None,
        wall_time_s: Optional[float] = None,
        produced_by: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """``INSERT OR REPLACE`` an artifact row. Idempotent under retries."""
        self._require_write()
        if state not in VALID_STATES:
            raise ValueError(f"invalid state {state!r}; must be one of {sorted(VALID_STATES)}")
        created = created_at or _utcnow_iso()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts"
                "(fingerprint, kind, spatial_key, recipe_id, location, state,"
                " bytes, wall_time_s, produced_by, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    fingerprint,
                    kind,
                    json.dumps(dict(spatial_key), sort_keys=True),
                    recipe_id,
                    location,
                    state,
                    bytes_,
                    wall_time_s,
                    produced_by,
                    created,
                ),
            )

    def add_edge(self, fingerprint: str, input_fingerprint: str) -> None:
        """``INSERT OR IGNORE`` one DAG edge."""
        self._require_write()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifact_inputs(fingerprint, input_fingerprint)"
                " VALUES (?,?)",
                (fingerprint, input_fingerprint),
            )

    def add_edges(self, fingerprint: str, input_fingerprints: Iterable[str]) -> None:
        for fp in input_fingerprints:
            self.add_edge(fingerprint, fp)

    def upsert_input_file(
        self,
        kind: str,
        key: str,
        spatial_key: Mapping[str, Any],
        *,
        bytes_: Optional[int] = None,
        mtime: Optional[float] = None,
        checksum: Optional[str] = None,
        source: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> None:
        """``INSERT OR REPLACE`` a raw-input registry row (ffi rows fed from ``ffi_list``, no re-stat)."""
        self._require_write()
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO input_files"
                "(kind, key, spatial_key, bytes, mtime, checksum, source, batch_id)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    kind,
                    key,
                    json.dumps(dict(spatial_key), sort_keys=True),
                    bytes_,
                    mtime,
                    checksum,
                    source,
                    batch_id,
                ),
            )

    def delete_artifact(self, fingerprint: str) -> None:
        """Remove one artifact + its edges (GC / test cleanup only)."""
        self._require_write()
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM artifacts WHERE fingerprint=?", (fingerprint,))
            conn.execute(
                "DELETE FROM artifact_inputs WHERE fingerprint=? OR input_fingerprint=?",
                (fingerprint, fingerprint),
            )

    def clear(self) -> None:
        """Wipe all rows (used by reindex to rebuild from scratch)."""
        self._require_write()
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM artifacts")
            conn.execute("DELETE FROM recipes")
            conn.execute("DELETE FROM artifact_inputs")
            conn.execute("DELETE FROM input_files")

    # -- reads (hot path; indexed only) ----------------------------------

    def artifact(self, fingerprint: str) -> Optional[ArtifactRow]:
        """Fetch one artifact row by fingerprint, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        return _row_to_artifact(row) if row is not None else None

    def recipe(self, recipe_id: str) -> Optional[RecipeRow]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recipes WHERE recipe_id=?", (recipe_id,)
            ).fetchone()
        return _row_to_recipe(row) if row is not None else None

    def _indexed_present(self, fingerprints: Sequence[str]) -> set[str]:
        if not fingerprints:
            return set()
        present: set[str] = set()
        with self._conn() as conn:
            # SQLite has a default limit of 999/32766 bound params depending
            # on build; chunk defensively.
            chunk = 500
            for i in range(0, len(fingerprints), chunk):
                batch = fingerprints[i : i + chunk]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT fingerprint FROM artifacts"
                    f" WHERE fingerprint IN ({placeholders}) AND state=?",
                    (*batch, STATE_COMPLETE),
                ).fetchall()
                present.update(r["fingerprint"] for r in rows)
        return present

    def missing_fingerprints(
        self,
        required_fps: Sequence[str],
        *,
        fallback_stat: bool = True,
    ) -> list[str]:
        """
        Fingerprints in *required_fps* not (yet) indexed as ``complete``.

        One indexed ``SELECT ... IN (...)`` -- O(len(required_fps)), never a
        directory walk. When ``fallback_stat=True`` (default), each
        still-missing fingerprint is individually re-checked via the
        injectable ``fs_probe`` (authoritative fallback on index lag, §10) --
        this touches only the missing keys, never the full required set, and
        never the whole store.
        """
        required = list(dict.fromkeys(str(fp) for fp in required_fps))
        present = self._indexed_present(required)
        missing = [fp for fp in required if fp not in present]
        if not missing or not fallback_stat:
            return missing
        still_missing = []
        for fp in missing:
            if not self._fs_probe(fp):
                still_missing.append(fp)
        return still_missing

    def scc_stage_complete(
        self,
        required_fps: Sequence[str],
        *,
        fallback_stat: bool = True,
    ) -> bool:
        """True iff every fingerprint in *required_fps* is indexed complete
        (or found via the missing-only fallback stat)."""
        if not required_fps:
            return True
        return len(self.missing_fingerprints(required_fps, fallback_stat=fallback_stat)) == 0

    def artifacts_by_kind_spatial(self, kind: str, spatial_key: Mapping[str, Any]) -> list[ArtifactRow]:
        """All artifacts for one ``(kind, spatial_key)`` -- e.g. every recipe
        ever built for one SCC's ``mapping``."""
        needle = json.dumps(dict(spatial_key), sort_keys=True)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE kind=? AND spatial_key=?",
                (kind, needle),
            ).fetchall()
        return [_row_to_artifact(r) for r in rows]

    def inputs_of(self, fingerprint: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT input_fingerprint FROM artifact_inputs WHERE fingerprint=?",
                (fingerprint,),
            ).fetchall()
        return [r["input_fingerprint"] for r in rows]

    def consumers_of(self, input_fingerprint: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT fingerprint FROM artifact_inputs WHERE input_fingerprint=?",
                (input_fingerprint,),
            ).fetchall()
        return [r["fingerprint"] for r in rows]

    def input_file(self, kind: str, key: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM input_files WHERE kind=? AND key=?", (kind, key)
            ).fetchone()
        return dict(row) if row is not None else None

    # -- stats ------------------------------------------------------------

    def stats(self) -> dict:
        """Row counts by kind/state -- cheap summary for ``syndiff bookkeeping stats``."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"]
            by_kind = conn.execute(
                "SELECT kind, state, COUNT(*) AS n FROM artifacts GROUP BY kind, state"
                " ORDER BY kind, state"
            ).fetchall()
            n_recipes = conn.execute("SELECT COUNT(*) AS n FROM recipes").fetchone()["n"]
            n_edges = conn.execute("SELECT COUNT(*) AS n FROM artifact_inputs").fetchone()["n"]
            n_input_files = conn.execute("SELECT COUNT(*) AS n FROM input_files").fetchone()["n"]
        by_kind_state: dict[str, dict[str, int]] = {}
        for row in by_kind:
            by_kind_state.setdefault(row["kind"], {})[row["state"]] = row["n"]
        return {
            "total_artifacts": total,
            "by_kind_state": by_kind_state,
            "n_recipes": n_recipes,
            "n_edges": n_edges,
            "n_input_files": n_input_files,
            "db_path": str(self.db_path),
        }


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRow:
    return ArtifactRow(
        fingerprint=row["fingerprint"],
        kind=row["kind"],
        spatial_key_json=row["spatial_key"],
        recipe_id=row["recipe_id"],
        location=row["location"],
        state=row["state"],
        bytes=row["bytes"],
        wall_time_s=row["wall_time_s"],
        produced_by=row["produced_by"],
        created_at=row["created_at"],
    )


def _row_to_recipe(row: sqlite3.Row) -> RecipeRow:
    return RecipeRow(
        recipe_id=row["recipe_id"],
        kind=row["kind"],
        params_json=row["params_json"],
        code_version=row["code_version"],
        git_sha=row["git_sha"],
        created_at=row["created_at"],
    )


def _default_fs_probe(_fingerprint: str) -> bool:
    """Default fallback probe: no location known here, so it always reports
    "not found" -- callers that need real stat-fallback behavior (§10: stat
    the *fingerprinted key path*, not the fingerprint alone) should pass a
    ``fs_probe`` closure bound to their location-resolution logic. The
    default is conservative (never claims false presence)."""
    return False


def _utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
