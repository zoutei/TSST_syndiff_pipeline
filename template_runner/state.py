"""SQLite bookkeeping for template pipeline runs.

The supervisor daemon is the sole writer of execution state. The CLI only
inserts new run/stage rows (``INSERT OR IGNORE``) and command intents. The
database is hardened for concurrent NFS access via WAL + busy_timeout.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

STAGE_NAMES = (
    "tess_ffi_download",
    "wcs_grouping",
    "mapping",
    "ps1_download",
    "ps1_process",
    "downsample",
)

STAGE_SHORT_NAMES: Dict[str, str] = {
    "tess_ffi_download": "tess_dl",
    "wcs_grouping": "wcs",
    "mapping": "map",
    "ps1_download": "ps1_dl",
    "ps1_process": "ps1_pr",
    "downsample": "down",
}

STAGE_DEPS: Dict[str, List[str]] = {
    "wcs_grouping": ["tess_ffi_download"],
    "mapping": ["wcs_grouping"],
    "ps1_download": ["mapping"],
    "ps1_process": ["ps1_download"],
    "downsample": ["mapping", "ps1_process"],
}

STAGE_POOL: Dict[str, str] = {
    "tess_ffi_download": "network",
    "wcs_grouping": "cpu_light",
    "mapping": "mapping",
    "ps1_download": "network",
    "ps1_process": "ps1_process",
    "downsample": "cpu_light",
}

# Stage statuses (single, explicit state machine).
STATUS_PENDING = "pending"
STATUS_READY = "ready"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_SKIPPED = "skipped"
STATUS_CANCELED = "canceled"
STATUS_EXTERNAL = "external"

# Terminal stage statuses (no further work scheduled).
TERMINAL_STATUSES = frozenset(
    {STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED, STATUS_CANCELED}
)
# Statuses that count as "dependency satisfied".
SATISFIED_STATUSES = frozenset({STATUS_SUCCESS, STATUS_SKIPPED})

# Run statuses.
RUN_RUNNING = "running"
RUN_STALLED = "stalled"
RUN_SUCCESS = "success"
RUN_FAILED = "failed"
RUN_CANCELED = "canceled"
ACTIVE_RUN_STATUSES = frozenset({RUN_RUNNING, RUN_STALLED})

# Command kinds (CLI -> daemon intents).
CMD_CANCEL = "cancel"
CMD_PAUSE = "pause"
CMD_RESUME = "resume"
CMD_RETRY = "retry"
CMD_FORCE_RERUN = "force_rerun"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageRunRow:
    run_id: str
    target_label: str
    stage: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    log_path: str | None = None
    error_tail: str | None = None
    executor: str | None = None
    native_id: int | None = None
    launch_token: str | None = None
    claimed_at: str | None = None
    submit_epoch: float | None = None


@dataclass
class CommandRow:
    id: int
    run_id: str
    kind: str
    args_json: str | None
    created_at: str | None
    processed_at: str | None

    def args(self) -> dict:
        if not self.args_json:
            return {}
        try:
            return json.loads(self.args_json)
        except json.JSONDecodeError:
            return {}


def downstream_stages(stage: str) -> List[str]:
    """All stages that (transitively) depend on *stage*."""
    out: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s, deps in STAGE_DEPS.items():
            if s in out:
                continue
            if stage in deps or any(d in out for d in deps):
                out.add(s)
                changed = True
    return [s for s in STAGE_NAMES if s in out]


class PipelineState:
    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT,
                    config_path TEXT,
                    targets_path TEXT,
                    status TEXT,
                    runs_root TEXT,
                    stages TEXT,
                    force_rerun INTEGER DEFAULT 0,
                    paused INTEGER DEFAULT 0,
                    stall_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS targets (
                    run_id TEXT,
                    target_label TEXT,
                    sector INTEGER,
                    camera INTEGER,
                    ccd INTEGER,
                    target_name TEXT,
                    enabled INTEGER,
                    PRIMARY KEY (run_id, target_label)
                );
                CREATE TABLE IF NOT EXISTS stage_runs (
                    run_id TEXT,
                    target_label TEXT,
                    stage TEXT,
                    status TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    exit_code INTEGER,
                    log_path TEXT,
                    error_tail TEXT,
                    executor TEXT,
                    native_id INTEGER,
                    launch_token TEXT,
                    claimed_at TEXT,
                    submit_epoch REAL,
                    PRIMARY KEY (run_id, target_label, stage)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT,
                    target_label TEXT,
                    stage TEXT,
                    artifact_type TEXT,
                    path TEXT,
                    verified_at TEXT,
                    PRIMARY KEY (run_id, target_label, stage, artifact_type)
                );
                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    kind TEXT,
                    args_json TEXT,
                    created_at TEXT,
                    processed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS daemon (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    pid INTEGER,
                    host TEXT,
                    started_at TEXT,
                    last_heartbeat TEXT
                );
                """
            )
            self._ensure_column(conn, "runs", "stall_reason", "TEXT")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def list_active_runs(self) -> List[dict]:
        return self.active_runs()

    def get_active_stages(self, run_id: str) -> List[str]:
        stages = self.selected_stages(run_id)
        return stages or list(STAGE_NAMES)

    def set_run_status(self, run_id: str, status: str, *, stall_reason: str | None = None) -> None:
        with self._conn() as conn:
            if stall_reason is not None:
                conn.execute(
                    "UPDATE runs SET status = ?, stall_reason = ? WHERE run_id = ?",
                    (status, stall_reason, run_id),
                )
            else:
                conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def running_jobs(self, run_id: str) -> List[StageRunRow]:
        return self.running_stage_runs(run_id)

    def requeue_running_stage(
        self, run_id: str, target_label: str, stage: str, *, error_tail: str | None = None
    ) -> None:
        self.requeue_to_ready(run_id, target_label, stage, error_tail=error_tail)

    def is_artifact_verified(self, run_id: str, target_label: str, stage: str) -> bool:
        return self.external_checked(run_id, target_label, stage)

    def cache_artifact_verified(
        self, run_id: str, target_label: str, stage: str, *, path: str = ""
    ) -> None:
        self.cache_external_check(run_id, target_label, stage, complete=True, path=path)

    def new_launch_token(self) -> str:
        return str(uuid.uuid4())

    def try_atomic_claim(
        self,
        run_id: str,
        target_label: str,
        stage: str,
        *,
        launch_token: str,
        executor: str,
        native_id: int,
        log_path: str,
        submit_epoch: float | None = None,
    ) -> bool:
        if not self.claim_ready(run_id, target_label, stage, launch_token):
            return False
        self.set_launch_descriptor(
            run_id,
            target_label,
            stage,
            executor=executor,
            native_id=native_id,
            submit_epoch=submit_epoch,
            log_path=log_path,
        )
        return True

    def promote_stages(self, run_id: str) -> int:
        promoted = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT target_label, stage, status FROM stage_runs WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            for row in rows:
                target_label = row["target_label"]
                stage = row["stage"]
                status = row["status"]
                if status == STATUS_BLOCKED and self.deps_satisfied(run_id, target_label, stage):
                    conn.execute(
                        "UPDATE stage_runs SET status = ? "
                        "WHERE run_id = ? AND target_label = ? AND stage = ?",
                        (STATUS_PENDING, run_id, target_label, stage),
                    )
                    status = STATUS_PENDING
                if status == STATUS_PENDING and self.deps_satisfied(
                    run_id, target_label, stage
                ):
                    conn.execute(
                        "UPDATE stage_runs SET status = ? "
                        "WHERE run_id = ? AND target_label = ? AND stage = ?",
                        (STATUS_READY, run_id, target_label, stage),
                    )
                    promoted += 1
        return promoted

    def fetch_ready_batch(self, run_id: str, pool: str, limit: int) -> List[StageRunRow]:
        stages_in_pool = [s for s in STAGE_NAMES if STAGE_POOL.get(s) == pool]
        if not stages_in_pool:
            return []
        placeholders = ",".join("?" for _ in stages_in_pool)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM stage_runs
                WHERE run_id = ? AND status = ? AND stage IN ({placeholders})
                ORDER BY target_label, stage
                LIMIT ?
                """,
                (run_id, STATUS_READY, *stages_in_pool, limit),
            ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    def reset_stages_for_force_rerun(
        self, run_id: str, target_labels: Sequence[str], stages: Sequence[str]
    ) -> None:
        with self._conn() as conn:
            for label in target_labels:
                for stage in stages:
                    conn.execute(
                        "UPDATE stage_runs SET status = ?, started_at = NULL, finished_at = NULL, "
                        "exit_code = NULL, error_tail = NULL, native_id = NULL, launch_token = NULL, "
                        "claimed_at = NULL, submit_epoch = NULL, log_path = NULL "
                        "WHERE run_id = ? AND target_label = ? AND stage = ?",
                        (STATUS_PENDING, run_id, label, stage),
                    )
                    conn.execute(
                        "DELETE FROM artifacts WHERE run_id = ? AND target_label = ? AND stage = ?",
                        (run_id, label, stage),
                    )

    def list_failed_stage_runs(self, run_id: str) -> List[StageRunRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? AND status = ? "
                "ORDER BY target_label, stage",
                (run_id, STATUS_FAILED),
            ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    def list_retryable_stage_runs(self, run_id: str) -> List[StageRunRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? AND status IN (?, ?, ?, ?) "
                "ORDER BY target_label, stage",
                (run_id, STATUS_FAILED, STATUS_CANCELED, STATUS_RUNNING, STATUS_BLOCKED),
            ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    def insert_command(
        self, kind: str, *, run_id: str | None = None, args: dict | None = None
    ) -> int:
        if not run_id:
            raise ValueError("run_id required for command intents")
        return self.enqueue_command(run_id, kind, args)

    def fetch_pending_commands(self, limit: int = 50) -> List[CommandRow]:
        return self.fetch_unprocessed_commands()[:limit]

    def apply_cancel_run(self, run_id: str) -> dict[str, int]:
        counts = self.cancel_run_stages(run_id)
        self.set_run_status(run_id, RUN_CANCELED)
        return {"canceled": counts["running"], "blocked": counts["other"]}

    def apply_retry_run(self, run_id: str, *, reset_downstream: bool = True) -> int:
        total = self.reopen_failed_canceled(run_id)
        self.set_run_status(run_id, RUN_RUNNING)
        return total

    def apply_force_rerun(
        self,
        run_id: str,
        target_labels: Sequence[str],
        stages: Sequence[str],
    ) -> None:
        """Daemon-side application of a force-rerun intent.

        Persists the run's ``force_rerun`` flag (so the daemon bypasses the
        artifact-skip path for these stages), resets the selected stages to
        ``pending`` (clearing cached artifact checks), and resumes the run.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET force_rerun = 1 WHERE run_id = ?", (run_id,)
            )
        self.reset_stages_for_force_rerun(run_id, target_labels, stages)
        self.set_run_status(run_id, RUN_RUNNING)

    def apply_retry_stage(
        self,
        run_id: str,
        target_label: str,
        stage: str,
        *,
        reset_downstream: bool = True,
    ) -> None:
        row = self.get_stage_run(run_id, target_label, stage)
        if row and row.status == STATUS_RUNNING:
            self.requeue_to_ready(run_id, target_label, stage, error_tail="Retry requested")
        else:
            self.reset_stage_for_retry(
                run_id, target_label, stage, reset_downstream=reset_downstream
            )
        self.set_run_status(run_id, RUN_RUNNING)

    def update_supervisor_heartbeat(self, pid: int) -> None:
        import socket

        host = socket.gethostname()
        if self.get_daemon() is None:
            self.set_daemon_running(pid, host)
        else:
            self.heartbeat_daemon(pid)

    def get_supervisor_status(self) -> dict | None:
        return self.get_daemon()

    def clear_supervisor(self) -> None:
        self.clear_daemon()

    # ------------------------------------------------------------------
    # Run creation / lookup
    # ------------------------------------------------------------------
    def create_run(
        self,
        run_id: str,
        config_path: str,
        targets_path: str,
        runs_root: str,
        targets: Sequence,
        stages: Sequence[str],
        *,
        force_rerun: bool = False,
    ) -> None:
        """Materialize the run plus the FULL 6-stage DAG per target.

        Stages inside *stages* start ``pending``; every other stage starts
        ``external`` (resolved to ``skipped`` once verified on disk).
        ``INSERT OR IGNORE`` keeps this idempotent for resubmits.
        """
        selected = set(stages)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, started_at, config_path, targets_path, status, runs_root, stages, force_rerun) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    _utc_now(),
                    config_path,
                    targets_path,
                    RUN_RUNNING,
                    runs_root,
                    ",".join(stages),
                    int(bool(force_rerun)),
                ),
            )
            for t in targets:
                conn.execute(
                    "INSERT OR IGNORE INTO targets "
                    "(run_id, target_label, sector, camera, ccd, target_name, enabled) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, t.label(), t.sector, t.camera, t.ccd, t.target_name, int(t.enabled)),
                )
                for stage in STAGE_NAMES:
                    status = STATUS_PENDING if stage in selected else STATUS_EXTERNAL
                    conn.execute(
                        "INSERT OR IGNORE INTO stage_runs (run_id, target_label, stage, status) "
                        "VALUES (?, ?, ?, ?)",
                        (run_id, t.label(), stage, status),
                    )

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def active_runs(self) -> List[dict]:
        placeholders = ",".join("?" for _ in ACTIVE_RUN_STATUSES)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM runs WHERE status IN ({placeholders}) ORDER BY started_at",
                tuple(ACTIVE_RUN_STATUSES),
            ).fetchall()
            return [dict(r) for r in rows]

    def set_paused(self, run_id: str, paused: bool) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET paused = ? WHERE run_id = ?", (int(paused), run_id))

    def is_paused(self, run_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT paused FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return bool(row and row["paused"])

    def selected_stages(self, run_id: str) -> List[str]:
        run = self.get_run(run_id) or {}
        raw = run.get("stages") or ""
        return [s for s in raw.split(",") if s]

    def get_run_target(self, run_id: str, sector: int, camera: int, ccd: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM targets WHERE run_id = ? AND sector = ? AND camera = ? AND ccd = ?",
                (run_id, sector, camera, ccd),
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Stage row read / write
    # ------------------------------------------------------------------
    def get_stage_run(self, run_id: str, target_label: str, stage: str) -> StageRunRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? AND target_label = ? AND stage = ?",
                (run_id, target_label, stage),
            ).fetchone()
            return StageRunRow(**dict(row)) if row else None

    def list_stage_runs(self, run_id: str) -> List[StageRunRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? ORDER BY target_label, stage",
                (run_id,),
            ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    def running_stage_runs(self, run_id: str | None = None) -> List[StageRunRow]:
        with self._conn() as conn:
            if run_id is None:
                rows = conn.execute(
                    "SELECT * FROM stage_runs WHERE status = ?", (STATUS_RUNNING,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM stage_runs WHERE run_id = ? AND status = ?",
                    (run_id, STATUS_RUNNING),
                ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    def update_stage_status(
        self,
        run_id: str,
        target_label: str,
        stage: str,
        status: str,
        *,
        started_at: str | None = None,
        finished_at: str | None = None,
        exit_code: int | None = None,
        log_path: str | None = None,
        error_tail: str | None = None,
        executor: str | None = None,
        native_id: int | None = None,
        launch_token: str | None = None,
        claimed_at: str | None = None,
        submit_epoch: float | None = None,
    ) -> None:
        fields = ["status = ?"]
        values: list = [status]
        for col, val in (
            ("started_at", started_at),
            ("finished_at", finished_at),
            ("exit_code", exit_code),
            ("log_path", log_path),
            ("error_tail", error_tail),
            ("executor", executor),
            ("native_id", native_id),
            ("launch_token", launch_token),
            ("claimed_at", claimed_at),
            ("submit_epoch", submit_epoch),
        ):
            if val is not None:
                fields.append(f"{col} = ?")
                values.append(val)
        values.extend([run_id, target_label, stage])
        sql = (
            f"UPDATE stage_runs SET {', '.join(fields)} "
            "WHERE run_id = ? AND target_label = ? AND stage = ?"
        )
        with self._conn() as conn:
            conn.execute(sql, values)

    def count_by_status(self, run_id: str) -> Dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM stage_runs WHERE run_id = ? GROUP BY status",
                (run_id,),
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    # ------------------------------------------------------------------
    # Dependency resolution (SINGLE source of truth)
    # ------------------------------------------------------------------
    def deps_satisfied(self, run_id: str, target_label: str, stage: str) -> bool:
        """True iff every ``STAGE_DEPS[stage]`` row is success/skipped."""
        deps = STAGE_DEPS.get(stage, [])
        if not deps:
            return True
        placeholders = ",".join("?" for _ in deps)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT stage, status FROM stage_runs "
                f"WHERE run_id = ? AND target_label = ? AND stage IN ({placeholders})",
                (run_id, target_label, *deps),
            ).fetchall()
        status_map = {r["stage"]: r["status"] for r in rows}
        return all(status_map.get(d) in SATISFIED_STATUSES for d in deps)

    @staticmethod
    def unmet_deps(stage: str, status_map: Dict[tuple[str, str], str], target_label: str) -> List[str]:
        """Deps of *stage* not yet success/skipped, given a (label,stage)->status map."""
        out: List[str] = []
        for dep in STAGE_DEPS.get(stage, []):
            if status_map.get((target_label, dep)) not in SATISFIED_STATUSES:
                out.append(dep)
        return out

    # ------------------------------------------------------------------
    # Atomic work claim
    # ------------------------------------------------------------------
    def claim_ready(self, run_id: str, target_label: str, stage: str, launch_token: str) -> bool:
        """Atomically transition ready -> running. Returns True iff this caller won."""
        now = _utc_now()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE stage_runs SET status = ?, launch_token = ?, claimed_at = ?, "
                "started_at = ?, finished_at = NULL, exit_code = NULL, error_tail = NULL "
                "WHERE run_id = ? AND target_label = ? AND stage = ? AND status = ?",
                (
                    STATUS_RUNNING,
                    launch_token,
                    now,
                    now,
                    run_id,
                    target_label,
                    stage,
                    STATUS_READY,
                ),
            )
            return cur.rowcount == 1

    def set_launch_descriptor(
        self,
        run_id: str,
        target_label: str,
        stage: str,
        *,
        executor: str,
        native_id: int | None,
        submit_epoch: float | None,
        log_path: str | None,
    ) -> None:
        """Record the durable launch descriptor after a successful launch."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE stage_runs SET executor = ?, native_id = ?, submit_epoch = ?, log_path = ? "
                "WHERE run_id = ? AND target_label = ? AND stage = ? AND status = ?",
                (executor, native_id, submit_epoch, log_path, run_id, target_label, stage, STATUS_RUNNING),
            )

    # ------------------------------------------------------------------
    # Promotion / blocking / requeue
    # ------------------------------------------------------------------
    def clear_launch_fields(self, run_id: str, target_label: str, stage: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE stage_runs SET executor = NULL, native_id = NULL, launch_token = NULL, "
                "claimed_at = NULL, submit_epoch = NULL "
                "WHERE run_id = ? AND target_label = ? AND stage = ?",
                (run_id, target_label, stage),
            )

    def mark_ready(self, run_id: str, target_label: str, stage: str) -> None:
        self.update_stage_status(run_id, target_label, stage, STATUS_READY)

    def mark_skipped(self, run_id: str, target_label: str, stage: str) -> None:
        self.update_stage_status(
            run_id, target_label, stage, STATUS_SKIPPED, finished_at=_utc_now(), exit_code=0
        )

    def block_downstream(self, run_id: str, target_label: str, failed_stage: str) -> None:
        with self._conn() as conn:
            for stage in downstream_stages(failed_stage):
                conn.execute(
                    "UPDATE stage_runs SET status = ? "
                    "WHERE run_id = ? AND target_label = ? AND stage = ? AND status IN (?, ?, ?)",
                    (
                        STATUS_BLOCKED,
                        run_id,
                        target_label,
                        stage,
                        STATUS_PENDING,
                        STATUS_READY,
                        STATUS_EXTERNAL,
                    ),
                )

    def requeue_to_ready(
        self, run_id: str, target_label: str, stage: str, *, error_tail: str | None = None
    ) -> None:
        """Move a ``running`` row back to ``ready`` (lost worker, no exit record)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE stage_runs SET status = ?, started_at = NULL, finished_at = NULL, "
                "exit_code = NULL, native_id = NULL, launch_token = NULL, claimed_at = NULL, "
                "submit_epoch = NULL, error_tail = ? "
                "WHERE run_id = ? AND target_label = ? AND stage = ? AND status = ?",
                (STATUS_READY, error_tail, run_id, target_label, stage, STATUS_RUNNING),
            )

    # ------------------------------------------------------------------
    # Retry / cancel
    # ------------------------------------------------------------------
    def reset_stage_for_retry(
        self, run_id: str, target_label: str, stage: str, reset_downstream: bool = True
    ) -> int:
        """Reopen *stage* (+downstream) to pending; clear external caches so they re-verify."""
        stages = [stage] + (downstream_stages(stage) if reset_downstream else [])
        count = 0
        with self._conn() as conn:
            for s in stages:
                cur = conn.execute(
                    "UPDATE stage_runs SET status = ?, started_at = NULL, finished_at = NULL, "
                    "exit_code = NULL, error_tail = NULL, native_id = NULL, launch_token = NULL, "
                    "claimed_at = NULL, submit_epoch = NULL "
                    "WHERE run_id = ? AND target_label = ? AND stage = ? AND status != ?",
                    (STATUS_PENDING, run_id, target_label, s, STATUS_EXTERNAL),
                )
                count += cur.rowcount
                conn.execute(
                    "DELETE FROM artifacts WHERE run_id = ? AND target_label = ? AND stage = ?",
                    (run_id, target_label, s),
                )
        return count

    def reopen_failed_canceled(self, run_id: str) -> int:
        """Reopen all failed/canceled stages (+downstream) to pending."""
        rows = self.list_stage_runs(run_id)
        seeds = [
            r for r in rows if r.status in (STATUS_FAILED, STATUS_CANCELED, STATUS_BLOCKED)
        ]
        total = 0
        for r in seeds:
            total += self.reset_stage_for_retry(
                run_id, r.target_label, r.stage, reset_downstream=True
            )
        return total

    def cancel_run_stages(self, run_id: str) -> Dict[str, int]:
        """Mark every non-terminal stage canceled (running rows finalized by daemon)."""
        now = _utc_now()
        counts = {"running": 0, "other": 0}
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE stage_runs SET status = ?, finished_at = ?, exit_code = ?, "
                "error_tail = ?, native_id = NULL, launch_token = NULL "
                "WHERE run_id = ? AND status = ?",
                (STATUS_CANCELED, now, 143, "Canceled by user", run_id, STATUS_RUNNING),
            )
            counts["running"] = cur.rowcount
            cur = conn.execute(
                "UPDATE stage_runs SET status = ? "
                "WHERE run_id = ? AND status IN (?, ?, ?, ?)",
                (
                    STATUS_CANCELED,
                    run_id,
                    STATUS_PENDING,
                    STATUS_READY,
                    STATUS_BLOCKED,
                    STATUS_EXTERNAL,
                ),
            )
            counts["other"] = cur.rowcount
        return counts

    # ------------------------------------------------------------------
    # External -> skipped completeness cache
    # ------------------------------------------------------------------
    def external_checked(self, run_id: str, target_label: str, stage: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM artifacts WHERE run_id = ? AND target_label = ? AND stage = ? "
                "AND artifact_type = ?",
                (run_id, target_label, stage, "external_check"),
            ).fetchone()
            return row is not None

    def cache_external_check(
        self, run_id: str, target_label: str, stage: str, *, complete: bool, path: str | None = None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts "
                "(run_id, target_label, stage, artifact_type, path, verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, target_label, stage, "external_check", "1" if complete else "0", _utc_now()),
            )

    def list_unchecked_external_stages(self, run_id: str) -> List[StageRunRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT s.* FROM stage_runs s "
                "WHERE s.run_id = ? AND s.status = ? AND NOT EXISTS ("
                "  SELECT 1 FROM artifacts a WHERE a.run_id = s.run_id "
                "  AND a.target_label = s.target_label AND a.stage = s.stage "
                "  AND a.artifact_type = ?)",
                (run_id, STATUS_EXTERNAL, "external_check"),
            ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Commands (CLI -> daemon intents)
    # ------------------------------------------------------------------
    def enqueue_command(self, run_id: str, kind: str, args: dict | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO commands (run_id, kind, args_json, created_at) VALUES (?, ?, ?, ?)",
                (run_id, kind, json.dumps(args or {}), _utc_now()),
            )
            return int(cur.lastrowid)

    def fetch_unprocessed_commands(self) -> List[CommandRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM commands WHERE processed_at IS NULL ORDER BY id"
            ).fetchall()
            return [CommandRow(**dict(r)) for r in rows]

    def mark_command_processed(self, command_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE commands SET processed_at = ? WHERE id = ?",
                (_utc_now(), command_id),
            )

    # ------------------------------------------------------------------
    # Daemon registry / heartbeat
    # ------------------------------------------------------------------
    def set_daemon_running(self, pid: int, host: str) -> None:
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO daemon (id, pid, host, started_at, last_heartbeat) "
                "VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET pid = excluded.pid, host = excluded.host, "
                "started_at = excluded.started_at, last_heartbeat = excluded.last_heartbeat",
                (pid, host, now, now),
            )

    def heartbeat_daemon(self, pid: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE daemon SET last_heartbeat = ?, pid = ? WHERE id = 1",
                (_utc_now(), pid),
            )

    def get_daemon(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM daemon WHERE id = 1").fetchone()
            return dict(row) if row else None

    def clear_daemon(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM daemon WHERE id = 1")
