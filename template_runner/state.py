"""SQLite bookkeeping for template pipeline runs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

STAGE_NAMES = (
    "tess_ffi_download",
    "wcs_grouping",
    "mapping",
    "ps1_download",
    "ps1_process",
    "downsample",
)

STAGE_DEPS: Dict[str, List[str]] = {
    "wcs_grouping": ["tess_ffi_download"],
    "mapping": ["wcs_grouping"],
    "ps1_download": ["mapping"],
    "ps1_process": ["ps1_download"],
    "downsample": ["wcs_grouping", "ps1_process"],
}

STAGE_POOL: Dict[str, str] = {
    "tess_ffi_download": "network",
    "wcs_grouping": "cpu_light",
    "mapping": "mapping",
    "ps1_download": "network",
    "ps1_process": "ps1_process",
    "downsample": "cpu_light",
}

STATUS_PENDING = "pending"
STATUS_READY = "ready"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_SKIPPED = "skipped"
STATUS_KILLED = "killed"

TERMINAL_STATUSES = frozenset({STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED, STATUS_KILLED})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageRunRow:
    run_id: str
    target_label: str
    stage: str
    status: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    log_path: str | None
    error_tail: str | None
    pid: int | None = None


class PipelineState:
    def __init__(self, db_path: str | Path):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
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
                    paused INTEGER DEFAULT 0
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
                    pid INTEGER,
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
                """
            )

    def create_run(
        self,
        run_id: str,
        config_path: str,
        targets_path: str,
        runs_root: str,
        targets: Sequence,
        stages: Sequence[str],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, started_at, config_path, targets_path, status, runs_root) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, _utc_now(), config_path, targets_path, "running", runs_root),
            )
            for t in targets:
                conn.execute(
                    "INSERT INTO targets (run_id, target_label, sector, camera, ccd, target_name, enabled) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, t.label(), t.sector, t.camera, t.ccd, t.target_name, int(t.enabled)),
                )
                for stage in stages:
                    conn.execute(
                        "INSERT INTO stage_runs (run_id, target_label, stage, status) VALUES (?, ?, ?, ?)",
                        (run_id, t.label(), stage, STATUS_PENDING),
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

    def set_run_status(self, run_id: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def set_paused(self, run_id: str, paused: bool) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET paused = ? WHERE run_id = ?", (int(paused), run_id))

    def is_paused(self, run_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute("SELECT paused FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return bool(row and row["paused"])

    def get_stage_run(self, run_id: str, target_label: str, stage: str) -> StageRunRow | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? AND target_label = ? AND stage = ?",
                (run_id, target_label, stage),
            ).fetchone()
            if not row:
                return None
            return StageRunRow(**dict(row))

    def list_stage_runs(self, run_id: str) -> List[StageRunRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? ORDER BY target_label, stage",
                (run_id,),
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
        pid: int | None = None,
    ) -> None:
        fields = ["status = ?"]
        values: list = [status]
        if started_at is not None:
            fields.append("started_at = ?")
            values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at = ?")
            values.append(finished_at)
        if exit_code is not None:
            fields.append("exit_code = ?")
            values.append(exit_code)
        if log_path is not None:
            fields.append("log_path = ?")
            values.append(log_path)
        if error_tail is not None:
            fields.append("error_tail = ?")
            values.append(error_tail)
        if pid is not None:
            fields.append("pid = ?")
            values.append(pid)
        values.extend([run_id, target_label, stage])
        sql = (
            f"UPDATE stage_runs SET {', '.join(fields)} "
            "WHERE run_id = ? AND target_label = ? AND stage = ?"
        )
        with self._conn() as conn:
            conn.execute(sql, values)

    def deps_satisfied(self, run_id: str, target_label: str, stage: str) -> bool:
        deps = STAGE_DEPS.get(stage, [])
        if not deps:
            return True
        with self._conn() as conn:
            for dep in deps:
                row = conn.execute(
                    "SELECT status FROM stage_runs WHERE run_id = ? AND target_label = ? AND stage = ?",
                    (run_id, target_label, dep),
                ).fetchone()
                if not row or row["status"] != STATUS_SUCCESS:
                    return False
        return True

    def promote_ready_stages(self, run_id: str, active_stages: Sequence[str]) -> int:
        promoted = 0
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT target_label, stage, status FROM stage_runs WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            status_map = {(r["target_label"], r["stage"]): r["status"] for r in rows}
            for (target_label, stage), status in status_map.items():
                if stage not in active_stages or status != STATUS_PENDING:
                    continue
                deps = STAGE_DEPS.get(stage, [])
                if all(status_map.get((target_label, d)) == STATUS_SUCCESS for d in deps):
                    conn.execute(
                        "UPDATE stage_runs SET status = ? WHERE run_id = ? AND target_label = ? AND stage = ?",
                        (STATUS_READY, run_id, target_label, stage),
                    )
                    promoted += 1
        return promoted

    def block_downstream(self, run_id: str, target_label: str, failed_stage: str) -> None:
        all_down: set[str] = set()
        for stage, deps in STAGE_DEPS.items():
            if failed_stage in deps:
                all_down.add(stage)
        changed = True
        while changed:
            changed = False
            for stage, deps in STAGE_DEPS.items():
                if stage in all_down:
                    continue
                if any(d in all_down for d in deps):
                    all_down.add(stage)
                    changed = True
        with self._conn() as conn:
            for stage in all_down:
                conn.execute(
                    "UPDATE stage_runs SET status = ? WHERE run_id = ? AND target_label = ? "
                    "AND stage = ? AND status IN (?, ?, ?)",
                    (
                        STATUS_BLOCKED,
                        run_id,
                        target_label,
                        stage,
                        STATUS_PENDING,
                        STATUS_READY,
                        STATUS_QUEUED,
                    ),
                )

    def count_by_status(self, run_id: str) -> Dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM stage_runs WHERE run_id = ? GROUP BY status",
                (run_id,),
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    def fetch_ready_batch(
        self, run_id: str, pool: str, limit: int, active_stages: Sequence[str]
    ) -> List[StageRunRow]:
        stages_in_pool = [s for s in active_stages if STAGE_POOL.get(s) == pool]
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

    def mark_queued(self, run_id: str, target_label: str, stage: str) -> None:
        self.update_stage_status(run_id, target_label, stage, STATUS_QUEUED)

    def reset_stages_for_force_rerun(
        self, run_id: str, target_labels: Sequence[str], stages: Sequence[str]
    ) -> None:
        """Reset selected stages to pending so they run again despite existing artifacts."""
        with self._conn() as conn:
            for label in target_labels:
                for stage in stages:
                    conn.execute(
                        "UPDATE stage_runs SET status = ?, started_at = NULL, finished_at = NULL, "
                        "exit_code = NULL, error_tail = NULL, pid = NULL, log_path = NULL "
                        "WHERE run_id = ? AND target_label = ? AND stage = ?",
                        (STATUS_PENDING, run_id, label, stage),
                    )

    def reset_stage_for_retry(
        self, run_id: str, target_label: str, stage: str, reset_downstream: bool = True
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE stage_runs SET status = ?, started_at = NULL, finished_at = NULL, "
                "exit_code = NULL, error_tail = NULL, pid = NULL "
                "WHERE run_id = ? AND target_label = ? AND stage = ?",
                (STATUS_READY, run_id, target_label, stage),
            )
            if reset_downstream:
                for ds in STAGE_NAMES:
                    if stage in STAGE_DEPS.get(ds, []):
                        conn.execute(
                            "UPDATE stage_runs SET status = ?, started_at = NULL, finished_at = NULL, "
                            "exit_code = NULL, error_tail = NULL, pid = NULL "
                            "WHERE run_id = ? AND target_label = ? AND stage = ?",
                            (STATUS_BLOCKED, run_id, target_label, ds),
                        )

    def running_pids(self, run_id: str) -> List[int]:
        return [int(r.pid) for r in self.running_jobs(run_id) if r.pid is not None]

    def running_jobs(self, run_id: str) -> List[StageRunRow]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM stage_runs WHERE run_id = ? AND status = ? AND pid IS NOT NULL",
                (run_id, STATUS_RUNNING),
            ).fetchall()
            return [StageRunRow(**dict(r)) for r in rows]

    def finalize_run_killed(self, run_id: str) -> dict[str, int]:
        """Mark active stage rows terminal after a user kill.

        - running -> killed (exit 143)
        - pending / ready / queued -> blocked
        """
        now = _utc_now()
        counts = {"killed": 0, "blocked": 0}
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE stage_runs
                SET status = ?, finished_at = ?, exit_code = ?, error_tail = ?, pid = NULL
                WHERE run_id = ? AND status = ?
                """,
                (STATUS_KILLED, now, 143, "Run killed by user", run_id, STATUS_RUNNING),
            )
            counts["killed"] = cur.rowcount
            cur = conn.execute(
                """
                UPDATE stage_runs
                SET status = ?, pid = NULL
                WHERE run_id = ? AND status IN (?, ?, ?)
                """,
                (STATUS_BLOCKED, run_id, STATUS_PENDING, STATUS_READY, STATUS_QUEUED),
            )
            counts["blocked"] = cur.rowcount
        return counts

    def mark_skipped_existing(self, run_id: str, target_label: str, stage: str) -> None:
        self.update_stage_status(
            run_id, target_label, stage, STATUS_SKIPPED, finished_at=_utc_now(), exit_code=0
        )
