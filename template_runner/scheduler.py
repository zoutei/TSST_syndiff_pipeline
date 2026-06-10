"""Multi-run supervisor daemon for template pipeline runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List

from syndiff_pipeline.template_runner import condor, daemon, launcher, logs, stages
from syndiff_pipeline.template_runner.deployment import load_handoff_root_from_deployment
from syndiff_pipeline.template_runner.run_context import resolve_run_context
from syndiff_pipeline.template_runner.workspace import record_deployment_path
from syndiff_pipeline.template_runner.runner_config import resolve_config
from syndiff_pipeline.template_runner.state import (
    RUN_CANCELED,
    SKIP_REASON_ARTIFACTS,
    SKIP_REASON_NOT_SELECTED,
    SKIP_REASON_STREAM,
    SKIP_REASON_SUPERSEDED,
    STATUS_BLOCKED,
    STATUS_CANCELED,
    STATUS_EXTERNAL,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    STAGE_DEPS,
    STAGE_POOL,
    TERMINAL_RUN_STATUSES,
    PipelineState,
    _utc_now,
    artifact_verify_needed,
    derive_run_final_status,
    downstream_stages,
    effective_stage_deps,
    run_stage_closure,
    stage_needs_artifact_verify_display,
)
from syndiff_pipeline.template_runner.verify_status import (
    clear_verify_in_flight,
    write_verify_in_flight,
)

if TYPE_CHECKING:
    from syndiff_pipeline.template_runner.verify_worker import (
        BackfillTask,
        VerifyOutcome,
        VerifyTask,
        VerifyTaskKey,
    )

log = logging.getLogger(__name__)

_shutdown = False
_lock_fd: int | None = None

NONTERMINAL_STATUSES = frozenset(
    {
        STATUS_PENDING,
        STATUS_READY,
        STATUS_RUNNING,
        STATUS_BLOCKED,
        STATUS_EXTERNAL,
    }
)

# Grace window after an atomic claim before a local job that has not yet written
# its status file is treated as lost. Heavy stage imports (numpy/zarr over NFS)
# can delay the first status write by many seconds, so keep this generous.
_LOCAL_START_GRACE_S = 300.0

# Cadence of the background heartbeat thread. It must be well under the
# staleness threshold used by the lifecycle layer (DEFAULT_HEARTBEAT_STALE_S).
_HEARTBEAT_INTERVAL_S = 15.0

# On SIGTERM, drain in-flight verify results briefly before dropping them.
_SHUTDOWN_VERIFY_DRAIN_S = 5.0

# If the host-local heartbeat file cannot be written for this long, something is
# badly wrong locally. Rather than linger as a zombie that holds the lock but
# looks dead, the supervisor exits so a fresh one (which reconciles in-flight
# jobs from durable status files) can take over.
_HEARTBEAT_FATAL_AFTER_S = 90.0


def _ensure_discord_bot_on_startup(handoff_root: str) -> None:
    from syndiff_pipeline.template_runner.discord_bot_control import (
        ensure_discord_bot_for_handoff_root,
    )

    try:
        result = ensure_discord_bot_for_handoff_root(handoff_root)
    except Exception:
        log.warning("Discord bot startup ensure failed", exc_info=True)
        return
    if result is None:
        return
    if result.spawned and result.pid:
        log.info("Started Discord bot pid=%s", result.pid)
    elif result.skipped_reason:
        log.warning("Discord bot not running: %s", result.skipped_reason)


def _age_seconds(iso_ts: str | None) -> float:
    """Age in seconds of an ISO-8601 timestamp; +inf if missing/unparseable."""
    if not iso_ts:
        return float("inf")
    try:
        ts = datetime.fromisoformat(iso_ts)
    except ValueError:
        return float("inf")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _handle_signal(signum, frame):
    global _shutdown
    log.warning("Received signal %s — shutting down supervisor gracefully", signum)
    _shutdown = True


def _write_local_heartbeat(handoff_root: str) -> None:
    """Write the host-local heartbeat file (NFS-independent liveness signal)."""
    path = logs.daemon_heartbeat_file(handoff_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(time.time()), encoding="utf-8")
    tmp.replace(path)


def _supervisor_heartbeat_loop(
    state: PipelineState, handoff_root: str, pid: int, interval_s: float
) -> None:
    """Keep liveness fresh while the main loop is busy launching stages.

    The host-local heartbeat file is the authoritative liveness signal because
    it does not depend on the (possibly wedged or full) NFS state DB. The DB
    heartbeat is updated best-effort, for cross-tool visibility only.
    """
    global _shutdown
    last_local_ok = time.monotonic()
    while not _shutdown:
        time.sleep(interval_s)
        if _shutdown:
            break
        try:
            _write_local_heartbeat(handoff_root)
            last_local_ok = time.monotonic()
        except Exception:
            log.exception("Failed to write local heartbeat file")
            if time.monotonic() - last_local_ok > _HEARTBEAT_FATAL_AFTER_S:
                log.error(
                    "Local heartbeat unwritable for >%ss — exiting so a fresh "
                    "supervisor can take over",
                    _HEARTBEAT_FATAL_AFTER_S,
                )
                _shutdown = True
                break
        try:
            state.update_supervisor_heartbeat(pid)
        except Exception:
            # DB heartbeat is best-effort; the local file is authoritative.
            log.warning("Failed to update DB heartbeat (best-effort)", exc_info=True)


def _write_summary(state: PipelineState, run_id: str, runs_root: str) -> None:
    counts = state.count_by_status(run_id)
    summary = {"run_id": run_id, "counts": counts, "updated_at": _utc_now()}
    logs.write_json_atomic(logs.summary_json_path(runs_root, run_id), summary)
    csv_path = logs.summary_csv_path(runs_root, run_id)
    rows = state.list_stage_runs(run_id)
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "target_label",
                "stage",
                "status",
                "started_at",
                "finished_at",
                "exit_code",
                "log_path",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "target_label": r.target_label,
                    "stage": r.stage,
                    "status": r.status,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "exit_code": r.exit_code,
                    "log_path": r.log_path,
                }
            )
    tmp.replace(csv_path)


def _load_run_context(state: PipelineState, run_id: str):
    run = state.get_run(run_id)
    if not run:
        return None
    runs_root = run["runs_root"]
    run_dir = logs.run_dir(runs_root, run_id)
    try:
        return resolve_run_context(run_dir=run_dir, run_id=run_id)
    except SystemExit as exc:
        # Missing/broken run directory must not crash the supervisor; skip it.
        log.error("Cannot load run context for %s: %s", run_id, exc)
        return None


def _read_status_file(
    runs_root: str, run_id: str, target_label: str, stage: str
) -> dict | None:
    return logs.read_json(logs.stage_status_path(runs_root, run_id, target_label, stage))


def _effective_exit_code(exit_code: int, log_path: str) -> int:
    """Reconcile misleading exit 0 when the stage log shows a signal shutdown."""
    if exit_code != 0:
        return exit_code
    tail = logs.read_log_tail(log_path, 30)
    if "Graceful shutdown initiated" in tail or "Initiating graceful shutdown" in tail:
        return 143
    return exit_code


def _notify_stage_outcome(
    state: PipelineState,
    run_id: str,
    *,
    target_label: str,
    stage: str,
    outcome: str,
    runs_root: str,
    finished_at: str,
    error_tail: str | None = None,
) -> None:
    ctx = _load_run_context(state, run_id)
    if ctx is None:
        return
    from syndiff_pipeline.template_runner.notifications import notifier_for_context

    notifier = notifier_for_context(state, ctx)
    if notifier is None:
        return
    notifier.notify_stage_outcome(
        run_id,
        runs_root,
        target_label=target_label,
        stage=stage,
        outcome=outcome,
        finished_at=finished_at,
        error_tail=error_tail,
    )


def _notify_run_retried(
    state: PipelineState,
    run_id: str,
    *,
    target_label: str | None = None,
    stage: str | None = None,
    reset_downstream: bool | None = None,
) -> None:
    ctx = _load_run_context(state, run_id)
    if ctx is None:
        return
    from syndiff_pipeline.template_runner.notifications import notifier_for_context

    notifier = notifier_for_context(state, ctx)
    if notifier is None:
        return
    notifier.notify_run_retried(
        run_id,
        ctx.cfg.runs_dir(),
        target_label=target_label,
        stage=stage,
        reset_downstream=reset_downstream,
    )


def _notify_run_canceled(state: PipelineState, run_id: str, running_before) -> None:
    ctx = _load_run_context(state, run_id)
    if ctx is None:
        return
    from syndiff_pipeline.template_runner.notifications import notifier_for_context

    notifier = notifier_for_context(state, ctx)
    if notifier is None:
        return
    runs_root = ctx.cfg.runs_dir()
    finished_at = _utc_now()
    for job in running_before:
        notifier.notify_stage_outcome(
            run_id,
            runs_root,
            target_label=job.target_label,
            stage=job.stage,
            outcome="canceled",
            finished_at=finished_at,
            error_tail="Canceled by user",
        )
    notifier.notify_run_canceled(run_id, runs_root)


def _finalize_stage(
    state: PipelineState,
    run_id: str,
    target_label: str,
    stage: str,
    *,
    runs_root: str,
    exit_code: int,
    log_path: str | None = None,
) -> None:
    log_path = log_path or str(logs.target_log_path(runs_root, run_id, target_label, stage))
    exit_code = _effective_exit_code(exit_code, log_path)
    error_tail = logs.read_log_tail(log_path, 20) if exit_code != 0 else ""
    finished_at = _utc_now()
    if exit_code == 0:
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            STATUS_SUCCESS,
            finished_at=finished_at,
            exit_code=0,
            log_path=log_path,
        )
        _notify_stage_outcome(
            state,
            run_id,
            target_label=target_label,
            stage=stage,
            outcome="success",
            runs_root=runs_root,
            finished_at=finished_at,
        )
    elif exit_code == 143:
        cancel_reason = error_tail or "Canceled (SIGTERM)"
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            STATUS_CANCELED,
            finished_at=finished_at,
            exit_code=exit_code,
            log_path=log_path,
            error_tail=cancel_reason,
        )
        _notify_stage_outcome(
            state,
            run_id,
            target_label=target_label,
            stage=stage,
            outcome="canceled",
            runs_root=runs_root,
            finished_at=finished_at,
            error_tail=cancel_reason,
        )
    else:
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            STATUS_FAILED,
            finished_at=finished_at,
            exit_code=exit_code,
            log_path=log_path,
            error_tail=error_tail,
        )
        state.block_downstream(run_id, target_label, stage)
        _notify_stage_outcome(
            state,
            run_id,
            target_label=target_label,
            stage=stage,
            outcome="failed",
            runs_root=runs_root,
            finished_at=finished_at,
            error_tail=error_tail,
        )
    state.clear_launch_fields(run_id, target_label, stage)


def reconcile_running_stages(
    state: PipelineState,
    run_id: str,
    ctx,
) -> dict[str, int]:
    """Self-healing reconcile for in-flight stage rows."""
    counts = {"adopted": 0, "completed": 0, "failed": 0, "requeued": 0, "still_running": 0}
    cfg = ctx.cfg
    runs_root = cfg.runs_dir()

    for job in state.running_jobs(run_id):
        executor = job.executor or cfg.stage_executor(job.stage)
        native_id = job.native_id
        status_doc = _read_status_file(runs_root, run_id, job.target_label, job.stage)

        if executor == "condor":
            if native_id is None:
                # Claimed but the cluster id was never recorded (daemon died
                # between claim and submit). Requeue once past the grace window.
                if _age_seconds(job.claimed_at) >= _LOCAL_START_GRACE_S:
                    died_reason = "Condor stage claimed but never submitted; requeued"
                    state.requeue_running_stage(
                        run_id,
                        job.target_label,
                        job.stage,
                        error_tail=died_reason,
                    )
                    _notify_stage_outcome(
                        state,
                        run_id,
                        target_label=job.target_label,
                        stage=job.stage,
                        outcome="died",
                        runs_root=runs_root,
                        finished_at=_utc_now(),
                        error_tail=died_reason,
                    )
                    counts["requeued"] += 1
                else:
                    counts["still_running"] += 1
                continue
            # Wall-clock submit epoch (DB-persisted) drives the poll grace.
            submit_epoch = job.submit_epoch if job.submit_epoch is not None else 0.0
            exit_code = condor.poll_cluster(int(native_id), submitted_at=submit_epoch)
            if exit_code is None:
                counts["still_running"] += 1
                continue
            log_path = str(
                logs.target_log_path(runs_root, run_id, job.target_label, job.stage)
            )
            _finalize_stage(
                state,
                run_id,
                job.target_label,
                job.stage,
                runs_root=runs_root,
                exit_code=int(exit_code),
                log_path=log_path,
            )
            effective = _effective_exit_code(
                int(exit_code),
                str(logs.target_log_path(runs_root, run_id, job.target_label, job.stage)),
            )
            counts["completed" if effective == 0 else "failed"] += 1
            continue

        # Local executor: the durable status file is authoritative for outcome,
        # never the in-memory Popen (which is gone after a daemon restart).
        token_ok = (
            status_doc is not None
            and status_doc.get("launch_token") == job.launch_token
        )

        if token_ok and status_doc.get("state") in ("exited", "success", "failed"):
            exit_code = int(status_doc.get("exit_code") or 0)
            _finalize_stage(
                state,
                run_id,
                job.target_label,
                job.stage,
                runs_root=runs_root,
                exit_code=exit_code,
            )
            counts["completed" if exit_code == 0 else "failed"] += 1
            continue

        pid = int(native_id) if native_id is not None else None
        alive = pid is not None and daemon.is_process_alive(pid)

        if alive and token_ok:
            # Our process is alive and the token matches: adopt, never relaunch.
            counts["adopted"] += 1
            counts["still_running"] += 1
            continue

        if (
            alive
            and not token_ok
            and _age_seconds(job.claimed_at) < _LOCAL_START_GRACE_S
        ):
            # Child alive but status file missing or still has a previous launch's
            # token (common on NFS right after relaunch). Trust DB native_id.
            counts["still_running"] += 1
            continue

        # Dead without an exit record, or stale/mismatched token past grace: requeue.
        _requeue_local_stage(
            state,
            run_id,
            job,
            runs_root=runs_root,
            reason="Local stage lost or stale; requeued",
            terminate_if_alive=alive,
        )
        counts["requeued"] += 1

    return counts


def _blocking_depth(stage: str, memo: dict[str, int] | None = None) -> int:
    memo = memo if memo is not None else {}
    if stage in memo:
        return memo[stage]
    deps = STAGE_DEPS.get(stage, [])
    if not deps:
        memo[stage] = 0
        return 0
    depth = max(_blocking_depth(dep, memo) for dep in deps) + 1
    memo[stage] = depth
    return depth


def _verify_outcome_still_applicable(state: PipelineState, key: VerifyTaskKey) -> bool:
    """True if a verify result may still be applied to SQLite for *key*."""
    run = state.get_run(key.run_id) or {}
    if run.get("force_rerun") and key.stage in set(state.get_active_stages(key.run_id)):
        # Selected stages are being force-rerun; do not artifact-skip them.
        return False
    row = state.get_stage_run(key.run_id, key.target_label, key.stage)
    if row is None or row.status not in (STATUS_PENDING, STATUS_EXTERNAL):
        return False
    if state.external_verify_complete(key.run_id, key.target_label, key.stage):
        return False
    return True


def _verify_worker():
    from syndiff_pipeline.template_runner.verify_worker import try_get_verify_worker

    return try_get_verify_worker()


def _cancel_verify_run(run_id: str) -> None:
    worker = _verify_worker()
    if worker is not None:
        worker.cancel_run(run_id)


def _cancel_verify_keys(keys: list[VerifyTaskKey]) -> None:
    worker = _verify_worker()
    if worker is not None:
        worker.cancel_keys(keys)


def _apply_verify_outcome(state: PipelineState, outcome: VerifyOutcome) -> int:
    """Persist one verify result; return 1 if the stage was skipped."""
    key = outcome.key
    if not _verify_outcome_still_applicable(state, key):
        return 0
    if outcome.error:
        state.cache_external_check(
            key.run_id, key.target_label, key.stage, complete=False
        )
        return 0
    if outcome.complete:
        state.mark_skipped(key.run_id, key.target_label, key.stage)
        state.cache_skip_reason(
            key.run_id, key.target_label, key.stage, SKIP_REASON_ARTIFACTS
        )
        state.cache_external_check(
            key.run_id,
            key.target_label,
            key.stage,
            complete=True,
            path=outcome.stable_path,
        )
        return 1
    state.cache_external_check(
        key.run_id, key.target_label, key.stage, complete=False
    )
    return 0


def _iter_verify_candidates(
    state: PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
) -> list[tuple]:
    """Collect uncached pending/external stages eligible for verification."""
    from syndiff_pipeline.template_runner.verify_worker import VerifyTaskKey

    active_stages = set(state.get_active_stages(run_id))
    cfg = ctx.cfg
    runs_root = cfg.runs_dir()
    candidates: list[tuple] = []
    rows_by_label: dict[str, list] = defaultdict(list)
    for row in state.list_stage_runs(run_id):
        rows_by_label[row.target_label].append(row)

    for target in ctx.targets:
        label = target.label()
        rows = rows_by_label.get(label)
        if not rows:
            continue
        resolved = resolve_config(target, cfg)
        for row in rows:
            if row.status not in (STATUS_PENDING, STATUS_EXTERNAL):
                continue
            if force_rerun and row.stage in active_stages:
                continue
            if row.stage == "ps1_download":
                if resolved.stages.ps1_process.ps1_source == "stream":
                    continue
                if state.get_skip_reason(run_id, label, "ps1_download") == SKIP_REASON_STREAM:
                    continue
            if (
                state.get_skip_reason(run_id, label, row.stage)
                in (SKIP_REASON_NOT_SELECTED, SKIP_REASON_SUPERSEDED)
            ):
                continue
            if row.status == STATUS_EXTERNAL:
                if not artifact_verify_needed(
                    state, run_id, label, row.stage, list(active_stages)
                ):
                    continue
            elif row.status == STATUS_PENDING:
                if row.stage in active_stages:
                    pass
                elif (
                    row.stage in run_stage_closure(active_stages)
                    and artifact_verify_needed(
                        state, run_id, label, row.stage, list(active_stages)
                    )
                ):
                    pass
                else:
                    continue
            else:
                continue
            if state.external_verify_complete(run_id, label, row.stage):
                continue
            if row.status == STATUS_PENDING and not state.deps_satisfied(
                run_id, label, row.stage, stages=resolved.stages
            ):
                continue
            manifest_path = str(
                logs.stage_manifest_path(runs_root, run_id, label, row.stage)
            )
            stable_path = str(
                logs.stable_stage_manifest_path(runs_root, label, row.stage)
            )
            candidates.append(
                (
                    VerifyTaskKey(run_id, label, row.stage),
                    row.status,
                    resolved,
                    manifest_path,
                    stable_path,
                )
            )

    def _sort_key(item: tuple) -> tuple:
        key, status, _resolved, _mp, _sp = item
        return (
            0 if status == STATUS_PENDING else 1,
            -_blocking_depth(key.stage),
            key.target_label,
            key.stage,
        )

    candidates.sort(key=_sort_key)
    return candidates


def _verify_backlog(
    state: PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
) -> tuple[int, int]:
    """Return (pending_candidate_count, in_flight_count) for artifact verify."""
    worker = _verify_worker()
    in_flight = worker.in_flight_count(run_id) if worker else 0
    pending = len(
        _iter_verify_candidates(state, run_id, ctx, force_rerun=force_rerun)
    )
    return pending, in_flight


def _collect_verify_status_by_run(
    state: PipelineState,
    runs: list[dict],
) -> dict[str, dict]:
    """Build per-run verify observability payload for host-local JSON."""
    worker = _verify_worker()
    by_run: dict[str, dict] = {}
    for run in runs:
        run_id = run["run_id"]
        ctx = _load_run_context(state, run_id)
        if ctx is None:
            by_run[run_id] = {"scan_running": 0, "scan_queued": 0, "active": []}
            continue
        force_rerun = bool((state.get_run(run_id) or {}).get("force_rerun"))
        pending, in_flight = _verify_backlog(
            state, run_id, ctx, force_rerun=force_rerun
        )
        active: list[list[str]] = []
        if worker is not None:
            active = [
                [key.target_label, key.stage]
                for key in worker.in_flight_keys(run_id)
            ]
        queued = max(0, pending - in_flight)
        by_run[run_id] = {
            "scan_running": in_flight,
            "scan_queued": queued,
            "active": active,
        }
    return by_run


def _run_verify_pass(
    state: PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
    budget: int,
    block: bool,
    block_timeout_s: float = 0.0,
) -> int:
    """Manifest fast path on main thread; full verify in background pool."""
    from syndiff_pipeline.template_runner.verify import check_manifests_only
    from syndiff_pipeline.template_runner.verify_worker import (
        BackfillTask,
        VerifyOutcome,
        VerifyTask,
        init_verify_worker,
    )

    worker = init_verify_worker(ctx.cfg.verify_max_workers)
    apply = lambda outcome: _apply_verify_outcome(state, outcome)
    max_in_flight = ctx.cfg.verify_max_workers
    budget_left = budget
    total = 0

    while budget_left > 0:
        worker.drain(apply, run_id=run_id, block=False)
        tasks: list[VerifyTask] = []
        backfills: list[BackfillTask] = []
        for key, _status, resolved, manifest_path, stable_path in _iter_verify_candidates(
            state, run_id, ctx, force_rerun=force_rerun
        ):
            if budget_left <= 0:
                break
            manifest_hit = check_manifests_only(
                resolved,
                key.stage,
                manifest_path=manifest_path,
                stable_manifest_path=stable_path,
            )
            if manifest_hit is True:
                budget_left -= 1
                if check_manifests_only(
                    resolved, key.stage, stable_manifest_path=stable_path
                ) is not True:
                    backfills.append(
                        BackfillTask(
                            manifest_path=manifest_path,
                            stable_path=stable_path,
                        )
                    )
                outcome = VerifyOutcome(
                    key=key,
                    complete=True,
                    stable_path=stable_path,
                    resolved=resolved,
                )
                total += apply(outcome)
                continue
            if worker.is_in_flight(key):
                continue
            if worker.in_flight_count(run_id) + len(tasks) >= max_in_flight:
                continue
            budget_left -= 1
            tasks.append(
                VerifyTask(
                    key=key,
                    manifest_path=manifest_path,
                    stable_path=stable_path,
                    resolved=resolved,
                )
            )

        if not tasks and not backfills:
            if worker.in_flight_count(run_id) == 0:
                break
            if not block:
                break
            total += worker.drain(
                apply,
                run_id=run_id,
                block=True,
                block_timeout_s=block_timeout_s,
            )
            continue

        worker.schedule_backfill(backfills)
        worker.schedule(tasks)
        total += worker.drain(
            apply,
            run_id=run_id,
            block=block,
            block_timeout_s=block_timeout_s,
        )
        if not block:
            break

    state.apply_superseded_skips(run_id, ctx.targets)
    return total


def _resolve_external_and_pending_skips(
    state: PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
    budget: int | None = None,
    block: bool = True,
) -> int:
    """Schedule artifact verification and optionally wait for it to finish."""
    if budget is None:
        budget = ctx.cfg.verify_budget_per_tick
    return _run_verify_pass(
        state,
        run_id,
        ctx,
        force_rerun=force_rerun,
        budget=budget,
        block=block,
    )


def _schedule_external_and_pending_skips(
    state: PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
    budget: int | None = None,
) -> None:
    """Non-blocking verify scheduling for the supervisor main loop."""
    if budget is None:
        budget = ctx.cfg.verify_budget_per_tick
    _run_verify_pass(
        state,
        run_id,
        ctx,
        force_rerun=force_rerun,
        budget=budget,
        block=False,
    )


def _cancel_verify_for_retry(
    run_id: str, target_label: str, stage: str, *, reset_downstream: bool
) -> None:
    from syndiff_pipeline.template_runner.verify_worker import VerifyTaskKey

    stages = [stage] + (
        downstream_stages(stage) if reset_downstream else []
    )
    keys = [VerifyTaskKey(run_id, target_label, s) for s in stages]
    _cancel_verify_keys(keys)


def _global_pool_running(state: PipelineState) -> dict[str, int]:
    """Running stage count per pool across ALL runs (global capacity)."""
    pool_running: dict[str, int] = defaultdict(int)
    for job in state.running_stage_runs(None):
        pool = STAGE_POOL.get(job.stage)
        if pool:
            pool_running[pool] += 1
    return pool_running


def _pool_capacity(pool_running: dict[str, int], pool_name: str, pool_cfg) -> int:
    return max(0, pool_cfg.max_concurrent - pool_running.get(pool_name, 0))


def _stall_reasons(state: PipelineState, run_id: str, ctx) -> List[str]:
    reasons: List[str] = []
    cfg = ctx.cfg
    pool_running = _global_pool_running(state)

    for row in state.list_stage_runs(run_id):
        if row.status not in (STATUS_PENDING, STATUS_READY, STATUS_BLOCKED, STATUS_EXTERNAL):
            continue
        label = row.target_label
        stage = row.stage
        if row.status == STATUS_BLOCKED:
            reasons.append(f"{label}/{stage}: blocked by upstream failure")
            continue
        target = next((t for t in ctx.targets if t.label() == label), None)
        stages = resolve_config(target, cfg).stages if target is not None else None
        if not state.deps_satisfied(run_id, label, stage, stages=stages):
            missing = []
            for dep in effective_stage_deps(stage, stages):
                dep_row = state.get_stage_run(run_id, label, dep)
                if dep_row is None or dep_row.status not in (STATUS_SUCCESS, STATUS_SKIPPED):
                    missing.append(f"{dep}={dep_row.status if dep_row else 'missing'}")
            reasons.append(f"{label}/{stage}: waiting on {', '.join(missing)}")
            continue
        if row.status in (STATUS_PENDING, STATUS_EXTERNAL):
            active_stages = state.get_active_stages(run_id)
            needs_verify = stage_needs_artifact_verify_display(
                state, run_id, label, stage, row.status, active_stages
            )
            if needs_verify and not state.external_verify_complete(
                run_id, label, stage
            ):
                reasons.append(f"{label}/{stage}: artifact verify queued")
            elif row.status == STATUS_PENDING:
                reasons.append(f"{label}/{stage}: pending promotion")
            else:
                reasons.append(f"{label}/{stage}: artifact verify queued")
            continue
        if row.status == STATUS_READY:
            pool = STAGE_POOL.get(stage)
            if pool:
                cap = cfg.resources.get(pool)
                if cap and pool_running[pool] >= cap.max_concurrent:
                    reasons.append(f"{label}/{stage}: pool {pool} saturated")
                else:
                    reasons.append(f"{label}/{stage}: ready but not claimed")
            else:
                reasons.append(f"{label}/{stage}: ready but not claimed")
    return reasons


def _try_launch_ready_row(
    state: PipelineState,
    run_id: str,
    ctx,
    row,
    *,
    pool_label: str,
    force_rerun: bool,
    active_stages: list[str],
    targets_by_label: dict,
    runs_root: str,
) -> bool:
    """Claim and launch one ready stage row. Returns True if launched."""
    if row.stage not in active_stages:
        return False
    target = targets_by_label.get(row.target_label)
    if target is None:
        return False

    cfg = ctx.cfg
    target_stages = resolve_config(target, cfg).stages
    if not state.deps_satisfied(run_id, row.target_label, row.stage, stages=target_stages):
        state.update_stage_status(run_id, row.target_label, row.stage, STATUS_PENDING)
        return False

    executor = cfg.stage_executor(row.stage)
    if executor == "condor" and row.native_id:
        condor.remove_cluster(int(row.native_id))

    launch_token = state.new_launch_token()
    if not state.claim_ready(run_id, row.target_label, row.stage, launch_token):
        return False

    if cfg.stage_executor(row.stage) == "local":
        logs.stage_status_path(
            runs_root, run_id, row.target_label, row.stage
        ).unlink(missing_ok=True)

    cmd = stages.build_stage_command(
        run_id,
        row.stage,
        str(ctx.run_dir),
        row.target_label,
        launch_token=launch_token,
        force_rerun=force_rerun,
    )
    log_path = str(logs.target_log_path(runs_root, run_id, row.target_label, row.stage))
    try:
        descriptor = launcher.launch_stage(
            cmd,
            cfg=cfg,
            stage=row.stage,
            runs_root=runs_root,
            run_id=run_id,
            target_label=row.target_label,
            launch_token=launch_token,
        )
    except Exception:
        log.exception("Launch failed for %s / %s; requeuing", row.target_label, row.stage)
        state.requeue_to_ready(run_id, row.target_label, row.stage, error_tail="Launch failed")
        return False

    state.set_launch_descriptor(
        run_id,
        row.target_label,
        row.stage,
        executor=descriptor.executor,
        native_id=descriptor.native_id,
        submit_epoch=descriptor.submit_epoch,
        log_path=log_path,
    )
    log.info(
        "Launched %s / %s (%s, %s, token=%s)",
        row.target_label,
        row.stage,
        pool_label,
        descriptor.executor,
        launch_token[:8],
    )
    return True


def _tick_run(state: PipelineState, run_id: str, ctx) -> None:
    run = state.get_run(run_id) or {}
    if state.is_paused(run_id):
        return

    force_rerun = bool(run.get("force_rerun"))
    reconcile_running_stages(state, run_id, ctx)
    state.apply_not_selected_skips(run_id, ctx.targets, ctx.cfg)
    state.apply_superseded_skips(run_id, ctx.targets)
    repaired = state.repair_orphaned_pending_upstream(run_id)
    if repaired:
        log.info(
            "Repaired %d orphaned pending upstream stage(s) in run %s",
            repaired,
            run_id,
        )
    _schedule_external_and_pending_skips(
        state,
        run_id,
        ctx,
        force_rerun=force_rerun,
        budget=ctx.cfg.verify_budget_per_tick,
    )
    targets_by_label = {t.label(): t for t in ctx.targets}
    target_stages_map = {
        label: resolve_config(target, ctx.cfg).stages
        for label, target in targets_by_label.items()
    }
    state.promote_stages(run_id, target_stages_map)

    cfg = ctx.cfg
    runs_root = cfg.runs_dir()
    active_stages = state.get_active_stages(run_id)

    # Pool capacity is enforced GLOBALLY across all active runs.
    pool_running = _global_pool_running(state)

    launch_kwargs = dict(
        force_rerun=force_rerun,
        active_stages=active_stages,
        targets_by_label=targets_by_label,
        runs_root=runs_root,
    )
    for row in state.fetch_ready_unpooled(run_id):
        _try_launch_ready_row(state, run_id, ctx, row, pool_label="unpooled", **launch_kwargs)

    for pool_name, pool_cfg in cfg.resources.items():
        capacity = _pool_capacity(pool_running, pool_name, pool_cfg)
        if capacity <= 0:
            continue
        batch = state.fetch_ready_batch(run_id, pool_name, capacity)
        for row in batch:
            _try_launch_ready_row(
                state, run_id, ctx, row, pool_label=pool_name, **launch_kwargs
            )

    counts = state.count_by_status(run_id)
    running = counts.get(STATUS_RUNNING, 0)
    launchable = sum(
        1
        for row in state.list_stage_runs(run_id)
        if row.status == STATUS_READY
        and state.deps_satisfied(
            run_id,
            row.target_label,
            row.stage,
            stages=target_stages_map.get(row.target_label),
        )
    )
    nonterminal = sum(counts.get(s, 0) for s in NONTERMINAL_STATUSES)

    prev_status = run.get("status")
    from syndiff_pipeline.template_runner.notifications import notifier_for_context

    notifier = notifier_for_context(state, ctx)

    if nonterminal == 0:
        final = derive_run_final_status(counts)
        prev_terminal = prev_status in TERMINAL_RUN_STATUSES
        state.set_run_status(run_id, final)
        if not prev_terminal:
            log.info("Run %s complete: %s", run_id, final)
            # Canceled runs already received notify_run_canceled when the intent
            # was applied; do not also emit run_completed(success).
            if notifier is not None and final != RUN_CANCELED:
                notifier.notify_run_completed(run_id, runs_root, outcome=final)
        elif final != prev_status:
            log.info(
                "Run %s terminal status corrected: %s -> %s",
                run_id,
                prev_status,
                final,
            )
        _write_summary(state, run_id, runs_root)
        return

    worker = _verify_worker()
    verify_in_flight = worker.in_flight_count(run_id) if worker else 0
    verify_pending, _ = _verify_backlog(
        state, run_id, ctx, force_rerun=bool(run.get("force_rerun"))
    )
    if (
        running == 0
        and launchable == 0
        and nonterminal > 0
        and verify_pending == 0
        and verify_in_flight == 0
    ):
        reasons = _stall_reasons(state, run_id, ctx)
        reason_text = "; ".join(reasons[:8])
        state.set_run_status(run_id, "stalled", stall_reason=reason_text)
        log.warning("Run %s stalled: %s", run_id, reason_text)
        if notifier is not None and prev_status != "stalled":
            notifier.notify_run_stalled(run_id, runs_root, stall_reason=reason_text)
    elif prev_status == "stalled" and (
        running > 0 or launchable > 0 or verify_pending > 0 or verify_in_flight > 0
    ):
        state.set_run_status(run_id, "running", stall_reason="")
        if notifier is not None:
            notifier.notify_run_resumed(run_id)

    _write_summary(state, run_id, runs_root)


def _requeue_local_stage(
    state: PipelineState,
    run_id: str,
    job,
    *,
    runs_root: str,
    reason: str,
    terminate_if_alive: bool,
) -> None:
    """Requeue a local running stage, terminating a live worker first."""
    if terminate_if_alive and job.native_id is not None:
        _terminate_job(job)
    log.info("Requeued %s / %s: %s", job.target_label, job.stage, reason)
    state.requeue_running_stage(
        run_id,
        job.target_label,
        job.stage,
        error_tail=reason,
    )
    _notify_stage_outcome(
        state,
        run_id,
        target_label=job.target_label,
        stage=job.stage,
        outcome="died",
        runs_root=runs_root,
        finished_at=_utc_now(),
        error_tail=reason,
    )


def _terminate_job(job) -> None:
    """Terminate a single running stage's worker (condor cluster or local pid)."""
    if job.native_id is None:
        return
    if job.executor == "condor":
        condor.remove_cluster(int(job.native_id))
    else:
        daemon.terminate_process_tree(int(job.native_id))


def _terminate_run_jobs(state: PipelineState, run_id: str) -> None:
    for job in state.running_stage_runs(run_id):
        _terminate_job(job)


def _apply_commands(state: PipelineState) -> None:
    for cmd in state.fetch_pending_commands():
        args = json.loads(cmd.args_json or "{}")
        try:
            if cmd.kind == "cancel" and cmd.run_id:
                # Terminate live workers BEFORE marking rows canceled so a
                # killed run never leaves orphaned processes/clusters running.
                running_before = state.running_stage_runs(cmd.run_id)
                _terminate_run_jobs(state, cmd.run_id)
                _cancel_verify_run(cmd.run_id)
                state.apply_cancel_run(cmd.run_id)
                _notify_run_canceled(state, cmd.run_id, running_before)
            elif cmd.kind == "pause" and cmd.run_id:
                state.set_paused(cmd.run_id, True)
            elif cmd.kind == "resume" and cmd.run_id:
                state.set_paused(cmd.run_id, False)
            elif cmd.kind == "retry" and cmd.run_id:
                if args.get("target_label") and args.get("stage"):
                    # If the targeted stage is still running, stop the worker
                    # first to avoid a duplicate when it is relaunched.
                    row = state.get_stage_run(
                        cmd.run_id, args["target_label"], args["stage"]
                    )
                    if row and row.status == STATUS_RUNNING:
                        _terminate_job(row)
                    reset_downstream = bool(args.get("reset_downstream", True))
                    state.apply_retry_stage(
                        cmd.run_id,
                        args["target_label"],
                        args["stage"],
                        reset_downstream=reset_downstream,
                    )
                    _cancel_verify_for_retry(
                        cmd.run_id,
                        args["target_label"],
                        args["stage"],
                        reset_downstream=reset_downstream,
                    )
                    _notify_run_retried(
                        state,
                        cmd.run_id,
                        target_label=args["target_label"],
                        stage=args["stage"],
                        reset_downstream=reset_downstream,
                    )
                else:
                    state.apply_retry_run(cmd.run_id)
                    _cancel_verify_run(cmd.run_id)
                    _notify_run_retried(state, cmd.run_id)
            elif cmd.kind == "force_rerun" and cmd.run_id:
                labels = args.get("target_labels") or []
                stages_arg = args.get("stages") or []
                # Stop any worker for a targeted stage first so the reset to
                # pending cannot orphan a live process / duplicate it on relaunch.
                for label in labels:
                    for stage in stages_arg:
                        row = state.get_stage_run(cmd.run_id, label, stage)
                        if row and row.status == STATUS_RUNNING:
                            _terminate_job(row)
                state.apply_force_rerun(cmd.run_id, labels, stages_arg)
                _cancel_verify_run(cmd.run_id)
            else:
                log.warning("Unknown or incomplete command id=%s kind=%s", cmd.id, cmd.kind)
        finally:
            state.mark_command_processed(cmd.id)


def run_supervisor_daemon(handoff_root: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [supervisor] %(message)s",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    global _lock_fd
    # Non-blocking exclusive lock: a second daemon must fail to acquire and exit,
    # not block forever waiting for the incumbent owner.
    from syndiff_pipeline.template_runner.workspace import state_db_path

    db_path = str(state_db_path(handoff_root))
    with daemon.daemon_lock(handoff_root, blocking=False) as fd:
        if fd is None:
            log.info("Another supervisor already owns the lock; exiting.")
            return 0
        _lock_fd = fd

        pid_path = logs.daemon_pid_path(handoff_root)
        pid = os.getpid()
        daemon.write_pid(pid_path, pid)
        state = PipelineState(db_path)
        # Establish liveness immediately (local file is authoritative) before
        # the first — potentially slow — scheduling pass begins.
        _write_local_heartbeat(handoff_root)
        state.update_supervisor_heartbeat(pid)

        heartbeat_thread = threading.Thread(
            target=_supervisor_heartbeat_loop,
            args=(state, handoff_root, pid, _HEARTBEAT_INTERVAL_S),
            name="supervisor-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        _ensure_discord_bot_on_startup(handoff_root)

        try:
            while not _shutdown:
                _apply_commands(state)

                for run in state.list_active_runs():
                    if _shutdown:
                        break
                    run_id = run["run_id"]
                    try:
                        ctx = _load_run_context(state, run_id)
                        if ctx is None:
                            continue
                        _tick_run(state, run_id, ctx)
                        # Honor cancel/pause/stop intents promptly even when a
                        # large active-run set makes a full pass slow.
                        _apply_commands(state)
                    except Exception:
                        # Isolate per-run failures so one bad run cannot take
                        # down scheduling for every other active run.
                        log.exception("Error while processing run %s", run_id)

                worker = _verify_worker()
                active_runs = state.list_active_runs()
                by_run = _collect_verify_status_by_run(state, active_runs)
                write_verify_in_flight(handoff_root, by_run)

                # Interruptible idle: wake early on shutdown instead of sleeping
                # through a SIGTERM.
                for _ in range(10):
                    if _shutdown:
                        break
                    time.sleep(0.1)
        finally:
            worker = _verify_worker()
            if worker is not None:
                if _shutdown:
                    try:
                        apply = lambda outcome: _apply_verify_outcome(state, outcome)
                        worker.drain(
                            apply,
                            block=True,
                            block_timeout_s=_SHUTDOWN_VERIFY_DRAIN_S,
                        )
                        worker.drain(apply, block=False)
                    except Exception:
                        log.exception("Error draining verify worker on shutdown")
                from syndiff_pipeline.template_runner.verify_worker import shutdown_verify_worker

                shutdown_verify_worker(wait=_shutdown)
            state.clear_supervisor()
            daemon.remove_pid_file(pid_path)
            try:
                logs.daemon_heartbeat_file(handoff_root).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                clear_verify_in_flight(handoff_root)
            except OSError:
                pass
    return 0


def run_scheduler(
    run_id: str,
    run_dir: str,
    stages_arg: str | None = None,
    force_rerun: bool = False,
) -> int:
    """Foreground single-run mode (debug): runs one tick loop inline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [scheduler] %(message)s",
    )
    ctx = resolve_run_context(run_dir=run_dir, run_id=run_id)
    state = PipelineState(ctx.cfg.state_db_path)
    active = stages.parse_stage_list(stages_arg)

    run_row = state.get_run(run_id)
    if run_row is not None:
        log.error(
            "Run %s already exists; choose a new --run-id for submit/run, "
            "or use syndiff-template retry for failed stages.",
            run_id,
        )
        return 1
    state.create_run(
        run_id,
        str(logs.run_config_path(ctx.run_dir)),
        str(logs.run_targets_path(ctx.run_dir)),
        ctx.cfg.runs_dir(),
        ctx.targets,
        active,
        force_rerun=force_rerun,
    )

    state.set_run_status(run_id, "running")
    while True:
        _tick_run(state, run_id, ctx)
        run = state.get_run(run_id) or {}
        if run.get("status") in ("success", "failed", "canceled"):
            break
        if run.get("status") == "stalled":
            log.error("Run stalled: %s", run.get("stall_reason"))
            return 1
        time.sleep(1.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Template pipeline supervisor")
    parser.add_argument("--daemon", action="store_true", help="Run global supervisor daemon")
    parser.add_argument(
        "--deployment",
        default=None,
        help="Path to deployment.yaml (required for --daemon)",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None, help="Path to run directory with frozen config")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)

    if args.daemon:
        if not args.deployment:
            raise SystemExit("--deployment required for --daemon")
        deploy_path = Path(args.deployment).expanduser().resolve()
        handoff_root = str(load_handoff_root_from_deployment(deploy_path))
        record_deployment_path(handoff_root, deploy_path)
        return run_supervisor_daemon(handoff_root)

    if not args.run_id or not args.run_dir:
        raise SystemExit("--run-id and --run-dir required without --daemon")
    return run_scheduler(
        args.run_id,
        args.run_dir,
        args.stages,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    raise SystemExit(main())
