"""Multi-run supervisor daemon for template pipeline runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

from syndiff_pipeline.template_runner import condor, daemon, launcher, logs, stages
from syndiff_pipeline.template_runner.run_context import resolve_run_context
from syndiff_pipeline.template_runner.runner_config import resolve_config
from syndiff_pipeline.template_runner.state import (
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
    PipelineState,
    _utc_now,
)
from syndiff_pipeline.template_runner.verify import (
    collect_stage_artifacts,
    manifest_valid,
    read_manifest,
    stage_complete,
    write_manifest,
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
    if exit_code == 0:
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            STATUS_SUCCESS,
            finished_at=_utc_now(),
            exit_code=0,
            log_path=log_path,
        )
    elif exit_code == 143:
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            STATUS_CANCELED,
            finished_at=_utc_now(),
            exit_code=exit_code,
            log_path=log_path,
            error_tail=error_tail or "Canceled (SIGTERM)",
        )
    else:
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            STATUS_FAILED,
            finished_at=_utc_now(),
            exit_code=exit_code,
            log_path=log_path,
            error_tail=error_tail,
        )
        state.block_downstream(run_id, target_label, stage)
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
                    state.requeue_running_stage(
                        run_id,
                        job.target_label,
                        job.stage,
                        error_tail="Condor stage claimed but never submitted; requeued",
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

        if alive and status_doc is None and _age_seconds(job.claimed_at) < _LOCAL_START_GRACE_S:
            # Freshly claimed: the child has not written its status file yet.
            counts["still_running"] += 1
            continue

        # Dead without an exit record, or stale/mismatched token: requeue to ready.
        state.requeue_running_stage(
            run_id,
            job.target_label,
            job.stage,
            error_tail="Local stage lost or stale; requeued",
        )
        counts["requeued"] += 1

    return counts


def _ensure_stable_manifest(resolved, stage: str, stable_path: str) -> None:
    """Write a cross-run completion manifest if a valid one is not already present.

    Called when a stage is confirmed complete (on disk or via a per-run
    manifest). Best-effort: any failure is logged and ignored so manifest
    bookkeeping never blocks scheduling.
    """
    existing = read_manifest(stable_path)
    if existing is not None and manifest_valid(existing, resolved, stage):
        return
    try:
        expected, produced, artifacts = collect_stage_artifacts(resolved, stage)
        write_manifest(stable_path, resolved, stage, artifacts, expected, produced)
    except Exception as exc:  # noqa: BLE001 - manifest write must never be fatal
        log.debug("Could not write stable manifest %s for %s: %s", stable_path, stage, exc)


def _resolve_external_and_pending_skips(
    state: PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
) -> int:
    if force_rerun:
        return 0
    skipped = 0
    cfg = ctx.cfg
    runs_root = cfg.runs_dir()
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
            # Cached completeness result -> never re-verify (kills the per-tick
            # NFS verify storm), regardless of complete/incomplete outcome.
            if state.external_checked(run_id, label, row.stage):
                continue
            # Only verify a pending stage once its deps are satisfied (about to
            # run). Pending stages with unmet deps must not trigger disk checks.
            if row.status == STATUS_PENDING and not state.deps_satisfied(
                run_id, label, row.stage
            ):
                continue
            manifest_path = str(
                logs.stage_manifest_path(runs_root, run_id, label, row.stage)
            )
            stable_path = str(
                logs.stable_stage_manifest_path(runs_root, label, row.stage)
            )
            complete = stage_complete(
                resolved,
                row.stage,
                manifest_path=manifest_path,
                stable_manifest_path=stable_path,
            )
            if complete:
                # Self-healing backfill: persist a cross-run manifest so the next
                # run skips re-scanning this already-complete output entirely.
                _ensure_stable_manifest(resolved, row.stage, stable_path)
                state.mark_skipped(run_id, label, row.stage)
                state.cache_external_check(
                    run_id, label, row.stage, complete=True, path=stable_path
                )
                skipped += 1
            elif row.status == STATUS_EXTERNAL:
                # An external (out-of-selection) stage that is not present on
                # disk will never be produced in this run: cache the negative
                # result so it is not re-verified every tick.
                state.cache_external_check(run_id, label, row.stage, complete=False)
    return skipped


def _global_pool_running(state: PipelineState) -> dict[str, int]:
    """Running stage count per pool across ALL runs (global capacity)."""
    pool_running: dict[str, int] = defaultdict(int)
    for job in state.running_stage_runs(None):
        pool_running[STAGE_POOL.get(job.stage, "?")] += 1
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
        if not state.deps_satisfied(run_id, label, stage):
            missing = []
            for dep in STAGE_DEPS.get(stage, []):
                dep_row = state.get_stage_run(run_id, label, dep)
                if dep_row is None or dep_row.status not in (STATUS_SUCCESS, STATUS_SKIPPED):
                    missing.append(f"{dep}={dep_row.status if dep_row else 'missing'}")
            reasons.append(f"{label}/{stage}: waiting on {', '.join(missing)}")
            continue
        if row.status in (STATUS_PENDING, STATUS_EXTERNAL):
            reasons.append(f"{label}/{stage}: pending promotion or artifact verify")
            continue
        if row.status == STATUS_READY:
            pool = STAGE_POOL.get(stage, "?")
            cap = cfg.resources.get(pool)
            if cap and pool_running[pool] >= cap.max_concurrent:
                reasons.append(f"{label}/{stage}: pool {pool} saturated")
            else:
                reasons.append(f"{label}/{stage}: ready but not claimed")
    return reasons


def _tick_run(state: PipelineState, run_id: str, ctx) -> None:
    run = state.get_run(run_id) or {}
    if state.is_paused(run_id):
        return

    force_rerun = bool(run.get("force_rerun"))
    reconcile_running_stages(state, run_id, ctx)
    _resolve_external_and_pending_skips(state, run_id, ctx, force_rerun=force_rerun)
    state.promote_stages(run_id)

    cfg = ctx.cfg
    runs_root = cfg.runs_dir()
    targets_by_label = {t.label(): t for t in ctx.targets}
    active_stages = state.get_active_stages(run_id)

    # Pool capacity is enforced GLOBALLY across all active runs.
    pool_running = _global_pool_running(state)

    for pool_name, pool_cfg in cfg.resources.items():
        capacity = _pool_capacity(pool_running, pool_name, pool_cfg)
        if capacity <= 0:
            continue
        batch = state.fetch_ready_batch(run_id, pool_name, capacity)
        for row in batch:
            if row.stage not in active_stages:
                continue
            if not state.deps_satisfied(run_id, row.target_label, row.stage):
                state.update_stage_status(
                    run_id, row.target_label, row.stage, STATUS_PENDING
                )
                continue

            target = targets_by_label.get(row.target_label)
            if target is None:
                continue

            executor = cfg.stage_executor(row.stage)
            # Clean up any stale Condor cluster recorded for a prior attempt
            # before we relaunch (best-effort; orphans also swept via audit files).
            if executor == "condor" and row.native_id:
                condor.remove_cluster(int(row.native_id))

            # Reserve the slot atomically BEFORE launching. A crash between
            # launch and claim could otherwise orphan a process that gets
            # relaunched on restart (the row would still be 'ready').
            launch_token = state.new_launch_token()
            if not state.claim_ready(run_id, row.target_label, row.stage, launch_token):
                continue

            cmd = stages.build_stage_command(
                run_id,
                row.stage,
                str(ctx.run_dir),
                row.target_label,
                launch_token=launch_token,
                force_rerun=force_rerun,
            )
            log_path = str(
                logs.target_log_path(runs_root, run_id, row.target_label, row.stage)
            )
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
                log.exception(
                    "Launch failed for %s / %s; requeuing", row.target_label, row.stage
                )
                state.requeue_to_ready(
                    run_id, row.target_label, row.stage, error_tail="Launch failed"
                )
                continue

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
                pool_name,
                descriptor.executor,
                launch_token[:8],
            )

    counts = state.count_by_status(run_id)
    running = counts.get(STATUS_RUNNING, 0)
    launchable = sum(
        1
        for row in state.list_stage_runs(run_id)
        if row.status == STATUS_READY and state.deps_satisfied(run_id, row.target_label, row.stage)
    )
    nonterminal = sum(counts.get(s, 0) for s in NONTERMINAL_STATUSES)

    if nonterminal == 0:
        final = "success" if counts.get(STATUS_FAILED, 0) == 0 else "failed"
        state.set_run_status(run_id, final)
        log.info("Run %s complete: %s", run_id, final)
        _write_summary(state, run_id, runs_root)
        return

    if running == 0 and launchable == 0 and nonterminal > 0:
        reasons = _stall_reasons(state, run_id, ctx)
        reason_text = "; ".join(reasons[:8])
        state.set_run_status(run_id, "stalled", stall_reason=reason_text)
        log.warning("Run %s stalled: %s", run_id, reason_text)
    elif run.get("status") == "stalled" and (running > 0 or launchable > 0):
        state.set_run_status(run_id, "running")

    _write_summary(state, run_id, runs_root)


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
                _terminate_run_jobs(state, cmd.run_id)
                state.apply_cancel_run(cmd.run_id)
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
                    state.apply_retry_stage(
                        cmd.run_id,
                        args["target_label"],
                        args["stage"],
                        reset_downstream=bool(args.get("reset_downstream", True)),
                    )
                else:
                    state.apply_retry_run(cmd.run_id)
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
            else:
                log.warning("Unknown or incomplete command id=%s kind=%s", cmd.id, cmd.kind)
        finally:
            state.mark_command_processed(cmd.id)


def run_supervisor_daemon(state_db_path: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [supervisor] %(message)s",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    global _lock_fd
    # Non-blocking exclusive lock: a second daemon must fail to acquire and exit,
    # not block forever waiting for the incumbent owner.
    with daemon.daemon_lock(state_db_path, blocking=False) as fd:
        if fd is None:
            log.info("Another supervisor already owns the lock; exiting.")
            return 0
        _lock_fd = fd

        pid_path = logs.daemon_pid_path(state_db_path)
        daemon.write_pid(pid_path, os.getpid())
        state = PipelineState(state_db_path)
        state.update_supervisor_heartbeat(os.getpid())

        try:
            while not _shutdown:
                state.update_supervisor_heartbeat(os.getpid())
                _apply_commands(state)

                for run in state.list_active_runs():
                    run_id = run["run_id"]
                    try:
                        ctx = _load_run_context(state, run_id)
                        if ctx is None:
                            continue
                        _tick_run(state, run_id, ctx)
                    except Exception:
                        # Isolate per-run failures so one bad run cannot take
                        # down scheduling for every other active run.
                        log.exception("Error while processing run %s", run_id)

                time.sleep(1.0)
        finally:
            state.clear_supervisor()
            daemon.remove_pid_file(pid_path)
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
    if run_row is None:
        state.create_run(
            run_id,
            str(logs.run_config_path(ctx.run_dir)),
            str(logs.run_targets_path(ctx.run_dir)),
            ctx.cfg.runs_dir(),
            ctx.targets,
            active,
            force_rerun=force_rerun,
        )
    elif force_rerun:
        state.reset_stages_for_force_rerun(
            run_id, [t.label() for t in ctx.targets], active
        )
        state.set_run_status(run_id, "running")

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
    parser.add_argument("--state-db", default=None, help="SQLite state DB path (daemon mode)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None, help="Path to run directory with frozen config")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args(argv)

    if args.daemon:
        if not args.state_db:
            raise SystemExit("--state-db required for --daemon")
        return run_supervisor_daemon(args.state_db)

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
