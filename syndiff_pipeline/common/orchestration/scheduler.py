"""Multi-run supervisor daemon for template pipeline runs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import signal
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List

from syndiff_pipeline.common.orchestration import condor, daemon, launcher, lease, logs, stage_liveness
from syndiff_pipeline.common.orchestration.deployment import load_workspace_root_from_deployment
from syndiff_pipeline.common.orchestration.workspace import record_deployment_path
import syndiff_pipeline.common.orchestration.state as pstate
from syndiff_pipeline.common.orchestration.verify_status import (
    clear_verify_in_flight,
    write_verify_in_flight,
)

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.verify_worker import (
        BackfillTask,
        VerifyOutcome,
        VerifyTask,
        VerifyTaskKey,
    )

log = logging.getLogger(__name__)

_shutdown = False
_lock_fd: int | None = None
_owned_lease: lease.Lease | None = None

NONTERMINAL_STATUSES = frozenset(
    {
        pstate.STATUS_PENDING,
        pstate.STATUS_READY,
        pstate.STATUS_RUNNING,
        pstate.STATUS_BLOCKED,
        pstate.STATUS_EXTERNAL,
    }
)

# Grace window after an atomic claim before a local job that has not yet written
# its status file is treated as lost. Heavy stage imports (numpy/zarr over NFS)
# can delay the first status write by many seconds, so keep this generous.
_LOCAL_START_GRACE_S = 300.0

# Cadence of the background heartbeat thread. It must be well under the
# staleness threshold used by the lifecycle layer (DEFAULT_HEARTBEAT_STALE_S).
_HEARTBEAT_INTERVAL_S = 15.0

# Provenance spool-ingest drain cadence (template_bookkeeping_plan.md §10/§15:
# "supervisor rotates + drains ... global, once per pass, throttled"). The
# supervisor is the sole DB writer, so this throttle just bounds how often we
# pay the rotate+drain cost per loop pass -- it is not a correctness lock.
_PROVENANCE_DRAIN_INTERVAL_S = 5.0
_last_provenance_drain_ts = 0.0

# On SIGTERM, drain in-flight verify results briefly before dropping them.
_SHUTDOWN_VERIFY_DRAIN_S = 5.0

# If the host-local heartbeat file cannot be written for this long, something is
# badly wrong locally. Rather than linger as a zombie that holds the lock but
# looks dead, the supervisor exits so a fresh one (which reconciles in-flight
# jobs from durable status files) can take over.
_HEARTBEAT_FATAL_AFTER_S = 90.0

# If the NFS lease cannot be renewed for this long, exit so another host can
# reclaim ownership after the lease goes stale.
_LEASE_RENEW_FATAL_AFTER_S = lease.DEFAULT_LEASE_STALE_S


def _start_in_process_discord_bot(workspace_root: str):
    """Start the Discord bot inside this supervisor process when configured."""
    from syndiff_pipeline.template_creation.orchestration.discord_bot import (
        InProcessDiscordBot,
    )
    from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
        should_start_in_process_bot,
    )

    should_start, reason, deploy_path = should_start_in_process_bot(workspace_root)
    if not should_start or deploy_path is None:
        if reason:
            log.info("Discord bot not started: %s", reason)
        return None
    bot = InProcessDiscordBot(deploy_path)
    if bot.start():
        return bot
    if bot.skipped_reason:
        log.warning("Discord bot not started: %s", bot.skipped_reason)
    return None


def _cleanup_legacy_bots_async(workspace_root: str) -> None:
    """Best-effort cleanup of legacy ``--detached`` bots only.

    Full singleton cleanup (all spawn styles) runs in ``InProcessDiscordBot.start``
    / ``stop`` so this background scan must not kill a freshly spawned bot.
    """
    from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
        discover_legacy_detached_bot_pids,
    )
    from syndiff_pipeline.common.orchestration import daemon as daemon_mod
    import signal as signal_mod

    try:
        pids = discover_legacy_detached_bot_pids(workspace_root)
        for pid in pids:
            try:
                daemon_mod.terminate_process_tree(pid, signal_mod.SIGTERM)
            except Exception:
                log.warning("Failed to terminate legacy Discord bot pid=%s", pid, exc_info=True)
        if pids:
            log.info(
                "Signaled %s legacy detached Discord bot process(es) for cleanup",
                len(pids),
            )
    except Exception:
        log.warning("Legacy Discord bot cleanup failed", exc_info=True)


def _maybe_respawn_discord_bot(workspace_root: str, discord_bot) -> object | None:
    """Respawn the Discord bot only when the lease is stale and no local bots remain."""
    from syndiff_pipeline.common.orchestration import lease as lease_mod
    from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
        discover_workspace_bot_pids,
        should_start_in_process_bot,
    )

    if discord_bot is not None and discord_bot.running:
        return discord_bot
    if lease_mod.bot_lease_is_alive(workspace_root):
        return discord_bot
    if discover_workspace_bot_pids(workspace_root):
        return discord_bot
    should_start, reason, deploy_path = should_start_in_process_bot(workspace_root)
    if not should_start or deploy_path is None:
        return discord_bot
    if discord_bot is None:
        from syndiff_pipeline.template_creation.orchestration.discord_bot import (
            InProcessDiscordBot,
        )

        discord_bot = InProcessDiscordBot(deploy_path)
    if discord_bot.start():
        log.info("Discord bot respawned after lease stale / no local bots")
        return discord_bot
    if discord_bot.skipped_reason:
        log.warning("Discord bot respawn skipped: %s", discord_bot.skipped_reason)
    return discord_bot


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


def _iso_to_epoch(iso_ts: str | None) -> float | None:
    """Parse ISO-8601 timestamp to UTC epoch seconds."""
    if not iso_ts:
        return None
    try:
        ts = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def _handle_signal(signum, frame):
    """Handle signal.
    
    Parameters
    ----------
    signum
    frame"""
    global _shutdown
    log.warning("Received signal %s — shutting down supervisor gracefully", signum)
    _shutdown = True


def _write_local_heartbeat(workspace_root: str) -> None:
    """Write the host-local heartbeat file (NFS-independent liveness signal)."""
    path = logs.daemon_heartbeat_file(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(str(time.time()), encoding="utf-8")
    tmp.replace(path)


def _supervisor_heartbeat_loop(
    state: pstate.PipelineState,
    workspace_root: str,
    pid: int,
    interval_s: float,
    owned_lease: lease.Lease,
) -> None:
    """Keep local heartbeat + NFS lease fresh; honor stop requests and fencing.

    The host-local heartbeat file detects a wedged local process independent of
    NFS. The shared lease is the cross-host ownership signal.
    """
    global _shutdown, _owned_lease
    last_local_ok = time.monotonic()
    last_lease_ok = time.monotonic()
    current_lease = owned_lease
    while not _shutdown:
        # Check stop / fencing frequently; sleep in short slices so remote stop
        # is honored well under the full renew interval.
        for _ in range(max(1, int(interval_s / 0.5))):
            if _shutdown:
                break
            stop = lease.read_stop_request(workspace_root)
            if lease.stop_targets_owner(stop, current_lease):
                log.warning(
                    "Stop request received (from=%s generation=%s) — shutting down",
                    stop.requested_by_host if stop else "?",
                    stop.target_generation if stop else None,
                )
                _shutdown = True
                break
            time.sleep(0.5)
        if _shutdown:
            break

        try:
            _write_local_heartbeat(workspace_root)
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
            renewed = lease.renew_lease(workspace_root, current_lease, pid=pid)
            if renewed is None:
                log.error(
                    "Lost lease ownership (generation=%s) — shutting down",
                    current_lease.generation,
                )
                _shutdown = True
                break
            current_lease = renewed
            _owned_lease = renewed
            last_lease_ok = time.monotonic()
        except Exception:
            log.exception("Failed to renew ownership lease")
            if time.monotonic() - last_lease_ok > _LEASE_RENEW_FATAL_AFTER_S:
                log.error(
                    "Lease renew failed for >%ss — exiting so another host can reclaim",
                    _LEASE_RENEW_FATAL_AFTER_S,
                )
                _shutdown = True
                break

        try:
            state.update_supervisor_heartbeat(pid)
        except Exception:
            # DB heartbeat is best-effort; lease + local file are authoritative.
            log.warning("Failed to update DB heartbeat (best-effort)", exc_info=True)


def _write_summary(state: pstate.PipelineState, run_id: str, runs_root: str) -> None:
    """Write summary.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    runs_root : str"""
    counts = state.count_by_status(run_id)
    summary = {"run_id": run_id, "counts": counts, "updated_at": pstate._utc_now()}
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


def _load_run_context(state: pstate.PipelineState, run_id: str):
    """Load run context for *run_id*, or None if the run cannot be loaded.

    Never raises: a broken run directory must not take down the supervisor.
    """
    run = state.get_run(run_id)
    if not run:
        return None
    runs_root = run["runs_root"]
    run_dir = logs.run_dir(runs_root, run_id)
    try:
        from syndiff_pipeline.common.orchestration.run_context import resolve_run_context

        return resolve_run_context(run_dir=run_dir, run_id=run_id)
    except Exception as exc:
        # Missing/broken run directory or bad targets.csv must not crash the
        # supervisor; skip this run until it is fixed or canceled.
        log.error("Cannot load run context for %s: %s", run_id, exc, exc_info=True)
        return None


def _read_status_file(
    runs_root: str, run_id: str, target_label: str, stage: str
) -> dict | None:
    """Read status file.
    
    Parameters
    ----------
    runs_root : str
    run_id : str
    target_label : str
    stage : str
    
    Returns
    -------
    dict | None"""
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
    state: pstate.PipelineState,
    run_id: str,
    *,
    target_label: str,
    stage: str,
    outcome: str,
    runs_root: str,
    finished_at: str,
    error_tail: str | None = None,
) -> None:
    """Notify stage outcome.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    target_label : str
    stage : str
    outcome : str
    runs_root : str
    finished_at : str
    error_tail : str | None, optional, default ``None``"""
    ctx = _load_run_context(state, run_id)
    if ctx is None:
        return
    from syndiff_pipeline.common.orchestration.notifications import notifier_for_context

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
    state: pstate.PipelineState,
    run_id: str,
    *,
    target_label: str | None = None,
    stage: str | None = None,
    reset_downstream: bool | None = None,
) -> None:
    """Notify run retried.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    target_label : str | None, optional, default ``None``
    stage : str | None, optional, default ``None``
    reset_downstream : bool | None, optional, default ``None``"""
    ctx = _load_run_context(state, run_id)
    if ctx is None:
        return
    from syndiff_pipeline.common.orchestration.notifications import notifier_for_context

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


def _notify_run_canceled(state: pstate.PipelineState, run_id: str, running_before) -> None:
    """Notify run canceled.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    running_before"""
    ctx = _load_run_context(state, run_id)
    if ctx is None:
        return
    from syndiff_pipeline.common.orchestration.notifications import notifier_for_context

    notifier = notifier_for_context(state, ctx)
    if notifier is None:
        return
    runs_root = ctx.cfg.runs_dir()
    finished_at = pstate._utc_now()
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


def _resolve_local_pid(
    job,
    status_doc: dict | None,
    *,
    token_ok: bool,
) -> tuple[int | None, bool, int | None]:
    """Resolve PID, liveness, and status-file PID for a local running stage."""
    status_pid: int | None = None
    if status_doc is not None:
        raw_status_pid = status_doc.get("pid")
        if raw_status_pid is not None:
            try:
                status_pid = int(raw_status_pid)
            except (TypeError, ValueError):
                status_pid = None

    pid = int(job.native_id) if job.native_id is not None else None
    if pid is None and token_ok and status_pid is not None:
        pid = status_pid

    alive = pid is not None and daemon.is_process_alive(pid)
    return pid, alive, status_pid


def _finalize_stage(
    state: pstate.PipelineState,
    run_id: str,
    target_label: str,
    stage: str,
    *,
    runs_root: str,
    exit_code: int,
    log_path: str | None = None,
) -> None:
    """Finalize stage.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    target_label : str
    stage : str
    runs_root : str
    exit_code : int
    log_path : str | None, optional, default ``None``"""
    log_path = log_path or str(logs.target_log_path(runs_root, run_id, target_label, stage))
    exit_code = _effective_exit_code(exit_code, log_path)
    error_tail = logs.read_log_tail(log_path, 20) if exit_code != 0 else ""
    finished_at = pstate._utc_now()
    if exit_code == 0:
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            pstate.STATUS_SUCCESS,
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
            pstate.STATUS_CANCELED,
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
            pstate.STATUS_FAILED,
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
    state: pstate.PipelineState,
    run_id: str,
    ctx,
) -> dict[str, int]:
    """Self-healing reconcile for in-flight stage rows."""
    counts = {"adopted": 0, "completed": 0, "failed": 0, "requeued": 0, "still_running": 0}
    cfg = ctx.cfg
    runs_root = cfg.runs_dir()

    running_jobs = list(state.running_jobs(run_id))
    condor_cluster_ids = [
        int(job.native_id)
        for job in running_jobs
        if (job.executor or cfg.stage_executor(job.stage)) == "condor"
        and job.native_id is not None
    ]
    cluster_status = (
        condor.query_clusters(condor_cluster_ids) if condor_cluster_ids else {}
    )

    for job in running_jobs:
        executor = job.executor or cfg.stage_executor(job.stage)
        native_id = job.native_id
        status_doc = _read_status_file(runs_root, run_id, job.target_label, job.stage)

        if executor == "condor":
            artifacts = condor.condor_artifact_paths(
                runs_root, run_id, job.target_label, job.stage
            )
            log_path = str(
                logs.target_log_path(runs_root, run_id, job.target_label, job.stage)
            )
            if native_id is None:
                adopted_id = condor.read_recorded_cluster_id(
                    runs_root, run_id, job.target_label, job.stage
                )
                if adopted_id is not None:
                    submit_epoch = job.submit_epoch
                    if submit_epoch is None:
                        submit_epoch = _iso_to_epoch(job.started_at)
                    state.set_launch_descriptor(
                        run_id,
                        job.target_label,
                        job.stage,
                        executor="condor",
                        native_id=int(adopted_id),
                        submit_epoch=submit_epoch,
                        log_path=log_path,
                    )
                    native_id = int(adopted_id)
                    counts["adopted"] += 1
                    cluster_status.update(condor.query_clusters([native_id]))
                elif _age_seconds(job.claimed_at) >= _LOCAL_START_GRACE_S:
                    died_reason = "Condor stage claimed but never submitted; requeued"
                    if _requeue_or_fail_stage(
                        state,
                        run_id,
                        job,
                        runs_root=runs_root,
                        reason=died_reason,
                        max_attempts=cfg.max_stage_attempts,
                        requeue_backoff_s=cfg.requeue_backoff_s,
                    ):
                        counts["requeued"] += 1
                    else:
                        counts["failed"] += 1
                else:
                    counts["still_running"] += 1
                if native_id is None:
                    continue
            # Wall-clock submit epoch (DB-persisted) drives the poll grace.
            submit_epoch = job.submit_epoch
            if submit_epoch is None:
                submit_epoch = _iso_to_epoch(job.started_at) or 0.0
            status, raw_exit = cluster_status.get(int(native_id), (None, None))
            hold_path = artifacts["hold"]
            if status is not None:
                stage_liveness.clear_poll_misses(artifacts["poll_misses"])
            exit_code = condor.poll_cluster_status(
                int(native_id),
                status,
                raw_exit,
                submitted_at=submit_epoch,
                hold_timeout_s=cfg.condor_hold_timeout_s,
                hold_path=hold_path,
                bad_machines_path=artifacts["bad_machines"],
            )
            if exit_code is None:
                if status in (condor._JOB_IDLE, condor._JOB_RUNNING):
                    eviction_host = condor.eviction_requeue_host(
                        artifacts["log"],
                        cluster_id=int(native_id),
                        eviction_state_path=artifacts["eviction_state"],
                    )
                    # A confirmed execute -> evict loop is stronger evidence
                    # than ordinary stage-log activity.  The Condor user log
                    # itself is updated by every execute/evict event, so
                    # gating this path on stage_output_recently_active()
                    # defeats host exclusion for exactly the clean repeated
                    # eviction pattern this detector handles.  Keep the
                    # liveness guard below for jobs that disappear from
                    # queue/history without a confirmed eviction pair.
                    if eviction_host:
                        tallies = condor.combined_eviction_tallies(
                            artifacts["log"].read_text(encoding="utf-8", errors="replace"),
                            cluster_id=int(native_id),
                        )
                        failure_count = int(tallies.get(eviction_host, 0))
                        condor.add_bad_machine(artifacts["bad_machines"], eviction_host)
                        condor.record_eviction_requeue(
                            artifacts["eviction_state"],
                            cluster_id=int(native_id),
                            host=eviction_host,
                            failure_count=failure_count,
                        )
                        reason = (
                            f"Condor immediate-evict loop on {eviction_host} "
                            f"({failure_count} failures); excluded host and requeued"
                        )
                        log.warning(
                            "Condor cluster %s evict loop on %s for %s / %s; requeueing",
                            native_id,
                            eviction_host,
                            job.target_label,
                            job.stage,
                        )
                        condor.remove_cluster(int(native_id), hold_path=hold_path)
                        if _requeue_or_fail_stage(
                            state,
                            run_id,
                            job,
                            runs_root=runs_root,
                            reason=reason,
                            # A bad/flaky execute host getting excluded and
                            # requeued is expected recovery, not stage
                            # failure -- attempts shares its counter with
                            # every other launch path, so budgeting this
                            # against the tight generic max_stage_attempts
                            # (default 3) means discovering 1-2 bad hosts in
                            # a small pool can exhaust the whole budget
                            # before ever reaching a good one. Give eviction
                            # requeues their own, much larger allowance;
                            # each retry excludes one more host, so this
                            # naturally bounds itself by the pool size.
                            max_attempts=cfg.max_eviction_stage_attempts,
                            requeue_backoff_s=cfg.requeue_backoff_s,
                            # Host exclusion plus requeue is an expected
                            # recovery path; notify only if retries are
                            # exhausted and the stage actually fails.
                            notify_on_requeue=False,
                        ):
                            counts["requeued"] += 1
                        else:
                            counts["failed"] += 1
                        continue
                counts["still_running"] += 1
                continue
            if status == condor._JOB_HELD:
                # Held past hold_timeout_s: bounded auto-retry for every hold
                # reason, not just the memory-cgroup signature (that one also
                # gets its bad host recorded via bad_machines_path above).
                # Same reasoning as the eviction-host path above: an
                # infra-side hold is expected recovery, not a stage failure.
                reason = (
                    f"Condor cluster {native_id} held past "
                    f"{cfg.condor_hold_timeout_s:.0f}s timeout; requeued"
                )
                if _requeue_or_fail_stage(
                    state,
                    run_id,
                    job,
                    runs_root=runs_root,
                    reason=reason,
                    max_attempts=cfg.max_eviction_stage_attempts,
                    requeue_backoff_s=cfg.requeue_backoff_s,
                    notify_on_requeue=False,
                ):
                    counts["requeued"] += 1
                else:
                    counts["failed"] += 1
                continue
            if exit_code == 1 and status is None:
                if stage_liveness.stage_output_recently_active(log_path, job.stage):
                    stage_liveness.clear_poll_misses(artifacts["poll_misses"])
                    log.info(
                        "Condor cluster %s not visible but %s / %s output is active; "
                        "keeping stage running",
                        native_id,
                        job.target_label,
                        job.stage,
                    )
                    counts["still_running"] += 1
                    continue
                misses = stage_liveness.record_poll_miss(artifacts["poll_misses"])
                if misses < stage_liveness.CONDOR_POLL_MISS_FAIL_THRESHOLD:
                    log.warning(
                        "Condor cluster %s not visible (miss %d/%d); deferring failure "
                        "for %s / %s",
                        native_id,
                        misses,
                        stage_liveness.CONDOR_POLL_MISS_FAIL_THRESHOLD,
                        job.target_label,
                        job.stage,
                    )
                    counts["still_running"] += 1
                    continue
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
            raw_exit = status_doc.get("exit_code")
            if raw_exit is None:
                _pid, alive, _status_pid = _resolve_local_pid(job, status_doc, token_ok=token_ok)
                if alive:
                    counts["still_running"] += 1
                    continue
                if _requeue_local_stage(
                    state,
                    run_id,
                    job,
                    runs_root=runs_root,
                    reason="Local stage exited without exit code; requeued",
                    terminate_if_alive=False,
                    max_attempts=cfg.max_stage_attempts,
                    requeue_backoff_s=cfg.requeue_backoff_s,
                ):
                    counts["requeued"] += 1
                else:
                    counts["failed"] += 1
                continue
            exit_code = int(raw_exit)
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

        pid, alive, status_pid = _resolve_local_pid(job, status_doc, token_ok=token_ok)

        if alive and token_ok:
            # Our process is alive and the token matches: adopt, never relaunch.
            if native_id is None and status_pid is not None:
                log_path = job.log_path or str(
                    logs.target_log_path(runs_root, run_id, job.target_label, job.stage)
                )
                state.set_launch_descriptor(
                    run_id,
                    job.target_label,
                    job.stage,
                    executor="local",
                    native_id=status_pid,
                    submit_epoch=job.submit_epoch,
                    log_path=log_path,
                )
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
        if _requeue_local_stage(
            state,
            run_id,
            job,
            runs_root=runs_root,
            reason="Local stage lost or stale; requeued",
            terminate_if_alive=alive,
            max_attempts=cfg.max_stage_attempts,
            requeue_backoff_s=cfg.requeue_backoff_s,
        ):
            counts["requeued"] += 1
        else:
            counts["failed"] += 1

    return counts


def _pipeline_spec():
    """Pipeline spec."""
    from syndiff_pipeline.pipeline_spec import get_syndiff_pipeline

    return get_syndiff_pipeline()


def _blocking_depth(stage: str, memo: dict[str, int] | None = None) -> int:
    """Blocking depth.
    
    Parameters
    ----------
    stage : str
    memo : dict[str, int] | None, optional, default ``None``
    
    Returns
    -------
    int"""
    memo = memo if memo is not None else {}
    if stage in memo:
        return memo[stage]
    spec = _pipeline_spec().get(stage)
    deps = list(spec.deps) if spec is not None else []
    if not deps:
        memo[stage] = 0
        return 0
    depth = max(_blocking_depth(dep, memo) for dep in deps) + 1
    memo[stage] = depth
    return depth


def _verify_outcome_still_applicable(state: pstate.PipelineState, key: VerifyTaskKey) -> bool:
    """True if a verify result may still be applied to SQLite for *key*."""
    run = state.get_run(key.run_id) or {}
    if run.get("force_rerun") and key.stage in set(state.get_active_stages(key.run_id)):
        # Selected stages are being force-rerun; do not artifact-skip them.
        return False
    row = state.get_stage_run(key.run_id, key.target_label, key.stage)
    if row is None or row.status not in (pstate.STATUS_PENDING, pstate.STATUS_EXTERNAL):
        return False
    if state.external_verify_complete(key.run_id, key.target_label, key.stage):
        return False
    return True


def _verify_worker():
    """Verify worker."""
    from syndiff_pipeline.common.orchestration.verify_worker import try_get_verify_worker

    return try_get_verify_worker()


def _cancel_verify_run(run_id: str) -> None:
    """Cancel verify run.
    
    Parameters
    ----------
    run_id : str"""
    worker = _verify_worker()
    if worker is not None:
        worker.cancel_run(run_id)


def _cancel_verify_keys(keys: list[VerifyTaskKey]) -> None:
    """Cancel verify keys.
    
    Parameters
    ----------
    keys : list[VerifyTaskKey]"""
    worker = _verify_worker()
    if worker is not None:
        worker.cancel_keys(keys)


def _apply_verify_outcome(state: pstate.PipelineState, outcome: VerifyOutcome) -> int:
    """Persist one verify result; return 1 if the stage was skipped."""
    key = outcome.key
    if not _verify_outcome_still_applicable(state, key):
        return 0
    if outcome.error:
        log.warning(
            "Verify error for %s/%s/%s: %s",
            key.run_id,
            key.target_label,
            key.stage,
            outcome.error,
        )
        return 0
    if outcome.complete:
        state.mark_skipped(key.run_id, key.target_label, key.stage)
        state.cache_skip_reason(
            key.run_id, key.target_label, key.stage, pstate.SKIP_REASON_ARTIFACTS
        )
        state.cache_external_check(
            key.run_id,
            key.target_label,
            key.stage,
            complete=True,
        )
        return 1
    state.cache_external_check(key.run_id, key.target_label, key.stage, complete=False)
    # An ``external`` stage is an upstream dependency that this run will not
    # produce.  Once its verification completes with ``False``, no scheduler
    # action can make it appear; repeatedly requeueing the same NFS scan leaves
    # the dependent stage permanently displayed as ``sc_q``.  Fail visibly so
    # the operator gets the missing prerequisite and can retry after repairing
    # it.  Verification exceptions remain non-terminal above: they may be
    # transient filesystem/DB failures rather than a missing artifact.
    row = state.get_stage_run(key.run_id, key.target_label, key.stage)
    if row is not None and row.status == pstate.STATUS_EXTERNAL:
        reason = (
            f"Required external artifact for stage {key.stage!r} was not found; "
            "the selected run cannot produce this prerequisite."
        )
        log.error("%s (%s/%s)", reason, key.run_id, key.target_label)
        state.update_stage_status(
            key.run_id,
            key.target_label,
            key.stage,
            pstate.STATUS_FAILED,
            finished_at=pstate._utc_now(),
            error_tail=reason,
        )
        state.block_downstream(key.run_id, key.target_label, key.stage)
    return 0


# Template stages with checkpoint-first verify short-circuit (plan §11).
# Stage name -> expected-fingerprint helper in provenance_checkpoint.
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    CHECKPOINT_STAGE_FINGERPRINTS,
    CHECKPOINT_STAGES,
)

# Backward-compatible alias for tests and external references.
_CHECKPOINT_STAGE_FINGERPRINTS = CHECKPOINT_STAGE_FINGERPRINTS

_TRUST_INDEX_MISS_REMEDIATION = (
    "run 'syndiff bookkeeping reindex', re-run the stage, or set "
    "bookkeeping.trust_index to false"
)


def _log_trust_index_checkpoint_miss(
    stage: str,
    run_id: str,
    target_label: str,
    *,
    miss_reason: str | None,
) -> None:
    """Emit BK-8 audit WARNING when trust_index fail-closes on a checkpoint miss."""
    if miss_reason == "store_unavailable":
        log.warning(
            "Provenance store unavailable for checkpoint verify on stage %s "
            "(%s/%s) with bookkeeping.trust_index; fail-closed (no legacy scan). "
            "Remediation: fix provenance.db access, or %s",
            stage,
            run_id,
            target_label,
            _TRUST_INDEX_MISS_REMEDIATION,
        )
    else:
        log.warning(
            "Checkpoint index miss for stage %s (%s/%s) with "
            "bookkeeping.trust_index; fail-closed (no legacy scan). "
            "Remediation: %s",
            stage,
            run_id,
            target_label,
            _TRUST_INDEX_MISS_REMEDIATION,
        )


def _checkpoint_hit(
    stage: str,
    key: "VerifyTaskKey",
    resolved,
    stable_path: str,
) -> tuple["VerifyOutcome | None", str | None]:
    """Checkpoint-first fast path for template stages (plan §11).

    Recomputes the stage checkpoint fingerprint fresh from *resolved* (a pure
    function, no filesystem access -- config drift naturally yields a
    different fingerprint and therefore a miss) and checks it against the
    provenance store with one indexed query. On a HIT, returns a
    legacy-shaped "ok" :class:`VerifyOutcome` so the caller can route it
    through the existing ``_apply_verify_outcome`` exactly as a scan success
    would. On ANY miss condition -- fingerprint not indexed, DB/store
    unavailable, provenance package absent, or any other exception -- returns
    ``None`` so the caller falls open, unchanged, to the legacy
    ``check_manifests_only`` / ``stage_absence_probe`` / background
    ``VerifyTask`` path. Never raises.
    """
    from syndiff_pipeline.common.orchestration.verify_worker import VerifyOutcome

    fingerprint_fn_name = CHECKPOINT_STAGE_FINGERPRINTS.get(stage)
    if fingerprint_fn_name is None:
        return None, None

    try:
        from syndiff_pipeline.common.provenance.store import ProvenanceStore
        from syndiff_pipeline.common.scc_paths import provenance_db_path
        from syndiff_pipeline.template_creation.orchestration import (
            provenance_checkpoint,
        )

        expected_fp_fn = getattr(provenance_checkpoint, fingerprint_fn_name)
        expected_fp = expected_fp_fn(resolved)
    except Exception:
        log.debug(
            "%s checkpoint check unavailable for %s/%s (falling open to legacy"
            " verify)",
            stage,
            key.run_id,
            key.target_label,
            exc_info=True,
        )
        return None, None

    try:
        store = ProvenanceStore(
            str(provenance_db_path(resolved.data_root)), read_only=True
        )
    except Exception:
        log.debug(
            "%s provenance store open failed for %s/%s",
            stage,
            key.run_id,
            key.target_label,
            exc_info=True,
        )
        return None, "store_unavailable"

    try:
        if not store.scc_stage_complete([expected_fp]):
            return None, "not_indexed"
    except Exception:
        log.debug(
            "%s checkpoint index query failed for %s/%s",
            stage,
            key.run_id,
            key.target_label,
            exc_info=True,
        )
        return None, "store_unavailable"

    return (
        VerifyOutcome(
            key=key,
            complete=True,
            stable_path=stable_path,
            resolved=resolved,
        ),
        None,
    )


@dataclass(frozen=True)
class _FastPathResult:
    """Outcome of one candidate's checkpoint/manifest/absence-probe fast path.

    Computed off the main thread by ``_run_fast_path_task`` (see
    ``_fast_path_check_bounded``) since each of the three checks it wraps
    reads from the shared NFS data root (provenance.db, manifests, skycell
    CSVs) and can block indefinitely under NFS contention. Exactly one of
    ``outcome`` / ``needs_full_verify`` is meaningful: ``outcome`` set means
    the candidate is resolved for this tick; ``needs_full_verify`` True means
    the caller should fall back to the background ``VerifyTask`` pool.
    """

    outcome: "VerifyOutcome | None" = None
    needs_full_verify: bool = False
    backfill: "BackfillTask | None" = None


def _run_fast_path_task(
    key: "VerifyTaskKey",
    resolved,
    manifest_path: str,
    stable_path: str,
    runner_cfg,
    meta,
    bookkeeping_trust_index: bool,
) -> "_FastPathResult":
    """Body of the checkpoint/manifest/absence-probe fast path (worker thread).

    Mirrors the per-candidate logic that used to run inline in
    ``_run_verify_pass``; moved here so it can be submitted to
    ``_fast_path_executor`` instead of blocking the daemon's main thread.
    """
    from syndiff_pipeline.common.orchestration.verify_worker import (
        BackfillTask,
        VerifyOutcome,
    )
    from syndiff_pipeline.template_creation.orchestration.verify import (
        AbsenceProbeResult,
        check_manifests_only,
        stage_absence_probe,
    )

    stage = key.stage
    checkpoint_result = _checkpoint_hit(stage, key, resolved, stable_path)
    checkpoint_outcome, miss_reason = (
        checkpoint_result if checkpoint_result is not None else (None, None)
    )
    if checkpoint_outcome is not None:
        return _FastPathResult(outcome=checkpoint_outcome)
    if bookkeeping_trust_index and stage in CHECKPOINT_STAGES:
        _log_trust_index_checkpoint_miss(
            stage, key.run_id, key.target_label, miss_reason=miss_reason
        )
        if miss_reason != "store_unavailable":
            return _FastPathResult(
                outcome=VerifyOutcome(
                    key=key, complete=False, stable_path=stable_path, resolved=resolved
                )
            )
        # store_unavailable: fall through to the legacy manifest/absence-probe
        # path below (see the identical comment previously inline here).

    manifest_hit = check_manifests_only(
        resolved,
        stage,
        manifest_path=manifest_path,
        stable_manifest_path=stable_path,
        runner_cfg=runner_cfg,
        meta=meta,
    )
    if manifest_hit is True:
        backfill = None
        if (
            check_manifests_only(
                resolved,
                stage,
                stable_manifest_path=stable_path,
                runner_cfg=runner_cfg,
                meta=meta,
            )
            is not True
        ):
            backfill = BackfillTask(manifest_path=manifest_path, stable_path=stable_path)
        return _FastPathResult(
            outcome=VerifyOutcome(
                key=key, complete=True, stable_path=stable_path, resolved=resolved
            ),
            backfill=backfill,
        )

    probe = stage_absence_probe(resolved, stage, runner_cfg=runner_cfg, meta=meta)
    if probe is AbsenceProbeResult.ABSENT:
        return _FastPathResult(
            outcome=VerifyOutcome(
                key=key, complete=False, stable_path=stable_path, resolved=resolved
            )
        )
    return _FastPathResult(needs_full_verify=True)


_FAST_PATH_MAX_WORKERS = 4
_FAST_PATH_SAME_TICK_TIMEOUT_S = 0.5
_FAST_PATH_BLOCK_POLL_S = 0.05
_fast_path_executor: "concurrent.futures.ThreadPoolExecutor | None" = None
_fast_path_lock = threading.Lock()
_fast_path_in_flight: dict = {}


def _get_fast_path_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _fast_path_executor
    with _fast_path_lock:
        if _fast_path_executor is None:
            _fast_path_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_FAST_PATH_MAX_WORKERS,
                thread_name_prefix="verify-fast-path",
            )
        return _fast_path_executor


def _fast_path_check_bounded(
    key: "VerifyTaskKey",
    resolved,
    manifest_path: str,
    stable_path: str,
    runner_cfg,
    meta,
    bookkeeping_trust_index: bool,
    *,
    timeout_s: float = _FAST_PATH_SAME_TICK_TIMEOUT_S,
) -> "_FastPathResult | None":
    """Run the checkpoint/manifest/absence-probe fast path off the main thread.

    Those checks read the provenance DB, manifests, and skycell CSVs from the
    shared NFS data root; any one of them stalling under NFS contention
    (D-state, uninterruptible -- see the nfs-contention-plscience-cluster
    postmortem) previously froze the daemon's single main scheduling thread
    entirely, blocking every run on every host, not just the one candidate
    that hit the stall. Submitting to a small dedicated pool keeps the common
    (fast) case unchanged -- the result is almost always ready well within
    *timeout_s* -- while a genuine stall only ties up one of a few worker
    slots instead of the whole daemon.

    Returns ``None`` if the check hasn't finished yet: the first time a given
    *key* is seen this waits up to *timeout_s*, but a *key* already in flight
    (e.g. re-encountered on a later tick, or a later iteration of the same
    verify pass) is polled with a zero timeout so a stuck candidate never
    costs more than one bounded wait. Callers must treat ``None`` exactly
    like "not resolved yet" -- leave state untouched and retry later.
    """
    executor = _get_fast_path_executor()
    with _fast_path_lock:
        fut = _fast_path_in_flight.get(key)
        freshly_submitted = fut is None
        if freshly_submitted:
            fut = executor.submit(
                _run_fast_path_task,
                key,
                resolved,
                manifest_path,
                stable_path,
                runner_cfg,
                meta,
                bookkeeping_trust_index,
            )
            _fast_path_in_flight[key] = fut
    try:
        result = fut.result(timeout=timeout_s if freshly_submitted else 0.0)
    except concurrent.futures.TimeoutError:
        return None
    with _fast_path_lock:
        _fast_path_in_flight.pop(key, None)
    return result


def _fast_path_in_flight_count(run_id: str) -> int:
    """Count fast-path candidates still in flight for *run_id*.

    Needed so blocking callers (``block=True``) don't return prematurely:
    the outer loop's "anything left to wait for?" check previously only
    looked at the slow ``ArtifactVerifyWorker`` pool, so a run whose
    candidates were all still pending in the fast-path pool (e.g. the very
    first tick, before any of them have had a chance to resolve) would be
    reported as fully drained when it wasn't.
    """
    with _fast_path_lock:
        return sum(1 for key in _fast_path_in_flight if key.run_id == run_id)


def reset_fast_path_pool_for_tests() -> None:
    """Tear down the fast-path executor/dedup state between unit tests."""
    global _fast_path_executor
    with _fast_path_lock:
        executor, _fast_path_executor = _fast_path_executor, None
        _fast_path_in_flight.clear()
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def _iter_verify_candidates(
    state: pstate.PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
) -> list[tuple]:
    """Collect uncached pending/external stages eligible for verification."""
    from syndiff_pipeline.common.orchestration.verify_worker import VerifyTaskKey

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
        from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

        resolved = resolve_config(target, cfg)
        for row in rows:
            if row.status not in (pstate.STATUS_PENDING, pstate.STATUS_EXTERNAL):
                continue
            if force_rerun and row.stage in active_stages:
                continue
            if row.stage == "ps1_download":
                if resolved.stages.ps1_process.ps1_source == "stream":
                    continue
                if state.get_skip_reason(run_id, label, "ps1_download") == pstate.SKIP_REASON_STREAM:
                    continue
            if (
                state.get_skip_reason(run_id, label, row.stage)
                in (pstate.SKIP_REASON_NOT_SELECTED, pstate.SKIP_REASON_SUPERSEDED)
            ):
                continue
            if row.status == pstate.STATUS_EXTERNAL:
                if not pstate.artifact_verify_needed(
                    state,
                    run_id,
                    label,
                    row.stage,
                    list(active_stages),
                    spec=state.pipeline_spec,
                ):
                    continue
            elif row.status == pstate.STATUS_PENDING:
                if row.stage in active_stages:
                    continue
                elif (
                    row.stage
                    in state.pipeline_spec.artifact_verify_closure(active_stages)
                    and pstate.artifact_verify_needed(
                        state,
                        run_id,
                        label,
                        row.stage,
                        list(active_stages),
                        spec=state.pipeline_spec,
                    )
                ):
                    pass
                else:
                    continue
            else:
                continue
            if row.status == pstate.STATUS_PENDING:
                if state.external_verify_attempted(run_id, label, row.stage):
                    continue
            elif state.external_verify_complete(run_id, label, row.stage):
                continue
            if row.status == pstate.STATUS_PENDING and not state.deps_satisfied(
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
        """Sort key.
        
        Parameters
        ----------
        item : tuple
        
        Returns
        -------
        tuple"""
        key, status, _resolved, _mp, _sp = item
        return (
            0 if status == pstate.STATUS_PENDING else 1,
            -_blocking_depth(key.stage),
            key.target_label,
            key.stage,
        )

    candidates.sort(key=_sort_key)
    return candidates


def _verify_backlog(
    state: pstate.PipelineState,
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


def collect_verify_status_for_run(
    state: pstate.PipelineState,
    run_id: str,
) -> dict | None:
    """Live artifact-scan observability for one run (daemon worker must be active)."""
    run = state.get_run(run_id)
    if not run:
        return None
    return _collect_verify_status_by_run(state, [run]).get(run_id)


def _collect_verify_status_by_run(
    state: pstate.PipelineState,
    runs: list[dict],
) -> dict[str, dict]:
    """Build per-run verify observability payload for host-local JSON."""
    worker = _verify_worker()
    by_run: dict[str, dict] = {}
    for run in runs:
        run_id = run["run_id"]
        try:
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
        except Exception:
            log.exception("Failed to collect verify status for run %s", run_id)
            by_run[run_id] = {"scan_running": 0, "scan_queued": 0, "active": []}
    return by_run


def _promote_warmed_external_checkpoints(
    state: pstate.PipelineState,
    run_id: str,
    ctx,
) -> int:
    """Promote EXTERNAL stages whose provenance checkpoint warmed since a fail-closed miss.

    Note: this still calls ``_checkpoint_hit`` synchronously on the main
    thread (unlike ``_run_verify_pass``, which routes it through
    ``_fast_path_check_bounded``). It only opens provenance.db, not the
    CSV/manifest reads that produced the observed daemon-wide freeze, so the
    risk is smaller, but it is not zero on a fully NFS-stalled data root.
    """
    if not ctx.cfg.bookkeeping_trust_index:
        return 0
    from syndiff_pipeline.common.orchestration.verify_worker import VerifyTaskKey
    from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

    promoted = 0
    active_stages = list(state.get_active_stages(run_id))
    cfg = ctx.cfg
    runs_root = cfg.runs_dir()
    targets_by_label = {t.label(): t for t in ctx.targets}
    for row in state.list_stage_runs(run_id):
        if row.status != pstate.STATUS_EXTERNAL:
            continue
        if row.stage not in CHECKPOINT_STAGES:
            continue
        if state.external_verify_complete(run_id, row.target_label, row.stage):
            continue
        if not state.external_verify_attempted(run_id, row.target_label, row.stage):
            continue
        if not pstate.artifact_verify_needed(
            state,
            run_id,
            row.target_label,
            row.stage,
            active_stages,
            spec=state.pipeline_spec,
        ):
            continue
        target = targets_by_label.get(row.target_label)
        if target is None:
            continue
        resolved = resolve_config(target, cfg)
        stable_path = str(
            logs.stable_stage_manifest_path(runs_root, row.target_label, row.stage)
        )
        key = VerifyTaskKey(run_id, row.target_label, row.stage)
        checkpoint_result = _checkpoint_hit(row.stage, key, resolved, stable_path)
        if checkpoint_result is None:
            continue
        outcome, _miss_reason = checkpoint_result
        if outcome is None or not outcome.complete:
            continue
        promoted += _apply_verify_outcome(state, outcome)
    return promoted


def _run_verify_pass(
    state: pstate.PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
    budget: int,
    block: bool,
    block_timeout_s: float = 0.0,
) -> int:
    """Manifest fast path off the main thread (bounded wait); full verify in background pool."""
    from syndiff_pipeline.common.orchestration.verify_worker import (
        BackfillTask,
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
            fast_path = _fast_path_check_bounded(
                key,
                resolved,
                manifest_path,
                stable_path,
                ctx.cfg,
                ctx.meta,
                ctx.cfg.bookkeeping_trust_index,
            )
            if fast_path is None:
                # Still running (or just stalled) in the fast-path pool; leave
                # state untouched and pick it up again on a later tick rather
                # than blocking the whole daemon on it.
                continue
            if fast_path.outcome is not None:
                budget_left -= 1
                total += apply(fast_path.outcome)
                if fast_path.backfill is not None:
                    backfills.append(fast_path.backfill)
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
                    runner_cfg=ctx.cfg,
                    meta=ctx.meta,
                )
            )

        if not tasks and not backfills:
            slow_in_flight = worker.in_flight_count(run_id)
            fast_in_flight = _fast_path_in_flight_count(run_id)
            if slow_in_flight == 0 and fast_in_flight == 0:
                break
            if not block:
                break
            if slow_in_flight > 0:
                total += worker.drain(
                    apply,
                    run_id=run_id,
                    block=True,
                    block_timeout_s=block_timeout_s,
                )
            else:
                # Only fast-path work is outstanding. An already-in-flight
                # fast-path candidate polls with a zero timeout (see
                # _fast_path_check_bounded), so a blocking caller would
                # busy-spin here without a short sleep between retries.
                time.sleep(_FAST_PATH_BLOCK_POLL_S)
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
    state: pstate.PipelineState,
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
    warmed = _promote_warmed_external_checkpoints(state, run_id, ctx)
    if warmed:
        log.info(
            "Promoted %d warmed external checkpoint stage(s) in run %s",
            warmed,
            run_id,
        )
    return _run_verify_pass(
        state,
        run_id,
        ctx,
        force_rerun=force_rerun,
        budget=budget,
        block=block,
    )


def _schedule_external_and_pending_skips(
    state: pstate.PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
    budget: int | None = None,
) -> None:
    """Non-blocking verify scheduling for the supervisor main loop."""
    if budget is None:
        budget = ctx.cfg.verify_budget_per_tick
    warmed = _promote_warmed_external_checkpoints(state, run_id, ctx)
    if warmed:
        log.info(
            "Promoted %d warmed external checkpoint stage(s) in run %s",
            warmed,
            run_id,
        )
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
    """Cancel verify for retry.
    
    Parameters
    ----------
    run_id : str
    target_label : str
    stage : str
    reset_downstream : bool"""
    from syndiff_pipeline.common.orchestration.verify_worker import VerifyTaskKey

    stages = [stage] + (
        _pipeline_spec().downstream_stages(stage) if reset_downstream else []
    )
    keys = [VerifyTaskKey(run_id, target_label, s) for s in stages]
    _cancel_verify_keys(keys)


def _global_pool_running(state: pstate.PipelineState) -> dict[str, int]:
    """Running stage count per pool across ALL runs (global capacity)."""
    pool_running: dict[str, int] = defaultdict(int)
    for job in state.running_stage_runs(None):
        stage_spec = _pipeline_spec().get(job.stage)
        if stage_spec is not None and stage_spec.pool:
            pool_running[stage_spec.pool] += 1
    return pool_running


def _pool_capacity(pool_running: dict[str, int], pool_name: str, pool_cfg) -> int:
    """Pool capacity.
    
    Parameters
    ----------
    pool_running : dict[str, int]
    pool_name : str
    pool_cfg
    
    Returns
    -------
    int"""
    return max(0, pool_cfg.max_concurrent - pool_running.get(pool_name, 0))


def _stall_reasons(state: pstate.PipelineState, run_id: str, ctx) -> List[str]:
    """Stall reasons.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    ctx
    
    Returns
    -------
    List[str]"""
    reasons: List[str] = []
    cfg = ctx.cfg
    pool_running = _global_pool_running(state)

    for row in state.list_stage_runs(run_id):
        if row.status not in (pstate.STATUS_PENDING, pstate.STATUS_READY, pstate.STATUS_BLOCKED, pstate.STATUS_EXTERNAL):
            continue
        label = row.target_label
        stage = row.stage
        if row.status == pstate.STATUS_BLOCKED:
            reasons.append(f"{label}/{stage}: blocked by upstream failure")
            continue
        target = next((t for t in ctx.targets if t.label() == label), None)
        from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

        stages = resolve_config(target, cfg).stages if target is not None else None
        if not state.deps_satisfied(run_id, label, stage, stages=stages):
            missing = []
            for dep in _pipeline_spec().effective_stage_deps(stage, stages):
                dep_row = state.get_stage_run(run_id, label, dep)
                if dep_row is None or dep_row.status not in (pstate.STATUS_SUCCESS, pstate.STATUS_SKIPPED):
                    missing.append(f"{dep}={dep_row.status if dep_row else 'missing'}")
            reasons.append(f"{label}/{stage}: waiting on {', '.join(missing)}")
            continue
        if row.status in (pstate.STATUS_PENDING, pstate.STATUS_EXTERNAL):
            active_stages = state.get_active_stages(run_id)
            needs_verify = pstate.stage_needs_artifact_verify_display(
                state, run_id, label, stage, row.status, active_stages
            )
            if needs_verify and not state.external_verify_complete(
                run_id, label, stage
            ):
                reasons.append(f"{label}/{stage}: artifact verify queued")
            elif row.status == pstate.STATUS_PENDING:
                reasons.append(f"{label}/{stage}: pending promotion")
            else:
                reasons.append(f"{label}/{stage}: artifact verify queued")
            continue
        if row.status == pstate.STATUS_READY:
            stage_spec = _pipeline_spec().get(stage)
            if stage_spec is not None and stage_spec.pool:
                pool = stage_spec.pool
                cap = cfg.resources.get(pool)
                if cap and pool_running[pool] >= cap.max_concurrent:
                    reasons.append(f"{label}/{stage}: pool {pool} saturated")
                else:
                    reasons.append(f"{label}/{stage}: ready but not claimed")
            else:
                reasons.append(f"{label}/{stage}: ready but not claimed")
    return reasons


def _try_launch_ready_row(
    state: pstate.PipelineState,
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

    from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

    cfg = ctx.cfg
    resolved = resolve_config(target, cfg)
    target_stages = resolved.stages
    if not state.deps_satisfied(run_id, row.target_label, row.stage, stages=target_stages):
        state.update_stage_status(run_id, row.target_label, row.stage, pstate.STATUS_PENDING)
        return False

    resources_override = None
    if row.stage == "ps1_process" and not force_rerun:
        from syndiff_pipeline.template_creation.orchestration import ps1_process_preflight

        plan = ps1_process_preflight.plan_ps1_process_launch(
            data_root=resolved.data_root,
            sector=target.sector,
            camera=target.camera,
            ccd=target.ccd,
            oversampling_factor=getattr(target_stages.mapping, "oversampling_factor", 1),
            params=target_stages.ps1_process,
            mapping_store_name=getattr(target_stages.mapping, "store_name", None),
        )
        if plan.decision == ps1_process_preflight.DECISION_SKIP:
            state.mark_skipped(run_id, row.target_label, row.stage)
            state.cache_skip_reason(run_id, row.target_label, row.stage, pstate.SKIP_REASON_ARTIFACTS)
            log.info(
                "ps1_process preflight skip for %s: %s", row.target_label, plan.reason
            )
            return True
        if plan.decision == ps1_process_preflight.DECISION_SMALL:
            resources_override = plan.resources
            log.info(
                "ps1_process preflight small job for %s: %s",
                row.target_label,
                plan.reason,
            )

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

    from syndiff_pipeline.template_creation.orchestration import dispatch

    cmd = dispatch.build_stage_command(
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
            resources_override=resources_override,
        )
    except Exception:
        log.exception("Launch failed for %s / %s; requeuing", row.target_label, row.stage)
        _requeue_or_fail_after_launch_failure(
            state,
            run_id,
            row.target_label,
            row.stage,
            cfg=cfg,
            runs_root=runs_root,
            reason="Launch failed",
        )
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


def _launch_force_overrides(
    state: pstate.PipelineState,
    run_id: str,
    ctx,
    *,
    force_rerun: bool,
    active_stages: list[str],
    targets_by_label: dict,
    runs_root: str,
) -> int:
    """Launch ready stages flagged force_launch, bypassing pool max_concurrent."""
    launched = 0
    launch_kwargs = dict(
        force_rerun=force_rerun,
        active_stages=active_stages,
        targets_by_label=targets_by_label,
        runs_root=runs_root,
    )
    for row in state.fetch_force_launch_ready(run_id):
        if row.status in pstate.TERMINAL_STATUSES:
            state.clear_force_launch(run_id, row.target_label, row.stage)
            continue
        if _try_launch_ready_row(
            state,
            run_id,
            ctx,
            row,
            pool_label="force_launch",
            **launch_kwargs,
        ):
            state.clear_force_launch(run_id, row.target_label, row.stage)
            launched += 1
            log.info(
                "Force-launched %s / %s (pool capacity bypassed)",
                row.target_label,
                row.stage,
            )
    return launched


def _tick_run(state: pstate.PipelineState, run_id: str, ctx) -> None:
    """Tick run.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str
    ctx"""
    run = state.get_run(run_id) or {}
    force_rerun = bool(run.get("force_rerun"))
    from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

    targets_by_label = {t.label(): t for t in ctx.targets}
    runs_root = ctx.cfg.runs_dir()
    active_stages = state.get_active_stages(run_id)
    force_kwargs = dict(
        force_rerun=force_rerun,
        active_stages=active_stages,
        targets_by_label=targets_by_label,
        runs_root=runs_root,
    )
    _launch_force_overrides(state, run_id, ctx, **force_kwargs)

    if state.is_paused(run_id):
        return

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
    if ctx.cfg.skip_artifact_verify:
        trusted = state.trust_external_artifacts(run_id, ctx.targets, active_stages)
        if trusted:
            log.info(
                "Trusted %d external stage(s) for run %s (skip_artifact_verify)",
                trusted,
                run_id,
            )
    else:
        _schedule_external_and_pending_skips(
            state,
            run_id,
            ctx,
            force_rerun=force_rerun,
            budget=ctx.cfg.verify_budget_per_tick,
        )
    target_stages_map = {
        label: resolve_config(target, ctx.cfg).stages
        for label, target in targets_by_label.items()
    }
    state.promote_stages(run_id, target_stages_map)
    _launch_force_overrides(state, run_id, ctx, **force_kwargs)

    cfg = ctx.cfg
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
    running = counts.get(pstate.STATUS_RUNNING, 0)
    launchable = sum(
        1
        for row in state.list_stage_runs(run_id)
        if row.status == pstate.STATUS_READY
        and state.deps_satisfied(
            run_id,
            row.target_label,
            row.stage,
            stages=target_stages_map.get(row.target_label),
        )
    )
    nonterminal = sum(counts.get(s, 0) for s in NONTERMINAL_STATUSES)

    prev_status = run.get("status")
    from syndiff_pipeline.common.orchestration.notifications import notifier_for_context

    notifier = notifier_for_context(state, ctx)

    if nonterminal == 0:
        final = pstate.derive_run_final_status(counts)
        prev_terminal = prev_status in pstate.TERMINAL_RUN_STATUSES
        state.set_run_status(run_id, final)
        if not prev_terminal:
            log.info("Run %s complete: %s", run_id, final)
            # Canceled runs already received notify_run_canceled when the intent
            # was applied; do not also emit run_completed(success).
            if notifier is not None and final != pstate.RUN_CANCELED:
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


def _requeue_or_fail_after_launch_failure(
    state: pstate.PipelineState,
    run_id: str,
    target_label: str,
    stage: str,
    *,
    cfg,
    runs_root: str,
    reason: str,
) -> None:
    """Requeue after launch failure, or finalize as failed after max attempts."""
    row = state.get_stage_run(run_id, target_label, stage)
    attempts = (row.attempts or 0) if row else 0
    if attempts >= cfg.max_stage_attempts:
        fail_reason = f"{reason} (gave up after {attempts} attempts)"
        finished_at = pstate._utc_now()
        log.warning(
            "Giving up on %s / %s after %d attempts: %s",
            target_label,
            stage,
            attempts,
            reason,
        )
        state.update_stage_status(
            run_id,
            target_label,
            stage,
            pstate.STATUS_FAILED,
            finished_at=finished_at,
            error_tail=fail_reason,
        )
        state.block_downstream(run_id, target_label, stage)
        state.clear_launch_fields(run_id, target_label, stage)
        _notify_stage_outcome(
            state,
            run_id,
            target_label=target_label,
            stage=stage,
            outcome="failed",
            runs_root=runs_root,
            finished_at=finished_at,
            error_tail=fail_reason,
        )
        return

    backoff_s = cfg.requeue_backoff_s * attempts
    log.info(
        "Requeued %s / %s after launch failure: %s (backoff %.1fs)",
        target_label,
        stage,
        reason,
        backoff_s,
    )
    state.requeue_to_ready(
        run_id,
        target_label,
        stage,
        error_tail=reason,
        backoff_s=backoff_s,
    )


def _requeue_or_fail_stage(
    state: pstate.PipelineState,
    run_id: str,
    job,
    *,
    runs_root: str,
    reason: str,
    max_attempts: int,
    requeue_backoff_s: float,
    terminate_if_alive: bool = False,
    notify_on_requeue: bool = True,
) -> bool:
    """Requeue a lost running stage, or finalize as failed after max attempts."""
    attempts = job.attempts or 0
    if attempts >= max_attempts:
        fail_reason = f"{reason} (gave up after {attempts} attempts)"
        finished_at = pstate._utc_now()
        log.warning(
            "Giving up on %s / %s after %d attempts: %s",
            job.target_label,
            job.stage,
            attempts,
            reason,
        )
        state.update_stage_status(
            run_id,
            job.target_label,
            job.stage,
            pstate.STATUS_FAILED,
            finished_at=finished_at,
            error_tail=fail_reason,
        )
        state.block_downstream(run_id, job.target_label, job.stage)
        state.clear_launch_fields(run_id, job.target_label, job.stage)
        _notify_stage_outcome(
            state,
            run_id,
            target_label=job.target_label,
            stage=job.stage,
            outcome="failed",
            runs_root=runs_root,
            finished_at=finished_at,
            error_tail=fail_reason,
        )
        return False

    if terminate_if_alive and job.native_id is not None:
        _terminate_job(job)
    backoff_s = requeue_backoff_s * attempts
    log.info("Requeued %s / %s: %s (backoff %.1fs)", job.target_label, job.stage, reason, backoff_s)
    state.requeue_running_stage(
        run_id,
        job.target_label,
        job.stage,
        error_tail=reason,
        backoff_s=backoff_s,
    )
    if notify_on_requeue:
        _notify_stage_outcome(
            state,
            run_id,
            target_label=job.target_label,
            stage=job.stage,
            outcome="died",
            runs_root=runs_root,
            finished_at=pstate._utc_now(),
            error_tail=reason,
        )
    return True


def _requeue_local_stage(
    state: pstate.PipelineState,
    run_id: str,
    job,
    *,
    runs_root: str,
    reason: str,
    terminate_if_alive: bool,
    max_attempts: int,
    requeue_backoff_s: float,
) -> bool:
    """Requeue a local running stage, terminating a live worker first."""
    return _requeue_or_fail_stage(
        state,
        run_id,
        job,
        runs_root=runs_root,
        reason=reason,
        max_attempts=max_attempts,
        requeue_backoff_s=requeue_backoff_s,
        terminate_if_alive=terminate_if_alive,
    )


def _terminate_job(job) -> None:
    """Terminate a single running stage's worker (condor cluster or local pid)."""
    if job.native_id is None:
        return
    if job.executor == "condor":
        condor.remove_cluster(int(job.native_id))
    else:
        daemon.terminate_process_tree(int(job.native_id))


def _terminate_run_jobs(state: pstate.PipelineState, run_id: str) -> None:
    """Terminate run jobs.
    
    Parameters
    ----------
    state : pstate.PipelineState
    run_id : str"""
    for job in state.running_stage_runs(run_id):
        _terminate_job(job)


def _apply_commands(state: pstate.PipelineState) -> None:
    """Apply commands.
    
    Parameters
    ----------
    state : pstate.PipelineState"""
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
                    try:
                        stage = _pipeline_spec().resolve_stage_name(args["stage"])
                    except ValueError as exc:
                        log.warning(
                            "Retry command id=%s ignored: %s",
                            cmd.id,
                            exc,
                        )
                        continue
                    target_label = args["target_label"]
                    # If the targeted stage is still running, stop the worker
                    # first to avoid a duplicate when it is relaunched.
                    row = state.get_stage_run(cmd.run_id, target_label, stage)
                    if row is None:
                        log.warning(
                            "Retry command id=%s ignored: no stage row for %s / %s "
                            "in run %s",
                            cmd.id,
                            target_label,
                            stage,
                            cmd.run_id,
                        )
                        continue
                    if row.status == pstate.STATUS_RUNNING:
                        _terminate_job(row)
                    reset_downstream = bool(args.get("reset_downstream", True))
                    state.apply_retry_stage(
                        cmd.run_id,
                        target_label,
                        stage,
                        reset_downstream=reset_downstream,
                    )
                    # A retry is an explicit signal that whatever caused past
                    # evictions (often oversized request_cpus/request_memory)
                    # may now be fixed -- don't let a stale host exclusion
                    # outlive the config change that caused it.
                    run_row = state.get_run(cmd.run_id)
                    if run_row:
                        artifacts = condor.condor_artifact_paths(
                            run_row["runs_root"],
                            cmd.run_id,
                            target_label,
                            stage,
                            mkdir=False,
                        )
                        condor.write_bad_machines(artifacts["bad_machines"], set())
                    _cancel_verify_for_retry(
                        cmd.run_id,
                        target_label,
                        stage,
                        reset_downstream=reset_downstream,
                    )
                    _notify_run_retried(
                        state,
                        cmd.run_id,
                        target_label=target_label,
                        stage=stage,
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
                        if row and row.status == pstate.STATUS_RUNNING:
                            _terminate_job(row)
                state.apply_force_rerun(cmd.run_id, labels, stages_arg)
                _cancel_verify_run(cmd.run_id)
            elif cmd.kind == "force_launch" and cmd.run_id:
                target_label = args.get("target_label")
                stage_raw = args.get("stage")
                if not target_label or not stage_raw:
                    log.warning(
                        "Incomplete force_launch command id=%s (need target_label + stage)",
                        cmd.id,
                    )
                else:
                    try:
                        stage = _pipeline_spec().resolve_stage_name(stage_raw)
                    except ValueError as exc:
                        log.warning(
                            "force_launch command id=%s ignored: %s",
                            cmd.id,
                            exc,
                        )
                        continue
                    row = state.get_stage_run(cmd.run_id, target_label, stage)
                    if row is None:
                        log.warning(
                            "force_launch: no stage row for %s / %s in run %s",
                            target_label,
                            stage,
                            cmd.run_id,
                        )
                    elif row.status == pstate.STATUS_RUNNING:
                        log.info(
                            "force_launch: %s / %s already running in run %s",
                            target_label,
                            stage,
                            cmd.run_id,
                        )
                    elif row.status in pstate.TERMINAL_STATUSES:
                        log.warning(
                            "force_launch: %s / %s is terminal (%s) in run %s",
                            target_label,
                            stage,
                            row.status,
                            cmd.run_id,
                        )
                    else:
                        state.set_force_launch(cmd.run_id, target_label, stage, enabled=True)
                        log.info(
                            "force_launch queued for %s / %s in run %s (status=%s)",
                            target_label,
                            stage,
                            cmd.run_id,
                            row.status,
                        )
            else:
                log.warning("Unknown or incomplete command id=%s kind=%s", cmd.id, cmd.kind)
        finally:
            state.mark_command_processed(cmd.id)


def _maybe_drain_provenance_spool(data_roots) -> None:
    """Throttled, best-effort drain of the provenance spool into ``provenance.db``.

    template_bookkeeping_plan.md §10/§15: "supervisor rotates each spool file
    (rename -> fresh fd), drains into ``provenance.db`` in one transaction
    (idempotent ``INSERT OR REPLACE``) ... Sole writer." This function is only
    ever called from the supervisor's own tick loop below, so the sole-writer
    invariant holds by construction -- no lock is taken here.

    Never raises: a missing/broken ``provenance`` package (it may still be
    mid-authoring, per the phased-PR rollout) or any per-``data_root`` failure
    must not affect scheduling. Imports are lazy and guarded so the scheduler
    keeps working even if the package cannot be imported at all.
    """
    global _last_provenance_drain_ts
    if not data_roots:
        return
    now = time.monotonic()
    if now - _last_provenance_drain_ts < _PROVENANCE_DRAIN_INTERVAL_S:
        return
    _last_provenance_drain_ts = now
    try:
        from syndiff_pipeline.common.scc_paths import (
            provenance_db_path,
            provenance_spool_dir,
        )
        from syndiff_pipeline.common.provenance.store import ProvenanceStore
        from syndiff_pipeline.common.provenance.ingest import drain_spool
    except Exception:
        log.debug(
            "Provenance package unavailable; skipping spool drain this pass",
            exc_info=True,
        )
        return
    for data_root in data_roots:
        try:
            store = ProvenanceStore(str(provenance_db_path(data_root)))
            drain_spool(store, provenance_spool_dir(data_root))
        except Exception:
            log.exception(
                "Provenance spool drain failed for data_root=%s (non-fatal)", data_root
            )


def run_supervisor_daemon(workspace_root: str) -> int:
    """Run supervisor daemon.
    
    Parameters
    ----------
    workspace_root : str
    
    Returns
    -------
    int"""
    global _shutdown, _lock_fd, _owned_lease
    _shutdown = False
    _owned_lease = None

    daemon.configure_process_logging("supervisor")
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    from syndiff_pipeline.common.orchestration.workspace import state_db_path

    db_path = str(state_db_path(workspace_root))
    pid = os.getpid()
    host = daemon.local_hostname()

    owned = lease.try_acquire_lease(workspace_root, host=host, pid=pid)
    if owned is None:
        log.info("Another supervisor already owns the lease; exiting.")
        return 0
    _owned_lease = owned
    # Keep lease/stop files permanently present for NFS visibility.
    lease.ensure_control_files(workspace_root)
    log.info(
        "Acquired supervisor lease generation=%s host=%s pid=%s",
        owned.generation,
        owned.host,
        owned.pid,
    )

    # Best-effort same-host flock; lease is authoritative and must not be blocked
    # by a stuck NFS flock.
    lock_cm = daemon.daemon_lock(workspace_root, blocking=False)
    fd = lock_cm.__enter__()
    _lock_fd = fd
    if fd is None:
        log.warning(
            "Could not acquire best-effort daemon.lock (lease still held); continuing"
        )

    discord_bot = None
    pid_path = logs.daemon_pid_path(workspace_root)
    try:
        daemon.write_process_identity(pid_path, pid, host=host)
        log.info("Supervisor started workspace=%s", workspace_root)
        state = pstate.PipelineState(db_path)
        _write_local_heartbeat(workspace_root)
        state.update_supervisor_heartbeat(pid)

        heartbeat_thread = threading.Thread(
            target=_supervisor_heartbeat_loop,
            args=(state, workspace_root, pid, _HEARTBEAT_INTERVAL_S, owned),
            name="supervisor-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        # Orphan /proc scan can be slow on large hosts — never block stop/heartbeat on it.
        threading.Thread(
            target=_cleanup_legacy_bots_async,
            args=(workspace_root,),
            name="legacy-bot-cleanup",
            daemon=True,
        ).start()
        discord_bot = _start_in_process_discord_bot(workspace_root)

        try:
            while not _shutdown:
                try:
                    _apply_commands(state)

                    provenance_data_roots: set[str] = set()
                    for run in state.list_active_runs():
                        if _shutdown:
                            break
                        run_id = run["run_id"]
                        try:
                            ctx = _load_run_context(state, run_id)
                            if ctx is None:
                                continue
                            data_root = getattr(ctx.cfg, "data_root", None)
                            if data_root:
                                provenance_data_roots.add(str(data_root))
                            _tick_run(state, run_id, ctx)
                            # Honor cancel/pause/stop intents promptly even when a
                            # large active-run set makes a full pass slow.
                            _apply_commands(state)
                        except Exception:
                            # Isolate per-run failures so one bad run cannot take
                            # down scheduling for every other active run.
                            log.exception("Error while processing run %s", run_id)

                    try:
                        discord_bot = _maybe_respawn_discord_bot(
                            workspace_root, discord_bot
                        )
                    except Exception:
                        log.warning("Discord bot watchdog failed", exc_info=True)

                    worker = _verify_worker()
                    active_runs = state.list_active_runs()
                    by_run = _collect_verify_status_by_run(state, active_runs)
                    write_verify_in_flight(workspace_root, by_run)
                    _maybe_drain_provenance_spool(provenance_data_roots)
                except Exception:
                    # Belt-and-braces around the whole tick: the per-run
                    # try/except above doesn't cover _apply_commands or
                    # state.list_active_runs() themselves (e.g. a
                    # StateDBUnavailableError from a stalled NFS connection
                    # open -- see state.py's _conn()). Losing one tick to a
                    # logged exception beats crashing the supervisor process.
                    log.exception("Error during supervisor tick")

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
                from syndiff_pipeline.common.orchestration.verify_worker import shutdown_verify_worker

                shutdown_verify_worker(wait=_shutdown)
            log.info("Supervisor shutting down workspace=%s", workspace_root)
            if discord_bot is not None:
                try:
                    discord_bot.stop()
                except Exception:
                    log.warning("Discord bot stop failed", exc_info=True)
            else:
                # Belt-and-suspenders when bot was never started / already None.
                try:
                    from syndiff_pipeline.template_creation.orchestration.discord_bot_control import (
                        stop_all_workspace_discord_bots,
                    )

                    stop_all_workspace_discord_bots(workspace_root)
                except Exception:
                    log.warning("Discord bot cleanup on shutdown failed", exc_info=True)
            try:
                state.clear_supervisor()
            except Exception:
                log.warning("Failed to clear supervisor row", exc_info=True)
            daemon.remove_pid_file(pid_path)
            try:
                logs.daemon_heartbeat_file(workspace_root).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                clear_verify_in_flight(workspace_root)
            except OSError:
                pass
            gen = _owned_lease.generation if _owned_lease is not None else owned.generation
            lease.clear_stop_request(workspace_root, only_generation=gen)
            lease.release_lease(workspace_root, host=host, pid=pid, generation=gen)
            _owned_lease = None
    finally:
        try:
            lock_cm.__exit__(None, None, None)
        except Exception:
            pass
        _lock_fd = None
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
    from syndiff_pipeline.common.orchestration.run_context import resolve_run_context
    from syndiff_pipeline.template_creation.orchestration import dispatch

    ctx = resolve_run_context(run_dir=run_dir, run_id=run_id)
    state = pstate.PipelineState(ctx.cfg.state_db_path)
    active = dispatch.parse_stage_list(stages_arg)

    run_row = state.get_run(run_id)
    if run_row is not None:
        log.error(
            "Run %s already exists; choose a new --run-id for submit/run, "
            "or use syndiff retry for failed stages.",
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
    from syndiff_pipeline.common.orchestration.run_setup import apply_post_create_run_setup

    apply_post_create_run_setup(state, run_id, ctx.targets, ctx.cfg, active)

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
    """Main.
    
    Parameters
    ----------
    argv : list[str] | None, optional, default ``None``
    
    Returns
    -------
    int"""
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
        workspace_root = str(load_workspace_root_from_deployment(deploy_path))
        record_deployment_path(workspace_root, deploy_path)
        return run_supervisor_daemon(workspace_root)

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
