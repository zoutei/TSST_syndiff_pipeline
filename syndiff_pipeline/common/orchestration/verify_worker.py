"""Background artifact verification for the supervisor scheduler.

On-disk ``stage_complete()`` checks can walk large NFS trees and parse
skycell CSVs for padding cells. Running them on the main scheduler thread
blocks reconcile, launches, and command handling. This module offloads
verification to a small thread pool; the main loop only schedules work and
applies finished results.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable

from syndiff_pipeline.template_creation.orchestration.runner_config import (
    ResolvedTargetConfig,
    RunnerConfig,
)
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    CHECKPOINT_STAGES,
    checkpoint_stage_indexed,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    copy_manifest_to_stable,
    stage_complete,
    write_stable_manifest,
)

log = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 1
_DRAIN_POLL_S = 0.05


@dataclass(frozen=True)
class VerifyTaskKey:
    """VerifyTaskKey."""
    run_id: str
    target_label: str
    stage: str


@dataclass(frozen=True)
class VerifyTask:
    """VerifyTask."""
    key: VerifyTaskKey
    manifest_path: str
    stable_path: str
    resolved: ResolvedTargetConfig
    runner_cfg: "RunnerConfig | None" = None
    meta: dict | None = None


@dataclass(frozen=True)
class BackfillTask:
    """BackfillTask."""
    manifest_path: str
    stable_path: str


@dataclass(frozen=True)
class VerifyOutcome:
    """VerifyOutcome."""
    key: VerifyTaskKey
    complete: bool
    stable_path: str
    resolved: ResolvedTargetConfig
    error: str | None = None


def _run_backfill_task(task: BackfillTask) -> None:
    """Run backfill task.
    
    Parameters
    ----------
    task : BackfillTask"""
    try:
        copy_manifest_to_stable(task.manifest_path, task.stable_path)
    except Exception as exc:  # noqa: BLE001 - backfill is best-effort
        log.debug(
            "Could not copy manifest %s -> %s: %s",
            task.manifest_path,
            task.stable_path,
            exc,
        )


def _run_verify_task(task: VerifyTask) -> VerifyOutcome:
    """Run verify task.
    
    Parameters
    ----------
    task : VerifyTask
    
    Returns
    -------
    VerifyOutcome"""
    try:
        if (
            task.runner_cfg is not None
            and task.runner_cfg.bookkeeping_trust_index
            and task.key.stage in CHECKPOINT_STAGES
        ):
            complete = checkpoint_stage_indexed(task.resolved, task.key.stage)
            return VerifyOutcome(
                key=task.key,
                complete=complete,
                stable_path=task.stable_path,
                resolved=task.resolved,
            )
        complete = stage_complete(
            task.resolved,
            task.key.stage,
            manifest_path=task.manifest_path,
            stable_manifest_path=task.stable_path,
            runner_cfg=task.runner_cfg,
            meta=task.meta,
        )
        if complete:
            try:
                write_stable_manifest(
                    task.resolved,
                    task.key.stage,
                    task.stable_path,
                    runner_cfg=task.runner_cfg,
                    meta=task.meta,
                )
            except Exception as exc:  # noqa: BLE001 - manifest write must not fail verify
                log.debug(
                    "Could not write stable manifest %s for %s: %s",
                    task.stable_path,
                    task.key.stage,
                    exc,
                )
        return VerifyOutcome(
            key=task.key,
            complete=complete,
            stable_path=task.stable_path,
            resolved=task.resolved,
        )
    except Exception as exc:  # noqa: BLE001 - verify must never kill the worker
        log.exception(
            "Artifact verify failed for %s / %s / %s",
            task.key.run_id,
            task.key.target_label,
            task.key.stage,
        )
        return VerifyOutcome(
            key=task.key,
            complete=False,
            stable_path=task.stable_path,
            resolved=task.resolved,
            error=str(exc),
        )


class ArtifactVerifyWorker:
    """ArtifactVerifyWorker."""
    def __init__(self, *, max_workers: int = _DEFAULT_MAX_WORKERS) -> None:
        """Init.
        
        Parameters
        ----------
        max_workers : int, optional, default ``_DEFAULT_MAX_WORKERS``"""
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="artifact-verify",
        )
        self._in_flight: dict[VerifyTaskKey, Future[VerifyOutcome]] = {}
        self._cancelled: set[VerifyTaskKey] = set()
        self._lock = threading.Lock()

    def schedule(self, tasks: Iterable[VerifyTask]) -> int:
        """Schedule.
        
        Parameters
        ----------
        tasks : Iterable[VerifyTask]
        
        Returns
        -------
        int"""
        scheduled = 0
        with self._lock:
            for task in tasks:
                if task.key in self._in_flight:
                    continue
                fut = self._executor.submit(_run_verify_task, task)
                self._in_flight[task.key] = fut
                scheduled += 1
        return scheduled

    def schedule_backfill(self, tasks: Iterable[BackfillTask]) -> int:
        """Copy per-run manifests to stable paths in the background (best-effort)."""
        scheduled = 0
        with self._lock:
            for task in tasks:
                self._executor.submit(_run_backfill_task, task)
                scheduled += 1
        return scheduled

    def is_in_flight(self, key: VerifyTaskKey) -> bool:
        """Is in flight.
        
        Parameters
        ----------
        key : VerifyTaskKey
        
        Returns
        -------
        bool"""
        with self._lock:
            return key in self._in_flight

    def in_flight_count(self, run_id: str | None = None) -> int:
        """In flight count.
        
        Parameters
        ----------
        run_id : str | None, optional, default ``None``
        
        Returns
        -------
        int"""
        with self._lock:
            if run_id is None:
                return len(self._in_flight)
            return sum(1 for key in self._in_flight if key.run_id == run_id)

    def in_flight_keys(self, run_id: str | None = None) -> list[VerifyTaskKey]:
        """In flight keys.
        
        Parameters
        ----------
        run_id : str | None, optional, default ``None``
        
        Returns
        -------
        list[VerifyTaskKey]"""
        with self._lock:
            keys = list(self._in_flight.keys())
        if run_id is not None:
            keys = [key for key in keys if key.run_id == run_id]
        return keys

    def cancel_keys(self, keys: Iterable[VerifyTaskKey]) -> int:
        """Drop in-flight verify futures for *keys*; return count cancelled."""
        cancelled = 0
        with self._lock:
            for key in keys:
                self._cancelled.add(key)
                fut = self._in_flight.pop(key, None)
                if fut is not None:
                    fut.cancel()
                    cancelled += 1
        return cancelled

    def cancel_run(self, run_id: str) -> int:
        """Cancel run.
        
        Parameters
        ----------
        run_id : str
        
        Returns
        -------
        int"""
        with self._lock:
            keys = [key for key in self._in_flight if key.run_id == run_id]
        return self.cancel_keys(keys)

    def drain(
        self,
        apply: Callable[[VerifyOutcome], int],
        *,
        run_id: str | None = None,
        block: bool = False,
        block_timeout_s: float = 0.0,
    ) -> int:
        """Apply finished verify outcomes via *apply*; return total skips applied."""
        total = 0
        deadline = (
            time.monotonic() + block_timeout_s if block_timeout_s > 0 else None
        )
        while True:
            total += self._drain_once(apply, run_id=run_id)
            if not block:
                break
            if self.in_flight_count(run_id) == 0:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(_DRAIN_POLL_S)
        return total

    def _drain_once(
        self,
        apply: Callable[[VerifyOutcome], int],
        *,
        run_id: str | None,
    ) -> int:
        """Drain once.
        
        Parameters
        ----------
        apply : Callable[[VerifyOutcome], int]
        run_id : str | None
        
        Returns
        -------
        int"""
        done: list[tuple[VerifyTaskKey, Future[VerifyOutcome]]] = []
        with self._lock:
            for key, fut in self._in_flight.items():
                if run_id is not None and key.run_id != run_id:
                    continue
                if fut.done():
                    done.append((key, fut))

        if not done:
            return 0

        applied = 0
        for key, fut in done:
            with self._lock:
                was_cancelled = key in self._cancelled
                if was_cancelled:
                    self._cancelled.discard(key)
                self._in_flight.pop(key, None)
            if was_cancelled:
                continue
            try:
                outcome = fut.result()
            except Exception:
                log.exception(
                    "Verify future failed for %s / %s / %s",
                    key.run_id,
                    key.target_label,
                    key.stage,
                )
                continue
            applied += apply(outcome)
        return applied

    def shutdown(self, *, wait: bool = True) -> None:
        """Shutdown.
        
        Parameters
        ----------
        wait : bool, optional, default ``True``"""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        with self._lock:
            self._in_flight.clear()
            self._cancelled.clear()


_worker: ArtifactVerifyWorker | None = None
_worker_lock = threading.Lock()


def init_verify_worker(max_workers: int = _DEFAULT_MAX_WORKERS) -> ArtifactVerifyWorker:
    """Create the singleton worker if absent (idempotent)."""
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = ArtifactVerifyWorker(max_workers=max_workers)
        return _worker


def get_verify_worker() -> ArtifactVerifyWorker:
    """Get verify worker.
    
    Returns
    -------
    ArtifactVerifyWorker"""
    return init_verify_worker()


def try_get_verify_worker() -> ArtifactVerifyWorker | None:
    """Return the worker if already initialized, without creating one."""
    with _worker_lock:
        return _worker


def verify_in_flight_count(run_id: str | None = None) -> int:
    """Return in-flight verify count inside the daemon process only.

    CLI tools must use ``verify_status.read_verify_in_flight`` instead; this
    function is not meaningful across processes and importing this module is
    expensive due to the ``verify`` dependency chain.
    """
    with _worker_lock:
        if _worker is None:
            return 0
        return _worker.in_flight_count(run_id)


def shutdown_verify_worker(*, wait: bool = True) -> None:
    """Shutdown verify worker.
    
    Parameters
    ----------
    wait : bool, optional, default ``True``"""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.shutdown(wait=wait)
            _worker = None


def reset_verify_worker_for_tests() -> None:
    """Tear down the singleton between unit tests.

    Also resets the scheduler's fast-path pool (deferred import to avoid a
    module cycle: scheduler.py already imports from this module at call
    time). The two pools share the same VerifyTaskKey namespace, so a stale
    fast-path future left over from a previous test can otherwise resolve a
    later test's candidate with an unrelated mocked result.
    """
    shutdown_verify_worker(wait=True)
    from syndiff_pipeline.common.orchestration import scheduler

    scheduler.reset_fast_path_pool_for_tests()
