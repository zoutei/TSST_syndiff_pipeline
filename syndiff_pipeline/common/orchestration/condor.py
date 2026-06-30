"""HTCondor job submission and polling via the Condor CLI."""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

_WRAPPER = Path(__file__).resolve().parent / "condor_wrapper.sh"

_JOB_REMOVED = 3
_JOB_COMPLETED = 4
_JOB_HELD = 5

_POLL_GRACE_SECONDS = 120.0
HOLD_TIMEOUT_S = 600.0

_submission_times: dict[int, float] = {}
_held_times: dict[int, float] = {}


@dataclass(frozen=True)
class CondorResourceRequest:
    request_cpus: int = 64
    request_memory_mb: int = 500_000
    requirements: str | None = "Memory >= 500000 && LoadAvg < 10"
    rank: str | None = "-LoadAvg"


def wrapper_path() -> Path:
    return _WRAPPER


def poll_grace_seconds() -> float:
    return _POLL_GRACE_SECONDS


def condor_artifact_paths(
    runs_root: str, run_id: str, target_label: str, stage: str
) -> dict[str, Path]:
    base = Path(runs_root) / run_id / "per_target" / target_label
    base.mkdir(parents=True, exist_ok=True)
    return {
        "stdout": base / f"{stage}.condor.stdout",
        "stderr": base / f"{stage}.condor.stderr",
        "log": base / f"{stage}.condor.log",
        "submit": base / f"{stage}.condor.submit",
        "clusters": base / f"{stage}.condor.clusters",
        "hold": base / f"{stage}.condor.hold",
    }


def _read_hold_epoch(hold_path: Path) -> float | None:
    try:
        line = hold_path.read_text(encoding="utf-8").strip()
        if not line:
            return None
        return float(line)
    except (OSError, ValueError):
        return None


def _write_hold_epoch(hold_path: Path, epoch: float) -> None:
    hold_path.parent.mkdir(parents=True, exist_ok=True)
    hold_path.write_text(f"{epoch}\n", encoding="utf-8")


def _clear_hold_epoch(hold_path: Path | None) -> None:
    if hold_path is None:
        return
    try:
        hold_path.unlink(missing_ok=True)
    except OSError:
        pass


def _resolve_held_since(
    cluster_id: int,
    *,
    hold_path: Path | None,
    now: float,
) -> float:
    if cluster_id in _held_times:
        return _held_times[cluster_id]
    if hold_path is not None:
        persisted = _read_hold_epoch(hold_path)
        if persisted is not None:
            _held_times[cluster_id] = persisted
            return persisted
    _held_times[cluster_id] = now
    if hold_path is not None:
        _write_hold_epoch(hold_path, now)
    return now


def _run_condor(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc


def _format_arguments(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _format_condor_environment(*, request_cpus: int | None = None) -> str | None:
    parts: list[str] = []
    conda_sh = os.environ.get("SYNDIFF_CONDA_SH")
    if conda_sh:
        conda_env = os.environ.get("SYNDIFF_CONDA_ENV", "syndiff")
        parts.extend(
            [
                f"SYNDIFF_CONDA_SH={shlex.quote(conda_sh)}",
                f"SYNDIFF_CONDA_ENV={shlex.quote(conda_env)}",
            ]
        )
    if request_cpus is not None and int(request_cpus) > 0:
        parts.append(f"SYNDIFF_REQUEST_CPUS={int(request_cpus)}")
    if not parts:
        return None
    return " ".join(parts)


def _parse_status_exit(parts: Sequence[str]) -> tuple[int | None, int | None]:
    if not parts:
        return None, None
    try:
        status = int(parts[0])
    except ValueError:
        return None, None
    exit_code: int | None = None
    if len(parts) > 1 and parts[1] not in ("undefined", "?"):
        try:
            exit_code = int(parts[1])
        except ValueError:
            exit_code = None
    return status, exit_code


def _record_cluster_submission(artifacts: dict[str, Path], cluster_id: int) -> None:
    clusters_path = artifacts["clusters"]
    with clusters_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{cluster_id}\n")


def write_submit_file(
    submit_path: Path,
    cmd: Sequence[str],
    artifacts: dict[str, Path],
    resources: CondorResourceRequest,
) -> None:
    if not _WRAPPER.is_file():
        raise FileNotFoundError(f"Condor wrapper missing: {_WRAPPER}")
    lines = [
        f"executable = {_WRAPPER}",
        f"arguments = {_format_arguments(cmd)}",
        "getenv = false",
        "should_transfer_files = NO",
        f"request_cpus = {resources.request_cpus}",
        f"request_memory = {resources.request_memory_mb}",
    ]
    environment = _format_condor_environment(request_cpus=resources.request_cpus)
    if environment:
        lines.append(f'environment = "{environment}"')
    if resources.requirements:
        lines.append(f"requirements = {resources.requirements}")
    if resources.rank:
        lines.append(f"rank = {resources.rank}")
    lines.extend(
        [
            f"output = {artifacts['stdout']}",
            f"error = {artifacts['stderr']}",
            f"log = {artifacts['log']}",
            "queue 1",
            "",
        ]
    )
    submit_path.write_text("\n".join(lines), encoding="utf-8")


def submit_job(
    cmd: Sequence[str],
    runs_root: str,
    run_id: str,
    target_label: str,
    stage: str,
    resources: CondorResourceRequest | None = None,
) -> tuple[int, float]:
    """Submit one stage command to Condor; return (cluster id, wall-clock submit epoch)."""
    resources = resources or CondorResourceRequest()
    artifacts = condor_artifact_paths(runs_root, run_id, target_label, stage)
    write_submit_file(artifacts["submit"], cmd, artifacts, resources)
    proc = _run_condor(["condor_submit", str(artifacts["submit"])])
    match = re.search(r"submitted to cluster (\d+)", proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse condor_submit output: {proc.stdout.strip()}")
    cluster_id = int(match.group(1))
    submit_epoch = time.time()
    _submission_times[cluster_id] = submit_epoch
    _record_cluster_submission(artifacts, cluster_id)
    log.info(
        "Submitted Condor cluster %s for %s / %s (cpus=%s mem=%sMB req=%r rank=%r)",
        cluster_id,
        target_label,
        stage,
        resources.request_cpus,
        resources.request_memory_mb,
        resources.requirements,
        resources.rank,
    )
    return cluster_id, submit_epoch


def _query_queue(cluster_id: int) -> tuple[int | None, int | None]:
    proc = _run_condor(
        ["condor_q", str(cluster_id), "-af", "JobStatus", "ExitCode"],
        check=False,
    )
    line = proc.stdout.strip()
    if not line:
        return None, None
    return _parse_status_exit(line.split())


def _query_history(cluster_id: int) -> tuple[int | None, int | None]:
    proc = _run_condor(
        ["condor_history", str(cluster_id), "-af", "JobStatus", "ExitCode", "-limit", "1"],
        check=False,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        return None, None
    return _parse_status_exit(line.split())


def _query_hold_reason(cluster_id: int) -> str | None:
    proc = _run_condor(
        ["condor_q", str(cluster_id), "-af", "HoldReason"],
        check=False,
    )
    line = proc.stdout.strip()
    return line or None


def query_clusters(cluster_ids: Sequence[int]) -> dict[int, tuple[int | None, int | None]]:
    """Batch-query Condor for multiple cluster ids."""
    if not cluster_ids:
        return {}
    unique_ids = list(dict.fromkeys(int(cluster_id) for cluster_id in cluster_ids))
    result: dict[int, tuple[int | None, int | None]] = {}
    proc = _run_condor(
        [
            "condor_q",
            *[str(cluster_id) for cluster_id in unique_ids],
            "-af",
            "ClusterId",
            "JobStatus",
            "ExitCode",
        ],
        check=False,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            cluster_id = int(parts[0])
        except ValueError:
            continue
        status, exit_code = _parse_status_exit(parts[1:])
        if status is not None:
            result[cluster_id] = (status, exit_code)
    for cluster_id in unique_ids:
        if cluster_id not in result:
            result[cluster_id] = _query_history(cluster_id)
    return result


def poll_cluster_status(
    cluster_id: int,
    status: int | None,
    exit_code: int | None,
    *,
    submitted_at: float | None = None,
    hold_timeout_s: float = HOLD_TIMEOUT_S,
    hold_path: Path | None = None,
) -> int | None:
    """Map a Condor JobStatus/ExitCode pair to a stage exit code, or None if still running."""
    if status is None:
        ts = submitted_at if submitted_at is not None else _submission_times.get(cluster_id)
        if ts is not None and time.time() - ts < _POLL_GRACE_SECONDS:
            return None
        log.warning("Condor cluster %s not found in queue or history", cluster_id)
        return 1
    if status == _JOB_COMPLETED:
        _held_times.pop(cluster_id, None)
        _clear_hold_epoch(hold_path)
        return exit_code if exit_code is not None else 0
    if status == _JOB_REMOVED:
        _held_times.pop(cluster_id, None)
        _clear_hold_epoch(hold_path)
        # condor_rm on a running job often records ExitCode 0 when the worker
        # handled SIGTERM cleanly; treat that as canceled (143), not success.
        if exit_code in (None, 0):
            return 143
        return exit_code
    if status == _JOB_HELD:
        now = time.time()
        held_since = _resolve_held_since(cluster_id, hold_path=hold_path, now=now)
        hold_reason = _query_hold_reason(cluster_id)
        log.warning(
            "Condor cluster %s held (reason: %s)",
            cluster_id,
            hold_reason or "unknown",
        )
        if now - held_since >= hold_timeout_s:
            log.warning(
                "Removing held Condor cluster %s after %.0fs timeout (reason: %s)",
                cluster_id,
                hold_timeout_s,
                hold_reason or "unknown",
            )
            _clear_hold_epoch(hold_path)
            remove_cluster(cluster_id)
            return 1
        return None
    return None


def poll_cluster(
    cluster_id: int,
    *,
    submitted_at: float | None = None,
    hold_timeout_s: float = HOLD_TIMEOUT_S,
    hold_path: Path | None = None,
) -> int | None:
    """Return None while running; otherwise the job exit code.

    *submitted_at* must be wall-clock epoch seconds (stored in DB), not monotonic.
    """
    status, exit_code = _query_queue(cluster_id)
    if status is None:
        status, exit_code = _query_history(cluster_id)
    return poll_cluster_status(
        cluster_id,
        status,
        exit_code,
        submitted_at=submitted_at,
        hold_timeout_s=hold_timeout_s,
        hold_path=hold_path,
    )


def remove_cluster(cluster_id: int, *, hold_path: Path | None = None) -> bool:
    proc = _run_condor(["condor_rm", str(cluster_id)], check=False)
    if proc.returncode == 0:
        log.info("Removed Condor cluster %s", cluster_id)
        _submission_times.pop(cluster_id, None)
        _held_times.pop(cluster_id, None)
        _clear_hold_epoch(hold_path)
        return True
    msg = (proc.stderr or proc.stdout or "").strip()
    log.warning("condor_rm %s failed (exit %s): %s", cluster_id, proc.returncode, msg)
    return False


def sweep_run_condor_clusters(state, cfg, run_id: str) -> int:
    removed = 0
    for job in state.running_jobs(run_id):
        cluster_id = job.native_id
        if cluster_id is None:
            continue
        executor = job.executor or cfg.stage_executor(job.stage)
        if executor != "condor":
            continue
        artifacts = condor_artifact_paths(
            runs_root, run_id, job.target_label, job.stage
        )
        if remove_cluster(int(cluster_id), hold_path=artifacts["hold"]):
            removed += 1
    return removed


def sweep_run_condor_audit_clusters(runs_root: str, run_id: str) -> int:
    removed = 0
    base = Path(runs_root) / run_id / "per_target"
    if not base.is_dir():
        return 0
    for clusters_path in base.rglob("*.condor.clusters"):
        for line in clusters_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cluster_id = int(line)
            except ValueError:
                continue
            status, _ = _query_queue(cluster_id)
            if status is None:
                continue
            if status in (_JOB_COMPLETED, _JOB_REMOVED):
                continue
            if remove_cluster(cluster_id):
                removed += 1
    return removed
