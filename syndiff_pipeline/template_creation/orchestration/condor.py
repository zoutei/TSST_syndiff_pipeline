"""HTCondor job submission and polling via the Condor CLI."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

_WRAPPER = Path(__file__).resolve().parent / "condor_wrapper.sh"

_JOB_COMPLETED = 4
_JOB_REMOVED = 3

_POLL_GRACE_SECONDS = 120.0

_submission_times: dict[int, float] = {}


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
    }


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
    parts = line.split()
    if not parts:
        return None, None
    status = int(parts[0])
    exit_code: int | None = None
    if len(parts) > 1 and parts[1] not in ("undefined", "?"):
        try:
            exit_code = int(parts[1])
        except ValueError:
            exit_code = None
    return status, exit_code


def _query_history(cluster_id: int) -> tuple[int | None, int | None]:
    proc = _run_condor(
        ["condor_history", str(cluster_id), "-af", "JobStatus", "ExitCode"],
        check=False,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        return None, None
    parts = line.split()
    status = int(parts[0])
    exit_code: int | None = None
    if len(parts) > 1 and parts[1] not in ("undefined", "?"):
        try:
            exit_code = int(parts[1])
        except ValueError:
            exit_code = None
    return status, exit_code


def poll_cluster(cluster_id: int, *, submitted_at: float | None = None) -> int | None:
    """Return None while running; otherwise the job exit code.

    *submitted_at* must be wall-clock epoch seconds (stored in DB), not monotonic.
    """
    status, exit_code = _query_queue(cluster_id)
    if status is None:
        status, exit_code = _query_history(cluster_id)
    if status is None:
        ts = submitted_at if submitted_at is not None else _submission_times.get(cluster_id)
        if ts is not None and time.time() - ts < _POLL_GRACE_SECONDS:
            return None
        log.warning("Condor cluster %s not found in queue or history", cluster_id)
        return 1
    if status == _JOB_COMPLETED:
        return exit_code if exit_code is not None else 0
    if status == _JOB_REMOVED:
        # condor_rm on a running job often records ExitCode 0 when the worker
        # handled SIGTERM cleanly; treat that as canceled (143), not success.
        if exit_code in (None, 0):
            return 143
        return exit_code
    return None


def remove_cluster(cluster_id: int) -> bool:
    proc = _run_condor(["condor_rm", str(cluster_id)], check=False)
    if proc.returncode == 0:
        log.info("Removed Condor cluster %s", cluster_id)
        _submission_times.pop(cluster_id, None)
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
        if remove_cluster(int(cluster_id)):
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
