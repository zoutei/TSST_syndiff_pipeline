"""HTCondor job submission and polling via the Condor CLI."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

_WRAPPER = Path(__file__).resolve().parent / "condor_wrapper.sh"

_JOB_IDLE = 1
_JOB_RUNNING = 2
_JOB_REMOVED = 3
_JOB_COMPLETED = 4
_JOB_HELD = 5

_POLL_GRACE_SECONDS = 120.0
HOLD_TIMEOUT_S = 600.0
_QUERY_RETRY_ATTEMPTS = 3
_QUERY_RETRY_DELAY_S = 0.5
_DISPLAY_QUERY_TIMEOUT_S = 15.0
CONDOR_EVICT_FAIL_THRESHOLD = 2

_EVENT_HEADER_RE = re.compile(r"^\s*\d{3}\s+\((\d+)\.\d+\.\d+\)")
_DISCONNECT_CYCLE_START_RE = re.compile(
    r"Job disconnected, attempting to reconnect", re.IGNORECASE
)
_RECONNECT_TARGET_RE = re.compile(r"reconnect to\s+([^\s,<]+)", re.IGNORECASE)
_NOT_FOUND_RE = re.compile(r"not found at execution machine", re.IGNORECASE)
_JOB_EXECUTING_HOST_RE = re.compile(
    r"Job executing on host.*?alias=([^&>\s]+)", re.IGNORECASE
)
_JOB_EVICTED_RE = re.compile(r"Job was evicted", re.IGNORECASE)

_HOLD_MEMORY_HOST_RE = re.compile(r"Error from\s+\S+@([^\s:]+):")
_HOLD_MEMORY_LIMIT_RE = re.compile(r"cgroup memory limit of\s+(\d+)\s+megabytes")
_HOLD_MEMORY_USAGE_RE = re.compile(r"[Ll]ast measured usage:\s+(\d+)\s+megabytes")
_HOLD_LOW_USAGE_RATIO_THRESHOLD = 0.5

_submission_times: dict[int, float] = {}
_held_times: dict[int, float] = {}


@dataclass(frozen=True)
class CondorResourceRequest:
    """CondorResourceRequest."""
    request_cpus: int = 64
    request_memory_mb: int = 500_000
    request_disk_kb: int | None = None
    host_stats_min_mem_mb: int = 128_000
    host_stats_max_load15: float = 10.0
    requirements: str | None = None
    rank: str | None = None


CONDOR_POLICY_ALLOWED = frozenset(
    {
        "request_cpus",
        "request_memory",
        "host_stats_min_mem_mb",
        "host_stats_max_load15",
    }
)
LEGACY_CONDOR_POLICY_KEYS = frozenset(
    {
        "requirements",
        "rank",
        "condor_requirements",
        "condor_rank",
    }
)


def parse_condor_policy_block(
    raw: dict | None,
    *,
    context: str,
    default_cpus: int,
    default_memory: int,
) -> tuple[int, int, int, float]:
    """Parse a YAML ``condor:`` block; reject legacy requirements/rank keys."""
    block = dict(raw or {})
    legacy = sorted(k for k in block if k in LEGACY_CONDOR_POLICY_KEYS)
    if legacy:
        raise ValueError(
            f"{context}: removed Condor keys {legacy}; use "
            "host_stats_min_mem_mb and host_stats_max_load15 instead"
        )
    unknown = sorted(k for k in block if k not in CONDOR_POLICY_ALLOWED)
    if unknown:
        raise ValueError(f"{context}: unknown condor keys {unknown}")
    request_memory = int(block.get("request_memory", default_memory))
    if "host_stats_min_mem_mb" not in block:
        log.warning(
            "%s: host_stats_min_mem_mb not set; defaulting the host-eligibility "
            "memory floor to request_memory (%dMB). Set it explicitly -- aliasing "
            "to the job's own memory request is often stricter than intended and "
            "can silently exclude every host.",
            context,
            request_memory,
        )
    min_mem = int(block.get("host_stats_min_mem_mb", request_memory))
    return (
        int(block.get("request_cpus", default_cpus)),
        request_memory,
        min_mem,
        float(block.get("host_stats_max_load15", 10.0)),
    )


def poll_grace_seconds() -> float:
    """Poll grace seconds.
    
    Returns
    -------
    float"""
    return _POLL_GRACE_SECONDS


def condor_status_label(status: int | None) -> str | None:
    """Map HTCondor JobStatus to a short display label."""
    if status == _JOB_IDLE:
        return "idle"
    if status == _JOB_RUNNING:
        return "running"
    if status == _JOB_HELD:
        return "held"
    return None


def format_condor_job_suffix(cluster_id: int, status: int | None) -> str:
    """Format a Condor queue-state suffix for progress detail (no leading space)."""
    label = condor_status_label(status)
    if label is None:
        return ""
    return f"condor {label} c{cluster_id}.0"


def condor_artifact_paths(
    runs_root: str,
    run_id: str,
    target_label: str,
    stage: str,
    *,
    mkdir: bool = True,
) -> dict[str, Path]:
    """Condor artifact paths.
    
    Parameters
    ----------
    runs_root : str
    run_id : str
    target_label : str
    stage : str
    mkdir : bool, optional, default ``True``
        When ``False``, paths are returned without creating directories (read-only use).
    
    Returns
    -------
    dict[str, Path]"""
    base = Path(runs_root) / run_id / "per_target" / target_label
    if mkdir:
        base.mkdir(parents=True, exist_ok=True)
    return {
        "stdout": base / f"{stage}.condor.stdout",
        "stderr": base / f"{stage}.condor.stderr",
        "log": base / f"{stage}.condor.log",
        "submit": base / f"{stage}.condor.submit",
        "clusters": base / f"{stage}.condor.clusters",
        "hold": base / f"{stage}.condor.hold",
        "poll_misses": base / f"{stage}.condor.poll_misses",
        "bad_machines": base / f"{stage}.condor.bad_machines",
        "eviction_state": base / f"{stage}.condor.eviction_state",
    }


def read_recorded_cluster_id(
    runs_root: str, run_id: str, target_label: str, stage: str
) -> int | None:
    """Read the last submitted cluster id from the durable ``*.condor.clusters`` file."""
    path = condor_artifact_paths(
        runs_root, run_id, target_label, stage, mkdir=False
    )["clusters"]
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return int(line)
        except ValueError:
            continue
    return None


def _read_hold_epoch(hold_path: Path) -> float | None:
    """Read hold epoch.
    
    Parameters
    ----------
    hold_path : Path
    
    Returns
    -------
    float | None"""
    try:
        line = hold_path.read_text(encoding="utf-8").strip()
        if not line:
            return None
        return float(line)
    except (OSError, ValueError):
        return None


def _write_hold_epoch(hold_path: Path, epoch: float) -> None:
    """Write hold epoch.
    
    Parameters
    ----------
    hold_path : Path
    epoch : float"""
    hold_path.parent.mkdir(parents=True, exist_ok=True)
    hold_path.write_text(f"{epoch}\n", encoding="utf-8")


def _clear_hold_epoch(hold_path: Path | None) -> None:
    """Clear hold epoch.
    
    Parameters
    ----------
    hold_path : Path | None"""
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
    """Resolve held since.
    
    Parameters
    ----------
    cluster_id : int
    hold_path : Path | None
    now : float
    
    Returns
    -------
    float"""
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


def _run_condor(
    args: Sequence[str],
    *,
    check: bool = True,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run condor.
    
    Parameters
    ----------
    args : Sequence[str]
    check : bool, optional, default ``True``
    timeout_s : float | None, optional, default ``None``
    
    Returns
    -------
    subprocess.CompletedProcess[str]"""
    proc = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc


def _format_arguments(cmd: Sequence[str]) -> str:
    """Format arguments.
    
    Parameters
    ----------
    cmd : Sequence[str]
    
    Returns
    -------
    str"""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _format_condor_environment(*, request_cpus: int | None = None) -> str | None:
    """Format condor environment.
    
    Parameters
    ----------
    request_cpus : int | None, optional, default ``None``
    
    Returns
    -------
    str | None"""
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
        parts.append(f"SYNDIFF_HYBRID_MAX_JOBS={int(request_cpus)}")
        # Stages that request multiple cores do their own process-level
        # parallelism (joblib/loky worker pools sized to request_cpus).
        # Without pinning BLAS/OMP thread pools to 1, each worker process can
        # independently spawn as many threads as visible cores, causing
        # severe oversubscription (e.g. 64 processes x 64 threads on a
        # 64-core box) that manifests as high "system" CPU / context-switch
        # overhead with near-zero forward progress.
        for _var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            parts.append(f"{_var}=1")
    benchmark_tag = os.environ.get("SYNDIFF_REMAP_BENCHMARK_TAG")
    if benchmark_tag:
        parts.append(f"SYNDIFF_REMAP_BENCHMARK_TAG={shlex.quote(benchmark_tag)}")
    # Debug/backfill-only downsample skycell restriction (2026-08-23), see
    # dispatch.py's own only_skycells hookup -- unset in every real
    # deployment, never a normal part of a submitted run's environment.
    only_skycells = os.environ.get("SYNDIFF_DOWNSAMPLE_ONLY_SKYCELLS")
    if only_skycells:
        parts.append(f"SYNDIFF_DOWNSAMPLE_ONLY_SKYCELLS={shlex.quote(only_skycells)}")
    if not parts:
        return None
    return " ".join(parts)


def _parse_status_exit(parts: Sequence[str]) -> tuple[int | None, int | None]:
    """Parse status exit.
    
    Parameters
    ----------
    parts : Sequence[str]
    
    Returns
    -------
    tuple[int | None, int | None]"""
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
    """Record cluster submission.
    
    Parameters
    ----------
    artifacts : dict[str, Path]
    cluster_id : int"""
    clusters_path = artifacts["clusters"]
    with clusters_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{cluster_id}\n")


def normalize_condor_host(host: str) -> str:
    """Normalize a Condor slot host to a Machine attribute value."""
    text = str(host or "").strip()
    if "@" in text:
        text = text.rsplit("@", 1)[-1]
    return text


def read_bad_machines(path: Path | str) -> set[str]:
    p = Path(path)
    if not p.is_file():
        return set()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return set()
    hosts = payload.get("hosts") if isinstance(payload, dict) else payload
    if not isinstance(hosts, list):
        return set()
    return {normalize_condor_host(str(host)) for host in hosts if str(host).strip()}


def write_bad_machines(path: Path | str, hosts: set[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(normalize_condor_host(host) for host in hosts if str(host).strip())
    p.write_text(json.dumps({"hosts": ordered}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_bad_machine(path: Path | str, host: str) -> bool:
    normalized = normalize_condor_host(host)
    if not normalized:
        return False
    hosts = read_bad_machines(path)
    if normalized in hosts:
        return False
    hosts.add(normalized)
    write_bad_machines(path, hosts)
    return True


def merge_requirements_with_exclusions(
    requirements: str | None,
    bad_machines: set[str],
) -> str | None:
    if not bad_machines:
        return requirements
    exclusions = " && ".join(
        f'Machine != "{host}"' for host in sorted(bad_machines)
    )
    base = (requirements or "True").strip() or "True"
    return f"({base}) && {exclusions}"


def tally_execution_eviction_failures(
    log_text: str, cluster_id: int | None = None
) -> dict[str, int]:
    """Count disconnect -> reconnect-failed ("not found at execution machine")
    cycles per execution host from a raw HTCondor user log.

    The real HTCondor event sequence for a job that matches a broken/vanished
    slot looks like::

        022 (CID.P.S) ... Job disconnected, attempting to reconnect
            Socket between submit and execute hosts closed unexpectedly
            Trying to reconnect to slot1_1@bad-host.example.com <...>
        ...
        024 (CID.P.S) ... Job reconnection failed
            Job not found at execution machine
            Can not reconnect to slot1_1@bad-host.example.com, rescheduling job
        ...
        004 (CID.P.S) ... Job was evicted.
            ...
            Job not found at execution machine
            ...

    ``Job not found at execution machine`` appears twice per cycle (once under
    the 024 event, once under the 004 event), so we count at most once per
    cycle, keyed to the host named in the preceding "reconnect to" line. A new
    "022 Job disconnected" line always starts a new cycle, which is essential
    because the *same* bad host is usually reused across many consecutive
    cycles.

    Because a stage's Condor log file is appended across every retry/cluster
    (the log path is stable per stage), *cluster_id* restricts the tally to
    events belonging to that specific cluster, avoiding misattributing an
    older, already-handled cluster's failures to a newly submitted one.
    """
    counts: dict[str, int] = {}
    current_host: str | None = None
    counted_for_cycle = False
    active_cluster: int | None = None
    target_cluster = int(cluster_id) if cluster_id is not None else None
    for line in log_text.splitlines():
        header = _EVENT_HEADER_RE.match(line)
        if header:
            try:
                active_cluster = int(header.group(1))
            except ValueError:
                active_cluster = None
        if target_cluster is not None and active_cluster != target_cluster:
            continue
        if _DISCONNECT_CYCLE_START_RE.search(line):
            current_host = None
            counted_for_cycle = False
            continue
        match = _RECONNECT_TARGET_RE.search(line)
        if match:
            current_host = normalize_condor_host(match.group(1))
            continue
        if current_host and not counted_for_cycle and _NOT_FOUND_RE.search(line):
            counts[current_host] = counts.get(current_host, 0) + 1
            counted_for_cycle = True
    return counts


def tally_rapid_reeviction_failures(
    log_text: str, cluster_id: int | None = None
) -> dict[str, int]:
    """Count clean execute -> evict cycles per host, no disconnect involved.

    Complements ``tally_execution_eviction_failures``, which only recognizes
    the disconnect/"not found at execution machine" pattern (an execute host
    vanishing off the network). This catches a different, equally real
    failure mode seen in production: a host's startd cleanly re-evicting the
    same job over and over within seconds of each (re)match -- plain "004
    Job was evicted" events with no disconnect/reconnect lines at all, e.g.
    a broken PREEMPT/RANK policy fighting the negotiator on one machine.
    Counts one failure per (``001`` executing-on-host, ``004`` evicted) pair
    with no other event in between; a job that runs to completion never
    increments anything. Both tallies feed the same host-exclusion path in
    ``eviction_requeue_host``.
    """
    counts: dict[str, int] = {}
    current_host: str | None = None
    active_cluster: int | None = None
    target_cluster = int(cluster_id) if cluster_id is not None else None
    for line in log_text.splitlines():
        header = _EVENT_HEADER_RE.match(line)
        if header:
            try:
                active_cluster = int(header.group(1))
            except ValueError:
                active_cluster = None
        if target_cluster is not None and active_cluster != target_cluster:
            continue
        exec_match = _JOB_EXECUTING_HOST_RE.search(line)
        if exec_match:
            current_host = normalize_condor_host(exec_match.group(1))
            continue
        if current_host and _JOB_EVICTED_RE.search(line):
            counts[current_host] = counts.get(current_host, 0) + 1
            current_host = None
    return counts


def combined_eviction_tallies(log_text: str, cluster_id: int | None = None) -> dict[str, int]:
    """Merge both eviction-detection heuristics into one per-host tally."""
    combined = tally_execution_eviction_failures(log_text, cluster_id=cluster_id)
    for host, count in tally_rapid_reeviction_failures(log_text, cluster_id=cluster_id).items():
        combined[host] = max(combined.get(host, 0), count)
    return combined


def _read_eviction_state(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_eviction_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_low_usage_memory_hold(
    hold_reason: str | None,
    *,
    usage_ratio_threshold: float = _HOLD_LOW_USAGE_RATIO_THRESHOLD,
) -> str | None:
    """Return the host to exclude when a cgroup memory-limit hold's measured
    usage is far below the limit (host-specific pressure, not genuine app need).

    Genuine memory-bloat holds (measured usage close to the limit) return
    None so a perfectly good host isn't excluded for a real app-level OOM.
    """
    if not hold_reason:
        return None
    host_match = _HOLD_MEMORY_HOST_RE.search(hold_reason)
    limit_match = _HOLD_MEMORY_LIMIT_RE.search(hold_reason)
    usage_match = _HOLD_MEMORY_USAGE_RE.search(hold_reason)
    if not (host_match and limit_match and usage_match):
        return None
    limit_mb = int(limit_match.group(1))
    if limit_mb <= 0:
        return None
    usage_mb = int(usage_match.group(1))
    if usage_mb / limit_mb >= usage_ratio_threshold:
        return None
    return normalize_condor_host(host_match.group(1))


def eviction_requeue_host(
    log_path: Path | str,
    *,
    cluster_id: int,
    eviction_state_path: Path | str,
    threshold: int = CONDOR_EVICT_FAIL_THRESHOLD,
) -> str | None:
    """Return a host to exclude when immediate-evict failures exceed *threshold*."""
    path = Path(log_path)
    if not path.is_file():
        return None
    tallies = combined_eviction_tallies(
        path.read_text(encoding="utf-8", errors="replace"), cluster_id=cluster_id
    )
    if not tallies:
        return None
    state = _read_eviction_state(Path(eviction_state_path))
    acted = state.get("acted_clusters")
    if not isinstance(acted, dict):
        acted = {}
    cluster_key = str(int(cluster_id))
    cluster_acted = acted.get(cluster_key)
    if not isinstance(cluster_acted, dict):
        cluster_acted = {}
    for host, count in tallies.items():
        if count < threshold:
            continue
        if int(cluster_acted.get(host, 0)) >= int(count):
            continue
        return host
    return None


def record_eviction_requeue(
    eviction_state_path: Path | str,
    *,
    cluster_id: int,
    host: str,
    failure_count: int,
) -> None:
    path = Path(eviction_state_path)
    state = _read_eviction_state(path)
    acted = state.get("acted_clusters")
    if not isinstance(acted, dict):
        acted = {}
    cluster_key = str(int(cluster_id))
    cluster_acted = acted.get(cluster_key)
    if not isinstance(cluster_acted, dict):
        cluster_acted = {}
    normalized = normalize_condor_host(host)
    cluster_acted[normalized] = int(failure_count)
    acted[cluster_key] = cluster_acted
    state["acted_clusters"] = acted
    _write_eviction_state(path, state)


def apply_bad_machine_exclusions(
    resources: CondorResourceRequest,
    artifacts: dict[str, Path],
) -> CondorResourceRequest:
    bad = read_bad_machines(artifacts["bad_machines"])
    if not bad:
        return resources
    merged = merge_requirements_with_exclusions(resources.requirements, bad)
    if merged == resources.requirements:
        return resources
    return replace(resources, requirements=merged)


def write_submit_file(
    submit_path: Path,
    cmd: Sequence[str],
    artifacts: dict[str, Path],
    resources: CondorResourceRequest,
) -> None:
    """Write submit file.
    
    Parameters
    ----------
    submit_path : Path
    cmd : Sequence[str]
    artifacts : dict[str, Path]
    resources : CondorResourceRequest"""
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
    if resources.request_disk_kb is not None:
        lines.append(f"request_disk = {int(resources.request_disk_kb)}")
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
    """Submit one stage command to Condor; return (cluster id, wall-clock submit epoch).

    Before writing a fresh submit file, rotates the previous attempt's
    ``stdout``/``stderr``/``log``/``submit`` artifacts aside into
    ``attempts/{prior_launch_token}/`` (see
    :func:`syndiff_pipeline.common.orchestration.logs.
    read_previous_launch_token`, :func:`...rotate_attempt_artifact`). Safe
    to self-determine here: this runs on the submit side, before the
    execute-side process's own status write (the thing that would
    otherwise overwrite the very status.json this reads) has happened.
    ``clusters``/``hold``/``poll_misses``/``bad_machines``/
    ``eviction_state`` are durable, cumulative-by-design state across
    resubmissions and are deliberately left alone.
    """
    resources = resources or CondorResourceRequest()
    artifacts = condor_artifact_paths(runs_root, run_id, target_label, stage)
    from syndiff_pipeline.common.orchestration.logs import (
        read_previous_launch_token,
        rotate_attempt_artifact,
    )

    prior_launch_token = read_previous_launch_token(runs_root, run_id, target_label, stage)
    for key in ("stdout", "stderr", "log", "submit"):
        rotate_attempt_artifact(artifacts[key], prior_launch_token)
    from syndiff_pipeline.common.orchestration.host_stats import apply_host_stats_policy

    resources = apply_host_stats_policy(resources)
    resources = apply_bad_machine_exclusions(resources, artifacts)
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
        "Submitted Condor cluster %s for %s / %s (cpus=%s mem=%sMB disk=%sKB req=%r rank=%r)",
        cluster_id,
        target_label,
        stage,
        resources.request_cpus,
        resources.request_memory_mb,
        resources.request_disk_kb,
        resources.requirements,
        resources.rank,
    )
    return cluster_id, submit_epoch


def _query_queue(
    cluster_id: int, *, timeout_s: float | None = _DISPLAY_QUERY_TIMEOUT_S
) -> tuple[int | None, int | None]:
    """Query queue.

    Parameters
    ----------
    cluster_id : int

    Returns
    -------
    tuple[int | None, int | None]"""
    try:
        proc = _run_condor(
            ["condor_q", str(cluster_id), "-af", "JobStatus", "ExitCode"],
            check=False,
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.warning("condor_q timed out after %.0fs for cluster %s", timeout_s or 0.0, cluster_id)
        return None, None
    line = proc.stdout.strip()
    if not line:
        return None, None
    return _parse_status_exit(line.split())


def _query_history(
    cluster_id: int, *, timeout_s: float | None = _DISPLAY_QUERY_TIMEOUT_S
) -> tuple[int | None, int | None]:
    """Query history.

    Parameters
    ----------
    cluster_id : int

    Returns
    -------
    tuple[int | None, int | None]"""
    try:
        proc = _run_condor(
            ["condor_history", str(cluster_id), "-af", "JobStatus", "ExitCode", "-limit", "1"],
            check=False,
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "condor_history timed out after %.0fs for cluster %s", timeout_s or 0.0, cluster_id
        )
        return None, None
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        return None, None
    return _parse_status_exit(line.split())


def _query_hold_reason(
    cluster_id: int, *, timeout_s: float | None = _DISPLAY_QUERY_TIMEOUT_S
) -> str | None:
    """Query hold reason.

    Parameters
    ----------
    cluster_id : int

    Returns
    -------
    str | None"""
    try:
        proc = _run_condor(
            ["condor_q", str(cluster_id), "-af", "HoldReason"],
            check=False,
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "condor_q (hold reason) timed out after %.0fs for cluster %s",
            timeout_s or 0.0,
            cluster_id,
        )
        return None
    line = proc.stdout.strip()
    return line or None


def _query_clusters_once(
    unique_ids: list[int], *, timeout_s: float | None = _DISPLAY_QUERY_TIMEOUT_S
) -> dict[int, tuple[int | None, int | None]]:
    result: dict[int, tuple[int | None, int | None]] = {}
    try:
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
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "condor_q timed out after %.0fs for %d cluster(s)", timeout_s or 0.0, len(unique_ids)
        )
        for cluster_id in unique_ids:
            result[cluster_id] = _query_history(cluster_id, timeout_s=timeout_s)
        return result
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


def query_clusters(cluster_ids: Sequence[int]) -> dict[int, tuple[int | None, int | None]]:
    """Batch-query Condor for multiple cluster ids (retries transient misses)."""
    if not cluster_ids:
        return {}
    unique_ids = list(dict.fromkeys(int(cluster_id) for cluster_id in cluster_ids))
    result: dict[int, tuple[int | None, int | None]] = {}
    for attempt in range(_QUERY_RETRY_ATTEMPTS):
        result = _query_clusters_once(unique_ids)
        missing = [
            cluster_id
            for cluster_id in unique_ids
            if result.get(cluster_id, (None, None))[0] is None
        ]
        if not missing or attempt + 1 >= _QUERY_RETRY_ATTEMPTS:
            break
        time.sleep(_QUERY_RETRY_DELAY_S)
    return result


def query_clusters_display(
    cluster_ids: Sequence[int],
    *,
    timeout_s: float | None = _DISPLAY_QUERY_TIMEOUT_S,
) -> dict[int, tuple[int | None, int | None]]:
    """Batch-query Condor queue state for progress display (no history fallback)."""
    if not cluster_ids:
        return {}
    unique_ids = list(dict.fromkeys(int(cluster_id) for cluster_id in cluster_ids))
    result: dict[int, tuple[int | None, int | None]] = {}
    try:
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
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "condor_q display query timed out after %.0fs for %d cluster(s)",
            timeout_s or 0.0,
            len(unique_ids),
        )
        return {}
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
            result[cluster_id] = (None, None)
    return result


def poll_cluster_status(
    cluster_id: int,
    status: int | None,
    exit_code: int | None,
    *,
    submitted_at: float | None = None,
    hold_timeout_s: float = HOLD_TIMEOUT_S,
    hold_path: Path | None = None,
    bad_machines_path: Path | None = None,
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
            if bad_machines_path is not None:
                bad_host = parse_low_usage_memory_hold(hold_reason)
                if bad_host is not None:
                    log.warning(
                        "Condor cluster %s held on %s with usage far below the "
                        "memory limit; excluding host",
                        cluster_id,
                        bad_host,
                    )
                    add_bad_machine(bad_machines_path, bad_host)
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
    """Remove cluster.
    
    Parameters
    ----------
    cluster_id : int
    hold_path : Path | None, optional, default ``None``
    
    Returns
    -------
    bool"""
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
    """Sweep run condor clusters.
    
    Parameters
    ----------
    state
    cfg
    run_id : str
    
    Returns
    -------
    int"""
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
    """Sweep run condor audit clusters.
    
    Parameters
    ----------
    runs_root : str
    run_id : str
    
    Returns
    -------
    int"""
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
