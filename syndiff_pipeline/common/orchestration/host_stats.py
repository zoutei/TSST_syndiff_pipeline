"""Live Condor ClassAd host memory/load → machine selection at submit time.

Memory/load come straight from HTCondor's own ClassAds: ``MemAvailableMB`` is
published by a ``STARTD_CRON`` job on every plscience execute host (real
``/proc/meminfo MemAvailable``, refreshed every 30s -- unlike Condor's own
``Memory`` attribute, which only tracks Condor-claimed capacity and misses
memory consumed outside Condor). ``DetectedMemory``/``LoadAvg`` are native
Condor attributes. There is no local sampler daemon or heartbeat file to
maintain; a host's data is exactly as fresh as the last time we queried
``condor_status``.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.condor import CondorResourceRequest

log = logging.getLogger(__name__)

_CONDOR_STATUS_TIMEOUT_S = 15.0
_CONDOR_STATUS_FIELDS = ("Machine", "DetectedMemory", "MemAvailableMB", "LoadAvg")


@dataclass(frozen=True)
class HostSample:
    """One host's live memory/load, as of the moment it was queried."""

    hostname: str
    mem_available_mb: int
    mem_total_mb: int
    load15: float


@dataclass(frozen=True)
class HostSelection:
    """Result of evaluating all expected hosts against thresholds."""

    eligible: tuple[HostSample, ...]
    excluded: dict[str, tuple[str, ...]]
    usable: bool


def expected_hosts() -> list[str]:
    return [f"plscience{n}.stsci.edu" for n in range(1, 16)]


def _parse_condor_status_output(stdout: str) -> dict[str, HostSample]:
    """Parse ``condor_status -af Machine DetectedMemory MemAvailableMB LoadAvg``.

    One row per Condor slot -- a partitionable host with claimed dynamic
    slots emits several rows for the same ``Machine``, but the fields we
    read are machine-wide (STARTD_CRON/native), so they're identical across
    a machine's rows; the first row seen wins. A field the STARTD hasn't
    published yet prints the literal token ``undefined``, which is treated
    as "no data for this host" (dropped), never as ``0``.
    """
    samples: dict[str, HostSample] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) != len(_CONDOR_STATUS_FIELDS) or "undefined" in parts:
            continue
        machine, detected_mem, mem_avail, load = parts
        if machine in samples:
            continue
        try:
            samples[machine] = HostSample(
                hostname=machine,
                mem_available_mb=int(float(mem_avail)),
                mem_total_mb=int(float(detected_mem)),
                load15=float(load),
            )
        except ValueError:
            continue
    return samples


def query_condor_host_samples(*, timeout_s: float = _CONDOR_STATUS_TIMEOUT_S) -> dict[str, HostSample]:
    """Live per-host memory/load, straight from the Condor collector."""
    try:
        proc = subprocess.run(
            ["condor_status", "-af", *_CONDOR_STATUS_FIELDS],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("condor_status timed out after %.0fs", timeout_s)
        return {}
    if proc.returncode != 0:
        log.warning(
            "condor_status failed (exit %d): %s",
            proc.returncode,
            proc.stderr.strip() or proc.stdout.strip(),
        )
        return {}
    return _parse_condor_status_output(proc.stdout)


def evaluate_host(
    sample: HostSample | None,
    *,
    min_mem_mb: int,
    max_load15: float,
) -> tuple[str, ...]:
    if sample is None:
        return ("missing",)
    reasons: list[str] = []
    if sample.mem_available_mb < min_mem_mb:
        reasons.append(f"low mem {sample.mem_available_mb}MB")
    if sample.load15 >= max_load15:
        reasons.append(f"high load15 {sample.load15:.2f}")
    return tuple(reasons)


def plan_host_selection(
    *,
    host_samples: dict[str, HostSample] | None = None,
    min_mem_mb: int,
    max_load15: float,
) -> HostSelection:
    """Split expected hosts into eligible (live data) vs excluded (missing).

    Low-mem/high-load hosts stay ``eligible`` here -- they're filtered out of
    matching by live ``MemAvailableMB``/``LoadAvg`` clauses in
    :func:`apply_host_stats_policy` instead, re-evaluated by the negotiator
    every cycle, not frozen at submit time. Only a host with no live data at
    all (down, decommissioned, or not yet publishing) has no such live
    substitute, so it's the only case that gets a hard ``Machine != ...``
    name exclusion here.
    """
    samples = host_samples if host_samples is not None else query_condor_host_samples()
    excluded: dict[str, tuple[str, ...]] = {}
    eligible: list[HostSample] = []
    for host in expected_hosts():
        sample = samples.get(host)
        if sample is None:
            excluded[host] = evaluate_host(sample, min_mem_mb=min_mem_mb, max_load15=max_load15)
        else:
            eligible.append(sample)
    eligible.sort(key=lambda s: (s.load15, s.hostname))
    usable = bool(samples) and bool(eligible)
    return HostSelection(
        eligible=tuple(eligible),
        excluded=excluded,
        usable=usable,
    )


def build_base_requirements(request_memory_mb: int) -> str:
    return f"Memory >= {int(request_memory_mb)}"


def format_machine_exclusions(hosts: Sequence[str]) -> str:
    if not hosts:
        return "# no exclusions"
    return " && ".join(f'Machine != "{host}"' for host in sorted(hosts))


def apply_host_stats_policy(
    resources: CondorResourceRequest,
    *,
    host_samples: dict[str, HostSample] | None = None,
) -> CondorResourceRequest:
    from syndiff_pipeline.common.orchestration.condor import merge_requirements_with_exclusions

    samples = host_samples if host_samples is not None else query_condor_host_samples()
    base = build_base_requirements(resources.request_memory_mb)
    selection = plan_host_selection(
        host_samples=samples,
        min_mem_mb=resources.host_stats_min_mem_mb,
        max_load15=resources.host_stats_max_load15,
    )
    if not selection.usable:
        log.warning(
            "host_stats: no usable condor_status samples (queried=%d eligible=%d); "
            "using Memory requirement and -LoadAvg rank",
            len(samples),
            len(selection.eligible),
        )
        return replace(
            resources,
            requirements=base,
            rank="-LoadAvg",
        )

    excluded_hosts = set(selection.excluded)
    requirements = merge_requirements_with_exclusions(base, excluded_hosts)
    # Live, continuously-reevaluated clauses using Condor's own STARTD-
    # published MemAvailableMB/LoadAvg (real host memory/load, including
    # non-Condor activity). Unlike the name-based exclusion above, these are
    # not frozen at submit time: the negotiator checks them against each
    # machine's *current* state on every cycle, so an idle job can still
    # match a host once its memory frees up or its load drops, with no
    # resubmission needed.
    requirements = (
        f"({requirements}) && (MemAvailableMB >= {int(resources.host_stats_min_mem_mb)}) "
        f"&& (LoadAvg <= {float(resources.host_stats_max_load15)})"
    )
    # Live rank -- Condor's own LoadAvg, re-evaluated fresh against each
    # candidate machine at match time. No pre-submission preference snapshot:
    # unlike a hostname-weighted rank baked from load15 at submit time, this
    # keeps responding to real conditions for as long as the job is idle.
    rank = "-LoadAvg"
    top = selection.eligible[0]
    log.info(
        "host_stats: %d eligible, %d excluded; top=%s load15=%.2f mem=%dMB",
        len(selection.eligible),
        len(selection.excluded),
        top.hostname,
        top.load15,
        top.mem_available_mb,
    )
    return replace(resources, requirements=requirements, rank=rank)
