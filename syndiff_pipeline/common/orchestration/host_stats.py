"""Cluster host sampler JSON → Condor machine selection at submit time."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from syndiff_pipeline.common.orchestration.condor import CondorResourceRequest

log = logging.getLogger(__name__)

_DEFAULT_STATS_DIR = "/home/kshukawa/.syndiff/host_stats"
DEFAULT_MAX_AGE_S = 300


def default_stats_dir() -> Path:
    return Path(os.environ.get("HOST_STATS_DIR", _DEFAULT_STATS_DIR))


# Back-compat alias for CLI default=...
DEFAULT_STATS_DIR = default_stats_dir()


@dataclass(frozen=True)
class HostSample:
    """One host heartbeat from the cluster sampler."""

    hostname: str
    login_hostname: str | None
    timestamp: int
    mem_available_mb: int
    mem_total_mb: int
    load1: float
    load5: float
    load15: float
    path: Path

    @property
    def age_s(self) -> int:
        return max(0, int(time.time()) - self.timestamp)


@dataclass(frozen=True)
class HostSelection:
    """Result of evaluating all expected hosts against thresholds."""

    eligible: tuple[HostSample, ...]
    excluded: dict[str, tuple[str, ...]]
    usable: bool


def normalize_condor_host(host: str) -> str:
    text = str(host or "").strip()
    if "@" in text:
        text = text.rsplit("@", 1)[-1]
    match = re.fullmatch(r"science(\d+)(?:\.stsci\.edu)?", text)
    if match:
        return f"plscience{match.group(1)}.stsci.edu"
    return text


def load_sample(path: Path) -> HostSample:
    payload = json.loads(path.read_text(encoding="utf-8"))
    hostname = normalize_condor_host(str(payload.get("hostname", path.stem)))
    login_hostname = payload.get("login_hostname")
    return HostSample(
        hostname=hostname,
        login_hostname=str(login_hostname) if login_hostname else None,
        timestamp=int(payload["timestamp"]),
        mem_available_mb=int(payload["mem_available_mb"]),
        mem_total_mb=int(payload["mem_total_mb"]),
        load1=float(payload["load1"]),
        load5=float(payload["load5"]),
        load15=float(payload["load15"]),
        path=path,
    )


def discover_samples(stats_dir: Path) -> dict[str, HostSample]:
    samples: dict[str, HostSample] = {}
    if not stats_dir.is_dir():
        return samples
    for path in sorted(stats_dir.glob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        try:
            sample = load_sample(path)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        samples[sample.hostname] = sample
    return samples


def expected_hosts() -> list[str]:
    return [f"plscience{n}.stsci.edu" for n in range(1, 16)]


def evaluate_host(
    sample: HostSample | None,
    *,
    min_mem_mb: int,
    max_load15: float,
    max_age_s: int,
) -> tuple[str, ...]:
    if sample is None:
        return ("missing",)
    reasons: list[str] = []
    if sample.age_s > max_age_s:
        reasons.append(f"stale {sample.age_s}s")
    if sample.mem_available_mb < min_mem_mb:
        reasons.append(f"low mem {sample.mem_available_mb}MB")
    if sample.load15 >= max_load15:
        reasons.append(f"high load15 {sample.load15:.2f}")
    return tuple(reasons)


def plan_host_selection(
    *,
    stats_dir: Path,
    min_mem_mb: int,
    max_load15: float,
    max_age_s: int = DEFAULT_MAX_AGE_S,
) -> HostSelection:
    samples = discover_samples(stats_dir)
    excluded: dict[str, tuple[str, ...]] = {}
    eligible: list[HostSample] = []
    for host in expected_hosts():
        sample = samples.get(host)
        reasons = evaluate_host(
            sample,
            min_mem_mb=min_mem_mb,
            max_load15=max_load15,
            max_age_s=max_age_s,
        )
        if reasons:
            excluded[host] = reasons
        elif sample is not None:
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


def build_load15_rank(eligible: Sequence[HostSample]) -> str | None:
    if not eligible:
        return None
    n = len(eligible)
    terms: list[str] = []
    for idx, sample in enumerate(eligible):
        weight = n - idx
        terms.append(f'(Machine == "{sample.hostname}") * {weight}')
    return " + ".join(terms)


def format_machine_exclusions(hosts: Sequence[str]) -> str:
    if not hosts:
        return "# no exclusions"
    return " && ".join(f'Machine != "{host}"' for host in sorted(hosts))


def apply_host_stats_policy(
    resources: CondorResourceRequest,
    *,
    stats_dir: Path | None = None,
    max_age_s: int = DEFAULT_MAX_AGE_S,
) -> CondorResourceRequest:
    from syndiff_pipeline.common.orchestration.condor import merge_requirements_with_exclusions

    stats_dir = stats_dir or default_stats_dir()
    base = build_base_requirements(resources.request_memory_mb)
    selection = plan_host_selection(
        stats_dir=stats_dir,
        min_mem_mb=resources.host_stats_min_mem_mb,
        max_load15=resources.host_stats_max_load15,
        max_age_s=max_age_s,
    )
    if not selection.usable:
        log.warning(
            "host_stats: no usable samples in %s (json=%d eligible=%d); "
            "using Memory requirement and -LoadAvg rank",
            stats_dir,
            len(discover_samples(stats_dir)),
            len(selection.eligible),
        )
        return replace(
            resources,
            requirements=base,
            rank="-LoadAvg",
        )

    excluded_hosts = set(selection.excluded)
    requirements = merge_requirements_with_exclusions(base, excluded_hosts)
    rank = build_load15_rank(selection.eligible)
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
