"""Worker-count helpers for Condor execute nodes and local runs."""

from __future__ import annotations

import multiprocessing
import os


def resolve_effective_n_jobs(
    n_jobs: int,
    *,
    stage_n_jobs: int | None = None,
) -> int:
    """Return parallel worker count capped by Condor allocation and visible CPUs.

    When *stage_n_jobs* is not given, ``SYNDIFF_REQUEST_CPUS`` (set in the
    Condor submit file) is the source of truth for *n_jobs*. When
    *stage_n_jobs* is given, it is an explicit per-stage override (e.g.
    ``hotpants_n_jobs: 1`` to bound OS4 field-template memory) and
    ``SYNDIFF_REQUEST_CPUS`` may only shrink it further, never widen it back
    out to the full Condor allocation.
    """
    env_raw = os.environ.get("SYNDIFF_REQUEST_CPUS", "").strip()
    env_cpus: int | None = None
    if env_raw:
        try:
            env_cpus = int(env_raw)
        except ValueError:
            env_cpus = None

    if stage_n_jobs is not None:
        base = int(stage_n_jobs)
        if env_cpus is not None:
            base = min(base, env_cpus)
    else:
        base = int(n_jobs or 1)
        if env_cpus is not None:
            base = env_cpus
    cpu_cap = multiprocessing.cpu_count() or 1
    return max(1, min(base, cpu_cap))
