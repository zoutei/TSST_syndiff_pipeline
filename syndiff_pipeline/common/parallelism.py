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

    On Condor execute nodes, ``SYNDIFF_REQUEST_CPUS`` (set in the submit file)
    is the source of truth. Otherwise use *stage_n_jobs* or global *n_jobs*.
    """
    base = int(stage_n_jobs if stage_n_jobs is not None else n_jobs or 1)
    env_raw = os.environ.get("SYNDIFF_REQUEST_CPUS", "").strip()
    if env_raw:
        try:
            base = int(env_raw)
        except ValueError:
            pass
    cpu_cap = multiprocessing.cpu_count() or 1
    return max(1, min(base, cpu_cap))
