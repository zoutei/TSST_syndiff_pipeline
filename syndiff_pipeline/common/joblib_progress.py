"""Optional tqdm wrappers for joblib Parallel workloads."""

from __future__ import annotations

import logging

from joblib import Parallel

log = logging.getLogger(__name__)


def parallel_map_with_optional_tqdm(
    delayed_calls,
    n_tasks: int,
    desc: str,
    n_jobs_eff: int,
    *,
    initializer=None,
    initargs=(),
    on_result=None,
    prefer: str | None = None,
):
    """Run *delayed_calls* with loky; show a tqdm bar when available.

    Always drains via ``return_as="generator"`` so ``on_result`` fires as each
    task completes (NFS-safe parent-only progress updates). Falls back to a
    blocking list collect only if the joblib build rejects ``return_as``.
    """
    parallel_kwargs: dict = {
        "n_jobs": n_jobs_eff,
        "backend": "loky",
    }
    if prefer is not None:
        parallel_kwargs["prefer"] = prefer
    if initializer is not None:
        parallel_kwargs["initializer"] = initializer
        parallel_kwargs["initargs"] = initargs

    def _collect(results_iter):
        out = []
        for item in results_iter:
            if on_result is not None:
                on_result(item)
            out.append(item)
        return out

    try:
        parallel_kwargs["return_as"] = "generator"
        gen = Parallel(**parallel_kwargs)(delayed_calls)
    except TypeError:
        log.debug(
            "joblib Parallel(return_as=...) unavailable; collecting after all tasks finish."
        )
        parallel_kwargs.pop("return_as", None)
        return _collect(Parallel(**parallel_kwargs)(delayed_calls))

    try:
        from tqdm.auto import tqdm

        return _collect(tqdm(gen, total=n_tasks, desc=desc, unit="task"))
    except ImportError:
        return _collect(gen)


def tqdm_iter(tasks: list, desc: str):
    """Iterate *tasks* with tqdm when available."""
    try:
        from tqdm.auto import tqdm

        return tqdm(tasks, desc=desc, unit="frame")
    except ImportError:
        return tasks
