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
    chunk_size: int | None = None,
):
    """Run *delayed_calls* with loky; show a tqdm bar when available.

    Always drains via ``return_as="generator"`` so ``on_result`` fires as each
    task completes (NFS-safe parent-only progress updates). Falls back to a
    blocking list collect only if the joblib build rejects ``return_as``.

    ``chunk_size``, when set, restarts the loky executor every *chunk_size*
    tasks. Long-running per-task native code (e.g. pyhotpants) can leak
    memory inside a worker process across many tasks; periodically recycling
    the whole pool bounds that growth on a predictable schedule instead of
    relying on joblib's crash-and-respawn-under-duress path, which re-runs
    the (possibly expensive) initializer chaotically as more workers die.
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

    calls = list(delayed_calls)
    step = chunk_size if chunk_size else (len(calls) or 1)
    chunks = [calls[i : i + step] for i in range(0, len(calls), step)] or [[]]

    try:
        from tqdm.auto import tqdm

        bar = tqdm(total=n_tasks, desc=desc, unit="task")
    except ImportError:
        bar = None

    out: list = []
    try:
        for chunk in chunks:
            kwargs = dict(parallel_kwargs)
            try:
                kwargs["return_as"] = "generator"
                gen = Parallel(**kwargs)(chunk)
            except TypeError:
                log.debug(
                    "joblib Parallel(return_as=...) unavailable; collecting "
                    "after each chunk finishes."
                )
                kwargs.pop("return_as", None)
                gen = Parallel(**kwargs)(chunk)
            for item in gen:
                if on_result is not None:
                    on_result(item)
                out.append(item)
                if bar is not None:
                    bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return out


def tqdm_iter(tasks: list, desc: str):
    """Iterate *tasks* with tqdm when available."""
    try:
        from tqdm.auto import tqdm

        return tqdm(tasks, desc=desc, unit="frame")
    except ImportError:
        return tasks
