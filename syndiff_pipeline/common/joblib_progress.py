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
):
    """Run *delayed_calls* with loky; show a tqdm bar when available."""
    parallel_kwargs = {
        "n_jobs": n_jobs_eff,
        "backend": "loky",
    }
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
        from tqdm.auto import tqdm
    except ImportError:
        return _collect(Parallel(**parallel_kwargs)(delayed_calls))
    try:
        parallel_kwargs["return_as"] = "generator"
        gen = Parallel(**parallel_kwargs)(delayed_calls)
        return _collect(tqdm(gen, total=n_tasks, desc=desc, unit="frame"))
    except TypeError:
        log.debug("joblib Parallel(return_as=...) unavailable; running without tqdm bar.")
        parallel_kwargs.pop("return_as", None)
        return _collect(Parallel(**parallel_kwargs)(delayed_calls))


def tqdm_iter(tasks: list, desc: str):
    """Iterate *tasks* with tqdm when available."""
    try:
        from tqdm.auto import tqdm

        return tqdm(tasks, desc=desc, unit="frame")
    except ImportError:
        return tasks
