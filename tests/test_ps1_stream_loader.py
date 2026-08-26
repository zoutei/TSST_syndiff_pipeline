"""Deterministic tests for the bounded PS1 stream skycell loader.

These tests do not contact PS1.  The component fetch is replaced with a
blocking in-memory function, while the executor's ``submit`` method records
which skycell each component belongs to.  That lets us exercise the two
different bounds independently: request workers (component futures) and
admitted complete skycells (prefetch slots).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import ps1_download as ps1


_COMPONENTS_PER_CELL = 12


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for loader activity")
        time.sleep(0.005)


def _cell_from_submit_callable(fn) -> str:
    """Extract the cell captured by ``_fetch_skycell_with_executor``'s closure."""
    for closure_cell in getattr(fn, "__closure__", ()) or ():
        value = closure_cell.cell_contents
        if isinstance(value, str) and value.startswith("skycell."):
            return value
    raise AssertionError("component task did not capture a skycell id")


class _BlockingFetch:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.active = 0
        self.peak_active = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, *_args, **_kwargs):
        with self._lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.started.set()
        try:
            # The timeout keeps a failed test from leaking a worker forever.
            self.release.wait(timeout=5)
        finally:
            with self._lock:
                self.active -= 1
        return {"data": np.zeros((1, 1), dtype=np.float32), "header": ""}


def _record_submissions(loader: ps1.StreamSkycellLoader) -> list[str]:
    submitted: list[str] = []
    submit_lock = threading.Lock()
    original_submit = loader._executor.submit

    def submit(fn, *args, **kwargs):
        with submit_lock:
            submitted.append(_cell_from_submit_callable(fn))
        return original_submit(fn, *args, **kwargs)

    # This is deliberately scoped to this run-scoped executor instance.
    loader._executor.submit = submit
    return submitted


def _run_fetch(loader: ps1.StreamSkycellLoader, cell: str, errors: list[BaseException]) -> None:
    try:
        loader.fetch_skycell_bands_masks_and_headers(cell)
    except BaseException as exc:  # report worker failures in the test thread
        errors.append(exc)


def test_stream_loader_caps_component_requests_and_keeps_cell_batches_fifo(monkeypatch):
    blocking_fetch = _BlockingFetch()
    monkeypatch.setattr(ps1, "download_and_process_band", blocking_fetch)
    loader = ps1.StreamSkycellLoader(max_inflight_requests=24, prefetch_cells=6)
    submitted = _record_submissions(loader)
    errors: list[BaseException] = []
    cells = [f"skycell.1234.{i:03d}" for i in range(6)]
    threads: list[threading.Thread] = []

    try:
        # Admit cells one at a time so the expected FIFO order is unambiguous;
        # each admission submits exactly the twelve FITS components as a batch.
        for index, cell in enumerate(cells, start=1):
            thread = threading.Thread(target=_run_fetch, args=(loader, cell, errors))
            thread.start()
            threads.append(thread)
            _wait_until(lambda n=index * _COMPONENTS_PER_CELL: len(submitted) >= n)

        _wait_until(lambda: blocking_fetch.peak_active >= 24)
        assert loader.effective_inflight_requests == 24
        assert blocking_fetch.peak_active <= 24
        assert submitted == [cell for cell in cells for _ in range(_COMPONENTS_PER_CELL)]
    finally:
        blocking_fetch.release.set()
        for thread in threads:
            thread.join(timeout=5)
        loader.close()

    assert not errors


def test_stream_loader_prefetch_slots_bound_admitted_cells(monkeypatch):
    blocking_fetch = _BlockingFetch()
    monkeypatch.setattr(ps1, "download_and_process_band", blocking_fetch)
    loader = ps1.StreamSkycellLoader(max_inflight_requests=24, prefetch_cells=2)
    submitted = _record_submissions(loader)
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []

    try:
        first = "skycell.1234.000"
        second = "skycell.1234.001"
        third = "skycell.1234.002"
        for cell, count in ((first, 12), (second, 24)):
            thread = threading.Thread(target=_run_fetch, args=(loader, cell, errors))
            thread.start()
            threads.append(thread)
            _wait_until(lambda n=count: len(submitted) >= n)

        third_thread = threading.Thread(target=_run_fetch, args=(loader, third, errors))
        third_thread.start()
        threads.append(third_thread)

        # Both complete-cell slots are occupied by blocked fetches.  The
        # third cell must not submit even one component until a slot frees.
        time.sleep(0.05)
        assert submitted == [first] * 12 + [second] * 12
        assert third_thread.is_alive()

        blocking_fetch.release.set()
        for thread in threads:
            thread.join(timeout=5)
        assert submitted == [first] * 12 + [second] * 12 + [third] * 12
    finally:
        blocking_fetch.release.set()
        for thread in threads:
            thread.join(timeout=5)
        loader.close()

    assert not errors


def test_stream_loader_close_drains_and_is_idempotent(monkeypatch):
    blocking_fetch = _BlockingFetch()
    monkeypatch.setattr(ps1, "download_and_process_band", blocking_fetch)
    loader = ps1.StreamSkycellLoader(max_inflight_requests=12, prefetch_cells=1)
    submitted = _record_submissions(loader)
    errors: list[BaseException] = []
    fetch_thread = threading.Thread(
        target=_run_fetch, args=(loader, "skycell.1234.000", errors)
    )
    fetch_thread.start()
    _wait_until(lambda: len(submitted) == _COMPONENTS_PER_CELL)

    close_done = threading.Event()

    def close_loader() -> None:
        loader.close()
        close_done.set()

    close_thread = threading.Thread(target=close_loader)
    close_thread.start()
    time.sleep(0.05)
    assert not close_done.is_set(), "close must drain outstanding component work"

    blocking_fetch.release.set()
    fetch_thread.join(timeout=5)
    close_thread.join(timeout=5)
    assert close_done.is_set()
    assert not errors

    # Repeated cleanup is safe, but using a closed loader is an explicit error.
    loader.close()
    with pytest.raises(RuntimeError, match="closed"):
        loader.fetch_skycell_bands_masks_and_headers("skycell.1234.001")
