"""Regression test for a padding_source-then-cache-hit cell never publishing
its combined_skycell record.

Background: ``process_coordinator`` handles three result task types.
``regular`` (freshly computed) results call ``_publish_combined`` to write
the raw ``combined_skycell`` record used by the shared convolved-store
publish path. ``padding_source`` results only populate ``band_cache`` (by
design -- they may never be needed as this SCC's own primary cell).
``regular_cache_hit`` results -- a cell first fetched as a padding_source,
later needed as this SCC's own primary row cell -- reused the cached image
via the fast path but never called ``_publish_combined`` either, silently
leaving a gap: ``_publish_canonical_convolved_snapshot``'s
``cell_dir/arrays.npz`` existence guard then (correctly, per its own
contract) skips the cell forever, since no record was ever written for it.

Confirmed in production: every CVZ SCC using ``use_shared_convolved_store``
had ~4.6-4.9% of skycells silently missing from the shared store this way,
uniformly, including SCCs whose row-assembly (a separate, already-fixed
bug) reported zero skipped skycells.
"""
from __future__ import annotations

import queue as _thread_queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from syndiff_pipeline.template_creation.processing.ps1_process import (
    process_coordinator,
)


class CacheHitPublishesCombinedRecordTests(unittest.TestCase):
    def test_regular_cache_hit_calls_publish_combined(self):
        # band_cache pre-populated exactly as the padding_source completion
        # branch would leave it -- avoids routing a fake bundle through the
        # real ProcessPoolExecutor/process_single_cell path, which needs a
        # fully-shaped bundle (WCS headers, PS1 arrays, ...) unrelated to
        # what this regression actually targets.
        combined_raw_queue: _thread_queue.Queue = _thread_queue.Queue()
        combined_cell_queue: _thread_queue.Queue = _thread_queue.Queue()

        image = np.ones((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.int32)
        band_cache = {
            "skycell.9999.001": {
                "combined_image": image,
                "combined_mask": mask,
                "headers_data": {},
                "removed_stars": [],
            }
        }

        cache_hit_bundle = {
            "task_type": "regular_cache_hit",
            "skycell_id": "skycell.9999.001",
            "projection": "skycell.9999",
            "row_id": 0,
            "x_coord": 0,
        }

        combined_raw_queue.put(cache_hit_bundle)
        combined_raw_queue.put(None)  # shutdown sentinel

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "syndiff_pipeline.template_creation.processing.combined_store.publish_combined_cell"
            ) as mock_publish:
                process_coordinator(
                    combined_raw_queue,
                    combined_cell_queue,
                    cell_buffer={},
                    num_workers=1,
                    band_cache=band_cache,
                    combined_store_data_root=tmp,
                    combined_store_recipe={"kind": "combined_skycell"},
                )

        mock_publish.assert_called_once()
        _, kwargs = mock_publish.call_args
        self.assertIs(kwargs["combined_image"], image)

    def test_cache_hit_without_band_cache_entry_does_not_publish(self):
        # Defensive: if band_cache somehow doesn't have the entry (the
        # existing "Expected cache hit ... not found; dropping" warning
        # path), nothing should be published either.
        combined_raw_queue: _thread_queue.Queue = _thread_queue.Queue()
        combined_cell_queue: _thread_queue.Queue = _thread_queue.Queue()

        cache_hit_bundle = {
            "task_type": "regular_cache_hit",
            "skycell_id": "skycell.9999.002",
            "projection": "skycell.9999",
            "row_id": 0,
            "x_coord": 0,
        }
        combined_raw_queue.put(cache_hit_bundle)
        combined_raw_queue.put(None)

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "syndiff_pipeline.template_creation.processing.combined_store.publish_combined_cell"
            ) as mock_publish:
                process_coordinator(
                    combined_raw_queue,
                    combined_cell_queue,
                    cell_buffer={},
                    num_workers=1,
                    band_cache={},
                    combined_store_data_root=tmp,
                    combined_store_recipe={"kind": "combined_skycell"},
                )

        mock_publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
