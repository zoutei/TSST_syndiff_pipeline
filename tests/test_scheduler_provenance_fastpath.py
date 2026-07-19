"""Tests for the scheduler's provenance fast path (PR3 cutover for ps1_process).

Exercises the private helpers directly (``_ps1_process_provenance_complete``,
``_drain_provenance_spool``) since they are the exact seam the daemon's verify
pass and tick loop call; this mirrors how ``tests/test_verify_worker.py`` tests
scheduler internals in this codebase.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from syndiff_pipeline.common.orchestration import scheduler
from syndiff_pipeline.common.provenance.ingest import drain_data_root
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, provenance_spool_dir
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    emit_ps1_process_checkpoint,
)


def _fake_resolved(data_root, **ps1_overrides):
    ps1_process = SimpleNamespace(
        psf_sigma=60.0,
        enable_saturation_correction=True,
        remove_saturated_stars=False,
        bright_star_mag_threshold=13.0,
        projections_limit=None,
    )
    for k, v in ps1_overrides.items():
        setattr(ps1_process, k, v)
    stages = SimpleNamespace(ps1_process=ps1_process)
    target = SimpleNamespace(sector=20, camera=1, ccd=1)
    return SimpleNamespace(target=target, stages=stages, data_root=str(data_root))


class ProvenanceFastPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Isolate the module-level caches from other tests / real usage.
        scheduler._PROVENANCE_STORE_CACHE.clear()
        scheduler._PROVENANCE_DRAIN_LAST.clear()

    def tearDown(self) -> None:
        scheduler._PROVENANCE_STORE_CACHE.clear()
        scheduler._PROVENANCE_DRAIN_LAST.clear()
        self._tmp.cleanup()

    def test_false_when_no_checkpoint_ever_emitted(self) -> None:
        resolved = _fake_resolved(self.root)
        self.assertFalse(scheduler._ps1_process_provenance_complete(resolved))

    def test_true_after_checkpoint_emitted_and_drained(self) -> None:
        resolved = _fake_resolved(self.root)
        emit_ps1_process_checkpoint(resolved, produced_by="run-1")
        store = ProvenanceStore(provenance_db_path(self.root))
        drain_data_root(store, self.root)
        # scheduler opens its OWN cached store instance for this data_root.
        self.assertTrue(scheduler._ps1_process_provenance_complete(resolved))

    def test_false_after_config_drift(self) -> None:
        resolved = _fake_resolved(self.root)
        emit_ps1_process_checkpoint(resolved, produced_by="run-1")
        store = ProvenanceStore(provenance_db_path(self.root))
        drain_data_root(store, self.root)
        self.assertTrue(scheduler._ps1_process_provenance_complete(resolved))

        drifted = _fake_resolved(self.root, psf_sigma=45.0)
        self.assertFalse(scheduler._ps1_process_provenance_complete(drifted))

    def test_never_raises_on_bad_data_root(self) -> None:
        resolved = _fake_resolved(self.root)
        resolved.data_root = None  # malformed input
        self.assertFalse(scheduler._ps1_process_provenance_complete(resolved))

    def test_drain_provenance_spool_ingests_and_throttles(self) -> None:
        resolved = _fake_resolved(self.root)
        emit_ps1_process_checkpoint(resolved, produced_by="run-1")
        spool_dir = provenance_spool_dir(self.root)
        self.assertEqual(len(list(spool_dir.glob("*.jsonl"))), 1)

        scheduler._drain_provenance_spool(str(self.root))
        self.assertEqual(list(spool_dir.glob("*.jsonl")), [])  # drained away

        # Emit again immediately; throttle window should suppress a second drain.
        emit_ps1_process_checkpoint(resolved, produced_by="run-2")
        scheduler._drain_provenance_spool(str(self.root))
        self.assertEqual(len(list(spool_dir.glob("*.jsonl"))), 1)  # NOT drained (throttled)

        # Force the throttle window to have elapsed; now it drains.
        scheduler._PROVENANCE_DRAIN_LAST[str(self.root)] = time.monotonic() - 3600
        scheduler._drain_provenance_spool(str(self.root))
        self.assertEqual(list(spool_dir.glob("*.jsonl")), [])

    def test_drain_noop_on_missing_data_root(self) -> None:
        scheduler._drain_provenance_spool(None)  # must not raise
        scheduler._drain_provenance_spool("")  # must not raise


if __name__ == "__main__":
    unittest.main()
