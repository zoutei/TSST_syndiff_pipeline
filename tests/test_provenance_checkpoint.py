"""Tests for the ps1_process provenance checkpoint (PR2/PR3 glue).

Uses a lightweight fake ``resolved`` (SimpleNamespace), mirroring
``tests/test_provenance_model.py``'s fake, extended with ``projections_limit``
and ``data_root`` which ``provenance_checkpoint`` additionally reads.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from syndiff_pipeline.common.provenance.ingest import drain_data_root
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, scc_convolved_zarr
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    emit_ps1_process_checkpoint,
    ps1_process_checkpoint_record,
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


class CheckpointRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_record_shape_and_location(self) -> None:
        resolved = _fake_resolved(self.root)
        record = ps1_process_checkpoint_record(resolved)
        self.assertEqual(record["kind"], "scc_assembly")
        self.assertEqual(record["state"], "complete")
        self.assertEqual(
            record["location"], str(scc_convolved_zarr(self.root, 20, 1, 1))
        )
        self.assertEqual(record["spatial_key"], {"sector": 20, "camera": 1, "ccd": 1})
        self.assertIn("psf_sigma", record["recipe"]["params"])
        self.assertIn("bright_star_mag_threshold", record["recipe"]["params"])
        self.assertIn("projections_limit", record["recipe"]["params"])

    def test_deterministic_fingerprint(self) -> None:
        r1 = ps1_process_checkpoint_record(_fake_resolved(self.root))
        r2 = ps1_process_checkpoint_record(_fake_resolved(self.root))
        self.assertEqual(r1["fingerprint"], r2["fingerprint"])

    def test_config_change_changes_fingerprint(self) -> None:
        base = ps1_process_checkpoint_record(_fake_resolved(self.root))
        changed = ps1_process_checkpoint_record(
            _fake_resolved(self.root, psf_sigma=45.0)
        )
        self.assertNotEqual(base["fingerprint"], changed["fingerprint"])

    def test_different_scc_changes_fingerprint(self) -> None:
        r1 = ps1_process_checkpoint_record(_fake_resolved(self.root))
        resolved2 = _fake_resolved(self.root)
        resolved2.target.ccd = 2
        r2 = ps1_process_checkpoint_record(resolved2)
        self.assertNotEqual(r1["fingerprint"], r2["fingerprint"])


class EmitAndIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_emit_then_drain_makes_store_report_complete(self) -> None:
        resolved = _fake_resolved(self.root)
        expected_fp = ps1_process_checkpoint_record(resolved)["fingerprint"]

        store = ProvenanceStore(provenance_db_path(self.root))
        self.assertFalse(store.scc_stage_complete([expected_fp]))

        emit_ps1_process_checkpoint(resolved, produced_by="run-123")
        drained = drain_data_root(store, self.root)
        self.assertEqual(drained, 1)
        self.assertTrue(store.scc_stage_complete([expected_fp]))

        art = store.get_artifact(expected_fp)
        self.assertEqual(art["produced_by"], "run-123")

    def test_stale_checkpoint_not_complete_after_config_change(self) -> None:
        resolved = _fake_resolved(self.root)
        emit_ps1_process_checkpoint(resolved, produced_by="run-1")
        store = ProvenanceStore(provenance_db_path(self.root))
        drain_data_root(store, self.root)

        # Simulate a config change: a different psf_sigma expects a DIFFERENT
        # fingerprint, which the store has never seen -> not complete. This is
        # the Merkle-invalidation guarantee the scheduler fast path relies on.
        changed = _fake_resolved(self.root, psf_sigma=45.0)
        changed_fp = ps1_process_checkpoint_record(changed)["fingerprint"]
        self.assertFalse(store.scc_stage_complete([changed_fp]))


if __name__ == "__main__":
    unittest.main()
