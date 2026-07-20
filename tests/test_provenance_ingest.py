"""Tests for ``common/provenance/ingest.py``: spool rotate/drain idempotency."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.ingest import drain_spool, rotate_spool_files
from syndiff_pipeline.common.provenance.publish import append_spool_record, build_record
from syndiff_pipeline.common.provenance.store import ProvenanceStore


class _TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.spool_dir = self.tmp / "bookkeeping" / "spool"
        self.store = ProvenanceStore(self.tmp / "bookkeeping" / "provenance.db")

    def _write_record(self, fp: str, *, inputs=(), recipe_params=None):
        rec = build_record(
            fp, "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, inputs, f"loc-{fp}",
            recipe_params=recipe_params,
        )
        append_spool_record(self.spool_dir, rec)
        return rec


class TestRotate(_TempCase):
    def test_rotate_renames_jsonl_to_draining(self):
        self._write_record("fp1")
        rotated = rotate_spool_files(self.spool_dir)
        self.assertEqual(len(rotated), 1)
        self.assertTrue(rotated[0].name.endswith(".jsonl.draining"))
        self.assertFalse(list(self.spool_dir.glob("*.jsonl")))

    def test_rotate_picks_up_leftover_draining_file(self):
        self._write_record("fp1")
        first = rotate_spool_files(self.spool_dir)
        self.assertEqual(len(first), 1)
        # Simulate a crash between rotate and delete: the .draining file is
        # still there on the next pass.
        second = rotate_spool_files(self.spool_dir)
        self.assertEqual(second, first)

    def test_rotate_on_empty_dir_is_a_noop(self):
        self.assertEqual(rotate_spool_files(self.spool_dir), [])
        self.assertEqual(rotate_spool_files(self.tmp / "does-not-exist"), [])


class TestDrainSpool(_TempCase):
    def test_drain_ingests_records_and_deletes_spool_file(self):
        self._write_record("fp1", recipe_params={"a": 1})
        result = drain_spool(self.store, self.spool_dir)
        self.assertEqual(result.files_drained, 1)
        self.assertEqual(result.records_ingested, 1)
        self.assertEqual(result.errors, [])
        self.assertEqual(list(self.spool_dir.glob("*")), [])

        row = self.store.artifact("fp1")
        self.assertIsNotNone(row)
        self.assertEqual(row.location, "loc-fp1")
        recipe = self.store.recipe("rid1")
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.params, {"a": 1})

    def test_drain_ingests_edges(self):
        self._write_record("fp1", inputs=["in1", "in2"])
        drain_spool(self.store, self.spool_dir)
        self.assertEqual(sorted(self.store.inputs_of("fp1")), ["in1", "in2"])

    def test_drain_multiple_records_one_file(self):
        self._write_record("fp1")
        self._write_record("fp2")
        self._write_record("fp3")
        result = drain_spool(self.store, self.spool_dir)
        self.assertEqual(result.records_ingested, 3)
        for fp in ("fp1", "fp2", "fp3"):
            self.assertIsNotNone(self.store.artifact(fp))

    def test_drain_is_idempotent_when_called_repeatedly(self):
        self._write_record("fp1")
        r1 = drain_spool(self.store, self.spool_dir)
        r2 = drain_spool(self.store, self.spool_dir)  # nothing left to drain
        self.assertEqual(r1.records_ingested, 1)
        self.assertEqual(r2.records_ingested, 0)
        self.assertEqual(r2.files_drained, 0)
        self.assertIsNotNone(self.store.artifact("fp1"))

    def test_reingesting_a_draining_file_after_simulated_crash_is_safe(self):
        """Simulate the crash-between-rotate-and-delete window: rotate
        manually, ingest is interrupted (we just don't delete), then a
        second drain pass must find the .draining file and finish cleanly
        without duplicating or corrupting anything."""
        self._write_record("fp1")
        rotated = rotate_spool_files(self.spool_dir)
        self.assertEqual(len(rotated), 1)
        # File is now {host}.{pid}.jsonl.draining and untouched by any ingest.
        result = drain_spool(self.store, self.spool_dir)
        self.assertEqual(result.files_drained, 1)
        self.assertEqual(result.records_ingested, 1)
        self.assertIsNotNone(self.store.artifact("fp1"))
        self.assertEqual(list(self.spool_dir.glob("*")), [])

    def test_malformed_line_is_skipped_not_fatal(self):
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        from syndiff_pipeline.common.provenance.publish import default_spool_path

        path = default_spool_path(self.spool_dir)
        path.write_text('{"not": "a valid record"}\nnot-json-at-all\n')
        # Rotate this hand-crafted file in ourselves (bypass append_spool_record).
        result = drain_spool(self.store, self.spool_dir)
        self.assertEqual(result.files_drained, 1)
        self.assertEqual(result.records_ingested, 0)
        self.assertGreaterEqual(result.records_skipped, 1)

    def test_recipe_shared_across_two_artifacts_is_insert_or_ignore(self):
        rec1 = build_record(
            "fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid-shared", 1, [], "loc1",
            recipe_params={"a": 1},
        )
        rec2 = build_record(
            "fp2", "mapping", {"s": 1, "c": 1, "k": 2}, "rid-shared", 1, [], "loc2",
            recipe_params={"a": 1},
        )
        append_spool_record(self.spool_dir, rec1)
        append_spool_record(self.spool_dir, rec2)
        result = drain_spool(self.store, self.spool_dir)
        self.assertEqual(result.records_ingested, 2)
        recipe = self.store.recipe("rid-shared")
        self.assertEqual(recipe.params, {"a": 1})
        self.assertIsNotNone(self.store.artifact("fp1"))
        self.assertIsNotNone(self.store.artifact("fp2"))


if __name__ == "__main__":
    unittest.main()
