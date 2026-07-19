"""Unit tests for the provenance SQLite store (stdlib unittest, tempfile DBs).

These exercise the derived-index contract in isolation from the rest of the
provenance package: schema idempotency, idempotent ingest, the hot-path
completeness query (including >999-fingerprint chunking and its
no-filesystem-access perf contract), edge replacement, recipe round-trips,
and input-file upsert semantics.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.provenance.store import ProvenanceStore


def _record(
    fp: str,
    *,
    kind: str = "convolved_skycell",
    recipe_id: str = "rid_default",
    params: dict | None = None,
    inputs: list[str] | None = None,
    state: str = "complete",
    spatial_key: dict | None = None,
    location: str | None = None,
    bytes: int | None = None,
) -> dict:
    """Build a well-formed ingest record with sensible defaults."""
    return {
        "fingerprint": fp,
        "kind": kind,
        "spatial_key": spatial_key
        if spatial_key is not None
        else {"projection": 1234, "skycell": 56},
        "recipe_id": recipe_id,
        "recipe": {
            "kind": kind,
            "params": params if params is not None else {"psf_sigma": 1.5, "radius": 470},
            "code_version": "1",
            "git_sha": "abc123",
        },
        "inputs": inputs if inputs is not None else [],
        "location": location,
        "state": state,
        "bytes": bytes,
        "wall_time_s": None,
        "produced_by": "run_x/host_y",
        "created_at": None,
    }


class ProvenanceStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "provenance.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def test_schema_init_idempotent(self) -> None:
        store1 = ProvenanceStore(self.db_path)
        # Opening a second time against the same DB must not raise / clobber.
        store2 = ProvenanceStore(self.db_path)
        # Both usable.
        self.assertEqual(store1.ingest_records([_record("fp1")]), 1)
        self.assertIsNotNone(store2.get_artifact("fp1"))

    def test_init_creates_parent_and_expands(self) -> None:
        nested = str(Path(self._tmp.name) / "a" / "b" / "provenance.db")
        store = ProvenanceStore(nested)
        self.assertTrue(Path(nested).parent.is_dir())
        self.assertEqual(store.scc_stage_complete([]), True)

    # ------------------------------------------------------------------
    # Ingest idempotency
    # ------------------------------------------------------------------
    def test_ingest_idempotent_single_row_and_recipe_dedup(self) -> None:
        store = ProvenanceStore(self.db_path)
        rec = _record("fp1", recipe_id="rid1")
        self.assertEqual(store.ingest_records([rec]), 1)
        # Ingest the identical record again.
        self.assertEqual(store.ingest_records([rec]), 1)

        # Exactly one artifact row.
        self.assertEqual(len(list(store.iter_artifacts())), 1)
        # Recipe deduped (INSERT OR IGNORE) — one row, params intact.
        recipe = store.get_recipe("rid1")
        self.assertIsNotNone(recipe)
        self.assertEqual(store.artifacts_by_recipe("rid1")[0]["fingerprint"], "fp1")

    def test_ingest_empty_batch(self) -> None:
        store = ProvenanceStore(self.db_path)
        self.assertEqual(store.ingest_records([]), 0)

    def test_ingest_replaces_state_and_fields(self) -> None:
        store = ProvenanceStore(self.db_path)
        store.ingest_records([_record("fp1", state="building")])
        self.assertEqual(store.get_artifact("fp1")["state"], "building")
        # Re-publish complete with a location.
        store.ingest_records([_record("fp1", state="complete", location="p/s/fp1")])
        art = store.get_artifact("fp1")
        self.assertEqual(art["state"], "complete")
        self.assertEqual(art["location"], "p/s/fp1")
        self.assertEqual(len(list(store.iter_artifacts())), 1)

    # ------------------------------------------------------------------
    # scc_stage_complete
    # ------------------------------------------------------------------
    def test_scc_stage_complete_all_vs_missing_vs_building(self) -> None:
        store = ProvenanceStore(self.db_path)
        store.ingest_records(
            [
                _record("a", state="complete"),
                _record("b", state="complete"),
                _record("c", state="building"),
            ]
        )
        # All complete -> True.
        self.assertTrue(store.scc_stage_complete(["a", "b"]))
        # One missing (never ingested) -> False.
        self.assertFalse(store.scc_stage_complete(["a", "b", "z"]))
        # One present but only 'building' -> False.
        self.assertFalse(store.scc_stage_complete(["a", "c"]))
        # Empty required set is vacuously complete.
        self.assertTrue(store.scc_stage_complete([]))

    def test_scc_stage_complete_no_filesystem_access(self) -> None:
        """Perf contract: completeness is pure-DB — works on fingerprints whose
        bytes were never written, and must not stat/scandir the store."""
        store = ProvenanceStore(self.db_path)
        # Records with location=None / bytes=None: nothing on disk.
        store.ingest_records(
            [_record("nofile1", location=None, bytes=None), _record("nofile2")]
        )

        # Fault-inject: any filesystem walk during the query must blow up.
        orig_scandir, orig_stat, orig_listdir = os.scandir, os.stat, os.listdir

        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError("scc_stage_complete touched the filesystem")

        os.scandir, os.stat, os.listdir = _boom, _boom, _boom
        try:
            self.assertTrue(store.scc_stage_complete(["nofile1", "nofile2"]))
            self.assertFalse(store.scc_stage_complete(["nofile1", "absent"]))
        finally:
            os.scandir, os.stat, os.listdir = orig_scandir, orig_stat, orig_listdir

    def test_scc_stage_complete_chunking_over_999(self) -> None:
        store = ProvenanceStore(self.db_path)
        n = 2500  # spans multiple 900-var chunks
        fps = [f"fp{i:05d}" for i in range(n)]
        store.ingest_records([_record(fp) for fp in fps])
        # Every one complete -> True across chunks.
        self.assertTrue(store.scc_stage_complete(fps))
        # Add one absent fingerprint anywhere -> False.
        self.assertFalse(store.scc_stage_complete(fps + ["absent"]))
        # Duplicates in the required set must not break the count.
        self.assertTrue(store.scc_stage_complete(fps + fps[:10]))

    # ------------------------------------------------------------------
    # missing_fingerprints
    # ------------------------------------------------------------------
    def test_missing_fingerprints_returns_exactly_absent(self) -> None:
        store = ProvenanceStore(self.db_path)
        store.ingest_records(
            [
                _record("a", state="complete"),
                _record("b", state="complete"),
                _record("c", state="building"),  # not complete -> missing
            ]
        )
        missing = store.missing_fingerprints(["a", "b", "c", "d"])
        # 'c' is only building, 'd' never ingested.
        self.assertEqual(missing, ["c", "d"])
        # Order preserved from the required list.
        self.assertEqual(store.missing_fingerprints(["d", "a", "c"]), ["d", "c"])
        # Nothing missing.
        self.assertEqual(store.missing_fingerprints(["a", "b"]), [])

    def test_missing_fingerprints_chunking_over_999(self) -> None:
        store = ProvenanceStore(self.db_path)
        fps = [f"fp{i:05d}" for i in range(2500)]
        store.ingest_records([_record(fp) for fp in fps[:-3]])  # last 3 absent
        self.assertEqual(store.missing_fingerprints(fps), fps[-3:])

    # ------------------------------------------------------------------
    # recipes
    # ------------------------------------------------------------------
    def test_get_recipe_round_trips_params(self) -> None:
        store = ProvenanceStore(self.db_path)
        params = {"z": 1, "a": {"nested": [3, 2, 1]}, "psf_sigma": 1.25}
        store.ingest_records([_record("fp1", recipe_id="rid1", params=params)])
        recipe = store.get_recipe("rid1")
        self.assertEqual(recipe["params"], params)
        self.assertEqual(recipe["kind"], "convolved_skycell")
        self.assertEqual(recipe["code_version"], "1")
        self.assertEqual(recipe["git_sha"], "abc123")
        # params_json is canonical (sorted keys).
        self.assertEqual(recipe["params_json"], '{"a":{"nested":[3,2,1]},"psf_sigma":1.25,"z":1}')

    def test_get_recipe_absent(self) -> None:
        store = ProvenanceStore(self.db_path)
        self.assertIsNone(store.get_recipe("nope"))

    # ------------------------------------------------------------------
    # edges
    # ------------------------------------------------------------------
    def test_edges_stored_and_replaced(self) -> None:
        store = ProvenanceStore(self.db_path)
        store.ingest_records([_record("child", inputs=["p1", "p2", "p3"])])
        self.assertEqual(store.artifact_inputs("child"), ["p1", "p2", "p3"])

        # Re-ingest with a different input set -> edges replaced, not merged.
        store.ingest_records([_record("child", inputs=["p2", "p4"])])
        self.assertEqual(store.artifact_inputs("child"), ["p2", "p4"])

        # Re-ingest with duplicate inputs -> deduped.
        store.ingest_records([_record("child", inputs=["p5", "p5", "p6"])])
        self.assertEqual(store.artifact_inputs("child"), ["p5", "p6"])

    # ------------------------------------------------------------------
    # get_artifact / iter_artifacts
    # ------------------------------------------------------------------
    def test_get_artifact_decodes_spatial_key(self) -> None:
        store = ProvenanceStore(self.db_path)
        sk = {"projection": 2000, "skycell": 42}
        store.ingest_records([_record("fp1", spatial_key=sk)])
        art = store.get_artifact("fp1")
        self.assertEqual(art["spatial_key"], sk)
        self.assertIsNone(store.get_artifact("absent"))

    def test_iter_artifacts_filter_by_kind(self) -> None:
        store = ProvenanceStore(self.db_path)
        store.ingest_records(
            [
                _record("a", kind="combined_skycell"),
                _record("b", kind="convolved_skycell"),
                _record("c", kind="combined_skycell"),
            ]
        )
        combined = [a["fingerprint"] for a in store.iter_artifacts("combined_skycell")]
        self.assertEqual(combined, ["a", "c"])
        self.assertEqual(len(list(store.iter_artifacts())), 3)

    # ------------------------------------------------------------------
    # input_files
    # ------------------------------------------------------------------
    def test_upsert_input_file_upsert_semantics(self) -> None:
        store = ProvenanceStore(self.db_path)
        store.upsert_input_file(
            "raw_skycell",
            "1234.056",
            {"projection": 1234, "skycell": 56},
            bytes=100,
            mtime="2026-01-01T00:00:00",
            batch_id="batchA",
        )
        rows = store.list_input_files("raw_skycell")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bytes"], 100)
        self.assertEqual(rows[0]["batch_id"], "batchA")
        self.assertEqual(rows[0]["spatial_key"], {"projection": 1234, "skycell": 56})

        # Upsert same (kind, key) -> one row, updated fields.
        store.upsert_input_file(
            "raw_skycell",
            "1234.056",
            {"projection": 1234, "skycell": 56},
            bytes=200,
            batch_id="batchB",
        )
        rows = store.list_input_files("raw_skycell")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bytes"], 200)
        self.assertEqual(rows[0]["batch_id"], "batchB")

        # Different key -> separate row.
        store.upsert_input_file(
            "raw_skycell", "1234.057", {"projection": 1234, "skycell": 57}
        )
        self.assertEqual(len(store.list_input_files("raw_skycell")), 2)
        # Different kind isolated.
        self.assertEqual(store.list_input_files("ffi"), [])


if __name__ == "__main__":
    unittest.main()
