"""Tests for ``common/provenance/store.py``: indexed queries + fault injection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.store import NoDirectoryWalkError, ProvenanceStore


class _TempStoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "bookkeeping" / "provenance.db"

    def make_store(self, **kwargs) -> ProvenanceStore:
        return ProvenanceStore(self.db_path, **kwargs)


class TestBasicReadWrite(_TempStoreCase):
    def test_upsert_and_fetch_artifact(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {"a": 1}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 20, "c": 1, "k": 1}, "rid1", "loc1")
        row = store.artifact("fp1")
        self.assertIsNotNone(row)
        self.assertEqual(row.kind, "mapping")
        self.assertEqual(row.spatial_key, {"s": 20, "c": 1, "k": 1})
        self.assertEqual(row.state, "complete")

    def test_artifact_missing_returns_none(self):
        store = self.make_store()
        self.assertIsNone(store.artifact("does-not-exist"))

    def test_edges_round_trip(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "diff_image", {}, 1)
        store.upsert_artifact("fp1", "diff_image", {"s": 1, "c": 1, "k": 1, "product_id": "t1"}, "rid1", "loc1")
        store.add_edges("fp1", ["in1", "in2", "in2"])  # duplicate is fine (INSERT OR IGNORE)
        self.assertEqual(sorted(store.inputs_of("fp1")), ["in1", "in2"])
        self.assertEqual(store.consumers_of("in1"), ["fp1"])

    def test_upsert_artifact_is_idempotent_replace(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {"a": 1}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1", "building")
        store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1", "complete")
        row = store.artifact("fp1")
        self.assertEqual(row.state, "complete")

    def test_invalid_state_rejected(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {}, 1)
        with self.assertRaises(ValueError):
            store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1", "bogus")

    def test_read_only_store_refuses_writes(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {}, 1)
        ro = self.make_store(read_only=True)
        with self.assertRaises(PermissionError):
            ro.upsert_recipe("rid2", "mapping", {}, 1)
        with self.assertRaises(PermissionError):
            ro.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1")


class TestMissingFingerprintsAndStageComplete(_TempStoreCase):
    def _seed_complete(self, store, fps):
        store.upsert_recipe("rid1", "diff_image", {}, 1)
        for i, fp in enumerate(fps):
            store.upsert_artifact(
                fp, "diff_image", {"s": 1, "c": 1, "k": 1, "product_id": f"t{i}"}, "rid1", f"loc{i}"
            )

    def test_missing_fingerprints_no_fallback(self):
        store = self.make_store()
        self._seed_complete(store, ["fp1", "fp2"])
        missing = store.missing_fingerprints(["fp1", "fp2", "fp3"], fallback_stat=False)
        self.assertEqual(missing, ["fp3"])

    def test_scc_stage_complete_true_when_all_present(self):
        store = self.make_store()
        self._seed_complete(store, ["fp1", "fp2", "fp3"])
        self.assertTrue(store.scc_stage_complete(["fp1", "fp2", "fp3"], fallback_stat=False))

    def test_scc_stage_complete_false_when_any_missing(self):
        store = self.make_store()
        self._seed_complete(store, ["fp1", "fp2"])
        self.assertFalse(store.scc_stage_complete(["fp1", "fp2", "fp3"], fallback_stat=False))

    def test_scc_stage_complete_vacuously_true_for_empty_required_set(self):
        store = self.make_store()
        self.assertTrue(store.scc_stage_complete([], fallback_stat=False))

    def test_building_state_does_not_count_as_complete(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1", "building")
        self.assertFalse(store.scc_stage_complete(["fp1"], fallback_stat=False))
        self.assertEqual(store.missing_fingerprints(["fp1"], fallback_stat=False), ["fp1"])

    def test_dedupes_required_fps(self):
        store = self.make_store()
        self._seed_complete(store, ["fp1"])
        missing = store.missing_fingerprints(["fp1", "fp1", "fp2", "fp2"], fallback_stat=False)
        self.assertEqual(missing, ["fp2"])

    def test_fallback_stat_only_touches_missing_keys(self):
        """§10: 'stat only the missing fingerprinted keys -- never the whole
        set'. Prove the fs_probe is called exactly once, for the one
        fingerprint absent from the index."""
        store = self.make_store()
        self._seed_complete(store, ["fp1", "fp2"])
        probed: list[str] = []

        def probe(fp: str) -> bool:
            probed.append(fp)
            return False

        store._fs_probe = probe  # type: ignore[attr-defined]
        missing = store.missing_fingerprints(["fp1", "fp2", "fp3"], fallback_stat=True)
        self.assertEqual(missing, ["fp3"])
        self.assertEqual(probed, ["fp3"])

    def test_fallback_stat_can_recover_a_freshly_published_but_unindexed_fp(self):
        store = self.make_store()
        self._seed_complete(store, ["fp1"])
        store._fs_probe = lambda fp: fp == "fp2"  # type: ignore[attr-defined]
        self.assertTrue(store.scc_stage_complete(["fp1", "fp2"], fallback_stat=True))


class TestFaultInjectionNoDirectoryWalk(_TempStoreCase):
    """Proves the indexed hot path (scc_stage_complete on a fully-indexed
    required set) never calls the injectable fs_probe -- i.e. never touches
    the filesystem -- which is the entire point of the graph replacing the
    O(cells) scan (§1 goal 6, §10, §18)."""

    def _raising_probe(self, fp: str) -> bool:
        raise NoDirectoryWalkError(
            f"fs_probe called for {fp!r}; the hot path must never stat when the index is complete"
        )

    def test_fully_indexed_required_set_never_probes_filesystem(self):
        store = self.make_store(fs_probe=self._raising_probe)
        store.upsert_recipe("rid1", "mapping", {}, 1)
        for fp in ("fp1", "fp2", "fp3"):
            store.upsert_artifact(fp, "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", f"loc-{fp}")

        # Must not raise: every fp is indexed complete, so the fallback path
        # (which would call the raising probe) is never reached.
        self.assertTrue(store.scc_stage_complete(["fp1", "fp2", "fp3"]))
        self.assertEqual(store.missing_fingerprints(["fp1", "fp2", "fp3"]), [])

    def test_disabling_fallback_never_probes_even_when_incomplete(self):
        store = self.make_store(fs_probe=self._raising_probe)
        store.upsert_recipe("rid1", "mapping", {}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1")
        # fp2 is missing from the index, but fallback_stat=False means the
        # scheduler-side caller explicitly opted out of any filesystem touch.
        self.assertFalse(store.scc_stage_complete(["fp1", "fp2"], fallback_stat=False))

    def test_incomplete_required_set_does_probe_only_the_gap(self):
        calls: list[str] = []

        def probe(fp: str) -> bool:
            calls.append(fp)
            return False

        store = self.make_store(fs_probe=probe)
        store.upsert_recipe("rid1", "mapping", {}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1")
        store.scc_stage_complete(["fp1", "fp2"])  # fp2 missing -> exactly one probe call
        self.assertEqual(calls, ["fp2"])


class TestStatsAndArtifactsByKindSpatial(_TempStoreCase):
    def test_stats_counts(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", "loc1")
        store.upsert_artifact("fp2", "mapping", {"s": 1, "c": 1, "k": 2}, "rid1", "loc2", "failed")
        stats = store.stats()
        self.assertEqual(stats["total_artifacts"], 2)
        self.assertEqual(stats["by_kind_state"]["mapping"]["complete"], 1)
        self.assertEqual(stats["by_kind_state"]["mapping"]["failed"], 1)

    def test_artifacts_by_kind_spatial(self):
        store = self.make_store()
        store.upsert_recipe("rid1", "mapping", {"a": 1}, 1)
        store.upsert_recipe("rid2", "mapping", {"a": 2}, 1)
        spatial = {"s": 1, "c": 1, "k": 1}
        store.upsert_artifact("fp1", "mapping", spatial, "rid1", "loc1")
        store.upsert_artifact("fp2", "mapping", spatial, "rid2", "loc2")
        store.upsert_artifact("fp3", "mapping", {"s": 1, "c": 1, "k": 2}, "rid1", "loc3")
        rows = store.artifacts_by_kind_spatial("mapping", spatial)
        self.assertEqual(sorted(r.fingerprint for r in rows), ["fp1", "fp2"])


if __name__ == "__main__":
    unittest.main()
