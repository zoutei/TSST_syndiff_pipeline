"""Tests for ``common/provenance/reindex.py``: offline rebuild from disk."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.fingerprint import fingerprint as make_fp
from syndiff_pipeline.common.provenance.fingerprint import recipe_id as make_rid
from syndiff_pipeline.common.provenance.ingest import drain_spool
from syndiff_pipeline.common.provenance.publish import publish_dir
from syndiff_pipeline.common.provenance.reindex import (
    discover_scc_dirs,
    reindex_data_root,
    reindex_scc_tree,
    reindex_shared_store,
)
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import ps1_combined_zarr_path, ps1_convolved_zarr_path


class _TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_root = Path(self._tmp.name)


class TestReindexSharedStore(_TempCase):
    def test_self_describing_artifact_is_verified_not_legacy(self):
        combined_root = ps1_combined_zarr_path(self.data_root)
        proj, skycell = "skycell1234", "2001"
        spatial_key = {"projection": proj, "skycell": skycell}
        params = {"gaia_version": "dr3"}
        rid = make_rid("combined_skycell", params, 1)
        fp = make_fp("combined_skycell", spatial_key, rid, [])

        def writer(d: Path):
            (d / "arrays.npz").write_bytes(b"fake")

        publish_dir(
            combined_root / proj / skycell, fp, "combined_skycell", spatial_key, rid, 1, [],
            writer, recipe_params=params,
        )

        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n_ok, n_legacy = reindex_shared_store(store, combined_root, "combined_skycell")
        self.assertEqual((n_ok, n_legacy), (1, 0))
        row = store.artifact(fp)
        self.assertIsNotNone(row)
        self.assertEqual(row.kind, "combined_skycell")
        recipe = store.recipe(rid)
        self.assertEqual(recipe.params, params)

    def test_tampered_sidecar_falls_back_to_legacy(self):
        combined_root = ps1_combined_zarr_path(self.data_root)
        proj, skycell = "skycell1234", "2001"

        def writer(d: Path):
            (d / "arrays.npz").write_bytes(b"fake")
            # Write a sidecar whose fingerprint does NOT match the directory
            # name it lives in (simulating drift/corruption).
            import json

            (d / "_provenance.json").write_text(
                json.dumps(
                    {
                        "fingerprint": "not-the-real-fp",
                        "kind": "combined_skycell",
                        "spatial_key": {"projection": proj, "skycell": skycell},
                        "recipe_id": "bogus",
                        "inputs": [],
                        "location": "irrelevant",
                    }
                )
            )

        dest_dir = combined_root / proj / skycell / "some-directory-name"
        dest_dir.mkdir(parents=True)
        writer(dest_dir)

        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n_ok, n_legacy = reindex_shared_store(store, combined_root, "combined_skycell")
        self.assertEqual((n_ok, n_legacy), (0, 1))
        stats = store.stats()
        self.assertIn("combined_skycell_legacy_unverified", stats["by_kind_state"])

    def test_missing_sidecar_falls_back_to_legacy(self):
        combined_root = ps1_combined_zarr_path(self.data_root)
        dest_dir = combined_root / "proj1" / "cell1" / "somefp"
        dest_dir.mkdir(parents=True)
        (dest_dir / "arrays.npz").write_bytes(b"legacy-no-sidecar")

        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n_ok, n_legacy = reindex_shared_store(store, combined_root, "combined_skycell")
        self.assertEqual((n_ok, n_legacy), (0, 1))

    def test_empty_or_missing_store_root_is_a_noop(self):
        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n_ok, n_legacy = reindex_shared_store(
            store, self.data_root / "does-not-exist", "combined_skycell"
        )
        self.assertEqual((n_ok, n_legacy), (0, 0))


class TestReindexSccTree(_TempCase):
    def _build_synthetic_scc(self, scc_dir: Path):
        (scc_dir / "remap" / "oversampling_2").mkdir(parents=True)
        (scc_dir / "remap" / "oversampling_2" / "remap_manifest.json").write_text("{}")
        (scc_dir / "templates" / "oversampling_2").mkdir(parents=True)
        (scc_dir / "templates" / "oversampling_2" / "foo.fits").write_text("x")
        (scc_dir / "convolved.zarr").mkdir(parents=True)
        (scc_dir / "mapping" / "oversampling_2").mkdir(parents=True)
        (scc_dir / "mapping" / "oversampling_2" / "foo.csv").write_text("x")

    def test_legacy_sweep_counts_and_kinds(self):
        scc_dir = self.data_root / "s0020" / "c1" / "k1"
        self._build_synthetic_scc(scc_dir)
        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n = reindex_scc_tree(store, scc_dir, 20, 1, 1)
        self.assertEqual(n, 4)
        stats = store.stats()
        for kind in (
            "scc_assembly_legacy_unverified",
            "remap_store_legacy_unverified",
            "downsample_legacy_unverified",
            "mapping_legacy_unverified",
        ):
            self.assertIn(kind, stats["by_kind_state"])

    def test_empty_oversampling_dirs_are_skipped(self):
        scc_dir = self.data_root / "s0020" / "c1" / "k1"
        (scc_dir / "templates" / "oversampling_2").mkdir(parents=True)  # empty
        (scc_dir / "mapping" / "oversampling_2").mkdir(parents=True)  # empty
        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n = reindex_scc_tree(store, scc_dir, 20, 1, 1)
        self.assertEqual(n, 0)

    def test_reindex_is_idempotent(self):
        scc_dir = self.data_root / "s0020" / "c1" / "k1"
        self._build_synthetic_scc(scc_dir)
        store = ProvenanceStore(self.data_root / "bookkeeping" / "provenance.db")
        n1 = reindex_scc_tree(store, scc_dir, 20, 1, 1)
        n2 = reindex_scc_tree(store, scc_dir, 20, 1, 1)
        self.assertEqual(n1, n2)
        # Re-running does not create duplicate rows (same synthetic fp both times).
        self.assertEqual(store.stats()["total_artifacts"], 4)


class TestDiscoverSccDirs(_TempCase):
    def test_discovers_nested_scc_dirs_only(self):
        (self.data_root / "s0020" / "c1" / "k1").mkdir(parents=True)
        (self.data_root / "s0020" / "c1" / "k2").mkdir(parents=True)
        (self.data_root / "s0021" / "c2" / "k3").mkdir(parents=True)
        (self.data_root / "ps1_skycells_zarr").mkdir(parents=True)  # not an SCC dir
        found = discover_scc_dirs(self.data_root)
        self.assertEqual(
            sorted((s, c, k) for s, c, k, _ in found),
            [(20, 1, 1), (20, 1, 2), (21, 2, 3)],
        )

    def test_missing_data_root_returns_empty(self):
        self.assertEqual(discover_scc_dirs(self.data_root / "nope"), [])


class TestReindexMatchesLiveIndex(_TempCase):
    """§18: 'reindex == live index'. Build a synthetic tree via the same
    publish_dir()+drain_spool() path a real producer/supervisor would use
    (the "live" index), then wipe the DB and rebuild purely from disk via
    reindex_data_root -- the resulting artifact/recipe sets must match."""

    def test_synthetic_tree_reindex_matches_live_built_index(self):
        combined_root = ps1_combined_zarr_path(self.data_root)
        convolved_root = ps1_convolved_zarr_path(self.data_root)
        spool_dir = self.data_root / "bookkeeping" / "spool"

        cells = [("proj1", "0001"), ("proj1", "0002"), ("proj2", "0001")]
        published = []
        for proj, skycell in cells:
            spatial_key = {"projection": proj, "skycell": skycell}
            params = {"gaia_version": "dr3"}
            rid = make_rid("combined_skycell", params, 1)
            fp = make_fp("combined_skycell", spatial_key, rid, [])

            def writer(d: Path):
                (d / "arrays.npz").write_bytes(b"fake")

            publish_dir(
                combined_root / proj / skycell, fp, "combined_skycell", spatial_key, rid, 1, [],
                writer, recipe_params=params, spool_dir=spool_dir,
            )
            published.append(fp)

            conv_rid = make_rid("convolved_skycell", {"psf_sigma": 60.0}, 1)
            conv_fp = make_fp("convolved_skycell", spatial_key, conv_rid, [fp])
            publish_dir(
                convolved_root / proj / skycell, conv_fp, "convolved_skycell", spatial_key,
                conv_rid, 1, [fp], writer, recipe_params={"psf_sigma": 60.0}, spool_dir=spool_dir,
            )
            published.append(conv_fp)

        # "Live" index: what the supervisor would build by draining the spool.
        live_store = ProvenanceStore(self.data_root / "bookkeeping" / "live.db")
        drain_spool(live_store, spool_dir)

        # Also add one legacy SCC tree, discovered by reindex's SCC sweep
        # (nothing publishes sidecars for it -- it predates this package).
        scc_dir = self.data_root / "s0020" / "c1" / "k1"
        (scc_dir / "convolved.zarr").mkdir(parents=True)
        legacy_store_for_scc = ProvenanceStore(self.data_root / "bookkeeping" / "scc_live.db")
        legacy_count_live = reindex_scc_tree(legacy_store_for_scc, scc_dir, 20, 1, 1)

        # Reindexed-from-scratch index: pure filesystem rebuild.
        rebuilt_store = ProvenanceStore(self.data_root / "bookkeeping" / "rebuilt.db")
        result = reindex_data_root(self.data_root, rebuilt_store, clear_first=True)

        self.assertEqual(result.shared_store_artifacts, len(published))
        self.assertEqual(result.shared_store_legacy, 0)
        self.assertEqual(result.scc_legacy_artifacts, legacy_count_live)
        self.assertEqual(result.sccs_scanned, 1)

        for fp in published:
            live_row = live_store.artifact(fp)
            rebuilt_row = rebuilt_store.artifact(fp)
            self.assertIsNotNone(live_row)
            self.assertIsNotNone(rebuilt_row)
            self.assertEqual(live_row.kind, rebuilt_row.kind)
            self.assertEqual(live_row.spatial_key, rebuilt_row.spatial_key)
            self.assertEqual(live_row.recipe_id, rebuilt_row.recipe_id)
            self.assertEqual(sorted(live_store.inputs_of(fp)), sorted(rebuilt_store.inputs_of(fp)))

        self.assertEqual(
            rebuilt_store.stats()["by_kind_state"].get("scc_assembly_legacy_unverified"),
            legacy_store_for_scc.stats()["by_kind_state"].get("scc_assembly_legacy_unverified"),
        )


if __name__ == "__main__":
    unittest.main()
