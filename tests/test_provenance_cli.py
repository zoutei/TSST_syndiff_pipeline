"""Tests for the ``syndiff bookkeeping`` CLI wiring (additive-only)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.cli import build_parser


class _TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_root = Path(self._tmp.name)


class TestBookkeepingWiredIntoTopLevelParser(unittest.TestCase):
    def test_bookkeeping_subcommand_registered(self):
        parser = build_parser()
        args = parser.parse_args(["bookkeeping", "stats", "--data-root", "/tmp/nonexistent"])
        self.assertEqual(args.command, "bookkeeping")
        self.assertEqual(args.bookkeeping_action, "stats")
        self.assertTrue(callable(args.func))

    def test_existing_subcommands_unaffected(self):
        parser = build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.command, "status")

    def test_bookkeeping_requires_an_action(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["bookkeeping"])


class TestBookkeepingStatsAndReindex(_TempCase):
    def test_stats_on_empty_data_root(self, capsys=None):
        parser = build_parser()
        args = parser.parse_args(["bookkeeping", "stats", "--data-root", str(self.data_root)])
        rc = args.func(args)
        self.assertEqual(rc, 0)

    def test_reindex_on_empty_data_root_succeeds(self):
        parser = build_parser()
        args = parser.parse_args(["bookkeeping", "reindex", "--data-root", str(self.data_root)])
        rc = args.func(args)
        self.assertEqual(rc, 0)

    def test_reindex_then_stats_reflects_legacy_scc_tree(self):
        scc_dir = self.data_root / "s0020" / "c1" / "k1"
        (scc_dir / "convolved.zarr").mkdir(parents=True)

        parser = build_parser()
        reindex_args = parser.parse_args(
            ["bookkeeping", "reindex", "--data-root", str(self.data_root)]
        )
        self.assertEqual(reindex_args.func(reindex_args), 0)

        from syndiff_pipeline.common.provenance.store import ProvenanceStore
        from syndiff_pipeline.common.scc_paths import provenance_db_path

        store = ProvenanceStore(provenance_db_path(self.data_root), read_only=True)
        stats = store.stats()
        self.assertIn("scc_assembly_legacy_unverified", stats["by_kind_state"])

    def test_query_missing_fingerprint_returns_nonzero(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "bookkeeping", "query", "--data-root", str(self.data_root),
                "--fingerprint", "does-not-exist",
            ]
        )
        rc = args.func(args)
        self.assertEqual(rc, 1)

    def test_query_requires_fingerprint_or_kind_and_spatial_key(self):
        parser = build_parser()
        args = parser.parse_args(["bookkeeping", "query", "--data-root", str(self.data_root)])
        with self.assertRaises(SystemExit):
            args.func(args)

    def test_query_by_kind_and_spatial_key(self):
        from syndiff_pipeline.common.provenance.store import ProvenanceStore
        from syndiff_pipeline.common.scc_paths import provenance_db_path

        store = ProvenanceStore(provenance_db_path(self.data_root))
        store.upsert_recipe("rid1", "mapping", {"a": 1}, 1)
        store.upsert_artifact("fp1", "mapping", {"s": 20, "c": 1, "k": 1}, "rid1", "loc1")

        parser = build_parser()
        args = parser.parse_args(
            [
                "bookkeeping", "query", "--data-root", str(self.data_root),
                "--kind", "mapping", "--spatial-key", json.dumps({"s": 20, "c": 1, "k": 1}),
            ]
        )
        rc = args.func(args)
        self.assertEqual(rc, 0)


class TestStandaloneCliModule(unittest.TestCase):
    def test_standalone_main_dispatches_stats(self):
        from syndiff_pipeline.common.provenance.cli import main as bookkeeping_main

        with tempfile.TemporaryDirectory() as tmp:
            rc = bookkeeping_main(["stats", "--data-root", tmp])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
