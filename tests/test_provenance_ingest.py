"""Tests for the supervisor-side spool ingest (provenance.ingest)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.provenance.ingest import drain_data_root, drain_spool
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, provenance_spool_dir


def _record(fp: str, kind: str = "combined_skycell", state: str = "complete") -> dict:
    return {
        "fingerprint": fp,
        "kind": kind,
        "spatial_key": {"projection": "1234", "skycell": "056"},
        "recipe_id": "recipe0001",
        "recipe": {
            "kind": kind,
            "params": {"psf_sigma": 40.0},
            "code_version": "1",
            "git_sha": None,
        },
        "inputs": [],
        "location": f"1234/skycell.1234.056/{fp}",
        "state": state,
        "bytes": 123,
        "wall_time_s": 1.5,
        "produced_by": "run-x",
    }


def _write_spool(spool_dir: Path, name: str, lines: list[str]) -> Path:
    spool_dir.mkdir(parents=True, exist_ok=True)
    path = spool_dir / name
    with open(path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    return path


class DrainSpoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.spool = self.root / "spool"
        self.store = ProvenanceStore(self.root / "provenance.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_drain_ingests_and_removes_spool(self) -> None:
        _write_spool(
            self.spool,
            "host1.111.jsonl",
            [json.dumps(_record("fp0000000000000000000001"))],
        )
        _write_spool(
            self.spool,
            "host2.222.jsonl",
            [json.dumps(_record("fp0000000000000000000002"))],
        )
        n = drain_spool(self.store, self.spool)
        self.assertEqual(n, 2)
        self.assertIsNotNone(self.store.get_artifact("fp0000000000000000000001"))
        self.assertIsNotNone(self.store.get_artifact("fp0000000000000000000002"))
        # Rotated files removed, no leftover *.jsonl.
        self.assertEqual(list(self.spool.glob("*.jsonl")), [])
        self.assertEqual(list(self.spool.glob("*.draining*")), [])

    def test_malformed_line_skipped_not_fatal(self) -> None:
        _write_spool(
            self.spool,
            "host1.111.jsonl",
            [
                json.dumps(_record("fp0000000000000000000010")),
                "{ this is not valid json",  # partial NFS append
                json.dumps(_record("fp0000000000000000000011")),
            ],
        )
        n = drain_spool(self.store, self.spool)
        self.assertEqual(n, 2)
        self.assertIsNotNone(self.store.get_artifact("fp0000000000000000000010"))
        self.assertIsNotNone(self.store.get_artifact("fp0000000000000000000011"))

    def test_record_missing_required_fields_skipped(self) -> None:
        _write_spool(
            self.spool,
            "host1.111.jsonl",
            [json.dumps({"kind": "combined_skycell"})],  # no fingerprint
        )
        n = drain_spool(self.store, self.spool)
        self.assertEqual(n, 0)

    def test_idempotent_redrain(self) -> None:
        _write_spool(
            self.spool,
            "host1.111.jsonl",
            [json.dumps(_record("fp0000000000000000000020"))],
        )
        self.assertEqual(drain_spool(self.store, self.spool), 1)
        # Re-emit the same record; must not duplicate the artifact.
        _write_spool(
            self.spool,
            "host1.111.jsonl",
            [json.dumps(_record("fp0000000000000000000020"))],
        )
        self.assertEqual(drain_spool(self.store, self.spool), 1)
        rows = list(self.store.iter_artifacts())
        self.assertEqual(len(rows), 1)

    def test_missing_spool_dir_is_noop(self) -> None:
        self.assertEqual(drain_spool(self.store, self.root / "nope"), 0)

    def test_drain_data_root_uses_canonical_paths(self) -> None:
        spool = provenance_spool_dir(self.root)
        _write_spool(
            spool,
            "host1.111.jsonl",
            [json.dumps(_record("fp0000000000000000000030"))],
        )
        store = ProvenanceStore(provenance_db_path(self.root))
        self.assertEqual(drain_data_root(store, self.root), 1)
        self.assertIsNotNone(store.get_artifact("fp0000000000000000000030"))


if __name__ == "__main__":
    unittest.main()
