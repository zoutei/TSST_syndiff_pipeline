"""Tests for provenance publish, reindex, and the bookkeeping CLI."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from syndiff_pipeline.common.provenance import cli
from syndiff_pipeline.common.provenance.publish import (
    STORE_PROVENANCE_BASENAME,
    emit_sidecar,
    publish_record,
    read_store_provenance,
    write_store_provenance,
)
from syndiff_pipeline.common.provenance.reindex import (
    LEGACY_STATE,
    reindex_data_root,
    reindex_shared_store,
)
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import (
    provenance_db_path,
    ps1_combined_zarr_path,
    ps1_skycells_zarr_path,
)


def _record(fp: str, projection: str = "1234", skycell: str = "056") -> dict:
    return {
        "fingerprint": fp,
        "kind": "combined_skycell",
        "spatial_key": {"projection": projection, "skycell": skycell},
        "recipe_id": "rid0000000000001",
        "recipe": {
            "kind": "combined_skycell",
            "params": {"psf_sigma": 40.0, "bright_star_mag_threshold": 13.0},
            "code_version": "1",
            "git_sha": None,
        },
        "inputs": [],
        "location": None,
        "state": "complete",
        "bytes": 100,
        "wall_time_s": 1.0,
        "produced_by": "run-x",
    }


class PublishTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_store_provenance_roundtrip(self) -> None:
        d = self.root / "art"
        d.mkdir()
        rec = _record("fp01")
        write_store_provenance(d, rec)
        self.assertTrue((d / STORE_PROVENANCE_BASENAME).is_file())
        self.assertEqual(read_store_provenance(d), rec)
        self.assertIsNone(read_store_provenance(self.root / "nope"))

    def test_emit_sidecar_appends_line(self) -> None:
        spool = emit_sidecar(_record("fp02"), self.root)
        emit_sidecar(_record("fp03"), self.root)
        lines = spool.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["fingerprint"], "fp02")

    def test_publish_record_atomic_rename_and_sidecar(self) -> None:
        tmp = self.root / "tmp_build"
        tmp.mkdir()
        (tmp / "data").write_bytes(b"bytes")  # stand-in for zarr arrays
        final = self.root / "store" / "1234" / "skycell.1234.056" / "fp04"
        out = publish_record(
            _record("fp04"), tmp_dir=tmp, final_dir=final, data_root=self.root
        )
        self.assertEqual(out, final)
        self.assertTrue((final / "data").is_file())
        self.assertTrue((final / STORE_PROVENANCE_BASENAME).is_file())
        self.assertFalse(tmp.exists())  # renamed away

    def test_publish_record_loser_discards_tmp(self) -> None:
        final = self.root / "store" / "fp05"
        final.mkdir(parents=True)
        (final / STORE_PROVENANCE_BASENAME).write_text("{}", encoding="utf-8")
        tmp = self.root / "tmp_build2"
        tmp.mkdir()
        out = publish_record(
            _record("fp05"), tmp_dir=tmp, final_dir=final, data_root=self.root, emit=False
        )
        self.assertEqual(out, final)
        self.assertFalse(tmp.exists())  # loser cleaned up, winner kept


class ReindexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = ProvenanceStore(self.root / "provenance.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_fp_dir(self, store_path: Path, proj: str, sky: str, fp: str, record=None) -> Path:
        d = store_path / proj / sky / fp
        d.mkdir(parents=True)
        (d / "data").write_bytes(b"x")
        if record is not None:
            write_store_provenance(d, record)
        return d

    def test_reindex_sidecar_and_legacy(self) -> None:
        store_path = self.root / "combined"
        self._make_fp_dir(store_path, "1234", "skycell.1234.056", "fpAAA", _record("fpAAA"))
        # No sidecar -> legacy_unverified.
        self._make_fp_dir(store_path, "1234", "skycell.1234.057", "fpBBB", None)
        counts = reindex_shared_store(self.store, store_path, "combined_skycell")
        self.assertEqual(counts, {"sidecar": 1, "legacy": 1})
        good = self.store.get_artifact("fpAAA")
        self.assertEqual(good["state"], "complete")
        legacy = self.store.get_artifact("fpBBB")
        self.assertEqual(legacy["state"], LEGACY_STATE)
        self.assertEqual(legacy["kind"], "combined_skycell")

    def test_reindex_is_rebuildable_end_to_end(self) -> None:
        """Publish to disk, then rebuild a fresh DB purely from the store."""
        store_path = ps1_combined_zarr_path(self.root)
        tmp = self.root / "build"
        tmp.mkdir()
        (tmp / "data").write_bytes(b"z")
        final = store_path / "1234" / "skycell.1234.099" / "fpCCC"
        publish_record(
            _record("fpCCC", skycell="099"),
            tmp_dir=tmp,
            final_dir=final,
            data_root=self.root,
            emit=False,
        )
        # Fresh DB knows nothing until reindex reads the self-describing store.
        fresh = ProvenanceStore(self.root / "fresh.db")
        self.assertIsNone(fresh.get_artifact("fpCCC"))
        reindex_data_root(fresh, self.root)
        art = fresh.get_artifact("fpCCC")
        self.assertIsNotNone(art)
        self.assertEqual(art["state"], "complete")
        self.assertIsNotNone(fresh.get_recipe(art["recipe_id"]))

    def test_reindex_raw_skycells_input_files(self) -> None:
        raw = ps1_skycells_zarr_path(self.root)
        (raw / "1234" / "skycell.1234.056").mkdir(parents=True)
        (raw / "1234" / "skycell.1234.057").mkdir(parents=True)
        result = reindex_data_root(self.store, self.root)
        self.assertEqual(result["raw_skycell"]["input_files"], 2)
        self.assertEqual(len(self.store.list_input_files("raw_skycell")), 2)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Seed a store with one published artifact.
        store_path = ps1_combined_zarr_path(self.root)
        tmp = self.root / "b"
        tmp.mkdir()
        (tmp / "data").write_bytes(b"z")
        final = store_path / "1234" / "skycell.1234.056" / "fpDDD"
        publish_record(
            _record("fpDDD"), tmp_dir=tmp, final_dir=final, data_root=self.root, emit=False
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, argv) -> tuple[int, dict]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(argv)
        out = buf.getvalue().strip()
        parsed = json.loads(out) if out else {}
        return rc, parsed

    def test_reindex_then_stats_and_query(self) -> None:
        rc, _ = self._run(["reindex", "--data-root", str(self.root)])
        self.assertEqual(rc, 0)
        rc, stats = self._run(["stats", "--data-root", str(self.root)])
        self.assertEqual(rc, 0)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["by_kind"]["combined_skycell"], 1)
        rc, art = self._run(
            ["query", "--data-root", str(self.root), "--fingerprint", "fpDDD"]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(art["fingerprint"], "fpDDD")
        self.assertIsNotNone(art["recipe"])

    def test_verify_reports_missing_on_disk(self) -> None:
        # Reindex records location; then delete the bytes to simulate drift.
        self._run(["reindex", "--data-root", str(self.root)])
        store = ProvenanceStore(provenance_db_path(self.root))
        art = store.get_artifact("fpDDD")
        # Location exists now -> verify clean.
        rc, out = self._run(["verify", "--data-root", str(self.root)])
        self.assertEqual(rc, 0)
        self.assertEqual(out["missing_on_disk"], 0)
        # Remove the artifact dir -> verify flags it.
        import shutil

        shutil.rmtree(art["location"])
        rc, out = self._run(["verify", "--data-root", str(self.root)])
        self.assertEqual(rc, 1)
        self.assertEqual(out["missing_on_disk"], 1)


if __name__ == "__main__":
    unittest.main()
