"""Tests for ``common/provenance/publish.py``: atomicity, crash-sim,
concurrency, and the O_APPEND spool."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.publish import (
    append_spool_record,
    build_record,
    default_spool_path,
    host_pid_tag,
    publish_dir,
    publish_record,
    try_publish_dir,
    try_publish_record,
)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


class TestPublishDirAtomicity(_TempDirCase):
    def test_success_writes_provenance_json_and_no_tmp_leftovers(self):
        dest_root = self.tmp / "store"

        def writer(d: Path):
            (d / "data.txt").write_text("hello")

        final = publish_dir(
            dest_root, "fp1", "combined_skycell",
            {"projection": "p1", "skycell": "2001"}, "rid1", 1, ["in1"], writer,
        )
        self.assertTrue(final.is_dir())
        self.assertEqual((final / "data.txt").read_text(), "hello")
        record = json.loads((final / "_provenance.json").read_text())
        self.assertEqual(record["fingerprint"], "fp1")
        self.assertEqual(record["inputs"], ["in1"])
        leftovers = list(dest_root.glob("_tmp_*"))
        self.assertEqual(leftovers, [])

    def test_crash_mid_write_leaves_only_tmp_orphan(self):
        dest_root = self.tmp / "store"

        def crashing_writer(d: Path):
            (d / "partial.txt").write_text("partial-bytes")
            raise RuntimeError("simulated crash mid-write")

        with self.assertRaises(RuntimeError):
            publish_dir(
                dest_root, "fpcrash", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [],
                crashing_writer,
            )
        final = dest_root / "fpcrash"
        self.assertFalse(final.exists(), "final key must never appear on a failed publish")
        # Our own cleanup runs in-process; a true SIGKILL would leave a
        # _tmp_* orphan instead of nothing -- either way, nothing but a
        # _tmp_-prefixed entry (or no entry at all) is ever left, and the
        # crucial invariant (no partial final key) holds in both cases.
        leftovers = list(dest_root.glob("*")) if dest_root.exists() else []
        for leftover in leftovers:
            self.assertTrue(leftover.name.startswith("_tmp_"))

    def test_concurrent_publish_of_same_fingerprint_is_idempotent(self):
        dest_root = self.tmp / "store"
        results: list[Path] = []
        errors: list[BaseException] = []

        def writer(d: Path):
            (d / "data.txt").write_text("same-content")

        def worker():
            try:
                p = publish_dir(
                    dest_root, "shared-fp", "combined_skycell",
                    {"projection": "p1", "skycell": "1"}, "rid1", 1, [], writer,
                )
                results.append(p)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8)
        final = dest_root / "shared-fp"
        self.assertTrue(final.is_dir())
        self.assertEqual((final / "data.txt").read_text(), "same-content")
        self.assertTrue(all(p == final for p in results))
        self.assertEqual(list(dest_root.glob("_tmp_*")), [])

    def test_spool_dir_receives_one_line_per_publish(self):
        dest_root = self.tmp / "store"
        spool_dir = self.tmp / "spool"

        def writer(d: Path):
            (d / "data.txt").write_text("x")

        publish_dir(
            dest_root, "fp1", "combined_skycell", {"projection": "p1", "skycell": "1"},
            "rid1", 1, [], writer, spool_dir=spool_dir,
        )
        files = list(spool_dir.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["fingerprint"], "fp1")


class TestPublishRecordAtomicity(_TempDirCase):
    def test_success_atomic_file_publish(self):
        dest_path = self.tmp / "diff" / "tess1_diff.fits"

        def writer(p: Path):
            p.write_bytes(b"fits-bytes")

        out = publish_record(
            dest_path, "fp1", "diff_image", {"s": 1, "c": 1, "k": 1, "product_id": "tess1"},
            "rid1", 1, ["ffi1"], writer,
        )
        self.assertEqual(out, dest_path)
        self.assertEqual(dest_path.read_bytes(), b"fits-bytes")
        self.assertEqual(list(dest_path.parent.glob("_tmp_*")), [])

    def test_crash_mid_write_leaves_only_tmp_orphan_and_no_dest(self):
        dest_path = self.tmp / "diff" / "tess1_diff.fits"

        def crashing_writer(p: Path):
            p.write_bytes(b"partial")
            raise RuntimeError("simulated crash")

        with self.assertRaises(RuntimeError):
            publish_record(
                dest_path, "fp1", "diff_image", {"s": 1, "c": 1, "k": 1, "product_id": "tess1"},
                "rid1", 1, [], crashing_writer,
            )
        self.assertFalse(dest_path.exists())
        leftovers = list(dest_path.parent.glob("*")) if dest_path.parent.exists() else []
        for leftover in leftovers:
            self.assertTrue(leftover.name.startswith("_tmp_"))

    def test_republish_overwrites_atomically(self):
        dest_path = self.tmp / "diff" / "tess1_diff.fits"
        dest_path.parent.mkdir(parents=True)
        dest_path.write_bytes(b"old-content")

        def writer(p: Path):
            p.write_bytes(b"new-content")

        publish_record(
            dest_path, "fp2", "diff_image", {"s": 1, "c": 1, "k": 1, "product_id": "tess1"},
            "rid1", 1, [], writer,
        )
        self.assertEqual(dest_path.read_bytes(), b"new-content")


class TestBestEffortWrappers(_TempDirCase):
    def test_try_publish_dir_swallows_exceptions(self):
        def crashing_writer(d: Path):
            raise RuntimeError("boom")

        result = try_publish_dir(
            self.tmp / "store", "fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [],
            crashing_writer,
        )
        self.assertIsNone(result)

    def test_try_publish_record_swallows_exceptions(self):
        def crashing_writer(p: Path):
            raise RuntimeError("boom")

        result = try_publish_record(
            self.tmp / "a.fits", "fp1", "diff_image", {"s": 1, "c": 1, "k": 1, "product_id": "t"},
            "rid1", 1, [], crashing_writer,
        )
        self.assertIsNone(result)

    def test_try_publish_dir_returns_path_on_success(self):
        def writer(d: Path):
            (d / "data.txt").write_text("ok")

        result = try_publish_dir(
            self.tmp / "store", "fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], writer
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.is_dir())


class TestSpoolAppend(_TempDirCase):
    def test_append_spool_record_o_append_multiple_lines(self):
        spool_dir = self.tmp / "spool"
        rec1 = build_record("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], "loc1")
        rec2 = build_record("fp2", "mapping", {"s": 1, "c": 1, "k": 2}, "rid1", 1, [], "loc2")
        p1 = append_spool_record(spool_dir, rec1)
        p2 = append_spool_record(spool_dir, rec2)
        self.assertEqual(p1, p2)  # same process -> same host.pid spool file
        lines = p1.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["fingerprint"], "fp1")
        self.assertEqual(json.loads(lines[1])["fingerprint"], "fp2")

    def test_default_spool_path_encodes_host_and_pid(self):
        spool_dir = self.tmp / "spool"
        path = default_spool_path(spool_dir)
        self.assertEqual(path.name, f"{host_pid_tag()}.jsonl")

    def test_concurrent_appends_from_threads_never_interleave_a_line(self):
        # All threads share this process's pid, so they append to the SAME
        # spool file; O_APPEND still guarantees each write() (one JSON line,
        # well under PIPE_BUF) lands whole, never interleaved mid-line.
        spool_dir = self.tmp / "spool"
        n = 40

        def worker(i: int):
            rec = build_record(f"fp{i}", "mapping", {"s": 1, "c": 1, "k": i}, "rid1", 1, [], f"loc{i}")
            append_spool_record(spool_dir, rec)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        files = list(spool_dir.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        lines = files[0].read_text().splitlines()
        self.assertEqual(len(lines), n)
        fps = set()
        for line in lines:
            record = json.loads(line)  # raises if a line got interleaved/corrupted
            fps.add(record["fingerprint"])
        self.assertEqual(fps, {f"fp{i}" for i in range(n)})


if __name__ == "__main__":
    unittest.main()
