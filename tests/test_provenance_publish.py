"""Tests for ``common/provenance/publish.py``: atomicity, crash-sim,
concurrency, and the O_APPEND spool."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance import publish as publish_mod
from syndiff_pipeline.common.provenance.publish import (
    append_spool_record,
    build_record,
    default_spool_path,
    git_sha,
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


class _GitShaCacheResetCase(unittest.TestCase):
    """Save/restore the module-level git_sha() cache around each test so
    tests can force a fresh resolution without leaking state to others."""

    def setUp(self):
        self._old_cache = publish_mod._git_sha_cache
        publish_mod._git_sha_cache = publish_mod._UNRESOLVED
        self.addCleanup(self._restore)

    def _restore(self):
        publish_mod._git_sha_cache = self._old_cache


class TestGitSha(_GitShaCacheResetCase):
    def test_resolves_a_40char_hex_sha_in_this_real_checkout(self):
        # This test file lives inside an actual git checkout, so a correctly
        # implemented git_sha() must resolve a real, full-length SHA here.
        sha = git_sha()
        self.assertIsNotNone(sha)
        self.assertEqual(len(sha), 40)
        int(sha, 16)  # raises ValueError if not valid hex

    def test_matches_git_rev_parse_head_directly(self):
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        self.assertEqual(git_sha(), expected)

    def test_cached_after_first_call_subprocess_not_rerun(self):
        first = git_sha()
        with mock.patch.object(publish_mod.subprocess, "run") as mocked:
            second = git_sha()
        mocked.assert_not_called()
        self.assertEqual(first, second)

    def test_never_raises_when_git_binary_missing(self):
        with mock.patch.object(
            publish_mod.subprocess, "run", side_effect=FileNotFoundError("no git")
        ):
            self.assertIsNone(git_sha())

    def test_never_raises_and_returns_none_on_timeout(self):
        with mock.patch.object(
            publish_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5),
        ):
            self.assertIsNone(git_sha())

    def test_none_on_nonzero_returncode(self):
        fake = mock.Mock(returncode=128, stdout="")
        with mock.patch.object(publish_mod.subprocess, "run", return_value=fake):
            self.assertIsNone(git_sha())

    def test_none_result_is_cached_not_retried_forever(self):
        with mock.patch.object(
            publish_mod.subprocess, "run", side_effect=FileNotFoundError("no git")
        ) as mocked:
            first = git_sha()
            second = git_sha()
        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(mocked.call_count, 1)


class TestBuildRecordCarriesGitSha(_GitShaCacheResetCase):
    def test_build_record_includes_git_sha_from_cache(self):
        with mock.patch.object(publish_mod, "_resolve_git_sha", return_value="deadbeef" * 5):
            record = build_record("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], "loc1")
        self.assertEqual(record["git_sha"], "deadbeef" * 5)

    def test_build_record_includes_none_when_git_unavailable(self):
        with mock.patch.object(publish_mod, "_resolve_git_sha", return_value=None):
            record = build_record("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], "loc1")
        self.assertIsNone(record["git_sha"])

    def test_meta_is_forwarded_verbatim(self):
        # Contract A3b depends on this: a later-wave agent adds
        # meta["run_id"] at diff-stage emit sites and relies on build_record
        # not filtering/renaming/normalizing the meta mapping in any way.
        meta = {"run_id": "20260827_120000", "custom_key": {"nested": [1, 2, 3]}}
        record = build_record(
            "fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], "loc1", meta=meta,
        )
        self.assertEqual(record["meta"], meta)

    def test_meta_omitted_when_falsy(self):
        record = build_record("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], "loc1", meta=None)
        self.assertNotIn("meta", record)
        record2 = build_record("fp1", "mapping", {"s": 1, "c": 1, "k": 1}, "rid1", 1, [], "loc1", meta={})
        self.assertNotIn("meta", record2)


if __name__ == "__main__":
    unittest.main()
