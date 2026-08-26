"""Regression test for the os.scandir-based rewrite of _count_npz_files.

Background: the original implementation (Path.rglob("*.npz") + Path.is_file())
re-stats every match with a fresh Path object, which over NFS is a second
round trip per file on top of the readdir that already found it. Verified
against a real ~351k-file remap cache (s0020_c3_k3 L4a) where this was the
dominant cost of an hours-long verify_remap call. Rewritten to use
os.scandir/os.DirEntry directly so the directory-entry type NFS already
returned is reused instead of re-stat'ing. This test locks down that the
new implementation counts identically to the documented contract: recursive,
".npz" suffix only, files only (not directories named "*.npz"), missing
directory returns 0.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.verify import _count_npz_files


class CountNpzFilesTests(unittest.TestCase):
    def test_missing_directory_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_count_npz_files(Path(tmp) / "does_not_exist"), 0)

    def test_counts_recursively_across_nested_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a" / "b").mkdir(parents=True)
            (root / "c").mkdir()
            (root / "a" / "one.npz").write_bytes(b"")
            (root / "a" / "b" / "two.npz").write_bytes(b"")
            (root / "c" / "three.npz").write_bytes(b"")
            (root / "root_level.npz").write_bytes(b"")
            self.assertEqual(_count_npz_files(root), 4)

    def test_ignores_non_npz_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "keep.npz").write_bytes(b"")
            (root / "skip.txt").write_bytes(b"")
            (root / "skip.npz.bak").write_bytes(b"")
            self.assertEqual(_count_npz_files(root), 1)

    def test_directory_named_like_npz_is_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "looks_like_a_file.npz").mkdir()
            (root / "real.npz").write_bytes(b"")
            self.assertEqual(_count_npz_files(root), 1)

    def test_empty_directory_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_count_npz_files(Path(tmp)), 0)


if __name__ == "__main__":
    unittest.main()
