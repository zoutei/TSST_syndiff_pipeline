"""SCC diff lane artifact naming."""

from __future__ import annotations

import unittest

from syndiff_pipeline.difference_imaging.orchestration import diff_store
from syndiff_pipeline.difference_imaging.support.ffi_naming import scc_diff_artifact_stem


class TestSccDiffNaming(unittest.TestCase):
    def test_stem_matches_basename_without_suffix(self):
        ffi_stem = "tess2020057105921-s0020-3-3"
        stem = scc_diff_artifact_stem(ffi_stem, "ks_d")
        self.assertEqual(
            diff_store.diff_artifact_basename(ffi_stem, "ks_d", suffix=""),
            f"{stem}"
        )
        self.assertEqual(stem, f"{ffi_stem}_ks_d")


if __name__ == "__main__":
    unittest.main()
