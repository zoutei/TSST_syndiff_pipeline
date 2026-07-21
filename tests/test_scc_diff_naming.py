"""SCC diff lane artifact naming."""

from __future__ import annotations

import unittest

from syndiff_pipeline.difference_imaging.orchestration import diff_store
from syndiff_pipeline.difference_imaging.support.ffi_naming import scc_diff_artifact_stem


class TestSccDiffNaming(unittest.TestCase):
    def test_stem_matches_basename_without_suffix(self):
        stem = scc_diff_artifact_stem("tess123", "ks_d")
        self.assertEqual(
            diff_store.diff_artifact_basename("tess123", "ks_d", suffix=""),
            f"{stem}"
        )
        self.assertEqual(stem, "tess123_ks_d")


if __name__ == "__main__":
    unittest.main()
