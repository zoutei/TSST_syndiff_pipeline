"""Tests for full/short pipeline stage name resolution."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.pipeline_spec import get_syndiff_pipeline, resolve_stage_name


class TestResolveStageName(unittest.TestCase):
    def test_full_name_unchanged(self):
        self.assertEqual(resolve_stage_name("mapping"), "mapping")
        self.assertEqual(resolve_stage_name("  mapping  "), "mapping")

    def test_short_name_resolves(self):
        self.assertEqual(resolve_stage_name("map"), "mapping")
        self.assertEqual(resolve_stage_name("down"), "downsample")
        self.assertEqual(resolve_stage_name("tess_dl"), "tess_ffi_download")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_stage_name("not_a_stage")
        msg = str(ctx.exception)
        self.assertIn("Unknown stage", msg)
        self.assertIn("mapping", msg)
        self.assertIn("map", msg)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            resolve_stage_name("   ")

    def test_parse_stage_list_accepts_short_names(self):
        pipeline = get_syndiff_pipeline()
        self.assertEqual(
            pipeline.parse_stage_list("map,down"),
            ["mapping", "downsample"],
        )


if __name__ == "__main__":
    unittest.main()
