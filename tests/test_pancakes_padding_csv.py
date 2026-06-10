"""Tests for mapping CSV padding metadata serialization."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing.pancakes import (
    finalize_master_skycells_csv,
    master_skycells_csv_paths,
    prepare_mapping_csv_workspace,
    update_skycells_with_padding_info,
)


class TestMasterSkycellsCsvPaths(unittest.TestCase):
    def test_default_and_oversampling_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial, final = master_skycells_csv_paths(tmp, 44, 2, 1)
            self.assertTrue(partial.endswith("tess_s0044_2_1_master_skycells_list.partial.csv"))
            self.assertTrue(final.endswith("tess_s0044_2_1_master_skycells_list.csv"))

            partial_os, final_os = master_skycells_csv_paths(tmp, 44, 2, 1, oversampling_factor=2)
            self.assertIn("oversampling_2", partial_os)
            self.assertTrue(
                partial_os.endswith("tess_s0044_2_1_master_skycells_list_os2.partial.csv")
            )
            self.assertTrue(final_os.endswith("tess_s0044_2_1_master_skycells_list_os2.csv"))

    def test_prepare_and_finalize_publish_final_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial, final = master_skycells_csv_paths(tmp, 22, 3, 3)
            Path(partial).parent.mkdir(parents=True, exist_ok=True)
            Path(final).write_text("stale\n", encoding="utf-8")
            Path(partial).write_text("NAME\nskycell.0001.0001\n", encoding="utf-8")

            prepare_mapping_csv_workspace(tmp, 22, 3, 3, overwrite=True)
            self.assertFalse(Path(partial).is_file())
            self.assertFalse(Path(final).is_file())

            Path(partial).write_text("NAME\nskycell.0001.0001\n", encoding="utf-8")
            published = finalize_master_skycells_csv(tmp, 22, 3, 3)
            self.assertEqual(published, final)
            self.assertFalse(Path(partial).is_file())
            self.assertTrue(Path(final).is_file())
            self.assertIn("skycell.0001.0001", Path(final).read_text(encoding="utf-8"))

    def test_finalize_requires_partial_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                finalize_master_skycells_csv(tmp, 22, 3, 3)


class TestPancakesPaddingCsv(unittest.TestCase):
    def test_update_skycells_stringifies_boolean_padding_fields(self):
        selected = pd.DataFrame({"NAME": ["skycell.2556.082"]})
        padding_results = {
            "skycell.2556.082": {
                "pad_skycell_top": "skycell.2556.083",
                "special_padding_needed": False,
                "edge_pixels_used": True,
                "good_side_fail": False,
                "special_padding_flags": [False, True, False, False, False, False, False, False],
            }
        }

        updated = update_skycells_with_padding_info(selected, padding_results)

        self.assertEqual(updated.at[0, "special_padding_needed"], "False")
        self.assertEqual(updated.at[0, "edge_pixels_used"], "True")
        self.assertEqual(updated.at[0, "good_side_fail"], "False")
        self.assertEqual(
            updated.at[0, "special_padding_flags"],
            "[False, True, False, False, False, False, False, False]",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "padding.csv"
            updated.to_csv(out, index=False)
            self.assertTrue(out.is_file())

    def test_update_skycells_csv_roundtrip_all_false(self):
        selected = pd.DataFrame({"NAME": ["skycell.2556.082"]})
        padding_results = {
            "skycell.2556.082": {
                "pad_skycell_top": "",
                "special_padding_needed": False,
                "edge_pixels_used": False,
                "good_side_fail": False,
                "special_padding_flags": [False] * 8,
            }
        }
        updated = update_skycells_with_padding_info(selected, padding_results)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "padding.csv"
            updated.to_csv(out, index=False)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
