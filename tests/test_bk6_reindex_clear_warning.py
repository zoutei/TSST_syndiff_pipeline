"""BK-6: full reindex clear warnings and background label inference."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from syndiff_pipeline.common.provenance.cli import cmd_bookkeeping_reindex
from syndiff_pipeline.common.provenance.gc import gc_report
from syndiff_pipeline.common.provenance.reindex import (
    REINDEX_CLEAR_PER_FFI_WARNING,
    _diff_kind_from_workspace_label,
    collect_reindex_clear_warnings,
    reindex_scc_tree,
)
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_spool_dir


class TestReindexClearWarnings(unittest.TestCase):
    def test_collect_warnings_always_includes_per_ffi_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            warnings = collect_reindex_clear_warnings(tmp)
        self.assertGreaterEqual(len(warnings), 1)
        self.assertEqual(warnings[0], REINDEX_CLEAR_PER_FFI_WARNING)

    def test_collect_warnings_flags_undrained_spool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            spool = provenance_spool_dir(data_root)
            spool.mkdir(parents=True)
            (spool / "host.123.jsonl").write_text('{"kind":"diff_image"}\n', encoding="utf-8")

            warnings = collect_reindex_clear_warnings(data_root)
        self.assertEqual(len(warnings), 2)
        self.assertIn("Undrained spool files", warnings[1])
        self.assertIn(str(spool), warnings[1])

    def test_cli_reindex_prints_clear_warning_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(data_root=tmp, incremental=False, config=None, site=None)
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = cmd_bookkeeping_reindex(args)
            self.assertEqual(rc, 0)
            err = buf.getvalue()
            self.assertIn("WARNING:", err)
            self.assertIn("spool-ingested", err)

    def test_cli_incremental_reindex_skips_clear_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(data_root=tmp, incremental=True, config=None, site=None)
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = cmd_bookkeeping_reindex(args)
            self.assertEqual(rc, 0)
            self.assertEqual(buf.getvalue(), "")


class TestBackgroundKindInference(unittest.TestCase):
    def test_hp_b_and_ks_b_labels_are_background(self) -> None:
        for label in ("hp_b", "ks_b", "ks_b_s"):
            with self.subTest(label=label):
                self.assertEqual(_diff_kind_from_workspace_label(label), "diff_background")

    def test_reindex_registers_hp_b_as_background_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            scc_dir = data_root / "s0020" / "c1" / "k1"
            label_dir = scc_dir / "diff" / "hp_b"
            label_dir.mkdir(parents=True)
            (label_dir / "tess2020019142923_hp_b.fits.fz").write_bytes(b"fake")

            store = ProvenanceStore(data_root / "bookkeeping" / "provenance.db")
            n = reindex_scc_tree(store, scc_dir, 20, 1, 1)
            self.assertEqual(n, 1)

            rows = store.artifacts_by_kind_spatial(
                "diff_background_legacy_unverified",
                {
                    "s": 20,
                    "c": 1,
                    "k": 1,
                    "workspace_label": "hp_b",
                },
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].location, str(label_dir))

    def test_gc_skips_tmp_label_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            scc_dir = data_root / "s0020" / "c1" / "k1"
            real = scc_dir / "diff" / "hp_d"
            real.mkdir(parents=True)
            (real / "frame.fits.fz").write_bytes(b"x")
            tmp_label = scc_dir / "diff" / "_tmp_publish"
            tmp_label.mkdir(parents=True)
            (tmp_label / "partial.fits.fz").write_bytes(b"x")

            report = gc_report(data_root)
            self.assertEqual(len(report.diff_recipe_dirs), 1)
            self.assertIn(str(real), report.diff_recipe_dirs)


if __name__ == "__main__":
    unittest.main()
