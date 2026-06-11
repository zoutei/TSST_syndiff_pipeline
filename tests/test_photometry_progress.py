"""Tests for forced-photometry progress sidecar helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.difference_imaging.stages import photometry_progress as pp


class TestPhotometryProgress(unittest.TestCase):
    def test_init_and_mark_epoch_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ws" / "lc_prf_on_diffs" / pp.PROGRESS_FILENAME
            pp.init_progress(
                path,
                output_label="lc_prf_on_diffs",
                diffs_input="hp_d",
                n_sources=1,
                epochs_total=5,
                phase="flux",
            )
            pp.mark_epoch_done(path)
            pp.mark_epoch_done(path)
            pp.mark_epoch_done(path)

            data = pp.read_progress(path)
            assert data is not None
            self.assertEqual(data["output_label"], "lc_prf_on_diffs")
            self.assertEqual(data["diffs_input"], "hp_d")
            self.assertEqual(data["epochs_done"], 3)
            self.assertEqual(data["phase"], "flux")

    def test_reset_epochs_done_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws_path = Path(tmp) / "ws" / "lc_prf_on_diffs" / pp.PROGRESS_FILENAME
            cli_path = Path(tmp) / "diff.photometry.progress.json"
            pp.init_progress_pair(
                ws_path,
                cli_path,
                output_label="lc_prf_on_diffs",
                diffs_input="hp_d",
                n_sources=2,
                epochs_total=4,
                phase="cutouts",
            )
            pp.record_epoch_progress(ws_path, cli_path)
            pp.record_epoch_progress(ws_path, cli_path)
            pp.reset_epochs_done_pair(ws_path, cli_path, phase="flux")
            pp.record_epoch_progress(ws_path, cli_path)
            pp.set_progress_phase_pair(ws_path, cli_path, "complete")

            ws = pp.read_progress(ws_path)
            cli = pp.read_progress(cli_path)
            assert ws is not None and cli is not None
            self.assertEqual(ws["epochs_done"], 1)
            self.assertEqual(ws["phase"], "complete")
            self.assertEqual(cli["n_sources"], 2)
            self.assertEqual(
                pp.format_progress_text(ws),
                "photometry lc_prf_on_diffs (2 src) complete 1/4",
            )

    def test_progress_path_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "per_target" / "target" / "diff.log"
            log_path.parent.mkdir(parents=True)
            log_path.touch()
            out_dir = Path(tmp) / "ws" / "lc_prf_on_diffs"
            self.assertEqual(
                pp.progress_path_for_diff_log(log_path).name,
                pp.CLI_PROGRESS_FILENAME,
            )
            self.assertEqual(
                pp.progress_path_for_output_workspace(out_dir).name,
                pp.PROGRESS_FILENAME,
            )


if __name__ == "__main__":
    unittest.main()
