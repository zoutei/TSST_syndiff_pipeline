"""Tests for Hotpants progress sidecar helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.difference_imaging.stages import hotpants_progress as hp


class TestHotpantsProgress(unittest.TestCase):
    def test_init_and_mark_frame_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ws" / "hp_d" / hp.PROGRESS_FILENAME
            hp.init_progress(
                path,
                diffs_label="hp_d",
                round_id=1,
                science="ffi",
                frames_total=3,
            )
            hp.mark_frame_done(path, success=True)
            hp.mark_frame_done(path, success=False)
            hp.mark_frame_done(path, success=True)

            data = hp.read_progress(path)
            assert data is not None
            self.assertEqual(data["diffs_label"], "hp_d")
            self.assertEqual(data["frames_done"], 3)
            self.assertEqual(data["frames_ok"], 2)
            self.assertEqual(data["phase"], "running")

    def test_dual_path_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws_path = Path(tmp) / "ws" / "hp2_d" / hp.PROGRESS_FILENAME
            cli_path = Path(tmp) / "diff.hotpants.progress.json"
            hp.init_progress_pair(
                ws_path,
                cli_path,
                diffs_label="hp2_d",
                round_id=1,
                science="ffi_wo_bkg",
                frames_total=2,
            )
            hp.record_frame_progress(ws_path, cli_path, success=True)
            hp.set_progress_phase_pair(ws_path, cli_path, "complete")

            ws = hp.read_progress(ws_path)
            cli = hp.read_progress(cli_path)
            assert ws is not None and cli is not None
            self.assertEqual(ws["diffs_label"], "hp2_d")
            self.assertEqual(cli["diffs_label"], "hp2_d")
            self.assertEqual(ws["frames_done"], 1)
            self.assertEqual(cli["phase"], "complete")
            self.assertEqual(
                hp.format_progress_text(ws),
                "hotpants hp2_d complete 1/2",
            )

    def test_progress_path_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "per_target" / "target" / "diff.log"
            log_path.parent.mkdir(parents=True)
            log_path.touch()
            diffs_dir = Path(tmp) / "ws" / "hp_d"
            self.assertEqual(
                hp.progress_path_for_diff_log(log_path).name,
                hp.CLI_PROGRESS_FILENAME,
            )
            self.assertEqual(
                hp.progress_path_for_diffs_workspace(diffs_dir).name,
                hp.PROGRESS_FILENAME,
            )


if __name__ == "__main__":
    unittest.main()
