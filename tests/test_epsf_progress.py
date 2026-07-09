"""Tests for ePSF progress sidecar helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from syndiff_pipeline.difference_imaging.stages import epsf_progress as ep


class TestEpsfProgress(unittest.TestCase):
    def test_init_and_mark_frame_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ws" / "epsf_r1" / ep.PROGRESS_FILENAME
            ep.init_progress(
                path,
                epsf_label="epsf_r1",
                diffs_input="hp_d",
                round_id=1,
                frames_total=3,
            )
            ep.mark_frame_done(path, success=True)
            ep.mark_frame_done(path, success=False)
            ep.mark_frame_done(path, success=True)

            data = ep.read_progress(path)
            assert data is not None
            self.assertEqual(data["epsf_label"], "epsf_r1")
            self.assertEqual(data["frames_done"], 3)
            self.assertEqual(data["frames_ok"], 2)
            self.assertEqual(data["phase"], "running")

    def test_dual_path_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws_path = Path(tmp) / "ws" / "epsf_r1" / ep.PROGRESS_FILENAME
            cli_path = Path(tmp) / "diff.epsf.progress.json"
            ep.init_progress_pair(
                ws_path,
                cli_path,
                epsf_label="epsf_r1",
                diffs_input="hp_d",
                round_id=1,
                frames_total=2,
            )
            ep.record_frame_progress(ws_path, cli_path, success=True)
            ep.set_progress_phase_pair(ws_path, cli_path, "complete")

            ws = ep.read_progress(ws_path)
            cli = ep.read_progress(cli_path)
            assert ws is not None and cli is not None
            self.assertEqual(ws["epsf_label"], "epsf_r1")
            self.assertEqual(cli["epsf_label"], "epsf_r1")
            self.assertEqual(ws["frames_done"], 1)
            self.assertEqual(cli["phase"], "complete")
            self.assertEqual(
                ep.format_progress_text(ws),
                "epsf epsf_r1 complete 1/2",
            )

    def test_merge_progress_with_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "epsf_r1"
            ws.mkdir(parents=True)
            sidecar = ws / ep.PROGRESS_FILENAME
            ep.init_progress(
                sidecar,
                epsf_label="epsf_r1",
                diffs_input="hp_d",
                round_id=1,
                frames_total=10,
                output_dir=str(ws),
            )
            ep.mark_frame_done(sidecar, success=True)
            (ws / "tess111_gridded_epsf.npz").write_bytes(b"x")
            (ws / "tess222_gridded_epsf.npz").write_bytes(b"x")
            (ws / "tess333_gridded_epsf.npz").write_bytes(b"x")
            merged = ep.read_progress_merged(sidecar, force_artifact_merge=True)
            assert merged is not None
            self.assertEqual(merged["frames_done"], 3)
            self.assertEqual(merged["frames_ok"], 3)
            self.assertEqual(
                ep.format_progress_text(merged),
                "epsf epsf_r1 3/10",
            )

    def test_fresh_sidecar_skips_artifact_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "epsf_r1"
            ws.mkdir(parents=True)
            sidecar = ws / ep.PROGRESS_FILENAME
            ep.init_progress(
                sidecar,
                epsf_label="epsf_r1",
                diffs_input="hp_d",
                round_id=1,
                frames_total=10,
                output_dir=str(ws),
            )
            ep.mark_frame_done(sidecar, success=True)
            (ws / "tess111_gridded_epsf.npz").write_bytes(b"x")
            (ws / "tess222_gridded_epsf.npz").write_bytes(b"x")
            with mock.patch.object(ep, "count_gridded_epsf_artifacts") as count:
                merged = ep.read_progress_merged(sidecar)
            count.assert_not_called()
            assert merged is not None
            self.assertEqual(merged["frames_done"], 1)

    def test_progress_path_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "per_target" / "target" / "diff.log"
            log_path.parent.mkdir(parents=True)
            log_path.touch()
            output_dir = Path(tmp) / "ws" / "epsf_r1"
            ws_path = ep.progress_path_for_output_workspace(output_dir)
            self.assertEqual(
                ep.progress_path_for_diff_log(log_path).name,
                ep.CLI_PROGRESS_FILENAME,
            )
            self.assertEqual(ws_path.name, ep.PROGRESS_FILENAME)
            self.assertEqual(ws_path.parent.name, "epsf_r1")


if __name__ == "__main__":
    unittest.main()
