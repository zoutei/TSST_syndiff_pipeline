"""Tests for centroids progress sidecar helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from syndiff_pipeline.difference_imaging.stages import centroids_progress as cp


class TestCentroidsProgress(unittest.TestCase):
    def test_init_and_mark_frame_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ws" / "centroids_r1" / cp.PROGRESS_FILENAME
            cp.init_progress(
                path,
                centroids_label="centroids_r1",
                diffs_input="hp_d",
                frames_total=3,
            )
            cp.mark_frame_done(path, success=True)
            cp.mark_frame_done(path, success=False)
            cp.mark_frame_done(path, success=True)

            data = cp.read_progress(path)
            assert data is not None
            self.assertEqual(data["centroids_label"], "centroids_r1")
            self.assertEqual(data["frames_done"], 3)
            self.assertEqual(data["frames_ok"], 2)
            self.assertEqual(data["phase"], "running")

    def test_dual_path_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws_path = Path(tmp) / "ws" / "centroids_r1" / cp.PROGRESS_FILENAME
            cli_path = Path(tmp) / "diff.centroids.progress.json"
            cp.init_progress_pair(
                ws_path,
                cli_path,
                centroids_label="centroids_r1",
                diffs_input="hp_d",
                frames_total=2,
            )
            cp.record_frame_progress(ws_path, cli_path, success=True)
            cp.set_progress_phase_pair(ws_path, cli_path, "complete")

            ws = cp.read_progress(ws_path)
            cli = cp.read_progress(cli_path)
            assert ws is not None and cli is not None
            self.assertEqual(ws["centroids_label"], "centroids_r1")
            self.assertEqual(cli["centroids_label"], "centroids_r1")
            self.assertEqual(ws["frames_done"], 1)
            self.assertEqual(cli["phase"], "complete")
            self.assertEqual(
                cp.format_progress_text(ws),
                "centroids centroids_r1 complete 1/2",
            )

    def test_merge_progress_with_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "centroids_r1"
            ws.mkdir(parents=True)
            sidecar = ws / cp.PROGRESS_FILENAME
            cp.init_progress(
                sidecar,
                centroids_label="centroids_r1",
                diffs_input="hp_d",
                frames_total=10,
                output_dir=str(ws),
            )
            cp.mark_frame_done(sidecar, success=True)
            (ws / "tess111_photresults.ecsv").write_bytes(b"x")
            (ws / "tess222_photresults.ecsv").write_bytes(b"x")
            (ws / "tess333_photresults.ecsv").write_bytes(b"x")
            merged = cp.read_progress_merged(sidecar, force_artifact_merge=True)
            assert merged is not None
            self.assertEqual(merged["frames_done"], 3)
            self.assertEqual(merged["frames_ok"], 3)
            self.assertEqual(
                cp.format_progress_text(merged),
                "centroids centroids_r1 3/10",
            )

    def test_fresh_sidecar_skips_artifact_glob(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "centroids_r1"
            ws.mkdir(parents=True)
            sidecar = ws / cp.PROGRESS_FILENAME
            cp.init_progress(
                sidecar,
                centroids_label="centroids_r1",
                diffs_input="hp_d",
                frames_total=10,
                output_dir=str(ws),
            )
            cp.mark_frame_done(sidecar, success=True)
            (ws / "tess111_photresults.ecsv").write_bytes(b"x")
            (ws / "tess222_photresults.ecsv").write_bytes(b"x")
            with mock.patch.object(cp, "count_photresults_artifacts") as count:
                merged = cp.read_progress_merged(sidecar)
            count.assert_not_called()
            assert merged is not None
            self.assertEqual(merged["frames_done"], 1)

    def test_progress_path_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "per_target" / "target" / "diff.log"
            log_path.parent.mkdir(parents=True)
            log_path.touch()
            output_dir = Path(tmp) / "ws" / "centroids_r1"
            ws_path = cp.progress_path_for_output_workspace(output_dir)
            self.assertEqual(
                cp.progress_path_for_diff_log(log_path).name,
                cp.CLI_PROGRESS_FILENAME,
            )
            self.assertEqual(ws_path.name, cp.PROGRESS_FILENAME)
            self.assertEqual(ws_path.parent.name, "centroids_r1")


if __name__ == "__main__":
    unittest.main()
