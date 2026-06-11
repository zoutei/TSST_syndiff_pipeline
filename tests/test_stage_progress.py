"""Tests for log-derived stage progress parsing."""
from __future__ import annotations

import argparse
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.cli import cmd_progress
from syndiff_pipeline.template_creation.orchestration.stage_progress import read_log_progress


class TestReadLogProgress(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.log_dir = Path(self.tmpdir.name)

    def _write_log(self, name: str, content: str) -> Path:
        path = self.log_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_ps1_download_finished_skycell(self):
        path = self._write_log(
            "ps1.log",
            "2026-06-08 INFO Finished skycell rings.v3.skycell.1234.567 (100/900)\n",
        )
        prog = read_log_progress(path, "ps1_download")
        self.assertIsNotNone(prog)
        self.assertEqual(prog.text, "100/900")
        self.assertEqual(prog.kind, "fraction")

    def test_ps1_download_dask_fallback(self):
        path = self._write_log(
            "ps1.log",
            "2026-06-08 INFO Dask progress: 342/1009 skycells finished (elapsed 120s)\n",
        )
        prog = read_log_progress(path, "ps1_download")
        self.assertEqual(prog.text, "342/1009")

    def test_ps1_download_total_only(self):
        path = self._write_log(
            "ps1.log",
            "2026-06-08 INFO Found 1009 total skycells to process\n",
        )
        prog = read_log_progress(path, "ps1_download")
        self.assertEqual(prog.text, "0/1009")

    def test_ps1_process_projection_row_progress(self):
        path = self._write_log(
            "ps1_pr.log",
            "\n".join(
                [
                    "INFO [Pipeline] Processing 19 projections",
                    # 2 projections completed; working on the 3rd (row 5/10)
                    "INFO [Pipeline] Progress: projection 2/19 row 5/10",
                ]
            ),
        )
        prog = read_log_progress(path, "ps1_process")
        self.assertEqual(prog.text, "2/19 projections 5/10 rows")
        self.assertEqual(prog.kind, "fraction")

    def test_ps1_process_fallback_counts_finished_projections(self):
        path = self._write_log(
            "ps1_pr.log",
            "\n".join(
                [
                    "INFO [Pipeline] Processing 19 projections",
                    "INFO [SequentialProcessor] --- Finished sequential processing for projection: 1920 ---",
                    "INFO [SequentialProcessor] --- Finished sequential processing for projection: 1921 ---",
                    "INFO [SequentialProcessor] --- Starting sequential processing for projection: 1922 ---",
                    "INFO [SequentialProcessor] --- Processing step for row 5/10: ROW ID 42 ---",
                ]
            ),
        )
        prog = read_log_progress(path, "ps1_process")
        self.assertEqual(prog.text, "2/19 projections 5/10 rows")

    def test_downsample_batch_progress(self):
        path = self._write_log(
            "down.log",
            "\n".join(
                [
                    "Processing 84 skycells in 12 batches...",
                    "Completed batch 1",
                    "Completed batch 2",
                    "Completed batch 3",
                ]
            ),
        )
        prog = read_log_progress(path, "downsample")
        self.assertEqual(prog.text, "~21/84")
        self.assertEqual(prog.kind, "fraction")

    def test_downsample_out_of_order_batch_fallback(self):
        path = self._write_log(
            "down.log",
            "\n".join(
                [
                    "Processing 84 skycells in 12 batches...",
                    "Completed batch 12",
                    "Completed batch 3",
                ]
            ),
        )
        prog = read_log_progress(path, "downsample")
        self.assertEqual(prog.text, "~14/84")

    def test_diff_hotpants_sidecar(self):
        log_path = self._write_log("diff.log", "Stage: hotpants\n")
        sidecar = log_path.parent / "diff.hotpants.progress.json"
        sidecar.write_text(
            (
                '{"diffs_label": "hp_d", "round_id": 1, "science": "ffi", '
                '"frames_total": 200, "frames_done": 45, "frames_ok": 44, '
                '"phase": "running"}\n'
            ),
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "diff")
        self.assertEqual(prog.text, "hotpants hp_d 45/200")
        self.assertEqual(prog.kind, "fraction")

    def test_diff_hotpants_log_fallback(self):
        path = self._write_log(
            "diff.log",
            "hotpants [hp2_d] round 1: 180/200 frames succeeded.\n",
        )
        prog = read_log_progress(path, "diff")
        self.assertEqual(prog.text, "hotpants hp2_d complete 180/200")
        self.assertEqual(prog.kind, "phase")

    def test_diff_photometry_sidecar(self):
        log_path = self._write_log("diff.log", "Stage: forced_photometry\n")
        sidecar = log_path.parent / "diff.photometry.progress.json"
        sidecar.write_text(
            (
                '{"output_label": "lc_prf_on_diffs", "diffs_input": "hp_d", '
                '"n_sources": 1, "epochs_total": 842, "epochs_done": 120, '
                '"phase": "flux", "updated_at": "2026-06-11T12:00:00+00:00"}\n'
            ),
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "diff")
        self.assertEqual(prog.text, "photometry lc_prf_on_diffs 120/842")
        self.assertEqual(prog.kind, "fraction")

    def test_diff_separate_sidecars_from_shared_log_path(self):
        """Hotpants and photometry must not share one CLI sidecar filename."""
        from syndiff_pipeline.difference_imaging.stages import (
            hotpants_progress as hp,
            photometry_progress as pp,
        )

        log_path = self._write_log("diff.log", "Stage: forced_photometry\n")
        hotpants_cli = hp.progress_path_for_diff_log(log_path)
        photometry_cli = pp.progress_path_for_diff_log(log_path)
        self.assertNotEqual(hotpants_cli, photometry_cli)

        ws_hp = self.log_dir / "ws" / "hp_d" / hp.PROGRESS_FILENAME
        ws_phot = self.log_dir / "ws" / "lc_prf_on_diffs" / pp.PROGRESS_FILENAME

        hp.init_progress_pair(
            ws_hp,
            hotpants_cli,
            diffs_label="hp_d",
            round_id=1,
            science="ffi",
            frames_total=1188,
        )
        hp.record_frame_progress(ws_hp, hotpants_cli, success=True)
        hp.set_progress_phase_pair(ws_hp, hotpants_cli, "complete")

        pp.init_progress_pair(
            ws_phot,
            photometry_cli,
            output_label="lc_prf_on_diffs",
            diffs_input="hp_d",
            n_sources=7,
            epochs_total=1188,
            phase="flux",
        )
        pp.record_epoch_progress(ws_phot, photometry_cli)

        self.assertTrue(hotpants_cli.is_file())
        self.assertTrue(photometry_cli.is_file())
        hp_data = hp.read_progress(hotpants_cli)
        pp_data = pp.read_progress(photometry_cli)
        self.assertIsNotNone(hp_data)
        self.assertIsNotNone(pp_data)
        self.assertIn("frames_total", hp_data)
        self.assertIn("epochs_total", pp_data)
        self.assertNotIn("epochs_total", hp_data)

        prog = read_log_progress(log_path, "diff")
        self.assertEqual(prog.text, "photometry lc_prf_on_diffs (7 src) 1/1188")
        self.assertEqual(prog.kind, "fraction")

    def test_diff_sidecar_prefers_newer_photometry(self):
        log_path = self._write_log("diff.log", "Stage: forced_photometry\n")
        hotpants_sidecar = log_path.parent / "diff.hotpants.progress.json"
        photometry_sidecar = log_path.parent / "diff.photometry.progress.json"
        hotpants_sidecar.write_text(
            (
                '{"diffs_label": "hp_d", "round_id": 1, "science": "ffi", '
                '"frames_total": 200, "frames_done": 200, "frames_ok": 200, '
                '"phase": "complete", "updated_at": "2026-06-11T11:00:00+00:00"}\n'
            ),
            encoding="utf-8",
        )
        photometry_sidecar.write_text(
            (
                '{"output_label": "lc_prf_on_diffs", "diffs_input": "hp_d", '
                '"n_sources": 1, "epochs_total": 842, "epochs_done": 50, '
                '"phase": "flux", "updated_at": "2026-06-11T12:00:00+00:00"}\n'
            ),
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "diff")
        self.assertEqual(prog.text, "photometry lc_prf_on_diffs 50/842")

    def test_downsample_sidecar_skycell_progress(self):
        log_path = self._write_log("downsample.log", "Processing 84 skycells in 12 batches...\n")
        sidecar = log_path.parent / "downsample.progress.json"
        sidecar.write_text(
            '{"total_skycells": 84, "skycells_done": 45, "phase": "parallel_batches"}\n',
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "45/84")
        self.assertEqual(prog.kind, "fraction")

    def test_downsample_sidecar_combining_phase(self):
        log_path = self._write_log("downsample.log", "Combining results...\n")
        sidecar = log_path.parent / "downsample.progress.json"
        sidecar.write_text(
            '{"total_skycells": 84, "skycells_done": 84, "phase": "combining"}\n',
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "combining")
        self.assertEqual(prog.kind, "phase")

    def test_downsample_sidecar_precomputing_shifts_zero_done(self):
        log_path = self._write_log("downsample.log", "Precomputing shifts for all offsets...\n")
        sidecar = log_path.parent / "downsample.progress.json"
        sidecar.write_text(
            '{"phase": "precomputing_shifts", "offsets_done": 0, "offsets_total": 10}\n',
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "shifts 0/10")
        self.assertEqual(prog.kind, "phase")

    def test_downsample_sidecar_precomputing_shifts(self):
        log_path = self._write_log("downsample.log", "Precomputing shifts for all offsets...\n")
        sidecar = log_path.parent / "downsample.progress.json"
        sidecar.write_text(
            '{"phase": "precomputing_shifts", "offsets_done": 3, "offsets_total": 10}\n',
            encoding="utf-8",
        )
        prog = read_log_progress(log_path, "downsample")
        self.assertEqual(prog.text, "shifts 3/10")
        self.assertEqual(prog.kind, "phase")

    def test_downsample_combining_phase(self):
        path = self._write_log(
            "down.log",
            "Completed batch 12\nCombining results...\n",
        )
        prog = read_log_progress(path, "downsample")
        self.assertEqual(prog.text, "combining")
        self.assertEqual(prog.kind, "phase")

    def test_tess_ffi_download_progress(self):
        path = self._write_log(
            "tess.log",
            "\n".join(
                [
                    "INFO Downloading 45 FITS file(s) to /data/tess ...",
                    "INFO FFI download progress: 20/45",
                ]
            ),
        )
        prog = read_log_progress(path, "tess_ffi_download")
        self.assertEqual(prog.text, "20/45")

    def test_mapping_phase(self):
        path = self._write_log("map.log", "MOC filtering complete.\n")
        prog = read_log_progress(path, "mapping")
        self.assertEqual(prog.text, "moc_filter")

    def test_wcs_grouping_elapsed_without_log(self):
        prog = read_log_progress(
            self.log_dir / "missing.log",
            "wcs_grouping",
            started_at="2020-01-01T00:00:00+00:00",
        )
        self.assertIsNotNone(prog)
        self.assertEqual(prog.kind, "elapsed")
        self.assertTrue(prog.text.endswith("m") or prog.text.endswith("s"))

    def test_missing_log_returns_none(self):
        prog = read_log_progress(self.log_dir / "nope.log", "ps1_download")
        self.assertIsNone(prog)

    def test_tail_reads_end_of_large_log(self):
        path = self.log_dir / "big.log"
        padding = "x" * 100_000
        path.write_text(padding + "Finished skycell foo (999/1000)\n", encoding="utf-8")
        prog = read_log_progress(path, "ps1_download", tail_bytes=4096)
        self.assertEqual(prog.text, "999/1000")

    def test_ps1_process_scans_past_verbose_tail(self):
        noise_line = "INFO [Gather] waiting for cell bundle\n"
        trailing_noise = noise_line * 4000
        progress_line = "INFO [Pipeline] Progress: projection 3/19 row 5/10\n"
        path = self._write_log(
            "ps1_pr.log",
            (noise_line * 2000) + progress_line + trailing_noise,
        )
        prog = read_log_progress(path, "ps1_process", tail_bytes=65536)
        self.assertIsNotNone(prog)
        self.assertEqual(prog.text, "3/19 projections 5/10 rows")


class TestCmdProgressDetail(unittest.TestCase):
    def test_prints_running_detail(self):
        buf = io.StringIO()
        args = argparse.Namespace(run_dir="/run", run_id="run_a", no_detail=False)

        fake_ctx = mock.Mock()
        fake_ctx.run_id = "run_a"
        fake_ctx.cfg.state_db_path = "/db.sqlite"
        fake_ctx.cfg.workspace_root = "/tmp/handoff"
        fake_ctx.cfg.runs_dir.return_value = "/runs"

        running_row = mock.Mock(
            target_label="s0041_c1_k2_2021udg",
            stage="ps1_download",
            log_path="/runs/run_a/per_target/s0041/ps1_download.log",
            started_at=None,
        )

        fake_state = mock.Mock()
        fake_state.count_by_status.return_value = {"running": 1, "success": 5}
        fake_state.get_run.return_value = {"status": "running"}
        fake_state.running_stage_runs.return_value = [running_row]

        with mock.patch(
            "syndiff_pipeline.common.orchestration.cli._resolve_run_from_args",
            return_value=fake_ctx,
        ), mock.patch(
            "syndiff_pipeline.common.orchestration.cli.PipelineState",
            return_value=fake_state,
        ), mock.patch(
            "syndiff_pipeline.common.orchestration.verify_status.read_verify_run_status",
            return_value={"scan_queued": 0, "scan_running": 0, "active": []},
        ), mock.patch(
            "syndiff_pipeline.template_creation.orchestration.run_report.read_log_progress",
            return_value=mock.Mock(text="342/1009", kind="fraction"),
        ), mock.patch("sys.stdout", buf):
            rc = cmd_progress(args)

        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("[run_a] status =", out)
        self.assertIn("s0041_c1_k2_2021udg ps1_dl: 342/1009", out)

    def test_no_detail_flag(self):
        buf = io.StringIO()
        args = argparse.Namespace(run_dir="/run", run_id="run_a", no_detail=True)

        fake_ctx = mock.Mock()
        fake_ctx.run_id = "run_a"
        fake_ctx.cfg.state_db_path = "/db.sqlite"
        fake_ctx.cfg.workspace_root = "/tmp/handoff"

        fake_state = mock.Mock()
        fake_state.count_by_status.return_value = {"running": 1}
        fake_state.get_run.return_value = {"status": "running"}

        with mock.patch(
            "syndiff_pipeline.common.orchestration.cli._resolve_run_from_args",
            return_value=fake_ctx,
        ), mock.patch(
            "syndiff_pipeline.common.orchestration.cli.PipelineState",
            return_value=fake_state,
        ), mock.patch(
            "syndiff_pipeline.common.orchestration.verify_status.read_verify_run_status",
            return_value={"scan_queued": 0, "scan_running": 0, "active": []},
        ), mock.patch("sys.stdout", buf):
            rc = cmd_progress(args)

        self.assertEqual(rc, 0)
        self.assertNotIn("ps1_dl:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
