"""Tests for stage log capture and bounded tail reads."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import logs


class TestReadLogTail(unittest.TestCase):
    def test_returns_last_n_lines_without_reading_entire_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.log"
            lines = [f"line-{i}" for i in range(5000)]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tail = logs.read_log_tail(path, n_lines=3, max_bytes=4096)
            self.assertEqual(tail, "line-4997\nline-4998\nline-4999")

    def test_missing_file_returns_empty(self):
        self.assertEqual(logs.read_log_tail("/no/such/file.log"), "")


class TestStageLogFdRedirect(unittest.TestCase):
    def test_stage_log_captures_stdout_via_fd_redirect(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            run_id = "run_a"
            label = "s0001_c1_k1_test"
            stage = "tess_ffi_download"
            with logs.stage_log(str(runs_root), run_id, label, stage, {"foo": "bar"}):
                print("captured-line", flush=True)
            log_path = logs.target_log_path(str(runs_root), run_id, label, stage)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("captured-line", text)
            self.assertIn("STAGE: tess_ffi_download", text)
            self.assertIn("Exit code:", text)


if __name__ == "__main__":
    unittest.main()
