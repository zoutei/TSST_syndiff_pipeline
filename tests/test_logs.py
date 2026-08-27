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


class TestRotateAttemptArtifact(unittest.TestCase):
    def test_moves_nonempty_file_into_attempts_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diff.log"
            path.write_text("old attempt content\n", encoding="utf-8")
            logs.rotate_attempt_artifact(path, "tok-1")
            self.assertFalse(path.exists())
            archived = Path(tmp) / "attempts" / "tok-1" / "diff.log"
            self.assertTrue(archived.is_file())
            self.assertEqual(archived.read_text(encoding="utf-8"), "old attempt content\n")

    def test_noop_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diff.log"
            logs.rotate_attempt_artifact(path, "tok-1")  # must not raise
            self.assertFalse((Path(tmp) / "attempts").exists())

    def test_noop_when_file_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diff.log"
            path.write_text("", encoding="utf-8")
            logs.rotate_attempt_artifact(path, "tok-1")
            self.assertTrue(path.is_file())
            self.assertFalse((Path(tmp) / "attempts").exists())

    def test_noop_when_launch_token_falsy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diff.log"
            path.write_text("content\n", encoding="utf-8")
            logs.rotate_attempt_artifact(path, None)
            self.assertTrue(path.is_file())
            self.assertFalse((Path(tmp) / "attempts").exists())


class TestStageLogRotatesPriorAttempt(unittest.TestCase):
    def test_second_attempt_archives_first_under_its_own_prior_token(self):
        # Mirrors real usage: the caller (run_stage.main) reads the prior
        # attempt's own token from status.json *before* this attempt's
        # status write clobbers it, then passes that (not this attempt's
        # own token) into stage_log.
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            run_id = "run_a"
            label = "s0001_c1_k1_test"
            stage = "diff"

            with logs.stage_log(str(runs_root), run_id, label, stage, {}):
                print("first-attempt-output", flush=True)

            with logs.stage_log(
                str(runs_root), run_id, label, stage, {}, prior_launch_token="tok-1"
            ):
                print("second-attempt-output", flush=True)

            log_path = logs.target_log_path(str(runs_root), run_id, label, stage)
            current = log_path.read_text(encoding="utf-8")
            self.assertIn("second-attempt-output", current)
            self.assertNotIn("first-attempt-output", current)

            archived = log_path.parent / "attempts" / "tok-1" / "diff.log"
            self.assertTrue(archived.is_file())
            self.assertIn("first-attempt-output", archived.read_text(encoding="utf-8"))

    def test_no_prior_token_keeps_legacy_truncate_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_root = Path(tmp) / "runs"
            run_id = "run_a"
            label = "s0001_c1_k1_test"
            stage = "diff"

            with logs.stage_log(str(runs_root), run_id, label, stage, {}):
                print("first", flush=True)
            with logs.stage_log(str(runs_root), run_id, label, stage, {}):
                print("second", flush=True)

            log_path = logs.target_log_path(str(runs_root), run_id, label, stage)
            current = log_path.read_text(encoding="utf-8")
            self.assertIn("second", current)
            self.assertNotIn("first", current)
            self.assertFalse((log_path.parent / "attempts").exists())


class TestReadPreviousLaunchToken(unittest.TestCase):
    def test_returns_none_when_status_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                logs.read_previous_launch_token(tmp, "run_a", "target", "diff")
            )

    def test_returns_token_from_status_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = logs.stage_status_path(tmp, "run_a", "target", "diff")
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                '{"launch_token": "abc-123", "state": "running"}', encoding="utf-8"
            )
            self.assertEqual(
                logs.read_previous_launch_token(tmp, "run_a", "target", "diff"),
                "abc-123",
            )

    def test_returns_none_on_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = logs.stage_status_path(tmp, "run_a", "target", "diff")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(
                logs.read_previous_launch_token(tmp, "run_a", "target", "diff")
            )

    def test_run_stage_reads_prior_token_before_overwriting_status(self):
        # End-to-end-ish: simulates run_stage.main's own ordering -- read
        # the prior token, THEN write the new attempt's status, and
        # confirm the value captured beforehand still reflects the OLD
        # attempt, not the one that just overwrote status.json.
        with tempfile.TemporaryDirectory() as tmp:
            status_path = logs.stage_status_path(tmp, "run_a", "target", "diff")
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                '{"launch_token": "old-token", "state": "exited"}', encoding="utf-8"
            )

            prior = logs.read_previous_launch_token(tmp, "run_a", "target", "diff")
            status_path.write_text(
                '{"launch_token": "new-token", "state": "running"}', encoding="utf-8"
            )

            self.assertEqual(prior, "old-token")
            self.assertEqual(
                logs.read_previous_launch_token(tmp, "run_a", "target", "diff"),
                "new-token",
            )


if __name__ == "__main__":
    unittest.main()
