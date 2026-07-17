"""Tests for oversampling-aware downsample progress display."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.stage_progress import (
    _parse_downsample_sidecar,
)
from syndiff_pipeline.template_creation.processing.downsample_progress import (
    init_progress,
    mark_skycell_done,
)


class TestDownsampleOversampleProgressDisplay(unittest.TestCase):
    def test_sidecar_shows_os_factor_in_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "per_target" / "t1" / "downsample.log"
            log_path.parent.mkdir(parents=True)
            progress_path = log_path.parent / "downsample.progress.json"
            init_progress(progress_path, total_skycells=100, batch_sizes=[100], oversampling_factor=8)
            mark_skycell_done(progress_path, 0)

            prog = _parse_downsample_sidecar(log_path)
            self.assertIsNotNone(prog)
            assert prog is not None
            self.assertEqual(prog.text, "1/100 os8")
            self.assertEqual(prog.kind, "fraction")


if __name__ == "__main__":
    unittest.main()
