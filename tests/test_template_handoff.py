"""Tests for event-level template symlink handoff."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.orchestration.template_handoff import (
    TEMPLATES_WS_LABEL,
    ensure_event_templates_symlink,
    event_templates_symlink_path,
)


class TestTemplateHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.event_dir = self.root / "events" / "s0020_c3_k3"
        self.physical = self.root / "data" / "shifted_downsampled" / "sector0020_camera3_ccd3"
        self.physical.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_event_templates_symlink_path(self):
        self.assertEqual(
            event_templates_symlink_path(self.event_dir),
            self.event_dir / "ws" / TEMPLATES_WS_LABEL,
        )

    def test_ensure_creates_symlink(self):
        link = ensure_event_templates_symlink(self.event_dir, self.physical)
        self.assertTrue(link.is_symlink())
        self.assertTrue(link.resolve() == self.physical.resolve())
        self.assertTrue((self.event_dir / "ws" / TEMPLATES_WS_LABEL).exists())

    def test_ensure_idempotent(self):
        link1 = ensure_event_templates_symlink(self.event_dir, self.physical)
        link2 = ensure_event_templates_symlink(self.event_dir, self.physical)
        self.assertEqual(link1, link2)
        self.assertTrue(link2.is_symlink())

    def test_ensure_refreshes_wrong_target(self):
        other = self.root / "data" / "other_templates"
        other.mkdir(parents=True)
        ensure_event_templates_symlink(self.event_dir, other)
        link = ensure_event_templates_symlink(self.event_dir, self.physical)
        self.assertEqual(link.resolve(), self.physical.resolve())

    def test_ensure_raises_if_blocking_file(self):
        self.event_dir.mkdir(parents=True)
        (self.event_dir / "ws").mkdir()
        blocker = self.event_dir / "ws" / TEMPLATES_WS_LABEL
        blocker.write_text("not a symlink")
        with self.assertRaises(FileExistsError):
            ensure_event_templates_symlink(self.event_dir, self.physical)

    def test_relative_symlink_target(self):
        link = ensure_event_templates_symlink(self.event_dir, self.physical)
        raw = os.readlink(link)
        self.assertFalse(os.path.isabs(raw))


if __name__ == "__main__":
    unittest.main()
