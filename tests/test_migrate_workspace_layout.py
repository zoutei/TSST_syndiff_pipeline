"""Tests for scripts/migrate_workspace_layout.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.migrate_workspace_layout import migrate_control_dir, normalize_events


class TestMigrateWorkspaceLayout(unittest.TestCase):
    def test_migrate_control_dir_moves_sqlite_and_daemon_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            (ws / "pipeline_state.sqlite").write_text("db", encoding="utf-8")
            (ws / "daemon.lock").write_text("", encoding="utf-8")
            moved, _ = migrate_control_dir(ws, dry_run=False)
            self.assertGreaterEqual(moved, 2)
            self.assertTrue((ws / "control" / "pipeline_state.sqlite").is_file())
            self.assertFalse((ws / "pipeline_state.sqlite").exists())

    def test_normalize_events_moves_flat_target_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            label = "s0023_c1_k3_2020ftl"
            target = ws / label
            target.mkdir(parents=True)
            (target / "cluster_template_job.json").write_text("{}", encoding="utf-8")
            moved, _ = normalize_events(ws, dry_run=False)
            self.assertEqual(moved, 1)
            self.assertTrue((ws / "events" / label / "cluster_template_job.json").is_file())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
