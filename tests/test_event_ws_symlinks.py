"""Tests for event-level ws symlink handoff (templates, ffis)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    FFIS_WS_LABEL,
    TEMPLATES_WS_LABEL,
    ensure_event_ffis_symlink,
    ensure_event_templates_symlink,
    event_ffis_symlink_path,
    event_templates_symlink_path,
    prune_stale_per_workspace_ffis_symlinks,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    clear_diff_workspace,
    link_master_workspace,
    master_root,
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


class TestFfiHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.event_dir = self.root / "events" / "s0020_c3_k3"
        self.ffi_leaf = self.root / "data" / "tess_ffi" / "s0020" / "cam3_ccd3"
        self.ffi_leaf.mkdir(parents=True)
        (self.ffi_leaf / "tess2020-s0020-3-3-0165-s_ffic.fits").write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_event_ffis_symlink_path(self):
        self.assertEqual(
            event_ffis_symlink_path(self.event_dir),
            self.event_dir / "ws" / FFIS_WS_LABEL,
        )

    def test_ensure_creates_symlink_at_ws_root(self):
        link = ensure_event_ffis_symlink(self.event_dir, self.ffi_leaf)
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.ffi_leaf.resolve())
        self.assertFalse((self.event_dir / "ws" / "hp_d" / FFIS_WS_LABEL).exists())

    def test_ensure_idempotent(self):
        link1 = ensure_event_ffis_symlink(self.event_dir, self.ffi_leaf)
        link2 = ensure_event_ffis_symlink(self.event_dir, self.ffi_leaf)
        self.assertEqual(link1, link2)

    def test_prune_stale_per_workspace_ffis_symlinks(self):
        ws_root = self.event_dir / "ws"
        (ws_root / "hp_d").mkdir(parents=True)
        (ws_root / "hp_e").mkdir(parents=True)
        stale_d = ws_root / "hp_d" / FFIS_WS_LABEL
        stale_e = ws_root / "hp_e" / FFIS_WS_LABEL
        stale_d.symlink_to(self.ffi_leaf)
        stale_e.symlink_to(self.ffi_leaf)

        removed = prune_stale_per_workspace_ffis_symlinks(self.event_dir)
        self.assertEqual(removed, 2)
        self.assertFalse(stale_d.exists())
        self.assertFalse(stale_e.exists())

    def test_link_master_workspace_creates_ws_ffis_not_per_workspace(self):
        out = self.event_dir
        ws = out / "ws"
        hp = ws / "hp_d"
        hp.mkdir(parents=True)
        (hp / "tess2020_hp_d.fits").write_bytes(b"x")

        refreshed = link_master_workspace(str(out), ffi_leaf=str(self.ffi_leaf))
        self.assertGreaterEqual(refreshed, 2)

        ffis_link = event_ffis_symlink_path(out)
        self.assertTrue(ffis_link.is_symlink())
        self.assertEqual(ffis_link.resolve(), self.ffi_leaf.resolve())
        self.assertFalse((hp / FFIS_WS_LABEL).exists())
        self.assertTrue(os.path.islink(os.path.join(master_root(str(out)), "tess2020_hp_d.fits")))

    def test_clear_diff_workspace_restores_ffis_symlink(self):
        ensure_event_ffis_symlink(self.event_dir, self.ffi_leaf)
        (self.event_dir / "ws" / "hp_d").mkdir(parents=True)
        (self.event_dir / "ws" / "hp_d" / "x.fits").write_bytes(b"x")

        clear_diff_workspace(self.event_dir)
        link = event_ffis_symlink_path(self.event_dir)
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.ffi_leaf.resolve())


if __name__ == "__main__":
    unittest.main()
