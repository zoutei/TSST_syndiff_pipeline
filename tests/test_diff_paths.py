"""Tests for difference-imaging workspace path helpers."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
    event_templates_symlink_path,
)
from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    absolutize_config,
)
from syndiff_pipeline.difference_imaging.stages.photometry import (
    write_lightcurve_diagnostic_plot,
)
from syndiff_pipeline.difference_imaging.support.paths import (
    KERNEL_RECONSTRUCTION_NPZ_BASENAME,
    MASTER_TESS_FFI_LINK,
    clear_diff_workspace,
    link_master_workspace,
    master_root,
    meta_workspace_dir_from_diffs_dir,
    meta_workspace_label,
    pipeline_plots_root,
)
from syndiff_pipeline.difference_imaging.stages.hotpants import (
    kernel_reconstruction_npz_path,
)


class TestPipelinePlotsRoot(unittest.TestCase):
    def test_default_subdir_under_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = os.path.join(tmp, "events", "s0020_c3_k3_2020ut")
            self.assertEqual(
                pipeline_plots_root(event),
                os.path.join(os.path.abspath(event), "ws", "debug_plots"),
            )

    def test_empty_subdir_returns_workspace_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            event = os.path.join(tmp, "event")
            self.assertEqual(
                pipeline_plots_root(event, ""),
                os.path.join(os.path.abspath(event), "ws"),
            )

    def test_absolutize_config_keeps_pipeline_plots_dir_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "config"
            site.mkdir()
            cfg = SynDiffConfig(
                output_dir=str(Path(tmp) / "events" / "test"),
                pipeline_plots_dir="debug_plots",
            )
            frozen = absolutize_config(cfg, site)
            self.assertEqual(frozen.pipeline_plots_dir, "debug_plots")
            self.assertEqual(
                pipeline_plots_root(frozen.output_dir, frozen.pipeline_plots_dir),
                str((Path(tmp) / "events" / "test" / "ws" / "debug_plots").resolve()),
            )


class TestLightcurveDiagnosticPlot(unittest.TestCase):
    def test_writes_empty_plot_when_all_flux_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame(
                {
                    "btjd": [100.0, 101.0, 102.0],
                    "flux": [float("nan")] * 3,
                    "eflux": [float("nan")] * 3,
                }
            )
            out = os.path.join(tmp, "lc_empty.png")
            path = write_lightcurve_diagnostic_plot(df, tmp, png_path=out, title_line="x")
            self.assertEqual(path, out)
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 1000)


class TestLinkMasterWorkspace(unittest.TestCase):
    def test_creates_absolute_fits_and_flat_ffi_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            hp = ws / "hp_d"
            hp.mkdir(parents=True)
            fits = hp / "tess2020_hp_d.fits"
            fits.write_bytes(b"SIMPLE  =                    T")

            ffi_leaf = Path(tmp) / "tess_ffi" / "s0020" / "cam3_ccd3"
            ffi_leaf.mkdir(parents=True)
            ffi = ffi_leaf / "tess2020-s0020-3-3-0165-s_ffic.fits"
            ffi.write_bytes(b"SIMPLE  =                    T")

            n1 = link_master_workspace(str(out), ffi_leaf=str(ffi_leaf))
            self.assertGreaterEqual(n1, 2)

            m_root = master_root(str(out))
            link_fits = os.path.join(m_root, "tess2020_hp_d.fits")
            self.assertTrue(os.path.islink(link_fits))
            self.assertEqual(os.readlink(link_fits), str(fits.resolve()))
            self.assertTrue(os.path.isfile(link_fits))

            ffi_link = os.path.join(m_root, ffi.name)
            self.assertTrue(os.path.islink(ffi_link))
            self.assertEqual(os.readlink(ffi_link), str(ffi.resolve()))
            self.assertFalse(os.path.exists(os.path.join(m_root, MASTER_TESS_FFI_LINK)))

            n2 = link_master_workspace(str(out), ffi_leaf=str(ffi_leaf))
            self.assertEqual(n2, 0)

    def test_creates_symlink_for_gzipped_ffi_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            ws.mkdir(parents=True)

            ffi_leaf = Path(tmp) / "tess_ffi" / "s0020" / "cam3_ccd3"
            ffi_leaf.mkdir(parents=True)
            ffi_gz = ffi_leaf / "tess2020-s0020-3-3-0165-s_ffic.fits.gz"
            ffi_gz.write_bytes(b"SIMPLE  =                    T")

            refreshed = link_master_workspace(str(out), ffi_leaf=str(ffi_leaf))
            self.assertGreaterEqual(refreshed, 1)

            m_root = master_root(str(out))
            link = os.path.join(m_root, ffi_gz.name)
            self.assertTrue(os.path.islink(link))
            self.assertEqual(os.readlink(link), str(ffi_gz.resolve()))
            self.assertTrue(os.path.isfile(link))

    def test_replaces_broken_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            hp = ws / "hp_d"
            hp.mkdir(parents=True)
            fits = hp / "frame.fits"
            fits.write_bytes(b"SIMPLE  =                    T")

            m_root = master_root(str(out))
            os.makedirs(m_root, exist_ok=True)
            stale = os.path.join(m_root, "frame.fits")
            os.symlink("/nonexistent/frame.fits", stale)

            refreshed = link_master_workspace(str(out))
            self.assertEqual(refreshed, 1)
            self.assertEqual(os.readlink(stale), str(fits.resolve()))
            self.assertTrue(os.path.isfile(stale))

    def test_removes_legacy_tess_ffi_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            hp = ws / "hp_d"
            hp.mkdir(parents=True)
            (hp / "a.fits").write_bytes(b"x")

            ffi_leaf = Path(tmp) / "tess_ffi" / "s0020" / "cam3_ccd3"
            ffi_leaf.mkdir(parents=True)
            ffi = ffi_leaf / "tess2020-s0020-3-3-0165-s_ffic.fits"
            ffi.write_bytes(b"x")

            m_root = master_root(str(out))
            os.makedirs(m_root, exist_ok=True)
            legacy = os.path.join(m_root, MASTER_TESS_FFI_LINK)
            os.symlink(str(Path(tmp) / "tess_ffi"), legacy)

            refreshed = link_master_workspace(str(out), ffi_leaf=str(ffi_leaf))
            self.assertGreaterEqual(refreshed, 2)
            self.assertFalse(os.path.exists(legacy))
            self.assertTrue(os.path.islink(os.path.join(m_root, ffi.name)))

    def test_prefers_gzip_when_both_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            hp = ws / "hp_d"
            hp.mkdir(parents=True)
            stem = "tess2020_hp_d"
            (hp / f"{stem}.fits").write_bytes(b"legacy")
            (hp / f"{stem}.fits.gz").write_bytes(b"gzip")

            link_master_workspace(str(out))
            m_root = master_root(str(out))
            link = os.path.join(m_root, f"{stem}.fits.gz")
            self.assertTrue(os.path.islink(link))
            self.assertEqual(os.readlink(link), str((hp / f"{stem}.fits.gz").resolve()))
            self.assertFalse(os.path.exists(os.path.join(m_root, f"{stem}.fits")))

    def test_skips_hotpants_stamp_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            hp = ws / "hp_d"
            stamps = ws / "hp_d_stamps"
            hp.mkdir(parents=True)
            stamps.mkdir(parents=True)
            (hp / "tess2020_hp_d.fits").write_bytes(b"x")
            (stamps / "tess2020_hp_d_stamps.fits").write_bytes(b"x")

            m_root = master_root(str(out))
            os.makedirs(m_root, exist_ok=True)
            stale = os.path.join(m_root, "tess2020_hp_d_stamps.fits")
            os.symlink(str((stamps / "tess2020_hp_d_stamps.fits").resolve()), stale)

            refreshed = link_master_workspace(str(out))
            self.assertGreaterEqual(refreshed, 1)
            self.assertTrue(os.path.islink(os.path.join(m_root, "tess2020_hp_d.fits")))
            self.assertFalse(os.path.exists(stale))
            self.assertEqual(
                sorted(os.listdir(m_root)),
                ["tess2020_hp_d.fits"],
            )

    def test_skips_master_subdir_when_scanning_workspaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            ws = out / "ws"
            hp = ws / "hp_d"
            hp.mkdir(parents=True)
            (hp / "a.fits").write_bytes(b"x")
            (ws / "master").mkdir(parents=True)

            refreshed = link_master_workspace(str(out))
            self.assertEqual(refreshed, 1)
            link_a = os.path.join(master_root(str(out)), "a.fits")
            self.assertTrue(os.path.islink(link_a))
            self.assertEqual(
                os.listdir(master_root(str(out))),
                ["a.fits"],
            )

    def test_clear_diff_workspace_restores_templates_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "event"
            physical = Path(tmp) / "templates_physical"
            physical.mkdir()
            ensure_event_templates_symlink(out, physical)
            (out / "ws" / "hp_d").mkdir(parents=True)
            (out / "ws" / "hp_d" / "x.fits").write_bytes(b"x")

            clear_diff_workspace(out)
            link = event_templates_symlink_path(out)
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), physical.resolve())


class TestMetaWorkspaceLabel(unittest.TestCase):
    def test_diffs_to_meta(self):
        self.assertEqual(meta_workspace_label("hp_d"), "hp_m")
        self.assertEqual(meta_workspace_label("ks_d"), "ks_m")

    def test_non_d_suffix(self):
        self.assertEqual(meta_workspace_label("diff_r1"), "diff_r1_m")

    def test_kernel_reconstruction_under_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            diffs = os.path.join(tmp, "ws", "hp_d")
            os.makedirs(diffs)
            path = kernel_reconstruction_npz_path(diffs)
            self.assertEqual(
                path,
                os.path.join(tmp, "ws", "hp_m", KERNEL_RECONSTRUCTION_NPZ_BASENAME),
            )
            self.assertEqual(
                meta_workspace_dir_from_diffs_dir(diffs),
                os.path.join(tmp, "ws", "hp_m"),
            )


if __name__ == "__main__":
    unittest.main()
