"""Tests for difference_imaging.orchestration.site_config."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target, load_targets
from syndiff_pipeline.common.scc_paths import event_scc_leaf, scc_ffi_dir, scc_templates_dir
from syndiff_pipeline.difference_imaging.orchestration.config import (
    SynDiffConfig,
    absolutize_config,
    load_config,
)
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    SitePaths,
    freeze_target_diff_config,
    load_diff_site_policy,
    resolve_diff_config,
    resolve_event_template_dir,
    write_frozen_diff_config,
)
from tests.site_fixtures import write_site_deployment


def _target() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=3,
        target_ra=210.219333,
        target_dec=81.846589,
        target_name="2020ut",
    )


def _write_diff_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "defaults:",
                "  n_jobs: 4",
                "paths:",
                "  template_base: shifted_downsampled",
                "pipeline:",
                "  - kind: shared_mask",
                "condor:",
                "  request_cpus: 4",
                "  request_memory: 32000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class TestSiteConfigLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"
        self.site.mkdir()
        self.handoff = self.root / "handoff"
        self.data = self.root / "data"
        write_site_deployment(
            self.site,
            workspace_root=str(self.handoff),
            data_root=str(self.data),
        )
        _write_diff_policy(self.site / "diff_config.yaml")
        target = _target()
        template_store = scc_templates_dir(self.data, target.sector, target.camera, target.ccd, oversampling_factor=1)
        template_store.mkdir(parents=True)
        (template_store / "template_manifest.json").write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_diff_site_policy(self):
        policy = load_diff_site_policy(self.site / "diff_config.yaml")
        self.assertEqual(policy.deployment_file, "deployment.yaml")
        self.assertEqual(len(policy.pipeline), 1)
        self.assertEqual(policy.defaults["n_jobs"], 4)
        self.assertEqual(policy.condor.request_cpus, 4)

    def test_load_diff_site_policy_rejects_legacy_condor_keys(self):
        path = self.site / "diff_config_legacy.yaml"
        path.write_text(
            (self.site / "diff_config.yaml").read_text(encoding="utf-8").replace(
                "request_memory: 32000",
                "request_memory: 32000\n  requirements: 'Memory >= 32000'",
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as ctx:
            load_diff_site_policy(path)
        self.assertIn("host_stats_min_mem_mb", str(ctx.exception))

    def test_load_diff_site_policy_rejects_empty_path(self):
        """Regression: an empty diff_config_path must raise, not resolve to CWD.

        A ``diff`` preset submitted with ``--config`` pointed straight at a
        diff_config.yaml (rather than a pipeline.yaml-style file that sets
        ``diff_config:``) leaves ``RunnerConfig.diff_config_path`` as ``""``.
        Previously ``Path("").resolve()`` silently became the process's CWD,
        so this only surfaced as a confusing IsADirectoryError deep inside
        Condor resource sizing, with affected stages requeuing forever.
        """
        with self.assertRaises(ValueError) as ctx:
            load_diff_site_policy("")
        self.assertIn("diff_config_path is empty", str(ctx.exception))

    def test_freeze_target_diff_config_absolutizes_paths(self):
        cfg = freeze_target_diff_config(self.site / "diff_config.yaml", _target())
        self.assertIsInstance(cfg, SynDiffConfig)
        self.assertTrue(Path(cfg.ffi_dir).is_absolute())
        self.assertTrue(Path(cfg.output_dir).is_absolute())
        self.assertTrue(Path(cfg.gaia_catalog).is_absolute())
        self.assertTrue(Path(cfg.template_dir).is_absolute())
        self.assertEqual(cfg.sector, 20)
        self.assertEqual(cfg.camera, 3)
        self.assertEqual(cfg.ccd, 3)
        self.assertEqual(cfg.target_ra, 210.219333)
        self.assertIn("events", cfg.output_dir)
        self.assertIn("2020ut", cfg.output_dir)
        self.assertIn("s0020_c3_k3", cfg.output_dir)
        self.assertEqual(cfg.ffi_dir, str((self.data / "s0020" / "c3" / "k3" / "ffi").resolve()))
        # Bundled straps/BSC are not injected; empty means packaged default at use time.
        self.assertEqual(cfg.straps_csv, "")
        self.assertEqual(cfg.bsc_catalog, "")
        self.assertEqual(
            cfg.gaia_catalog,
            str(
                (
                    self.data
                    / "s0020" / "c3" / "k3"
                    / "catalogs"
                    / "gaia_catalog_s0020_3_3.csv"
                ).resolve()
            ),
        )

    def test_prefers_gaia_catalog_pipeline_in_workspace(self):
        target = _target()
        event_dir = event_scc_leaf(self.handoff, target.event_name(), target.sector, target.camera, target.ccd)
        ws = event_dir / "ws"
        ws.mkdir(parents=True, exist_ok=True)
        pipeline_csv = ws / "gaia_catalog_pipeline.csv"
        pipeline_csv.write_text("x,y\n", encoding="utf-8")
        cfg = freeze_target_diff_config(self.site / "diff_config.yaml", target)
        self.assertEqual(cfg.gaia_catalog, str(pipeline_csv.resolve()))

    def test_write_frozen_diff_config_round_trip(self):
        cfg = freeze_target_diff_config(self.site / "diff_config.yaml", _target())
        out = self.root / "frozen" / "diff_config.yaml"
        write_frozen_diff_config(cfg, out)
        loaded = load_config(str(out))
        self.assertTrue(Path(loaded.output_dir).is_absolute())
        self.assertEqual(loaded.pipeline, cfg.pipeline)
        self.assertEqual(loaded.n_jobs, 4)

    def test_example_site_files_exist(self):
        example = SitePaths.from_site_dir(_ROOT / "config")
        self.assertTrue(example.template_config.is_file())
        self.assertTrue(example.diff_config.is_file())
        self.assertTrue(example.deployment_example.is_file())
        policy = load_diff_site_policy(example.diff_config)
        self.assertTrue(policy.pipeline)
        self.assertEqual(policy.condor.request_memory, 100_000)
        targets = load_targets(_ROOT / "config" / "targets_example.csv")
        self.assertGreater(len(targets), 0)

    def test_resolve_event_template_dir_via_scc_store(self):
        target = _target()
        resolved = resolve_event_template_dir(
            event_scc_leaf(self.handoff, target.event_name(), target.sector, target.camera, target.ccd),
            data_root=self.data,
            sector=target.sector,
            camera=target.camera,
            ccd=target.ccd,
        )
        self.assertTrue(resolved.endswith("s0020/c3/k3/templates/oversampling_1"))

    def test_resolve_diff_config_honors_template_store_name(self):
        target = _target()
        named = scc_templates_dir(
            self.data,
            target.sector,
            target.camera,
            target.ccd,
            oversampling_factor=1,
            store_name="no_l4b",
        )
        named.mkdir(parents=True)
        (named / "template_manifest.json").write_text("{}", encoding="utf-8")
        policy_path = self.site / "diff_config_named.yaml"
        policy_path.write_text(
            "\n".join(
                [
                    "deployment_file: deployment.yaml",
                    "defaults:",
                    "  n_jobs: 4",
                    "  oversampling_factor: 1",
                    "paths:",
                    "  template_store_name: no_l4b",
                    "pipeline:",
                    "  - kind: shared_mask",
                    "condor:",
                    "  request_cpus: 4",
                    "  request_memory: 32000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = freeze_target_diff_config(policy_path, target)
        self.assertTrue(cfg.template_dir.endswith("templates_no_l4b/oversampling_1"))
        self.assertEqual(Path(cfg.template_dir).resolve(), named.resolve())

    def test_resolve_event_template_dir_missing_raises(self):
        target = _target()
        with self.assertRaises(FileNotFoundError):
            resolve_event_template_dir(
                event_scc_leaf(self.handoff, "missing_target", target.sector, target.camera, target.ccd),
                data_root=self.data / "empty",
                sector=target.sector,
                camera=target.camera,
                ccd=target.ccd,
            )

    def test_absolutize_config_from_relative_paths(self):
        cfg = SynDiffConfig(
            ffi_dir="tess_ffi",
            output_dir="events/test",
            gaia_catalog="catalogs/sector_0020/camera_3/ccd_3/gaia.csv",
            template_dir="shifted_downsampled/sector0020_camera3_ccd3",
        )
        frozen = absolutize_config(cfg, self.data)
        self.assertEqual(frozen.ffi_dir, str((self.data / "tess_ffi").resolve()))
        self.assertEqual(frozen.output_dir, str((self.data / "events" / "test").resolve()))


if __name__ == "__main__":
    unittest.main()
