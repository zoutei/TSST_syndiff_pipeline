"""Tests for diff-aware verify and manifest plumbing (R1)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import logs
from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.support.manifest import manifest_path_from_output_dir
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    parse_stage_params,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    MANIFEST_SCHEMA_VERSION,
    config_fingerprint,
    manifest_valid,
    stage_complete,
    verify_stage,
    write_stable_manifest,
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
                "  n_jobs: 2",
                "paths:",
                "  template_base: shifted_downsampled",
                "pipeline:",
                "  - kind: shared_mask",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class TestVerifyDiffCli(unittest.TestCase):
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
        template_leaf = self.data / "shifted_downsampled" / "sector0020_camera3_ccd3"
        template_leaf.mkdir(parents=True)
        (template_leaf / "group_1").mkdir()
        (template_leaf / "group_1" / "ps1_template.fits").write_bytes(b"SIMPLE  = T")

        self.target = _target()
        self.event_dir = self.handoff / "events" / self.target.label()
        ensure_event_templates_symlink(self.event_dir, template_leaf)
        (self.event_dir / "cluster_template_job.json").write_text(
            json.dumps({"reference_ffi_path": "/tmp/ref.fits"}),
            encoding="utf-8",
        )

        self.runner = RunnerConfig(
            workspace_root=str(self.handoff),
            runs_root=str(self.handoff / "runs"),
            diff_config_path=str(self.site / "diff_config.yaml"),
            stages=parse_stage_params({"diff": {"executor": "condor"}}),
        )
        self.resolved = resolve_config(self.target, self.runner)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_diff_outputs(self) -> None:
        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")
        ws_root = self.event_dir / "ws"
        ws_root.mkdir(parents=True, exist_ok=True)
        (ws_root / SHARED_MASK_FITS_BASENAME).write_bytes(b"SIMPLE  = T")

    def test_verify_stage_diff_with_runner_cfg(self):
        self._write_diff_outputs()
        result = verify_stage(self.resolved, "diff", runner_cfg=self.runner)
        self.assertTrue(result.ok)
        self.assertEqual(result.stage, "diff")

    def test_write_stable_manifest_diff_does_not_raise(self):
        self._write_diff_outputs()
        stable_path = logs.stable_stage_manifest_path(
            self.runner.runs_dir(), self.target.label(), "diff"
        )
        write_stable_manifest(
            self.resolved,
            "diff",
            stable_path,
            runner_cfg=self.runner,
        )
        self.assertTrue(Path(stable_path).is_file())

    def test_verify_diff_missing_manifest_csv(self):
        ws_hp = self.event_dir / "ws" / "hp_d"
        ws_hp.mkdir(parents=True)
        (ws_hp / "frame.fits").write_bytes(b"SIMPLE  = T")

        result = verify_stage(self.resolved, "diff", runner_cfg=self.runner)
        self.assertFalse(result.ok)
        self.assertIn("Missing frame manifest CSV", result.message)

    def test_verify_diff_partial_ws_only_master(self):
        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")
        master = self.event_dir / "ws" / "master"
        master.mkdir(parents=True)
        (master / "placeholder.fits").write_bytes(b"SIMPLE  = T")

        result = verify_stage(self.resolved, "diff", runner_cfg=self.runner)
        self.assertFalse(result.ok)
        self.assertIn("Final pipeline outputs missing", result.message)

    def test_stage_complete_diff_stale_fingerprint(self):
        artifact = self.event_dir / "ws" / "hp_d" / "frame.fits"
        manifest_path = logs.stage_manifest_path(
            self.runner.runs_dir(), "run_a", self.target.label(), "diff"
        )
        stale_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stage": "diff",
            "config_fingerprint": "stale",
            "expected_count": 1,
            "produced_count": 1,
            "artifacts": [str(artifact)],
        }
        logs.write_json_atomic(manifest_path, stale_manifest)

        self.assertFalse(
            manifest_valid(
                stale_manifest,
                self.resolved,
                "diff",
                runner_cfg=self.runner,
            )
        )
        self.assertFalse(
            stage_complete(
                self.resolved,
                "diff",
                manifest_path=str(manifest_path),
                runner_cfg=self.runner,
            )
        )
        self.assertNotEqual(
            config_fingerprint(self.resolved, "diff", runner_cfg=self.runner),
            "stale",
        )

    def test_stage_complete_diff_manifest_first(self):
        self._write_diff_outputs()
        artifact = self.event_dir / "ws" / "hp_d" / "frame.fits"
        manifest_path = logs.stage_manifest_path(
            self.runner.runs_dir(), "run_a", self.target.label(), "diff"
        )
        logs.write_json_atomic(
            manifest_path,
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "stage": "diff",
                "config_fingerprint": config_fingerprint(
                    self.resolved, "diff", runner_cfg=self.runner
                ),
                "expected_count": 1,
                "produced_count": 1,
                "artifacts": [str(artifact)],
            },
        )

        self.assertTrue(
            stage_complete(
                self.resolved,
                "diff",
                manifest_path=str(manifest_path),
                runner_cfg=self.runner,
            )
        )


if __name__ == "__main__":
    unittest.main()
