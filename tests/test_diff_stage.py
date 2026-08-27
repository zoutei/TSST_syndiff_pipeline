"""Tests for the diff pipeline stage (WP-I)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import condor
from syndiff_pipeline.common.scc_paths import event_scc_leaf
from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
)
from syndiff_pipeline.common.orchestration.spec import StageRunContext
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.stages import (
    DIFF_STAGE,
    execute_diff_stage,
    write_diff_manifest,
)
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.pipeline_spec import STAGE_DEPS, STAGE_NAMES, STAGE_POOL, SYNDIFF_PIPELINE
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    load_runner_config,
    parse_stage_params,
)
from syndiff_pipeline.template_creation.orchestration.verify import verify_diff
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
                "condor:",
                "  request_cpus: 4",
                "  request_memory: 32000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class TestDiffPipelineSpec(unittest.TestCase):
    def test_diff_split_stages_composed(self):
        # diff is split into diff_prep -> background_estimate -> diff (the
        # tail: hotpants/epsf/centroids/temporal_wcs), shown to operators as
        # one "diff" stage but three real DAG nodes.
        self.assertEqual(len(STAGE_NAMES), 11)
        self.assertEqual(
            STAGE_NAMES[-5:], ("diff_prep", "background_estimate", "diff", "photometry", "star")
        )
        self.assertEqual(STAGE_DEPS["diff_prep"], ["downsample"])
        self.assertEqual(STAGE_DEPS["background_estimate"], ["diff_prep"])
        self.assertEqual(STAGE_DEPS["diff"], ["background_estimate"])
        self.assertEqual(STAGE_POOL["diff_prep"], "diff_prep")
        self.assertEqual(STAGE_POOL["background_estimate"], "background_estimate")
        self.assertEqual(STAGE_POOL["diff"], "diff")

    def test_diff_resource_pool_default(self):
        cfg = RunnerConfig(stages=parse_stage_params({}))
        self.assertIn("diff", cfg.resources)
        self.assertEqual(cfg.resources["diff"].max_concurrent, 2)

    def test_diff_condor_resources_from_site_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            site.mkdir()
            write_site_deployment(
                site,
                workspace_root=str(root / "handoff"),
                data_root=str(root / "data"),
            )
            _write_diff_policy(site / "diff_config.yaml")
            runner = RunnerConfig(
                stages=parse_stage_params({"diff": {"executor": "condor"}}),
                diff_config_path=str(site / "diff_config.yaml"),
            )
            resources = DIFF_STAGE.condor_resources(runner)
            self.assertIsInstance(resources, condor.CondorResourceRequest)
            self.assertEqual(resources.request_cpus, 4)
            self.assertEqual(resources.request_memory_mb, 32000)


class TestDiffStageExecution(unittest.TestCase):
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

        self.target = _target()
        self.event_dir = event_scc_leaf(
            self.handoff,
            self.target.event_name(),
            self.target.sector,
            self.target.camera,
            self.target.ccd,
        )
        self.event_dir.mkdir(parents=True, exist_ok=True)
        template_leaf = (
            self.data
            / "s0020" / "c3" / "k3"
            / "templates"
            / "oversampling_1"
        )
        template_leaf.mkdir(parents=True)
        (template_leaf / "group_1").mkdir()
        (template_leaf / "group_1" / "ps1_template.fits").write_bytes(b"SIMPLE  = T")
        (self.event_dir / "event_job.json").write_text(
            json.dumps({"reference_ffi_path": "/tmp/ref.fits"}),
            encoding="utf-8",
        )

        self.runner = RunnerConfig(
            workspace_root=str(self.handoff),
            runs_root=str(self.handoff / "runs"),
            diff_config_path=str(self.site / "diff_config.yaml"),
            stages=parse_stage_params({"diff": {"executor": "condor"}}),
        )
        self.runs_root = self.runner.runs_root
        self.run_id = "test_run"
        run_dir = Path(self.runs_root) / self.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "per_target").mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _ctx(self) -> StageRunContext:
        return StageRunContext(
            run_id=self.run_id,
            runs_root=self.runs_root,
            target_label=self.target.label(),
            target=self.target,
            runner_cfg=self.runner,
            meta={"source_diff_config_path": str(self.site / "diff_config.yaml")},
        )

    @mock.patch(
        "syndiff_pipeline.difference_imaging.orchestration.execute.run_config_pipeline"
    )
    def test_execute_freezes_config_and_runs_pipeline(self, run_mock):
        ctx = self._ctx()
        expected, produced, artifacts = execute_diff_stage(ctx)
        frozen = (
            Path(self.runs_root)
            / self.run_id
            / "per_target"
            / self.target.label()
            / "diff_config.yaml"
        )
        self.assertTrue(frozen.is_file())
        run_mock.assert_called_once()
        self.assertGreaterEqual(expected, 1)
        self.assertGreaterEqual(produced, 0)
        self.assertIsInstance(artifacts, list)

    @mock.patch(
        "syndiff_pipeline.difference_imaging.orchestration.execute.run_config_pipeline"
    )
    def test_execute_diff_stage_force_rerun_preserves_ws(self, run_mock):
        ws_hp = self.event_dir / "ws" / "hp_d"
        ws_hp.mkdir(parents=True)
        stale_fits = ws_hp / "stale.fits"
        stale_fits.write_bytes(b"SIMPLE  = T")
        gaia_csv = self.event_dir / "ws" / "gaia_catalog_pipeline.csv"
        gaia_csv.parent.mkdir(parents=True, exist_ok=True)
        gaia_csv.write_text("source_id,ra,dec\n", encoding="utf-8")

        ctx = self._ctx()
        ctx.force_rerun = True
        execute_diff_stage(ctx)

        self.assertTrue(stale_fits.is_file())
        self.assertTrue((self.event_dir / "ws" / "hp_d").is_dir())
        self.assertTrue(gaia_csv.is_file())
        run_mock.assert_called_once()

    def test_verify_requires_ws_and_manifest(self):
        from syndiff_pipeline.difference_imaging.support.manifest import (
            manifest_path_from_output_dir,
        )
        from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
            _scc_lane_root,
            frozen_diff_config_for_verify,
            resolve_diff_site_config_path,
        )
        from syndiff_pipeline.common.scc_paths import resolve_scc_diff_bookkeeping_dir
        from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
            DIFF_JOB_BASENAME,
            FRAMES_CSV_BASENAME,
        )

        from syndiff_pipeline.template_creation.orchestration.runner_config import resolve_config

        resolved = resolve_config(self.target, self.runner)
        result = verify_diff(resolved, self.runner)
        self.assertFalse(result.ok)

        # An empty (header-only) manifest keeps the PR-D1 indexed-completeness
        # check falling open (no product IDs to require), so this test
        # exercises the legacy scc_diff_lane_complete fallback: bookkeeping
        # (diff_job.json + frames.csv) plus the final stage's SCC-lane
        # artifact (shared_mask.fits under the SCC diff lane root, *not* the
        # event workspace -- shared_mask moved SCC-side in the workspace-split
        # migration).
        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")

        cfg = frozen_diff_config_for_verify(
            resolve_diff_site_config_path(meta=None, runner_cfg=self.runner),
            self.target,
            meta=None,
        )
        lane_root = _scc_lane_root(cfg)
        lane_root.mkdir(parents=True, exist_ok=True)
        (lane_root / SHARED_MASK_FITS_BASENAME).write_bytes(b"SIMPLE  = T")

        bk_dir = resolve_scc_diff_bookkeeping_dir(
            cfg.data_root,
            int(cfg.sector),
            int(cfg.camera),
            int(cfg.ccd),
            oversampling_factor=max(1, int(getattr(cfg, "oversampling_factor", 1) or 1)),
            template_store_name=getattr(cfg, "template_store_name", None),
        )
        bk_dir.mkdir(parents=True, exist_ok=True)
        (bk_dir / DIFF_JOB_BASENAME).write_text("{}", encoding="utf-8")
        (bk_dir / FRAMES_CSV_BASENAME).write_text("ffi_product_id\n", encoding="utf-8")

        result = verify_diff(resolved, self.runner)
        self.assertTrue(result.ok)

    def test_write_diff_manifest_round_trip(self):
        ctx = self._ctx()
        manifest_path = (
            Path(self.runs_root)
            / self.run_id
            / "per_target"
            / self.target.label()
            / "diff.manifest.json"
        )
        payload = write_diff_manifest(
            manifest_path, ctx, [str(self.event_dir / "ws")], 1, 1
        )
        self.assertEqual(payload["stage"], "diff")
        self.assertIn("config_fingerprint", payload)
        self.assertTrue(manifest_path.is_file())

    def test_site_config_loads_diff_path(self):
        site_config = self.site / "pipeline.yaml"
        site_config.write_text(
            "\n".join(
                [
                    "deployment_file: deployment.yaml",
                    "diff_config: diff_config.yaml",
                    "stages:",
                    "  diff:",
                    "    executor: condor",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cfg = load_runner_config(site_config)
        self.assertTrue(cfg.diff_config_path.endswith("diff_config.yaml"))
        self.assertEqual(cfg.stages.diff.executor, "condor")


if __name__ == "__main__":
    unittest.main()
