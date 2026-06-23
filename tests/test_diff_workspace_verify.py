"""Tests for workspace-aware diff verification (debug ws_{id}/ trees)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
)
from syndiff_pipeline.common.orchestration.spec import StageRunContext
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
    collect_diff_workspace_artifacts,
    diff_workspace_complete,
    frozen_diff_config_for_verify,
)
from syndiff_pipeline.difference_imaging.orchestration.stages import _diff_config_fingerprint
from syndiff_pipeline.difference_imaging.support.manifest import manifest_path_from_output_dir
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    RunnerConfig,
    parse_stage_params,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.verify import verify_diff
from tests.site_fixtures import write_site_deployment


def _target() -> Target:
    return Target(
        sector=22,
        camera=3,
        ccd=3,
        target_ra=228.479042,
        target_dec=52.722981,
        target_name="2020dgc",
    )


def _write_single_kernel_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "defaults:",
                "  n_jobs: 2",
                "  workspace_run_id: dbg",
                "paths:",
                "  template_base: shifted_downsampled",
                "pipeline:",
                "  - kind: shared_mask",
                "  - kind: kernel_fit",
                "    output: kernel_fit",
                "  - kind: convolved_templates",
                "    inputs:",
                "      kernel_fit: kernel_fit",
                "    output: tmpl_conv",
                "  - kind: kernel_subtract",
                "    inputs:",
                "      convolved: tmpl_conv",
                "    output:",
                "      diffs: kd_d",
                "  - kind: forced_photometry",
                "    inputs:",
                "      diffs: kd_d",
                "    output: lc_prf_on_diffs",
                "condor:",
                "  request_cpus: 4",
                "  request_memory: 32000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class TestDiffWorkspaceVerify(unittest.TestCase):
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
        _write_single_kernel_policy(self.site / "diff_config.yaml")

        self.target = _target()
        self.event_dir = self.handoff / "events" / self.target.label()
        self.event_dir.mkdir(parents=True, exist_ok=True)
        template_leaf = self.data / "shifted_downsampled" / "sector0022_camera3_ccd3"
        template_leaf.mkdir(parents=True)
        ensure_event_templates_symlink(self.event_dir, template_leaf)
        (self.event_dir / "cluster_template_job.json").write_text(
            json.dumps({"reference_ffi_path": "/tmp/ref.fits"}),
            encoding="utf-8",
        )

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")

        self.runner = RunnerConfig(
            workspace_root=str(self.handoff),
            runs_root=str(self.handoff / "runs"),
            diff_config_path=str(self.site / "diff_config.yaml"),
            stages=parse_stage_params({"diff": {"executor": "condor"}}),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _cfg(self, *, meta: dict | None = None):
        return frozen_diff_config_for_verify(
            self.site / "diff_config.yaml",
            self.target,
            meta=meta,
        )

    def test_canonical_ws_ignored_when_debug_id_set(self):
        canonical = self.event_dir / "ws" / "kd_d"
        canonical.mkdir(parents=True)
        (canonical / "frame.fits").write_bytes(b"SIMPLE  = T")
        (self.event_dir / "ws" / "lc_prf_on_diffs" / "lightcurve.csv").parent.mkdir(
            parents=True, exist_ok=True
        )
        (self.event_dir / "ws" / "lc_prf_on_diffs" / "lightcurve.csv").write_text(
            "btjd,flux\n", encoding="utf-8"
        )

        cfg = self._cfg()
        self.assertFalse(diff_workspace_complete(cfg, self.event_dir))

        result = verify_diff(resolve_config(self.target, self.runner), self.runner)
        self.assertFalse(result.ok)

    def test_partial_debug_tree_incomplete_without_lightcurve(self):
        ws_dbg = self.event_dir / "ws_dbg"
        (ws_dbg / "kernel_fit").mkdir(parents=True, exist_ok=True)
        (ws_dbg / "kernel_fit" / "kernel_fit_meta.json").write_text("{}", encoding="utf-8")
        (ws_dbg / "kernel_fit" / "kernel_r2.npz").write_bytes(b"PK")
        kd = ws_dbg / "kd_d"
        kd.mkdir(parents=True)
        (kd / "tess0001_kd_d.fits").write_bytes(b"SIMPLE  = T")

        cfg = self._cfg()
        self.assertFalse(diff_workspace_complete(cfg, self.event_dir))

    def test_complete_when_final_lightcurve_present_in_debug_tree(self):
        ws_dbg = self.event_dir / "ws_dbg" / "lc_prf_on_diffs"
        ws_dbg.mkdir(parents=True)
        (ws_dbg / "lightcurve.csv").write_text("btjd,flux\n1,2\n", encoding="utf-8")

        cfg = self._cfg()
        self.assertTrue(diff_workspace_complete(cfg, self.event_dir))

        result = verify_diff(resolve_config(self.target, self.runner), self.runner)
        self.assertTrue(result.ok)
        self.assertIn("ws_dbg", result.message)

    def test_meta_workspace_run_id_override(self):
        cfg_yaml = self._cfg()
        self.assertEqual(cfg_yaml.workspace_run_id, "dbg")

        cfg_meta = self._cfg(meta={"workspace_run_id": "cli_debug"})
        self.assertEqual(cfg_meta.workspace_run_id, "cli_debug")

        ws = self.event_dir / "ws_cli_debug" / "lc_prf_on_diffs"
        ws.mkdir(parents=True)
        (ws / "lightcurve.csv").write_text("btjd,flux\n", encoding="utf-8")

        self.assertTrue(diff_workspace_complete(cfg_meta, self.event_dir))
        self.assertFalse(diff_workspace_complete(cfg_yaml, self.event_dir))

    def test_fingerprint_includes_workspace_run_id(self):
        ctx = StageRunContext(
            run_id="run_a",
            runs_root=str(self.handoff / "runs"),
            target_label=self.target.label(),
            target=self.target,
            runner_cfg=self.runner,
            meta={"source_diff_config_path": str(self.site / "diff_config.yaml")},
        )
        fp_default = _diff_config_fingerprint(ctx)

        ctx.meta["workspace_run_id"] = "other"
        fp_override = _diff_config_fingerprint(ctx)

        self.assertNotEqual(fp_default, fp_override)

    def test_collect_artifacts_uses_debug_tree(self):
        ws_dbg = self.event_dir / "ws_dbg" / "kd_d"
        ws_dbg.mkdir(parents=True)
        fits = ws_dbg / "tess0001_kd_d.fits"
        fits.write_bytes(b"SIMPLE  = T")

        canonical = self.event_dir / "ws" / "kd_d"
        canonical.mkdir(parents=True)
        (canonical / "other.fits").write_bytes(b"SIMPLE  = T")

        cfg = self._cfg()
        artifacts = collect_diff_workspace_artifacts(cfg, self.event_dir)
        artifact_str = "\n".join(artifacts)
        self.assertIn(str(fits.resolve()), artifact_str)
        self.assertNotIn(str((canonical / "other.fits").resolve()), artifact_str)


class TestSharedMaskOnlyVerify(unittest.TestCase):
    def test_shared_mask_only_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            site.mkdir()
            handoff = root / "handoff"
            data = root / "data"
            write_site_deployment(
                site,
                workspace_root=str(handoff),
                data_root=str(data),
            )
            policy = site / "diff_config.yaml"
            policy.write_text(
                "\n".join(
                    [
                        "deployment_file: deployment.yaml",
                        "paths:",
                        "  template_base: shifted_downsampled",
                        "pipeline:",
                        "  - kind: shared_mask",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            target = _target()
            event_dir = handoff / "events" / target.label()
            event_dir.mkdir(parents=True)
            template_leaf = data / "shifted_downsampled" / "sector0022_camera3_ccd3"
            template_leaf.mkdir(parents=True)
            ensure_event_templates_symlink(event_dir, template_leaf)
            manifest_csv = Path(manifest_path_from_output_dir(str(event_dir), None))
            manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")
            ws = event_dir / "ws"
            ws.mkdir(exist_ok=True)
            (ws / SHARED_MASK_FITS_BASENAME).write_bytes(b"SIMPLE  = T")

            runner = RunnerConfig(
                workspace_root=str(handoff),
                diff_config_path=str(policy),
            )
            cfg = frozen_diff_config_for_verify(policy, target)
            self.assertTrue(diff_workspace_complete(cfg, event_dir))

            result = verify_diff(resolve_config(target, runner), runner)
            self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
