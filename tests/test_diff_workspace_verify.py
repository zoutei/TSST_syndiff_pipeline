"""Tests for SCC-aware diff verification."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    resolve_scc_diff_bookkeeping_dir,
    scc_diff_dir,
)
from syndiff_pipeline.common.orchestration.spec import StageRunContext
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
    collect_diff_workspace_artifacts,
    diff_workspace_complete,
    frozen_diff_config_for_verify,
    scc_diff_lane_complete,
)
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
)
from syndiff_pipeline.difference_imaging.orchestration.stages import _diff_config_fingerprint
from syndiff_pipeline.difference_imaging.support.manifest import manifest_path_from_output_dir
from syndiff_pipeline.difference_imaging.support.paths import SHARED_MASK_FITS_BASENAME
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    load_runner_config,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.verify import verify_diff
from tests.site_fixtures import write_unified_site_config


def _target() -> Target:
    return Target(
        sector=22,
        camera=3,
        ccd=3,
        target_ra=228.479042,
        target_dec=52.722981,
        target_name="2020dgc",
    )


_SINGLE_KERNEL_DIFF_POLICY = {
    "defaults": {"n_jobs": 2},
    "paths": {"template_base": "shifted_downsampled"},
    "pipeline": [
        {"kind": "shared_mask"},
        {"kind": "kernel_fit", "output": "kernel_fit"},
        {
            "kind": "convolved_templates",
            "inputs": {"kernel_fit": "kernel_fit"},
            "output": "tmpl_conv",
        },
        {
            "kind": "background_estimate",
            "inputs": {"convolved": "tmpl_conv"},
            "output": {"diffs": "ks_d"},
        },
    ],
    "condor": {"request_cpus": 4, "request_memory": 32000},
}


def _write_single_kernel_policy(site: Path, *, workspace_root: str, data_root: str) -> None:
    write_unified_site_config(
        site / "pipeline.yaml",
        workspace_root=workspace_root,
        data_root=data_root,
        diff=_SINGLE_KERNEL_DIFF_POLICY,
        stages={"diff": {"executor": "condor"}},
    )


def _write_scc_bookkeeping(data_root: Path, target: Target) -> Path:
    bk = resolve_scc_diff_bookkeeping_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        oversampling_factor=1,
        template_store_name=None,
    )
    bk.mkdir(parents=True, exist_ok=True)
    (bk / FRAMES_CSV_BASENAME).write_text("ffi_product_id\n", encoding="utf-8")
    (bk / DIFF_JOB_BASENAME).write_text(
        json.dumps({"schema_version": 2, "mapping_grid": {"nx": 1, "ny": 1}}),
        encoding="utf-8",
    )
    return bk


class TestDiffWorkspaceVerify(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"
        self.site.mkdir()
        self.handoff = self.root / "handoff"
        self.data = self.root / "data"
        _write_single_kernel_policy(
            self.site, workspace_root=str(self.handoff), data_root=str(self.data)
        )

        self.target = _target()
        self.event_dir = event_scc_leaf(
            self.handoff,
            self.target.event_name(),
            self.target.sector,
            self.target.camera,
            self.target.ccd,
        )
        self.event_dir.mkdir(parents=True, exist_ok=True)
        (self.event_dir / "event_job.json").write_text(
            json.dumps({"reference_ffi_path": "/tmp/ref.fits"}),
            encoding="utf-8",
        )

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")

        self.runner = load_runner_config(self.site / "pipeline.yaml")
        self.runner.runs_root = str(self.handoff / "runs")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _cfg(self, *, meta: dict | None = None):
        return frozen_diff_config_for_verify(
            self.target,
            meta=meta,
            runner_cfg=self.runner,
        )

    def _lane_root(self) -> Path:
        return scc_diff_dir(
            self.data,
            self.target.sector,
            self.target.camera,
            self.target.ccd,
        )

    def test_incomplete_without_scc_lane_outputs(self):
        _write_scc_bookkeeping(self.data, self.target)
        cfg = self._cfg()
        self.assertFalse(scc_diff_lane_complete(cfg))
        self.assertFalse(diff_workspace_complete(cfg, self.event_dir))

    def test_complete_when_final_diff_on_scc_lane(self):
        _write_scc_bookkeeping(self.data, self.target)
        lane = self._lane_root()
        ks_d = lane / "ks_d"
        ks_d.mkdir(parents=True)
        (ks_d / "tess2020019142923-s0022-3-3_ks_d.fits.fz").write_bytes(b"SIMPLE  = T")

        cfg = self._cfg()
        self.assertTrue(scc_diff_lane_complete(cfg))
        self.assertTrue(diff_workspace_complete(cfg, self.event_dir))

        result = verify_diff(resolve_config(self.target, self.runner), self.runner)
        self.assertTrue(result.ok)
        self.assertIn("SCC diff lane complete", result.message)

    def test_event_ws_ignored_when_scc_lane_complete(self):
        _write_scc_bookkeeping(self.data, self.target)
        lane = self._lane_root()
        ks_d = lane / "ks_d"
        ks_d.mkdir(parents=True)
        (ks_d / "tess2020019142923-s0022-3-3_ks_d.fits.fz").write_bytes(b"SIMPLE  = T")

        canonical = self.event_dir / "ws" / "ks_d"
        canonical.mkdir(parents=True)
        (canonical / "frame.fits").write_bytes(b"SIMPLE  = T")

        cfg = self._cfg()
        self.assertTrue(diff_workspace_complete(cfg, self.event_dir))

    def test_fingerprint_stable_without_workspace_run_id(self):
        ctx = StageRunContext(
            run_id="run_a",
            runs_root=str(self.handoff / "runs"),
            target_label=self.target.label(),
            target=self.target,
            runner_cfg=self.runner,
            meta={},
        )
        fp_default = _diff_config_fingerprint(ctx)

        ctx.meta["workspace_run_id"] = "other"
        fp_override = _diff_config_fingerprint(ctx)

        self.assertNotEqual(fp_default, fp_override)

    def test_collect_artifacts_from_scc_lane(self):
        _write_scc_bookkeeping(self.data, self.target)
        lane = self._lane_root()
        ks_d = lane / "ks_d"
        ks_d.mkdir(parents=True)
        fits = ks_d / "tess2020019142923-s0022-3-3_ks_d.fits.fz"
        fits.write_bytes(b"SIMPLE  = T")

        canonical = self.event_dir / "ws" / "ks_d"
        canonical.mkdir(parents=True)
        (canonical / "other.fits").write_bytes(b"SIMPLE  = T")

        cfg = self._cfg()
        artifacts = collect_diff_workspace_artifacts(cfg, self.event_dir)
        artifact_str = "\n".join(artifacts)
        self.assertIn(str(fits.resolve()), artifact_str)
        self.assertNotIn(str((canonical / "other.fits").resolve()), artifact_str)


class TestSharedMaskOnlyVerify(unittest.TestCase):
    def test_shared_mask_only_complete_on_scc_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            site.mkdir()
            handoff = root / "handoff"
            data = root / "data"
            write_unified_site_config(
                site / "pipeline.yaml",
                workspace_root=str(handoff),
                data_root=str(data),
                diff={
                    "paths": {"template_base": "shifted_downsampled"},
                    "pipeline": [{"kind": "shared_mask"}],
                },
            )
            target = _target()
            event_dir = event_scc_leaf(
                handoff,
                target.event_name(),
                target.sector,
                target.camera,
                target.ccd,
            )
            event_dir.mkdir(parents=True)
            manifest_csv = Path(manifest_path_from_output_dir(str(event_dir), None))
            manifest_csv.write_text("ffi_product_id\n", encoding="utf-8")
            _write_scc_bookkeeping(data, target)
            lane = scc_diff_dir(data, target.sector, target.camera, target.ccd)
            lane.mkdir(parents=True, exist_ok=True)
            (lane / SHARED_MASK_FITS_BASENAME).write_bytes(b"SIMPLE  = T")

            runner = load_runner_config(site / "pipeline.yaml")
            cfg = frozen_diff_config_for_verify(target, runner_cfg=runner)
            self.assertTrue(diff_workspace_complete(cfg, event_dir))

            result = verify_diff(resolve_config(target, runner), runner)
            self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
