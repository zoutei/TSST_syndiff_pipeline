"""Tests for fast absence probes before full artifact verify."""
from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    AbsenceProbeResult,
    stage_absence_probe,
)


def _resolved(tmp: Path) -> ResolvedTargetConfig:
    target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
    return ResolvedTargetConfig(
        target=target,
        data_root=str(tmp / "data"),
        ffi_dir=str(tmp / "data" / "tess_ffi"),
        event_dir=str(tmp / "events" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            downsample=DownsampleStageParams(),
        ),
        mapping_root=str(tmp / "mapping"),
        zarr_dir=str(tmp / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "shifted_downsampled"),
    )


class TestStageAbsenceProbe(unittest.TestCase):
    def test_wcs_absent_on_fresh_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            self.assertEqual(
                stage_absence_probe(resolved, "wcs_grouping"),
                AbsenceProbeResult.ABSENT,
            )

    def test_mapping_absent_on_fresh_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            self.assertEqual(
                stage_absence_probe(resolved, "mapping"),
                AbsenceProbeResult.ABSENT,
            )

    def test_tess_absent_with_no_files_or_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            self.assertEqual(
                stage_absence_probe(resolved, "tess_ffi_download"),
                AbsenceProbeResult.ABSENT,
            )

    def test_tess_maybe_present_with_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            ffi_leaf = (
                tmp
                / "data"
                / "tess_ffi"
                / "s0022"
                / "cam3_ccd3"
            )
            ffi_leaf.mkdir(parents=True)
            (ffi_leaf / "tess2020019142923-s0022-3-3-0165-s_ffic.fits").write_bytes(b"x")
            self.assertEqual(
                stage_absence_probe(resolved, "tess_ffi_download"),
                AbsenceProbeResult.MAYBE_PRESENT,
            )

    def test_ps1_download_absent_without_zarr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            self.assertEqual(
                stage_absence_probe(resolved, "ps1_download"),
                AbsenceProbeResult.ABSENT,
            )

    def test_ps1_download_maybe_present_with_empty_zarr_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            zarr_path = tmp / "data" / "ps1_skycells_zarr" / "ps1_skycells.zarr"
            zarr_path.mkdir(parents=True)
            self.assertEqual(
                stage_absence_probe(resolved, "ps1_download"),
                AbsenceProbeResult.MAYBE_PRESENT,
            )

    def test_ps1_process_absent_without_convolved_zarr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            self.assertEqual(
                stage_absence_probe(resolved, "ps1_process"),
                AbsenceProbeResult.ABSENT,
            )


class TestVerifyPassAbsenceProbe(unittest.TestCase):
    def test_absent_probe_skips_worker_schedule(self):
        from syndiff_pipeline.common.orchestration.scheduler import _run_verify_pass
        from syndiff_pipeline.common.orchestration.verify_worker import (
            reset_verify_worker_for_tests,
            shutdown_verify_worker,
        )
        from tests.test_daemon_behavior import _minimal_run_setup

        reset_verify_worker_for_tests()
        target = Target(22, 3, 3, 228.0, 52.0, "2020dgc")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                state, ctx, run_id, _runs_root = _minimal_run_setup(
                    tmp_path, [target], active_stages=["wcs_grouping"]
                )
                label = target.label()
                state.update_stage_status(
                    run_id, label, "tess_ffi_download", "success", exit_code=0
                )

                scheduled: list = []

                def _capture_schedule(tasks):
                    scheduled.extend(tasks)

                with unittest.mock.patch(
                    "syndiff_pipeline.common.orchestration.verify_worker.ArtifactVerifyWorker.schedule",
                    side_effect=_capture_schedule,
                ):
                    _run_verify_pass(
                        state, run_id, ctx, force_rerun=False, budget=8, block=False
                    )

                self.assertEqual(scheduled, [])
                self.assertTrue(
                    state.external_verify_attempted(run_id, label, "wcs_grouping")
                )
        finally:
            shutdown_verify_worker(wait=False)
            reset_verify_worker_for_tests()


if __name__ == "__main__":
    unittest.main()
