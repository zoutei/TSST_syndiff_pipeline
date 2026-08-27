"""BK-7: shared convolved store flag wiring through stage_params, dispatch, verify."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration import dispatch
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    RemapStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
    parse_stage_params,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    ps1_process_uses_shared_convolved_store,
    resolve_downsample_convolved_dir,
    resolve_ps1_process_checkpoint_location,
    verify_ps1_process,
)
from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path, scc_convolved_zarr
from tests.test_provenance_checkpoint import (
    _FakeMappingParams,
    _FakePs1ProcessParams,
    _FakeResolved,
    _FakeStages,
    _FakeTarget,
    emit_scc_assembly_checkpoint,
)


def _template_stages(**ps1_overrides) -> TemplateStageParams:
    return parse_stage_params({"ps1_process": ps1_overrides})


class TestPs1ProcessStageParamsSharedStore(unittest.TestCase):
    def test_defaults_off(self):
        stages = parse_stage_params({})
        self.assertFalse(stages.ps1_process.use_shared_convolved_store)
        self.assertTrue(stages.ps1_process.write_per_scc_convolved_zarr)

    def test_parse_shared_store_flags(self):
        stages = parse_stage_params(
            {
                "ps1_process": {
                    "use_shared_convolved_store": True,
                    "write_per_scc_convolved_zarr": False,
                }
            }
        )
        self.assertTrue(stages.ps1_process.use_shared_convolved_store)
        self.assertFalse(stages.ps1_process.write_per_scc_convolved_zarr)

    def test_rejects_dual_write_when_shared_on(self):
        with self.assertRaises(ValueError):
            parse_stage_params(
                {
                    "ps1_process": {
                        "use_shared_convolved_store": True,
                        "write_per_scc_convolved_zarr": True,
                    }
                }
            )


class TestDispatchPs1ProcessKwargs(unittest.TestCase):
    def _resolved(self, pp: Ps1ProcessStageParams) -> ResolvedTargetConfig:
        target = Target(
            sector=20,
            camera=1,
            ccd=1,
            target_ra=1.0,
            target_dec=2.0,
            target_name="bk7",
        )
        return ResolvedTargetConfig(
            target=target,
            data_root="/data",
            ffi_dir="/data/s0020/c1/k1/ffi",
            event_dir="/ws/events/bk7/s0020_c1_k1",
            skycell_wcs_csv="/data/skycell_wcs.csv",
            stages=TemplateStageParams(
                wcs_grouping=WcsGroupingStageParams(),
                mapping=MappingStageParams(),
                ps1_download=Ps1DownloadStageParams(),
                ps1_process=pp,
                remap=RemapStageParams(),
                downsample=DownsampleStageParams(),
            ),
            mapping_root="/data/s0020/c1/k1/mapping/oversampling_1",
            zarr_dir="/data/ps1_skycells_zarr",
            template_output_base="/data/s0020/c1/k1/templates/oversampling_1",
        )

    @mock.patch("syndiff_pipeline.template_creation.processing.ps1_process.run_modern_sliding_window_pipeline")
    def test_dispatch_forces_per_scc_off_when_shared_on(self, run_mock):
        run_mock.return_value = {
            "expected_count": 1,
            "produced_count": 1,
            "artifacts": [],
        }
        resolved = self._resolved(
            Ps1ProcessStageParams(
                use_shared_convolved_store=True,
                write_per_scc_convolved_zarr=False,
            )
        )
        dispatch._execute_template_stage(resolved, "ps1_process")
        run_mock.assert_called_once()
        kwargs = run_mock.call_args.kwargs
        self.assertTrue(kwargs["use_shared_convolved_store"])
        self.assertFalse(kwargs["write_per_scc_convolved_zarr"])

    @mock.patch("syndiff_pipeline.template_creation.processing.ps1_process.run_modern_sliding_window_pipeline")
    def test_dispatch_legacy_defaults(self, run_mock):
        run_mock.return_value = {
            "expected_count": 0,
            "produced_count": 0,
            "artifacts": [],
        }
        resolved = self._resolved(Ps1ProcessStageParams())
        dispatch._execute_template_stage(resolved, "ps1_process")
        kwargs = run_mock.call_args.kwargs
        self.assertFalse(kwargs["use_shared_convolved_store"])
        self.assertTrue(kwargs["write_per_scc_convolved_zarr"])
        self.assertEqual(kwargs["stream_max_inflight_requests"], 24)
        self.assertEqual(kwargs["stream_prefetch_cells"], 6)

    @mock.patch("syndiff_pipeline.template_creation.processing.ps1_process.run_modern_sliding_window_pipeline")
    def test_dispatch_passes_stream_loader_controls(self, run_mock):
        run_mock.return_value = {
            "expected_count": 0,
            "produced_count": 0,
            "artifacts": [],
        }
        resolved = self._resolved(
            Ps1ProcessStageParams(
                ps1_source="stream",
                stream_max_inflight_requests=12,
                stream_prefetch_cells=3,
            )
        )
        dispatch._execute_template_stage(resolved, "ps1_process")
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(kwargs["stream_max_inflight_requests"], 12)
        self.assertEqual(kwargs["stream_prefetch_cells"], 3)


class TestCheckpointAndVerifyResolvers(unittest.TestCase):
    def test_checkpoint_location_legacy_by_default(self):
        resolved = _FakeResolved(data_root="/data/root")
        self.assertEqual(
            str(resolve_ps1_process_checkpoint_location(resolved)),
            str(scc_convolved_zarr("/data/root", 20, 1, 1)),
        )
        self.assertFalse(ps1_process_uses_shared_convolved_store(resolved))

    def test_checkpoint_location_shared_when_flag_on(self):
        resolved = _FakeResolved(
            data_root="/data/root",
            stages=_FakeStages(
                _FakeMappingParams(),
                _FakePs1ProcessParams(
                    use_shared_convolved_store=True,
                    write_per_scc_convolved_zarr=False,
                ),
            ),
        )
        self.assertEqual(
            str(resolve_ps1_process_checkpoint_location(resolved)),
            str(ps1_convolved_zarr_path("/data/root")),
        )

    def test_emit_scc_assembly_location_follows_shared_flag(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            resolved = _FakeResolved(
                data_root=tmp,
                stages=_FakeStages(
                    _FakeMappingParams(),
                    _FakePs1ProcessParams(
                        use_shared_convolved_store=True,
                        write_per_scc_convolved_zarr=False,
                    ),
                ),
            )
            emit_scc_assembly_checkpoint(resolved)
            from syndiff_pipeline.common.scc_paths import provenance_spool_dir

            records = []
            spool = Path(provenance_spool_dir(tmp))
            for path in spool.glob("*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0]["location"],
                str(ps1_convolved_zarr_path(tmp)),
            )

    def test_verify_shared_store_counts_published_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            mapping_dir = (
                data_root / "s0020" / "c1" / "k1" / "mapping" / "oversampling_1"
            )
            mapping_dir.mkdir(parents=True)
            csv_path = mapping_dir / "tess_s0020_1_1_master_skycells_list.csv"
            csv_path.write_text(
                "NAME,projection,y,x,NAXIS1,NAXIS2\n"
                "skycell.1111.001,skycell.1111,0,0,32,32\n",
                encoding="utf-8",
            )
            cell_dir = (
                data_root
                / "ps1_skycells_zarr"
                / "ps1_convolved.zarr"
                / "skycell.1111"
                / "001"
                / "fp0001"
            )
            cell_dir.mkdir(parents=True)
            (cell_dir / "arrays.npz").write_bytes(b"x")

            target = Target(
                sector=20,
                camera=1,
                ccd=1,
                target_ra=1.0,
                target_dec=2.0,
                target_name="bk7",
            )
            resolved = ResolvedTargetConfig(
                target=target,
                data_root=str(data_root),
                ffi_dir=str(data_root / "s0020" / "c1" / "k1" / "ffi"),
                event_dir=str(data_root / "events" / "bk7" / "s0020_c1_k1"),
                skycell_wcs_csv=str(data_root / "skycell_wcs.csv"),
                stages=TemplateStageParams(
                    wcs_grouping=WcsGroupingStageParams(),
                    mapping=MappingStageParams(oversampling_factor=1),
                    ps1_download=Ps1DownloadStageParams(),
                    ps1_process=Ps1ProcessStageParams(
                        use_shared_convolved_store=True,
                        write_per_scc_convolved_zarr=False,
                    ),
                    remap=RemapStageParams(),
                    downsample=DownsampleStageParams(),
                ),
                mapping_root=str(mapping_dir),
                zarr_dir=str(data_root / "ps1_skycells_zarr"),
                template_output_base=str(
                    data_root / "s0020" / "c1" / "k1" / "templates" / "oversampling_1"
                ),
            )
            result = verify_ps1_process(resolved)
            self.assertTrue(result.ok, result.message)
            self.assertEqual(
                resolve_downsample_convolved_dir(resolved),
                str(ps1_convolved_zarr_path(data_root)),
            )


if __name__ == "__main__":
    unittest.main()
