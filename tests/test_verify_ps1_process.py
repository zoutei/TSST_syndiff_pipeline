"""Tests for ps1_process artifact verification."""
from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import zarr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_runner.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_runner.stage_params import (
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
    DownsampleStageParams,
)
from syndiff_pipeline.template_runner.targets import Target
from syndiff_pipeline.template_runner.verify import verify_ps1_process


def _resolved(tmp: Path, csv_path: Path, zarr_path: Path, projections_limit: int | None = None):
    target = Target(
        sector=44,
        camera=2,
        ccd=1,
        target_ra=85.0,
        target_dec=16.0,
        target_name="2021aesq",
    )
    mapping_root = csv_path.parent.parent.parent.parent
    return ResolvedTargetConfig(
        target=target,
        data_root=str(zarr_path.parent.parent),
        ffi_dir=str(tmp / "ffi"),
        handoff_dir=str(tmp / "handoff" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        gaia_credentials=None,
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(projections_limit=projections_limit),
            downsample=DownsampleStageParams(),
        ),
        mapping_root=str(mapping_root),
        zarr_dir=str(tmp / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "shifted_downsampled"),
    )


class TestVerifyPs1Process(unittest.TestCase):
    def _write_mapping_csv(self, path: Path, rows: list[tuple[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["NAME,projection", *(f"{name},{proj}" for name, proj in rows)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_empty_zarr_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "skycell_pixel_mapping" / "sector_0044" / "camera_2" / "ccd_1" / "tess_s0044_2_1_master_skycells_list.csv"
            self._write_mapping_csv(csv_path, [("skycell.1520.080", "1520")])
            zarr_path = tmp / "convolved_results" / "sector_0044_camera_2_ccd_1.zarr"
            zarr_path.mkdir(parents=True)
            zarr.open(str(zarr_path), mode="w")

            result = verify_ps1_process(_resolved(tmp, csv_path, zarr_path))
            self.assertFalse(result.ok)
            self.assertIn("0/1", result.message)

    def test_partial_zarr_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "skycell_pixel_mapping" / "sector_0044" / "camera_2" / "ccd_1" / "tess_s0044_2_1_master_skycells_list.csv"
            self._write_mapping_csv(
                csv_path,
                [
                    ("skycell.1520.080", "1520"),
                    ("skycell.1520.081", "1520"),
                ],
            )
            zarr_path = tmp / "convolved_results" / "sector_0044_camera_2_ccd_1.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root.create_array("skycell.1520.080_data", shape=(8, 8), dtype="f4")[:] = 1.0

            result = verify_ps1_process(_resolved(tmp, csv_path, zarr_path))
            self.assertFalse(result.ok)
            self.assertIn("Partial", result.message)
            self.assertIn("1/2", result.message)

    def test_complete_zarr_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "skycell_pixel_mapping" / "sector_0044" / "camera_2" / "ccd_1" / "tess_s0044_2_1_master_skycells_list.csv"
            self._write_mapping_csv(
                csv_path,
                [
                    ("skycell.1520.080", "1520"),
                    ("skycell.1520.081", "1520"),
                ],
            )
            zarr_path = tmp / "convolved_results" / "sector_0044_camera_2_ccd_1.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root.create_array("skycell.1520.080_data", shape=(8, 8), dtype="f4")[:] = 1.0
            root.create_array("skycell.1520.081_data", shape=(8, 8), dtype="f4")[:] = 1.0

            result = verify_ps1_process(_resolved(tmp, csv_path, zarr_path))
            self.assertTrue(result.ok)
            self.assertIn("2/2", result.message)

    def test_arrays_without_chunks_are_not_saved(self):
        # An array directory that exists with a positive shape but no materialized
        # chunk must count as NOT saved (catches interrupted/empty writes).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "skycell_pixel_mapping" / "sector_0044" / "camera_2" / "ccd_1" / "tess_s0044_2_1_master_skycells_list.csv"
            self._write_mapping_csv(csv_path, [("skycell.1520.080", "1520")])
            zarr_path = tmp / "convolved_results" / "sector_0044_camera_2_ccd_1.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root.create_array("skycell.1520.080_data", shape=(8, 8), dtype="f4")

            result = verify_ps1_process(_resolved(tmp, csv_path, zarr_path))
            self.assertFalse(result.ok)
            self.assertIn("0/1", result.message)

    def test_tuple_skycell_ids_are_normalized(self):
        # Regression: expected_convolved_skycells yields (name, index) tuples;
        # verification must compare on the name alone, otherwise a complete store
        # is mis-reported as 0/N and the stage is needlessly re-run.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "skycell_pixel_mapping" / "sector_0044" / "camera_2" / "ccd_1" / "tess_s0044_2_1_master_skycells_list.csv"
            self._write_mapping_csv(csv_path, [("skycell.1520.080", "1520")])
            zarr_path = tmp / "convolved_results" / "sector_0044_camera_2_ccd_1.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root.create_array("skycell.1520.080_data", shape=(8, 8), dtype="f4")[:] = 1.0
            root.create_array("skycell.1520.081_data", shape=(8, 8), dtype="f4")[:] = 1.0

            with unittest.mock.patch(
                "syndiff_pipeline.template_runner.verify.expected_convolved_skycells",
                return_value=[("skycell.1520.080", 0), ("skycell.1520.081", 1)],
            ):
                result = verify_ps1_process(_resolved(tmp, csv_path, zarr_path))
            self.assertTrue(result.ok)
            self.assertIn("2/2", result.message)

    def test_projections_limit_reduces_expected_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "skycell_pixel_mapping" / "sector_0044" / "camera_2" / "ccd_1" / "tess_s0044_2_1_master_skycells_list.csv"
            self._write_mapping_csv(
                csv_path,
                [
                    ("skycell.1520.080", "1520"),
                    ("skycell.1520.081", "1520"),
                    ("skycell.1922.042", "1922"),
                ],
            )
            zarr_path = tmp / "convolved_results" / "sector_0044_camera_2_ccd_1.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            root.create_array("skycell.1520.080_data", shape=(8, 8), dtype="f4")[:] = 1.0
            root.create_array("skycell.1520.081_data", shape=(8, 8), dtype="f4")[:] = 1.0

            result = verify_ps1_process(_resolved(tmp, csv_path, zarr_path, projections_limit=1))
            self.assertTrue(result.ok)
            self.assertIn("2/2", result.message)


if __name__ == "__main__":
    unittest.main()
