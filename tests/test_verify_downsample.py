"""Tests for per-offset completeness of downsample artifact verification."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_creation.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.verify import verify_downsample


def _resolved(tmp: Path, *, single_offset: bool) -> ResolvedTargetConfig:
    target = Target(
        sector=22,
        camera=3,
        ccd=3,
        target_ra=228.0,
        target_dec=52.0,
        target_name="2020dgc",
    )
    return ResolvedTargetConfig(
        target=target,
        data_root=str(tmp / "data"),
        ffi_dir=str(tmp / "data" / "tess_ffi"),
        handoff_dir=str(tmp / "handoff" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            downsample=DownsampleStageParams(single_offset=single_offset),
        ),
        mapping_root=str(tmp / "mapping"),
        zarr_dir=str(tmp / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "shifted_downsampled"),
    )


def _write_cluster_job(handoff_dir: Path, offsets: list[tuple[float, float]]) -> None:
    handoff_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "sector": 22,
        "camera": 3,
        "ccd": 3,
        "x_min": 0,
        "y_min": 0,
        "x_max": 2048,
        "y_max": 2048,
        "groups": [{"group_dx": dx, "group_dy": dy} for dx, dy in offsets],
    }
    (handoff_dir / "cluster_template_job.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _offset_fits_name(dx: float, dy: float) -> str:
    return f"syndiff_template_s0022_3_3_dx{dx:.3f}_dy{dy:.3f}.fits.gz"


def _out_dir(resolved: ResolvedTargetConfig) -> Path:
    base = Path(resolved.template_output_base)
    return base / "sector0022_camera3_ccd3"


class TestVerifyDownsampleMultiOffset(unittest.TestCase):
    def test_partial_offsets_fails_then_complete_passes(self):
        offsets = [(0.0, 0.0), (0.5, -0.5), (1.0, 0.25)]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp, single_offset=False)
            _write_cluster_job(Path(resolved.handoff_dir), offsets)
            out_dir = _out_dir(resolved)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Only SOME of the expected per-offset FITS present -> partial.
            for dx, dy in offsets[:2]:
                (out_dir / _offset_fits_name(dx, dy)).write_bytes(b"fits")

            result = verify_downsample(resolved)
            self.assertFalse(result.ok)
            self.assertIn("Partial downsample", result.message)
            self.assertIn("2/3", result.message)

            # Now ALL expected per-offset FITS present -> complete.
            for dx, dy in offsets[2:]:
                (out_dir / _offset_fits_name(dx, dy)).write_bytes(b"fits")

            result = verify_downsample(resolved)
            self.assertTrue(result.ok)
            self.assertIn("All 3 offset FITS present", result.message)


class TestVerifyDownsampleSingleOffset(unittest.TestCase):
    def test_single_offset_requires_only_zero_offset(self):
        # Multiple offsets in the cluster job, but single_offset=True collapses
        # the expectation to the single [0, 0] offset FITS.
        offsets = [(0.0, 0.0), (0.5, -0.5), (1.0, 0.25)]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp, single_offset=True)
            _write_cluster_job(Path(resolved.handoff_dir), offsets)
            out_dir = _out_dir(resolved)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Nothing on disk yet -> missing the single expected offset.
            result = verify_downsample(resolved)
            self.assertFalse(result.ok)
            self.assertIn("0/1", result.message)

            # Only the [0, 0] offset FITS is required for single_offset.
            (out_dir / _offset_fits_name(0.0, 0.0)).write_bytes(b"fits")
            result = verify_downsample(resolved)
            self.assertTrue(result.ok)
            self.assertIn("All 1 offset FITS present", result.message)


class TestVerifyDownsampleLegacyFits(unittest.TestCase):
    def test_uncompressed_fits_still_verifies(self):
        offsets = [(0.0, 0.0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp, single_offset=False)
            _write_cluster_job(Path(resolved.handoff_dir), offsets)
            out_dir = _out_dir(resolved)
            out_dir.mkdir(parents=True, exist_ok=True)

            legacy_name = "syndiff_template_s0022_3_3_dx0.000_dy0.000.fits"
            (out_dir / legacy_name).write_bytes(b"fits")

            result = verify_downsample(resolved)
            self.assertTrue(result.ok)
            self.assertIn("All 1 offset FITS present", result.message)


if __name__ == "__main__":
    unittest.main()
