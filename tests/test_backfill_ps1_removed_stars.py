"""Tests for scripts/backfill_ps1_removed_stars.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from astropy.io import fits

from scripts.backfill_ps1_removed_stars import _backfill_target
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.verify import (
    event_dir_ps1_removed_stars_csv_path,
    ps1_process_removed_stars_csv_path,
)


def _resolved(tmp: Path) -> ResolvedTargetConfig:
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
        event_dir=str(tmp / "events" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            downsample=DownsampleStageParams(),
        ),
        mapping_root=str(tmp / "mapping"),
        zarr_dir=str(tmp / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "shifted_downsampled"),
    )


class TestBackfillPs1RemovedStars(unittest.TestCase):
    def test_backfill_writes_event_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)

            event_dir = Path(resolved.event_dir)
            event_dir.mkdir(parents=True)

            ref_fits = tmp / "ref_ffi.fits"
            data = np.zeros((64, 64), dtype=np.float32)
            hdu0 = fits.PrimaryHDU()
            hdu1 = fits.ImageHDU(data=data)
            hdu1.header.update(
                {
                    "NAXIS1": 64,
                    "NAXIS2": 64,
                    "CRPIX1": 32.0,
                    "CRPIX2": 32.0,
                    "CRVAL1": 228.0,
                    "CRVAL2": 52.0,
                    "CDELT1": -0.01,
                    "CDELT2": 0.01,
                    "CTYPE1": "RA---TAN",
                    "CTYPE2": "DEC--TAN",
                }
            )
            fits.HDUList([hdu0, hdu1]).writeto(ref_fits, overwrite=True)

            (event_dir / "cluster_template_job.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sector": 22,
                        "camera": 3,
                        "ccd": 3,
                        "x_min": 0,
                        "y_min": 0,
                        "x_max": 64,
                        "y_max": 64,
                        "reference_ffi_path": str(ref_fits),
                        "groups": [{"group_dx": 0.0, "group_dy": 0.0}],
                    }
                ),
                encoding="utf-8",
            )

            removed_src = ps1_process_removed_stars_csv_path(resolved)
            removed_src.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "source_id": 1234567890123456789,
                        "ra": 228.0,
                        "dec": 52.0,
                        "tess_mag": 12.5,
                    }
                ]
            ).to_csv(removed_src, index=False)

            status = _backfill_target(resolved, dry_run=False, force=False)
            self.assertEqual(status, "wrote")
            out_csv = event_dir_ps1_removed_stars_csv_path(resolved)
            self.assertTrue(out_csv.is_file())

            status = _backfill_target(resolved, dry_run=False, force=False)
            self.assertEqual(status, "skip_exists")


if __name__ == "__main__":
    unittest.main()
