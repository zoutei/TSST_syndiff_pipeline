"""Tests for SCC mapping debug plots."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.template_creation.processing.mapping_plots import (
    MAPPING_PROJECTION_OVERLAY_BASENAME,
    write_mapping_projection_overlay,
)


class TestMappingProjectionOverlay(unittest.TestCase):
    def _write_synthetic_master(self, path: Path) -> None:
        master = np.full((32, 32), -1, dtype=np.int32)
        master[4:16, 4:16] = 0
        master[4:16, 16:28] = 1
        master[16:28, 4:28] = 2
        names = np.array(["skycell.1000.001", "skycell.1000.002", "skycell.2000.001"])
        ids = np.array([0, 1, 2], dtype=np.int32)
        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(master)
        hdu2 = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="SKYCELL", format="32A", array=names),
                fits.Column(name="SKYCIND", format="J", array=ids),
            ]
        )
        fits.HDUList([hdu0, hdu1, hdu2]).writeto(path, overwrite=True)

    def test_writes_projection_overlay_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            master_path = tmp / "master.fits"
            csv_path = tmp / "skycells.csv"
            out_path = tmp / MAPPING_PROJECTION_OVERLAY_BASENAME
            self._write_synthetic_master(master_path)
            pd.DataFrame(
                {
                    "NAME": [
                        "skycell.1000.001",
                        "skycell.1000.002",
                        "skycell.2000.001",
                    ],
                    "projection": ["1000", "1000", "2000"],
                }
            ).to_csv(csv_path, index=False)

            result = write_mapping_projection_overlay(
                master_path,
                csv_path,
                out_path,
                sector=20,
                camera=3,
                ccd=3,
            )
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.is_file())
            self.assertGreater(result.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
