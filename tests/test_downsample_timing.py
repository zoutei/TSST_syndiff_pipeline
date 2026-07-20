"""Tests for downsample batch timing logs and log_level."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import zarr
from astropy.io import fits

from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    parse_stage_params,
)
from syndiff_pipeline.template_creation.processing.downsample import (
    process_skycell_batch,
    process_skycell_batch_from_arrays,
)


def _make_identity_reg_fits(path: Path, shape: tuple[int, int]) -> None:
    h, w = shape
    assignment = np.full((h, w), -1, dtype=np.int32)
    assignment[1 : h - 1, 1 : w - 1] = 0
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=assignment)
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


class TestDownsampleTiming(unittest.TestCase):
    def _synthetic_case(self, tmp: Path) -> dict:
        skycell_name = "skycell.1.2"
        shape = (4, 4)
        ps1_data = np.arange(1, shape[0] * shape[1] + 1, dtype=np.float32).reshape(shape)
        ps1_mask = np.zeros(shape, dtype=np.uint32)
        reg_path = tmp / "reg.fits"
        _make_identity_reg_fits(reg_path, shape)
        zarr_path = tmp / "convolved.zarr"
        root = zarr.open(str(zarr_path), mode="w")
        root[f"{skycell_name}_data"] = ps1_data
        root[f"{skycell_name}_mask"] = ps1_mask
        offsets = np.array([[0.0, 0.0]], dtype=np.float64)
        shifts_dict = {
            (0.0, 0.0): pd.DataFrame(
                {"NAME": [skycell_name], "shift_x": [0], "shift_y": [0]}
            )
        }
        return {
            "skycell_name": skycell_name,
            "ps1_data": ps1_data,
            "ps1_mask": ps1_mask,
            "reg_path": reg_path,
            "zarr_path": zarr_path,
            "offsets": offsets,
            "shifts_dict": shifts_dict,
            "base_shape": shape,
            "roi_bounds": (0, 0, shape[1], shape[0]),
        }

    def test_batch_summary_line_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._synthetic_case(Path(tmpdir))
            buf = io.StringIO()
            with redirect_stdout(buf):
                process_skycell_batch(
                    batch_idx=0,
                    reg_files=[str(case["reg_path"])],
                    skycell_names=[case["skycell_name"]],
                    offsets=case["offsets"],
                    shifts_dict=case["shifts_dict"],
                    base_tess_shape=case["base_shape"],
                    zarr_path=case["zarr_path"],
                    roi_bounds=case["roi_bounds"],
                    total_batches=3,
                )
            line = next(
                ln for ln in buf.getvalue().splitlines() if ln.startswith("[downsample] batch")
            )
            self.assertIn("batch 1/3 done skycells=1", line)
            self.assertIn("elapsed=", line)
            self.assertIn("zarr=", line)
            self.assertIn("regmap=", line)
            self.assertIn("binning=", line)

    def test_debug_emits_per_skycell_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case = self._synthetic_case(Path(tmpdir))
            buf = io.StringIO()
            with redirect_stdout(buf):
                process_skycell_batch_from_arrays(
                    batch_idx=0,
                    reg_files=[str(case["reg_path"])],
                    skycell_names=[case["skycell_name"]],
                    arrays={case["skycell_name"]: (case["ps1_data"], case["ps1_mask"])},
                    offsets=case["offsets"],
                    shifts_dict=case["shifts_dict"],
                    base_tess_shape=case["base_shape"],
                    roi_bounds=case["roi_bounds"],
                    log_level="DEBUG",
                )
            out = buf.getvalue()
            self.assertIn("[downsample] skycell skycell.1.2", out)
            self.assertIn("regmap=", out)

    def test_log_level_stage_params(self):
        params = parse_stage_params({"downsample": {"log_level": "debug"}})
        self.assertEqual(params.downsample.log_level, "DEBUG")

    def test_invalid_log_level_raises(self):
        with self.assertRaises(ValueError):
            DownsampleStageParams(log_level="TRACE")


if __name__ == "__main__":
    unittest.main()
