"""Tests for star mini-downsample and downsample array injection."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import zarr
from astropy.io import fits

from syndiff_pipeline.star.mini_downsample import (
    convolve_star_only_cutout,
    downsample_star_arrays,
    write_star_mini_templates,
)
from syndiff_pipeline.template_creation.processing.convolution_utils import (
    apply_gaussian_convolution,
)
from syndiff_pipeline.template_creation.processing.downsample import (
    combine_sparse_downsample_results,
    process_skycell_batch,
    process_skycell_batch_from_arrays,
)


def _make_identity_reg_fits(path: Path, shape: tuple[int, int]) -> None:
    """Registration map with a leading unmapped (-1) group (production-like)."""
    h, w = shape
    assignment = np.full((h, w), -1, dtype=np.int32)
    assignment[1 : h - 1, 1 : w - 1] = 0
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=assignment)
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def _make_synthetic_skycell_case(
    tmp: Path,
    shape: tuple[int, int] = (4, 4),
) -> dict:
    skycell_name = "skycell.1.2"
    ps1_data = np.arange(1, shape[0] * shape[1] + 1, dtype=np.float32).reshape(shape)
    ps1_mask = np.zeros(shape, dtype=np.uint32)
    inner = ps1_data[1 : shape[0] - 1, 1 : shape[1] - 1]

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
    base_shape = shape
    roi_bounds = (0, 0, shape[1], shape[0])

    return {
        "skycell_name": skycell_name,
        "ps1_data": ps1_data,
        "ps1_mask": ps1_mask,
        "inner_sum": float(inner.sum()),
        "reg_path": reg_path,
        "zarr_path": zarr_path,
        "offsets": offsets,
        "shifts_dict": shifts_dict,
        "base_shape": base_shape,
        "roi_bounds": roi_bounds,
    }


class TestStarMiniDownsample(unittest.TestCase):
    def test_process_skycell_batch_from_arrays_matches_zarr_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case = _make_synthetic_skycell_case(Path(tmpdir))
            common_kwargs = dict(
                batch_idx=0,
                reg_files=[str(case["reg_path"])],
                skycell_names=[case["skycell_name"]],
                offsets=case["offsets"],
                shifts_dict=case["shifts_dict"],
                base_tess_shape=case["base_shape"],
                roi_bounds=case["roi_bounds"],
            )

            zarr_result = process_skycell_batch(
                zarr_path=case["zarr_path"],
                **common_kwargs,
            )
            arrays_result = process_skycell_batch_from_arrays(
                arrays={
                    case["skycell_name"]: (case["ps1_data"], case["ps1_mask"])
                },
                **common_kwargs,
            )

            for zarr_part, arrays_part in zip(zarr_result, arrays_result):
                np.testing.assert_array_equal(zarr_part, arrays_part)

            zarr_dense = combine_sparse_downsample_results(
                [zarr_result],
                case["offsets"],
                case["base_shape"],
                case["roi_bounds"],
            )
            arrays_dense = downsample_star_arrays(
                arrays={case["skycell_name"]: (case["ps1_data"], case["ps1_mask"])},
                reg_files=[str(case["reg_path"])],
                skycell_names=[case["skycell_name"]],
                offsets=case["offsets"],
                shifts_dict=case["shifts_dict"],
                base_tess_shape=case["base_shape"],
                roi_bounds=case["roi_bounds"],
            )
            np.testing.assert_allclose(zarr_dense, arrays_dense)
            self.assertEqual(zarr_dense.shape, (1, 3, 4, 4))
            self.assertAlmostEqual(float(zarr_dense[0, 0, 0, 0]), case["inner_sum"])

    def test_process_skycell_batch_regression_identity_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case = _make_synthetic_skycell_case(Path(tmpdir))
            indices, sums, counts, mask_counts = process_skycell_batch(
                batch_idx=0,
                reg_files=[str(case["reg_path"])],
                skycell_names=[case["skycell_name"]],
                offsets=case["offsets"],
                shifts_dict=case["shifts_dict"],
                base_tess_shape=case["base_shape"],
                zarr_path=case["zarr_path"],
                roi_bounds=case["roi_bounds"],
            )
            self.assertEqual(len(indices), 1)
            dense = combine_sparse_downsample_results(
                [(indices, sums, counts, mask_counts)],
                case["offsets"],
                case["base_shape"],
                case["roi_bounds"],
            )
            self.assertAlmostEqual(float(dense[0, 0, 0, 0]), case["inner_sum"])
            self.assertEqual(float(dense[0, 1, 0, 0]), 4.0)
            self.assertEqual(float(dense[0, 2, 0, 0]), 0.0)

    def test_convolve_star_only_cutout_matches_production_and_stays_in_window(self):
        shape = (2000, 2000)
        image = np.zeros(shape, dtype=np.float32)
        cy, cx = 1000, 1000
        flux = 123.45
        image[cy, cx] = flux
        margin_px = 470
        psf_sigma = 60.0

        convolved, origin = convolve_star_only_cutout(
            image,
            psf_sigma=psf_sigma,
            margin_px=margin_px,
        )
        y0, x0 = origin
        self.assertGreaterEqual(y0, cy - margin_px)
        self.assertGreaterEqual(x0, cx - margin_px)
        self.assertLessEqual(y0 + convolved.shape[0], cy + margin_px + 1)
        self.assertLessEqual(x0 + convolved.shape[1], cx + margin_px + 1)

        expected = apply_gaussian_convolution(
            image[y0 : y0 + convolved.shape[0], x0 : x0 + convolved.shape[1]],
            sigma=psf_sigma,
            radius=margin_px,
            cval=0.0,
        )
        np.testing.assert_allclose(convolved, expected, rtol=0, atol=0)

        full_embed = np.zeros(shape, dtype=np.float32)
        full_embed[y0 : y0 + convolved.shape[0], x0 : x0 + convolved.shape[1]] = convolved
        outside = full_embed.copy()
        outside[y0 : y0 + convolved.shape[0], x0 : x0 + convolved.shape[1]] = 0.0
        self.assertEqual(float(np.nansum(np.abs(outside))), 0.0)

    def test_convolve_star_only_cutout_conserves_flux_without_nans(self):
        shape = (4800, 4800)
        image = np.zeros(shape, dtype=np.float32)
        cy, cx = 2400, 2400
        yy, xx = np.mgrid[cy - 5 : cy + 6, cx - 5 : cx + 6]
        blob = 1000.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
        image[cy - 5 : cy + 6, cx - 5 : cx + 6] = blob.astype(np.float32)
        input_flux = float(np.sum(image))

        convolved, _origin = convolve_star_only_cutout(image, psf_sigma=60.0)
        self.assertFalse(np.isnan(convolved).any())
        self.assertAlmostEqual(float(np.sum(convolved)), input_flux, delta=1.0)
        self.assertGreater(np.count_nonzero(convolved > 0.01 * np.max(convolved)), 100)

    def test_write_star_mini_templates_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "mini_templates"
            arrays = np.zeros((1, 3, 4, 4), dtype=np.float32)
            arrays[0, 0] = np.arange(16, dtype=np.float32).reshape(4, 4)
            arrays[0, 1] = 1
            arrays[0, 2] = 0
            offsets = np.array([[0.25, -0.5]], dtype=np.float64)
            metadata = {
                "gaia_source_id": 1060421588522505216,
                "tic_id": 142748283,
                "sector": 20,
                "camera": 3,
                "ccd": 2,
            }

            written = write_star_mini_templates(
                out_dir,
                arrays,
                offsets=offsets,
                roi_origin=(10, 20),
                host_identifier_metadata=metadata,
            )
            self.assertEqual(len(written), 1)
            self.assertTrue(Path(written[0]).is_file())

            with fits.open(written[0]) as hdul:
                self.assertEqual(hdul[1].name, "FLUX_SUM")
                self.assertEqual(hdul[2].name, "COUNT")
                self.assertEqual(hdul[3].name, "MASK")
                header = hdul[1].header
                self.assertEqual(header["XMIN"], 10)
                self.assertEqual(header["YMIN"], 20)
                self.assertEqual(header["GAIA_SOURCE_ID"], 1060421588522505216)
                self.assertEqual(header["TIC_ID"], 142748283)
                np.testing.assert_allclose(hdul[1].data, arrays[0, 0])
                np.testing.assert_array_equal(hdul[2].data, arrays[0, 1].astype(np.int32))
                np.testing.assert_array_equal(hdul[3].data, arrays[0, 2].astype(np.int32))


if __name__ == "__main__":
    unittest.main()
