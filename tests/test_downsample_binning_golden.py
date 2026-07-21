"""Golden tests: vectorized downsample binning matches reference loop semantics."""
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
from astropy.io import fits

from syndiff_pipeline.template_creation.processing.downsample import (
    _aggregate_sorted_groups,
    _process_skycell_registration_binning,
    combine_sparse_downsample_results,
)


def _binning_reference_loop(
    ps1_rav: np.ndarray,
    ps1_mask_rav: np.ndarray,
    breaks: np.ndarray,
    ignore_mask: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original per-group Python loop used before vectorization."""
    n_groups = len(breaks) - 1
    sums = np.zeros(n_groups, dtype=np.float32)
    counts = np.zeros(n_groups, dtype=np.int32)
    mask_counts = np.zeros(n_groups, dtype=np.int32)

    for i in range(n_groups):
        slice_data = ps1_rav[breaks[i] : breaks[i + 1]]
        slice_mask = ps1_mask_rav[breaks[i] : breaks[i + 1]]
        ignored_pixels = (slice_mask & ignore_mask) > 0
        counts[i] = len(slice_data)
        sums[i] = np.nansum(slice_data[~ignored_pixels])
        mask_counts[i] = np.sum(slice_mask != 0)

    return sums, counts, mask_counts


def _combine_reference_loop(
    combined_indices: np.ndarray,
    combined_sums: np.ndarray,
    combined_counts: np.ndarray,
    combined_mask_counts: np.ndarray,
    offsets: np.ndarray,
    base_tess_shape: tuple[int, int],
    roi_bounds: tuple[int, int, int, int],
    oversampling_factor: int,
) -> np.ndarray:
    """Original scatter loop used before vectorization."""
    x_min, y_min, x_max, y_max = roi_bounds
    roi_h = y_max - y_min
    roi_w = x_max - x_min
    out_h = roi_h * oversampling_factor
    out_w = roi_w * oversampling_factor
    combined_results = np.zeros((len(offsets), 3, out_h, out_w), dtype=np.float32)

    for i, idx in enumerate(combined_indices):
        if oversampling_factor > 1:
            os_width = base_tess_shape[1] * oversampling_factor
            y_os = idx // os_width
            x_os = idx % os_width
            y_base = y_os // oversampling_factor
            x_base = x_os // oversampling_factor
            sub_y = y_os % oversampling_factor
            sub_x = x_os % oversampling_factor
            if x_min <= x_base < x_max and y_min <= y_base < y_max:
                out_y = (y_base - y_min) * oversampling_factor + sub_y
                out_x = (x_base - x_min) * oversampling_factor + sub_x
            else:
                continue
        else:
            y_base = idx // base_tess_shape[1]
            x_base = idx % base_tess_shape[1]
            if x_min <= x_base < x_max and y_min <= y_base < y_max:
                out_y = y_base - y_min
                out_x = x_base - x_min
            else:
                continue

        if 0 <= out_y < out_h and 0 <= out_x < out_w:
            for offset_idx in range(len(offsets)):
                combined_results[offset_idx, 0, out_y, out_x] = combined_sums[i, offset_idx]
                combined_results[offset_idx, 1, out_y, out_x] = combined_counts[i, offset_idx]
                combined_results[offset_idx, 2, out_y, out_x] = combined_mask_counts[i, offset_idx]

    return combined_results


def _make_assignment_map(
    shape: tuple[int, int],
    pixel_groups: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    """Build a registration map: pixel index -> list of (y, x) PS1 coords."""
    h, w = shape
    assignment = np.full((h, w), -1, dtype=np.int32)
    for pix_idx, coords in pixel_groups.items():
        for y, x in coords:
            assignment[y, x] = pix_idx
    return assignment


class TestDownsampleBinningGolden(unittest.TestCase):
    def test_aggregate_sorted_groups_matches_reference(self):
        rng = np.random.default_rng(42)
        n = 200
        ps1_rav = rng.normal(size=n).astype(np.float32)
        ps1_rav[rng.choice(n, size=10, replace=False)] = np.nan
        ps1_mask_rav = rng.integers(0, 1 << 14, size=n, dtype=np.uint32)
        ignore_mask = 1 << 12

        pind = np.repeat(np.arange(8), np.linspace(5, 30, 8, dtype=int))
        sort_ind = np.argsort(pind)
        breaks = np.where(np.diff(pind[sort_ind]) > 0)[0] + 1
        breaks = np.append(breaks, len(sort_ind))
        group_starts = breaks[:-1]

        ref_sums, ref_counts, ref_mask_counts = _binning_reference_loop(
            ps1_rav[sort_ind],
            ps1_mask_rav[sort_ind],
            breaks,
            ignore_mask,
        )
        vec_sums, vec_counts, vec_mask_counts = _aggregate_sorted_groups(
            ps1_rav[sort_ind],
            ps1_mask_rav[sort_ind],
            group_starts,
            ignore_mask,
        )

        np.testing.assert_allclose(vec_sums, ref_sums, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(vec_counts, ref_counts)
        np.testing.assert_array_equal(vec_mask_counts, ref_mask_counts)

    def test_aggregate_sorted_groups_handles_narrow_mask_dtype(self):
        """PS1 mask arrays on disk can be uint8; ignore_mask bits beyond uint8
        range (e.g. bit 12 = 4096) must not raise on the bitwise AND — a
        narrower-than-the-bit mask simply never matches that bit."""
        n = 50
        ps1_rav = np.ones(n, dtype=np.float32)
        ps1_mask_rav = np.zeros(n, dtype=np.uint8)
        ps1_mask_rav[::3] = 1  # some bit-0 flags, well within uint8
        ignore_mask = 1 << 12

        group_starts = np.array([0, 10, 25], dtype=np.intp)
        sums, counts, mask_counts = _aggregate_sorted_groups(
            ps1_rav, ps1_mask_rav, group_starts, ignore_mask
        )
        # No uint8 value can have bit 12 set, so nothing is ignored: sums ==
        # counts (all ones, none dropped).
        np.testing.assert_array_equal(sums, counts.astype(np.float32))

    def test_process_skycell_binning_nans_and_ignore_bits(self):
        shape = (6, 6)
        assignment = _make_assignment_map(
            shape,
            {
                0: [(1, 1), (1, 2), (2, 1)],
                5: [(3, 3), (3, 4)],
            },
        )
        ps1_data = np.arange(1, shape[0] * shape[1] + 1, dtype=np.float32).reshape(shape)
        ps1_data[1, 2] = np.nan
        ps1_mask = np.zeros(shape, dtype=np.uint32)
        ps1_mask[2, 1] = 1 << 12  # ignored for sums, still counted in counts

        offsets = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        shifts_dict = {
            (0.0, 0.0): pd.DataFrame(
                {"NAME": ["skycell.9.9"], "shift_x": [0], "shift_y": [0]}
            ),
            (0.5, 0.5): pd.DataFrame(
                {"NAME": ["skycell.9.9"], "shift_x": [1], "shift_y": [0]}
            ),
        }

        result = _process_skycell_registration_binning(
            ps1_assignment=assignment,
            ps1_data=ps1_data,
            ps1_mask=ps1_mask,
            skycell_name="skycell.9.9",
            offsets=offsets,
            shifts_dict=shifts_dict,
            base_tess_shape=(4, 4),
            roi_bounds=(0, 0, 4, 4),
            oversampling_factor=1,
            ignore_mask=1 << 12,
        )
        self.assertIsNotNone(result)
        tess_pixels, sums, counts, mask_counts = result

        # Pixel 0: values 8, nan, 15 (ignored) -> sum 8, count 3, mask_count 1
        pix0 = np.where(tess_pixels == 0)[0][0]
        self.assertEqual(int(counts[pix0, 0]), 3)
        self.assertAlmostEqual(float(sums[pix0, 0]), 8.0)
        self.assertEqual(int(mask_counts[pix0, 0]), 1)
        self.assertEqual(int(mask_counts[pix0, 0]), 1)

        # Second offset uses roll(0,1): different grouping
        self.assertEqual(sums.shape[1], 2)

    def test_combine_sparse_matches_reference_single_and_multi_offset(self):
        base_shape = (8, 8)
        roi_bounds = (1, 1, 6, 6)
        offsets = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        indices = np.array([10, 11, 50, 51], dtype=int)
        sums = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            dtype=np.float32,
        )
        counts = np.array(
            [[1, 10], [2, 20], [3, 30], [4, 40]],
            dtype=np.int32,
        )
        mask_counts = np.zeros_like(counts)

        batch = [(indices, sums, counts, mask_counts)]
        vec = combine_sparse_downsample_results(
            batch, offsets, base_shape, roi_bounds, oversampling_factor=1
        )
        ref = _combine_reference_loop(
            indices, sums, counts, mask_counts, offsets, base_shape, roi_bounds, 1
        )
        np.testing.assert_allclose(vec, ref, rtol=0, atol=0)

    def test_combine_sparse_oversampling_and_duplicate_indices(self):
        os_factor = 2
        base_shape = (4, 4)
        roi_bounds = (0, 0, 4, 4)
        offsets = np.array([[0.0, 0.0]], dtype=np.float64)
        os_width = base_shape[1] * os_factor

        # Two contributions to the same oversampled pixel (index 5)
        idx_a = 5
        idx_b = 5
        sums = np.array([[1.5], [2.5]], dtype=np.float32)
        counts = np.array([[2], [3]], dtype=np.int32)
        mask_counts = np.array([[1], [0]], dtype=np.int32)

        batch = [
            (np.array([idx_a], dtype=int), sums[:1], counts[:1], mask_counts[:1]),
            (np.array([idx_b], dtype=int), sums[1:], counts[1:], mask_counts[1:]),
        ]
        vec = combine_sparse_downsample_results(
            batch, offsets, base_shape, roi_bounds, oversampling_factor=os_factor
        )

        indices_cat = np.array([idx_a, idx_b], dtype=int)
        ref = _combine_reference_loop(
            indices_cat,
            sums,
            counts,
            mask_counts,
            offsets,
            base_shape,
            roi_bounds,
            os_factor,
        )
        # combine deduplicates via np.add.at before scatter
        unique_idx, inv = np.unique(indices_cat, return_inverse=True)
        ref_sums = np.zeros((len(unique_idx), 1), dtype=np.float32)
        ref_counts = np.zeros((len(unique_idx), 1), dtype=np.int32)
        ref_mask = np.zeros((len(unique_idx), 1), dtype=np.int32)
        np.add.at(ref_sums, inv, sums)
        np.add.at(ref_counts, inv, counts)
        np.add.at(ref_mask, inv, mask_counts)
        ref_dedup = _combine_reference_loop(
            unique_idx,
            ref_sums,
            ref_counts,
            ref_mask,
            offsets,
            base_shape,
            roi_bounds,
            os_factor,
        )
        np.testing.assert_allclose(vec, ref_dedup, rtol=0, atol=0)

        y_os = idx_a // os_width
        x_os = idx_a % os_width
        out_y = y_os
        out_x = x_os
        self.assertAlmostEqual(float(vec[0, 0, out_y, out_x]), 4.0)
        self.assertEqual(float(vec[0, 1, out_y, out_x]), 5.0)
        self.assertEqual(float(vec[0, 2, out_y, out_x]), 1.0)

    def test_end_to_end_binning_and_combine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shape = (5, 5)
            assignment = np.full(shape, -1, dtype=np.int32)
            assignment[1:4, 1:4] = 0  # maps to TESS pixel 0

            reg_path = Path(tmpdir) / "reg.fits"
            hdu0 = fits.PrimaryHDU()
            hdu1 = fits.ImageHDU(data=assignment)
            fits.HDUList([hdu0, hdu1]).writeto(reg_path, overwrite=True)

            ps1_data = np.ones(shape, dtype=np.float32)
            ps1_data[2, 2] = np.nan
            ps1_mask = np.zeros(shape, dtype=np.uint32)
            ps1_mask[2, 2] = 1 << 12

            offsets = np.array([[0.0, 0.0]], dtype=np.float64)
            shifts_dict = {
                (0.0, 0.0): pd.DataFrame(
                    {"NAME": ["skycell.1.1"], "shift_x": [0], "shift_y": [0]}
                )
            }

            contrib = _process_skycell_registration_binning(
                ps1_assignment=assignment,
                ps1_data=ps1_data,
                ps1_mask=ps1_mask,
                skycell_name="skycell.1.1",
                offsets=offsets,
                shifts_dict=shifts_dict,
                base_tess_shape=shape,
                roi_bounds=(0, 0, shape[1], shape[0]),
                oversampling_factor=1,
                ignore_mask=1 << 12,
            )
            self.assertIsNotNone(contrib)
            dense = combine_sparse_downsample_results(
                [contrib],
                offsets,
                shape,
                (0, 0, shape[1], shape[0]),
            )
            # 9 pixels, nan+ignored at center excluded from sum
            self.assertAlmostEqual(float(dense[0, 0, 0, 0]), 8.0)
            self.assertEqual(float(dense[0, 1, 0, 0]), 9.0)


if __name__ == "__main__":
    unittest.main()
