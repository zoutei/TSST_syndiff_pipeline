"""Unit tests for hybrid L4a/L4b mask helpers."""

from __future__ import annotations

import unittest

import numpy as np

from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
    abutting_rim_ps1_mask,
    apply_hybrid_patch,
    build_l4a_hybrid_assignment,
    needs_recompute_mask,
    roll_assignment,
    tess_ownership_boundary,
)


class TestHybridRegmaps(unittest.TestCase):
    def test_ownership_boundary_on_checkerboard(self):
        tid = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.int64,
        )
        b = tess_ownership_boundary(tid)
        self.assertTrue(b[0, 1])  # 0|1 horizontal
        self.assertTrue(b[1, 0])  # 0|2 vertical
        self.assertFalse(b[0, 0])

    def test_needs_recompute_r1_covers_seam(self):
        tid = np.full((8, 8), 1, dtype=np.int64)
        tid[:, 4:] = 2
        tid[6:, :] = -1
        mask = needs_recompute_mask(tid, R=1)
        # seam column and footprint edge should be flagged
        self.assertTrue(mask[:, 3].any() or mask[:, 4].any())
        self.assertTrue(mask[5, :].any())

    def test_apply_hybrid_patch_replaces_masked(self):
        linear = np.arange(16, dtype=np.int64).reshape(4, 4)
        exact = linear + 100
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        out = apply_hybrid_patch(linear, exact, mask)
        np.testing.assert_array_equal(out[mask], exact[mask])
        np.testing.assert_array_equal(out[~mask], linear[~mask])

    def test_abutting_rim_ps1_mask(self):
        tid = np.array([[10, 10, 20], [10, 20, 20], [-1, 20, 30]], dtype=np.int64)
        rim = abutting_rim_ps1_mask(tid, [10, 20])
        self.assertEqual(int(rim.sum()), 7)  # three 10s + four 20s
        self.assertFalse(rim[2, 2])  # 30 not in border set

    def test_abutting_rim_ps1_mask_accepts_float_tid(self):
        """Regmap FITS files store TESS_PIXEL_MAP as whole-number floats."""
        tid = np.array(
            [[10, 10, 20], [10, 20, 20], [-1, 20, 30]], dtype=np.float32
        )
        rim = abutting_rim_ps1_mask(tid, [10, 20])
        self.assertEqual(int(rim.sum()), 7)
        self.assertFalse(rim[2, 2])

    def test_roll_assignment_roundtrip_shape(self):
        frozen = np.arange(12, dtype=np.int64).reshape(3, 4)
        rolled = roll_assignment(frozen, 1, -2, convention="assignment")
        self.assertEqual(rolled.shape, frozen.shape)

    def test_build_l4a_hybrid_assignment_patches_mask(self):
        frozen = np.full((10, 10), 5, dtype=np.int64)
        frozen[:, 5:] = 9
        exact = frozen + 1000
        hybrid, mask = build_l4a_hybrid_assignment(frozen, 0, 0, exact, hybrid_R=1)
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(hybrid[mask], exact[mask])
        np.testing.assert_array_equal(hybrid[~mask], frozen[~mask])

    def test_seam_roll_is_exact_for_shift_per_key(self):
        from syndiff_pipeline.template_creation.processing.hybrid_regmaps import (
            invalid_border_margin,
            seam_roll_is_exact_for_shift,
            stencil_roll_is_exact,
        )

        tid = np.full((20, 20), -1, dtype=np.int64)
        tid[5:15, 5:15] = 1
        self.assertGreater(invalid_border_margin(tid, R=1), 4)
        self.assertTrue(seam_roll_is_exact_for_shift(tid, 1, 0, R=1))
        self.assertFalse(seam_roll_is_exact_for_shift(tid, 10, 0, R=1))
        self.assertEqual(
            stencil_roll_is_exact(tid, 10, R=1),
            seam_roll_is_exact_for_shift(tid, 10, 0, R=1),
        )


if __name__ == "__main__":
    unittest.main()
