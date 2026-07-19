"""Locks in the PR5 seam-correction finding as a real regression test.

Uses the REAL production convolution function
(``convolution_utils.apply_gaussian_convolution``, at the pipeline's actual
defaults sigma=60/radius=470) to prove, with synthetic but controlled arrays,
that Gaussian convolution's linearity gives an EXACT (not approximate)
cross-projection seam correction strategy for Phase 2:

    convolve(canonical_gap_zero) + convolve(patch_alone) == convolve(correct)

where ``canonical_gap_zero`` is the same-projection-only padded array (the
sky-addressed, shareable product) with the cross-projection gap left at zero,
and ``patch_alone`` is an all-zero array of the same shape with ONLY the
reprojected neighbor content placed in the gap. See
``doc/template_bookkeeping_plan.md`` §13 for the full writeup and why this
matters: without this correction, leaving the gap zero-filled produces a real,
systematic flux deficit (tens of percent) within one truncation radius of a
true cross-projection seam — this is NOT validated as "close enough" to skip.
"""

from __future__ import annotations

import unittest

import numpy as np

from syndiff_pipeline.template_creation.processing.convolution_utils import (
    apply_gaussian_convolution,
)

# Match production defaults (ps1_process passes psf_sigma; convolution_utils'
# own default radius=470 is what the pipeline actually uses).
SIGMA = 60.0
RADIUS = 470
PAD_SIZE = 480


def _synthetic_field(width: int, height: int, *, background: float, seed: int) -> np.ndarray:
    """Background + a handful of Gaussian point sources, deterministic per seed."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    arr = np.full((height, width), background, dtype=np.float64)
    for _ in range(20):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        amp = rng.uniform(50, 500)
        sig = rng.uniform(1.5, 4.0)
        arr += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))
    return arr


class SeamCorrectionLinearityTests(unittest.TestCase):
    """Exact-correction proof: convolve(A) + convolve(B) == convolve(A+B)."""

    def setUp(self) -> None:
        self.height, self.width = 1200, 900
        self.gap_x0 = self.width // 2 - PAD_SIZE // 2
        self.gap_x1 = self.width // 2 + PAD_SIZE // 2

        local = _synthetic_field(self.width, self.height, background=100.0, seed=11)
        neighbor = _synthetic_field(self.width, self.height, background=100.0, seed=22)

        self.correct = local.copy()
        self.correct[:, self.gap_x0 : self.gap_x1] = neighbor[:, self.gap_x0 : self.gap_x1]

        self.canonical_gap_zero = local.copy()
        self.canonical_gap_zero[:, self.gap_x0 : self.gap_x1] = 0.0

        self.patch_alone = np.zeros_like(local)
        self.patch_alone[:, self.gap_x0 : self.gap_x1] = neighbor[:, self.gap_x0 : self.gap_x1]

    def test_additive_seam_correction_is_numerically_exact(self) -> None:
        conv_correct = apply_gaussian_convolution(self.correct, sigma=SIGMA, radius=RADIUS, cval=0.0)
        conv_canonical = apply_gaussian_convolution(
            self.canonical_gap_zero, sigma=SIGMA, radius=RADIUS, cval=0.0
        )
        conv_patch = apply_gaussian_convolution(self.patch_alone, sigma=SIGMA, radius=RADIUS, cval=0.0)

        reconstructed = conv_canonical + conv_patch
        # Exact to floating-point precision (proves this is a real linear
        # decomposition, not a "close enough" approximation): tight atol.
        np.testing.assert_allclose(reconstructed, conv_correct, atol=1e-8, rtol=1e-10)

    def test_uncorrected_gap_zero_fill_has_material_seam_bias(self) -> None:
        """Guards against silently 'simplifying' Phase 2 to skip the correction.

        If someone ships the canonical (gap-zero) cell as-is near a seam
        without the additive patch correction, this asserts the resulting
        error is large enough to matter (not a rounding-level discrepancy),
        so that shortcut can never be mistaken for "good enough" by a future
        change that weakens or removes the correction step.
        """
        conv_correct = apply_gaussian_convolution(self.correct, sigma=SIGMA, radius=RADIUS, cval=0.0)
        conv_canonical = apply_gaussian_convolution(
            self.canonical_gap_zero, sigma=SIGMA, radius=RADIUS, cval=0.0
        )
        center_y = self.height // 2
        at_seam_edge = self.gap_x0 - 1
        local_flux = conv_correct[center_y, at_seam_edge]
        err_fraction = abs(conv_canonical[center_y, at_seam_edge] - local_flux) / local_flux
        # Empirically ~50% at the immediate seam edge; assert it's at least
        # material (>10%) so this test fails loudly if the synthetic setup
        # ever drifts into a regime where the bias looks negligible.
        self.assertGreater(err_fraction, 0.10)


if __name__ == "__main__":
    unittest.main()
