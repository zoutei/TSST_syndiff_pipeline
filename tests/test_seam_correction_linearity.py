"""Re-landed science guardrails for PR5's cross-projection seam correction
(``doc/template_bookkeeping_plan.md`` §13, decision #4).

The canonical convolved cell (plan §13) is convolved on a master array
padded by **same-projection** neighbors only; anything a cross-projection
neighbor would have contributed is missing at publish time. The plan
documents two validated findings this module re-derives against the real
production convolution (``convolution_utils.apply_gaussian_convolution``,
sigma~60, radius~470 regime) instead of a synthetic stand-in:

(a) **Linearity** (to ~1e-12 relative, plan claims ~1e-15): Gaussian
    convolution is linear, so
    ``convolve(A + B) == convolve(A) + convolve(B)``.
    This is *why* the exact seam correction can be computed as
    ``convolve(canonical cell with the gap zeroed) + convolve(the
    reprojected neighbor patch alone, at its true position)``, added --
    without ever needing the two arrays' union to exist in memory at once.

(b) **Bias guard**: the "just zero the gap and convolve" shortcut is *not*
    a safe approximation. It silently drops the neighbor's blurred-in
    contribution near the seam -- a large, sharply-edge-localized flux
    deficit that decays away over about one truncation radius. This test
    quantifies that deficit to document why the exact correction in (a) is
    mandatory, not optional polish.

Geometry used for both tests: a two-region synthetic domain split at
column ``gap`` --

    A = canonical cell, gap zeroed  (flux only in columns [0, gap))
    B = neighbor patch, alone       (flux only in columns [gap, W))
    A + B = the (unavailable in production) union of both cells

``cval=0.0`` is used throughout (not the production default ``np.nan``) so
NaN never contaminates the linearity/deficit arithmetic -- consistent with
``test_convolution_utils.py``'s existing use of ``cval=0.0`` for flux-
conserving synthetic tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing.convolution_utils import (
    apply_gaussian_convolution,
)

# Production regime (ps1_process.py's convolution step / convolved_store's
# DEFAULT_PSF_SIGMA / DEFAULT_RADIUS): sigma=60, radius=470 -> truncate~7.83.
SIGMA = 60.0
RADIUS = 470

# 600x600 is comfortably larger than RADIUS (dask_image's map_overlap depth
# must be smaller than the array extent) while staying fast: ~0.5s/call.
_H, _W = 600, 600
_GAP = _W // 2
_ROW = _H // 2  # sample far from the top/bottom array edges

_CANONICAL_FLUX = 50.0  # flat "canonical cell" background level
_NEIGHBOR_FLUX = 500.0  # flat "neighbor patch" level (brighter, so its
# missing contribution is unambiguous against the canonical background)


def _two_region_domain() -> tuple[np.ndarray, np.ndarray]:
    """Build (A, B): canonical-with-gap-zeroed and neighbor-patch-alone."""
    a = np.zeros((_H, _W), dtype=np.float64)
    a[:, :_GAP] = _CANONICAL_FLUX
    b = np.zeros((_H, _W), dtype=np.float64)
    b[:, _GAP:] = _NEIGHBOR_FLUX
    return a, b


@pytest.fixture(scope="module")
def convolved_regime():
    """Convolve A, B, and A+B once and share across both tests below."""
    a, b = _two_region_domain()
    conv_a = apply_gaussian_convolution(a, sigma=SIGMA, radius=RADIUS, cval=0.0)
    conv_b = apply_gaussian_convolution(b, sigma=SIGMA, radius=RADIUS, cval=0.0)
    conv_full = apply_gaussian_convolution(a + b, sigma=SIGMA, radius=RADIUS, cval=0.0)
    return a, b, conv_a, conv_b, conv_full


def test_gaussian_convolution_is_linear(convolved_regime):
    """convolve(A) + convolve(B) == convolve(A + B) to ~1e-12 relative.

    This is the exact property the seam correction relies on: the canonical
    cell's convolution and the reprojected neighbor patch's convolution can
    be computed independently (as production does, one per row-step) and
    simply added -- the result is mathematically identical to having
    convolved the true, unavailable union of both cells' data.
    """
    _, _, conv_a, conv_b, conv_full = convolved_regime

    reconstructed = conv_a + conv_b
    max_abs_err = float(np.max(np.abs(conv_full - reconstructed)))
    max_abs_val = float(np.max(np.abs(conv_full)))
    rel_err = max_abs_err / max_abs_val

    assert rel_err < 1e-10, (
        f"Gaussian convolution linearity violated beyond floating-point noise: "
        f"rel_err={rel_err:.3e} (expected ~1e-15-1e-12)"
    )


def test_naive_zero_gap_shortcut_has_significant_edge_flux_deficit(convolved_regime):
    """Quantify why skipping the neighbor-patch correction is unsafe.

    ``naive`` = what production would compute if it just zeroed the
    cross-projection gap and convolved (the unsafe shortcut).
    ``exact`` = ``naive + convolve(neighbor patch alone)`` = the correction
    from decision #4 / plan §13 = (by the linearity test above)
    ``convolve(A + B)``, i.e. what you'd get if the neighbor's real data had
    been available.

    The deficit (``exact - naive``) must be a large fraction of the true
    flux immediately at the seam edge, and must decay to negligible as you
    move away from the edge into the interior of the canonical cell --
    exactly the "~50% at the edge, tapering over ~1 truncation radius"
    shape the plan documents.
    """
    _, _, conv_a, conv_b, conv_full = convolved_regime

    naive = conv_a
    exact = conv_full  # == conv_a + conv_b, established by the linearity test
    deficit = exact - naive  # == conv_b: exactly the missing neighbor contribution

    def relative_deficit_at(offset_from_edge: int) -> float:
        col = _GAP - offset_from_edge  # inside the canonical region, near the seam
        return float(deficit[_ROW, col] / exact[_ROW, col])

    near_edge = relative_deficit_at(5)
    mid_range = relative_deficit_at(100)
    far_from_edge = relative_deficit_at(250)

    # (1) Significant deficit right at the seam: the naive shortcut is
    # missing the majority of the true local flux there.
    assert near_edge > 0.5, (
        f"expected a large (>50%) relative flux deficit at the seam edge from "
        f"the naive zero-gap shortcut, got {near_edge:.1%}"
    )

    # (2) The deficit decays away from the edge...
    assert mid_range < near_edge
    assert far_from_edge < mid_range

    # (3) ...to something negligible about half a truncation radius (~235px)
    # into the canonical cell's interior, away from the seam.
    assert far_from_edge < 0.01, (
        f"expected the flux deficit to have decayed to <1% by 250px from the "
        f"seam (~half the {RADIUS}px truncation radius), got {far_from_edge:.2%}"
    )

    # Sanity: the deficit is strictly positive everywhere sampled (naive
    # always *underestimates* -- it never overshoots).
    assert naive[_ROW, _GAP - 5] < exact[_ROW, _GAP - 5]
    assert naive[_ROW, _GAP - 250] < exact[_ROW, _GAP - 250]
