"""API-independent oracle for the MAPGRID=3 paired-padding contract.

This fixture deliberately does not call ``MappingGrid`` or any diff-stage
pairing helper.  It records the geometry and expected result that P1/P6 must
implement.  The kernel support is measured from the current Hotpants defaults:
``rkernel=int(2.5 * sci_fwhm)=4`` and the existing four-pixel spare margin
therefore fixes the native template margin at ``P=8``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syndiff_pipeline.common.mapping_grid import compute_conv_pad_native, compute_rkernel
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams


@dataclass(frozen=True)
class PairedPaddingOracle:
    """Small deterministic paired science/template case for all four edges."""

    science_shape: tuple[int, int]
    pad: int
    template_support: np.ndarray
    science_padded: np.ndarray
    science_invalid: np.ndarray
    expected_trimmed: np.ndarray


def _reference_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reference zero-boundary convolution, written independently of stages."""
    radius = kernel.shape[0] // 2
    out = np.zeros_like(image, dtype=np.float64)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            total = 0.0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    iy, ix = y + dy, x + dx
                    if 0 <= iy < image.shape[0] and 0 <= ix < image.shape[1]:
                        total += kernel[dy + radius, dx + radius] * image[iy, ix]
            out[y, x] = total
    return out


def _oracle() -> PairedPaddingOracle:
    # The production defaults are sci_fwhm=1.88 and a four-pixel spare margin.
    hp = HotpantsParams()
    rkernel = compute_rkernel(hp.sci_fwhm)
    assert rkernel == 4, "update the oracle when the Hotpants default changes"
    pad = compute_conv_pad_native(rkernel, template_conv_pad_spare_px=4)
    assert pad == 8, "the MAPGRID=3 margin must be reviewed with the kernel change"

    science_shape = (6, 8)
    template_shape = tuple(dim + 2 * pad for dim in science_shape)
    template = np.zeros(template_shape, dtype=np.float64)

    # Impulses on each science edge and corner.  They are represented in the
    # template-support array, not copied from the fabricated science pad.
    science_points = {
        "top_left": (0, 0),
        "top": (0, science_shape[1] // 2),
        "top_right": (0, science_shape[1] - 1),
        "left": (science_shape[0] // 2, 0),
        "right": (science_shape[0] // 2, science_shape[1] - 1),
        "bottom_left": (science_shape[0] - 1, 0),
        "bottom": (science_shape[0] - 1, science_shape[1] // 2),
        "bottom_right": (science_shape[0] - 1, science_shape[1] - 1),
    }
    for value, (y, x) in enumerate(science_points.values(), start=1):
        template[pad + y, pad + x] = float(value)

    # Also place template-only support impulses one kernel radius outside each
    # science edge.  These are the pixels that would be lost if the template
    # were cropped to S before convolution.
    support_points = {
        "support_top_left": (pad - rkernel, pad - rkernel),
        "support_top": (pad - rkernel, pad + science_shape[1] // 2),
        "support_top_right": (pad - rkernel, pad + science_shape[1] - 1 + rkernel),
        "support_left": (pad + science_shape[0] // 2, pad - rkernel),
        "support_right": (pad + science_shape[0] // 2, pad + science_shape[1] - 1 + rkernel),
        "support_bottom_left": (pad + science_shape[0] - 1 + rkernel, pad - rkernel),
        "support_bottom": (pad + science_shape[0] - 1 + rkernel, pad + science_shape[1] // 2),
        "support_bottom_right": (
            pad + science_shape[0] - 1 + rkernel,
            pad + science_shape[1] - 1 + rkernel,
        ),
    }
    for value, (y, x) in enumerate(support_points.values(), start=10):
        template[y, x] = float(value)

    # An asymmetric, radius-four kernel makes every edge contribution
    # observable and catches orientation/trim mistakes without depending on
    # pyhotpants internals.
    kernel = np.zeros((2 * rkernel + 1, 2 * rkernel + 1), dtype=np.float64)
    kernel[rkernel, rkernel] = 1.0
    kernel[rkernel - rkernel, rkernel] = 2.0  # source from the top
    kernel[rkernel + rkernel, rkernel] = 3.0  # source from the bottom
    kernel[rkernel, rkernel - rkernel] = 4.0  # source from the left
    kernel[rkernel, rkernel + rkernel] = 5.0  # source from the right
    kernel[rkernel - rkernel, rkernel - rkernel] = 6.0  # upper-left support
    kernel[rkernel - rkernel, rkernel + rkernel] = 7.0  # upper-right support
    kernel[rkernel + rkernel, rkernel - rkernel] = 8.0  # lower-left support
    kernel[rkernel + rkernel, rkernel + rkernel] = 9.0  # lower-right support

    convolved = _reference_convolution(template, kernel)
    expected = convolved[pad : pad + science_shape[0], pad : pad + science_shape[1]]

    science_padded = np.zeros(template_shape, dtype=np.float64)
    science_invalid = np.ones(template_shape, dtype=bool)
    science_padded[pad : pad + science_shape[0], pad : pad + science_shape[1]] = 0.0
    science_invalid[pad : pad + science_shape[0], pad : pad + science_shape[1]] = False
    return PairedPaddingOracle(
        science_shape=science_shape,
        pad=pad,
        template_support=template,
        science_padded=science_padded,
        science_invalid=science_invalid,
        expected_trimmed=expected,
    )


def test_oracle_locks_effective_hotpants_support_and_margin():
    oracle = _oracle()
    assert oracle.pad == 8
    assert oracle.template_support.shape == (22, 24)
    assert oracle.science_padded.shape == oracle.template_support.shape


def test_oracle_requires_neutral_invalid_science_padding_and_matched_shape():
    oracle = _oracle()
    p = oracle.pad
    valid = np.s_[p : p + oracle.science_shape[0], p : p + oracle.science_shape[1]]
    assert np.all(oracle.science_padded[~oracle.science_invalid] == 0.0)
    assert np.all(oracle.science_invalid[:p, :])
    assert np.all(oracle.science_invalid[:, :p])
    assert np.all(oracle.science_invalid[p + oracle.science_shape[0] :, :])
    assert np.all(oracle.science_invalid[:, p + oracle.science_shape[1] :])
    assert not np.any(oracle.science_invalid[valid])


def test_oracle_trimmed_result_preserves_all_four_edges_and_corners():
    oracle = _oracle()
    # These are explicit nonzero expected edge/corner responses, rather than
    # an assertion about an implementation's crop helper.
    expected = oracle.expected_trimmed
    assert expected.shape == oracle.science_shape
    assert np.all(expected[np.ix_([0, -1], [0, -1])] > 0.0)
    assert np.all(expected[[0, -1], [oracle.science_shape[1] // 2]] > 0.0)
    assert np.all(expected[[oracle.science_shape[0] // 2], [0, -1]] > 0.0)
    np.testing.assert_array_equal(
        expected,
        np.array(
            [
                [148.0, 0.0, 0.0, 15.0, 28.0, 0.0, 0.0, 87.0],
                [81.0, 0.0, 0.0, 72.0, 69.0, 0.0, 0.0, 24.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [56.0, 0.0, 0.0, 25.0, 16.0, 0.0, 0.0, 75.0],
                [16.0, 0.0, 0.0, 21.0, 10.0, 0.0, 0.0, 6.0],
                [305.0, 0.0, 0.0, 40.0, 79.0, 0.0, 0.0, 161.0],
            ]
        ),
    )
