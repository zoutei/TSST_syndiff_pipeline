"""H.2: precomputed-basis convolution must reproduce pyhotpants's own
block-based spatial convolution (``jit_spatial_convolve``) exactly, and
patch-level scatter-add must reproduce whole-image convolution exactly
(the "expand by half-kernel, don't crop before summing" invariant) --
see spicy-squishing-ritchie.md Part H."""

from __future__ import annotations

import numpy as np
import pytest

from hotpants.pure.convolution import jit_spatial_convolve
from hotpants.pure.kernel import calculate_kernel_basis

from syndiff_pipeline.difference_imaging.stages.convolved_templates_patch_cache import (
    dilate_footprint_for_patch_convolution,
    kernel_coeffs_at_block_center,
    precompute_basis_valid_maps,
    recombine_basis_maps_full,
    scatter_add_patch_valid_maps,
)


def _random_basis(seed: int = 0):
    rng = np.random.default_rng(seed)
    sigma_gauss = [0.6, 1.4]
    deg_fixe = [2, 1]
    ker_order = 1
    basis = calculate_kernel_basis((9, 9), sigma_gauss, deg_fixe)
    basis_arr = np.asarray(basis, dtype=np.float64)
    n_comp_ker = basis_arr.shape[0]
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_needed = 1 + 1 + (n_comp_ker - 1) * n_spatial
    kernel_sol = rng.normal(size=n_needed)
    return basis_arr, kernel_sol, n_comp_ker, ker_order, rng


def test_kernel_coeffs_reconstructs_jit_make_kernel_local_kernel():
    """kernel_coeffs_at_block_center, dotted with kernel_vecs, must equal
    jit_make_kernel's own local_kernel output exactly."""
    from hotpants.pure.convolution import jit_make_kernel

    basis_arr, kernel_sol, n_comp_ker, ker_order, rng = _random_basis(seed=1)
    hw_kernel = basis_arr.shape[1] // 2
    r_pix_x, r_pix_y = 40.0, 40.0
    cx, cy = 17.3, 22.1

    local_kernel_real = jit_make_kernel(
        kernel_sol, 0, hw_kernel, r_pix_x, r_pix_y, n_comp_ker, ker_order,
        basis_arr, cx, cy,
    )
    coeffs = kernel_coeffs_at_block_center(
        kernel_sol,
        r_pix_x=r_pix_x, r_pix_y=r_pix_y,
        n_comp_ker=n_comp_ker, ker_order=ker_order,
        block_center_x=cx, block_center_y=cy,
    )
    local_kernel_mine = np.tensordot(coeffs, basis_arr, axes=(0, 0))
    np.testing.assert_allclose(local_kernel_mine, local_kernel_real, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kc_step", [3, 7, 100])
def test_recombine_matches_jit_spatial_convolve(kc_step):
    """Whole-image (no patch decomposition) recombination must match
    jit_spatial_convolve's real output to float32-vs-float64 precision --
    today's production convolve_template_with_kernel_solution runs
    jit_spatial_convolve with the image cast to float32 internally, while
    this module runs entirely in float64, so exact bit agreement is not
    expected (this module is more precise, not less) -- only agreement
    to float32 machine precision (~1e-6 relative) is."""
    basis_arr, kernel_sol, n_comp_ker, ker_order, rng = _random_basis(seed=2)
    hw_kernel = basis_arr.shape[1] // 2
    ny, nx = 40, 45
    image = rng.normal(size=(ny, nx)).astype(np.float64)

    variance = np.zeros_like(image, dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=np.int32)
    conv_real, _var, _mask = jit_spatial_convolve(
        image.astype(np.float32), kernel_sol, variance, mask,
        kc_step, hw_kernel, n_comp_ker, ker_order, basis_arr,
        False, 0.0, 1,
    )

    basis_maps = precompute_basis_valid_maps(image, basis_arr)
    conv_mine = recombine_basis_maps_full(
        basis_maps, kernel_sol,
        ny_hr=ny, nx_hr=nx, hw_kernel=hw_kernel, kc_step=kc_step,
        n_comp_ker=n_comp_ker, ker_order=ker_order, oversample=1,
    )

    conv_real64 = conv_real.astype(np.float64)
    # Border (never written by jit_spatial_convolve) must be exactly zero
    # on both sides.
    np.testing.assert_array_equal(conv_mine[:hw_kernel, :], 0.0)
    np.testing.assert_array_equal(conv_mine[-hw_kernel:, :], 0.0)
    np.testing.assert_array_equal(conv_real64[:hw_kernel, :], 0.0)
    np.testing.assert_array_equal(conv_real64[-hw_kernel:, :], 0.0)

    interior_mine = conv_mine[hw_kernel:-hw_kernel, hw_kernel:-hw_kernel]
    interior_real = conv_real64[hw_kernel:-hw_kernel, hw_kernel:-hw_kernel]
    scale = np.abs(interior_real).max()
    np.testing.assert_allclose(interior_mine, interior_real, rtol=0, atol=max(1e-4, scale * 2e-5))


def test_patch_scatter_add_matches_whole_image_convolution():
    """sum_i conv(patch_i, basis_k) must equal conv(sum_i patch_i, basis_k)
    when each patch's valid-mode convolution is computed over its own
    footprint dilated by the kernel half-width and NOT cropped back before
    summing -- the precise form of "expand the edges, add the flux from
    these"."""
    basis_arr, kernel_sol, n_comp_ker, ker_order, rng = _random_basis(seed=3)
    kh = basis_arr.shape[1]
    hw_kernel = kh // 2
    ny, nx = 30, 34

    # Two overlapping-support synthetic "skycell" patches with sharp edges,
    # each nonzero only within its own footprint.
    full = np.zeros((ny, nx), dtype=np.float64)
    patch_a = np.zeros((ny, nx), dtype=np.float64)
    patch_a[2:15, 3:20] = rng.normal(size=(13, 17))
    patch_b = np.zeros((ny, nx), dtype=np.float64)
    patch_b[10:28, 12:30] = rng.normal(size=(18, 18))
    full = patch_a + patch_b

    ny_valid, nx_valid = ny - 2 * hw_kernel, nx - 2 * hw_kernel

    def _patch_valid_maps(patch: np.ndarray, footprint: tuple[slice, slice]):
        ys, xs = footprint
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        py0, py1, px0, px1 = dilate_footprint_for_patch_convolution(
            y0, y1, x0, x1, hw_kernel=hw_kernel, array_shape=(ny, nx)
        )
        sub = patch[py0:py1, px0:px1]
        maps = precompute_basis_valid_maps(sub, basis_arr)
        return maps, (py0, px0)

    patch_maps = [
        _patch_valid_maps(patch_a, (slice(2, 15), slice(3, 20))),
        _patch_valid_maps(patch_b, (slice(10, 28), slice(12, 30))),
    ]
    summed_valid = scatter_add_patch_valid_maps(
        patch_maps, n_basis=basis_arr.shape[0], ny_valid=ny_valid, nx_valid=nx_valid
    )

    whole_valid = precompute_basis_valid_maps(full, basis_arr)

    np.testing.assert_allclose(summed_valid, whole_valid, rtol=0, atol=1e-9)

    # And, composed all the way through recombination, must match
    # jit_spatial_convolve on the whole (non-decomposed) image too.
    conv_from_patches = recombine_basis_maps_full(
        summed_valid, kernel_sol,
        ny_hr=ny, nx_hr=nx, hw_kernel=hw_kernel, kc_step=5,
        n_comp_ker=n_comp_ker, ker_order=ker_order, oversample=1,
    )
    variance = np.zeros((ny, nx), dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=np.int32)
    conv_real, _var, _mask = jit_spatial_convolve(
        full.astype(np.float32), kernel_sol, variance, mask,
        5, hw_kernel, n_comp_ker, ker_order, basis_arr,
        False, 0.0, 1,
    )
    interior_mine = conv_from_patches[hw_kernel:-hw_kernel, hw_kernel:-hw_kernel]
    interior_real = conv_real.astype(np.float64)[hw_kernel:-hw_kernel, hw_kernel:-hw_kernel]
    scale = np.abs(interior_real).max()
    np.testing.assert_allclose(interior_mine, interior_real, rtol=0, atol=max(1e-4, scale * 2e-5))
