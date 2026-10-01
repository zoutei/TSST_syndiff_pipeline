# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Permanent equivalence gates for the gather-free packed renderer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import epsf_model as EM


def _render_pair(local, pix_x, pix_y, off_x, off_y):
    cx, cy = EM.core_centroid_xy(local)
    got = EM.render_packed_pixels_banded(
        local, pix_x, pix_y, off_x, off_y, cx, cy,
    )
    got = got / (EM.renorm_scalar(local, cx, cy)[:, None] + 1e-12)

    shifted = EM.recenter_grid_core(local, clip_nonneg=False, n_iter=1)
    ref = EM.render_physical_pixels_blocksum(
        shifted,
        pix_x - off_x[:, None],
        pix_y - off_y[:, None],
    )
    return got, ref


@pytest.mark.parametrize("stamp_physical", [13, 15])
def test_packed_banded_matches_gather_reference_value_and_gradient(stamp_physical):
    _, g_size, _ = EM.node_geometry(stamp_physical)
    rng = np.random.default_rng(183 + stamp_physical)
    local = jnp.asarray(rng.uniform(0.01, 1.0, size=(2, g_size, g_size)), dtype=jnp.float32)
    # Non-contiguous and edge-reaching integer-lattice supports.
    half = stamp_physical // 2
    px = jnp.asarray([
        [-half, -3, 0, 2, half],
        [-half, -1, 1, 4, half],
    ], dtype=jnp.float32)
    py = jnp.asarray([
        [half, -2, 0, 3, -half],
        [-half, 2, -3, 1, half],
    ], dtype=jnp.float32)
    ox = jnp.asarray([0.37, -0.49], dtype=jnp.float32)
    oy = jnp.asarray([-0.42, 0.31], dtype=jnp.float32)

    got, ref = _render_pair(local, px, py, ox, oy)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=3e-5, atol=3e-5)

    def band_loss(grid, xoff, yoff):
        return jnp.sum(_render_pair(grid, px, py, xoff, yoff)[0] ** 2)

    def ref_loss(grid, xoff, yoff):
        return jnp.sum(_render_pair(grid, px, py, xoff, yoff)[1] ** 2)

    grads_got = jax.grad(band_loss, argnums=(0, 1, 2))(local, ox, oy)
    grads_ref = jax.grad(ref_loss, argnums=(0, 1, 2))(local, ox, oy)
    for actual, expected in zip(grads_got, grads_ref):
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=2e-4, atol=2e-4,
        )
