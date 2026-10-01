# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""fine_neighbour_penalty couples only the pixel-scale part of adjacent ePSF nodes."""

import numpy as np
import jax.numpy as jnp

from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import loss as L

G = EM.NODE_GRID_SIZE


def _gauss(sigma, cx=0.0, cy=0.0):
    c = (G - 1) / 2.0
    y, x = np.mgrid[:G, :G].astype(np.float64)
    g = np.exp(-0.5 * (((x - c - cx) / sigma) ** 2 + ((y - c - cy) / sigma) ** 2))
    return g / g.sum()


def _pair(a, b):
    """(1, 1, 2, G, G): two horizontally adjacent nodes."""
    return jnp.asarray(np.stack([a, b])[None, None], dtype=jnp.float32)


def test_identical_nodes_cost_nothing():
    base = _gauss(5.0)
    assert float(L.fine_neighbour_penalty(_pair(base, base))) == 0.0


def test_smooth_shape_change_is_nearly_free():
    # a 10% width change and a 1-sample shift: real field variation, all broad
    a, b = _gauss(5.0), _gauss(5.5, cx=1.0)
    fine = float(L.fine_neighbour_penalty(_pair(a, b)))
    full = float(L.node_smoothness_penalty(_pair(a, b)))
    # The split is soft (Gaussian, sigma=2 samples): a 1-sample shift of a sigma=5
    # profile leaks ~3.5% into the fine part -- still ~30x less than a comb.
    assert fine < 0.05 * full, (fine, full)


def test_pixel_scale_comb_is_fully_penalised():
    a = _gauss(5.0)
    idx = np.arange(G)
    comb = np.where((idx[:, None] % 4 == 0) & (idx[None, :] % 4 == 0), 1.0, 0.0) - 1.0 / 16
    b = a * (1.0 + 0.05 * comb)
    fine = float(L.fine_neighbour_penalty(_pair(a, b)))
    full = float(L.node_smoothness_penalty(_pair(a, b)))
    assert fine > 0.8 * full, (fine, full)


def test_default_weight_is_off():
    assert L.LossWeights().lambda_fine_nbr == 0.0
