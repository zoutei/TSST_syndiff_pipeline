# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for ``epsf_model.footprint_weight_grid`` (task T1-3: footprint-metric gauge).

Additive, default-off: ``decode_epsf_modes``'s existing ``weight_grid`` kwarg
already accepts any (G, G) array, so this is a new grid-builder rather than a
change to the decode/gauge machinery itself -- every existing caller (which
never passes ``weight_grid``) is untouched.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from syndiff_pipeline.forward_model import epsf_model as EM


def test_uniform_profile_is_a_disk_matching_radius():
    g = EM.NODE_GRID_SIZE
    radius = 4.243
    wg = np.asarray(EM.footprint_weight_grid(g, radius_px=radius, profile="uniform"))
    coord = np.asarray(EM.node_coord_1d(n_grid=g))
    r = np.sqrt(coord[None, :] ** 2 + coord[:, None] ** 2)
    expected = (r <= radius).astype(np.float32)
    np.testing.assert_array_equal(wg, expected)
    assert wg.sum() > 0


def test_smaller_radius_gives_a_strict_subset():
    g = EM.NODE_GRID_SIZE
    wide = np.asarray(EM.footprint_weight_grid(g, radius_px=7.071, profile="uniform")) > 0
    narrow = np.asarray(EM.footprint_weight_grid(g, radius_px=4.243, profile="uniform")) > 0
    assert narrow.sum() < wide.sum()
    assert np.all(wide[narrow])  # narrow is contained in wide


def test_shot_noise_profile_requires_base():
    g = EM.NODE_GRID_SIZE
    try:
        EM.footprint_weight_grid(g, profile="shot_noise")
    except ValueError as exc:
        assert "base" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_shot_noise_profile_downweights_bright_core():
    """A star's core (high `base` density) gets a lower shot-noise weight than
    its wings, at fixed radius -- the opposite of the uniform profile, which
    treats every sample in the disk equally."""
    g = EM.NODE_GRID_SIZE
    coord = np.asarray(EM.node_coord_1d(n_grid=g))
    r2 = coord[None, :] ** 2 + coord[:, None] ** 2
    base = np.exp(-r2 / (2 * 0.5 ** 2)).astype(np.float32)
    base = base / base.sum()
    wg = np.asarray(
        EM.footprint_weight_grid(g, radius_px=6.0, profile="shot_noise",
                                  base=jnp.asarray(base), flux=1500.0, sky=100.0)
    )
    center = int(round(EM.NODE_CENTER_INDEX))
    edge = 2
    assert wg[center, center] < wg[center, center + edge]


def test_unknown_profile_rejected():
    try:
        EM.footprint_weight_grid(EM.NODE_GRID_SIZE, profile="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_drop_in_for_decode_epsf_modes_weight_grid_kwarg():
    """The whole point: this is usable wherever decode_epsf_modes already
    accepts weight_grid, with no change to that function. Gauges 1/3 (zero
    sum, unit-norm-independent structure) still hold; only gauge 2's metric
    (and hence the exact decoded values) changes.
    """
    rng = np.random.default_rng(7)
    G = EM.NODE_GRID_SIZE
    base_raw = jnp.asarray(rng.normal(size=(2, 2, G, G)), dtype=jnp.float32)
    base = EM.decode_epsf_base(base_raw)
    raw = jnp.asarray(rng.normal(size=(1, 2, 2, G, G)), dtype=jnp.float32)

    default = EM.decode_epsf_modes(raw, base)
    footprint_grid = EM.footprint_weight_grid(G, radius_px=4.243)
    alt = EM.decode_epsf_modes(raw, base, weight_grid=footprint_grid)

    # Both remain flux-neutral (gauge 1) regardless of the gauge-2 metric used.
    assert float(np.max(np.abs(np.asarray(default).sum(axis=(-1, -2))))) < 2e-4
    assert float(np.max(np.abs(np.asarray(alt).sum(axis=(-1, -2))))) < 2e-4
    # A narrower gauge-2 metric changes the decoded mode (different projection).
    assert float(jnp.max(jnp.abs(default - alt))) > 1e-6


def test_narrow_gauge_zeroes_the_inner_product_in_its_own_metric():
    """Gauge 2's exact guarantee: after decoding with ``weight_grid=W``, the
    decoded mode is exactly orthogonal to (mean-subtracted) base under the
    inner product weighted by ``W``. Using the narrow footprint grid as ``W``
    must zero that narrow-metric inner product; it need not (and in general
    does not) zero the wide-metric one, or a raw in-aperture flux ratio --
    those are different questions (see S2's findings.md Q4).
    """
    G = EM.NODE_GRID_SIZE
    coord = np.asarray(EM.node_coord_1d(n_grid=G))
    r2 = coord[None, :] ** 2 + coord[:, None] ** 2
    base_np = np.exp(-r2 / (2 * 1.0 ** 2)).astype(np.float32)
    base_np = base_np / base_np.sum()
    base = jnp.asarray(base_np)

    from scipy.ndimage import laplace
    raw = jnp.asarray(laplace(base_np).astype(np.float32))[None]

    narrow_radius = 4.243
    narrow_w = EM.footprint_weight_grid(G, radius_px=narrow_radius)
    narrow_gauged = np.asarray(EM.decode_epsf_modes(raw, base, weight_grid=narrow_w))[0]

    base_meansub = base_np - base_np.mean()
    inner = float(np.sum(np.asarray(narrow_w) * narrow_gauged * base_meansub))
    assert abs(inner) < 1e-4 * float(np.sum(np.asarray(narrow_w) * base_meansub ** 2))
