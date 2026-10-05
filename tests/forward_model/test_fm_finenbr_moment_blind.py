# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Moment-blind fine-neighbour penalty, and moment_blind as the default coupling (2026-09-29)."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import scene_fit as SF

G = 63
S = 4.0  # samples per native pixel


def gauss(sx, sy, x0=0.0, y0=0.0, g=G):
    c = (np.arange(g) - (g - 1) / 2) / S
    yy, xx = np.meshgrid(c, c, indexing="ij")
    e = np.exp(-0.5 * ((xx - x0) ** 2 / sx ** 2 + (yy - y0) ** 2 / sy ** 2))
    return e / e.sum()


def field_from(fn, r=3, c=3):
    return jnp.asarray(np.stack([np.stack([fn(i, j) for j in range(c)]) for i in range(r)]), jnp.float32)


@pytest.fixture(scope="module")
def varied_field():
    """Nodes differ in width, anisotropy and centroid plus a small common pixel-scale pattern."""
    rng = np.random.default_rng(0)
    common = rng.normal(size=(G, G)) * 2e-5
    return field_from(lambda i, j: gauss(0.9 + 0.08 * i, 0.9 + 0.05 * j, 0.03 * j, -0.02 * i) + common)


def test_moment_blind_exerts_no_width_or_shift_force(varied_field):
    E = varied_field
    gen = L._low_order_generators(E)
    for mode, blind in (("plain", False), ("moment_blind", True)):
        g = jax.grad(lambda e: L.fine_nbr_penalty(e[None], mode))(E)
        for k in (1, 2, 3, 4, 5):  # dx, dy, dxx, dyy, dxy
            d = gen[:, :, k]
            force = jnp.sum(g * d, axis=(-2, -1)) / jnp.linalg.norm(d.reshape(3, 3, -1), axis=-1)
            ratio = float(jnp.max(jnp.abs(force) / jnp.linalg.norm(g.reshape(3, 3, -1), axis=-1)))
            if blind:
                assert ratio < 1e-3, (k, ratio)
            elif k in (3, 4):
                assert ratio > 0.05, (k, ratio)  # the plain penalty DOES push width


def test_moment_blind_still_penalises_pixel_noise(varied_field):
    E = varied_field
    noise = jnp.asarray(np.random.default_rng(1).normal(size=E.shape) * 1e-4, jnp.float32)
    ref = float(L.fine_neighbour_penalty(noise[None]))
    for mode in L.FINE_NBR_MODES:
        assert float(L.fine_nbr_penalty((E + noise)[None], mode)) > float(L.fine_nbr_penalty(E[None], mode)) + 0.1 * ref


def test_moment_blind_small_for_pure_low_order_field():
    E = field_from(lambda i, j: gauss(0.9 + 0.1 * i, 0.9 + 0.1 * j, 0.02 * j, 0.0))
    plain = float(L.fine_nbr_penalty(E[None], "plain"))
    mb = float(L.fine_nbr_penalty(E[None], "moment_blind"))
    assert plain > 0 and mb < 0.05 * plain, (mb, plain)


def test_moment_blind_identical_nodes_and_degenerate_grids():
    E = field_from(lambda i, j: gauss(1.0, 1.1))
    assert float(L.fine_neighbour_penalty_moment_blind(E[None])) == pytest.approx(0.0, abs=1e-18)
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda e: L.fine_neighbour_penalty_moment_blind(e[None]))(E))))
    row = field_from(lambda i, j: gauss(1.0 + 0.1 * j, 1.0), r=1, c=3)
    assert np.isfinite(float(L.fine_neighbour_penalty_moment_blind(row[None])))


def test_sum_and_mean_share_normalisation(varied_field):
    E = varied_field[None]
    n = L.fine_neighbour_count(E.shape)
    assert float(L.fine_neighbour_sum_moment_blind(E)) / n == pytest.approx(
        float(L.fine_neighbour_penalty_moment_blind(E)), rel=1e-6)
    # plain mode of the dispatcher is exactly the pre-existing penalty
    assert float(L.fine_nbr_penalty(E, "plain")) == float(L.fine_neighbour_penalty(E))


def test_prior_sum_follows_mode():
    E = field_from(lambda i, j: gauss(0.9 + 0.1 * i, 0.9 + 0.1 * j))
    params = {"epsf_base_raw": E}
    with pytest.raises(ValueError):
        L.fine_nbr_penalty(E[None], "nope")
    # decoded base is not the raw field; compare against the functions on the decoded base
    base = L.decoded_epsf_base(params)[None]
    sig = 1e-4
    assert float(L.fine_nbr_prior_sum(params, sig)) == pytest.approx(
        float(L.fine_neighbour_sum_moment_blind(base)) / (2 * sig ** 2), rel=1e-5)
    assert float(L.fine_nbr_prior_sum(params, sig, "plain")) == pytest.approx(
        float(L.fine_neighbour_sum(base)) / (2 * sig ** 2), rel=1e-5)


def test_defaults_are_moment_blind():
    assert L.FINE_NBR_MODE_DEFAULT == "moment_blind"
    assert L.LossWeights().fine_nbr_mode == "moment_blind"


def _args(**kw):
    base = dict(resume=False, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_scene_fit_new_run_defaults_to_moment_blind(tmp_path):
    a = _args(); SF.resolve_g8_defaults(a, tmp_path)
    assert a.fine_nbr_mode == "moment_blind"
    a = SF.build_parser().parse_args(["--scene-dir", "x", "--out-dir", "y"])
    assert a.fine_nbr_mode is None and a.lambda_fine_nbr == 1e8   # resolved in run()


def test_scene_fit_resume_keeps_the_mode_it_started_with(tmp_path):
    (tmp_path / "fit_meta.json").write_text(json.dumps({"lambda_fine_nbr": 1e8}))  # pre-change run
    a = _args(resume=True); SF.resolve_g8_defaults(a, tmp_path)
    assert a.fine_nbr_mode == "plain"
    (tmp_path / "fit_meta.json").write_text(json.dumps({"fine_nbr_mode": "moment_blind"}))
    a = _args(resume=True); SF.resolve_g8_defaults(a, tmp_path)
    assert a.fine_nbr_mode == "moment_blind"
    a = _args(resume=True, fine_nbr_mode="plain"); SF.resolve_g8_defaults(a, tmp_path)
    assert a.fine_nbr_mode == "plain"   # explicit flag wins


@pytest.mark.parametrize("mode,bs,tol", [("moment_blind_pair", 0.0, 0.08), ("moment_blind", 1.0, 0.15)])
def test_reduced_bases_stay_nearly_blind_and_penalise_noise(varied_field, mode, bs, tol):
    """pair-mean / smoothed-generator bases: width/shift force much smaller than the plain
    penalty's (not exactly zero -- the price for not freeing each node's own junk)."""
    E = varied_field
    gen = L._low_order_generators(E)
    fplain = jax.grad(lambda e: L.fine_nbr_penalty(e[None], "plain"))(E)
    g = jax.grad(lambda e: L.fine_nbr_penalty(e[None], mode, basis_sigma=bs))(E)
    for k in (3, 4):
        d = gen[:, :, k]
        f_red = float(jnp.max(jnp.abs(jnp.sum(g * d, axis=(-2, -1)))))
        f_pl = float(jnp.max(jnp.abs(jnp.sum(fplain * d, axis=(-2, -1)))))
        assert f_red < tol * f_pl, (mode, k, f_red / f_pl)
    noise = jnp.asarray(np.random.default_rng(2).normal(size=E.shape) * 1e-4, jnp.float32)
    assert float(L.fine_nbr_penalty((E + noise)[None], mode, basis_sigma=bs)) > \
        float(L.fine_nbr_penalty(E[None], mode, basis_sigma=bs)) + 0.1 * float(L.fine_neighbour_penalty(noise[None]))


def test_pair_basis_frees_less_than_own_basis():
    """A node-pair difference made of each node's own pixel-scale pattern hides in the
    own-node derivative basis more than in the pair-mean basis."""
    rng = np.random.default_rng(3)
    E = field_from(lambda i, j: gauss(1.0, 1.0) * (1 + 0.05 * rng.normal(size=(G, G))), r=1, c=2)
    own = float(L.fine_nbr_penalty(E[None], "moment_blind"))
    pair = float(L.fine_nbr_penalty(E[None], "moment_blind_pair"))
    assert pair > own
