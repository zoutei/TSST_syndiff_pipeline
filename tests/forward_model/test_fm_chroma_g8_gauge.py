# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Raw-P vs mean-removed-base gauge for the global colour model (2026-09-24)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import epsf_model as EM
from syndiff_pipeline.forward_model import loss as L


def _psf(g_size: int = EM.NODE_GRID_SIZE, sigma_px: float = 0.8) -> jnp.ndarray:
    c = EM.node_center_for_grid(g_size)
    ax = (np.arange(g_size) - c) / EM.OVERSAMPLE
    x, y = np.meshgrid(ax, ax)
    r2 = x ** 2 + y ** 2
    # Gaussian core plus a Moffat-like wing so the generators are not all identical.
    p = np.exp(-r2 / (2 * sigma_px ** 2)) + 0.02 * (1 + r2) ** -2
    return jnp.asarray(p / p.sum(), jnp.float32)


PAIRS = [
    (EM.chroma_kurt_plain_field, EM.chroma_kurt_plain_field_raw),
    (EM.chroma_blur_field, EM.chroma_blur_field_raw),
    (EM.chroma_dilation_field, EM.chroma_dilation_field_raw),
    (EM.chroma_trefoil_a_field, EM.chroma_trefoil_a_field_raw),
]


@pytest.mark.parametrize("mean_fn,raw_fn", PAIRS)
def test_raw_gauge_is_orthogonal_to_raw_base(mean_fn, raw_fn):
    p = _psf()
    w = EM.canonical_mode_weight_grid(int(p.shape[-1]))
    q = raw_fn(p)
    overlap = float(jnp.sum(w * q * p) / jnp.sqrt(jnp.sum(w * q * q) * jnp.sum(w * p * p)))
    assert abs(overlap) < 1e-4


def test_mean_gauge_leaves_flat_sheet_raw_does_not():
    p = _psf()
    q_mean = np.asarray(EM.chroma_dilation_field(p))
    q_raw = np.asarray(EM.chroma_dilation_field_raw(p))
    corner = (slice(0, 3), slice(0, 3))
    # A pure PSF is ~0 in the grid corner, so anything there is the gauge's sheet.
    assert abs(q_mean[corner].mean()) > 1e3 * abs(q_raw[corner].mean())
    # Raw and mean gauges differ by (a constant) + (a multiple of P) only.
    d = q_mean - q_raw
    coef = np.polyfit(np.asarray(p).ravel(), d.ravel(), 1)
    assert np.allclose(d, np.polyval(coef, np.asarray(p)), atol=1e-6 * np.abs(q_mean).max())


def test_generator_registry_has_raw_twins():
    for name in ("kurt_plain", "blur", "dilation", "tre_a", "tre_b"):
        assert name + "_raw" in L.CHROMA_FIELD_GENERATORS


def _fake_ctx(n: int = 5):
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        chroma_delta=jnp.asarray(rng.normal(size=n), jnp.float32),
        chroma_axis=(-55.66, 2098.93),
        x_lin=jnp.asarray(rng.uniform(0, 2048, n), jnp.float32),
        y_lin=jnp.asarray(rng.uniform(0, 2048, n), jnp.float32),
        chroma_g8_gauge="raw", chroma_g8_no_dil=True,
    )


def test_blur_order_zero_extra_terms_match_global_blur():
    ctx = _fake_ctx(); occ = jnp.arange(5)
    c8 = jnp.asarray([0.01, -0.1, 0.03, -0.03, 0.006, -0.02, 0.0, -0.005], jnp.float32)
    c10 = jnp.concatenate([c8, jnp.zeros(2, jnp.float32)])
    f8 = L._chroma_g8_slot_terms({"chroma_g8": c8}, ctx, occ)[3]
    f10 = L._chroma_g8_slot_terms({"chroma_g8": c10}, ctx, occ)[3]
    assert set(f8) == set(f10) and "dilation_raw" not in f8
    for k in f8:
        np.testing.assert_array_equal(np.asarray(f8[k]), np.asarray(f10[k]))


def test_blur_radial_terms_follow_distance_to_axis():
    ctx = _fake_ctx(); occ = jnp.arange(5)
    c = jnp.asarray([0, 0, 0, 0, 0, 0.0, 0, 0, 0.5, 0.25], jnp.float32)  # blur = 0.5 r + 0.25 r^2
    blur = np.asarray(L._chroma_g8_slot_terms({"chroma_g8": c}, ctx, occ)[3]["blur_raw"])
    r = np.hypot(ctx.chroma_axis[0] - np.asarray(ctx.x_lin), ctx.chroma_axis[1] - np.asarray(ctx.y_lin)) / 1000.0
    np.testing.assert_allclose(blur, np.asarray(ctx.chroma_delta) * (0.5 * r + 0.25 * r ** 2), rtol=1e-5)


def test_extras_dil_r_and_axis_oriented_astig():
    ctx = _fake_ctx(); occ = jnp.arange(5)
    ctx.chroma_g8_no_dil = False
    ctx.chroma_g8_extras = ("dil_r", "astig0", "astig_r")
    c = jnp.asarray([0, 0, 0, 0, 0, 0, 0.02, 0, 0.01, 0.003, 0.002], jnp.float32)
    f = L._chroma_g8_slot_terms({"chroma_g8": c}, ctx, occ)[3]
    d = np.asarray(ctx.chroma_delta)
    vx = ctx.chroma_axis[0] - np.asarray(ctx.x_lin); vy = ctx.chroma_axis[1] - np.asarray(ctx.y_lin)
    r = np.hypot(vx, vy) / 1000.0; phi = np.arctan2(vy, vx)
    np.testing.assert_allclose(np.asarray(f["dilation_raw"]), d * (0.02 + 0.01 * r), rtol=1e-5)
    a = 0.003 + 0.002 * r
    np.testing.assert_allclose(np.asarray(f["aniso_raw"]), d * a * np.cos(2 * phi), rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(f["shear_raw"]), d * a * np.sin(2 * phi), rtol=1e-4, atol=1e-7)
    assert "aniso_raw" in L.CHROMA_FIELD_GENERATORS and "shear_raw" in L.CHROMA_FIELD_GENERATORS


def test_extras_length_mismatch_raises():
    ctx = _fake_ctx(); ctx.chroma_g8_extras = ("dil_r",)
    with pytest.raises(ValueError):
        L._chroma_g8_slot_terms({"chroma_g8": jnp.zeros(8, jnp.float32)}, ctx, jnp.arange(5))


def test_axis_oriented_astig_is_a_stretch_along_the_axis_direction():
    # For phi = 0 the combination is pure aniso (stretch along x); for phi = 45 deg pure shear.
    p = _psf()
    a0 = np.asarray(EM.chroma_aniso_field_raw(p)); s0 = np.asarray(EM.chroma_shear_field_raw(p))
    for phi, want in ((0.0, a0), (np.pi / 4, s0)):
        got = np.cos(2 * phi) * a0 + np.sin(2 * phi) * s0
        np.testing.assert_allclose(got, want, atol=1e-6 * np.abs(a0).max())


def _args(**kw):
    from types import SimpleNamespace
    base = dict(resume=False, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_scene_fit_defaults_are_the_chosen_model(tmp_path):
    from syndiff_pipeline.forward_model import scene_fit as SF
    a = _args(); SF.resolve_g8_defaults(a, tmp_path)
    # A3 (2026-09-29): #14 + quad colour shift + colour elongation planes
    assert (a.chroma_g8_gauge, a.chroma_g8_extras) == ("raw", "dil_r,sq0,sq1,q1_0,q1_x,q1_y,q2_0,q2_x,q2_y")


def test_resume_of_a_pre_decision_run_keeps_its_model(tmp_path):
    import json
    from syndiff_pipeline.forward_model import scene_fit as SF
    (tmp_path / "fit_meta.json").write_text(json.dumps({"chroma_model": "global8"}))  # no g8 keys
    a = _args(resume=True); SF.resolve_g8_defaults(a, tmp_path)
    assert (a.chroma_g8_gauge, a.chroma_g8_extras) == ("mean", "")
    (tmp_path / "fit_meta.json").write_text(json.dumps({"chroma_g8_gauge": "raw", "chroma_g8_extras": "dil_r,astig0"}))
    a = _args(resume=True); SF.resolve_g8_defaults(a, tmp_path)
    assert (a.chroma_g8_gauge, a.chroma_g8_extras) == ("raw", "dil_r,astig0")


def test_explicit_g8_flags_win(tmp_path):
    from syndiff_pipeline.forward_model import scene_fit as SF
    a = _args(chroma_g8_gauge="mean", chroma_g8_extras=""); SF.resolve_g8_defaults(a, tmp_path)
    assert (a.chroma_g8_gauge, a.chroma_g8_extras) == ("mean", "")
