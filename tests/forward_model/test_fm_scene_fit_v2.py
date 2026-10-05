# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""scene_fit driver for colour model v2: colour file, <delta^2>, g8 warm start, rejection gate."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import scene_fit as SF

C, A, N = SF.ROLE_CONTRIB, SF.ROLE_ANCHOR, SF.ROLE_NUISANCE


# ---------------------------------------------------------------- g8 warm start

def test_carry_g8_keeps_and_zero_pads():
    old = np.arange(1, 10, dtype=float)                       # 8 base + dil_r
    got = SF.carry_g8(old, ("dil_r",), ("dil_r", "sq0", "sq1"))
    np.testing.assert_array_equal(got, np.r_[old, 0.0, 0.0])
    np.testing.assert_array_equal(SF.carry_g8(old, ("dil_r",), ("dil_r",)), old)


def test_carry_g8_raises_on_non_prefix_and_bad_length():
    old = np.ones(9)
    with pytest.raises(ValueError, match="must start with"):
        SF.carry_g8(old, ("dil_r",), ("sq0", "sq1"))
    with pytest.raises(ValueError, match="must start with"):
        SF.carry_g8(old, ("dil_r",), ())
    with pytest.raises(ValueError, match="need 8"):
        SF.carry_g8(np.ones(10), (), ("sq0", "sq1"))


def _params(g8=None):
    p = {"epsf_base_raw": jnp.zeros((3, 3, 4, 4)), "wcs_coeff": jnp.ones(2)}
    if g8 is not None:
        p["chroma_g8"] = jnp.asarray(g8, jnp.float32)
    return p


def test_set_chroma_model_keeps_incoming_g8():
    old = np.linspace(0.1, 0.9, 9)
    p = SF.set_chroma_model(_params(old), "global8", halo=False, g8_extras=("dil_r", "sq0", "sq1"),
                            g8_source_extras=("dil_r",))
    np.testing.assert_allclose(np.asarray(p["chroma_g8"]), np.r_[old, 0, 0], rtol=1e-6)
    assert p["wcs_coeff"] is not None


def test_set_chroma_model_init_overrides_and_fresh_is_zero():
    init = list(range(11))
    p = SF.set_chroma_model(_params(np.ones(9)), "global8", halo=False, g8_init=init,
                            g8_extras=("dil_r", "sq0", "sq1"))
    np.testing.assert_array_equal(np.asarray(p["chroma_g8"]), init)
    p = SF.set_chroma_model(_params(), "global8", halo=False, g8_extras=("dil_r",))
    np.testing.assert_array_equal(np.asarray(p["chroma_g8"]), np.zeros(9))
    with pytest.raises(ValueError, match="g8_source_extras"):
        SF.set_chroma_model(_params(np.ones(9)), "global8", halo=False, g8_extras=("dil_r",))


def _wargs(**kw):
    base = dict(init_params_file=None, init_from=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_warm_start_source_meta_and_defaults(tmp_path):
    pf = tmp_path / "params_stage3.npz"
    src = SF.warm_start_source(_wargs(init_params_file=str(pf)))           # no meta -> #14
    assert (src["gauge"], src["extras"], src["fit_meta"]) == ("raw", ("dil_r",), None)
    (tmp_path / "fit_meta.json").write_text(json.dumps({"chroma_model": "global8"}))
    src = SF.warm_start_source(_wargs(init_from=str(tmp_path)))            # pre-decision meta
    assert (src["gauge"], src["extras"]) == ("mean", ())
    assert src["params"] == str(tmp_path / "params.npz")
    (tmp_path / "fit_meta.json").write_text(json.dumps(
        {"chroma_g8_gauge": "raw", "chroma_g8_extras": "dil_r,sq0", "chroma_g8_blur_order": 1}))
    src = SF.warm_start_source(_wargs(init_params_file=str(pf), init_from="/nonexistent"))
    assert src["extras"] == ("dil_r", "sq0", "blur_r")
    assert SF.warm_start_source(_wargs()) is None


def test_resume_restores_colour_file(tmp_path):
    (tmp_path / "fit_meta.json").write_text(json.dumps(
        {"chroma_g8_gauge": "raw", "chroma_g8_extras": "dil_r", "colour_file": "/x/u.csv"}))
    a = SimpleNamespace(resume=True, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None, colour_file=None)
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.colour_file == "/x/u.csv"
    a = SimpleNamespace(resume=True, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None, colour_file="")
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.colour_file == ""                                              # explicit BP-RP wins
    a = SimpleNamespace(resume=False, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None, colour_file=None)
    SF.resolve_g8_defaults(a, tmp_path)
    assert a.colour_file is None


# ---------------------------------------------------------------- colour file

BIG = 6917528997577384320                   # > 2**53: must not go through float


def _csv(path, rows):
    path.write_text("source_id,colour\n" + "".join(f"{s},{c}\n" for s, c in rows))
    return path


def test_colour_from_file_match_nan_unmatched(tmp_path):
    f = _csv(tmp_path / "c.csv", [(BIG, 1.5), (BIG + 1, "nan"), (7, ""), (99, 3.0)])
    sid = np.array([BIG, BIG + 1, 7, 8, 9], dtype=np.int64)
    fb = np.array([0.1, 0.2, 0.3, 0.4, np.nan])
    c, counts = SF.colour_from_file(f, sid, fb)
    np.testing.assert_array_equal(c, [1.5, 0.2, 0.3, 0.4, np.nan])
    assert counts == {"n_file_rows": 4, "matched": 1, "fallback": 3, "no_colour": 1}


def _fake_scene(role, colour_scene, sbi, n_bundle, flux=None, noise=None):
    s = object.__new__(SF.Scene)
    role = np.asarray(role, np.int64)
    n = role.size
    s.role, s.N = role, n
    s.colour = np.full(n_bundle, -9.0)
    s.colour[sbi] = colour_scene
    s.colour_source = "bp_rp"
    s.z = {"star_bundle_index": np.asarray(sbi), "source_id": np.arange(100, 100 + n, dtype=np.int64),
           "tess_flux": np.ones(n) if flux is None else np.asarray(flux, float),
           "core": np.array([True, False]),
           "noise": np.ones((n, 2)) if noise is None else np.asarray(noise, float)}
    return s


def test_use_colour_file_writes_the_scene_stars_bundle_slots(tmp_path):
    sbi = np.array([5, 0, 3])                                             # scene star -> bundle row
    s = _fake_scene([C, A, N], [0.5, 1.0, 1.5], sbi, n_bundle=7)
    f = _csv(tmp_path / "c.csv", [(101, 2.0), (102, "nan")])            # star 1 matched, 2 NaN
    counts = s.use_colour_file(f)
    assert counts["matched"] == 1 and counts["fallback"] == 2
    np.testing.assert_array_equal(s.colour, [2.0, -9, -9, 1.5, -9, 0.5, -9])
    assert s.colour_source == str(f)


def test_delta2_mean_global_matches_colour_ref_population():
    c = np.array([0.0, 1.0, 2.0, np.nan, 10.0])
    s = _fake_scene([C, C, A, A, N], c, np.arange(5), n_bundle=5)
    cref = s.colour_ref()
    assert cref == pytest.approx(1.0)                                      # nuisance + NaN excluded
    assert s.delta2_mean(cref) == pytest.approx(2.0 / 3.0)


def test_delta2_mean_per_population_is_weighted_per_role():
    c = np.array([0.0, 2.0, 1.0, 3.0, 10.0])
    flux = np.array([1.0, 3.0, 1.0, 1.0, 1.0])                            # weight ~ flux^2
    s = _fake_scene([C, C, A, A, N], c, np.arange(5), n_bundle=5, flux=flux)
    cref = s.colour_ref_per_population()
    w = np.array([1.0, 9.0])
    mc = np.sum(w * c[:2]) / 10.0
    assert cref[C] == pytest.approx(mc) and cref[A] == pytest.approx(2.0)
    d2 = s.delta2_mean(cref)
    assert d2[C] == pytest.approx(np.sum(w * (c[:2] - mc) ** 2) / 10.0)
    assert d2[A] == pytest.approx(1.0)
    assert d2[N] == d2[C]


# ---------------------------------------------------------------- rejection gate

@pytest.mark.parametrize("stage,step,every,conv,last,want", [
    (2, 50, 0, True, None, False),        # --reject-every 0: never, even when converged
    (3, 60, 0, True, 5, False),
    (2, 60, 10, False, None, True),       # periodic
    (2, 61, 10, False, None, False),
    (2, 40, 10, False, None, False),      # burn-in
    (3, 61, 10, True, None, True),        # convergence-forced (rejection on)
    (3, 61, 10, True, 0, False),          # ... unless the last refresh changed nothing
    (1, 60, 10, True, 5, False),          # never in stage 1
])
def test_refresh_due(stage, step, every, conv, last, want):
    assert SF.refresh_due(stage, step, reject_every=every, burn_in=50, converged=conv,
                          last_refresh_changed=last) is want


# ---------------------------------------------------------------- A3 default (2026-09-29)
A3 = "dil_r,sq0,sq1,q1_0,q1_x,q1_y,q2_0,q2_x,q2_y"


def test_default_is_a3_and_resume_keeps_old_model(tmp_path):
    a = SimpleNamespace(resume=False, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None, colour_file=None)
    SF.resolve_g8_defaults(a, tmp_path)
    assert (a.chroma_g8_gauge, a.chroma_g8_extras) == ("raw", A3)
    (tmp_path / "fit_meta.json").write_text(json.dumps({"chroma_g8_gauge": "raw", "chroma_g8_extras": "dil_r"}))
    a = SimpleNamespace(resume=True, chroma_g8_gauge=None, chroma_g8_extras=None, fine_nbr_mode=None, colour_file=None)
    SF.resolve_g8_defaults(a, tmp_path)
    assert (a.chroma_g8_gauge, a.chroma_g8_extras) == ("raw", "dil_r")    # a #14 run resumes as #14


def test_warm_start_without_meta_assumes_c14_and_pads_to_a3(tmp_path):
    a = SimpleNamespace(init_params_file=str(tmp_path / "params.npz"), init_from=None)
    src = SF.warm_start_source(a)
    assert src["fit_meta"] is None and src["extras"] == ("dil_r",)
    new = SF.g8_extras_tuple(A3)
    out = SF.carry_g8(np.arange(9.0), src["extras"], new)
    assert out.shape == (17,) and np.array_equal(out[:9], np.arange(9.0)) and not out[9:].any()


def test_early_stop_patience_default_is_200():
    a = SF.build_parser().parse_args(["--scene-dir", "x", "--out-dir", "y"])
    assert a.early_stop_patience == 200
