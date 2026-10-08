"""Unit tests for the Paper-1 final-recipe chain additions (init/nbr/folds stages, fit.init, CCD colour shift)."""
import types

import jax  # noqa: F401  (before pandas/pyarrow)
import jax.numpy as jnp
import numpy as np
import pytest

from syndiff_pipeline.forward_model import fit as FM_FIT
from syndiff_pipeline.forward_model import loss as L
from syndiff_pipeline.forward_model import scene_fit as SF
from syndiff_pipeline.forward_model.chain import cli as CLI
from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain import crossfit as X
from syndiff_pipeline.forward_model.chain import fit as F
from syndiff_pipeline.forward_model.chain import neighbours as NB
from syndiff_pipeline.forward_model.recipe import recipe_argv

from chain_fixtures import raw_config, write_config


def _cfg(tmp_path, **over):
    return C.config_from_dict(raw_config(tmp_path, **over))


def _nbr(tmp_path, **kw):
    return {"ledger": str(tmp_path / "cand.csv"), **kw}


# ------------------------------------------------------------------ 1. config
def test_config_defaults_without_new_sections(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.neighbours is None
    x = cfg.crossfit
    assert (x.n_folds, x.seed, x.tile, x.pattern, x.reference_folds) == (5, 20260929, 128, "diagonal", None)
    assert cfg.fit.init == "static"


def test_config_parses_new_sections(tmp_path):
    cfg = _cfg(tmp_path, neighbours=_nbr(tmp_path, tmax=16.5, gate_override=True), fit={"init": "photutils"},
               crossfit={"n_folds": 4, "seed": 7, "tile": 64, "pattern": "group",
                         "reference_folds": str(tmp_path / "folds.npz")})
    assert cfg.neighbours == C.NeighboursCfg(ledger=tmp_path / "cand.csv", tmax=16.5, gate_override=True)
    assert cfg.fit.init == "photutils"
    assert cfg.crossfit == C.CrossfitCfg(4, 7, 64, "group", tmp_path / "folds.npz")
    assert _cfg(tmp_path, neighbours=_nbr(tmp_path)).neighbours.tmax == 17.0


@pytest.mark.parametrize("mut,msg", [
    (lambda t: {"neighbours": _nbr(t, bogus=1)}, "unknown key"),
    (lambda t: {"neighbours": _nbr(t, tmax=0)}, "tmax"),
    (lambda t: {"neighbours": _nbr(t, tmax=35)}, "tmax"),
    (lambda t: {"neighbours": {"ledger": "rel/cand.csv"}}, "absolute"),
    (lambda t: {"fit": {"init": "foo"}}, "fit.init"),
    (lambda t: {"crossfit": {"pattern": "foo"}}, "crossfit.pattern"),
])
def test_config_bad_values(tmp_path, mut, msg):
    with pytest.raises(C.ConfigError, match=msg):
        _cfg(tmp_path, **mut(tmp_path))


def test_config_hash_changes(tmp_path):
    h0 = _cfg(tmp_path).config_hash()
    assert _cfg(tmp_path, neighbours=_nbr(tmp_path)).config_hash() != h0
    assert _cfg(tmp_path, crossfit={"n_folds": 4}).config_hash() != h0
    assert _cfg(tmp_path, fit={"init": "photutils"}).config_hash() != h0


# ------------------------------------------------------------------ 2. fit_dirs / train_scene
def test_fit_dirs_scene_choice(tmp_path):
    plain, nb = _cfg(tmp_path), _cfg(tmp_path, neighbours=_nbr(tmp_path))
    assert F.fit_dirs(plain, "boot") == (plain.stage_dir("scene_boot"), plain.stage_dir("fit"))
    assert F.fit_dirs(plain, "refit")[0] == plain.stage_dir("scene_final")
    assert F.fit_dirs(nb, "boot") == (nb.stage_dir("nbr_boot"), nb.stage_dir("fit"))
    assert F.fit_dirs(nb, "refit")[0] == nb.stage_dir("nbr_final")
    assert F.fit_dirs(nb, "refit", warm=True)[1] == nb.stage_dir("refit") / "warm"
    assert plain.stage_dir("scene_boot").name == "scene_boot" and nb.stage_dir("nbr_final").name == "nbr_final"
    assert X.train_scene(plain, "final").name == "scene_final" and X.train_scene(nb, "boot").name == "nbr_boot"
    with pytest.raises(ValueError):
        X.train_scene(plain, "x")


# ------------------------------------------------------------------ 3. fit_command init
def test_fit_command_init_selection(tmp_path):
    ip = tmp_path / "p.npz"
    inputs = {"colour_file": str(tmp_path / "c.csv"), "init_params": str(ip)}
    st = _cfg(tmp_path, inputs=inputs)
    ph = _cfg(tmp_path, inputs=inputs, fit={"init": "photutils"})

    def init_of(cfg, **kw):
        cmd = F.fit_command(cfg, tmp_path / "s", tmp_path / "o", **kw)
        return cmd[cmd.index("--init-params-file") + 1]

    assert init_of(st) == str(ip) and init_of(st, which="final") == str(ip)
    assert init_of(ph) == str(ph.out_root / "init_boot/params_init.npz")
    assert init_of(ph, which="boot") == str(ph.out_root / "init_boot/params_init.npz")
    assert init_of(ph, which="final") == str(ph.out_root / "init_final/params_init.npz")
    over = tmp_path / "override.npz"
    assert init_of(ph, init_params=over, which="final") == str(over)
    assert init_of(st, init_params=over) == str(over)
    warm = F.fit_command(ph, tmp_path / "s", tmp_path / "o", warm=True)
    boot = ph.stage_dir("fit")
    assert warm[warm.index("--init-params-file") + 1] == str(boot / "params.npz")
    assert warm[warm.index("--init-state-file") + 1] == str(boot / "state_stage3.npz")
    assert warm[warm.index("--steps-per-stage") + 1] == "0,0,2000"


# ------------------------------------------------------------------ 4. prior flags
def test_check_prior_flags(tmp_path):
    cfg = _cfg(tmp_path, fit={"recipe": "paper1_final"})
    assert F.check_prior_flags(cfg) == {"lambda-fine-nbr": 0.0, "lambda-local-poly": 3.0e8, "local-poly-window": 7.0}
    bad = _cfg(tmp_path, fit={"recipe": "paper1_final", "extra_flags": ["--lambda-local-poly", "1e6"]})
    with pytest.raises(ValueError, match="overrides"):
        F.check_prior_flags(bad)
    bad2 = _cfg(tmp_path, fit={"recipe": "paper1_final", "extra_flags": ["--lambda-local-poly=1e6"]})
    with pytest.raises(ValueError, match="overrides"):
        F.check_prior_flags(bad2)


# ------------------------------------------------------------------ 5. recipe
def test_paper1_final_recipe_parses():
    a = SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", "o", *recipe_argv("paper1_final")])
    want = dict(chroma_model="global8", colour_gauge="global", chroma_axis="tesspoint", huber_delta=1e6,
                reject_every=0, lambda_fine_nbr=0, lambda_local_poly=3e8, local_poly_window=7,
                lr_per_stage="1e-3,3e-4,1e-4", epsf_lr_scale=2.5, chroma_lr_scale=10, stop_rule="smoothed",
                steps_per_stage="3000,20000,20000", star_weight_cap_tmag=10.5, anchors_train_epsf=True,
                chroma_ccd_shift="frozen0")
    for k, v in want.items():
        assert getattr(a, k) == v, k
    assert SF.build_parser().parse_args(["--scene-dir", "s", "--out-dir", "o"]).chroma_ccd_shift == "off"


# ------------------------------------------------------------------ 6. CCD shift leaf
@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_ccd_shift_leaf_always_frozen(stage):
    assert FM_FIT._leaf_labels(stage, freeze_wcs=False)["chroma_ccd_shift"] == "frozen"


def test_ccd_shift_leaf_last():
    assert FM_FIT.ALL_OPTIONAL_LEAVES[-1] == "chroma_ccd_shift"
    assert FM_FIT.CHROMA_CCD_SHIFT_LEAVES == ("chroma_ccd_shift",)


def _ctx(delta):
    return types.SimpleNamespace(chroma_delta=jnp.asarray(delta, jnp.float32))


def test_chroma_slot_terms_identity_without_leaf(monkeypatch):
    base = (jnp.arange(3.0), jnp.ones(3), 2 * jnp.ones(3), {})
    monkeypatch.setattr(L, "_colour_slot_terms", lambda p, c, o: base)
    ctx, occ = _ctx([0.1, 0.2, 0.3, 0.4]), jnp.array([0, 1, 3])
    got = L.chroma_slot_terms({}, ctx, occ)
    want = L.BW.merge_slot_terms({}, ctx, occ, base)
    assert got is want or all(np.array_equal(g, w) for g, w in zip(got[:3], want[:3]))
    monkeypatch.setattr(L, "_colour_slot_terms", lambda p, c, o: None)
    assert L.chroma_slot_terms({}, ctx, occ) is None


def test_chroma_slot_terms_with_leaf(monkeypatch):
    ctx, occ = _ctx([0.1, 0.2, 0.3, 0.4]), jnp.array([0, 1, 3])
    delta = np.array([0.1, 0.2, 0.4], np.float32)
    params = {"chroma_ccd_shift": jnp.array([0.5, -2.0], jnp.float32)}
    # colour model off
    monkeypatch.setattr(L, "_colour_slot_terms", lambda p, c, o: None)
    m, sx, sy, f = L.chroma_slot_terms(params, ctx, occ)
    np.testing.assert_allclose(m, delta); np.testing.assert_allclose(sx, delta * 0.5)
    np.testing.assert_allclose(sy, delta * -2.0); assert f == {}
    # colour model on: shifts are added
    base = (jnp.asarray(delta), jnp.ones(3), 2 * jnp.ones(3), {"dilation": 1})
    monkeypatch.setattr(L, "_colour_slot_terms", lambda p, c, o: base)
    m, sx, sy, f = L.chroma_slot_terms(params, ctx, occ)
    np.testing.assert_allclose(sx, 1 + delta * 0.5); np.testing.assert_allclose(sy, 2 + delta * -2.0)
    assert f == {"dilation": 1}
    # zero leaf = no change
    z = L.chroma_slot_terms({"chroma_ccd_shift": jnp.zeros(2)}, ctx, occ)
    np.testing.assert_allclose(z[1], 1.0); np.testing.assert_allclose(z[2], 2.0)
    with pytest.raises(ValueError, match="shape"):
        L.chroma_slot_terms({"chroma_ccd_shift": jnp.zeros(3)}, ctx, occ)
    with pytest.raises(ValueError, match="colour offsets"):
        L.chroma_slot_terms(params, types.SimpleNamespace(chroma_delta=None), occ)


# ------------------------------------------------------------------ 7. pad_table
def test_pad_table():
    t = {"fold": np.array([0, 1, 2]), "held": np.array([True, False, True]), "w": np.array([1.0, 2.0, 3.0]),
         "grid": np.arange(6).reshape(3, 2), "tiles": np.arange(5), "source_id": np.array([1, 2, 3])}
    sid = np.array([1, 2, 3, 10, 11])
    out = NB.pad_table(t, 3, 2, sid)
    assert out["fold"].tolist() == [0, 1, 2, -1, -1]
    assert out["held"].tolist() == [True, False, True, False, False] and out["held"].dtype == bool
    assert np.isnan(out["w"][3:]).all() and out["w"][:3].tolist() == [1, 2, 3]
    assert out["grid"].shape == (5, 2) and (out["grid"][3:] == -1).all()
    assert out["tiles"] is t["tiles"] or np.array_equal(out["tiles"], np.arange(5))
    assert out["source_id"].tolist() == sid.tolist()


# ------------------------------------------------------------------ 8. axis cells / cell width
@pytest.mark.parametrize("ccd,want", [(1, ("2,2", "0,0")), (3, ("2,2", "0,0")), (2, ("2,0", "0,2")), (4, ("2,0", "0,2"))])
def test_axis_cells(tmp_path, ccd, want):
    cfg = _cfg(tmp_path, scc={"sector": 24, "camera": 2, "ccd": ccd}, stem=f"tess2020120182919-s0024-2-{ccd}")
    assert X.axis_cells(cfg) == want


def test_cell_width():
    def e(n, cxx, cyy):
        return {"n": n, "cxx": cxx, "cyy": cyy}
    none = {"n": 99, "cxx": None, "cyy": None}
    cell = {f"{m},{c}": e(1000, 50.0, 50.0) for m in range(5) for c in range(4)}   # outside the used rows/cols
    cell.update({"1,0": e(10, 2.0, 4.0), "1,1": e(30, 1.0, 1.0), "2,0": none, "2,1": e(20, 0.0, 2.0)})
    w, n = X.cell_width({"cells": {"2,2": cell}}, "2,2")
    assert n == 60 and w == pytest.approx((10 * 3.0 + 30 * 1.0 + 20 * 1.0) / 60)
    allnone = {f"{m},{c}": none for m in range(5) for c in range(4)}
    w, n = X.cell_width({"cells": {"0,0": allnone}}, "0,0")
    assert n == 0 and np.isnan(w)


# ------------------------------------------------------------------ 9. union / topology
def _tiny_scene(centres, S=3):
    N = len(centres)
    k = np.arange(S * S)
    cx = np.array([c[0] for c in centres]); cy = np.array([c[1] for c in centres])
    xx = cx[:, None] + (k % S - S // 2); yy = cy[:, None] + (k // S - S // 2)
    ids, uid, owner = {}, np.zeros((N, S * S), np.int32), np.zeros((N, S * S), bool)
    for i in range(N):
        for j in range(S * S):
            key = (int(xx[i, j]), int(yy[i, j]))
            if key not in ids:
                ids[key] = len(ids); owner[i, j] = True
            uid[i, j] = ids[key]
    U = len(ids)
    val = {key: (0.5 * key[0] + 10.0 * key[1]) for key in ids}
    data = np.array([[val[(int(xx[i, j]), int(yy[i, j]))] for j in range(S * S)] for i in range(N)], np.float32)
    return dict(n_union=np.int64(U), stamp=np.int64(S), uid=uid, owner=owner, cx=cx, cy=cy, data=data,
                noise=np.ones((N, S * S), np.float32), valid=np.ones((N, S * S), bool),
                finite=np.ones((N, S * S), bool), role=np.array([1] * N, np.int8)), ids


def test_union_arrays_roundtrip():
    z, ids = _tiny_scene([(10, 10), (11, 10), (30, 30)])
    U = int(z["n_union"])
    assert U == 9 + 3 + 9 and U == len(ids)
    u = NB.union_arrays(z)
    assert u["data"].shape == (U + 1,) and u["noise"][U] == 1
    for key in ("data", "valid", "finite"):
        assert np.array_equal(u[key][z["uid"]], z[key])
    assert np.array_equal(u["x"][z["uid"]], z["cx"][:, None] + (np.arange(9) % 3 - 1))
    bad = dict(z); bad["data"] = z["data"].copy(); bad["data"][1, 0] += 1.0     # a duplicate copy disagrees
    with pytest.raises(AssertionError, match="inconsistent"):
        NB.union_arrays(bad)
    bad2 = dict(z); bad2["owner"] = z["owner"].copy(); bad2["owner"][1, :] = True
    with pytest.raises(AssertionError, match="exactly one owner"):
        NB.union_arrays(bad2)


def test_rebuild_topology_pairs_and_islands():
    z, _ = _tiny_scene([(10, 10), (11, 10), (30, 30)])
    t = NB.rebuild_topology(z)
    pairs = set(zip(t["pair_i"].tolist(), t["pair_j"].tolist()))
    assert pairs == {(0, 1)}
    assert t["island"][0] == t["island"][1] != t["island"][2]
    assert t["tiers"].tolist() == [1, 2] and t["tier2_star_idx"].shape == (1, 2) and t["tier1_star_idx"].shape == (1, 1)
    assert sorted(t["tier2_star_idx"][0].tolist()) == [0, 1]
    # shared pixels invalid for star 0 -> no weighted common pixel -> the geometric pair is disconnected
    z2 = dict(z); z2["valid"] = z["valid"].copy()
    shared = np.isin(z["uid"][0], z["uid"][1])
    z2["valid"][0, shared] = False
    t2 = NB.rebuild_topology(z2)
    assert len(t2["pair_i"]) == 0 and len(set(t2["island"].tolist())) == 3
    # inactive pixels also disconnect
    act = np.ones(int(z["n_union"]) + 1, bool); act[z["uid"][0][shared]] = False
    assert len(NB.rebuild_topology(z, act)["pair_i"]) == 0
    with pytest.raises(ValueError):
        NB.rebuild_topology(z, np.ones(3, bool))


# ------------------------------------------------------------------ 10. CLI
def test_cli_parses_new_stages(tmp_path):
    p = CLI.build_parser()
    a = p.parse_args(["folds_boot", "--config", "X", "--fold", "2"])
    assert (a.stage, a.fold, a.summarise) == ("folds_boot", 2, False)
    a = p.parse_args(["nbr_final", "--config", "X"])
    assert a.stage == "nbr_final" and a.fold is None
    a = p.parse_args(["folds_final", "--config", "X", "--summarise"])
    assert a.summarise is True
    for s in ("init_boot", "init_final", "nbr_boot", "folds_final"):
        assert s in CLI.ALL_STAGES and s in C.STAGES


# ---------------------------------------------------------------- verified-ledger neighbour source (assoc_r2)
def test_config_assoc_r2_source(tmp_path):
    cat = tmp_path / "gaia.csv"; cat.write_text("source_id,ra,dec,pmra,pmdec\n")
    cfg = _cfg(tmp_path, neighbours={"ledger": str(tmp_path / "n.parquet"), "tmax": 17, "source": "assoc_r2",
                                     "gaia_catalog": str(cat)})
    assert cfg.neighbours.source == "assoc_r2" and cfg.neighbours.gaia_catalog == cat and not cfg.neighbours.gate_override
    assert _cfg(tmp_path, neighbours={"ledger": str(tmp_path / "c.csv")}).neighbours.source == "candidates"
    for bad in ({"source": "assoc_r2"},                                              # no gaia_catalog
                {"source": "assoc_r2", "gaia_catalog": str(cat), "gate_override": True},
                {"source": "foo"}):
        with pytest.raises(C.ConfigError):
            _cfg(tmp_path, neighbours={"ledger": str(tmp_path / "n.parquet"), **bad})


def test_assoc_r2_table(tmp_path):
    import pandas as pd
    big = 4611686018427387905                       # > 2**53: must survive exactly (no float64 round trip)
    t = pd.DataFrame({"gaia_id": [str(big), "17", None], "ra": [10.0, 20.0, 30.0], "dec": [1.0, 2.0, 3.0],
                      "tess_mag": [12.0, 16.5, np.nan], "phot_g_mean_mag": [12.5, 17.0, np.nan],
                      "phot_bp_mean_mag": [13.0, 17.6, np.nan], "phot_rp_mean_mag": [12.0, 16.4, np.nan],
                      "canonical_entity": ["a", "b", "c"], "link_kind": ["centre_removed", "enclosed_core_trigger", "x"],
                      "is_region_trigger": [False, True, False], "in_trigger_core": [False, False, False]})
    t.to_parquet(tmp_path / "n.parquet")
    pd.DataFrame({"source_id": [big], "ra": [10.001], "dec": [1.001], "pmra": [5.0], "pmdec": [-3.0]}).to_csv(
        tmp_path / "gaia.csv", index=False)
    rows, info = NB.assoc_r2_table(tmp_path / "n.parquet", tmp_path / "gaia.csv")
    assert info == dict(n_rows=3, n_gaia_rows=2, n_without_gaia=1, n_not_in_gaia_catalogue=1,
                        gaia_catalog=str(tmp_path / "gaia.csv"))
    r = rows.set_index("source_id")
    assert big in r.index and 17 in r.index and rows.source_id.dtype == np.int64
    assert (r.loc[big, "ra"], r.loc[big, "dec"], r.loc[big, "pmra"]) == (10.001, 1.001, 5.0)   # catalogue astrometry
    assert (r.loc[17, "ra"], r.loc[17, "dec"]) == (20.0, 2.0) and np.isnan(r.loc[17, "pmra"])  # table fallback, no pm
    assert np.isclose(r.loc[big, "bp_rp"], 1.0) and (rows.ref_epoch == 2016.0).all()


def test_assoc_r2_table_catalogue_without_pm(tmp_path):
    import pandas as pd
    pd.DataFrame({"gaia_id": ["5"], "ra": [1.0], "dec": [2.0], "tess_mag": [15.0], "phot_g_mean_mag": [15.5],
                  "phot_bp_mean_mag": [16.0], "phot_rp_mean_mag": [15.0], "canonical_entity": ["a"], "link_kind": ["centre_removed"],
                  "is_region_trigger": [False], "in_trigger_core": [False]}).to_parquet(tmp_path / "n.parquet")
    pd.DataFrame({"source_id": [5], "ra": [1.0001], "dec": [2.0001]}).to_csv(tmp_path / "gaia.csv", index=False)
    rows, info = NB.assoc_r2_table(tmp_path / "n.parquet", tmp_path / "gaia.csv")
    assert info["n_not_in_gaia_catalogue"] == 0 and rows.ra.iloc[0] == 1.0001 and np.isnan(rows.pmra.iloc[0])
