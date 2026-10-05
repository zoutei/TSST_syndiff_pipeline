"""Per-band chain (f01-f04): store-weight-aware split, f01b grouping regression, f03 submit, f04 weight bookkeeping;
slow parity tests against the e2e F1 products."""
import json
from pathlib import Path

import numpy as np
import pytest

E2E_STALE = pytest.mark.xfail(reason=("input gone: the e2e combined/convolved cells of projection 2484 are no longer in /astro/armin/koji/syndiff/data/ps1_skycells_zarr (checked 2026-10-05); re-pin to the v3 per-band pilot"), strict=False)

from syndiff_pipeline.template_creation.processing import perband as PB
from syndiff_pipeline.template_creation.processing.combined_store import DEFAULT_BAND_WEIGHTS

import b_fixtures as BF
from syndiff_pipeline.forward_model.chain.perband import f01b_lists, f02_band_cells, f03_band_contrib, f04_reduce
from syndiff_pipeline.forward_model.chain.perband import paths as PP

D13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}


def _raw(seed=1, shape=(30, 36)):
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    bands = {b: rng.normal(5.0, 3.0, shape).astype(np.float32) for b in PB.BANDS}
    bands["z"][3, 4] = np.nan
    hdr = {}
    for k, b in enumerate(PB.BANDS):
        h = fits.Header()
        h["BOFFSET"], h["BSOFTEN"], h["EXPTIME"] = 100.0 + k, 50.0 + 3 * k, 30.0 + k
        hdr[b] = h.tostring()
    return bands, hdr


def _stored(wb, seed=2):
    """A 'stored combined cell': the weighted band sum with a few pixels zeroed (remove_background's Z)."""
    before = PB.sum_bands(wb)
    after = before.copy()
    rng = np.random.default_rng(seed)
    after[rng.random(after.shape) < 0.3] = 0.0
    return before, after


# ------------------------------------------------------------------ band weights
def test_production_store_split_is_bit_identical_to_old_path():
    bands, hdr = _raw()
    wb_old = PB.weighted_band_images(bands, hdr)                    # DEFAULT weights, as before the change
    before, after = _stored(wb_old)
    old = PB.split_like_combined(wb_old, after, combined_before=before)
    wb, new, b2 = PB.split_with_store_weights(bands, hdr, after, dict(DEFAULT_BAND_WEIGHTS),
                                              expected_band_weights=dict(DEFAULT_BAND_WEIGHTS))
    assert np.array_equal(b2, before, equal_nan=True)
    for b in PB.BANDS:
        assert wb[b].dtype == wb_old[b].dtype and np.array_equal(wb[b], wb_old[b], equal_nan=True)
        assert new[b].dtype == np.float32 and np.array_equal(new[b], old[b], equal_nan=True)


def test_adopted_store_split_uses_the_stores_weights():
    bands, hdr = _raw()
    wb = PB.weighted_band_images(bands, hdr, weights=D13)
    before, after = _stored(wb)
    _, split, b2 = PB.split_with_store_weights(bands, hdr, after, D13, expected_band_weights=D13)
    assert np.array_equal(b2, before, equal_nan=True)
    # C_b = Z * w'_b F_b, summing back to the stored (D13-weighted) cell to float32 rounding
    res = PB.split_residual(split, after)
    assert res["n_nan_mismatch"] == 0 and res["max_rel_to_peak"] < 1e-6
    zeroed = PB.zeroed_mask(before, after)
    assert np.all(split["r"][zeroed] == 0)
    # and it differs from a production-weight split by exactly the weight ratio in the kept pixels
    wb_p = PB.weighted_band_images(bands, hdr)
    keep = ~zeroed & np.isfinite(wb_p["r"])
    ratio = split["r"][keep].astype(np.float64) / wb_p["r"][keep].astype(np.float64)
    nz = wb_p["r"][keep] != 0
    assert np.allclose(ratio[nz], D13["r"] / DEFAULT_BAND_WEIGHTS["r"], rtol=1e-5)


def test_split_refuses_mismatching_weights():
    bands, hdr = _raw()
    wb = PB.weighted_band_images(bands, hdr, weights=D13)
    _, after = _stored(wb)
    with pytest.raises(ValueError, match="band_weights"):
        PB.split_with_store_weights(bands, hdr, after, D13, expected_band_weights=dict(DEFAULT_BAND_WEIGHTS))
    with pytest.raises(ValueError):
        PB.band_weights_from_recipe({"band_weights": {"r": 1.0}})
    with pytest.raises(ValueError):
        PB.band_weights_from_recipe({})


def test_weight_rescale_both_store_cases():
    adopted = json.loads(Path(BF.ADOPTED).read_text()) if Path(BF.ADOPTED).exists() else None
    s_prod = PB.weight_rescale(DEFAULT_BAND_WEIGHTS, D13)
    assert s_prod == {b: D13[b] / DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS}
    if adopted:      # bit-identical to the constants the e2e run used
        assert [s_prod[b] for b in PB.BANDS] == adopted["scale_vs_production"]
    s_same = PB.weight_rescale(D13, D13)
    assert all(v == 1.0 for v in s_same.values())          # exactly 1: no second rescale of an adopted store
    assert PB.same_band_weights(D13, dict(D13)) and not PB.same_band_weights(D13, DEFAULT_BAND_WEIGHTS)


def test_chain_band_weights_modes(tmp_path):
    (tmp_path / "adopted.json").write_text(json.dumps({"weights_rizy": [D13[b] for b in PB.BANDS]}))
    cfg = BF.make_cfg(tmp_path)
    assert PP.chain_band_weights(cfg) == {b: DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS}
    cfg2 = BF.make_cfg(tmp_path, extra_inputs={"combined_store_weights": "adopted"})
    assert PP.chain_band_weights(cfg2) == D13
    with pytest.raises(ValueError):
        PP.chain_band_weights(BF.make_cfg(tmp_path, extra_inputs={"combined_store_weights": "bogus"}))


def test_store_recipe_carries_weights(tmp_path):
    (tmp_path / "adopted.json").write_text(json.dumps({"weights_rizy": [D13[b] for b in PB.BANDS]}))
    cfg_p = BF.make_cfg(tmp_path)
    cfg_a = BF.make_cfg(tmp_path, extra_inputs={"combined_store_weights": "adopted"})
    rp = f02_band_cells.store_recipe(cfg_p, PP.chain_band_weights(cfg_p))
    ra = f02_band_cells.store_recipe(cfg_a, PP.chain_band_weights(cfg_a))
    assert PB.band_weights_from_recipe(rp) == {b: DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS}
    assert PB.band_weights_from_recipe(ra) == D13
    from syndiff_pipeline.template_creation.processing.combined_store import combined_recipe_id
    assert combined_recipe_id(rp) != combined_recipe_id(ra)       # distinct store fingerprints


def test_load_store_cell_reads_recorded_weights(tmp_path, monkeypatch):
    from syndiff_pipeline.template_creation.processing import combined_store as CS
    cell = tmp_path / "ps1_skycells_zarr/ps1_combined.zarr/skycell.2274/083/fp1"
    cell.mkdir(parents=True)
    np.savez(cell / CS._ARRAYS_FILENAME, combined_image=np.ones((3, 3), np.float32), combined_mask=np.zeros((3, 3), np.int32))
    (cell / CS._HEADERS_FILENAME).write_text("{}")
    (cell / CS._REMOVED_STARS_FILENAME).write_text("[]")
    (cell / "_provenance.json").write_text(json.dumps({"recipe_params": {"band_weights": D13}}))
    monkeypatch.setattr(CS, "resolve_combined_fingerprint_for_recipe", lambda *a, **k: "fp1")
    hit, w = f02_band_cells.load_store_cell(tmp_path, "skycell.2274.083", {"band_weights": dict(DEFAULT_BAND_WEIGHTS)})
    assert hit["combined_image"].shape == (3, 3)
    assert w == D13                                  # the sidecar's, not the requested recipe's
    (cell / "_provenance.json").unlink()
    _, w2 = f02_band_cells.load_store_cell(tmp_path, "skycell.2274.083", {"band_weights": dict(DEFAULT_BAND_WEIGHTS)})
    assert w2 == {b: DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS}      # falls back to the requested recipe
    monkeypatch.setattr(CS, "resolve_combined_fingerprint_for_recipe", lambda *a, **k: None)
    assert f02_band_cells.load_store_cell(tmp_path, "skycell.2274.083", {}) == (None, None)


def _cells_fixture(tmp_path, weights_by_cell):
    cfg = BF.make_cfg(tmp_path, extra_inputs={})
    P = PP.chain_paths(cfg)
    P.perband.mkdir(parents=True)
    P.band_cells.mkdir(parents=True)
    names = sorted(weights_by_cell)
    P.cells_json.write_text(json.dumps({"all_band_cells": names, "cells": names}))
    for n, w in weights_by_cell.items():
        chk = {} if w is None else {"store_band_weights": w}
        np.savez(P.band_cells / f"{n}.npz", r=np.zeros(2, np.float32), check=json.dumps(chk))
    return cfg, P


def test_f04_store_weights_bookkeeping(tmp_path):
    # legacy cells (no recorded weights) are the production store
    cfg, P = _cells_fixture(tmp_path / "a", {"skycell.1.001": None, "skycell.1.002": dict(DEFAULT_BAND_WEIGHTS)})
    out = f04_reduce.collect_store_weights(P, cfg)
    assert out["store_band_weights"] == {b: DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS}
    # a mixed store is an error
    cfg, P = _cells_fixture(tmp_path / "b", {"skycell.1.001": dict(DEFAULT_BAND_WEIGHTS), "skycell.1.002": D13})
    with pytest.raises(ValueError, match="different store weights"):
        f04_reduce.collect_store_weights(P, cfg)
    # an adopted-weight store against a production-assuming chain is an error ...
    cfg, P = _cells_fixture(tmp_path / "c", {"skycell.1.001": D13, "skycell.1.002": D13})
    with pytest.raises(ValueError, match="chain-assumed"):
        f04_reduce.collect_store_weights(P, cfg)
    # ... and fine when the chain is configured for it
    (tmp_path / "c").mkdir(exist_ok=True)
    (tmp_path / "c" / "adopted.json").write_text(json.dumps({"weights_rizy": [D13[b] for b in PB.BANDS]}))
    cfg_a = BF.make_cfg(tmp_path / "c", extra_inputs={"combined_store_weights": "adopted"})
    out = f04_reduce.collect_store_weights(P, cfg_a)
    assert out["store_band_weights"] == D13


# ------------------------------------------------------------------ f01b grouping regression
def test_f01b_group_key_includes_row_layout():
    """Lists giving the same neighbour NAMES but a different row layout (cells/x-indices of rows R-1..R+1) must NOT
    collapse into one candidate (skycell.2486.029: own list 0.81 vs 1e-7 relative blur error)."""
    nb = lambda md, c, R, X: [("n1", 0, 0), ("n2", 0, 0)]       # identical neighbour names for every list
    md1 = {"rows": {0: [("a", 0), ("b", 1)], 1: [("c", 0), ("d", 1)], 2: [("e", 0)]}}
    md2 = {"rows": {0: [("a", 0), ("b", 1)], 1: [("c", 0), ("d", 1)], 2: [("e", 0), ("f", 1)]}}     # row R+1 has an extra cell
    md3 = {"rows": {0: [("a", 0), ("b", 1)], 1: [("c", 0), ("d", 1)], 2: [("e", 0)], 7: [("z", 0)]}}  # differs far from R only
    k1, k2, k3 = (f01b_lists.group_key(m, "c", nb) for m in (md1, md2, md3))
    assert k1[0] == k2[0] == k3[0]          # same names
    assert k1 != k2                          # different layout of row R+1: distinct candidates
    assert k1 == k3                          # rows outside R-1..R+1 do not matter
    sets = {k1: ["own"], k2: ["x", "y", "z"]}
    ranked = f01b_lists.rank_candidates(sets, "own")
    assert [ps for _k, ps in ranked] == [["own"], ["x", "y", "z"]]      # own list first
    sets = {k1: ["p"], k2: ["x", "y"]}
    assert [ps for _k, ps in f01b_lists.rank_candidates(sets, "own")] == [["x", "y"], ["p"]]   # then most-shared first


# ------------------------------------------------------------------ f03 Condor array
def test_f03_submit_array_text(tmp_path):
    cfg = BF.make_cfg(tmp_path)
    sub = f03_band_contrib.submit(cfg, n_chunks=7, n_jobs=3)
    t = sub.read_text()
    assert "queue 7" in t and "request_cpus = 8" in t and "request_memory = 80000" in t
    assert "f03_band_contrib" in t and "$(Process)" in t and "--jobs 3" in t
    assert t.count("f03_$(Process).out") == 1 and t.count("f03_$(Process).err") == 1
    assert "PYTHONPATH=" in t and str(cfg.code.forward_model_root) in t


def test_paths_from_config(tmp_path):
    P = PP.chain_paths(BF.make_cfg(tmp_path))
    assert P.perband == tmp_path / "out/perband" and P.band_cells == tmp_path / "out/perband/band_cells"
    assert P.mapping_dir == tmp_path / "out/mapping/oversampling_4"
    assert P.master_name == "tess_s0024_2_2_master_pixels2skycells_os4.fits.fz"
    assert P.scc == tmp_path / "data/s0024/c2/k2"
    assert P.frames == [BF.STEM, BF.UNSEEN]
    cfg = BF.make_cfg(tmp_path)
    cfg2 = BF.make_cfg(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "band_cells": str(tmp_path / "bc")})
    assert PP.chain_paths(cfg2).band_cells == tmp_path / "bc"


# ------------------------------------------------------------------ slow parity (e2e F1 products)
slow = pytest.mark.slow


def _e2e_f1(tmp_path):
    """tmp F1 config whose perband inputs are the e2e products (symlinks), outputs fresh."""
    BF.need(BF.E2E_RUN / "perband/out/F1/cells.json", BF.E2E_RUN / "mapping/oversampling_4", BF.E2E_RUN / "perband/out/band_cells",
            BF.E2E_RUN / "perband/out/contrib_F1")
    cfg = BF.f1_cfg(tmp_path)
    BF.link(tmp_path / "F1/mapping", BF.E2E_RUN / "mapping")
    return cfg


@slow
def test_f01_select_matches_e2e(tmp_path):
    cfg = _e2e_f1(tmp_path)
    from syndiff_pipeline.forward_model.chain.perband import f01_select
    out = f01_select.run(cfg)
    ref = json.loads((BF.E2E_RUN / "perband/out/F1/cells.json").read_text())
    assert out == json.loads(json.dumps(ref)) or json.loads(json.dumps(out)) == ref


@E2E_STALE
@slow
def test_f02_band_cell_matches_e2e_bitwise(tmp_path):
    """Rebuild one band cell from the combined store + raw zarr; bitwise equal to the e2e cell; records production weights."""
    cfg = _e2e_f1(tmp_path)
    P = PP.chain_paths(cfg)
    name = "skycell.2484.045"
    ref = BF.E2E_RUN / "perband/out/band_cells" / f"{name}.npz"
    BF.need(ref, P.raw_zarr)
    cfg2 = BF.f1_cfg(tmp_path, band_cells=str(tmp_path / "bc"))
    (tmp_path / "bc").mkdir()
    _, chk = f02_band_cells.one(cfg2, name)
    assert "error" not in chk, chk
    assert chk["store_band_weights"] == {b: DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS}
    BF.arrays_equal(tmp_path / "bc" / f"{name}.npz", ref, keys=list(PB.BANDS))
    old = json.loads(str(np.load(ref)["check"]))
    assert chk["max_rel_to_peak"] == old["max_rel_to_peak"] and chk["zeroed_frac"] == old["zeroed_frac"]


@E2E_STALE
@slow
def test_f03_contrib_matches_e2e_bitwise(tmp_path):
    """Two cells (one plain, one seam-corrected cross-projection) re-run from the e2e band cells + lists: contrib arrays bitwise."""
    cfg = _e2e_f1(tmp_path)
    P = PP.chain_paths(cfg)
    cells = json.loads((BF.E2E_RUN / "perband/out/F1/cells.json").read_text())
    BF.link(P.perband / "cells.json", BF.E2E_RUN / "perband/out/F1/cells.json")
    BF.link(P.perband / "publisher_lists.json", BF.E2E_RUN / "perband/out/F1/publisher_lists.json")
    cfg = BF.f1_cfg(tmp_path, band_cells=str(BF.E2E_RUN / "perband/out/band_cells"))
    xp = sorted(cells["xproj"])[0]
    plain = next(c for c in cells["cells"] if c not in cells["xproj"])
    for name in (plain, xp):
        ref = BF.E2E_RUN / "perband/out/contrib_F1" / f"{name}.npz"
        BF.need(ref)
        _, chk = f03_band_contrib.one(cfg, name)
        assert "error" not in chk, chk
        BF.arrays_equal(P.contrib / f"{name}.npz", ref, skip=("check",))
        old = json.loads(str(np.load(ref)["check"]))
        assert chk["list_chosen"] == old["list_chosen"] and chk["xproj"] == old["xproj"]
        assert chk["max_rel_to_peak"] == old["max_rel_to_peak"]
    assert json.loads(str(np.load(P.contrib / f"{xp}.npz")["check"]))["xproj"]


@slow
def test_f04_reduce_matches_e2e_bitwise(tmp_path):
    cfg = _e2e_f1(tmp_path)
    P = PP.chain_paths(cfg)
    BF.link(P.perband / "cells.json", BF.E2E_RUN / "perband/out/F1/cells.json")
    BF.link(P.perband / "contrib", BF.E2E_RUN / "perband/out/contrib_F1")
    cfg = BF.f1_cfg(tmp_path, band_cells=str(BF.E2E_RUN / "perband/out/band_cells"))
    f04_reduce.run(cfg, make_figure=False)
    ref = BF.E2E_RUN / "perband/out/band_templates/F1/a3"
    for k in ("T_r", "T_i", "T_z", "T_y", "T_prodpath", "count", "T_sum"):
        a, b = np.load(P.band_templates / f"{k}.npy"), np.load(ref / f"{k}.npy")
        assert np.array_equal(a, b), k
    sw = json.loads(P.store_weights_json.read_text())
    assert sw["store_band_weights"] == {b: DEFAULT_BAND_WEIGHTS[b] for b in PB.BANDS} and not sw["missing"]
    new, old = json.loads((P.band_templates / "sum_check.json").read_text()), json.loads((ref / "sum_check.json").read_text())
    for k in ("bandsum_vs_template_max_rel", "prodpath_vs_template_max_rel", "band_share", "flux"):
        assert new[k] == old[k], k


# ------------------------------------------------------------------ CLI stage modules (chain.cli DISPATCH names)
@pytest.mark.parametrize("stage,mod", [("select", "perband.select"), ("lists", "perband.lists"), ("band_cells", "perband.band_cells"),
                                       ("contrib", "perband.contrib"), ("reduce", "perband.reduce"), ("kernels", "kernels"),
                                       ("hotpants", "hotpants_ref"), ("final", "final"), ("score", "score")])
def test_cli_dispatch_targets_exist(stage, mod):
    import importlib
    import inspect
    from syndiff_pipeline.forward_model.chain import cli
    assert cli.DISPATCH[stage] == mod
    m = importlib.import_module(f"syndiff_pipeline.forward_model.chain.{mod}")
    assert "cfg" in inspect.signature(m.run).parameters


def test_contrib_stage_condor_writes_array(tmp_path, monkeypatch):
    from syndiff_pipeline.forward_model.chain import condor
    from syndiff_pipeline.forward_model.chain.perband import contrib
    monkeypatch.setattr(condor, "submit", lambda sub: f"submitted {sub.name}")
    sub = contrib.run(BF.make_cfg(tmp_path), condor=True, args=["--n-chunks", "5", "--jobs", "2"])
    assert "queue 5" in sub.read_text() and "--jobs 2" in sub.read_text()


def test_reduce_stage_refuses_incomplete(tmp_path, monkeypatch):
    from syndiff_pipeline.forward_model.chain.perband import f04_reduce, reduce as RD
    monkeypatch.setattr(f04_reduce, "run", lambda cfg, make_figure=True: {"missing": ["skycell.1.001"], "errors": {}})
    with pytest.raises(RuntimeError, match="incomplete"):
        RD.run(BF.make_cfg(tmp_path))
