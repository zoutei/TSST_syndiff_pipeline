"""template_calibration (zpcell): config parsing, f04 reduce unit tests on synthetic contribs, and slow parity tests against
the run4 products and the out-of-tree reference T' (bandw_kernel_shift_20261009)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import b_fixtures as BF
from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain.config import ConfigError
from syndiff_pipeline.forward_model.chain.perband import f04_reduce
from syndiff_pipeline.forward_model.chain.perband import paths as PP

BANDS = ("r", "i", "z", "y")
KEYS = ("prod",) + BANDS
CELLS = ("skycell.1.001", "skycell.1.002")
SHP = (4, 5)


def _raw(tmp_path, **tc):
    raw = {"field": "T1", "scc": {"sector": 24, "camera": 2, "ccd": 2}, "stem": BF.STEM, "unseen_stem": BF.UNSEEN,
           "data_root": str(tmp_path / "data"), "out_root": str(tmp_path / "out"),
           "inputs": {"colour_file": str(tmp_path / "colour.csv"), "adopted_weights": str(tmp_path / "adopted.json")},
           "code": {"forward_model_root": str(Path(C.__file__).resolve().parents[3])}}
    if tc:
        raw["template_calibration"] = tc
    return raw


# ------------------------------------------------------------------ config
def test_default_is_off_and_hash_unchanged(tmp_path):
    cfg = C.config_from_dict(_raw(tmp_path))
    assert cfg.template_calibration == C.TemplateCalibrationCfg("off", None)
    d = cfg.to_dict()
    assert "template_calibration" not in d                       # off section does not enter the hash
    off = C.config_from_dict(_raw(tmp_path, mode="off"))
    assert off.config_hash() == cfg.config_hash()
    # an on section does change the hash
    on = C.config_from_dict(_raw(tmp_path, mode="zpcell", factors=str(tmp_path / "f.json")))
    assert on.config_hash() != cfg.config_hash() and on.to_dict()["template_calibration"]["mode"] == "zpcell"


def test_existing_config_hash_matches_dict_without_section():
    p = Path(C.__file__).parent / "configs" / "F1.yaml"
    import yaml
    cfg = C.load_config(p)
    d = cfg.to_dict()
    assert "template_calibration" not in d
    import hashlib
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
    assert cfg.config_hash() == hashlib.sha256(blob.encode()).hexdigest()


def test_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="mode"):
        C.config_from_dict(_raw(tmp_path, mode="bogus"))
    with pytest.raises(ConfigError, match="factors"):
        C.config_from_dict(_raw(tmp_path, mode="zpcell"))
    with pytest.raises(ConfigError, match="unknown key"):
        C.config_from_dict(_raw(tmp_path, mode="off", extra=1))
    with pytest.raises(ConfigError, match="off"):
        C.config_from_dict(_raw(tmp_path, mode="off", factors="/x/f.json"))


def test_factors_field_substitution(tmp_path):
    cfg = C.config_from_dict(_raw(tmp_path, mode="zpcell", factors="/x/factors_{field}_hyb.json"))
    assert cfg.template_calibration.factors == Path("/x/factors_T1_hyb.json")


# ------------------------------------------------------------------ f04 on synthetic contribs
@pytest.fixture(autouse=True)
def _grid(monkeypatch, request):
    if request.node.get_closest_marker("slow"):
        return
    class G:
        def array_shape_os(self):
            return SHP
    monkeypatch.setattr("syndiff_pipeline.common.mapping_grid.load_mapping_grid_from_master", lambda *a, **k: G())


def _setup(tmp_path, factors=None, field="T1", tc_mode="zpcell"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = _raw(tmp_path)
    if tc_mode == "zpcell":
        fp = tmp_path / "factors.json"
        fp.write_text(json.dumps({"field": field, "bands": list(BANDS), "reference": "unit", "global_norm": 1.0,
                                  "zpcell": factors if factors is not None else {}}))
        raw["template_calibration"] = {"mode": "zpcell", "factors": str(fp)}
    cfg = C.config_from_dict(raw, config_path=tmp_path / "cfg.yaml")
    P = PP.chain_paths(cfg)
    P.contrib.mkdir(parents=True)
    P.band_cells.mkdir(parents=True)
    P.cells_json.write_text(json.dumps({"all_band_cells": list(CELLS), "cells": list(CELLS)}))
    rng = np.random.default_rng(3)
    for n in CELLS:
        np.savez(P.band_cells / f"{n}.npz", r=np.zeros(2, np.float32), check=json.dumps({}))
        kw = {"check": json.dumps({"max_rel_to_peak": 1e-7})}
        for k in KEYS:
            pix = rng.choice(SHP[0] * SHP[1], 9, replace=False).astype(np.int64)
            pix = np.concatenate([pix, pix[:3]])
            kw[f"a3_pix_{k}"] = pix
            kw[f"a3_sum_{k}"] = rng.normal(5.0, 1.0, pix.size)
        kw["a3_count"] = np.ones(kw["a3_pix_prod"].size)
        np.savez(P.contrib / f"{n}.npz", **kw)
    return cfg, P


def _load(P):
    return {k: np.load(P.band_templates / f"{k}.npy") for k in ("T_r", "T_i", "T_z", "T_y", "T_sum", "T_prodpath", "count")}


def _hand(P, fac):
    acc = {b: np.zeros(SHP[0] * SHP[1]) for b in BANDS}
    for n in CELLS:
        z = np.load(P.contrib / f"{n}.npz")
        for b in BANDS:
            np.add.at(acc[b], z[f"a3_pix_{b}"], z[f"a3_sum_{b}"] / fac[n][b])
    return {b: v.reshape(SHP) for b, v in acc.items()}


def _fac(val=None):
    rng = np.random.default_rng(9)
    return {n: {b: (1.0 if val else float(rng.uniform(0.8, 1.3))) for b in BANDS} for n in CELLS}


def test_off_mode_unchanged_and_unrecorded(tmp_path):
    cfg, P = _setup(tmp_path, tc_mode="off")
    s = f04_reduce.run(cfg, make_figure=False)
    out = _load(P)
    ref = _hand(P, {n: {b: 1.0 for b in BANDS} for n in CELLS})
    for b in BANDS:
        assert np.array_equal(out[f"T_{b}"], ref[b])
    assert "template_calibration" not in s


def test_ones_factors_bitwise_equal_to_off(tmp_path):
    cfg0, P0 = _setup(tmp_path / "a", tc_mode="off")
    f04_reduce.run(cfg0, make_figure=False)
    cfg1, P1 = _setup(tmp_path / "b", factors=_fac(1))
    s1 = f04_reduce.run(cfg1, make_figure=False)
    a, b = _load(P0), _load(P1)
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
    assert s1["template_calibration"]["n_cells_calibrated"] == 2


def test_known_factors_hand_computed_and_gate_uncalibrated(tmp_path):
    fac = _fac()
    cfg, P = _setup(tmp_path / "a", factors=fac)
    s = f04_reduce.run(cfg, make_figure=False)
    out = _load(P)
    hand = _hand(P, fac)
    for b in BANDS:
        assert np.array_equal(out[f"T_{b}"], hand[b]), b
    assert np.array_equal(out["T_sum"], sum(hand[b] for b in BANDS))
    # prod path, count and the sum check are the uncalibrated ones
    cfg0, P0 = _setup(tmp_path / "b", tc_mode="off")
    s0 = f04_reduce.run(cfg0, make_figure=False)
    o0 = _load(P0)
    assert np.array_equal(out["T_prodpath"], o0["T_prodpath"]) and np.array_equal(out["count"], o0["count"])
    for k in ("bandsum_vs_template_max_rel", "prodpath_vs_template_max_rel", "bandsum_vs_template_p999_rel"):
        assert s[k] == s0[k], k
    tc = s["template_calibration"]
    assert tc["mode"] == "zpcell" and len(tc["factors_sha256"]) == 64 and tc["reference"] == "unit" and tc["global_norm"] == 1.0
    for b in BANDS:
        v = [fac[n][b] for n in CELLS]
        assert tc["factor_min"][b] == min(v) and tc["factor_max"][b] == max(v) and tc["factor_median"][b] == float(np.median(v))
        assert tc["flux_calibrated"][b] == float(out[f"T_{b}"].sum())
        assert tc["flux_uncalibrated"][b] == float(o0[f"T_{b}"].sum())
    assert json.loads((P.band_templates / "sum_check.json").read_text())["template_calibration"] == tc


def test_missing_cell_raises(tmp_path):
    fac = _fac()
    fac.pop(CELLS[1])
    cfg, _ = _setup(tmp_path, factors=fac)
    with pytest.raises(ConfigError, match="lack 1 cell"):
        f04_reduce.run(cfg, make_figure=False)


def test_field_mismatch_and_bad_factors_raise(tmp_path):
    cfg, _ = _setup(tmp_path / "a", factors=_fac(1), field="F9")
    with pytest.raises(ConfigError, match="field"):
        f04_reduce.run(cfg, make_figure=False)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        fac = _fac(1)
        fac[CELLS[0]]["z"] = bad
        cfg, _ = _setup(tmp_path / f"b{bad}", factors=fac)
        with pytest.raises(ConfigError, match="finite"):
            f04_reduce.run(cfg, make_figure=False)


def test_reduce_stage_provenance_lists_factors_only_when_on(tmp_path, monkeypatch):
    from syndiff_pipeline.forward_model.chain.perband import reduce as RD
    seen = []
    monkeypatch.setattr(RD, "write_provenance", lambda stage, cfg, inputs: seen.append(sorted(inputs)))
    monkeypatch.setattr(RD, "mark_done", lambda stage: None)
    monkeypatch.setattr(f04_reduce, "run", lambda cfg, make_figure=True: {"missing": [], "errors": {}})
    for tag, mode in (("a", "off"), ("b", "zpcell")):
        cfg, P = _setup(tmp_path / tag, factors=_fac(1), tc_mode=mode)
        RD.run(cfg)
    assert "template_calibration_factors" not in seen[0] and "template_calibration_factors" in seen[1]


# ------------------------------------------------------------------ slow integration (run4 + reference)
slow = pytest.mark.slow
_spec = importlib.util.spec_from_file_location("verify_port", Path(__file__).resolve().parents[2] / "tools/template_zp/verify_port.py")
VP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VP)


@pytest.fixture
def big_tmp(request):
    """Scratch dir on /astro (the band templates are ~2.6 GB each; /tmp is too small), removed afterwards."""
    import shutil
    d = VP.OUT / "pytest_scratch" / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _need(field):
    BF.need(VP.RUN4 / field / "perband/cells.json", VP.RUN4 / field / "perband/contrib", VP.RUN4 / field / "perband/band_templates/T_r.npy",
            VP.RUN4 / field / "perband/provenance.json", VP.RUN4 / field / "mapping", VP.FACT / f"factors_{field}_hyb.json",
            VP.REF / field / "zpcell_hyb/T_r.npy")


@slow
@pytest.mark.parametrize("field", ["F1"])
def test_ones_factors_bitwise_equal_to_run4(big_tmp, field):
    tmp_path = big_tmp
    _need(field)
    VP.reduce_field(field, tmp_path, VP.ones_factors(field, tmp_path / "ones.json"))
    for k in ("T_r", "T_i", "T_z", "T_y", "T_sum", "T_prodpath", "count"):
        a = np.load(tmp_path / "perband/band_templates" / f"{k}.npy")
        b = np.load(VP.RUN4 / field / f"perband/band_templates/{k}.npy")
        assert np.array_equal(a, b), k


@slow
@pytest.mark.parametrize("field", ["F1"])
def test_real_factors_match_reference_tprime(big_tmp, field):
    tmp_path = big_tmp
    _need(field)
    VP.reduce_field(field, tmp_path, VP.FACT / f"factors_{field}_hyb.json")
    for b in BANDS:
        a = np.load(tmp_path / f"perband/band_templates/T_{b}.npy").astype(np.float32)
        ref = np.load(VP.REF / field / f"zpcell_hyb/T_{b}.npy")
        np.testing.assert_allclose(a, ref, rtol=1e-6, atol=0, err_msg=b)
