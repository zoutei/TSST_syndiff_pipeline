"""Slow parity tests: re-run stages on the e2e F1 inputs and compare with the e2e products (skipped if absent)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from syndiff_pipeline.forward_model.chain import compare as K
from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain import scene as S
from syndiff_pipeline.forward_model.chain import wcs_export as W

from chain_fixtures import CHAIN_DIR

E = Path("/astro/armin/koji/syndiff/dev_runs/e2e_f1_20260930")
SRC = Path("/home/kshukawa/syndiff_pipeline/dev/forward_epsf_wcs/output/runs/sffi_colour_20260924/scenes/F1_s24c2k2")
HP = E / "s12_bootstrap/diff_tvwcs/hp_d/tess2020120182919-s0024-2-2_hp_d.fits.fz"
DROP = Path("/astro/armin/koji/syndiff/dev_runs/seam_doublecount_20260929/nbr_f4_fitstars_drop_gt1sigma.csv")

pytestmark = pytest.mark.slow


def _need(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"{p} not available")


def _f1_cfg(tmp_path):
    raw = yaml.safe_load((CHAIN_DIR / "configs/F1.yaml").read_text())
    raw["out_root"] = str(tmp_path / "F1")
    return C.config_from_dict(raw)


def test_scene_swap_demote_bitwise(tmp_path):
    _need(SRC, HP, DROP, E / "s3_fit/scene")
    out = tmp_path / "scene"
    S.build_scene(SRC, out, hp_d=HP, exclusion_csv=DROP, exclusion_strict=True)   # strict = e2e demote_stars assert
    a, b = np.load(out / "scene_bundle.npz"), np.load(E / "s3_fit/scene/scene_bundle.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k], equal_nan=True), k
    mine = json.loads((out / "scene_meta.json").read_text())
    for k in ("csv_sha256", "n_listed", "n_matched_in_scene", "n_listed_not_in_scene", "strict"):   # new provenance keys
        mine["demoted"].pop(k)
    assert mine == json.loads((E / "s3_fit/scene/scene_meta.json").read_text())


def test_wcs_export_bitwise(tmp_path):
    _need(E / "s3_fit/fit/params.npz", E / "run/wcs/e2e_f1_v1", E / "s3_fit/scene", C.load_config(CHAIN_DIR / "configs/F1.yaml").reference.tvwcs_store,
          C.load_config(CHAIN_DIR / "configs/F1.yaml").reference.old_store)
    cfg = _f1_cfg(tmp_path)
    out = tmp_path / "wcs"
    res = W.export_wcs(cfg, fit_dir=E / "s3_fit/fit", scene_dir=E / "s3_fit/scene", out_dir=out, version="e2e_f1_v1")
    assert res["gateA"]["pass"] and res["gateB"]["pass"]
    ref = E / "run/wcs"
    assert (out / "e2e_f1_v1/models/orbit_00.npz").read_bytes() == (ref / "e2e_f1_v1/models/orbit_00.npz").read_bytes()
    import pandas as pd
    pd.testing.assert_frame_equal(pd.read_parquet(out / "e2e_f1_v1/frames.parquet"), pd.read_parquet(ref / "e2e_f1_v1/frames.parquet"))
    ja, jb = res, json.loads((ref / "wcs_export_gates.json").read_text())
    ja = json.loads(json.dumps(ja))
    ja.pop("store"), jb.pop("store")
    assert ja == jb
    fa, fb = np.load(out / "gateB_fields.npz"), np.load(ref / "gateB_fields.npz")
    assert sorted(fa.files) == sorted(fb.files) and all(np.array_equal(fa[k], fb[k]) for k in fa.files)


def _same(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    else:
        assert a == b, (path, a, b)


def test_compare_equals_e2e(tmp_path):
    ref = E / "s8_refit/compare_fits.json"
    BO = Path("/astro/armin/koji/syndiff/dev_runs/prior_bakeoff_20260930/runs")
    R = E / "s8_refit"
    fits = {"boot": E / "s3_fit/fit", "R1_final_xsec": R / "R1_final_xsec/fit", "R2_final_warm": R / "R2_final_warm/fit",
            "rehearsal_oldw": E / "s3_rehearsal/fit", "lp7s_fold0": BO / "F1_lp7s_fold0/fit", "lp7s_fold1": BO / "F1_lp7s_fold1/fit"}
    pairs = [("R1_final_xsec", "boot", "refit on FINAL image, same init + recipe (the D14 test)"),
             ("R2_final_warm", "boot", "warm continuation on FINAL image from the calibration"),
             ("boot", "rehearsal_oldw", "bootstrap weights: adopted vs production (yardstick)"),
             ("lp7s_fold1", "lp7s_fold0", "split-half: different 80% star sets (yardstick)")]
    _need(ref, E / "s3_fit/scene", *[fits[k] / "params.npz" for k in ("boot", "R1_final_xsec")])
    res = K.compare_fits(fits, pairs, E / "s3_fit/scene", tmp_path)
    _same(json.loads(json.dumps(res)), json.loads(ref.read_text()))
    assert (tmp_path / "compare_fits.md").read_text() == (R / "compare_fits.md").read_text()
