"""Colour-selected faint-star background exclusion (``tessreduce_faint_star_*``; dev_runs/ksb_stability_20261008)."""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from syndiff_pipeline.common.provenance.fingerprint import recipe_id
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue
from syndiff_pipeline.difference_imaging.orchestration import stage_params as sp
from syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual import (
    faint_star_exclusion_from_catalog,
    star_wing_exclusion,
    star_wing_exclusion_from_catalog,
)

RED_FLOOR5 = [[13.5, 7], [14.0, 6], [14.5, 5], [16.0, 5]]
SHAPE = (60, 90)


def _catalog(tmp_path):
    rows = [
        # x, y, tess_mag, bp, rp
        (10.0, 10.0, 14.2, 18.0, 16.3),  # red, faint -> selected (r=5)
        (40.0, 10.0, 14.2, 17.0, 16.3),  # blue -> not selected (colour 0.7)
        (70.0, 10.0, 12.0, 18.0, 16.0),  # red but brighter than tmag_min
        (10.0, 40.0, 13.2, 17.4, 16.0),  # colour 1.4, r=7
        (40.0, 40.0, 14.2, np.nan, 16.0),  # NaN colour
        (70.0, 40.0, 16.5, 20.0, 18.0),  # fainter than table
    ]
    path = tmp_path / "cat.csv"
    pd.DataFrame(rows, columns=["x", "y", "tess_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]).to_csv(path, index=False)
    return str(path)


def test_selection_and_radius_table(tmp_path):
    csv = _catalog(tmp_path)
    got = faint_star_exclusion_from_catalog(csv, SHAPE, RED_FLOOR5, 13.0, 1.2)
    want = star_wing_exclusion(SHAPE, np.array([10.0, 10.0]), np.array([10.0, 40.0]), np.array([14.2, 13.2]), RED_FLOOR5)
    assert np.array_equal(got, want)
    assert got[10, 10] and got[10, 15] and not got[10, 16]  # r = 5 for T=14.2
    assert got[40, 17] and not got[40, 18]  # r = 7 for T=13.2


def test_no_colour_cut_selects_by_tmag_only_and_nan_colour_included(tmp_path):
    csv = _catalog(tmp_path)
    got = faint_star_exclusion_from_catalog(csv, SHAPE, RED_FLOOR5, 13.0, None)
    assert got[10, 40] and got[40, 40]  # blue and NaN-colour stars now selected
    assert not got[10, 70]  # T=12 < tmag_min
    assert not got[40, 70]  # T=16.5 beyond the table


def test_nan_colour_not_selected_with_cut(tmp_path):
    csv = _catalog(tmp_path)
    assert not faint_star_exclusion_from_catalog(csv, SHAPE, RED_FLOOR5, 13.0, 1.2)[40, 40]


def test_missing_colour_columns_error(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"x": [1.0], "y": [1.0], "tess_mag": [14.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="phot_bp_mean_mag"):
        faint_star_exclusion_from_catalog(str(p), SHAPE, RED_FLOOR5, 13.0, 1.2)
    assert faint_star_exclusion_from_catalog(str(p), SHAPE, RED_FLOOR5, 13.0, None).any()


def test_union_with_wing_disks_and_none_when_unset(tmp_path, monkeypatch):
    from syndiff_pipeline.difference_imaging.orchestration import execute

    csv = _catalog(tmp_path)
    monkeypatch.setattr(execute, "_diff_lane_root_dir", lambda cfg, ctx: str(tmp_path))
    monkeypatch.setattr(execute, "GAIA_CATALOG_PIPELINE_BASENAME", "cat.csv")
    wing = [[13.0, 6]]
    kw = dict(tessreduce_star_wing_radii=None, tessreduce_faint_star_radii=None,
              tessreduce_faint_star_tmag_min=None, tessreduce_faint_star_bp_rp_min=None)
    ns = lambda **o: type("P", (), {**kw, **o})()  # noqa: E731
    assert execute._background_exclusion_for_stage(None, None, ns(), SHAPE) is None
    w = execute._background_exclusion_for_stage(None, None, ns(tessreduce_star_wing_radii=wing), SHAPE)
    assert np.array_equal(w, star_wing_exclusion_from_catalog(csv, SHAPE, wing))
    f = execute._background_exclusion_for_stage(
        None, None, ns(tessreduce_faint_star_radii=RED_FLOOR5, tessreduce_faint_star_tmag_min=13.0,
                       tessreduce_faint_star_bp_rp_min=1.2), SHAPE)
    assert np.array_equal(f, faint_star_exclusion_from_catalog(csv, SHAPE, RED_FLOOR5, 13.0, 1.2))
    both = execute._background_exclusion_for_stage(
        None, None, ns(tessreduce_star_wing_radii=wing, tessreduce_faint_star_radii=RED_FLOOR5,
                       tessreduce_faint_star_tmag_min=13.0, tessreduce_faint_star_bp_rp_min=1.2), SHAPE)
    assert np.array_equal(both, w | f) and both.sum() > max(w.sum(), f.sum())


# ---- stage_params ----------------------------------------------------------------------------------------------

FAINT = dict(tessreduce_faint_star_radii=RED_FLOOR5, tessreduce_faint_star_tmag_min=13, tessreduce_faint_star_bp_rp_min=1.2)


@pytest.mark.parametrize("kind,parse", [("background_estimate", sp.parse_background_estimate),
                                        ("kernel_fit", sp.parse_kernel_fit)])
def test_stage_params_accept_and_validate(kind, parse):
    p = parse({"kind": kind, **FAINT}, 0)
    assert p.tessreduce_faint_star_radii == RED_FLOOR5
    assert p.tessreduce_faint_star_tmag_min == 13.0 and p.tessreduce_faint_star_bp_rp_min == 1.2
    p = parse({"kind": kind, **{**FAINT, "tessreduce_faint_star_bp_rp_min": None}}, 0)
    assert p.tessreduce_faint_star_bp_rp_min is None
    for bad, msg in [
        ({"tessreduce_faint_star_radii": [[14.0, 5], [13.0, 4]]}, "faint_star_radii"),
        ({"tessreduce_faint_star_tmag_min": None}, "tmag_min is required"),
        ({"tessreduce_faint_star_tmag_min": "x"}, "finite number"),
        ({"tessreduce_faint_star_bp_rp_min": float("nan")}, "finite number"),
    ]:
        with pytest.raises(ValueError, match=msg):
            parse({"kind": kind, **{**FAINT, **bad}}, 0)
    with pytest.raises(ValueError, match="need tessreduce_faint_star_radii"):
        parse({"kind": kind, "tessreduce_faint_star_tmag_min": 13.0}, 0)


def test_unset_config_unchanged_params_and_recipe():
    ks = sp.parse_background_estimate({"kind": "background_estimate", "tessreduce_star_wing_radii": [[9.0, 6]]}, 0)
    kf = sp.parse_kernel_fit({"kind": "kernel_fit"}, 0)
    for p in (ks, kf):
        assert p.tessreduce_faint_star_radii is None
        recipe = provenance_glue.diff_recipe("diff_image", p)["params"]
        flat = repr(recipe)
        assert "faint_star" not in flat
        base = {f.name for f in dataclasses.fields(p)} - set(sp.KernelFitParams._RECIPE_OMIT_WHEN_NONE)
        assert set(dataclasses.asdict(p)) - set(sp.KernelFitParams._RECIPE_OMIT_WHEN_NONE) == base
    # set keys do enter the recipe and change its id
    ks2 = sp.parse_background_estimate({"kind": "background_estimate", **FAINT}, 0)
    r0 = provenance_glue.diff_recipe("diff_image", ks)
    r2 = provenance_glue.diff_recipe("diff_image", ks2)
    assert "tessreduce_faint_star_radii" in repr(r2["params"])
    assert recipe_id("diff_image", r0["params"], r0["code_version"]) != recipe_id("diff_image", r2["params"], r2["code_version"])


# ---- golden ----------------------------------------------------------------------------------------------------

LANE = "/astro/armin/koji/syndiff/dev_runs/paper_dataset_20261001/data_root/s0024/c2/k2/diff_linear"
GOLD = "/astro/armin/koji/syndiff/dev_runs/ksb_stability_20261008/offset_masks/red_floor5_extra.npy"


def test_golden_red_floor5_mask_matches_ksb_stability_record():
    if not (os.path.exists(f"{LANE}/gaia_catalog_pipeline.csv") and os.path.exists(f"{LANE}/diff_config.yaml")
            and os.path.exists(GOLD)):
        pytest.skip("S24 C2K2 paper-dataset lane / golden mask not available")
    cfg = yaml.safe_load(open(f"{LANE}/diff_config.yaml"))
    stage = next(s for s in cfg["pipeline"] if s.get("kind") == "background_estimate")
    p = sp.parse_background_estimate({**stage, **FAINT}, 0)
    csv = f"{LANE}/gaia_catalog_pipeline.csv"
    shape = (2048, 2048)
    wing = star_wing_exclusion_from_catalog(csv, shape, p.tessreduce_star_wing_radii)
    faint = faint_star_exclusion_from_catalog(
        csv, shape, p.tessreduce_faint_star_radii, p.tessreduce_faint_star_tmag_min, p.tessreduce_faint_star_bp_rp_min)
    gold = np.load(GOLD)
    assert gold.shape == shape
    assert np.array_equal(wing | faint, gold), int(((wing | faint) != gold).sum())


# ---- tessreduce_residual_exclude_percentile ---------------------------------------------------------------------

def _synthetic_frame():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:80, 0:80]
    img = 0.02 * xx + 0.5 * np.sin(yy / 9.0) + rng.normal(0, 0.3, (80, 80))
    mask = np.zeros((80, 80), dtype=np.int32)
    mask[20:40, 20:40] = 1
    mask[:, ::7] |= 4
    return img, mask


def test_residual_exclude_percentile_unset_is_bit_identical_and_set_changes(monkeypatch):
    from syndiff_pipeline.difference_imaging.stages.background import tessreduce_residual as TR

    img, mask = _synthetic_frame()
    seen = []
    orig = TR.Background2D

    def spy(*a, **k):
        if "fill_value" in k:  # the residual-surface call (other Background2D calls fix exclude_percentile=50)
            seen.append(k.get("exclude_percentile", "absent"))
        return orig(*a, **k)

    monkeypatch.setattr(TR, "Background2D", spy)
    base, _, _ = TR.estimate_tessreduce_residual_background(img, mask)
    n_unset = len(seen)
    assert set(seen) == {"absent"}
    again, _, _ = TR.estimate_tessreduce_residual_background(img, mask, residual_exclude_percentile=None)
    assert np.array_equal(base, again)
    seen.clear()
    got, _, _ = TR.estimate_tessreduce_residual_background(img, mask, residual_exclude_percentile=50.0)
    assert seen == [50.0] * n_unset and n_unset >= 1
    assert not np.array_equal(base, got)


@pytest.mark.parametrize("kind,parse", [("background_estimate", sp.parse_background_estimate),
                                        ("kernel_fit", sp.parse_kernel_fit)])
def test_residual_exclude_percentile_params(kind, parse):
    assert parse({"kind": kind}, 0).tessreduce_residual_exclude_percentile is None
    assert parse({"kind": kind, "tessreduce_residual_exclude_percentile": 50}, 0).tessreduce_residual_exclude_percentile == 50.0
    for bad in (0, -1, 101, "x", float("nan"), True):
        with pytest.raises(ValueError, match="exclude_percentile"):
            parse({"kind": kind, "tessreduce_residual_exclude_percentile": bad}, 0)
    p0 = parse({"kind": kind}, 0)
    p1 = parse({"kind": kind, "tessreduce_residual_exclude_percentile": 50}, 0)
    assert "exclude_percentile" not in repr(provenance_glue.diff_recipe("diff_image", p0)["params"])
    assert "exclude_percentile" in repr(provenance_glue.diff_recipe("diff_image", p1)["params"])
