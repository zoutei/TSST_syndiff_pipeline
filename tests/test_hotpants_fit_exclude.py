"""Hotpants fit-only exclusion (``hp_star_wing_radii``) and ``hp_bgo: -1`` (dev_runs/bkg_offset_20261006)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams, parse_hotpants
from syndiff_pipeline.difference_imaging.stages import hotpants as HP
from syndiff_pipeline.forward_model.chain.bootstrap import lane_star_wing_radii

TABLE = [[9.0, 12], [11.0, 8], [13.0, 6]]


def _stage(**kw):
    return {"kind": "hotpants", "output": {"diffs": "d", "convolved": "c"}, **kw}


def test_parse_accepts_bgo_minus_one_and_wing_radii():
    hp = parse_hotpants(_stage(hp_bgo=-1, hp_star_wing_radii=TABLE), 0)
    assert hp.hp_bgo == -1
    assert hp.hp_star_wing_radii == TABLE


def test_parse_defaults_unchanged():
    hp = parse_hotpants(_stage(), 0)
    assert hp.hp_bgo == 3
    assert hp.hp_star_wing_radii is None


@pytest.mark.parametrize("bad", [[[11.0, 8], [9.0, 12]], [[9.0, 0]], [[9.0]]])
def test_parse_rejects_bad_wing_tables(bad):
    with pytest.raises(ValueError, match="hp_star_wing_radii"):
        parse_hotpants(_stage(hp_star_wing_radii=bad), 0)


def test_parse_rejects_bgo_below_minus_one():
    with pytest.raises(ValueError, match="hp_bgo"):
        parse_hotpants(_stage(hp_bgo=-2), 0)


def _frame(tmp_path, *, blob: bool, fit_only: bool):
    rng = np.random.default_rng(5)
    n = 120
    tpl = np.full((n, n), 50.0)
    sci = np.full((n, n), 50.0)
    stars = [(20, 20), (60, 25), (95, 30), (25, 70), (65, 65), (100, 95), (40, 100)]
    yy, xx = np.mgrid[-6:7, -6:7]
    psf = np.exp(-(xx**2 + yy**2) / (2 * 1.1**2))
    for x, y in stars:
        tpl[y - 6:y + 7, x - 6:x + 7] += 400 * psf
        sci[y - 6:y + 7, x - 6:x + 7] += 500 * psf
    tpl += rng.normal(0, 0.3, tpl.shape)
    sci += rng.normal(0, 0.3, sci.shape)
    exclude = np.zeros((n, n), dtype=bool)
    exclude[62:66, 68:71] = True  # inside the substamp box of the (65, 65) star
    if blob:
        sci[exclude] += 30.0
    hp = HotpantsParams(hp_sigma_gauss=[1.0, 1.88], hp_ngauss=2, hp_deg_fixe=[2, 2], hp_ko=0, hp_bgo=-1,
                        stamp_mode="connected_regions", region_min_npix=20, write_stamps=False)
    cfg = HP.build_hotpants_config(hp, str(tmp_path / "d"), str(tmp_path / "c"), "s", write_stamps=False,
                                   sci_shape=sci.shape)
    res = HP.run_hotpants_frame(sci, np.full_like(sci, 0.3), tpl, np.zeros((n, n), dtype=np.int32),
                                np.array(stars, dtype=float), cfg, oversample=1,
                                fit_only_exclude=exclude if fit_only else None)
    assert res["success"], res["error_msg"]
    return res, exclude


def test_fit_only_exclude_keeps_light_out_of_the_fit(tmp_path):
    clean, _ = _frame(tmp_path, blob=False, fit_only=True)
    blob, exclude = _frame(tmp_path, blob=True, fit_only=True)
    k0 = clean["kernel_params_arrays"]["kernel_solution"]
    k1 = blob["kernel_params_arrays"]["kernel_solution"]
    np.testing.assert_allclose(k1, k0, rtol=0, atol=1e-9)
    # the blob is in the difference image, not absorbed by the kernel
    assert np.median(blob["diff"][exclude] - clean["diff"][exclude]) == pytest.approx(30.0, abs=0.5)


def test_fit_only_exclude_not_flagged_in_output_mask(tmp_path):
    with_ex, exclude = _frame(tmp_path, blob=False, fit_only=True)
    without, _ = _frame(tmp_path, blob=False, fit_only=False)
    assert with_ex["mask"].shape == exclude.shape
    m1, m0 = np.asarray(with_ex["mask"]), np.asarray(without["mask"])
    assert not np.any(m1[exclude] & HP._HOTPANTS_FLAG_INPUT_MASK)
    # pyhotpants still marks the kernel footprint around the excluded pixels FLAG_OK_CONV (0x40, "convolved over a
    # flagged pixel, OK"); the chain treats {0, 64} as good (_tk.run_match), so only that benign bit may differ
    assert not np.any((m1 ^ m0) & ~0x40)


def test_fit_only_exclude_shape_mismatch_fails(tmp_path):
    sci = np.full((40, 40), 10.0)
    hp = HotpantsParams(stamp_mode="connected_regions", hp_bgo=-1, write_stamps=False)
    cfg = HP.build_hotpants_config(hp, str(tmp_path / "d"), str(tmp_path / "c"), "s", write_stamps=False,
                                   sci_shape=sci.shape)
    res = HP.run_hotpants_frame(sci, sci, sci, np.zeros((40, 40), dtype=np.int32), np.array([[20.0, 20.0]]), cfg,
                                oversample=1, fit_only_exclude=np.zeros((10, 10), dtype=bool))
    assert not res["success"] and "fit_only_exclude" in res["error_msg"]


def test_lane_star_wing_radii_reads_frozen_background_table(tmp_path):
    (tmp_path / "diff_config.yaml").write_text(yaml.safe_dump({"pipeline": [
        {"kind": "kernel_fit", "tessreduce_star_wing_radii": [[1.0, 2]]},
        {"kind": "background_estimate", "tessreduce_star_wing_radii": TABLE},
    ]}))
    assert lane_star_wing_radii(tmp_path) == [[9.0, 12], [11.0, 8], [13.0, 6]]


def test_lane_star_wing_radii_missing_raises(tmp_path):
    (tmp_path / "diff_config.yaml").write_text(yaml.safe_dump({"pipeline": [{"kind": "background_estimate"}]}))
    with pytest.raises(ValueError, match="tessreduce_star_wing_radii"):
        lane_star_wing_radii(tmp_path)
