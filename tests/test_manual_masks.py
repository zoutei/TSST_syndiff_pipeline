"""Manual extra-mask file (masking/manual.py) and its wiring into the shared mask."""
from __future__ import annotations

import os
import shutil
import stat

import numpy as np
import pandas as pd
import pytest
import yaml

from syndiff_pipeline.difference_imaging.masking import bits
from syndiff_pipeline.difference_imaging.masking.api import generate_shared_mask_catalog
from syndiff_pipeline.difference_imaging.masking.manual import (
    apply_manual_masks,
    load_manual_regions,
    parse_region,
    rasterize_region,
)
from syndiff_pipeline.difference_imaging.masking.settings import (
    AsteroidMaskSettings,
    MaskSettings,
    SharedMaskSettings,
    TnsMaskSettings,
    mask_settings_from_dict,
    mask_settings_to_dict,
)
from syndiff_pipeline.difference_imaging.masking.shared import build_static_mask

CROP = {"x_min": 100, "x_max": 180, "y_min": 50, "y_max": 110, "shape": (60, 80)}


def _write(tmp_path, doc, name="manual_masks.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc))
    return p


DOC = {
    "version": 1,
    "masks": [
        {"sector": 24, "camera": 2, "ccd": 2, "regions": [
            {"kind": "column", "x": 110, "reason": "bad col"},
            {"kind": "rect", "x": [120, 125], "y": [60, 70], "bit": "sat_cross", "reason": "bleed"},
            {"kind": "circle", "x": 150.0, "y": 80.0, "r": 3, "reason": "spot"},
        ]},
        {"sector": 25, "camera": 1, "ccd": 1, "regions": [{"kind": "column", "x": 5, "reason": "other"}]},
    ],
}


def test_parse_filters_scc_and_defaults(tmp_path):
    p = _write(tmp_path, DOC)
    r = load_manual_regions(p, 24, 2, 2)
    assert [x["kind"] for x in r] == ["column", "rect", "circle"]
    assert [x["bit_value"] for x in r] == [bits.EDGE, bits.SAT_CROSS, bits.EDGE]
    assert load_manual_regions(p, 24, 2, 3) == []
    assert len(load_manual_regions(p, 25, 1, 1)) == 1


@pytest.mark.parametrize("bad,msg", [
    ({"kind": "blob", "reason": "r", "x": 1}, "unknown kind"),
    ({"kind": "column", "x": 3}, "reason"),
    ({"kind": "column", "x": 3, "reason": "r", "foo": 1}, "unknown keys"),
    ({"kind": "rect", "x": [5, 5], "y": [0, 3], "reason": "r"}, "max > min"),
    ({"kind": "rect", "x": [5, 8], "reason": "r"}, "missing 'y'"),
    ({"kind": "circle", "x": 1, "y": 1, "r": 0, "reason": "r"}, "positive"),
    ({"kind": "column", "x": 3.5, "reason": "r"}, "integer"),
    ({"kind": "column", "x": 3, "bit": "strap", "reason": "r"}, "bit must be"),
])
def test_validation_errors_name_entry(bad, msg, tmp_path):
    doc = {"version": 1, "masks": [{"sector": 24, "camera": 2, "ccd": 2, "regions": [bad]}]}
    with pytest.raises(ValueError, match=msg) as e:
        load_manual_regions(_write(tmp_path, doc), 24, 2, 2)
    assert "masks[0].regions[0]" in str(e.value)


def test_file_level_validation(tmp_path):
    with pytest.raises(ValueError, match="version"):
        load_manual_regions(_write(tmp_path, {"version": 2, "masks": []}), 24, 2, 2)
    with pytest.raises(ValueError, match="unknown keys"):
        load_manual_regions(_write(tmp_path, {"version": 1, "masks": [
            {"sector": 24, "camera": 2, "ccd": 2, "regions": [], "extra": 1}]}), 24, 2, 2)


def test_rasterize_crop_offset_and_clipping():
    col = parse_region({"kind": "column", "x": 110, "reason": "r"}, "w")
    m = rasterize_region((60, 80), CROP, col)
    assert m[:, 10].all() and m.sum() == 60  # crop-local x = 110 - 100
    clipped = parse_region({"kind": "rect", "x": [95, 103], "y": [40, 53], "reason": "r"}, "w")
    m = rasterize_region((60, 80), CROP, clipped)
    assert m.sum() == 3 * 3 and m[:3, :3].all()  # full-FFI x 100..102, y 50..52
    outside = parse_region({"kind": "rect", "x": [0, 10], "y": [0, 10], "reason": "r"}, "w")
    assert not rasterize_region((60, 80), CROP, outside).any()
    ycol = parse_region({"kind": "column", "x": 111, "y": [60, 70], "reason": "r"}, "w")
    m = rasterize_region((60, 80), CROP, ycol)
    assert m[:, 11].sum() == 10 and m[10:20, 11].all()
    circ = parse_region({"kind": "circle", "x": 150.0, "y": 80.0, "r": 3, "reason": "r"}, "w")
    m = rasterize_region((60, 80), CROP, circ)
    assert m[30, 50] and m[30, 53] and not m[30, 54] and not m[33 + 1, 50]
    edge_circ = parse_region({"kind": "circle", "x": 100.0, "y": 50.0, "r": 4, "reason": "r"}, "w")
    assert rasterize_region((60, 80), CROP, edge_circ).sum() > 0


def test_apply_bits_and_noop():
    base = np.full((60, 80), bits.STRAP, dtype=np.int16)
    assert apply_manual_masks(base, CROP, None) is base and apply_manual_masks(base, CROP, []) is base
    regs = [parse_region(r, "w") for r in DOC["masks"][0]["regions"]]
    out = apply_manual_masks(base, CROP, regs)
    assert out.dtype == np.int16 and (base == bits.STRAP).all()  # input untouched
    assert out[0, 10] == bits.STRAP | bits.EDGE
    assert out[15, 22] == bits.STRAP | bits.SAT_CROSS
    assert out[5, 5] == bits.STRAP


def _settings(**kw):
    return MaskSettings(
        shared=SharedMaskSettings(style=kw.pop("style", "empirical"), include_straps=False, include_edges=False,
                                  ps1_min_hit_count=0),
        tns=TnsMaskSettings(enabled=False), asteroids=AsteroidMaskSettings(enabled=False), **kw)


@pytest.mark.parametrize("style", ["empirical", "tessreduce"])
def test_static_mask_identical_without_regions_and_ored_with(style, tmp_path):
    image = np.zeros((60, 80))
    gaia = pd.DataFrame({"x": [20.0], "y": [20.0], "mag": [11.0]})
    kw = dict(straps_csv="/nonexistent/straps.csv")
    base = build_static_mask(image, gaia, CROP, settings=_settings(style=style), **kw)
    same = build_static_mask(image, gaia, CROP, settings=_settings(style=style, manual_masks=[]), **kw)
    assert np.array_equal(base, same)
    regs = [parse_region(r, "w") for r in DOC["masks"][0]["regions"]]
    with_m = build_static_mask(image, gaia, CROP, settings=_settings(style=style, manual_masks=regs), **kw)
    assert np.array_equal(with_m, apply_manual_masks(base, CROP, regs))
    assert (with_m[:, 10] & bits.EDGE).all()


def test_settings_dict_and_key_roundtrip():
    d0 = mask_settings_to_dict(MaskSettings())
    assert "manual_mask_file" not in d0 and "manual_masks" not in d0
    s = mask_settings_from_dict({"manual_mask_file": "x.yaml"})
    assert s.manual_mask_file == "x.yaml" and mask_settings_to_dict(s)["manual_mask_file"] == "x.yaml"
    regs = [parse_region(r, "w") for r in DOC["masks"][0]["regions"]]
    d1 = mask_settings_to_dict(MaskSettings(manual_masks=regs))
    assert d1["manual_masks"][0]["kind"] == "column" and "bit_value" not in str(d1)


@pytest.mark.skipif(shutil.which("fpack") is None, reason="fpack (syndiff env) required to write shared_mask")
def test_shared_mask_stage_end_to_end_freezes_copy(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    _write(site, DOC)
    image = np.zeros((60, 80))
    gaia = pd.DataFrame({"x": [20.0], "y": [20.0], "mag": [11.0], "tess_mag": [11.0]})

    def run(lane, ccd):
        return generate_shared_mask_catalog(
            ref_image=image, gaia_df=gaia, crop_bounds=CROP, lane_root=lane, data_root=tmp_path, sector=24,
            camera=2, ccd=ccd, straps_csv="/nonexistent/straps.csv", settings=_settings(), site_dir=site)

    lane = tmp_path / "lane_a"
    cat = run(lane, 2)
    assert (cat.static[:, 10] & bits.EDGE).all()
    frozen = lane / "manual_masks.yaml"
    assert frozen.is_file()
    assert stat.S_IMODE(frozen.stat().st_mode) == 0o444
    fz = load_manual_regions(frozen, 24, 2, 2)
    assert len(fz) == 3  # round-trips and only holds this SCC
    assert yaml.safe_load(frozen.read_text())["masks"].__len__() == 1
    run(lane, 2)  # re-run over an existing 444 file works
    # other SCC: identical to no file, nothing frozen
    lane_b = tmp_path / "lane_b"
    other = run(lane_b, 3)
    ref = generate_shared_mask_catalog(
        ref_image=image, gaia_df=gaia, crop_bounds=CROP, lane_root=tmp_path / "lane_c", data_root=tmp_path,
        sector=24, camera=2, ccd=3, straps_csv="/nonexistent/straps.csv", settings=_settings())
    assert np.array_equal(other.static, ref.static) and not (lane_b / "manual_masks.yaml").exists()
