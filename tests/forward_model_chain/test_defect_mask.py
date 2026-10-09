"""inputs.defect_mask (saturation bleed / bad columns): Hotpants reference mask bit, final match rejection, and
scene_final masking (the bootstrap scene is left alone)."""
import json

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.forward_model.chain import _tk as TK
from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain import hotpants_ref as H
from syndiff_pipeline.forward_model.chain import scene as S

from chain_fixtures import make_hp_d, make_scene, raw_config


def _defect_file(path, shape=(40, 40), cols=(8,)):
    d = np.zeros(shape, np.uint8)
    d[:, list(cols)] = 2                                  # bit 2 = bad column (rollout convention)
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(d)]).writeto(path)
    return path


def test_config_parses_defect_mask(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "defect_mask": str(tmp_path / "d.fits.fz")}))
    assert cfg.inputs.defect_mask == tmp_path / "d.fits.fz"
    assert C.config_from_dict(raw_config(tmp_path)).inputs.defect_mask is None


def test_defect_bit_is_set_and_rejected_by_the_match(tmp_path):
    p = _defect_file(tmp_path / "d.fits.fz", cols=(3,))
    cfg = C.config_from_dict(raw_config(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "defect_mask": str(p)}))
    defect = H.load_defect_mask(cfg, (40, 40))
    assert defect.sum() == 40 and defect[:, 3].all()
    m0 = np.zeros((40, 40), np.int32)
    m0[0, 0] = 64                                         # FLAG_OK_CONV stays acceptable
    m0[0, 3] = 64
    m = H.apply_defect_bit(m0, defect)
    assert m.dtype == np.int32 and (m[:, 3] & H.DEFECT_BIT).all() and m[0, 0] == 64
    assert H.DEFECT_BIT > 0x8000                          # above every pyhotpants flag
    good, _ = TK.selection(np.ones((40, 40)), m, pd.DataFrame({"x": [], "y": [], "tmag": []}))
    assert not good[:, 3].any() and good[:, 4].all() and good[0, 0]
    assert H.apply_defect_bit(m0, None) is m0
    assert H.load_defect_mask(C.config_from_dict(raw_config(tmp_path)), (40, 40)) is None


def test_mask_defects_marks_valid_and_demotes(tmp_path):
    src = tmp_path / "src"
    make_scene(src)
    z, meta = S._load(src)
    summ = S.mask_defects(z, meta, _defect_file(tmp_path / "d.fits.fz"))
    v = z["valid"].reshape(6, 5, 5)
    assert not v[0][:, 2].any() and v[0][:, [0, 1, 3, 4]].all() and v[1].all()
    assert summ["n_stamp_px_masked"] == 2 * 5 and summ["n_demoted"] == 2 and summ["n_defect_px"] == 40
    assert sorted(meta["defect_mask"]["demoted_source_ids"]) == [1000, 1004] and meta["defect_mask"]["sha256"]
    assert "straps_masked" not in meta or not meta["straps_masked"]


def test_strap_and_defect_masks_agree_on_the_same_columns(tmp_path):
    """mask_straps after the refactor = mask_defects on the same pixel map (shared numerics)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    m = np.zeros((40, 40), np.int16)
    m[:, 8] = 4
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(m)]).writeto(ws / "shared_mask.fits.fz")
    make_scene(tmp_path / "a", workspace=ws)
    za, ma = S._load(tmp_path / "a")
    zb, mb = S._load(tmp_path / "a")
    S.mask_straps(za, ma, src_dir=tmp_path / "a")
    S.mask_defects(zb, mb, _defect_file(tmp_path / "d.fits.fz"))
    assert np.array_equal(za["valid"], zb["valid"]) and np.array_equal(za["role"], zb["role"])
    assert ma["n_roles"] == mb["n_roles"]


def test_run_scene_applies_defect_mask_to_final_only(tmp_path):
    src = tmp_path / "src"
    make_scene(src)
    hp = make_hp_d(tmp_path / "hp.fits.fz")
    p = _defect_file(tmp_path / "d.fits.fz")
    cfg = C.config_from_dict(raw_config(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "source_scene": str(src),
                                                         "bootstrap_hp_d": str(hp), "defect_mask": str(p)}))
    b = S.run_scene(cfg, "boot")
    f = S.run_scene(cfg, "final", hp_d=hp)
    mb = json.loads((b / "scene_meta.json").read_text())
    mf = json.loads((f / "scene_meta.json").read_text())
    assert "defect_mask" not in mb and mf["defect_mask"]["n_stamp_px_masked"] == 10
    assert json.loads((f / "provenance.json").read_text())["inputs"]["defect_mask"]["sha256"]
    zb = np.load(b / "scene_bundle.npz")
    zf = np.load(f / "scene_bundle.npz")
    assert zb["valid"].sum() - zf["valid"].sum() == 10


def test_mask_defects_demotes_only_stars_it_touched(tmp_path):
    """A star that already fails the core rule but whose stamp the defect map misses keeps its role (run3_rf5 bug:
    313 untouched F1 stars were demoted when the defect step re-applied the rule to every star)."""
    src = tmp_path / "src"
    make_scene(src)
    z, meta = S._load(src)
    v = z["valid"].copy()
    v[1, :] = False                                       # star 1 (cx=14) already fails the rule, role 1
    z["valid"] = v
    role1 = int(z["role"][1])
    assert role1 != 2
    summ = S.mask_defects(z, meta, _defect_file(tmp_path / "d.fits.fz"))   # column 8: stars 0 and 4 only
    assert z["role"][1] == role1 and summ["n_demoted"] == 2
    assert sorted(meta["defect_mask"]["demoted_source_ids"]) == [1000, 1004]
    z2, meta2 = S._load(src)
    z2["valid"] = v.copy()
    S.mask_defects(z2, meta2, _defect_file(tmp_path / "e.fits.fz", cols=()))   # empty map: nothing changes
    assert z2["role"][1] == role1 and meta2["defect_mask"]["n_demoted"] == 0
