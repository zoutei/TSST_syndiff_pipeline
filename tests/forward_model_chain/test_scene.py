import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from syndiff_pipeline.forward_model.chain import config as C
from syndiff_pipeline.forward_model.chain import scene as S

from chain_fixtures import make_hp_d, make_scene, raw_config


def _load(d):
    return dict(np.load(d / "scene_bundle.npz")), json.loads((d / "scene_meta.json").read_text())


def test_swap_replaces_data_noise_only(tmp_path):
    src = tmp_path / "src"
    z0 = make_scene(src)
    hp = make_hp_d(tmp_path / "hp.fits.fz", value=3.0, noise=2.0)
    z, meta = S._load(src)
    n = S.swap_data(z, meta, src, hp)
    assert n == int((z0["finite"] & (z0["data"] != 3.0)).sum())
    assert (z["data"] == 3.0).all() and (z["noise"] == 2.0).all()
    for k in ("role", "valid", "cx", "cy", "source_id"):
        assert np.array_equal(z[k], z0[k])
    assert z["data"].dtype == np.float32 and meta["d14_swap"]["hp_d"].endswith("hp.fits.fz")


def test_swap_keeps_nonfinite_and_out_of_bounds(tmp_path):
    src = tmp_path / "src"
    z0 = make_scene(src)
    fin = z0["finite"].copy()
    fin[0, 0] = False
    z0["finite"] = fin
    np.savez(src / "scene_bundle.npz", **z0)
    hp = make_hp_d(tmp_path / "hp.fits", shape=(40, 40))
    z, meta = S._load(src)
    S.swap_data(z, meta, src, hp)
    assert z["data"][0, 0] == z0["data"][0, 0]                      # not finite -> untouched


def test_swap_rejects_bad_hp(tmp_path):
    src = tmp_path / "src"
    make_scene(src)
    hp = make_hp_d(tmp_path / "hp.fits", noise=0.0)                   # noise must be > 0
    z, meta = S._load(src)
    with pytest.raises(ValueError, match="non-finite"):
        S.swap_data(z, meta, src, hp)


def test_demote(tmp_path):
    src = tmp_path / "src"
    z0 = make_scene(src)
    csv = tmp_path / "drop.csv"
    pd.DataFrame({"source_id": [1001, 1002]}).to_csv(csv, index=False)
    z, meta = S._load(src)
    assert S.demote(z, meta, csv) == 2
    assert z["role"].tolist() == [0, 2, 2, 1, 1, 2] and z["role"].dtype == z0["role"].dtype
    d = meta["demoted"]
    assert (d["n_listed"], d["n_matched_in_scene"], d["n_listed_not_in_scene"], d["strict"]) == (2, 2, 0, False)
    import hashlib
    assert d["csv_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert meta["n_roles"] == {"contrib": 1, "anchor": 2, "nuisance": 3}
    assert meta["demoted"]["roles_before"] == [1, 4, 1] and meta["demoted"]["roles_after"] == [1, 2, 3]
    # list larger than the scene: unmatched ids are counted, not an error (default)
    pd.DataFrame({"source_id": [1001, 99999, 88888, 99999]}).to_csv(csv, index=False)
    z, meta = S._load(src)
    assert S.demote(z, meta, csv) == 1
    d = meta["demoted"]
    assert (d["n_listed"], d["n_matched_in_scene"], d["n_listed_not_in_scene"]) == (3, 1, 2)
    assert z["role"].tolist() == [0, 2, 1, 1, 1, 2]
    # strict mode = the e2e assert
    with pytest.raises(ValueError, match="1 of 4"):
        S.demote(*S._load(src), csv, strict=True)


def _strap_workspace(tmp_path, cols):
    ws = tmp_path / "ws"
    ws.mkdir()
    m = np.zeros((40, 40), np.int16)
    m[:, cols] = 4 | 1
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(m)]).writeto(ws / "shared_mask.fits.fz")
    return ws


def test_strap_mask_marks_valid_and_demotes(tmp_path):
    ws = _strap_workspace(tmp_path, [8])        # column x=8: centre column of stars 0 and 4 (cx=8), stamp 5
    src = tmp_path / "src"
    make_scene(src, workspace=ws)
    z, meta = S._load(src)
    summ = S.mask_straps(z, meta, src_dir=src)
    # stamps with cx=8 cover columns 6..10: 5 px per row masked
    v = z["valid"].reshape(6, 5, 5)
    assert not v[0][:, 2].any() and v[0][:, [0, 1, 3, 4]].all()
    assert v[1].all()                                                    # cx=14: untouched (cols 12..16)
    assert summ["n_stamp_px_masked"] == 2 * 5 and summ["n_demoted"] == 2
    assert meta["straps_masked"] and 4 in meta["masked_bits"]
    # core_radius 1 -> 5 core px of which the centre column holds 3 -> 2 valid < min_core_valid(3): demoted
    assert z["role"][0] == 2 and z["role"][4] == 2 and z["role"][1] == 1
    assert meta["n_roles"]["nuisance"] == 3
    assert sorted(meta["strap_mask"]["demoted_source_ids"]) == [1000, 1004]


def test_strap_mask_noop_when_already_masked(tmp_path):
    src = tmp_path / "src"
    make_scene(src, workspace=tmp_path / "does_not_exist")
    z, meta = S._load(src)
    meta["straps_masked"] = True
    z0 = {k: v.copy() for k, v in z.items()}
    out = S.mask_straps(z, meta, src_dir=src)
    assert "skipped" in out and all(np.array_equal(z[k], z0[k]) for k in z)


def test_strap_mask_needs_shared_mask_scene(tmp_path):
    src = tmp_path / "src"
    make_scene(src, mask_source="MaskCatalog.mask_at(full)")
    with pytest.raises(ValueError, match="shared mask"):
        S.mask_straps(*S._load(src), src_dir=src)


def test_build_scene_order_and_files(tmp_path):
    src = tmp_path / "src"
    make_scene(src)
    hp = make_hp_d(tmp_path / "hp.fits.fz")
    csv = tmp_path / "drop.csv"
    pd.DataFrame({"source_id": [1003]}).to_csv(csv, index=False)
    out = tmp_path / "out"
    summ = S.build_scene(src, out, hp_d=hp, exclusion_csv=csv)
    assert summ["demoted"]["n_matched_in_scene"] == 1
    z, meta = _load(out)
    assert (z["data"] == 3.0).all() and z["role"][3] == 2 and "strap_mask" not in meta
    assert summ["n_roles"] == meta["n_roles"]
    # no hp_d / csv: a plain copy
    S.build_scene(src, tmp_path / "copy")
    z0, _ = _load(src)
    zc, _ = _load(tmp_path / "copy")
    assert all(np.array_equal(z0[k], zc[k]) for k in z0)


def test_find_hp_d(tmp_path):
    with pytest.raises(C.ConfigError, match="found 0"):
        S.find_hp_d(tmp_path, "stemA")
    p = make_hp_d(tmp_path / "sub/hp_d/stemA_hp_d.fits.fz")
    assert S.find_hp_d(tmp_path, "stemA") == p
    make_hp_d(tmp_path / "other/stemA_hp_d.fits.fz")
    with pytest.raises(C.ConfigError, match="found 2"):
        S.find_hp_d(tmp_path, "stemA")


def test_run_scene_stage(tmp_path):
    src = tmp_path / "src"
    make_scene(src)
    hp = make_hp_d(tmp_path / "hp.fits.fz")
    cfg = C.config_from_dict(raw_config(tmp_path, inputs={"colour_file": str(tmp_path / "c.csv"), "source_scene": str(src),
                                                         "bootstrap_hp_d": str(hp)}))
    st = S.run_scene(cfg, "boot")
    assert C.is_done(st) and st == cfg.stage_dir("scene_boot")
    prov = json.loads((st / "provenance.json").read_text())
    assert prov["inputs"]["hp_d"]["sha256"]
    t = (st / "DONE").stat().st_mtime_ns
    S.run_scene(cfg, "boot")                                              # idempotent
    assert (st / "DONE").stat().st_mtime_ns == t
    # final: hp_d must be discoverable or given
    with pytest.raises(C.ConfigError):
        S.run_scene(cfg, "final")
    S.run_scene(cfg, "final", hp_d=hp)
    assert C.is_done(cfg.stage_dir("scene_final"))


def test_run_scene_requires_source(tmp_path):
    cfg = C.config_from_dict(raw_config(tmp_path))
    with pytest.raises(C.ConfigError, match="source_scene"):
        S.run_scene(cfg, "boot")


def _toy_scene(n=3, S=5):
    import numpy as np
    z = {"stamp": np.int32(S), "data": np.ones((n, S * S), np.float32), "noise": np.ones((n, S * S), np.float32),
         "valid": np.ones((n, S * S), bool), "role": np.zeros(n, np.int8), "source_id": np.arange(n, dtype=np.int64)}
    z["uid"] = np.arange(n * S * S, dtype=np.int32).reshape(n, S * S)
    z["uid"][1, 0] = z["uid"][0, 0]           # stamps 0 and 1 share union pixel 0
    meta = {"core_radius": 1.0, "min_core_valid": 5}
    return z, meta


def test_negative_outlier_mask_masks_shared_pixels_and_demotes_on_core():
    import numpy as np
    from syndiff_pipeline.forward_model.chain import scene as SC
    z, meta = _toy_scene()
    z["data"][0, 0] = -1000.0                  # corner pixel of stamp 0 (shared with stamp 1): mask, no demotion
    z["data"][2, 12] = -1000.0                 # centre of stamp 2: masked -> centre invalid -> role 2
    s = SC.mask_negative_outliers(z, meta, nsigma=300)
    assert not z["valid"][0, 0] and not z["valid"][1, 0] and not z["valid"][2, 12]
    assert z["valid"].sum() == 3 * 25 - 3 and list(z["role"]) == [0, 0, 2]
    assert s["n_union_px_masked"] == 2 and s["n_demoted"] == 1 and meta["n_roles"]["nuisance"] == 1
    z2, meta2 = _toy_scene()
    z2["data"][0, 3] = -200.0                  # above the threshold: untouched
    assert SC.mask_negative_outliers(z2, meta2)["n_stamp_px_masked"] == 0 and z2["valid"].all()


def test_guard_refuses_broken_swapped_data():
    import numpy as np
    import pytest
    from syndiff_pipeline.forward_model.chain import scene as SC
    z, _ = _toy_scene()
    SC.guard_swapped_data(z, "x")
    z["data"][1, 4] = 1e10
    with pytest.raises(ValueError, match="refusing"):
        SC.guard_swapped_data(z, "x")
    z["data"][1, 4] = np.nan
    with pytest.raises(ValueError):
        SC.guard_swapped_data(z, "x")
