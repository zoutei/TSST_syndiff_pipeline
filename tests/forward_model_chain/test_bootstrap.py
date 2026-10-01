"""Fast synthetic tests for forward_model.chain.bootstrap (no /astro access)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.forward_model.chain import bootstrap as bs


def test_robust_stats_ignores_nonfinite_and_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 2.0, 20000)
    x[:50] = 1e6
    x[50:60] = np.nan
    s = bs.robust_stats(x)
    assert s["n"] == 19990
    assert abs(s["median"] - 0.5) < 0.05
    assert abs(s["mad_sigma"] - 2.0) < 0.1
    assert s["std"] > 1000  # std is NOT robust, MAD is


def test_robust_stats_empty():
    assert bs.robust_stats(np.full(3, np.nan))["n"] == 0


def test_hpd_good_mask_accepts_bit32_and_rejects_bad_noise():
    diff = np.ones((2, 3))
    noise = np.array([[1.0, 0.0, 1.0], [np.nan, 1.0, 1.0]])
    mask = np.array([[0, 0, 32], [0, 4, 0]])
    g = bs.hpd_good_mask(diff, noise, mask)
    assert g.tolist() == [[True, False, True], [False, False, True]]


def test_hpd_stats_counts_strict_mask0():
    rng = np.random.default_rng(1)
    noise = np.full((100, 100), 2.0)
    diff = rng.normal(0, 2.0, (100, 100))
    mask = np.zeros((100, 100), int)
    mask[:, :50] = 32
    s = bs.hpd_stats(diff, noise, mask)
    assert s["n"] == 10000 and s["n_strict_mask0"] == 5000
    assert abs(s["mad_sigma"] - 1.0) < 0.05


def test_frame_group_offsets_and_convolved_lookup(tmp_path):
    lane = tmp_path
    (lane / "tmpl_conv").mkdir()
    pd.DataFrame({"ffi_basename": ["tessA-s0001-1-1-0180-s_ffic.fits", "tessB-s0001-1-1-0180-s_ffic.fits"],
                  "group_id": [-1, 2], "group_dx": [np.nan, -0.0], "group_dy": [np.nan, -0.01]}).to_csv(lane / "frames.csv", index=False)
    pd.DataFrame({"group_id": [0, 2], "group_dx": [0.01, 0.0], "group_dy": [-0.02, -0.01],
                  "template_path": ["a", "b"], "convolved_path": ["/x/c0.fits.fz", "/x/c2.fits.fz"]}).to_csv(
        lane / "tmpl_conv" / "convolved_templates.csv", index=False)
    assert bs.frame_group_offsets(lane, "tessB-s0001-1-1") == (0.0, -0.01)
    # preferred source: <scc>/remap_linear/oversampling_1/point_drift_table.csv (sibling of the lane dir)
    lane2 = tmp_path / "scc" / "diff_linear"
    (lane2).mkdir(parents=True)
    d = tmp_path / "scc" / "remap_linear" / "oversampling_1"
    d.mkdir(parents=True)
    pd.DataFrame({"filename": ["tessC-s0001-1-1-0180-s_ffic.fits"], "group_id": [3], "group_dx": [0.02], "group_dy": [-0.03]}).to_csv(
        d / "point_drift_table.csv", index=False)
    assert bs.frame_group_offsets(lane2, "tessC-s0001-1-1") == (0.02, -0.03)
    assert str(bs.convolved_template_for_frame(lane, "tessB-s0001-1-1")) == "/x/c2.fits.fz"
    with pytest.raises(ValueError):
        bs.frame_group_offsets(lane, "tessA-s0001-1-1")  # frame without a group
    with pytest.raises(KeyError):
        bs.frame_group_offsets(lane, "nope")


def test_private_data_root_symlinks_and_idempotent(tmp_path):
    src = tmp_path / "data"
    scc = src / "s0024" / "c2" / "k2"
    for d in ("catalogs", "ffi", "wcs"):
        (scc / d).mkdir(parents=True)
    (scc / "ffi_list.parquet").write_bytes(b"x")
    z = src / "ps1_skycells_zarr"
    for d in ("ps1_skycells.zarr", "ps1_combined.zarr", "ps1_convolved.zarr"):
        (z / d).mkdir(parents=True)
    priv = tmp_path / "priv"
    for _ in range(2):
        bs.make_private_data_root(priv, src, 24, 2, 2, scc / "ffi" / "f.fits")
    assert (priv / "s0024/c2/k2/ffi").is_symlink()
    assert (priv / "s0024/c2/k2/ffi_list.parquet").read_bytes() == b"x"  # copied, not linked
    assert not (priv / "s0024/c2/k2/ffi_list.parquet").is_symlink()
    assert (priv / "ps1_skycells_zarr/ps1_convolved.zarr").is_symlink()
    assert (priv / "bookkeeping").is_dir()


def test_estimate_ks_b_harmonic_vs_biharmonic_smooth_sky():
    """On a smooth synthetic sky with a masked hole the harmonic fill recovers the background to < noise and shape is kept."""
    n = 96
    yy, xx = np.mgrid[:n, :n]
    sky = 0.5 + 0.01 * xx + 0.005 * yy
    rng = np.random.default_rng(3)
    ffi = sky + rng.normal(0, 0.05, sky.shape)
    conv = np.zeros_like(ffi)
    mask = np.zeros((n, n), np.int16)
    mask[30:60, 30:60] = 1
    b = bs.estimate_ks_b(ffi, conv, mask, fill_method="harmonic", boundary_k=5)
    assert b.shape == (n, n)
    inner = np.abs(b - sky)[20:76, 20:76]
    assert np.nanmedian(inner) < 0.1
    with pytest.raises(ValueError):
        bs.estimate_ks_b(ffi, conv[:-1], mask)


def test_hp_recipe_matches_reference_values():
    r = bs.HP_RECIPE
    assert r["hp_ko"] == 2 and r["hp_bgo"] == 0 and r["hp_nss"] == 100
    assert r["hp_sigma_gauss"] == [0.752, 1.88, 3.76]
    assert (r["hp_nstampx"], r["hp_nstampy"]) == (10, 10)
    assert r["write_kernel_solutions"] is True
    assert bs.OVERSAMPLING == 4 and bs.PSF_SIGMA == 40.0


def test_star_mask_pad_forwarded_only_when_nonzero(monkeypatch):
    seen = {}

    def fake(res, mask, **kw):
        seen.update(kw)
        return res * 0, res * 0, res * 0

    import syndiff_pipeline.difference_imaging.stages.background.tessreduce_residual as tr
    monkeypatch.setattr(tr, "estimate_tessreduce_residual_background", fake)
    a = np.zeros((4, 4))
    bs.estimate_ks_b(a, a, np.zeros((4, 4), np.int16))
    assert "star_mask_pad_px" not in seen and seen["fill_method"] == "harmonic"
    bs.estimate_ks_b(a, a, np.zeros((4, 4), np.int16), star_mask_pad_px=3)
    assert seen["star_mask_pad_px"] == 3
    with pytest.raises(ValueError):
        bs.estimate_ks_b(a, a, np.zeros((4, 4), np.int16), star_mask_pad_px=-1)


def test_lane_dir_resolution_and_check(tmp_path):
    class In: lane_dir = None
    class Cfg:
        inputs = In()
        raw = {}
        out_root = tmp_path
    c = Cfg()
    assert bs.lane_dir(c) == tmp_path / "lane_f1"
    assert bs.lane_dir(c, "/x/y") == Path("/x/y")
    c.inputs.lane_dir = tmp_path / "L"
    assert bs.lane_dir(c) == tmp_path / "L"
    lane = tmp_path / "L"
    (lane / "ks_b").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as e:
        bs.check_lane(lane, "stem")
    assert "shared_mask" in str(e.value) and "substamp_stars" in str(e.value) and "ks_b" in str(e.value)
    for f in ("shared_mask.fits.fz", "hotpants_substamp_stars.csv", "ks_b/stem_ks_b.fits.fz"):
        (lane / f).write_bytes(b"")
    assert set(bs.check_lane(lane, "stem")) == {"shared_mask", "substamp_stars", "ks_b"}


def test_template_recipe_uses_chain_band_weights():
    """The D13 weights through bootstrap's recipe path give the dataset store's recipe id (e17a198a), the production
    defaults do not: a weight mix-up can only miss cells, never load a wrong-weight template."""
    from syndiff_pipeline.template_creation.processing.combined_store import combined_recipe_id, production_combined_recipe
    base = {"remove_saturated_stars": True, "enable_saturation_correction": False}
    d13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}
    assert combined_recipe_id(production_combined_recipe({**base, "band_weights": d13})) == "e17a198a4942aa2d"
    assert combined_recipe_id(production_combined_recipe(base)) != "e17a198a4942aa2d"
