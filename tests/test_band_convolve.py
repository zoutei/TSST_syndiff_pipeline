"""Tests for template_creation.processing.band_convolve (convolve at downsample, from band cells).

Uses an injected in-memory loader (no on-disk store) and a tiny synthetic projection:
two rows (2 cells + 1 cell), cell 520 x 520 so the 480-px overlaps/pads of the canonical cell apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing import band_convolve as bc
from syndiff_pipeline.template_creation.processing import canonical_cell as cc
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import perband

W = 520
PROJ = "9999"
NAMES = {"A": f"skycell.{PROJ}.000", "B": f"skycell.{PROJ}.001", "C": f"skycell.{PROJ}.010"}


def _df():
    rows = [
        (NAMES["A"], 0, 0), (NAMES["B"], 1, 0), (NAMES["C"], 0, 1),
    ]
    return pd.DataFrame(
        {"NAME": [r[0] for r in rows], "projection": [int(PROJ)] * 3, "x": [r[1] for r in rows],
         "y": [r[2] for r in rows], "NAXIS1": W, "NAXIS2": W}
    ).set_index("NAME", drop=False)


def _band_cells(seed):
    rng = np.random.default_rng(seed)
    out = {}
    for b in perband.BANDS:
        img = rng.normal(0.0, 1.0, (W, W)).astype(np.float32)
        img[rng.random((W, W)) < 0.6] = 0.0  # Z zeroing
        out[b] = img
    out["r"][5:7, 9:11] = np.nan
    nan = np.zeros((W, W), bool)
    for v in out.values():
        nan |= np.isnan(v)
    return {b: np.where(nan, np.float32(np.nan), v) for b, v in out.items()}


@pytest.fixture(scope="module")
def store():
    cells = {n: _band_cells(i) for i, n in enumerate(NAMES.values())}
    masks = {n: np.full((W, W), 3 * i, dtype=np.uint16) for i, n in enumerate(NAMES.values())}
    calls = {"n": 0}

    def loader(data_root, projection, cell, fp):
        name = f"{projection}.{cell}"
        calls["n"] += 1
        if name not in cells:
            return None
        return {"band_images": cells[name], "combined_mask": masks[name],
                "removal_band_weights": dict(REMOVAL)}

    return cells, masks, loader, calls


REMOVAL = {"r": 0.25, "i": 0.25, "z": 0.25, "y": 0.25}
D13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}
RECIPE = cs.combined_recipe(band_weights=REMOVAL, gaia_version="none",
                            remove_saturated_stars=False, bright_star_removal="none")


def _source(store, **kw):
    cells, masks, loader, calls = store
    return bc.BandCellPlaneSource("/nonexistent", RECIPE, loader=loader, **kw)


def _manual_plane(cells, name, w=None, factors=None):
    """float64 sum_b (w_b / f_{s,b}) Z F_b, the explicit formula."""
    w = REMOVAL if w is None else w
    out = np.zeros((W, W))
    for b in perband.BANDS:
        out += (w[b] / (1.0 if factors is None else factors[name][b])) * cells[name][b].astype(np.float64)
    return out


def test_equals_canonical_cell_of_summed_planes(store):
    cells = store[0]
    df = _df()
    name = NAMES["A"]
    got = bc.load_padding_aware_convolved_cell_from_band_cells(
        "/nonexistent", name, skycell_df=df, psf_sigma=40.0, combined_recipe=RECIPE, source=_source(store))
    assert got is not None
    img, mask = got
    md = cc.metadata_for_cell(df, name)
    ref = cc.canonical_cell_image(name, md, lambda n: None if n not in cells else _manual_plane(cells, n),
                                  40.0, 470, window_dtype=np.float64)
    assert img.dtype == np.float32 and mask.dtype == np.uint16
    np.testing.assert_array_equal(np.isnan(img), np.isnan(ref))
    assert np.array_equal(img, ref, equal_nan=True)  # no seam, same code, same inputs -> identical
    assert np.all(mask == 0)  # own mask (cell A index 0)


def test_blur_of_sum_equals_sum_of_band_blurs(store):
    """Linearity: one blur of the summed plane == sum_b of per-band blurs (1e-6 of peak)."""
    cells = store[0]
    df = _df()
    name = NAMES["C"]
    img, _ = bc.load_padding_aware_convolved_cell_from_band_cells(
        "/nonexistent", name, skycell_df=df, psf_sigma=40.0, combined_recipe=RECIPE, source=_source(store))
    md = cc.metadata_for_cell(df, name)
    # the store is unweighted: weight each band cell (removal weights) before the per-band blur
    scaled = {n: {b: cells[n][b].astype(np.float64) * REMOVAL[b] for b in perband.BANDS} for n in cells}
    per_band = perband.convolve_band_cells(name, md, lambda n: scaled.get(n), 40.0, 470)
    s = sum(per_band[b].astype(np.float64) for b in perband.BANDS)
    fin = np.isfinite(s) & np.isfinite(img)
    assert fin.any()
    assert np.abs(s[fin] - img[fin]).max() / np.abs(img[fin]).max() < 1e-6


def _factors(seed=11):
    rng = np.random.default_rng(seed)
    return {n: {b: float(rng.uniform(0.9, 1.1)) for b in perband.BANDS} for n in NAMES.values()}


@pytest.mark.parametrize("name", ["A", "C"])
def test_blur_of_weighted_factored_sum_equals_manual_float64_formula(store, name):
    """Template weights w_b and per-cell factors f_{s,b} act BEFORE the blur: the loader's image
    equals the blur of the manual float64 plane, and the linearity formula
    sum_b blur(Z F_b (w_b / f_{s,b})) agrees to 1e-6 of peak."""
    cells = store[0]
    df = _df()
    nm = NAMES[name]
    fac = _factors()
    got = bc.load_padding_aware_convolved_cell_from_band_cells(
        "/nonexistent", nm, skycell_df=df, psf_sigma=40.0, combined_recipe=RECIPE,
        source=_source(store, weights=D13, band_factors=fac))
    img, _ = got
    md = cc.metadata_for_cell(df, nm)
    ref = cc.canonical_cell_image(nm, md, lambda n: None if n not in cells else _manual_plane(cells, n, D13, fac),
                                  40.0, 470, window_dtype=np.float64)
    assert np.array_equal(img, ref, equal_nan=True)
    scaled = {n: {b: cells[n][b].astype(np.float64) * (D13[b] / fac[n][b]) for b in perband.BANDS} for n in cells}
    per_band = perband.convolve_band_cells(nm, md, lambda n: scaled.get(n), 40.0, 470)
    s = sum(per_band[b].astype(np.float64) for b in perband.BANDS)
    fin = np.isfinite(s) & np.isfinite(img)
    assert fin.any() and np.abs(s[fin] - img[fin]).max() / np.abs(img[fin]).max() < 1e-6


def test_template_weights_change_the_plane_default_is_removal(store):
    cells = store[0]
    a = _source(store).image(NAMES["A"])
    b = _source(store, weights=D13).image(NAMES["A"])
    assert a.dtype == np.float64
    np.testing.assert_array_equal(a, _manual_plane(cells, NAMES["A"]))  # default = recorded removal weights
    assert not np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-3)
    np.testing.assert_array_equal(b, _manual_plane(cells, NAMES["A"], D13))


def test_factors_strict_and_each_cell_uses_its_own(store):
    fac = _factors()
    src = _source(store, band_factors={k: v for k, v in fac.items() if k != NAMES["B"]})
    src.get(NAMES["A"])
    with pytest.raises(ValueError, match="no silent f=1"):
        src.get(NAMES["B"])
    cells = store[0]
    full = _source(store, band_factors=fac)
    np.testing.assert_array_equal(full.image(NAMES["B"]), _manual_plane(cells, NAMES["B"], None, fac))
    np.testing.assert_array_equal(full.image(NAMES["A"]), _manual_plane(cells, NAMES["A"], None, fac))


def test_zpcell_validation_errors(tmp_path):
    good = {n: {b: 1.01 for b in perband.BANDS} for n in NAMES.values()}
    assert bc.validate_band_factors(good, cells=list(NAMES.values()))
    with pytest.raises(ValueError, match="lacks"):
        bc.validate_band_factors(good, cells=list(NAMES.values()) + ["skycell.9999.099"])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bc.validate_band_factors({"x": {**good[NAMES["A"]], "z": bad}})
    with pytest.raises(ValueError, match="every band"):
        bc.validate_band_factors({"x": {"r": 1.0, "i": 1.0}})
    import json

    p = tmp_path / "f.json"
    p.write_text(json.dumps(good))
    assert bc.load_band_factors_json(p, cells=list(NAMES.values())) == good
    p.write_text(json.dumps({"bands": list(perband.BANDS), "zpcell": good}))
    assert bc.load_band_factors_json(p) == good
    p.write_text(json.dumps({"bands": ["r", "i", "z"], "zpcell": good}))
    with pytest.raises(ValueError, match="bands"):
        bc.load_band_factors_json(p)


def test_missing_neighbour_gives_none(store):
    cells, masks, loader, calls = store

    def partial(data_root, projection, cell, fp):
        return None if cell == "001" else loader(data_root, projection, cell, fp)

    src = bc.BandCellPlaneSource("/nonexistent", RECIPE, loader=partial)
    assert bc.load_padding_aware_convolved_cell_from_band_cells(
        "/nonexistent", NAMES["A"], skycell_df=_df(), psf_sigma=40.0, combined_recipe=RECIPE, source=src) is None
    assert src.get("skycell.9999.000") is not None


def test_lru_cache_reuses_planes(store):
    src = _source(store)
    src.get(NAMES["A"]); src.get(NAMES["A"]); src.get(NAMES["B"])
    assert src.n_loads == 2


def test_downsample_flag_defaults_off_and_fingerprint_unchanged():
    from syndiff_pipeline.template_creation.orchestration.stage_params import (
        DOWNSAMPLE_ALLOWED, DownsampleStageParams)

    ds = DownsampleStageParams()
    assert ds.convolve_from_band_cells is False
    assert "convolve_from_band_cells" in DOWNSAMPLE_ALLOWED
    assert ds.band_weights is None and ds.zpcell_factors_path is None
    assert {"band_weights", "zpcell_factors_path"} <= DOWNSAMPLE_ALLOWED


PINNED_DOWNSAMPLE_CONFIG_FP = "c5681f5a6c349952"  # computed on the unmodified HEAD 0497b866
PINNED_DOWNSAMPLE_RECIPE_KEYS = {
    "apply_inter_skycell", "apply_intra_skycell", "ignore_mask_bits", "mapping_grid", "output_base",
    "output_store_name", "oversampling_factor", "remap_store_name", "single_offset"}


def _resolved(**kw):
    from types import SimpleNamespace

    from syndiff_pipeline.template_creation.orchestration.stage_params import (
        DownsampleStageParams, MappingStageParams)

    return SimpleNamespace(
        target=SimpleNamespace(sector=26, camera=3, ccd=1),
        stages=SimpleNamespace(downsample=DownsampleStageParams(**kw), mapping=MappingStageParams()),
        template_output_base="x", downsample_remap_store_name=None)


def test_unset_flags_leave_existing_fingerprints_unchanged():
    from syndiff_pipeline.common.provenance import model
    from syndiff_pipeline.template_creation.orchestration import verify

    r = _resolved()
    assert verify.config_fingerprint(r, "downsample") == PINNED_DOWNSAMPLE_CONFIG_FP
    assert set(model.downsample_recipe_params(r)) == PINNED_DOWNSAMPLE_RECIPE_KEYS
    # band-cell flag alone (no weights/factors) adds exactly the one key it always added
    on = _resolved(convolve_from_band_cells=True)
    assert set(model.downsample_recipe_params(on)) == PINNED_DOWNSAMPLE_RECIPE_KEYS | {"convolve_from_band_cells"}


def test_template_weights_and_factors_enter_downsample_fingerprint(tmp_path):
    import json

    from syndiff_pipeline.common.provenance import model
    from syndiff_pipeline.template_creation.orchestration import verify

    base = _resolved(convolve_from_band_cells=True)
    w = _resolved(convolve_from_band_cells=True, band_weights=D13)
    w2 = _resolved(convolve_from_band_cells=True, band_weights={**D13, "r": 0.255})
    fp = lambda r: verify.config_fingerprint(r, "downsample")  # noqa: E731
    assert len({fp(base), fp(w), fp(w2)}) == 3
    assert model.downsample_recipe_params(w)["band_weights"] == D13
    f1 = tmp_path / "f1.json"
    f1.write_text(json.dumps({"x": {b: 1.0 for b in perband.BANDS}}))
    z1 = _resolved(convolve_from_band_cells=True, zpcell_factors_path=str(f1))
    fz1 = fp(z1)
    assert fz1 != fp(base) and model.downsample_recipe_params(z1)["zpcell_factors"]["sha256"] != "unreadable"
    f1.write_text(json.dumps({"x": {b: 1.1 for b in perband.BANDS}}))  # content change -> new fingerprint
    assert fp(z1) != fz1


def test_weights_and_factors_need_the_band_cell_flag_and_valid_weights():
    with pytest.raises(ValueError, match="convolve_from_band_cells"):
        _resolved(band_weights=D13)
    with pytest.raises(ValueError, match="convolve_from_band_cells"):
        _resolved(zpcell_factors_path="f.json")
    with pytest.raises(ValueError):
        _resolved(convolve_from_band_cells=True, band_weights={"r": 1.0})
    with pytest.raises(ValueError):
        _resolved(convolve_from_band_cells=True, band_weights={**D13, "y": -0.1})


def test_recipe_params_only_record_flag_when_on():
    from types import SimpleNamespace

    from syndiff_pipeline.common.provenance import model
    from syndiff_pipeline.template_creation.orchestration.stage_params import (
        DownsampleStageParams, MappingStageParams)

    mp = MappingStageParams()

    def params(flag):
        ds = DownsampleStageParams(convolve_from_band_cells=flag)
        resolved = SimpleNamespace(stages=SimpleNamespace(downsample=ds, mapping=mp),
                                   template_output_base="x", downsample_remap_store_name=None)
        return model.downsample_recipe_params(resolved)

    off, on = params(False), params(True)
    assert "convolve_from_band_cells" not in off
    assert on.pop("convolve_from_band_cells") is True
    assert on == off


def test_loader_switch_dispatches_to_band_cells(monkeypatch):
    from syndiff_pipeline.template_creation.processing import linear_downsample as LD

    seen = {}

    def fake(data_root, skycell, **kw):
        seen.update(data_root=data_root, skycell=skycell, **kw)
        return "SENTINEL"

    monkeypatch.setattr(bc, "load_padding_aware_convolved_cell_from_band_cells", fake)
    df = _df()
    got = LD._load_ps1_skycell(
        NAMES["A"], data_root="/r", shared_convolved_store=True, legacy_zarr_path=None, zstore_cache={},
        skycell_df=df, psf_sigma=40.0, combined_recipe=RECIPE, convolve_from_band_cells=True)
    assert got == "SENTINEL" and seen["skycell"] == NAMES["A"] and seen["combined_recipe"] is RECIPE
    with pytest.raises(ValueError):
        LD._load_ps1_skycell(
            NAMES["A"], data_root="/r", shared_convolved_store=True, legacy_zarr_path=None, zstore_cache={},
            skycell_df=df, psf_sigma=40.0, combined_recipe=None, convolve_from_band_cells=True)


# ---------------------------------------------------------------- recorded-absent neighbours / seam sources
def _partial_source(store, data_root, missing="001"):
    cells, masks, loader, calls = store

    def partial(dr, projection, cell, fp):
        return None if cell == missing else loader(dr, projection, cell, fp)

    return bc.BandCellPlaneSource(data_root, RECIPE, loader=partial)


def test_recorded_absent_neighbour_stays_nan_like_the_old_path(store, tmp_path):
    from syndiff_pipeline.template_creation.processing import band_cell_store as bcs

    cells = store[0]
    df, name = _df(), NAMES["A"]
    bcs.write_absent_cell(tmp_path, NAMES["B"], {})  # PS1 has no data for cell B (A's right neighbour)
    got = bc.load_padding_aware_convolved_cell_from_band_cells(
        tmp_path, name, skycell_df=df, psf_sigma=40.0, combined_recipe=RECIPE,
        source=_partial_source(store, tmp_path))
    assert got is not None
    img, mask = got
    md = cc.metadata_for_cell(df, name)
    ref = cc.canonical_cell_image(
        name, md, lambda n: None if n in (NAMES["B"],) or n not in cells else _manual_plane(cells, n),
        40.0, 470, window_dtype=np.float64)
    assert np.array_equal(img, ref, equal_nan=True)
    full = cc.canonical_cell_image(name, md, lambda n: _manual_plane(cells, n) if n in cells else None,
                                   40.0, 470, window_dtype=np.float64)
    assert not np.array_equal(img, full, equal_nan=True)  # the absent neighbour contributes nothing (NaN->0 in the blur)


def test_unrecorded_missing_neighbour_still_fails(store, tmp_path):
    assert bc.load_padding_aware_convolved_cell_from_band_cells(
        tmp_path, NAMES["A"], skycell_df=_df(), psf_sigma=40.0, combined_recipe=RECIPE,
        source=_partial_source(store, tmp_path)) is None
    # a different cell recorded absent does not excuse it
    from syndiff_pipeline.template_creation.processing import band_cell_store as bcs

    bcs.write_absent_cell(tmp_path, NAMES["C"], {})
    assert bc.load_padding_aware_convolved_cell_from_band_cells(
        tmp_path, NAMES["A"], skycell_df=_df(), psf_sigma=40.0, combined_recipe=RECIPE,
        source=_partial_source(store, tmp_path)) is None


def _seam_df():
    d = _df()
    d["pad_skycell_left"] = ""
    d.loc[NAMES["A"], "pad_skycell_left"] = "skycell.8888.000"
    d.loc["skycell.8888.000"] = d.loc[NAMES["B"]].copy()
    d.loc["skycell.8888.000", ["NAME", "pad_skycell_left"]] = ["skycell.8888.000", ""]
    d.loc["skycell.8888.000", "projection"] = 8888
    return d


@pytest.mark.parametrize("recorded_absent", [True, False])
def test_seam_source_unavailable_behaves_as_the_old_path(store, tmp_path, monkeypatch, recorded_absent):
    """Old path: ``padding_correction._location_correction`` raises PaddingCorrectionError when the fetch of a
    required cross-projection source returns None. The band-cell fetch returns None for an absent AND for a
    missing source alike, so the behaviour is the same (no invented fallback)."""
    from syndiff_pipeline.template_creation.processing import band_cell_store as bcs
    from syndiff_pipeline.template_creation.processing import padding_correction as PC

    if recorded_absent:
        bcs.write_absent_cell(tmp_path, "skycell.8888.000", {})
    seen = []

    def stub(*, neighbors, fetch_image, skycell, location, **kw):
        for n in neighbors:
            seen.append(n)
            if fetch_image(n) is None:
                raise PC.PaddingCorrectionError(f"required combined skycell {n} for {skycell}/{location} is unavailable")
        return np.zeros(kw["cell_shape"])

    monkeypatch.setattr(PC, "_location_correction", stub)
    monkeypatch.setattr(PC, "_cell_wcs", lambda row: None)
    src = bc.BandCellPlaneSource(tmp_path, RECIPE, loader=store[2])
    with pytest.raises(PC.PaddingCorrectionError, match="skycell.8888.000"):
        bc.load_padding_aware_convolved_cell_from_band_cells(
            tmp_path, NAMES["A"], skycell_df=_seam_df(), psf_sigma=40.0, combined_recipe=RECIPE, source=src)
    assert seen == ["skycell.8888.000"]
