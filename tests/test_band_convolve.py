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
        return {"band_images": cells[name], "combined_mask": masks[name]}

    return cells, masks, loader, calls


RECIPE = cs.combined_recipe(band_weights={"r": 0.25, "i": 0.25, "z": 0.25, "y": 0.25}, gaia_version="none",
                            remove_saturated_stars=False, bright_star_removal="none")


def _source(store, **kw):
    cells, masks, loader, calls = store
    return bc.BandCellPlaneSource("/nonexistent", RECIPE, loader=loader,
                                  summer=lambda ld: perband.sum_bands(ld["band_images"]), **kw)


def test_equals_canonical_cell_of_summed_planes(store):
    cells = store[0]
    df = _df()
    name = NAMES["A"]
    got = bc.load_padding_aware_convolved_cell_from_band_cells(
        "/nonexistent", name, skycell_df=df, psf_sigma=40.0, combined_recipe=RECIPE, source=_source(store))
    assert got is not None
    img, mask = got
    md = cc.metadata_for_cell(df, name)
    ref = cc.canonical_cell_image(name, md, lambda n: None if n not in cells else perband.sum_bands(cells[n]),
                                  40.0, 470)
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
    per_band = perband.convolve_band_cells(name, md, lambda n: cells.get(n), 40.0, 470)
    s = sum(per_band[b].astype(np.float64) for b in perband.BANDS)
    fin = np.isfinite(s) & np.isfinite(img)
    assert fin.any()
    assert np.abs(s[fin] - img[fin]).max() / np.abs(img[fin]).max() < 1e-6


def test_missing_neighbour_gives_none(store):
    cells, masks, loader, calls = store

    def partial(data_root, projection, cell, fp):
        return None if cell == "001" else loader(data_root, projection, cell, fp)

    src = bc.BandCellPlaneSource("/nonexistent", RECIPE, loader=partial,
                                 summer=lambda ld: perband.sum_bands(ld["band_images"]))
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
