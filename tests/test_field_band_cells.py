"""Field-mode band-cell route (``stages.downsample.convolve_from_band_cells`` with geometry_mode field)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.orchestration import verify
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
)
from syndiff_pipeline.template_creation.processing import band_cell_store
from syndiff_pipeline.template_creation.processing import band_convolve as bc
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolution_utils
from syndiff_pipeline.template_creation.processing import field_downsample as fd
from syndiff_pipeline.template_creation.processing import field_templates as ft
from syndiff_pipeline.template_creation.processing.field_templates import contrib_path, load_contrib

REMOVAL = {"r": 0.25, "i": 0.25, "z": 0.25, "y": 0.25}
D13 = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}
RECIPE = cs.combined_recipe(
    band_weights=REMOVAL, gaia_version="none", remove_saturated_stars=False, bright_star_removal="none"
)


def _resolved(**kw):
    ds = DownsampleStageParams(**kw)
    return SimpleNamespace(
        target=SimpleNamespace(sector=26, camera=3, ccd=1),
        stages=SimpleNamespace(downsample=ds, mapping=MappingStageParams()),
        template_output_base="x",
        downsample_remap_store_name=None,
    )


def _fp(r):
    return verify.config_fingerprint(r, "downsample")


# ------------------------------------------------------------------ fingerprint params
def test_field_route_params_none_when_flag_off():
    assert bc.field_route_fingerprint_params(_resolved()) is None
    assert bc.field_route_fingerprint_params(_resolved(geometry_mode="linear")) is None


def test_field_route_params_none_for_linear():
    assert bc.field_route_fingerprint_params(_resolved(geometry_mode="linear", convolve_from_band_cells=True)) is None


def test_field_route_params_dict_for_field():
    p = bc.field_route_fingerprint_params(_resolved(convolve_from_band_cells=True))
    assert set(p) == {"route", "blur_method", "combined_recipe_id", "psf_sigma", "kernel_radius", "canonical_geometry"}
    assert p["route"] == bc.ROUTE_NAME
    assert p["blur_method"] == "dask"
    assert p["psf_sigma"] == 40.0 and p["kernel_radius"] == bc.DEFAULT_RADIUS
    assert len(p["combined_recipe_id"]) == 16


def test_field_fingerprint_depends_on_blur_method_and_weights():
    base = _resolved(convolve_from_band_cells=True)
    fft = _resolved(convolve_from_band_cells=True, blur_method="fft")
    dask = _resolved(convolve_from_band_cells=True, blur_method="dask")
    w = _resolved(convolve_from_band_cells=True, band_weights=D13)
    off = _resolved()
    assert _fp(fft) != _fp(dask)
    assert _fp(base) == _fp(dask)  # None == "dask"
    assert _fp(w) != _fp(base)
    assert len({_fp(base), _fp(fft), _fp(w), _fp(off)}) == 4


def test_linear_fingerprint_unaffected_by_field_extras(monkeypatch):
    from syndiff_pipeline.common.provenance import model

    r = _resolved(geometry_mode="linear", convolve_from_band_cells=True, band_weights=D13)
    fp, params = _fp(r), model.downsample_recipe_params(r)
    assert "field_band_route" not in params
    monkeypatch.setattr(bc, "field_route_fingerprint_params", lambda resolved: None)
    assert _fp(r) == fp
    assert model.downsample_recipe_params(r) == params


def test_field_recipe_params_record_route():
    from syndiff_pipeline.common.provenance import model

    p = model.downsample_recipe_params(_resolved(convolve_from_band_cells=True, blur_method="fft"))
    assert p["field_band_route"]["blur_method"] == "fft"
    assert "field_band_route" not in model.downsample_recipe_params(_resolved())


# ------------------------------------------------------------------ provenance + header
def test_band_cell_route_provenance_weights_source():
    kw = dict(combined_recipe=RECIPE, blur_method="fft", psf_sigma=40.0)
    a = bc.band_cell_route_provenance(band_weights=None, **kw)
    assert a["band_weights_source"] == "removal_weights_in_cells"
    assert a["band_weights"] == {b: 0.25 for b in "rizy"}
    b = bc.band_cell_route_provenance(band_weights=D13, **kw)
    assert b["band_weights_source"] == "config"
    assert b["band_weights"] == D13
    assert a["blur_method"] == "fft" and a["route"] == bc.ROUTE_NAME and a["zpcell"] == "off"
    assert a["combined_recipe_id"] == b["combined_recipe_id"] == bc.combined_recipe_id(RECIPE)
    assert bc.combined_recipe_id(dict(RECIPE)) == bc.combined_recipe_id(RECIPE)
    other = cs.combined_recipe(
        band_weights=D13, gaia_version="none", remove_saturated_stars=False, bright_star_removal="none"
    )
    assert bc.combined_recipe_id(other) != bc.combined_recipe_id(RECIPE)


def _hdr(provenance):
    return ft.build_field_fits_header(sector=1, camera=2, ccd=3, group_id=4, provenance=provenance)


def test_header_without_template_source_unchanged():
    base = {"mapping_fingerprint": "abc", "intra_skycell_R": 2}
    plain = _hdr(base)
    assert not any(k.startswith("BANDW") or k in ("BANDSRC", "CONVBC", "BCRECIPE", "BLURM") for k in plain)
    assert _hdr(dict(base)).tostring() == plain.tostring()
    assert _hdr({**base, "template_source": None}).tostring() == plain.tostring()


def test_header_with_template_source_has_band_cards():
    src = bc.band_cell_route_provenance(combined_recipe=RECIPE, band_weights=D13, blur_method="fft", psf_sigma=40.0)
    h = _hdr({"mapping_fingerprint": "abc", "template_source": src})
    assert h["BANDWR"] == 0.254 and h["BANDWI"] == 0.4368 and h["BANDWZ"] == 0.1654 and h["BANDWY"] == 0.1438
    assert h["BANDSRC"] == "band_cells" and h["CONVBC"] is True and h["BWTMODE"] == "unweighted"
    assert h["BWSRC"] == "config" and h["ZPMODE"] == "off" and h["BLURM"] == "fft"
    assert h["PSFSIG"] == 40.0 and h["BLURRAD"] == bc.DEFAULT_RADIUS
    assert h["CANONGEO"] == src["canonical_geometry"] and h["BCRECIPE"] == src["combined_recipe_id"]
    assert h["MAPFP"] == "abc"


# ------------------------------------------------------------------ availability gate
def _df(names):
    return pd.DataFrame({"NAME": names}).set_index("NAME", drop=False)


def _patch_store(monkeypatch, *, need, absent=(), missing=()):
    monkeypatch.setattr(band_cell_store, "required_band_cells", lambda df, cells: list(need))
    monkeypatch.setattr(band_cell_store, "is_absent_cell", lambda root, n: n in absent)

    def resolve(root, proj, cell, recipe):
        return None if f"{proj}.{cell}" in missing else "fp"

    monkeypatch.setattr(band_cell_store, "resolve_band_cells_fingerprint_for_recipe", resolve)


def _validate(required, master, df_names, **kw):
    return fd._validate_l5_band_cell_availability(
        master_skycells=set(master), data_root="/r", skycell_df=_df(df_names), combined_recipe=RECIPE,
        required_skycells=set(required), **kw)


A, B, C = "skycell.1.000", "skycell.1.001", "skycell.1.010"


def test_validate_all_present(monkeypatch):
    _patch_store(monkeypatch, need=[A, B, C])
    batches = [(A, {}), (B, {})]
    kept, skipped, absent = _validate([A, B], [A, B, C], [A, B, C], skycell_batches=batches)
    assert kept == batches and skipped == [] and absent == set()


def test_validate_missing_band_cell_raises(monkeypatch):
    _patch_store(monkeypatch, need=[A, B, C], missing={B})
    with pytest.raises(fd.L5CompletenessError) as ei:
        _validate([A, B], [A, B, C], [A, B, C])
    d = ei.value.diagnostics
    assert d["absent_from_band_store"] == [B]
    assert d["absent_from_master"] == [] and d["absent_from_mapping_csv"] == []


def test_validate_ps1_absent_tolerated(monkeypatch):
    _patch_store(monkeypatch, need=[A, B], absent={B}, missing={B})
    kept, _, absent = _validate([A, B], [A, B], [A, B], skycell_batches=[(A, {})])
    assert absent == {B} and kept == [(A, {})]


def test_validate_required_not_in_master(monkeypatch):
    _patch_store(monkeypatch, need=[A])
    with pytest.raises(fd.L5CompletenessError) as ei:
        _validate([A, C], [A], [A, C])
    assert ei.value.diagnostics["absent_from_master"] == [C]


def test_validate_required_not_in_mapping_csv(monkeypatch):
    _patch_store(monkeypatch, need=[A])
    with pytest.raises(fd.L5CompletenessError) as ei:
        _validate([A, C], [A, C], [A])
    assert ei.value.diagnostics["absent_from_mapping_csv"] == [C]


# ------------------------------------------------------------------ loader branch
@pytest.fixture
def worker():
    fd._reset_l5_worker()
    yield
    fd._reset_l5_worker()


def test_band_branch_loader_dispatch_and_blur_method(monkeypatch, worker):
    seen = {}

    def fake(data_root, skycell, **kw):
        seen.update(data_root=data_root, skycell=skycell, method=convolution_utils._default_method(), **kw)
        return "SENTINEL"

    monkeypatch.setattr(bc, "load_padding_aware_convolved_cell_from_band_cells", fake)
    df = _df([A])
    fd._init_l5_worker({
        "convolve_from_band_cells": True, "blur_method": "fft", "skycell_df": df, "psf_sigma": 40.0,
        "combined_recipe": RECIPE, "band_weights": D13, "data_root": "/r"})
    assert fd._load_ps1_skycell_for_l5(A) == "SENTINEL"
    assert seen["method"] == "fft" and seen["data_root"] == "/r" and seen["skycell"] == A
    assert seen["skycell_df"] is df and seen["psf_sigma"] == 40.0
    assert seen["combined_recipe"] is RECIPE and seen["band_weights"] == D13
    assert convolution_utils._METHOD_OVERRIDE == []  # context exited


# ------------------------------------------------------------------ absent-cell path in the batch
def _batch_payload(tmp_path, store, absent):
    from syndiff_pipeline.template_creation.processing.field_abutting import abutting_undirected_pairs

    master = np.zeros((4, 6), dtype=np.int32)
    master[:, :3] = 10
    master[:, 3:] = 20
    name_to_id = {"skycell.1.1": 10, "skycell.1.2": 20}
    return {
        "store": str(store), "rebuild_field_store": True, "mapping_root": str(tmp_path),
        "sector": 1, "camera": 1, "ccd": 1, "oversampling_factor": 1, "scratch_regmaps": {},
        "zarr_path": str(tmp_path / "dummy.zarr"), "name_to_id": name_to_id,
        "id_to_name": {v: k for k, v in name_to_id.items()}, "master_map": master,
        "pair_ids": abutting_undirected_pairs(master),
        "neighbours_by_id": fd._neighbours_by_skycell_id(abutting_undirected_pairs(master)),
        "group_shifts_by_gid": {0: {}, 1: {}}, "epoch_index": None,
        "exact_cache_l4a_dir": str(tmp_path / "l4a"), "exact_cache_l4b_dir": str(tmp_path / "l4b"),
        "base_tess_shape": (4, 6), "roi_bounds": (0, 0, 6, 4), "ignore_mask": 0, "intra_skycell_R": 1,
        "ps1_absent_skycells": frozenset(absent),
    }


_BUCKETS = {(1, 0, ((20, 0, 1),)): [(0, 1, 0), (1, 1, 0)]}


def test_batch_absent_ps1_cell_writes_empty_contribs(monkeypatch, tmp_path: Path, worker):
    store = tmp_path / "templates"
    (store / "contribs").mkdir(parents=True)
    monkeypatch.setattr(fd, "_load_ps1_skycell_for_l5", lambda _sc: None)
    monkeypatch.setattr(fd, "_read_regmap_assignment_l5", lambda sc: pytest.fail("regmap must not be read"))
    fd._init_l5_worker(_batch_payload(tmp_path, store, {"skycell.1.1"}))
    res = fd._l5_skycell_batch("skycell.1.1", _BUCKETS)
    assert res["n_writes"] == 2 and res["n_nonempty"] == 0 and res["n_compose"] == 0
    for gid in (0, 1):
        p = contrib_path(store, "skycell.1.1", 1, 0, group_id=gid)
        assert p.is_file()
        assert len(load_contrib(p)["indices"]) == 0


def test_batch_missing_cell_not_ps1_absent_raises(monkeypatch, tmp_path: Path, worker):
    store = tmp_path / "templates"
    (store / "contribs").mkdir(parents=True)
    monkeypatch.setattr(fd, "_load_ps1_skycell_for_l5", lambda _sc: None)
    fd._init_l5_worker(_batch_payload(tmp_path, store, {"skycell.1.2"}))  # a different cell is absent
    with pytest.raises(fd.L5CompletenessError):
        fd._l5_skycell_batch("skycell.1.1", _BUCKETS)
    assert not list((store / "contribs").glob("*.npz"))
