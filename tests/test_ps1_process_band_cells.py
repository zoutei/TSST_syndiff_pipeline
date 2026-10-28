"""ps1_process side of the band-cell store: publish_band_cells / skip_convolved plumbing."""

from __future__ import annotations

import dataclasses
import pathlib
import queue as _thread_queue
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from syndiff_pipeline.template_creation.orchestration import verify as V
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    Ps1ProcessStageParams,
    parse_stage_params,
)
from syndiff_pipeline.template_creation.processing import band_cell_store as bcs
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import perband
from syndiff_pipeline.template_creation.processing import ps1_process as pp

from test_band_cell_store import (  # noqa: E402  (tests/ has no package; pytest rootdir import)
    CELL, PROJ, SKYCELL, _bundle, _headers, _raw_bands, _same_bits,
)
from test_verify_absence_probe import _resolved  # noqa: E402

# Fingerprints computed from the pre-change verify.py (git HEAD 278461c1) for the same inputs.
SPROJ = f"skycell.{PROJ}"
GOLDEN_DEFAULT = "743329f5987acd0e"
GOLDEN_WEIGHTED = "7f49695dde013407"


# ---------------------------------------------------------------- flags off == old behaviour
def test_flags_default_off_and_fingerprint_unchanged():
    p = Ps1ProcessStageParams()
    assert p.publish_band_cells is False and p.skip_convolved is False
    r = _resolved(pathlib.Path("/x"))
    assert V.config_fingerprint(r, "ps1_process") == GOLDEN_DEFAULT
    r.stages.ps1_process = Ps1ProcessStageParams(
        band_weights={"r": 0.3, "i": 0.3, "z": 0.2, "y": 0.2}, psf_sigma=33.0
    )
    assert V.config_fingerprint(r, "ps1_process") == GOLDEN_WEIGHTED


def test_flags_on_change_the_fingerprint():
    r = _resolved(pathlib.Path("/x"))
    base = V.config_fingerprint(r, "ps1_process")
    r.stages.ps1_process = Ps1ProcessStageParams(publish_band_cells=True)
    a = V.config_fingerprint(r, "ps1_process")
    r.stages.ps1_process = Ps1ProcessStageParams(skip_convolved=True)
    b = V.config_fingerprint(r, "ps1_process")
    assert len({base, a, b}) == 3


def test_flags_parse_from_site_config_and_default_effective_config_unchanged():
    raw = {"ps1_process": {"publish_band_cells": True, "skip_convolved": True}}
    st = parse_stage_params(raw)
    assert st.ps1_process.publish_band_cells and st.ps1_process.skip_convolved
    default = dataclasses.asdict(parse_stage_params({"ps1_process": {}}).ps1_process)
    legacy = {k: v for k, v in default.items() if k not in ("publish_band_cells", "skip_convolved")}
    assert default["publish_band_cells"] is False and default["skip_convolved"] is False
    assert legacy == {k: v for k, v in dataclasses.asdict(Ps1ProcessStageParams()).items()
                      if k in legacy}


def test_effective_convolve_flags():
    assert pp._effective_convolve_flags(True, False, False) == (True, False)
    assert pp._effective_convolve_flags(False, True, False) == (False, True)
    assert pp._effective_convolve_flags(True, False, True) == (False, False)
    assert pp._effective_convolve_flags(False, True, True) == (False, False)


# ---------------------------------------------------------------- band combiner + single cell
def _raw_bundle(bands):
    return {
        "skycell_id": SKYCELL, "projection": f"skycell.{PROJ}", "row_id": 0, "x_coord": 0,
        "task_type": "regular", "bands_data": bands,
        "masks_data": {b: np.zeros(bands["r"].shape, np.uint16) for b in perband.BANDS},
        "headers_data": _headers(),
        "weights_data": {b: np.full(bands["r"].shape, 0.25, np.float32) for b in perband.BANDS},
        "headers_weight_data": _headers(), "remove_saturated_stars": False,
    }


def _combine(bands, **kw):
    raw_q, out_q = _thread_queue.Queue(), _thread_queue.Queue()
    raw_q.put(_raw_bundle(bands))
    raw_q.put(None)
    pp.band_combiner_worker(raw_q, out_q, **kw)
    return out_q.get_nowait()


def test_band_combiner_attaches_unweighted_bands_only_when_asked():
    bands = _raw_bands(3, nan_in_r=False)
    off = _combine(bands)
    assert "band_images" not in off
    w = {"r": 0.3, "i": 0.3, "z": 0.2, "y": 0.2}
    on = _combine(bands, band_weights=w, band_cells_dtype="float64")
    assert set(on["band_images"]) == set(perband.BANDS)
    # UNWEIGHTED float64 bands (unit weights even though the combiner used ``w``): the removal
    # weighted float64 sum, cast to float32, is the combiner's float32 combined image within 2 ulp
    unit = perband.weighted_band_images(bands, _headers(), weights=perband.UNIT_WEIGHTS)
    for b in perband.BANDS:
        assert np.array_equal(on["band_images"][b], unit[b])
    tot = perband.weighted_sum(on["band_images"], w).astype(np.float32)
    ulp = np.spacing(np.abs(on["combined_image"]))
    assert np.all(np.abs(tot.astype(np.float64) - on["combined_image"]) <= 2 * ulp)


def test_flags_off_single_cell_result_has_no_band_payload():
    b = _bundle(_raw_bands(), None, "float32")
    b.pop("band_images")
    res = pp.process_single_cell(b)
    assert "band_images_shm" not in res
    out = pp._materialize_shm_result(res)
    assert "band_images" not in out


def test_publish_combined_result_publishes_band_cells_under_same_fingerprint(tmp_path):
    recipe = cs.combined_recipe(gaia_version="none")
    res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
    assert set(res["band_images"]) == set(perband.BANDS)
    info = pp.publish_combined_result(str(tmp_path), recipe, res)
    assert info is not None
    assert "band_images" not in res  # freed from the result
    assert bcs.band_cells_complete(tmp_path, SPROJ, CELL, info["fingerprint"])
    assert bcs.resolve_band_cells_fingerprint_for_recipe(tmp_path, SPROJ, CELL, recipe) == info["fingerprint"]
    comb = cs.try_load_combined_cell(tmp_path, SPROJ, CELL, info["fingerprint"])
    loaded = bcs.try_load_band_cells(tmp_path, SPROJ, CELL, info["fingerprint"])
    d = np.abs(bcs.combined_from_band_cells(loaded).astype(np.float64) - comb["combined_image"])
    ok = np.isfinite(comb["combined_image"])
    assert np.nanmax(d[ok]) / np.abs(comb["combined_image"][ok]).max() < 1e-6
    assert np.array_equal(np.isnan(loaded["band_images"]["r"]), np.isnan(comb["combined_image"]))


def test_publish_combined_result_without_bands_publishes_no_band_cells(tmp_path):
    recipe = cs.combined_recipe(gaia_version="none")
    b = _bundle(_raw_bands(), None, "float32")
    b.pop("band_images")
    res = pp._materialize_shm_result(pp.process_single_cell(b))
    info = pp.publish_combined_result(str(tmp_path), recipe, res)
    assert info is not None
    assert not bcs.band_cell_dir(tmp_path, SPROJ, CELL, info["fingerprint"]).exists()


# ---------------------------------------------------------------- ingest: combined hit needs band cells
def _ingest(tmp_path, recipe, monkeypatch, required):
    calls = {"n": 0}

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return {}, {}, {}, {}, {}

    monkeypatch.setattr(pp, "fetch_skycell_bands_masks_and_headers", fake_fetch)
    tq, rq = _thread_queue.Queue(), _thread_queue.Queue()
    tq.put((SKYCELL, f"skycell.{PROJ}", 0, 0, "regular"))
    tq.put(None)
    pp.ingest_worker(tq, rq, ps1_source="stream", band_cache={},
                     combined_store_data_root=str(tmp_path), combined_store_recipe=recipe,
                     band_cells_required=required)
    return calls["n"], rq


def test_ingest_combined_hit_without_band_cells_reprocesses_when_required(tmp_path, monkeypatch):
    recipe = cs.combined_recipe(gaia_version="none")
    res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
    res.pop("band_images")  # combined cell only
    assert pp.publish_combined_result(str(tmp_path), recipe, res) is not None
    n, rq = _ingest(tmp_path, recipe, monkeypatch, required=False)
    assert n == 0 and rq.get_nowait()["task_type"] == "regular_cache_hit"
    n, rq = _ingest(tmp_path, recipe, monkeypatch, required=True)
    assert n == 1 and rq.empty()


def test_ingest_combined_hit_with_band_cells_is_a_hit_when_required(tmp_path, monkeypatch):
    recipe = cs.combined_recipe(gaia_version="none")
    res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
    assert pp.publish_combined_result(str(tmp_path), recipe, res) is not None
    n, rq = _ingest(tmp_path, recipe, monkeypatch, required=True)
    assert n == 0 and rq.get_nowait()["task_type"] == "regular_cache_hit"


# ---------------------------------------------------------------- skip_convolved writes no convolved files
def test_skip_convolved_row_step_convolves_and_publishes_nothing(tmp_path, monkeypatch):
    from syndiff_pipeline.template_creation.processing import convolution_utils, convolved_store

    def boom(*a, **k):
        raise AssertionError("convolution/publish must not run with skip_convolved")

    monkeypatch.setattr(convolution_utils, "apply_gaussian_convolution", boom)
    monkeypatch.setattr(convolved_store, "publish_convolved_cell", boom)
    monkeypatch.setattr(pp, "_publish_canonical_convolved_snapshot", boom)
    monkeypatch.setattr(pp, "_gather_cells_for_row", lambda *a, **k: [])
    monkeypatch.setattr(pp, "assemble_row_from_bundles", lambda *a, **k: ({}, {}))
    shared, per_scc = pp._effective_convolve_flags(True, True, True)
    state = pp.ProcessingState(current_array=np.zeros((8, 8), np.float32), next_array=np.zeros((8, 8), np.float32))
    cfg = mock.MagicMock()
    out = pp.process_row_step_from_queue(
        state, cfg, {}, 0, None, _thread_queue.Queue(), {}, 40.0, {}, "skycell.1",
        convolved_store_data_root=str(tmp_path) if shared else None,
        combined_store_recipe=cs.combined_recipe(gaia_version="none"),
        convolved_store_recipe=None,
        write_per_scc_convolved_zarr=per_scc,
    )
    assert out[0] == {}
    assert not (tmp_path / "ps1_skycells_zarr" / "ps1_convolved.zarr").exists()


# ---------------------------------------------------------------- verify / probe / preflight
def _publish_cells(tmp_path, recipe, with_bands):
    res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
    if not with_bands:
        res.pop("band_images")
    return pp.publish_combined_result(str(tmp_path), recipe, res)


def _resolved_skip(tmp, **flags):
    r = _resolved(tmp)
    r.stages.ps1_process = Ps1ProcessStageParams(
        enable_saturation_correction=False, remove_saturated_stars=False, **flags
    )
    return r


def test_skip_convolved_checkpoint_location_and_verify(tmp_path):
    from syndiff_pipeline.common import scc_paths

    recipe_params = dict(enable_saturation_correction=False, remove_saturated_stars=False)
    recipe = cs.production_combined_recipe(recipe_params)
    names = [SKYCELL]

    # combined-only completeness (skip_convolved without publish_band_cells)
    r = _resolved_skip(tmp_path, skip_convolved=True)
    assert V.resolve_ps1_process_checkpoint_location(r) == scc_paths.ps1_combined_zarr_path(r.data_root)
    assert V.resolve_ps1_process_checkpoint_location(
        _resolved_skip(tmp_path, skip_convolved=True, publish_band_cells=True)
    ) == scc_paths.ps1_band_cells_zarr_path(r.data_root)
    # flags off: the old locations
    assert "convolved" in str(V.resolve_ps1_process_checkpoint_location(_resolved_skip(tmp_path))).lower()

    with mock.patch.object(V, "expected_ps1_process_skycells", return_value=names):
        r = _resolved_skip(tmp_path, skip_convolved=True)
        r.data_root = str(tmp_path)
        assert not V.verify_ps1_process(r).ok
        assert V.stage_absence_probe(r, "ps1_process").name == "ABSENT"
        info = _publish_cells(tmp_path, recipe, with_bands=False)
        assert info
        assert V.verify_ps1_process(r).ok  # combined cell suffices
        assert V.stage_absence_probe(r, "ps1_process").name == "MAYBE_PRESENT"

        rb = _resolved_skip(tmp_path, skip_convolved=True, publish_band_cells=True)
        rb.data_root = str(tmp_path)
        assert not V.verify_ps1_process(rb).ok  # combined only: band cells missing
        assert not bcs.band_cell_dir(tmp_path, SPROJ, CELL, info["fingerprint"]).exists()
        res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
        pp.publish_combined_result(str(tmp_path), recipe, res)
        assert V.verify_ps1_process(rb).ok
    # nothing under ps1_convolved.zarr
    assert not (tmp_path / "ps1_skycells_zarr" / "ps1_convolved.zarr").exists()


def test_preflight_full_profile_when_skip_convolved():
    from syndiff_pipeline.template_creation.orchestration import ps1_process_preflight as pf

    params = Ps1ProcessStageParams(use_shared_convolved_store=True, write_per_scc_convolved_zarr=False,
                                   skip_convolved=True)
    with mock.patch("syndiff_pipeline.template_creation.processing.ps1_process.expected_convolved_skycells",
                    return_value=["skycell.1.001"]):
        plan = pf.plan_ps1_process_launch(data_root="/nonexistent", sector=1, camera=1, ccd=1,
                                          oversampling_factor=1, params=params)
    assert plan.decision == pf.DECISION_FULL and "skip_convolved" in plan.reason


def test_cell_published_for_skip_convolved(tmp_path):
    recipe = cs.combined_recipe(gaia_version="none")
    assert not pp._cell_published_for_skip_convolved(str(tmp_path), SKYCELL, recipe, False)
    assert _publish_cells(tmp_path, recipe, with_bands=False)
    assert pp._cell_published_for_skip_convolved(str(tmp_path), SKYCELL, recipe, False)
    assert not pp._cell_published_for_skip_convolved(str(tmp_path), SKYCELL, recipe, True)


# ---------------------------------------------------------------- skip_convolved through the real pipeline
import pandas as pd  # noqa: E402


def _mapping_df():
    rows = []
    for i, c in enumerate(("033", "034", "035")):
        rows.append({
            "NAME": f"skycell.2484.{c}", "projection": "2484", "y": 0, "x": i,
            "NAXIS1": SIZE_, "NAXIS2": SIZE_,
            "pad_skycell_left": "skycell.2484.032" if i == 0 else "",   # same projection: not a source
            "pad_skycell_right": "skycell.2485.001/skycell.2485.002" if i == 2 else "",
            "pad_skycell_top": "skycell.2483.010" if i == 1 else "",
        })
    return pd.DataFrame(rows)


SIZE_ = 48


def _f01_cells(df):
    """The cell set f01_select computes: cells | same_nbrs | xproj_src."""
    from syndiff_pipeline.template_creation.processing.padding_correction import cross_projection_padding_spec

    d = df.set_index("NAME", drop=False)
    cells = sorted(d.index)
    md = {}
    need = set(cells)
    for name in cells:
        proj = str(d.loc[name, "projection"])
        if proj not in md:
            md[proj] = pp.extract_projection_metadata(d, proj)
        nb = pp._find_projection_neighbors(md[proj], name, int(d.loc[name, "y"]), int(d.loc[name, "x"]))
        need |= {n for n, _, _ in nb}
        need |= {s["neighbor"] for s in cross_projection_padding_spec(d.loc[name])}
    return need


def _run_skip(tmp_path, monkeypatch, load_hook=None, **flags):
    from syndiff_pipeline.template_creation.processing import convolution_utils

    df = _mapping_df()
    loaded = []

    def fake_load(zarr_store, projection, skycell_id):
        loaded.append(skycell_id)
        bands = _raw_bands(len(loaded), nan_in_r=False)
        masks = {b: np.zeros((SIZE_, SIZE_), np.uint16) for b in perband.BANDS}
        wts = {b: np.full((SIZE_, SIZE_), 0.25, np.float32) for b in perband.BANDS}
        out = (bands, masks, wts, _headers(), _headers())
        return out if load_hook is None else load_hook(skycell_id, *out)

    def boom(*a, **k):
        raise AssertionError("Gaussian blur must never run with skip_convolved")

    monkeypatch.setattr(convolution_utils, "apply_gaussian_convolution", boom)
    monkeypatch.setattr(pp, "sequential_processor", boom)  # no row assembly / window / padding
    monkeypatch.setattr(pp, "process_row_step_from_queue", boom)
    monkeypatch.setattr(pp, "apply_cross_projection_padding", boom)
    monkeypatch.setattr(pp.zarr_utils, "load_skycell_bands_masks_and_headers", fake_load)
    monkeypatch.setattr(pp, "get_projections_from_csv", lambda p: ["2484"])
    monkeypatch.setattr(pp, "load_csv_data", lambda p: df)
    monkeypatch.setattr(pp, "expected_convolved_skycells", lambda *a, **k: set())
    monkeypatch.setattr(pp, "zarr", mock.MagicMock())
    csv = tmp_path / "master.csv"
    csv.write_text("NAME\n")
    res = pp.run_modern_sliding_window_pipeline(
        sector=1, camera=1, ccd=1, data_root=str(tmp_path), mapping_csv_path=str(csv),
        remove_saturated_stars=False, num_ingest_workers=2, skip_convolved=True,
        use_shared_convolved_store=True, **flags,
    )
    return df, res, loaded


def test_skip_convolved_pipeline_publishes_exactly_f01_cells_without_blur(tmp_path, monkeypatch):
    import threading

    before = threading.active_count()
    df, res, loaded = _run_skip(tmp_path, monkeypatch, publish_band_cells=True)
    assert "error" not in res, res
    need = _f01_cells(df)
    assert {c for c in loaded} == need and len(loaded) == len(need)  # each ingested once
    recipe = cs.production_combined_recipe(
        dict(enable_saturation_correction=False, remove_saturated_stars=False))
    for name in need:
        proj, cell = cs._projection_and_cell(name)
        assert bcs.resolve_band_cells_fingerprint_for_recipe(tmp_path, proj, cell, recipe) is not None, name
        assert cs.resolve_combined_fingerprint_for_recipe(tmp_path, proj, cell, recipe) is not None, name
    assert sorted(res["produced_skycells"]) == sorted(need)
    # nothing convolved written anywhere
    assert not (tmp_path / "ps1_skycells_zarr" / "ps1_convolved.zarr").exists()
    assert not list(tmp_path.glob("convolved_results/*.zarr"))
    assert threading.active_count() <= before + 1  # no leftover pipeline threads

    # second run: everything already published -> nothing is ingested again
    df, res2, loaded2 = _run_skip(tmp_path, monkeypatch, publish_band_cells=True)
    assert "error" not in res2 and loaded2 == []


def test_skip_convolved_combined_only_when_band_cells_off(tmp_path, monkeypatch):
    df, res, loaded = _run_skip(tmp_path, monkeypatch)
    assert "error" not in res and len(loaded) == len(_f01_cells(df))
    for name in _f01_cells(df):
        proj, cell = cs._projection_and_cell(name)
        assert not bcs.band_cell_dir(tmp_path, proj, cell, "x").parent.exists()
