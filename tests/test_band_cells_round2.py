"""Round-2 review fixes of the band-cell store: missing PS1 band, provenance record key, checkpoint
keying, config validation, early downsample check, threaded writer."""

from __future__ import annotations

import json
import queue as _thread_queue
import shutil
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.common import scc_paths
from syndiff_pipeline.common.provenance.gc import gc_report
from syndiff_pipeline.common.provenance.ingest import drain_spool
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.template_creation.orchestration import provenance_checkpoint as PC
from syndiff_pipeline.template_creation.orchestration import verify as V
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    Ps1ProcessStageParams,
    parse_stage_params,
)
from syndiff_pipeline.template_creation.processing import band_cell_store as bcs
from syndiff_pipeline.template_creation.processing import band_convolve
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import linear_downsample as LD
from syndiff_pipeline.template_creation.processing import perband
from syndiff_pipeline.template_creation.processing import ps1_process as pp

from test_band_cell_store import CELL, PROJ, SIZE, SKYCELL, _bundle, _headers, _raw_bands
from test_ps1_process_band_cells import (
    SPROJ, _combine, _f01_cells, _mapping_df, _run_skip,
)
from test_verify_absence_probe import _resolved

RECIPE = dict(enable_saturation_correction=False, remove_saturated_stars=False)


# ------------------------------------------------------------------ 1. missing PS1 band
def _combiner_bundle_without(band: str, seed=7):
    """A reduced bundle exactly as band_combiner_worker makes it for a cell lacking ``band``."""
    bands = _raw_bands(seed, nan_in_r=False)
    raw = {
        "skycell_id": SKYCELL, "projection": SPROJ, "row_id": 0, "x_coord": 0, "task_type": "regular",
        "bands_data": {b: a for b, a in bands.items() if b != band},
        "masks_data": {b: np.zeros((SIZE, SIZE), np.uint16) for b in bands if b != band},
        "headers_data": {b: h for b, h in _headers().items() if b != band},
        "weights_data": {b: np.full((SIZE, SIZE), 0.25, np.float32) for b in bands if b != band},
        "headers_weight_data": {b: h for b, h in _headers().items() if b != band},
        "remove_saturated_stars": False,
    }
    rq, oq = _thread_queue.Queue(), _thread_queue.Queue()
    rq.put(raw)
    rq.put(None)
    pp.band_combiner_worker(rq, oq, band_cells_dtype="float32")
    return oq.get_nowait()


@pytest.mark.parametrize("band", ["y", "r"])
def test_cell_lacking_a_ps1_band_publishes_zero_plane_and_matches_combiner(tmp_path, band):
    recipe = cs.production_combined_recipe(RECIPE)
    bundle = _combiner_bundle_without(band)
    assert bundle["band_missing"] == [band] and band not in bundle["band_images"]
    res = pp._materialize_shm_result(pp.process_single_cell(bundle))
    assert res["band_missing"] == [band] and set(res["band_images"]) == set(perband.BANDS)
    c = res["combined_image"]
    info = pp.publish_combined_result(str(tmp_path), recipe, dict(res))
    assert info and bcs.band_cells_complete(tmp_path, SPROJ, CELL, info["fingerprint"])
    loaded = bcs.try_load_band_cells(tmp_path, SPROJ, CELL, info["fingerprint"])
    assert loaded["missing_bands"] == [band]
    z = loaded["band_images"][band]
    assert np.all(z[~np.isnan(z)] == 0) and np.array_equal(np.isnan(z), np.isnan(c))
    # combined_from_band_cells == the combiner's combined cell (4 float32 ulp), NaN pattern equal
    total = bcs.combined_from_band_cells(loaded)
    ok = np.isfinite(c)
    assert np.array_equal(np.isnan(total), np.isnan(c))
    assert np.max(np.abs(total[ok].astype(np.float64) - c[ok])) <= 4 * np.spacing(np.abs(c[ok]).max())
    # BandCellPlaneSource with the recipe's own weights gives the same plane
    src = band_convolve.BandCellPlaneSource(tmp_path, recipe)
    plane = src.image(SKYCELL)
    assert plane is not None and plane.dtype == np.float64
    assert np.array_equal(np.isnan(plane), np.isnan(c))
    assert np.max(np.abs(plane[ok] - c[ok])) <= 4 * np.spacing(np.abs(c[ok]).max())
    # explicit weights == perband.weighted_sum on the three present bands only
    w = cs.DEFAULT_BAND_WEIGHTS
    only = {b: a for b, a in loaded["band_images"].items() if b != band}
    assert np.array_equal(perband.weighted_sum(loaded["band_images"], w), perband.weighted_sum(only, w),
                          equal_nan=True)
    # with zpcell factors the zero band still contributes 0
    f = {b: 1.1 for b in perband.BANDS}
    assert np.array_equal(perband.weighted_sum(loaded["band_images"], w, f), perband.weighted_sum(only, w, f),
                          equal_nan=True)


def test_missing_bands_must_name_real_bands(tmp_path):
    recipe = cs.production_combined_recipe(RECIPE)
    res = pp._materialize_shm_result(pp.process_single_cell(_combiner_bundle_without("y")))
    assert bcs.publish_band_cells(
        tmp_path, SPROJ, CELL, band_images=res["band_images"], combined_mask=res["combined_mask"],
        headers_data={}, removed_stars=[], recipe=recipe, missing_bands=["q"]) is None


# ------------------------------------------------------------------ 1b. hard failure in skip mode only
def test_skip_convolved_fails_hard_when_a_band_publish_fails(tmp_path, monkeypatch):
    real = bcs.publish_band_cells
    monkeypatch.setattr(bcs, "publish_band_cells", lambda *a, **k: None)
    df, res, loaded = _run_skip(tmp_path, monkeypatch, publish_band_cells=True)
    assert "error" in res and "not published" in res["error"] and "band cells" in res["error"]
    # the verify gate of that store must not pass either
    monkeypatch.setattr(bcs, "publish_band_cells", real)


def test_skip_convolved_fails_hard_when_a_cell_is_lost(tmp_path, monkeypatch):
    real = pp.process_single_cell
    def lossy(bundle):
        return None if bundle["skycell_id"].endswith(".034") else real(bundle)
    monkeypatch.setattr(pp, "process_single_cell", lossy)
    df, res, loaded = _run_skip(tmp_path, monkeypatch, publish_band_cells=True)
    assert "error" in res and "skycell.2484.034" in res["error"]


def test_non_skip_band_publish_failure_stays_best_effort(tmp_path, monkeypatch):
    recipe = cs.production_combined_recipe(RECIPE)
    monkeypatch.setattr(bcs, "publish_band_cells", lambda *a, **k: None)
    res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
    info = pp.publish_combined_result(str(tmp_path), recipe, res)
    assert info is not None  # combined cell published, band failure only logged


# ------------------------------------------------------------------ 1c. config validation
def test_publish_band_cells_refuses_saturation_correction():
    with pytest.raises(ValueError, match="enable_saturation_correction"):
        Ps1ProcessStageParams(publish_band_cells=True, enable_saturation_correction=True)
    with pytest.raises(ValueError, match="enable_saturation_correction"):
        parse_stage_params({"ps1_process": {"publish_band_cells": True, "enable_saturation_correction": True}})
    Ps1ProcessStageParams(publish_band_cells=True)
    Ps1ProcessStageParams(enable_saturation_correction=True)  # unaffected without the flag


def test_run_refuses_publish_band_cells_with_saturation_correction(tmp_path):
    out = pp.run_modern_sliding_window_pipeline(
        sector=1, camera=1, ccd=1, data_root=str(tmp_path), mapping_csv_path=str(tmp_path / "x.csv"),
        publish_band_cells=True, enable_saturation_correction=True, remove_saturated_stars=False)
    assert "error" in out


# ------------------------------------------------------------------ 5. config + early downsample check
def test_convolve_from_band_cells_needs_linear_geometry():
    with pytest.raises(ValueError, match="geometry_mode: linear"):
        parse_stage_params({"downsample": {"convolve_from_band_cells": True}})
    with pytest.raises(ValueError, match="geometry_mode: linear"):
        parse_stage_params({"downsample": {"convolve_from_band_cells": True, "geometry_mode": "field"},
                            "wcs_grouping": {"geometry_mode": "linear"}})
    st = parse_stage_params({"downsample": {"convolve_from_band_cells": True, "geometry_mode": "linear"}})
    assert st.downsample.convolve_from_band_cells
    parse_stage_params({"downsample": {"geometry_mode": "field"}})  # flag off: untouched


def test_downsample_start_check_lists_missing_cells_then_passes(tmp_path, monkeypatch):
    df = _mapping_df().set_index("NAME", drop=False)
    names = sorted(df.index)
    recipe = cs.production_combined_recipe(RECIPE)
    need = bcs.required_band_cells(df, names)
    assert set(need) == _f01_cells(_mapping_df())
    with pytest.raises(RuntimeError, match=rf"band cells missing for {len(need)}/{len(need)} cells.*skycell\.") as ei:
        LD.check_band_cells_available(tmp_path, df, names, recipe)
    assert "publish_band_cells" in str(ei.value)
    _run_skip(tmp_path, monkeypatch, publish_band_cells=True)  # publishes exactly the f01 set
    assert LD.check_band_cells_available(tmp_path, df, names, recipe) == need
    # remove one xproj source: the early check names it
    victim = [n for n in need if n not in names][0]
    proj, cell = cs._projection_and_cell(victim)
    d = next(bcs.band_cells_root(tmp_path).joinpath(proj, cell).iterdir())
    shutil.rmtree(d)
    with pytest.raises(RuntimeError, match=victim.replace(".", r"\.")):
        LD.check_band_cells_available(tmp_path, df, names, recipe)


# ------------------------------------------------------------------ 2. provenance records
def _spool_records(data_root):
    out = []
    for p in sorted((Path(data_root) / "bookkeeping" / "spool").glob("*.jsonl")):
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def test_band_record_has_its_own_fingerprint_and_ingest_keeps_both_rows(tmp_path):
    recipe = cs.production_combined_recipe(RECIPE)
    res = pp._materialize_shm_result(pp.process_single_cell(_bundle(_raw_bands(), None, "float32")))
    info = pp.publish_combined_result(str(tmp_path), recipe, res)
    fp = info["fingerprint"]
    band_dir = bcs.band_cell_dir(tmp_path, SPROJ, CELL, fp)
    assert band_dir.name == fp  # on-disk name stays the combined fingerprint
    recs = {r["kind"]: r for r in _spool_records(tmp_path)}
    assert recs["combined_skycell"]["fingerprint"] == fp
    band = recs[bcs.KIND]
    assert band["fingerprint"] != fp and fp in band["inputs"]
    assert band["location"] == str(band_dir)
    assert bcs.band_cell_record_fingerprint(band_dir) == band["fingerprint"]
    assert band["recipe_id"] != recs["combined_skycell"]["recipe_id"]

    store = ProvenanceStore(scc_paths.provenance_db_path(tmp_path))
    drain_spool(store, scc_paths.provenance_spool_dir(tmp_path))
    comb_row, band_row = store.artifact(fp), store.artifact(band["fingerprint"])
    assert comb_row is not None and comb_row.kind == "combined_skycell"
    assert band_row is not None and band_row.kind == bcs.KIND
    assert comb_row.location != band_row.location
    assert fp in store.inputs_of(band["fingerprint"])
    # gc: nothing missing, and no orphan (the dir under the combined root is known by the combined row)
    rep = gc_report(tmp_path)
    assert rep.db_missing_files == [] and rep.orphan_fingerprint_dirs == []


def test_record_fingerprint_is_deterministic_and_distinct():
    recipe = cs.production_combined_recipe(RECIPE)
    a = bcs.band_record_fingerprint(SPROJ, CELL, recipe, ["x"], "FP")
    assert a == bcs.band_record_fingerprint(SPROJ, CELL, recipe, ["x"], "FP")
    assert a[0] != "FP" and a != bcs.band_record_fingerprint(SPROJ, CELL, recipe, ["x"], "FP2")


# ------------------------------------------------------------------ 3. checkpoint keying
def _cp_resolved(tmp_path, **flags):
    r = _resolved(tmp_path)
    r.stages.ps1_process = Ps1ProcessStageParams(
        enable_saturation_correction=False, remove_saturated_stars=False, **flags)
    csv = scc_paths.scc_mapping_master_skycells_csv(r.data_root, 22, 3, 3, oversampling_factor=1)
    csv.parent.mkdir(parents=True, exist_ok=True)
    _mapping_df().to_csv(csv, index=False)
    return r


def _publish_all(data_root, with_bands=True):
    recipe = cs.production_combined_recipe(RECIPE)
    for i, name in enumerate(_mapping_df()["NAME"]):
        b = _bundle(_raw_bands(i), None, "float32")
        b["skycell_id"] = name
        if not with_bands:
            b.pop("band_images")
        res = pp._materialize_shm_result(pp.process_single_cell(b))
        assert pp.publish_combined_result(str(data_root), recipe, res)


def test_scc_assembly_is_keyed_on_skip_convolved_and_resolves_from_band_cells(tmp_path):
    r_off = _cp_resolved(tmp_path)
    r_skip = _cp_resolved(tmp_path, skip_convolved=True)
    r_band = _cp_resolved(tmp_path, skip_convolved=True, publish_band_cells=True)
    # an OLD convolved store with a complete-looking cell for every mapping cell is present
    old = scc_paths.ps1_convolved_zarr_path(r_off.data_root)
    for name in _mapping_df()["NAME"]:
        proj, cell = cs._projection_and_cell(name)
        (old / proj / cell / "oldfp").mkdir(parents=True)
    _publish_all(r_off.data_root)
    fp_off = PC.expected_scc_assembly_fingerprint(r_off)
    fp_skip = PC.expected_scc_assembly_fingerprint(r_skip)
    fp_band = PC.expected_scc_assembly_fingerprint(r_band)
    assert len({fp_off, fp_skip, fp_band}) == 3
    # inputs come from the band-cell / combined store, never from the old convolved store
    ins = PC._resolve_convolved_skycell_fingerprints(r_band)
    assert len(ins) == 3 and "oldfp" not in ins
    assert set(ins).isdisjoint(set(PC._resolve_convolved_skycell_fingerprints(r_skip)))
    assert len(PC._resolve_convolved_skycell_fingerprints(r_skip)) == 3
    # a missing band cell: inputs empty, expected fingerprint differs from the complete one
    proj, cell = cs._projection_and_cell(_mapping_df()["NAME"].iloc[1])
    shutil.rmtree(next(bcs.band_cells_root(r_band.data_root).joinpath(proj, cell).iterdir()))
    assert PC._resolve_convolved_skycell_fingerprints(r_band) == []
    assert PC.expected_scc_assembly_fingerprint(r_band) != fp_band
    # the combined-only gate is still complete (band cells are not required there)
    assert PC.expected_scc_assembly_fingerprint(r_skip) == fp_skip


def test_trust_index_not_fooled_by_missing_band_cell(tmp_path):
    r = _cp_resolved(tmp_path, skip_convolved=True, publish_band_cells=True)
    _publish_all(r.data_root)
    PC.emit_scc_assembly_checkpoint(r)
    store = ProvenanceStore(scc_paths.provenance_db_path(r.data_root))
    drain_spool(store, scc_paths.provenance_spool_dir(r.data_root))
    assert PC.checkpoint_stage_indexed(r, "ps1_process")
    proj, cell = cs._projection_and_cell(_mapping_df()["NAME"].iloc[0])
    shutil.rmtree(next(bcs.band_cells_root(r.data_root).joinpath(proj, cell).iterdir()))
    assert not PC.checkpoint_stage_indexed(r, "ps1_process")


def test_flag_off_scc_assembly_recipe_params_unchanged():
    from syndiff_pipeline.common.provenance import model

    r = _resolved(Path("/x"))
    p = model.scc_assembly_recipe_params(r)
    assert "skip_convolved" not in p and "publish_band_cells" not in p
    r.stages.ps1_process = Ps1ProcessStageParams(skip_convolved=True, publish_band_cells=True)
    p = model.scc_assembly_recipe_params(r)
    assert p["skip_convolved"] is True and p["publish_band_cells"] is True


# ------------------------------------------------------------------ 4. writer
def test_write_npz_is_a_normal_npz_at_any_level(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"band_r": rng.normal(size=(30, 40)).astype(np.float32),
              "combined_mask": rng.integers(0, 9, (30, 40)).astype(np.uint16)}
    sizes = {}
    for lvl in (1, 6, 9):
        p = tmp_path / f"a{lvl}.npz"
        bcs._write_npz(p, arrays, lvl)
        with np.load(p, allow_pickle=False) as z:
            assert sorted(z.files) == sorted(arrays)
            for k, v in arrays.items():
                assert z[k].dtype == v.dtype and np.array_equal(z[k], v)
        sizes[lvl] = p.stat().st_size
    assert sizes[9] <= sizes[1]


def test_band_publish_pool_is_bounded_and_close_waits():
    pool = pp._BandPublishPool(workers=2, max_pending=3)
    done, running, peak = [], [0], [0]
    lock = threading.Lock()

    def work(i):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        done.append(i)

    for i in range(10):
        pool.submit(work, i)
    pool.close()
    assert sorted(done) == list(range(10)) and peak[0] <= 2


def test_skip_convolved_pipeline_uses_the_pool_and_publishes_everything(tmp_path, monkeypatch):
    used = []
    real = pp._BandPublishPool.submit
    monkeypatch.setattr(pp._BandPublishPool, "submit",
                        lambda self, fn, *a, **k: (used.append(threading.current_thread().name),
                                                   real(self, fn, *a, **k))[1])
    df, res, loaded = _run_skip(tmp_path, monkeypatch, publish_band_cells=True)
    assert "error" not in res and len(used) == len(_f01_cells(df))
