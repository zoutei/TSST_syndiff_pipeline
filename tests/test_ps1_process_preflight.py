"""Tests for the ps1_process pre-launch delta/small-job policy
(``ps1_process_preflight.plan_ps1_process_launch``): skip when every OS-target
skycell is already canonical in the shared store, size a small Condor job
when only a handful are missing, and fail closed to the full profile
whenever the shared-store path can't be established.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.orchestration import ps1_process_preflight as pf
from syndiff_pipeline.template_creation.orchestration.stage_params import Ps1ProcessStageParams
from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolved_store as vs

SECTOR, CAMERA, CCD = 51, 4, 2


def _params(**overrides) -> Ps1ProcessStageParams:
    return Ps1ProcessStageParams(use_shared_convolved_store=True, write_per_scc_convolved_zarr=False, **overrides)


def _publish_canonical(tmp_path: Path, projection: str, cell: str, combined_recipe: dict, convolved_recipe: dict) -> None:
    rng = np.random.default_rng(hash((projection, cell)) % (2**32))
    combined_image = rng.random((8, 8)).astype(np.float32)
    combined_mask = rng.integers(0, 4, size=(8, 8)).astype(np.uint16)
    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, projection, cell)
    info = cs.publish_combined_cell(
        tmp_path, projection, cell,
        combined_image=combined_image, combined_mask=combined_mask,
        headers_data={"r": "R"}, removed_stars=[],
        recipe=combined_recipe, input_fingerprints=[raw_fp],
    )
    assert info is not None
    published = vs.publish_convolved_cell(
        tmp_path, projection, cell,
        convolved_image=combined_image, convolved_mask=combined_mask,
        headers_data={"r": "R"}, removed_stars=[],
        recipe=convolved_recipe, combined_fingerprint=info["fingerprint"],
    )
    assert published is not None


def _patch_expected_cells(monkeypatch, target_cells: list[str], os1_cells: list[str]) -> None:
    def _fake(data_root, sector, camera, ccd, *, oversampling_factor=1, mapping_csv_path=None, projections_limit=None):
        return list(target_cells) if int(oversampling_factor) != 1 else list(os1_cells)

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.ps1_process.expected_convolved_skycells",
        _fake,
    )


def test_skip_when_all_cells_already_canonical(tmp_path, monkeypatch):
    cells = ["skycell.2333.090", "skycell.2333.091"]
    _patch_expected_cells(monkeypatch, target_cells=cells, os1_cells=cells)
    params = _params()
    combined_recipe = cs.production_combined_recipe(params, data_root=tmp_path, sector=SECTOR, camera=CAMERA, ccd=CCD)
    convolved_recipe = vs.convolved_recipe(params)
    for cell_name in cells:
        _publish_canonical(tmp_path, "skycell.2333", cell_name.split(".")[-1], combined_recipe, convolved_recipe)

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_SKIP
    assert plan.missing_cells == frozenset()
    assert plan.resources is None


def test_small_job_when_one_cell_missing(tmp_path, monkeypatch):
    cells = ["skycell.2333.090", "skycell.2333.091"]
    os1_cells = ["skycell.2333.090"]  # OS4-only: skycell.2333.091
    _patch_expected_cells(monkeypatch, target_cells=cells, os1_cells=os1_cells)
    params = _params(small_job_max_skycells=32, small_job_min_memory_mb=25_000, small_job_memory_per_skycell_mb=2_500)
    combined_recipe = cs.production_combined_recipe(params, data_root=tmp_path, sector=SECTOR, camera=CAMERA, ccd=CCD)
    convolved_recipe = vs.convolved_recipe(params)
    _publish_canonical(tmp_path, "skycell.2333", "090", combined_recipe, convolved_recipe)
    # "091" deliberately left unpublished -- it's the missing cell.

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_SMALL
    assert plan.missing_cells == frozenset({"skycell.2333.091"})
    assert plan.target_only_cells == frozenset({"skycell.2333.091"})
    assert plan.resources is not None
    assert plan.resources.request_memory_mb == 25_000  # floor: 1 cell * 2500 < 25000 floor
    assert plan.resources.request_cpus == 1


def test_small_job_memory_scales_with_missing_count(tmp_path, monkeypatch):
    cells = [f"skycell.2333.{i:03d}" for i in range(20)]
    _patch_expected_cells(monkeypatch, target_cells=cells, os1_cells=[])
    params = _params(small_job_max_skycells=32, small_job_request_cpus=16)
    # Nothing published: all 20 cells are "missing".

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_SMALL
    assert len(plan.missing_cells) == 20
    assert plan.resources.request_memory_mb == 50_000  # 20 * 2500
    assert plan.resources.request_cpus == 16  # min(16, 20)


def test_full_when_missing_exceeds_small_job_ceiling(tmp_path, monkeypatch):
    cells = [f"skycell.2333.{i:03d}" for i in range(5)]
    _patch_expected_cells(monkeypatch, target_cells=cells, os1_cells=[])
    params = _params(small_job_max_skycells=2)

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_FULL
    assert len(plan.missing_cells) == 5
    assert plan.resources is None


def test_full_when_shared_store_disabled(tmp_path, monkeypatch):
    cells = ["skycell.2333.090"]
    _patch_expected_cells(monkeypatch, target_cells=cells, os1_cells=cells)
    params = Ps1ProcessStageParams(use_shared_convolved_store=False)

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_FULL
    assert "disabled" in plan.reason


def test_full_closed_when_mapping_inventory_unresolvable(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no mapping csv")

    monkeypatch.setattr(
        "syndiff_pipeline.template_creation.processing.ps1_process.expected_convolved_skycells",
        _raise,
    )
    params = _params()

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_FULL
    assert plan.missing_cells == frozenset()
    assert "could not resolve" in plan.reason


def test_small_job_disabled_via_zero_ceiling(tmp_path, monkeypatch):
    cells = ["skycell.2333.090"]
    _patch_expected_cells(monkeypatch, target_cells=cells, os1_cells=[])
    params = _params(small_job_max_skycells=0)

    plan = pf.plan_ps1_process_launch(
        data_root=str(tmp_path), sector=SECTOR, camera=CAMERA, ccd=CCD,
        oversampling_factor=4, params=params,
    )

    assert plan.decision == pf.DECISION_FULL
