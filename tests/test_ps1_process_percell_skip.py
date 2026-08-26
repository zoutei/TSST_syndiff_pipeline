"""Tests for the per-cell canonical-skip + local-window convolution change
in ``ps1_process._publish_canonical_convolved_snapshot`` (see
``ps1_process_percell_skip_plan.md``).

Covers:
- zero-compute when every cell in the row is already canonical,
- local ±radius windowed convolution when few cells are missing,
- whole-row fallback when many cells are missing,
- numerical equivalence between the local-window path and a direct
  full-array convolution reference,
- already-canonical cells are never re-published.

Uses small synthetic arrays with a tiny ``psf_sigma``/``radius`` so tests run
fast; the fingerprint chain itself never touches real PS1 data (mirrors
``test_convolved_store_skip_check.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolution_utils
from syndiff_pipeline.template_creation.processing import convolved_store as vs
from syndiff_pipeline.template_creation.processing import ps1_process as pp

_PROJECTION = "skycell.2058"
_PSF_SIGMA = 2.0
_RADIUS = 5
_CELL_W = 50
_CELL_H = 30
_ARRAY_H = 100


def _cell_name(i: int) -> str:
    return f"{_PROJECTION}.{i:03d}"


def _build_state(n_cells: int, array_width: int, rng_seed: int = 0) -> pp.ProcessingState:
    rng = np.random.default_rng(rng_seed)
    array = rng.random((_ARRAY_H, array_width)).astype(np.float32)
    state = pp.ProcessingState(current_array=array)
    y0, y1 = 30, 30 + _CELL_H
    for i in range(n_cells):
        x0 = 20 + i * _CELL_W
        x1 = x0 + _CELL_W
        name = _cell_name(i)
        state.cell_locations[name] = (x0, x1, y0, y1)
        state.current_masks[name] = np.zeros((_CELL_H, _CELL_W), dtype=np.uint16)
        state.cell_metadata[name] = {"headers_data": {}, "removed_stars": []}
    return state


def _publish_combined_for_all(tmp_path: Path, state: pp.ProcessingState, combined_recipe: dict) -> None:
    for name in state.cell_locations:
        _, cell = name.rsplit(".", 1)
        raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, _PROJECTION, cell)
        cs.publish_combined_cell(
            tmp_path,
            _PROJECTION,
            cell,
            combined_image=np.zeros((_CELL_H, _CELL_W), dtype=np.float32),
            combined_mask=np.zeros((_CELL_H, _CELL_W), dtype=np.uint16),
            headers_data={},
            removed_stars=[],
            recipe=combined_recipe,
            input_fingerprints=[raw_fp],
        )


def _mark_canonical(tmp_path: Path, cell: str, combined_recipe: dict, convolved_recipe: dict) -> None:
    """Publish a (dummy-content) convolved record for ``cell`` under the
    exact fingerprint chain that would make it "already canonical"."""
    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, _PROJECTION, cell)
    combined_fp = cs.resolve_combined_fingerprint_for_recipe(
        tmp_path, _PROJECTION, cell, combined_recipe, raw_fp=raw_fp,
    )
    assert combined_fp is not None
    vs.publish_convolved_cell(
        tmp_path,
        _PROJECTION,
        cell,
        convolved_image=np.zeros((_CELL_H, _CELL_W), dtype=np.float32),
        convolved_mask=np.zeros((_CELL_H, _CELL_W), dtype=np.uint16),
        headers_data={},
        removed_stars=[],
        recipe=convolved_recipe,
        combined_fingerprint=combined_fp,
    )


def test_zero_convolution_calls_when_fully_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe(psf_sigma=_PSF_SIGMA, radius=_RADIUS)

    state = _build_state(n_cells=4, array_width=20 + 4 * _CELL_W + 20)
    _publish_combined_for_all(tmp_path, state, combined_recipe)
    for name in state.cell_locations:
        _, cell = name.rsplit(".", 1)
        _mark_canonical(tmp_path, cell, combined_recipe, convolved_recipe)

    calls = []
    real = convolution_utils.apply_gaussian_convolution

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(convolution_utils, "apply_gaussian_convolution", _spy)

    pp._publish_canonical_convolved_snapshot(
        state, _PROJECTION, _PSF_SIGMA, str(tmp_path), combined_recipe, convolved_recipe,
    )

    assert calls == []


def test_local_window_path_used_for_few_missing_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe(psf_sigma=_PSF_SIGMA, radius=_RADIUS)

    n_cells = 6  # 1/6 missing < ROW_FALLBACK_THRESHOLD (0.2) -> local-window path
    state = _build_state(n_cells=n_cells, array_width=20 + n_cells * _CELL_W + 20)
    _publish_combined_for_all(tmp_path, state, combined_recipe)
    for i in range(n_cells - 1):
        _mark_canonical(tmp_path, f"{i:03d}", combined_recipe, convolved_recipe)
    # cell n_cells-1 left un-published -> the only "missing" cell this row.

    whole_row_calls = []
    real_whole_row = pp._convolve_whole_row_snapshot

    def _spy_whole_row(*args, **kwargs):
        whole_row_calls.append(1)
        return real_whole_row(*args, **kwargs)

    monkeypatch.setattr(pp, "_convolve_whole_row_snapshot", _spy_whole_row)

    published = {}
    real_publish = vs.publish_convolved_cell

    def _spy_publish(data_root, projection, cell, **kwargs):
        published[cell] = kwargs
        return real_publish(data_root, projection, cell, **kwargs)

    monkeypatch.setattr(vs, "publish_convolved_cell", _spy_publish)

    pp._publish_canonical_convolved_snapshot(
        state, _PROJECTION, _PSF_SIGMA, str(tmp_path), combined_recipe, convolved_recipe,
    )

    assert whole_row_calls == []
    # Only the missing cell should have been (re-)published.
    assert list(published.keys()) == [f"{n_cells - 1:03d}"]

    for i in range(n_cells):
        cell = f"{i:03d}"
        assert vs.skycell_already_canonical(
            tmp_path, _PROJECTION, cell, combined_recipe, convolved_recipe,
        ) is True


def test_whole_row_fallback_used_when_many_cells_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe(psf_sigma=_PSF_SIGMA, radius=_RADIUS)

    n_cells = 2  # 1/2 missing >= ROW_FALLBACK_THRESHOLD (0.2) -> whole-row fallback
    state = _build_state(n_cells=n_cells, array_width=20 + n_cells * _CELL_W + 20)
    _publish_combined_for_all(tmp_path, state, combined_recipe)
    _mark_canonical(tmp_path, "000", combined_recipe, convolved_recipe)
    # cell "001" left un-published -> missing.

    local_calls = []
    real_local = pp._convolve_local_windows_for_missing_cells

    def _spy_local(*args, **kwargs):
        local_calls.append(1)
        return real_local(*args, **kwargs)

    monkeypatch.setattr(pp, "_convolve_local_windows_for_missing_cells", _spy_local)

    pp._publish_canonical_convolved_snapshot(
        state, _PROJECTION, _PSF_SIGMA, str(tmp_path), combined_recipe, convolved_recipe,
    )

    assert local_calls == []
    assert vs.skycell_already_canonical(tmp_path, _PROJECTION, "001", combined_recipe, convolved_recipe) is True


def test_local_window_result_matches_full_array_convolution(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe(psf_sigma=_PSF_SIGMA, radius=_RADIUS)

    n_cells = 6
    state = _build_state(n_cells=n_cells, array_width=20 + n_cells * _CELL_W + 20, rng_seed=42)
    _publish_combined_for_all(tmp_path, state, combined_recipe)
    for i in range(n_cells - 1):
        _mark_canonical(tmp_path, f"{i:03d}", combined_recipe, convolved_recipe)
    missing_cell_name = _cell_name(n_cells - 1)
    x0, x1, y0, y1 = state.cell_locations[missing_cell_name]

    # Reference: convolve the whole array directly and crop the cell.
    reference_full = convolution_utils.apply_gaussian_convolution(
        state.current_array.copy(), sigma=_PSF_SIGMA, radius=_RADIUS,
    )
    reference_cell = reference_full[y0:y1, x0:x1]

    pp._publish_canonical_convolved_snapshot(
        state, _PROJECTION, _PSF_SIGMA, str(tmp_path), combined_recipe, convolved_recipe,
    )

    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, _PROJECTION, f"{n_cells - 1:03d}")
    combined_fp = cs.resolve_combined_fingerprint_for_recipe(
        tmp_path, _PROJECTION, f"{n_cells - 1:03d}", combined_recipe, raw_fp=raw_fp,
    )
    convolved_fp = vs.resolve_convolved_fingerprint_for_recipe(
        tmp_path, _PROJECTION, f"{n_cells - 1:03d}", convolved_recipe, combined_fp,
    )
    loaded = vs.try_load_convolved_cell(tmp_path, _PROJECTION, f"{n_cells - 1:03d}", convolved_fp)
    assert loaded is not None
    np.testing.assert_allclose(loaded["convolved_image"], reference_cell, rtol=1e-5, atol=1e-6)
