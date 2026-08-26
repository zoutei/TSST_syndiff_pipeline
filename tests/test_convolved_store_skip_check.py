"""Tests for ``convolved_store.skycell_already_canonical`` -- the per-cell
canonical-skip check used by the ``ps1_process`` percell-skip optimization
(see ``ps1_process_percell_skip_plan.md``).

These tests never touch real PS1 data: ``raw_skycell_input_fingerprint``
degrades gracefully to a stable ``{"status": "missing"}`` token when the raw
zarr group isn't on disk, which is enough to exercise the full fingerprint
chain deterministically against ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolved_store as vs

_PROJECTION = "skycell.1234"
_CELL = "000"


def _publish_combined(tmp_path: Path, recipe: dict) -> str:
    rng = np.random.default_rng(0)
    combined_image = rng.random((16, 16)).astype(np.float32)
    combined_mask = rng.integers(0, 4, size=(16, 16)).astype(np.uint16)
    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, _PROJECTION, _CELL)
    info = cs.publish_combined_cell(
        tmp_path,
        _PROJECTION,
        _CELL,
        combined_image=combined_image,
        combined_mask=combined_mask,
        headers_data={"r": "R"},
        removed_stars=[],
        recipe=recipe,
        input_fingerprints=[raw_fp],
    )
    assert info is not None
    return info["fingerprint"]


def _publish_convolved(tmp_path: Path, recipe: dict, combined_fp: str) -> str:
    rng = np.random.default_rng(1)
    convolved_image = rng.random((16, 16)).astype(np.float32)
    convolved_mask = rng.integers(0, 4, size=(16, 16)).astype(np.uint16)
    info = vs.publish_convolved_cell(
        tmp_path,
        _PROJECTION,
        _CELL,
        convolved_image=convolved_image,
        convolved_mask=convolved_mask,
        headers_data={"r": "R"},
        removed_stars=[],
        recipe=recipe,
        combined_fingerprint=combined_fp,
    )
    assert info is not None
    return info["fingerprint"]


def test_not_canonical_when_nothing_published(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, combined_recipe, convolved_recipe,
    ) is False


def test_not_canonical_when_only_combined_published(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    _publish_combined(tmp_path, combined_recipe)
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, combined_recipe, convolved_recipe,
    ) is False


def test_canonical_when_both_published_under_matching_recipes(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    combined_fp = _publish_combined(tmp_path, combined_recipe)
    _publish_convolved(tmp_path, convolved_recipe, combined_fp)
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, combined_recipe, convolved_recipe,
    ) is True


def test_not_canonical_when_combined_recipe_differs(tmp_path: Path):
    published_combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    combined_fp = _publish_combined(tmp_path, published_combined_recipe)
    _publish_convolved(tmp_path, convolved_recipe, combined_fp)

    # Caller's own combined recipe differs (e.g. a different gaia_version) --
    # must not be treated as canonical even though *a* convolved payload
    # exists for this cell under a different upstream recipe.
    different_combined_recipe = cs.combined_recipe(gaia_version="dr3-mismatch")
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, different_combined_recipe, convolved_recipe,
    ) is False


def test_not_canonical_when_convolved_recipe_differs(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    published_convolved_recipe = vs.convolved_recipe()
    combined_fp = _publish_combined(tmp_path, combined_recipe)
    _publish_convolved(tmp_path, published_convolved_recipe, combined_fp)

    different_convolved_recipe = vs.convolved_recipe(psf_sigma=45.0)
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, combined_recipe, different_convolved_recipe,
    ) is False


def test_raw_fp_shortcut_matches_recomputed_value(tmp_path: Path):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()
    combined_fp = _publish_combined(tmp_path, combined_recipe)
    _publish_convolved(tmp_path, convolved_recipe, combined_fp)

    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, _PROJECTION, _CELL)
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, combined_recipe, convolved_recipe, raw_fp=raw_fp,
    ) is True


def test_never_raises_on_internal_failure(tmp_path: Path, monkeypatch):
    combined_recipe = cs.combined_recipe()
    convolved_recipe = vs.convolved_recipe()

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cs, "resolve_combined_fingerprint_for_recipe", _boom)
    assert vs.skycell_already_canonical(
        tmp_path, _PROJECTION, _CELL, combined_recipe, convolved_recipe,
    ) is False
