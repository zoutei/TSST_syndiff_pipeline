"""Regression tests for the shared cross-sector combined/convolved store
recipe-contamination bug.

Both ``ps1_combined.zarr`` and ``ps1_convolved.zarr`` are content-addressed
by fingerprint but shared across sectors/runs: an unrelated run can publish
a *different* recipe (e.g. ``remove_saturated_stars=False``) for the exact
same sky cell, and that publish can easily have a newer mtime than the
correct one. Before the deterministic recipe-matched resolvers existed,
``field_downsample._discover_shared_convolved_fp`` and
``padding_correction._discover_shared_combined_fp`` picked "whichever
fingerprint is newest", which could silently reintroduce saturated stars
that ``ps1_process`` had already correctly removed. These tests assert the
newest-mtime entry never wins once a caller supplies its own recipe.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import convolved_store as vs
from syndiff_pipeline.template_creation.processing import field_downsample as fd
from syndiff_pipeline.template_creation.processing import padding_correction as pc

_PROJECTION = "skycell.2556"
_CELL = "080"
_SKYCELL = f"{_PROJECTION}.{_CELL}"


def _touch_older(path: Path, seconds: float) -> None:
    """Push a published fingerprint dir's mtime into the past."""
    now = os.stat(path).st_mtime
    older = now - seconds
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            os.utime(p, (older, older))
    os.utime(path, (older, older))


def _publish_combined(tmp_path: Path, *, remove_saturated_stars: bool) -> dict:
    recipe = cs.combined_recipe(remove_saturated_stars=remove_saturated_stars)
    raw_fp = cs.raw_skycell_input_fingerprint(tmp_path, _PROJECTION, _CELL)
    info = cs.publish_combined_cell(
        tmp_path,
        _PROJECTION,
        _CELL,
        combined_image=np.full((4, 4), 1.0 if remove_saturated_stars else 999.0, dtype=np.float64),
        combined_mask=np.zeros((4, 4), dtype=np.uint16),
        headers_data={},
        removed_stars=[],
        recipe=recipe,
        input_fingerprints=[raw_fp],
        producer="test",
    )
    assert info is not None
    return info


def _publish_convolved(tmp_path: Path, *, combined_fp: str, psf_sigma: float) -> dict:
    recipe = vs.convolved_recipe(psf_sigma=psf_sigma)
    info = vs.publish_convolved_cell(
        tmp_path,
        _PROJECTION,
        _CELL,
        convolved_image=np.full((4, 4), psf_sigma, dtype=np.float32),
        convolved_mask=np.zeros((4, 4), dtype=np.uint16),
        headers_data={},
        removed_stars=[],
        recipe=recipe,
        combined_fingerprint=combined_fp,
    )
    assert info is not None
    return info


def test_combined_discovery_prefers_recipe_match_over_newer_wrong_mtime(tmp_path: Path):
    correct = _publish_combined(tmp_path, remove_saturated_stars=True)
    # Simulate the correct publish happening first, then an unrelated run
    # publishing a different (wrong-for-us) recipe later -- newer mtime.
    wrong = _publish_combined(tmp_path, remove_saturated_stars=False)
    assert correct["fingerprint"] != wrong["fingerprint"]

    my_recipe = cs.combined_recipe(remove_saturated_stars=True)
    fp = pc._discover_shared_combined_fp(
        tmp_path, _PROJECTION, _CELL, combined_recipe=my_recipe,
    )
    assert fp == correct["fingerprint"]

    image = pc._load_combined_image(tmp_path, _PROJECTION, _CELL, combined_recipe=my_recipe)
    assert image is not None
    np.testing.assert_array_equal(image, np.full((4, 4), 1.0))


def test_combined_discovery_without_recipe_falls_back_to_mtime_with_warning(
    tmp_path: Path, caplog,
):
    correct = _publish_combined(tmp_path, remove_saturated_stars=True)
    _touch_older(correct["dir"], seconds=3600)
    wrong = _publish_combined(tmp_path, remove_saturated_stars=False)
    assert correct["fingerprint"] != wrong["fingerprint"]

    with caplog.at_level("WARNING"):
        fp = pc._discover_shared_combined_fp(tmp_path, _PROJECTION, _CELL)
    # No recipe context at all: legacy newest-mtime fallback still applies.
    assert fp == wrong["fingerprint"]


def test_convolved_discovery_prefers_recipe_match_over_newer_wrong_mtime(tmp_path: Path):
    combined_info = _publish_combined(tmp_path, remove_saturated_stars=True)
    correct = _publish_convolved(
        tmp_path, combined_fp=combined_info["fingerprint"], psf_sigma=40.0,
    )
    wrong = _publish_convolved(
        tmp_path, combined_fp=combined_info["fingerprint"], psf_sigma=60.0,
    )
    assert correct["fingerprint"] != wrong["fingerprint"]

    my_combined_recipe = cs.combined_recipe(remove_saturated_stars=True)
    fp = fd._discover_shared_convolved_fp(
        tmp_path, _PROJECTION, _CELL,
        psf_sigma=40.0, combined_recipe=my_combined_recipe,
    )
    assert fp == correct["fingerprint"]

    loaded = fd._try_load_shared_convolved_arrays(
        tmp_path, _SKYCELL, psf_sigma=40.0, combined_recipe=my_combined_recipe,
    )
    assert loaded is not None
    data, _mask = loaded
    np.testing.assert_array_equal(data, np.full((4, 4), 40.0, dtype=np.float32))


def test_convolved_discovery_without_recipe_falls_back_to_mtime_with_warning(
    tmp_path: Path,
):
    combined_info = _publish_combined(tmp_path, remove_saturated_stars=True)
    correct = _publish_convolved(
        tmp_path, combined_fp=combined_info["fingerprint"], psf_sigma=40.0,
    )
    _touch_older(correct["dir"], seconds=3600)
    wrong = _publish_convolved(
        tmp_path, combined_fp=combined_info["fingerprint"], psf_sigma=60.0,
    )
    assert correct["fingerprint"] != wrong["fingerprint"]

    fp = fd._discover_shared_convolved_fp(tmp_path, _PROJECTION, _CELL)
    assert fp == wrong["fingerprint"]


def test_convolved_discovery_with_missing_recipe_never_uses_pointer_or_mtime(tmp_path: Path):
    """A production L5 call must fail its audit, not borrow another recipe."""
    wrong_combined = _publish_combined(tmp_path, remove_saturated_stars=False)
    wrong = _publish_convolved(
        tmp_path, combined_fp=wrong_combined["fingerprint"], psf_sigma=40.0,
    )

    requested = cs.combined_recipe(remove_saturated_stars=True)
    assert fd._discover_shared_convolved_fp(
        tmp_path, _PROJECTION, _CELL, psf_sigma=40.0, combined_recipe=requested,
    ) is None

    payload = {
        "data_root": str(tmp_path),
        "shared_convolved_store": True,
        "psf_sigma": 40.0,
        "combined_recipe": requested,
        "legacy_zarr_path": None,
    }
    assert not fd._convolved_skycell_available(payload, _SKYCELL)
    assert wrong["fingerprint"]


def test_padding_combined_discovery_with_missing_recipe_never_uses_pointer_or_mtime(tmp_path: Path):
    _publish_combined(tmp_path, remove_saturated_stars=False)
    requested = cs.combined_recipe(remove_saturated_stars=True)
    assert pc._discover_shared_combined_fp(
        tmp_path, _PROJECTION, _CELL, combined_recipe=requested,
    ) is None


def test_l5_completeness_audit_requires_the_requested_shared_recipe(tmp_path: Path):
    """The pre-worker audit rejects a complete-but-wrong shared artifact."""
    wrong_combined = _publish_combined(tmp_path, remove_saturated_stars=False)
    _publish_convolved(tmp_path, combined_fp=wrong_combined["fingerprint"], psf_sigma=40.0)
    requested = cs.combined_recipe(remove_saturated_stars=True)
    payload = {
        "data_root": str(tmp_path),
        "shared_convolved_store": True,
        "psf_sigma": 40.0,
        "combined_recipe": requested,
    }

    with pytest.raises(fd.L5CompletenessError) as excinfo:
        fd._validate_l5_convolved_completeness(
            master_skycells={_SKYCELL},
            required_skycells={_SKYCELL},
            payload=payload,
        )
    assert excinfo.value.diagnostics["absent_from_store"] == [_SKYCELL]
