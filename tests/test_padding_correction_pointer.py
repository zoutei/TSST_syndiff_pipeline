"""``padding_correction._discover_shared_combined_fp`` fingerprint discovery.

Prefers the ``combined_store`` "current" pointer selection when one has
been published, but falls back to the newest complete payload by mtime --
mirroring ``field_downsample._discover_shared_convolved_fp``'s discovery for
the convolved store. This fallback is required in practice: nothing in
production calls ``combined_store.update_current_pointer`` today (it exists
but is never wired into ``ps1_process``), so a strict pointer-only
requirement would make every published combined cell permanently
undiscoverable and the cross-projection correction unusable on real data.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import padding_correction as pc


def _publish(tmp_path: Path, *, projection: str = "1234", cell: str = "070", raw_fp: str = "raw_a"):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint16)
    recipe = cs.combined_recipe(gaia_version="none")
    info = cs.publish_combined_cell(
        tmp_path,
        projection,
        cell,
        combined_image=image,
        combined_mask=mask,
        headers_data={"r": "R"},
        removed_stars=[],
        recipe=recipe,
        input_fingerprints=[raw_fp],
        producer="test",
    )
    assert info is not None
    return image, mask, info


def test_discover_returns_none_when_never_published(tmp_path: Path):
    assert pc._discover_shared_combined_fp(tmp_path, "1234", "070") is None


def test_discover_falls_back_to_mtime_when_pointer_never_set(tmp_path: Path):
    """A real published cell without a pointer update must still be
    discoverable via the mtime fallback -- production never writes the
    pointer, so requiring one would make the correction unusable."""
    _image, _mask, info = _publish(tmp_path)
    assert pc._discover_shared_combined_fp(tmp_path, "1234", "070") == info["fingerprint"]


def test_discover_returns_current_pointer_fingerprint(tmp_path: Path):
    _image, _mask, info = _publish(tmp_path)
    ok = cs.update_current_pointer(tmp_path, "1234", "070", info["fingerprint"])
    assert ok is True

    fp = pc._discover_shared_combined_fp(tmp_path, "1234", "070")
    assert fp == info["fingerprint"]


def test_discover_ignores_newer_mtime_non_current_fingerprint(tmp_path: Path):
    """The actual regression this fix guards against: a second, newer-mtime
    publish (e.g. a stale/losing concurrent writer, or an old schema-version
    cell touched by an unrelated process) must NOT be picked over the
    pointer's actual target, even though it's newer on disk."""
    _image1, _mask1, info1 = _publish(tmp_path, raw_fp="raw_a")
    ok = cs.update_current_pointer(tmp_path, "1234", "070", info1["fingerprint"])
    assert ok is True

    # A second, different-content publish under the SAME (projection, cell)
    # but a different raw_fp -> different fingerprint, definitely newer mtime,
    # but current.json still points at info1's fingerprint.
    time.sleep(0.01)
    _image2, _mask2, info2 = _publish(tmp_path, raw_fp="raw_b_newer")
    assert info2["fingerprint"] != info1["fingerprint"]

    fp = pc._discover_shared_combined_fp(tmp_path, "1234", "070")
    assert fp == info1["fingerprint"], "must follow the pointer, not the newest-mtime fingerprint dir"


def test_load_combined_image_uses_pointer(tmp_path: Path):
    image, _mask, info = _publish(tmp_path)
    ok = cs.update_current_pointer(tmp_path, "1234", "070", info["fingerprint"])
    assert ok is True

    loaded = pc._load_combined_image(tmp_path, "1234", "070")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, image.astype(np.float64))


def test_load_combined_image_falls_back_to_mtime_without_pointer(tmp_path: Path):
    image, _mask, _info = _publish(tmp_path)
    loaded = pc._load_combined_image(tmp_path, "1234", "070")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, image.astype(np.float64))
