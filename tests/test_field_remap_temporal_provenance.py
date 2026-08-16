from __future__ import annotations

import json
import os
from pathlib import Path

from syndiff_pipeline.template_creation.processing.field_remap import (
    _mapping_fingerprint,
    _temporal_remap_cache_is_current,
)


def test_mapping_fingerprint_changes_when_regmap_changes(tmp_path: Path):
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    (mapping / "tess_s0020_3_3_master_pixels2skycells_os4.fits").write_bytes(b"master")
    (mapping / "tess_s0020_3_3_master_skycells_list_os4.csv").write_text("NAME\n")
    regmap = mapping / "skycell.regmap.fits"
    regmap.write_bytes(b"one")
    first = _mapping_fingerprint(mapping, 20, 3, 3, oversampling_factor=4)
    regmap.write_bytes(b"two-two")
    os.utime(regmap, None)
    second = _mapping_fingerprint(mapping, 20, 3, 3, oversampling_factor=4)
    assert first != second


def test_temporal_cache_requires_manifest_and_all_provenance(tmp_path: Path):
    expected = {
        "temporal_wcs_version": "v1",
        "temporal_wcs_fingerprint": "wcs-a",
        "mapping_fingerprint": "map-a",
    }
    assert not _temporal_remap_cache_is_current(tmp_path, expected)
    manifest = tmp_path / "remap_manifest.json"
    manifest.write_text(json.dumps(expected))
    assert _temporal_remap_cache_is_current(tmp_path, expected)
    manifest.write_text(json.dumps({**expected, "mapping_fingerprint": "map-b"}))
    assert not _temporal_remap_cache_is_current(tmp_path, expected)
