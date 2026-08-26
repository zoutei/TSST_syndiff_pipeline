"""Fail-closed selection tests for immutable shared artifact versions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing.artifact_pointer import ArtifactPointerError


def _published(tmp_path: Path):
    info = cs.publish_combined_cell(
        tmp_path, "1234", "070", combined_image=np.ones((2, 2)),
        combined_mask=np.zeros((2, 2), dtype=np.uint16), headers_data={"r": "R"},
        removed_stars=[], recipe=cs.combined_recipe(gaia_version="none"),
        input_fingerprints=["raw"], producer="test",
    )
    assert info is not None
    return info


def test_dangling_current_pointer_fails_closed(tmp_path: Path):
    info = _published(tmp_path)
    root = Path(info["dir"]).parent
    (root / "current.json").write_text(json.dumps({
        "schema_version": 1, "kind": cs.KIND,
        "spatial_key": {"projection": "1234", "skycell": "070"},
        "fingerprint": "does-not-exist", "recipe_id": info["recipe_id"],
    }))
    with pytest.raises(ArtifactPointerError, match="incomplete"):
        cs.resolve_current_combined_ref(tmp_path, "1234", "070")


def test_pointer_target_sidecar_mismatch_fails_closed(tmp_path: Path):
    info = _published(tmp_path)
    assert cs.update_current_pointer(tmp_path, "1234", "070", info["fingerprint"])
    sidecar_path = Path(info["dir"]) / "_provenance.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["spatial_key"] = {"projection": "wrong", "skycell": "070"}
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ArtifactPointerError, match="provenance mismatch"):
        cs.resolve_current_combined_ref(tmp_path, "1234", "070")
