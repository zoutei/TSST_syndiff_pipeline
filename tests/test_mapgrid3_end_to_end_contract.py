"""Synthetic end-to-end verifier for the MAPGRID=3 paired-padding handoff.

This test intentionally stays at the artifact/array-contract level.  It does
not run a real SCC, download data, invoke Hotpants, or exercise a production
OS4 lane.  The fixture connects the published master geometry, the remap and
L5 provenance gates, the convolved-cell inventory, and the final paired
science/template pad-and-trim operation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from syndiff_pipeline.common.grid_pairing import (
    pad_mask_to_template,
    prepare_science_template_pairing,
    trim_padded_products,
)
from syndiff_pipeline.common.mapping_grid import (
    MappingGrid,
    load_mapping_grid_from_master,
)
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    _validate_template_remap_provenance,
)
from syndiff_pipeline.template_creation.processing.convolved_store import (
    convolved_store_inventory,
    publish_convolved_cell,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    validate_frozen_field_geometry,
)
from syndiff_pipeline.template_creation.processing.pancakes import (
    master_skycells_csv_paths,
    save_master_mapping,
)


def _grid() -> MappingGrid:
    # Deliberately asymmetric S/T dimensions make accidental one-sided crops
    # and transposed slices visible in the assertions below.
    return MappingGrid.from_science_bounds(10, 0, 18, 6, pad=2, oversampling=1)


def _field_sidecar(grid: MappingGrid, *, temporal: bool = True) -> dict:
    return {
        "schema_version": 4,
        "geometry_mode": "temporal_wcs" if temporal else "field",
        "mapping_grid": grid.to_mapping_dict(),
        "base_tess_shape": list(grid.array_shape_native()),
        "oversampling_factor": 1,
        "science_pad_policy": "neutral_invalid",
        "template_support_bounds_ffi": {
            "x_min": grid.template_xmin,
            "x_max": grid.template_xmax,
            "y_min": grid.template_ymin,
            "y_max": grid.template_ymax,
        },
        "pad_native": {
            "left": grid.pad_left,
            "right": grid.pad_right,
            "bottom": grid.pad_bottom,
            "top": grid.pad_top,
        },
        "geometry_provenance": {
            "temporal_wcs_fingerprint": "wcs-v3",
            "temporal_wcs_frame_contract_fingerprint": "frame-v3",
        },
    }


def test_mapgrid3_artifacts_and_pairing_form_one_geometry_contract(tmp_path: Path):
    """Master -> remap/L5 -> inventory -> paired trim preserves T and S."""
    grid = _grid()

    # P2: publish and reload the authoritative full-FFI template-support map.
    mapping_root = tmp_path / "mapping"
    selected = pd.DataFrame({"NAME": ["skycell.1234.001"]})
    save_master_mapping(
        np.zeros(grid.array_shape_native(), dtype=np.int32),
        selected,
        "ffi.fits",
        fits.Header({"SECTOR": 1}),
        grid.array_shape_native(),
        str(mapping_root),
        1,
        1,
        1,
        oversampling_factor=1,
        mapping_grid=grid,
    )
    master = next(mapping_root.rglob("*master_pixels2skycells.fits.fz"))
    loaded = load_mapping_grid_from_master(master)
    assert loaded.geometry_fingerprint == grid.geometry_fingerprint
    assert loaded.template_ffi_bounds()["shape"] == grid.array_shape_native()
    csv_partial, _ = master_skycells_csv_paths(str(mapping_root), 1, 1, 1, 1)
    published = pd.read_csv(csv_partial)
    assert set(published["GEOMFP"].astype(str)) == {grid.geometry_fingerprint}
    assert set(published["SCIENCE_PAD_POLICY"].astype(str)) == {"neutral_invalid"}

    # P4/P5: both stores must agree with the same immutable T geometry and
    # temporal provenance before consumers can proceed.
    template_store = tmp_path / "templates"
    template_store.mkdir()
    (template_store / "field_mode_assembly.json").write_text(
        json.dumps(_field_sidecar(grid))
    )
    remap_store = tmp_path / "remap"
    remap_store.mkdir()
    (remap_store / "remap_manifest.json").write_text(
        json.dumps(
            {
                "temporal_wcs_fingerprint": "wcs-v3",
                "temporal_wcs_frame_contract_fingerprint": "frame-v3",
            }
        )
    )
    side = validate_frozen_field_geometry(template_store, grid)
    assert side["template_support_bounds_ffi"]["x_min"] == grid.template_xmin
    _validate_template_remap_provenance(template_store, remap_store, grid)

    # P3: inventory is only complete when each expected cell has a valid
    # provenance sidecar, not merely an image payload.
    convolved_root = tmp_path / "convolved"
    recipe = {"geometry_fingerprint": grid.geometry_fingerprint, "mapgrid": 3}
    publish_convolved_cell(
        convolved_root,
        "skycell.1234",
        "001",
        convolved_image=np.ones((3, 3), dtype=np.float32),
        convolved_mask=np.zeros((3, 3), dtype=np.uint16),
        headers_data={"MAPGRID": 3, "GEOMFP": grid.geometry_fingerprint},
        removed_stars=[],
        recipe=recipe,
        combined_fingerprint="combined-v3",
    )
    inventory = convolved_store_inventory(convolved_root, ["skycell.1234.001"])
    assert inventory["present"] == ["skycell.1234.001"]
    assert inventory["missing"] == []

    # P6: science is neutral-padded and invalid-masked on all four sides,
    # paired with T at exactly equal shape, then trimmed canonically to S.
    science = np.arange(48, dtype=np.float64).reshape(6, 8)
    template = np.arange(120, dtype=np.float64).reshape(grid.array_shape_native())
    science_padded, template_paired = prepare_science_template_pairing(
        science, template, grid
    )
    assert science_padded.shape == template_paired.shape == grid.array_shape_native()
    mask = pad_mask_to_template(np.zeros_like(science, dtype=bool), grid)
    assert np.all(mask[: grid.pad_top, :])
    assert np.all(mask[-grid.pad_bottom :, :])
    assert np.all(mask[:, : grid.pad_left])
    assert np.all(mask[:, -grid.pad_right :])
    assert not np.any(mask[grid.science_slice_native()])
    trimmed = trim_padded_products(template_paired - science_padded, grid=grid)
    assert trimmed.shape == science.shape
    np.testing.assert_array_equal(
        trimmed,
        template[grid.science_slice_native()] - science,
    )


def test_mapgrid3_geometry_and_temporal_provenance_mismatch_fail_closed(tmp_path: Path):
    grid = _grid()
    side = _field_sidecar(grid)
    side["science_pad_policy"] = "zero_valid"
    store = tmp_path / "templates"
    store.mkdir()
    (store / "field_mode_assembly.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="neutral_invalid"):
        validate_frozen_field_geometry(store, grid)

    side["science_pad_policy"] = "neutral_invalid"
    side["mapping_grid"]["geometry_fingerprint"] = "wrong"
    (store / "field_mode_assembly.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="geometry mismatch"):
        validate_frozen_field_geometry(store, grid)

    # Restore geometry and check the independent P4->P5 temporal handoff.
    side = _field_sidecar(grid)
    (store / "field_mode_assembly.json").write_text(json.dumps(side))
    remap = tmp_path / "remap"
    remap.mkdir()
    (remap / "remap_manifest.json").write_text(
        json.dumps({"temporal_wcs_fingerprint": "stale-wcs"})
    )
    with pytest.raises(ValueError, match="temporal_wcs_fingerprint"):
        _validate_template_remap_provenance(store, remap, grid)
