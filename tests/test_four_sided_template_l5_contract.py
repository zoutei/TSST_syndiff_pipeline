"""Focused MAPGRID=3 L5/template-support contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.template_creation.processing.field_templates import (
    validate_frozen_field_geometry,
    write_contrib,
)


def _v3_grid() -> MappingGrid:
    return MappingGrid.from_science_bounds(2, 0, 6, 4, pad=1, oversampling=1)


def _sidecar(grid: MappingGrid) -> dict:
    return {
        "schema_version": 4,
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
    }


def test_mapgrid3_l5_sidecar_requires_neutral_invalid_policy(tmp_path: Path) -> None:
    grid = _v3_grid()
    side = _sidecar(grid)
    side.pop("science_pad_policy")
    (tmp_path / "field_mode_assembly.json").write_text(json.dumps(side))
    with pytest.raises(ValueError, match="neutral_invalid"):
        validate_frozen_field_geometry(tmp_path, grid)


def test_mapgrid3_l5_sidecar_accepts_exact_template_support(tmp_path: Path) -> None:
    grid = _v3_grid()
    (tmp_path / "field_mode_assembly.json").write_text(json.dumps(_sidecar(grid)))
    loaded = validate_frozen_field_geometry(tmp_path, grid)
    assert loaded["template_support_bounds_ffi"] == {
        "x_min": 1,
        "x_max": 7,
        "y_min": -1,
        "y_max": 5,
    }


def test_mapgrid3_support_shape_is_not_science_shape() -> None:
    grid = _v3_grid()
    assert grid.array_shape_native() == (6, 6)  # T
    assert (grid.science_ymax - grid.science_ymin,
            grid.science_xmax - grid.science_xmin) == (4, 4)  # S
    assert grid.science_slice_native() == (slice(1, 5), slice(1, 5))
