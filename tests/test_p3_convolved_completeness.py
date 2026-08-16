from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.template_creation.processing.convolved_store import (
    convolved_store_inventory,
    publish_convolved_cell,
)
from syndiff_pipeline.template_creation.processing.ps1_process import (
    expected_convolved_skycells,
    master_skycell_inventory,
)


def _write_master(path: Path, *, mapgrid: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "NAME": ["skycell.1234.001", "skycell.1234.002", "skycell.1234.003"],
            "projection": ["skycell.1234"] * 3,
            "MAPGRID": [mapgrid] * 3,
            "GEOMFP": ["geom-v3"] * 3,
            "COORDFRM": ["full_ffi"] * 3,
        }
    ).to_csv(path, index=False)


def test_os_master_inventory_is_exact_and_requires_mapgrid3(tmp_path: Path):
    csv = tmp_path / "master.csv"
    _write_master(csv)
    assert expected_convolved_skycells(
        str(tmp_path), 1, 1, 1, oversampling_factor=4, mapping_csv_path=str(csv)
    ) == ["skycell.1234.001", "skycell.1234.002", "skycell.1234.003"]
    inv = master_skycell_inventory(
        str(tmp_path), 1, 1, 1, oversampling_factor=4, mapping_csv_path=str(csv)
    )
    assert inv["count"] == 3
    assert inv["geometry_fingerprint"] == "geom-v3"

    _write_master(csv, mapgrid=2)
    with pytest.raises(ValueError, match="MAPGRID=3"):
        expected_convolved_skycells(
            str(tmp_path), 1, 1, 1, oversampling_factor=4, mapping_csv_path=str(csv)
        )


def test_convolved_inventory_reports_missing_edge_and_invalid_provenance(tmp_path: Path):
    root = tmp_path / "data"
    recipe = {"psf_sigma": 60.0, "radius": 150, "mode": "nearest", "padding": "same_projection_only"}
    publish_convolved_cell(
        root,
        "skycell.1234",
        "001",
        convolved_image=np.ones((4, 4), dtype=np.float32),
        convolved_mask=np.zeros((4, 4), dtype=np.uint16),
        headers_data={},
        removed_stars=[],
        recipe=recipe,
        combined_fingerprint="combined-fp",
    )
    report = convolved_store_inventory(
        root,
        ["skycell.1234.001", "skycell.1234.002", "skycell.1234.999"],
    )
    assert report["present"] == ["skycell.1234.001"]
    assert report["missing"] == ["skycell.1234.002", "skycell.1234.999"]

    # A payload without its provenance sidecar is not availability.
    cell_root = root / "ps1_skycells_zarr" / "ps1_convolved.zarr" / "skycell.1234" / "001"
    dirs = [p for p in cell_root.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    dirs = list(dirs[0].iterdir())
    sidecar = next(p for p in dirs if p.name == "_provenance.json")
    sidecar.unlink()
    report = convolved_store_inventory(root, ["skycell.1234.001"])
    assert report["missing"] == ["skycell.1234.001"]
