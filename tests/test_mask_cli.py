"""syndiff mask export CLI and SCC lane catalog helpers."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.difference_imaging.masking.asteroids import (
    BTJD_EPOCH_OFFSET,
    load_asteroid_products,
    normalize_asteroid_times_btjd,
)
from syndiff_pipeline.difference_imaging.masking.cli import (
    _lanes_with_shared_mask,
    _resolve_lane_root,
    cmd_export,
)
from syndiff_pipeline.difference_imaging.masking.ffi_mask import (
    load_catalog_for_scc_lane,
    load_ffi_times_table_for_lane,
)


def _write_minimal_scc_lane(tmp_path: Path) -> tuple[Path, Path]:
    """SCC tree with shared mask, mapping master, asteroids, WCS manifest."""
    data_root = tmp_path / "data"
    scc = data_root / "s0020" / "c3" / "k3"
    lane = scc / "diff_linear"
    lane.mkdir(parents=True)

    grid = MappingGrid(ffi_xmin=0, ffi_ymin=-8, ffi_xmax=8, ffi_ymax=8)
    sci_shape = grid.science_ffi_bounds()["shape"]
    static = np.zeros(sci_shape, dtype=np.int16)
    static[1, 1] = 1
    fits.PrimaryHDU(static).writeto(lane / "shared_mask.fits.fz", overwrite=True)

    mapping = scc / "mapping" / "oversampling_1"
    mapping.mkdir(parents=True)
    hdr = fits.Header()
    for key, val in grid.to_fits_header_updates().items():
        hdr[key] = val
    master = mapping / "tess_s0020_3_3_master_pixels2skycells.fits.fz"
    master_data = np.zeros(grid.array_shape_native(), dtype=np.int32)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(data=master_data, header=hdr)]
    ).writeto(master, overwrite=True)

    ast_dir = (
        data_root
        / "catalogs"
        / "sector_0020"
        / "camera_3"
        / "ccd_3"
        / "asteroids"
    )
    ast_dir.mkdir(parents=True)
    iv = pd.DataFrame(
        {
            "target_id": ["a"],
            "row": [3],
            "col": [3],
            "cadence_lo": [0],
            "cadence_hi": [0],
        }
    )
    tm = pd.DataFrame({"cadence": [0, 1], "btjd": [10.0, 10.02]})
    iv.to_parquet(ast_dir / "pixel_intervals.parquet", index=False)
    tm.to_parquet(ast_dir / "asteroid_ffi_times.parquet", index=False)

    (lane / "wcs").mkdir()
    pd.DataFrame(
        {
            "stem": ["tess1000000000001-s0020-3-3"],
            "btjd": [10.0],
        }
    ).to_csv(lane / "wcs" / "per_ffi_coeffs.csv", index=False)

    deploy = tmp_path / "config" / "deployment.yaml"
    deploy.parent.mkdir(parents=True)
    deploy.write_text(
        f"data_root: {data_root}\nworkspace_root: {tmp_path / 'workspace'}\n"
    )
    return lane, deploy


def test_normalize_asteroid_times_btjd_from_jd():
    jd = 2458897.5
    tm = pd.DataFrame({"cadence": [0, 1], "btjd": [jd, jd + 0.5]})
    out = normalize_asteroid_times_btjd(tm)
    assert out is not None
    assert abs(float(out["btjd"].iloc[0]) - (jd - BTJD_EPOCH_OFFSET)) < 1e-6


def test_normalize_asteroid_times_btjd_idempotent():
    tm = pd.DataFrame({"cadence": [0], "btjd": [1899.31]})
    out = normalize_asteroid_times_btjd(tm)
    assert float(out["btjd"].iloc[0]) == pytest.approx(1899.31)


def test_load_asteroid_products_normalizes_jd(tmp_path):
    iv = pd.DataFrame(
        {
            "target_id": ["a"],
            "row": [50],
            "col": [50],
            "cadence_lo": [0],
            "cadence_hi": [0],
        }
    )
    tm = pd.DataFrame({"cadence": [0], "btjd": [2458897.5]})
    iv.to_parquet(tmp_path / "pixel_intervals.parquet", index=False)
    tm.to_parquet(tmp_path / "asteroid_ffi_times.parquet", index=False)
    _, loaded_tm = load_asteroid_products(tmp_path)
    assert loaded_tm is not None
    assert float(loaded_tm["btjd"].iloc[0]) < 3000.0


def test_lanes_with_shared_mask(tmp_path):
    scc = tmp_path / "s0022" / "c3" / "k3"
    lane_a = scc / "diff_linear"
    lane_b = scc / "diff_field"
    lane_a.mkdir(parents=True)
    lane_b.mkdir(parents=True)
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.int16)).writeto(
        lane_a / "shared_mask.fits.fz", overwrite=True
    )
    hits = _lanes_with_shared_mask(scc)
    assert len(hits) == 1
    assert hits[0][0] == "linear"
    assert hits[0][1] == lane_a


def test_resolve_lane_root_auto(tmp_path):
    scc = tmp_path / "s0022" / "c3" / "k3"
    lane = scc / "diff_linear"
    lane.mkdir(parents=True)
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.int16)).writeto(
        lane / "shared_mask.fits.fz", overwrite=True
    )
    root, store = _resolve_lane_root(
        tmp_path, 22, 3, 3, lane=None, site=None
    )
    assert store == "linear"
    assert root == lane


def test_load_catalog_for_scc_lane(tmp_path):
    lane, _ = _write_minimal_scc_lane(tmp_path)
    data_root = tmp_path / "data"
    cat = load_catalog_for_scc_lane(
        lane, data_root=data_root, sector=20, camera=3, ccd=3
    )
    assert cat.static.shape == (8, 8)
    assert cat.has_temporal()


def test_load_ffi_times_from_wcs_csv(tmp_path):
    lane, _ = _write_minimal_scc_lane(tmp_path)
    manifest = load_ffi_times_table_for_lane(
        lane, data_root=tmp_path / "data", sector=20, camera=3, ccd=3
    )
    assert "btjd" in manifest.columns
    assert manifest.iloc[0]["filename"].startswith("tess")


def test_parse_scc_arg_prefixed():
    from syndiff_pipeline.difference_imaging.masking.cli import _parse_scc_arg

    assert _parse_scc_arg("s0022/c3/k3") == (22, 3, 3)
    assert _parse_scc_arg("22/3/3") == (22, 3, 3)


def test_cmd_export_writes_mask_fits(tmp_path):
    lane, deploy = _write_minimal_scc_lane(tmp_path)
    out_parent = lane / "debug_plots" / "masks"
    args = argparse.Namespace(
        site=str(deploy.parent),
        deployment=str(deploy),
        scc="20/3/3",
        sector=None,
        camera=None,
        ccd=None,
        ffi="tess1000000000001",
        lane="linear",
        out=None,
        which="full",
        overwrite=True,
    )
    with patch(
        "syndiff_pipeline.difference_imaging.masking.cli._default_out_path",
        return_value=out_parent / "mask_full_tess1000000000001.fits",
    ):
        rc = cmd_export(args)
    assert rc == 0
    out = out_parent / "mask_full_tess1000000000001.fits"
    assert out.is_file()
    data = fits.getdata(out)
    assert data[1, 1] == 1
