"""BK-7: field_downsample shared convolved-store cell loading.

When ``use_shared_convolved_store`` is on, ``resolve_downsample_convolved_dir``
points at ``ps1_convolved.zarr`` (``{projection}/{cell}/{fp}/arrays.npz``).
L5 must load via ``convolved_store.try_load_convolved_cell``, not legacy flat
``{skycell}_data`` zarr keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from syndiff_pipeline.common.scc_paths import (
    PS1_CONVOLVED_ZARR_BASENAME,
    ps1_convolved_zarr_path,
)
from syndiff_pipeline.template_creation.processing import convolved_store as vs
from syndiff_pipeline.template_creation.processing import field_downsample as fd


_UPSTREAM_COMBINED_FP = "combined_fp_field_downsample_test"


def _publish_cell(tmp_path: Path, *, projection: str = "skycell.1234", cell: str = "000"):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint16)
    info = vs.publish_convolved_cell(
        tmp_path,
        projection,
        cell,
        convolved_image=image,
        convolved_mask=mask,
        headers_data={"r": "R"},
        removed_stars=[],
        recipe=vs.convolved_recipe(),
        combined_fingerprint=_UPSTREAM_COMBINED_FP,
    )
    assert info is not None
    return image, mask, info


def test_detects_shared_vs_legacy_store_paths(tmp_path: Path):
    shared = ps1_convolved_zarr_path(tmp_path)
    assert fd._is_shared_convolved_store_path(shared)
    assert fd._is_shared_convolved_store_path(tmp_path / PS1_CONVOLVED_ZARR_BASENAME)
    assert not fd._is_shared_convolved_store_path(
        tmp_path / "sector_0020_camera_1_ccd_1.zarr"
    )
    assert not fd._is_shared_convolved_store_path(tmp_path / "convolved.zarr")


def test_shared_store_loader_returns_arrays(tmp_path: Path):
    image, mask, info = _publish_cell(tmp_path)
    skycell = "skycell.1234.000"

    loaded = fd._try_load_shared_convolved_arrays(tmp_path, skycell)
    assert loaded is not None
    data, m = loaded
    np.testing.assert_array_equal(data, image)
    np.testing.assert_array_equal(m, mask)

    fp = fd._discover_shared_convolved_fp(tmp_path, "skycell.1234", "000")
    assert fp == info["fingerprint"]


def test_shared_store_loader_returns_none_when_missing(tmp_path: Path):
    assert fd._try_load_shared_convolved_arrays(tmp_path, "skycell.1234.000") is None


def _write_legacy_flat_cell(
    zpath: Path, skycell: str, image: np.ndarray, mask: np.ndarray
) -> None:
    root = zarr.open(str(zpath), mode="w")
    root[f"{skycell}_data"] = image
    root[f"{skycell}_mask"] = mask


def test_legacy_flat_zarr_path_still_works(tmp_path: Path):
    skycell = "skycell.9999.001"
    image = np.full((3, 3), 7.0, dtype=np.float32)
    mask = np.ones((3, 3), dtype=np.int32)
    zpath = tmp_path / "sector_0020_camera_1_ccd_1.zarr"
    _write_legacy_flat_cell(zpath, skycell, image, mask)

    zstore = zarr.open(str(zpath), mode="r")
    data, m = fd._load_zarr_skycell(zstore, skycell)
    np.testing.assert_array_equal(data, image)
    np.testing.assert_array_equal(m, mask)


def test_l5_loader_shared_first_then_legacy_fallback(tmp_path: Path):
    image, mask, _info = _publish_cell(tmp_path)
    skycell = "skycell.1234.000"

    # Legacy zarr with different values — shared hit must win.
    legacy = tmp_path / "legacy.zarr"
    _write_legacy_flat_cell(
        legacy,
        skycell,
        np.full((4, 4), -1.0, dtype=np.float32),
        np.zeros((4, 4), dtype=np.int32),
    )

    fd._reset_l5_worker()
    fd._init_l5_worker(
        {
            "data_root": str(tmp_path),
            "shared_convolved_store": True,
            "zarr_path": str(ps1_convolved_zarr_path(tmp_path)),
            "legacy_zarr_path": str(legacy),
        }
    )
    data, m = fd._load_ps1_skycell_for_l5(skycell)
    np.testing.assert_array_equal(data, image)
    np.testing.assert_array_equal(m, mask)
    fd._reset_l5_worker()


def test_l5_loader_falls_back_to_legacy_when_shared_cell_missing(tmp_path: Path):
    skycell = "skycell.5555.002"
    image = np.arange(9, dtype=np.float32).reshape(3, 3)
    mask = np.zeros((3, 3), dtype=np.int32)
    legacy = tmp_path / "legacy.zarr"
    _write_legacy_flat_cell(legacy, skycell, image, mask)

    # Shared root exists but has no cell for this skycell.
    ps1_convolved_zarr_path(tmp_path).mkdir(parents=True)

    fd._reset_l5_worker()
    fd._init_l5_worker(
        {
            "data_root": str(tmp_path),
            "shared_convolved_store": True,
            "zarr_path": str(ps1_convolved_zarr_path(tmp_path)),
            "legacy_zarr_path": str(legacy),
        }
    )
    data, m = fd._load_ps1_skycell_for_l5(skycell)
    np.testing.assert_array_equal(data, image)
    np.testing.assert_array_equal(m, mask)
    fd._reset_l5_worker()


def test_l5_loader_legacy_only_when_shared_flag_off(tmp_path: Path):
    skycell = "skycell.7777.003"
    image = np.ones((2, 2), dtype=np.float32) * 3.0
    mask = np.zeros((2, 2), dtype=np.int32)
    zpath = tmp_path / "sector_0020_camera_1_ccd_1.zarr"
    _write_legacy_flat_cell(zpath, skycell, image, mask)

    fd._reset_l5_worker()
    fd._init_l5_worker(
        {
            "data_root": str(tmp_path),
            "shared_convolved_store": False,
            "zarr_path": str(zpath),
            "legacy_zarr_path": None,
        }
    )
    data, m = fd._load_ps1_skycell_for_l5(skycell)
    np.testing.assert_array_equal(data, image)
    np.testing.assert_array_equal(m, mask)
    fd._reset_l5_worker()


def test_shared_miss_without_legacy_raises(tmp_path: Path):
    ps1_convolved_zarr_path(tmp_path).mkdir(parents=True)
    fd._reset_l5_worker()
    fd._init_l5_worker(
        {
            "data_root": str(tmp_path),
            "shared_convolved_store": True,
            "zarr_path": str(ps1_convolved_zarr_path(tmp_path)),
            "legacy_zarr_path": None,
        }
    )
    with pytest.raises(FileNotFoundError, match="No shared convolved cell"):
        fd._load_ps1_skycell_for_l5("skycell.1234.000")
    fd._reset_l5_worker()
