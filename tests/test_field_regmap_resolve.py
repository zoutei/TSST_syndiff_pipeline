"""Regmap resolution must stay inside the matching oversampling tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from syndiff_pipeline.template_creation.processing.field_remap import (
    _find_regmap,
    _mapping_scc_dir,
    _master_pixels2skycells_path,
)


def test_find_regmap_prefers_native_not_os2(tmp_path: Path):
    mapping = tmp_path / "skycell_pixel_mapping"
    native = mapping / "sector_0020" / "camera_3" / "ccd_3"
    os2 = mapping / "oversampling_2" / "sector_0020" / "camera_3" / "ccd_3"
    native.mkdir(parents=True)
    os2.mkdir(parents=True)
    sk = "skycell.2587.092"
    native_file = native / f"tess_s20_3_3_{sk}.fits.fz"
    os2_file = os2 / f"tess_s20_3_3_{sk}_os2.fits.fz"
    native_file.write_bytes(b"native")
    os2_file.write_bytes(b"os2")

    found = _find_regmap(mapping, 20, 3, 3, sk, oversampling_factor=1)
    assert found == native_file

    found_os2 = _find_regmap(mapping, 20, 3, 3, sk, oversampling_factor=2)
    assert found_os2 == os2_file


def test_find_regmap_resolves_legacy_gz(tmp_path: Path):
    """Backward-compatible read: gzip-only regmaps still resolve."""
    mapping = tmp_path / "skycell_pixel_mapping"
    native = mapping / "sector_0020" / "camera_3" / "ccd_3"
    native.mkdir(parents=True)
    sk = "skycell.2587.092"
    native_file = native / f"tess_s20_3_3_{sk}.fits.gz"
    native_file.write_bytes(b"native")

    found = _find_regmap(mapping, 20, 3, 3, sk, oversampling_factor=1)
    assert found == native_file


def test_master_path_native(tmp_path: Path):
    mapping = tmp_path / "skycell_pixel_mapping"
    scc = mapping / "sector_0020" / "camera_3" / "ccd_3"
    scc.mkdir(parents=True)
    master = scc / "tess_s0020_3_3_master_pixels2skycells.fits.fz"
    master.write_bytes(b"m")
    # Poison glob with an oversampled master that would win a naive **/ glob.
    os2 = mapping / "oversampling_2" / "sector_0020" / "camera_3" / "ccd_3"
    os2.mkdir(parents=True)
    (os2 / "tess_s0020_3_3_master_pixels2skycells_os2.fits.fz").write_bytes(b"bad")

    assert _master_pixels2skycells_path(mapping, 20, 3, 3) == master
    assert _mapping_scc_dir(mapping, 20, 3, 3).resolve() == scc.resolve()


def test_master_path_resolves_legacy_gz(tmp_path: Path):
    mapping = tmp_path / "skycell_pixel_mapping"
    scc = mapping / "sector_0020" / "camera_3" / "ccd_3"
    scc.mkdir(parents=True)
    master = scc / "tess_s0020_3_3_master_pixels2skycells.fits.gz"
    master.write_bytes(b"m")

    assert _master_pixels2skycells_path(mapping, 20, 3, 3) == master


def test_find_regmap_missing_raises(tmp_path: Path):
    mapping = tmp_path / "skycell_pixel_mapping"
    (mapping / "sector_0020" / "camera_3" / "ccd_3").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        _find_regmap(mapping, 20, 3, 3, "skycell.1.1", oversampling_factor=1)
