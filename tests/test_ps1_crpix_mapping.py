"""PS1 CRPIX (+0.5, IPP convention) in the mapping stage: table load, WCS, GEOMFP, marker, verify."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.common.mapping_grid import MappingGrid, load_mapping_grid_from_master
from syndiff_pipeline.common.ps1_wcs import (
    PS1_CRPIX_COLUMN,
    apply_ps1_crpix_convention,
    ps1_crpix_offset,
    table_ps1_crpix_offset,
)
from syndiff_pipeline.template_creation.orchestration.bundled_assets import skycell_wcs_csv
from syndiff_pipeline.template_creation.orchestration.verify import verify_mapping
from syndiff_pipeline.template_creation.processing import pancakes

from tests.test_verify_mapping_mapgrid import _resolved, _tiny_grid, _write_master


def _raw() -> pd.DataFrame:
    return pd.read_csv(skycell_wcs_csv())


def test_bundled_table_ipp_adds_half_pixel(monkeypatch):
    monkeypatch.delenv("SYNDIFF_PS1_CRPIX_CONVENTION", raising=False)
    raw = _raw()
    out = apply_ps1_crpix_convention(raw)
    np.testing.assert_allclose(out["CRPIX1"], raw["CRPIX1"] + 0.5)
    np.testing.assert_allclose(out["CRPIX2"], raw["CRPIX2"] + 0.5)
    assert set(out[PS1_CRPIX_COLUMN]) == {0.5}
    assert table_ps1_crpix_offset(out) == 0.5


def test_bundled_table_fits_convention_is_raw(monkeypatch):
    monkeypatch.setenv("SYNDIFF_PS1_CRPIX_CONVENTION", "fits")
    raw = _raw()
    out = apply_ps1_crpix_convention(raw)
    np.testing.assert_array_equal(out["CRPIX1"], raw["CRPIX1"])
    assert set(out[PS1_CRPIX_COLUMN]) == {0.0}
    assert table_ps1_crpix_offset(out) == 0.0


def test_mapping_path_applies_convention_once():
    src = inspect.getsource(pancakes.process_tess_image_optimized)
    assert "apply_ps1_crpix_convention(pd.read_csv(skycell_wcs_csv))" in src
    # no other raw read of the bundled table in the module
    assert inspect.getsource(pancakes).count("pd.read_csv(skycell_wcs_csv") == 1


def test_get_ps1_wcs_information_shifts_half_pixel(monkeypatch):
    monkeypatch.delenv("SYNDIFF_PS1_CRPIX_CONVENTION", raising=False)
    raw = _raw().iloc[100]
    corr = apply_ps1_crpix_convention(_raw()).iloc[100]
    _, w_raw, _ = pancakes.get_ps1_wcs_information(raw)
    hdr, w_cor, _ = pancakes.get_ps1_wcs_information(corr)
    assert hdr["PS1CRPIX"] == 0.5
    ra, dec = w_raw.pixel_to_world_values(np.array([1000.0, 3000.0]), np.array([500.0, 2500.0]))
    x0, y0 = w_raw.world_to_pixel_values(ra, dec)
    x1, y1 = w_cor.world_to_pixel_values(ra, dec)
    np.testing.assert_allclose(x1 - x0, 0.5, atol=1e-6)
    np.testing.assert_allclose(y1 - y0, 0.5, atol=1e-6)


def _grid_args():
    return dict(nx=2136, ny=2078, conv_pad_native=8, oversampling=4)


def test_geomfp_legacy_unchanged_and_ipp_differs():
    legacy = MappingGrid.from_ffi_shape(**_grid_args())
    legacy_explicit = MappingGrid.from_ffi_shape(**_grid_args(), ps1_crpix=0.0)
    ipp = MappingGrid.from_ffi_shape(**_grid_args(), ps1_crpix=0.5)
    assert "ps1_crpix" not in legacy.geometry_recipe()
    assert legacy.geometry_fingerprint == legacy_explicit.geometry_fingerprint
    assert ipp.geometry_recipe()["ps1_crpix"] == 0.5
    assert ipp.geometry_fingerprint != legacy.geometry_fingerprint
    # the legacy recipe hashes exactly as before the change (recipe dict without the new key)
    import hashlib, json
    expected = hashlib.sha256(
        json.dumps(legacy.geometry_recipe(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    assert legacy.geometry_fingerprint == expected


def test_grid_roundtrips_marker_through_fits_header_and_dict():
    ipp = MappingGrid.from_ffi_shape(**_grid_args(), ps1_crpix=0.5)
    hdr = ipp.to_fits_header_updates()
    assert hdr["PS1CRPIX"] == 0.5
    from astropy.io import fits

    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    assert MappingGrid.from_fits_header(h).geometry_fingerprint == ipp.geometry_fingerprint
    assert MappingGrid.from_mapping_dict(ipp.to_mapping_dict()).geometry_fingerprint == ipp.geometry_fingerprint
    # a legacy header (no PS1CRPIX) validates against its own stored GEOMFP regardless of the active convention
    leg = MappingGrid.from_ffi_shape(**_grid_args())
    h2 = fits.Header()
    for k, v in leg.to_fits_header_updates().items():
        if k != "PS1CRPIX":
            h2[k] = v
    assert MappingGrid.from_fits_header(h2).geometry_fingerprint == leg.geometry_fingerprint


def test_marker_survives_master_csv_write_and_rewrite(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNDIFF_PS1_CRPIX_CONVENTION", raising=False)
    df = apply_ps1_crpix_convention(_raw().iloc[:3])
    grid = MappingGrid.from_ffi_shape(**_grid_args(), ps1_crpix=0.5)
    partial, final = pancakes.master_skycells_csv_paths(str(tmp_path), 22, 3, 3, oversampling_factor=4)
    pancakes.save_updated_skycell_csv(df, str(tmp_path), 22, 3, 3, oversampling_factor=4, mapping_grid=grid)
    back = pd.read_csv(partial)
    assert table_ps1_crpix_offset(back) == 0.5
    np.testing.assert_allclose(back["CRPIX1"], df["CRPIX1"])
    assert set(back["GEOMFP"]) == {grid.geometry_fingerprint}
    # rewrite from the re-read table must not correct twice
    again = apply_ps1_crpix_convention(back)
    np.testing.assert_allclose(again["CRPIX1"], df["CRPIX1"])
    pancakes.save_updated_skycell_csv(again, str(tmp_path), 22, 3, 3, oversampling_factor=4, mapping_grid=grid)
    assert table_ps1_crpix_offset(pd.read_csv(partial)) == 0.5
    assert pancakes.finalize_master_skycells_csv(str(tmp_path), 22, 3, 3, oversampling_factor=4) == final


def _write_csv_marker(leaf: Path, marker: float | None) -> None:
    leaf.mkdir(parents=True, exist_ok=True)
    cols = "NAME,projection" + (f",{PS1_CRPIX_COLUMN}" if marker is not None else "")
    row = "skycell.0001.0001,0001" + (f",{marker}" if marker is not None else "")
    (leaf / "tess_s0022_3_3_master_skycells_list.csv").write_text(f"{cols}\n{row}\n", encoding="utf-8")


def test_verify_mapping_reports_marker(tmp_path, caplog, monkeypatch):
    monkeypatch.delenv("SYNDIFF_PS1_CRPIX_CONVENTION", raising=False)
    leaf = tmp_path / "mapping"
    _write_csv_marker(leaf, 0.5)
    _write_master(leaf, _tiny_grid())
    res = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert res.ok and "PS1CRPIX=0.5" in res.message
    assert not any("PS1CRPIX" in r.message and r.levelname == "WARNING" for r in caplog.records)


def test_verify_mapping_legacy_csv_reports_zero_and_passes(tmp_path, caplog, monkeypatch):
    monkeypatch.delenv("SYNDIFF_PS1_CRPIX_CONVENTION", raising=False)
    leaf = tmp_path / "mapping"
    _write_csv_marker(leaf, None)
    _write_master(leaf, _tiny_grid())
    with caplog.at_level("WARNING"):
        res = verify_mapping(_resolved(tmp_path, mapping_root=leaf))
    assert res.ok and "PS1CRPIX=0.0" in res.message
    assert any("PS1CRPIX" in r.getMessage() for r in caplog.records)  # warns: active convention is 0.5
    assert ps1_crpix_offset() == 0.5
