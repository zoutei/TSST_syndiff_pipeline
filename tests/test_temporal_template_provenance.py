from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.template_creation.processing.field_downsample import (
    materialize_field_fits_for_store,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    build_field_fits_header,
    write_native_debug_fits,
    write_contrib,
)


def _grid():
    return MappingGrid(
        ffi_xmin=0, ffi_xmax=2, ffi_ymin=0, ffi_ymax=2, oversampling=4
    )


def test_temporal_provenance_is_serialized_and_written_to_fits():
    provenance = {
        "geometry_mode": "temporal_wcs",
        "temporal_wcs_version": "temporal_cheb5_bspline_v1",
        "temporal_wcs_fingerprint": "wcs-fingerprint",
        "mapping_fingerprint": "mapping-fingerprint",
        "remap_fingerprint": "remap-fingerprint",
    }
    manifest = FieldManifest(
        geometry_mode="field", scope="scc", assembly="sparse_sum",
        materialize_fits=True, sector=20, camera=3, ccd=3,
        contribs_dir="contribs", groups=[], provenance=provenance,
    ).to_dict()
    assert manifest["provenance"] == provenance
    hdr = build_field_fits_header(
        sector=20, camera=3, ccd=3, group_id=0, oversampling_factor=4,
        provenance=provenance, mapping_grid=_grid(),
    )
    assert hdr["TVWCSVER"] == provenance["temporal_wcs_version"]
    assert hdr["TVWCSFP"] == provenance["temporal_wcs_fingerprint"]
    assert hdr["MAPFP"] == provenance["mapping_fingerprint"]
    assert hdr["REMAPFP"] == provenance["remap_fingerprint"]


def test_temporal_materialization_writes_lane_specific_debug_fits(tmp_path: Path):
    store = tmp_path / "s0020" / "c3" / "k3" / "templates_tvwcs" / "oversampling_4"
    write_contrib(
        store, "skycell.1.1", 0, 0,
        indices=np.array([0, 1, 3], dtype=np.int64),
        flux_sum=np.array([2.0, 4.0, 6.0]), count=np.ones(3), group_id=0,
    )
    shifts = pd.DataFrame({"group_id": [0], "skycell": ["skycell.1.1"], "sx_int": [0], "sy_int": [0]})
    result = materialize_field_fits_for_store(
        store, shifts, sector=20, camera=3, ccd=3,
        base_tess_shape=(8, 8), oversampling_factor=4,
        provenance={"geometry_mode": "temporal_wcs", "temporal_wcs_version": "v1"},
        mapping_grid=_grid(),
    )
    assert len(result["debug_fits"]) == 1
    debug = Path(result["debug_fits"][0])
    assert debug.parent == store.parents[1] / "debug_plots" / "templates_tvwcs_os4" / "fits"
    with fits.open(debug) as hdul:
        assert hdul[1].header["TVWCSVER"] == "v1"
        assert hdul[1].header["OVERSAMP"] == 4
    native = Path(result["native_debug_fits"][0])
    assert native == debug.with_name("syndiff_field_s0020_3_3_os4_gid0_native.fits.fz")


def test_native_debug_fits_sums_flux_and_count(tmp_path: Path):
    source = tmp_path / "source.fits"
    flux = np.arange(16, dtype=np.float32).reshape(4, 4)
    count = np.ones((4, 4), dtype=np.float32)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(flux, name="COMPRESSED_IMAGE"),
        fits.ImageHDU(count, name="COUNT"),
    ]).writeto(source)

    output = write_native_debug_fits(source, oversampling_factor=2)
    assert Path(output).name == "source_native.fits.fz"
    with fits.open(output) as hdul:
        np.testing.assert_allclose(hdul[1].data, [[10, 18], [42, 50]])
        np.testing.assert_allclose(hdul[2].data, [[4, 4], [4, 4]])
        assert hdul[1].header["OVERSAMP"] == 1
        assert hdul[1].header["SRCOS"] == 2
        assert hdul[1].header["BLKSUM"] == 2


def test_native_debug_fits_rejects_non_divisible_shape(tmp_path: Path):
    source = tmp_path / "source.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((3, 4), dtype=np.float32)),
        fits.ImageHDU(np.ones((3, 4), dtype=np.float32), name="COUNT"),
    ]).writeto(source)

    import pytest
    with pytest.raises(ValueError, match="not divisible"):
        write_native_debug_fits(source, oversampling_factor=2)
