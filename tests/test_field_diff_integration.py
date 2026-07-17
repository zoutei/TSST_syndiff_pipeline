"""Field-mode diff integration: assemble -> loaders -> FFI assembler -> verify ->
kernel group lookup, exercised across module seams on a synthetic SCC store."""

import numpy as np
import pandas as pd
import pytest

from syndiff_pipeline.difference_imaging.stages.convolved_templates import (
    lookup_convolved_path_by_group_id,
)
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    FieldModeTemplateContext,
    assemble_field_template_for_ffi,
    build_field_mode_count_loader,
    build_field_mode_template_loader,
)
from syndiff_pipeline.template_creation.processing.field_downsample import (
    assemble_field_group_count,
    assemble_field_group_flux,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    verify_field_store,
    write_contrib,
    write_template_manifest,
    FieldManifest,
)

NY, NX = 10, 12


def _flat(y, x):
    return y * NX + x


@pytest.fixture
def field_store(tmp_path):
    """A tiny SCC store: group 0 over skycells A (shift +1,0) and B (0,0)."""
    store = tmp_path / "field_templates" / "sector_0001_camera_1_ccd_1"
    (store / "contribs").mkdir(parents=True)

    # A contributes flux 100 at (2,3) count 2; B contributes flux 50 at (5,6) count 1.
    write_contrib(
        store, "skycell.1.1", 1, 0,
        indices=np.array([_flat(2, 3)]), flux_sum=np.array([100.0]),
        count=np.array([2.0]), mask_count=np.array([0.0]),
    )
    write_contrib(
        store, "skycell.2.2", 0, 0,
        indices=np.array([_flat(5, 6)]), flux_sum=np.array([50.0]),
        count=np.array([1.0]), mask_count=np.array([0.0]),
    )
    shifts_df = pd.DataFrame(
        [
            dict(group_id=0, skycell="skycell.1.1", sx_int=1, sy_int=0),
            dict(group_id=0, skycell="skycell.2.2", sx_int=0, sy_int=0),
        ]
    )
    shifts_df.to_parquet(store / "template_group_shifts.parquet")
    write_template_manifest(
        store,
        FieldManifest(
            geometry_mode="field", scope="scc", assembly="sparse_sum",
            materialize_fits=False, sector=1, camera=1, ccd=1,
            contribs_dir="contribs", groups=[{"group_id": 0}],
        ),
    )
    return store, shifts_df


def test_assemble_group_flux_and_count(field_store):
    store, shifts_df = field_store
    flux = assemble_field_group_flux(store, shifts_df, 0, shape=(NY, NX))
    assert flux[2, 3] == pytest.approx(100.0 / 2)   # mean flux
    assert flux[5, 6] == pytest.approx(50.0 / 1)
    count = assemble_field_group_count(store, shifts_df, 0, shape=(NY, NX))
    assert count[2, 3] == 2 and count[5, 6] == 1


def _ctx(store, shifts_df):
    return FieldModeTemplateContext(
        store_root=str(store), shifts_df=shifts_df,
        base_tess_shape=(NY, NX), template_roi_bounds=(0, 0, NX, NY),
    )


def test_template_and_count_loaders_crop(field_store):
    store, shifts_df = field_store
    ctx = _ctx(store, shifts_df)
    crop = {"x_min": 2, "x_max": 8, "y_min": 1, "y_max": 7}
    flux_crop = build_field_mode_template_loader(ctx, crop)(0)
    assert flux_crop.shape == (6, 6)
    # (2,3) full -> (1,1) crop-local; (5,6) full -> (4,4) crop-local
    assert flux_crop[1, 1] == pytest.approx(50.0)
    assert flux_crop[4, 4] == pytest.approx(50.0)
    count_crop = build_field_mode_count_loader(ctx, crop)(0)
    assert count_crop[1, 1] == 2 and count_crop[4, 4] == 1


def test_assemble_template_for_ffi_by_name(field_store):
    store, shifts_df = field_store
    ctx = _ctx(store, shifts_df)
    manifest = pd.DataFrame(
        {
            "filename": ["tess2020007-0001-1-1_ffic.fits.gz"],
            "path": ["/data/tess2020007-0001-1-1_ffic.fits.gz"],
            "group_id": [0],
        }
    )
    # by name (basename) and by full path must agree; full FFI shape
    big = assemble_field_template_for_ffi(ctx, manifest, "tess2020007-0001-1-1_ffic.fits.gz")
    assert big.shape == (NY, NX)
    assert big[2, 3] == pytest.approx(50.0)
    big2 = assemble_field_template_for_ffi(ctx, manifest, manifest.loc[0, "path"])
    assert np.array_equal(big, big2)


def test_verify_field_store_marker(field_store):
    store, _ = field_store
    keys = [("skycell.1.1", 1, 0), ("skycell.2.2", 0, 0)]
    assert verify_field_store(store, required_keys=keys)["ok"]
    bad = verify_field_store(store, required_keys=keys + [("skycell.9.9", 3, 3)])
    assert not bad["ok"]


def test_kernel_convolved_group_lookup():
    table = pd.DataFrame(
        [
            {"group_id": 0, "convolved_path": "/w/convolved_template_gid0.fits.gz"},
            {"group_id": 7, "convolved_path": "/w/convolved_template_gid7.fits.gz"},
        ]
    )
    assert lookup_convolved_path_by_group_id(table, 7).endswith("gid7.fits.gz")
    with pytest.raises(FileNotFoundError):
        lookup_convolved_path_by_group_id(table, 3)
