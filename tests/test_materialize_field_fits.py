"""Tests for materialized field template FITS (hybrid contrib assemble path)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.common.mapping_grid import MAPGRID_VERSION, MappingGrid
from syndiff_pipeline.common.template_coverage import template_coverage_ffi_bounds
from syndiff_pipeline.difference_imaging.support.template_resolution import (
    find_field_fits_by_group_id,
    parse_field_gid_from_filename,
)
from syndiff_pipeline.template_creation.processing.field_downsample import (
    assemble_field_group_flux,
    materialize_field_fits_for_store,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    _roi_bounds_to_assemble_crop,
    field_fits_basename,
    write_contrib,
    write_template_manifest,
    validate_frozen_field_geometry,
)

NY, NX = 8, 10


def _flat(y: int, x: int) -> int:
    return y * NX + x


def _plain_write_field_group_fits(out_path, flux, count, *, header=None):
    from astropy.io import fits

    plain = Path(str(out_path).replace(".fits.fz", ".fits").replace(".fits.gz", ".fits"))
    plain.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header(header) if header is not None else fits.Header()
    count_hdr = hdr.copy()
    count_hdr["EXTNAME"] = "COUNT"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(np.asarray(flux, dtype=np.float32), header=hdr),
            fits.ImageHDU(np.asarray(count, dtype=np.float32), header=count_hdr, name="COUNT"),
        ]
    )
    hdul.writeto(plain, overwrite=True)
    return str(plain)


class TestMaterializeFieldFits(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = Path(self._tmpdir.name) / "store"
        self.store.mkdir()
        (self.store / "contribs").mkdir()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_frozen_geometry_validation_rejects_mismatched_grid(self):
        grid = MappingGrid(
            ffi_xmin=0, ffi_ymin=0, ffi_xmax=NX, ffi_ymax=NY,
            oversampling=1, conv_pad_native=0,
        )
        (self.store / "field_mode_assembly.json").write_text(json.dumps({
            "schema_version": 3,
            "mapping_grid": grid.to_mapping_dict(),
            "base_tess_shape": [NY, NX],
            "geometry_provenance": {"temporal_wcs_frame_contract_fingerprint": "fp-a"},
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
        }))
        self.assertEqual(
            validate_frozen_field_geometry(
                self.store, grid,
                expected_provenance={"temporal_wcs_frame_contract_fingerprint": "fp-a"},
            )["schema_version"],
            3,
        )
        other = MappingGrid(
            ffi_xmin=0, ffi_ymin=0, ffi_xmax=NX - 1, ffi_ymax=NY,
            oversampling=1, conv_pad_native=0,
        )
        with self.assertRaises(ValueError, msg="geometry mismatch"):
            validate_frozen_field_geometry(self.store, other)

    def test_oversampled_crop_uses_mapping_grid_local_coordinates(self):
        grid = MappingGrid(
            ffi_xmin=44,
            ffi_ymin=-8,
            ffi_xmax=2092,
            ffi_ymax=2048,
            oversampling=4,
            conv_pad_native=8,
        )
        self.assertEqual(
            _roi_bounds_to_assemble_crop(
                (44, -8, 2092, 2048),
                oversampling_factor=4,
                mapping_grid=grid,
            ),
            (0, 8192, 0, 8224),
        )

    def _shifts_legacy(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(group_id=0, skycell="skycell.1.1", sx_int=0, sy_int=0),
                dict(group_id=1, skycell="skycell.2.2", sx_int=1, sy_int=0),
            ]
        )

    def _shifts_pair_state(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(group_id=0, skycell="skycell.1.1", sx_int=0, sy_int=0),
                dict(group_id=1, skycell="skycell.1.1", sx_int=0, sy_int=0),
            ]
        )

    def _seed_group_contribs(self) -> None:
        write_contrib(
            self.store,
            "skycell.1.1",
            0,
            0,
            indices=np.array([_flat(1, 2)], dtype=np.int64),
            flux_sum=np.array([10.0]),
            count=np.array([2.0]),
            group_id=0,
        )
        write_contrib(
            self.store,
            "skycell.2.2",
            1,
            0,
            indices=np.array([_flat(4, 5)], dtype=np.int64),
            flux_sum=np.array([20.0]),
            count=np.array([4.0]),
            group_id=1,
        )

    def _seed_pair_state_contribs(self) -> None:
        write_contrib(
            self.store,
            "skycell.1.1",
            0,
            0,
            indices=np.array([_flat(2, 3)], dtype=np.int64),
            flux_sum=np.array([30.0]),
            count=np.array([3.0]),
            group_id=0,
        )
        write_contrib(
            self.store,
            "skycell.1.1",
            0,
            0,
            indices=np.array([_flat(2, 3)], dtype=np.int64),
            flux_sum=np.array([60.0]),
            count=np.array([6.0]),
            group_id=1,
        )
        (self.store / "field_mode_assembly.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "group_scoped_contribs": True,
                }
            )
            + "\n"
        )

    @mock.patch(
        "syndiff_pipeline.template_creation.processing.field_downsample.write_field_group_fits",
        side_effect=_plain_write_field_group_fits,
    )
    def test_materialize_writes_fits_per_group_matching_assemble(self, _mock_write):
        self._seed_group_contribs()
        shifts_df = self._shifts_legacy()
        shifts_df.to_parquet(self.store / "template_group_shifts.parquet")
        write_template_manifest(
            self.store,
            FieldManifest(
                geometry_mode="field",
                scope="scc",
                assembly="sparse_sum",
                materialize_fits=True,
                sector=20,
                camera=3,
                ccd=3,
                contribs_dir="contribs",
                groups=[{"group_id": 0}, {"group_id": 1}],
            ),
        )

        result = materialize_field_fits_for_store(
            self.store,
            shifts_df,
            sector=20,
            camera=3,
            ccd=3,
            base_tess_shape=(NY, NX),
            provenance={
                "intra_skycell_R": 1,
                "n_intra_skycell_keys": 12,
                "n_inter_skycell_pair_states": 0,
            },
        )

        self.assertEqual(result["n_groups"], 2)
        for gid in (0, 1):
            expected = assemble_field_group_flux(
                self.store,
                shifts_df,
                gid,
                shape=(NY, NX),
                group_scoped_contribs=True,
            )
            fits_path = find_field_fits_by_group_id(self.store, gid)
            self.assertIsNotNone(fits_path)
            with fits.open(fits_path) as hdul:
                data = hdul[0].data.astype(np.float64)
            np.testing.assert_allclose(data, expected)
            self.assertEqual(
                parse_field_gid_from_filename(field_fits_basename(20, 3, 3, gid)),
                gid,
            )

        sidecar = json.loads((self.store / "materialized_fits.json").read_text())
        self.assertEqual(sidecar["provenance"]["intra_skycell_R"], 1)
        self.assertEqual(sidecar["provenance"]["n_intra_skycell_keys"], 12)

    @mock.patch(
        "syndiff_pipeline.template_creation.processing.field_downsample.write_field_group_fits",
        side_effect=_plain_write_field_group_fits,
    )
    def test_materialize_writes_mapgrid_headers(self, _mock_write):
        self._seed_group_contribs()
        shifts_df = self._shifts_legacy()
        shifts_df.to_parquet(self.store / "template_group_shifts.parquet")
        grid = MappingGrid(
            ffi_xmin=0,
            ffi_ymin=0,
            ffi_xmax=NX,
            ffi_ymax=NY,
            oversampling=1,
            conv_pad_native=0,
        )

        materialize_field_fits_for_store(
            self.store,
            shifts_df,
            sector=20,
            camera=3,
            ccd=3,
            base_tess_shape=(NY, NX),
            mapping_grid=grid,
        )

        fits_path = find_field_fits_by_group_id(self.store, 0)
        self.assertIsNotNone(fits_path)
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            self.assertEqual(int(hdr["MAPGRID"]), MAPGRID_VERSION)
            self.assertEqual(int(hdr["CONVPAD"]), 0)
            self.assertEqual(int(hdr["XMIN"]), 0)
            self.assertEqual(int(hdr["YMIN"]), 0)
            self.assertEqual(int(hdr["XMAX"]), NX)
            self.assertEqual(int(hdr["YMAX"]), NY)
            self.assertEqual(str(hdr["COORDFRM"]).strip(), "full_ffi")
        cov = template_coverage_ffi_bounds(str(fits_path))
        self.assertEqual(cov["x_min"], 0)
        self.assertEqual(cov["y_min"], 0)
        self.assertEqual(cov["shape"], (NY, NX))

    @mock.patch(
        "syndiff_pipeline.template_creation.processing.field_downsample.write_field_group_fits",
        side_effect=_plain_write_field_group_fits,
    )
    def test_pair_state_group_qualified_contribs(self, _mock_write):
        self._seed_pair_state_contribs()
        shifts_df = self._shifts_pair_state()
        shifts_df.to_parquet(self.store / "template_group_shifts.parquet")

        materialize_field_fits_for_store(
            self.store,
            shifts_df,
            sector=1,
            camera=1,
            ccd=1,
            base_tess_shape=(NY, NX),
            provenance={"intra_skycell_R": 1},
        )

        flux_g0 = assemble_field_group_flux(
            self.store, shifts_df, 0, shape=(NY, NX)
        )
        flux_g1 = assemble_field_group_flux(
            self.store, shifts_df, 1, shape=(NY, NX)
        )
        self.assertAlmostEqual(float(flux_g0[2, 3]), 30.0)
        self.assertAlmostEqual(float(flux_g1[2, 3]), 60.0)

        with fits.open(find_field_fits_by_group_id(self.store, 0)) as hdul:
            np.testing.assert_allclose(hdul[0].data, flux_g0)
        with fits.open(find_field_fits_by_group_id(self.store, 1)) as hdul:
            np.testing.assert_allclose(hdul[0].data, flux_g1)

    @mock.patch(
        "syndiff_pipeline.template_creation.processing.field_downsample.write_field_group_fits",
        side_effect=_plain_write_field_group_fits,
    )
    def test_materialize_tolerates_missing_contrib_in_group(self, _mock_write):
        write_contrib(
            self.store,
            "skycell.1.1",
            0,
            0,
            indices=np.array([_flat(0, 0)], dtype=np.int64),
            flux_sum=np.array([1.0]),
            count=np.array([1.0]),
            group_id=0,
        )
        shifts_df = pd.DataFrame(
            [
                dict(group_id=0, skycell="skycell.1.1", sx_int=0, sy_int=0),
                dict(group_id=0, skycell="skycell.9.9", sx_int=0, sy_int=0),
            ]
        )
        result = materialize_field_fits_for_store(
            self.store,
            shifts_df,
            sector=1,
            camera=1,
            ccd=1,
            base_tess_shape=(NY, NX),
        )
        self.assertEqual(result["n_groups"], 1)
        expected = assemble_field_group_flux(
            self.store,
            shifts_df,
            0,
            shape=(NY, NX),
            present_only=True,
            group_scoped_contribs=True,
        )
        with fits.open(find_field_fits_by_group_id(self.store, 0)) as hdul:
            np.testing.assert_allclose(hdul[0].data, expected)

    @mock.patch(
        "syndiff_pipeline.template_creation.processing.field_downsample.write_field_group_fits",
        side_effect=_plain_write_field_group_fits,
    )
    def test_materialize_skips_group_with_no_contribs(self, _mock_write):
        write_contrib(
            self.store,
            "skycell.2.2",
            1,
            0,
            indices=np.array([_flat(4, 5)], dtype=np.int64),
            flux_sum=np.array([20.0]),
            count=np.array([4.0]),
            group_id=1,
        )
        shifts_df = pd.DataFrame(
            [
                dict(group_id=0, skycell="skycell.9.9", sx_int=0, sy_int=0),
                dict(group_id=1, skycell="skycell.2.2", sx_int=1, sy_int=0),
            ]
        )
        result = materialize_field_fits_for_store(
            self.store,
            shifts_df,
            sector=1,
            camera=1,
            ccd=1,
            base_tess_shape=(NY, NX),
        )
        self.assertEqual(result["n_groups"], 1)
        self.assertEqual(result["skipped_groups"][0]["group_id"], 0)
        self.assertIsNone(find_field_fits_by_group_id(self.store, 0))
        self.assertIsNotNone(find_field_fits_by_group_id(self.store, 1))

    def test_materialize_false_writes_no_fits(self):
        self._seed_group_contribs()
        fits_dir = self.store / "fits"
        self.assertFalse(fits_dir.exists())
        # Lazy-only path: no call to materialize_field_fits_for_store.
        self.assertFalse((self.store / "materialized_fits.json").exists())


@unittest.skipUnless(shutil.which("fpack"), "fpack not on PATH")
class TestMaterializeFieldFitsFpack(unittest.TestCase):
    def test_write_field_group_fits_fpack_roundtrip(self):
        from syndiff_pipeline.template_creation.processing.field_templates import (
            write_field_group_fits,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "syndiff_field_s0001_1_1_gid0.fits"
            flux = np.ones((4, 5), dtype=np.float64) * 3.5
            count = np.ones((4, 5), dtype=np.float64) * 2.0
            written = write_field_group_fits(out, flux, count)
            self.assertTrue(Path(written).is_file())
            self.assertTrue(str(written).endswith(".fits.fz"))


if __name__ == "__main__":
    unittest.main()
