"""Tests for SCC-scoped ffi_list inventory and cache-only WCS helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.download import manifest_basename_from_local
from syndiff_pipeline.common.scc_paths import scc_ffi_list_csv, scc_ffi_list_parquet
from syndiff_pipeline.common.wcs_header_cache import (
    ensure_scc_ffi_list,
    extract_ffi_header_record,
    ffi_list_csv_path,
    ffi_list_is_complete,
    ffi_list_parquet_path,
    header_from_cached_row,
    load_ffi_list,
    median_crval_from_cache,
    rebuild_scc_ffi_list,
    upsert_ffi_list_rows,
    wcs_from_cached_row,
)


def _write_test_ffi(
    path: Path,
    *,
    crval1: float = 100.0,
    crval2: float = 10.0,
    include_wcs: bool = True,
    sip: bool = False,
) -> None:
    hdu0 = fits.PrimaryHDU()
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 32
    hdr["NAXIS2"] = 32
    if include_wcs:
        hdr["CRVAL1"] = crval1
        hdr["CRVAL2"] = crval2
        hdr["CRPIX1"] = 16.0
        hdr["CRPIX2"] = 16.0
        hdr["CD1_1"] = -0.0001
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 0.0001
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        if sip:
            hdr["A_ORDER"] = 2
            hdr["B_ORDER"] = 2
            hdr["A_0_2"] = 1e-7
            hdr["B_0_2"] = 1e-7
    hdr["DATE-OBS"] = "2020-01-01T00:00:00"
    hdu1 = fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32), header=hdr)
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


class TestFfiListPaths(unittest.TestCase):
    def test_parquet_and_csv_under_scc(self):
        with tempfile.TemporaryDirectory() as tmp:
            pq = ffi_list_parquet_path(tmp, 20, 3, 3)
            csv = ffi_list_csv_path(tmp, 20, 3, 3)
            self.assertEqual(pq, scc_ffi_list_parquet(tmp, 20, 3, 3))
            self.assertEqual(csv, scc_ffi_list_csv(tmp, 20, 3, 3))
            self.assertTrue(str(pq).endswith("s0020/c3/k3/ffi_list.parquet"))
            self.assertTrue(str(csv).endswith("s0020/c3/k3/ffi_list.csv"))


class TestFfiListIngest(unittest.TestCase):
    def test_logical_filename_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "tess_s0020_3_3_ffic.fits"
            _write_test_ffi(plain)
            fz = Path(str(plain) + ".fz")
            plain.rename(fz)
            row = extract_ffi_header_record(fz, open_fits=wcs_grouping.open_fits_memmap)
            self.assertEqual(row["filename"], "tess_s0020_3_3_ffic.fits")
            self.assertTrue(row["wcs_ok"])

    def test_bad_wcs_still_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_ffic.fits"
            _write_test_ffi(path, include_wcs=False)
            row = extract_ffi_header_record(path, open_fits=wcs_grouping.open_fits_memmap)
            self.assertFalse(row["wcs_ok"])
            self.assertEqual(row["filename"], "bad_ffic.fits")

    def test_header_cards_sip_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sip_ffic.fits"
            _write_test_ffi(path, sip=True)
            row = extract_ffi_header_record(path, open_fits=wcs_grouping.open_fits_memmap)
            hdr = header_from_cached_row(pd.Series(row))
            self.assertIn("A_ORDER", hdr)
            wcs_from_cached_row(pd.Series(row))

    def test_upsert_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "ffi_list.parquet"
            path = Path(tmp) / "one_ffic.fits"
            _write_test_ffi(path, crval1=50.0)
            row1 = extract_ffi_header_record(path, open_fits=wcs_grouping.open_fits_memmap)
            upsert_ffi_list_rows(pq, [row1])
            _write_test_ffi(path, crval1=60.0)
            row2 = extract_ffi_header_record(path, open_fits=wcs_grouping.open_fits_memmap)
            upsert_ffi_list_rows(pq, [row2])
            df = load_ffi_list(pq)
            self.assertEqual(len(df), 1)
            hdr = header_from_cached_row(df.loc["one_ffic.fits"])
            self.assertAlmostEqual(float(hdr["CRVAL1"]), 60.0)

    def test_is_complete_with_bad_wcs_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_ffic.fits"
            _write_test_ffi(path, include_wcs=False)
            row = extract_ffi_header_record(path, open_fits=wcs_grouping.open_fits_memmap)
            df = pd.DataFrame([row]).set_index("filename")
            self.assertTrue(ffi_list_is_complete([path], df))

    def test_ensure_missing_keys_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
            ffi_dir.mkdir(parents=True)
            p1 = ffi_dir / "a_ffic.fits"
            p2 = ffi_dir / "b_ffic.fits"
            _write_test_ffi(p1, crval1=1.0)
            _write_test_ffi(p2, crval1=2.0)
            pq = ffi_list_parquet_path(data_root, 20, 1, 1)
            row1 = extract_ffi_header_record(p1, open_fits=wcs_grouping.open_fits_memmap)
            upsert_ffi_list_rows(pq, [row1])
            with unittest.mock.patch(
                "syndiff_pipeline.common.wcs_header_cache.extract_ffi_header_record",
                wraps=extract_ffi_header_record,
            ) as mocked:
                ensure_scc_ffi_list(
                    data_root,
                    20,
                    1,
                    1,
                    [p1, p2],
                    open_fits=wcs_grouping.open_fits_memmap,
                )
                self.assertEqual(mocked.call_count, 1)
            df = load_ffi_list(pq)
            self.assertEqual(len(df), 2)

    def test_no_wcs_cache_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
            ffi_dir.mkdir(parents=True)
            path = ffi_dir / "x_ffic.fits"
            _write_test_ffi(path)
            legacy = data_root / "s0020" / "c1" / "k1" / "wcs_cache.parquet"
            pd.DataFrame({"filename": ["x_ffic.fits.fz"], "CRVAL1": [999.0]}).to_parquet(
                legacy, index=False
            )
            df = ensure_scc_ffi_list(
                data_root,
                20,
                1,
                1,
                [path],
                open_fits=wcs_grouping.open_fits_memmap,
            )
            self.assertIn("header_cards", df.columns)
            hdr = header_from_cached_row(df.loc["x_ffic.fits"])
            self.assertAlmostEqual(float(hdr["CRVAL1"]), 100.0)


class TestCacheOnlyWcsTable(unittest.TestCase):
    def test_median_crval_from_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            rows = []
            for i, ra in enumerate((100.0, 102.0, 104.0)):
                p = Path(tmp) / f"f{i}_ffic.fits"
                _write_test_ffi(p, crval1=ra)
                paths.append(str(p))
                rows.append(
                    extract_ffi_header_record(p, open_fits=wcs_grouping.open_fits_memmap)
                )
            df = pd.DataFrame(rows).set_index("filename")
            med_ra, _ = median_crval_from_cache(df, paths)
            self.assertAlmostEqual(med_ra, 102.0)

    def test_build_wcs_table_from_cache_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, ra in enumerate((200.0, 200.1)):
                p = Path(tmp) / f"t{i}_ffic.fits"
                _write_test_ffi(p, crval1=ra, crval2=5.0)
                paths.append(str(p))
            target_ra, target_dec = 200.05, 5.0
            direct = wcs_grouping.build_wcs_table(paths, target_ra, target_dec)
            rows = [
                extract_ffi_header_record(p, open_fits=wcs_grouping.open_fits_memmap)
                for p in paths
            ]
            df = pd.DataFrame(rows).set_index("filename")
            cached = wcs_grouping.build_wcs_table_from_cache(
                df, paths, target_ra, target_dec
            )
            for col in ("delta_x", "delta_y", "x_pix", "y_pix", "wcs_ok"):
                np.testing.assert_allclose(
                    direct[col].astype(float),
                    cached[col].astype(float),
                    rtol=0,
                    atol=1e-6,
                    err_msg=col,
                )


class TestResolveSccReferenceFfiCacheOnly(unittest.TestCase):
    def test_no_fits_opens_when_ffi_list_warm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
                resolve_scc_reference_ffi,
            )
            from syndiff_pipeline.common.orchestration.targets import Target
            from syndiff_pipeline.template_creation.orchestration.runner_config import (
                ResolvedTargetConfig,
            )
            from syndiff_pipeline.template_creation.orchestration.stage_params import (
                DownsampleStageParams,
                MappingStageParams,
                Ps1DownloadStageParams,
                Ps1ProcessStageParams,
                RemapStageParams,
                TemplateStageParams,
                WcsGroupingStageParams,
            )

            data_root = tmp / "data"
            ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
            ffi_dir.mkdir(parents=True)
            paths = []
            for i in range(3):
                p = ffi_dir / f"tess-s0020-1-1-{i:04d}_ffic.fits"
                _write_test_ffi(p, crval1=100.0 + i * 0.01)
                paths.append(str(p))
            rebuild_scc_ffi_list(
                data_root, 20, 1, 1, paths, open_fits=wcs_grouping.open_fits_memmap
            )
            target = Target(
                target_name="test",
                sector=20,
                camera=1,
                ccd=1,
                target_ra=100.01,
                target_dec=10.0,
            )
            resolved = ResolvedTargetConfig(
                target=target,
                data_root=str(data_root),
                ffi_dir=str(ffi_dir),
                event_dir=str(tmp / "ws" / "events" / "e1" / "s0020_c1_k1"),
                skycell_wcs_csv=str(tmp / "skycell.csv"),
                stages=TemplateStageParams(
                    wcs_grouping=WcsGroupingStageParams(),
                    mapping=MappingStageParams(),
                    ps1_download=Ps1DownloadStageParams(),
                    ps1_process=Ps1ProcessStageParams(),
                    remap=RemapStageParams(),
                    downsample=DownsampleStageParams(),
                ),
                mapping_root=str(data_root / "s0020" / "c1" / "k1" / "mapping"),
                zarr_dir=str(data_root / "ps1_skycells_zarr"),
                template_output_base=str(data_root / "s0020" / "c1" / "k1" / "templates"),
            )
            with unittest.mock.patch(
                "syndiff_pipeline.common.wcs_grouping.extract_wcs_from_ffi",
                side_effect=AssertionError("extract_wcs_from_ffi should not run"),
            ):
                ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            self.assertTrue(Path(ref).is_file())


if __name__ == "__main__":
    unittest.main()
