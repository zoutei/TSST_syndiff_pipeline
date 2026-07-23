"""Tests for SCC point-drift / reference-FFI drift dedupe."""

from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from syndiff_pipeline.common import wcs_grouping
from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import scc_remap_dir, scc_templates_dir
from syndiff_pipeline.common.wcs_header_cache import rebuild_scc_ffi_list
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    bootstrap_scc_diff_linear,
)
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
from syndiff_pipeline.template_creation.processing.linear_downsample import (
    LINEAR_ASSEMBLY_BASENAME,
)
from syndiff_pipeline.template_creation.processing.scc_reference_ffi import (
    load_mapping_reference_ffi,
    mapping_run_meta_path,
    resolve_cached_or_select_reference_ffi,
    resolve_scc_point_drift_table,
    resolve_scc_reference_ffi,
    scc_wcs_drift_debug_plot_path,
    write_scc_wcs_drift_debug_plot,
)


def _write_test_ffi(
    path: Path,
    *,
    crval1: float = 100.0,
    crval2: float = 10.0,
) -> None:
    hdu0 = fits.PrimaryHDU()
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 32
    hdr["NAXIS2"] = 32
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
    hdr["DATE-OBS"] = "2020-01-01T00:00:00"
    hdu1 = fits.ImageHDU(data=np.zeros((32, 32), dtype=np.float32), header=hdr)
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def _make_resolved(tmp: Path, *, oversampling_factor: int = 1) -> ResolvedTargetConfig:
    data_root = tmp / "data"
    ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
    ffi_dir.mkdir(parents=True)
    paths = []
    for i in range(5):
        p = ffi_dir / f"tess202001914292{i}-s0020-1-1-0165-s_ffic.fits"
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
        target_ra=100.02,
        target_dec=10.0,
    )
    return ResolvedTargetConfig(
        target=target,
        data_root=str(data_root),
        ffi_dir=str(ffi_dir),
        event_dir=str(tmp / "ws" / "events" / "e1" / "s0020_c1_k1"),
        skycell_wcs_csv=str(tmp / "skycell.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(offset_threshold=0.01),
            mapping=MappingStageParams(oversampling_factor=oversampling_factor),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            remap=RemapStageParams(),
            downsample=DownsampleStageParams(oversampling_factor=oversampling_factor),
        ),
        mapping_root=str(data_root / "s0020" / "c1" / "k1" / "mapping"),
        zarr_dir=str(data_root / "ps1_skycells_zarr"),
        template_output_base=str(data_root / "s0020" / "c1" / "k1" / "templates"),
    )


class TestResolveSccReferenceFfiNoPlot(unittest.TestCase):
    def test_mapping_default_does_not_write_debug_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            with unittest.mock.patch(
                "syndiff_pipeline.template_creation.processing.scc_reference_ffi."
                "write_scc_wcs_drift_debug_plot"
            ) as plot_mock:
                ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            self.assertTrue(Path(ref).is_file())
            plot_mock.assert_not_called()
            self.assertFalse(scc_wcs_drift_debug_plot_path(resolved).is_file())


class TestLightweightRefLoad(unittest.TestCase):
    def test_downsample_force_rerun_does_not_rewrite_mapping_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ref1 = resolve_scc_reference_ffi(resolved, force_rerun=True)
            meta_path = mapping_run_meta_path(resolved)
            before = json.loads(meta_path.read_text(encoding="utf-8"))

            # Simulate downsample --force-rerun path: never reselect.
            with unittest.mock.patch(
                "syndiff_pipeline.common.wcs_grouping.choose_reference_ffi_path",
                side_effect=AssertionError("must not reselect reference FFI"),
            ):
                ref2 = resolve_cached_or_select_reference_ffi(resolved)

            self.assertEqual(Path(ref1).resolve(), Path(ref2).resolve())
            after = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(before["reference_ffi_path"], after["reference_ffi_path"])
            self.assertEqual(before["recorded_at"], after["recorded_at"])

    def test_missing_run_meta_falls_back_to_resolve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            self.assertIsNone(load_mapping_reference_ffi(resolved))
            ref = resolve_cached_or_select_reference_ffi(resolved)
            self.assertTrue(Path(ref).is_file())
            self.assertEqual(load_mapping_reference_ffi(resolved), ref)


class TestPointDriftAndPlot(unittest.TestCase):
    def test_cold_path_builds_wcs_table_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            store = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=1, store_name="linear"
            )
            real_build = (
                "syndiff_pipeline.common.wcs_grouping.build_wcs_table_from_cache"
            )
            with unittest.mock.patch(
                real_build, wraps=wcs_grouping.build_wcs_table_from_cache
            ) as build_mock:
                wcs_table, drift = resolve_scc_point_drift_table(
                    resolved, ref_ffi_path=ref, store_root=store, force_rerun=True
                )
                write_scc_wcs_drift_debug_plot(
                    resolved, ref, wcs_table=wcs_table, force_rerun=True
                )
            self.assertEqual(build_mock.call_count, 1)
            self.assertEqual(drift.shape[0], 5)
            self.assertIn("group_id", wcs_table.columns)
            self.assertTrue(scc_wcs_drift_debug_plot_path(resolved).is_file())

    def test_cache_hit_skips_wcs_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            store = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=1, store_name="linear"
            )
            resolve_scc_point_drift_table(
                resolved, ref_ffi_path=ref, store_root=store, force_rerun=True
            )
            with unittest.mock.patch(
                "syndiff_pipeline.common.wcs_grouping.build_wcs_table_from_cache",
                side_effect=AssertionError("should not rebuild"),
            ):
                wcs_table, _ = resolve_scc_point_drift_table(
                    resolved, ref_ffi_path=ref, store_root=store, force_rerun=False
                )
            self.assertIn("group_id", wcs_table.columns)

    def test_stale_png_overwritten_from_point_drift_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            plot_path = scc_wcs_drift_debug_plot_path(resolved)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_bytes(b"stale-median-crval-png")

            store = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=1, store_name="linear"
            )
            wcs_table, _ = resolve_scc_point_drift_table(
                resolved, ref_ffi_path=ref, store_root=store, force_rerun=True
            )
            with unittest.mock.patch(
                "syndiff_pipeline.common.wcs_grouping.plot_wcs_drift_and_template_assignment",
            ) as plot_fn:
                def _fake_plot(table, out, **kwargs):
                    Path(out).write_bytes(b"point-drift-png")
                    return out

                plot_fn.side_effect = _fake_plot
                write_scc_wcs_drift_debug_plot(
                    resolved, ref, wcs_table=wcs_table, force_rerun=False
                )
            self.assertEqual(plot_path.read_bytes(), b"point-drift-png")
            plot_fn.assert_called_once()
            passed = plot_fn.call_args[0][0]
            self.assertIn("group_id", passed.columns)
            n_groups = int(passed.loc[passed["group_id"] >= 0, "group_id"].nunique())
            self.assertGreaterEqual(n_groups, 1)

    def test_plot_with_wcs_table_does_not_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            store = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=1, store_name="linear"
            )
            wcs_table, _ = resolve_scc_point_drift_table(
                resolved, ref_ffi_path=ref, store_root=store, force_rerun=True
            )
            with unittest.mock.patch(
                "syndiff_pipeline.common.wcs_grouping.build_wcs_table_from_cache",
                side_effect=AssertionError("plot must not rebuild"),
            ), unittest.mock.patch(
                "syndiff_pipeline.common.wcs_grouping.plot_wcs_drift_and_template_assignment",
            ) as plot_fn:
                plot_fn.side_effect = lambda table, out, **kw: Path(out).write_bytes(b"x")
                write_scc_wcs_drift_debug_plot(
                    resolved, ref, wcs_table=wcs_table, force_rerun=True
                )
            plot_fn.assert_called_once()

    def test_explicit_reference_ffi_override_in_point_drift_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ffi_dir = Path(resolved.ffi_dir)
            override = sorted(ffi_dir.glob("*.fits"))[2]
            ref = resolve_scc_reference_ffi(
                resolved, force_rerun=True, override_path=str(override)
            )
            self.assertEqual(Path(ref).resolve(), override.resolve())
            store = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=1, store_name="linear"
            )
            resolve_scc_point_drift_table(
                resolved, ref_ffi_path=ref, store_root=store, force_rerun=True
            )
            meta = json.loads((store / "point_drift_meta.json").read_text())
            self.assertEqual(
                Path(meta["reference_ffi_path"]).resolve(), override.resolve()
            )
            self.assertIn("group_id", pd.read_csv(store / "point_drift_table.csv").columns)

    def test_oversampling_4_lane_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir), oversampling_factor=4)
            ref = resolve_scc_reference_ffi(resolved, force_rerun=True)
            store = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=4, store_name="linear"
            )
            resolve_scc_point_drift_table(
                resolved, ref_ffi_path=ref, store_root=store, force_rerun=True
            )
            self.assertTrue((store / "point_drift_table.csv").is_file())
            self.assertIn("oversampling_4", str(store))
            os1 = scc_remap_dir(
                resolved.data_root, 20, 1, 1, oversampling_factor=1, store_name="linear"
            )
            self.assertFalse((os1 / "point_drift_table.csv").is_file())


class TestRemapPointVsPerSkycell(unittest.TestCase):
    def test_per_skycell_does_not_require_point_drift_plot(self):
        """Remap drift_source=per_skycell must not depend on point-drift PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _make_resolved(Path(tmpdir))
            ref = resolve_scc_reference_ffi(
                resolved, force_rerun=True, write_debug_plot=False
            )
            self.assertTrue(Path(ref).is_file())
            self.assertFalse(scc_wcs_drift_debug_plot_path(resolved).is_file())


def test_bootstrap_scc_diff_linear_smoke(tmp_path):
    data_root = tmp_path / "data"
    sector, camera, ccd = 20, 1, 1
    grid = MappingGrid.from_ffi_shape(2048, 2048)
    os_factor = 1

    tmpl = scc_templates_dir(
        data_root, sector, camera, ccd, oversampling_factor=os_factor, store_name="linear"
    )
    tmpl.mkdir(parents=True)
    sidecar = {
        "schema_version": 3,
        "mapping_grid": grid.to_mapping_dict(),
        "base_tess_shape": list(grid.array_shape_native()),
        "oversampling_factor": os_factor,
    }
    (tmpl / LINEAR_ASSEMBLY_BASENAME).write_text(json.dumps(sidecar))

    remap = scc_remap_dir(
        data_root, sector, camera, ccd, oversampling_factor=os_factor, store_name="linear"
    )
    remap.mkdir(parents=True)
    names = [
        "tess2020019142923-s0020-1-1-0165-s_ffic.fits",
        "tess2020019142924-s0020-1-1-0165-s_ffic.fits",
        "tess2020019142925-s0020-1-1-0165-s_ffic.fits",
    ]
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "filename": name,
                "group_id": i % 2,
                "group_dx": 0.01 * (i % 2),
                "group_dy": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(remap / "point_drift_table.csv", index=False)

    ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
    ffi_dir.mkdir(parents=True)
    for name in names:
        (ffi_dir / name).write_bytes(b"")

    result = bootstrap_scc_diff_linear(
        data_root=data_root,
        sector=sector,
        camera=camera,
        ccd=ccd,
        template_store_name="linear",
        output_store_name="smoke",
        remap_store_name="linear",
    )
    assert result.crop_bounds == grid.science_ffi_bounds()
    assert result.frames_csv_path.is_file()
    assert len(result.frames_df) == 3
    assert set(result.frames_df["group_id"].tolist()) == {0, 1}


if __name__ == "__main__":
    unittest.main()
