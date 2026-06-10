"""Tests for tess_ffi_download and ps1_download artifact verification."""
from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import zarr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_creation.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.verify import (
    stage_complete,
    verify_ps1_download,
    verify_tess_ffi_download,
)


def _resolved(tmp: Path, csv_path: Path | None = None) -> ResolvedTargetConfig:
    target = Target(
        sector=22,
        camera=3,
        ccd=3,
        target_ra=228.0,
        target_dec=52.0,
        target_name="2020dgc",
    )
    mapping_root = (
        csv_path.parent.parent.parent.parent
        if csv_path is not None
        else tmp / "skycell_pixel_mapping"
    )
    return ResolvedTargetConfig(
        target=target,
        data_root=str(tmp / "data"),
        ffi_dir=str(tmp / "data" / "tess_ffi"),
        handoff_dir=str(tmp / "handoff" / target.label()),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            downsample=DownsampleStageParams(),
        ),
        mapping_root=str(mapping_root),
        zarr_dir=str(tmp / "data" / "ps1_skycells_zarr"),
        template_output_base=str(tmp / "shifted_downsampled"),
    )


def _write_tesscurl_manifest(path: Path, sector: int, camera: int, ccd: int, basenames: list[str]) -> None:
    lines = [
        "#!/bin/bash",
        *(
            f'curl -C - -o {name} https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/{name}'
            for name in basenames
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mapping_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["NAME,projection", *(f"{name},{proj}" for name, proj in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_complete_ps1_skycell(root, skycell_name: str) -> None:
    projection_id = skycell_name.split(".")[1]
    if projection_id not in root:
        root.create_group(projection_id)
    group = root[projection_id].create_group(skycell_name)
    # Write actual chunk data: the fast verifier treats an array as complete only
    # when at least one chunk is materialized on disk (mirrors the real download,
    # which always writes pixel data).
    for band in ("r", "i", "z", "y"):
        group.create_array(band, shape=(4, 4), dtype="f4")[:] = 1.0
        group.create_array(f"{band}_mask", shape=(4, 4), dtype="u1")[:] = 1
        group.create_array(f"{band}_wt", shape=(4, 4), dtype="f4")[:] = 1.0


class TestVerifyTessFfiDownload(unittest.TestCase):
    def test_no_files_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            ffi_leaf = Path(resolved.ffi_dir) / "s0022" / "cam3_ccd3"
            _write_tesscurl_manifest(
                ffi_leaf / "tesscurl_sector_22_ffic.sh",
                sector=22,
                camera=3,
                ccd=3,
                basenames=["tess2020019142923-s0022-3-3-0165-s_ffic.fits"],
            )

            result = verify_tess_ffi_download(resolved)
            self.assertFalse(result.ok)
            self.assertIn("Partial FFI download", result.message)
            self.assertIn("0/1", result.message)

    def test_partial_download_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            ffi_leaf = Path(resolved.ffi_dir) / "s0022" / "cam3_ccd3"
            names = [
                "tess2020019142923-s0022-3-3-0165-s_ffic.fits",
                "tess2020019142924-s0022-3-3-0166-s_ffic.fits",
            ]
            _write_tesscurl_manifest(ffi_leaf / "tesscurl_sector_22_ffic.sh", 22, 3, 3, names)
            (ffi_leaf / names[0]).write_bytes(b"fits")

            result = verify_tess_ffi_download(resolved)
            self.assertFalse(result.ok)
            self.assertIn("Partial FFI download", result.message)
            self.assertIn("1/2", result.message)

    def test_complete_download_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            ffi_leaf = Path(resolved.ffi_dir) / "s0022" / "cam3_ccd3"
            names = [
                "tess2020019142923-s0022-3-3-0165-s_ffic.fits",
                "tess2020019142924-s0022-3-3-0166-s_ffic.fits",
            ]
            _write_tesscurl_manifest(ffi_leaf / "tesscurl_sector_22_ffic.sh", 22, 3, 3, names)
            for name in names:
                (ffi_leaf / name).write_bytes(b"fits")

            result = verify_tess_ffi_download(resolved)
            self.assertTrue(result.ok)
            self.assertIn("2 FFI files", result.message)

    def test_manifest_unavailable_is_unknown_tristate(self):
        # When the tesscurl manifest is unavailable, expected_ffi_basenames
        # returns None and completeness is genuinely undeterminable: the result
        # must be tri-state unknown (ok=False, unknown=True) and stage_complete
        # must treat it as NOT complete (so it is not skipped, but also not a
        # forced rerun-triggering hard FAIL).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            with unittest.mock.patch(
                "syndiff_pipeline.common.download.expected_ffi_basenames",
                return_value=None,
            ):
                result = verify_tess_ffi_download(resolved)
                self.assertFalse(result.ok)
                self.assertTrue(result.unknown)
                self.assertIn("manifest unavailable", result.message)
                self.assertFalse(stage_complete(resolved, "tess_ffi_download"))


class TestVerifyPs1Download(unittest.TestCase):
    def test_missing_mapping_csv_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            resolved = _resolved(tmp)
            result = verify_ps1_download(resolved)
            self.assertFalse(result.ok)
            self.assertIn("Master skycells CSV missing", result.message)

    def test_partial_zarr_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = (
                tmp
                / "skycell_pixel_mapping"
                / "sector_0022"
                / "camera_3"
                / "ccd_3"
                / "tess_s0022_3_3_master_skycells_list.csv"
            )
            _write_mapping_csv(
                csv_path,
                [
                    ("skycell.1520.080", "1520"),
                    ("skycell.1520.081", "1520"),
                ],
            )
            resolved = _resolved(tmp, csv_path)
            zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            _write_complete_ps1_skycell(root, "skycell.1520.080")

            result = verify_ps1_download(resolved)
            self.assertFalse(result.ok)
            self.assertIn("Partial PS1 zarr", result.message)
            self.assertIn("1/2", result.message)

    def test_complete_zarr_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = (
                tmp
                / "skycell_pixel_mapping"
                / "sector_0022"
                / "camera_3"
                / "ccd_3"
                / "tess_s0022_3_3_master_skycells_list.csv"
            )
            _write_mapping_csv(
                csv_path,
                [
                    ("skycell.1520.080", "1520"),
                    ("skycell.1520.081", "1520"),
                ],
            )
            resolved = _resolved(tmp, csv_path)
            zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            _write_complete_ps1_skycell(root, "skycell.1520.080")
            _write_complete_ps1_skycell(root, "skycell.1520.081")

            result = verify_ps1_download(resolved)
            self.assertTrue(result.ok)
            self.assertIn("2/2", result.message)

    def test_skycell_missing_weight_arrays_is_incomplete(self):
        # Mirrors real older projections (e.g. 1347/1348) that have band + mask
        # arrays but no *_wt: ps1_process consumes weights, so the writer's
        # 12-array definition treats these as incomplete and the verifier must
        # agree (rather than falsely passing them).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = (
                tmp
                / "skycell_pixel_mapping"
                / "sector_0022"
                / "camera_3"
                / "ccd_3"
                / "tess_s0022_3_3_master_skycells_list.csv"
            )
            _write_mapping_csv(csv_path, [("skycell.1520.080", "1520")])
            resolved = _resolved(tmp, csv_path)
            zarr_path = Path(resolved.zarr_dir) / "ps1_skycells.zarr"
            root = zarr.open(str(zarr_path), mode="w")
            group = root.create_group("1520").create_group("skycell.1520.080")
            for band in ("r", "i", "z", "y"):
                group.create_array(band, shape=(4, 4), dtype="f4")[:] = 1.0
                group.create_array(f"{band}_mask", shape=(4, 4), dtype="u1")[:] = 1

            result = verify_ps1_download(resolved)
            self.assertFalse(result.ok)
            self.assertIn("0/1", result.message)


if __name__ == "__main__":
    unittest.main()
