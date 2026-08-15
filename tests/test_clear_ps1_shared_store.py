"""BK-7 (revised): force-rerun must NOT touch the shared convolved store.

Originally this scope-cleared "only this SCC's own expected cells" before
reprocessing. That was wrong: ``expected_ps1_process_skycells`` includes
cross-projection padding neighbors that routinely belong to OTHER SCCs
whose footprints overlap this one (CVZ-style repeated-pointing campaigns).
Since the store is keyed by sky position only (no sector/camera/ccd in the
path), the old clear deleted neighboring SCCs' already-published cells and
never republished them -- confirmed by exact skycell-set-intersection
arithmetic against real data loss across four separate CVZ SCC re-runs.
The store is content-addressed/fingerprinted, so clearing was never
required for correctness."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path, scc_convolved_zarr
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    RemapStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    clear_ps1_process_artifacts,
    verify_ps1_process,
)


def _resolved(
    data_root: Path,
    *,
    use_shared_convolved_store: bool = False,
    sector: int = 20,
    camera: int = 1,
    ccd: int = 1,
    mapping_csv_rows: str = (
        "NAME,projection,y,x,NAXIS1,NAXIS2\n"
        "skycell.1111.001,skycell.1111,0,0,32,32\n"
        "skycell.1111.002,skycell.1111,0,0,32,32\n"
    ),
) -> ResolvedTargetConfig:
    mapping_dir = (
        data_root / f"s{sector:04d}" / f"c{camera}" / f"k{ccd}" / "mapping" / "oversampling_1"
    )
    mapping_dir.mkdir(parents=True, exist_ok=True)
    csv_path = mapping_dir / f"tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list.csv"
    csv_path.write_text(mapping_csv_rows, encoding="utf-8")

    target = Target(
        sector=sector,
        camera=camera,
        ccd=ccd,
        target_ra=1.0,
        target_dec=2.0,
        target_name="clear-test",
    )
    return ResolvedTargetConfig(
        target=target,
        data_root=str(data_root),
        ffi_dir=str(data_root / f"s{sector:04d}" / f"c{camera}" / f"k{ccd}" / "ffi"),
        event_dir=str(data_root / "events" / "clear-test" / f"s{sector:04d}_c{camera}_k{ccd}"),
        skycell_wcs_csv=str(data_root / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(
                use_shared_convolved_store=use_shared_convolved_store,
                write_per_scc_convolved_zarr=not use_shared_convolved_store,
            ),
            remap=RemapStageParams(),
            downsample=DownsampleStageParams(),
        ),
        mapping_root=str(mapping_dir),
        zarr_dir=str(data_root / "ps1_skycells_zarr"),
        template_output_base=str(
            data_root / f"s{sector:04d}" / f"c{camera}" / f"k{ccd}" / "templates" / "oversampling_1"
        ),
    )


def _publish_shared_cell(data_root: Path, skycell_name: str, *, fp: str = "fp0001") -> Path:
    parts = skycell_name.split(".")
    projection = ".".join(parts[:2])
    cell = parts[2]
    cell_dir = (
        ps1_convolved_zarr_path(data_root) / projection / cell / fp
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "arrays.npz").write_bytes(b"x")
    return cell_dir.parent


class TestClearPs1SharedStore(unittest.TestCase):
    def test_legacy_mode_clears_per_scc_zarr_not_shared_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            resolved = _resolved(data_root, use_shared_convolved_store=False)
            legacy = scc_convolved_zarr(data_root, 20, 1, 1)
            legacy.mkdir(parents=True)
            (legacy / "skycell.1111.001_data").write_text("x", encoding="utf-8")
            csv_path = Path(str(legacy).replace(".zarr", "_removed_stars.csv"))
            csv_path.write_text("source_id\n1\n", encoding="utf-8")

            other_cell = _publish_shared_cell(data_root, "skycell.9999.042")

            removed = clear_ps1_process_artifacts(resolved)

            self.assertEqual(
                set(removed),
                {str(legacy), str(csv_path)},
            )
            self.assertFalse(legacy.exists())
            self.assertFalse(csv_path.exists())
            self.assertTrue(other_cell.is_dir())
            self.assertTrue((other_cell / "fp0001" / "arrays.npz").is_file())

    def test_shared_mode_does_not_touch_shared_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            resolved = _resolved(data_root, use_shared_convolved_store=True)
            shared_root = ps1_convolved_zarr_path(data_root)

            cell_001 = _publish_shared_cell(data_root, "skycell.1111.001")
            cell_002 = _publish_shared_cell(data_root, "skycell.1111.002")
            other_cell = _publish_shared_cell(data_root, "skycell.9999.042")

            legacy = scc_convolved_zarr(data_root, 20, 1, 1)
            legacy.mkdir(parents=True)
            (legacy / "stale_data").write_text("x", encoding="utf-8")
            csv_path = Path(str(legacy).replace(".zarr", "_removed_stars.csv"))
            csv_path.write_text("source_id\n1\n", encoding="utf-8")

            removed = set(clear_ps1_process_artifacts(resolved))

            self.assertEqual(removed, {str(csv_path)})
            self.assertTrue(cell_001.is_dir(), "shared mode must not remove this SCC's own cells")
            self.assertTrue(cell_002.is_dir(), "shared mode must not remove this SCC's own cells")
            self.assertTrue(shared_root.is_dir())
            self.assertTrue(other_cell.is_dir(), "shared mode must not remove overlapping neighbor SCCs' cells")
            self.assertTrue(legacy.is_dir(), "shared mode must not remove legacy per-SCC zarr")
            self.assertFalse(csv_path.exists())

            result = verify_ps1_process(resolved)
            self.assertTrue(result.ok, "already-published cells must remain verifiably complete")

    def test_shared_clear_is_idempotent_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            resolved = _resolved(data_root, use_shared_convolved_store=True)
            cell = _publish_shared_cell(data_root, "skycell.1111.001")

            first = clear_ps1_process_artifacts(resolved)
            self.assertEqual(first, [])
            self.assertEqual(clear_ps1_process_artifacts(resolved), [])
            self.assertTrue(cell.is_dir())


if __name__ == "__main__":
    unittest.main()
