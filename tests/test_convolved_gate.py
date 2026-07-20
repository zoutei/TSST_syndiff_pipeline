"""Tests for common.provenance.convolved_gate (PR5 real numeric gate).

The gate used to just check that shared-store files existed and load, with
an inline comment admitting it never compared against the legacy per-SCC
``convolved.zarr`` -- so ``report["pass"]`` could never catch a wrong
implementation. This module tests the rewritten version: a real numeric
comparison, projection-level cross-projection-padding eligibility filtering,
and an explicit (non-stub) ``pass: False`` when no valid comparison exists.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from syndiff_pipeline.common.provenance.convolved_gate import convolved_gate_report
from syndiff_pipeline.common.scc_paths import ps1_convolved_zarr_path, scc_convolved_zarr
from syndiff_pipeline.template_creation.processing import convolved_store as cvs


def _write_master_csv(
    data_root: Path,
    sector: int,
    camera: int,
    ccd: int,
    rows: list[dict],
) -> None:
    """Legacy-layout CSV fallback ``find_csv_file`` accepts (see csv_utils.py)."""
    csv_dir = (
        data_root
        / "skycell_pixel_mapping"
        / f"sector_{sector:04d}"
        / f"camera_{camera}"
        / f"ccd_{ccd}"
    )
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"tess_s{sector:04d}_{camera}_{ccd}_master_skycells_list.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _write_legacy_convolved_cell(legacy_path: Path, skycell_name: str, array: np.ndarray) -> None:
    store = zarr.open(str(legacy_path), mode="a")
    compressor = {"name": "zstd", "configuration": {"level": 3}}
    name = f"{skycell_name}_data"
    if name in store:
        del store[name]
    store.create_array(name=name, data=array, chunks=array.shape, compressors=[compressor], fill_value=np.nan)


def _publish_shared_cell(data_root: Path, projection: str, cell: str, array: np.ndarray) -> None:
    recipe = cvs.convolved_recipe(psf_sigma=20.0)
    info = cvs.publish_convolved_cell(
        data_root,
        projection,
        cell,
        convolved_image=array,
        convolved_mask=np.zeros(array.shape, dtype=np.uint16),
        headers_data={"r": "H"},
        removed_stars=[],
        recipe=recipe,
        combined_fingerprint="test-combined-fp",
    )
    assert info is not None, "setup: failed to publish shared convolved cell"


class ConvolvedGateNumericComparisonTests(unittest.TestCase):
    SECTOR, CAMERA, CCD = 20, 3, 3

    def _clean_projection_rows(self) -> list[dict]:
        # No pad_skycell_* columns at all -> identify_all_padding_sources
        # finds zero requirements for every row -> the whole projection is
        # "clean" (eligible for comparison).
        return [
            {"projection": "skycell.1111", "y": 0, "NAME": "skycell.1111.001", "x": 0, "NAXIS1": 32, "NAXIS2": 32},
        ]

    def test_pass_when_shared_matches_legacy_on_a_padding_free_cell(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _write_master_csv(data_root, self.SECTOR, self.CAMERA, self.CCD, self._clean_projection_rows())

            rng = np.random.default_rng(0)
            array = rng.random((32, 32)).astype(np.float32)

            legacy_path = scc_convolved_zarr(data_root, self.SECTOR, self.CAMERA, self.CCD)
            _write_legacy_convolved_cell(legacy_path, "skycell.1111.001", array)
            _publish_shared_cell(data_root, "skycell.1111", "001", array)

            report = convolved_gate_report(
                data_root, sector=self.SECTOR, camera=self.CAMERA, ccd=self.CCD, sample_cells=10
            )

            self.assertTrue(report["pass"], report)
            self.assertEqual(len(report["compared"]), 1)
            self.assertEqual(report["compared"][0]["max_abs_diff"], 0.0)
            self.assertEqual(report["failures"], [])
            self.assertIn("skycell.1111", report["cross_projection_padding_diagnostics"]["clean_projections"])

    def test_fail_when_shared_diverges_from_legacy(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _write_master_csv(data_root, self.SECTOR, self.CAMERA, self.CCD, self._clean_projection_rows())

            rng = np.random.default_rng(1)
            legacy_array = rng.random((32, 32)).astype(np.float32)
            shared_array = legacy_array.copy()
            shared_array[5, 5] += 50.0  # a real, large divergence

            legacy_path = scc_convolved_zarr(data_root, self.SECTOR, self.CAMERA, self.CCD)
            _write_legacy_convolved_cell(legacy_path, "skycell.1111.001", legacy_array)
            _publish_shared_cell(data_root, "skycell.1111", "001", shared_array)

            report = convolved_gate_report(
                data_root, sector=self.SECTOR, camera=self.CAMERA, ccd=self.CCD, sample_cells=10
            )

            self.assertFalse(report["pass"], report)
            self.assertEqual(len(report["failures"]), 1)
            self.assertAlmostEqual(report["failures"][0]["max_abs_diff"], 50.0, places=4)

    def test_no_pass_stub_when_no_eligible_cells(self) -> None:
        """Every projection has a cross-projection padding requirement ->
        no valid comparison is possible -> pass must be False, not a silent
        stub pass on zero real comparisons."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            rows = [
                {
                    "projection": "skycell.1111",
                    "y": 0,
                    "NAME": "skycell.1111.001",
                    "x": 0,
                    "NAXIS1": 32,
                    "NAXIS2": 32,
                    # A single-cell row is simultaneously "first"/"last" in
                    # analyze_cell_positions, so a "left" padding requirement
                    # passes _parse_row_padding_requirements_df's validity
                    # check regardless of top/bottom row position.
                    "pad_skycell_left": "skycell.2222.099",
                },
            ]
            _write_master_csv(data_root, self.SECTOR, self.CAMERA, self.CCD, rows)

            legacy_path = scc_convolved_zarr(data_root, self.SECTOR, self.CAMERA, self.CCD)
            array = np.zeros((32, 32), dtype=np.float32)
            _write_legacy_convolved_cell(legacy_path, "skycell.1111.001", array)
            _publish_shared_cell(data_root, "skycell.1111", "001", array)

            report = convolved_gate_report(
                data_root, sector=self.SECTOR, camera=self.CAMERA, ccd=self.CCD, sample_cells=10
            )

            self.assertFalse(report["pass"])
            self.assertEqual(report["eligible_cell_count"], 0)
            self.assertIn("note", report)
            self.assertEqual(report["compared"], [])

    def test_missing_legacy_or_shared_store_reports_error_not_pass(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            report = convolved_gate_report(
                data_root, sector=self.SECTOR, camera=self.CAMERA, ccd=self.CCD
            )
            self.assertFalse(report["pass"])
            self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
