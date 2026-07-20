"""Tests for field-mode on-demand template loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from syndiff_pipeline.difference_imaging.support.template_resolution import (
    FieldModeTemplateContext,
    build_field_mode_template_loader,
    maybe_load_field_mode_template_context,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    write_contrib,
    write_template_manifest,
)


class TestFieldModeLoader(unittest.TestCase):
    def test_build_loader_assembles_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            store.mkdir()
            (store / "contribs").mkdir()
            ny, nx = 10, 12
            # flat index 1*12+2 = 14 gets flux 6
            write_contrib(
                store,
                "skycell.1.1",
                0,
                0,
                indices=np.array([14], dtype=np.int64),
                flux_sum=np.array([6.0]),
                count=np.array([2.0]),
                group_id=0,
            )
            shifts = pd.DataFrame(
                {
                    "group_id": np.array([0], dtype=np.int32),
                    "skycell": ["skycell.1.1"],
                    "sx_int": np.array([0], dtype=np.int16),
                    "sy_int": np.array([0], dtype=np.int16),
                    "qx": np.array([0.0], dtype=np.float32),
                    "qy": np.array([0.0], dtype=np.float32),
                    "cache_key": ["qx0"],
                }
            )
            ctx = FieldModeTemplateContext(
                store_root=str(store),
                shifts_df=shifts,
                base_tess_shape=(ny, nx),
                template_roi_bounds=(0, 0, nx, ny),
            )
            loader = build_field_mode_template_loader(
                ctx, {"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 3}
            )
            arr = loader(0)
            self.assertEqual(arr.shape, (3, 4))
            # mean flux at (1,2) within crop = 6/2 = 3
            self.assertAlmostEqual(float(arr[1, 2]), 3.0)

    def _write_maybe_load_fixture(self, tmp: str, *, schema_version: int) -> tuple[Path, Path]:
        event = Path(tmp) / "event"
        store = Path(tmp) / "store"
        event.mkdir()
        store.mkdir()
        (store / "contribs").mkdir()
        write_template_manifest(
            store,
            FieldManifest(
                geometry_mode="field",
                scope="scc",
                assembly="sparse_sum",
                materialize_fits=False,
                sector=20,
                camera=3,
                ccd=3,
                contribs_dir="contribs",
                groups=[{"group_id": 0, "n_frames": 1}],
            ),
        )
        payload = {
            "schema_version": schema_version,
            "store_root": str(store),
            "base_tess_shape": [10, 12],
            "roi_bounds": [0, 0, 12, 10],
            "oversampling_factor": 1,
            "ignore_mask": 0,
        }
        if schema_version >= 2:
            payload.update(
                {
                    "group_scoped_contribs": True,
                }
            )
        (store / "field_mode_assembly.json").write_text(json.dumps(payload))
        pd.DataFrame(
            {
                "group_id": [0],
                "skycell": ["skycell.1.1"],
                "sx_int": [0],
                "sy_int": [0],
                "qx": [0.0],
                "qy": [0.0],
                "cache_key": ["x"],
            }
        ).to_parquet(event / "template_group_shifts.parquet", index=False)
        return store, event

    def test_maybe_load_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, event = self._write_maybe_load_fixture(tmp, schema_version=1)
            ctx = maybe_load_field_mode_template_context(store, event)
            self.assertIsNotNone(ctx)
            self.assertEqual(ctx.base_tess_shape, (10, 12))
            self.assertEqual(ctx.oversampling_factor, 1)

    def test_maybe_load_context_schema_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, event = self._write_maybe_load_fixture(tmp, schema_version=2)
            ctx = maybe_load_field_mode_template_context(store, event)
            self.assertIsNotNone(ctx)
            self.assertEqual(ctx.store_root, str(store))
            self.assertEqual(ctx.base_tess_shape, (10, 12))

    def test_loader_scales_native_crop_when_oversampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            store.mkdir()
            (store / "contribs").mkdir()
            # HR full shape 20x24 = native 10x12 at F=2
            ny, nx = 20, 24
            # flat index in HR: y=3,x=5 -> 3*24+5 = 77, flux 8 count 2 -> mean 4
            write_contrib(
                store,
                "skycell.1.1",
                0,
                0,
                indices=np.array([77], dtype=np.int64),
                flux_sum=np.array([8.0]),
                count=np.array([2.0]),
                group_id=0,
            )
            shifts = pd.DataFrame(
                {
                    "group_id": np.array([0], dtype=np.int32),
                    "skycell": ["skycell.1.1"],
                    "sx_int": np.array([0], dtype=np.int16),
                    "sy_int": np.array([0], dtype=np.int16),
                    "qx": np.array([0.0], dtype=np.float32),
                    "qy": np.array([0.0], dtype=np.float32),
                    "cache_key": ["qx0"],
                }
            )
            ctx = FieldModeTemplateContext(
                store_root=str(store),
                shifts_df=shifts,
                base_tess_shape=(ny, nx),
                template_roi_bounds=(0, 0, nx, ny),
                oversampling_factor=2,
            )
            # native crop [0,4) x [0,3) -> HR [0,8) x [0,6)
            loader = build_field_mode_template_loader(
                ctx, {"x_min": 0, "x_max": 4, "y_min": 0, "y_max": 3}
            )
            arr = loader(0)
            self.assertEqual(arr.shape, (6, 8))
            self.assertAlmostEqual(float(arr[3, 5]), 4.0)


if __name__ == "__main__":
    unittest.main()
