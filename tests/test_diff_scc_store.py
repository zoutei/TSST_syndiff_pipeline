"""PR-D2 SCC-scoped diff store: write-through, index pointers, second-event reuse."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from syndiff_pipeline.common.scc_paths import scc_diff_workspace_dir, scc_diff_workspace_index_path
from syndiff_pipeline.difference_imaging.orchestration import diff_store
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.hotpants import _write_image_fits
import numpy as np


class TestDiffSccStore(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = Path(self._tmp()) / "data"
        self.event1_ws = Path(self._tmp()) / "event1" / "ws"
        self.event2_ws = Path(self._tmp()) / "event2" / "ws"
        self.event1_ws.mkdir(parents=True)
        self.event2_ws.mkdir(parents=True)

    def _tmp(self) -> Path:
        import tempfile

        if not hasattr(self, "_tmpdir"):
            self._tmpdir = tempfile.mkdtemp(prefix="diff_scc_store_")
        return Path(self._tmpdir)

    def test_scc_primary_write_records_workspace_index(self) -> None:
        hp = HotpantsParams()
        recipe_fp = diff_store.recipe_fp_for_artifact("diff_image", hp)
        self.assertIsNotNone(recipe_fp)

        write_path, scc_primary = diff_store.resolve_diff_write_path(
            data_root=str(self.data_root),
            sck=(20, 3, 3),
            kind="diff_image",
            stage_label="diffs_r1",
            product_id="tess123",
            label="diffs_r1",
            params=hp,
            workspace_path=self.event1_ws / "diffs_r1" / "tess123_diffs_r1.fits.fz",
        )
        self.assertTrue(scc_primary)
        _write_image_fits(str(write_path), np.zeros((4, 4), dtype=np.float32))

        diff_store.record_scc_artifact_pointer(
            workspace_root=str(self.event1_ws),
            product_id="tess123",
            label="diffs_r1",
            scc_path=str(write_path),
            kind="diff_image",
            fingerprint="fp_test",
            stage_label="diffs_r1",
            recipe_fp=str(recipe_fp),
        )

        index_path = scc_diff_workspace_index_path(self.event1_ws)
        self.assertTrue(index_path.is_file())
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertIn("tess123:diffs_r1", data["artifacts"])
        self.assertEqual(data["artifacts"]["tess123:diffs_r1"]["scc_path"], str(write_path))

        stage_dir = scc_diff_workspace_dir(
            self.data_root, 20, 3, 3, store_name=None, workspace_label="diffs_r1", recipe_fp=str(recipe_fp)
        )
        self.assertTrue(stage_dir.is_dir())
        self.assertTrue(write_path.is_file())

    def test_second_event_materializes_from_scc_store(self) -> None:
        hp = HotpantsParams()
        recipe_fp = diff_store.recipe_fp_for_artifact("diff_image", hp)
        assert recipe_fp is not None

        scc_path = diff_store.scc_diff_artifact_path(
            str(self.data_root),
            20,
            3,
            3,
            "diffs_r1",
            recipe_fp,
            "tess456",
            "diffs_r1",
        )
        _write_image_fits(str(scc_path), np.ones((8, 8), dtype=np.float32) * 3.0)
        diff_store.record_scc_artifact_pointer(
            workspace_root=str(self.event1_ws),
            product_id="tess456",
            label="diffs_r1",
            scc_path=str(scc_path),
            kind="diff_image",
            fingerprint="fp_event1",
            stage_label="diffs_r1",
            recipe_fp=recipe_fp,
        )

        event2_dest = self.event2_ws / "diffs_r1" / "tess456_diffs_r1.fits.fz"
        event2_dest.parent.mkdir(parents=True, exist_ok=True)
        self.assertFalse(event2_dest.is_file())

        ok = diff_store.try_materialize_workspace_artifact(
            data_root=str(self.data_root),
            sck=(20, 3, 3),
            kind="diff_image",
            stage_label="diffs_r1",
            product_id="tess456",
            label="diffs_r1",
            params=hp,
            workspace_dest=event2_dest,
            workspace_root=str(self.event2_ws),
        )
        self.assertTrue(ok)
        self.assertTrue(event2_dest.is_file())

        index2 = json.loads(
            scc_diff_workspace_index_path(self.event2_ws).read_text(encoding="utf-8")
        )
        self.assertIn("tess456:diffs_r1", index2["artifacts"])

    def test_no_scc_context_uses_workspace_path(self) -> None:
        hp = HotpantsParams()
        ws = self.event1_ws / "diffs_r1" / "tess1_diffs_r1.fits.fz"
        write_path, scc_primary = diff_store.resolve_diff_write_path(
            data_root=str(self.data_root),
            sck=None,
            kind="diff_image",
            stage_label="diffs_r1",
            product_id="tess1",
            label="diffs_r1",
            params=hp,
            workspace_path=ws,
        )
        self.assertFalse(scc_primary)
        self.assertEqual(write_path, ws)

    def test_no_data_root_uses_workspace_path(self) -> None:
        hp = HotpantsParams()
        ws = self.event1_ws / "diffs_r1" / "tess1b_diffs_r1.fits.fz"
        write_path, scc_primary = diff_store.resolve_diff_write_path(
            data_root=None,
            sck=(20, 3, 3),
            kind="diff_image",
            stage_label="diffs_r1",
            product_id="tess1b",
            label="diffs_r1",
            params=hp,
            workspace_path=ws,
        )
        self.assertFalse(scc_primary)
        self.assertEqual(write_path, ws)

    def test_resolve_diff_write_path_scc_primary(self) -> None:
        hp = HotpantsParams()
        ws = self.event1_ws / "diffs_r1" / "tess789_diffs_r1.fits.fz"
        write_path, scc_primary = diff_store.resolve_diff_write_path(
            data_root=str(self.data_root),
            sck=(20, 3, 3),
            kind="diff_image",
            stage_label="diffs_r1",
            product_id="tess789",
            label="diffs_r1",
            params=hp,
            workspace_path=ws,
            output_store_name="l4_split_smoke",
        )
        self.assertTrue(scc_primary)
        self.assertIn("diff_l4_split_smoke", str(write_path))
        self.assertNotEqual(write_path, ws)

    def test_write_through_lands_under_named_lane(self) -> None:
        """PR-4b: with data_root + SCC context, artifact path is under diff_{lane}/."""
        hp = HotpantsParams()
        lane = "field_smoke"
        ws = self.event1_ws / "hp_d" / "tess_wt_hp_d.fits.fz"
        write_path, scc_primary = diff_store.resolve_diff_write_path(
            data_root=str(self.data_root),
            sck=(20, 1, 1),
            kind="diff_image",
            stage_label="hp_d",
            product_id="tess_wt",
            label="hp_d",
            params=hp,
            workspace_path=ws,
            output_store_name=lane,
        )
        self.assertTrue(scc_primary)
        parts = write_path.parts
        self.assertIn(f"diff_{lane}", parts)
        self.assertIn("hp_d", parts)
        self.assertTrue(str(write_path).endswith("tess_wt_hp_d.fits.fz"))

        # Simulate SCC-primary write-through (no Hotpants): file lands on lane path.
        write_path.parent.mkdir(parents=True, exist_ok=True)
        _write_image_fits(str(write_path), np.ones((6, 6), dtype=np.float32))
        self.assertTrue(write_path.is_file())
        expected_root = scc_diff_workspace_dir(
            self.data_root,
            20,
            1,
            1,
            store_name=lane,
            workspace_label="hp_d",
            recipe_fp=str(diff_store.recipe_fp_for_artifact("diff_image", hp)),
        )
        self.assertEqual(write_path.parent, expected_root)
        self.assertFalse(ws.is_file())


if __name__ == "__main__":
    unittest.main()
