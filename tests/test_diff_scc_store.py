"""PR-D2 SCC-scoped diff store: mirror, index pointers, second-event reuse."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from syndiff_pipeline.common.scc_paths import scc_diff_stage_dir, scc_diff_workspace_index_path
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

    def test_mirror_records_workspace_index(self) -> None:
        src = Path(self._tmp()) / "src.fits.fz"
        _write_image_fits(str(src), np.zeros((4, 4), dtype=np.float32))
        hp = HotpantsParams()
        recipe_fp = diff_store.recipe_fp_for_artifact("diff_image", hp)
        self.assertIsNotNone(recipe_fp)

        dest = diff_store.publish_mirror(
            publish_scc=True,
            data_root=str(self.data_root),
            sector=20,
            camera=3,
            ccd=3,
            stage_label="diffs_r1",
            recipe_fp=str(recipe_fp),
            product_id="tess123",
            label="diffs_r1",
            source_path=str(src),
            fingerprint="fp_test",
            workspace_root=str(self.event1_ws),
            kind="diff_image",
        )
        self.assertIsNotNone(dest)
        self.assertTrue(Path(dest).is_file())

        index_path = scc_diff_workspace_index_path(self.event1_ws)
        self.assertTrue(index_path.is_file())
        data = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertIn("tess123:diffs_r1", data["artifacts"])

        stage_dir = scc_diff_stage_dir(
            self.data_root, 20, 3, 3, "diffs_r1", str(recipe_fp)
        )
        self.assertTrue(stage_dir.is_dir())

    def test_second_event_materializes_from_scc_store(self) -> None:
        src = Path(self._tmp()) / "event1_product.fits.fz"
        _write_image_fits(str(src), np.ones((8, 8), dtype=np.float32) * 3.0)
        hp = HotpantsParams()
        recipe_fp = diff_store.recipe_fp_for_artifact("diff_image", hp)
        assert recipe_fp is not None

        diff_store.publish_mirror(
            publish_scc=True,
            data_root=str(self.data_root),
            sector=20,
            camera=3,
            ccd=3,
            stage_label="diffs_r1",
            recipe_fp=recipe_fp,
            product_id="tess456",
            label="diffs_r1",
            source_path=str(src),
            fingerprint="fp_event1",
            workspace_root=str(self.event1_ws),
            kind="diff_image",
        )

        event2_dest = self.event2_ws / "diffs_r1" / "tess456_diffs_r1.fits.fz"
        event2_dest.parent.mkdir(parents=True, exist_ok=True)
        self.assertFalse(event2_dest.is_file())

        ok = diff_store.try_materialize_workspace_artifact(
            publish_scc=True,
            data_root=str(self.data_root),
            sector=20,
            camera=3,
            ccd=3,
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

    def test_publish_scc_off_skips_mirror(self) -> None:
        src = Path(self._tmp()) / "noop.fits.fz"
        _write_image_fits(str(src), np.zeros((2, 2), dtype=np.float32))
        dest = diff_store.publish_mirror(
            publish_scc=False,
            data_root=str(self.data_root),
            sector=20,
            camera=3,
            ccd=3,
            stage_label="diffs_r1",
            recipe_fp="abc",
            product_id="tess1",
            label="diffs_r1",
            source_path=str(src),
            fingerprint=None,
            workspace_root=str(self.event1_ws),
        )
        self.assertIsNone(dest)


if __name__ == "__main__":
    unittest.main()
