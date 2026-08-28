"""SCC-scoped diff store: flat lane layout and write-path resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from syndiff_pipeline.common.scc_paths import scc_diff_label_dir, scc_diff_workspace_dir
from syndiff_pipeline.difference_imaging.orchestration import diff_store
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.hotpants import _write_image_fits
import numpy as np


class TestDiffSccStore(unittest.TestCase):
    def setUp(self) -> None:
        self.data_root = Path(self._tmp()) / "data"
        self.event1_ws = Path(self._tmp()) / "event1" / "ws"

    def _tmp(self) -> Path:
        import tempfile

        if not hasattr(self, "_tmpdir"):
            self._tmpdir = tempfile.mkdtemp(prefix="diff_scc_store_")
        return Path(self._tmpdir)

    def test_flat_lane_write_path(self) -> None:
        hp = HotpantsParams()
        ffi_stem = "tess2020057105921-s0020-3-3"
        write_path = diff_store.resolve_diff_write_path(
            data_root=str(self.data_root),
            sck=(20, 3, 3),
            kind="diff_image",
            stage_label="diffs_r1",
            ffi_stem=ffi_stem,
            label="diffs_r1",
            params=hp,
        )
        expected = (
            scc_diff_label_dir(self.data_root, 20, 3, 3, store_name=None, label="diffs_r1")
            / f"{ffi_stem}_diffs_r1.fits.fz"
        )
        self.assertEqual(write_path, expected)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        _write_image_fits(str(write_path), np.zeros((4, 4), dtype=np.float32))
        self.assertTrue(write_path.is_file())

    def test_scc_diff_artifact_path_named_lane(self) -> None:
        hp = HotpantsParams()
        lane = "field_smoke"
        ffi_stem = "tess2020057105921-s0020-1-1"
        write_path = diff_store.resolve_diff_write_path(
            data_root=str(self.data_root),
            sck=(20, 1, 1),
            kind="diff_image",
            stage_label="hp_d",
            ffi_stem=ffi_stem,
            label="hp_d",
            params=hp,
            output_store_name=lane,
        )
        self.assertIn("diff_field_smoke", str(write_path))
        self.assertEqual(
            write_path.parent,
            scc_diff_workspace_dir(
                self.data_root,
                20,
                1,
                1,
                store_name=lane,
                workspace_label="hp_d",
            ),
        )
        self.assertTrue(str(write_path).endswith(f"{ffi_stem}_hp_d.fits.fz"))

    def test_resolve_diff_write_path_requires_scc_context(self) -> None:
        hp = HotpantsParams()
        with self.assertRaises(ValueError):
            diff_store.resolve_diff_write_path(
                data_root="",
                sck=(20, 3, 3),
                kind="diff_image",
                stage_label="diffs_r1",
                ffi_stem="tess2020-s0020-3-3",
                label="diffs_r1",
                params=hp,
            )

    def test_recipe_fp_for_artifact_still_available(self) -> None:
        hp = HotpantsParams()
        recipe_fp = diff_store.recipe_fp_for_artifact("diff_image", hp)
        self.assertIsNotNone(recipe_fp)

    def test_scc_primary_write_does_not_create_event_ws_fits(self) -> None:
        """SCC-primary hotpants writes to data_root lane, not events/.../ws/hp_d/."""
        from astropy.io import fits
        from unittest.mock import patch

        from syndiff_pipeline.difference_imaging.stages import hotpants as hp_mod

        root = Path(self._tmp())
        event_ws_hp_d = root / "events" / "s0020_c3_k3" / "ws" / "hp_d"
        event_ws_hp_d.mkdir(parents=True)
        event_ws_hp_c = root / "events" / "s0020_c3_k3" / "ws" / "hp_c"
        event_ws_hp_c.mkdir(parents=True)
        dirs = hp_mod.HotpantsWorkspaceDirs(
            diffs=str(event_ws_hp_d),
            convolved=str(event_ws_hp_c),
        )
        shape = (8, 8)
        product_id = "2020057105921"
        ffi_stem = f"tess{product_id}-s0020-3-3"
        hp = HotpantsParams(write_convolved=False, write_bkg=False, write_stamps=False)
        sci = np.ones(shape, dtype=np.float64)
        tmpl = np.ones(shape, dtype=np.float64) * 0.5

        with (
            patch.object(hp_mod.wcs_grouping, "crop_ffi_header", return_value=fits.Header()),
            patch.object(hp_mod, "_load_template_cropped", return_value=tmpl),
            patch.object(hp_mod, "_resolve_linear_template_pad", return_value=0),
            patch.object(hp_mod, "_load_ffi_cropped", return_value=(sci, np.ones(shape))),
            patch.object(hp_mod, "kernel_sum_at_center", return_value=0.02),
            patch.object(
                hp_mod,
                "run_hotpants_frame",
                return_value={
                    "success": True,
                    "diff": sci,
                    "bkg": None,
                    "convolved": None,
                    "noise": None,
                    "mask": None,
                    "kernel_params_arrays": {},
                },
            ),
        ):
            result = hp_mod._process_one_frame(
                ffi_path=f"/fake/{ffi_stem}.fits",
                product_id=product_id,
                group_id=0,
                hp=hp,
                template_path_map={0: "/fake/template.fits"},
                mask=np.zeros(shape, dtype=np.uint8),
                crop_bounds={"x_min": 0, "x_max": 8, "y_min": 0, "y_max": 8},
                ref_stars_xy=np.array([[4.0, 4.0]]),
                dirs=dirs,
                round_id=1,
                sck=(20, 3, 3),
                data_root=str(self.data_root),
            )

        self.assertTrue(result["success"])
        scc_fits = (
            scc_diff_label_dir(self.data_root, 20, 3, 3, store_name=None, label="hp_d")
            / f"{ffi_stem}_hp_d.fits.fz"
        )
        self.assertTrue(scc_fits.is_file(), f"expected SCC write at {scc_fits}")
        self.assertFalse(
            any(event_ws_hp_d.glob("*.fits*")),
            "event ws/hp_d must not receive diff FITS in SCC-primary mode",
        )


if __name__ == "__main__":
    unittest.main()
