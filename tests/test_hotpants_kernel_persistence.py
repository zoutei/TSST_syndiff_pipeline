"""Tests for optional per-frame Hotpants kernel persistence."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages import hotpants
from syndiff_pipeline.difference_imaging.stages.kernel import load_frame_kernel


class TestFrameKernelsDir(unittest.TestCase):
    def test_sibling_naming(self):
        self.assertEqual(
            hotpants.frame_kernels_dir("/data/ws/ks_d"),
            os.path.abspath("/data/ws/ks_d_kernels"),
        )
        self.assertEqual(
            hotpants.frame_kernels_dir("/data/ws/hp_d"),
            os.path.abspath("/data/ws/hp_d_kernels"),
        )


class TestWriteLoadFrameKernel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.kernels_dir = Path(self._tmp.name) / "hp_d_kernels"
        self.product_id = "tess2026039233236"
        self.hp = HotpantsParams()
        self.hp_config = hotpants.build_hotpants_config(
            hp=self.hp,
            diff_dir=str(Path(self._tmp.name) / "hp_d"),
            convolved_dir=str(Path(self._tmp.name) / "hp_c"),
            frame_stem=f"{self.product_id}_hp_d",
            write_stamps=False,
        )
        self.kernel_solution = np.linspace(0.1, 1.0, self.hp_config.n_comp_total + 1)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        hotpants.write_frame_kernel_npz(
            str(self.kernels_dir),
            self.product_id,
            self.kernel_solution,
            self.hp_config,
        )
        loaded_ks, hp_fields = load_frame_kernel(str(self.kernels_dir), self.product_id)
        np.testing.assert_allclose(loaded_ks, self.kernel_solution)
        self.assertEqual(
            set(hp_fields),
            {
                "rkernel",
                "ko",
                "bgo",
                "ngauss",
                "deg_fixe",
                "sigma_gauss",
                "use_pca",
            },
        )
        self.assertEqual(hp_fields["rkernel"], int(self.hp_config.rkernel))
        self.assertEqual(hp_fields["ngauss"], int(self.hp_config.ngauss))

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_frame_kernel(str(self.kernels_dir), self.product_id)
        self.assertIn(f"{self.product_id}_kernel.npz", str(ctx.exception))


class TestProcessOneFrameKernelPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.diffs = root / "ws" / "hp_d"
        self.convolved = root / "ws" / "hp_c"
        self.data_root = root / "data"
        self.diffs.mkdir(parents=True)
        self.convolved.mkdir(parents=True)
        self.dirs = hotpants.HotpantsWorkspaceDirs(
            diffs=str(self.diffs),
            convolved=str(self.convolved),
        )
        self.shape = (16, 16)
        self.product_id = "tess2026039233236"
        self.crop_bounds = {"x_min": 0, "x_max": 16, "y_min": 0, "y_max": 16}
        self.kernel_solution = np.ones(42, dtype=np.float64)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_frame(self, *, write_kernel_solutions: bool):
        hp = HotpantsParams(
            write_convolved=False,
            write_bkg=False,
            write_stamps=False,
            write_kernel_solutions=write_kernel_solutions,
        )
        sci = np.ones(self.shape, dtype=np.float64)
        tmpl = np.ones(self.shape, dtype=np.float64) * 0.5
        with (
            patch.object(hotpants.wcs_grouping, "crop_ffi_header", return_value=fits.Header()),
            patch.object(hotpants, "_load_template_cropped", return_value=tmpl),
            patch.object(hotpants, "_resolve_linear_template_pad", return_value=0),
            patch.object(hotpants, "_load_ffi_cropped", return_value=(sci, np.ones(self.shape))),
            patch.object(hotpants, "write_diff_noise_mask_fits"),
            patch.object(
                hotpants,
                "kernel_sum_at_center",
                return_value=0.02,
            ),
            patch.object(hotpants, "run_hotpants_frame") as mock_run,
        ):
            mock_run.return_value = {
                "success": True,
                "diff": sci,
                "bkg": None,
                "convolved": None,
                "noise": None,
                "mask": None,
                "kernel_params_arrays": {
                    "kernel_solution": self.kernel_solution,
                },
            }
            return hotpants._process_one_frame(
                ffi_path="/fake/ffi.fits",
                product_id=self.product_id,
                group_id=0,
                hp=hp,
                template_path_map={0: "/fake/template.fits"},
                mask=np.zeros(self.shape, dtype=np.uint8),
                crop_bounds=self.crop_bounds,
                ref_stars_xy=np.array([[8.0, 8.0]]),
                dirs=self.dirs,
                round_id=1,
                # SCC-only storage (041e996): hotpants always resolves its
                # write path via data_root/sck (resolve_diff_write_path) --
                # there is no workspace-only fallback any more, so both are
                # required even though write_diff_noise_mask_fits is mocked
                # out below and never touches disk.
                sck=(20, 3, 3),
                data_root=str(self.data_root),
            )

    def test_default_does_not_write_kernels(self):
        self.assertFalse(HotpantsParams().write_kernel_solutions)
        self._run_frame(write_kernel_solutions=False)
        kernels_dir = hotpants.frame_kernels_dir(str(self.diffs))
        self.assertFalse(Path(kernels_dir).exists())

    def test_enabled_writes_kernel_npz(self):
        self._run_frame(write_kernel_solutions=True)
        kernels_dir = hotpants.frame_kernels_dir(str(self.diffs))
        npz_path = hotpants.frame_kernel_npz_path(kernels_dir, self.product_id)
        self.assertTrue(Path(npz_path).is_file())
        with np.load(npz_path, allow_pickle=False) as data:
            self.assertIn("kernel_solution", data)
            np.testing.assert_allclose(
                data["kernel_solution"], self.kernel_solution
            )


if __name__ == "__main__":
    unittest.main()
