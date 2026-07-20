"""Tests for downsample regmap scratch staging and skycell checkpoint resume."""
from __future__ import annotations

import errno
import gzip
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from astropy.io import fits

from syndiff_pipeline.template_creation.orchestration.stage_params import (
    parse_stage_params,
)
from syndiff_pipeline.template_creation.processing.downsample import (
    checkpoint_dir_for_output,
    checkpoint_npz_path,
    combine_sparse_downsample_results,
    is_valid_skycell_checkpoint,
    load_skycell_checkpoint,
    process_skycell_batch_from_arrays,
    resolve_stage_regmaps_to_scratch,
    save_skycell_checkpoint,
    scan_completed_skycell_checkpoints,
    stage_regmap_files_to_scratch,
)


def _make_identity_reg_fits(path: Path, shape: tuple[int, int]) -> None:
    h, w = shape
    assignment = np.full((h, w), -1, dtype=np.int32)
    assignment[1 : h - 1, 1 : w - 1] = 0
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=assignment)
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def _make_identity_reg_fits_gz(path: Path, shape: tuple[int, int]) -> None:
    with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
        _make_identity_reg_fits(Path(tmp.name), shape)
        with open(tmp.name, "rb") as f_in, gzip.open(path, "wb") as f_out:
            f_out.write(f_in.read())


def _skycell_case(tmp: Path, name: str, pixel_value: int, shape: tuple[int, int] = (4, 4)) -> dict:
    ps1_data = np.full(shape, float(pixel_value), dtype=np.float32)
    ps1_mask = np.zeros(shape, dtype=np.uint32)
    reg_path = tmp / f"{name}.fits"
    _make_identity_reg_fits(reg_path, shape)
    return {
        "skycell_name": name,
        "ps1_data": ps1_data,
        "ps1_mask": ps1_mask,
        "reg_path": reg_path,
        "inner_sum": float(ps1_data[1 : shape[0] - 1, 1 : shape[1] - 1].sum()),
    }


class TestRegmapScratchStaging(unittest.TestCase):
    def test_stage_copies_fits_gz_as_is(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shape = (4, 4)
            src = tmp / "tess_s20_3_3_skycell.1.2.fits.gz"
            _make_identity_reg_fits_gz(src, shape)
            scratch = tmp / "scratch"
            src_size = src.stat().st_size

            local_paths, scratch_dir, n_staged, _elapsed = stage_regmap_files_to_scratch(
                [str(src)],
                sector=20,
                camera=3,
                ccd=3,
                oversampling_factor=1,
                scratch_base=scratch,
            )

            self.assertEqual(n_staged, 1)
            self.assertEqual(len(local_paths), 1)
            dest = Path(local_paths[0])
            self.assertTrue(dest.is_file())
            self.assertTrue(str(dest).endswith(".fits.gz"))
            self.assertEqual(dest.name, src.name)
            self.assertAlmostEqual(dest.stat().st_size, src_size, delta=1)
            self.assertIn("syndiff_downsample_regmaps_0020_3_3", str(scratch_dir))

            with fits.open(dest) as hdul:
                self.assertEqual(hdul[1].data.shape, shape)

            # Second call reuses existing scratch copy.
            local_paths2, _, n_staged2, _ = stage_regmap_files_to_scratch(
                [str(src)],
                sector=20,
                camera=3,
                ccd=3,
                oversampling_factor=1,
                scratch_base=scratch,
            )
            self.assertEqual(n_staged2, 0)
            self.assertEqual(local_paths2, local_paths)

    def test_stage_enospc_raises_errno_28(self):
        """Helper surfaces ENOSPC so callers can fall back to NFS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "reg.fits.gz"
            _make_identity_reg_fits_gz(src, (3, 3))
            scratch = tmp / "scratch"

            real_copy2 = shutil.copy2

            def _boom(src_path, dst_path, *args, **kwargs):
                raise OSError(errno.ENOSPC, "No space left on device")

            with mock.patch(
                "syndiff_pipeline.template_creation.processing.downsample.shutil.copy2",
                side_effect=_boom,
            ):
                with self.assertRaises(OSError) as ctx:
                    stage_regmap_files_to_scratch(
                        [str(src)],
                        sector=20,
                        camera=3,
                        ccd=3,
                        oversampling_factor=4,
                        scratch_base=scratch,
                    )
            self.assertEqual(ctx.exception.errno, errno.ENOSPC)
            # Avoid unused-var lint for patched symbol in some runners
            self.assertTrue(callable(real_copy2))

    def test_auto_detect_condor_scratch(self):
        self.assertFalse(resolve_stage_regmaps_to_scratch(None))
        self.assertFalse(resolve_stage_regmaps_to_scratch(False))
        self.assertTrue(resolve_stage_regmaps_to_scratch(True))
        with mock.patch.dict(os.environ, {"_CONDOR_SCRATCH_DIR": "/var/condor/scratch"}):
            self.assertTrue(resolve_stage_regmaps_to_scratch(None))

    def test_staging_summary_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "reg.fits.gz"
            _make_identity_reg_fits_gz(src, (3, 3))
            local_paths, scratch_dir, n_staged, elapsed = stage_regmap_files_to_scratch(
                [str(src)],
                sector=1,
                camera=2,
                ccd=3,
                oversampling_factor=4,
                scratch_base=tmp / "scratch",
            )
            line = (
                f"[downsample] staged {n_staged}/{len(local_paths)} regmaps to scratch "
                f"{scratch_dir} in {elapsed:.1f}s (zarr stays on NFS)"
            )
            self.assertIn("staged 1/1 regmaps", line)
            self.assertIn("_os4", str(scratch_dir))
            self.assertTrue(str(local_paths[0]).endswith(".fits.gz"))


class TestSkycellCheckpointResume(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "_partial"
            path = checkpoint_npz_path(ckpt_dir, "skycell.9.8")
            indices = np.array([0, 5], dtype=np.int64)
            sums = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            counts = np.array([[1, 1], [2, 2]], dtype=np.int32)
            mask_counts = np.array([[0, 1], [1, 0]], dtype=np.int32)
            save_skycell_checkpoint(path, indices, sums, counts, mask_counts)
            self.assertTrue(is_valid_skycell_checkpoint(path))
            loaded = load_skycell_checkpoint(path)
            np.testing.assert_array_equal(loaded[0], indices)
            np.testing.assert_array_equal(loaded[1], sums)
            self.assertIn("skycell.9.8", scan_completed_skycell_checkpoints(ckpt_dir))

    def test_resume_skips_preexisting_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            shape = (4, 4)
            sc_a = _skycell_case(tmp, "skycell.1.1", pixel_value=10)
            sc_b = _skycell_case(tmp, "skycell.2.2", pixel_value=20)

            offsets = np.array([[0.0, 0.0]], dtype=np.float64)
            shifts_dict = {
                (0.0, 0.0): pd.DataFrame(
                    {
                        "NAME": [sc_a["skycell_name"], sc_b["skycell_name"]],
                        "shift_x": [0, 0],
                        "shift_y": [0, 0],
                    }
                )
            }
            base_shape = shape
            roi_bounds = (0, 0, shape[1], shape[0])

            # First run: bin skycell A only and persist checkpoint (simulate mid-run kill).
            arrays_a = {sc_a["skycell_name"]: (sc_a["ps1_data"], sc_a["ps1_mask"])}
            result_a = process_skycell_batch_from_arrays(
                batch_idx=0,
                reg_files=[str(sc_a["reg_path"])],
                skycell_names=[sc_a["skycell_name"]],
                arrays=arrays_a,
                offsets=offsets,
                shifts_dict=shifts_dict,
                base_tess_shape=base_shape,
                roi_bounds=roi_bounds,
                checkpoint_dir=tmp / "out" / "_partial",
                checkpoint_skycells=True,
            )

            ckpt_dir = checkpoint_dir_for_output(tmp / "out")
            self.assertTrue(is_valid_skycell_checkpoint(checkpoint_npz_path(ckpt_dir, sc_a["skycell_name"])))

            # Second run: A loads checkpoint; B is computed fresh. Combine uses this batch only.
            arrays_both = {
                sc_a["skycell_name"]: (sc_a["ps1_data"], sc_a["ps1_mask"]),
                sc_b["skycell_name"]: (sc_b["ps1_data"], sc_b["ps1_mask"]),
            }
            buf = io.StringIO()
            with redirect_stdout(buf):
                result_both = process_skycell_batch_from_arrays(
                    batch_idx=0,
                    reg_files=[str(sc_a["reg_path"]), str(sc_b["reg_path"])],
                    skycell_names=[sc_a["skycell_name"], sc_b["skycell_name"]],
                    arrays=arrays_both,
                    offsets=offsets,
                    shifts_dict=shifts_dict,
                    base_tess_shape=base_shape,
                    roi_bounds=roi_bounds,
                    checkpoint_dir=ckpt_dir,
                    checkpoint_skycells=True,
                    log_level="DEBUG",
                )
            self.assertIn("loaded from checkpoint", buf.getvalue())

            combined = combine_sparse_downsample_results(
                [result_both],
                offsets,
                base_shape,
                roi_bounds,
            )
            flux = combined[0, 0]
            self.assertGreater(float(flux.sum()), 0.0)
            # Both skycells map inner pixels to TESS index 0 -> output [0, 0].
            self.assertAlmostEqual(float(flux[0, 0]), sc_a["inner_sum"] + sc_b["inner_sum"], places=3)

    def test_stage_params_wiring(self):
        params = parse_stage_params(
            {
                "downsample": {
                    "checkpoint_skycells": True,
                    "stage_regmaps_to_scratch": False,
                }
            }
        )
        self.assertTrue(params.downsample.checkpoint_skycells)
        self.assertFalse(params.downsample.stage_regmaps_to_scratch)


if __name__ == "__main__":
    unittest.main()
