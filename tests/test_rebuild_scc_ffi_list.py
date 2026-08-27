"""Smoke tests for scripts/rebuild_scc_ffi_list.py."""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "rebuild_scc_ffi_list.py"


def _write_gz_ffi(gz_path: Path) -> None:
    plain = gz_path.with_suffix("")  # .fits from .fits.gz
    hdu0 = fits.PrimaryHDU()
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 8
    hdr["NAXIS2"] = 8
    hdr["CRVAL1"] = 100.0
    hdr["CRVAL2"] = 10.0
    hdr["CRPIX1"] = 4.0
    hdr["CRPIX2"] = 4.0
    hdr["CD1_1"] = -0.0001
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 0.0001
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["DATE-OBS"] = "2020-01-01T00:00:00"
    hdu1 = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32), header=hdr)
    fits.HDUList([hdu0, hdu1]).writeto(plain, overwrite=True)
    with open(plain, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    plain.unlink()


class TestRebuildSccFfiListScript(unittest.TestCase):
    def test_limit_smoke_skip_fpack(self):
        """--skip-fpack on a tiny .fits tree writes parquet + csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
            ffi_dir.mkdir(parents=True)
            plain = ffi_dir / "tess-s0020-1-1-0001_ffic.fits"
            hdu0 = fits.PrimaryHDU()
            hdr = fits.Header()
            hdr.update(
                {
                    "NAXIS": 2,
                    "NAXIS1": 8,
                    "NAXIS2": 8,
                    "CRVAL1": 100.0,
                    "CRVAL2": 10.0,
                    "CRPIX1": 4.0,
                    "CRPIX2": 4.0,
                    "CD1_1": -0.0001,
                    "CD2_2": 0.0001,
                    "CTYPE1": "RA---TAN",
                    "CTYPE2": "DEC--TAN",
                    "DATE-OBS": "2020-01-01T00:00:00",
                }
            )
            hdu1 = fits.ImageHDU(data=np.zeros((8, 8), dtype=np.float32), header=hdr)
            fits.HDUList([hdu0, hdu1]).writeto(plain, overwrite=True)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "20",
                    "1",
                    "1",
                    "--data-root",
                    str(data_root),
                    "--skip-fpack",
                ],
                capture_output=True,
                text=True,
                cwd=str(_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            pq = data_root / "s0020" / "c1" / "k1" / "ffi_list.parquet"
            csv = data_root / "s0020" / "c1" / "k1" / "ffi_list.csv"
            self.assertTrue(pq.is_file(), proc.stdout)
            self.assertTrue(csv.is_file(), proc.stdout)
            self.assertIn("ffi_list: 1 rows", proc.stdout)
            self.assertIn("ffi_list.csv", proc.stdout)

    @unittest.skipUnless(shutil.which("fpack"), "fpack required for gz→fz smoke")
    def test_limit_smoke_with_fpack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            ffi_dir = data_root / "s0020" / "c1" / "k1" / "ffi"
            ffi_dir.mkdir(parents=True)
            gz = ffi_dir / "tess-s0020-1-1-0001_ffic.fits.gz"
            _write_gz_ffi(gz)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPT),
                    "20",
                    "1",
                    "1",
                    "--data-root",
                    str(data_root),
                    "--limit",
                    "1",
                ],
                capture_output=True,
                text=True,
                cwd=str(_ROOT),
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            self.assertFalse(gz.is_file())
            fz = ffi_dir / "tess-s0020-1-1-0001_ffic.fits.fz"
            self.assertTrue(fz.is_file())
            self.assertTrue((data_root / "s0020" / "c1" / "k1" / "ffi_list.csv").is_file())


if __name__ == "__main__":
    unittest.main()
