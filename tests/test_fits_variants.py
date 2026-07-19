"""Tests for tri-format FITS path helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.fits_variants import (
    FITS_FPACK_EXT,
    FITS_GZIP_EXT,
    FITS_PLAIN_EXT,
    canonical_fits_path_key,
    fits_logical_path,
    resolve_fits_variant,
    resolve_stem_in_directory,
    strip_fits_storage_suffix,
    try_resolve_fits_variant,
)


class TestFitsVariants(unittest.TestCase):
    def test_strip_all_suffixes(self):
        self.assertEqual(strip_fits_storage_suffix("a.fits.fz"), "a")
        self.assertEqual(strip_fits_storage_suffix("a.fits.gz"), "a")
        self.assertEqual(strip_fits_storage_suffix("a.fits"), "a")
        self.assertEqual(strip_fits_storage_suffix("/x/a.fits.fz"), "a")

    def test_logical_and_canonical_keys(self):
        base = "/data/frame.fits"
        self.assertEqual(fits_logical_path("/data/frame.fits.fz"), base)
        self.assertEqual(fits_logical_path("/data/frame.fits.gz"), base)
        k_fz = canonical_fits_path_key("/data/frame.fits.fz")
        k_gz = canonical_fits_path_key("/data/frame.fits.gz")
        k_pl = canonical_fits_path_key("/data/frame.fits")
        self.assertEqual(k_fz, k_gz)
        self.assertEqual(k_fz, k_pl)

    def test_resolve_prefers_fz(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = "tess123_hp_d"
            plain = Path(tmp) / f"{stem}.fits"
            gz = Path(tmp) / f"{stem}.fits.gz"
            fz = Path(tmp) / f"{stem}.fits.fz"
            plain.write_bytes(b"p")
            gz.write_bytes(b"g")
            fz.write_bytes(b"f")
            self.assertEqual(resolve_stem_in_directory(tmp, stem), str(fz))
            self.assertEqual(
                resolve_fits_variant(plain),
                plain,
            )  # exact path wins when present
            plain.unlink()
            resolved = resolve_fits_variant(Path(tmp) / f"{stem}.fits")
            self.assertEqual(resolved, fz)

    def test_resolve_falls_back_gz_then_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = "frame"
            gz = Path(tmp) / f"{stem}{FITS_GZIP_EXT}"
            gz.write_bytes(b"g")
            self.assertEqual(
                str(resolve_fits_variant(Path(tmp) / f"{stem}{FITS_FPACK_EXT}")),
                str(gz),
            )
            gz.unlink()
            plain = Path(tmp) / f"{stem}{FITS_PLAIN_EXT}"
            plain.write_bytes(b"p")
            self.assertEqual(
                str(resolve_fits_variant(Path(tmp) / f"{stem}{FITS_GZIP_EXT}")),
                str(plain),
            )
            self.assertIsNone(try_resolve_fits_variant(Path(tmp) / "missing.fits"))


if __name__ == "__main__":
    unittest.main()
