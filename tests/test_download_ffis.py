"""Tests for TESS FFI download helpers (fpack-aware discovery)."""
from __future__ import annotations

import gzip
import io
import os
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.download import (
    _GZIP_MAGIC,
    _download_ffis_via_tesscurl,
    _stream_url_to_fpack_fits,
    compress_spoc_ffi_to_fpack,
    list_local_ffis,
    local_ffi_manifest_basenames,
    resolve_local_ffi_path,
    spoc_ffi_basename_from_local,
)


class TestListLocalFfis(unittest.TestCase):
    def test_prefers_fits_fz_over_gz_and_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leaf = Path(tmpdir)
            stem = "tess2020019142923-s0022-3-3-0165-s_ffic"
            (leaf / f"{stem}.fits").write_bytes(b"raw")
            (leaf / f"{stem}.fits.gz").write_bytes(b"gz")
            fz_path = leaf / f"{stem}.fits.fz"
            fz_path.write_bytes(b"fz")

            paths = list_local_ffis(str(leaf), sector=22, camera=3, ccd=3)
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).name, fz_path.name)

    def test_manifest_basenames_map_fz_to_fits(self):
        paths = [
            "/data/tess_ffi/s0022/cam3_ccd3/tess2020019142923-s0022-3-3-0165-s_ffic.fits.fz",
        ]
        basenames = local_ffi_manifest_basenames(paths)
        self.assertEqual(
            basenames,
            {"tess2020019142923-s0022-3-3-0165-s_ffic.fits"},
        )


class TestResolveLocalFfiPath(unittest.TestCase):
    def test_prefers_fpack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            plain = os.path.join(tmpdir, stem)
            gz = plain + ".gz"
            fz = plain + ".fz"
            Path(plain).write_bytes(b"plain")
            Path(gz).write_bytes(b"gz")
            Path(fz).write_bytes(b"fz")
            resolved = resolve_local_ffi_path(tmpdir, stem)
            self.assertEqual(resolved, fz)

    def test_falls_back_to_gz_then_plain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            plain = os.path.join(tmpdir, stem)
            Path(plain).write_bytes(b"plain")
            self.assertEqual(resolve_local_ffi_path(tmpdir, stem), plain)
            gz = plain + ".gz"
            Path(gz).write_bytes(b"gz")
            self.assertEqual(resolve_local_ffi_path(tmpdir, stem), gz)


class TestCompressSpocFfi(unittest.TestCase):
    def test_round_trip_removes_plain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plain = os.path.join(
                tmpdir, "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            )
            data = np.ones((4, 4), dtype=np.float32)
            fits.writeto(plain, data, overwrite=True)
            fz = compress_spoc_ffi_to_fpack(plain)
            self.assertFalse(os.path.isfile(plain))
            self.assertTrue(fz.endswith(".fits.fz"))
            from syndiff_pipeline.common.fits_io import open_fits

            with open_fits(fz) as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)


class TestStreamUrlToFpackFits(unittest.TestCase):
    def _mock_urlopen_chunks(self, chunks: list[bytes]):
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                if not chunks:
                    return b""
                if size < 0 or size >= len(chunks[0]):
                    return chunks.pop(0)
                head = chunks[0][:size]
                chunks[0] = chunks[0][size:]
                return head

        return unittest.mock.patch(
            "syndiff_pipeline.common.download.urlopen",
            return_value=FakeResp(),
        )

    def test_uncompressed_fits_streams_to_fpack_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plain = io.BytesIO()
            data = np.ones((4, 4), dtype=np.float32)
            fits.writeto(plain, data, overwrite=True)
            plain.seek(0)
            fz_dest = os.path.join(
                tmpdir, "tess2020019142923-s0022-3-3-0165-s_ffic.fits.fz"
            )
            plain_path = fz_dest[: -len(".fz")]

            with self._mock_urlopen_chunks([plain.read()]):
                _stream_url_to_fpack_fits(
                    "https://example.invalid/file", fz_dest, 30.0
                )

            self.assertTrue(os.path.isfile(fz_dest))
            self.assertFalse(os.path.isfile(plain_path))
            from syndiff_pipeline.common.fits_io import open_fits

            with open_fits(fz_dest) as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)

    def test_pre_gzip_payload_gunzipped_then_fpacked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.BytesIO()
            data = np.ones((4, 4), dtype=np.float32)
            fits.writeto(buf, data, overwrite=True)
            gz_bytes = gzip.compress(buf.getvalue())
            self.assertEqual(gz_bytes[:2], _GZIP_MAGIC)
            fz_dest = os.path.join(
                tmpdir, "tess2020019142923-s0022-3-3-0165-s_ffic.fits.fz"
            )

            with self._mock_urlopen_chunks([gz_bytes]):
                _stream_url_to_fpack_fits(
                    "https://example.invalid/file", fz_dest, 30.0
                )

            self.assertTrue(os.path.isfile(fz_dest))
            from syndiff_pipeline.common.fits_io import open_fits

            with open_fits(fz_dest) as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)


class TestTesscurlDownloadFpack(unittest.TestCase):
    def test_download_compresses_to_fz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bn = "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            script = (
                f"#!/bin/bash\n"
                f"curl -C - -o {bn} https://example.invalid/{bn}\n"
            )
            script_path = os.path.join(tmpdir, "tesscurl_sector_22_ffic.sh")
            Path(script_path).write_text(script, encoding="utf-8")

            def fake_fetch(url, timeout):
                if url.endswith("_ffic.sh"):
                    return script.encode()
                raise AssertionError(f"unexpected fetch {url}")

            def fake_stream(url, fz_dest, timeout):
                buf = io.BytesIO()
                fits.writeto(buf, np.zeros((2, 2), dtype=np.float32), overwrite=True)
                buf.seek(0)
                plain = fz_dest[: -len(".fz")]
                Path(plain).write_bytes(buf.read())
                compress_spoc_ffi_to_fpack(plain)

            with unittest.mock.patch(
                "syndiff_pipeline.common.download._fetch_bytes", side_effect=fake_fetch
            ), unittest.mock.patch(
                "syndiff_pipeline.common.download._stream_url_to_fpack_fits",
                side_effect=fake_stream,
            ):
                paths = _download_ffis_via_tesscurl(22, 3, 3, tmpdir, overwrite=False)

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith(".fits.fz"))
            self.assertFalse(os.path.isfile(os.path.join(tmpdir, bn)))
            self.assertEqual(
                spoc_ffi_basename_from_local(paths[0]),
                bn,
            )

    def test_parallel_downloads_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stems = [
                "tess2020019142923-s0022-3-3-0165-s_ffic.fits",
                "tess2020019142924-s0022-3-3-0166-s_ffic.fits",
                "tess2020019142925-s0022-3-3-0167-s_ffic.fits",
                "tess2020019142926-s0022-3-3-0168-s_ffic.fits",
            ]
            lines = [
                f"curl -C - -o {stem} https://example.invalid/{stem}\n"
                for stem in stems
            ]
            script = "#!/bin/bash\n" + "".join(lines)
            script_path = os.path.join(tmpdir, "tesscurl_sector_22_ffic.sh")
            Path(script_path).write_text(script, encoding="utf-8")
            active = {"n": 0, "peak": 0}
            lock = threading.Lock()

            def fake_fetch(url, timeout):
                if url.endswith("_ffic.sh"):
                    return script.encode()
                raise AssertionError(f"unexpected fetch {url}")

            def fake_stream(url, fz_dest, timeout):
                with lock:
                    active["n"] += 1
                    active["peak"] = max(active["peak"], active["n"])
                try:
                    time.sleep(0.05)
                    Path(fz_dest).write_bytes(b"fz")
                finally:
                    with lock:
                        active["n"] -= 1

            with unittest.mock.patch(
                "syndiff_pipeline.common.download._fetch_bytes", side_effect=fake_fetch
            ), unittest.mock.patch(
                "syndiff_pipeline.common.download._stream_url_to_fpack_fits",
                side_effect=fake_stream,
            ):
                paths = _download_ffis_via_tesscurl(
                    22, 3, 3, tmpdir, overwrite=False, max_workers=4
                )

            self.assertEqual(len(paths), len(stems))
            self.assertGreaterEqual(active["peak"], 2)
            for path in paths:
                self.assertTrue(path.endswith(".fits.fz"))


if __name__ == "__main__":
    unittest.main()
