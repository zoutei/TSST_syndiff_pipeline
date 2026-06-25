"""Tests for TESS FFI download helpers (gzip-aware discovery)."""
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
    _stream_url_to_gzip_fits,
    compress_spoc_ffi_to_gzip,
    list_local_ffis,
    local_ffi_manifest_basenames,
    resolve_local_ffi_path,
    spoc_ffi_basename_from_local,
)


class TestListLocalFfis(unittest.TestCase):
    def test_prefers_fits_gz_over_fits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leaf = Path(tmpdir)
            stem = "tess2020019142923-s0022-3-3-0165-s_ffic"
            (leaf / f"{stem}.fits").write_bytes(b"raw")
            gz_path = leaf / f"{stem}.fits.gz"
            gz_path.write_bytes(b"gz")

            paths = list_local_ffis(str(leaf), sector=22, camera=3, ccd=3)
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).name, gz_path.name)

    def test_manifest_basenames_map_gz_to_fits(self):
        paths = [
            "/data/tess_ffi/s0022/cam3_ccd3/tess2020019142923-s0022-3-3-0165-s_ffic.fits.gz",
        ]
        basenames = local_ffi_manifest_basenames(paths)
        self.assertEqual(
            basenames,
            {"tess2020019142923-s0022-3-3-0165-s_ffic.fits"},
        )


class TestResolveLocalFfiPath(unittest.TestCase):
    def test_prefers_gzip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            plain = os.path.join(tmpdir, stem)
            gz = plain + ".gz"
            Path(plain).write_bytes(b"plain")
            Path(gz).write_bytes(b"gz")
            resolved = resolve_local_ffi_path(tmpdir, stem)
            self.assertEqual(resolved, gz)

    def test_falls_back_to_plain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stem = "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            plain = os.path.join(tmpdir, stem)
            Path(plain).write_bytes(b"plain")
            self.assertEqual(resolve_local_ffi_path(tmpdir, stem), plain)


class TestCompressSpocFfi(unittest.TestCase):
    def test_round_trip_removes_plain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plain = os.path.join(
                tmpdir, "tess2020019142923-s0022-3-3-0165-s_ffic.fits"
            )
            data = np.ones((4, 4), dtype=np.float32)
            fits.writeto(plain, data, overwrite=True)
            gz = compress_spoc_ffi_to_gzip(plain)
            self.assertFalse(os.path.isfile(plain))
            self.assertTrue(gz.endswith(".fits.gz"))
            with fits.open(gz) as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)


class TestStreamUrlToGzipFits(unittest.TestCase):
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

    def test_uncompressed_fits_streams_to_gzip_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plain = io.BytesIO()
            data = np.ones((4, 4), dtype=np.float32)
            fits.writeto(plain, data, overwrite=True)
            plain.seek(0)
            gz_dest = os.path.join(
                tmpdir, "tess2020019142923-s0022-3-3-0165-s_ffic.fits.gz"
            )
            plain_path = gz_dest[:-3]

            with self._mock_urlopen_chunks([plain.read()]):
                _stream_url_to_gzip_fits("https://example.invalid/file", gz_dest, 30.0)

            self.assertTrue(os.path.isfile(gz_dest))
            self.assertFalse(os.path.isfile(plain_path))
            with fits.open(gz_dest) as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)

    def test_pre_gzip_payload_not_double_compressed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = _GZIP_MAGIC + b"rest-of-gzip-bytes"
            gz_dest = os.path.join(
                tmpdir, "tess2020019142923-s0022-3-3-0165-s_ffic.fits.gz"
            )

            with self._mock_urlopen_chunks([payload]):
                _stream_url_to_gzip_fits("https://example.invalid/file", gz_dest, 30.0)

            self.assertEqual(Path(gz_dest).read_bytes(), payload)


class TestTesscurlDownloadGzip(unittest.TestCase):
    def test_download_compresses_to_gz(self):
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

            def fake_stream(url, gz_dest, timeout):
                buf = io.BytesIO()
                fits.writeto(buf, np.zeros((2, 2), dtype=np.float32), overwrite=True)
                buf.seek(0)
                with gzip.open(gz_dest, "wb") as out:
                    out.write(buf.read())

            with unittest.mock.patch(
                "syndiff_pipeline.common.download._fetch_bytes", side_effect=fake_fetch
            ), unittest.mock.patch(
                "syndiff_pipeline.common.download._stream_url_to_gzip_fits",
                side_effect=fake_stream,
            ):
                paths = _download_ffis_via_tesscurl(22, 3, 3, tmpdir, overwrite=False)

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith(".fits.gz"))
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

            def fake_stream(url, gz_dest, timeout):
                with lock:
                    active["n"] += 1
                    active["peak"] = max(active["peak"], active["n"])
                try:
                    time.sleep(0.05)
                    buf = io.BytesIO()
                    fits.writeto(buf, np.zeros((2, 2), dtype=np.float32), overwrite=True)
                    buf.seek(0)
                    with gzip.open(gz_dest, "wb") as out:
                        out.write(buf.read())
                finally:
                    with lock:
                        active["n"] -= 1

            with unittest.mock.patch(
                "syndiff_pipeline.common.download._fetch_bytes", side_effect=fake_fetch
            ), unittest.mock.patch(
                "syndiff_pipeline.common.download._stream_url_to_gzip_fits",
                side_effect=fake_stream,
            ):
                paths = _download_ffis_via_tesscurl(
                    22, 3, 3, tmpdir, overwrite=False, max_workers=4
                )

            self.assertEqual(len(paths), len(stems))
            self.assertGreaterEqual(active["peak"], 2)
            for path in paths:
                self.assertTrue(path.endswith(".fits.gz"))


if __name__ == "__main__":
    unittest.main()
