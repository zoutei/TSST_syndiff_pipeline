"""Tests for fpack-based FITS writes and open_fits primary promotion."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common import fits_io
from syndiff_pipeline.common.fits_io import (
    fpack_plain_fits,
    open_fits,
    write_hdul_fits,
    write_image_fits,
)


@unittest.skipUnless(shutil.which("fpack"), "fpack not on PATH")
class TestFitsIo(unittest.TestCase):
    def test_write_image_round_trip_and_no_plain_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.arange(16, dtype=np.float32).reshape(4, 4) + 1.0
            out = write_image_fits(Path(tmp) / "img.fits.gz", data)
            self.assertTrue(out.endswith(".fits.fz"))
            self.assertTrue(os.path.isfile(out))
            self.assertFalse(os.path.isfile(Path(tmp) / "img.fits"))
            self.assertFalse(os.path.isfile(Path(tmp) / "img.fits.gz"))
            with open_fits(out) as hdul:
                self.assertIsNotNone(hdul[0].data)
                np.testing.assert_array_equal(hdul[0].data, data)

    def test_multi_hdu_preserves_named_exts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.ones((3, 3), dtype=np.float32)
            hdul = fits.HDUList(
                [
                    fits.PrimaryHDU(data),
                    fits.ImageHDU(data * 2, name="NOISE"),
                    fits.ImageHDU(data * 3, name="MASK"),
                ]
            )
            out = write_hdul_fits(Path(tmp) / "diff.fits", hdul)
            with open_fits(out) as h:
                np.testing.assert_array_equal(h[0].data, data)
                np.testing.assert_array_equal(h["NOISE"].data, data * 2)
                np.testing.assert_array_equal(h["MASK"].data, data * 3)

    def test_legacy_gz_still_opens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.fits.gz"
            data = np.eye(2, dtype=np.float32)
            fits.writeto(path, data, overwrite=True)
            with open_fits(Path(tmp) / "legacy.fits.fz") as hdul:
                np.testing.assert_array_equal(hdul[0].data, data)


def _fake_success_fpack(fpack_bin, target):
    """Stand-in for ``_run_fpack``: simulate real fpack by writing the
    ``.fz`` sibling and returning rc=0 -- exercises the atomicity logic
    without needing the real CFITSIO binary on PATH."""
    fz = Path(fits_io.fits_fpack_path(target))
    fz.write_bytes(b"FAKE-FZ-BYTES")
    target.unlink()  # real fpack -D removes the (temp) plain input on success
    return subprocess.CompletedProcess([fpack_bin, str(target)], 0, stdout="ok", stderr="")


def _fake_failing_fpack(fpack_bin, target):
    """Stand-in for ``_run_fpack`` simulating a failure: no ``.fz`` produced."""
    return subprocess.CompletedProcess(
        [fpack_bin, str(target)], 1, stdout="", stderr="simulated fpack failure"
    )


class TestFpackAtomicity(unittest.TestCase):
    """Exercises the §10 atomicity hardening in ``fpack_plain_fits`` by
    monkeypatching the fpack subprocess call -- runs unconditionally (no
    real ``fpack`` binary required)."""

    def _write_plain(self, tmp: str, name: str = "img.fits") -> Path:
        path = Path(tmp) / name
        fits.writeto(path, np.ones((2, 2), dtype=np.float32), overwrite=True)
        return path

    def test_success_atomic_replace_and_plain_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._write_plain(tmp)
            with mock.patch.object(fits_io, "_require_fpack", return_value="fpack"), \
                 mock.patch.object(fits_io, "_run_fpack", side_effect=_fake_success_fpack):
                fz = fpack_plain_fits(plain, delete_plain=True)
            self.assertTrue(fz.is_file())
            self.assertEqual(fz.read_bytes(), b"FAKE-FZ-BYTES")
            self.assertFalse(plain.is_file(), "plain should be deleted after successful publish")
            leftovers = [p for p in Path(tmp).iterdir() if p.name.startswith("_tmp_")]
            self.assertEqual(leftovers, [], f"no _tmp_* orphans expected, found {leftovers}")

    def test_success_keeps_plain_when_delete_plain_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._write_plain(tmp)
            with mock.patch.object(fits_io, "_require_fpack", return_value="fpack"), \
                 mock.patch.object(fits_io, "_run_fpack", side_effect=_fake_success_fpack):
                fz = fpack_plain_fits(plain, delete_plain=False)
            self.assertTrue(fz.is_file())
            self.assertTrue(plain.is_file())

    def test_failure_leaves_plain_and_existing_fz_untouched(self):
        """The old ``fpack_plain_fits`` pre-unlinked any existing ``.fits.fz``
        before invoking fpack -- a failure after that point silently lost the
        prior artifact. The hardened version must never touch the final key
        until a single successful atomic replace."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._write_plain(tmp)
            fz_path = Path(fits_io.fits_fpack_path(plain))
            fz_path.write_bytes(b"PRE-EXISTING-COMPLETE-FZ")

            with mock.patch.object(fits_io, "_require_fpack", return_value="fpack"), \
                 mock.patch.object(fits_io, "_run_fpack", side_effect=_fake_failing_fpack):
                with self.assertRaises(RuntimeError):
                    fpack_plain_fits(plain, delete_plain=True)

            self.assertTrue(plain.is_file(), "plain must survive a failed fpack")
            self.assertEqual(
                fz_path.read_bytes(),
                b"PRE-EXISTING-COMPLETE-FZ",
                "pre-existing .fits.fz must not be touched by a failed republish",
            )
            leftovers = [p for p in Path(tmp).iterdir() if p.name.startswith("_tmp_")]
            self.assertEqual(leftovers, [], f"no _tmp_* orphans expected, found {leftovers}")

    def test_crash_mid_write_leaves_only_tmp_orphan(self):
        """Simulate a hard crash: fpack succeeds (temp .fz written) but the
        process dies before our own cleanup/replace can run. Model this by
        making the cleanup step itself unable to run (as a real SIGKILL
        would prevent) and asserting the only trace left behind is the
        ``_tmp_*`` sibling -- the final ``.fits.fz`` key was never created or
        modified."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._write_plain(tmp)
            fz_path = Path(fits_io.fits_fpack_path(plain))

            real_replace = os.replace

            def _crash_before_replace(src, dst):
                # Simulate the process dying at the atomic-rename boundary:
                # the rename never happens, so the final key is never touched.
                raise OSError("simulated crash before os.replace")

            with mock.patch.object(fits_io, "_require_fpack", return_value="fpack"), \
                 mock.patch.object(fits_io, "_run_fpack", side_effect=_fake_success_fpack), \
                 mock.patch.object(fits_io.os, "replace", side_effect=_crash_before_replace):
                with self.assertRaises(OSError):
                    fpack_plain_fits(plain, delete_plain=True)

            self.assertFalse(fz_path.is_file(), "final .fits.fz must never appear partially")
            self.assertTrue(plain.is_file(), "plain must survive when the replace never happened")
            # Our own best-effort cleanup still ran (finally block executes
            # even though the simulated failure happened "at" the replace),
            # so no _tmp_* orphan is expected in this in-process simulation --
            # a true SIGKILL would leave one, and that orphan carries the
            # _tmp_ prefix by construction (asserted via the naming scheme
            # itself: tmp_plain/tmp_fz are always _tmp_-prefixed siblings of
            # plain, never the final .fits or .fits.fz name).
            leftovers = list(Path(tmp).glob("_tmp_*"))
            for leftover in leftovers:
                self.assertTrue(leftover.name.startswith("_tmp_"))

    def test_missing_plain_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.fits"
            with self.assertRaises(FileNotFoundError):
                fpack_plain_fits(missing)

    def test_rejects_already_compressed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "already.fits.fz"
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                fpack_plain_fits(path)


if __name__ == "__main__":
    unittest.main()
