"""Unit tests for PRF photometry wiring in the star runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.star import cli, runner


class TestBuildPrfMethod(unittest.TestCase):
    def test_build_prf_method_constructs_tess_prf_with_full_ffi_xy(self):
        ctx = SimpleNamespace(camera=3, ccd=2, sector=20)
        sentinel_prf = object()
        fake_tess_prf_cls = mock.Mock(return_value=sentinel_prf)
        fake_prf_module = SimpleNamespace(TESS_PRF=fake_tess_prf_cls)

        with mock.patch.dict(sys.modules, {"PRF": fake_prf_module}):
            with mock.patch(
                "syndiff_pipeline.difference_imaging.stages.photometry."
                "resolve_tess_prf_localdatadir",
                return_value="/fake/localdatadir/",
            ):
                method = runner._build_prf_method(ctx, x_ref=123.4, y_ref=567.8)

        self.assertEqual(method["name"], "prf")
        self.assertEqual(method["type"], "prf")
        self.assertIs(method["epsf_model"], sentinel_prf)
        self.assertEqual(method["psf_size"], 11)
        self.assertEqual(method["phot_bkg_poly_order"], 3)

        fake_tess_prf_cls.assert_called_once_with(
            3, 2, 20, 123.4, 567.8, localdatadir="/fake/localdatadir/"
        )

    def test_build_prf_method_raises_helpful_error_without_prf_package(self):
        ctx = SimpleNamespace(camera=3, ccd=2, sector=20)
        with mock.patch.dict(sys.modules, {"PRF": None}):
            with self.assertRaises(ImportError):
                runner._build_prf_method(ctx, x_ref=1.0, y_ref=1.0)

    def test_run_parser_accepts_submit_and_run(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "--site",
                "config",
                "--target-name",
                "20/3/2",
            ]
        )
        self.assertEqual(args.command, "run")
        self.assertEqual(args.target_name, "20/3/2")

        submit_args = parser.parse_args(
            [
                "submit",
                "--site",
                "config",
                "--run-id",
                "star_test",
            ]
        )
        self.assertEqual(submit_args.command, "submit")
        self.assertEqual(submit_args.run_id, "star_test")


if __name__ == "__main__":
    unittest.main()
