"""Tests for :mod:`stage_params`."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    SHARED_MASK_ALLOWED,
    parse_background_rough,
    parse_forced_photometry,
    parse_hotpants,
    validate_stage_keys,
)


class TestStageParams(unittest.TestCase):
    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            validate_stage_keys(
                {"kind": "shared_mask", "foo": 1},
                0,
                "shared_mask",
                SHARED_MASK_ALLOWED,
            )
        self.assertIn("unknown keys", str(ctx.exception))

    def test_hotpants_defaults_merge(self):
        hp = parse_hotpants(
            {"kind": "hotpants", "inputs": {}, "output": {"diffs": "a", "convolved": "b"}},
            0,
        )
        self.assertEqual(hp.sci_fwhm, 1.88)
        self.assertEqual(hp.hp_ko, 2)

    def test_hotpants_override(self):
        hp = parse_hotpants(
            {
                "kind": "hotpants",
                "inputs": {},
                "output": {"diffs": "a", "convolved": "b"},
                "sci_fwhm": 2.5,
                "hp_ko": 3,
            },
            0,
        )
        self.assertEqual(hp.sci_fwhm, 2.5)
        self.assertEqual(hp.hp_ko, 3)

    def test_hotpants_sigma_gauss_override(self):
        hp = parse_hotpants(
            {
                "kind": "hotpants",
                "inputs": {},
                "output": {"diffs": "a", "convolved": "b"},
                "hp_sigma_gauss": [0.752, 1.88, 3.76],
                "hp_ngauss": 3,
            },
            0,
        )
        from syndiff_pipeline.difference_imaging.stages.hotpants import (
            _kernel_scale_pixels,
            _resolved_sigma_gauss,
            build_hotpants_config,
        )

        self.assertEqual(_resolved_sigma_gauss(hp), [0.752, 1.88, 3.76])
        self.assertAlmostEqual(_kernel_scale_pixels(hp), 1.88)
        cfg = build_hotpants_config(hp, "/tmp", "/tmp", "t", write_stamps=False)
        self.assertEqual(cfg.rkernel, 4)
        self.assertEqual(cfg.rss, 4)
        self.assertEqual(list(cfg.sigma_gauss), [0.752, 1.88, 3.76])

    def test_hotpants_write_flags_default_true(self):
        hp = parse_hotpants(
            {"kind": "hotpants", "inputs": {}, "output": {"diffs": "a"}},
            0,
        )
        self.assertTrue(hp.write_convolved)
        self.assertTrue(hp.write_bkg)
        self.assertTrue(hp.write_stamps)
        self.assertTrue(hp.write_kernel_params)

    def test_hotpants_write_flags_override(self):
        hp = parse_hotpants(
            {
                "kind": "hotpants",
                "inputs": {},
                "output": {"diffs": "a"},
                "write_convolved": False,
                "write_bkg": False,
                "write_stamps": False,
                "write_kernel_params": False,
            },
            0,
        )
        self.assertFalse(hp.write_convolved)
        self.assertFalse(hp.write_bkg)
        self.assertFalse(hp.write_stamps)
        self.assertFalse(hp.write_kernel_params)

    def test_background_rough_bkg_source_hunt(self):
        sp = parse_background_rough(
            {
                "kind": "background_rough",
                "inputs": {"diffs": "d", "bkg": "h"},
                "output": "out",
                "bkg_source_hunt": False,
            },
            0,
        )
        self.assertFalse(sp.bkg_source_hunt)

    def test_forced_photometry_requires_psf_type(self):
        with self.assertRaises(ValueError):
            parse_forced_photometry(
                {"kind": "forced_photometry", "inputs": {"diffs": "x"}, "output": "y"},
                0,
            )

    def test_forced_photometry_epsf(self):
        p = parse_forced_photometry(
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "x", "epsf": "e"},
                "output": "y",
                "psf_type": "epsf",
                "phot_snap": "ref",
            },
            0,
        )
        self.assertEqual(p.psf_type, "epsf")
        self.assertEqual(p.phot_snap, "ref")
        self.assertEqual(p.phot_debug_stamp_size, 25)

    def test_forced_photometry_debug_stamp_size(self):
        p = parse_forced_photometry(
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "x", "epsf": "e"},
                "output": "y",
                "psf_type": "prf",
                "phot_debug_stamp_size": 31,
            },
            0,
        )
        self.assertEqual(p.phot_debug_stamp_size, 31)


if __name__ == "__main__":
    unittest.main()
