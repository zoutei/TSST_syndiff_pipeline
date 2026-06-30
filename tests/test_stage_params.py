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
    parse_background,
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
        self.assertEqual(hp.hp_normalize, "t")

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

    def test_hotpants_write_flags_override(self):
        hp = parse_hotpants(
            {
                "kind": "hotpants",
                "inputs": {},
                "output": {"diffs": "a"},
                "write_convolved": False,
                "write_bkg": False,
                "write_stamps": False,
            },
            0,
        )
        self.assertFalse(hp.write_convolved)
        self.assertFalse(hp.write_bkg)
        self.assertFalse(hp.write_stamps)

    def test_background_steps_parse(self):
        bp = parse_background(
            {
                "kind": "background",
                "inputs": {"diffs": "d", "bkg": "h"},
                "output": "out",
                "steps": {
                    "spatial": {"enabled": True, "box_size": 32},
                    "temporal": {"enabled": False},
                    "strap": {"enabled": True, "qe_floor": 1.01},
                },
            },
            0,
        )
        self.assertTrue(bp.spatial.enabled)
        self.assertEqual(bp.spatial.box_size, 32)
        self.assertFalse(bp.temporal.enabled)
        self.assertTrue(bp.strap.enabled)
        self.assertAlmostEqual(bp.strap.qe_floor, 1.01)

    def test_forced_photometry_requires_methods(self):
        with self.assertRaises(ValueError):
            parse_forced_photometry(
                {"kind": "forced_photometry", "inputs": {"diffs": "x"}, "output": "y"},
                0,
            )

    def test_forced_photometry_rejects_legacy_psf_type(self):
        with self.assertRaises(ValueError) as ctx:
            parse_forced_photometry(
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "x", "epsf": "e"},
                    "output": "y",
                    "psf_type": "epsf",
                },
                0,
            )
        self.assertIn("methods", str(ctx.exception))

    def test_forced_photometry_multi_method(self):
        p = parse_forced_photometry(
            {
                "kind": "forced_photometry",
                "inputs": {"diffs": "x", "epsf": "e"},
                "output": "y",
                "methods": [
                    {
                        "name": "epsf",
                        "type": "psf",
                        "psf_type": "epsf",
                        "phot_snap": "ref",
                    },
                    {
                        "name": "ap3",
                        "type": "aperture",
                        "tar_ap": 3,
                    },
                ],
            },
            0,
        )
        self.assertEqual(len(p.methods), 2)
        self.assertEqual(p.methods[0].name, "epsf")
        self.assertEqual(p.methods[0].psf_type, "epsf")
        self.assertEqual(p.methods[0].phot_snap, "ref")
        self.assertEqual(p.methods[1].name, "ap3")
        self.assertEqual(p.methods[1].tar_ap, 3)

    def test_forced_photometry_duplicate_method_name_rejected(self):
        with self.assertRaises(ValueError):
            parse_forced_photometry(
                {
                    "kind": "forced_photometry",
                    "inputs": {"diffs": "x"},
                    "output": "y",
                    "methods": [
                        {"name": "prf", "type": "psf", "psf_type": "prf"},
                        {"name": "prf", "type": "aperture"},
                    ],
                },
                0,
            )


if __name__ == "__main__":
    unittest.main()
