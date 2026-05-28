"""Tests for :mod:`stage_params`."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_params import (
    parse_background_rough,
    parse_forced_photometry,
    parse_hotpants,
    validate_stage_keys,
    WCS_GROUPING_ALLOWED,
)


class TestStageParams(unittest.TestCase):
    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            validate_stage_keys(
                {"kind": "wcs_grouping", "foo": 1},
                0,
                "wcs_grouping",
                WCS_GROUPING_ALLOWED,
            )
        self.assertIn("unknown keys", str(ctx.exception))

    def test_hotpants_defaults_merge(self):
        hp = parse_hotpants(
            {"kind": "hotpants", "inputs": {}, "output": {"diffs": "a", "convolved": "b"}},
            0,
        )
        self.assertEqual(hp.sci_fwhm, 1.0)
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


if __name__ == "__main__":
    unittest.main()
