"""Tests for additional / per-event forced photometry config."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration.config import (
    normalize_additional_forced_targets,
)
from syndiff_pipeline.difference_imaging.stages import photometry as ph
from syndiff_pipeline.difference_imaging.orchestration.execute import (
    _forced_photometry_lightcurve_plot_path,
)
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    freeze_target_diff_config,
    load_diff_site_policy,
)
from tests.site_fixtures import write_site_deployment
from syndiff_pipeline.common.orchestration.event_ws_symlinks import (
    ensure_event_templates_symlink,
)


def _target_2020ut() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=3,
        target_ra=210.219333,
        target_dec=81.846589,
        target_name="2020ut",
    )


def _target_other() -> Target:
    return Target(
        sector=23,
        camera=1,
        ccd=3,
        target_ra=185.015708,
        target_dec=5.343289,
        target_name="2020ftl",
    )


class TestNormalizeAdditionalForcedTargets(unittest.TestCase):
    def test_sky_mode(self):
        out = normalize_additional_forced_targets(
            [{"name": "bottom", "ra": 210.27, "dec": 81.87}]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["position_mode"], "sky")
        self.assertEqual(out[0]["name"], "bottom")
        self.assertAlmostEqual(out[0]["ra"], 210.27)

    def test_offset_mode(self):
        out = normalize_additional_forced_targets(
            [{"name": "offset_top", "dx": 0, "dy": -7}]
        )
        self.assertEqual(out[0]["position_mode"], "offset")
        self.assertEqual(out[0]["dy"], -7.0)

    def test_fixed_mode(self):
        out = normalize_additional_forced_targets(
            [{"name": "pix", "x": 512.0, "y": 600.0}]
        )
        self.assertEqual(out[0]["position_mode"], "fixed")
        self.assertEqual(out[0]["x"], 512.0)

    def test_duplicate_names_rejected(self):
        raw = [
            {"name": "a", "dx": 1, "dy": 0},
            {"name": "a", "dx": 2, "dy": 0},
        ]
        with self.assertRaises(ValueError):
            normalize_additional_forced_targets(raw)

    def test_multiple_modes_rejected(self):
        with self.assertRaises(ValueError):
            normalize_additional_forced_targets(
                [{"name": "bad", "ra": 1.0, "dec": 2.0, "dx": 0, "dy": 0}]
            )


class TestResolveForcedTargetXy(unittest.TestCase):
    def test_offset_from_primary(self):
        wcs = pd.DataFrame({"path": ["a.fits", "b.fits"]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        primary = np.array([[10.0, 20.0], [12.0, 22.0]], dtype=np.float64)
        spec = {"position_mode": "offset", "dx": 7.0, "dy": -7.0, "name": "off"}
        out = ph.resolve_forced_target_xy(spec, primary, wcs, crop_bounds)
        np.testing.assert_allclose(out, primary + np.array([7.0, -7.0]))

    def test_fixed_broadcast(self):
        wcs = pd.DataFrame({"path": ["a.fits", "b.fits", "c.fits"]})
        crop_bounds = {"x_min": 0.0, "y_min": 0.0, "shape": (100, 100)}
        primary = np.full((3, 2), np.nan)
        spec = {"position_mode": "fixed", "x": 512.0, "y": 600.0, "name": "pix"}
        out = ph.resolve_forced_target_xy(spec, primary, wcs, crop_bounds)
        expected = np.full((3, 2), [512.0, 600.0])
        np.testing.assert_allclose(out, expected)


def _primary_forced_photometry_spec(target_ra: float, target_dec: float) -> dict:
    """Mirror execute.py primary phot_specs entry for forced_photometry."""
    return {
        "position_mode": "sky",
        "ra": float(target_ra),
        "dec": float(target_dec),
    }


def _sky_ra_dec_from_spec(pt: dict) -> tuple[float, float]:
    """Mirror execute.py sky-mode ra/dec extraction in forced_photometry loop."""
    mode = pt.get("position_mode", "sky")
    if mode == "sky":
        return float(pt["ra"]), float(pt["dec"])
    return float("nan"), float("nan")


class TestForcedPhotometryLightcurvePlotPath(unittest.TestCase):
    def test_primary_lightcurve_png(self):
        pdir = "/event/debug_plots"
        lc = _forced_photometry_lightcurve_plot_path(
            pdir, "lc_prf_on_diffs", "prf", None
        )
        self.assertEqual(
            lc, "/event/debug_plots/lightcurve_lc_prf_on_diffs_prf.png"
        )

    def test_extra_gets_method_and_target_in_name(self):
        pdir = "/event/debug_plots"
        lc = _forced_photometry_lightcurve_plot_path(
            pdir, "lc_prf_on_diffs", "prf", "offset_top"
        )
        self.assertEqual(
            lc,
            "/event/debug_plots/lightcurve_lc_prf_on_diffs_prf_offset_top.png",
        )

    def test_lightcurve_csv_basename_primary_and_extra(self):
        self.assertEqual(ph.lightcurve_csv_basename("prf"), "lightcurve_prf.csv")
        self.assertEqual(
            ph.lightcurve_csv_basename("prf", "offset_top"),
            "lightcurve_prf_offset_top.csv",
        )


class TestPrimaryForcedPhotometrySpec(unittest.TestCase):
    def test_primary_spec_includes_ra_dec(self):
        t = _target_2020ut()
        pt = _primary_forced_photometry_spec(t.target_ra, t.target_dec)
        ra_log, dec_log = _sky_ra_dec_from_spec(pt)
        self.assertAlmostEqual(ra_log, t.target_ra)
        self.assertAlmostEqual(dec_log, t.target_dec)

    def test_legacy_primary_spec_without_ra_raises(self):
        pt = {"position_mode": "sky"}
        with self.assertRaises(KeyError):
            _sky_ra_dec_from_spec(pt)


class TestSitePolicyForcedTargets(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"
        self.site.mkdir()
        self.handoff = self.root / "handoff"
        self.data = self.root / "data"
        write_site_deployment(
            self.site,
            workspace_root=str(self.handoff),
            data_root=str(self.data),
        )
        template_leaf = self.data / "shifted_downsampled" / "sector0020_camera3_ccd3"
        template_leaf.mkdir(parents=True)
        (template_leaf / "syndiff_template_s0020_3_3.fits").write_bytes(b"")
        ensure_event_templates_symlink(
            self.handoff / "events" / _target_2020ut().label(), template_leaf
        )
        template_leaf23 = self.data / "shifted_downsampled" / "sector0023_camera1_ccd3"
        template_leaf23.mkdir(parents=True)
        (template_leaf23 / "syndiff_template_s0023_1_3.fits").write_bytes(b"")
        ensure_event_templates_symlink(
            self.handoff / "events" / _target_other().label(), template_leaf23
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_policy(self, extra: str) -> Path:
        path = self.site / "diff_config.yaml"
        path.write_text(
            "\n".join(
                [
                    "deployment_file: deployment.yaml",
                    "defaults:",
                    "  n_jobs: 4",
                    "additional_forced_targets:",
                    "  - name: offset_top",
                    "    dx: 0",
                    "    dy: -7",
                    "  - name: offset_bottom",
                    "    dx: 0",
                    "    dy: 7",
                    "  - name: offset_right",
                    "    dx: 7",
                    "    dy: 0",
                    "  - name: offset_left",
                    "    dx: -7",
                    "    dy: 0",
                    "pipeline:",
                    "  - kind: shared_mask",
                    extra,
                    "condor:",
                    "  request_cpus: 4",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_load_policy_parses_per_event_force_targets(self):
        path = self._write_policy(
            "\n".join(
                [
                    "per_event_force_targets:",
                    "  s0020_c3_k3_2020ut:",
                    "    - name: bottom",
                    "      ra: 210.2696410",
                    "      dec: 81.8743734",
                    "    - name: fixed_pixel_check",
                    "      x: 512.0",
                    "      y: 600.0",
                ]
            )
        )
        policy = load_diff_site_policy(path)
        self.assertEqual(len(policy.additional_forced_targets), 4)
        self.assertIn("s0020_c3_k3_2020ut", policy.per_event_force_targets)
        self.assertEqual(len(policy.per_event_force_targets["s0020_c3_k3_2020ut"]), 2)

    def test_freeze_2020ut_merges_global_and_per_event(self):
        path = self._write_policy(
            "\n".join(
                [
                    "per_event_force_targets:",
                    "  s0020_c3_k3_2020ut:",
                    "    - name: bottom",
                    "      ra: 210.2696410",
                    "      dec: 81.8743734",
                    "    - name: fixed_pixel_check",
                    "      x: 512.0",
                    "      y: 600.0",
                ]
            )
        )
        cfg = freeze_target_diff_config(path, _target_2020ut())
        self.assertEqual(len(cfg.additional_forced_targets), 6)
        names = [t["name"] for t in cfg.additional_forced_targets]
        self.assertIn("offset_top", names)
        self.assertIn("bottom", names)
        self.assertIn("fixed_pixel_check", names)

    def test_freeze_other_target_gets_global_only(self):
        path = self._write_policy(
            "\n".join(
                [
                    "per_event_force_targets:",
                    "  s0020_c3_k3_2020ut:",
                    "    - name: bottom",
                    "      ra: 210.2696410",
                    "      dec: 81.8743734",
                ]
            )
        )
        cfg = freeze_target_diff_config(path, _target_other())
        self.assertEqual(len(cfg.additional_forced_targets), 4)
        self.assertEqual(
            [t["position_mode"] for t in cfg.additional_forced_targets],
            ["offset", "offset", "offset", "offset"],
        )


if __name__ == "__main__":
    unittest.main()
