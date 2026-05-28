"""Tests for TESSVectors-aware reference FFI selection."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Repo root (parent of ``tests/``) on path for ``import wcs_grouping``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wcs_grouping import (
    attach_tessvector_earth_moon_angles,
    choose_reference_ffi_path,
    plot_wcs_drift_and_template_assignment,
)


class TestChooseReferenceFfiPath(unittest.TestCase):
    def _base_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "path": ["/ffi/a.fits", "/ffi/b.fits", "/ffi/c.fits"],
                "wcs_ok": [True, True, True],
                "btjd": [100.0, 101.0, 102.0],
                "delta_x": [0.0, 0.2, 0.4],
                "delta_y": [0.0, 0.0, 0.0],
                "delta_x_raw": [0.0, 0.2, 0.4],
                "delta_y_raw": [0.0, 0.0, 0.0],
                "group_id": [0, 0, 0],
            }
        )

    def test_prefers_row_meeting_residual_and_angles_closest_to_median_smoothed(self):
        """Middle row has large raw–smooth residual; pick among outer rows nearest median."""
        df = self._base_table()
        # Median smoothed dx = 0.2, dy = 0.0. Row b: raw dx differs from smooth → high residual.
        df.loc[1, "delta_x_raw"] = 1.0
        df.loc[1, "delta_y_raw"] = 0.0
        df["earth_deg"] = [50.0, 50.0, 50.0]
        df["moon_deg"] = [30.0, 30.0, 30.0]
        chosen = choose_reference_ffi_path(
            df,
            earth_deg_min=45.0,
            moon_deg_min=25.0,
            max_smoothed_residual=0.05,
        )
        # Rows 0 and 2 pass residual; distances to (0.2, 0): 0.04 and 0.04 — first wins.
        self.assertEqual(chosen, "/ffi/a.fits")

    def test_angle_fallback_still_respects_median_smoothed(self):
        """When no row passes Earth/Moon cuts, fallback uses all usable rows."""
        df = self._base_table()
        df["earth_deg"] = [10.0, 50.0, 10.0]
        df["moon_deg"] = [30.0, 30.0, 30.0]
        chosen = choose_reference_ffi_path(
            df,
            earth_deg_min=45.0,
            moon_deg_min=25.0,
            max_smoothed_residual=0.05,
        )
        # Only row 1 passes angles; median dx=0.2 → row b is sole angle-qualified candidate.
        self.assertEqual(chosen, "/ffi/b.fits")


class TestAttachTessvectorAngles(unittest.TestCase):
    def test_interpolation_from_local_csv(self):
        csv = """# comment
MidTime,Earth_Camera_Angle,Moon_Camera_Angle
99.0,40.0,20.0
101.0,50.0,30.0
103.0,60.0,40.0
"""
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, "TessVectors_S020_C3_FFI.csv")
            with open(fname, "w") as fh:
                fh.write(csv)
            wcs_table = pd.DataFrame(
                {
                    "path": ["/x.fits"],
                    "wcs_ok": [True],
                    "btjd": [101.0],
                    "delta_x": [0.0],
                    "delta_y": [0.0],
                }
            )
            out = attach_tessvector_earth_moon_angles(
                wcs_table, sector=20, camera=3, tessvectors_data_path=tmp
            )
            self.assertAlmostEqual(float(out["earth_deg"].iloc[0]), 50.0, places=5)
            self.assertAlmostEqual(float(out["moon_deg"].iloc[0]), 30.0, places=5)


class TestDriftPlot(unittest.TestCase):
    def test_plot_writes_with_four_panels(self):
        df = pd.DataFrame(
            {
                "path": ["/a.fits", "/b.fits"],
                "wcs_ok": [True, True],
                "btjd": [100.0, 101.0],
                "delta_x": [0.0, 0.1],
                "delta_y": [0.0, 0.0],
                "delta_x_raw": [0.0, 0.1],
                "delta_y_raw": [0.0, 0.0],
                "group_id": [0, 0],
                "earth_deg": [50.0, 48.0],
                "moon_deg": [30.0, 28.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            outp = os.path.join(tmp, "drift.png")
            r = plot_wcs_drift_and_template_assignment(
                df,
                outp,
                ref_ffi_path="/b.fits",
                ref_earth_deg_min=45.0,
                ref_moon_deg_min=25.0,
            )
            self.assertEqual(r, outp)
            self.assertTrue(os.path.isfile(outp))
            self.assertGreater(os.path.getsize(outp), 1000)


if __name__ == "__main__":
    unittest.main()
