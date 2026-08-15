"""Tests for compact star×frame WCS audit matrices."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from syndiff_pipeline.difference_imaging.wcs.audit_matrix import (
    AUDIT_NPZ,
    STATUS_CLIPPED,
    STATUS_MISSING,
    STATUS_USED,
    StarAuditMatrixWriter,
    build_star_index,
    load_stars_fit_audit,
    to_long_dataframe,
)


class TestAuditMatrix(unittest.TestCase):
    def _gaia_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_id": [101, 102, 103],
                "ra": [240.1, 240.2, 240.3],
                "dec": [54.1, 54.2, 54.3],
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
            }
        )

    def test_build_star_index_from_phot_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            phot_path = Path(tmp) / "frame_photresults.ecsv"
            Table.from_pandas(
                pd.DataFrame({"x_init": [1.0, 2.0], "y_init": [4.0, 5.0]})
            ).write(phot_path, format="ascii.ecsv", overwrite=True)

            index = build_star_index(self._gaia_df(), phot_paths=[phot_path])
            self.assertEqual(len(index.source_id), 2)
            self.assertEqual(index.row_lookup[101], 0)
            self.assertEqual(index.row_lookup[102], 1)

    def test_writer_round_trip_and_status_encoding(self):
        gaia = self._gaia_df()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            phot_path = out / "frame_photresults.ecsv"
            Table.from_pandas(
                pd.DataFrame({"x_init": [1.0, 2.0, 3.0], "y_init": [4.0, 5.0, 6.0]})
            ).write(phot_path, format="ascii.ecsv", overwrite=True)

            index = build_star_index(gaia, phot_paths=[phot_path])
            writer = StarAuditMatrixWriter(
                index,
                stems=["stem_a", "stem_b"],
                btjd=[2458000.0, 2458001.0],
                out_dir=out,
            )
            writer.write_frame(
                0,
                np.array([101, 102], dtype=np.int64),
                np.array([0.1, 0.2], dtype=np.float32),
                np.array([-0.1, -0.2], dtype=np.float32),
                np.array([True, False]),
            )
            writer.write_frame(
                1,
                np.array([103], dtype=np.int64),
                np.array([0.3], dtype=np.float32),
                np.array([0.4], dtype=np.float32),
                np.array([True]),
            )
            npz_path = writer.finalize()

            self.assertEqual(npz_path.name, AUDIT_NPZ)
            audit = load_stars_fit_audit(npz_path)
            self.assertEqual(audit["du"].shape, (3, 2))
            self.assertEqual(audit["status"][0, 0], STATUS_USED)
            self.assertEqual(audit["status"][1, 0], STATUS_CLIPPED)
            self.assertEqual(audit["status"][2, 1], STATUS_USED)
            self.assertTrue(np.isnan(audit["du"][2, 0]))
            self.assertEqual(audit["status"][2, 0], STATUS_MISSING)

            long_df = to_long_dataframe(audit)
            self.assertEqual(len(long_df), 3)
            self.assertIn("hypot_resid", long_df.columns)
            self.assertEqual(set(long_df["stem"]), {"stem_a", "stem_b"})

    def test_write_audit_frame_helper(self):
        gaia = self._gaia_df()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            phot_path = out / "frame_photresults.ecsv"
            Table.from_pandas(
                pd.DataFrame({"x_init": [1.0], "y_init": [4.0]})
            ).write(phot_path, format="ascii.ecsv", overwrite=True)
            index = build_star_index(gaia, phot_paths=[phot_path])
            writer = StarAuditMatrixWriter(
                index,
                stems=["stem_a"],
                btjd=[2458000.0],
                out_dir=out,
            )
            writer.write_audit_frame(
                {
                    "frame_idx": 0,
                    "source_id": np.array([101], dtype=np.int64),
                    "du": np.array([0.5], dtype=np.float32),
                    "dv": np.array([-0.5], dtype=np.float32),
                    "keep_mask": np.array([True]),
                }
            )
            audit = load_stars_fit_audit(writer.finalize())
            self.assertAlmostEqual(float(audit["du"][0, 0]), 0.5)
            self.assertEqual(int(audit["status"][0, 0]), STATUS_USED)


class TestPerFfiWcsAuditIntegration(unittest.TestCase):
    def test_run_writes_npz_audit(self):
        from syndiff_pipeline.difference_imaging.stages.per_ffi_wcs import (
            run_per_ffi_wcs_all_frames,
        )
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            parse_per_ffi_wcs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / "lane"
            centroids_dir = lane / "centroids_r1"
            hp_d_dir = lane / "hp_d"
            wcs_dir = lane / "wcs"
            centroids_dir.mkdir(parents=True)
            hp_d_dir.mkdir(parents=True)

            stems = ["tess_a", "tess_b"]
            gaia_rows = []
            for i, stem in enumerate(stems):
                x_init = float(i + 1)
                y_init = float(i + 10)
                gaia_rows.append(
                    {
                        "source_id": 1000 + i,
                        "ra": 240.0 + i * 0.01,
                        "dec": 54.0 + i * 0.01,
                        "x": x_init,
                        "y": y_init,
                    }
                )
                phot = Table()
                phot["id"] = [1]
                phot["group_id"] = [1]
                phot["group_size"] = [1]
                phot["local_bkg"] = [0.0]
                phot["x_init"] = [x_init]
                phot["y_init"] = [y_init]
                phot["flux_init"] = [1000.0]
                phot["x_fit"] = [x_init + 0.01]
                phot["y_fit"] = [y_init + 0.01]
                phot["flux_fit"] = [1000.0]
                phot["x_err"] = [0.01]
                phot["y_err"] = [0.01]
                phot["flux_err"] = [1.0]
                phot["n_pixels_fit"] = [25]
                phot["qfit"] = [0.05]
                phot["cfit"] = [0.0]
                phot["reduced_chi2"] = [1.0]
                phot["flags"] = [0]
                phot.write(
                    centroids_dir / f"{stem}_photresults.ecsv",
                    format="ascii.ecsv",
                    overwrite=True,
                )

                hdr = fits.Header()
                hdr["XMIN"] = 0
                hdr["YMIN"] = 0
                hdr["XMAX"] = 100
                hdr["YMAX"] = 100
                hdr["TSTART"] = 2458000.0 + i
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(data=np.zeros((100, 100)), header=hdr),
                    ]
                ).writeto(hp_d_dir / f"{stem}_hp_d.fits", overwrite=True)

            index = {stem: f"{stem}_photresults.ecsv" for stem in stems}
            (centroids_dir / "centroids_index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            gaia_df = pd.DataFrame(gaia_rows)
            params = parse_per_ffi_wcs(
                {
                    "kind": "per_ffi_wcs",
                    "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
                    "output": "wcs",
                    "min_stars": 1,
                    "sip_degree": 2,
                },
                0,
            )

            class _Cfg:
                n_jobs = 1

            run_per_ffi_wcs_all_frames(
                str(lane),
                gaia_df,
                _Cfg(),
                params,
                str(wcs_dir),
                centroids_label="centroids_r1",
                hp_d_label="hp_d",
                force_rerun=True,
                sector=22,
                camera=3,
                ccd=3,
            )

            audit_path = wcs_dir / AUDIT_NPZ
            self.assertTrue(audit_path.is_file())
            audit = load_stars_fit_audit(audit_path)
            self.assertEqual(audit["du"].shape[1], 2)
            self.assertGreater(np.count_nonzero(audit["status"]), 0)


if __name__ == "__main__":
    unittest.main()
