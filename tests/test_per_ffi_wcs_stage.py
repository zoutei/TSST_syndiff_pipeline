"""Tests for per_ffi_wcs stage wiring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.diff_verify import (
    _scc_final_stage_complete,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    parse_per_ffi_wcs,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline
from syndiff_pipeline.difference_imaging.stages.per_ffi_wcs import (
    COEFFS_CSV,
    list_frames_for_lane,
)


class TestPerFfiWcsStage(unittest.TestCase):
    def test_validate_pipeline_accepts_per_ffi_wcs(self):
        cfg = SynDiffConfig(
            output_dir="/tmp/event",
            pipeline=[
                {"external_workspaces": ["centroids_r1", "hp_d"]},
                {
                    "kind": "per_ffi_wcs",
                    "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
                    "output": "wcs",
                    "sip_degree": 5,
                },
            ],
            sector=20,
            camera=3,
            ccd=3,
        )
        validate_pipeline(cfg)

    def test_parse_per_ffi_wcs_defaults(self):
        stage = {
            "kind": "per_ffi_wcs",
            "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
            "output": "wcs",
        }
        params = parse_per_ffi_wcs(stage, 0)
        self.assertEqual(params.sip_degree, 5)
        self.assertEqual(params.min_stars, 50)

    def test_list_frames_for_lane_from_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp)
            centroids_dir = lane / "centroids_r1"
            hp_d_dir = lane / "hp_d"
            centroids_dir.mkdir()
            hp_d_dir.mkdir()

            stem = "tess2020050192921-s0020-3-3"
            phot_name = f"{stem}_photresults.ecsv"
            (centroids_dir / phot_name).write_text(
                "# %ECSV 1.0\n"
                "# ---\n"
                "# schema: astropy-2.0\n"
                "x_init y_init\n"
                "1.0 2.0\n",
                encoding="utf-8",
            )
            index = {stem: phot_name}
            (centroids_dir / "centroids_index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            hdr = fits.Header()
            hdr["XMIN"] = 100
            hdr["YMIN"] = 200
            hdr["XMAX"] = 300
            hdr["YMAX"] = 400
            hdr["TSTART"] = 2457000.0
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data=np.zeros((200, 200)), header=hdr),
                ]
            ).writeto(hp_d_dir / f"{stem}_hp_d.fits", overwrite=True)

            frames = list_frames_for_lane(
                lane, centroids_label="centroids_r1", hp_d_label="hp_d"
            )
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].stem, stem)

    def test_scc_final_stage_complete_per_ffi_wcs(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp)
            wcs_dir = lane / "wcs"
            wcs_dir.mkdir()
            (wcs_dir / COEFFS_CSV).write_text("stem,btjd\n", encoding="utf-8")
            cfg = SynDiffConfig(
                output_dir=str(lane),
                pipeline=[
                    {
                        "kind": "per_ffi_wcs",
                        "inputs": {"centroids": "centroids_r1", "diffs": "hp_d"},
                        "output": "wcs",
                    }
                ],
                sector=20,
                camera=3,
                ccd=3,
                data_root=str(lane),
            )
            self.assertTrue(_scc_final_stage_complete(cfg, lane))


if __name__ == "__main__":
    unittest.main()
