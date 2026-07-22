"""ePSF gridded emit: real diff_image fingerprints, fail-open without loc:."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    freeze_target_diff_config,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    EpsfParams,
    HotpantsParams,
)
from syndiff_pipeline.difference_imaging.stages import gridded_epsf
from syndiff_pipeline.difference_imaging.support.manifest import (
    manifest_path_from_output_dir,
)
from tests.site_fixtures import write_site_deployment


def _write_hotpants_epsf_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "paths:",
                "  template_base: shifted_downsampled",
                "pipeline:",
                "  - kind: hotpants",
                "    output:",
                "      diffs: hp_d",
                "      convolved: hp_c",
                "  - kind: epsf",
                "    inputs:",
                "      diffs: hp_d",
                "    output: epsf_r1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestGriddedEpsfProvenanceEmit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
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
        _write_hotpants_epsf_policy(self.site / "diff_config.yaml")

        self.target = Target(
            sector=20,
            camera=3,
            ccd=3,
            target_ra=228.479042,
            target_dec=52.722981,
            target_name="2020dgc",
        )
        self.event_dir = self.handoff / self.target.event_name()
        self.event_dir.mkdir(parents=True, exist_ok=True)

        self.ffi_dir = self.root / "ffis"
        self.ffi_dir.mkdir()
        self.product_id = "tess0001"
        self.ffi_name = "tess0001-s0020-3-3-0001-s_ffic.fits"
        self.ffi_path = self.ffi_dir / self.ffi_name
        self.ffi_path.write_bytes(b"SIMPLE  = T")

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "filename": [self.ffi_name],
                "path": [str(self.ffi_path)],
                "wcs_ok": [True],
                "group_id": [0],
            }
        ).to_csv(manifest_csv, index=False)

        self.cfg = freeze_target_diff_config(self.site / "diff_config.yaml", self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)
        self.cfg.output_dir = str(self.event_dir)

    def _downsample_fp(self) -> str:
        from syndiff_pipeline.common.provenance.fingerprint import (
            RECIPE_SCHEMA_VERSION,
            fingerprint,
            recipe_id,
        )
        from syndiff_pipeline.common.provenance.model import SccKey

        spatial = SccKey(20, 3, 3).to_dict()
        rid = recipe_id("downsample", {"test": True}, RECIPE_SCHEMA_VERSION)
        return fingerprint("downsample", spatial, rid, [])

    def _emit_diff_image(self, downsample_fp: str) -> str:
        inputs = pg.diff_image_input_fingerprints(
            sector=20,
            camera=3,
            ccd=3,
            ffi_path=str(self.ffi_path),
            downsample_fp=downsample_fp,
        )
        loc = self.root / "tess0001_hp_d.fits"
        loc.write_bytes(b"SIMPLE  = T")
        fp = pg.emit_diff_artifact(
            kind="diff_image",
            sector=20,
            camera=3,
            ccd=3,
            product_id=self.product_id,
            label="hp_d",
            params=HotpantsParams(),
            location=str(loc),
            input_fingerprints=inputs,
            data_root=str(self.cfg.data_root),
        )
        self.assertIsNotNone(fp)
        return fp

    def test_build_diff_image_fps_matches_emit(self):
        downsample_fp = self._downsample_fp()
        diff_fp = self._emit_diff_image(downsample_fp)
        diff_path = str(self.root / f"{self.product_id}_hp_d.fits")
        Path(diff_path).write_bytes(b"SIMPLE  = T")

        with patch.object(
            pg, "resolve_downsample_fingerprint_from_cfg", return_value=downsample_fp
        ):
            built = gridded_epsf.build_diff_image_fps(
                self.cfg,
                [diff_path],
                diffs_input="hp_d",
                sck=(20, 3, 3),
            )

        self.assertEqual(built.get(self.product_id), diff_fp)

    def test_emit_uses_real_diff_image_fp(self):
        real_fp = "real_diff_image_fp_abc123"
        output_dir = str(self.event_dir / "epsf_r1")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        diff_path = self.event_dir / f"{self.product_id}_hp_d.fits"
        fits.writeto(diff_path, np.zeros((8, 8), dtype=np.float32), overwrite=True)

        stack = np.ones((4, 5, 5), dtype=np.float64)
        grid_xypos = [(4.0, 4.0), (4.0, 4.0), (4.0, 4.0), (4.0, 4.0)]

        gridded_epsf._init_gridded_epsf_worker(
            pd.DataFrame({"x": [4.0], "y": [4.0], "phot_rp_mean_mag": [10.0]}),
            EpsfParams(tile_nx=1, tile_ny=1, psf_size=2, min_stars_per_tile=1),
            output_dir,
            None,
            skip_existing=False,
            sck=(20, 3, 3),
            data_root=str(self.cfg.data_root),
            epsf_label="epsf_r1",
            workspace_root=str(self.handoff),
            diff_image_fps={self.product_id: real_fp},
        )

        with patch.object(
            gridded_epsf, "build_gridded_psf_for_frame", return_value=(object(), grid_xypos, stack)
        ), patch.object(pg, "upstream_label_edge") as mock_loc_edge, patch.object(
            pg, "emit_diff_artifact"
        ) as mock_emit:
            result = gridded_epsf._fit_one_frame_task(0, str(diff_path))

        self.assertTrue(result[2])
        mock_loc_edge.assert_not_called()
        mock_emit.assert_called_once()
        self.assertEqual(mock_emit.call_args.kwargs["input_fingerprints"], [real_fp])
        self.assertFalse(
            any(
                str(fp).startswith("loc:")
                for fp in mock_emit.call_args.kwargs["input_fingerprints"]
            )
        )

    def test_emit_skipped_without_diff_image_fp(self):
        output_dir = str(self.event_dir / "epsf_r1")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        diff_path = self.event_dir / f"{self.product_id}_hp_d.fits"
        fits.writeto(diff_path, np.zeros((8, 8), dtype=np.float32), overwrite=True)

        stack = np.ones((4, 5, 5), dtype=np.float64)
        grid_xypos = [(4.0, 4.0), (4.0, 4.0), (4.0, 4.0), (4.0, 4.0)]

        gridded_epsf._init_gridded_epsf_worker(
            pd.DataFrame({"x": [4.0], "y": [4.0], "phot_rp_mean_mag": [10.0]}),
            EpsfParams(tile_nx=1, tile_ny=1, psf_size=2, min_stars_per_tile=1),
            output_dir,
            None,
            skip_existing=False,
            sck=(20, 3, 3),
            data_root=str(self.cfg.data_root),
            epsf_label="epsf_r1",
            workspace_root=str(self.handoff),
            diff_image_fps={},
        )

        with patch.object(
            gridded_epsf, "build_gridded_psf_for_frame", return_value=(object(), grid_xypos, stack)
        ), patch.object(pg, "upstream_label_edge") as mock_loc_edge, patch.object(
            pg, "emit_diff_artifact"
        ) as mock_emit:
            result = gridded_epsf._fit_one_frame_task(0, str(diff_path))

        self.assertTrue(result[2])
        mock_loc_edge.assert_not_called()
        mock_emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
