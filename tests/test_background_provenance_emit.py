"""Background stage provenance emit: real diff_image edges, no loc: tokens."""

from __future__ import annotations

import json
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
from syndiff_pipeline.common.provenance.ingest import drain_spool
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    provenance_db_path,
    provenance_spool_dir,
    resolve_scc_diff_bookkeeping_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
)
from syndiff_pipeline.difference_imaging.orchestration import diff_verify as dv
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.config import save_config
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.background.pipeline import BackgroundParams
from syndiff_pipeline.difference_imaging.stages.background import io as bkg_io
from syndiff_pipeline.difference_imaging.support.ffi_naming import (
    workspace_frame_fits_basename,
    workspace_frame_stem,
)
from syndiff_pipeline.difference_imaging.support.manifest import (
    manifest_path_from_output_dir,
)
from syndiff_pipeline.difference_imaging.support.paths import DIFF_CONFIG_SNAPSHOT_BASENAME
from tests.site_fixtures import resolve_target_diff_config, write_site_deployment, write_unified_site_config


def _target() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=3,
        target_ra=228.479042,
        target_dec=52.722981,
        target_name="2020dgc",
    )


_HOTPANTS_BACKGROUND_DIFF_POLICY = {
    "paths": {"template_base": "shifted_downsampled"},
    "pipeline": [
        {
            "kind": "hotpants",
            "output": {"diffs": "hp_d", "convolved": "hp_c"},
        },
        {
            "kind": "background_temporal_smoothing",
            "inputs": {"diffs": "hp_d"},
            "output": "bkg_s1",
        },
    ],
}


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestBackgroundProvenanceEmit(unittest.TestCase):
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
        write_unified_site_config(
            self.site / "pipeline.yaml",
            workspace_root=str(self.handoff),
            data_root=str(self.data),
            diff=_HOTPANTS_BACKGROUND_DIFF_POLICY,
        )

        self.target = _target()
        self.event_dir = event_scc_leaf(
            self.handoff,
            self.target.event_name(),
            self.target.sector,
            self.target.camera,
            self.target.ccd,
        )
        self.event_dir.mkdir(parents=True, exist_ok=True)

        self.ffi_dir = self.root / "ffis"
        self.ffi_dir.mkdir()
        self.ffi_paths: dict[str, str] = {}
        for pid, name in (
            ("tess0001", "tess0001-s0020-3-3-0001-s_ffic.fits"),
            ("tess0002", "tess0002-s0020-3-3-0002-s_ffic.fits"),
        ):
            p = self.ffi_dir / name
            p.write_bytes(b"SIMPLE  = T")
            self.ffi_paths[pid] = str(p)

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "filename": [
                    Path(self.ffi_paths["tess0001"]).name,
                    Path(self.ffi_paths["tess0002"]).name,
                ],
                "path": [self.ffi_paths["tess0001"], self.ffi_paths["tess0002"]],
                "wcs_ok": [True, True],
                "group_id": [0, 0],
            }
        ).to_csv(manifest_csv, index=False)

        self.cfg = resolve_target_diff_config(self.site, self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)
        self.cfg.output_dir = str(self.event_dir)

        self.ws_tree = self.event_dir / "ws"
        self.ws_tree.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, str(self.ws_tree / DIFF_CONFIG_SNAPSHOT_BASENAME))

        self.diff_dir = self.ws_tree / "hp_d"
        self.diff_dir.mkdir(parents=True, exist_ok=True)
        for pid in self.ffi_paths:
            stem = workspace_frame_stem(pid, "hp_d")
            (self.diff_dir / workspace_frame_fits_basename(stem)).write_bytes(b"SIMPLE  = T")
        self.out_dir = self.ws_tree / "bkg_s1"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_scc_handoff()

    def _write_scc_handoff(self) -> None:
        bk = resolve_scc_diff_bookkeeping_dir(
            self.data, self.target.sector, self.target.camera, self.target.ccd
        )
        bk.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "path": list(self.ffi_paths.values()),
                "ffi_basename": [Path(p).name for p in self.ffi_paths.values()],
                "wcs_ok": [True, True],
                "group_id": [0, 0],
            }
        ).to_csv(bk / FRAMES_CSV_BASENAME, index=False)
        (bk / DIFF_JOB_BASENAME).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "sector": self.target.sector,
                    "camera": self.target.camera,
                    "ccd": self.target.ccd,
                    "geometry_mode": "field",
                    "mapping_grid": {
                        "sector": self.target.sector,
                        "camera": self.target.camera,
                        "ccd": self.target.ccd,
                    },
                    "crop_bounds": {"x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10},
                }
            ),
            encoding="utf-8",
        )

    def _seed_downsample(self) -> str:
        from syndiff_pipeline.common.provenance.fingerprint import (
            RECIPE_SCHEMA_VERSION,
            fingerprint,
            recipe_id,
        )
        from syndiff_pipeline.common.provenance.model import SccKey

        spatial = SccKey(20, 3, 3).to_dict()
        rid = recipe_id("downsample", {"test": True}, RECIPE_SCHEMA_VERSION)
        downsample_fp = fingerprint("downsample", spatial, rid, [])
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        store.upsert_recipe(rid, "downsample", {"test": True}, RECIPE_SCHEMA_VERSION)
        store.upsert_artifact(
            downsample_fp,
            "downsample",
            {"s": 20, "c": 3, "k": 3},
            rid,
            "/tmp/templates",
        )
        return downsample_fp

    def _expected_diff_image_fps(self, downsample_fp: str) -> dict[str, str]:
        hp = HotpantsParams()
        out: dict[str, str] = {}
        for pid, ffi_path in self.ffi_paths.items():
            inputs = pg.diff_image_input_fingerprints(
                sector=20,
                camera=3,
                ccd=3,
                ffi_path=ffi_path,
                downsample_fp=downsample_fp,
            )
            fp = pg.diff_kind_fingerprint(
                "diff_image",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="hp_d",
                params=hp,
                input_fingerprints=inputs,
            )
            self.assertIsNotNone(fp)
            out[pid] = fp
        return out

    def _frame_records(self) -> list[bkg_io.FrameRecord]:
        ffi_paths = [self.ffi_paths["tess0001"], self.ffi_paths["tess0002"]]
        wcs_table = pd.DataFrame(
            {
                "path": ffi_paths,
                "btjd": [2458000.0, 2458001.0],
                "group_id": [0, 0],
            }
        )
        return bkg_io.build_frame_records(
            ffi_paths, wcs_table, str(self.diff_dir), bkg_dir=None
        )

    def test_emit_uses_real_diff_image_fingerprint(self) -> None:
        downsample_fp = self._seed_downsample()
        expected_diff_fps = self._expected_diff_image_fps(downsample_fp)
        records = self._frame_records()
        stack = np.zeros((len(records), 4, 4), dtype=np.float32)

        bkg_io.write_per_frame_fits(
            str(self.out_dir),
            stack,
            records,
            sck=(20, 3, 3),
            data_root=str(self.cfg.data_root),
            background_params=BackgroundParams(),
        )

        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        for rec in records:
            spatial = pg.ffi_spatial_key(20, 3, 3, rec.product_id, "bkg_s1")
            rows = store.artifacts_by_kind_spatial("diff_background", spatial)
            self.assertEqual(len(rows), 1, rec.product_id)
            inputs = store.inputs_of(rows[0].fingerprint)
            self.assertEqual(inputs, [expected_diff_fps[rec.product_id]])
            self.assertFalse(any(str(i).startswith("loc:") for i in inputs))

    def test_emit_matches_indexed_verify_inputs(self) -> None:
        downsample_fp = self._seed_downsample()
        records = self._frame_records()
        stack = np.zeros((len(records), 4, 4), dtype=np.float32)
        bkg_params = BackgroundParams()

        bkg_io.write_per_frame_fits(
            str(self.out_dir),
            stack,
            records,
            sck=(20, 3, 3),
            data_root=str(self.cfg.data_root),
            background_params=bkg_params,
        )

        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))
        frames = dv.load_diff_frames_for_verify(self.cfg, self.event_dir)
        stage = {
            "kind": "background_temporal_smoothing",
            "inputs": {"diffs": "hp_d"},
            "output": "bkg_s1",
        }

        for rec in records:
            indexed_inputs = dv._indexed_input_fingerprints(
                cfg=self.cfg,
                stage=stage,
                kind="diff_background",
                frames_df=frames,
                product_id=rec.product_id,
                downsample_fp=downsample_fp,
            )
            self.assertIsNotNone(indexed_inputs)
            spatial = pg.ffi_spatial_key(20, 3, 3, rec.product_id, "bkg_s1")
            rows = store.artifacts_by_kind_spatial("diff_background", spatial)
            self.assertEqual(len(rows), 1)
            emit_inputs = store.inputs_of(rows[0].fingerprint)
            self.assertEqual(emit_inputs, indexed_inputs)

    def test_skips_emit_when_diff_image_unresolved(self) -> None:
        records = self._frame_records()
        stack = np.zeros((len(records), 4, 4), dtype=np.float32)

        bkg_io.write_per_frame_fits(
            str(self.out_dir),
            stack,
            records,
            sck=(20, 3, 3),
            data_root=str(self.cfg.data_root),
            background_params=BackgroundParams(),
        )

        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        for rec in records:
            spatial = pg.ffi_spatial_key(20, 3, 3, rec.product_id, "bkg_s1")
            rows = store.artifacts_by_kind_spatial("diff_background", spatial)
            self.assertEqual(rows, [], rec.product_id)


if __name__ == "__main__":
    unittest.main()
