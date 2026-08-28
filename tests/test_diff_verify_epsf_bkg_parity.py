"""BK-4: indexed verify input-fingerprint parity for epsf and diff_background."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
    scc_diff_bookkeeping_dir,
)
from syndiff_pipeline.difference_imaging.orchestration import diff_verify as dv
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    EpsfParams,
    HotpantsParams,
)
from syndiff_pipeline.difference_imaging.support.manifest import (
    manifest_path_from_output_dir,
)
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


_HOTPANTS_EPSF_DIFF_POLICY = {
    "paths": {"template_base": "shifted_downsampled"},
    "pipeline": [
        {
            "kind": "hotpants",
            "output": {"diffs": "hp_d", "convolved": "hp_c"},
        },
        {
            "kind": "epsf",
            "inputs": {"diffs": "hp_d"},
            "output": "epsf_r1",
        },
    ],
}


class _ParityBase(unittest.TestCase):
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
            diff=_HOTPANTS_EPSF_DIFF_POLICY,
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
        self.ffi_paths = {}
        for pid, name in (
            ("tess0001", "tess0001-s0020-3-3-0001-s_ffic.fits"),
            ("tess0002", "tess0002-s0020-3-3-0002-s_ffic.fits"),
        ):
            p = self.ffi_dir / name
            p.write_bytes(b"SIMPLE  = T")
            self.ffi_paths[pid] = str(p)

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        frames_frame = pd.DataFrame(
            {
                "filename": [
                    Path(self.ffi_paths["tess0001"]).name,
                    Path(self.ffi_paths["tess0002"]).name,
                ],
                "path": [self.ffi_paths["tess0001"], self.ffi_paths["tess0002"]],
                "wcs_ok": [True, True],
                "group_id": [0, 0],
            }
        )
        frames_frame.to_csv(manifest_csv, index=False)

        # SCC-only storage: load_diff_frames_for_verify (the primary indexed
        # verify path) reads bookkeeping/diff/oversampling_N/frames.csv +
        # diff_job.json, not the legacy event-level manifest above. Seed both
        # so indexed-parity checks below can resolve the frame manifest.
        bk = scc_diff_bookkeeping_dir(
            self.data, self.target.sector, self.target.camera, self.target.ccd
        )
        bk.mkdir(parents=True, exist_ok=True)
        frames_frame.to_csv(bk / FRAMES_CSV_BASENAME, index=False)
        (bk / DIFF_JOB_BASENAME).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "sector": self.target.sector,
                    "camera": self.target.camera,
                    "ccd": self.target.ccd,
                    "mapping_grid": {
                        "sector": self.target.sector,
                        "camera": self.target.camera,
                        "ccd": self.target.ccd,
                    },
                }
            ),
            encoding="utf-8",
        )

        self.cfg = resolve_target_diff_config(self.site, self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)

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

    def _seed_downsample(self) -> str:
        downsample_fp = self._downsample_fp()
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        from syndiff_pipeline.common.provenance.fingerprint import RECIPE_SCHEMA_VERSION, recipe_id

        rid = recipe_id("downsample", {"test": True}, RECIPE_SCHEMA_VERSION)
        store.upsert_recipe(rid, "downsample", {"test": True}, RECIPE_SCHEMA_VERSION)
        store.upsert_artifact(
            downsample_fp,
            "downsample",
            {"s": 20, "c": 3, "k": 3},
            rid,
            "/tmp/templates",
        )
        return downsample_fp

    def _emit_diff_images(self, downsample_fp: str) -> dict[str, str]:
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
            loc = self.root / f"{pid}_hp_d.fits"
            loc.write_bytes(b"SIMPLE  = T")
            fp = pg.emit_diff_artifact(
                kind="diff_image",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="hp_d",
                params=hp,
                location=str(loc),
                input_fingerprints=inputs,
                data_root=str(self.cfg.data_root),
            )
            self.assertIsNotNone(fp)
            out[pid] = fp
        return out


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestEpsfIndexedParity(_ParityBase):
    def test_emit_indexed_verify_parity(self):
        # Per-frame fingerprint scheme (own diff image only) -- explicit
        # epsf_mode="per_frame" since orbit_binned (the new default) uses a
        # different scheme (see gridded_epsf_orbit.py F1) that this indexed
        # fast-path deliberately falls open on.
        downsample_fp = self._seed_downsample()
        diff_fps = self._emit_diff_images(downsample_fp)
        epsf_params = EpsfParams(epsf_mode="per_frame")
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        for pid, diff_fp in diff_fps.items():
            inputs = pg.epsf_input_fingerprints(diff_fp)
            self.assertEqual(inputs, [diff_fp])
            loc = self.root / f"{pid}_epsf.npz"
            loc.write_bytes(b"PK")
            pg.emit_diff_artifact(
                kind="epsf",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="epsf_r1",
                params=epsf_params,
                location=str(loc),
                input_fingerprints=inputs,
                data_root=str(self.cfg.data_root),
                is_fits=False,
            )
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        stage = {
            "kind": "epsf",
            "inputs": {"diffs": "hp_d"},
            "output": "epsf_r1",
            "epsf_mode": "per_frame",
        }
        self.assertTrue(dv.diff_stage_complete_indexed(self.cfg, self.event_dir, stage))

    def test_falls_open_without_downsample(self):
        stage = {
            "kind": "epsf",
            "inputs": {"diffs": "hp_d"},
            "output": "epsf_r1",
        }
        self.assertIsNone(dv.diff_stage_complete_indexed(self.cfg, self.event_dir, stage))


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestDiffBackgroundIndexedParity(_ParityBase):
    def test_hotpants_internal_emit_matches_indexed_inputs(self):
        """ffi_fp vector from emit matches indexed helper for hotpants-internal bkg."""
        hp = HotpantsParams()
        frames = dv.load_diff_frames_for_verify(self.cfg, self.event_dir)
        for pid, ffi_path in self.ffi_paths.items():
            ffi_fp = pg.ffi_input_fingerprint(20, 3, 3, ffi_path)
            emit_inputs = pg.diff_background_input_fingerprints(ffi_fp)
            indexed_inputs = dv._indexed_input_fingerprints(
                cfg=self.cfg,
                stage={"kind": "hotpants", "output": {"diffs": "hp_d", "phot_bkg": "hp_b"}},
                kind="diff_background",
                frames_df=frames,
                product_id=pid,
            )
            self.assertEqual(indexed_inputs, emit_inputs)
            emit_fp = pg.diff_kind_fingerprint(
                "diff_background",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="hp_b",
                params=hp,
                input_fingerprints=emit_inputs,
            )
            indexed_fp = pg.diff_kind_fingerprint(
                "diff_background",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="hp_b",
                params=hp,
                input_fingerprints=indexed_inputs,
            )
            self.assertEqual(emit_fp, indexed_fp)

    def test_falls_open_when_ffi_unresolved(self):
        frames = dv.load_diff_frames_for_verify(self.cfg, self.event_dir)
        result = dv._indexed_input_fingerprints(
            cfg=self.cfg,
            stage={"kind": "hotpants", "output": {"diffs": "hp_d"}},
            kind="diff_background",
            frames_df=frames,
            product_id="tess9999",
        )
        self.assertIsNone(result)


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestSccManifestPreference(_ParityBase):
    def test_indexed_verify_prefers_scc_over_event_manifest(self):
        """When both manifests exist, SCC row count drives required product ids."""
        bk = scc_diff_bookkeeping_dir(
            self.data, self.target.sector, self.target.camera, self.target.ccd
        )
        bk.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "path": [self.ffi_paths["tess0001"], self.ffi_paths["tess0002"]],
                "ffi_basename": [
                    Path(self.ffi_paths["tess0001"]).name,
                    Path(self.ffi_paths["tess0002"]).name,
                ],
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
                    "mapping_grid": {
                        "sector": self.target.sector,
                        "camera": self.target.camera,
                        "ccd": self.target.ccd,
                    },
                }
            ),
            encoding="utf-8",
        )

        downsample_fp = self._seed_downsample()
        diff_fps = self._emit_diff_images(downsample_fp)
        epsf_params = EpsfParams()
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        # Only ingest epsf for tess0001 — event manifest has one row but SCC has two.
        pid = "tess0001"
        inputs = pg.epsf_input_fingerprints(diff_fps[pid])
        loc = self.root / f"{pid}_epsf.npz"
        loc.write_bytes(b"PK")
        pg.emit_diff_artifact(
            kind="epsf",
            sector=20,
            camera=3,
            ccd=3,
            product_id=pid,
            label="epsf_r1",
            params=epsf_params,
            location=str(loc),
            input_fingerprints=inputs,
            data_root=str(self.cfg.data_root),
            is_fits=False,
        )
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        frames = dv.load_diff_frames_for_verify(self.cfg, self.event_dir)
        self.assertEqual(len(frames), 2)

        stage = {
            "kind": "epsf",
            "inputs": {"diffs": "hp_d"},
            "output": "epsf_r1",
        }
        self.assertFalse(dv.diff_stage_complete_indexed(self.cfg, self.event_dir, stage))


if __name__ == "__main__":
    unittest.main()
