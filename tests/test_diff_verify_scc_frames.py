"""BK-5: SCC-primary frame manifest for diff indexed verify (no event bind)."""

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
    HotpantsParams,
    BackgroundEstimateParams,
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


_HOTPANTS_DIFF_POLICY = {
    "paths": {"template_base": "shifted_downsampled"},
    "pipeline": [
        {"kind": "shared_mask"},
        {"kind": "hotpants", "output": {"diffs": "diffs", "convolved": "convolved"}},
    ],
}

_KERNEL_SUBTRACT_DIFF_POLICY = {
    "paths": {"template_base": "shifted_downsampled"},
    "pipeline": [
        {"kind": "background_estimate", "output": {"diffs": "ks_d"}},
    ],
}


def _write_hotpants_policy(site: Path, *, workspace_root: str, data_root: str) -> None:
    write_unified_site_config(
        site / "pipeline.yaml",
        workspace_root=workspace_root,
        data_root=data_root,
        diff=_HOTPANTS_DIFF_POLICY,
    )


def _write_kernel_subtract_policy(site: Path, *, workspace_root: str, data_root: str) -> None:
    write_unified_site_config(
        site / "pipeline.yaml",
        workspace_root=workspace_root,
        data_root=data_root,
        diff=_KERNEL_SUBTRACT_DIFF_POLICY,
    )


class _SccFramesBase(unittest.TestCase):
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

        self._write_scc_handoff()

    def _write_scc_handoff(self) -> None:
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

    def _seed_downsample(self, cfg) -> str:
        downsample_fp = self._downsample_fp()
        store = ProvenanceStore(str(provenance_db_path(cfg.data_root)))
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


class TestLoadDiffFramesForVerify(_SccFramesBase):
    def test_prefers_scc_bookkeeping_without_event_manifest(self):
        _write_hotpants_policy(self.site, workspace_root=str(self.handoff), data_root=str(self.data))
        cfg = resolve_target_diff_config(self.site, self.target)
        cfg.ffi_dir = str(self.ffi_dir)

        event_manifest = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        self.assertFalse(event_manifest.is_file())

        frames = dv.load_diff_frames_for_verify(cfg, self.event_dir)
        self.assertEqual(len(frames), 2)
        self.assertIn("path", frames.columns)
        self.assertTrue(str(frames.iloc[0]["path"]).endswith(".fits"))

    def test_raises_when_no_scc_handoff(self):
        _write_hotpants_policy(self.site, workspace_root=str(self.handoff), data_root=str(self.data))
        cfg = resolve_target_diff_config(self.site, self.target)
        cfg.ffi_dir = str(self.ffi_dir)

        bk = scc_diff_bookkeeping_dir(
            self.data, self.target.sector, self.target.camera, self.target.ccd
        )
        for name in (FRAMES_CSV_BASENAME, DIFF_JOB_BASENAME):
            path = bk / name
            if path.is_file():
                path.unlink()

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "filename": [Path(self.ffi_paths["tess0001"]).name],
                "path": [self.ffi_paths["tess0001"]],
                "wcs_ok": [True],
                "group_id": [0],
            }
        ).to_csv(manifest_csv, index=False)

        with self.assertRaises(FileNotFoundError):
            dv.load_diff_frames_for_verify(cfg, self.event_dir)


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestSccIndexedVerify(_SccFramesBase):
    def setUp(self) -> None:
        super().setUp()
        _write_hotpants_policy(self.site, workspace_root=str(self.handoff), data_root=str(self.data))
        self.cfg = resolve_target_diff_config(self.site, self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)

    def test_workspace_complete_without_event_manifest(self):
        downsample_fp = self._seed_downsample(self.cfg)
        hp = HotpantsParams()
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        for pid, ffi_path in self.ffi_paths.items():
            inputs = pg.diff_image_input_fingerprints(
                sector=20,
                camera=3,
                ccd=3,
                ffi_path=ffi_path,
                downsample_fp=downsample_fp,
            )
            loc = self.root / f"{pid}_diffs.fits"
            loc.write_bytes(b"SIMPLE  = T")
            pg.emit_diff_artifact(
                kind="diff_image",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="diffs",
                params=hp,
                location=str(loc),
                input_fingerprints=inputs,
                data_root=str(self.cfg.data_root),
            )
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        event_manifest = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        self.assertFalse(event_manifest.is_file())
        # No ws/ workspace tree under SCC-scoped output_dir (wave A-3);
        # completeness is driven entirely by SCC bookkeeping + indexed
        # provenance, so no directory needs to exist under event_dir here.
        self.assertTrue(dv.diff_workspace_complete(self.cfg, self.event_dir))

    def test_indexed_verify_uses_scc_frames(self):
        downsample_fp = self._seed_downsample(self.cfg)
        hp = HotpantsParams()
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        for pid, ffi_path in self.ffi_paths.items():
            inputs = pg.diff_image_input_fingerprints(
                sector=20,
                camera=3,
                ccd=3,
                ffi_path=ffi_path,
                downsample_fp=downsample_fp,
            )
            loc = self.root / f"{pid}_diffs.fits"
            loc.write_bytes(b"SIMPLE  = T")
            pg.emit_diff_artifact(
                kind="diff_image",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="diffs",
                params=hp,
                location=str(loc),
                input_fingerprints=inputs,
                data_root=str(self.cfg.data_root),
            )
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        stage = {"kind": "hotpants", "output": {"diffs": "diffs", "convolved": "convolved"}}
        self.assertTrue(dv.diff_stage_complete_indexed(self.cfg, self.event_dir, stage))


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestKernelSubtractIndexedVerify(_SccFramesBase):
    def setUp(self) -> None:
        super().setUp()
        _write_kernel_subtract_policy(self.site, workspace_root=str(self.handoff), data_root=str(self.data))
        self.cfg = resolve_target_diff_config(self.site, self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)

    def test_kernel_subtract_indexed_kind(self):
        downsample_fp = self._seed_downsample(self.cfg)
        ks = BackgroundEstimateParams()
        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        for pid, ffi_path in self.ffi_paths.items():
            inputs = pg.diff_image_input_fingerprints(
                sector=20,
                camera=3,
                ccd=3,
                ffi_path=ffi_path,
                downsample_fp=downsample_fp,
            )
            loc = self.root / f"{pid}_ks_d.fits"
            loc.write_bytes(b"SIMPLE  = T")
            pg.emit_diff_artifact(
                kind="diff_image",
                sector=20,
                camera=3,
                ccd=3,
                product_id=pid,
                label="ks_d",
                params=ks,
                location=str(loc),
                input_fingerprints=inputs,
                data_root=str(self.cfg.data_root),
            )
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        stage = {"kind": "background_estimate", "output": {"diffs": "ks_d"}}
        self.assertTrue(dv.diff_stage_complete_indexed(self.cfg, self.event_dir, stage))


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestMissingFingerprintsResume(_SccFramesBase):
    def test_diff_image_complete_in_store_fail_open_without_db(self):
        _write_hotpants_policy(self.site, workspace_root=str(self.handoff), data_root=str(self.data))
        cfg = resolve_target_diff_config(self.site, self.target)
        cfg.ffi_dir = str(self.ffi_dir)
        result = pg.diff_image_complete_in_store(
            sector=20,
            camera=3,
            ccd=3,
            product_id="tess0001",
            label="diffs",
            params=HotpantsParams(),
            ffi_path=self.ffi_paths["tess0001"],
            data_root=str(cfg.data_root),
        )
        self.assertIsNone(result)

    def test_diff_image_complete_in_store_true_when_indexed(self):
        _write_hotpants_policy(self.site, workspace_root=str(self.handoff), data_root=str(self.data))
        cfg = resolve_target_diff_config(self.site, self.target)
        cfg.ffi_dir = str(self.ffi_dir)
        downsample_fp = self._seed_downsample(cfg)
        hp = HotpantsParams()
        inputs = pg.diff_image_input_fingerprints(
            sector=20,
            camera=3,
            ccd=3,
            ffi_path=self.ffi_paths["tess0001"],
            downsample_fp=downsample_fp,
        )
        store = ProvenanceStore(str(provenance_db_path(cfg.data_root)))
        loc = self.root / "tess0001_diffs.fits"
        loc.write_bytes(b"SIMPLE  = T")
        pg.emit_diff_artifact(
            kind="diff_image",
            sector=20,
            camera=3,
            ccd=3,
            product_id="tess0001",
            label="diffs",
            params=hp,
            location=str(loc),
            input_fingerprints=inputs,
            data_root=str(cfg.data_root),
        )
        drain_spool(store, provenance_spool_dir(cfg.data_root))

        self.assertTrue(
            pg.diff_image_complete_in_store(
                sector=20,
                camera=3,
                ccd=3,
                product_id="tess0001",
                label="diffs",
                params=hp,
                ffi_path=self.ffi_paths["tess0001"],
                downsample_fp=downsample_fp,
                data_root=str(cfg.data_root),
            )
        )


if __name__ == "__main__":
    unittest.main()
