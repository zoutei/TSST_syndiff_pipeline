"""Tests for the PR-D1 indexed completeness path in ``diff_verify.py``.

Covers: emit → drain_spool → ``diff_stage_complete_indexed`` True for the same
frame/params (BK-4 parity), falling open (``None``) when provenance is
unavailable, and ``diff_workspace_complete`` preferring the indexed answer.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.scc_paths import event_scc_leaf, provenance_db_path, provenance_spool_dir
from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.provenance.ingest import drain_spool
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.difference_imaging.orchestration import diff_verify as dv
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    freeze_target_diff_config,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.support.manifest import (
    manifest_path_from_output_dir,
)
from tests.site_fixtures import write_site_deployment


def _target() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=3,
        target_ra=228.479042,
        target_dec=52.722981,
        target_name="2020dgc",
    )


def _write_hotpants_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "paths:",
                "  template_base: shifted_downsampled",
                "pipeline:",
                "  - kind: shared_mask",
                "  - kind: hotpants",
                "    output:",
                "      diffs: diffs",
                "      convolved: convolved",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class _BaseCase(unittest.TestCase):
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
        _write_hotpants_policy(self.site / "diff_config.yaml")

        self.target = _target()
        self.event_dir = event_scc_leaf(
            self.handoff,
            self.target.event_name(),
            self.target.sector,
            self.target.camera,
            self.target.ccd,
        )
        self.event_dir.mkdir(parents=True, exist_ok=True)
        (self.event_dir / "event_job.json").write_text(
            '{"reference_ffi_path": "/tmp/ref.fits"}', encoding="utf-8"
        )

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

        self.cfg = freeze_target_diff_config(self.site / "diff_config.yaml", self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)

    def _hotpants_stage(self) -> dict:
        return {"kind": "hotpants", "output": {"diffs": "diffs", "convolved": "convolved"}}

    def _downsample_fp(self) -> str:
        """Synthetic SCC downsample fingerprint (empty-input recipe stand-in)."""
        from syndiff_pipeline.common.provenance.fingerprint import (
            RECIPE_SCHEMA_VERSION,
            fingerprint,
            recipe_id,
        )
        from syndiff_pipeline.common.provenance.model import SccKey

        spatial = SccKey(20, 3, 3).to_dict()
        rid = recipe_id("downsample", {"test": True}, RECIPE_SCHEMA_VERSION)
        return fingerprint("downsample", spatial, rid, [])


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestDiffStageCompleteIndexed(_BaseCase):
    def test_data_root_threaded_from_site_config(self):
        self.assertEqual(self.cfg.data_root, str(self.data.resolve()))

    def test_none_when_store_empty(self):
        # Without a resolvable downsample edge, indexed check falls open.
        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertIsNone(result)

    def test_emit_ingest_indexed_verify_parity(self):
        """emit → drain_spool → indexed verify True (same inputs as emit)."""
        downsample_fp = self._downsample_fp()
        # Seed the downsample node so verify's store lookup / explicit path matches emit.
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

        hp = HotpantsParams()
        for pid, ffi_path in self.ffi_paths.items():
            inputs = pg.diff_image_input_fingerprints(
                sector=20,
                camera=3,
                ccd=3,
                ffi_path=ffi_path,
                downsample_fp=downsample_fp,
            )
            self.assertIsNotNone(inputs)
            loc = self.root / f"{pid}_diffs.fits"
            loc.write_bytes(b"SIMPLE  = T")
            fp = pg.emit_diff_artifact(
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
            self.assertIsNotNone(fp)

        drain_spool(store, provenance_spool_dir(self.cfg.data_root))

        # Point verify at the seeded downsample via store lookup (spatial key match).
        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertTrue(result)

    def test_partial_ingest_is_incomplete(self):
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

        hp = HotpantsParams()
        pid = "tess0001"
        inputs = pg.diff_image_input_fingerprints(
            sector=20, camera=3, ccd=3, ffi_path=self.ffi_paths[pid], downsample_fp=downsample_fp
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

        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertFalse(result)

    def test_unsupported_stage_kind_falls_open(self):
        result = dv.diff_stage_complete_indexed(
            self.cfg, self.event_dir, {"kind": "forced_photometry", "output": "lc"}
        )
        self.assertIsNone(result)

    def test_workspace_complete_prefers_indexed_true(self):
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

        hp = HotpantsParams()
        for pid, ffi_path in self.ffi_paths.items():
            inputs = pg.diff_image_input_fingerprints(
                sector=20, camera=3, ccd=3, ffi_path=ffi_path, downsample_fp=downsample_fp
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

        ws_dir = dv.diff_workspace_root(self.cfg, self.event_dir)
        ws_dir.mkdir(parents=True, exist_ok=True)
        self.assertTrue(dv.diff_workspace_complete(self.cfg, self.event_dir))


class TestFallsOpenWithoutPackage(_BaseCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved = (
            pg.PROVENANCE_AVAILABLE,
            pg._SPOOL_AVAILABLE,
            pg._STORE_AVAILABLE,
        )
        pg.PROVENANCE_AVAILABLE = False
        pg._SPOOL_AVAILABLE = False
        pg._STORE_AVAILABLE = False
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        (
            pg.PROVENANCE_AVAILABLE,
            pg._SPOOL_AVAILABLE,
            pg._STORE_AVAILABLE,
        ) = self._saved

    def test_indexed_check_returns_none(self):
        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertIsNone(result)

    def test_workspace_complete_falls_back_to_legacy_marker(self):
        self.assertFalse(dv.diff_workspace_complete(self.cfg, self.event_dir))

        ws_dir = dv.diff_workspace_root(self.cfg, self.event_dir) / "diffs"
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "tess0001_diffs.fits").write_bytes(b"SIMPLE  = T")
        self.assertTrue(dv.diff_workspace_complete(self.cfg, self.event_dir))


if __name__ == "__main__":
    unittest.main()
