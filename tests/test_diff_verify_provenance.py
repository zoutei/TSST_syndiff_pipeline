"""Tests for the PR-D1 indexed completeness path in ``diff_verify.py``.

Covers: ``diff_stage_complete_indexed`` answering True/False against a temp
``ProvenanceStore`` with synthetic rows (fingerprints computed the same way
``provenance_glue``/the save-site emit calls compute them), falling open
(``None``) cleanly when the provenance package is made unavailable, and
``diff_workspace_complete`` preferring the indexed answer over the legacy
marker check while still falling back to it when the store can't answer.
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

from syndiff_pipeline.common.scc_paths import event_scc_leaf, provenance_db_path
from syndiff_pipeline.common.orchestration.targets import Target
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

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "filename": [
                    "tess0001-s0020-3-3-0001-s_ffic.fits",
                    "tess0002-s0020-3-3-0002-s_ffic.fits",
                ],
                "wcs_ok": [True, True],
                "group_id": [0, 0],
            }
        ).to_csv(manifest_csv, index=False)

        self.cfg = freeze_target_diff_config(self.site / "diff_config.yaml", self.target)

    def _hotpants_stage(self) -> dict:
        return {"kind": "hotpants", "output": {"diffs": "diffs", "convolved": "convolved"}}


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestDiffStageCompleteIndexed(_BaseCase):
    def test_data_root_threaded_from_site_config(self):
        self.assertEqual(self.cfg.data_root, str(self.data.resolve()))

    def _required_fps(self) -> list[str]:
        hp = HotpantsParams()
        return [
            pg.diff_kind_fingerprint(
                "diff_image",
                sector=self.cfg.sector,
                camera=self.cfg.camera,
                ccd=self.cfg.ccd,
                product_id=pid,
                label="diffs",
                params=hp,
            )
            for pid in ("tess0001", "tess0002")
        ]

    def test_none_when_store_empty(self):
        # No rows ingested yet -> indexed check answers False (definitively,
        # not fall-open) once the provenance package is present but nothing
        # has been recorded.
        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertFalse(result)

    def test_true_once_rows_ingested(self):
        from syndiff_pipeline.common.provenance.store import ProvenanceStore

        fps = self._required_fps()
        db_path = provenance_db_path(self.cfg.data_root)
        store = ProvenanceStore(db_path)
        for i, fp in enumerate(fps):
            rid = f"rid{i}"
            store.upsert_recipe(rid, "diff_image", {"x": i}, 1)
            store.upsert_artifact(
                fp,
                "diff_image",
                {"s": self.cfg.sector, "c": self.cfg.camera, "k": self.cfg.ccd, "product_id": f"tess000{i+1}"},
                rid,
                f"/tmp/loc{i}.fits",
            )

        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertTrue(result)

    def test_partial_ingest_is_incomplete(self):
        from syndiff_pipeline.common.provenance.store import ProvenanceStore

        fps = self._required_fps()
        db_path = provenance_db_path(self.cfg.data_root)
        store = ProvenanceStore(db_path)
        store.upsert_recipe("rid0", "diff_image", {"x": 0}, 1)
        store.upsert_artifact(
            fps[0],
            "diff_image",
            {"s": self.cfg.sector, "c": self.cfg.camera, "k": self.cfg.ccd, "product_id": "tess0001"},
            "rid0",
            "/tmp/loc0.fits",
        )
        result = dv.diff_stage_complete_indexed(self.cfg, self.event_dir, self._hotpants_stage())
        self.assertFalse(result)

    def test_unsupported_stage_kind_falls_open(self):
        result = dv.diff_stage_complete_indexed(
            self.cfg, self.event_dir, {"kind": "forced_photometry", "output": "lc"}
        )
        self.assertIsNone(result)

    def test_workspace_complete_prefers_indexed_true(self):
        """diff_workspace_complete should trust an indexed True even though no
        legacy marker files exist under ws/ at all."""
        from syndiff_pipeline.common.provenance.store import ProvenanceStore

        fps = self._required_fps()
        db_path = provenance_db_path(self.cfg.data_root)
        store = ProvenanceStore(db_path)
        for i, fp in enumerate(fps):
            rid = f"rid{i}"
            store.upsert_recipe(rid, "diff_image", {"x": i}, 1)
            store.upsert_artifact(
                fp,
                "diff_image",
                {"s": self.cfg.sector, "c": self.cfg.camera, "k": self.cfg.ccd, "product_id": f"tess000{i+1}"},
                rid,
                f"/tmp/loc{i}.fits",
            )
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
        # No legacy marker present either -> overall False, but critically no
        # exception and no false-positive from a broken indexed path.
        self.assertFalse(dv.diff_workspace_complete(self.cfg, self.event_dir))

        ws_dir = dv.diff_workspace_root(self.cfg, self.event_dir) / "diffs"
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "tess0001_diffs.fits").write_bytes(b"SIMPLE  = T")
        self.assertTrue(dv.diff_workspace_complete(self.cfg, self.event_dir))


if __name__ == "__main__":
    unittest.main()
