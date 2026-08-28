"""BK-4b: photometry provenance emit uses real fingerprints and fail-opens without loc:."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
)
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    EpsfParams,
    HotpantsParams,
    PsfPhotometryMethodParams,
)
from syndiff_pipeline.difference_imaging.stages.photometry import (
    ForcedPhotTargetSpec,
    _photometry_provenance_input_fingerprints,
    _try_emit_photometry_provenance,
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


class _PhotometryProvBase(unittest.TestCase):
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
        self.product_id = "tess0001"
        self.ffi_path = self.ffi_dir / "tess0001-s0020-3-3-0001-s_ffic.fits"
        self.ffi_path.write_bytes(b"SIMPLE  = T")

        manifest_csv = Path(manifest_path_from_output_dir(str(self.event_dir), None))
        manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "filename": [self.ffi_path.name],
                "path": [str(self.ffi_path)],
                "wcs_ok": [True],
                "group_id": [0],
            }
        ).to_csv(manifest_csv, index=False)

        self.cfg = resolve_target_diff_config(self.site, self.target)
        self.cfg.ffi_dir = str(self.ffi_dir)
        self.cfg.output_dir = str(self.event_dir)
        self.cfg.target_name = self.target.target_name

        self.diff_path = self.event_dir / "ws" / "hp_d" / f"{self.product_id}_hp_d.fits.fz"
        self.diff_path.parent.mkdir(parents=True, exist_ok=True)
        self.diff_path.write_bytes(b"SIMPLE  = T")

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

    def _expected_diff_image_fp(self, downsample_fp: str) -> str:
        hp = HotpantsParams()
        inputs = pg.diff_image_input_fingerprints(
            sector=20,
            camera=3,
            ccd=3,
            ffi_path=str(self.ffi_path),
            downsample_fp=downsample_fp,
        )
        self.assertIsNotNone(inputs)
        return pg.diff_kind_fingerprint(
            "diff_image",
            sector=20,
            camera=3,
            ccd=3,
            product_id=self.product_id,
            label="hp_d",
            params=hp,
            input_fingerprints=inputs,
        )

    def _expected_epsf_fp(self, diff_image_fp: str) -> str:
        epsf_params = EpsfParams()
        inputs = pg.epsf_input_fingerprints(diff_image_fp)
        self.assertEqual(inputs, [diff_image_fp])
        return pg.diff_kind_fingerprint(
            "epsf",
            sector=20,
            camera=3,
            ccd=3,
            product_id=self.product_id,
            label="epsf_r1",
            params=epsf_params,
            input_fingerprints=inputs,
        )


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestPhotometryProvenanceInputFingerprints(_PhotometryProvBase):
    def test_fails_open_without_downsample(self):
        out = _photometry_provenance_input_fingerprints(
            self.cfg,
            [str(self.diff_path)],
            diffs_label="hp_d",
        )
        self.assertIsNone(out)

    def test_real_diff_image_fp_without_loc(self):
        downsample_fp = self._seed_downsample()
        expected = self._expected_diff_image_fp(downsample_fp)
        self.assertIsNotNone(expected)

        out = _photometry_provenance_input_fingerprints(
            self.cfg,
            [str(self.diff_path)],
            diffs_label="hp_d",
        )
        self.assertEqual(out, [expected])
        self.assertTrue(all(not fp.startswith("loc:") for fp in out))

    def test_includes_epsf_fp_when_workspace_configured(self):
        downsample_fp = self._seed_downsample()
        diff_fp = self._expected_diff_image_fp(downsample_fp)
        epsf_fp = self._expected_epsf_fp(diff_fp)

        out = _photometry_provenance_input_fingerprints(
            self.cfg,
            [str(self.diff_path)],
            diffs_label="hp_d",
            epsf_workspace="epsf_r1",
        )
        self.assertEqual(sorted(out), sorted([diff_fp, epsf_fp]))
        self.assertTrue(all(not fp.startswith("loc:") for fp in out))

    def test_path_only_would_have_been_loc_skips(self):
        """Bare path with no recipe context must not mint loc: via upstream_label_edge."""
        bare_cfg = mock.Mock(
            pipeline=[],
            sector=20,
            camera=3,
            ccd=3,
            ffi_dir="",
            output_dir="",
            manifest="",
            data_root=str(self.data),
        )
        out = _photometry_provenance_input_fingerprints(
            bare_cfg,
            ["/tmp/no_such_diff.fits"],
        )
        self.assertIsNone(out)


@unittest.skipUnless(
    pg.PROVENANCE_AVAILABLE and pg._SPOOL_AVAILABLE,
    "provenance spool unavailable",
)
class TestPhotometryProvenanceEmit(_PhotometryProvBase):
    def _spool_records(self) -> list[dict]:
        spool = provenance_spool_dir(self.cfg.data_root)
        records: list[dict] = []
        for path in Path(spool).glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        return records

    def test_emit_skipped_when_only_loc_would_be_available(self):
        method = PsfPhotometryMethodParams(name="psf_epsf", psf_type="epsf")
        _try_emit_photometry_provenance(
            cfg=self.cfg,
            method=method,
            diff_paths=[str(self.diff_path)],
            phot_targets=[
                ForcedPhotTargetSpec(
                    target_xy=(10.0, 10.0),
                    csv_basename="lightcurve.csv",
                    plot_source_label="primary",
                )
            ],
            output_dir=str(self.event_dir / "lc_out"),
            output_label="lc",
            diffs_input="hp_d",
            stage_epsf_workspace="epsf_r1",
            gridded_epsf_by_workspace={"epsf_r1": object()},
        )
        self.assertEqual(self._spool_records(), [])

    def test_emit_records_real_edges(self):
        downsample_fp = self._seed_downsample()
        diff_fp = self._expected_diff_image_fp(downsample_fp)
        epsf_fp = self._expected_epsf_fp(diff_fp)
        method = PsfPhotometryMethodParams(name="psf_epsf", psf_type="epsf")
        out_dir = self.event_dir / "lc_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        _try_emit_photometry_provenance(
            cfg=self.cfg,
            method=method,
            diff_paths=[str(self.diff_path)],
            phot_targets=[
                ForcedPhotTargetSpec(
                    target_xy=(10.0, 10.0),
                    csv_basename="lightcurve.csv",
                    plot_source_label="primary",
                )
            ],
            output_dir=str(out_dir),
            output_label="lc",
            diffs_input="hp_d",
            stage_epsf_workspace="epsf_r1",
            gridded_epsf_by_workspace={"epsf_r1": object()},
        )

        records = self._spool_records()
        self.assertEqual(len(records), 1)
        inputs = records[0]["inputs"]
        self.assertEqual(sorted(inputs), sorted([diff_fp, epsf_fp]))
        self.assertTrue(all(not fp.startswith("loc:") for fp in inputs))

        store = ProvenanceStore(str(provenance_db_path(self.cfg.data_root)))
        drain_spool(store, provenance_spool_dir(self.cfg.data_root))
        emitted_fp = records[0]["fingerprint"]
        self.assertEqual(sorted(store.inputs_of(emitted_fp)), sorted([diff_fp, epsf_fp]))

    def test_emit_never_raises(self):
        broken_cfg = mock.Mock(side_effect=RuntimeError("boom"))
        method = PsfPhotometryMethodParams(name="psf_epsf", psf_type="epsf")
        _try_emit_photometry_provenance(
            cfg=broken_cfg,
            method=method,
            diff_paths=[str(self.diff_path)],
            phot_targets=[],
            output_dir="/tmp/x",
            output_label="lc",
            diffs_input="hp_d",
            stage_epsf_workspace=None,
            gridded_epsf_by_workspace=None,
        )


if __name__ == "__main__":
    unittest.main()
