"""BK-4 / BK-4b extras: scc_assembly convolved inputs + photometry two-event lineage."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.ingest import drain_spool
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, provenance_spool_dir
from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    PsfPhotometryMethodParams,
)
from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    expected_scc_assembly_fingerprint,
    emit_scc_assembly_checkpoint,
)
from tests.test_provenance_checkpoint import _FakeMappingParams, _FakePs1ProcessParams, _FakeResolved, _FakeStages, _FakeTarget


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestSccAssemblyConvolvedInputs(unittest.TestCase):
    def test_inputs_nonempty_when_cells_resolvable(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            # Minimal mapping CSV + unique published convolved cell.
            mapping_dir = (
                data_root
                / "s0020"
                / "c1"
                / "k1"
                / "mapping"
                / "oversampling_2"
            )
            mapping_dir.mkdir(parents=True)
            csv_path = mapping_dir / "tess_s0020_1_1_master_skycells_list_os2.csv"
            csv_path.write_text(
                "NAME,projection,y,x,NAXIS1,NAXIS2\n"
                "skycell.2246.000,2246,0,0,4800,4800\n",
                encoding="utf-8",
            )
            cell_fp = "abc123convolvedcell0001"
            cell_dir = (
                data_root
                / "ps1_skycells_zarr"
                / "ps1_convolved.zarr"
                / "2246"
                / "skycell.2246.000"
                / cell_fp
            )
            cell_dir.mkdir(parents=True)
            (cell_dir / "_provenance.json").write_text(
                json.dumps({"fingerprint": cell_fp}), encoding="utf-8"
            )
            # Payload markers so _payload_complete isn't required for sidecar read
            (cell_dir / "arrays.npz").write_bytes(b"x")
            (cell_dir / "headers.json").write_text("{}", encoding="utf-8")
            (cell_dir / "removed_stars.json").write_text("[]", encoding="utf-8")

            resolved = _FakeResolved(
                data_root=str(data_root),
                target=_FakeTarget(sector=20, camera=1, ccd=1),
                stages=_FakeStages(_FakeMappingParams(), _FakePs1ProcessParams()),
            )
            # Ensure mapping recipe attrs exist
            resolved.stages.mapping.pad_distance = 0.0
            resolved.stages.mapping.overwrite = False

            emit_scc_assembly_checkpoint(resolved)
            records = []
            spool = Path(provenance_spool_dir(str(data_root)))
            for path in spool.glob("*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        records.append(json.loads(line))
            self.assertEqual(len(records), 1)
            inputs = records[0]["inputs"]
            self.assertGreaterEqual(len(inputs), 2)  # mapping + >=1 convolved
            self.assertIn(cell_fp, inputs)
            self.assertEqual(
                records[0]["fingerprint"], expected_scc_assembly_fingerprint(resolved)
            )


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestPhotometryTwoEventLineage(unittest.TestCase):
    def test_two_events_share_upstream_differ_by_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            shared_diff_fp = "shareddiffimagefingerprint1"
            shared_epsf_fp = "sharedepsffingerprint00001"
            method = PsfPhotometryMethodParams(name="psf_epsf")

            fp_a = pg.emit_photometry_artifact(
                event="event_a",
                sector=20,
                camera=3,
                ccd=3,
                method=method.name,
                label="lc",
                params=method,
                location=str(data_root / "event_a" / "lc.csv"),
                input_fingerprints=[shared_diff_fp, shared_epsf_fp],
                data_root=str(data_root),
            )
            fp_b = pg.emit_photometry_artifact(
                event="event_b",
                sector=20,
                camera=3,
                ccd=3,
                method=method.name,
                label="lc",
                params=method,
                location=str(data_root / "event_b" / "lc.csv"),
                input_fingerprints=[shared_diff_fp, shared_epsf_fp],
                data_root=str(data_root),
            )
            self.assertIsNotNone(fp_a)
            self.assertIsNotNone(fp_b)
            self.assertNotEqual(fp_a, fp_b)

            store = ProvenanceStore(str(provenance_db_path(str(data_root))))
            drain_spool(store, provenance_spool_dir(str(data_root)))
            row_a = store.artifact(fp_a)
            row_b = store.artifact(fp_b)
            self.assertIsNotNone(row_a)
            self.assertIsNotNone(row_b)
            self.assertEqual(row_a.spatial_key["event"], "event_a")
            self.assertEqual(row_b.spatial_key["event"], "event_b")
            self.assertEqual(sorted(store.inputs_of(fp_a)), sorted([shared_diff_fp, shared_epsf_fp]))
            self.assertEqual(sorted(store.inputs_of(fp_b)), sorted([shared_diff_fp, shared_epsf_fp]))


if __name__ == "__main__":
    unittest.main()
