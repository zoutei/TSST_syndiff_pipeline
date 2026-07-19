"""Tests for the supervisor's throttled provenance spool drain (PR2, plan §10/§15).

``_maybe_drain_provenance_spool`` is called from the supervisor's tick loop
(``run_supervisor_daemon``) next to ``write_verify_in_flight``. It must:

- be a no-op with no active-run data roots;
- drain each distinct ``data_root`` seen among active runs, throttled to at
  most once per ``_PROVENANCE_DRAIN_INTERVAL_S``;
- never raise -- neither when the ``provenance`` package can't be imported
  at all nor when a per-``data_root`` drain fails.

These tests run against the real ``provenance.store``/``provenance.ingest``/
``provenance.publish``/``scc_paths`` modules (all landed on this branch as of
this PR), draining real spool files into real (temp-directory) SQLite
provenance stores -- true end-to-end coverage of the supervisor's ingest
wiring, not a mocked contract.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration import scheduler
from syndiff_pipeline.common.provenance.fingerprint import RECIPE_SCHEMA_VERSION, fingerprint, recipe_id
from syndiff_pipeline.common.provenance.publish import append_spool_record, build_record
from syndiff_pipeline.common.provenance.store import ProvenanceStore
from syndiff_pipeline.common.scc_paths import provenance_db_path, provenance_spool_dir


_STORE_MODULE = "syndiff_pipeline.common.provenance.store"
_INGEST_MODULE = "syndiff_pipeline.common.provenance.ingest"
_PUBLISH_MODULE = "syndiff_pipeline.common.provenance.publish"


def _write_one_spool_record(data_root: str, *, spatial_key: dict) -> str:
    """Append one well-formed spool record under *data_root* via the real
    publish path; returns its fingerprint."""
    params = {"psf_sigma": 2.5}
    rid = recipe_id("scc_assembly", params, RECIPE_SCHEMA_VERSION)
    fp = fingerprint("scc_assembly", spatial_key, rid, [])
    record = build_record(
        fp, "scc_assembly", spatial_key, rid, RECIPE_SCHEMA_VERSION, [], "/legacy/convolved.zarr",
        recipe_params=params,
    )
    append_spool_record(provenance_spool_dir(data_root), record)
    return fp


class TestMaybeDrainProvenanceSpool(unittest.TestCase):
    def setUp(self):
        self._orig_ts = scheduler._last_provenance_drain_ts
        scheduler._last_provenance_drain_ts = 0.0

    def tearDown(self):
        scheduler._last_provenance_drain_ts = self._orig_ts

    def test_no_data_roots_is_a_noop(self):
        with mock.patch(
            "syndiff_pipeline.common.orchestration.scheduler.time.monotonic",
            return_value=100.0,
        ):
            scheduler._maybe_drain_provenance_spool(set())
        self.assertEqual(scheduler._last_provenance_drain_ts, 0.0)

    def test_import_failure_is_swallowed(self):
        # Simulate the provenance package being unimportable (mid-authoring
        # window, broken install, ...): setting a module to None in
        # sys.modules makes `import` raise ImportError.
        with mock.patch.dict(sys.modules, {_STORE_MODULE: None, _INGEST_MODULE: None}):
            try:
                scheduler._maybe_drain_provenance_spool({"/data/root"})
            except Exception as exc:  # pragma: no cover - the assertion is the point
                self.fail(f"_maybe_drain_provenance_spool raised: {exc!r}")

    def test_drains_each_data_root_into_its_own_real_store(self):
        with tempfile.TemporaryDirectory() as root_a, tempfile.TemporaryDirectory() as root_b:
            fp_a = _write_one_spool_record(root_a, spatial_key={"s": 1, "c": 1, "k": 1, "os": 2})
            fp_b = _write_one_spool_record(root_b, spatial_key={"s": 2, "c": 2, "k": 2, "os": 2})

            with mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.time.monotonic",
                return_value=1000.0,
            ):
                scheduler._maybe_drain_provenance_spool({root_a, root_b})

            store_a = ProvenanceStore(str(provenance_db_path(root_a)))
            store_b = ProvenanceStore(str(provenance_db_path(root_b)))
            self.assertIsNotNone(store_a.artifact(fp_a))
            self.assertIsNotNone(store_b.artifact(fp_b))
            # Cross-check: root_a's store must not see root_b's artifact.
            self.assertIsNone(store_a.artifact(fp_b))
            self.assertEqual(scheduler._last_provenance_drain_ts, 1000.0)

    def test_throttled_within_interval(self):
        with tempfile.TemporaryDirectory() as root_a:
            _write_one_spool_record(root_a, spatial_key={"s": 1, "c": 1, "k": 1, "os": 2})

            with mock.patch(
                f"{_INGEST_MODULE}.drain_spool"
            ) as mock_drain, mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.time.monotonic",
                side_effect=[1000.0, 1000.5, 1001.0],
            ):
                scheduler._maybe_drain_provenance_spool({root_a})  # ts=1000.0 -> drains
                scheduler._maybe_drain_provenance_spool({root_a})  # ts=1000.5 -> throttled
                scheduler._maybe_drain_provenance_spool({root_a})  # ts=1001.0 -> throttled

            mock_drain.assert_called_once()

    def test_drains_again_after_interval_elapses(self):
        with tempfile.TemporaryDirectory() as root_a:
            _write_one_spool_record(root_a, spatial_key={"s": 1, "c": 1, "k": 1, "os": 2})

            with mock.patch(
                f"{_INGEST_MODULE}.drain_spool"
            ) as mock_drain, mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.time.monotonic",
                side_effect=[1000.0, 1006.0],
            ):
                scheduler._maybe_drain_provenance_spool({root_a})  # drains
                scheduler._maybe_drain_provenance_spool({root_a})  # 6s later -> drains again

            self.assertEqual(mock_drain.call_count, 2)

    def test_per_data_root_failure_is_isolated(self):
        with tempfile.TemporaryDirectory() as root_bad, tempfile.TemporaryDirectory() as root_good:
            _write_one_spool_record(root_bad, spatial_key={"s": 1, "c": 1, "k": 1, "os": 2})
            fp_good = _write_one_spool_record(root_good, spatial_key={"s": 2, "c": 2, "k": 2, "os": 2})

            real_store_cls = ProvenanceStore

            def flaky_store(db_path, **kwargs):
                if str(db_path).startswith(root_bad):
                    raise RuntimeError("boom")
                return real_store_cls(db_path, **kwargs)

            with mock.patch(
                f"{_STORE_MODULE}.ProvenanceStore", side_effect=flaky_store
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler.time.monotonic",
                return_value=2000.0,
            ):
                try:
                    scheduler._maybe_drain_provenance_spool({root_bad, root_good})
                except Exception as exc:  # pragma: no cover - the assertion is the point
                    self.fail(f"_maybe_drain_provenance_spool raised: {exc!r}")

            store_good = real_store_cls(str(provenance_db_path(root_good)))
            self.assertIsNotNone(store_good.artifact(fp_good))


if __name__ == "__main__":
    unittest.main()
