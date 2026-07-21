"""Tests for the ``scc_assembly`` provenance checkpoint (PR2, plan §11).

All tests here run against the *real* ``provenance.fingerprint`` /
``provenance.model`` / ``provenance.publish`` / ``scc_paths`` modules (all
landed on this branch as of this PR) -- determinism, config-sensitivity, and
the sidecar record shape are real, unmocked, end-to-end assertions against a
temp ``data_root``.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
    SCC_ASSEMBLY_KIND,
    emit_scc_assembly_checkpoint,
    expected_scc_assembly_fingerprint,
)
from syndiff_pipeline.common.scc_paths import provenance_spool_dir, scc_convolved_zarr


class _FakeMappingParams:
    def __init__(
        self,
        oversampling_factor: int = 2,
        *,
        x_left_dead: int = 44,
        x_right_dead: int = 44,
        y_edge_strip: int = 30,
        template_conv_pad_spare_px: int = 4,
        sci_fwhm: float = 1.88,
        pad_distance: float = 0.0,
        overwrite: bool = False,
    ) -> None:
        self.oversampling_factor = oversampling_factor
        self.x_left_dead = x_left_dead
        self.x_right_dead = x_right_dead
        self.y_edge_strip = y_edge_strip
        self.template_conv_pad_spare_px = template_conv_pad_spare_px
        self.sci_fwhm = sci_fwhm
        self.pad_distance = pad_distance
        self.overwrite = overwrite


class _FakePs1ProcessParams:
    def __init__(
        self,
        *,
        projections_limit=None,
        psf_sigma: float = 2.5,
        enable_saturation_correction: bool = True,
        remove_saturated_stars: bool = True,
        bright_star_mag_threshold: float = 12.0,
        use_shared_convolved_store: bool = False,
        write_per_scc_convolved_zarr: bool = True,
    ) -> None:
        self.projections_limit = projections_limit
        self.psf_sigma = psf_sigma
        self.enable_saturation_correction = enable_saturation_correction
        self.remove_saturated_stars = remove_saturated_stars
        self.bright_star_mag_threshold = bright_star_mag_threshold
        self.use_shared_convolved_store = use_shared_convolved_store
        self.write_per_scc_convolved_zarr = write_per_scc_convolved_zarr


class _FakeStages:
    def __init__(self, mapping: _FakeMappingParams, ps1_process: _FakePs1ProcessParams) -> None:
        self.mapping = mapping
        self.ps1_process = ps1_process


class _FakeTarget:
    def __init__(self, sector: int = 20, camera: int = 1, ccd: int = 1) -> None:
        self.sector = sector
        self.camera = camera
        self.ccd = ccd


class _FakeResolved:
    """Minimal duck-typed stand-in for ``ResolvedTargetConfig``.

    Only the attributes ``provenance_checkpoint`` actually reads are
    populated, so this test is decoupled from ``runner_config``'s real
    dataclass shape.
    """

    def __init__(
        self,
        *,
        data_root: str = "/data/root",
        target: _FakeTarget | None = None,
        stages: _FakeStages | None = None,
    ) -> None:
        self.data_root = data_root
        self.target = target or _FakeTarget()
        self.stages = stages or _FakeStages(_FakeMappingParams(), _FakePs1ProcessParams())


def _resolved(*, data_root: str = "/data/root", **ps1_process_overrides) -> _FakeResolved:
    return _FakeResolved(
        data_root=data_root,
        stages=_FakeStages(
            _FakeMappingParams(),
            _FakePs1ProcessParams(**ps1_process_overrides),
        ),
    )


class TestExpectedSccAssemblyFingerprint(unittest.TestCase):
    """Real (unmocked) determinism / config-sensitivity against fingerprint+model."""

    def test_deterministic_for_identical_config(self):
        fp1 = expected_scc_assembly_fingerprint(_resolved())
        fp2 = expected_scc_assembly_fingerprint(_resolved())
        self.assertEqual(fp1, fp2)
        self.assertIsInstance(fp1, str)
        self.assertTrue(fp1)

    def test_psf_sigma_changes_fingerprint(self):
        fp_base = expected_scc_assembly_fingerprint(_resolved(psf_sigma=2.5))
        fp_changed = expected_scc_assembly_fingerprint(_resolved(psf_sigma=3.0))
        self.assertNotEqual(fp_base, fp_changed)

    def test_saturation_correction_flag_changes_fingerprint(self):
        fp_base = expected_scc_assembly_fingerprint(
            _resolved(enable_saturation_correction=True)
        )
        fp_changed = expected_scc_assembly_fingerprint(
            _resolved(enable_saturation_correction=False)
        )
        self.assertNotEqual(fp_base, fp_changed)

    def test_bright_star_mag_threshold_changes_fingerprint(self):
        fp_base = expected_scc_assembly_fingerprint(
            _resolved(bright_star_mag_threshold=12.0)
        )
        fp_changed = expected_scc_assembly_fingerprint(
            _resolved(bright_star_mag_threshold=13.0)
        )
        self.assertNotEqual(fp_base, fp_changed)

    def test_projections_limit_changes_fingerprint(self):
        fp_base = expected_scc_assembly_fingerprint(_resolved(projections_limit=None))
        fp_changed = expected_scc_assembly_fingerprint(_resolved(projections_limit=5))
        self.assertNotEqual(fp_base, fp_changed)

    def test_oversampling_changes_fingerprint(self):
        resolved_os2 = _FakeResolved(
            stages=_FakeStages(_FakeMappingParams(oversampling_factor=2), _FakePs1ProcessParams())
        )
        resolved_os4 = _FakeResolved(
            stages=_FakeStages(_FakeMappingParams(oversampling_factor=4), _FakePs1ProcessParams())
        )
        fp2 = expected_scc_assembly_fingerprint(resolved_os2)
        fp4 = expected_scc_assembly_fingerprint(resolved_os4)
        self.assertNotEqual(fp2, fp4)

    def test_sector_camera_ccd_changes_fingerprint(self):
        resolved_a = _FakeResolved(target=_FakeTarget(sector=20, camera=1, ccd=1))
        resolved_b = _FakeResolved(target=_FakeTarget(sector=20, camera=1, ccd=2))
        self.assertNotEqual(
            expected_scc_assembly_fingerprint(resolved_a),
            expected_scc_assembly_fingerprint(resolved_b),
        )

    def test_unrelated_field_does_not_change_fingerprint(self):
        # data_root affects location, not identity (plan §5: fingerprint is a
        # pure function of kind/spatial_key/recipe/inputs, not of location).
        resolved_a = _FakeResolved(data_root="/data/root/a")
        resolved_b = _FakeResolved(data_root="/data/root/b")
        self.assertEqual(
            expected_scc_assembly_fingerprint(resolved_a),
            expected_scc_assembly_fingerprint(resolved_b),
        )


def _read_spool_records(data_root: str) -> list[dict]:
    """Read every JSON line from every spool file under *data_root* (test-only)."""
    spool_dir = Path(provenance_spool_dir(data_root))
    if not spool_dir.is_dir():
        return []
    records: list[dict] = []
    for path in sorted(spool_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestEmitSccAssemblyCheckpoint(unittest.TestCase):
    """End-to-end tests against the real ``provenance.publish``/``scc_paths``
    modules, writing into a temp ``data_root``."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.data_root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_emit_writes_exactly_one_well_formed_sidecar_record(self):
        resolved = _resolved(data_root=self.data_root)
        emit_scc_assembly_checkpoint(resolved)

        records = _read_spool_records(self.data_root)
        self.assertEqual(len(records), 1)
        record = records[0]

        self.assertEqual(record["kind"], SCC_ASSEMBLY_KIND)
        self.assertEqual(record["fingerprint"], expected_scc_assembly_fingerprint(resolved))
        self.assertEqual(record["spatial_key"], {"s": 20, "c": 1, "k": 1, "os": 2})
        self.assertEqual(
            record["recipe_params"],
            {
                "projections_limit": None,
                "psf_sigma": 2.5,
                "enable_saturation_correction": True,
                "remove_saturated_stars": True,
                "bright_star_mag_threshold": 12.0,
                "mapping_grid": {
                    "x_left_dead": 44,
                    "x_right_dead": 44,
                    "y_edge_strip": 30,
                    "conv_pad_native": 8,
                    "oversampling_factor": 2,
                },
            },
        )
        from syndiff_pipeline.template_creation.orchestration.provenance_checkpoint import (
            expected_mapping_fingerprint,
        )

        self.assertEqual(record["inputs"], [expected_mapping_fingerprint(resolved)])
        self.assertEqual(record["state"], "complete")
        self.assertEqual(
            record["location"],
            str(scc_convolved_zarr(self.data_root, 20, 1, 1)),
        )
        # No bytes moved: emitting the checkpoint must not touch the
        # (nonexistent, in this test) convolved.zarr location at all.
        self.assertFalse(Path(record["location"]).exists())

    def test_emit_is_idempotent_across_repeated_calls(self):
        # Same config -> same fingerprint -> two spool lines with identical
        # content (append-only spool; dedup happens at ingest via INSERT OR
        # REPLACE, not here).
        resolved = _resolved(data_root=self.data_root)
        emit_scc_assembly_checkpoint(resolved)
        emit_scc_assembly_checkpoint(resolved)
        records = _read_spool_records(self.data_root)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["fingerprint"], records[1]["fingerprint"])

    def test_emit_does_not_raise_when_publish_raises(self):
        resolved = _resolved(data_root=self.data_root)
        with mock.patch(
            "syndiff_pipeline.common.provenance.publish.append_spool_record",
            side_effect=RuntimeError("spool disk full"),
        ):
            try:
                emit_scc_assembly_checkpoint(resolved)
            except Exception as exc:  # pragma: no cover - the assertion is the point
                self.fail(f"emit_scc_assembly_checkpoint raised: {exc!r}")
        self.assertEqual(_read_spool_records(self.data_root), [])

    def test_emit_does_not_raise_when_provenance_publish_is_absent(self):
        # Simulate the provenance package being unimportable (e.g. mid an
        # in-flight authoring window, or a broken install): setting a module
        # to None in sys.modules makes `import` raise ImportError.
        resolved = _resolved(data_root=self.data_root)
        with mock.patch.dict(
            sys.modules, {"syndiff_pipeline.common.provenance.publish": None}
        ):
            try:
                emit_scc_assembly_checkpoint(resolved)
            except Exception as exc:  # pragma: no cover - the assertion is the point
                self.fail(f"emit_scc_assembly_checkpoint raised: {exc!r}")
        self.assertEqual(_read_spool_records(self.data_root), [])


if __name__ == "__main__":
    unittest.main()
