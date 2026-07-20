"""Verify rejects L4b-lite manifests and enforces dual-cache pair_state layout."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.scc_paths import event_scc_leaf, scc_templates_dir
from syndiff_pipeline.template_creation.orchestration.runner_config import ResolvedTargetConfig
from syndiff_pipeline.template_creation.orchestration.stage_params import (
    DownsampleStageParams,
    MappingStageParams,
    Ps1DownloadStageParams,
    Ps1ProcessStageParams,
    RemapStageParams,
    TemplateStageParams,
    WcsGroupingStageParams,
    parse_stage_params,
)
from syndiff_pipeline.template_creation.orchestration.verify import (
    verify_downsample_field_mode,
    verify_remap,
)
from syndiff_pipeline.template_creation.processing.field_remap import (
    EXACT_CACHE_L4A_DIRNAME,
    EXACT_CACHE_L4B_DIRNAME,
    EXACT_CACHE_LEGACY_DIRNAME,
    REMAP_MANIFEST_NAME,
    remap_root,
)
from syndiff_pipeline.template_creation.processing.field_templates import (
    MANIFEST_NAME,
    contrib_path,
    field_templates_root,
)


def _resolved(
    tmp: Path,
    *,
    l4b_policy: str = "none",
    downsample_l4b_policy: str | None = None,
) -> ResolvedTargetConfig:
    target = Target(20, 1, 1, 210.0, 81.0, "2020ut")
    data_root = tmp / "data"
    return ResolvedTargetConfig(
        target=target,
        data_root=str(data_root),
        ffi_dir=str(data_root / "s0020" / "c1" / "k1" / "ffi"),
        event_dir=str(
            event_scc_leaf(tmp, target.event_name(), target.sector, target.camera, target.ccd)
        ),
        skycell_wcs_csv=str(tmp / "skycell_wcs.csv"),
        stages=TemplateStageParams(
            wcs_grouping=WcsGroupingStageParams(geometry_mode="field"),
            mapping=MappingStageParams(oversampling_factor=1),
            ps1_download=Ps1DownloadStageParams(),
            ps1_process=Ps1ProcessStageParams(),
            remap=RemapStageParams(l4b_policy=l4b_policy),
            downsample=DownsampleStageParams(
                geometry_mode="field",
                l4b_policy=downsample_l4b_policy
                if downsample_l4b_policy is not None
                else l4b_policy,
            ),
        ),
        mapping_root=str(data_root / "s0020" / "c1" / "k1" / "mapping" / "oversampling_1"),
        zarr_dir=str(data_root / "ps1_skycells_zarr"),
        template_output_base=str(
            scc_templates_dir(data_root, target.sector, target.camera, target.ccd, oversampling_factor=1)
        ),
    )


def _write_remap_store(
    resolved: ResolvedTargetConfig,
    *,
    manifest: dict,
    l4a_npz: int = 0,
    l4b_npz: int = 0,
    legacy_npz: int = 0,
) -> Path:
    t = resolved.target
    store = remap_root(
        resolved.data_root,
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=1,
    )
    store.mkdir(parents=True, exist_ok=True)
    (store / REMAP_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    (store / "shift_schedule.npz").write_bytes(b"npz")
    (store / "template_group_shifts.parquet").write_bytes(b"parquet")
    if l4a_npz:
        l4a = store / EXACT_CACHE_L4A_DIRNAME
        l4a.mkdir(parents=True, exist_ok=True)
        for i in range(l4a_npz):
            (l4a / f"skycell.1.{i}_sx+0_sy+0_exact.npz").write_bytes(b"npz")
    if l4b_npz:
        l4b = store / EXACT_CACHE_L4B_DIRNAME
        l4b.mkdir(parents=True, exist_ok=True)
        for i in range(l4b_npz):
            (l4b / f"pair_10__20_sx+0_sy+0_sx+1_sy+0_rim.npz").write_bytes(b"npz")
    if legacy_npz:
        legacy = store / EXACT_CACHE_LEGACY_DIRNAME
        legacy.mkdir(parents=True, exist_ok=True)
        for i in range(legacy_npz):
            (legacy / f"legacy_{i}.npz").write_bytes(b"npz")
    return store


def _write_field_store(
    resolved: ResolvedTargetConfig,
    *,
    group_scoped: bool = False,
    l4b_policy: str = "none",
) -> Path:
    t = resolved.target
    store = field_templates_root(
        resolved.data_root,
        t.sector,
        t.camera,
        t.ccd,
        oversampling_factor=1,
    )
    store.mkdir(parents=True, exist_ok=True)
    (store / MANIFEST_NAME).write_text(json.dumps({"geometry_mode": "field"}) + "\n")
    (store / "field_mode_assembly.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "l4b_policy": l4b_policy,
                "group_scoped_contribs": group_scoped,
            }
        )
        + "\n"
    )
    contrib_dir = store / "contribs"
    contrib_dir.mkdir(parents=True, exist_ok=True)
    if group_scoped:
        p = contrib_path(store, "skycell.1.1", 0, 0, group_id=7)
        keys = [[7, "skycell.1.1", 0, 0]]
        schema_version = 2
    else:
        p = contrib_path(store, "skycell.1.1", 0, 0)
        keys = [["skycell.1.1", 0, 0]]
        schema_version = 1
    import numpy as np

    np.savez(p, indices=np.array([1], dtype=np.int32), flux_sum=np.array([1.0]))
    event_dir = Path(resolved.event_dir)
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / "field_contrib_keys.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "l4b_policy": l4b_policy,
                "keys": keys,
            }
        )
        + "\n"
    )
    return store


class TestVerifyRejectsL4bLite(unittest.TestCase):
    def test_verify_remap_rejects_include_abutting_border_exact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "include_abutting_border_exact": True,
                    "l4b_policy": "none",
                },
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("include_abutting_border_exact", result.message)
            self.assertIn("exact_cache_l4a", result.message)

    def test_verify_remap_rejects_lite_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "l4b_policy": "lite",
                },
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("deprecated l4b_policy", result.message)

    def test_verify_remap_rejects_polluted_legacy_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "none",
                    "n_exact_keys": 2,
                },
                legacy_npz=2,
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("polluted", result.message)
            self.assertIn(EXACT_CACHE_LEGACY_DIRNAME, result.message)


class TestVerifyPairStateDualCache(unittest.TestCase):
    def test_verify_remap_pair_state_requires_l4b_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir), l4b_policy="pair_state")
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "pair_state",
                    "n_exact_keys": 2,
                    "n_l4b_pair_states": 1,
                },
                l4a_npz=2,
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn(EXACT_CACHE_L4B_DIRNAME, result.message)

    def test_verify_remap_pair_state_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir), l4b_policy="pair_state")
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "pair_state",
                    "n_exact_keys": 3,
                    "n_l4b_pair_states": 2,
                },
                l4a_npz=2,
                l4b_npz=1,
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("L4a cache incomplete", result.message)

    def test_verify_remap_pair_state_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir), l4b_policy="pair_state")
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "pair_state",
                    "n_exact_keys": 2,
                    "n_l4b_pair_states": 1,
                },
                l4a_npz=2,
                l4b_npz=1,
            )
            result = verify_remap(resolved)
            self.assertTrue(result.ok)

    def test_verify_downsample_pair_state_requires_gid_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir), l4b_policy="pair_state")
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "pair_state",
                    "n_exact_keys": 1,
                    "n_l4b_pair_states": 1,
                },
                l4a_npz=1,
                l4b_npz=1,
            )
            _write_field_store(resolved, group_scoped=False, l4b_policy="pair_state")
            result = verify_downsample_field_mode(resolved)
            self.assertFalse(result.ok)
            self.assertIn("group-qualified", result.message)

    def test_verify_downsample_pair_state_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir), l4b_policy="pair_state")
            _write_remap_store(
                resolved,
                manifest={
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "pair_state",
                    "n_exact_keys": 1,
                    "n_l4b_pair_states": 1,
                },
                l4a_npz=1,
                l4b_npz=1,
            )
            _write_field_store(resolved, group_scoped=True, l4b_policy="pair_state")
            result = verify_downsample_field_mode(resolved)
            self.assertTrue(result.ok)


class TestStageParamsL4bPolicy(unittest.TestCase):
    def test_rejects_invalid_l4b_policy(self):
        with self.assertRaises(ValueError) as ctx:
            parse_stage_params({"remap": {"l4b_policy": "lite"}})
        self.assertIn("l4b_policy", str(ctx.exception))

    def test_accepts_pair_state_and_require_l4b_cache(self):
        stages = parse_stage_params(
            {
                "remap": {"l4b_policy": "pair_state"},
                "downsample": {
                    "l4b_policy": "pair_state",
                    "require_l4b_cache": True,
                },
            }
        )
        self.assertEqual(stages.remap.l4b_policy, "pair_state")
        self.assertEqual(stages.downsample.l4b_policy, "pair_state")
        self.assertTrue(stages.downsample.require_l4b_cache)


if __name__ == "__main__":
    unittest.main()
