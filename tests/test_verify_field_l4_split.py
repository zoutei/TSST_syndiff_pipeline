"""Verify rejects legacy toggle manifests and enforces dual-cache field layout."""

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


def _v3_manifest(**overrides) -> dict:
    payload = {
        "schema_version": 3,
        "cache_quantum_ps1_px": 1.0,
        "keying": "absolute",
        "intra_skycell_R": 1,
        "n_shift_epochs": 0,
        "n_pair_epochs": 0,
        "n_intra_skycell_keys": 0,
        "n_inter_skycell_pair_states": 0,
    }
    payload.update(overrides)
    if "n_intra_skycell_keys" in overrides and "n_shift_epochs" not in overrides:
        payload["n_shift_epochs"] = int(overrides["n_intra_skycell_keys"])
    if "n_inter_skycell_pair_states" in overrides and "n_pair_epochs" not in overrides:
        payload["n_pair_epochs"] = int(overrides["n_inter_skycell_pair_states"])
    return payload


def _resolved(tmp: Path) -> ResolvedTargetConfig:
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
            remap=RemapStageParams(),
            downsample=DownsampleStageParams(geometry_mode="field"),
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
    # Schema-v3 epoch stub artifacts (required whenever Exact NPZs are claimed)
    if l4a_npz or l4b_npz or True:
        import numpy as np
        (store / "shift_epochs.parquet").write_bytes(b"parquet")
        (store / "pair_epochs.parquet").write_bytes(b"parquet")
        (store / "epoch_group_members.parquet").write_bytes(b"parquet")
        np.savez_compressed(
            store / "gid_epoch_index.npz",
            l4a_skycell=np.array([], dtype=object),
            l4a_gid=np.zeros(0, dtype=np.int32),
            l4a_sx=np.zeros(0, dtype=np.int32),
            l4a_sy=np.zeros(0, dtype=np.int32),
            l4a_epoch_id=np.zeros(0, dtype=np.int32),
            l4b_pair_lo=np.zeros(0, dtype=np.int32),
            l4b_pair_hi=np.zeros(0, dtype=np.int32),
            l4b_gid=np.zeros(0, dtype=np.int32),
            l4b_sx_lo=np.zeros(0, dtype=np.int32),
            l4b_sy_lo=np.zeros(0, dtype=np.int32),
            l4b_sx_hi=np.zeros(0, dtype=np.int32),
            l4b_sy_hi=np.zeros(0, dtype=np.int32),
            l4b_pair_epoch_id=np.zeros(0, dtype=np.int32),
        )
        np.save(store / "group_id_per_frame.npy", np.zeros(1, dtype=np.int32))
    if l4a_npz:
        l4a = store / EXACT_CACHE_L4A_DIRNAME
        l4a.mkdir(parents=True, exist_ok=True)
        for i in range(l4a_npz):
            cell = l4a / f"skycell.1.{i}"
            cell.mkdir(parents=True, exist_ok=True)
            (cell / f"e0_sx+1_sy+0_exact.npz").write_bytes(b"npz")
    if l4b_npz:
        l4b = store / EXACT_CACHE_L4B_DIRNAME
        pair = l4b / "pair_10__20"
        pair.mkdir(parents=True, exist_ok=True)
        for i in range(l4b_npz):
            (pair / f"e{i}_sx+0_sy+0_sx+1_sy+0_rim.npz").write_bytes(b"npz")
    if legacy_npz:
        legacy = store / EXACT_CACHE_LEGACY_DIRNAME
        legacy.mkdir(parents=True, exist_ok=True)
        for i in range(legacy_npz):
            (legacy / f"legacy_{i}.npz").write_bytes(b"npz")
    return store


def _write_field_store(resolved: ResolvedTargetConfig) -> Path:
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
                "group_scoped_contribs": True,
            }
        )
        + "\n"
    )
    p = contrib_path(store, "skycell.1.1", 0, 0, group_id=7)
    keys = [[7, "skycell.1.1", 0, 0]]
    import numpy as np

    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, indices=np.array([1], dtype=np.int32), flux_sum=np.array([1.0]))
    event_dir = Path(resolved.event_dir)
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / "field_contrib_keys.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "keys": keys,
            }
        )
        + "\n"
    )
    return store


class TestVerifyRejectsLegacyManifests(unittest.TestCase):
    def test_verify_remap_rejects_include_abutting_border_exact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(include_abutting_border_exact=True),
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("include_abutting_border_exact", result.message)

    def test_verify_remap_rejects_toggle_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest={
                    "schema_version": 2,
                    "cache_quantum_ps1_px": 1.0,
                    "keying": "absolute",
                    "apply_hybrid_exact": True,
                    "l4b_policy": "none",
                },
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("removed geometry toggles", result.message)

    def test_verify_remap_rejects_polluted_legacy_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(n_intra_skycell_keys=2),
                legacy_npz=2,
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("polluted", result.message)
            self.assertIn(EXACT_CACHE_LEGACY_DIRNAME, result.message)


class TestVerifyFieldDualCache(unittest.TestCase):
    def test_verify_remap_requires_inter_skycell_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(
                    n_intra_skycell_keys=2,
                    n_inter_skycell_pair_states=1,
                ),
                l4a_npz=2,
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn(EXACT_CACHE_L4B_DIRNAME, result.message)

    def test_verify_remap_intra_cache_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(
                    n_intra_skycell_keys=3,
                    n_inter_skycell_pair_states=1,
                ),
                l4a_npz=2,
                l4b_npz=1,
            )
            result = verify_remap(resolved)
            self.assertFalse(result.ok)
            self.assertIn("intra-skycell cache incomplete", result.message)

    def test_verify_remap_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(
                    n_intra_skycell_keys=2,
                    n_inter_skycell_pair_states=1,
                ),
                l4a_npz=2,
                l4b_npz=1,
            )
            result = verify_remap(resolved)
            self.assertTrue(result.ok)

    def test_verify_downsample_requires_gid_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(
                    n_intra_skycell_keys=1,
                    n_inter_skycell_pair_states=1,
                ),
                l4a_npz=1,
                l4b_npz=1,
            )
            store = field_templates_root(
                resolved.data_root, 20, 1, 1, oversampling_factor=1
            )
            store.mkdir(parents=True, exist_ok=True)
            (store / MANIFEST_NAME).write_text("{}")
            event_dir = Path(resolved.event_dir)
            event_dir.mkdir(parents=True, exist_ok=True)
            (event_dir / "field_contrib_keys.json").write_text(
                json.dumps({"schema_version": 1, "keys": [["skycell.1.1", 0, 0]]})
            )
            result = verify_downsample_field_mode(resolved)
            self.assertFalse(result.ok)
            self.assertIn("group-qualified", result.message)

    def test_verify_downsample_ok(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = _resolved(Path(tmpdir))
            _write_remap_store(
                resolved,
                manifest=_v3_manifest(
                    n_intra_skycell_keys=1,
                    n_inter_skycell_pair_states=1,
                ),
                l4a_npz=1,
                l4b_npz=1,
            )
            _write_field_store(resolved)
            result = verify_downsample_field_mode(resolved)
            self.assertTrue(result.ok)


class TestStageParamsFieldGeometry(unittest.TestCase):
    def test_rejects_removed_remap_keys(self):
        with self.assertRaises(ValueError) as ctx:
            parse_stage_params({"remap": {"l4b_policy": "pair_state"}})
        self.assertIn("l4b_policy", str(ctx.exception))

    def test_rejects_removed_downsample_keys(self):
        with self.assertRaises(ValueError) as ctx:
            parse_stage_params({"downsample": {"apply_hybrid_exact": True}})
        self.assertIn("apply_hybrid_exact", str(ctx.exception))

    def test_accepts_intra_skycell_R(self):
        stages = parse_stage_params({"remap": {"intra_skycell_R": 2}})
        self.assertEqual(stages.remap.intra_skycell_R, 2)


if __name__ == "__main__":
    unittest.main()
