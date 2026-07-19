"""Unit tests for the provenance model (recipes, artifacts, per-kind params).

Uses a lightweight fake ``resolved`` (SimpleNamespace) mirroring the real
ResolvedTargetConfig shape: ``.target`` (sector/camera/ccd) and ``.stages.*``
carrying the stage-param attributes the builders read.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from syndiff_pipeline.common.provenance import model


def _fake_resolved(**overrides):
    """Build a fake ResolvedTargetConfig-shaped object.

    Attribute names mirror ``template_creation/orchestration/stage_params.py``
    (MappingStageParams, Ps1ProcessStageParams, DownsampleStageParams,
    WcsGroupingStageParams) and the ``.templates`` alias for ``downsample``.
    """
    mapping = SimpleNamespace(
        oversampling_factor=2,
        pad_distance=480,
        tess_buffer=150,
        edge_exclusion=10,
    )
    ps1_process = SimpleNamespace(
        psf_sigma=60.0,
        enable_saturation_correction=True,
        remove_saturated_stars=False,
        bright_star_mag_threshold=13.0,
    )
    downsample = SimpleNamespace(
        oversampling_factor=2,
        single_offset=False,
        ignore_mask_bits=[12],
        geometry_mode="field",
    )
    wcs_grouping = SimpleNamespace(
        offset_threshold=0.01,
        wcs_drift_savgol_window=11,
        wcs_drift_savgol_polyorder=2,
        crop_mode="full",
    )
    # ``.templates`` is a property alias for ``downsample`` on the real object.
    stages = SimpleNamespace(
        mapping=mapping,
        ps1_process=ps1_process,
        downsample=downsample,
        templates=downsample,
        wcs_grouping=wcs_grouping,
    )
    target = SimpleNamespace(sector=20, camera=1, ccd=1)
    resolved = SimpleNamespace(target=target, stages=stages)
    for k, v in overrides.items():
        setattr(resolved, k, v)
    return resolved


class TestSpatialKeys(unittest.TestCase):
    def test_skycell_key(self):
        self.assertEqual(
            model.skycell_spatial_key(1234, 56),
            {"projection": 1234, "skycell": 56},
        )

    def test_scc_key(self):
        self.assertEqual(
            model.scc_spatial_key(20, 1, 1, 2),
            {"sector": 20, "camera": 1, "ccd": 1, "oversampling": 2},
        )

    def test_scc_no_os_key(self):
        self.assertEqual(
            model.scc_no_os_spatial_key(20, 1, 1),
            {"sector": 20, "camera": 1, "ccd": 1},
        )


class TestRecipeParams(unittest.TestCase):
    def setUp(self):
        self.resolved = _fake_resolved()

    def test_kinds_registry(self):
        self.assertEqual(
            model.KINDS,
            frozenset(
                {
                    "ffi_set",
                    "raw_skycell",
                    "source_catalog",
                    "mapping",
                    "wcs_group",
                    "combined_skycell",
                    "convolved_skycell",
                    "scc_assembly",
                    "template",
                }
            ),
        )

    def test_mapping_params(self):
        self.assertEqual(
            model.recipe_params("mapping", self.resolved),
            {"oversampling_factor": 2, "pad_distance": 480, "tess_buffer": 150},
        )

    def test_combined_params(self):
        self.assertEqual(
            model.recipe_params("combined_skycell", self.resolved, gaia_version="dr3"),
            {
                "enable_saturation_correction": True,
                "remove_saturated_stars": False,
                "bright_star_mag_threshold": 13.0,
                "gaia_version": "dr3",
            },
        )

    def test_convolved_params(self):
        self.assertEqual(
            model.recipe_params("convolved_skycell", self.resolved),
            {"psf_sigma": 60.0, "radius": 470, "mode": "constant"},
        )

    def test_convolved_params_override(self):
        p = model.recipe_params("convolved_skycell", self.resolved, radius=200, mode="reflect")
        self.assertEqual(p["radius"], 200)
        self.assertEqual(p["mode"], "reflect")

    def test_template_params(self):
        self.assertEqual(
            model.recipe_params("template", self.resolved),
            {
                "oversampling_factor": 2,
                "single_offset": False,
                "ignore_mask_bits": [12],
                "geometry_mode": "field",
            },
        )

    def test_wcs_group_params(self):
        self.assertEqual(
            model.recipe_params("wcs_group", self.resolved),
            {
                "offset_threshold": 0.01,
                "wcs_drift_savgol_window": 11,
                "wcs_drift_savgol_polyorder": 2,
                "crop_mode": "full",
            },
        )

    def test_stub_builders(self):
        self.assertEqual(
            model.recipe_params("ffi_set", self.resolved),
            {"sector": 20, "camera": 1, "ccd": 1},
        )
        self.assertEqual(
            model.recipe_params("raw_skycell", self.resolved, version_token="tok"),
            {"version_token": "tok"},
        )
        self.assertEqual(
            model.recipe_params("source_catalog", self.resolved, gaia_version="dr3"),
            {"gaia_version": "dr3"},
        )
        self.assertEqual(
            model.recipe_params("scc_assembly", self.resolved),
            {"edge_exclusion": 10},
        )

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            model.recipe_params("nope", self.resolved)


class TestRecipeAndArtifact(unittest.TestCase):
    def setUp(self):
        self.resolved = _fake_resolved()

    def test_recipe_id_stable_and_length(self):
        r1 = model.build_recipe("convolved_skycell", self.resolved)
        r2 = model.build_recipe("convolved_skycell", self.resolved)
        self.assertEqual(r1.recipe_id(), r2.recipe_id())
        self.assertEqual(len(r1.recipe_id()), 16)

    def test_recipe_id_changes_with_param(self):
        r1 = model.build_recipe("convolved_skycell", self.resolved)
        r2 = model.build_recipe("convolved_skycell", self.resolved, radius=999)
        self.assertNotEqual(r1.recipe_id(), r2.recipe_id())

    def test_artifact_fingerprint_stable_and_length(self):
        recipe = model.build_recipe("convolved_skycell", self.resolved)
        sk = model.skycell_spatial_key(1234, 56)
        a1 = model.Artifact("convolved_skycell", sk, recipe, inputs=["aaa", "bbb"])
        a2 = model.Artifact("convolved_skycell", sk, recipe, inputs=["bbb", "aaa"])
        # Input ordering must not affect identity (sorted in fingerprint()).
        self.assertEqual(a1.fingerprint(), a2.fingerprint())
        self.assertEqual(len(a1.fingerprint()), 24)

    def test_merkle_input_change_changes_fingerprint(self):
        recipe = model.build_recipe("convolved_skycell", self.resolved)
        sk = model.skycell_spatial_key(1234, 56)
        a1 = model.Artifact("convolved_skycell", sk, recipe, inputs=["aaa"])
        a2 = model.Artifact("convolved_skycell", sk, recipe, inputs=["ccc"])
        self.assertNotEqual(a1.fingerprint(), a2.fingerprint())

    def test_merkle_recipe_change_changes_fingerprint(self):
        sk = model.skycell_spatial_key(1234, 56)
        r1 = model.build_recipe("convolved_skycell", self.resolved)
        r2 = model.build_recipe("convolved_skycell", self.resolved, radius=999)
        a1 = model.Artifact("convolved_skycell", sk, r1, inputs=["aaa"])
        a2 = model.Artifact("convolved_skycell", sk, r2, inputs=["aaa"])
        self.assertNotEqual(a1.fingerprint(), a2.fingerprint())

    def test_to_record_shape(self):
        recipe = model.build_recipe("combined_skycell", self.resolved, gaia_version="dr3")
        sk = model.skycell_spatial_key(1234, 56)
        art = model.Artifact(
            "combined_skycell",
            sk,
            recipe,
            inputs=["raw_fp", "cat_fp"],
            location="proj/cell/fp",
            state="complete",
            bytes=1024,
            wall_time_s=1.5,
            produced_by="host.pid",
        )
        rec = art.to_record()
        self.assertEqual(
            set(rec.keys()),
            {
                "fingerprint",
                "kind",
                "spatial_key",
                "recipe_id",
                "recipe",
                "inputs",
                "location",
                "state",
                "bytes",
                "wall_time_s",
                "produced_by",
            },
        )
        self.assertEqual(
            set(rec["recipe"].keys()),
            {"kind", "params", "code_version", "git_sha"},
        )
        self.assertEqual(rec["fingerprint"], art.fingerprint())
        self.assertEqual(rec["recipe_id"], recipe.recipe_id())
        self.assertEqual(rec["kind"], "combined_skycell")
        self.assertEqual(rec["inputs"], ["raw_fp", "cat_fp"])
        self.assertEqual(rec["location"], "proj/cell/fp")
        self.assertEqual(rec["state"], "complete")


if __name__ == "__main__":
    unittest.main()
