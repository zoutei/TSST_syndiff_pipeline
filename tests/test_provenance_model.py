"""Tests for ``common/provenance/model.py``: spatial keys, kind registry,
and recipe_params builders."""

from __future__ import annotations

import sys
import tempfile
import unittest
from types import SimpleNamespace

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.common.provenance.fingerprint import recipe_id


def _fake_resolved(**overrides):
    """Minimal duck-typed stand-in for ResolvedTargetConfig -- only the
    attributes the recipe_params builders actually read."""
    mapping = SimpleNamespace(
        oversampling_factor=2,
        pad_distance=480,
        overwrite=True,
        x_left_dead=44,
        x_right_dead=44,
        y_edge_strip=30,
        sci_fwhm=1.88,
        template_conv_pad_spare_px=4,
    )
    ps1_process = SimpleNamespace(
        projections_limit=None,
        psf_sigma=60.0,
        enable_saturation_correction=True,
        remove_saturated_stars=False,
        bright_star_mag_threshold=13.0,
    )
    remap = SimpleNamespace(
        cache_quantum_ps1_px=1.0,
        keying="absolute",
        intra_skycell_R=1,
        store_name=None,
    )
    downsample = SimpleNamespace(
        oversampling_factor=2,
        single_offset=False,
        ignore_mask_bits=[12],
        output_base=None,
        output_store_name=None,
        apply_intra_skycell=True,
        apply_inter_skycell=True,
    )
    ps1_download = SimpleNamespace(overwrite=False, use_local_files=False)
    stages = SimpleNamespace(
        mapping=mapping,
        ps1_process=ps1_process,
        remap=remap,
        downsample=downsample,
        ps1_download=ps1_download,
    )
    resolved = SimpleNamespace(stages=stages, template_output_base="templates", downsample_remap_store_name=None)
    for k, v in overrides.items():
        setattr(resolved, k, v)
    return resolved


class TestSpatialKeys(unittest.TestCase):
    def test_skycell_key_to_dict(self):
        k = model.SkycellKey("skycell1234", "2001")
        self.assertEqual(k.to_dict(), {"projection": "skycell1234", "skycell": "2001"})

    def test_skycell_key_rejects_empty(self):
        with self.assertRaises(ValueError):
            model.SkycellKey("", "2001")

    def test_scc_key_optional_os(self):
        self.assertEqual(model.SccKey(20, 1, 1).to_dict(), {"s": 20, "c": 1, "k": 1})
        self.assertEqual(
            model.SccKey(20, 1, 1, os=2).to_dict(), {"s": 20, "c": 1, "k": 1, "os": 2}
        )

    def test_scc_key_rejects_negative(self):
        with self.assertRaises(ValueError):
            model.SccKey(-1, 1, 1)

    def test_scc_key_rejects_non_positive_os(self):
        with self.assertRaises(ValueError):
            model.SccKey(20, 1, 1, os=0)

    def test_scc_key_optional_store_name(self):
        self.assertEqual(
            model.SccKey(20, 1, 1, os=2, store_name="hybrid_r2").to_dict(),
            {"s": 20, "c": 1, "k": 1, "os": 2, "store_name": "hybrid_r2"},
        )

    def test_scc_ffi_key(self):
        k = model.SccFfiKey(20, 1, 1, "tess2020019142923")
        self.assertEqual(
            k.to_dict(), {"s": 20, "c": 1, "k": 1, "product_id": "tess2020019142923"}
        )

    def test_scc_ffi_key_rejects_empty_product_id(self):
        with self.assertRaises(ValueError):
            model.SccFfiKey(20, 1, 1, "")

    def test_scc_ffi_key_includes_label_when_set(self):
        k = model.SccFfiKey(20, 1, 1, "tess2020019142923", label="hp_d")
        self.assertEqual(
            k.to_dict(),
            {"s": 20, "c": 1, "k": 1, "product_id": "tess2020019142923", "label": "hp_d"},
        )

    def test_diff_labels_yield_distinct_fingerprints(self):
        from syndiff_pipeline.common.provenance.fingerprint import fingerprint

        base = {"s": 20, "c": 3, "k": 3, "product_id": "tess2020019142923"}
        recipe = {"kind": "diff_image", "params": {"HotpantsParams": {"nrx": 1}}, "code_version": 1}
        fp_a = fingerprint("diff_image", {**base, "label": "hp_d"}, recipe, ())
        fp_b = fingerprint("diff_image", {**base, "label": "kn_d"}, recipe, ())
        self.assertNotEqual(fp_a, fp_b)

    def test_event_key(self):
        k = model.EventKey("2020dgc", 20, 1, 1)
        self.assertEqual(k.to_dict(), {"event": "2020dgc", "s": 20, "c": 1, "k": 1})

    def test_event_key_rejects_empty_event(self):
        with self.assertRaises(ValueError):
            model.EventKey("", 20, 1, 1)


class TestKindRegistry(unittest.TestCase):
    def test_all_fifteen_kinds_present(self):
        self.assertEqual(len(model.ALL_KINDS), 15)
        self.assertEqual(set(model.KIND_REGISTRY.keys()), set(model.ALL_KINDS))

    def test_template_and_diff_kinds_partition(self):
        self.assertEqual(len(model.TEMPLATE_KINDS), 10)
        self.assertEqual(len(model.DIFF_KINDS), 5)
        self.assertEqual(set(model.TEMPLATE_KINDS) & set(model.DIFF_KINDS), set())

    def test_spatial_key_kinds_are_valid(self):
        for kind, spec in model.KIND_REGISTRY.items():
            self.assertIn(
                spec.spatial_key_kind, model.SPATIAL_KEY_KINDS, msg=f"kind={kind}"
            )

    def test_legacy_unverified_kind_naming(self):
        self.assertEqual(model.legacy_unverified_kind("mapping"), "mapping_legacy_unverified")
        # Idempotent: applying twice does not double-suffix.
        self.assertEqual(
            model.legacy_unverified_kind(model.legacy_unverified_kind("mapping")),
            "mapping_legacy_unverified",
        )


class TestTemplateRecipeParamsBuilders(unittest.TestCase):
    def test_mapping_recipe_params_matches_verify_py_fields(self):
        resolved = _fake_resolved()
        params = model.mapping_recipe_params(resolved)
        self.assertEqual(
            params,
            {
                "oversampling_factor": 2,
                "pad_distance": 480,
                "overwrite": True,
                "mapping_grid": {
                    "x_left_dead": 44,
                    "x_right_dead": 44,
                    "y_edge_strip": 30,
                    "conv_pad_native": 8,
                    "oversampling_factor": 2,
                },
            },
        )

    def test_ps1_process_recipe_params(self):
        resolved = _fake_resolved()
        params = model.ps1_process_recipe_params(resolved)
        self.assertEqual(
            params,
            {
                "projections_limit": None,
                "psf_sigma": 60.0,
                "enable_saturation_correction": True,
                "remove_saturated_stars": False,
                "bright_star_mag_threshold": 13.0,
            },
        )

    def test_remap_store_recipe_params(self):
        resolved = _fake_resolved()
        params = model.remap_store_recipe_params(resolved)
        self.assertEqual(
            params,
            {
                "oversampling_factor": 2,
                "cache_quantum_ps1_px": 1.0,
                "keying": "absolute",
                "intra_skycell_R": 1,
                "store_name": "",
                "mapping_grid": {
                    "x_left_dead": 44,
                    "x_right_dead": 44,
                    "y_edge_strip": 30,
                    "conv_pad_native": 8,
                    "oversampling_factor": 2,
                },
            },
        )

    def test_downsample_recipe_params_falls_back_to_template_output_base(self):
        resolved = _fake_resolved()
        params = model.downsample_recipe_params(resolved)
        self.assertEqual(params["output_base"], "templates")
        self.assertEqual(params["ignore_mask_bits"], [12])
        self.assertEqual(params["output_store_name"], "")
        self.assertEqual(params["remap_store_name"], "")
        self.assertTrue(params["apply_intra_skycell"])
        self.assertTrue(params["apply_inter_skycell"])
        self.assertEqual(params["mapping_grid"]["oversampling_factor"], 2)

    def test_downsample_recipe_params_prefers_stage_output_base(self):
        resolved = _fake_resolved()
        resolved.stages.downsample.output_base = "custom_base"
        params = model.downsample_recipe_params(resolved)
        self.assertEqual(params["output_base"], "custom_base")

    def test_ps1_download_recipe_params(self):
        resolved = _fake_resolved()
        self.assertEqual(
            model.ps1_download_recipe_params(resolved),
            {"overwrite": False, "use_local_files": False},
        )

    def test_ffi_set_and_raw_skycell_are_empty(self):
        self.assertEqual(model.ffi_set_recipe_params(), {})
        self.assertEqual(model.raw_skycell_recipe_params(), {})

    def test_source_catalog_recipe_params_default_and_override(self):
        self.assertEqual(model.source_catalog_recipe_params(), {"gaia_version": "dr3"})
        self.assertEqual(
            model.source_catalog_recipe_params(gaia_version="dr2", mag_threshold=13.0),
            {"gaia_version": "dr2", "mag_threshold": 13.0},
        )

    def test_combined_skycell_recipe_params_folds_in_gaia_version(self):
        resolved = _fake_resolved()
        params = model.combined_skycell_recipe_params(resolved, gaia_version="dr3")
        self.assertEqual(params["gaia_version"], "dr3")
        self.assertEqual(params["psf_sigma"], 60.0)

    def test_convolved_skycell_recipe_params(self):
        resolved = _fake_resolved()
        params = model.convolved_skycell_recipe_params(resolved)
        self.assertEqual(params["psf_sigma"], 60.0)
        self.assertEqual(params["padding"], "same_projection_only")

    def test_scc_assembly_matches_ps1_process_plus_mapping_grid(self):
        resolved = _fake_resolved()
        asm = model.scc_assembly_recipe_params(resolved)
        proc = model.ps1_process_recipe_params(resolved)
        self.assertEqual({k: v for k, v in asm.items() if k != "mapping_grid"}, proc)
        self.assertIn("mapping_grid", asm)

    def test_ignore_mask_bits_is_a_plain_list_not_the_original_object(self):
        resolved = _fake_resolved()
        original = resolved.stages.downsample.ignore_mask_bits
        params = model.downsample_recipe_params(resolved)
        self.assertEqual(params["ignore_mask_bits"], original)
        self.assertIsNot(params["ignore_mask_bits"], original)


class TestDiffRecipeParamsBuilders(unittest.TestCase):
    def test_shared_mask_recipe_params_from_dataclass(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            SharedMaskParams,
        )
        from syndiff_pipeline.difference_imaging.masking.settings import (
            MaskSettings,
            SharedMaskSettings,
        )

        p = SharedMaskParams(ref_mag_min=12.5)
        params = model.shared_mask_recipe_params(p)
        self.assertEqual(params["ref_mag_min"], 12.5)
        self.assertIn("mask_settings", params)
        self.assertIn("shared", params["mask_settings"])
        self.assertNotIn("mask_settings", params.get("mask_settings", {}))

        custom = MaskSettings(shared=SharedMaskSettings(bright_maglim=11.0))
        params2 = model.shared_mask_recipe_params(p, mask_settings=custom)
        self.assertEqual(params2["mask_settings"]["shared"]["bright_maglim"], 11.0)
        self.assertNotEqual(params["mask_settings"], params2["mask_settings"])

    def test_shared_mask_recipe_params_loads_yaml_path_on_params(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            SharedMaskParams,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            yaml_a = tmp_path / "mask_a.yaml"
            yaml_b = tmp_path / "mask_b.yaml"
            yaml_a.write_text("shared:\n  bright_maglim: 11.5\n", encoding="utf-8")
            yaml_b.write_text("shared:\n  bright_maglim: 12.5\n", encoding="utf-8")

            p = SharedMaskParams(mask_settings=str(yaml_a))
            recipe_a = model.shared_mask_recipe_params(p)
            self.assertEqual(recipe_a["mask_settings"]["shared"]["bright_maglim"], 11.5)
            self.assertNotIn("mask_settings", {k for k in recipe_a if k != "mask_settings"})
            for value in recipe_a.values():
                if isinstance(value, str) and value.endswith(".yaml"):
                    self.fail(f"recipe contains raw path string: {value!r}")

            p_b = SharedMaskParams(mask_settings=str(yaml_b))
            recipe_b = model.shared_mask_recipe_params(p_b)
            self.assertNotEqual(
                recipe_id("shared_mask", recipe_a, 2),
                recipe_id("shared_mask", recipe_b, 2),
            )

    def test_shared_mask_recipe_params_same_yaml_contents_same_recipe(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            SharedMaskParams,
        )

        contents = "shared:\n  bright_maglim: 10.0\n  strapsize: 7\n"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path_one = tmp_path / "one" / "mask_settings.yaml"
            path_two = tmp_path / "two" / "mask_settings.yaml"
            path_one.parent.mkdir(parents=True)
            path_two.parent.mkdir(parents=True)
            path_one.write_text(contents, encoding="utf-8")
            path_two.write_text(contents, encoding="utf-8")

            recipe_one = model.shared_mask_recipe_params(
                SharedMaskParams(mask_settings=str(path_one))
            )
            recipe_two = model.shared_mask_recipe_params(
                SharedMaskParams(mask_settings=str(path_two))
            )
            self.assertEqual(recipe_one["mask_settings"], recipe_two["mask_settings"])
            self.assertEqual(
                recipe_id("shared_mask", recipe_one, 2),
                recipe_id("shared_mask", recipe_two, 2),
            )

    def test_shared_mask_recipe_params_explicit_kwarg_overrides_path(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            SharedMaskParams,
        )
        from syndiff_pipeline.difference_imaging.masking.settings import (
            MaskSettings,
            SharedMaskSettings,
        )

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "mask_settings.yaml"
            yaml_path.write_text("shared:\n  bright_maglim: 9.0\n", encoding="utf-8")
            custom = MaskSettings(shared=SharedMaskSettings(bright_maglim=14.0))
            recipe = model.shared_mask_recipe_params(
                SharedMaskParams(mask_settings=str(yaml_path)),
                mask_settings=custom,
            )
            self.assertEqual(recipe["mask_settings"]["shared"]["bright_maglim"], 14.0)

    def test_mapping_grid_recipe_fragment_converts_ffi_block_with_nx_ny(self):
        from syndiff_pipeline.common.mapping_grid import MappingGrid

        grid = MappingGrid.from_ffi_shape(2048, 2048, oversampling=2)
        block = {**grid.to_mapping_dict(), "nx": 2048, "ny": 2048}
        self.assertEqual(
            model.mapping_grid_recipe_fragment(block),
            {
                "x_left_dead": 44,
                "x_right_dead": 44,
                "y_edge_strip": 30,
                "conv_pad_native": 8,
                "oversampling_factor": 2,
            },
        )

    def test_epsf_recipe_params_from_dataclass(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import EpsfParams

        p = EpsfParams(tile_nx=5)
        params = model.epsf_recipe_params(p)
        self.assertEqual(params["tile_nx"], 5)

    def test_diff_image_recipe_params_single_hotpants(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams

        p = HotpantsParams(sci_fwhm=2.0)
        params = model.diff_image_recipe_params(p)
        self.assertEqual(set(params.keys()), {"HotpantsParams"})
        self.assertEqual(params["HotpantsParams"]["sci_fwhm"], 2.0)

    def test_diff_image_recipe_params_kernel_fit_and_subtract_merge(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            KernelFitParams,
            KernelSubtractParams,
        )

        fit = KernelFitParams(weighting_factor=0.7)
        sub = KernelSubtractParams(phot_box_size=6)
        params = model.diff_image_recipe_params(fit, sub)
        self.assertEqual(set(params.keys()), {"KernelFitParams", "KernelSubtractParams"})
        self.assertEqual(params["KernelFitParams"]["weighting_factor"], 0.7)
        self.assertEqual(params["KernelSubtractParams"]["phot_box_size"], 6)

    def test_diff_image_recipe_params_requires_at_least_one(self):
        with self.assertRaises(ValueError):
            model.diff_image_recipe_params()

    def test_photometry_recipe_params(self):
        from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
            AperturePhotometryMethodParams,
        )

        p = AperturePhotometryMethodParams(name="ap1", tar_ap=3.0)
        params = model.photometry_recipe_params(p)
        self.assertEqual(params["name"], "ap1")
        self.assertEqual(params["tar_ap"], 3.0)

    def test_recipe_params_reject_non_dataclass_non_mapping(self):
        with self.assertRaises(TypeError):
            model.shared_mask_recipe_params(object())


if __name__ == "__main__":
    unittest.main()
