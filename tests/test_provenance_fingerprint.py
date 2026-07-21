"""Golden + determinism tests for ``common/provenance/fingerprint.py``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.fingerprint import (
    RECIPE_SCHEMA_VERSION,
    canonical,
    fingerprint,
    recipe_id,
)


class TestCanonicalGolden(unittest.TestCase):
    """Exact byte-for-byte pins. Any change here is a breaking change to
    every fingerprint already on disk."""

    def test_dict_key_sort_and_float_round(self):
        out = canonical({"b": 1, "a": 2.0000000001, "c": [1, 2, 3]})
        self.assertEqual(out, b'{"a":2.0,"b":1,"c":[1,2,3]}')

    def test_negative_zero_normalized_and_tuple_becomes_list(self):
        out = canonical({"z": -0.0, "y": (1, 2, 3), "x": "hello"})
        self.assertEqual(out, b'{"x":"hello","y":[1,2,3],"z":0.0}')

    def test_nested_dict_and_float_precision(self):
        out = canonical({"nested": {"b": 2, "a": 1}, "floats": [1.23456789012, 0.1]})
        self.assertEqual(out, b'{"floats":[1.23456789,0.1],"nested":{"a":1,"b":2}}')

    def test_scalars(self):
        self.assertEqual(canonical(None), b"null")
        self.assertEqual(canonical(True), b"true")
        self.assertEqual(canonical(False), b"false")
        self.assertEqual(canonical(42), b"42")
        self.assertEqual(canonical(3.14159265358979), b"3.141592654")
        self.assertEqual(canonical([]), b"[]")
        self.assertEqual(canonical({}), b"{}")

    def test_nan_and_inf_rejected(self):
        with self.assertRaises(ValueError):
            canonical(float("nan"))
        with self.assertRaises(ValueError):
            canonical(float("inf"))
        with self.assertRaises(ValueError):
            canonical(float("-inf"))

    def test_bytes_rejected(self):
        with self.assertRaises(TypeError):
            canonical(b"raw bytes")

    def test_numpy_scalars_and_arrays_coerced(self):
        np = self._numpy()
        out = canonical({"x": np.float64(1.5), "y": np.int64(3), "z": np.array([1, 2, 3])})
        self.assertEqual(out, b'{"x":1.5,"y":3,"z":[1,2,3]}')

    @staticmethod
    def _numpy():
        import numpy as np

        return np


class TestRecipeIdAndFingerprintGolden(unittest.TestCase):
    def test_recipe_schema_version_is_two(self):
        self.assertEqual(RECIPE_SCHEMA_VERSION, 2)

    def test_recipe_id_pinned(self):
        rid = recipe_id(
            "mapping",
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
            RECIPE_SCHEMA_VERSION,
        )
        self.assertEqual(rid, "3a001d8636a2e0e5")
        self.assertEqual(len(rid), 16)

    def test_fingerprint_pinned(self):
        rid = recipe_id(
            "mapping",
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
            RECIPE_SCHEMA_VERSION,
        )
        fp = fingerprint("mapping", {"s": 20, "c": 1, "k": 1, "os": 2}, rid, ["aaa111", "bbb222"])
        self.assertEqual(fp, "fbd30ecff49511442462bd5b")
        self.assertEqual(len(fp), 24)


class TestDeterminism(unittest.TestCase):
    def test_recipe_id_independent_of_dict_key_order(self):
        rid_a = recipe_id("mapping", {"a": 1, "b": 2, "c": 3}, 1)
        rid_b = recipe_id("mapping", {"c": 3, "a": 1, "b": 2}, 1)
        self.assertEqual(rid_a, rid_b)

    def test_fingerprint_independent_of_input_order(self):
        rid = recipe_id("mapping", {"a": 1}, 1)
        fp_a = fingerprint("mapping", {"s": 1, "c": 1, "k": 1}, rid, ["x", "y", "z"])
        fp_b = fingerprint("mapping", {"s": 1, "c": 1, "k": 1}, rid, ["z", "x", "y"])
        self.assertEqual(fp_a, fp_b)

    def test_fingerprint_independent_of_spatial_key_dict_order(self):
        rid = recipe_id("mapping", {"a": 1}, 1)
        fp_a = fingerprint("mapping", {"s": 1, "c": 2, "k": 3}, rid, [])
        fp_b = fingerprint("mapping", {"k": 3, "s": 1, "c": 2}, rid, [])
        self.assertEqual(fp_a, fp_b)

    def test_repeated_calls_are_stable(self):
        params = {"oversampling_factor": 2, "pad_distance": 480}
        rid1 = recipe_id("mapping", params, 1)
        rid2 = recipe_id("mapping", params, 1)
        self.assertEqual(rid1, rid2)


class TestMerkleInvalidation(unittest.TestCase):
    """Flipping one param must change the recipe_id (and thus the
    fingerprint) for the affected kind, and must NOT change a sibling
    artifact's fingerprint that doesn't depend on it -- "invalidation is
    automatic and exact" (§3)."""

    def test_flip_one_param_changes_recipe_id(self):
        base = {"oversampling_factor": 2, "pad_distance": 480, "overwrite": True}
        flipped = {**base, "pad_distance": 481}
        self.assertNotEqual(
            recipe_id("mapping", base, 1),
            recipe_id("mapping", flipped, 1),
        )

    def test_flip_one_param_changes_fingerprint_and_downstream_only(self):
        # Upstream artifact "mapping" at two param variants.
        mapping_params_a = {"oversampling_factor": 2, "pad_distance": 480}
        mapping_params_b = {"oversampling_factor": 2, "pad_distance": 481}
        rid_a = recipe_id("mapping", mapping_params_a, 1)
        rid_b = recipe_id("mapping", mapping_params_b, 1)
        spatial = {"s": 20, "c": 1, "k": 1, "os": 2}
        fp_mapping_a = fingerprint("mapping", spatial, rid_a, [])
        fp_mapping_b = fingerprint("mapping", spatial, rid_b, [])
        self.assertNotEqual(fp_mapping_a, fp_mapping_b)

        # A downstream artifact ("remap_store") that consumes mapping's
        # fingerprint as an input must also change when mapping changes...
        remap_rid = recipe_id("remap_store", {"keying": "absolute"}, 1)
        fp_remap_a = fingerprint("remap_store", spatial, remap_rid, [fp_mapping_a])
        fp_remap_b = fingerprint("remap_store", spatial, remap_rid, [fp_mapping_b])
        self.assertNotEqual(fp_remap_a, fp_remap_b)

        # ...but a sibling artifact that does NOT consume mapping at all
        # (e.g. an unrelated raw_skycell input node) must be completely
        # unaffected by the mapping param flip -- "exactly the downstream
        # cone" changes, nothing else.
        sibling_rid = recipe_id("raw_skycell", {}, 1)
        sibling_spatial = {"projection": "skycell1234", "skycell": "2001"}
        fp_sibling_before = fingerprint("raw_skycell", sibling_spatial, sibling_rid, [])
        fp_sibling_after = fingerprint("raw_skycell", sibling_spatial, sibling_rid, [])
        self.assertEqual(fp_sibling_before, fp_sibling_after)
        self.assertNotIn(fp_sibling_before, (fp_mapping_a, fp_mapping_b, fp_remap_a, fp_remap_b))

    def test_same_params_different_kind_differ(self):
        # Same params/code_version but a different kind string must not
        # collide -- kind is part of the hashed payload.
        params = {"a": 1}
        rid_mapping = recipe_id("mapping", params, 1)
        rid_downsample = recipe_id("downsample", params, 1)
        self.assertNotEqual(rid_mapping, rid_downsample)

    def test_code_version_bump_changes_recipe_id(self):
        params = {"a": 1}
        rid_v1 = recipe_id("mapping", params, 1)
        rid_v2 = recipe_id("mapping", params, 2)
        self.assertNotEqual(rid_v1, rid_v2)


if __name__ == "__main__":
    unittest.main()
