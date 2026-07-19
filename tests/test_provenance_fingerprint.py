"""Unit tests for the provenance fingerprint contract.

Golden bytes are pinned so any drift in canonical serialization (which would
silently change every fingerprint) is caught immediately.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.provenance.fingerprint import (
    RECIPE_SCHEMA_VERSION,
    canonical,
    code_version,
    fingerprint,
    git_sha,
    recipe_id,
)

# A representative nested dict exercising every normalization rule: sorted keys,
# bool, None, int, str, tuple/list order, integral floats (2.0 -> 2), -0.0 -> 0,
# and float rounding (1.2000000001 -> 1.2).
GOLDEN_OBJ = {
    "kind": "combined_skycell",
    "spatial_key": {"projection": 1234, "skycell": 56},
    "params": {
        "bright_star_mag_threshold": 15.0,
        "enable_saturation_correction": True,
        "gaia_version": "dr3",
        "psf_sigma": 1.2000000001,
        "radius": 470,
        "nested": [1, 2.0, -0.0, {"b": 3, "a": 4}],
        "maybe": None,
    },
}

GOLDEN_BYTES = (
    b'{"kind":"combined_skycell","params":{"bright_star_mag_threshold":15,'
    b'"enable_saturation_correction":true,"gaia_version":"dr3","maybe":null,'
    b'"nested":[1,2,0,{"a":4,"b":3}],"psf_sigma":1.2,"radius":470},'
    b'"spatial_key":{"projection":1234,"skycell":56}}'
)


class CanonicalTests(unittest.TestCase):
    def test_golden_bytes(self):
        self.assertEqual(canonical(GOLDEN_OBJ), GOLDEN_BYTES)

    def test_returns_bytes(self):
        self.assertIsInstance(canonical(GOLDEN_OBJ), bytes)

    def test_dict_key_order_invariant(self):
        a = {"a": 1, "b": 2, "c": {"x": 1, "y": 2}}
        b = {"c": {"y": 2, "x": 1}, "b": 2, "a": 1}
        self.assertEqual(canonical(a), canonical(b))

    def test_list_order_preserved(self):
        self.assertNotEqual(canonical([1, 2, 3]), canonical([3, 2, 1]))

    def test_tuple_serializes_like_list(self):
        self.assertEqual(canonical((1, 2, 3)), canonical([1, 2, 3]))
        self.assertEqual(
            canonical({"k": (1, "a", True)}), canonical({"k": [1, "a", True]})
        )

    def test_scalars(self):
        self.assertEqual(canonical(None), b"null")
        self.assertEqual(canonical(True), b"true")
        self.assertEqual(canonical(False), b"false")
        self.assertEqual(canonical(7), b"7")
        self.assertEqual(canonical("hi"), b'"hi"')


class FloatNormalizationTests(unittest.TestCase):
    def test_within_epsilon_identical(self):
        self.assertEqual(canonical(1.0), canonical(1.0000000001))
        self.assertEqual(canonical(1.0), canonical(1))
        self.assertEqual(
            canonical({"v": 3.14159265359}), canonical({"v": 3.141592653588})
        )

    def test_integral_float_collapses_to_int(self):
        self.assertEqual(canonical(2.0), b"2")
        self.assertEqual(canonical(2.0), canonical(2))

    def test_negative_zero_normalized(self):
        self.assertEqual(canonical(-0.0), canonical(0.0))
        self.assertEqual(canonical(-0.0), b"0")

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            canonical(float("nan"))
        with self.assertRaises(ValueError):
            canonical({"x": float("nan")})

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            canonical(float("inf"))
        with self.assertRaises(ValueError):
            canonical([float("-inf")])

    def test_genuine_float_preserved(self):
        self.assertEqual(canonical(1.5), b"1.5")


class CodeVersionTests(unittest.TestCase):
    def test_code_version(self):
        self.assertEqual(code_version(), str(RECIPE_SCHEMA_VERSION))
        self.assertEqual(code_version(), "1")

    def test_git_sha_never_raises(self):
        sha = git_sha()
        self.assertTrue(sha is None or isinstance(sha, str))


class RecipeIdTests(unittest.TestCase):
    def test_length_and_hex(self):
        rid = recipe_id("mapping", {"oversampling_factor": 2}, "1")
        self.assertEqual(len(rid), 16)
        int(rid, 16)  # raises if not hex

    def test_stable(self):
        params = {"a": 1, "b": 2.0}
        self.assertEqual(
            recipe_id("k", params, "1"), recipe_id("k", {"b": 2, "a": 1}, "1")
        )

    def test_param_change_changes_id(self):
        base = recipe_id("k", {"psf_sigma": 1.2}, "1")
        changed = recipe_id("k", {"psf_sigma": 1.3}, "1")
        self.assertNotEqual(base, changed)

    def test_code_version_change_changes_id(self):
        self.assertNotEqual(
            recipe_id("k", {"a": 1}, "1"), recipe_id("k", {"a": 1}, "2")
        )


class FingerprintTests(unittest.TestCase):
    def test_length_and_hex(self):
        fp = fingerprint("combined_skycell", {"projection": 1, "skycell": 2}, "abc", [])
        self.assertEqual(len(fp), 24)
        int(fp, 16)  # raises if not hex

    def test_input_order_invariant(self):
        sk = {"projection": 1, "skycell": 2}
        self.assertEqual(
            fingerprint("k", sk, "r", ["z", "a", "m"]),
            fingerprint("k", sk, "r", ["m", "a", "z"]),
        )

    def test_spatial_key_order_invariant(self):
        self.assertEqual(
            fingerprint("k", {"a": 1, "b": 2}, "r", []),
            fingerprint("k", {"b": 2, "a": 1}, "r", []),
        )

    def test_merkle_input_change_propagates(self):
        sk = {"projection": 1, "skycell": 2}
        base = fingerprint("k", sk, "r", ["input_a"])
        changed = fingerprint("k", sk, "r", ["input_b"])
        self.assertNotEqual(base, changed)

    def test_merkle_recipe_change_propagates(self):
        # Changing one param mints a new recipe_id, which changes the fingerprint.
        sk = {"projection": 1, "skycell": 2}
        rid1 = recipe_id("convolved_skycell", {"psf_sigma": 1.2}, "1")
        rid2 = recipe_id("convolved_skycell", {"psf_sigma": 1.3}, "1")
        self.assertNotEqual(
            fingerprint("convolved_skycell", sk, rid1, ["parent"]),
            fingerprint("convolved_skycell", sk, rid2, ["parent"]),
        )


if __name__ == "__main__":
    unittest.main()
