"""Unit tests for SCC field_templates sparse store + assemble."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from syndiff_pipeline.template_creation.processing.field_templates import (
    FieldManifest,
    assemble_group_from_contribs,
    contrib_basename,
    contrib_path,
    field_templates_root,
    load_contrib,
    parse_contrib_basename,
    verify_field_store,
    write_contrib,
    write_template_manifest,
)


class TestFieldTemplates(unittest.TestCase):
    def test_paths_and_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = field_templates_root(tmp, 20, 3, 3)
            self.assertTrue(
                str(root).endswith("s0020/c3/k3/templates/oversampling_1")
            )
            name = contrib_basename("skycell.2588.036", -2, 5)
            self.assertEqual(name, "skycell.2588.036_sx-2_sy+5.npz")
            self.assertEqual(parse_contrib_basename(name), ("skycell.2588.036", -2, 5))

    def test_write_assemble_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = field_templates_root(tmp, 20, 3, 3)
            ny, nx = 6, 8
            # two skycells contribute to overlapping flat indices
            write_contrib(
                root,
                "skycell.1.1",
                0,
                0,
                indices=np.array([0, 1, 8], dtype=np.int64),
                flux_sum=np.array([1.0, 2.0, 3.0]),
                count=np.array([1.0, 1.0, 1.0]),
                mask_count=np.array([0.0, 0.0, 1.0]),
            )
            write_contrib(
                root,
                "skycell.1.2",
                1,
                -1,
                indices=np.array([1, 9], dtype=np.int64),
                flux_sum=np.array([4.0, 5.0]),
                count=np.array([1.0, 1.0]),
            )
            write_template_manifest(
                root,
                FieldManifest(
                    geometry_mode="field",
                    scope="scc",
                    assembly="sparse_sum",
                    materialize_fits=False,
                    sector=20,
                    camera=3,
                    ccd=3,
                    contribs_dir="contribs",
                    groups=[{"group_id": 0, "n_frames": 2}],
                ),
            )
            out = assemble_group_from_contribs(
                root,
                [("skycell.1.1", 0, 0), ("skycell.1.2", 1, -1)],
                shape=(ny, nx),
                crop=(0, 4, 0, 2),
            )
            self.assertEqual(out["flux_sum"].shape, (2, 4))
            # flat index 1 = (0,1) gets 2+4=6
            self.assertAlmostEqual(float(out["flux_sum"][0, 1]), 6.0)
            v = verify_field_store(
                root,
                required_keys=[("skycell.1.1", 0, 0), ("skycell.1.2", 1, -1)],
            )
            self.assertTrue(v["ok"])

    def test_assemble_skip_count_and_mask(self):
        # Regression for the hotpants field-mode template loader hot path:
        # need_count=False/need_mask=False must still give the correct flux
        # sum, must not decompress the unused NPZ members, and must return
        # empty placeholders (not stale/zero-shaped-wrong) for the skipped
        # planes.
        with tempfile.TemporaryDirectory() as tmp:
            root = field_templates_root(tmp, 20, 3, 3)
            ny, nx = 6, 8
            write_contrib(
                root,
                "skycell.1.1",
                0,
                0,
                indices=np.array([0, 1, 8], dtype=np.int64),
                flux_sum=np.array([1.0, 2.0, 3.0]),
                count=np.array([1.0, 1.0, 1.0]),
                mask_count=np.array([0.0, 0.0, 1.0]),
            )
            write_contrib(
                root,
                "skycell.1.2",
                1,
                -1,
                indices=np.array([1, 9], dtype=np.int64),
                flux_sum=np.array([4.0, 5.0]),
                count=np.array([1.0, 1.0]),
            )
            write_template_manifest(
                root,
                FieldManifest(
                    geometry_mode="field",
                    scope="scc",
                    assembly="sparse_sum",
                    materialize_fits=False,
                    sector=20,
                    camera=3,
                    ccd=3,
                    contribs_dir="contribs",
                    groups=[{"group_id": 0, "n_frames": 2}],
                ),
            )
            out = assemble_group_from_contribs(
                root,
                [("skycell.1.1", 0, 0), ("skycell.1.2", 1, -1)],
                shape=(ny, nx),
                crop=(0, 4, 0, 2),
                need_count=False,
                need_mask=False,
            )
            self.assertEqual(out["flux_sum"].shape, (2, 4))
            self.assertAlmostEqual(float(out["flux_sum"][0, 1]), 6.0)
            self.assertEqual(out["count"].size, 0)
            self.assertEqual(out["mask_count"].size, 0)

    def test_load_contrib_keys_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = field_templates_root(tmp, 20, 3, 3)
            write_contrib(
                root,
                "skycell.1.1",
                0,
                0,
                indices=np.array([0, 1], dtype=np.int64),
                flux_sum=np.array([1.0, 2.0]),
                count=np.array([1.0, 1.0]),
                mask_count=np.array([0.0, 1.0]),
            )
            path = contrib_path(root, "skycell.1.1", 0, 0)
            full = load_contrib(path)
            self.assertEqual(set(full.keys()) & {"count", "mask_count"}, {"count", "mask_count"})
            subset = load_contrib(path, keys=["indices", "flux_sum"])
            self.assertEqual(set(subset.keys()), {"indices", "flux_sum"})

    def test_verify_missing_contrib(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = field_templates_root(tmp, 1, 1, 1)
            root.mkdir(parents=True)
            (root / "template_manifest.json").write_text("{}")
            (root / "contribs").mkdir()
            v = verify_field_store(root, required_keys=[("skycell.9.9", 0, 0)])
            self.assertFalse(v["ok"])


if __name__ == "__main__":
    unittest.main()
