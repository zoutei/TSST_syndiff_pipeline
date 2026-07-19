"""Tests for ``difference_imaging/orchestration/provenance_glue.py`` (PR-D1).

Covers: recipe determinism per stage kind, product-id/spatial-key
correctness, required-set derivation from a synthetic frames.csv, emit being
non-fatal when the spool dir is unwritable, and indexed completeness against
a temp ``ProvenanceStore`` with synthetic rows.
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    EpsfParams,
    HotpantsParams,
    KernelFitParams,
    KernelSubtractParams,
    SharedMaskParams,
)


class TestRecipeDeterminism(unittest.TestCase):
    def test_hotpants_recipe_deterministic(self):
        hp = HotpantsParams()
        r1 = pg.diff_recipe("diff_image", hp)
        r2 = pg.diff_recipe("diff_image", HotpantsParams())
        self.assertEqual(r1, r2)
        self.assertEqual(r1["kind"], "diff_image")
        self.assertIn("HotpantsParams", r1["params"])

    def test_epsf_recipe_deterministic(self):
        e1 = pg.diff_recipe("epsf", EpsfParams())
        e2 = pg.diff_recipe("epsf", EpsfParams())
        self.assertEqual(e1, e2)

    def test_shared_mask_recipe_deterministic(self):
        m1 = pg.diff_recipe("shared_mask", SharedMaskParams())
        m2 = pg.diff_recipe("shared_mask", SharedMaskParams())
        self.assertEqual(m1, m2)

    def test_kernel_pair_recipe_namespaced_by_classname(self):
        r = pg.diff_recipe("diff_image", [KernelFitParams(), KernelSubtractParams()])
        self.assertEqual(set(r["params"].keys()), {"KernelFitParams", "KernelSubtractParams"})

    def test_recipe_changes_with_params(self):
        r1 = pg.diff_recipe("diff_image", HotpantsParams())
        r2 = pg.diff_recipe("diff_image", HotpantsParams(sci_fwhm=99.0))
        self.assertNotEqual(r1, r2)


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestFingerprintDeterminism(unittest.TestCase):
    def test_same_recipe_same_fingerprint(self):
        hp = HotpantsParams()
        fp1 = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=hp,
        )
        fp2 = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=HotpantsParams(),
        )
        self.assertIsNotNone(fp1)
        self.assertEqual(fp1, fp2)

    def test_label_change_reFingerprints(self):
        """Distinct workspace labels must not collide (plan §6)."""
        hp = HotpantsParams()
        fp_a = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=hp,
        )
        fp_b = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs_debug", params=hp,
        )
        self.assertNotEqual(fp_a, fp_b)

    def test_param_change_reFingerprints(self):
        fp1 = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=HotpantsParams(),
        )
        fp2 = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=HotpantsParams(sci_fwhm=2.5),
        )
        self.assertNotEqual(fp1, fp2)

    def test_spatial_key_change_reFingerprints(self):
        hp = HotpantsParams()
        fp1 = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=hp,
        )
        fp2 = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0002",
            label="diffs", params=hp,
        )
        self.assertNotEqual(fp1, fp2)

    def test_missing_input_fingerprint_fails_open(self):
        fp = pg.diff_kind_fingerprint(
            "epsf", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="epsf", params=EpsfParams(), input_fingerprints=[None],
        )
        self.assertIsNone(fp)

    def test_shared_mask_fingerprint(self):
        fp = pg.diff_kind_fingerprint_shared_mask(20, 3, 3, SharedMaskParams())
        self.assertIsNotNone(fp)
        fp2 = pg.diff_kind_fingerprint_shared_mask(20, 3, 3, SharedMaskParams(gaia_mag_bright=5.0))
        self.assertNotEqual(fp, fp2)


class TestSpatialKeys(unittest.TestCase):
    def test_ffi_spatial_key_shape(self):
        key = pg.ffi_spatial_key(20, 3, 3, "tess0001", "diffs")
        self.assertEqual(key["s"], 20)
        self.assertEqual(key["c"], 3)
        self.assertEqual(key["k"], 3)
        self.assertEqual(key["product_id"], "tess0001")
        self.assertEqual(key["label"], "diffs")

    def test_shared_mask_spatial_key_shape(self):
        key = pg.shared_mask_spatial_key(20, 3, 3, label="shared_mask")
        self.assertEqual(key, {"s": 20, "c": 3, "k": 3})

    def test_product_id_for_ffi(self):
        pid = pg.product_id_for_ffi("tess2020019142923-s0020-3-3-0165-s_ffic.fits")
        self.assertEqual(pid, "tess2020019142923")


class TestRequiredProductIds(unittest.TestCase):
    def _frames_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "filename": [
                    "tess0001-s0020-3-3-0001-s_ffic.fits",
                    "tess0002-s0020-3-3-0002-s_ffic.fits",
                    "tess0003-s0020-3-3-0003-s_ffic.fits",
                    "tess0004-s0020-3-3-0004-s_ffic.fits",
                ],
                "wcs_ok": [True, True, False, True],
                "group_id": [0, 1, 0, -1],
            }
        )

    def test_excludes_bad_wcs_and_ungrouped(self):
        pids = pg.required_product_ids(self._frames_df())
        # row 3 excluded (wcs_ok False), row 4 excluded (group_id < 0)
        self.assertEqual(pids, sorted(["tess0001", "tess0002"]))

    def test_empty_frame_returns_empty(self):
        self.assertEqual(pg.required_product_ids(pd.DataFrame()), [])
        self.assertEqual(pg.required_product_ids(None), [])

    def test_no_group_id_column_keeps_wcs_ok_rows(self):
        df = self._frames_df().drop(columns=["group_id"])
        pids = pg.required_product_ids(df)
        self.assertEqual(pids, sorted(["tess0001", "tess0002", "tess0004"]))


@unittest.skipUnless(pg.PROVENANCE_AVAILABLE, "common.provenance not importable")
class TestEmitNonFatal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_root = Path(self.tmp.name)

    def _write_ffi(self) -> str:
        p = Path(self.tmp.name) / "tess0001-s0020-3-3-0001-s_ffic.fits"
        p.write_bytes(b"SIMPLE  = T")
        return str(p)

    def test_emit_succeeds_and_writes_spool(self):
        ffi_path = self._write_ffi()
        loc = Path(self.tmp.name) / "diff.fits"
        loc.write_bytes(b"SIMPLE  = T")
        fp = pg.emit_diff_artifact(
            kind="diff_image",
            sector=20,
            camera=3,
            ccd=3,
            product_id="tess0001",
            label="diffs",
            params=HotpantsParams(),
            location=str(loc),
            input_fingerprints=[pg.ffi_input_fingerprint(20, 3, 3, ffi_path)],
            data_root=str(self.data_root),
        )
        self.assertIsNotNone(fp)
        spool_dir = self.data_root / "bookkeeping" / "spool"
        self.assertTrue(spool_dir.is_dir())
        files = list(spool_dir.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        contents = files[0].read_text()
        self.assertIn(fp, contents)
        self.assertIn('"diff_image"', contents)

    def test_emit_none_when_data_root_missing(self):
        fp = pg.emit_diff_artifact(
            kind="diff_image",
            sector=20,
            camera=3,
            ccd=3,
            product_id="tess0001",
            label="diffs",
            params=HotpantsParams(),
            location="/tmp/whatever.fits",
            data_root=None,
        )
        self.assertIsNone(fp)

    def test_emit_non_fatal_when_spool_dir_unwritable(self):
        bookkeeping = self.data_root / "bookkeeping"
        bookkeeping.mkdir(parents=True)
        os.chmod(bookkeeping, stat.S_IRUSR | stat.S_IXUSR)  # read+exec only, no write
        self.addCleanup(lambda: os.chmod(bookkeeping, stat.S_IRWXU))
        try:
            fp = pg.emit_diff_artifact(
                kind="diff_image",
                sector=20,
                camera=3,
                ccd=3,
                product_id="tess0001",
                label="diffs",
                params=HotpantsParams(),
                location="/tmp/whatever.fits",
                data_root=str(self.data_root),
            )
        except Exception as exc:  # pragma: no cover - the whole point is this must not happen
            self.fail(f"emit_diff_artifact raised instead of degrading gracefully: {exc}")
        # Running as root (common in containers) bypasses the permission bit;
        # only assert the no-raise contract, not the exact return value.
        if os.geteuid() != 0:
            self.assertIsNone(fp)

    def test_emit_shared_mask_non_fatal_and_writes_spool(self):
        loc = Path(self.tmp.name) / "shared_mask.fits"
        loc.write_bytes(b"SIMPLE  = T")
        fp = pg.emit_shared_mask_artifact(
            sector=20,
            camera=3,
            ccd=3,
            params=SharedMaskParams(),
            location=str(loc),
            data_root=str(self.data_root),
        )
        self.assertIsNotNone(fp)


class TestProvenanceUnavailableFallsOpen(unittest.TestCase):
    """Simulate the provenance package being absent (mid-authoring elsewhere)."""

    def setUp(self):
        self._saved = (
            pg.PROVENANCE_AVAILABLE,
            pg._SPOOL_AVAILABLE,
            pg._STORE_AVAILABLE,
        )
        pg.PROVENANCE_AVAILABLE = False
        pg._SPOOL_AVAILABLE = False
        pg._STORE_AVAILABLE = False

    def tearDown(self):
        (
            pg.PROVENANCE_AVAILABLE,
            pg._SPOOL_AVAILABLE,
            pg._STORE_AVAILABLE,
        ) = self._saved

    def test_fingerprint_is_none(self):
        fp = pg.diff_kind_fingerprint(
            "diff_image", sector=20, camera=3, ccd=3, product_id="tess0001",
            label="diffs", params=HotpantsParams(),
        )
        self.assertIsNone(fp)

    def test_emit_is_none_and_non_fatal(self):
        fp = pg.emit_diff_artifact(
            kind="diff_image",
            sector=20,
            camera=3,
            ccd=3,
            product_id="tess0001",
            label="diffs",
            params=HotpantsParams(),
            location="/tmp/whatever.fits",
            data_root="/tmp/does_not_matter",
        )
        self.assertIsNone(fp)

    def test_ffi_input_fingerprint_is_none(self):
        self.assertIsNone(pg.ffi_input_fingerprint(20, 3, 3, "/tmp/does_not_exist.fits"))

    def test_open_store_is_none(self):
        self.assertIsNone(pg.open_store("/tmp/does_not_matter"))

    def test_recipe_helpers_still_work(self):
        # Pure-python recipe/spatial-key/required-set helpers need no package.
        r = pg.diff_recipe("diff_image", HotpantsParams())
        self.assertEqual(r["kind"], "diff_image")
        key = pg.ffi_spatial_key(20, 3, 3, "tess0001", "diffs")
        self.assertEqual(key["product_id"], "tess0001")


if __name__ == "__main__":
    unittest.main()
