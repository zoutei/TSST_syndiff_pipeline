"""Tests for diff template handoff bootstrap and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from astropy.io import fits
import numpy as np

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.common.scc_paths import event_scc_leaf, resolve_scc_diff_bookkeeping_dir
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig
from syndiff_pipeline.difference_imaging.orchestration.execute import _load_template_handoff
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline


def _write_scc_diff_bookkeeping(data_root: Path, ref_fits: Path) -> None:
    """Seed ``bookkeeping/diff/oversampling_1/{diff_job.json,frames.csv}``.

    Mirrors the SCC-only handoff ``ensure_scc_diff_handoff`` reuses when a
    matching ``diff_job.json`` (schema_version >= 2) and ``frames.csv``
    already exist (`scc_bootstrap.py`) -- this is what
    ``_load_template_handoff``/``load_scc_diff_handoff_for_config`` now read
    instead of the legacy per-event ``event_job.json``/``frames.csv`` that
    ``bind`` used to write.
    """
    grid = MappingGrid(
        ffi_xmin=-8,
        ffi_ymin=-8,
        ffi_xmax=72,
        ffi_ymax=72,
        science_xmin_ffi=0,
        science_ymin_ffi=0,
        science_xmax_ffi=64,
        science_ymax_ffi=64,
    )
    bk_dir = resolve_scc_diff_bookkeeping_dir(
        data_root, 20, 3, 3, oversampling_factor=1, template_store_name=None
    )
    bk_dir.mkdir(parents=True, exist_ok=True)
    diff_job = {
        "schema_version": 2,
        "sector": 20,
        "camera": 3,
        "ccd": 3,
        "geometry_mode": "field",
        "mapping_grid": grid.to_mapping_dict(),
        "crop_bounds": grid.science_ffi_bounds(),
        "template_store_name": None,
        "output_store_name": None,
        "remap_store_name": None,
        "oversampling_factor": 1,
        "event_name": "2020ut",
    }
    (bk_dir / DIFF_JOB_BASENAME).write_text(json.dumps(diff_job), encoding="utf-8")
    pd.DataFrame(
        {
            "path": [str(ref_fits)],
            "ffi_basename": [ref_fits.name],
            "group_id": [0],
            "wcs_ok": [True],
        }
    ).to_csv(bk_dir / FRAMES_CSV_BASENAME, index=False)


class TestDiffHandoffBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.event_dir = event_scc_leaf(self.root, "2020ut", 20, 3, 3)
        ref = self.root / "ref.fits"
        data = np.zeros((64, 64), dtype=np.float32)
        hdu1 = fits.ImageHDU(data=data)
        hdu1.header["NAXIS1"] = 64
        hdu1.header["NAXIS2"] = 64
        hdu1.header["CRPIX1"] = 32.0
        hdu1.header["CRPIX2"] = 32.0
        hdu1.header["CRVAL1"] = 100.0
        hdu1.header["CRVAL2"] = 10.0
        hdu1.header["CDELT1"] = -0.01
        hdu1.header["CDELT2"] = 0.01
        hdu1.header["CTYPE1"] = "RA---TAN"
        hdu1.header["CTYPE2"] = "DEC--TAN"
        fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(ref, overwrite=True)
        self.ref_fits = ref
        _write_scc_diff_bookkeeping(self.data, ref)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_handoff_inherits_cluster_crop(self):
        cfg = SynDiffConfig(
            data_root=str(self.data),
            output_dir=str(self.event_dir),
            target_ra=100.0,
            target_dec=10.0,
            pipeline=[{"kind": "shared_mask"}],
        )
        wcs_table, crop, ref, thresh = _load_template_handoff(
            cfg, str(self.event_dir), None
        )
        self.assertEqual(len(wcs_table), 1)
        self.assertEqual(crop["x_max"], 64)
        self.assertEqual(ref, str(self.ref_fits))
        # SCC-only handoff no longer reads a per-job offset_threshold; it is
        # the fixed CLAUDE.md invariant (1 PS1 px quantization -> 0.01 TESS
        # px), hardcoded in load_scc_diff_handoff_for_config (scc_bootstrap.py).
        self.assertEqual(thresh, 0.01)

    def test_scc_handoff_crop_is_always_full_science_bounds(self):
        """SCC diff crop comes from the mapping grid, never from the target.

        The per-event ``crop_mode: target_box`` / ``crop_box_size`` knobs were
        dropped by the SCC-only storage refactor (041e996), which deleted the
        ``wcs_grouping.resolve_diff_crop_bounds(cfg, out)`` call from
        ``execute.py``. ``scc_bootstrap`` reads neither key -- both the linear
        and field paths set ``crop_bounds = mapping_grid.science_ffi_bounds()``
        unconditionally -- so the crop is a property of the SCC, not of whichever
        event happens to reference it. That is the intended behaviour under
        SCC-scoped diff; wave A-3 removed the keys themselves (``SynDiffConfig``
        no longer has ``crop_mode``/``crop_box_size`` at all).

        Regression guard: the handoff crop always equals the mapping grid's
        own ``science_ffi_bounds()`` -- there is no config knob left that
        could shrink or otherwise change it.
        """
        grid = MappingGrid(
            ffi_xmin=-8,
            ffi_ymin=-8,
            ffi_xmax=72,
            ffi_ymax=72,
            science_xmin_ffi=0,
            science_ymin_ffi=0,
            science_xmax_ffi=64,
            science_ymax_ffi=64,
        )
        cfg = SynDiffConfig(
            data_root=str(self.data),
            output_dir=str(self.event_dir),
            target_ra=100.0,
            target_dec=10.0,
            pipeline=[{"kind": "shared_mask"}],
        )
        _, crop, _, _ = _load_template_handoff(cfg, str(self.event_dir), None)

        expected = grid.science_ffi_bounds()
        self.assertEqual(tuple(crop["shape"]), tuple(expected["shape"]))
        for key in ("x_min", "x_max", "y_min", "y_max"):
            self.assertEqual(crop[key], expected[key])

    def test_missing_manifest_raises(self):
        cfg = SynDiffConfig(output_dir=str(self.root / "empty"), pipeline=[])
        with self.assertRaises(RuntimeError):
            _load_template_handoff(cfg, str(self.root / "empty"), None)

    def test_validate_rejects_bind_stage(self):
        cfg = SynDiffConfig(
            pipeline=[{"kind": "bind"}],
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pipeline(cfg)
        self.assertIn("unknown kind 'bind'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
