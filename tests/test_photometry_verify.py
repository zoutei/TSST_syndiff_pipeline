"""Tests for photometry orchestration verify gates."""

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
from syndiff_pipeline.common.scc_paths import (
    event_scc_leaf,
    scc_diff_dir,
    scc_diff_label_dir,
)
from syndiff_pipeline.difference_imaging.orchestration.scc_bootstrap import (
    DIFF_JOB_BASENAME,
    FRAMES_CSV_BASENAME,
)
from syndiff_pipeline.difference_imaging.stages.photometry import lightcurve_csv_basename
from syndiff_pipeline.difference_imaging.support.paths import photometry_root
from syndiff_pipeline.photometry.orchestration.verify import (
    collect_photometry_artifacts,
    photometry_complete,
    scc_diff_lane_complete,
    verify_photometry_prerequisites,
)
from syndiff_pipeline.photometry.site_config import (
    PhotometryRunConfig,
    PhotometrySitePolicy,
    resolve_photometry_run_config,
)
from tests.site_fixtures import write_site_deployment


def _target() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=2,
        target_ra=228.479042,
        target_dec=52.722981,
        target_name="2020ftl",
    )


def _write_photometry_policy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "deployment_file: deployment.yaml",
                "defaults:",
                "  photometry_run_id: smoke_phot",
                "paths:",
                "  inputs:",
                "    diffs: hp_d",
                "    epsf: epsf_r1",
                "pipeline:",
                "  - kind: astrometry",
                "  - kind: forced_photometry",
                "    inputs:",
                "      diffs: hp_d",
                "      epsf: epsf_r1",
                "    output: lc_gepsf",
                "    methods:",
                "      - name: gepsf",
                "        type: psf",
                "        psf_type: epsf",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_scc_bookkeeping(data_root: Path, target: Target) -> Path:
    from syndiff_pipeline.common.scc_paths import scc_diff_bookkeeping_dir

    bk = scc_diff_bookkeeping_dir(
        data_root,
        target.sector,
        target.camera,
        target.ccd,
        oversampling_factor=1,
        template_store_name=None,
    )
    bk.mkdir(parents=True, exist_ok=True)
    (bk / FRAMES_CSV_BASENAME).write_text("product_id\ntess1\n", encoding="utf-8")
    (bk / DIFF_JOB_BASENAME).write_text(
        json.dumps({"schema_version": 2, "mapping_grid": {"nx": 1, "ny": 1}}),
        encoding="utf-8",
    )
    return bk


class TestSccDiffLaneComplete(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.target = _target()
        self.run_config = PhotometryRunConfig(
            photometry_run_id="smoke_phot",
            diffs_label="hp_d",
            epsf_label="epsf_r1",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_incomplete_without_lane_outputs(self) -> None:
        _write_scc_bookkeeping(self.data, self.target)
        self.assertFalse(
            scc_diff_lane_complete(
                self.run_config,
                data_root=self.data,
                sector=self.target.sector,
                camera=self.target.camera,
                ccd=self.target.ccd,
            )
        )

    def test_complete_with_diffs_and_epsf_index(self) -> None:
        _write_scc_bookkeeping(self.data, self.target)
        hp_d = scc_diff_label_dir(
            self.data,
            self.target.sector,
            self.target.camera,
            self.target.ccd,
            store_name=None,
            label="hp_d",
        )
        hp_d.mkdir(parents=True)
        (hp_d / "tess2020057105921-s0020-3-2_hp_d.fits.fz").write_bytes(b"SIMPLE  = T")
        epsf = scc_diff_label_dir(
            self.data,
            self.target.sector,
            self.target.camera,
            self.target.ccd,
            store_name=None,
            label="epsf_r1",
        )
        epsf.mkdir(parents=True)
        (epsf / "gridded_epsf_index.json").write_text("{}", encoding="utf-8")

        self.assertTrue(
            scc_diff_lane_complete(
                self.run_config,
                data_root=self.data,
                sector=self.target.sector,
                camera=self.target.camera,
                ccd=self.target.ccd,
            )
        )


class TestPhotometryComplete(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.event_dir = event_scc_leaf(
            self.root / "handoff",
            _target().event_name(),
            20,
            3,
            2,
        )
        self.event_dir.mkdir(parents=True)
        self.run_config = PhotometryRunConfig(
            photometry_run_id="smoke_phot",
            pipeline=[
                {"kind": "astrometry"},
                {
                    "kind": "forced_photometry",
                    "output": "lc_gepsf",
                    "methods": [{"name": "gepsf"}],
                },
            ],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_astrometry_only_complete_with_json(self) -> None:
        astro_only = PhotometryRunConfig(
            photometry_run_id="smoke_phot",
            pipeline=[{"kind": "astrometry"}],
        )
        phot_root = Path(photometry_root(str(self.event_dir), "smoke_phot"))
        phot_root.mkdir(parents=True)
        (phot_root / "astrometry_result.json").write_text("{}", encoding="utf-8")
        self.assertTrue(photometry_complete(astro_only, self.event_dir))

    def test_forced_photometry_incomplete_without_csv(self) -> None:
        phot_root = Path(photometry_root(str(self.event_dir), "smoke_phot"))
        (phot_root / "lc_gepsf").mkdir(parents=True)
        self.assertFalse(photometry_complete(self.run_config, self.event_dir))

    def test_forced_photometry_complete_with_csv(self) -> None:
        phot_root = Path(photometry_root(str(self.event_dir), "smoke_phot"))
        lc_dir = phot_root / "lc_gepsf"
        lc_dir.mkdir(parents=True)
        (lc_dir / lightcurve_csv_basename("gepsf")).write_text("btjd,flux\n", encoding="utf-8")
        self.assertTrue(photometry_complete(self.run_config, self.event_dir))


class TestMultiStageForcedPhotometry(unittest.TestCase):
    """Several forced_photometry stages sharing one astrometry stage (e.g. one
    per position source: xy_free / ffi_wcs / temporal_wcs) -- every stage's
    output must be checked, not just the first."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.event_dir = event_scc_leaf(
            self.root / "handoff", _target().event_name(), 20, 3, 2
        )
        self.event_dir.mkdir(parents=True)
        self.run_config = PhotometryRunConfig(
            photometry_run_id="smoke_phot",
            pipeline=[
                {"kind": "astrometry"},
                {
                    "kind": "forced_photometry",
                    "output": "lc_xyfree",
                    "methods": [{"name": "xyfree"}],
                },
                {
                    "kind": "forced_photometry",
                    "output": "lc_ffiwcs",
                    "methods": [{"name": "ffiwcs"}],
                },
            ],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_incomplete_when_only_first_stage_has_csv(self) -> None:
        phot_root = Path(photometry_root(str(self.event_dir), "smoke_phot"))
        lc1 = phot_root / "lc_xyfree"
        lc1.mkdir(parents=True)
        (lc1 / lightcurve_csv_basename("xyfree")).write_text("btjd,flux\n", encoding="utf-8")
        self.assertFalse(photometry_complete(self.run_config, self.event_dir))

    def test_complete_when_all_stages_have_csv(self) -> None:
        phot_root = Path(photometry_root(str(self.event_dir), "smoke_phot"))
        for label, name in (("lc_xyfree", "xyfree"), ("lc_ffiwcs", "ffiwcs")):
            lc = phot_root / label
            lc.mkdir(parents=True)
            (lc / lightcurve_csv_basename(name)).write_text("btjd,flux\n", encoding="utf-8")
        self.assertTrue(photometry_complete(self.run_config, self.event_dir))

    def test_collect_artifacts_includes_all_stages(self) -> None:
        phot_root = Path(photometry_root(str(self.event_dir), "smoke_phot"))
        for label, name in (("lc_xyfree", "xyfree"), ("lc_ffiwcs", "ffiwcs")):
            lc = phot_root / label
            lc.mkdir(parents=True)
            (lc / lightcurve_csv_basename(name)).write_text("btjd,flux\n", encoding="utf-8")
        artifacts = collect_photometry_artifacts(self.run_config, self.event_dir)
        basenames = {Path(a).name for a in artifacts}
        self.assertIn(lightcurve_csv_basename("xyfree"), basenames)
        self.assertIn(lightcurve_csv_basename("ffiwcs"), basenames)


class TestVerifyPhotometryPrerequisites(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.site = self.root / "site"
        self.site.mkdir()
        self.handoff = self.root / "handoff"
        self.data = self.root / "data"
        write_site_deployment(
            self.site,
            workspace_root=str(self.handoff),
            data_root=str(self.data),
        )
        _write_photometry_policy(self.site / "photometry_config.yaml")
        self.target = _target()
        self.policy = PhotometrySitePolicy(
            deployment_file="deployment.yaml",
            config_path=str(self.site / "photometry_config.yaml"),
            pipeline=[
                {"kind": "astrometry"},
                {
                    "kind": "forced_photometry",
                    "output": "lc_gepsf",
                    "methods": [{"name": "gepsf"}],
                },
            ],
            defaults={"photometry_run_id": "smoke_phot"},
            paths={"inputs": {"diffs": "hp_d", "epsf": "epsf_r1"}},
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rejects_missing_coordinates(self) -> None:
        bad = Target(
            sector=20,
            camera=3,
            ccd=2,
            target_ra=None,
            target_dec=None,
            target_name="2020ftl",
        )
        ok, msg = verify_photometry_prerequisites(self.policy, bad, site_dir=self.site)
        self.assertFalse(ok)
        self.assertIn("target_ra", msg)

    def test_rejects_incomplete_scc_lane(self) -> None:
        ok, msg = verify_photometry_prerequisites(
            self.policy, self.target, site_dir=self.site
        )
        self.assertFalse(ok)
        self.assertIn("SCC diff lane incomplete", msg)

    def test_passes_when_lane_ready(self) -> None:
        _write_scc_bookkeeping(self.data, self.target)
        lane = scc_diff_dir(self.data, self.target.sector, self.target.camera, self.target.ccd)
        hp_d = lane / "hp_d"
        hp_d.mkdir(parents=True)
        (hp_d / "tess2020057105921-s0020-3-2_hp_d.fits.fz").write_bytes(b"SIMPLE  = T")
        epsf = lane / "epsf_r1"
        epsf.mkdir(parents=True)
        (epsf / "gridded_epsf_index.json").write_text("{}", encoding="utf-8")

        ok, msg = verify_photometry_prerequisites(
            self.policy, self.target, site_dir=self.site
        )
        self.assertTrue(ok, msg)

    def test_resolve_run_config_matches_policy(self) -> None:
        run_config = resolve_photometry_run_config(
            self.policy, self.target, site_dir=self.site
        )
        self.assertEqual(run_config.diffs_label, "hp_d")
        self.assertEqual(run_config.epsf_label, "epsf_r1")
        self.assertEqual(run_config.photometry_run_id, "smoke_phot")


if __name__ == "__main__":
    unittest.main()
