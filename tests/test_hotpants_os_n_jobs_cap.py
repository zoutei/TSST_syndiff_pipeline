"""Tests for the HOTPANTS_OS_N_JOBS per-worker thread-pool cap.

Regression coverage for a real oversubscription risk: pyhotpants's own
internal precompute_basis_lr_maps step sizes its ThreadPoolExecutor from
os.cpu_count() (or HOTPANTS_OS_N_JOBS if set) with no awareness of how many
other hotpants_n_jobs frame workers are co-resident. Without dividing the
Condor allocation across workers, N concurrent frame workers each try to
grab the full allocation for that one internal step.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages.hotpants import (
    _hotpants_loky_initializer,
    _resolve_hotpants_os_n_jobs,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    HOTPANTS_ALLOWED,
    HotpantsParams,
    parse_hotpants,
)


class TestResolveHotpantsOsNJobs(unittest.TestCase):
    def test_serial_worker_returns_none(self):
        hp = HotpantsParams()
        self.assertIsNone(_resolve_hotpants_os_n_jobs(hp, n_workers=1))

    def test_explicit_override_wins(self):
        hp = HotpantsParams(hotpants_os_n_jobs=7)
        self.assertEqual(_resolve_hotpants_os_n_jobs(hp, n_workers=4), 7)

    def test_explicit_override_applies_in_serial_case_too(self):
        # Regression: precompute_basis_lr_maps launches up to n_ker
        # concurrent threads *per frame*, independent of hotpants_n_jobs --
        # an explicit cap must be reachable even at n_workers=1, where it
        # was previously short-circuited to None unconditionally.
        hp = HotpantsParams(hotpants_os_n_jobs=12)
        self.assertEqual(_resolve_hotpants_os_n_jobs(hp, n_workers=1), 12)

    def test_auto_divides_syndiff_request_cpus(self):
        hp = HotpantsParams()
        with mock.patch.dict(os.environ, {"SYNDIFF_REQUEST_CPUS": "64"}, clear=False):
            self.assertEqual(_resolve_hotpants_os_n_jobs(hp, n_workers=4), 16)

    def test_auto_falls_back_to_cpu_count(self):
        hp = HotpantsParams()
        env = dict(os.environ)
        env.pop("SYNDIFF_REQUEST_CPUS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("os.cpu_count", return_value=32):
                self.assertEqual(_resolve_hotpants_os_n_jobs(hp, n_workers=4), 8)

    def test_never_below_one(self):
        hp = HotpantsParams()
        with mock.patch.dict(os.environ, {"SYNDIFF_REQUEST_CPUS": "3"}, clear=False):
            self.assertEqual(_resolve_hotpants_os_n_jobs(hp, n_workers=8), 1)


class TestLokyInitializerSetsEnv(unittest.TestCase):
    def test_sets_env_var_when_given(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOTPANTS_OS_N_JOBS", None)
            _hotpants_loky_initializer(
                mask=None,
                ref_stars_xy=None,
                hp=HotpantsParams(),
                template_path_map={},
                crop_bounds={},
                workspace_dirs=None,
                round_id=1,
                legacy_bkg_sidecar=False,
                hotpants_os_n_jobs=16,
            )
            self.assertEqual(os.environ.get("HOTPANTS_OS_N_JOBS"), "16")
            os.environ.pop("HOTPANTS_OS_N_JOBS", None)

    def test_leaves_env_unset_when_none(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOTPANTS_OS_N_JOBS", None)
            _hotpants_loky_initializer(
                mask=None,
                ref_stars_xy=None,
                hp=HotpantsParams(),
                template_path_map={},
                crop_bounds={},
                workspace_dirs=None,
                round_id=1,
                legacy_bkg_sidecar=False,
                hotpants_os_n_jobs=None,
            )
            self.assertNotIn("HOTPANTS_OS_N_JOBS", os.environ)


class TestHotpantsOsNJobsConfigParsing(unittest.TestCase):
    def test_allowed_key_and_parses(self):
        self.assertIn("hotpants_os_n_jobs", HOTPANTS_ALLOWED)
        hp = parse_hotpants({"kind": "hotpants", "hotpants_os_n_jobs": 12}, 0)
        self.assertEqual(hp.hotpants_os_n_jobs, 12)

    def test_default_is_none(self):
        hp = parse_hotpants({"kind": "hotpants"}, 0)
        self.assertIsNone(hp.hotpants_os_n_jobs)


if __name__ == "__main__":
    unittest.main()
