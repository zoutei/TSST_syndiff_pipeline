"""Tests for Condor-aware n_jobs resolution."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.parallelism import resolve_effective_n_jobs


class TestResolveEffectiveNJobs(unittest.TestCase):
    def test_cfg_n_jobs_without_env(self):
        with mock.patch("multiprocessing.cpu_count", return_value=16):
            self.assertEqual(resolve_effective_n_jobs(4), 4)

    def test_condor_env_overrides_cfg(self):
        with mock.patch.dict(os.environ, {"SYNDIFF_REQUEST_CPUS": "8"}, clear=False):
            with mock.patch("multiprocessing.cpu_count", return_value=16):
                self.assertEqual(resolve_effective_n_jobs(4), 8)

    def test_caps_at_cpu_count(self):
        with mock.patch.dict(os.environ, {"SYNDIFF_REQUEST_CPUS": "64"}, clear=False):
            with mock.patch("multiprocessing.cpu_count", return_value=8):
                self.assertEqual(resolve_effective_n_jobs(4), 8)

    def test_stage_n_jobs_used_when_no_condor_env(self):
        with mock.patch("multiprocessing.cpu_count", return_value=16):
            self.assertEqual(
                resolve_effective_n_jobs(8, stage_n_jobs=2),
                2,
            )

    def test_minimum_one(self):
        with mock.patch("multiprocessing.cpu_count", return_value=0):
            self.assertEqual(resolve_effective_n_jobs(0), 1)


if __name__ == "__main__":
    unittest.main()
