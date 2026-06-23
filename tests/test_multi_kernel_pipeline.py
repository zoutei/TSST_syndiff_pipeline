"""Tests for multi-kernel diff pipeline configs."""

from __future__ import annotations

import unittest
from pathlib import Path

from syndiff_pipeline.difference_imaging.orchestration.site_config import (
    load_diff_site_policy,
)
from syndiff_pipeline.difference_imaging.orchestration.validate import validate_pipeline
from syndiff_pipeline.difference_imaging.orchestration.config import SynDiffConfig


class TestMultiKernelPipelineConfigs(unittest.TestCase):
    def test_resume_yaml_validates_with_site_defaults(self):
        repo = Path(__file__).resolve().parents[1]
        path = repo / "config" / "diff_config_multi_kernel_resume.yaml"
        policy = load_diff_site_policy(path)
        cfg = SynDiffConfig(pipeline=list(policy.pipeline))
        for key, val in policy.defaults.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        for item in policy.additional_forced_targets:
            cfg.additional_forced_targets = list(policy.additional_forced_targets)
        validate_pipeline(cfg)

    def test_full_multi_kernel_yaml_validates(self):
        repo = Path(__file__).resolve().parents[1]
        path = repo / "config" / "diff_config_multi_kernel.yaml"
        policy = load_diff_site_policy(path)
        cfg = SynDiffConfig(pipeline=list(policy.pipeline))
        for key, val in policy.defaults.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
        validate_pipeline(cfg)


if __name__ == "__main__":
    unittest.main()
