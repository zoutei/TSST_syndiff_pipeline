"""Guardrail: template-stage execution resources must stay out of the manifest
fingerprint.

The template side already gets this right -- ``verify.config_fingerprint`` is a
hand-curated *inclusion* list that names only science params, so
``condor_request_*``/``n_jobs``/``small_job_*`` never reach the hash and are
retunable on a live run. Nothing enforced that, though: adding a resource key to
one of those ``parts`` lists would silently make every template manifest
sensitive to a worker count, exactly the failure the diff side just had to be
dug out of.

These tests pin the property so a future edit has to break a test to break it.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    load_runner_config,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.verify import config_fingerprint

# stage -> a resource-only override that must NOT move the fingerprint.
RESOURCE_OVERRIDES = {
    "mapping": [
        {"condor_request_cpus": 64},
        {"condor_request_memory": 250_000},
        {"host_stats_min_mem_mb": 250_000},
        {"executor": "local"},
    ],
    "ps1_process": [
        {"condor_request_cpus": 8},
        {"condor_request_memory": 200_000},
        {"num_ingest_workers": 4},
        {"stream_max_inflight_requests": 8},
        {"small_job_request_cpus": 4},
        {"small_job_min_memory_mb": 110_000},
        {"small_job_memory_per_skycell_mb": 5_000},
    ],
    "remap": [
        {"n_jobs": 1},
        {"condor_request_cpus": 8},
        {"condor_request_memory": 64_000},
        {"stage_regmaps_to_scratch": True},
    ],
    "downsample": [
        {"n_jobs": 1},
        {"skycells_per_batch": 3},
        {"condor_request_cpus": 8},
        {"condor_request_memory": 64_000},
    ],
}

# A science param per stage that MUST still move the fingerprint, so the tests
# above cannot pass by the fingerprint simply ignoring everything.
RECIPE_OVERRIDES = {
    "mapping": {"oversampling_factor": 4},
    "ps1_process": {"psf_sigma": 12.5},
    "remap": {"intra_skycell_R": 7},
    "downsample": {"single_offset": True},
}


def _target() -> Target:
    return Target(
        sector=20,
        camera=3,
        ccd=3,
        target_ra=228.0,
        target_dec=52.0,
        target_name="2020ut",
    )


def _resolved(tmp: Path, stage: str, overrides: dict):
    cfg_path = tmp / f"{stage}_{abs(hash(str(overrides))) % 10**8}.yaml"
    doc = {
        "config_schema_version": 2,
        "workspace_root": str(tmp / "workspace"),
        "data_root": str(tmp / "data"),
        "runs_root": str(tmp / "workspace" / "runs"),
        "state_db_path": str(tmp / "state.sqlite"),
        "skycell_wcs_csv": "x.csv",
        "stages": {stage: dict(overrides)} if overrides else {},
    }
    cfg_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return resolve_config(_target(), load_runner_config(str(cfg_path)))


class TestTemplateResourceParamsNotFingerprinted(unittest.TestCase):
    def test_resource_overrides_do_not_move_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for stage, overrides_list in RESOURCE_OVERRIDES.items():
                base = config_fingerprint(_resolved(tmp, stage, {}), stage)
                for overrides in overrides_list:
                    with self.subTest(stage=stage, overrides=overrides):
                        self.assertEqual(
                            base,
                            config_fingerprint(_resolved(tmp, stage, overrides), stage),
                            f"{stage}: {overrides} is an execution resource and must "
                            f"not participate in the manifest fingerprint",
                        )

    def test_recipe_overrides_still_move_fingerprint(self):
        """Guard against a vacuous test: real science params must still count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for stage, overrides in RECIPE_OVERRIDES.items():
                with self.subTest(stage=stage, overrides=overrides):
                    base = config_fingerprint(_resolved(tmp, stage, {}), stage)
                    self.assertNotEqual(
                        base,
                        config_fingerprint(_resolved(tmp, stage, overrides), stage),
                        f"{stage}: {overrides} changes outputs and must be hashed",
                    )


if __name__ == "__main__":
    unittest.main()
