"""Tests for deployment path loading and daemon discovery."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.deployment import load_workspace_root_from_deployment
from syndiff_pipeline.common.orchestration.workspace import (
    discover_alive_workspace_roots,
    load_recorded_deployment_path,
    record_deployment_path,
)


class TestDeploymentPathLoading(unittest.TestCase):
    def test_load_workspace_root_from_deployment_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            deploy = base / "deployment.yaml"
            handoff = base / "handoff"
            deploy.write_text(
                f"workspace_root: {handoff}\ndata_root: {base / 'data'}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                str(load_workspace_root_from_deployment(deploy)),
                str(handoff.resolve()),
            )

    def test_record_and_load_deployment_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = base / "deployment.yaml"
            deploy.write_text(
                f"workspace_root: {handoff}\ndata_root: {base / 'data'}\n",
                encoding="utf-8",
            )
            record_deployment_path(handoff, deploy)
            loaded = load_recorded_deployment_path(handoff)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, deploy.resolve())


class TestDaemonDiscovery(unittest.TestCase):
    def test_discover_returns_list(self):
        roots = discover_alive_workspace_roots()
        self.assertIsInstance(roots, list)


if __name__ == "__main__":
    unittest.main()
