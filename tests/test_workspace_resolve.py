"""Tests for deployment path loading and daemon discovery."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.deployment import load_workspace_root_from_deployment
from syndiff_pipeline.common.orchestration.workspace import (
    deployment_candidates,
    discover_alive_workspace_roots,
    handoff_cache_path,
    load_handoff_cache,
    load_recorded_deployment_path,
    record_deployment_path,
    record_handoff_cache,
    resolve_handoff_fast,
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


class TestHandoffCache(unittest.TestCase):
    def test_record_and_load_handoff_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = base / "deployment.yaml"
            deploy.write_text(
                f"workspace_root: {handoff}\ndata_root: {base / 'data'}\n",
                encoding="utf-8",
            )
            cache_path = handoff_cache_path()
            old = cache_path.read_text(encoding="utf-8") if cache_path.is_file() else None
            try:
                record_handoff_cache(handoff, deploy)
                loaded = load_handoff_cache()
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(
                    str(Path(loaded["workspace_root"]).resolve()),
                    str(handoff.resolve()),
                )
                self.assertEqual(
                    str(Path(loaded["deployment_path"]).resolve()),
                    str(deploy.resolve()),
                )
            finally:
                if old is None:
                    cache_path.unlink(missing_ok=True)
                else:
                    cache_path.write_text(old, encoding="utf-8")

    def test_resolve_handoff_fast_uses_deployment_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = base / "deployment.yaml"
            deploy.write_text(
                f"workspace_root: {handoff}\ndata_root: {base / 'data'}\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                "os.environ",
                {"SYNDIFF_DEPLOYMENT": str(deploy)},
                clear=False,
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.scheduler_control.daemon_is_alive",
                return_value=True,
            ):
                resolved = resolve_handoff_fast(require_daemon=True)
            self.assertEqual(resolved, str(handoff.resolve()))

    def test_deployment_candidates_includes_env_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            handoff = base / "handoff"
            handoff.mkdir()
            deploy = base / "deployment.yaml"
            deploy.write_text(
                f"workspace_root: {handoff}\ndata_root: {base / 'data'}\n",
                encoding="utf-8",
            )
            cache_path = handoff_cache_path()
            old = cache_path.read_text(encoding="utf-8") if cache_path.is_file() else None
            try:
                record_handoff_cache(handoff, deploy)
                with mock.patch.dict(
                    "os.environ",
                    {"SYNDIFF_DEPLOYMENT": str(deploy)},
                    clear=False,
                ):
                    candidates = deployment_candidates()
                self.assertEqual(candidates[0], deploy.resolve())
            finally:
                if old is None:
                    cache_path.unlink(missing_ok=True)
                else:
                    cache_path.write_text(old, encoding="utf-8")


class TestDaemonDiscovery(unittest.TestCase):
    def test_discover_returns_list(self):
        roots = discover_alive_workspace_roots()
        self.assertIsInstance(roots, list)


if __name__ == "__main__":
    unittest.main()
