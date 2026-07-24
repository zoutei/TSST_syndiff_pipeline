"""Tests for Discord bot channel matching."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.discord_bot import (
    _channel_matches,
    _condor_shell_argv,
    cluster_status_trigger,
    condor_shell_trigger,
    run_cluster_status_command,
    run_condor_shell_command,
)


class TestChannelMatches(unittest.TestCase):
    def test_same_channel(self):
        msg = SimpleNamespace(channel=SimpleNamespace(id=123, parent_id=None, parent=None))
        self.assertTrue(_channel_matches(msg, 123))

    def test_thread_parent_id(self):
        msg = SimpleNamespace(
            channel=SimpleNamespace(id=999, parent_id=123, parent=None),
        )
        self.assertTrue(_channel_matches(msg, 123))

    def test_other_channel(self):
        msg = SimpleNamespace(channel=SimpleNamespace(id=456, parent_id=None, parent=None))
        self.assertFalse(_channel_matches(msg, 123))


class TestCondorShellCommands(unittest.TestCase):
    def test_condor_shell_trigger_exact_match(self):
        self.assertEqual(condor_shell_trigger("condor_q"), "condor_q")
        self.assertEqual(condor_shell_trigger("  CONDOR_QN  "), "condor_qn")
        self.assertEqual(condor_shell_trigger("condor_status -tla"), "condor_status -tla")
        self.assertIsNone(condor_shell_trigger("condor_q -all"))
        self.assertIsNone(condor_shell_trigger("star_lc_verify"))

    def test_condor_qn_argv_uses_current_user(self):
        with mock.patch.dict(os.environ, {"USER": "kshukawa"}, clear=False):
            argv = _condor_shell_argv("condor_qn")
        self.assertEqual(
            argv,
            [
                "condor_q",
                "kshukawa",
                "-af",
                "ClusterId",
                "ProcId",
                "JobStatus",
                "RemoteHost",
            ],
        )

    def test_condor_status_tla_argv(self):
        self.assertEqual(
            _condor_shell_argv("condor_status -tla"),
            ["condor_status", "-af", "Name", "TotalLoadAvg"],
        )

    def test_run_condor_shell_command_formats_output(self):
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(
                stdout="OWNER BATCH_NAME\nkshukawa ID: 2\n",
                stderr="",
                returncode=0,
            )
            messages = run_condor_shell_command("condor_q")
        self.assertEqual(len(messages), 1)
        self.assertIn("**condor_q**", messages[0])
        self.assertIn("kshukawa ID: 2", messages[0])

    def test_run_condor_shell_command_missing_binary(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            messages = run_condor_shell_command("condor_status")
        self.assertIn("not found", messages[0])


class TestClusterStatusCommand(unittest.TestCase):
    def test_cluster_status_trigger_substring(self):
        self.assertTrue(cluster_status_trigger("how is the cluster?"))
        self.assertTrue(cluster_status_trigger("syndiff cluster"))
        self.assertFalse(cluster_status_trigger("condor_q"))

    def test_run_cluster_status_command_formats_output(self):
        with mock.patch(
            "syndiff_pipeline.common.orchestration.host_stats_cli.render_cluster_table_text",
            return_value="HOST  SLOT\nplscience1.stsci.edu  515GB",
        ):
            messages = run_cluster_status_command()
        self.assertEqual(len(messages), 1)
        self.assertIn("**syndiff cluster**", messages[0])
        self.assertIn("plscience1.stsci.edu", messages[0])


if __name__ == "__main__":
    unittest.main()
