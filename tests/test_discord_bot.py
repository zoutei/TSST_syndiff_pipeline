"""Tests for Discord bot channel matching."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.orchestration.discord_bot import _channel_matches


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


if __name__ == "__main__":
    unittest.main()
