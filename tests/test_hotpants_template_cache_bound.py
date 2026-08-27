"""Tests for the bounded per-process template cache in hotpants.py.

Regression coverage for a real OOM: field-mode (tvwcs/OS4) group_id changes
nearly every frame under drift tracking, and the unbounded dict previously
used here retained one large assembled template array per distinct group_id
ever seen in the process -- growing without limit over a full-SCC run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.stages.hotpants import _BoundedTemplateCache
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    HOTPANTS_ALLOWED,
    HotpantsParams,
    parse_hotpants,
)


class TestBoundedTemplateCache(unittest.TestCase):
    def test_evicts_oldest_beyond_maxsize(self):
        cache = _BoundedTemplateCache(maxsize=2)
        cache[1] = "a"
        cache[2] = "b"
        cache[3] = "c"
        self.assertNotIn(1, cache)
        self.assertIn(2, cache)
        self.assertIn(3, cache)

    def test_get_refreshes_recency(self):
        cache = _BoundedTemplateCache(maxsize=2)
        cache[1] = "a"
        cache[2] = "b"
        _ = cache[1]  # touch 1, making 2 the least-recently-used
        cache[3] = "c"
        self.assertIn(1, cache)
        self.assertNotIn(2, cache)
        self.assertIn(3, cache)

    def test_maxsize_at_least_one(self):
        cache = _BoundedTemplateCache(maxsize=0)
        cache[1] = "a"
        cache[2] = "b"
        self.assertEqual(len(cache._data), 1)
        self.assertIn(2, cache)


class TestTemplateCacheMaxGroupsParam(unittest.TestCase):
    def test_default_is_bounded(self):
        hp = HotpantsParams()
        self.assertEqual(hp.template_cache_max_groups, 2)

    def test_allowed_key_present(self):
        self.assertIn("template_cache_max_groups", HOTPANTS_ALLOWED)

    def test_parse_overrides_default(self):
        hp = parse_hotpants({"kind": "hotpants", "template_cache_max_groups": 5}, 0)
        self.assertEqual(hp.template_cache_max_groups, 5)


if __name__ == "__main__":
    unittest.main()
