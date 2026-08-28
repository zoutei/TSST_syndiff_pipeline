"""Tests for Contract A2 — ``shared_mask`` recipe hashes the *resolved* mask policy.

Covers ``provenance_glue.diff_recipe`` / ``diff_kind_fingerprint_shared_mask`` /
``emit_shared_mask_artifact``'s new ``mask_settings=`` keyword and its
resolution order: explicit ``mask_settings=`` wins > existing
``params.mask_settings`` path-load > model (``MaskSettings()``) defaults.

The hard acceptance gate (measured by the supervisor before this change
landed): ``recipe_id("shared_mask", ...)`` for ``SharedMaskParams()`` with no
``mask_settings`` override MUST remain ``ee172cefa01e2aaf`` — it is the only
``shared_mask`` recipe_id in the live provenance database. Every existing
on-disk shared_mask artifact is keyed on this literal; if it moves, the whole
history is orphaned.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.stage_params import SharedMaskParams

try:
    from syndiff_pipeline.common.provenance.fingerprint import recipe_id
    from syndiff_pipeline.difference_imaging.masking.settings import (
        MaskSettings,
        SharedMaskSettings,
    )

    _PROVENANCE_IMPORTABLE = True
except Exception:  # pragma: no cover
    _PROVENANCE_IMPORTABLE = False

# The single shared_mask recipe_id in the live provenance DB. Must never move
# for the default policy (mask_settings=None, or an explicit MaskSettings()).
LIVE_DEFAULT_RECIPE_ID = "ee172cefa01e2aaf"


@unittest.skipUnless(
    pg.PROVENANCE_AVAILABLE and _PROVENANCE_IMPORTABLE, "common.provenance not importable"
)
class TestSharedMaskRecipeMaskSettingsKwarg(unittest.TestCase):
    def _recipe_id(self, **kwargs):
        r = pg.diff_recipe("shared_mask", SharedMaskParams(), **kwargs)
        return recipe_id("shared_mask", r["params"], r["code_version"])

    def test_default_none_matches_live_recipe_id(self):
        """Acceptance gate: mask_settings=None must be byte-identical to today."""
        self.assertEqual(self._recipe_id(), LIVE_DEFAULT_RECIPE_ID)

    def test_explicit_default_mask_settings_matches_live_recipe_id(self):
        """An explicit MaskSettings() (all defaults) must also hash to the same id."""
        self.assertEqual(
            self._recipe_id(mask_settings=MaskSettings()), LIVE_DEFAULT_RECIPE_ID
        )

    def test_non_default_policy_changes_recipe_id(self):
        """A materially different policy must produce a *different* id."""
        custom = MaskSettings(shared=SharedMaskSettings(bright_maglim=14.0))
        self.assertNotEqual(self._recipe_id(mask_settings=custom), LIVE_DEFAULT_RECIPE_ID)

    def test_explicit_mask_settings_wins_over_params_path(self):
        """Explicit mask_settings= must win even when params.mask_settings also
        points at a (different) YAML file — resolution order: explicit first.
        """
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "mask_settings.yaml"
            yaml_path.write_text(
                "shared:\n  bright_maglim: 9.5\n  faint_maglim: 16.0\n",
                encoding="utf-8",
            )
            params = SharedMaskParams(mask_settings=str(yaml_path))
            explicit = MaskSettings(shared=SharedMaskSettings(bright_maglim=14.0))

            r_path_only = pg.diff_recipe("shared_mask", params)
            r_explicit = pg.diff_recipe("shared_mask", params, mask_settings=explicit)

            self.assertEqual(
                r_path_only["params"]["mask_settings"]["shared"]["bright_maglim"], 9.5
            )
            self.assertEqual(
                r_explicit["params"]["mask_settings"]["shared"]["bright_maglim"], 14.0
            )
            self.assertNotEqual(r_path_only["params"], r_explicit["params"])

    def test_params_mask_settings_path_still_loads_when_no_explicit_kwarg(self):
        """Regression guard: the pre-existing path-load behaviour (no explicit
        mask_settings= kwarg) must be unaffected by this change.
        """
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "mask_settings.yaml"
            yaml_path.write_text(
                "shared:\n  bright_maglim: 9.5\n  faint_maglim: 16.0\n",
                encoding="utf-8",
            )
            r_default = pg.diff_recipe("shared_mask", SharedMaskParams())
            r_custom = pg.diff_recipe(
                "shared_mask", SharedMaskParams(mask_settings=str(yaml_path))
            )
            self.assertNotEqual(r_default["params"], r_custom["params"])
            self.assertEqual(
                r_custom["params"]["mask_settings"]["shared"]["bright_maglim"], 9.5
            )


@unittest.skipUnless(
    pg.PROVENANCE_AVAILABLE and _PROVENANCE_IMPORTABLE, "common.provenance not importable"
)
class TestDiffKindFingerprintSharedMaskMaskSettingsKwarg(unittest.TestCase):
    """``diff_kind_fingerprint_shared_mask`` / ``emit_shared_mask_artifact`` thread
    the same mask_settings= override into the fingerprint (not just diff_recipe).
    """

    def test_fingerprint_changes_with_explicit_mask_settings(self):
        fp_default = pg.diff_kind_fingerprint_shared_mask(
            20, 3, 3, SharedMaskParams()
        )
        fp_custom = pg.diff_kind_fingerprint_shared_mask(
            20,
            3,
            3,
            SharedMaskParams(),
            mask_settings=MaskSettings(shared=SharedMaskSettings(bright_maglim=14.0)),
        )
        self.assertIsNotNone(fp_default)
        self.assertIsNotNone(fp_custom)
        self.assertNotEqual(fp_default, fp_custom)

    def test_fingerprint_none_matches_explicit_default(self):
        fp_none = pg.diff_kind_fingerprint_shared_mask(20, 3, 3, SharedMaskParams())
        fp_explicit_default = pg.diff_kind_fingerprint_shared_mask(
            20, 3, 3, SharedMaskParams(), mask_settings=MaskSettings()
        )
        self.assertEqual(fp_none, fp_explicit_default)


if __name__ == "__main__":
    unittest.main()
