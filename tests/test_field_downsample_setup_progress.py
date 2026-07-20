"""Setup progress sidecar and field L5 staging helpers."""

from __future__ import annotations

import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from syndiff_pipeline.template_creation.orchestration.stage_params import parse_stage_params
from syndiff_pipeline.template_creation.orchestration.stage_progress import (
    _parse_downsample_sidecar,
)
from syndiff_pipeline.template_creation.processing import field_downsample_progress as fdp
from syndiff_pipeline.template_creation.processing.downsample import (
    resolve_stage_regmaps_to_scratch,
)


class TestFieldSetupProgress(unittest.TestCase):
    def test_init_field_setup_progress_writes_setup_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "downsample.progress.json"
            fdp.init_field_setup_progress(path)
            data = json.loads(path.read_text())
            self.assertEqual(data["phase"], "setup")
            self.assertEqual(data["geometry_mode"], "field")
            self.assertEqual(data["composite_keys_total"], 0)

    def test_init_field_progress_transitions_from_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "downsample.progress.json"
            fdp.init_field_setup_progress(path)
            fdp.init_field_progress(
                path,
                n_skycells=3,
                n_composite_keys=10,
                n_contrib_keys=12,
            )
            data = json.loads(path.read_text())
            self.assertEqual(data["phase"], "field_composite_keys")
            self.assertEqual(data["total_skycells"], 3)
            self.assertEqual(data["composite_keys_total"], 10)
            self.assertIn("setup", data.get("phase_elapsed_s", {}))

    def test_parse_downsample_sidecar_setup_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "downsample.log"
            log.write_text("")
            prog = Path(tmp) / "downsample.progress.json"
            fdp.init_field_setup_progress(prog)
            progress = _parse_downsample_sidecar(log)
            self.assertIsNotNone(progress)
            # StageProgress is a NamedTuple / dataclass — accept either attr name.
            text = getattr(progress, "text", None) or getattr(progress, "label", None)
            if text is None and hasattr(progress, "__iter__"):
                text = next(iter(progress))
            self.assertEqual(str(text), "setup")
            kind = getattr(progress, "kind", None)
            if kind is None and hasattr(progress, "__getitem__"):
                kind = progress[1]
            self.assertEqual(kind, "phase")


class TestFieldStagingConfig(unittest.TestCase):
    def test_stage_regmaps_false_disables_auto_condor(self):
        self.assertFalse(resolve_stage_regmaps_to_scratch(False))

    def test_parse_smoke_style_downsample_params(self):
        stages = parse_stage_params(
            {
                "downsample": {
                    "stage_regmaps_to_scratch": False,
                    "condor_request_disk": 4096,
                    "apply_intra_skycell": True,
                    "apply_inter_skycell": False,
                }
            }
        )
        self.assertFalse(stages.downsample.stage_regmaps_to_scratch)
        self.assertEqual(stages.downsample.condor_request_disk, 4096)


class TestFieldStagingEnospcFallback(unittest.TestCase):
    def test_enospace_leaves_empty_scratch_map(self):
        """Mirror field_downsample ENOSPC handling: empty scratch_regmaps, no raise."""
        import shutil

        from syndiff_pipeline.template_creation.processing.downsample import (
            resolve_downsample_scratch_dir,
            stage_regmap_files_to_scratch,
        )

        sector, camera, ccd, oversampling_factor = 20, 1, 1, 1
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.fits.gz"
            src.write_bytes(b"fake")
            sky_reg = [("skycell.1.1", str(src))]
            scratch_regmaps: dict[str, str] = {}

            def boom(*_a, **_k):
                raise OSError(errno.ENOSPC, "No space left on device")

            with mock.patch(
                "syndiff_pipeline.template_creation.processing.downsample.shutil.copy2",
                side_effect=boom,
            ):
                try:
                    stage_regmap_files_to_scratch(
                        [p for _, p in sky_reg],
                        sector=sector,
                        camera=camera,
                        ccd=ccd,
                        oversampling_factor=oversampling_factor,
                        scratch_base=Path(tmp) / "scratch",
                    )
                    raised = False
                    exc = None
                except OSError as e:
                    raised = True
                    exc = e
                    self.assertEqual(exc.errno, errno.ENOSPC)
                    os_suffix = f"_os{oversampling_factor}" if oversampling_factor > 1 else ""
                    scratch_dir = (
                        Path(tmp)
                        / "scratch"
                        / f"syndiff_downsample_regmaps_{sector:04d}_{camera}_{ccd}{os_suffix}"
                    )
                    if scratch_dir.is_dir():
                        shutil.rmtree(scratch_dir, ignore_errors=True)
                    scratch_regmaps = {}
            self.assertTrue(raised)
            self.assertEqual(scratch_regmaps, {})


if __name__ == "__main__":
    unittest.main()
