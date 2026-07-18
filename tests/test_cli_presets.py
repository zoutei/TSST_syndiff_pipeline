"""Tests for syndiff CLI stage presets."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from syndiff_pipeline.common.orchestration import cli as orch_cli
from syndiff_pipeline.common.orchestration.cli import DIFF_STAGE, preset_stages
from syndiff_pipeline.template_creation.orchestration.stages import TEMPLATE_STAGES
from syndiff_pipeline.cli import build_execution_parser, main, parse_execution_argv
from syndiff_pipeline.cli import preset_stages as entry_preset_stages


class TestPresetStageLists(unittest.TestCase):
    def test_template_preset_matches_template_stages(self):
        expected = [spec.name for spec in TEMPLATE_STAGES]
        self.assertEqual(preset_stages("template"), expected)

    def test_all_preset_removed(self):
        with self.assertRaises(ValueError):
            preset_stages("all")

    def test_diff_preset_is_diff_only(self):
        self.assertEqual(preset_stages("diff"), [DIFF_STAGE])

    def test_entry_point_reexports_preset_stages(self):
        self.assertIs(entry_preset_stages, preset_stages)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            preset_stages("smoke")


class TestExecutionParser(unittest.TestCase):
    def test_parse_execution_argv_sets_preset_for_template_scc(self):
        _, verb, args = parse_execution_argv(
            [
                "template",
                "submit",
                "--config",
                "/tmp/config.yaml",
                "--scc",
                "/tmp/sccs.csv",
            ]
        )
        self.assertEqual(verb, "submit")
        self.assertEqual(args.preset, "template")
        self.assertEqual(args.scc, "/tmp/sccs.csv")
        self.assertIsNone(args.stages)

    def test_parse_execution_argv_sets_preset_for_template_inline_scc(self):
        _, verb, args = parse_execution_argv(
            [
                "template",
                "submit",
                "--config",
                "/tmp/config.yaml",
                "--sector",
                "22",
                "--camera",
                "3",
                "--ccd",
                "3",
            ]
        )
        self.assertEqual(verb, "submit")
        self.assertEqual(args.preset, "template")
        self.assertEqual(args.sector, 22)
        self.assertEqual(args.camera, 3)
        self.assertEqual(args.ccd, 3)

    def test_all_preset_fails(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_execution_argv(
                [
                    "all",
                    "submit",
                    "--config",
                    "/tmp/config.yaml",
                    "--targets",
                    "/tmp/targets.csv",
                ]
            )
        self.assertIn("all", str(ctx.exception))

    def test_main_rejects_all_preset(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["all", "submit", "--config", "/tmp/config.yaml", "--targets", "/tmp/t.csv"])
        self.assertIn("removed", str(ctx.exception).lower())

    def test_stages_override_clears_preset(self):
        _, _, args = parse_execution_argv(
            [
                "template",
                "run",
                "--config",
                "/tmp/config.yaml",
                "--scc",
                "/tmp/sccs.csv",
                "--stages",
                "mapping,downsample",
            ]
        )
        self.assertIsNone(args.preset)
        self.assertEqual(args.stages, "mapping,downsample")

    def test_site_resolves_template_config(self):
        site = _ROOT / "config"
        if not site.is_dir():
            self.skipTest("config not present")
        _, _, args = parse_execution_argv(
            [
                "diff",
                "submit",
                "--site",
                str(site),
                "--targets",
                str(site / "targets_example.csv"),
            ]
        )
        self.assertEqual(args.config, str((site / "pipeline.yaml").resolve()))

    def test_build_execution_parser_prog(self):
        parser = build_execution_parser("template", "submit")
        self.assertEqual(parser.prog, "syndiff template submit")

    def test_template_parser_has_no_targets(self):
        parser = build_execution_parser("template", "submit")
        action_dests = {a.dest for a in parser._actions}
        self.assertNotIn("targets", action_dests)

    def test_diff_parser_requires_targets(self):
        parser = build_execution_parser("diff", "submit")
        action_dests = {a.dest for a in parser._actions}
        self.assertIn("targets", action_dests)


class TestDiffRunGuard(unittest.TestCase):
    def test_diff_run_without_target_name_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                f"workspace_root: {tmp}\ndata_root: {tmp}\n",
                encoding="utf-8",
            )
            targets_path = Path(tmp) / "targets.csv"
            targets_path.write_text(
                "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
                "22,3,3,228.0,52.0,2020dgc,true\n",
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "diff",
                        "run",
                        "--config",
                        str(cfg_path),
                        "--targets",
                        str(targets_path),
                    ]
                )
            self.assertIn("--target-name", str(ctx.exception))


class TestLocalSubmitPatch(unittest.TestCase):
    def test_local_sets_diff_executor_in_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        f"workspace_root: {handoff}",
                        f"data_root: {handoff / 'data'}",
                        "deployment_file: deployment.yaml",
                        "stages:",
                        "  diff:",
                        "    executor: condor",
                    ]
                ),
                encoding="utf-8",
            )
            (Path(tmp) / "deployment.yaml").write_text(
                f"workspace_root: {handoff}\ndata_root: {handoff / 'data'}\n",
                encoding="utf-8",
            )
            targets_path = Path(tmp) / "targets.csv"
            targets_path.write_text(
                "sector,camera,ccd,target_ra,target_dec,target_name,enabled\n"
                "22,3,3,228.0,52.0,2020dgc,true\n",
                encoding="utf-8",
            )

            _, _, args = parse_execution_argv(
                [
                    "diff",
                    "submit",
                    "--config",
                    str(cfg_path),
                    "--targets",
                    str(targets_path),
                    "--local",
                ]
            )
            with mock.patch.object(orch_cli, "ensure_daemon_running"), mock.patch.object(
                orch_cli, "record_deployment_path"
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.run_setup.apply_post_create_run_setup",
                return_value=mock.Mock(),
            ):
                orch_cli.cmd_submit(args)

            runs = sorted((handoff / "runs").glob("*"))
            self.assertTrue(runs)
            frozen = runs[0] / "config.yaml"
            self.assertTrue(frozen.is_file())
            raw = yaml.safe_load(frozen.read_text(encoding="utf-8"))
            self.assertEqual(raw["stages"]["diff"]["executor"], "local")


class TestTemplateSubmitFreezesScc(unittest.TestCase):
    def test_template_submit_materializes_scc_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff = Path(tmp) / "handoff"
            handoff.mkdir()
            cfg_path = Path(tmp) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        f"workspace_root: {handoff}",
                        f"data_root: {handoff / 'data'}",
                        "deployment_file: deployment.yaml",
                    ]
                ),
                encoding="utf-8",
            )
            (Path(tmp) / "deployment.yaml").write_text(
                f"workspace_root: {handoff}\ndata_root: {handoff / 'data'}\n",
                encoding="utf-8",
            )
            scc_path = Path(tmp) / "sccs.csv"
            scc_path.write_text(
                "sector,camera,ccd,enabled\n22,3,3,true\n",
                encoding="utf-8",
            )

            _, _, args = parse_execution_argv(
                [
                    "template",
                    "submit",
                    "--config",
                    str(cfg_path),
                    "--scc",
                    str(scc_path),
                ]
            )
            with mock.patch.object(orch_cli, "ensure_daemon_running"), mock.patch.object(
                orch_cli, "record_deployment_path"
            ), mock.patch(
                "syndiff_pipeline.common.orchestration.run_setup.apply_post_create_run_setup",
                return_value=mock.Mock(),
            ):
                orch_cli.cmd_submit(args)

            runs = sorted((handoff / "runs").glob("*"))
            self.assertTrue(runs)
            frozen_scc = runs[0] / "scc.csv"
            self.assertTrue(frozen_scc.is_file())
            self.assertIn("22,3,3", frozen_scc.read_text(encoding="utf-8"))


class TestVerifySiteScope(unittest.TestCase):
    def test_verify_site_resolves_config(self):
        site = _ROOT / "config"
        if not site.is_dir():
            self.skipTest("config not present")
        args = orch_cli.build_parser().parse_args(
            [
                "verify",
                "--site",
                str(site),
                "--targets",
                str(site / "targets_example.csv"),
            ]
        )
        orch_cli._resolve_config_from_site(args)
        self.assertEqual(args.config, str((site / "pipeline.yaml").resolve()))


if __name__ == "__main__":
    unittest.main()
