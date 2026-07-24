"""Tests for star gridded ePSF build/load and gepsf photometry wiring."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.difference_imaging.stages import gridded_epsf
from syndiff_pipeline.star import diff_runner
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.epsf_runner import ensure_star_epsf_catalog, epsf_workspace_dir
from syndiff_pipeline.star.identifiers import ResolvedHost
from syndiff_pipeline.star import runner
from syndiff_pipeline.star.site_config import (
    StarEpsfConfig,
    StarRunConfig,
    load_star_site_policy,
    load_star_targets,
    find_star_target_row,
    resolve_star_run_config,
)
from syndiff_pipeline.star.windowed_photometry import run_windowed_forced_photometry


def _minimal_ctx(tmp: Path, *, ws_name: str = "ws") -> StarEventContext:
    crop_bounds = {
        "x_min": 0,
        "y_min": 0,
        "x_max": 64,
        "y_max": 64,
        "shape": (64, 64),
    }
    event = tmp / "events" / "s20_astrometry" / "s0020_c3_k2"
    ws = event / ws_name
    data = tmp / "data" / "s0020" / "c3" / "k2" / "diff"
    for sub in ("hp_d", "hp_c", "ks_b_s", "hp_d_kernels", "epsf_r1"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("hp_d", "hp_c", "ks_b_s", "hp_d_kernels"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    ffi_name = "tess2019358235923-s0020-3-2-0165-s_ffic.fits.gz"
    ffi_path = tmp / ffi_name
    manifest = pd.DataFrame(
        {
            "path": [str(ffi_path)],
            "btjd": [2459000.1],
            "group_id": [0],
            "group_dx": [0.0],
            "group_dy": [0.0],
        }
    )
    manifest.to_csv(event / "frames.csv", index=False)
    gaia = pd.DataFrame(
        {
            "ra": [100.0],
            "dec": [20.0],
            "x": [32.0],
            "y": [32.0],
            "phot_rp_mean_mag": [10.0],
        }
    )
    gaia.to_csv(tmp / "gaia.csv", index=False)
    return StarEventContext(
        target=Target(
            sector=20,
            camera=3,
            ccd=2,
            target_ra=0.0,
            target_dec=0.0,
            target_name="s20_astrometry",
        ),
        event_dir=str(event),
        workspace_root=str(tmp / "workspace"),
        data_root=str(tmp / "data"),
        cluster_job_path=str(event / "event_job.json"),
        cluster_job=crop_bounds,
        crop_bounds=crop_bounds,
        mapping_dir=str(tmp / "mapping"),
        mapping_csv=str(tmp / "mapping" / "map.csv"),
        master_mapping_fits=str(tmp / "mapping" / "master.fits.fz"),
        gaia_catalog_path=str(tmp / "gaia.csv"),
        templates_dir=str(tmp / "templates"),
        reference_ffi_path=str(tmp / "ref.fits"),
        sector=20,
        camera=3,
        ccd=2,
        baseline_workspace_dir=str(ws),
        baseline_diffs_label="hp_d",
        baseline_diffs_dir=str(data / "hp_d"),
        baseline_convolved_dir=str(data / "hp_c"),
        baseline_phot_bkg_dir=str(data / "ks_b_s"),
        baseline_phot_bkg_label="ks_b_s",
        baseline_kernels_dir=str(data / "hp_d_kernels"),
        output_store_name=None,
    )


def _write_fake_gridded_epsf_workspace(out_dir: Path, ffi_stem: str = "tess2019358235923") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(16.0, 16.0), (48.0, 16.0), (16.0, 48.0), (48.0, 48.0)]
    npz_path = gridded_epsf.gridded_epsf_npz_path(str(out_dir), ffi_stem)
    gridded_epsf.save_gridded_epsf_npz(npz_path, stack, grid_xypos, oversampling=2)
    index_path = out_dir / gridded_epsf.GRIDDED_EPSF_INDEX_BASENAME
    index_path.write_text(json.dumps({ffi_stem: npz_path}), encoding="utf-8")


def _resolved_host() -> ResolvedHost:
    return ResolvedHost(
        input_kind="gaia",
        input_value=1060421588522505216,
        tic_id=None,
        gaia_source_id=1060421588522505216,
        ra=0.0,
        dec=0.0,
        phot_g_mean_mag=12.0,
        phot_bp_mean_mag=None,
        phot_rp_mean_mag=None,
        resolution_method="test",
        label=None,
    )


class TestStarEpsfGepsfConfig(unittest.TestCase):
    def test_production_yaml_loads_with_explicit_wiring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            site = Path(tmpdir)
            policy_path = site / "star_config.yaml"
            policy_path.write_text(
                (_ROOT / "config" / "star_config_epsf_gepsf.yaml")
                .read_text(encoding="utf-8")
                .replace(
                    "  requirements: 'Memory >= 100000 && LoadAvg < 10 && Machine != \"plscience10.stsci.edu\"'\n  rank: \"-LoadAvg\"\n",
                    "  host_stats_min_mem_mb: 100000\n  host_stats_max_load15: 10.0\n",
                ),
                encoding="utf-8",
            )
            targets_path = site / "star_targets.csv"
            targets_path.write_text(
                (_ROOT / "config" / "star_targets_example.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            policy = load_star_site_policy(policy_path)
            rows = load_star_targets(targets_path, site_dir=site)
        row = find_star_target_row(rows, "s0020_c3_k2")
        run_cfg = resolve_star_run_config(policy, row, site_dir=site)

        self.assertIsNotNone(run_cfg.epsf)
        assert run_cfg.epsf is not None
        self.assertEqual(run_cfg.epsf.output, "epsf_r1")
        self.assertEqual(run_cfg.epsf.diffs, "hp_d")
        gepsf = run_cfg.photometry_methods[0]
        self.assertEqual(gepsf["name"], "gepsf")
        self.assertEqual(gepsf["inputs"]["epsf"], "epsf_r1")
        self.assertEqual(gepsf["epsf_workspace"], "epsf_r1")


class TestStarEpsfRunner(unittest.TestCase):
    def test_load_existing_epsf_catalog_without_build_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = _minimal_ctx(tmp_path)
            out = Path(epsf_workspace_dir(ctx, "epsf_r1"))
            _write_fake_gridded_epsf_workspace(out)

            catalog = ensure_star_epsf_catalog(
                ctx,
                "epsf_r1",
                build_cfg=None,
                overwrite=False,
            )
            self.assertEqual(catalog.workspace_dir, str(out))
            self.assertIn("tess2019358235923", catalog.index)

    def test_missing_epsf_raises_without_build_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _minimal_ctx(Path(tmp))
            with self.assertRaises(FileNotFoundError):
                ensure_star_epsf_catalog(ctx, "epsf_r1", build_cfg=None, overwrite=False)

    def test_build_epsf_uses_explicit_diffs_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = _minimal_ctx(tmp_path)
            hp_d = Path(ctx.baseline_diffs_dir)
            from syndiff_pipeline.difference_imaging.stages import hotpants

            hotpants._write_image_fits(
                str(hp_d / "tess2019358235923-s0020-3-2_hp_d.fits.fz"),
                np.zeros((64, 64), dtype=np.float32),
            )

            build_cfg = StarEpsfConfig(
                enabled=True,
                diffs="hp_d",
                output="epsf_r1",
                tile_nx=1,
                tile_ny=1,
                epsf_n_jobs=1,
            )
            out_dir = epsf_workspace_dir(ctx, "epsf_r1")
            fake_stack = np.ones((1, 1, 121), dtype=np.float64)

            with mock.patch(
                "syndiff_pipeline.star.epsf_runner.epsf_fitting.fit_epsf_all_frames",
                return_value=(fake_stack, [(32.0, 32.0)], ["tess2019358235923"], [True]),
            ) as mock_fit:
                with mock.patch(
                    "syndiff_pipeline.star.epsf_runner.catalog_from_workspace",
                    return_value=gridded_epsf.GriddedEpsfCatalog(
                        workspace_dir=out_dir,
                        index={"tess2019358235923": "fake.npz"},
                    ),
                ):
                    with mock.patch(
                        "syndiff_pipeline.star.epsf_runner.workspace_has_gridded_epsf",
                        return_value=False,
                    ), mock.patch(
                        "syndiff_pipeline.star.epsf_runner.load_catalog_for_event",
                        return_value=None,
                    ), mock.patch(
                        "syndiff_pipeline.star.epsf_runner.os.path.isfile",
                        return_value=True,
                    ), mock.patch(
                        "syndiff_pipeline.common.wcs_header_cache.load_ffi_list",
                        return_value=pd.DataFrame(),
                    ):
                        ensure_star_epsf_catalog(
                            ctx,
                            "epsf_r1",
                            build_cfg=build_cfg,
                            overwrite=True,
                        )

            mock_fit.assert_called_once()
            self.assertEqual(mock_fit.call_args.kwargs["diffs_input"], "hp_d")
            self.assertEqual(mock_fit.call_args.kwargs["epsf_label"], "epsf_r1")


class TestStarGepsfMethodWiring(unittest.TestCase):
    def test_resolve_photometry_methods_attaches_catalog_by_label(self):
        ctx = mock.Mock()
        sentinel = object()
        catalogs = {"epsf_r1": sentinel}
        methods = [
            {
                "name": "gepsf",
                "type": "psf",
                "psf_type": "epsf",
                "inputs": {"epsf": "epsf_r1"},
                "epsf_workspace": "epsf_r1",
            }
        ]
        resolved = runner._resolve_photometry_methods(
            ctx,
            methods=methods,
            x_ref=1.0,
            y_ref=2.0,
            epsf_catalogs=catalogs,
        )
        self.assertIs(resolved[0]["gridded_catalog"], sentinel)

    def test_resolve_photometry_methods_missing_catalog_raises(self):
        ctx = mock.Mock()
        methods = [
            {
                "name": "gepsf",
                "type": "psf",
                "psf_type": "epsf",
                "inputs": {"epsf": "epsf_r1"},
                "epsf_workspace": "epsf_r1",
            }
        ]
        with self.assertRaisesRegex(ValueError, "inputs.epsf='epsf_r1'"):
            runner._resolve_photometry_methods(
                ctx,
                methods=methods,
                x_ref=1.0,
                y_ref=2.0,
                epsf_catalogs={},
            )

    def test_runner_loads_catalog_per_required_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ctx = _minimal_ctx(tmp_path)
            out = Path(epsf_workspace_dir(ctx, "epsf_r1"))
            _write_fake_gridded_epsf_workspace(out)

            hosts_csv = tmp_path / "hosts.csv"
            hosts_csv.write_text("tic_id,gaia_source_id,label\n", encoding="utf-8")
            run_config = StarRunConfig(
                photometry_methods=[
                    {
                        "name": "gepsf",
                        "type": "psf",
                        "psf_type": "epsf",
                        "inputs": {"epsf": "epsf_r1"},
                        "epsf_workspace": "epsf_r1",
                        "fit_shape": 11,
                        "aperture_radius": 2,
                        "psf_grouper_min_separation": 10,
                    }
                ],
                epsf=StarEpsfConfig(enabled=True, output="epsf_r1", diffs="hp_d"),
                stars_file=str(hosts_csv),
            )

            labels = []
            with mock.patch(
                "syndiff_pipeline.star.runner.ensure_star_epsf_catalog",
                side_effect=lambda _ctx, label, **kwargs: (
                    labels.append(label),
                    gridded_epsf.catalog_from_workspace(str(out)),
                )[1],
            ) as mock_ensure:
                with mock.patch(
                    "syndiff_pipeline.star.runner.load_star_hosts_file",
                    return_value=[],
                ):
                    with mock.patch(
                        "syndiff_pipeline.star.runner.validate_star_prerequisites"
                    ):
                        runner.run_star_pipeline(ctx, run_config=run_config, validate=False)

            mock_ensure.assert_called_once()
            self.assertEqual(labels, ["epsf_r1"])
            call_kwargs = mock_ensure.call_args.kwargs
            self.assertIs(call_kwargs["build_cfg"], run_config.epsf)


class TestWindowedGepsfPhotometry(unittest.TestCase):
    def test_gepsf_batch_writes_lightcurve_csv(self):
        host = _resolved_host()
        host_xy = (12.0, 12.0)
        ffi_stem = "tess2019358235923"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            epsf_dir = tmp_path / "epsf_r1"
            _write_fake_gridded_epsf_workspace(epsf_dir, ffi_stem=ffi_stem)
            catalog = gridded_epsf.catalog_from_workspace(str(epsf_dir))
            assert catalog is not None

            stamp_paths = []
            for i in range(2):
                stamp = np.full((25, 25), 5.0, dtype=np.float64)
                stamp[int(round(host_xy[1])), int(round(host_xy[0]))] += 20.0 + i
                path = tmp_path / "stamps" / f"{ffi_stem}_{i}.fits.fz"
                path.parent.mkdir(parents=True, exist_ok=True)
                written = diff_runner.write_star_diff_stamp(
                    str(path),
                    stamp.astype(np.float32),
                    window_origin=(10 + i, 20 + i),
                    host_local_xy=(10 + i + host_xy[0], 20 + i + host_xy[1]),
                )
                stamp_paths.append(written)

            out_dir = tmp_path / "lc"
            dfs = run_windowed_forced_photometry(
                stamp_paths,
                host=host,
                methods=[
                    {
                        "name": "gepsf",
                        "type": "psf",
                        "psf_type": "epsf",
                        "inputs": {"epsf": "epsf_r1"},
                        "gridded_catalog": catalog,
                        "fit_shape": 11,
                        "aperture_radius": 2,
                        "psf_grouper_min_separation": 10,
                    }
                ],
                output_dir=str(out_dir),
                time_values=[2459000.1, 2459000.2],
            )

            self.assertIn("gepsf", dfs)
            self.assertEqual(len(dfs["gepsf"]), 2)
            csv_path = out_dir / f"lightcurve_gepsf_gaia_{host.gaia_source_id}.csv"
            self.assertTrue(csv_path.is_file())

    def test_gepsf_missing_frame_model_writes_nan(self):
        host = _resolved_host()
        host_xy = (12.0, 12.0)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            epsf_dir = tmp_path / "epsf_r1"
            _write_fake_gridded_epsf_workspace(epsf_dir, ffi_stem="known_frame")
            catalog = gridded_epsf.catalog_from_workspace(str(epsf_dir))
            assert catalog is not None

            stamp = np.full((25, 25), 5.0, dtype=np.float64)
            path = tmp_path / "stamps" / "missing_frame.fits.fz"
            path.parent.mkdir(parents=True, exist_ok=True)
            written = diff_runner.write_star_diff_stamp(
                str(path),
                stamp.astype(np.float32),
                window_origin=(10, 20),
                host_local_xy=(10 + host_xy[0], 20 + host_xy[1]),
            )

            dfs = run_windowed_forced_photometry(
                [written],
                host=host,
                methods=[
                    {
                        "name": "gepsf",
                        "type": "psf",
                        "psf_type": "epsf",
                        "gridded_catalog": catalog,
                        "fit_shape": 11,
                    }
                ],
                output_dir=str(tmp_path / "lc"),
            )
            self.assertTrue(np.isnan(dfs["gepsf"]["flux"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
