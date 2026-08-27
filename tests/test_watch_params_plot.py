from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class WatchParamsPlotTests(unittest.TestCase):
    def test_load_wcs_history_from_checkpoints_orders_and_dedupes(self):
        from unittest import mock

        from dev.forward_epsf_wcs import fit as FIT
        from dev.forward_epsf_wcs.checkpoint_history import save_params_leaves_npz
        from dev.forward_epsf_wcs.diagnostics import watch_params_plot as wpp

        slim = {k: np.asarray(np.zeros((2, 2), dtype=np.float32)) for k in FIT.STAGE_LEAVES}
        tracks_by_step = {
            20: np.ones((4, 3), dtype=np.float64),
            40: np.ones((4, 3), dtype=np.float64),
            60: np.ones((4, 3), dtype=np.float64) * 2.0,
        }

        def coeff_for_path(_params, _bundle):
            step = order[idx[0]]
            idx[0] += 1
            return tracks_by_step[step]

        order = [20, 40, 60]
        idx = [0]
        with tempfile.TemporaryDirectory() as raw:
            artifacts = Path(raw) / "artifacts"
            ckpt = artifacts / "checkpoints"
            ckpt.mkdir(parents=True)
            for step in (60, 20, 40):
                save_params_leaves_npz(ckpt / f"params_s1_step{step:05d}.npz", slim)
            bundle = mock.Mock()
            with mock.patch.object(wpp, "coeff_tracks", side_effect=coeff_for_path):
                history = wpp.load_wcs_history_from_checkpoints(artifacts, bundle)
        self.assertEqual(history["tracks"].shape[0], 2)
        np.testing.assert_array_equal(history["step"], np.array([20, 60], dtype=np.int32))

    def test_resolve_latest_params_prefers_numbered_checkpoint(self):
        from dev.forward_epsf_wcs import fit as FIT
        from dev.forward_epsf_wcs.checkpoint_history import save_params_leaves_npz
        from dev.forward_epsf_wcs.diagnostics.watch_params_plot import resolve_latest_params

        slim_latest = {k: np.asarray(np.ones((2, 2), dtype=np.float32)) for k in FIT.STAGE_LEAVES}
        slim_ckpt = {k: np.asarray(np.zeros((2, 2), dtype=np.float32)) for k in FIT.STAGE_LEAVES}
        with tempfile.TemporaryDirectory() as raw:
            artifacts = Path(raw) / "artifacts"
            ckpt = artifacts / "checkpoints"
            ckpt.mkdir(parents=True)
            save_params_leaves_npz(ckpt / "params_s1_step00020.npz", slim_ckpt)
            save_params_leaves_npz(artifacts / "params_latest.npz", slim_latest)
            params = resolve_latest_params(artifacts)
        self.assertIsNotNone(params)
        np.testing.assert_allclose(np.asarray(params["wcs_coeff"]), np.zeros((2, 2)))

    def test_render_watch_wcs_plot_writes_png(self):
        from dev.forward_epsf_wcs.diagnostics.watch_params_plot import render_watch_wcs_plot

        btjd = np.linspace(2400.0, 2401.0, 5)
        init = np.zeros((4, btjd.size))
        history = {
            "stage": np.array([1], dtype=np.int32),
            "step": np.array([20], dtype=np.int32),
            "tracks": np.ones((1, 4, btjd.size)) * 0.1,
        }
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "watch_live_wcs.png"
            written = render_watch_wcs_plot(
                btjd=btjd,
                init_tracks=init,
                history=history,
                out_path=out,
                run_id="test-run",
                stage=1,
                step=20,
            )
            self.assertEqual(written, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_render_watch_wcs_plot_init_vs_current_without_checkpoints(self):
        from dev.forward_epsf_wcs.diagnostics.watch_params_plot import render_watch_wcs_plot

        btjd = np.linspace(2400.0, 2401.0, 5)
        init = np.zeros((4, btjd.size))
        current = np.ones((4, btjd.size)) * 0.2
        history = {
            "stage": np.array([], dtype=np.int32),
            "step": np.array([], dtype=np.int32),
            "tracks": np.zeros((0, 4, btjd.size)),
        }
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "watch_live_wcs.png"
            written = render_watch_wcs_plot(
                btjd=btjd,
                init_tracks=init,
                history=history,
                current_tracks=current,
                out_path=out,
                run_id="test-run",
                stage=3,
                step=888,
            )
            self.assertEqual(written, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_render_watch_epsf_plot_writes_png(self):
        from unittest import mock

        from dev.forward_epsf_wcs.diagnostics import watch_params_plot as wpp

        grid = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(1, 1, 5, 5)
        btjd = np.linspace(2400.0, 2401.0, 4)
        basis = np.eye(4)
        bundle = mock.Mock(w_frame_basis=basis)
        init_params = {"epsf_base_raw": np.zeros(1), "w_coeff": np.zeros((1, 4))}
        latest_params = {"epsf_base_raw": np.zeros(1), "w_coeff": np.ones((1, 4)) * 0.1}
        with mock.patch.object(wpp.L, "decoded_epsf_base", side_effect=[grid, grid * 2.0]):
            with tempfile.TemporaryDirectory() as raw:
                out = Path(raw) / "watch_live_epsf.png"
                written = wpp.render_watch_epsf_plot(
                    init_params=init_params,
                    latest_params=latest_params,
                    btjd=btjd,
                    bundle=bundle,
                    out_path=out,
                    run_id="test-run",
                    stage=2,
                    step=100,
                )
                self.assertEqual(written, out)
                self.assertTrue(out.is_file())
                self.assertGreater(out.stat().st_size, 0)

    def test_maybe_render_epsf_stage2_snapshot_writes_once(self):
        from unittest import mock

        from dev.forward_epsf_wcs.diagnostics import watch_params_plot as wpp

        btjd = np.linspace(2400.0, 2401.0, 4)
        basis = np.eye(4)
        bundle = mock.Mock(w_frame_basis=basis)
        grid = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(1, 1, 5, 5)
        init_params = {"epsf_base_raw": np.zeros(1), "w_coeff": np.zeros((1, 4))}
        stage2_params = {"epsf_base_raw": np.zeros(1), "w_coeff": np.ones((1, 4)) * 0.05}

        with tempfile.TemporaryDirectory() as raw:
            artifacts = Path(raw) / "artifacts"
            artifacts.mkdir()
            (artifacts / "params_stage2.npz").write_bytes(b"stub")
            state: dict = {}
            with mock.patch.object(wpp.FIT, "load_params_npz", return_value=stage2_params):
                with mock.patch.object(wpp.L, "decoded_epsf_base", side_effect=[grid, grid * 1.5]):
                    ok = wpp.maybe_render_epsf_stage2_snapshot(
                        artifacts=artifacts,
                        init_params=init_params,
                        bundle=bundle,
                        btjd=btjd,
                        run_id="run",
                        stage=3,
                        state=state,
                    )
            out = artifacts / "plots" / wpp.WATCH_LIVE_EPSF_STAGE2_PLOT_NAME
            self.assertTrue(ok)
            self.assertTrue(out.is_file())
            ok2 = wpp.maybe_render_epsf_stage2_snapshot(
                artifacts=artifacts,
                init_params=init_params,
                bundle=bundle,
                btjd=btjd,
                run_id="run",
                stage=3,
                state=state,
            )
            self.assertFalse(ok2)

    def test_maybe_refresh_params_plots_stage_gating(self):
        from unittest import mock

        from dev.forward_epsf_wcs.diagnostics import watch_params_plot as wpp

        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            artifacts = run_dir / "artifacts"
            artifacts.mkdir()
            (artifacts / "params_latest.npz").write_bytes(b"stub")
            state: dict = {}
            bundle = Path(raw) / "bundle.npz"
            bundle.write_bytes(b"bundle")

            with mock.patch.object(wpp.FB, "load_fit_bundle") as load_bundle:
                with mock.patch.object(wpp.FIT, "load_params_npz", return_value={"wcs_coeff": np.zeros(1)}):
                    with mock.patch.object(wpp, "load_init_params", return_value={"wcs_coeff": np.zeros(1)}):
                        with mock.patch.object(wpp, "load_wcs_history_from_checkpoints", return_value={
                            "stage": np.array([1], dtype=np.int32),
                            "step": np.array([20], dtype=np.int32),
                            "tracks": np.zeros((1, 2, 3)),
                        }):
                            with mock.patch.object(wpp, "coeff_tracks", return_value=np.zeros((2, 3))):
                                wcs_out = artifacts / "plots" / "w.png"
                                wcs_out.parent.mkdir(parents=True, exist_ok=True)
                                wcs_out.write_bytes(b"png")
                                with mock.patch.object(wpp, "render_watch_wcs_plot", return_value=wcs_out) as wcs_plot:
                                    with mock.patch.object(wpp, "render_watch_epsf_plot") as epsf_plot:
                                        load_bundle.return_value = mock.Mock(
                                            meta={"selected_frame_btjd": [1.0, 2.0, 3.0]},
                                        )
                                        ok = wpp.maybe_refresh_params_plots(
                                            run_dir,
                                            run_id="run",
                                            bundle_path=bundle,
                                            remote={"stage": 1, "step": 10},
                                            state=state,
                                        )
            self.assertTrue(ok)
            wcs_plot.assert_called_once()
            epsf_plot.assert_not_called()

            with mock.patch.object(wpp.FB, "load_fit_bundle") as load_bundle:
                with mock.patch.object(wpp.FIT, "load_params_npz", return_value={"wcs_coeff": np.zeros(1)}):
                    with mock.patch.object(wpp, "load_init_params", return_value={"wcs_coeff": np.zeros(1)}):
                        with mock.patch.object(wpp, "load_wcs_history_from_checkpoints", return_value={
                            "stage": np.array([1], dtype=np.int32),
                            "step": np.array([20], dtype=np.int32),
                            "tracks": np.zeros((1, 2, 3)),
                        }):
                            with mock.patch.object(wpp, "coeff_tracks", return_value=np.zeros((2, 3))):
                                wcs_out = artifacts / "plots" / "w.png"
                                epsf_out = artifacts / "plots" / "e.png"
                                wcs_out.parent.mkdir(parents=True, exist_ok=True)
                                wcs_out.write_bytes(b"png")
                                epsf_out.write_bytes(b"png")
                                with mock.patch.object(wpp, "render_watch_wcs_plot", return_value=wcs_out):
                                    with mock.patch.object(wpp, "render_watch_epsf_plot", return_value=epsf_out) as epsf_plot:
                                        load_bundle.return_value = mock.Mock(
                                            meta={"selected_frame_btjd": [1.0, 2.0, 3.0]},
                                        )
                                        state2: dict = {}
                                        ok2 = wpp.maybe_refresh_params_plots(
                                            run_dir,
                                            run_id="run",
                                            bundle_path=bundle,
                                            remote={"stage": 2, "step": 10},
                                            state=state2,
                                        )
            self.assertTrue(ok2)
            epsf_plot.assert_called_once()

            state3: dict = {}
            ok3 = wpp.maybe_refresh_params_plots(
                run_dir,
                run_id="run",
                bundle_path=bundle,
                remote={"stage": 0, "step": 0},
                state=state3,
            )
            self.assertFalse(ok3)


if __name__ == "__main__":
    unittest.main()
