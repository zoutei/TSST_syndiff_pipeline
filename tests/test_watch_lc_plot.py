from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class WatchLcPlotTests(unittest.TestCase):
    def test_spatial_grid_star_indices_orders_low_xy_bottom_left(self):
        from dev.forward_epsf_wcs import aperture_correction as AC
        from dev.forward_epsf_wcs.diagnostics.watch_lc_plot import spatial_grid_star_indices

        n_t = 5
        table = AC.PrimaryFluxTable(
            flux=np.ones((4, n_t)),
            sigma_f=np.ones((4, n_t)),
            x=np.array([[0, 0, 0, 0, 0], [10, 10, 10, 10, 10], [0, 0, 0, 0, 0], [10, 10, 10, 10, 10]]),
            y=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [10, 10, 10, 10, 10], [10, 10, 10, 10, 10]]),
            active=np.ones((4, n_t), dtype=bool),
            group_index=np.arange(4),
            slot_index=np.zeros(4, dtype=int),
            star_index=np.arange(4),
            tess_mag=np.array([10.0, 11.0, 12.0, 13.0]),
        )
        positions = spatial_grid_star_indices(table, np.arange(4), grid_size=2)
        self.assertEqual(len(positions), 4)
        # lowest x,y (star 0) should land bottom-left (row=1, col=0)
        bottom_left = next(p for p in positions if p[2] == 0)
        self.assertEqual(bottom_left[:2], (1, 0))

    def test_render_watch_ac_w0_plot_writes_png(self):
        from dev.forward_epsf_wcs import aperture_correction as AC
        from dev.forward_epsf_wcs.diagnostics.watch_lc_plot import render_watch_ac_w0_plot

        btjd = np.linspace(2400.0, 2401.0, 8)
        result = AC.ApertureResult(
            A_coeff=np.zeros((2, 2, 3)),
            A_node=np.ones((2, 2, btjd.size)) * (1.0 + 0.01 * np.sin(np.linspace(0, 1, btjd.size))),
            A_star=np.ones((1, btjd.size)),
            F=np.array([1.0]),
            flux_corr=np.ones((1, btjd.size)),
            frame_basis=np.eye(btjd.size, 3),
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 1.0]),
            n_iter=1,
            rms_resid=0.01,
        )
        w0 = 0.02 * np.sin(np.linspace(0, 1.5, btjd.size))
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "watch_live_ac_w0.png"
            written = render_watch_ac_w0_plot(
                btjd=btjd,
                result=result,
                w0=w0,
                out_path=out,
                run_id="test-run",
                stage=3,
                step=100,
            )
            self.assertEqual(written, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)

    def test_render_watch_lc_grid_plot_writes_png(self):
        from dev.forward_epsf_wcs import aperture_correction as AC
        from dev.forward_epsf_wcs.diagnostics.watch_lc_plot import render_watch_lc_grid_plot

        btjd = np.linspace(2400.0, 2401.0, 6)
        n_stars = 4
        flux = np.ones((n_stars, btjd.size)) * np.linspace(0.98, 1.02, btjd.size)
        corr = flux * 0.995
        table = AC.PrimaryFluxTable(
            flux=flux,
            sigma_f=np.full((n_stars, btjd.size), 0.01),
            x=np.array([[0, 0, 0, 0, 0, 0], [5, 5, 5, 5, 5, 5], [0, 0, 0, 0, 0, 0], [5, 5, 5, 5, 5, 5]]),
            y=np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [5, 5, 5, 5, 5, 5], [5, 5, 5, 5, 5, 5]]),
            active=np.ones((n_stars, btjd.size), dtype=bool),
            group_index=np.arange(n_stars),
            slot_index=np.zeros(n_stars, dtype=int),
            star_index=np.arange(n_stars),
            tess_mag=np.array([9.5, 10.0, 10.5, 11.0]),
            btjd=btjd,
        )
        chi2_red = np.full((n_stars, btjd.size), 650.0)
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "watch_live_lc_grid.png"
            written = render_watch_lc_grid_plot(
                table=table,
                flux_corr=corr,
                btjd=btjd,
                chi2_red=chi2_red,
                out_path=out,
                run_id="test-run",
                stage=3,
                step=50,
                grid_size=2,
            )
            self.assertEqual(written, out)
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
