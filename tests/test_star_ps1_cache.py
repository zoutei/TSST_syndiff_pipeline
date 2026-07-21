"""Tests for star PS1 zarr cache."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.star.context import StarEventContext
from syndiff_pipeline.star.ps1_cache import (
    ensure_skycell_cached,
    load_skycell_bands_for_source,
    load_skycell_bands_from_cache,
    ps1_skycells_zarr_paths,
)


def _ctx(tmp: Path) -> StarEventContext:
    return StarEventContext(
        target=Target(
            sector=20,
            camera=3,
            ccd=2,
            target_ra=120.0,
            target_dec=30.0,
            target_name="s20_astrometry",
        ),
        event_dir=str(tmp / "event"),
        workspace_root=str(tmp / "workspace"),
        data_root=str(tmp / "data"),
        cluster_job_path=str(tmp / "event" / "cluster_template_job.json"),
        cluster_job={},
        crop_bounds={"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20, "shape": (20, 20)},
        mapping_dir=str(tmp / "data" / "skycell_pixel_mapping"),
        mapping_csv=str(tmp / "data" / "mapping.csv"),
        master_mapping_fits=str(tmp / "data" / "master.fits.fz"),
        gaia_catalog_path=str(tmp / "data" / "gaia.csv"),
        templates_dir=str(tmp / "templates"),
        reference_ffi_path=str(tmp / "ref.fits"),
        sector=20,
        camera=3,
        ccd=2,
        baseline_workspace_dir=str(tmp / "event" / "ws"),
        baseline_diffs_label="hp_d",
        baseline_diffs_dir=str(tmp / "event" / "ws" / "hp_d"),
        baseline_convolved_dir=str(tmp / "event" / "ws" / "hp_c"),
        baseline_phot_bkg_dir=str(tmp / "event" / "ws" / "ks_b_s"),
        baseline_phot_bkg_label="ks_b_s",
        baseline_kernels_dir=str(tmp / "event" / "ws" / "hp_d_kernels"),
    )


class TestStarPs1Cache(unittest.TestCase):
    def test_zarr_paths_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx(Path(tmpdir))
            zarr_path, lock_file = ps1_skycells_zarr_paths(ctx)
            self.assertEqual(
                zarr_path,
                Path(tmpdir) / "data" / "ps1_skycells_zarr" / "ps1_skycells.zarr",
            )
            self.assertEqual(
                lock_file,
                Path(tmpdir) / "data" / "ps1_skycells_zarr" / "ps1_skycells.zarr.lock",
            )

    def test_ensure_skycell_cached_downloads_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx(Path(tmpdir))
            skycell = "skycell.2582.071"
            fake_root = MagicMock()
            fake_writer = MagicMock()

            with (
                patch(
                    "syndiff_pipeline.star.ps1_cache.initialize_zarr_store",
                    return_value=fake_root,
                ) as mock_init,
                patch(
                    "syndiff_pipeline.star.ps1_cache.count_complete_arrays",
                    side_effect=[0, 12],
                ),
                patch(
                    "syndiff_pipeline.star.ps1_cache.download_and_store_skycell",
                ) as mock_download,
                patch(
                    "syndiff_pipeline.star.ps1_cache.ZarrWriter",
                    return_value=fake_writer,
                ),
            ):
                path1 = ensure_skycell_cached(skycell, ctx)
                path2 = ensure_skycell_cached(skycell, ctx)

            self.assertEqual(path1, path2)
            self.assertEqual(mock_init.call_count, 2)
            mock_download.assert_called_once()

    def test_load_skycell_bands_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx(Path(tmpdir))
            skycell = "skycell.2582.071"
            payload = ({"r": "img"}, {"r": "mask"}, {}, {"r": "hdr"}, {})

            with (
                patch(
                    "syndiff_pipeline.star.ps1_cache.ensure_skycell_cached",
                    return_value=Path(tmpdir) / "cache.zarr",
                ),
                patch("syndiff_pipeline.star.ps1_cache.zarr.open", return_value=MagicMock()),
                patch(
                    "syndiff_pipeline.star.ps1_cache._load_skycell_from_store",
                    return_value=payload,
                ),
            ):
                result = load_skycell_bands_from_cache(skycell, ctx)

            self.assertEqual(result, payload)

    def test_load_skycell_bands_from_stream_never_touches_zarr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx(Path(tmpdir))
            skycell = "skycell.2582.071"
            payload = ({"r": "img"}, {"r": "mask"}, {}, {"r": "hdr"}, {})

            with (
                patch(
                    "syndiff_pipeline.star.ps1_cache.fetch_skycell_bands_masks_and_headers",
                    return_value=payload,
                ) as mock_fetch,
                patch("syndiff_pipeline.star.ps1_cache.ensure_skycell_cached") as mock_cache,
            ):
                result = load_skycell_bands_for_source(
                    skycell,
                    ctx,
                    ps1_source="stream",
                )

            self.assertEqual(result, payload)
            mock_fetch.assert_called_once_with(skycell)
            mock_cache.assert_not_called()

    def test_zarr_local_only_fails_on_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx(Path(tmpdir))
            skycell = "skycell.2582.071"
            with self.assertRaises(FileNotFoundError):
                load_skycell_bands_for_source(
                    skycell,
                    ctx,
                    ps1_source="zarr_local_only",
                )


if __name__ == "__main__":
    unittest.main()
