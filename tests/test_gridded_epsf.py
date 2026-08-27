"""Unit tests for photutils gridded ePSF fitting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from photutils.psf import GriddedPSFModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration.stage_params import EpsfParams
from syndiff_pipeline.difference_imaging.stages import gridded_epsf


def _gaussian_stamp(y0: float, x0: float, shape: tuple[int, int], sigma: float = 1.2) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def test_prepare_gaia_matches_starpositioningscript_filter():
    """Reference script: phot_rp_mean_mag < 12.95 before the frame loop."""
    from syndiff_pipeline.difference_imaging.stages.epsf import (
        tess_mag_from_gaia_phot,
    )

    n = 200
    g = np.linspace(8.0, 16.0, n)
    df = pd.DataFrame(
        {
            "ra": np.linspace(100.0, 101.0, n),
            "dec": np.linspace(20.0, 21.0, n),
            "phot_g_mean_mag": g,
            "phot_bp_mean_mag": g + 0.1,
            "phot_rp_mean_mag": g - 0.1,
        }
    )
    expected_tess_mag = tess_mag_from_gaia_phot(g, g + 0.1, g - 0.1)
    params = EpsfParams(tess_mag_max=12.95)
    out = gridded_epsf.prepare_gaia_for_gridded_epsf(df, params)
    assert len(out) == int((expected_tess_mag < 12.95).sum())
    assert out["tess_mag"].max() < 12.95


def test_prepare_gaia_null_mag_max_uses_reference_default():
    """Frozen configs with tess_mag_max: null still apply the 12.95 reference cut."""
    from syndiff_pipeline.difference_imaging.stages.epsf import (
        tess_mag_from_gaia_phot,
    )

    n = 100
    g = np.linspace(8.0, 16.0, n)
    df = pd.DataFrame(
        {
            "ra": np.linspace(100.0, 101.0, n),
            "dec": np.linspace(20.0, 21.0, n),
            "phot_g_mean_mag": g,
            "phot_bp_mean_mag": g + 0.1,
            "phot_rp_mean_mag": g - 0.1,
        }
    )
    expected_tess_mag = tess_mag_from_gaia_phot(g, g + 0.1, g - 0.1)
    params = EpsfParams(tess_mag_max=None)
    out = gridded_epsf.prepare_gaia_for_gridded_epsf(df, params)
    assert len(out) == int((expected_tess_mag < 12.95).sum())


def test_build_gridded_psf_synthetic():
    ny, nx = 128, 128
    image = np.zeros((ny, nx), dtype=np.float64)
    stars = [(32.0, 32.0), (40.0, 35.0), (36.0, 28.0), (34.0, 38.0), (38.0, 32.0)]
    for y, x in stars:
        image += 50.0 * _gaussian_stamp(y, x, (ny, nx))
    image += np.random.default_rng(0).normal(0, 0.05, image.shape)

    gaia = pd.DataFrame(
        {
            "ra": [100.0] * len(stars),
            "dec": [20.0] * len(stars),
            "x": [s[1] for s in stars],
            "y": [s[0] for s in stars],
            "phot_g_mean_mag": [10.0] * len(stars),
            "phot_bp_mean_mag": [10.2] * len(stars),
            "phot_rp_mean_mag": [9.8] * len(stars),
        }
    )
    params = EpsfParams(tile_nx=1, tile_ny=1, psf_size=5, min_stars_per_tile=3)
    gaia = gridded_epsf.prepare_gaia_for_gridded_epsf(gaia, params)
    model, grid_xypos, stack, _n_stars = gridded_epsf.build_gridded_psf_for_frame(
        image, gaia, params
    )
    assert model is not None
    assert stack is not None
    assert stack.ndim == 3
    assert len(grid_xypos) == 1
    assert isinstance(model, GriddedPSFModel)


def test_gridded_npz_roundtrip(tmp_path):
    # photutils requires >= 4 grid nodes (not 2 or 3)
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    path = str(tmp_path / "tess123_gridded_epsf.npz")
    gridded_epsf.save_gridded_epsf_npz(path, stack, grid_xypos, oversampling=2)
    model = gridded_epsf.load_gridded_psf_model(path)
    assert model.data.shape == (4, 11, 11)
    assert len(model.grid_xypos) == 4


def test_gridded_npz_roundtrip_with_n_stars(tmp_path):
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    path = str(tmp_path / "tess124_gridded_epsf.npz")
    gridded_epsf.save_gridded_epsf_npz(
        path, stack, grid_xypos, oversampling=2, n_stars=[3, 7, 0, 12]
    )
    z = np.load(path, allow_pickle=False)
    try:
        assert list(z["n_stars"]) == [3, 7, 0, 12]
    finally:
        z.close()


def test_gridded_npz_roundtrip_without_n_stars_omits_field(tmp_path):
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    path = str(tmp_path / "tess125_gridded_epsf.npz")
    gridded_epsf.save_gridded_epsf_npz(path, stack, grid_xypos, oversampling=2)
    z = np.load(path, allow_pickle=False)
    try:
        assert "n_stars" not in z.files
    finally:
        z.close()


def test_write_gridded_epsf_frame_plot_title_includes_star_count(tmp_path):
    from syndiff_pipeline.difference_imaging.support import plot as plot_mod

    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    npz_path = str(tmp_path / "tess126_gridded_epsf.npz")
    gridded_epsf.save_gridded_epsf_npz(
        npz_path, stack, grid_xypos, oversampling=2, n_stars=[3, 7, 0, 12]
    )

    captured_titles: list[str] = []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    orig_set_title = matplotlib.axes.Axes.set_title

    def _capture_set_title(self, label, *a, **k):
        captured_titles.append(label)
        return orig_set_title(self, label, *a, **k)

    import matplotlib.axes as _maxes

    _maxes.Axes.set_title = _capture_set_title
    try:
        png_path = str(tmp_path / "epsf_r1_tess126.png")
        out = plot_mod.write_gridded_epsf_frame_plot(npz_path, png_path)
    finally:
        _maxes.Axes.set_title = orig_set_title

    assert out == png_path
    assert os.path.isfile(png_path)
    assert any("N=3" in t for t in captured_titles)
    assert any("N=7" in t for t in captured_titles)
    assert any("N=0" in t for t in captured_titles)
    assert any("N=12" in t for t in captured_titles)


def test_fit_epsf_section_returns_stamp():
    ny, nx = 64, 64
    image = np.zeros((ny, nx))
    xs = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
    ys = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
    for x, y in zip(xs, ys):
        image += 30 * _gaussian_stamp(y, x, (ny, nx))
    stars_tbl = Table()
    stars_tbl["x"] = xs
    stars_tbl["y"] = ys
    stamp = gridded_epsf.fit_epsf_section(
        image, stars_tbl, extract_size=15, oversampling=2, maxiters=5
    )
    assert stamp is not None
    assert stamp.ndim == 2
    assert np.isfinite(stamp).all()


def test_fit_epsf_section_star_usage_out_reports_all_candidates_used():
    """No excluded stars in a clean, well-separated fit -- every candidate
    ends up in used_xy, none in excluded_xy, and no positions are lost."""
    ny, nx = 64, 64
    image = np.zeros((ny, nx))
    xs = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
    ys = np.array([20.0, 22.0, 24.0, 26.0, 28.0])
    for x, y in zip(xs, ys):
        image += 30 * _gaussian_stamp(y, x, (ny, nx))
    stars_tbl = Table()
    stars_tbl["x"] = xs
    stars_tbl["y"] = ys
    usage: dict = {}
    stamp = gridded_epsf.fit_epsf_section(
        image, stars_tbl, extract_size=15, oversampling=2, maxiters=5, star_usage_out=usage
    )
    assert stamp is not None
    assert len(usage["used_xy"]) + len(usage["excluded_xy"]) == len(stars_tbl)
    all_xy = {tuple(xy) for xy in usage["used_xy"]} | {tuple(xy) for xy in usage["excluded_xy"]}
    assert all_xy == {(float(x), float(y)) for x, y in zip(xs, ys)}


def test_is_valid_gridded_epsf_npz(tmp_path):
    path = str(tmp_path / "tess123_gridded_epsf.npz")
    assert gridded_epsf._is_valid_gridded_epsf_npz(path) is False
    (tmp_path / "tess123_gridded_epsf.npz").write_bytes(b"not-npz")
    assert gridded_epsf._is_valid_gridded_epsf_npz(path) is False
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    gridded_epsf.save_gridded_epsf_npz(path, stack, grid_xypos, oversampling=2)
    assert gridded_epsf._is_valid_gridded_epsf_npz(path) is True


def test_filter_stars_geometric_mask():
    mask = np.ones((40, 40), dtype=bool)
    mask[0:12, 0:12] = False
    stars = Table()
    stars["x"] = [20.0, 5.0]
    stars["y"] = [20.0, 5.0]
    kept = gridded_epsf._filter_stars_geometric_mask(stars, mask, box_radius=7)
    assert len(kept) == 1
    assert float(kept["x"][0]) == 5.0


def test_epsf_defaults_match_reference():
    params = EpsfParams()
    assert params.tile_nx == 5
    assert params.tile_ny == 5
    assert params.psf_size == 15
    assert params.epsf_stamp_border_crop == 8
    assert params.epsf_smoothing_kernel == "quadratic"


def test_stamp_border_crop_applied(monkeypatch):
    ny, nx = 64, 64
    image = np.zeros((ny, nx))
    gaia = pd.DataFrame(
        {
            "ra": [100.0],
            "dec": [20.0],
            "x": [32.0],
            "y": [32.0],
            "phot_g_mean_mag": [10.0],
            "phot_bp_mean_mag": [10.1],
            "phot_rp_mean_mag": [9.9],
        }
    )
    stamp = np.ones((21, 21), dtype=np.float64)

    def _fake_fit(*_a, **_k):
        return stamp

    monkeypatch.setattr(gridded_epsf, "fit_epsf_section", _fake_fit)
    params = EpsfParams(
        tile_nx=1,
        tile_ny=1,
        psf_size=5,
        min_stars_per_tile=1,
        epsf_stamp_border_crop=8,
    )
    gaia = gridded_epsf.prepare_gaia_for_gridded_epsf(gaia, params)
    _model, _centers, stack, _n_stars = gridded_epsf.build_gridded_psf_for_frame(
        image, gaia, params
    )
    assert stack is not None
    assert stack.shape == (1, 5, 5)


def _worker_wcs_context(monkeypatch, tmp_path, stem: str = "tess111"):
    gaia_base = pd.DataFrame(
        {"ra": [100.0], "dec": [20.0], "phot_rp_mean_mag": [10.0]}
    )

    def _fake_gaia(*_a, **_k):
        return pd.DataFrame({"x": [10.0], "y": [10.0], "phot_rp_mean_mag": [10.0]})

    monkeypatch.setattr(
        "syndiff_pipeline.common.wcs_grouping.gaia_science_xy_for_frame",
        _fake_gaia,
    )
    return gaia_base, {
        "ffi_list_df": pd.DataFrame(),
        "science_bounds": {"x_min": 0, "y_min": 0, "shape": (32, 32)},
        "ffi_path_by_stem": {stem: str(tmp_path / "ffi.fits")},
        "data_root": str(tmp_path / "data"),
        "sck": (20, 1, 1),
        "epsf_label": "epsf_r1",
    }


def _scc_epsf_npz_path(tmp_path, stem: str):
    from syndiff_pipeline.difference_imaging.orchestration import diff_store

    path = diff_store.resolve_diff_write_path(
        data_root=str(tmp_path / "data"),
        sck=(20, 1, 1),
        kind="epsf",
        stage_label="epsf_r1",
        ffi_stem=stem,
        label="epsf_r1",
        params=EpsfParams(),
        suffix=".npz",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_centroids_defaults_and_mag_filter():
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
        CentroidsParams,
    )
    from syndiff_pipeline.difference_imaging.stages import centroids

    from syndiff_pipeline.difference_imaging.stages.epsf import (
        tess_mag_from_gaia_phot,
    )

    params = CentroidsParams()
    assert params.aperture_radius == 4.0
    assert params.psf_grouper_min_separation == 10.0
    assert params.centroids_max_group_size == 5
    assert params.tess_mag_min == 7.5
    g = np.array([7.0, 7.6, 12.0, 13.0])
    df = pd.DataFrame(
        {
            "phot_g_mean_mag": g,
            "phot_bp_mean_mag": g + 0.1,
            "phot_rp_mean_mag": g - 0.1,
        }
    )
    expected_tess_mag = tess_mag_from_gaia_phot(g, g + 0.1, g - 0.1)
    out = centroids._filter_gaia_for_centroids(df, params)
    kept = (expected_tess_mag > 7.5) & (expected_tess_mag < 12.95)
    assert list(out["tess_mag"]) == list(expected_tess_mag[kept])


def test_centroids_attach_gaia_metadata():
    from astropy.table import Table
    from syndiff_pipeline.difference_imaging.stages import centroids

    gaia = pd.DataFrame(
        {
            "source_id": [101, 102],
            "ra": [10.0, 11.0],
            "dec": [20.0, 21.0],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "phot_rp_mean_mag": [10.0, 11.0],
        }
    )
    phot = Table(
        {
            "x_init": [1.0, 2.0],
            "y_init": [3.0, 4.0],
            "flux_fit": [0.1, 0.2],
        }
    )
    out = centroids._attach_gaia_metadata(phot, gaia)
    df = out.to_pandas()
    assert list(df["source_id"]) == [101, 102]
    assert "ra" in df.columns


def test_split_oversized_group_caps_size_and_keeps_all_points():
    from syndiff_pipeline.difference_imaging.stages import centroids

    rng = np.random.default_rng(0)
    n = 23
    # Tight cluster: every point within 2px of its neighbors, so the base
    # SourceGrouper(min_separation=10) would put all 23 in one group.
    x = rng.uniform(0, 2, n)
    y = rng.uniform(0, 2, n)
    pieces = centroids._split_oversized_group(x, y, max_group_size=5, start_sep=10.0)
    all_idx = np.concatenate(pieces)
    assert sorted(all_idx.tolist()) == list(range(n))  # no drops, no duplicates
    assert all(len(p) <= 5 for p in pieces)


def test_split_oversized_group_noop_when_already_small():
    from syndiff_pipeline.difference_imaging.stages import centroids

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    pieces = centroids._split_oversized_group(x, y, max_group_size=5, start_sep=10.0)
    assert len(pieces) == 1
    assert sorted(pieces[0].tolist()) == [0, 1, 2]


def test_split_oversized_group_falls_back_on_coincident_points():
    from syndiff_pipeline.difference_imaging.stages import centroids

    # 12 exactly-coincident points -- shrinking min_separation never
    # separates them, so this must hit the positional-chunking fallback
    # and still terminate, still keep every point.
    n = 12
    x = np.zeros(n)
    y = np.zeros(n)
    pieces = centroids._split_oversized_group(x, y, max_group_size=5, start_sep=10.0)
    all_idx = np.concatenate(pieces)
    assert sorted(all_idx.tolist()) == list(range(n))
    assert all(len(p) <= 5 for p in pieces)


def test_capped_source_grouper_caps_group_size():
    from syndiff_pipeline.difference_imaging.stages import centroids

    rng = np.random.default_rng(1)
    n = 30
    x = rng.uniform(0, 3, n)
    y = rng.uniform(0, 3, n)
    grouper = centroids._CappedSourceGrouper(min_separation=10.0, max_group_size=5)
    ids = grouper(x, y)
    assert len(ids) == n
    counts = pd.Series(ids).value_counts()
    assert counts.max() <= 5
    assert counts.sum() == n


def test_centroids_writes_inline_debug_residual_without_refit(tmp_path, monkeypatch):
    """A debug-selected frame writes its residual FITS from the fit already
    computed -- _photometry_one_frame must be called exactly once per frame,
    never a second time for the debug output."""
    from astropy.io import fits
    from astropy.table import Table
    from syndiff_pipeline.common import wcs_grouping
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
        CentroidsParams,
    )
    from syndiff_pipeline.difference_imaging.stages import centroids
    from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
        GriddedEpsfCatalog,
    )
    from syndiff_pipeline.difference_imaging.support.plot import (
        centroids_residual_fits_path,
    )

    stem = "tess111"
    diff_path = tmp_path / f"{stem}_hp_d.fits"
    fits.writeto(diff_path, np.zeros((16, 16), dtype=np.float32), overwrite=True)

    output_dir = str(tmp_path / "centroids_out")
    debug_plot_dir = str(tmp_path / "debug_plots")

    catalog = GriddedEpsfCatalog(workspace_dir=str(tmp_path), index={stem: "dummy.npz"})
    monkeypatch.setattr(GriddedEpsfCatalog, "load_model", lambda self, ffi_stem: object())

    def _fake_gaia_science_xy(gaia_df, ffi_path, ffi_list_df, science_bounds):
        out = gaia_df.copy()
        out["x"] = [8.0]
        out["y"] = [8.0]
        return out

    monkeypatch.setattr(
        wcs_grouping, "gaia_science_xy_for_frame", _fake_gaia_science_xy
    )

    call_count = {"n": 0}

    class _FakePsfPhot:
        def make_residual_image(self, diff_img):
            return np.zeros_like(diff_img)

    def _fake_photometry_one_frame(diff_img, model, gaia_df, params):
        call_count["n"] += 1
        table = Table({"x_init": [8.0], "y_init": [8.0], "flux_fit": [1.0]})
        return table, _FakePsfPhot()

    monkeypatch.setattr(centroids, "_photometry_one_frame", _fake_photometry_one_frame)

    gaia_base = pd.DataFrame(
        {
            "ra": [10.0],
            "dec": [1.0],
            "phot_g_mean_mag": [10.0],
            "phot_bp_mean_mag": [10.1],
            "phot_rp_mean_mag": [9.9],
        }
    )
    params = CentroidsParams()
    ffi_list_df = pd.DataFrame({"stem": [stem]})
    science_bounds = {"x_min": 0, "y_min": 0, "shape": (16, 16)}
    ffi_path_by_stem = {stem: str(tmp_path / f"{stem}.fits")}

    centroids._init_centroids_worker(
        gaia_base,
        catalog,
        params,
        output_dir,
        skip_existing=False,
        sck=None,
        data_root=None,
        centroids_label="centroids_r1",
        ffi_list_df=ffi_list_df,
        science_bounds=science_bounds,
        ffi_path_by_stem=ffi_path_by_stem,
        debug_stems=frozenset({stem}),
        debug_plot_dir=debug_plot_dir,
    )

    result = centroids._centroids_one_frame_task(0, str(diff_path))
    assert result[1] == stem
    assert result[2] is True  # ok
    assert call_count["n"] == 1  # fit ran exactly once -- no refit for debug output

    residual_path = centroids_residual_fits_path(debug_plot_dir, "centroids_r1", stem)
    assert os.path.isfile(residual_path)


def test_centroids_no_debug_residual_when_stem_not_selected(tmp_path, monkeypatch):
    """Frames outside debug_stems must not get a residual FITS at all."""
    from astropy.io import fits
    from astropy.table import Table
    from syndiff_pipeline.common import wcs_grouping
    from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
        CentroidsParams,
    )
    from syndiff_pipeline.difference_imaging.stages import centroids
    from syndiff_pipeline.difference_imaging.stages.gridded_epsf import (
        GriddedEpsfCatalog,
    )
    from syndiff_pipeline.difference_imaging.support.plot import (
        centroids_residual_fits_path,
    )

    stem = "tess111"
    diff_path = tmp_path / f"{stem}_hp_d.fits"
    fits.writeto(diff_path, np.zeros((16, 16), dtype=np.float32), overwrite=True)

    output_dir = str(tmp_path / "centroids_out")
    debug_plot_dir = str(tmp_path / "debug_plots")

    catalog = GriddedEpsfCatalog(workspace_dir=str(tmp_path), index={stem: "dummy.npz"})
    monkeypatch.setattr(GriddedEpsfCatalog, "load_model", lambda self, ffi_stem: object())

    def _fake_gaia_science_xy(gaia_df, ffi_path, ffi_list_df, science_bounds):
        out = gaia_df.copy()
        out["x"] = [8.0]
        out["y"] = [8.0]
        return out

    monkeypatch.setattr(
        wcs_grouping, "gaia_science_xy_for_frame", _fake_gaia_science_xy
    )

    class _FakePsfPhot:
        def make_residual_image(self, diff_img):
            return np.zeros_like(diff_img)

    def _fake_photometry_one_frame(diff_img, model, gaia_df, params):
        table = Table({"x_init": [8.0], "y_init": [8.0], "flux_fit": [1.0]})
        return table, _FakePsfPhot()

    monkeypatch.setattr(centroids, "_photometry_one_frame", _fake_photometry_one_frame)

    gaia_base = pd.DataFrame(
        {
            "ra": [10.0],
            "dec": [1.0],
            "phot_g_mean_mag": [10.0],
            "phot_bp_mean_mag": [10.1],
            "phot_rp_mean_mag": [9.9],
        }
    )
    params = CentroidsParams()
    ffi_list_df = pd.DataFrame({"stem": [stem]})
    science_bounds = {"x_min": 0, "y_min": 0, "shape": (16, 16)}
    ffi_path_by_stem = {stem: str(tmp_path / f"{stem}.fits")}

    centroids._init_centroids_worker(
        gaia_base,
        catalog,
        params,
        output_dir,
        skip_existing=False,
        sck=None,
        data_root=None,
        centroids_label="centroids_r1",
        ffi_list_df=ffi_list_df,
        science_bounds=science_bounds,
        ffi_path_by_stem=ffi_path_by_stem,
        debug_stems=frozenset(),  # nothing selected
        debug_plot_dir=debug_plot_dir,
    )

    result = centroids._centroids_one_frame_task(0, str(diff_path))
    assert result[2] is True

    residual_path = centroids_residual_fits_path(debug_plot_dir, "centroids_r1", stem)
    assert not os.path.isfile(residual_path)


def test_fit_one_frame_skips_existing_npz(tmp_path, monkeypatch):
    output_dir = str(tmp_path)
    stem = "tess111"
    path = _scc_epsf_npz_path(tmp_path, stem)
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    gridded_epsf.save_gridded_epsf_npz(path, stack, grid_xypos, oversampling=2)

    calls: list[int] = []

    def _boom(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("should not fit when npz exists")

    monkeypatch.setattr(gridded_epsf, "build_gridded_psf_for_frame", _boom)
    gaia_base, wcs_ctx = _worker_wcs_context(monkeypatch, tmp_path, stem)
    gridded_epsf._init_gridded_epsf_worker(
        gaia_base,
        EpsfParams(),
        output_dir,
        None,
        skip_existing=True,
        **wcs_ctx,
    )
    result = gridded_epsf._fit_one_frame_task(
        0, str(tmp_path / f"{stem}_hp_d.fits")
    )
    assert result[2] is True
    assert result[5] is True
    assert calls == []


def test_fit_one_frame_refits_when_skip_existing_disabled(tmp_path, monkeypatch):
    output_dir = str(tmp_path)
    stem = "tess111"
    path = _scc_epsf_npz_path(tmp_path, stem)
    stack = np.ones((4, 11, 11), dtype=np.float64)
    grid_xypos = [(32.0, 32.0), (96.0, 32.0), (32.0, 96.0), (96.0, 96.0)]
    gridded_epsf.save_gridded_epsf_npz(path, stack, grid_xypos, oversampling=2)

    calls: list[int] = []

    def _fake_fit(*_args, **_kwargs):
        calls.append(1)
        return None, grid_xypos, None, []

    monkeypatch.setattr(gridded_epsf, "build_gridded_psf_for_frame", _fake_fit)
    gaia_base, wcs_ctx = _worker_wcs_context(monkeypatch, tmp_path, stem)
    gridded_epsf._init_gridded_epsf_worker(
        gaia_base,
        EpsfParams(),
        output_dir,
        None,
        skip_existing=False,
        **wcs_ctx,
    )
    ny, nx = 32, 32
    diff_path = tmp_path / f"{stem}_hp_d.fits"
    from astropy.io import fits

    fits.writeto(diff_path, np.zeros((ny, nx), dtype=np.float32), overwrite=True)
    result = gridded_epsf._fit_one_frame_task(0, str(diff_path))
    assert result[5] is False
    assert calls == [1]
