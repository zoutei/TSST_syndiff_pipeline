"""Unit tests for photutils gridded ePSF fitting."""

from __future__ import annotations

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
    n = 200
    df = pd.DataFrame(
        {
            "ra": np.linspace(100.0, 101.0, n),
            "dec": np.linspace(20.0, 21.0, n),
            "phot_rp_mean_mag": np.linspace(8.0, 16.0, n),
        }
    )
    params = EpsfParams(mag_max_rp=12.95)
    out = gridded_epsf.prepare_gaia_for_gridded_epsf(df, params)
    assert len(out) == int((df["phot_rp_mean_mag"] < 12.95).sum())
    assert out["phot_rp_mean_mag"].max() < 12.95


def test_prepare_gaia_null_mag_max_uses_reference_default():
    """Frozen configs with mag_max_rp: null still apply the 12.95 reference cut."""
    n = 100
    df = pd.DataFrame(
        {
            "ra": np.linspace(100.0, 101.0, n),
            "dec": np.linspace(20.0, 21.0, n),
            "phot_rp_mean_mag": np.linspace(8.0, 16.0, n),
        }
    )
    params = EpsfParams(mag_max_rp=None)
    out = gridded_epsf.prepare_gaia_for_gridded_epsf(df, params)
    assert len(out) == int((df["phot_rp_mean_mag"] < 12.95).sum())


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
    model, grid_xypos, stack = gridded_epsf.build_gridded_psf_for_frame(
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
            "phot_rp_mean_mag": [10.0],
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
    _model, _centers, stack = gridded_epsf.build_gridded_psf_for_frame(
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

    params = CentroidsParams()
    assert params.aperture_radius == 4.0
    assert params.psf_grouper_min_separation == 10.0
    assert params.mag_min_rp == 7.5
    df = pd.DataFrame({"phot_rp_mean_mag": [7.0, 7.6, 12.0, 13.0]})
    out = centroids._filter_gaia_for_centroids(df, params)
    assert list(out["phot_rp_mean_mag"]) == [7.6, 12.0]


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
        return None, grid_xypos, None

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
