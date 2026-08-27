"""Unit tests for orbit-binned gridded ePSF (gridded_epsf_orbit.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.difference_imaging.orchestration import provenance_glue as pg
from syndiff_pipeline.difference_imaging.orchestration.stage_params import (
    EpsfParams,
    parse_epsf,
)
from syndiff_pipeline.difference_imaging.stages import gridded_epsf, gridded_epsf_orbit as geo


# ── EpsfParams / parse_epsf ─────────────────────────────────────────────────


def test_epsf_mode_defaults_to_orbit_binned():
    params = EpsfParams()
    assert params.epsf_mode == "orbit_binned"
    assert params.epsf_per_orbit == 5
    assert params.epsf_frames_per_anchor == 20
    assert params.epsf_stack_before_fit is True


def test_parse_epsf_accepts_per_frame():
    params = parse_epsf({"kind": "epsf", "epsf_mode": "per_frame"}, 0)
    assert params.epsf_mode == "per_frame"


def test_parse_epsf_rejects_bad_mode():
    with pytest.raises(ValueError):
        parse_epsf({"kind": "epsf", "epsf_mode": "bogus"}, 0)


def test_parse_epsf_rejects_bad_orbit_params():
    with pytest.raises(ValueError):
        parse_epsf({"kind": "epsf", "epsf_per_orbit": 0}, 0)
    with pytest.raises(ValueError):
        parse_epsf({"kind": "epsf", "epsf_anchor_edge_fraction": 0.6}, 0)


# ── Anchor placement ─────────────────────────────────────────────────────────


def test_anchor_target_phases_degenerate_single_anchor():
    phases = geo.anchor_target_phases(1, 0.12, 3.0)
    assert phases.tolist() == [0.5]


def test_anchor_target_phases_zero_anchors():
    assert geo.anchor_target_phases(0, 0.12, 3.0).size == 0


def test_anchor_target_phases_canonical_five_places_per_orbit():
    """Production default (epsf_per_orbit=5): one in the middle, two at the
    very ends, two at edge_fraction in from each end."""
    phases = geo.anchor_target_phases(5, edge_fraction=0.12, edge_boost=3.0)
    assert phases.tolist() == pytest.approx([0.0, 0.12, 0.5, 0.88, 1.0])


def test_anchor_target_phases_two_anchors_are_both_endpoints():
    phases = geo.anchor_target_phases(2, edge_fraction=0.12, edge_boost=3.0)
    assert phases.tolist() == [0.0, 1.0]


def test_anchor_target_phases_general_odd_n_keeps_endpoints_and_midpoint():
    n = 7
    phases = geo.anchor_target_phases(n, edge_fraction=0.12, edge_boost=3.0)
    assert phases.shape == (n,)
    assert np.all(np.diff(phases) > 0)
    assert phases[0] == pytest.approx(0.0)
    assert phases[-1] == pytest.approx(1.0)
    assert phases[n // 2] == pytest.approx(0.5)
    # Symmetric about the midpoint.
    assert np.allclose(phases, 1.0 - phases[::-1])


# ── Anchor frame selection ───────────────────────────────────────────────────


def test_select_anchor_frames_excludes_quality_bad_window():
    btjds = np.linspace(0, 10, 50)
    quality_ok = np.ones(50, dtype=bool)
    quality_ok[10:13] = False
    anchors = geo.select_anchor_frames(
        btjds=btjds,
        quality_ok=quality_ok,
        n_anchors=5,
        frames_per_anchor=8,
        edge_fraction=0.12,
        edge_boost=3.0,
        max_expand=20,
    )
    assert len(anchors) == 5
    for a in anchors:
        assert all(quality_ok[p] for p in a.window_frame_pos)
        assert not (set(a.window_frame_pos) & {10, 11, 12})


def test_select_anchor_frames_window_bounded_by_max_expand():
    n = 40
    btjds = np.linspace(0, 10, n)
    quality_ok = np.zeros(n, dtype=bool)
    quality_ok[19] = True  # only the anchor's own frame is good
    anchors = geo.select_anchor_frames(
        btjds=btjds,
        quality_ok=quality_ok,
        n_anchors=1,
        frames_per_anchor=8,
        edge_fraction=0.12,
        edge_boost=3.0,
        max_expand=2,
    )
    assert len(anchors) == 1
    # Only radius<=2 around the anchor is searched -- can't reach 8 frames.
    assert len(anchors[0].window_frame_pos) < 8
    assert set(anchors[0].window_frame_pos).issubset(set(range(17, 22)))


def test_select_anchor_frames_degenerate_short_orbit_reuses_all_frames():
    btjds = np.linspace(0, 1, 3)
    quality_ok = np.ones(3, dtype=bool)
    anchors = geo.select_anchor_frames(
        btjds=btjds,
        quality_ok=quality_ok,
        n_anchors=5,
        frames_per_anchor=8,
        edge_fraction=0.12,
        edge_boost=3.0,
        max_expand=20,
    )
    # select_anchor_frames caps n_anchors at the number of available frames
    # (a second safety net independent of the orchestration-level F6 relax
    # logic in fit_gridded_epsf_orbit_binned) and must not crash.
    assert len(anchors) == 3
    for a in anchors:
        assert set(a.window_frame_pos).issubset({0, 1, 2})


def test_select_anchor_frames_empty_input():
    assert geo.select_anchor_frames(
        btjds=np.array([]),
        quality_ok=np.array([], dtype=bool),
        n_anchors=5,
        frames_per_anchor=8,
        edge_fraction=0.12,
        edge_boost=3.0,
        max_expand=20,
    ) == []


def test_select_anchor_frames_all_nan_btjd_returns_empty():
    btjds = np.full(10, np.nan)
    quality_ok = np.ones(10, dtype=bool)
    assert (
        geo.select_anchor_frames(
            btjds=btjds,
            quality_ok=quality_ok,
            n_anchors=3,
            frames_per_anchor=4,
            edge_fraction=0.12,
            edge_boost=3.0,
            max_expand=10,
        )
        == []
    )


# ── Stacked-mode combine / mask union ────────────────────────────────────────


def test_nanmean_combine_ignores_nan_pixels():
    a = np.array([[1.0, np.nan], [3.0, 4.0]])
    b = np.array([[3.0, 5.0], [np.nan, 6.0]])
    out = geo._nanmean_combine([a, b])
    assert out[0, 0] == pytest.approx(2.0)
    assert out[0, 1] == pytest.approx(5.0)
    assert out[1, 0] == pytest.approx(3.0)
    assert out[1, 1] == pytest.approx(5.0)


def test_frame_reject_mask_static_fallback():
    mask_2d = np.zeros((4, 4), dtype=bool)
    mask_2d[0, 0] = True
    out = geo._frame_reject_mask(mask_catalog=None, mask_2d=mask_2d, btjd=1.0)
    assert out is mask_2d


def test_frame_reject_mask_catalog_union_semantics(monkeypatch):
    class _FakeCatalog:
        def mask_at(self, btjd, which="full"):
            return np.array([[0, 1], [0, 0]], dtype=np.int32)

    captured = {}

    def _fake_epsf_reject_mask(raw):
        captured["raw"] = raw
        return raw.astype(bool)

    monkeypatch.setattr(
        "syndiff_pipeline.difference_imaging.masking.bits.epsf_reject_mask",
        _fake_epsf_reject_mask,
    )
    out = geo._frame_reject_mask(mask_catalog=_FakeCatalog(), mask_2d=None, btjd=1.0)
    assert out.tolist() == [[False, True], [False, False]]


def test_fit_anchor_stacked_unions_window_masks(monkeypatch, tmp_path):
    ny, nx = 16, 16
    imgs = []
    masks = []
    for i in range(3):
        img = np.full((ny, nx), float(i), dtype=np.float64)
        p = tmp_path / f"diff_{i}.fits"
        from astropy.io import fits

        fits.PrimaryHDU(img).writeto(p)
        imgs.append(str(p))
        m = np.zeros((ny, nx), dtype=bool)
        m[i, i] = True  # each frame masks a different pixel
        masks.append(m)

    captured = {}

    def _fake_build(diff_image, gaia_df, epsf_params, *, mask_2d=None, frame_label="", star_usage_out=None):
        captured["diff_image"] = diff_image
        captured["mask_2d"] = mask_2d
        return None, [(1.0, 1.0)], np.ones((1, 3, 3)), [7]

    monkeypatch.setattr(geo, "build_gridded_psf_for_frame", _fake_build)
    monkeypatch.setattr(
        geo, "gaia_science_xy_for_frame", lambda gaia, path, ffi_list_df, bounds: gaia
    )

    grid_xypos, stack, n_stars = geo.fit_anchor_stacked(
        window_diff_paths=imgs,
        window_masks=masks,
        anchor_ffi_path=imgs[1],
        gaia_base=pd.DataFrame({"ra": [1.0], "dec": [2.0]}),
        epsf_params=EpsfParams(),
        ffi_list_df=pd.DataFrame(),
        science_bounds={},
        frame_label="anchor",
    )
    assert stack is not None
    assert n_stars == [7]
    # Mean-combine of [0,1,2] at every pixel -> 1.0 everywhere.
    assert np.allclose(captured["diff_image"], 1.0)
    # Union mask: pixels (0,0), (1,1), (2,2) all rejected.
    union = captured["mask_2d"]
    assert union[0, 0] and union[1, 1] and union[2, 2]
    assert not union[3, 3]


def test_fit_anchor_stacked_writes_star_selection_debug_output(monkeypatch, tmp_path):
    """When debug_plot_dir is set, the anchor's own fit (no re-fit) writes a
    DS9 region + star-selection PNG using build_gridded_psf_for_frame's
    star_usage_out."""
    from astropy.io import fits

    img = np.zeros((16, 16), dtype=np.float64)
    p = tmp_path / "diff_0.fits"
    fits.PrimaryHDU(img).writeto(p)

    def _fake_build(diff_image, gaia_df, epsf_params, *, mask_2d=None, frame_label="", star_usage_out=None):
        if star_usage_out is not None:
            star_usage_out["used_xy"] = [(2.0, 3.0), (5.0, 6.0)]
            star_usage_out["excluded_xy"] = [(9.0, 9.0)]
        return None, [(1.0, 1.0)], np.ones((1, 3, 3)), [3]

    monkeypatch.setattr(geo, "build_gridded_psf_for_frame", _fake_build)
    monkeypatch.setattr(
        geo, "gaia_science_xy_for_frame", lambda gaia, path, ffi_list_df, bounds: gaia
    )

    debug_dir = str(tmp_path / "debug_plots" / "epsf_r1")
    grid_xypos, stack, n_stars = geo.fit_anchor_stacked(
        window_diff_paths=[str(p)],
        window_masks=[None],
        anchor_ffi_path=str(p),
        gaia_base=pd.DataFrame({"ra": [1.0], "dec": [2.0]}),
        epsf_params=EpsfParams(),
        ffi_list_df=pd.DataFrame(),
        science_bounds={},
        frame_label="anchor_stem",
        debug_plot_dir=debug_dir,
        epsf_label="epsf_r1",
    )
    assert stack is not None

    from syndiff_pipeline.difference_imaging.support.plot import (
        epsf_star_selection_png_path,
        epsf_star_selection_region_path,
    )

    region_path = epsf_star_selection_region_path(debug_dir, "epsf_r1", "anchor_stem")
    png_path = epsf_star_selection_png_path(debug_dir, "epsf_r1", "anchor_stem")
    assert os.path.isfile(region_path)
    assert os.path.isfile(png_path)
    region_text = open(region_path, encoding="utf-8").read()
    assert "used_0" in region_text and "color=blue" in region_text
    assert "excluded_0" in region_text and "color=red" in region_text


def test_fit_anchor_stacked_no_debug_output_without_plot_dir(monkeypatch, tmp_path):
    from astropy.io import fits

    img = np.zeros((16, 16), dtype=np.float64)
    p = tmp_path / "diff_0.fits"
    fits.PrimaryHDU(img).writeto(p)

    def _fake_build(diff_image, gaia_df, epsf_params, *, mask_2d=None, frame_label="", star_usage_out=None):
        assert star_usage_out is None
        return None, [(1.0, 1.0)], np.ones((1, 3, 3)), [3]

    monkeypatch.setattr(geo, "build_gridded_psf_for_frame", _fake_build)
    monkeypatch.setattr(
        geo, "gaia_science_xy_for_frame", lambda gaia, path, ffi_list_df, bounds: gaia
    )

    _grid_xypos, stack, _n_stars = geo.fit_anchor_stacked(
        window_diff_paths=[str(p)],
        window_masks=[None],
        anchor_ffi_path=str(p),
        gaia_base=pd.DataFrame({"ra": [1.0], "dec": [2.0]}),
        epsf_params=EpsfParams(),
        ffi_list_df=pd.DataFrame(),
        science_bounds={},
        frame_label="anchor_stem",
    )
    assert stack is not None
    assert not (tmp_path / "debug_plots").exists()


# ── Fingerprint helpers (F1) ─────────────────────────────────────────────────

pytestmark_provenance = pytest.mark.skipif(
    not pg.PROVENANCE_AVAILABLE, reason="common.provenance not importable"
)


@pytestmark_provenance
def test_anchor_epsf_fingerprint_none_on_missing_window_fp():
    fp = geo.anchor_epsf_fingerprint(
        sector=20,
        camera=3,
        ccd=3,
        anchor_product_id="tess0001",
        epsf_label="epsf_r1",
        epsf_params=EpsfParams(),
        window_diff_image_fps=["abc", None, "def"],
    )
    assert fp is None


@pytestmark_provenance
def test_anchor_epsf_fingerprint_order_independent():
    kwargs = dict(
        sector=20,
        camera=3,
        ccd=3,
        anchor_product_id="tess0001",
        epsf_label="epsf_r1",
        epsf_params=EpsfParams(),
    )
    fp_a = geo.anchor_epsf_fingerprint(
        window_diff_image_fps=["fp1", "fp2", "fp3"], **kwargs
    )
    fp_b = geo.anchor_epsf_fingerprint(
        window_diff_image_fps=["fp3", "fp1", "fp2"], **kwargs
    )
    assert fp_a is not None
    assert fp_a == fp_b


@pytestmark_provenance
def test_anchor_epsf_fingerprint_changes_with_window_membership():
    kwargs = dict(
        sector=20,
        camera=3,
        ccd=3,
        anchor_product_id="tess0001",
        epsf_label="epsf_r1",
        epsf_params=EpsfParams(),
    )
    fp_a = geo.anchor_epsf_fingerprint(
        window_diff_image_fps=["fp1", "fp2"], **kwargs
    )
    fp_b = geo.anchor_epsf_fingerprint(
        window_diff_image_fps=["fp1", "fp2", "fp3"], **kwargs
    )
    assert fp_a != fp_b


@pytestmark_provenance
def test_interpolated_epsf_fingerprint_depends_on_both_anchors():
    kwargs = dict(
        sector=20,
        camera=3,
        ccd=3,
        product_id="tess0005",
        epsf_label="epsf_r1",
        epsf_params=EpsfParams(),
    )
    fp_ab = geo.interpolated_epsf_fingerprint(
        neighbor_anchor_fps=["anchor_a_fp", "anchor_b_fp"], **kwargs
    )
    fp_ac = geo.interpolated_epsf_fingerprint(
        neighbor_anchor_fps=["anchor_a_fp", "anchor_c_fp"], **kwargs
    )
    assert fp_ab is not None
    assert fp_ab != fp_ac

    fp_clamped = geo.interpolated_epsf_fingerprint(
        neighbor_anchor_fps=["anchor_a_fp"], **kwargs
    )
    assert fp_clamped is not None
    assert fp_clamped != fp_ab


@pytestmark_provenance
def test_interpolated_epsf_fingerprint_none_on_missing_anchor():
    fp = geo.interpolated_epsf_fingerprint(
        sector=20,
        camera=3,
        ccd=3,
        product_id="tess0005",
        epsf_label="epsf_r1",
        epsf_params=EpsfParams(),
        neighbor_anchor_fps=[None, "anchor_b_fp"],
    )
    assert fp is None


# ── fit_epsf_section_multi (pooled generalization, regression + new) ────────


def _gaussian_stamp(y0: float, x0: float, shape: tuple[int, int], sigma: float = 1.2) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))


def _synthetic_section(stars, shape=(64, 64), amp=30.0):
    image = np.zeros(shape)
    for y, x in stars:
        image += amp * _gaussian_stamp(y, x, shape)
    return image


def test_fit_epsf_section_single_frame_matches_legacy_call():
    stars = [(20.0, 20.0), (22.0, 22.0), (24.0, 24.0), (26.0, 26.0), (28.0, 28.0)]
    image = _synthetic_section(stars)
    stars_tbl = Table()
    stars_tbl["x"] = [s[1] for s in stars]
    stars_tbl["y"] = [s[0] for s in stars]
    stamp = gridded_epsf.fit_epsf_section(
        image, stars_tbl, extract_size=15, oversampling=2, maxiters=5
    )
    assert stamp is not None
    assert np.isfinite(stamp).all()


def test_fit_epsf_section_multi_pools_across_frames():
    stars = [(20.0, 20.0), (22.0, 22.0), (24.0, 24.0)]
    stars_tbl = Table()
    stars_tbl["x"] = [s[1] for s in stars]
    stars_tbl["y"] = [s[0] for s in stars]
    frames = [
        (_synthetic_section(stars), stars_tbl, None),
        (_synthetic_section(stars), stars_tbl, None),
        (_synthetic_section(stars), stars_tbl, None),
    ]
    stamp = gridded_epsf.fit_epsf_section_multi(
        frames, extract_size=15, oversampling=2, maxiters=5
    )
    assert stamp is not None
    assert np.isfinite(stamp).all()


# ── BTJD fallback for manifests with no btjd column (linear/CVZ lane bug) ──


def _make_ffi_list_df(rows: dict[str, str]) -> pd.DataFrame:
    from astropy.io import fits

    from syndiff_pipeline.common.wcs_header_cache import _header_cards_bytes

    records = []
    for filename, date_obs in rows.items():
        hdr = fits.Header()
        if date_obs is not None:
            hdr["DATE-OBS"] = date_obs
        records.append({"filename": filename, "header_cards": _header_cards_bytes(hdr)})
    return pd.DataFrame(records).set_index("filename")


def test_resolve_btjd_by_stem_fills_from_date_obs_when_manifest_has_no_btjd():
    """Regression: linear-mode frame manifests carry no btjd column at all,
    so btjd_by_stem_from_manifest returns {} -- must not silently leave
    every frame's BTJD as NaN (which makes select_anchor_frames return zero
    anchors for every orbit, observed on a real s0050/c4/k4 smoke run)."""
    ffi_list_df = _make_ffi_list_df(
        {"a.fits": "2020-01-01T00:00:00.000", "b.fits": "2020-01-02T00:00:00.000"}
    )
    ffi_path_by_stem = {"pidA": "a.fits", "pidB": "b.fits"}
    resolved = geo._resolve_btjd_by_stem(["pidA", "pidB"], {}, ffi_list_df, ffi_path_by_stem)
    assert set(resolved) == {"pidA", "pidB"}
    assert all(np.isfinite(v) for v in resolved.values())
    assert resolved["pidB"] > resolved["pidA"]


def test_resolve_btjd_by_stem_keeps_existing_finite_values():
    ffi_list_df = _make_ffi_list_df({"a.fits": "2020-01-01T00:00:00.000"})
    resolved = geo._resolve_btjd_by_stem(
        ["pidA"], {"pidA": 123.456}, ffi_list_df, {"pidA": "a.fits"}
    )
    assert resolved["pidA"] == 123.456


def test_resolve_btjd_by_stem_overrides_nan_placeholder():
    ffi_list_df = _make_ffi_list_df({"a.fits": "2020-01-01T00:00:00.000"})
    resolved = geo._resolve_btjd_by_stem(
        ["pidA"], {"pidA": float("nan")}, ffi_list_df, {"pidA": "a.fits"}
    )
    assert np.isfinite(resolved["pidA"])


def test_resolve_btjd_by_stem_missing_date_obs_stays_absent():
    ffi_list_df = _make_ffi_list_df({"a.fits": None})
    resolved = geo._resolve_btjd_by_stem(["pidA"], {}, ffi_list_df, {"pidA": "a.fits"})
    assert "pidA" not in resolved


# ── TESS-mag star selection + isolation filter (dev/forward_epsf_wcs parity) ─


def test_epsf_tess_mag_defaults_and_isolation_validation():
    assert EpsfParams().tess_mag_max == 12.95
    assert EpsfParams().tess_mag_min is None
    assert parse_epsf({"kind": "epsf", "tess_mag_max": 11.0}, 0).tess_mag_max == 11.0
    with pytest.raises(ValueError):
        parse_epsf({"kind": "epsf", "epsf_isolation_min_sep_px": -1.0}, 0)


def test_prepare_gaia_for_gridded_epsf_always_uses_tess_mag():
    df = pd.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [1.0, 2.0],
            "phot_g_mean_mag": [8.0, 15.0],
            "phot_bp_mean_mag": [8.1, 15.2],
            "phot_rp_mean_mag": [7.9, 14.8],
        }
    )
    params = EpsfParams(tess_mag_min=None, tess_mag_max=11.0)
    out = gridded_epsf.prepare_gaia_for_gridded_epsf(df, params)
    assert len(out) == 1
    assert "tess_mag" in out.columns


def test_prepare_gaia_for_gridded_epsf_defers_filter_when_isolation_enabled():
    """Isolation needs the FULL candidate+neighbor pool -- the parent-process
    prefilter must not narrow to the mag window when isolation is on."""
    df = pd.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [1.0, 2.0],
            "phot_g_mean_mag": [8.0, 15.0],
            "phot_bp_mean_mag": [8.1, 15.2],
            "phot_rp_mean_mag": [7.9, 14.8],
        }
    )
    params = EpsfParams(
        tess_mag_min=7.0, tess_mag_max=11.0,
        epsf_isolation_min_sep_px=6.0,
    )
    out = gridded_epsf.prepare_gaia_for_gridded_epsf(df, params)
    assert len(out) == 2  # unfiltered -- both rows still present
    assert "tess_mag" in out.columns


def test_apply_epsf_isolation_filter_matches_forward_epsf_wcs_rule():
    df = pd.DataFrame(
        {
            "phot_g_mean_mag": [8.0, 8.0, 8.0, 8.0, 12.9],
            "phot_bp_mean_mag": [8.1, 8.1, 8.1, 8.1, 13.0],
            "phot_rp_mean_mag": [7.9, 7.9, 7.9, 7.9, 12.8],
            "x": [10.0, 100.0, 103.0, 200.0, 106.0],
            "y": [10.0, 100.0, 100.0, 200.0, 100.0],
        }
    )
    df = gridded_epsf._ensure_tess_mag_column(df)
    out = gridded_epsf.apply_epsf_isolation_filter(
        df, mag_min=7.0, mag_max=11.0, min_sep_px=6.0, neighbor_mag_max=13.0,
        mag_col="tess_mag",
    )
    # Isolated candidates (0, far from everything) and (3, far from everything)
    # survive; the close pair (1, 2, 3px apart, well under the 6px radius)
    # and the faint neighbor's contaminated candidate are dropped.
    assert sorted(out["x"].tolist()) == [10.0, 200.0]


def test_apply_epsf_isolation_filter_neighbor_pool_includes_faint_end():
    """A star just outside the primary mag window (e.g. 11-13) must still
    count as a disqualifying neighbor -- this is the bug a naive
    'filter to mag window first, then check isolation' ordering would cause."""
    df = pd.DataFrame(
        {
            "phot_g_mean_mag": [8.0, 12.5],
            "phot_bp_mean_mag": [8.1, 12.6],
            "phot_rp_mean_mag": [7.9, 12.4],
            "x": [100.0, 102.0],
            "y": [100.0, 100.0],
        }
    )
    df = gridded_epsf._ensure_tess_mag_column(df)
    out = gridded_epsf.apply_epsf_isolation_filter(
        df, mag_min=7.0, mag_max=11.0, min_sep_px=6.0, neighbor_mag_max=13.0,
        mag_col="tess_mag",
    )
    assert len(out) == 0  # the bright candidate is disqualified by its faint neighbor


def test_apply_epsf_isolation_filter_empty_when_no_x_column():
    df = pd.DataFrame({"tess_mag": [8.0]})
    assert len(gridded_epsf.apply_epsf_isolation_filter(
        df, mag_min=7.0, mag_max=11.0, min_sep_px=6.0, neighbor_mag_max=13.0,
    )) == 0


def test_fit_epsf_section_multi_empty_frames_returns_none():
    stars_tbl = Table()
    stars_tbl["x"] = []
    stars_tbl["y"] = []
    assert (
        gridded_epsf.fit_epsf_section_multi(
            [(_synthetic_section([]), stars_tbl, None)],
            extract_size=15,
            oversampling=2,
            maxiters=5,
        )
        is None
    )
