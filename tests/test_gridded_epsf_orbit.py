"""Unit tests for orbit-binned gridded ePSF (gridded_epsf_orbit.py)."""

from __future__ import annotations

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


def test_anchor_target_phases_denser_near_edges():
    """With edge_boost>1, quantile spacing near the edges must be tighter
    than the corresponding uniform spacing (denser sampling)."""
    n = 7
    phases = geo.anchor_target_phases(n, edge_fraction=0.15, edge_boost=4.0)
    assert phases.shape == (n,)
    assert np.all(np.diff(phases) > 0)
    assert phases.min() >= 0.0 and phases.max() <= 1.0
    uniform_gap = 1.0 / n
    edge_gap = phases[1] - phases[0]
    middle_gap = phases[n // 2 + 1] - phases[n // 2]
    assert edge_gap < uniform_gap
    assert middle_gap > edge_gap


def test_anchor_target_phases_no_edge_boost_is_uniform_quantiles():
    n = 5
    phases = geo.anchor_target_phases(n, edge_fraction=0.12, edge_boost=1.0)
    expected = (np.arange(n) + 0.5) / n
    assert np.allclose(phases, expected, atol=1e-3)


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

    def _fake_build(diff_image, gaia_df, epsf_params, *, mask_2d=None, frame_label=""):
        captured["diff_image"] = diff_image
        captured["mask_2d"] = mask_2d
        return None, [(1.0, 1.0)], np.ones((1, 3, 3))

    monkeypatch.setattr(geo, "build_gridded_psf_for_frame", _fake_build)
    monkeypatch.setattr(
        geo, "gaia_science_xy_for_frame", lambda gaia, path, ffi_list_df, bounds: gaia
    )

    grid_xypos, stack = geo.fit_anchor_stacked(
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
    # Mean-combine of [0,1,2] at every pixel -> 1.0 everywhere.
    assert np.allclose(captured["diff_image"], 1.0)
    # Union mask: pixels (0,0), (1,1), (2,2) all rejected.
    union = captured["mask_2d"]
    assert union[0, 0] and union[1, 1] and union[2, 2]
    assert not union[3, 3]


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
