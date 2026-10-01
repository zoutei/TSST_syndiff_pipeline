"""Tests for the ``footprint_v1`` star-removal convention in band_utils.

All inputs are synthetic and deterministic.  PS1 cells overlap by 480 px, so a
star centred just outside cell X can leave its halo inside X; ``footprint_v1``
must remove it from both cells, ``segment_v0`` (legacy) does not.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing.band_utils import (
    REMOVAL_CONVENTION,
    REMOVAL_CONVENTION_LEGACY,
    build_sep_background_segmentation,
    remove_background,
    select_catalog_for_cell,
    star_footprint_radius,
)

CELL = 600
SKY_W = 1000  # cell X = sky cols 0..599, cell Y = sky cols 400..999 (200 px shared)
X0_Y = 400


def _gauss(shape, cx, cy, amp, sigma):
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))


def _sky(stars, shape=(CELL, SKY_W), seed=1, noise=0.05):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise, shape)
    for cx, cy, amp, sig in stars:
        data += _gauss(shape, cx, cy, amp, sig)
    return data.astype(np.float32)


def _cut_cell(sky, x0):
    """Cut a CELLxCELL window and NaN its outermost row/column like PS1."""
    cell = sky[:, x0 : x0 + CELL].copy()
    cell[0, :] = cell[-1, :] = np.nan
    cell[:, 0] = cell[:, -1] = np.nan
    uncert = np.full_like(cell, 0.1)
    return cell, uncert


def _sky_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 30.0]
    w.wcs.crpix = [500.5, 300.5]
    w.wcs.cdelt = [-7e-5, 7e-5]
    return w


def _cell_wcs(x0):
    w = _sky_wcs()
    w.wcs.crpix = [500.5 - x0, 300.5]
    return w


def _catalog(rows):
    """rows: (sky_x, sky_y, tess_mag, source_id) -> Gaia-like frame."""
    w = _sky_wcs()
    xs = np.array([r[0] for r in rows], dtype=float)
    ys = np.array([r[1] for r in rows], dtype=float)
    ra, dec = w.pixel_to_world_values(xs, ys)
    tm = np.array([r[2] for r in rows], dtype=float)
    return pd.DataFrame({
        "source_id": pd.array([r[3] for r in rows], dtype="Int64"),
        "ra": ra,
        "dec": dec,
        "phot_g_mean_mag": tm + 0.43,  # no BP/RP -> T = G - 0.43
        "phot_bp_mean_mag": np.nan,
        "phot_rp_mean_mag": np.nan,
    })


# Bright star S centred 13 px beyond cell X's right edge (col 599) -> sky x=612.
S_XY = (612.0, 300.0)
STARS = [
    (S_XY[0], S_XY[1], 2000.0, 8.0),  # halo reaches ~34 px, i.e. >10 px into X
    (500.0, 150.0, 30.0, 2.5),   # faint, uncatalogued, in the shared strip
    (150.0, 450.0, 30.0, 2.5),
]


def _run(cell, uncert, cat_px, convention):
    return remove_background(
        cell.copy(), uncert, sigma=2.5, sigma_mask=50, mask=None,
        remove_saturated_stars=True, gaia_catalog_pixels=cat_px,
        bright_star_mag_threshold=13.0, convention=convention,
    )


class TestSeamStar:
    def setup_method(self):
        self.sky = _sky(STARS)
        self.cx, self.ux = _cut_cell(self.sky, 0)
        self.cy, self.uy = _cut_cell(self.sky, X0_Y)
        # S (bright), N (faint neighbour inside S's halo, in Y but outside X).
        self.cat = _catalog([(S_XY[0], S_XY[1], 10.0, 111), (606.0, 308.0, 16.0, 222)])

    def _strip(self, data_x, data_y):
        # Shared sky columns 401..598 (skip each cell's NaN edge column).
        return data_x[:, 401:599], data_y[:, 1:199]

    def test_footprint_v1_removes_in_both_cells(self):
        cat_x = select_catalog_for_cell(self.cat, _cell_wcs(0), (CELL, CELL))
        cat_y = select_catalog_for_cell(self.cat, _cell_wcs(X0_Y), (CELL, CELL))
        assert list(cat_x["source_id"]) == [111]  # N is faint and off-cell
        assert not cat_x["in_cell"].iloc[0]
        assert cat_x["pixel_x"].iloc[0] >= CELL
        assert cat_y["in_cell"].all() and len(cat_y) == 2

        out_x, rec_x = _run(self.cx, self.ux, cat_x, REMOVAL_CONVENTION)
        out_y, rec_y = _run(self.cy, self.uy, cat_y, REMOVAL_CONVENTION)

        y = int(S_XY[1])
        # Halo pixels of S inside X (cols 582..598) are zeroed in both cells.
        assert np.all(out_x[y - 5 : y + 6, 582:599] == 0)
        assert np.all(out_y[y - 5 : y + 6, 582 - X0_Y : 599 - X0_Y] == 0)
        sx, sy = self._strip(out_x, out_y)
        np.testing.assert_array_equal(sx, sy)
        # The uncatalogued faint star in the shared strip survives in both.
        assert sx[150, 500 - 401] > 5.0

        reasons_x = {(r["source_id"], r["removal_reason"]) for r in rec_x}
        assert reasons_x == {(111, "catalog_bright_star")}
        assert rec_x[0]["pixel_x"] >= CELL  # off-cell record keeps outside position
        reasons_y = {(r["source_id"], r["removal_reason"]) for r in rec_y}
        assert reasons_y == {(111, "catalog_bright_star"), (222, "catalog_neighbor")}

    def test_legacy_in_cell_only_leaves_halo_in_x(self):
        """Documents the seam bug: X keeps S's halo, Y removes it."""
        def in_cell_only(wcs):
            c = select_catalog_for_cell(self.cat, wcs, (CELL, CELL))
            return c[c["in_cell"]].reset_index(drop=True)

        cat_x = in_cell_only(_cell_wcs(0))
        cat_y = in_cell_only(_cell_wcs(X0_Y))
        assert len(cat_x) == 0
        out_x, _ = _run(self.cx, self.ux, cat_x, REMOVAL_CONVENTION_LEGACY)
        out_y, _ = _run(self.cy, self.uy, cat_y, REMOVAL_CONVENTION_LEGACY)

        y = int(S_XY[1])
        assert np.any(out_x[y - 5 : y + 6, 582:599] > 0)
        assert np.all(out_y[y - 5 : y + 6, 582 - X0_Y : 599 - X0_Y] == 0)


def _ring_star(size=300, c=150):
    """Bright star whose halo is cut into two SEP segments (left/right)."""
    data = _gauss((size, size), c, c, 500.0, 4.0).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - c, yy - c)
    data[(r > 11) & (np.abs(xx - c) <= 1)] = 0.0  # vertical cut through the halo
    return data, np.full_like(data, 0.1)


def _row(px, py, tmag, sid=1, in_cell=None):
    d = {
        "source_id": sid, "pixel_x": px, "pixel_y": py, "tess_mag": tmag,
        "ra": 1.0, "dec": 2.0, "phot_g_mean_mag": tmag,
        "phot_bp_mean_mag": np.nan, "phot_rp_mean_mag": np.nan,
    }
    if in_cell is not None:
        d["in_cell"] = in_cell
    return d


class TestSplitHalo:
    def test_precondition_two_segments(self):
        data, unc = _ring_star()
        res = build_sep_background_segmentation(data, unc, sigma=2.5, sigma_mask=50,
                                                close_bright_mask=True)
        assert len(np.unique(res.segmap[res.segmap > 0])) >= 2

    def test_footprint_removes_all_legacy_removes_part(self):
        data, unc = _ring_star()
        cat = pd.DataFrame([_row(150.0, 150.0, 10.0)])
        new, rec = remove_background(data.copy(), unc, gaia_catalog_pixels=cat,
                                     convention=REMOVAL_CONVENTION)
        old, _ = remove_background(data.copy(), unc, gaia_catalog_pixels=cat,
                                   convention=REMOVAL_CONVENTION_LEGACY)
        assert np.count_nonzero(new) == 0
        assert np.count_nonzero(old) > 0
        assert [r["removal_reason"] for r in rec] == ["catalog_bright_star"]


class TestOffCellGuards:
    def _cell(self):
        size = 300
        data = _gauss((size, size), size - 12, 150, 30.0, 6.0).astype(np.float32)
        return data, np.full_like(data, 0.1)

    def test_star_beyond_radius_not_removed(self):
        data, unc = self._cell()
        w = 300
        far = pd.DataFrame([_row(w + 400.0, 150.0, 10.0, in_cell=False)])
        out, rec = remove_background(data.copy(), unc, gaia_catalog_pixels=far,
                                     convention=REMOVAL_CONVENTION)
        assert rec == []
        assert out[150, w - 12] > 5.0

        near = pd.DataFrame([_row(w + 100.0, 150.0, 10.0, in_cell=False)])
        out, rec = remove_background(data.copy(), unc, gaia_catalog_pixels=near,
                                     convention=REMOVAL_CONVENTION)
        assert [r["removal_reason"] for r in rec] == ["catalog_bright_star"]
        assert out[150, w - 12] == 0.0

    def test_lookup_on_empty_sky_removes_nothing(self):
        size = 300
        data = _gauss((size, size), 100, 150, 30.0, 3.0).astype(np.float32)  # far from edge
        unc = np.full_like(data, 0.1)
        row = pd.DataFrame([_row(size + 20.0, 150.0, 10.0, in_cell=False)])
        out, rec = remove_background(data.copy(), unc, gaia_catalog_pixels=row,
                                     convention=REMOVAL_CONVENTION)
        assert rec == []
        assert out[150, 100] > 5.0

    def test_unrelated_source_peaking_inside_not_removed(self):
        """Footprint contains the lookup pixel but peaks 40 px in: not halo light."""
        size = 300
        data = _gauss((size, size), size - 40, 150, 2.0, 20.0).astype(np.float32)
        unc = np.full_like(data, 0.1)
        row = pd.DataFrame([_row(size + 13.0, 150.0, 10.0, in_cell=False)])
        out, rec = remove_background(data.copy(), unc, gaia_catalog_pixels=row,
                                     convention=REMOVAL_CONVENTION)
        res = build_sep_background_segmentation(data, unc, sigma=2.5, sigma_mask=50,
                                                close_bright_mask=True)
        assert res.segmap[150, size - 1] > 0  # precondition: lookup pixel is in a segment
        assert rec == []
        assert out[150, size - 40] > 1.0

    def test_faint_halo_past_nan_edge_column_removed(self):
        """Halo reaches only ~3 px past a NaN edge column: lookup steps inward."""
        size = 300
        data = _gauss((size, size), size + 12, 150, 10.0, 6.0).astype(np.float32)
        data[:, size - 1] = np.nan
        unc = np.full_like(data, 0.1)
        row = pd.DataFrame([_row(size + 12.0, 150.0, 12.0, in_cell=False)])
        out, rec = remove_background(data.copy(), unc, gaia_catalog_pixels=row,
                                     convention=REMOVAL_CONVENTION)
        assert [r["removal_reason"] for r in rec] == ["catalog_bright_star"]
        assert np.all(out[145:156, size - 4 : size - 1] == 0)

    def test_unknown_convention_raises(self):
        data, unc = self._cell()
        with pytest.raises(ValueError):
            remove_background(data, unc, convention="nope")


class TestSaturationPass:
    def test_whole_component_zeroed_no_star(self):
        data, unc = _ring_star()
        mask = np.zeros(data.shape, dtype=np.uint16)
        mask[148:153, 148:153] = 0x0020 | 0x1000
        out, rec = remove_background(data.copy(), unc, mask=mask,
                                     convention=REMOVAL_CONVENTION)
        assert np.count_nonzero(out) == 0  # both halves, not just one segment
        assert [r["removal_reason"] for r in rec] == ["quality_flag_no_star"]
        assert rec[0]["seg_flux"] > 0
        assert abs(rec[0]["seg_centroid_x"] - 150) < 3 and abs(rec[0]["seg_centroid_y"] - 150) < 3

    def test_faint_catalog_star_in_component_is_quality_flag_star(self):
        data, unc = _ring_star()
        mask = np.zeros(data.shape, dtype=np.uint16)
        mask[148:153, 148:153] = 0x0020 | 0x1000
        cat = pd.DataFrame([_row(150.0, 150.0, 16.0, sid=77)])
        out, rec = remove_background(data.copy(), unc, mask=mask, gaia_catalog_pixels=cat,
                                     convention=REMOVAL_CONVENTION)
        assert np.count_nonzero(out) == 0
        assert [(r["source_id"], r["removal_reason"]) for r in rec] == [(77, "quality_flag_star")]


def _legacy_fixture():
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 0.05, (200, 200))
    data += _gauss((200, 200), 60, 60, 300.0, 3.0)
    data += _gauss((200, 200), 150, 120, 100.0, 2.0)
    data += _gauss((200, 200), 100, 30, 20.0, 2.0)
    data += _gauss((200, 200), 30, 150, 20.0, 2.0)  # uncatalogued, kept
    data = data.astype(np.float32)
    unc = np.full_like(data, 0.1)
    mask = np.zeros(data.shape, dtype=np.uint16)
    mask[58:63, 58:63] = 0x0020 | 0x1000
    mask[28:33, 98:103] = 0x0020 | 0x1000
    cat = pd.DataFrame([
        _row(60.0, 60.0, 10.0, sid=11),
        _row(61.0, 59.0, 15.0, sid=12),
        _row(150.0, 120.0, 11.0, sid=13),
        _row(-5.0, 20.0, 9.0, sid=14),
    ])
    return data, unc, mask, cat


# sha256 of (processed data bytes + repr(records)) from the pre-change implementation.
_LEGACY_SHA = "d02d60eec27010d3d216603a0144788397af2e03facd648911382c92466c2787"


class TestLegacyBitIdentical:
    def test_segment_v0_matches_pre_change_output(self):
        data, unc, mask, cat = _legacy_fixture()
        out, rec = remove_background(
            data.copy(), unc, sigma=2.5, sigma_mask=50, mask=mask,
            gaia_catalog_pixels=cat, convention="segment_v0",
        )
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(out).tobytes())
        h.update(repr(rec).encode())
        assert h.hexdigest() == _LEGACY_SHA


class TestFootprintRadiusAndSelection:
    def test_radius_values(self):
        assert star_footprint_radius(13.0) == pytest.approx(160.0)
        assert star_footprint_radius(10.0) == pytest.approx(160.0)
        assert star_footprint_radius(8.0) == pytest.approx(160.0 * 10 ** 0.4)
        assert star_footprint_radius(5.0) == pytest.approx(480.0)
        arr = star_footprint_radius(np.array([13.0, 5.0]))
        np.testing.assert_allclose(arr, [160.0, 480.0])

    def test_selection_by_distance_and_magnitude(self):
        w = _cell_wcs(0)
        # Distances beyond the right edge (col 599): R(11)=160, so 100 in, 300 out.
        rows = [
            (300.0, 300.0, 16.0, 1),          # in cell, faint: kept
            (599.0 + 100.0, 300.0, 11.0, 2),  # off-cell, bright, near: kept
            (599.0 + 300.0, 300.0, 11.0, 3),  # off-cell, bright, far: dropped
            (599.0 + 20.0, 300.0, 16.0, 4),   # off-cell, faint: dropped
            (599.0 + 300.0, 300.0, 5.0, 5),   # off-cell, very bright, R=480: kept
        ]
        out = select_catalog_for_cell(_catalog(rows), w, (CELL, CELL))
        assert sorted(out["source_id"].tolist()) == [1, 2, 5]
        by_id = out.set_index("source_id")
        assert by_id.loc[1, "in_cell"] and not by_id.loc[2, "in_cell"]
        assert {"pixel_x", "pixel_y", "tess_mag", "in_cell"} <= set(out.columns)

    def test_empty_catalog(self):
        out = select_catalog_for_cell(pd.DataFrame(), _cell_wcs(0), (CELL, CELL))
        assert len(out) == 0
