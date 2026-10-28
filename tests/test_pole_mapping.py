"""Pole-adjacent mapping: LONPOLE of the PS1 pole cell, second-pass assignment, and the
``stages.mapping.max_extrapolate_px`` plumbing (fingerprint unchanged when unset)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.common.orchestration.targets import Target
from syndiff_pipeline.common.provenance import model
from syndiff_pipeline.template_creation.orchestration.runner_config import (
    load_runner_config,
    resolve_config,
)
from syndiff_pipeline.template_creation.orchestration.stage_params import MappingStageParams
from syndiff_pipeline.template_creation.orchestration.verify import config_fingerprint
from syndiff_pipeline.template_creation.processing import pancakes as P

CDELT = 6.9444446e-05
KEYS = ["NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "PC1_1", "PC1_2",
        "PC2_1", "PC2_2", "CDELT1", "CDELT2", "RADESYS", "CTYPE1", "CTYPE2"]


def _row(name, crval1, crval2, crpix1=500.5, crpix2=500.5, n=1000, cx=5, cy=5):
    return dict(NAME=name, CRVAL1=crval1, CRVAL2=crval2, CRPIX1=crpix1, CRPIX2=crpix2,
                CDELT1=CDELT, CDELT2=CDELT, PC1_1=-1.0, PC1_2=0.0, PC2_1=-0.0, PC2_2=1.0,
                NAXIS1=n, NAXIS2=n, RADESYS="FK5", CTYPE1="RA---TAN", CTYPE2="DEC--TAN",
                x=cx, y=cy)


def _wcs(row):
    return WCS(fits.Header({k: row[k] for k in KEYS}))


def _table(rows):
    df = pd.DataFrame(rows)
    rd = P.calculate_radec_center(df)
    df["RA"], df["DEC"] = rd[:, 0], rd[:, 1]
    return df


def _calc(row, x, y):
    return P.calculate_radec(x, y, row["CRVAL1"], row["CRVAL2"], row["CRPIX1"], row["CRPIX2"],
                             row["PC1_1"], row["PC1_2"], row["PC2_1"], row["PC2_2"],
                             row["CDELT1"], row["CDELT2"])


REAL_POLE_CRVAL2 = 89.999999999997   # CRVAL2 of the real PS1 skycell.2643.004/.005 headers (lonpole_check)
GRID = [(x, y) for x in np.linspace(0, 6239, 9) for y in np.linspace(0, 6239, 9)]


def _sep_arcsec(w1, w2, pts):
    from astropy.coordinates import SkyCoord

    x, y = np.array(pts).T
    a = SkyCoord(*w1.pixel_to_world_values(x, y), unit="deg")
    b = SkyCoord(*w2.pixel_to_world_values(x, y), unit="deg")
    return a.separation(b).arcsec


def test_table_pole_row_wcs_matches_real_header_wcs():
    table_row = pd.Series(_row("skycell.2643.004", 0.0, 90.0, crpix1=6000.0, crpix2=29040.5, n=6240))
    real_row = dict(table_row, CRVAL2=REAL_POLE_CRVAL2)  # what the real image header carries
    real = _wcs(real_row)
    assert real.wcs.lonpole == 180.0
    # unfixed astropy on the table row is the 180 deg flip (documents the bug)
    assert _sep_arcsec(_wcs(table_row), real, GRID).max() > 1e3
    # the helper used by get_ps1_wcs_information
    _, fixed, shape = P.get_ps1_wcs_information(table_row)
    assert shape == (6240, 6240)
    assert fixed.wcs.lonpole == 180.0
    assert _sep_arcsec(fixed, real, GRID).max() < 1e-3
    # inverse direction too
    x, y = np.array(GRID).T
    wra, wdec = real.pixel_to_world_values(x, y)
    px, py = fixed.world_to_pixel_values(wra, wdec)
    assert np.abs(px - x).max() < 1e-3 and np.abs(py - y).max() < 1e-3
    # numba convention (LONPOLE=180) agrees with the fixed WCS, away from the pole's RA ill-conditioning
    for gx, gy in [(100.0, 200.0), (3000.0, 4000.0), (6000.0, 100.0)]:
        ra, dec = _calc(table_row, gx, gy)
        wra_, wdec_ = fixed.pixel_to_world_values(gx, gy)
        assert abs((ra - wra_ + 180) % 360 - 180) * np.cos(np.deg2rad(wdec_)) < 1e-6


def test_header_helper_pole_only_and_non_pole_unchanged():
    from syndiff_pipeline.common.ps1_wcs import set_pole_lonpole_header, set_pole_lonpole_wcs

    for crval2, expect in [(90.0, True), (89.999999999997, False), (60.0, False), (-90.0, False)]:
        h = fits.Header({k: _row("x", 0.0, crval2)[k] for k in KEYS})
        set_pole_lonpole_header(h)
        assert ("LONPOLE" in h) is expect
        w = WCS(fits.Header({k: _row("x", 0.0, crval2)[k] for k in KEYS}))
        before = w.wcs.lonpole
        set_pole_lonpole_wcs(w)
        assert (w.wcs.lonpole == 180.0) if expect else (w.wcs.lonpole == before)


def test_crval2_60_wcs_bitwise_unchanged():
    row = pd.Series(_row("skycell.1200.044", 123.4, 60.0))
    _, w_new, _ = P.get_ps1_wcs_information(row)
    w_old = _wcs(row)  # what the pre-fix code built
    x, y = np.array(GRID).T
    a = w_new.pixel_to_world_values(x, y)
    b = w_old.pixel_to_world_values(x, y)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert "LONPOLE" not in P.get_ps1_wcs_information(row)[0]


def test_cross_projection_cell_wcs_pole_row():
    from syndiff_pipeline.template_creation.processing.cross_projection_geometry import cell_wcs_from_row

    row = pd.Series(_row("skycell.2643.004", 0.0, 90.0))
    assert cell_wcs_from_row(row).wcs.lonpole == 180.0


# ---------------------------------------------------------------------------
# assign_unmapped_by_pixel_geometry on a small synthetic table
# ---------------------------------------------------------------------------

BUF = dict(edge_buffer_large=100, edge_buffer_small=20, buffer=50)


@pytest.fixture(scope="module")
def table():
    a = _row("skycell.1200.055", 10.0, 60.0)                       # index 0
    b = _row("skycell.1200.056", 10.0, 60.0, crpix1=500.5 - 1000)  # index 1: B px = A px - 1000
    p = _row("skycell.2643.055", 0.0, 90.0)                        # index 2: pole cell
    far = _row("skycell.0001.055", 200.0, -30.0)                   # index 3: nowhere near
    return _table([a, b, p, far])


def _world(table, j, x, y):
    return _wcs(table.iloc[j]).pixel_to_world_values(x, y)


def _run(table, pts, ids=None, **kw):
    """pts: list of (j_cell, x, y, (tess_y, tess_x)) -> world coords through cell j's WCS."""
    ra = np.array([_world(table, j, x, y)[0] for j, x, y, _ in pts])
    dec = np.array([_world(table, j, x, y)[1] for j, x, y, _ in pts])
    pix = np.array([t for *_, t in pts], dtype=np.float64)
    if ids is None:
        ids = np.full(len(pts), -1, dtype=np.int32)
    cx = np.array([0.0, 1000.0, 5000.0, 9000.0])
    cy = np.zeros(4)
    return P.assign_unmapped_by_pixel_geometry(ids, table, ra, dec, cx, cy, pix, **BUF, **kw)


def test_tiers_and_default_no_extrapolation(table):
    pts = [
        (0, 500.0, 500.0, (0.0, 500.0)),    # tier 1: inside A buffered rect only
        (0, 900.0, 500.0, (0.0, 900.0)),    # tier 1 tie A/B (B px x=-100 -> outside) check below
        (0, 990.0, 500.0, (0.0, 990.0)),    # tier 2: in A image, outside rect (x1=980)
        (0, 300.0, -50.0, (0.0, 300.0)),    # tier 3: 50 px outside A image
        (0, 300.0, -5000.0, (0.0, 300.0)),  # far outside everything
        (2, 300.0, 500.0, (0.0, 5000.0)),   # pole cell tier 1
    ]
    ids, info = _run(table, pts)
    assert ids.dtype == np.int32
    assert list(ids[[0, 1, 2]]) == [0, 0, 0]
    assert ids[3] == -1 and ids[4] == -1          # default max_extrapolate_px = 0 never extrapolates
    assert ids[5] == 2
    assert info["tier1"] == 3 and info["tier2"] == 1 and info["tier3"] == 0 and info["left"] == 2

    ids100, info100 = _run(table, pts, max_extrapolate_px=100.0)
    assert ids100[3] == 0 and ids100[4] == -1
    assert info100["tier3"] == 1 and info100["left"] == 1
    assert (ids100[[0, 1, 2, 5]] == ids[[0, 1, 2, 5]]).all()


def test_tier1_tie_goes_to_closest_projection_centre(table):
    # A pixel x=900 -> B pixel x=-100: not in B.  Use x=1050 on A == B x=50: outside B rect (x0=100).
    # Real overlap: B pixel x=150 == A pixel x=1150 (outside A image) -> only B.  Build a true tie
    # with two cells sharing one WCS:
    a = _row("skycell.1200.055", 10.0, 60.0)
    b = _row("skycell.1200.066", 10.0, 60.0, cx=6)
    t = _table([a, b])
    pts = [(0, 500.0, 500.0, (0.0, 1.0)), (0, 500.0, 500.0, (0.0, 999.0))]
    cx = np.array([0.0, 1000.0])
    ra, dec = zip(*[_wcs(a).pixel_to_world_values(500.0, 500.0)] * 2)
    pix = np.array([p[3] for p in pts])
    ids, _ = P.assign_unmapped_by_pixel_geometry(
        np.full(2, -1, dtype=np.int32), t, np.array(ra), np.array(dec), cx, np.zeros(2), pix, **BUF)
    assert list(ids) == [0, 1]


def test_assigned_entries_never_modified_and_noop_when_nothing_unmapped(table):
    pts = [
        (0, 500.0, 500.0, (0.0, 500.0)),
        (0, 500.0, 500.0, (0.0, 500.0)),
        (2, 300.0, 500.0, (0.0, 5000.0)),
    ]
    start = np.array([3, -1, 1], dtype=np.int32)  # 0 and 2 preassigned (wrong on purpose)
    ids, info = _run(table, pts, ids=start.copy())
    assert ids[0] == 3 and ids[2] == 1
    assert ids[1] == 0 and info["n_unmapped_in"] == 1

    full = np.array([0, 1, 2], dtype=np.int32)
    ids2, info2 = _run(table, pts, ids=full.copy())
    assert np.array_equal(ids2, full) and ids2.dtype == full.dtype
    assert info2["n_unmapped_in"] == 0 and info2["left"] == 0
    assert ids2 is not full  # a copy, input untouched


# ---------------------------------------------------------------------------
# stage-param plumbing and fingerprint pins
# ---------------------------------------------------------------------------

PINNED_MAPPING_FINGERPRINT = "10220412b88e15e5"  # computed on main before this change


def _resolved(tmp: Path, overrides: dict):
    cfg_path = tmp / "cfg.yaml"
    doc = {
        "config_schema_version": 2,
        "workspace_root": str(tmp / "workspace"),
        "data_root": str(tmp / "data"),
        "runs_root": str(tmp / "workspace" / "runs"),
        "state_db_path": str(tmp / "state.sqlite"),
        "skycell_wcs_csv": "x.csv",
        "stages": {"mapping": dict(overrides)} if overrides else {},
    }
    cfg_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    t = Target(sector=20, camera=3, ccd=3, target_ra=228.0, target_dec=52.0, target_name="2020ut")
    return resolve_config(t, load_runner_config(str(cfg_path)))


def test_stage_param_default_and_validation():
    assert MappingStageParams().max_extrapolate_px == 0.0
    assert MappingStageParams(max_extrapolate_px=200).max_extrapolate_px == 200.0
    for bad in (-1, -0.001, float("nan"), float("inf"), "abc", None):
        with pytest.raises(ValueError):
            MappingStageParams(max_extrapolate_px=bad)


def test_flag_off_fingerprint_and_recipe_pinned_flag_on_moves():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        off = _resolved(tmp, {})
        assert off.stages.mapping.max_extrapolate_px == 0.0
        assert config_fingerprint(off, "mapping") == PINNED_MAPPING_FINGERPRINT
        assert config_fingerprint(_resolved(tmp, {"max_extrapolate_px": 0}), "mapping") == PINNED_MAPPING_FINGERPRINT
        assert model.mapping_recipe_params(off) == {
            "oversampling_factor": 1, "pad_distance": 480, "overwrite": True,
            "mapping_grid": {"x_left_dead": 44, "x_right_dead": 44, "y_edge_strip": 30,
                             "conv_pad_native": 8, "oversampling_factor": 1, "mapgrid_version": 3},
        }
        on = _resolved(tmp, {"max_extrapolate_px": 200})
        assert on.stages.mapping.max_extrapolate_px == 200.0
        assert config_fingerprint(on, "mapping") != PINNED_MAPPING_FINGERPRINT
        assert model.mapping_recipe_params(on)["max_extrapolate_px"] == 200.0


def test_per_scc_override_reaches_stage_params():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        r = _resolved(tmp, {"max_extrapolate_px": 200})
        assert r.stages.mapping.max_extrapolate_px == 200.0
        with pytest.raises(ValueError):
            _resolved(tmp, {"max_extrapolate_px": -5})
