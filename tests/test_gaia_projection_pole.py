"""Pole-adjacent projection catalogues: RA-chunked download + tangent-box filter (no network)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing import gaia_projection_catalog as gpc

POLE_PROJ = "2643"   # north-cap cell, CRVAL2 = 90
WRAP_PROJ = "2642"   # CRVAL2 = 86, RA-wrapping footprint
NORMAL_PROJ = "2486"


def _sky(n=4000, dec_min=80.0, seed=1):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(dec_min)), 1.0, n)))
    return pd.DataFrame({
        "source_id": pd.array(rng.permutation(n) + 1000, dtype="Int64"),
        "ra": ra, "dec": dec,
        "phot_g_mean_mag": 12.0, "phot_bp_mean_mag": 12.5,
        "phot_rp_mean_mag": rng.uniform(10, 20, n),
    })


def _bbox_downloader(sky, calls):
    """Mock Gaia: returns all rows inside the bbox of the polygon passed in."""
    def _dl(ra, dec, mag):
        calls.append((float(np.min(ra)), float(np.max(ra)), float(np.min(dec)), float(np.max(dec)), mag))
        m = (sky.ra >= np.min(ra)) & (sky.ra <= np.max(ra)) & (sky.dec >= np.min(dec)) & (sky.dec <= np.max(dec))
        return sky[m].sample(frac=1.0, random_state=3).reset_index(drop=True)
    return _dl


def test_pole_detection():
    for p in (POLE_PROJ, WRAP_PROJ, "2598", "2634"):
        ra, dec = gpc.projection_footprint_polygon(p)
        assert gpc.polygon_needs_pole_path(ra, dec), p
    for p in (NORMAL_PROJ, "2599", "2386"):
        ra, dec = gpc.projection_footprint_polygon(p)
        assert not gpc.polygon_needs_pole_path(ra, dec), p


def test_dec_range_contains_pole_only_for_cap():
    assert gpc._pole_dec_range(POLE_PROJ, gpc.DEFAULT_MARGIN_PX)[1] == 90.0
    assert gpc._pole_dec_range(WRAP_PROJ, gpc.DEFAULT_MARGIN_PX)[1] < 90.0


@pytest.mark.parametrize("proj", [POLE_PROJ, WRAP_PROJ])
def test_pole_path_matches_bruteforce(tmp_path, proj):
    sky = _sky()
    calls = []
    out = gpc.ensure_projection_catalog(tmp_path, proj, downloader=_bbox_downloader(sky, calls))
    assert len(calls) == gpc.POLE_RA_CHUNKS and all(c[4] is None for c in calls)
    got = pd.read_parquet(out)
    # independent brute force: tangent-plane box via the cells' own geometry
    crval1, crval2, u0, u1, v0, v1 = gpc._projection_tangent_box(proj)
    from astropy.wcs import WCS
    w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [crval1, crval2]
    w.wcs.crpix = [0, 0]; w.wcs.cdelt = [1, 1]
    if crval2 >= 90:
        w.wcs.lonpole = 180.0
    u, v = w.wcs_world2pix(sky.ra.values, sky.dec.values, 1)
    exp = sky[(u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)]
    assert len(exp) > 0
    assert set(got.source_id.astype("int64")) == set(exp.source_id.astype("int64"))
    sid = got.source_id.astype("int64").to_numpy()
    assert len(set(sid)) == len(sid) and (np.diff(sid) > 0).all()
    meta = json.load(open(str(out) + ".meta.json"))
    assert meta["download_mode"] == "ra_chunked"
    assert meta["content_sha256"] == gpc.content_sha256(gpc._read_catalog(out))


def test_pole_path_deterministic(tmp_path):
    sky = _sky()
    a = gpc.ensure_projection_catalog(tmp_path / "a", POLE_PROJ, downloader=_bbox_downloader(sky, []))
    b = gpc.ensure_projection_catalog(tmp_path / "b", POLE_PROJ, downloader=_bbox_downloader(sky.sample(frac=1, random_state=9), []))
    assert a.read_bytes() == b.read_bytes()
    assert json.load(open(str(a) + ".meta.json"))["content_sha256"] == json.load(open(str(b) + ".meta.json"))["content_sha256"]


def test_chunk_merge_dedups_edge_rows():
    rows = pd.DataFrame({"source_id": pd.array([1, 2, 3], dtype="Int64"), "ra": [14.99, 15.0, 15.01], "dec": [88.0] * 3})
    def fetch(ra_rng, dec_rng):  # edge row 2 returned by both neighbouring chunks
        return rows[(rows.ra >= ra_rng[0]) & (rows.ra <= ra_rng[1])]
    df = gpc.download_ra_chunked(POLE_PROJ, fetch=fetch)
    assert sorted(df.source_id.astype("int64")) == [1, 2, 3]


def test_non_pole_path_unchanged(tmp_path):
    """Non-pole projections: one polygon call, old filter, no download_mode key."""
    calls = []
    def dl(ra, dec, mag):
        calls.append((len(ra), mag))
        crval1, crval2, *_ = gpc._projection_tangent_box(NORMAL_PROJ)
        w = gpc._projection_tan_wcs(crval1, crval2)
        pts = np.array([[0.0, 0.0], [0.5, 0.3], [-0.4, 0.2], [30.0, 30.0]])
        r, d = w.wcs_pix2world(pts[:, 0], pts[:, 1], 1)
        return pd.DataFrame({"source_id": pd.array([50, 10, 30, 20], dtype="Int64"), "ra": r, "dec": d,
                             "phot_g_mean_mag": 12.0, "phot_bp_mean_mag": 12.5, "phot_rp_mean_mag": 11.5})
    out = gpc.ensure_projection_catalog(tmp_path, NORMAL_PROJ, downloader=dl)
    assert calls == [(200, None)]
    df = pd.read_parquet(out)
    assert list(df.source_id) == [10, 30, 50]
    meta = json.load(open(str(out) + ".meta.json"))
    assert "download_mode" not in meta and "ra_chunks" not in meta


def test_polygon_function_bit_identical():
    # the refactor must not change the polygon (golden values of the first vertices of 2486)
    ra, dec = gpc.projection_footprint_polygon(NORMAL_PROJ)
    assert ra.shape == (200,) and dec.shape == (200,)
    crval1, crval2, u0, u1, v0, v1 = gpc._projection_tangent_box(NORMAL_PROJ)
    w = gpc._projection_tan_wcs(crval1, crval2)
    r2, d2 = w.wcs_pix2world([u0], [v0], 1)
    assert ra[0] == np.mod(r2[0], 360.0) and dec[0] == d2[0]


def test_provenance_in_meta_only(tmp_path):
    sky = _sky()
    a = gpc.ensure_projection_catalog(tmp_path / "a", POLE_PROJ, downloader=_bbox_downloader(sky, []))
    b = gpc.ensure_projection_catalog(tmp_path / "b", POLE_PROJ, downloader=_bbox_downloader(sky, []),
                                      provenance={"built_by_branch": "x", "built_by_commit": "abc"})
    ma, mb = json.load(open(str(a) + ".meta.json")), json.load(open(str(b) + ".meta.json"))
    assert mb["built_by_commit"] == "abc" and "built_by_commit" not in ma
    for k in ("content_sha256", "removal_content_sha256", "removal_fingerprint_sha256", "file_sha256", "n_rows"):
        assert ma[k] == mb[k]
    assert a.read_bytes() == b.read_bytes()
    assert gpc.projection_catalog_fingerprint(tmp_path / "a", POLE_PROJ) == gpc.projection_catalog_fingerprint(tmp_path / "b", POLE_PROJ)
    assert mb["download_dec_range"][1] == 90.0 and mb["ra_chunks"] == gpc.POLE_RA_CHUNKS
