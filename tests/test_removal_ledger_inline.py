"""Inline removal ledger in ps1_process: identical pixels, published ledger, safe fallback."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.template_creation.processing import ps1_process as pp
from syndiff_pipeline.template_creation.processing.removal_ledger import inline as rl

SHAPE = (400, 400)
CELL = "skycell.2528.005"


def _header():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [270.0, 66.0]
    w.wcs.crpix = [200.5, 200.5]
    w.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    h = w.to_header()
    h["NAXIS1"], h["NAXIS2"] = SHAPE[1], SHAPE[0]
    h["MJD-OBS"] = 56000.0
    return h.tostring(), w


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    img = rng.normal(0.0, 1.0, SHAPE)
    stars = [(200.0, 200.0, 5e5, 6.0), (90.0, 310.0, 3e3, 2.0), (300.0, 80.0, 2e3, 2.0), (215.0, 230.0, 8e2, 2.0)]
    for x, y, flux, sig in stars:
        img += flux / (2 * np.pi * sig**2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig**2))
    return img.astype(np.float32), stars


def _catalog(w, stars):
    rows = []
    mags = [(9.0, 9.4, 8.5), (16.0, 16.4, 15.5), (16.5, 16.9, 16.0), (19.5, 20.0, 19.0)]
    for i, ((x, y, _f, _s), (g, bp, rp)) in enumerate(zip(stars, mags)):
        ra, dec = w.all_pix2world(x, y, 0)
        rows.append(dict(source_id=1000 + i, ra=float(ra), dec=float(dec), pmra=0.0, pmdec=0.0,
                         phot_g_mean_mag=g, phot_bp_mean_mag=bp, phot_rp_mean_mag=rp))
    df = pd.DataFrame(rows)
    df["source_id"] = df["source_id"].astype("Int64")
    return df


def _bundle(tmp_path, *, ledger):
    header, w = _header()
    img, stars = _scene()
    cat = _catalog(w, stars)
    b = dict(
        skycell_id=CELL, projection="skycell.2528", row_id=0, x_coord=0,
        combined_image=img, combined_uncert=np.ones(SHAPE, np.float32),
        combined_mask=np.zeros(SHAPE, np.int32), headers_data={"r": header},
        remove_saturated_stars=True, bright_star_mag_threshold=13.0,
        gaia_catalog=cat[cat.phot_rp_mean_mag < 18].reset_index(drop=True),
    )
    if ledger:
        b["ledger"] = dict(data_root=str(tmp_path), projection="skycell.2528", skycell="005",
                           combined_fingerprint="fp_test", recipe_id="rid", gaia_fingerprint="g:abc",
                           gaia_store="test", gaia_all=cat)
    return b


def _run(bundle):
    result = pp.process_single_cell(bundle)
    assert result is not None
    return pp._materialize_shm_result(result)


def test_ledger_does_not_change_pixels_and_publishes(tmp_path):
    off = _run(_bundle(tmp_path, ledger=False))
    on = _run(_bundle(tmp_path, ledger=True))
    assert np.array_equal(off["combined_image"], on["combined_image"], equal_nan=True)
    pd.testing.assert_frame_equal(pd.DataFrame(off["removed_stars"]), pd.DataFrame(on["removed_stars"]))
    assert on["ledger_path"] and on["ledger_fingerprint"] == "fp_test"
    path = rl.published_ledger(tmp_path, "skycell.2528", "005", "fp_test")
    assert path is not None and str(path) == on["ledger_path"]
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "complete_cell_pixels"
    assert manifest["source_accounting_status"] == "catalogue_scoped"
    with np.load(path / "geometry.npz") as z:
        assert "segmentation_union" in z.files and "sep_bright_mask" in z.files
    ids = rl.associated_gaia_ids(path)
    assert 1000 in ids  # the bright trigger star


def test_ledger_failure_falls_back_to_unchanged_removal(tmp_path, monkeypatch):
    off = _run(_bundle(tmp_path, ledger=False))

    def boom(*a, **k):
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(rl, "capture_removal", boom)
    on = _run(_bundle(tmp_path, ledger=True))
    assert np.array_equal(off["combined_image"], on["combined_image"], equal_nan=True)
    assert on["ledger_path"] is None
    err = rl.ledger_fingerprint_dir(tmp_path, "skycell.2528", "005", "fp_test") / "error.json"
    assert "synthetic capture failure" in json.loads(err.read_text())["error"]


def test_projection_catalog_order():
    order = pp._projection_catalog_order(
        ["skycell.2528", "skycell.2529"],
        {"skycell.2600.001": "skycell.2600", "skycell.2601.002": "skycell.2601"},
        {("skycell.2529", 3): {"skycell.2600.001"}, ("skycell.2528", 1): {"skycell.2601.002"}},
    )
    assert order == ["2528", "2601", "2529", "2600"]


class _FakeProvider:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def catalog(self, pid, subset="removal"):
        self.calls.append((pid, subset))
        if pid not in self.frames:
            raise RuntimeError("download failed")
        return self.frames[pid]

    def wait(self, pid, timeout=None):
        return self.catalog(pid)


def test_cell_catalog_resolution_uses_own_projection():
    a = pd.DataFrame(dict(source_id=pd.array([1, 2], dtype="Int64"), ra=[0.0, 1.0], dec=[0.0, 1.0]))
    b = pd.DataFrame(dict(source_id=pd.array([2, 3], dtype="Int64"), ra=[1.0, 2.0], dec=[1.0, 2.0]))
    prov = _FakeProvider({"2528": a, "2529": b})
    assert pp._cell_removal_catalog(prov, "skycell.2528.005") is a
    assert pp._cell_removal_catalog(prov, "skycell.9999.001") is None  # failure -> guard refuses the cell
    union = pp._removal_catalog_for_cells(prov, ["skycell.2528.005", "skycell.2529.010", "skycell.2528.006"])
    assert union.source_id.tolist() == [1, 2, 3]
    assert pp._catalog_failures(prov, ["2528", "9999"]) == {"9999": "RuntimeError: download failed"}
