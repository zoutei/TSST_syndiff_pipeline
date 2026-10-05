"""Tests for per-projection Gaia catalogues (no network; downloader injected)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from syndiff_pipeline.template_creation.processing import gaia_projection_catalog as gpc
from syndiff_pipeline.template_creation.orchestration.bundled_assets import skycell_wcs_csv

PROJ = "2486"


def _cells():
    t = pd.read_csv(skycell_wcs_csv())
    return t[t["NAME"].str.startswith(f"skycell.{PROJ}.")]


def _fake_downloader_factory(calls):
    def _dl(ra, dec, mag_limit):
        calls.append((len(ra), mag_limit))
        c = _cells().iloc[0]
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [c.CRVAL1, c.CRVAL2]
        w.wcs.crpix = [0.0, 0.0]
        w.wcs.cdelt = [1.0, 1.0]
        # Tangent-plane offsets (deg): a few inside the footprint, two far outside.
        pts = np.array([[0.0, 0.0], [0.5, 0.3], [-0.4, 0.2], [30.0, 30.0], [-40.0, 10.0]])
        r, d = w.wcs_pix2world(pts[:, 0], pts[:, 1], 1)
        return pd.DataFrame({
            "source_id": pd.array([50, 10, 30, 20, 40], dtype="Int64"),
            "ra": r, "dec": d,
            "phot_g_mean_mag": 12.0, "phot_bp_mean_mag": 12.5, "phot_rp_mean_mag": 11.5,
            "parallax": 1.0,
        })
    return _dl


class TestProjectionId:
    @pytest.mark.parametrize("v", ["skycell.2486.085", "skycell.2486", "2486", 2486])
    def test_parsing(self, v):
        assert gpc.projection_id(v) == "2486"

    def test_zero_pad_and_bad(self):
        assert gpc.projection_id("skycell.635.001") == "0635"
        with pytest.raises(ValueError):
            gpc.projection_id("tess")


class TestPolygon:
    def test_contains_every_cell_centre_and_is_scc_independent(self):
        ra, dec = gpc.projection_footprint_polygon(PROJ)
        ra2, dec2 = gpc.projection_footprint_polygon("skycell.2486.085")
        np.testing.assert_array_equal(ra, ra2)
        np.testing.assert_array_equal(dec, dec2)
        assert len(ra) == 4 * 50

        from syndiff_pipeline.template_creation.processing import pancakes

        cells = _cells()
        assert len(cells) > 1
        centres = []
        for c in cells.itertuples(index=False):
            w = WCS(naxis=2)
            w.wcs.ctype = [c.CTYPE1, c.CTYPE2]
            w.wcs.crval = [c.CRVAL1, c.CRVAL2]
            w.wcs.crpix = [c.CRPIX1, c.CRPIX2]
            w.wcs.cdelt = [c.CDELT1, c.CDELT2]
            w.wcs.pc = [[c.PC1_1, c.PC1_2], [c.PC2_1, c.PC2_2]]
            centres.append(w.wcs_pix2world([[c.NAXIS1 / 2.0, c.NAXIS2 / 2.0]], 0)[0])
        centres = np.array(centres)
        df = pd.DataFrame({"ra": centres[:, 0], "dec": centres[:, 1]})
        kept = pancakes.filter_gaia_dataframe_to_polygon(df, ra, dec)
        assert len(kept) == len(df)

    def test_margin_grows_polygon(self):
        _, d0 = gpc.projection_footprint_polygon(PROJ, margin_px=0)
        _, d1 = gpc.projection_footprint_polygon(PROJ, margin_px=480)
        assert (d1.max() - d1.min()) > (d0.max() - d0.min())


class TestEnsureCatalog:
    def test_download_once_and_meta(self, tmp_path):
        calls = []
        dl = _fake_downloader_factory(calls)
        path = gpc.ensure_projection_catalog(tmp_path, PROJ, downloader=dl)
        assert path == gpc.projection_catalog_path(tmp_path, PROJ)
        assert path.name == "proj_2486.parquet"
        assert gpc.GAIA_PROJECTION_SCHEME in str(path)
        meta_path = path.with_name(path.name + ".meta.json")
        assert path.is_file() and meta_path.is_file()
        assert len(calls) == 1 and calls[0][1] == 18.0

        meta = json.loads(meta_path.read_text())
        assert meta["scheme"] == gpc.GAIA_PROJECTION_SCHEME
        assert meta["projection"] == PROJ and meta["release"] == "gaiadr3"
        assert meta["margin_px"] == gpc.DEFAULT_MARGIN_PX == 600 and meta["magnitude_limit"] == 18
        assert len(meta["polygon_ra"]) == len(meta["polygon_dec"]) == 200
        assert meta["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

        df = gpc.load_projection_catalog(tmp_path, PROJ)
        assert meta["n_rows"] == len(df) == 3  # two far-away stars filtered out
        assert df["source_id"].tolist() == [10, 30, 50]  # sorted by source_id
        assert str(df["source_id"].dtype) == "Int64"
        assert meta["content_sha256"] == gpc.content_sha256(df)

        before = path.read_bytes()
        again = gpc.ensure_projection_catalog(tmp_path, PROJ, downloader=dl)
        assert again == path and len(calls) == 1
        assert path.read_bytes() == before

        fp = gpc.projection_catalog_fingerprint(tmp_path, PROJ)
        assert fp == f"{gpc.GAIA_PROJECTION_SCHEME}:{meta['content_sha256'][:24]}"

    def test_content_hash_independent_of_file_bytes_and_row_order(self, tmp_path):
        """Two data roots that built the same projection get the same fingerprint."""
        dl = _fake_downloader_factory([])
        gpc.ensure_projection_catalog(tmp_path / "a", PROJ, downloader=dl)
        gpc.ensure_projection_catalog(tmp_path / "b", PROJ, downloader=dl)
        assert (gpc.projection_catalog_fingerprint(tmp_path / "a", PROJ)
                == gpc.projection_catalog_fingerprint(tmp_path / "b", PROJ))
        df = gpc.load_projection_catalog(tmp_path / "a", PROJ)
        assert gpc.content_sha256(df.iloc[::-1]) == gpc.content_sha256(df)

    def test_absent(self, tmp_path):
        assert gpc.projection_catalog_fingerprint(tmp_path, PROJ) is None
        with pytest.raises(FileNotFoundError):
            gpc.load_projection_catalog(tmp_path, PROJ)


class TestLoadForProjections:
    def test_deduplicates_on_source_id(self, tmp_path):
        d = tmp_path
        for proj, ids in (("2486", [1, 2, 3]), ("2487", [3, 4])):
            p = gpc.projection_catalog_path(d, proj)
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "source_id": pd.array(ids, dtype="Int64"),
                "ra": 1.0, "dec": 2.0,
                "phot_g_mean_mag": 12.0, "phot_bp_mean_mag": 12.5, "phot_rp_mean_mag": 11.5,
            }).to_parquet(p, index=False)
        out = gpc.load_catalog_for_projections(d, ["2486", "skycell.2487.001", "2486"])
        assert out["source_id"].tolist() == [1, 2, 3, 4]
