"""Band-cell provenance cards in linear-mode template FITS headers (flag on) and byte-identity (flag off)."""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from syndiff_pipeline.template_creation.processing import linear_downsample as ld

CARDS = ("BANDWR", "BANDWI", "BANDWZ", "BANDWY", "BANDSRC", "CONVBC")
W = {"r": 0.254, "i": 0.4368, "z": 0.1654, "y": 0.1438}


def _base_header():
    h = fits.Header()
    h["OVERSAMP"] = 1
    h["XMIN"], h["XMAX"], h["YMIN"], h["YMAX"] = 0, 6, 0, 4
    return h


def _write(tmp_path, header):
    groups = [(0, 0.0, 0.0), (1, 0.25, -0.125)]
    sums = {g: np.full((4, 6), 1.5, np.float32) for g, _, _ in groups}
    cnt = {g: np.ones((4, 6), np.int32) for g, _, _ in groups}
    msk = {g: np.zeros((4, 6), np.int32) for g, _, _ in groups}
    return ld._write_group_templates(
        tmp_path, groups, header, sums, cnt, msk, sector=26, camera=3, ccd=1, oversampling_factor=1,
    )


def test_flag_off_headers_unchanged(tmp_path):
    off = _write(tmp_path, _base_header())
    for p in off:
        with fits.open(p) as h:
            for hdu in h:
                assert not any(k in hdu.header for k in CARDS)
    ref = _base_header()
    with fits.open(off[0]) as h:
        for k in ref:
            assert h[1].header[k] == ref[k]


def test_flag_on_cards_all_extensions_and_diff_stage_reader(tmp_path):
    hdr = _base_header()
    ld.add_band_cell_header_cards(hdr, W)
    (tmp_path / "b").mkdir()
    on = _write(tmp_path / "b", hdr)
    for p in on:
        with fits.open(p) as h:
            assert [x.name for x in h][1:] == ["FLUX_SUM", "COUNT", "MASK"]
            for hdu in h:
                assert hdu.header["BANDWR"] == 0.254
                assert hdu.header["BANDWI"] == 0.4368
                assert hdu.header["BANDWZ"] == 0.1654
                assert hdu.header["BANDWY"] == 0.1438
                assert hdu.header["BANDSRC"] == "band_cells"
                assert hdu.header["CONVBC"] is True
                assert hdu.header.comments["BANDWR"] == "template band weight"
    # added cards are the only difference
    (tmp_path / "c").mkdir()
    off = _write(tmp_path / "c", _base_header())
    with fits.open(on[1]) as a, fits.open(off[1]) as b:
        extra = [k for k in a[1].header if k not in b[1].header]
        assert sorted(extra) == sorted(CARDS)
        np.testing.assert_array_equal(a[1].data, b[1].data)


def test_diff_stage_template_readers_tolerate_cards(tmp_path):
    from syndiff_pipeline.common import template_coverage as tc

    ph = _base_header()
    pn = _write(tmp_path, ph)[0]
    (tmp_path / "w").mkdir()
    hdr = _base_header()
    ld.add_band_cell_header_cards(hdr, W)
    pw = _write(tmp_path / "w", hdr)[0]
    crop = {"x_min": 1, "x_max": 4, "y_min": 1, "y_max": 3}
    # the readers the diff/hotpants stages use: same answers with and without the cards
    assert tc.template_coverage_ffi_bounds(pw) == tc.template_coverage_ffi_bounds(pn)
    assert tc.template_crop_slices(pw, crop) == tc.template_crop_slices(pn, crop)
    np.testing.assert_array_equal(tc.load_template_count_cropped(pw, crop), tc.load_template_count_cropped(pn, crop))
