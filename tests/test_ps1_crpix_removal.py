"""PS1 CRPIX convention (+0.5) applied where raw PS1 headers feed star removal."""
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from syndiff_pipeline.template_creation.processing import combined_store as cs
from syndiff_pipeline.template_creation.processing import correct_saturation as csat
from syndiff_pipeline.template_creation.processing import ps1_process

ENV = "SYNDIFF_PS1_CRPIX_CONVENTION"
SHAPE = (400, 400)


def _header_str():
    h = fits.Header()
    h["NAXIS"] = 2
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = 150.0
    h["CRVAL2"] = 2.0
    h["CRPIX1"] = 200.0
    h["CRPIX2"] = 200.0
    h["CDELT1"] = 6.9444446e-05
    h["CDELT2"] = 6.9444446e-05
    h["PC1_1"] = -1.0
    h["PC1_2"] = 0.0
    h["PC2_1"] = 0.0
    h["PC2_2"] = 1.0
    return h.tostring()


def _catalog():
    # star near the cell centre
    return pd.DataFrame(dict(
        ra=[150.0 + 3e-4], dec=[2.0 + 2e-4], phot_g_mean_mag=[12.0],
        phot_bp_mean_mag=[12.3], phot_rp_mean_mag=[11.6],
    ))


def _select(monkeypatch, conv):
    if conv is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, conv)
    out = ps1_process.select_removal_catalog(_catalog(), _header_str(), SHAPE, skycell_id="x")
    return float(out.pixel_x.iloc[0]), float(out.pixel_y.iloc[0])


def test_select_removal_catalog_shift(monkeypatch):
    x0, y0 = _select(monkeypatch, "fits")
    x1, y1 = _select(monkeypatch, None)
    assert x1 - x0 == pytest.approx(0.5, abs=1e-6)
    assert y1 - y0 == pytest.approx(0.5, abs=1e-6)


def test_cone_prefilter_uses_corrected_wcs(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    seen = []
    from syndiff_pipeline.common import ps1_wcs

    orig = ps1_wcs.ps1_wcs_from_header
    monkeypatch.setattr(ps1_wcs, "ps1_wcs_from_header", lambda h: seen.append(1) or orig(h))
    out = ps1_process.catalog_cone_prefilter(_catalog(), _header_str(), SHAPE)
    assert len(out) == 1 and seen


def test_ledger_association_uses_corrected_wcs(monkeypatch):
    from syndiff_pipeline.template_creation.processing.removal_ledger import inline, matching

    got = {}

    def fake_gaia_sources(df, wcs, epoch_year=None):
        got["crpix"] = np.array(wcs.wcs.crpix)
        raise RuntimeError("stop")

    monkeypatch.setattr(matching, "gaia_sources", fake_gaia_sources)
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(RuntimeError):
        inline.associate_gaia(None, None, None, _header_str(), _catalog())
    assert np.allclose(got["crpix"], [200.5, 200.5])
    monkeypatch.setenv(ENV, "fits")
    with pytest.raises(RuntimeError):
        inline.associate_gaia(None, None, None, _header_str(), _catalog())
    assert np.allclose(got["crpix"], [200.0, 200.0])


@pytest.mark.parametrize("conv,expect", [(None, 0.5), ("fits", 0.0)])
def test_ledger_metadata_records_offset(monkeypatch, conv, expect):
    from syndiff_pipeline.template_creation.processing.removal_ledger import inline as rl

    if conv is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, conv)
    seen = {}
    monkeypatch.setattr(rl, "capture_removal", lambda *a, **k: ("res", [], object(), None))
    monkeypatch.setattr(rl, "associate_gaia", lambda *a, **k: (None, None, None, None))
    monkeypatch.setattr(rl, "publish_cell_ledger", lambda *a, **k: seen.update(k) or "dest")
    bundle = dict(
        skycell_id="c", combined_image=None, combined_uncert=None, combined_mask=None,
        headers_data={"i": _header_str()},
        ledger=dict(combined_fingerprint="fp", data_root="/nonexistent", projection="p",
                    skycell="c", gaia_all=_catalog()),
    )
    ps1_process._capture_cell_ledger(bundle, None, ps1_process.band_utils.REMOVAL_CONVENTION_FOOTPRINT,
                                     remove_saturated_stars=True, bright_star_mag_threshold=13.0)
    assert seen["metadata"]["ps1_crpix_offset"] == expect
    assert seen["metadata"]["wcs_header"] == _header_str()


def test_correct_saturation_row_wcs(monkeypatch):
    got = {}

    def fake_filter(cat, wcs, *a):
        got["crpix"] = np.array(wcs.wcs.crpix)
        return cat.iloc[0:0]

    monkeypatch.setattr(csat, "filter_catalog_for_region", fake_filter)
    monkeypatch.delenv(ENV, raising=False)
    data = np.zeros((50, 50))
    csat.apply_saturation_to_row(
        data, {}, {"c": (0, 50, 0, 50)},
        [dict(x_coord=0, headers_data={"i": _header_str()})], _catalog(),
    )
    assert np.allclose(got["crpix"], [200.5, 200.5])


def _recipe():
    cfg = dict(enable_saturation_correction=False, remove_saturated_stars=True,
               bright_star_mag_threshold=13.0, band_weights=dict(r=1.0, i=1.0, z=1.0, y=1.0))
    return cs.production_combined_recipe(cfg)


def test_fingerprint_independent_of_convention(monkeypatch):
    """The CRPIX convention is deliberately NOT part of the combined recipe: footprint_v1 removal is
    bit-identical under both conventions in 25/26 real cells and differs by 2e-4 of the flux in one
    (dev_runs/ps1_wcs_convention_20261020/removal_ab), so cached cells stay valid."""
    monkeypatch.setenv(ENV, "fits")
    legacy = _recipe()
    monkeypatch.delenv(ENV, raising=False)
    new = _recipe()
    assert "ps1_crpix_offset" not in new
    assert cs.combined_recipe_id(new) == cs.combined_recipe_id(legacy)


def test_reference_header_is_deterministic():
    """One header per skycell regardless of band arrival order: i, else r, z, y."""
    from syndiff_pipeline.common.ps1_wcs import ps1_reference_header
    h = {"y": "Y", "z": "Z", "i": "I", "r": "R"}
    assert ps1_reference_header(h) == "I"
    assert ps1_reference_header(dict(reversed(list(h.items())))) == "I"
    assert ps1_reference_header({"z": "Z", "r": "R"}) == "R"
    assert ps1_reference_header({"y": "Y", "z": "Z"}) == "Z"
    assert ps1_reference_header({"g": "G"}) == "G"
