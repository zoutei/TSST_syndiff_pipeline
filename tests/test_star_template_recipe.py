import numpy as np
import pytest
import yaml
from syndiff_pipeline.star.template_recipe import TemplateRecipe, retained_component


def test_frozen_weights_and_width(tmp_path):
    weights = dict(r=0.1, i=0.2, z=0.3, y=0.4)
    (tmp_path / "f.yaml").write_text(
        yaml.safe_dump(
            dict(
                stages=dict(
                    ps1_process=dict(psf_sigma=40),
                    downsample=dict(
                        band_weights=weights,
                        ignore_mask_bits=[12],
                        zpcell_factors_path=None,
                    ),
                )
            )
        )
    )
    (tmp_path / "cells.json").write_text("{}")
    path = tmp_path / "operator.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                frozen_config="f.yaml",
                kernel_radius=470,
                geometry="canonical_v3",
                cell_index="cells.json",
                removal_recipe_id="test",
                registration_grid=dict(width=2064, origin_full_ffi=[36, -8]),
            )
        )
    )
    recipe = TemplateRecipe.load(path, dict(band_weights=weights))
    assert (recipe.sigma, recipe.weights, recipe.veto_bits) == (40, weights, 4096)
    with pytest.raises(ValueError, match="disagree"):
        TemplateRecipe.load(path, dict(band_weights={}))


def test_retained_footprint_and_blend():
    stack = np.zeros((4, 10, 10))
    stack[:, 5, 5:7] = 1
    stack[:, 0, 0] = 2
    out, flag = retained_component(stack, (15, 25), (10, 20), (100, 100), [(16, 25)])
    assert flag == "shared_PS1_footprint"
    assert np.all(out.sum(axis=(1, 2)) == 2)


def test_unweighted_donor_uses_frozen_support_and_cell_factors(tmp_path):
    import json
    from types import SimpleNamespace

    from astropy.io import fits
    from astropy.wcs import WCS
    import pandas as pd

    from syndiff_pipeline.star.template_recipe import read_donor

    cell = "skycell.2000.001"
    archive = tmp_path / "skycell.2000" / "001" / "id"
    archive.mkdir(parents=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [135, 12]
    wcs.wcs.crpix = [7, 8]
    wcs.wcs.cd = np.diag([-0.001, 0.001])
    hdr = wcs.to_header()
    hdr["MJD-OBS"] = 57388
    (archive / "headers.json").write_text(json.dumps({"i": hdr.tostring()}))
    provenance = archive / "_provenance.json"
    prov = dict(recipe_id="frozen-removal", meta=dict(band_weighted=False, missing_bands=[]))
    provenance.write_text(json.dumps(prov))
    mask = np.zeros((21, 21), "u2")
    mask[10, 11] = 4096
    values = [3., 5., 7., 11.]
    bands = "rizy"
    planes = {}
    for band, value in zip(bands, values):
        data = np.zeros(mask.shape, "f4")
        data[7, 6] = value
        data[10, 11] = np.nan
        data[15, 16] = 10
        planes[f"band_{band}"] = data
    np.savez(archive / "arrays.npz", combined_mask=mask, **planes)
    (tmp_path / "index.json").write_text(json.dumps({cell: str(archive)}))
    (tmp_path / "factors.json").write_text(json.dumps({cell: dict(zip(bands, (2, 3, 4, 5)))}))
    spec = tmp_path / "operator.yaml"
    spec.write_text(yaml.safe_dump(dict(
        psf_sigma=40, kernel_radius=470, band_weights=dict.fromkeys(bands, .25),
        ignore_mask_bits=[12], zpcell_factors_path="factors.json",
        geometry="canonical_v3", removal_recipe_id="frozen-removal",
        cell_index="index.json", registration_grid=dict(width=21, origin_full_ffi=[0, 0]),
    )))
    recipe = TemplateRecipe.load(spec)
    host = SimpleNamespace(ra=135, dec=12, pmra=None, pmdec=None, gaia_source_id=101)
    row = pd.Series(dict(hdr) | {"NAXIS1": 21, "NAXIS2": 21,
        "CD1_1": -.001, "CD1_2": 0., "CD2_1": 0., "CD2_2": .001})
    donor = read_donor(recipe, cell, row, host, tmp_path)
    valid, calibrated = donor.read((0, 0, 21, 21))
    assert not valid[10, 11]
    assert donor.flag == "retained"
    np.testing.assert_array_equal(calibrated[:, 15, 16], 0)
    np.testing.assert_allclose(calibrated[:, 7, 6], np.array(values) / (2, 3, 4, 5))
    prov["meta"]["band_weighted"] = True
    provenance.write_text(json.dumps(prov))
    with pytest.raises(ValueError, match="unweighted"):
        read_donor(recipe, cell, row, host, tmp_path)


def test_row_overlap_is_single_writer():
    from types import SimpleNamespace
    import pandas as pd
    from syndiff_pipeline.star.component_window import ComponentOperator
    from syndiff_pipeline.star.template_recipe import Donor

    name_a, name_b = "skycell.2000.001", "skycell.2000.011"
    shape = (510, 512)
    rows = pd.DataFrame([
        dict(NAME=name, projection=2000, x=1, y=row, NAXIS1=shape[1], NAXIS2=shape[0])
        for name, row in ((name_a, 0), (name_b, 1))
    ])
    bands = np.ones((4, *shape), dtype="f4")
    bits = np.zeros((shape[0], shape[1] // 8), "u1")
    donors = {name: Donor(shape, bits, bits, (0, 0), value, "retained")
              for name, value in ((name_a, bands), (name_b, bands * 2))}
    current = ComponentOperator(SimpleNamespace(sigma=4, radius=10, geometry="canonical_v3"),
                                rows, donors.__getitem__)
    previous = ComponentOperator(
        SimpleNamespace(sigma=4, radius=10, geometry="june_row", cells=donors),
        rows, donors.__getitem__,
    )
    rect = (60, 0, 61, 24)
    valid, light = current.canonical(name_b, rect, True)
    assert valid.all()
    np.testing.assert_array_equal(light, 1)
    _, historical = previous.canonical(name_b, rect, True)
    np.testing.assert_array_equal(historical[:, :10], 1)
    np.testing.assert_array_equal(historical[:, 10:], 2)
