"""Tests for the ``starmodel_v1`` star-removal convention (production since combined schema 3).

``footprint_v1`` zeroed a bright star's whole 8-connected footprint, so a faint
source whose segment touched the bright star's halo was deleted with it: the
template held a hole at every removed star (dev_runs/removal_hole_20261001).
``starmodel_v1`` subtracts a radial model of the star and zeroes only its core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_band_utils_removal import _catalog, _cell_wcs, _gauss  # noqa: E402

from syndiff_pipeline.template_creation.processing.band_utils import (  # noqa: E402
    REMOVAL_CONVENTION,
    REMOVAL_CONVENTION_FOOTPRINT,
    REMOVAL_CONVENTION_STARMODEL,
    remove_background,
    select_catalog_for_cell,
)

N = 600
BX, BY = 300.0, 300.0            # bright star (T = 10)
FX, FY = 360.0, 300.0            # faint neighbour on its halo (T = 16), 60 px away
FAINT_AMP, FAINT_SIG = 30.0, 2.5


def _cell(seed=3):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 0.05, (N, N))
    img += _gauss((N, N), BX, BY, 3000.0, 6.0) + _gauss((N, N), BX, BY, 2.0, 40.0)  # core + broad halo (detected, below the core threshold at 60 px)
    img += _gauss((N, N), FX, FY, FAINT_AMP, FAINT_SIG)
    img = img.astype(np.float32)
    img[0, :] = img[-1, :] = np.nan
    img[:, 0] = img[:, -1] = np.nan
    return img, np.full_like(img, 0.1)


def _cat():
    # _catalog takes sky (x, y); the cell WCS at x0 = 0 is the sky WCS.
    return select_catalog_for_cell(_catalog([(BX, BY, 10.0, 111), (FX, FY, 16.0, 222)]), _cell_wcs(0), (N, N))


def _run(convention, mask=None, img=None):
    img0, unc = _cell()
    img = img0 if img is None else img
    return remove_background(img.copy(), unc, sigma=2.5, sigma_mask=50, mask=mask, remove_saturated_stars=True,
                             gaia_catalog_pixels=_cat(), bright_star_mag_threshold=13.0, convention=convention)


def _box(img, x, y, r):
    return float(np.nansum(img[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1]))


def test_conventions_and_recipe_agree():
    from syndiff_pipeline.template_creation.processing.combined_store import BRIGHT_STAR_REMOVAL_CONVENTION

    assert REMOVAL_CONVENTION_STARMODEL == "starmodel_v1"
    assert BRIGHT_STAR_REMOVAL_CONVENTION == REMOVAL_CONVENTION  # the recipe records what production runs


def test_faint_neighbour_on_the_halo_is_kept():
    true_flux = FAINT_AMP * 2 * np.pi * FAINT_SIG ** 2
    v1, _ = _run(REMOVAL_CONVENTION_FOOTPRINT)
    sm, _ = _run(REMOVAL_CONVENTION_STARMODEL)
    assert _box(v1, FX, FY, 8) < 0.05 * true_flux           # footprint_v1 deletes it (the hole)
    assert abs(_box(sm, FX, FY, 8) - true_flux) < 0.25 * true_flux


def test_bright_star_light_is_removed():
    img, _ = _cell()
    sm, recs = _run(REMOVAL_CONVENTION_STARMODEL)
    star_flux = 3000.0 * 2 * np.pi * 36 + 2.0 * 2 * np.pi * 1600
    yy, xx = np.mgrid[0:N, 0:N]
    away = np.hypot(xx - FX, yy - FY) > 10
    near = (np.hypot(xx - BX, yy - BY) < 200) & away
    assert abs(float(np.nansum(sm[near]))) < 0.01 * star_flux
    assert [r["source_id"] for r in recs if r["removal_reason"] == "catalog_bright_star"] == [111]


def test_uncatalogued_saturated_blob_is_still_zeroed():
    img, _ = _cell()
    img += _gauss((N, N), 100.0, 500.0, 500.0, 3.0).astype(np.float32)
    mask = np.zeros((N, N), dtype=np.uint16)
    mask[498:503, 98:103] = 0x0020 | 0x1000
    sm, recs = _run(REMOVAL_CONVENTION_STARMODEL, mask=mask, img=img)
    assert _box(sm, 100, 500, 6) == 0.0
    assert any(r["removal_reason"] == "quality_flag_no_star" for r in recs)
    # ...but the bright star's own saturated core does not drag its neighbour into that pass
    mask2 = np.zeros((N, N), dtype=np.uint16)
    mask2[298:303, 298:303] = 0x0020 | 0x1000
    sm2, _ = _run(REMOVAL_CONVENTION_STARMODEL, mask=mask2)
    true_flux = FAINT_AMP * 2 * np.pi * FAINT_SIG ** 2
    assert abs(_box(sm2, FX, FY, 8) - true_flux) < 0.25 * true_flux


def test_background_is_zeroed_away_from_stars():
    sm, _ = _run(REMOVAL_CONVENTION_STARMODEL)
    assert np.mean(sm[20:80, 20:80] == 0) > 0.9
