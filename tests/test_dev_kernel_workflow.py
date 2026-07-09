"""Tests for dev/kernel_experiment (no Hotpants runtime required for unit tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dev.kernel_experiment.background import photutils_background_masked
from dev.kernel_experiment.compare import compare_kernels
from dev.kernel_experiment.context import find_template_by_offset
from dev.kernel_experiment.kernel import (
    build_kernel_basis,
    convolve_template_with_kernel_solution,
)
from syndiff_pipeline.difference_imaging.orchestration.stage_params import HotpantsParams
from syndiff_pipeline.difference_imaging.stages.hotpants import parse_syndiff_template_filename


def test_build_kernel_basis_shape():
    hp = HotpantsParams(sci_fwhm=1.0, hp_ngauss=3, hp_deg_fixe=[6, 4, 2])
    basis = build_kernel_basis(hp)
    assert basis.ndim == 3
    h, w = basis.shape[1], basis.shape[2]
    assert h == w and h % 2 == 1


def test_photutils_background_masked_fallback_on_tiny_image():
    image = np.ones((8, 8))
    mask = np.zeros((8, 8), dtype=np.int16)
    mask[0, :] = 1
    bkg = photutils_background_masked(image, mask, box_size=16)
    assert bkg.shape == image.shape
    assert np.isfinite(bkg).all()


def test_compare_kernels_identical():
    k = np.eye(5)
    cmp = compare_kernels(k, k.copy())
    assert cmp.l2 == pytest.approx(0.0)
    assert cmp.linf == pytest.approx(0.0)
    assert cmp.correlation == pytest.approx(1.0)


def test_parse_syndiff_template_dx0_dy0():
    name = "syndiff_template_s0020_3_3_dx0.000_dy0.000.fits.gz"
    parsed = parse_syndiff_template_filename(name)
    assert parsed is not None
    assert parsed.dx == pytest.approx(0.0)
    assert parsed.dy == pytest.approx(0.0)


def test_variant_label_names():
    from dev.kernel_experiment.masks import variant_label

    assert variant_label("gaia_shared", "gaia_shared") == "gaia__gaia"
    assert variant_label("gaia_shared", "sat_catalog") == "gaia__sat"
    assert variant_label("sat_catalog", "gaia_shared") == "sat__gaia"
    assert variant_label("sat_catalog", "sat_catalog") == "sat__sat"


def test_mask_for_variant_requires_sat_mask():
    from dev.kernel_experiment.masks import mask_for_variant

    shared = np.zeros((4, 4), dtype=np.int16)
    with pytest.raises(ValueError, match="sat_star_catalog_mask"):
        mask_for_variant(shared_mask=shared, sat_star_catalog_mask=None, variant="sat_catalog")

    sat = np.ones((4, 4), dtype=np.int16)
    out = mask_for_variant(shared_mask=shared, sat_star_catalog_mask=sat, variant="sat_catalog")
    assert np.array_equal(out, sat)


def test_catalog_df_with_bsc_appends_rows_and_bumps_bright_mags():
    from dev.kernel_experiment.masks import _catalog_df_with_bsc

    ps1 = __import__("pandas").DataFrame({"x": [1.0], "y": [2.0], "mag": [14.0]})
    bsc = __import__("pandas").DataFrame({"x": [5.0], "y": [6.0], "vmag": [5.5]})
    merged = _catalog_df_with_bsc(ps1, bsc)
    assert len(merged) == 2
    assert merged.iloc[-1]["mag"] == pytest.approx(7.01)


def test_parse_crop_bounds_from_targets_reg(tmp_path):
    reg = tmp_path / "targets.reg"
    reg.write_text(
        "# crop-local image coords (1-based); FFI ROI origin x_min=1112 y_min=992 size=1024x1024\n",
        encoding="utf-8",
    )
    from dev.kernel_experiment.crops import parse_crop_bounds_from_targets_reg

    bounds = parse_crop_bounds_from_targets_reg(str(tmp_path))
    assert bounds is not None
    assert bounds["x_min"] == 1112
    assert bounds["y_min"] == 992
    assert bounds["shape"] == (1024, 1024)

    tmpl = tmp_path / "syndiff_template_s0020_3_3_dx0.000_dy0.000.fits"
    tmpl.write_bytes(b"")
    other = tmp_path / "syndiff_template_s0020_3_3_dx0.010_dy0.000.fits"
    other.write_bytes(b"")
    found = find_template_by_offset(tmp_path, dx=0.0, dy=0.0)
    assert found.endswith("dx0.000_dy0.000.fits")


def test_load_ps1_removed_in_crop_reprojects_when_csv_roi_differs(tmp_path):
    """Full-chip CSV x,y must not be used when diff ROI is a target box."""
    from astropy.io import fits

    from dev.kernel_experiment.masks import _load_ps1_removed_in_crop

    ref_fits = tmp_path / "ref_ffi.fits"
    data = np.zeros((128, 128), dtype=np.float32)
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data=data)
    hdu1.header.update(
        {
            "NAXIS1": 128,
            "NAXIS2": 128,
            "CRPIX1": 64.0,
            "CRPIX2": 64.0,
            "CRVAL1": 210.0,
            "CRVAL2": 80.0,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
        }
    )
    fits.HDUList([hdu0, hdu1]).writeto(ref_fits, overwrite=True)

    from syndiff_pipeline.common.wcs_grouping import world_ra_dec_to_pixel
    from astropy.wcs import WCS

    wcs = WCS(hdu1.header)
    star_ra, star_dec = 210.0, 80.0
    x_ffi, y_ffi = world_ra_dec_to_pixel(wcs, star_ra, star_dec)

    crop_bounds = {
        "x_min": 40,
        "y_min": 30,
        "x_max": 104,
        "y_max": 94,
        "shape": (64, 64),
    }

    csv_path = tmp_path / "ps1_removed_stars.csv"
    pd.DataFrame(
        [
            {
                "source_id": 1234567890123456789,
                "ra": star_ra,
                "dec": star_dec,
                "tess_mag": 11.5,
                # Wrong crop-local coords (as if written for full-chip origin).
                "x": x_ffi,
                "y": y_ffi,
            }
        ]
    ).to_csv(csv_path, index=False)

    out = _load_ps1_removed_in_crop(str(csv_path), crop_bounds, str(ref_fits))
    assert len(out) == 1
    assert out.iloc[0]["x"] == pytest.approx(x_ffi - crop_bounds["x_min"], abs=0.5)
    assert out.iloc[0]["y"] == pytest.approx(y_ffi - crop_bounds["y_min"], abs=0.5)
    assert out.iloc[0]["mag"] == pytest.approx(11.5)
