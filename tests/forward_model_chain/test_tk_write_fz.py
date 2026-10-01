"""Chain FITS products are written the production way (common/fits_io: fpack -g -q 0 -> ZQUANTIZ 'NONE')."""
import numpy as np
import pytest
from astropy.io import fits

from syndiff_pipeline.forward_model.chain import _tk as TK


def test_write_fz_is_lossless_fpack_with_zquantiz_none(tmp_path):
    rng = np.random.default_rng(1)
    diff = rng.normal(0, 3, (64, 48)).astype(np.float32)
    diff[3, 4] = np.nan
    noise = np.abs(rng.normal(1, 0.1, (64, 48))).astype(np.float32)
    mask = rng.integers(0, 64, (64, 48)).astype(np.int32)
    hdr = fits.Header({"EXTNAME": "X", "BUNIT": "electrons/s"})
    primary = fits.Header({"FFISTEM": "t"})
    out = TK.write_fz(tmp_path / "a" / "x_hp_d.fits.fz", primary, [(diff, hdr), (noise, hdr), (mask, hdr)])
    assert out.name == "x_hp_d.fits.fz" and not (tmp_path / "a" / "x_hp_d.fits").exists()
    with fits.open(out) as h:
        assert np.array_equal(h[1].data, diff, equal_nan=True) and np.array_equal(h[2].data, noise)
        assert np.array_equal(h[3].data, mask) and h[0].header["FFISTEM"] == "t" and h[1].header["BUNIT"] == "electrons/s"
    with fits.open(out, disable_image_compression=True) as raw:
        assert raw[1].header["ZQUANTIZ"] == "NONE" and raw[2].header["ZQUANTIZ"] == "NONE"
        assert raw[1].header["ZCMPTYPE"].startswith("GZIP")
