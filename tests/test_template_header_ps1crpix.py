"""Template headers must round-trip the MappingGrid (incl. PS1CRPIX = 0.5) through MappingGrid.from_fits_header.

Regression (2026-10-28, S26 C2K2): linear/field template writers cast every non-string MappingGrid card to int, so
PS1CRPIX 0.5 became 0 while GEOMFP kept the 0.5 fingerprint -> from_fits_header raised 'FITS GEOMFP does not match'."""
import inspect

from syndiff_pipeline.common.mapping_grid import MappingGrid
from syndiff_pipeline.template_creation.processing import field_templates, linear_downsample


def _grid(ps1_crpix):
    return MappingGrid.from_ffi_shape(2078, 2136, oversampling=1, ps1_crpix=ps1_crpix)


def test_field_header_round_trips_ps1crpix():
    for off in (0.0, 0.5):
        g = _grid(off)
        hdr = field_templates.build_field_fits_header(sector=26, camera=2, ccd=2, group_id=0, mapping_grid=g)
        back = MappingGrid.from_fits_header(hdr)
        assert back.ps1_crpix == off
        assert back.geometry_fingerprint == g.geometry_fingerprint


def test_linear_writer_keeps_float_cards():
    src = inspect.getsource(linear_downsample)
    assert "isinstance(value, (str, bool, float))" in src
