"""GWCS adapter for a fixed-epoch temporal Chebyshev detector WCS.

The temporal NPZ model remains authoritative.  This module supplies a
pickle-safe, fixed-BTJD GWCS view so callers can use the standard low-level
and high-level WCS APIs without reconstructing an Astropy WCS per coordinate.
"""

from __future__ import annotations

from astropy import coordinates as coord
from astropy import units as u
from astropy.modeling import Model
from gwcs import coordinate_frames as cf
from gwcs import wcs


class _PixelToSky(Model):
    n_inputs = 2
    n_outputs = 2
    input_units = {"x": u.pix, "y": u.pix}

    def __init__(self, temporal_model, btjd, **kwargs):
        super().__init__(**kwargs)
        self.temporal_model = temporal_model
        self.btjd = float(btjd)

    def evaluate(self, x, y):
        ra, dec = self.temporal_model.pixel_to_world(
            x.to_value(u.pix), y.to_value(u.pix), self.btjd
        )
        return ra * u.deg, dec * u.deg


class _SkyToPixel(Model):
    n_inputs = 2
    n_outputs = 2
    input_units = {"x": u.deg, "y": u.deg}

    def __init__(self, temporal_model, btjd, **kwargs):
        super().__init__(**kwargs)
        self.temporal_model = temporal_model
        self.btjd = float(btjd)

    def evaluate(self, ra, dec):
        x, y = self.temporal_model.world_to_pixel_values(
            ra.to_value(u.deg), dec.to_value(u.deg), self.btjd
        )
        return x * u.pix, y * u.pix


def build_fixed_time_gwcs(temporal_model, btjd: float, *, name: str = "temporal_wcs"):
    """Build a GWCS detector-to-ICRS view for one BTJD."""
    forward = _PixelToSky(temporal_model, btjd, name=f"{name}_pixel_to_sky")
    inverse = _SkyToPixel(temporal_model, btjd, name=f"{name}_sky_to_pixel")
    forward.inverse = inverse
    detector = cf.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )
    sky = cf.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )
    # Keep the detector bounds in the temporal model manifest.  GWCS bounding
    # boxes require unit-bearing bounds in some Astropy versions and are not
    # needed for coordinate evaluation; assigning a bare tuple here would make
    # valid pixel quantities fail before the transform is called.
    return wcs.WCS([(detector, forward), (sky, None)], name=name)


__all__ = ["build_fixed_time_gwcs"]
