"""Full-FFI boundary for the crop-local temporal WCS model.

The fitted temporal Chebyshev model intentionally remains in the science
crop's pixel frame.  This module is the only production boundary that
translates that frame to the full detector coordinates used by mapping and
remapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalWcsAdapter:
    """Astropy-compatible temporal WCS view with a full-FFI pixel contract."""

    model: object
    btjd: float
    model_origin_ffi: tuple[float, float]

    def __post_init__(self):
        origin = np.asarray(self.model_origin_ffi, dtype=float)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("model_origin_ffi must contain two finite coordinates")
        object.__setattr__(self, "model_origin_ffi", (float(origin[0]), float(origin[1])))
        object.__setattr__(self, "btjd", float(self.btjd))

    def full_ffi_to_model_local(self, x_ffi, y_ffi):
        ox, oy = self.model_origin_ffi
        return np.asarray(x_ffi, dtype=float) - ox, np.asarray(y_ffi, dtype=float) - oy

    def model_local_to_full_ffi(self, x_local, y_local):
        ox, oy = self.model_origin_ffi
        return np.asarray(x_local, dtype=float) + ox, np.asarray(y_local, dtype=float) + oy

    def pixel_to_world(self, x_ffi, y_ffi):
        x_local, y_local = self.full_ffi_to_model_local(x_ffi, y_ffi)
        return self.model.pixel_to_world(x_local, y_local, self.btjd)

    def all_pix2world(self, x, y=None, origin=0):
        # Astropy-WCS convenience form: all_pix2world(pixel_array_Nx2, origin).
        # Detected when the caller passes only two positional args and the
        # second one is a bare scalar "origin" rather than a real y array.
        packed_call = (
            y is not None and np.ndim(y) == 0 and np.asarray(x).ndim >= 1
            and np.asarray(x).shape[-1] == 2 and origin == 0
        )
        if packed_call:
            x, y = np.asarray(x)[..., 0], np.asarray(x)[..., 1]
        if y is None:
            pixels = np.asarray(x)
            if pixels.shape[-1] != 2:
                raise ValueError("pixel coordinates must have a final dimension of 2")
            ra, dec = self.pixel_to_world(pixels[..., 0], pixels[..., 1])
            return np.stack((ra, dec), axis=-1)
        ra, dec = self.pixel_to_world(x, y)
        if packed_call:
            return np.stack((ra, dec), axis=-1)
        return ra, dec

    def pixel_to_world_values(self, x, y=None):
        if y is None:
            pixels = np.asarray(x)
            if pixels.shape[-1] != 2:
                raise ValueError("pixel coordinates must have a final dimension of 2")
            x, y = pixels[..., 0], pixels[..., 1]
        return self.pixel_to_world(x, y)

    def world_to_pixel_values(self, ra, dec):
        x_local, y_local = self.model.world_to_pixel_values(ra, dec, self.btjd)
        return self.model_local_to_full_ffi(x_local, y_local)

    def at_time(self, btjd):
        """Return a view at another cadence while retaining the same frame contract."""
        return type(self)(self.model, float(btjd), self.model_origin_ffi)


__all__ = ["TemporalWcsAdapter"]
