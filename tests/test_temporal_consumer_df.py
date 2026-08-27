"""Focused D--F regressions for full-FFI temporal-WCS consumers."""

from __future__ import annotations

import numpy as np

from syndiff_pipeline.difference_imaging.wcs.temporal_adapter import TemporalWcsAdapter
from syndiff_pipeline.template_creation.processing import field_remap


class _RawModel:
    """Tiny crop-local model exposing the runtime methods used by the adapter."""

    def pixel_to_world(self, x, y, btjd):
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def world_to_pixel_values(self, ra, dec, btjd):
        return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


def test_exact_workers_use_full_ffi_adapter_at_os1_and_os4(monkeypatch):
    class _Store:
        def __init__(self, root):
            self.root = root

        def for_stem(self, stem):
            return TemporalWcsAdapter(_RawModel(), 12.0, (44.0, 0.0)), 12.0

    monkeypatch.setattr(
        "syndiff_pipeline.difference_imaging.wcs.temporal_cheb.TemporalChebWcsStore",
        _Store,
    )
    for factor in (1, 4):
        field_remap._reset_remap_worker()
        field_remap._REMAP_WORKER.update(
            {
                "wcs_mode": "temporal",
                "temporal_wcs_dir": "/unused",
                "frame_filenames": ["ffi_a"],
                "oversampling_factor": factor,
            }
        )
        wcs = field_remap._worker_frame_wcs(0)
        assert isinstance(wcs, TemporalWcsAdapter)
        # The worker receives FFI coordinates; the adapter alone translates
        # them to the crop-local model (the old raw path returned 44 here).
        assert np.allclose(wcs.pixel_to_world_values([44.0], [0.0]), ([0.0], [0.0]))

