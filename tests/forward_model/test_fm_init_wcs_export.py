# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for init-study WCS coeff loading."""

from __future__ import annotations

import numpy as np
import pytest

from syndiff_pipeline.forward_model import temporal as T
from syndiff_pipeline.forward_model.init_study.init_wcs_export import load_init_wcs_coeff


class _Frame:
    def __init__(self, stem: str, btjd: float):
        self.stem = stem
        self.btjd = btjd


def test_load_init_wcs_coeff_validates_stems(tmp_path):
    stems = ["a", "b", "c"]
    btjd = np.array([1.0, 2.0, 3.0])
    wcs_tb = T.build_temporal_basis(btjd, n_interior=2, uniform_knots=True)
    coeff = np.zeros((30, wcs_tb.n_basis), dtype=np.float32)
    path = tmp_path / "wcs_coeff.npz"
    np.savez(
        path,
        wcs_coeff=coeff,
        wcs_frame_basis=np.asarray(wcs_tb.frame_basis, dtype=np.float32),
        btjd=btjd,
        stems=np.asarray(stems, dtype=object),
        n_terms=15,
        cheb_poly_degree=4,
    )
    frames = [_Frame(s, float(b)) for s, b in zip(stems, btjd)]
    out, meta = load_init_wcs_coeff(path, frames, wcs_tb, cheb_degree=4)
    assert out.shape == coeff.shape
    assert meta["n_frames"] == 3

    with pytest.raises(ValueError, match="stems mismatch"):
        load_init_wcs_coeff(path, [_Frame("x", 1.0)], wcs_tb, cheb_degree=4)
