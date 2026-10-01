# Migrated from dev/forward_epsf_wcs (dev commit 7c8af4f) by tools/forward_model_transplant.py.
"""Tests for ``temporal.override_w_frame_basis``/``save_w_frame_basis_sidecar``
(task T1-2: swap in a wider temporal basis without touching fit_bundle.py's
2GB npz or the concurrently-edited train_from_bundle.py/loss.py/fit.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from syndiff_pipeline.forward_model import temporal as T


def test_override_replaces_basis_and_checks_frame_count():
    fake_bundle = SimpleNamespace(w_frame_basis=np.zeros((100, 7), dtype=np.float32))
    new = np.ones((100, 17), dtype=np.float32)
    out = T.override_w_frame_basis(fake_bundle, new)
    assert out is fake_bundle
    assert fake_bundle.w_frame_basis.shape == (100, 17)
    np.testing.assert_array_equal(fake_bundle.w_frame_basis, new)


def test_override_rejects_frame_count_mismatch():
    fake_bundle = SimpleNamespace(w_frame_basis=np.zeros((100, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        T.override_w_frame_basis(fake_bundle, np.ones((99, 17), dtype=np.float32))


def test_override_rejects_non_2d():
    fake_bundle = SimpleNamespace(w_frame_basis=np.zeros((100, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        T.override_w_frame_basis(fake_bundle, np.ones((100,), dtype=np.float32))


def test_sidecar_round_trips(tmp_path):
    btjd = np.linspace(2718.64, 2730.35, 500)
    basis = T.build_temporal_basis_gap_aware(btjd, gap_btjd=(2724.4307 - 0.02, 2724.4307 + 0.02))
    path = tmp_path / "w_frame_basis_sidecar.npz"
    T.save_w_frame_basis_sidecar(path, basis)

    loaded = np.load(path)
    np.testing.assert_allclose(loaded["w_frame_basis"], np.asarray(basis.frame_basis))
    np.testing.assert_allclose(loaded["btjd_ref"], basis.btjd_ref)
    np.testing.assert_allclose(loaded["btjd_scale"], basis.btjd_scale)

    fake_bundle = SimpleNamespace(w_frame_basis=np.zeros((btjd.shape[0], 7), dtype=np.float32))
    T.override_w_frame_basis(fake_bundle, loaded["w_frame_basis"])
    assert fake_bundle.w_frame_basis.shape[1] == basis.n_basis
