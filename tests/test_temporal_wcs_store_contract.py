"""Small, data-root-free contracts for the temporal-WCS runtime artifact."""

import json

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import (
    TemporalChebWcs,
    TemporalChebWcsStore,
    chebyshev_design,
    temporal_frame_contract,
)
from syndiff_pipeline.difference_imaging.wcs.temporal_adapter import TemporalWcsAdapter


def _reference_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [100.0, 100.0]
    wcs.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _store(tmp_path, *, duplicate=False):
    model = TemporalChebWcs.from_reference_wcs(
        _reference_wcs(), center=[100.0, 100.0], half_extents=[100.0, 100.0],
        btjd_ref=100.0, btjd_scale=1.0,
    )
    model_path = tmp_path / "models" / "orbit_00.npz"
    model.save(model_path)
    rows = pd.DataFrame(
        {
            "stem": ["ffi_a", "ffi_a" if duplicate else "ffi_b"],
            "btjd": [100.1, 100.9],
            "orbit_index": [0, 0],
        }
    )
    rows.to_parquet(tmp_path / "frames.parquet", index=False)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "model_kind": "temporal_wcs",
                "spatial_basis": "chebyshev",
                "spatial_degree": 5,
                "frame_contract": temporal_frame_contract(origin_ffi=(44, 0), shape=(200, 200)),
                "models": [{"orbit_index": 0, "path": "models/orbit_00.npz"}],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_chebyshev_design_has_stable_total_degree_order():
    design = chebyshev_design(np.array([0.0, 0.5]), np.array([0.0, -0.5]), 5)
    assert design.shape == (2, 21)
    # The first term is T0(x)T0(y), followed by the degree-one terms.
    assert np.allclose(design[:, 0], 1.0)
    assert np.allclose(design[:, 1], np.array([0.0, -0.5]))
    assert np.allclose(design[:, 2], np.array([0.0, 0.5]))


def test_store_resolves_stem_and_returns_frame_time(tmp_path):
    store = TemporalChebWcsStore(_store(tmp_path))
    model, btjd = store.for_stem("ffi_b")
    assert isinstance(model, TemporalWcsAdapter)
    assert btjd == pytest.approx(100.9)
    # The orbit model is cached, so all stems in one orbit share the object.
    model_a, _ = store.for_stem("ffi_a")
    assert isinstance(model_a, TemporalWcsAdapter)
    assert len(store.fingerprint) == 64


def test_store_resolves_standard_spoc_ffi_basename(tmp_path):
    root = _store(tmp_path)
    rows = pd.read_parquet(root / "frames.parquet")
    rows.loc[0, "stem"] = "tess2020007215923-s0020-3-3"
    rows.to_parquet(root / "frames.parquet", index=False)

    store = TemporalChebWcsStore(root)
    model, btjd = store.for_stem(
        "tess2020007215923-s0020-3-3-0165-s_ffic.fits.fz"
    )
    assert isinstance(model, TemporalWcsAdapter)
    assert btjd == pytest.approx(100.1)


def test_store_rejects_duplicate_stem_rows(tmp_path):
    store = TemporalChebWcsStore(_store(tmp_path, duplicate=True))
    with pytest.raises(ValueError, match="duplicate"):
        store.for_stem("ffi_a")


def test_store_rejects_missing_frame_contract(tmp_path):
    root = _store(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    del manifest["frame_contract"]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="frame contract"):
        TemporalChebWcsStore(root)
