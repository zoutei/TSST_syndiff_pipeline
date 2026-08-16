"""Runtime temporal-WCS index coverage for FFIs absent from the fit lane."""

import hashlib
import json

import pandas as pd
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.common.scc_paths import (
    scc_ffi_dir,
    scc_ffi_list_parquet,
    scc_per_ffi_wcs_dir,
    scc_temporal_wcs_dir,
)

from syndiff_pipeline.difference_imaging.stages.temporal_wcs import (
    ORBIT_BSPLINE_PREDICTION,
    _extend_runtime_frame_index,
    republish_temporal_runtime_index,
)
from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcs


def _fitted_frames():
    return pd.DataFrame(
        {
            "stem": ["tess1000000000000-s0020-3-3", "tess1000000010000-s0020-3-3"],
            "btjd": [100.0, 200.0],
            "frame_index": [0, 1],
            "n_stars_qc": [100, 100],
            "fit_provenance": ["fit", "fit"],
            "median_residual": [0.1, 0.1],
            "orbit_index": [0, 1],
        }
    )


def _ffi_list():
    return pd.DataFrame(
        {
            "filename": [
                "tess1000000000000-s0020-3-3-0165-s_ffic.fits",
                "tess1000000005000-s0020-3-3-0165-s_ffic.fits",
                "tess1000000010000-s0020-3-3-0165-s_ffic.fits",
            ],
            "date_obs": [
                "2018-07-19T12:00:00.000",
                "2018-10-27T12:00:00.000",
                "2019-02-04T12:00:00.000",
            ],
            "wcs_ok": [True, False, True],
        }
    ).set_index("filename")


def test_runtime_index_adds_header_wcs_failure_as_orbit_prediction():
    ffi_list = _ffi_list()
    runtime = _extend_runtime_frame_index(
        _fitted_frames(),
        ffi_list_df=ffi_list,
        ffi_filenames=list(ffi_list.index),
    )

    assert len(runtime) == 3
    predicted = runtime[runtime["fit_provenance"] == ORBIT_BSPLINE_PREDICTION]
    assert len(predicted) == 1
    row = predicted.iloc[0]
    assert row["stem"] == "tess1000000005000-s0020-3-3"
    # The added cadence is assigned to one of the existing B-spline models;
    # publication must never create a model-less third orbit.
    assert row["orbit_index"] in {0, 1}
    assert row["fit_frame_index"] == -1
    assert row["runtime_source"] == "ffi_list_no_header_wcs"
    assert not bool(row["ffi_wcs_ok"])
    assert runtime["frame_index"].tolist() == list(range(3))
    assert set(runtime["orbit_index"]) == {0, 1}


def test_runtime_index_fails_when_an_added_ffi_has_no_date_obs():
    ffi_list = _ffi_list()
    ffi_list.loc[
        "tess1000000005000-s0020-3-3-0165-s_ffic.fits", "date_obs"
    ] = None
    with pytest.raises(RuntimeError, match="lacks DATE-OBS"):
        _extend_runtime_frame_index(
            _fitted_frames(),
            ffi_list_df=ffi_list,
            ffi_filenames=list(ffi_list.index),
        )


def test_runtime_index_rejects_centroid_frame_absent_from_local_ffi_set():
    ffi_list = _ffi_list().iloc[:2]
    with pytest.raises(RuntimeError, match="absent from local FFI set"):
        _extend_runtime_frame_index(
            _fitted_frames(),
            ffi_list_df=ffi_list,
            ffi_filenames=list(ffi_list.index),
        )


def _reference_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [100.0, 100.0]
    wcs.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def test_republish_only_updates_runtime_index_not_models_or_per_ffi(tmp_path):
    data_root = tmp_path / "data"
    ffi_dir = scc_ffi_dir(data_root, 20, 3, 3)
    ffi_dir.mkdir(parents=True)
    ffi_list = _ffi_list()
    for name in ffi_list.index:
        (ffi_dir / name).touch()
    ffi_list.reset_index().to_parquet(scc_ffi_list_parquet(data_root, 20, 3, 3), index=False)

    model_dir = scc_temporal_wcs_dir(data_root, 20, 3, 3)
    per_dir = scc_per_ffi_wcs_dir(data_root, 20, 3, 3)
    (model_dir / "models").mkdir(parents=True)
    per_dir.mkdir(parents=True)
    frame_df = _fitted_frames()
    frame_df.to_parquet(model_dir / "frames.parquet", index=False)
    model = TemporalChebWcs.from_reference_wcs(
        _reference_wcs(), center=[100.0, 100.0], half_extents=[100.0, 100.0]
    )
    models = []
    for orbit in (0, 1):
        rel = f"models/orbit_{orbit:02d}.npz"
        path = model_dir / rel
        model.save(path)
        models.append(
            {
                "orbit_index": orbit,
                "start": orbit,
                "end": orbit + 1,
                "path": rel,
                "fingerprint": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "model_kind": "temporal_wcs",
        "version": "temporal_cheb5_bspline_v1",
        "spatial_basis": "chebyshev",
        "spatial_degree": 5,
        "n_frames": 2,
        "n_fit": 2,
        "models": models,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    per_manifest = {"kind": "per_ffi_cheb5", "models": {
        stem: f"models/{stem}.npz" for stem in frame_df["stem"]
    }}
    (per_dir / "manifest.json").write_text(json.dumps(per_manifest), encoding="utf-8")

    orbit_bytes = [(model_dir / spec["path"]).read_bytes() for spec in models]
    per_manifest_bytes = (per_dir / "manifest.json").read_bytes()
    result = republish_temporal_runtime_index(data_root, 20, 3, 3)

    assert result["n_frames"] == 3
    assert result["n_fit"] == 2
    assert result["n_orbit_bspline_predictions"] == 1
    assert [(model_dir / spec["path"]).read_bytes() for spec in models] == orbit_bytes
    assert (per_dir / "manifest.json").read_bytes() == per_manifest_bytes
    updated = json.loads((model_dir / "manifest.json").read_text())
    assert updated["n_frames"] == 3
    assert updated["n_orbit_bspline_predictions"] == 1
    runtime = pd.read_parquet(model_dir / "frames.parquet")
    assert (runtime["fit_provenance"] == ORBIT_BSPLINE_PREDICTION).sum() == 1
