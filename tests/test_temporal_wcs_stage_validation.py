"""Fail-closed publication checks for the production temporal-WCS stage."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from syndiff_pipeline.difference_imaging.stages.temporal_wcs import (
    MODEL_VERSION,
    _validate_published_artifacts,
)
from syndiff_pipeline.difference_imaging.wcs.temporal_cheb import TemporalChebWcs


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _artifact(tmp_path, *, bad_model=False, bad_frames=False):
    per = tmp_path / "per_ffi_cheb5"
    model_dir = tmp_path / "temporal_cheb5_bspline_v1"
    (model_dir / "models").mkdir(parents=True)
    per.mkdir()
    model = TemporalChebWcs.from_reference_wcs(
        _wcs(), center=[100, 100], half_extents=[100, 100], poly_degree=5,
        btjd_ref=1.0, btjd_scale=1.0,
    )
    path = model_dir / "models" / "orbit_00.npz"
    model.save(path)
    import hashlib
    fp = hashlib.sha256(path.read_bytes()).hexdigest()
    frames = [SimpleNamespace(stem="fit"), SimpleNamespace(stem="rejected")]
    frame_df = pd.DataFrame({
        "stem": ["fit", "rejected"] if not bad_frames else ["rejected", "fit"],
        "btjd": [1.0, 1.5], "frame_index": [0, 1],
        "fit_provenance": ["fit", "predicted"],
        "median_residual": [0.1, np.nan], "orbit_index": [0, 0],
    })
    manifest = {
        "model_kind": "temporal_wcs", "version": MODEL_VERSION,
        "n_frames": 2, "models": [{"orbit_index": 0, "start": 0, "end": 2,
        "path": "models/orbit_00.npz", "fingerprint": fp}],
    }
    if bad_model:
        manifest["models"][0]["fingerprint"] = "0" * 64
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    frame_df.to_parquet(model_dir / "frames.parquet", index=False)
    per_manifest = {"kind": "per_ffi_cheb5", "models": {"fit": "models/fit.npz"}}
    (per / "manifest.json").write_text(json.dumps(per_manifest), encoding="utf-8")
    return per, model_dir, frame_df, frames


def test_validation_allows_rejected_frame_as_temporal_prediction(tmp_path):
    per, model_dir, frame_df, frames = _artifact(tmp_path)
    _validate_published_artifacts(per, model_dir, frame_df, frames, [])


@pytest.mark.parametrize("case,match", [
    ("bad_model", "fingerprint mismatch"),
    ("bad_frames", "frame manifest is not in input-frame order"),
])
def test_validation_rejects_stale_or_misaligned_publication(tmp_path, case, match):
    per, model_dir, frame_df, frames = _artifact(tmp_path, **{case: True})
    with pytest.raises(RuntimeError, match=match):
        _validate_published_artifacts(per, model_dir, frame_df, frames, [])


def test_validation_rejects_missing_temporal_model_file(tmp_path):
    per, model_dir, frame_df, frames = _artifact(tmp_path)
    (model_dir / "models" / "orbit_00.npz").unlink()
    with pytest.raises(RuntimeError, match="missing temporal model"):
        _validate_published_artifacts(per, model_dir, frame_df, frames, [])


def test_validation_rejects_frame_without_orbit_assignment(tmp_path):
    per, model_dir, frame_df, frames = _artifact(tmp_path)
    frame_df.loc[1, "orbit_index"] = -1
    with pytest.raises(RuntimeError, match="no temporal model"):
        _validate_published_artifacts(per, model_dir, frame_df, frames, [])
